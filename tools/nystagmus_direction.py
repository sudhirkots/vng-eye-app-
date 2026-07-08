"""CLINICAL NYSTAGMUS DIRECTION detector (Dr. K, 2026-07-04) — detect BEATS, not velocity spikes.

Primary output: is there rhythmic jerk nystagmus, and the fast-phase direction. Works on an iris-centre trace
CSV (frame, eye, cx, cy, status). A true FAST PHASE must (Fast-Phase Acceptance Rule): be much faster than the
local slow drift, occur in BOTH eyes at nearly the same time with the same direction, be OPPOSITE the local
slow-phase drift, repeat rhythmically, and obey a refractory interval. Everything else is rejected with a
reason. Image convention: +x = image-right = patient's LEFT; +y = down.

Usage: nystagmus_direction.py [trace.csv]   (default step2_points.csv)
"""
import csv, math, sys
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parent.parent
CLIP = "nystagmus at rest to left in right vestibular neuritis"
OUTDIR = REPO / "outputs" / f"{CLIP}_tracked" / "step2_fixed_circle"
CSVF = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR / "step2_points.csv"
TAG = CSVF.stem
FPS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0   # pass the clip's real fps (25/30/39...)
REFR = int(0.20*FPS)     # refractory: >= 0.2 s between beats (max 5/s)
KCONJ = 3                # conjugacy window (frames) between the two eyes
WDRIFT = int(0.40*FPS)   # window to estimate local slow-phase drift

def gsmooth(x, s=1.0):
    r = max(1, int(3*s)); k = np.exp(-0.5*(np.arange(-r, r+1)/s)**2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")[:len(x)]

def interp_nan(x):
    x = x.copy(); idx = np.arange(len(x)); g = ~np.isnan(x)
    if g.sum() < 2: return np.nan_to_num(x)
    x[~g] = np.interp(idx[~g], idx[g], x[g]); return x

def peaks(x, thr, mindist):
    out = []
    for i in range(1, len(x)-1):
        if x[i] >= thr and x[i] >= x[i-1] and x[i] > x[i+1]:
            if out and i-out[-1] < mindist:
                if x[i] > x[out[-1]]: out[-1] = i
            else: out.append(i)
    return out

# ---- load both eyes ----
rows = list(csv.DictReader(open(CSVF, encoding="utf-8")))
NF = max(int(r["frame"]) for r in rows) + 1
pos = {}; valid = {}; scl = {}; seed_src = {}
has_scl = "scl_left" in rows[0]; has_src = "seed_source" in rows[0]
for ek in ("L", "R"):
    cx = np.full(NF, np.nan); cy = np.full(NF, np.nan); vv = np.zeros(NF, bool)
    nL = np.full(NF, np.nan); nR = np.full(NF, np.nan); ar = np.full(NF, np.nan)
    for r in rows:
        if r["eye"] != ek: continue
        f = int(r["frame"])
        if has_src and r.get("seed_source"): seed_src[ek] = r["seed_source"]
        if r["status"] == "needs_rescue" or not r["cx"]: continue
        cx[f], cy[f], vv[f] = float(r["cx"]), float(r["cy"]), True
        if has_scl and r.get("scl_left"): nL[f], nR[f], ar[f] = float(r["scl_left"]), float(r["scl_right"]), float(r["disc_area"])
    pos[ek] = (interp_nan(cx), interp_nan(cy)); valid[ek] = vv; scl[ek] = (nL, nR, ar)

def local_drift(v, i):
    a, b = max(0, i-WDRIFT), min(len(v), i+WDRIFT+1)
    seg = np.concatenate([v[a:max(a, i-2)], v[i+3:b]])
    return float(np.median(seg)) if seg.size else 0.0

PRESENT = [ek for ek in ("L", "R") if valid[ek].any()]      # eyes actually tracked (1 or 2; conjugate)

def _track_score(ek):
    """Per-eye trackability at load: valid fraction penalised by impossible-jump fraction. Only used to DROP a
    starkly-noisy eye when the other is clearly clean (single-eye rescue). Does NOT touch thresholds/direction."""
    if not valid[ek].any(): return 0.0
    cx, cy = pos[ek]; d = np.hypot(np.diff(cx), np.diff(cy))
    ar = scl[ek][2]; med = np.nanmedian(ar) if np.isfinite(ar).any() else np.nan
    diam = 2*np.sqrt(med/np.pi) if (med == med and med > 0) else 40.0
    return float(valid[ek].mean()) * (1.0 - min(1.0, float((d > 1.5*diam).mean())*3))

# Eye selection (two-eye vs single-eye low-conjugacy fallback) is finalized BELOW, after the windowed
# detector + per-eye quality helpers exist -- see "CONJUGACY-AWARE EYE SELECTION (Fix 1)".

def combined(axis):
    """Use the PRESENT eye(s). Nystagmus is conjugate, so ONE clear iris is enough; if both are tracked,
    average them (cuts per-eye noise). Detrend only the VERY-slow head/camera drift (>3s) so the nystagmus
    (cycles < ~2s) is KEPT. Velocity is NOT high-passed -- the slow-phase drift IS the signal."""
    ds = [pos[ek][axis]-gsmooth(pos[ek][axis], 90.0) for ek in PRESENT]
    comb = np.mean(ds, 0) if ds else np.zeros(NF)
    v = np.gradient(gsmooth(comb, 1.5))
    vmask = np.any([valid[ek] for ek in PRESENT], 0) if PRESENT else np.zeros(NF, bool)
    return comb, v, vmask

CONF_MIN = 0.30                                   # below this -> "no clear directional nystagmus"
# Top-level category thresholds (from the 8-clip calibration: real nystagmus 0.41-0.63, all 3 normals <=0.26).
# NYST_CONF sits in the clean gap; TEND_CONF is kept just under it so the 3 known normals stay no_nystagmus.
NYST_CONF = 0.35                                   # >= this -> nystagmus_likely
TEND_CONF = 0.30                                   # [TEND_CONF, NYST_CONF) -> insufficient_beats (jerk tendency, <3 clean beats)
TRACK_MIN = 0.60                                   # valid-tracking fraction below this -> uncertain_tracking (never call "normal")
QWIN = int(2.0*FPS)                                # signal-quality / per-eye analysis window (~2 s)
# --- single-eye low-conjugacy fallback (Fix 1, 2026-07-08) ---------------------------------------
# When the two eyes' motion does NOT correlate (low conjugacy) the two-eye average is meaningless and the
# clip was being forced to uncertain_tracking. Instead: judge each eye ALONE and, if exactly one eye is
# clean AND shows STRONG horizontal jerk evidence, analyse that eye. Safety: the single-eye bar is STRICTER
# than the two-eye NYST_CONF (a lone noisy eye must not fake nystagmus), and is HORIZONTAL-ONLY for now
# (the vertical channel is under separate repair, Fix 2 -- a lone vertical spike must not create a call).
CONJ_LOW    = 0.20            # inter-eye horizontal-velocity corr below this = "low conjugacy" (matches SQ gate)
SINGLE_CONF = 0.45            # a SINGLE eye must clear this (NYST_CONF 0.35 + a 0.10 single-eye safety margin)
SINGLE_CONS = 0.60           # ...and be internally direction-consistent across windows
SINGLE_QMIN = 0.60           # ...and be usable (tracked+plausible) in >= this fraction of frames
SINGLE_WIN  = 3              # ...over at least this many usable ~2 s windows
SINGLE_BLINK = 0.35          # ...and NOT be blink/artifact-dominated (hard-fail frames above this -> reject)
VERT_MARGIN = 1.5                                  # vertical must beat horizontal by this factor to be called
VERT_MIN = 0.50                                    # ...and clear this absolute confidence (vertical is rare;
#                                                    lid/blink noise mimics it, so demand strong evidence)
def pick_axis(h, v):
    return v if (v["conf"] > VERT_MARGIN*max(h["conf"], 1e-6) and v["conf"] >= VERT_MIN) else h

SPIKE_K = 6.0                                     # winsorize velocity at median +/- SPIKE_K*MAD before scoring.
#   Blinks and the startup transient inject 1-2 isolated one-sided velocity spikes; the cubic skew term is
#   dominated by them, which saturated the VERTICAL channel and produced confident wrong-axis calls on
#   horizontal clips (2026-07-07 calibration). Capping the spikes keeps genuine repeated fast phases (MANY
#   tail frames, sign+count preserved) while removing lone blink spikes (1 frame -> negligible).
def asymmetry(v):
    """Motion asymmetry of a velocity segment. Returns the FAST-phase sign (+1 = image-right/patient-LEFT for
    H, = down for V) and the component asymmetries. Clinician logic: eye spends MORE time drifting the slow way;
    the OPPOSITE way is faster but briefer; the rare high-velocity tail (skew) points to the fast phase."""
    med = np.median(v); mad = 1.4826*np.median(np.abs(v-med)) + 1e-9
    v = np.clip(v, med - SPIKE_K*mad, med + SPIKE_K*mad)         # spike/blink-robust (see SPIKE_K note above)
    floor = max(0.4, 0.5*np.median(np.abs(v)))
    pos, neg = v > floor, v < -floor; npos, nneg = int(pos.sum()), int(neg.sum())
    if npos+nneg < 8: return None
    sp_pos = float(v[pos].mean()) if npos else 0.0; sp_neg = float(-v[neg].mean()) if nneg else 0.0
    time_asym = (nneg-npos)/(npos+nneg)                          # >0: more slow frames drift right -> fast LEFT(+)
    speed_asym = (sp_pos-sp_neg)/(sp_pos+sp_neg+1e-9)            # >0: left flicks faster -> fast LEFT(+)
    sk = ((v-v.mean())**3).mean()/((v.std()+1e-9)**3)           # >0: positive tail -> fast LEFT(+)
    votes = [np.sign(time_asym), np.sign(speed_asym), np.sign(sk)]
    fast = int(np.sign(sum(votes))) or 1
    agree = abs(sum(votes))/3.0
    strength = (abs(time_asym)+abs(speed_asym)+min(1.0, abs(sk)/2))/3.0
    return dict(fast=fast, agree=agree, strength=strength, time_asym=time_asym, speed_asym=speed_asym, skew=sk,
                slow_frac=max(npos, nneg)/(npos+nneg))

def windowed(v, axis, win=int(2.0*FPS), step=int(1.0*FPS)):
    A = asymmetry(v)
    if A is None: return dict(fast=0, conf=0.0, axis=axis, cons=0.0, A=None)
    signs = [w["fast"] for a in range(0, max(1, NF-win), step) if (w := asymmetry(v[a:a+win]))]
    signs = np.array(signs) if signs else np.array([A["fast"]])
    cons = float((signs == A["fast"]).mean())                   # cross-window consistency
    conf = A["agree"]*min(1.0, A["strength"]*2.2)*cons
    return dict(fast=A["fast"], conf=conf, axis=axis, cons=cons, A=A)

def name_axis(fast, axis):
    if not fast: return "none"
    return ("LEFT-beating" if fast > 0 else "RIGHT-beating") if axis == "H" else ("DOWN-beating" if fast > 0 else "UP-beating")

# ---- per-frame signal quality per eye (used by eye selection AND the signal-quality layer below) ----
def eye_frame_quality(ek):
    cx, cy = pos[ek]; tracked = valid[ek]
    nL, nR, ar = scl[ek]; tot = np.nan_to_num(nL)+np.nan_to_num(nR)
    fin = np.isfinite(ar); has_area = fin.any() and np.nanmedian(ar) > 0
    med_area = np.median(ar[fin]) if fin.any() else np.nan
    diam = 2*np.sqrt(med_area/np.pi) if has_area else 40.0
    med_tot = np.median(tot[tot > 0]) if (tot > 0).any() else 1.0
    # (2) plausible disc area (only where area is reported)
    area_ok = np.where(fin & has_area, (ar > 0.35*med_area) & (ar < 2.8*med_area), True)
    # (7) blink / occlusion: area collapse, sclera collapse, or lost track
    blink = (~tracked) | (fin & has_area & (ar < 0.35*med_area)) | (tot < 0.12*med_tot)
    # (3) impossible centroid jump (> ~1.5 iris diameters between two tracked frames)
    d = np.hypot(np.diff(cx, prepend=cx[0]), np.diff(cy, prepend=cy[0]))
    jump = (d > 1.5*diam) & tracked & np.roll(tracked, 1)
    # (1) centroid inside a plausible orbit range (robust: within ~6 MAD of its own tracked trajectory)
    def inrange(a):
        av = a[tracked]
        if av.size < 5: return np.ones(NF, bool)
        c = np.median(av); s = 1.4826*np.median(np.abs(av-c)) + 1e-6
        return np.abs(a-c) < 6*s
    orbit_ok = inrange(cx) & inrange(cy)
    # (6) plausible sclera balance
    bal = np.where(tot > 0, (np.nan_to_num(nL)-np.nan_to_num(nR))/(tot+1e-9), np.nan)
    scl_ok = np.where(np.isfinite(bal), (np.abs(bal) < 0.985) & (tot > 0.12*med_tot), True)
    hard = tracked & (blink | jump | (~area_ok) | (~orbit_ok) | (~scl_ok))    # tracked but a hard failure
    good = tracked & area_ok & (~blink) & (~jump) & orbit_ok & scl_ok
    q = np.zeros(NF, int)
    q[tracked & ~hard] = 2                            # usable (tracked + plausible)
    q[good] = 3                                       # good
    q[hard] = 1                                       # poor
    return q                                          # 0 (missing) where not tracked

# ================= CONJUGACY-AWARE EYE SELECTION (Fix 1: single-eye low-conjugacy fallback) =================
def _eye_vel(ek, axis):                               # same detrend/velocity as combined(), one eye
    d = pos[ek][axis] - gsmooth(pos[ek][axis], 90.0)
    return np.gradient(gsmooth(d, 1.5))
def _eye_quality(ek):
    q = eye_frame_quality(ek); nus = 0
    for a in range(0, NF, QWIN):
        seg = q[a:a+QWIN]
        if seg.size and (seg >= 2).mean() >= 0.5: nus += 1
    return dict(usable=float((q >= 2).mean()), good=float((q == 3).mean()),
                blink=float((q == 1).mean()), n_usable=nus)
def eye_evidence(ek):                                 # this eye's own jerk evidence (whole-clip) + tracking
    Hd = windowed(_eye_vel(ek, 0), "H"); Vd = windowed(_eye_vel(ek, 1), "V")
    dom = pick_axis(Hd, Vd); ax = "H" if dom is Hd else "V"
    return dict(ek=ek, conf=dom["conf"], fast=dom["fast"], ax=ax, cons=dom["cons"],
                dir=name_axis(dom["fast"], ax), Hc=Hd["conf"], Vc=Vd["conf"], **_eye_quality(ek))
def _conjugacy():
    if len(PRESENT) != 2: return None
    vL = np.gradient(gsmooth(pos["L"][0], 1.5)); vR = np.gradient(gsmooth(pos["R"][0], 1.5))
    m = valid["L"] & valid["R"]
    if m.sum() > 10 and vL[m].std() > 1e-6 and vR[m].std() > 1e-6:
        return float(np.corrcoef(vL[m], vR[m])[0, 1])
    return None
def _reliable(e): return e["usable"] >= SINGLE_QMIN and e["n_usable"] >= SINGLE_WIN and e["blink"] <= SINGLE_BLINK
def _strong(e):   return _reliable(e) and e["ax"] == "H" and e["conf"] >= SINGLE_CONF and e["cons"] >= SINGLE_CONS

CONJ = _conjugacy(); DROPPED = None; L_EV = R_EV = None
if len(PRESENT) == 2:
    L_EV, R_EV = eye_evidence("L"), eye_evidence("R")
    if CONJ is None or CONJ >= CONJ_LOW:                 # adequate conjugacy -> both eyes (keep the stark drop)
        sc = {ek: _track_score(ek) for ek in PRESENT}
        lo, hi = min(sc, key=sc.get), max(sc, key=sc.get)
        if sc[lo] < 0.50 and sc[hi] > 0.80:
            DROPPED = lo; PRESENT = [hi]; ANALYSIS_MODE = f"single_eye_fallback_{hi}"
            EYE_REASON = f"{hi} kept; {lo} poorly tracked (track score {sc[lo]:.2f})"
        else:
            ANALYSIS_MODE = "both_eyes_conjugate"
            EYE_REASON = (f"conjugacy {CONJ:.2f} >= {CONJ_LOW}" if CONJ is not None else "single-eye csv")
    else:                                               # LOW conjugacy -> judge each eye ALONE (Fix 1)
        sL, sR = _strong(L_EV), _strong(R_EV)
        if sL and sR and L_EV["fast"] == R_EV["fast"]:
            best = "L" if L_EV["conf"] >= R_EV["conf"] else "R"
            PRESENT = [best]; ANALYSIS_MODE = f"single_eye_fallback_{best}"
            EYE_REASON = f"low conjugacy {CONJ:.2f}; both eyes agree ({L_EV['dir']}), using stronger {best}"
        elif sL and sR:                                 # both strong but OPPOSITE direction
            ANALYSIS_MODE = "uncertain_conflicting_eyes"
            EYE_REASON = f"low conjugacy {CONJ:.2f}; eyes disagree (L {L_EV['dir']} {L_EV['conf']:.2f}, R {R_EV['dir']} {R_EV['conf']:.2f})"
        elif sL or sR:
            best = "L" if sL else "R"; e = L_EV if sL else R_EV
            PRESENT = [best]; ANALYSIS_MODE = f"single_eye_fallback_{best}"
            EYE_REASON = f"low conjugacy {CONJ:.2f}; only {best} clean+strong ({e['dir']} {e['conf']:.2f})"
        else:
            ANALYSIS_MODE = "uncertain_poor_both_eyes"
            EYE_REASON = f"low conjugacy {CONJ:.2f}; neither eye clears single-eye bar (L {L_EV['dir']} {L_EV['conf']:.2f}, R {R_EV['dir']} {R_EV['conf']:.2f})"
elif len(PRESENT) == 1:
    ANALYSIS_MODE = f"single_eye_only_{PRESENT[0]}"; EYE_REASON = "only one eye tracked"
else:
    ANALYSIS_MODE = "uncertain_poor_both_eyes"; EYE_REASON = "no eye tracked"

if DROPPED:                                    EYE_MODE = "one-eye (auto: dropped noisy eye)"
elif "fallback" in ANALYSIS_MODE:              EYE_MODE = "one-eye (low-conjugacy fallback)"
elif len(PRESENT) == 1:                        EYE_MODE = "one-eye (single marked eye)"
else:                                          EYE_MODE = "two-eye"
EYES_USED = list(PRESENT)

combH, vH, vmH = combined(0); combV, vV, vmV = combined(1)

# ---- GAZE ZONES: SCLERA-BALANCE / CANTHUS-PROXIMITY (anatomical, head-motion invariant) ----
# per eye, balance = (sclera LEFT of iris - sclera RIGHT of iris)/(total). More sclera LEFT = iris shifted
# image-right = patient looking LEFT. Combine both eyes. Extreme = one side's sclera ~0 (iris at a canthus).
if has_scl and PRESENT:
    bals = []; exs = []
    for ek in PRESENT:
        nL, nR, _ = scl[ek]; tot = np.nan_to_num(nL)+np.nan_to_num(nR)
        b = np.where(tot > 0, (np.nan_to_num(nL)-np.nan_to_num(nR))/(tot+1e-9), np.nan)
        bals.append(interp_nan(b))
        exs.append((np.minimum(np.nan_to_num(nL), np.nan_to_num(nR))/(tot+1e-9) < 0.12).astype(float))
    inst = np.mean(bals, 0)                               # sclera balance over present eye(s) (+1 = patient LEFT)
    extreme = np.mean(exs, 0)
    # SUSTAINED GAZE: suppress within-beat oscillation (~2s window) -> held eye position; reference to the
    # clip's habitual baseline (median) = primary. The nystagmus drift no longer moves the gaze zone.
    sustained = gsmooth(inst, 2.0*FPS)
    gaze = sustained - np.median(sustained)              # sustained gaze deviation from habitual/primary
    ext = gsmooth(extreme, 2.0*FPS) > 0.5                     # eye(s) near a canthus (already a fraction)
    TG, TX = 0.25, 0.55
    zone = np.where(ext | (np.abs(gaze) > TX), np.where(gaze > 0, "extreme-left", "extreme-right"),
                    np.where(gaze > TG, "left", np.where(gaze < -TG, "right", "primary")))
else:                                                    # fallback: old image-position zoning
    relL = pos["L"][0]-np.median(pos["L"][0]); relR = pos["R"][0]-np.median(pos["R"][0])
    gaze = gsmooth(0.5*(relL+relR), 8.0); Tg = max(12.0, 0.6*np.std(gaze))
    zone = np.where(gaze > Tg, "left", np.where(gaze < -Tg, "right", "primary"))

def zone_asym(v, idx):
    if len(idx) < int(1.5*FPS): return dict(enough=False, n=len(idx), fast=0, conf=0.0)
    A = asymmetry(v[idx])
    if A is None: return dict(enough=False, n=len(idx), fast=0, conf=0.0)
    runs = []; s = idx[0]; prev = idx[0]                  # contiguous runs for cross-window consistency
    for i in idx[1:]:
        if i == prev+1: prev = i
        else: runs.append((s, prev)); s = i; prev = i
    runs.append((s, prev))
    signs = [w["fast"] for a, b in runs if b-a+1 >= int(1.0*FPS) and (w := asymmetry(v[a:b+1]))]
    cons = float(np.mean([sg == A["fast"] for sg in signs])) if signs else 0.6
    return dict(enough=True, n=len(idx), fast=A["fast"], conf=A["agree"]*min(1.0, A["strength"]*2.2)*cons, A=A, cons=cons)

def zone_report(zname):
    idx = np.where(zone == zname)[0]
    h = zone_asym(vH, idx); v = zone_asym(vV, idx)
    dom = pick_axis(h, v); ax = "H" if dom is h else "V"
    return dict(zname=zname, n=len(idx), h=h, v=v, dom=dom, ax=ax,
                label=(name_axis(dom["fast"], ax) if dom["enough"] and dom["conf"] >= CONF_MIN else
                       ("no jerk nystagmus" if dom["enough"] else "not enough data")))

ZONES = ["primary", "left", "right", "extreme-left", "extreme-right"]
present = [z for z in ZONES if int((zone == z).sum()) > 0]
ZR = {z: zone_report(z) for z in present}
H = windowed(vH, "H"); V = windowed(vV, "V"); main = pick_axis(H, V)
def name(d): return name_axis(d["fast"], d["axis"])

zoning = "sclera-balance (anatomical)" if has_scl else "image-position (fallback; head-motion prone)"
print(f"=== GAZE-ZONE NYSTAGMUS CLASSIFICATION === trace: {TAG}   zoning: {zoning}")
print(f"  {'zone':14}{'time':8}{'direction':18}{'conf':7}{'notes'}")
for z in present:
    r = ZR[z]; dom = r["dom"]
    direction = name_axis(dom["fast"], r["ax"]) if (dom["enough"] and dom["conf"] >= CONF_MIN) else \
                ("no jerk nystagmus" if dom["enough"] else "not enough data")
    notes = []
    if "extreme" in z: notes.append("tracking/interpretation less reliable")
    if dom["enough"] and 0 < dom["conf"] < CONF_MIN: notes.append("below confidence threshold")
    if dom["enough"] and r["ax"] == "V" and r["h"]["enough"]:
        notes.append(f"(H was {name_axis(r['h']['fast'],'H')} {r['h']['conf']:.2f})")
    print("  %-14s%4.1fs   %-18s%.2f   %s" % (z, r["n"]/FPS, direction, dom["conf"], "; ".join(notes)))
print(f"  whole-clip reference: {name(main)} (conf {main['conf']:.2f})   [CONF_MIN={CONF_MIN}]")

# ================= SIGNAL-QUALITY LAYER (per-frame -> per-window -> overall) =================
# Formal check that the EllSeg-centroid signal is TRUSTWORTHY before any nystagmus call. Seven checks:
#  (1) centroid inside its own plausible orbit range, (2) plausible disc area, (3) no impossible jumps,
#  (4) stable tracking %, (5) L/R conjugacy (both eyes), (6) plausible sclera balance, (7) not blink/occlusion.
# Per-frame quality in {good=3, usable=2, poor=1, missing=0}. RULE: overall poor/missing -> uncertain_tracking
# (NEVER no_nystagmus). This does NOT measure velocity and makes no VNG claim -- it only gates trust.
# (QWIN + eye_frame_quality are defined ABOVE, before eye selection, so the fallback can reuse them.)

def signal_quality():
    FQ = np.stack([eye_frame_quality(ek) for ek in PRESENT]).max(0) if PRESENT else np.zeros(NF, int)
    conj = None                                       # (5) conjugacy needs both eyes
    if len(PRESENT) == 2:
        vL = np.gradient(gsmooth(pos["L"][0], 1.5)); vR = np.gradient(gsmooth(pos["R"][0], 1.5))
        m = valid["L"] & valid["R"]
        if m.sum() > 10 and vL[m].std() > 1e-6 and vR[m].std() > 1e-6:
            conj = float(np.corrcoef(vL[m], vR[m])[0, 1])
    wl = []                                           # (4) per-window tracking stability
    for a in range(0, NF, QWIN):
        seg = FQ[a:a+QWIN]
        if seg.size == 0: continue
        pf, uf, gf = (seg >= 1).mean(), (seg >= 2).mean(), (seg == 3).mean()
        wl.append("missing" if pf < 0.5 else "poor" if uf < 0.5 else "usable" if gf < 0.5 else "good")
    n_usable = sum(w in ("good", "usable") for w in wl)
    vfrac, ufrac, gfrac = float((FQ >= 1).mean()), float((FQ >= 2).mean()), float((FQ == 3).mean())
    conj_bad = conj is not None and conj < CONJ_LOW
    if not PRESENT or vfrac < 0.50:                       overall = "missing_signal"
    elif ufrac < 0.50 or n_usable < 2 or conj_bad:       overall = "poor_signal"
    elif gfrac < 0.50:                                   overall = "usable_signal"
    else:                                                overall = "good_signal"
    per_eye = {}                                      # per-eye tracking quality (reporting)
    for ek in PRESENT:
        q = eye_frame_quality(ek)
        per_eye[ek] = dict(valid=float((q >= 1).mean()), good=float((q == 3).mean()),
                           src=seed_src.get(ek, "unknown"))
    return dict(FQ=FQ, overall=overall, valid_frac=vfrac, usable_frac=ufrac, good_frac=gfrac,
                n_usable=n_usable, n_windows=len(wl), conj=conj, conj_bad=conj_bad, per_eye=per_eye)

def fast_jump_evidence():
    """Qualitative candidate fast phases: abrupt CORRECTIVE velocity spikes opposite the local slow-phase drift.
    NOT a velocity measurement -- it only counts visible corrective jumps + their dominant direction and the
    longest same-direction run (a direct proxy for the '>=3 beats in succession' definition)."""
    v = vH; slow = gsmooth(v, 0.8*FPS); resid = v - slow
    mad = 1.4826*np.median(np.abs(resid - np.median(resid))) + 1e-6
    jumps = []
    for i in np.where(np.abs(resid) > 4.0*mad)[0]:
        if slow[i] != 0 and np.sign(resid[i]) == -np.sign(slow[i]):       # opposite slow drift = corrective
            if not jumps or i - jumps[-1][0] >= REFR: jumps.append((int(i), int(np.sign(resid[i]))))
    if not jumps: return dict(n=0, dir=0, run=0)
    signs = [s for _, s in jumps]; run = best = 1
    for k in range(1, len(signs)):
        run = run+1 if signs[k] == signs[k-1] else 1; best = max(best, run)
    return dict(n=len(jumps), dir=int(np.sign(sum(signs))) or 1, run=best)

SQ = signal_quality(); FJ = fast_jump_evidence()

# ---- TOP-LEVEL CATEGORY (gated by signal quality FIRST) ----
# Evidence for a call = strongest sustained same-direction slow-phase asymmetry (whole-clip or any NON-extreme
# gaze zone). Extreme-gaze zones report separately but never drive the headline (tracking there less reliable).
zone_ev = [(ZR[z]["dom"]["conf"], ZR[z]["dom"]["fast"], ZR[z]["ax"], z)
           for z in present if "extreme" not in z and ZR[z]["dom"]["enough"]]
ev_conf, ev_fast, ev_ax, ev_zone = max(zone_ev + [(main["conf"], main["fast"], main["axis"], "whole-clip")])
ev_dir = name_axis(ev_fast, ev_ax)
zloc = "" if ev_zone in ("primary", "whole-clip") else f" ({ev_zone} gaze)"

if SQ["overall"] in ("poor_signal", "missing_signal") or not PRESENT:
    category = "uncertain_tracking"
    if ANALYSIS_MODE == "uncertain_conflicting_eyes":
        detail = f"conflicting_eye_signals - {EYE_REASON}; not interpretable"
    elif ANALYSIS_MODE == "uncertain_poor_both_eyes":
        detail = f"{SQ['overall']} - {EYE_REASON}; NOT interpretable as normal"
    else:
        detail = f"{SQ['overall']} - eye-position signal not trustworthy; NOT interpretable as normal"
elif ev_conf >= NYST_CONF:
    strength = "clear" if ev_conf >= 0.55 else "probable"
    category = "nystagmus_likely"; detail = f"{ev_dir}{zloc} - {strength} (conf {ev_conf:.2f})"
elif ev_conf >= TEND_CONF:
    category = "insufficient_beats"; detail = f"a {ev_dir}{zloc} jerk tendency, but <3 clean beats (conf {ev_conf:.2f})"
else:
    category = "no_nystagmus"; detail = f"clear tracking, no repeated jerk pattern (conf {ev_conf:.2f})"

# ---- CLINICAL PATTERN SUMMARY ----
def zone_call(z):                                    # confident signed direction in a zone, else None
    r = ZR.get(z)
    return (r["ax"], r["dom"]["fast"]) if (r and r["dom"]["enough"] and r["dom"]["conf"] >= NYST_CONF) else None
if category == "uncertain_tracking":                 pattern = "uncertain_tracking"
elif category == "no_nystagmus":                     pattern = "no nystagmus"
elif category == "insufficient_beats":               pattern = "no definite nystagmus (sub-threshold tendency)"
elif ev_ax == "V":                                   pattern = "vertical nystagmus"
else:
    lz, rz = zone_call("left"), zone_call("right")   # gaze-evoked = direction CHANGES between L and R gaze
    if lz and rz and lz[0] == "H" and rz[0] == "H" and lz[1] != rz[1]:
        pattern = "gaze-evoked direction-changing nystagmus"
    else:
        pattern = "direction-fixed nystagmus"

srcs = sorted({SQ["per_eye"][ek]["src"] for ek in EYES_USED}) or ["unknown"]
seed_label = "/".join(srcs)
print(f"\n>>> CATEGORY: {category}")
print(f"    {detail}")
print(f"    clinical pattern: {pattern}")
print(f"    seed source: {seed_label}   analysis: {EYE_MODE}, eye(s) used: {'+'.join(EYES_USED) or 'none'}"
      + (f"   [dropped {DROPPED}: poorly tracked]" if DROPPED else ""))
print(f"    analysis_mode: {ANALYSIS_MODE}")
print(f"    eye-selection reason: {EYE_REASON}")
if CONJ is not None:
    print(f"    inter-eye conjugacy: {CONJ:.2f}  ({'LOW -> judged each eye alone' if CONJ < CONJ_LOW else 'adequate -> both eyes'})")
if L_EV and R_EV:
    print("  per-eye INDEPENDENT evidence (whole-clip):")
    for e in (R_EV, L_EV):
        print(f"    - {e['ek']} eye: {e['dir']} conf {e['conf']:.2f} (cons {e['cons']:.2f}, axis {e['ax']}) | "
              f"usable {e['usable']:.0%}, good {e['good']:.0%}, blink {e['blink']:.0%}, usable-windows {e['n_usable']}")
print("  per-eye tracking quality:")
for ek in ("R", "L"):
    if ek in SQ["per_eye"]:
        pe = SQ["per_eye"][ek]
        print(f"    - {ek} eye: tracked {pe['valid']:.0%}, good {pe['good']:.0%}   ({pe['src']})")
    elif ek == DROPPED:
        print(f"    - {ek} eye: DROPPED (poorly tracked; excluded so it cannot spoil the result)")
print("  evidence components:")
mA = main["A"] or {}
print(f"    - slow-phase asymmetry score : {ev_conf:.2f}  (agree {mA.get('agree',0):.2f}, strength {mA.get('strength',0):.2f})")
fjdir = name_axis(FJ["dir"], "H") if FJ["n"] else "n/a"
print(f"    - candidate fast-jumps        : {FJ['n']} corrective (longest same-dir run {FJ['run']}, {fjdir})")
print(f"    - direction consistency       : {main['cons']:.2f}")
print(f"    - usable windows              : {SQ['n_usable']}/{SQ['n_windows']}")
conjs = f", conjugacy {SQ['conj']:.2f}" if SQ["conj"] is not None else ", single-eye"
print(f"    - tracking quality            : {SQ['overall']} (tracked {SQ['valid_frac']:.0%}, good {SQ['good_frac']:.0%}{conjs})")
if category != "uncertain_tracking":
    print(f"    [qualitative screening at {FPS:.0f} fps - no velocity / no VNG metrics; jerk nystagmus only]")
