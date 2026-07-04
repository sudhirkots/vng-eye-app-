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
pos = {}; valid = {}; scl = {}
has_scl = "scl_left" in rows[0]
for ek in ("L", "R"):
    cx = np.full(NF, np.nan); cy = np.full(NF, np.nan); vv = np.zeros(NF, bool)
    nL = np.full(NF, np.nan); nR = np.full(NF, np.nan); ar = np.full(NF, np.nan)
    for r in rows:
        if r["eye"] != ek: continue
        f = int(r["frame"])
        if r["status"] == "needs_rescue" or not r["cx"]: continue
        cx[f], cy[f], vv[f] = float(r["cx"]), float(r["cy"]), True
        if has_scl and r.get("scl_left"): nL[f], nR[f], ar[f] = float(r["scl_left"]), float(r["scl_right"]), float(r["disc_area"])
    pos[ek] = (interp_nan(cx), interp_nan(cy)); valid[ek] = vv; scl[ek] = (nL, nR, ar)

def local_drift(v, i):
    a, b = max(0, i-WDRIFT), min(len(v), i+WDRIFT+1)
    seg = np.concatenate([v[a:max(a, i-2)], v[i+3:b]])
    return float(np.median(seg)) if seg.size else 0.0

PRESENT = [ek for ek in ("L", "R") if valid[ek].any()]      # eyes actually tracked (1 or 2; conjugate)

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

def asymmetry(v):
    """Motion asymmetry of a velocity segment. Returns the FAST-phase sign (+1 = image-right/patient-LEFT for
    H, = down for V) and the component asymmetries. Clinician logic: eye spends MORE time drifting the slow way;
    the OPPOSITE way is faster but briefer; the rare high-velocity tail (skew) points to the fast phase."""
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

combH, vH, vmH = combined(0); combV, vV, vmV = combined(1)

def name_axis(fast, axis):
    if not fast: return "none"
    return ("LEFT-beating" if fast > 0 else "RIGHT-beating") if axis == "H" else ("DOWN-beating" if fast > 0 else "UP-beating")

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
    dom = h if h["conf"] >= v["conf"] else v; ax = "H" if dom is h else "V"
    return dict(zname=zname, n=len(idx), h=h, v=v, dom=dom, ax=ax,
                label=(name_axis(dom["fast"], ax) if dom["enough"] and dom["conf"] >= CONF_MIN else
                       ("no clear nystagmus" if dom["enough"] else "not enough data")))

ZONES = ["primary", "left", "right", "extreme-left", "extreme-right"]
present = [z for z in ZONES if int((zone == z).sum()) > 0]
ZR = {z: zone_report(z) for z in present}
H = windowed(vH, "H"); V = windowed(vV, "V"); main = H if H["conf"] >= V["conf"] else V
def name(d): return name_axis(d["fast"], d["axis"])

zoning = "sclera-balance (anatomical)" if has_scl else "image-position (fallback; head-motion prone)"
print(f"=== GAZE-ZONE NYSTAGMUS CLASSIFICATION === trace: {TAG}   zoning: {zoning}")
print(f"  {'zone':14}{'time':8}{'direction':18}{'conf':7}{'notes'}")
for z in present:
    r = ZR[z]; dom = r["dom"]
    direction = name_axis(dom["fast"], r["ax"]) if (dom["enough"] and dom["conf"] >= CONF_MIN) else \
                ("no clear nystagmus" if dom["enough"] else "not enough data")
    notes = []
    if "extreme" in z: notes.append("tracking/interpretation less reliable")
    if dom["enough"] and 0 < dom["conf"] < CONF_MIN: notes.append("below confidence threshold")
    if dom["enough"] and r["ax"] == "V" and r["h"]["enough"]:
        notes.append(f"(H was {name_axis(r['h']['fast'],'H')} {r['h']['conf']:.2f})")
    print("  %-14s%4.1fs   %-18s%.2f   %s" % (z, r["n"]/FPS, direction, dom["conf"], "; ".join(notes)))
print(f"  whole-clip reference: {name(main)} (conf {main['conf']:.2f})   [CONF_MIN={CONF_MIN}]")
