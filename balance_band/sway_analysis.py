"""BALANCE BAND — waist-worn IMU to evaluate the balance system of anyone (Dr. K, 2026-09-23).

A belt over the lower back (L3–L5, near the body's centre of gravity) carries an IMU (accelerometer +
gyroscope). Its tilt closely tracks the centre of gravity (COG) angle, whether the person corrects sway at
the ankles or at the hips. (A shin band, --site shin, is also supported, but it only sees the ankle part of
sway; a second shin band alongside the waist, to tell ankle from hip strategy, is planned.)
For anyone: a healthy person, someone who feels unsteady, or any patient (vestibular, neuropathy,
cerebellar, parkinsonism, the elderly, ...).

SESSION (one CSV per step, all in one folder, band NOT moved between steps):

  los.csv      LIMITS OF STABILITY. Stand still ~3 s (-> the CENTRE), then lean as far as possible
               FORWARD, back to centre, BACKWARD, centre, LEFT, centre, RIGHT, centre — without stepping.
               -> how far the COG can go in each direction before balance is lost.
  eo_firm.csv  eyes open,   firm floor   20 s   reference: vision + somatosensory + vestibular
  ec_firm.csv  eyes closed, firm floor   20 s   vision removed          -> how much VISION matters
  eo_foam.csv  eyes open,   on foam      20 s   somatosensory degraded  -> how much SOMATOSENSORY input matters
  ec_foam.csv  eyes closed, on foam      20 s   both removed            -> balance rests mainly on VESTIBULAR input

Each CSV: t, ax, ay, az, gx, gy, gz [, fall] [, phase]
  t in s; accel in g or m/s^2 (only its direction is used); gyro in deg/s (or --gyro-units rad);
  fall = 1 from the moment the examiner presses "lost balance / stepped / grabbed";
  phase (los.csv only) = the instruction on screen at that moment: centre / forward / backward / left / right.

BAND ORIENTATION IS FOUND AUTOMATICALLY: the band can be strapped on any way round. With the phase labels
in los.csv, "down" comes from the quiet stance (centre) and "forward" from the forward lean, so the sensor
axes never have to be known. Without labels, give --forward/--left, or +x forward / +y left is assumed.

For each standing condition we report the sway angle, how it compares with eyes-open-firm, and how much of
the person's OWN limit of stability the sway used up. Qualitative read + honest confidence: bad or missing
data give INSUFFICIENT / UNCLEAR, never a false "normal".

Usage:
    python balance_band/sway_analysis.py <session_folder> [--site waist|shin] [--json out.json]
                                         [--forward +x --left +y]      (only if los.csv has no phase labels)
    python balance_band/sway_analysis.py <empty_folder> --demo backward [--demo-tilted]  # synthetic session
    (demo patterns: normal, weak_vestibular, weak_somatosensory, weak_visual, backward)

Body frame: F = person's forward, L = person's LEFT, U = up.
  AP angle > 0 = COG FORWARD of centre;  ML angle > 0 = COG to the person's LEFT.
Thresholds marked PROVISIONAL are placeholders until healthy controls are recorded on this band.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

CONDITIONS = ("eo_firm", "ec_firm", "eo_foam", "ec_foam")
LABEL = {"eo_firm": "Eyes open, firm", "ec_firm": "Eyes closed, firm",
         "eo_foam": "Eyes open, foam", "ec_foam": "Eyes closed, foam"}
DIRS = ("forward", "backward", "left", "right")
SITES = ("waist", "shin")

# ---- quality gates (failing these -> INSUFFICIENT, never "normal") ----
MIN_FS_HZ = 20.0
WARN_FS_HZ = 50.0          # >= 100 Hz recommended
MIN_DURATION_S = 10.0      # a completed standing trial shorter than this is insufficient
PLANNED_DURATION_S = 20.0
CENTRE_S = 2.0             # quiet stance at the start of los.csv that defines the centre
LOS_SCRIPT = [("centre", 3), ("forward", 6), ("centre", 3), ("backward", 6), ("centre", 3),
              ("left", 6), ("centre", 3), ("right", 6), ("centre", 3)]      # phase, seconds (as the app shows)
AXES_MIN_TILT_DEG = 1.5    # the forward lean must tilt the band at least this much to find "forward"
AXES_AGREE = 0.7           # cos(angle) the other leans must agree with the found axes
MIN_LOS_DEG = 1.0          # a lean smaller than this = direction not attempted

# ---- signal processing ----
COMP_TAU_S = 1.0           # complementary filter: gyro below ~1 s, gravity above
SWAY_LP_HZ = 2.5           # postural sway lives below ~2 Hz; faster content is sensor noise / jolts

# ---- PROVISIONAL interpretation thresholds (calibrate on healthy controls from THIS band) ----
RATIO_INCREASED = 2.0      # removing ONE sense doubles the sway
RATIO_EC_FOAM = 4.0        # removing vision AND somatosensory input together
LOS_USED_HIGH = 70.0       # sway reaching >= 70 % of the person's limit -> near the edge (fall risk)
LOS_BACK_RATIO = 0.5       # backward limit < half the forward limit -> reduced backward stability
LOS_SIDE_RATIO = 0.6       # one side < 60 % of the other -> side-to-side asymmetry
LEAN_DEG = 2.0             # mean ML offset from the eyes-open-firm position that counts as a lean
# Sensory ratios: each condition gets a stability score 0-100 (100 = no sway, 0 = sway reached the person's
# own limit of stability, or a fall); each ratio = that condition's score / eyes-open-firm score.
# Each sense has its OWN cut-off, because even healthy people do worst on eyes-closed foam.
SENSORY = (("somatosensory", "ec_firm", "eyes closed, firm"),
           ("visual", "eo_foam", "eyes open, foam"),
           ("vestibular", "ec_foam", "eyes closed, foam"))
RATIO_LOW = {"somatosensory": 80.0, "visual": 70.0, "vestibular": 50.0}   # PROVISIONAL, % of reference
WEAK_TEXT = {
    "somatosensory": "Weak use of SOMATOSENSORY input (feet and joints): balance drops when the eyes close on a "
                     "firm floor — the person leans on vision.",
    "visual": "Weak use of VISION: balance drops on foam with eyes open — the person leans on firm support "
              "under the feet.",
    "vestibular": "Weak use of VESTIBULAR input: balance drops, or fails, when mainly vestibular input is left "
                  "(eyes closed on foam)."}
REF_SCORE_MIN = 40.0       # below this the reference itself is too unsteady for ratios to mean much
LOB_DEG = 8.0              # sway beyond this from the trial's centre, unmarked -> flag for review


# ------------------------------------------------------------------ axes / geometry
def axis_vec(s):
    s = s.strip().lower(); v = np.zeros(3)
    v["xyz".index(s[-1])] = -1.0 if s.startswith("-") else 1.0
    return v

def body_matrix(forward="+x", left="+y"):
    """Rows = body axes (F, L, U) in sensor coords; U = F x L (right-handed)."""
    f, l = axis_vec(forward), axis_vec(left)
    if abs(f @ l) > 0.5:
        raise ValueError("--forward and --left must be different sensor axes")
    return np.vstack([f, l, np.cross(f, l)])

def align_rotation(g_body):
    """Rotation taking the measured gravity direction at the centre onto +U (fixes a tilted band)."""
    a = g_body / np.linalg.norm(g_body); b = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


# ------------------------------------------------------------------ loading / filtering
def load_trial(path):
    path = Path(path)
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return None
    d = {k: np.array([float(r[k]) for r in rows]) for k in ("t", "ax", "ay", "az", "gx", "gy", "gz")}
    d["fall"] = np.array([int(float(r.get("fall") or 0)) for r in rows])
    d["phase"] = np.array([(r.get("phase") or "").strip().lower() for r in rows])
    return d

def lowpass(x, fs, fc):
    """Zero-phase 2nd-order low-pass (two one-pole passes, forward and backward)."""
    a = math.exp(-2 * math.pi * fc / fs)

    def one_pole(seq):
        out = np.empty_like(seq); out[0] = seq[0]
        for i in range(1, len(seq)):
            out[i] = a * out[i-1] + (1 - a) * seq[i]
        return out
    y = np.asarray(x, float)
    for _ in range(2):
        y = one_pole(one_pole(y)[::-1])[::-1]      # forward then backward -> no time shift
    return y

def sample_rate(t):
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return 1.0 / dt if dt > 0 else 0.0

def body_signals(tr, B, R, gyro_rad=False):
    acc = np.stack([tr["ax"], tr["ay"], tr["az"]], 1) @ B.T @ R.T
    gyr = np.stack([tr["gx"], tr["gy"], tr["gz"]], 1) @ B.T @ R.T
    if gyro_rad:
        gyr = np.degrees(gyr)
    return acc, gyr

def sway_angles(tr, B, R, gyro_rad=False, lp=True):
    """Complementary filter -> AP and ML COG angles (deg) from centre, + angular rates (deg/s)."""
    t = tr["t"]; acc, gyr = body_signals(tr, B, R, gyro_rad)
    aF, aL, aU = acc[:, 0], acc[:, 1], acc[:, 2]
    ap_acc = np.degrees(np.arctan2(-aF, np.hypot(aL, aU)))    # forward lean -> gravity appears toward -F
    ml_acc = np.degrees(np.arctan2(-aL, np.hypot(aF, aU)))    # left lean    -> gravity appears toward -L
    ap_rate, ml_rate = gyr[:, 1], -gyr[:, 0]                  # about +L = top forward; about -F = top left
    n0 = max(1, int(np.searchsorted(t, t[0] + 0.5)))          # start from 0.5 s of gravity, not one sample
    ap = np.empty_like(t); ml = np.empty_like(t)
    ap[0], ml[0] = ap_acc[:n0].mean(), ml_acc[:n0].mean()
    for i in range(1, len(t)):
        dt = max(t[i] - t[i-1], 1e-4); a = COMP_TAU_S / (COMP_TAU_S + dt)
        ap[i] = a * (ap[i-1] + ap_rate[i] * dt) + (1 - a) * ap_acc[i]
        ml[i] = a * (ml[i-1] + ml_rate[i] * dt) + (1 - a) * ml_acc[i]
    fs = sample_rate(t)
    if lp and fs > 2.5 * SWAY_LP_HZ:
        ap, ml = lowpass(ap, fs, SWAY_LP_HZ), lowpass(ml, fs, SWAY_LP_HZ)
    return ap, ml, ap_rate, ml_rate, np.linalg.norm(acc, axis=1)

def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


# ------------------------------------------------------------------ band orientation from the LOS recording
def _lean_direction(acc, g_hat, mask):
    """Horizontal direction (sensor coords) in which gravity APPEARS to move during a lean, and the tilt (deg)."""
    a = acc[mask]
    if len(a) < 5:
        return None, 0.0
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(a @ g_hat, -1, 1)))
    peak = a[tilt >= 0.7 * tilt.max()]                     # the held part of the lean
    h = peak.mean(0) - g_hat
    h -= (h @ g_hat) * g_hat                               # keep only the horizontal part
    n = np.linalg.norm(h)
    return (h / n if n > 0 else None), float(tilt.max())

def auto_axes(los):
    """Body axes (rows F, L, U in sensor coords) from a phase-labelled LOS recording -> (B, note) or (None, why)."""
    if los is None or not np.any(los["phase"] == "forward") or not np.any(los["phase"] == "centre"):
        return None, "no phase labels in los.csv"
    acc = np.stack([los["ax"], los["ay"], los["az"]], 1)
    centre = np.flatnonzero(los["phase"] == "centre")
    breaks = np.flatnonzero(np.diff(centre) > 1)
    first = centre[:breaks[0] + 1] if breaks.size else centre   # the opening quiet stance
    g = acc[first].mean(0); g_hat = g / np.linalg.norm(g)       # UP (the accelerometer reads +1 g upward)
    d, tilt = _lean_direction(acc, g_hat, los["phase"] == "forward")
    if d is None or tilt < AXES_MIN_TILT_DEG:
        return None, f"forward lean too small ({tilt:.1f} deg) to find the band's forward direction"
    F = -d                                                 # leaning forward, gravity appears toward -F
    L = np.cross(g_hat, F)
    B = np.vstack([F, L, g_hat])
    odd = []
    for ph, want in (("backward", -F), ("left", L), ("right", -L)):
        dd, tt = _lean_direction(acc, g_hat, los["phase"] == ph)
        if dd is not None and tt >= AXES_MIN_TILT_DEG and -dd @ want < AXES_AGREE:
            odd.append(ph)
    return B, (f"the {'/'.join(odd)} lean did not point where expected — were the leans done in the "
               "order shown? Check the result." if odd else "")


# ------------------------------------------------------------------ step 1: limits of stability
def analyse_los(tr, B, R, gyro_rad=False):
    out = {"status": "insufficient", "notes": []}
    fs = sample_rate(tr["t"]); out["fs_hz"] = round(fs, 1)
    if fs < MIN_FS_HZ:
        out["notes"].append(f"sampling {fs:.0f} Hz < {MIN_FS_HZ:.0f} Hz"); return out
    ap, ml, *_ = sway_angles(tr, B, R, gyro_rad)
    fall = np.flatnonzero(tr["fall"] > 0); stop = int(fall[0]) if fall.size else len(ap)
    if fall.size:
        out["notes"].append(f"lost balance at {tr['t'][stop]-tr['t'][0]:.1f} s while leaning — "
                            "limits are measured up to that moment")
    ap, ml, ph = ap[:stop], ml[:stop], tr["phase"][:stop]
    if len(ap) < int(3 * fs):
        out["notes"].append("recording too short"); return out
    lim = {"forward": pct(ap, 99), "backward": -pct(ap, 1), "left": pct(ml, 99), "right": -pct(ml, 1)}
    if np.any(ph != ""):     # labelled: each limit only from its own lean, so sway elsewhere cannot count
        sel = {"forward": (ap, 1), "backward": (ap, -1), "left": (ml, 1), "right": (ml, -1)}
        for d, (x, sgn) in sel.items():
            m = ph == d
            lim[d] = sgn * pct(x[m], 99 if sgn > 0 else 1) if m.any() else 0.0
    out["limits_deg"] = {k: round(max(v, 0.0), 1) for k, v in lim.items()}
    missing = [k for k, v in lim.items() if v < MIN_LOS_DEG]
    if missing:
        out["notes"].append("no lean recorded " + "/".join(missing) + " — not attempted, or could not lean at all")
    out["missing"] = missing
    out["status"] = "complete" if not missing else "partial"
    return out


# ------------------------------------------------------------------ steps 2-5: standing conditions
def analyse_trial(tr, B, R, los=None, gyro_rad=False):
    out = {"status": "insufficient", "notes": []}
    t = tr["t"]
    if len(t) < 3:
        out["notes"].append("too few samples"); return out
    fs = sample_rate(t); out["fs_hz"] = round(fs, 1)
    if fs < MIN_FS_HZ:
        out["notes"].append(f"sampling {fs:.0f} Hz < {MIN_FS_HZ:.0f} Hz"); return out
    if fs < WARN_FS_HZ:
        out["notes"].append(f"low sampling rate ({fs:.0f} Hz; >=100 Hz recommended)")
    dts = np.diff(t)
    if np.any(dts > 5.0 / fs):
        out["notes"].append(f"{int(np.sum(dts > 5.0/fs))} data gap(s)")

    ap, ml, ap_rate, ml_rate, amag = sway_angles(tr, B, R, gyro_rad)
    fall_idx = np.flatnonzero(tr["fall"] > 0); fell = fall_idx.size > 0
    stop = int(fall_idx[0]) if fell else len(t)
    seg = slice(0, stop)
    dur = float(t[stop-1] - t[0]) if stop > 1 else 0.0
    out["duration_s"] = round(dur, 1)
    if stop > 1:
        over = np.flatnonzero(np.hypot(ap[seg] - np.median(ap[seg]), ml[seg] - np.median(ml[seg])) > LOB_DEG)
        if over.size and not fell:
            out["notes"].append(f"sway exceeded {LOB_DEG:.0f} deg at {t[over[0]]-t[0]:.1f} s — possible loss of "
                                "balance not marked by the examiner; please check")
    g = np.median(amag)
    if g > 0 and stop > 1 and np.mean(np.abs(amag[seg] / g - 1.0) > 0.25) > 0.05:
        out["notes"].append("large jolts (stepping, or the band slipping?) — angles less reliable")

    if fell:
        out["status"] = "fell"; out["fall_at_s"] = round(float(t[stop] - t[0]), 1)
        k = max(0, stop - int(0.5 * fs)); dap, dml = ap[stop-1] - ap[k], ml[stop-1] - ml[k]
        if max(abs(dap), abs(dml)) > 0.5:
            out["fall_direction"] = (("forward" if dap > 0 else "backward") if abs(dap) >= abs(dml)
                                     else ("to the LEFT" if dml > 0 else "to the RIGHT"))
    elif dur < MIN_DURATION_S:
        out["notes"].append(f"only {dur:.0f} s recorded (< {MIN_DURATION_S:.0f} s) and no fall marked")
        return out
    else:
        out["status"] = "complete"
        if dur < 0.8 * PLANNED_DURATION_S:
            out["notes"].append(f"shorter than planned ({dur:.0f} s of {PLANNED_DURATION_S:.0f} s)")
    if stop < max(3, int(1.0 * fs)):      # fell at once: no sway metrics — the fall IS the finding
        return out

    a, m = ap[seg], ml[seg]
    ac, mc = a - a.mean(), m - m.mean()
    ev = np.clip(np.linalg.eigvalsh(np.cov(np.vstack([ac, mc]))), 0, None)
    out.update({
        "ap_range90_deg": round(pct(a, 95) - pct(a, 5), 2),
        "ml_range90_deg": round(pct(m, 95) - pct(m, 5), 2),
        "ap_rms_deg": round(float(np.sqrt(np.mean(ac**2))), 2),
        "ml_rms_deg": round(float(np.sqrt(np.mean(mc**2))), 2),
        "ellipse95_deg2": round(float(math.pi * 5.991 * math.sqrt(ev[0] * ev[1])), 2),
        "mean_ap_deg": round(float(a.mean()), 2), "mean_ml_deg": round(float(m.mean()), 2),
    })
    out["sway_deg"] = round(math.hypot(out["ap_range90_deg"], out["ml_range90_deg"]), 2)

    # how much of the person's own limit of stability did this trial use?
    if los and "limits_deg" in los:
        reach = {"forward": pct(a, 99), "backward": -pct(a, 1), "left": pct(m, 99), "right": -pct(m, 1)}
        used = {d: 100.0 * max(reach[d], 0) / los["limits_deg"][d]
                for d in DIRS if los["limits_deg"][d] >= MIN_LOS_DEG}
        if used:
            d = max(used, key=used.get)
            out["los_used_pct"] = round(min(used[d], 999.0)); out["los_used_dir"] = d
    if fell:
        out["los_used_pct"] = 100                  # a fall = the limit was reached, by definition
        out["los_used_dir"] = {"forward": "forward", "backward": "backward", "to the LEFT": "left",
                               "to the RIGHT": "right"}.get(out.get("fall_direction"), out.get("los_used_dir"))
    return out


# ------------------------------------------------------------------ session
def analyse_session(folder, forward=None, left=None, gyro_rad=False, site="waist"):
    """forward/left: sensor axes (e.g. '+x', '+y'). Leave both None to find them from the LOS recording."""
    folder = Path(folder)
    los_tr = load_trial(folder / "los.csv")
    trials = {c: load_trial(folder / f"{c}.csv") for c in CONDITIONS}
    if site not in SITES:
        raise ValueError(f"site must be one of {SITES}")
    res = {"site": site, "los": None, "trials": {}, "comparisons": {}, "findings": [], "caveats": []}
    if forward or left:
        B = body_matrix(forward or "+x", left or "+y")
        res["axes"] = {"method": "given", "forward": forward or "+x", "left": left or "+y"}
    else:
        B, why = auto_axes(los_tr)
        if B is not None:
            res["axes"] = {"method": "found automatically from the limits-of-stability leans",
                           "forward": np.round(B[0], 3).tolist(), "left": np.round(B[1], 3).tolist()}
            if why:
                res["caveats"].append("Band orientation: " + why)
        else:
            B = body_matrix("+x", "+y")
            res["axes"] = {"method": "ASSUMED (+x forward, +y left)", "forward": "+x", "left": "+y"}
            res["caveats"].append(f"Band orientation NOT known ({why}); assumed sensor +x = forward, +y = left. "
                                  "If that is wrong, the forward/backward and left/right findings are wrong.")

    # CENTRE = quiet stance at the start of the LOS recording (else the start of eyes-open-firm)
    ref, secs = (los_tr, CENTRE_S) if los_tr is not None else (trials["eo_firm"], 1.0)
    if ref is None:
        res["overall"] = "INSUFFICIENT — no recordings found"; return res
    n0 = max(3, min(len(ref["t"]), int(np.searchsorted(ref["t"], ref["t"][0] + secs))))
    g0 = np.array([ref["ax"][:n0].mean(), ref["ay"][:n0].mean(), ref["az"][:n0].mean()]) @ B.T
    R = align_rotation(g0)
    tilt = math.degrees(math.acos(np.clip(g0[2] / np.linalg.norm(g0), -1, 1)))
    if tilt > 25:
        res["caveats"].append(f"band sits {tilt:.0f} deg off vertical at the centre — check --forward/--left")

    if los_tr is None:
        res["caveats"].append("No limits-of-stability recording — sway cannot be expressed as % of the "
                              "person's own limits; centre taken from the start of eyes-open-firm.")
    else:
        res["los"] = analyse_los(los_tr, B, R, gyro_rad)
    for c in CONDITIONS:
        res["trials"][c] = ({"status": "not done", "notes": []} if trials[c] is None
                            else analyse_trial(trials[c], B, R, res["los"], gyro_rad))
    interpret(res)
    return res


def sensory_ratios(res):
    """Stability score per condition and the three sensory ratios (%), into res['sensory']."""
    T = res["trials"]
    score = {}
    for c in CONDITIONS:
        if T[c]["status"] == "fell":
            score[c] = 0.0
        elif T[c]["status"] == "complete" and "los_used_pct" in T[c]:
            score[c] = round(max(0.0, 100.0 - T[c]["los_used_pct"]), 1)
    out = {"scores": score, "ratios": {}, "low": [], "notes": []}
    res["sensory"] = out
    if not res.get("los") or "limits_deg" not in res["los"]:
        out["notes"].append("Sensory ratios need the limits-of-stability step (step 1) — not available.")
        return
    ref = score.get("eo_firm")
    if not ref:
        out["notes"].append("Sensory ratios need a usable eyes-open-firm trial — not available.")
        return
    if ref < REF_SCORE_MIN:
        out["notes"].append(f"Unsteady even with eyes open on a firm floor (score {ref:.0f}/100), so the sensory "
                            "ratios are unreliable — the problem is broader than one sense.")
    for sense, c, _ in SENSORY:
        if c in score:
            r = round(min(100.0, 100.0 * score[c] / ref))
            out["ratios"][sense] = r
            if r < RATIO_LOW[sense]:
                out["low"].append(sense)
        else:
            out["notes"].append(f"{sense.capitalize()} ratio: condition not usable ({T[c]['status']}).")


def interpret(res):
    T, F, C, L = res["trials"], res["findings"], res["comparisons"], res["los"]

    # --- limits of stability ---
    if L and "limits_deg" in L:
        lim = L["limits_deg"]
        if lim["forward"] >= MIN_LOS_DEG and lim["backward"] < LOS_BACK_RATIO * lim["forward"]:
            F.append(f"Reduced BACKWARD limit of stability ({lim['backward']:.1f} deg vs {lim['forward']:.1f} deg "
                     "forward) — backward falls are the risk (seen e.g. in parkinsonism and in older fallers).")
        lo, hi = sorted([("left", lim["left"]), ("right", lim["right"])], key=lambda x: x[1])
        if hi[1] >= MIN_LOS_DEG and lo[1] < LOS_SIDE_RATIO * hi[1]:
            F.append(f"Side-to-side asymmetry: can lean much less to the {lo[0].upper()} "
                     f"({lo[1]:.1f} vs {hi[1]:.1f} deg).")

    usable = lambda c: T[c]["status"] in ("complete", "fell")
    if T["eo_firm"]["status"] == "fell":
        F.append("Cannot stand feet-together with eyes open on a firm floor — severe imbalance; the sensory "
                 "conditions cannot be compared.")
        res["overall"] = "SEVERE IMBALANCE (fails the easiest condition)"; _caveats(res); return
    base = T["eo_firm"].get("sway_deg") if T["eo_firm"]["status"] == "complete" else None
    if not base:
        res["overall"] = ("INSUFFICIENT — eyes-open-firm reference trial "
                          f"{T['eo_firm']['status']}; cannot compare conditions"); _caveats(res); return

    def rel(c):
        if T[c]["status"] == "fell": return "fell"
        if T[c]["status"] != "complete" or T[c].get("sway_deg") is None: return None
        return round(T[c]["sway_deg"] / base, 2)
    C["x_ref_ec_firm"] = rel("ec_firm")          # sway as a multiple of eyes-open-firm (descriptive)
    C["x_ref_eo_foam"] = rel("eo_foam")
    C["x_ref_ec_foam"] = rel("ec_foam")

    for c in ("ec_firm", "eo_foam", "ec_foam"):
        if T[c]["status"] == "fell":
            d = T[c].get("fall_direction")
            F.append(f"{LABEL[c]}: lost balance at {T[c]['fall_at_s']} s" + (f", falling {d}" if d else "") + ".")

    # --- which sense is used poorly (-> where to focus rehabilitation) ---
    sensory_ratios(res)
    S = res["sensory"]
    if S["ratios"]:
        weak = list(S["low"])
    else:                     # no limits of stability: judge from the sway multiples instead (cruder)
        up = lambda r, thr: r == "fell" or (isinstance(r, float) and r >= thr)
        weak = [sense for sense, key, thr in (("somatosensory", "x_ref_ec_firm", RATIO_INCREASED),
                                              ("visual", "x_ref_eo_foam", RATIO_INCREASED),
                                              ("vestibular", "x_ref_ec_foam", RATIO_EC_FOAM)) if up(C[key], thr)]
        if weak:
            res["caveats"].append("Without step 1 the weak sense is judged from how many times the sway rose "
                                  "(x ref), not from the sensory ratios — cruder.")
    for x in weak:
        F.append(WEAK_TEXT[x])

    near = [(c, T[c]["los_used_pct"], T[c]["los_used_dir"]) for c in CONDITIONS
            if T[c].get("los_used_pct", 0) >= LOS_USED_HIGH and T[c]["status"] == "complete"]
    for c, p, d in near:
        F.append(f"{LABEL[c]}: sway used {p}% of the {d} limit of stability — close to the edge (fall risk).")

    m0 = T["eo_firm"]["mean_ml_deg"]
    side = [np.sign(T[c]["mean_ml_deg"] - m0) for c in ("ec_firm", "eo_foam", "ec_foam")
            if "mean_ml_deg" in T[c] and abs(T[c]["mean_ml_deg"] - m0) >= LEAN_DEG]
    if len(side) >= 2 and abs(sum(side)) == len(side):
        F.append(f"Consistent lean to the person's {'LEFT' if side[0] > 0 else 'RIGHT'} when senses are removed.")

    missing = [LABEL[c] for c in CONDITIONS if not usable(c)]
    if missing:
        res["caveats"].append("Not usable: " + ", ".join(missing) + " — comparisons with these are unclear.")
    ref_score = S["scores"].get("eo_firm")
    if ref_score is not None and S["ratios"] and ref_score < REF_SCORE_MIN:
        res["overall"] = "UNSTEADY EVEN WITH ALL SENSES AVAILABLE"
    elif weak:
        res["overall"] = "WEAK " + " + ".join(x.upper() for x in weak) + " USE"
    elif missing:
        res["overall"] = "UNCLEAR — no weak sense on the usable trials, but some conditions are missing"
    else:
        res["overall"] = "NO WEAK SENSE on the conditions tested"
        if not S["ratios"]:
            res["caveats"].append("Relative comparison only: someone who sways a lot in EVERY condition can still "
                                  "read 'no weak sense' — record step 1 so sway can be judged against their limits.")
    if not weak and any("limit" in f.lower() or "lean" in f.lower() for f in F):
        res["overall"] += " — but see stability findings"
    _caveats(res)


def _caveats(res):
    if res.get("site") == "shin":
        res["caveats"].append("Shin band: sees only the ankle (inverted-pendulum) part of sway. On foam or near the "
                              "limits people bend at the hips, which a shin sensor under-reads — use the waist band.")
    else:
        res["caveats"].append("Waist band: measures the centre of gravity's tilt but cannot say whether the person "
                              "corrected at the ankles or the hips (that needs a second, shin band).")
    res["caveats"].append("Foam DEGRADES rather than abolishes foot/ankle sensation, so eyes-closed-on-foam "
                          "leans mainly — not purely — on vestibular input.")
    res["caveats"].append("Thresholds are PROVISIONAL until healthy controls are recorded on this band: sensory-"
                          "ratio cut-offs somatosensory %d%% / visual %d%% / vestibular %d%%, %d%% of limit = near "
                          "the edge, backward limit < %d%% of forward."
                          % (RATIO_LOW["somatosensory"], RATIO_LOW["visual"], RATIO_LOW["vestibular"],
                             LOS_USED_HIGH, 100 * LOS_BACK_RATIO))


# ------------------------------------------------------------------ report
def report_text(res):
    out = [f"BALANCE BAND — {res.get('site', 'waist')} IMU, standing balance"]
    if res.get("axes"):
        out.append(f"Band orientation: {res['axes']['method']}")
    out.append("")
    L = res["los"]
    if L:
        out.append("Step 1 — Limits of stability (how far the COG can lean from centre without stepping):")
        if "limits_deg" in L:
            out.append("   " + "   ".join(f"{d} {L['limits_deg'][d]:.1f}°" for d in DIRS))
        out += [f"   ({n})" for n in L["notes"]]
        out.append("")
    out.append(f"{'Condition':<19}{'status':<12}{'AP 90%':>7}{'ML 90%':>7}{'sway':>6}{'x ref':>7}"
               f"{'% limit':>9}   notes")
    base = res["trials"]["eo_firm"].get("sway_deg")
    for c in CONDITIONS:
        t = res["trials"][c]
        f = lambda k: f"{t[k]:.1f}" if k in t else "–"
        ratio = f"{t['sway_deg']/base:.1f}" if (base and "sway_deg" in t) else "–"
        used = f"{t['los_used_pct']}% {t['los_used_dir'][:1].upper()}" if "los_used_pct" in t else "–"
        st = t["status"] + (f"@{t['fall_at_s']}s" if t["status"] == "fell" else "")
        out.append(f"{LABEL[c]:<19}{st:<12}{f('ap_range90_deg'):>7}{f('ml_range90_deg'):>7}{f('sway_deg'):>6}"
                   f"{ratio:>7}{used:>9}   {'; '.join(t['notes'])}")
    out.append("  (degrees; 90% = 5th–95th percentile; sway = combined AP/ML; x ref = vs eyes open firm;")
    out.append("   % limit = furthest excursion as % of that direction's limit of stability, F/B/L/R)")
    out += ["", f"OVERALL: {res['overall']}"] + [f"  • {x}" for x in res["findings"]]
    S = res.get("sensory")
    if S:
        out += ["", "Sensory ratios — how well balance holds when relying mainly on one sense",
                "(100% = as steady as eyes open on a firm floor):"]
        if S["scores"]:
            out.append("   Stability score /100:  " + "   ".join(
                f"{LABEL[c]} {S['scores'][c]:.0f}" + (" (fell)" if res["trials"][c]["status"] == "fell" else "")
                for c in CONDITIONS if c in S["scores"]))
        for sense, c, how in SENSORY:
            if sense in S["ratios"]:
                flag = "   <- LOW" if sense in S["low"] else ""
                out.append(f"   {sense.capitalize():<14}{S['ratios'][sense]:>4}%   ({how} vs eyes open, firm;"
                           f" provisional cut-off {RATIO_LOW[sense]:.0f}%){flag}")
        out += [f"   ({n})" for n in S["notes"]]
        if S["ratios"]:
            out.append("   Focus for rehabilitation: " + (", ".join(x.upper() for x in S["low"]) if S["low"]
                       else "no single weak sense on these provisional cut-offs"))
            out.append("   (The three are separate abilities and do not add up to 100%. Vestibular is naturally the"
                       " lowest even in healthy people, hence its lower cut-off.)")
    if res["caveats"]:
        out += ["", "Caveats:"] + [f"  - {x}" for x in res["caveats"]]
    return "\n".join(out)


# ------------------------------------------------------------------ synthetic demo sessions
# per condition: (AP amp, ML amp[, fall time[, ML lean]]) in deg;  "los": F, B, L, R limits
DEMO = {
    "normal":             {"los": (7, 5, 5, 5), "eo_firm": (0.6, 0.4), "ec_firm": (0.8, 0.5),
                           "eo_foam": (0.9, 0.7), "ec_foam": (1.6, 1.1)},
    "weak_vestibular":    {"los": (7, 5, 5, 5), "eo_firm": (0.6, 0.4), "ec_firm": (0.8, 0.5),
                           "eo_foam": (0.9, 0.7), "ec_foam": (3.5, 3.0, 8.0)},
    "weak_somatosensory": {"los": (7, 5, 5, 5), "eo_firm": (0.6, 0.4), "ec_firm": (3.2, 2.6),
                           "eo_foam": (1.0, 0.8), "ec_foam": (1.8, 1.3)},
    "weak_visual":        {"los": (7, 5, 5, 5), "eo_firm": (0.6, 0.4), "ec_firm": (0.9, 0.6),
                           "eo_foam": (3.8, 3.2), "ec_foam": (1.8, 1.3)},
    "backward":           {"los": (5, 1.5, 3.5, 3.5), "eo_firm": (0.7, 0.5), "ec_firm": (1.0, 0.7),
                           "eo_foam": (1.1, 0.8), "ec_foam": (2.2, 1.6, 11.0)},
}

def _smooth_noise(n, fs, rng, fc=0.6):
    x = rng.standard_normal(n + int(4 * fs)); a = math.exp(-2 * math.pi * fc / fs)
    for _ in range(2):
        y = np.empty_like(x); y[0] = x[0]
        for i in range(1, len(x)): y[i] = a * y[i-1] + (1 - a) * x[i]
        x = y
    x = x[-n:]; return x / (x.std() + 1e-12)

def _write(path, t, ap, ml, fall, rng, phase=None, mount=None):
    apr, mlr = np.radians(ap), np.radians(ml)
    aF, aL = -np.sin(apr), -np.sin(mlr); aU = np.sqrt(np.clip(1 - aF**2 - aL**2, 0, 1))
    acc = np.stack([aF, aL, aU], 1) + 0.004 * rng.standard_normal((len(t), 3))
    gyr = np.stack([-np.gradient(ml, t), np.gradient(ap, t), np.zeros(len(t))], 1)
    gyr += 0.3 * rng.standard_normal((len(t), 3)) + np.array([0.4, -0.3, 0.1])     # noise + gyro bias
    if mount is not None:          # rows of `mount` = body F, L, U in sensor coords -> express in the sensor frame
        acc, gyr = acc @ mount, gyr @ mount
    head = ["t", "ax", "ay", "az", "gx", "gy", "gz", "fall"] + (["phase"] if phase is not None else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(head)
        for i in range(len(t)):
            w.writerow([f"{t[i]:.3f}", *(f"{v:.5f}" for v in acc[i]), *(f"{v:.4f}" for v in gyr[i]), fall[i]]
                       + ([phase[i]] if phase is not None else []))

def random_mount(rng):
    """A random band orientation (a proper rotation), as if strapped on any way round."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))

def write_demo(pattern, folder, fs=100.0, dur=20.0, seed=1, tilted=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); spec = DEMO[pattern]
    mount = random_mount(np.random.default_rng(seed + 100)) if tilted else None
    # limits of stability, following LOS_SCRIPT: lean out over 2 s and hold; come back during the next centre
    lim = dict(zip(DIRS, spec["los"]))
    vec = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1)}
    n = int(sum(d for _, d in LOS_SCRIPT) * fs); t = np.arange(n) / fs
    ap = 0.1 * _smooth_noise(n, fs, rng); ml = 0.1 * _smooth_noise(n, fs, rng)
    phase = np.empty(n, dtype=object); s = 0.0
    for name, d in LOS_SCRIPT:
        phase[(t >= s) & (t < s + d)] = name
        if name in vec:
            x = np.clip((t - s) / 2, 0, 1) - np.clip((t - s - d) / 2, 0, 1)
            prof = 0.5 - 0.5 * np.cos(np.pi * x)
            ap += lim[name] * vec[name][0] * prof; ml += lim[name] * vec[name][1] * prof
        s += d
    _write(folder / "los.csv", t, ap, ml, np.zeros(n, int), rng, phase=phase, mount=mount)
    for c in CONDITIONS:
        s = spec[c]; amp_ap, amp_ml = s[0], s[1]
        fall_t = s[2] if len(s) > 2 else None; lean = s[3] if len(s) > 3 else 0.0
        n = int(dur * fs); t = np.arange(n) / fs
        ap = amp_ap / 3.3 * _smooth_noise(n, fs, rng)
        ml = amp_ml / 3.3 * _smooth_noise(n, fs, rng) + lean
        fall = np.zeros(n, int)
        if fall_t:
            k = int(fall_t * fs); ramp = np.clip((t - t[k] + 0.6) / 0.6, 0, None) ** 2
            if pattern == "backward":
                ap = ap - 6.0 * ramp               # falls BACKWARD
            else:
                ml = ml - 6.0 * ramp               # falls to the right
            fall[k:] = 1
        _write(folder / f"{c}.csv", t, ap, ml, fall, rng, mount=mount)
    return folder


def main(argv=None):
    p = argparse.ArgumentParser(description="Balance-band standing-balance analysis")
    p.add_argument("folder", help="session folder: los.csv, eo_firm.csv, ec_firm.csv, eo_foam.csv, ec_foam.csv")
    p.add_argument("--forward", help="sensor axis pointing to the person's front (e.g. +x, -z); "
                                     "omit to find it automatically from los.csv")
    p.add_argument("--left", help="sensor axis pointing to the person's LEFT; omit for automatic")
    p.add_argument("--site", default="waist", choices=SITES,
                   help="where the band is worn: waist (lower back, recommended) or shin")
    p.add_argument("--gyro-units", default="deg", choices=["deg", "rad"])
    p.add_argument("--demo", choices=sorted(DEMO), help="first write a synthetic session of this pattern")
    p.add_argument("--demo-tilted", action="store_true", help="demo: band strapped on at a random orientation")
    p.add_argument("--json", help="also write the full result to this JSON file")
    a = p.parse_args(argv)
    if a.demo:
        write_demo(a.demo, a.folder, tilted=a.demo_tilted)
    res = analyse_session(a.folder, a.forward, a.left, a.gyro_units == "rad", a.site)
    print(report_text(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
