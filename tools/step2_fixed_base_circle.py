"""STEP 2 — Fixed Base Circle From Sclera-Bounded Iris  (fresh; does NOT use the old almond method).

Builds directly on the LOCKED Step 1 (checkpoint/ellseg-anchor): EllSeg gives the approximate iris
location only. Step 2 then:

  1. In/around the EllSeg-anchored region, take the dark iris blob (the dark component that overlaps the
     EllSeg disc most — EllSeg tells us WHERE).
  2. Find its LIMBUS points = the dark-iris edge that ABUTS relatively uniform white/pink sclera
     (especially the left/right = medial/lateral margins).
  3. CALIBRATE a base radius R from the clearest, largest FULL-circle evidence across the whole clip
     (free-radius circle fits on frames with wide angular limbus coverage + low residual). Then LOCK R.
  4. Per frame, fit a circle of FIXED radius R (fit the CENTRE only) to the limbus points.
     -> iris size stays constant; no frame-to-frame radius shimmer from EllSeg.

Deliberately NOT handled yet (later steps): oval foreshortening at side gaze, eyelid clipping, canthus
exclusion, sector shape, torsion/texture. This step only proves a STABLE fixed-size circle.

Outputs (in outputs/<clip>_tracked/step2_fixed_circle/):
  step2_fixed_circle.mp4   overlay: limbus points + fixed-R circle + centre, both eyes
  step2_diag.png           per-frame 4 panels: original / EllSeg anchor / limbus points / fitted circle
  step2_radius_stability.png  free per-frame radius (jittery) vs the LOCKED flat R  (the pass/fail plot)
  step2_centre_trace.png   fixed-circle centre (cx) across the clip, both eyes
  step2_points.csv         frame, eye, cx, cy, R, n_limbus, coverage_deg, free_r

Run with the rit-nets venv python.
"""
import sys, os, json, math, csv
from pathlib import Path
import numpy as np, cv2

ELL = Path(os.environ.get("ELLSEG_DIR", Path.home() / "rit-nets" / "EllSeg"))
sys.path.insert(0, str(ELL))
for _a, _t in (("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import torch
from modelSummary import model_dict

REPO = Path(__file__).resolve().parent.parent
CLIP = os.environ.get("RIT_CLIP", "nystagmus at rest to left in right vestibular neuritis")
VIDEO = REPO / "samples" / f"{CLIP}.mp4"
OUTBASE = REPO / "outputs" / f"{CLIP}_tracked"
GT = OUTBASE / "RIT_ground_truth"
OUTDIR = OUTBASE / "step2_fixed_circle"
OUTDIR.mkdir(parents=True, exist_ok=True)
OPH, OPW = 240, 320
DIAG_EYE = "R"
DIAG_FRAMES = [40, 100, 160, 220, 235, 286, 335, 400, 470, 610]   # spread across the clip (+235,286 = side gaze)
COV_GOOD = 150      # deg of limbus arc that counts as strong "full-circle" evidence for calibration
RES_GOOD = 0.12     # max relative fit residual (residual / r) for calibration frames
SUPPORT_MIN = 0.45  # SIDE-GAZE RESCUE GATE cond.1b: min fraction of the fixed circle overlapping the EllSeg DISC
ANCHOR_MAX = 0.60   # cond.1a: max centre-to-EllSeg-anchor distance, as a fraction of R (drift off-iris fails)

model = model_dict["ritnet_v3"]
nd = torch.load(ELL / "weights" / "all.git_ok", map_location="cpu", weights_only=False)
model.load_state_dict(nd["state_dict"], strict=True); model.eval()

def preprocess(gray):
    h, w = gray.shape; sc = OPW / w
    nw, nh = int(round(w*sc)), int(round(h*sc))
    img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    if OPH >= nh:
        pad = OPH-nh; top = pad//2; img = np.pad(img, ((top, pad-top), (0, 0))); valid = (top, nh)
    else:
        top = (nh-OPH)//2; img = img[top:top+OPH, :]; valid = (0, OPH)
    imgn = (img-img.mean())/(img.std()+1e-6)
    return torch.from_numpy(imgn).unsqueeze(0).unsqueeze(0).to(torch.float32), valid

def seg_forward(x):
    with torch.no_grad():
        x4, x3, x2, x1, xc = model.enc(x); seg = model.dec(x4, x3, x2, x1, xc)
    return seg.max(1)[1][0].numpy().astype(np.uint8)

def ellseg_region(gray_full, cx, cy, hw, hh, W, H):
    """LOCKED Step 1: EllSeg anchor. Returns crop window + iris/pupil disc mask (crop coords)."""
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    if disc.sum() < 30: return None
    m = cv2.moments(disc)
    return dict(win=(x0, y0, x1, y1), disc=disc, cx=m["m10"]/m["m00"], cy=m["m01"]/m["m00"])

# ---- Step-2 geometry helpers (generic; independent of the old method) ----
SECTOR = 55   # deg: medial/lateral limbus = edge points within +-SECTOR of horizontal (left & right)

def sclera_and_dark(bgr, gray):
    """TRUE white/pink sclera vs skin. Sclera = BRIGHT and LOW-SATURATION (whitish); brown/orange skin is
    bright but HIGH-saturation -> rejected. Adaptive: split saturation of the bright pixels by Otsu, keep the
    low-sat (white) side. Then drop speckle components but KEEP both scleral lobes (iris splits sclera in two).
    """
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); S = hsv[:, :, 1]
    bright = gray >= thr
    bs = S[bright]
    if bs.size >= 30:
        st, _ = cv2.threshold(bs.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        st = float(np.clip(st, 25, 90))                # white sclera stays desaturated; skin is more saturated
    else:
        st = 60.0
    sclera = (bright & (S <= st)).astype(np.uint8)      # TRUE-LIMBUS RULE: bright + desaturated = real sclera
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sclera, 8)
    if n > 1:
        tot = int(sclera.sum()); keep = np.zeros_like(sclera)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= max(40, 0.05 * tot): keep[lab == i] = 1   # keep both lobes, drop specks
        sclera = keep
    dark = (gray < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # strip thin lashes
    return thr, sclera, dark

def iris_blob_at_anchor(dark, disc):
    """The dark component that overlaps the EllSeg disc most = the iris (EllSeg anchor picks the blob)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    best, bov = None, 0
    for i in range(1, n):
        comp = (lab == i)
        ov = int((comp & (disc > 0)).sum())
        if ov > bov: bov, best = ov, comp
    if best is None or bov < 10: return None
    return best.astype(np.uint8)

MINARC = 7   # a real limbus arc spans at least this many contour vertices

def _contiguous_runs(mask):
    """Runs of True on a CYCLIC boolean -> list of index-arrays (original contour indices)."""
    n = len(mask)
    if not mask.any(): return []
    if mask.all(): return [np.arange(n)]
    start = int(np.where(~mask)[0][0]); order = (np.arange(n) + start) % n
    m = mask[order]; runs = []; i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            runs.append(order[i:j]); i = j
        else:
            i += 1
    return runs

def limbus_arcs(iris_blob, sclera, dcx, dcy, R_hint):
    """MEDIAL/LATERAL LIMBUS ARC + PARTIAL ARC COMPLETION RULES. Walk the iris-blob contour; keep only
    vertices that (a) abut TRUE sclera and (b) lie in the medial/lateral sectors. Group into contiguous
    arcs; drop short/scattered runs (skin/lash noise never forms a long smooth arc). Score each arc by
    length x sclera-contact x convexity x radius-consistency, keep the best on each side. ONE clean arc is
    enough. Returns (pts, left_xs, right_xs) where pts are the winning arc vertices."""
    cnts, _ = cv2.findContours(iris_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        e = np.empty((0, 2)); return e, np.empty(0), np.empty(0)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    if len(c) < MINARC:
        e = np.empty((0, 2)); return e, np.empty(0), np.empty(0)
    sclera_d = cv2.dilate(sclera, np.ones((5, 5), np.uint8))
    Hc, Wc = iris_blob.shape
    xs, ys = c[:, 0], c[:, 1]
    ang = np.degrees(np.arctan2(ys - dcy, xs - dcx))
    ml = (np.abs(ang) <= SECTOR) | (np.abs(np.abs(ang) - 180) <= SECTOR)   # medial/lateral only
    touch = np.zeros(len(c), bool)                                          # abuts TRUE sclera?
    for k, (x, y) in enumerate(c):
        if not ml[k]: continue
        y0 = max(0, y-2); x0 = max(0, x-2)
        touch[k] = bool(sclera_d[y0:y+3, x0:x+3].any())
    good = ml & touch
    runs = [r for r in _contiguous_runs(good) if len(r) >= MINARC]
    if not runs:
        e = np.empty((0, 2)); return e, np.empty(0), np.empty(0)
    best_left = best_right = None; bl = br = 0.0
    for r in runs:
        p = c[r].astype(np.float64)
        mang = np.degrees(np.arctan2(p[:, 1].mean() - dcy, p[:, 0].mean() - dcx))
        side = "R" if abs(mang) <= 90 else "L"
        # radius-consistency: how close the arc sits to R_hint from the anchor centre
        rr = np.hypot(p[:, 0] - dcx, p[:, 1] - dcy)
        rad_ok = float(np.mean(np.abs(rr - R_hint) < max(6.0, 0.30 * R_hint)))
        score = len(r) * (0.5 + rad_ok)
        if side == "R" and score > br: br, best_right = score, p
        if side == "L" and score > bl: bl, best_left = score, p
    keep = [a for a in (best_left, best_right) if a is not None]
    pts = np.vstack(keep) if keep else np.empty((0, 2))
    lx = best_left[:, 0] if best_left is not None else np.empty(0)
    rx = best_right[:, 0] if best_right is not None else np.empty(0)
    return pts, lx, rx

def fit_circle_free(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones(len(x))]); b = x*x + y*y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = sol; r = math.sqrt(max(1e-6, c + a*a + bb*bb))
    d = np.hypot(x - a, y - bb); res = float(np.mean(np.abs(d - r)))
    return float(a), float(bb), float(r), res

def coverage_deg(pts, cx, cy):
    ang = (np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360)
    return len(set((ang // 10).astype(int).tolist())) * 10

def fit_centre_fixed(pts, R, c0):
    """Fit ONLY the centre of a circle of KNOWN radius R (2 DOF) -> stable centre from a partial arc."""
    c = np.asarray(c0, float).copy()
    for _ in range(8):
        d = c - pts; nn = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        votes = pts + R * (d / nn[:, None])
        inl = np.abs(nn - R) < max(3.0, 0.20 * R)
        c = votes[inl].mean(0) if int(inl.sum()) >= 6 else votes.mean(0)
    return c

# ---- seeds from the orbit-lock ground truth (same as Step 1) ----
meta = json.load(open(GT / "meta.json", encoding="utf-8"))
def src_pts(e):
    X0, Y0, s = e["X0"], e["Y0"], e["scale"]; return np.array([[x/s+X0, y/s+Y0] for x, y in e["orbit_crop"]], np.float32)
eyes = {}
for ek in ("L", "R"):
    ents = sorted([v for v in meta.values() if v.get("eye") == ek and "orbit_crop" in v], key=lambda v: v["frame"])
    if not ents: continue
    ws, hs = [], []
    for v in ents:
        (cx, cy), (a1, a2), ang = cv2.fitEllipse(src_pts(v)); ws.append(max(a1, a2)); hs.append(min(a1, a2))
    (cx0, cy0), _, _ = cv2.fitEllipse(src_pts(ents[0]))
    eyes[ek] = dict(ow=float(np.median(ws)), oh=float(np.median(hs)), seed=(cx0, cy0))
    print(f"{ek}: opening ~{eyes[ek]['ow']:.0f}x{eyes[ek]['oh']:.0f}px seed=({cx0:.0f},{cy0:.0f})")

# ================= PASS 1 — EllSeg once; collect limbus geometry + calibrate R =================
cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); FPS = cap.get(5) or 30.0
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
cache = {}                     # (fi,ek) -> dict(win, pts, seed=(cx,cy), free_r, cov, res, width)
diag_raw = {}                  # fi -> render materials for the DIAG_EYE
widths = {ek: [] for ek in eyes}     # medial-lateral iris WIDTH (both sides present) -> base radius
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek, e in eyes.items():
        st = state[ek]; grow = 1.0 + min(st["lost"], 6) * 0.25
        reg = ellseg_region(gray, st["cx"], st["cy"], e["ow"]*0.6*grow, e["oh"]*0.6*grow, W, H)
        if reg is None: st["lost"] += 1; continue
        x0, y0, x1, y1 = reg["win"]
        bgr_c = frame[y0:y1, x0:x1]; g_c = gray[y0:y1, x0:x1]
        thr, sclera, dark = sclera_and_dark(bgr_c, g_c)
        blob = iris_blob_at_anchor(dark, reg["disc"])
        seed_full = (reg["cx"] + x0, reg["cy"] + y0)
        if blob is None:
            st.update(cx=seed_full[0], cy=seed_full[1], lost=st["lost"]+1); continue
        r_hint = max(20.0, math.sqrt(int(reg["disc"].sum()) / math.pi))   # EllSeg's own size = soft arc-radius hint
        pts, lx, rx = limbus_arcs(blob, sclera, reg["cx"], reg["cy"], r_hint)
        width = None
        if len(lx) >= 3 and len(rx) >= 3:                     # BOTH medial & lateral limbus present (frontal-ish)
            width = float(np.median(rx) - np.median(lx))       # iris WIDTH = medial-lateral span at the anchor
            if 12 <= width <= 320: widths[ek].append(width)
        # TRACKING follows the reliable EllSeg centroid (locked Step 1), NOT the circle fit -- the sparse-arc
        # free fit is unstable and would let the window climb onto the brow. EllSeg locates; anatomy refines.
        st.update(cx=seed_full[0], cy=seed_full[1], lost=0)
        if len(pts) >= 6:
            a, b, fr, res = fit_circle_free(pts)
            cov = coverage_deg(pts, a, b)
            dys, dxs = np.where(reg["disc"])                   # store EllSeg disc compactly (bbox submask) for the gate
            dbb = (int(dxs.min()), int(dys.min()), int(dxs.max())+1, int(dys.max())+1)
            dm = reg["disc"][dbb[1]:dbb[3], dbb[0]:dbb[2]].copy()
            # STEP 2c: "how flat" from EllSeg disc ASPECT; "which way tilted" from ANATOMY (iris - sclera).
            asp = 1.0                                              # disc aspect = minor/major of the EllSeg disc
            dcs, _ = cv2.findContours(reg["disc"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if dcs:
                dcm = max(dcs, key=cv2.contourArea)
                if len(dcm) >= 5:
                    (_, _), (da1, da2), _ = cv2.fitEllipse(dcm)
                    asp = min(da1, da2) / max(da1, da2, 1e-6)
            Ms = cv2.moments(sclera)                               # anatomical gaze DIRECTION: sclera centroid -> iris
            scx, scy = (Ms["m10"]/Ms["m00"], Ms["m01"]/Ms["m00"]) if Ms["m00"] > 0 else (reg["cx"], reg["cy"])
            gdir = (reg["cx"] - scx, reg["cy"] - scy)
            cache[(fi, ek)] = dict(win=(x0, y0, x1, y1), pts=pts, seed=(reg["cx"], reg["cy"]),
                                   free_r=fr, cov=cov, res=res, width=width, disc_bb=dbb, disc_m=dm,
                                   aspect=asp, gdir=gdir)
        if ek == DIAG_EYE and fi in DIAG_FRAMES and (fi, ek) in cache:
            diag_raw[fi] = dict(crop=bgr_c.copy(), disc=reg["disc"].copy(), sclera=sclera.copy(),
                                dark=dark.copy(), seed=(reg["cx"], reg["cy"]), pts=cache[(fi, ek)]["pts"].copy())
cap.release()

# CALIBRATE base R from the medial-lateral WIDTH. Frontal frames show the FULL width (largest); side gaze
# foreshortens it. Take a high percentile of the clean widths -> the un-foreshortened iris diameter -> R.
R = {}
for ek in eyes:
    arr = np.array(widths[ek])
    if arr.size >= 5:
        diam = float(np.percentile(arr, 80)); R[ek] = diam / 2.0
        print(f"[calibrate] {ek}: base R = {R[ek]:.1f}px  (medial-lateral width p80={diam:.0f} from "
              f"{arr.size} two-sided frames; width range {arr.min():.0f}-{arr.max():.0f})")
    else:
        frs = np.array([c["free_r"] for (f, e2), c in cache.items() if e2 == ek])
        R[ek] = float(np.median(frs)) if frs.size else 40.0
        print(f"[calibrate] {ek}: base R = {R[ek]:.1f}px  (FALLBACK median free-r; only {arr.size} two-sided frames)")

# STEP 2c — temporally SMOOTH the disc-aspect ("how flat") and the anatomical gaze direction ("which way"),
# per eye, to kill EllSeg's frame-to-frame jitter while keeping the gaze-driven trend.
CTH_FLOOR = 0.22        # flattest oval we allow
DIR_CONF = 0.22         # |gdir| (as fraction of R) needed to trust the tilt direction when the oval is flat
def _roll_med(vals, w=9):
    n = len(vals); out = np.array(vals, float); h = w // 2
    for i in range(n): out[i] = np.median(vals[max(0, i-h):i+h+1])
    return out
for ek in eyes:
    fs = sorted(f for (f, e2) in cache if e2 == ek)
    if not fs: continue
    asp = _roll_med(np.array([cache[(f, ek)]["aspect"] for f in fs]))
    gx = _roll_med(np.array([cache[(f, ek)]["gdir"][0] for f in fs]))
    gy = _roll_med(np.array([cache[(f, ek)]["gdir"][1] for f in fs]))
    for k, f in enumerate(fs):
        cache[(f, ek)]["cth"] = float(np.clip(asp[k], CTH_FLOOR, 1.0))
        cache[(f, ek)]["gdir_s"] = (float(gx[k]), float(gy[k]))
    print(f"[oval] {ek}: disc-aspect cth {asp.min():.2f}-{asp.max():.2f}; "
          f"|gdir| median {np.median(np.hypot(gx, gy)):.0f}px (R={R[ek]:.0f})")

def make_oval(cth, gdir, Rk):
    """major = 2R (fixed). minor = 2R*cos th, cos th = smoothed EllSeg disc aspect (how flat). Long axis angle
    = anatomical gaze direction + 90 (which way tilted). Returns (major, minor, ang, cth, |gdir|)."""
    gn = float(math.hypot(gdir[0], gdir[1]))
    major = 2.0 * Rk; minor = major * cth
    ang = (math.degrees(math.atan2(gdir[1], gdir[0])) + 90.0) if gn > 2 else 0.0
    return major, minor, ang, cth, gn

# ================= PASS 2 — no EllSeg; fixed-R centre fit + SIDE-GAZE RESCUE GATE + render =================
def gate_support(disc, ctr, Rk):
    """SIDE-GAZE RESCUE GATE metric: fraction of the fixed-R circle that overlaps the EllSeg IRIS DISC.
    Uses the disc (true iris segmentation, no lash/shadow) not the Otsu-dark blob: at side gaze the disc is a
    sliver -> the big rigid circle overlaps little of it -> low support -> withhold. Captures both 'evidence
    too narrow' and 'circle mostly outside the true iris' in one number."""
    circ = np.zeros(disc.shape, np.uint8); cv2.circle(circ, (int(ctr[0]), int(ctr[1])), int(round(Rk)), 1, -1)
    a = int(circ.sum())
    return int(((circ > 0) & (disc > 0)).sum()) / max(1, a)

def disc_from_cache(c, shape):
    disc = np.zeros(shape, np.uint8); x0, y0, x1, y1 = c["disc_bb"]; disc[y0:y1, x0:x1] = c["disc_m"]; return disc

def oval_support(disc, ctr, major, minor, ang):
    """Fraction of the projected OVAL that overlaps the EllSeg iris disc (the Step-2c gate metric)."""
    ov = np.zeros(disc.shape, np.uint8)
    cv2.ellipse(ov, (int(ctr[0]), int(ctr[1])), (int(major/2), int(minor/2)), ang, 0, 360, 1, -1)
    a = int(ov.sum())
    return int(((ov > 0) & (disc > 0)).sum()) / max(1, a)

def gate_decision(disc, seed, pts, ctr, Rk):
    """TWO conditions, BOTH required for fixed_circle_ok (Dr. K, 2026-07-04):
      (1a) circle centre stays near the EllSeg anchor  AND  (1b) circle overlaps the EllSeg iris disc enough;
      (2)  a valid medial/lateral sclera-facing limbus arc exists.
    Measured against EllSeg-disc + limbus evidence, NOT generic Otsu dark (lid/skin shadow must not pass)."""
    support = gate_support(disc, ctr, Rk)
    dist = float(np.hypot(ctr[0] - seed[0], ctr[1] - seed[1]))
    cond_anchor = dist <= ANCHOR_MAX * Rk
    cond_support = support >= SUPPORT_MIN
    cond_arc = len(pts) >= MINARC
    ok = cond_anchor and cond_support and cond_arc
    return ok, support, dist

cap = cv2.VideoCapture(str(VIDEO))
vw = cv2.VideoWriter(str(OUTDIR / "step2_fixed_circle.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
trace = []; last_good = {ek: None for ek in eyes}; status_diag = {}; rescued_info = []
counts = {ek: {"circle": 0, "oval": 0, "rescue": 0} for ek in eyes}
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek in eyes:
        c = cache.get((fi, ek))
        if c is None: continue
        x0, y0, x1, y1 = c["win"]; pts = c["pts"]; Rk = R[ek]
        disc = disc_from_cache(c, (y1-y0, x1-x0))
        ctr = fit_centre_fixed(pts, Rk, np.array(c["seed"], float))
        major, minor, ang, cth, gn = make_oval(c["cth"], c["gdir_s"], Rk)   # STEP 2c gaze-oval
        support = oval_support(disc, ctr, major, minor, ang)           # gate measured on the OVAL vs EllSeg disc
        dist = float(np.hypot(ctr[0]-c["seed"][0], ctr[1]-c["seed"][1]))
        dir_ok = (cth >= 0.85) or (gn >= DIR_CONF * Rk)                # flat oval needs a confident tilt direction
        ok_frame = (len(pts) >= MINARC) and (dist <= ANCHOR_MAX*Rk) and (support >= SUPPORT_MIN) and dir_ok
        for px, py in pts.astype(int):                                   # limbus arc points (small yellow)
            cv2.circle(frame, (px + x0, py + y0), 1, (0, 220, 220), -1)
        if ok_frame:
            cx, cy = float(ctr[0] + x0), float(ctr[1] + y0)
            cv2.ellipse(frame, (int(cx), int(cy)), (int(major/2), int(minor/2)), ang, 0, 360, (0, 0, 255), 2)  # gaze-oval (red)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)             # centre (yellow)
            last_good[ek] = (cx, cy, major, minor, ang)
            near_circle = cth >= 0.92
            counts[ek]["circle" if near_circle else "oval"] += 1
            st_lab = "fixed_circle_ok" if near_circle else "projected_oval_ok"
            trace.append((fi, ek, round(cx, 2), round(cy, 2), round(Rk, 1), round(cth, 3), round(support, 2), st_lab))
        else:                                                            # withhold; carry forward last good oval
            counts[ek]["rescue"] += 1; st_lab = "needs_rescue"
            reasons = []
            if len(pts) < MINARC: reasons.append("no_valid_arc")
            if dist > ANCHOR_MAX * Rk: reasons.append("far_from_anchor")
            if support < SUPPORT_MIN: reasons.append("low_disc_support")
            if not dir_ok: reasons.append("ambiguous_gaze")
            rescued_info.append((fi, ek, support, dist, len(pts) >= MINARC, "+".join(reasons), (x0, y0, x1, y1)))
            if last_good[ek] is not None:
                cx, cy, mj, mn, an = last_good[ek]
                cv2.ellipse(frame, (int(cx), int(cy)), (int(mj/2), int(mn/2)), an, 0, 360, (150, 150, 150), 1)  # faint carried oval
                cv2.putText(frame, "needs_rescue", (x0, max(18, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                trace.append((fi, ek, round(cx, 2), round(cy, 2), round(Rk, 1), round(cth, 3), round(support, 2), st_lab))
            else:
                cv2.putText(frame, "needs_rescue", (x0, max(18, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if ek == DIAG_EYE and fi in DIAG_FRAMES:
            status_diag[fi] = (st_lab, support, dist, cth)
    cv2.putText(frame, f"f{fi}  STEP2c gaze-oval + rescue gate", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}  STEP2c gaze-oval + rescue gate", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    vw.write(frame)
cap.release(); vw.release()

with open(OUTDIR / "step2_points.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["frame", "eye", "cx", "cy", "R", "cos_theta", "support", "status"])
    w.writerows(trace)
print("gate:", {ek: counts[ek] for ek in eyes})
print(f"--- {DIAG_EYE}-eye status at diagnostic frames ---")
for f in DIAG_FRAMES:
    if f in status_diag:
        lab, sup, dist, cth = status_diag[f]
        print(f"  f{f}: {lab:18s} support={sup:.2f} dist={dist:.0f}px cos_th={cth:.2f} (minor/major)")

# ---- diagnostic montage (DIAG_EYE): original / EllSeg anchor / limbus points / fitted fixed-R circle ----
PW = 300
def fitw(im): h = int(PW*im.shape[0]/im.shape[1]); return cv2.resize(im, (PW, h))
rows = []
for f in DIAG_FRAMES:
    d = diag_raw.get(f)
    if d is None: continue
    crop, disc, sclera, pts = d["crop"], d["disc"], d["sclera"], d["pts"]
    p1 = crop.copy()
    p2 = crop.copy()
    cv2.drawContours(p2, cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
    cv2.circle(p2, (int(d["seed"][0]), int(d["seed"][1])), 4, (0, 255, 255), -1)
    p3 = crop.copy(); p3[sclera > 0] = (0.7*p3[sclera > 0] + np.array([255, 60, 0])*0.3).astype(np.uint8)
    for px, py in pts.astype(int): cv2.circle(p3, (px, py), 2, (0, 0, 255), -1)
    p4 = crop.copy()
    ek = DIAG_EYE; Rk = R[ek]; cc = cache[(f, ek)]
    ctr = fit_centre_fixed(pts, Rk, np.array(d["seed"], float))
    major, minor, ang, cth, gn = make_oval(cc["cth"], cc["gdir_s"], Rk)
    support = oval_support(d["disc"], ctr, major, minor, ang)
    dir_ok = (cth >= 0.85) or (gn >= DIR_CONF * Rk)
    okf = (len(pts) >= MINARC) and (float(np.hypot(ctr[0]-d["seed"][0], ctr[1]-d["seed"][1])) <= ANCHOR_MAX*Rk) and (support >= SUPPORT_MIN) and dir_ok
    cv2.circle(p4, (int(ctr[0]), int(ctr[1])), int(round(Rk)), (255, 120, 0), 1)   # OLD fixed circle (thin blue), compare
    gv = cc["gdir_s"]; gnn = math.hypot(*gv) + 1e-6                                 # anatomical gaze arrow (magenta)
    cv2.arrowedLine(p4, (int(ctr[0]), int(ctr[1])), (int(ctr[0]+gv[0]/gnn*40), int(ctr[1]+gv[1]/gnn*40)), (255, 0, 255), 2, tipLength=0.3)
    if okf:
        col = (0, 0, 255) if cth < 0.92 else (0, 200, 0)           # red oval / green near-circle
        cv2.ellipse(p4, (int(ctr[0]), int(ctr[1])), (int(major/2), int(minor/2)), ang, 0, 360, col, 2)   # NEW gaze-oval
        lab = "projected_oval_ok" if cth < 0.92 else "fixed_circle_ok"
    else:
        col = (0, 165, 255)
        cv2.ellipse(p4, (int(ctr[0]), int(ctr[1])), (int(major/2), int(minor/2)), ang, 0, 360, (150, 150, 150), 1)   # withheld -> faint
        lab = "needs_rescue"
    cv2.circle(p4, (int(ctr[0]), int(ctr[1])), 4, col, -1)
    txt = f"{lab} cth={cth:.2f} s={support:.2f}"
    cv2.putText(p4, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(p4, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    panels = [fitw(p) for p in (p1, p2, p3, p4)]
    hmax = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, hmax-p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
    row = np.hstack([cv2.copyMakeBorder(p, 0, 0, 0, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels])
    cv2.putText(row, f"f{f}", (6, hmax-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(row, f"f{f}", (6, hmax-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    rows.append(row)
if rows:
    head = np.full((30, rows[0].shape[1], 3), 20, np.uint8)
    for i, lab in enumerate(["ORIGINAL", "EllSeg ANCHOR", "TRUE LIMBUS (med/lat, sclera)", "GAZE-OVAL vs circle"]):
        cv2.putText(head, lab, (i*(PW+3)+6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    w = max(r.shape[1] for r in rows); rr = [cv2.copyMakeBorder(r, 0, 0, 0, w-r.shape[1], cv2.BORDER_CONSTANT) for r in ([head]+rows)]
    cv2.imwrite(str(OUTDIR / "step2_diag.png"), np.vstack(rr))

# ---- plots: radius stability (pass/fail) + centre trace ----
def plot(series, colors, hline, title, ylab, path, ylo=None, yhi=None, PWc=1500, Hc=340):
    canvas = np.full((Hc, PWc, 3), 255, np.uint8)
    L, Rm, T, Bm = 60, 20, 30, 30
    allv = [v for s in series.values() for (_, v) in s if v is not None] + ([hline] if hline else [])
    if not allv: cv2.imwrite(path, canvas); return
    ylo = min(allv) if ylo is None else ylo; yhi = max(allv) if yhi is None else yhi
    if yhi - ylo < 1e-6: yhi = ylo + 1
    def X(f): return int(L + (Rm and 0) + (PWc-L-Rm) * f / max(1, NF-1))
    def Y(v): return int(T + (Hc-T-Bm) * (1 - (v-ylo)/(yhi-ylo)))
    cv2.rectangle(canvas, (L, T), (PWc-Rm, Hc-Bm), (210, 210, 210), 1)
    if hline is not None:
        cv2.line(canvas, (L, Y(hline)), (PWc-Rm, Y(hline)), (0, 0, 0), 1)
        cv2.putText(canvas, f"locked R={hline:.0f}", (PWc-Rm-150, Y(hline)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for ek, s in series.items():
        col = colors[ek]; prev = None
        for f, v in s:
            if v is None: prev = None; continue
            p = (X(f), Y(v))
            if prev is not None: cv2.line(canvas, prev, p, col, 1)
            prev = p
    cv2.putText(canvas, title, (L, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, f"{ylab}  [{ylo:.0f}..{yhi:.0f}]", (L, Hc-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    for i, ek in enumerate(series): cv2.putText(canvas, ek, (PWc-Rm-40, 20+18*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[ek], 2)
    cv2.imwrite(path, canvas)

COL = {"L": (200, 80, 0), "R": (0, 80, 220)}
free_series = {ek: [(f, cache[(f, ek)]["free_r"]) if (f, ek) in cache else (f, None) for f in range(NF)] for ek in eyes}
cx_series = {ek: [] for ek in eyes}
bykey = {(f, ek): (cx, cy) for (f, ek, cx, cy, *_ ) in trace}
for ek in eyes:
    cx_series[ek] = [(f, bykey[(f, ek)][0]) if (f, ek) in bykey else (f, None) for f in range(NF)]
plot(free_series, COL, hline=np.mean([R[ek] for ek in eyes]),
     title="STEP 2 radius stability: per-frame FREE fit (jittery) vs LOCKED R (flat) -> should NOT track the jitter",
     ylab="radius px", path=str(OUTDIR / "step2_radius_stability.png"), ylo=0)
plot(cx_series, COL, hline=None,
     title="STEP 2 fixed-circle centre: horizontal position cx across the clip (both eyes)",
     ylab="cx px", path=str(OUTDIR / "step2_centre_trace.png"))

# ---- RESCUE REVIEW SHEET: representative needs_rescue frames (early/mid/late, both eyes) ----
from collections import Counter
rc = {ek: Counter() for ek in eyes}
for (f, ek, sup, dist, ha, reason, win) in rescued_info: rc[ek][reason or "(none)"] += 1
print("rescue reasons:", {ek: dict(rc[ek]) for ek in eyes})
print("no-evidence frames (no arc / EllSeg miss, never drawn):",
      {ek: NF - sum(1 for (f, e2) in cache if e2 == ek) for ek in eyes})

thirds = [("early", 0, NF//3), ("mid", NF//3, 2*NF//3), ("late", 2*NF//3, NF)]
sel = []
for ek in eyes:
    for nm, lo, hi in thirds:
        cand = [r for r in rescued_info if r[1] == ek and lo <= r[0] < hi]
        if not cand: continue
        for i in np.linspace(0, len(cand)-1, min(2, len(cand))).astype(int):
            sel.append((nm,) + cand[i])
need = sorted(set(s[1] for s in sel)); frames_r = {}
capr = cv2.VideoCapture(str(VIDEO))
for fi_ in need:
    capr.set(cv2.CAP_PROP_POS_FRAMES, fi_); okr, frr = capr.read()
    if okr: frames_r[fi_] = frr
capr.release()

def _panel(nm, fi_, ek, sup, dist, ha, reason, win):
    x0, y0, x1, y1 = win; crop = frames_r[fi_][y0:y1, x0:x1].copy()
    c = cache.get((fi_, ek))
    if c is not None:
        disc = disc_from_cache(c, (y1-y0, x1-x0))
        cv2.drawContours(crop, cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 200, 0), 2)
        pts = c["pts"]
        for px, py in pts.astype(int): cv2.circle(crop, (px, py), 2, (0, 220, 220), -1)
        ctr = fit_centre_fixed(pts, R[ek], np.array(c["seed"], float))
        mj, mn, an, _, _ = make_oval(c["cth"], c["gdir_s"], R[ek])
        cv2.ellipse(crop, (int(ctr[0]), int(ctr[1])), (int(mj/2), int(mn/2)), an, 0, 360, (150, 150, 150), 1)   # rejected oval (faint)
    crop = fitw(crop)
    bar = np.full((56, crop.shape[1], 3), 25, np.uint8)
    for i, t in enumerate([f"f{fi_} {ek} [{nm}]", f"s={sup:.2f} d={dist:.0f} arc={'Y' if ha else 'N'}", reason or "(gate)"]):
        cv2.putText(bar, t, (4, 16+i*17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255) if i == 2 else (255, 255, 255), 1)
    return np.vstack([crop, bar])

if sel:
    panels = [_panel(*s) for s in sel if s[1] in frames_r]
    hmax = max(p.shape[0] for p in panels); wmax = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, hmax-p.shape[0], 0, wmax-p.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels]
    rowsr = [np.hstack(panels[i:i+4]) for i in range(0, len(panels), 4)]
    wmx = max(r.shape[1] for r in rowsr)
    rowsr = [cv2.copyMakeBorder(r, 0, 0, 0, wmx-r.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for r in rowsr]
    cv2.imwrite(str(OUTDIR / "step2_rescue_review.png"), np.vstack(rowsr))
    print("rescue review ->", OUTDIR / "step2_rescue_review.png")

print("painted frames:", {ek: sum(1 for (f, e2) in cache if e2 == ek) for ek in eyes}, "of", NF)
print("R locked:", {k: round(v, 1) for k, v in R.items()})
print("video ->", OUTDIR / "step2_fixed_circle.mp4")
print("diag  ->", OUTDIR / "step2_diag.png")
print("plots ->", OUTDIR / "step2_radius_stability.png", "&", OUTDIR / "step2_centre_trace.png")
print("csv   ->", OUTDIR / "step2_points.csv")
