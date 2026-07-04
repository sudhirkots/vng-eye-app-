"""STEP 2c-i — History-Based Iris Gaze Rule (gaze DIRECTION layer only; NO oval drawn).

Gaze direction comes from the iris centre's MOVEMENT HISTORY relative to the primary (straight-ahead)
position -- NOT from the sclera centroid (that was noisy/sign-ambiguous). Sclera is used only to find the
limbus arc for the refined centre. Builds on the locked EllSeg anchor + fixed-radius refinement.

Pipeline:
  1. EllSeg anchor per frame (reliable LOCATION, Step 1).
  2. Refined fixed-radius circle centre from the medial/lateral limbus arc, when reliable.
  3. iris centre = refined-if-reliable else raw EllSeg centre; forward-fill gaps.
  4. SMOOTH the centre trajectory (rolling median) -> kills EllSeg shimmer.
  5. primary = median smoothed centre over the roundest-disc (frontal) frames, per eye.
  6. gaze vector = smoothed centre - primary; direction + magnitude + stability -> confidence + status.

Outputs (outputs/<clip>_tracked/step2c_gaze/):
  step2c_gaze.mp4          per-frame: primary (cyan cross), centre trail, smoothed centre, gaze arrow, status
  step2c_gaze_keyframes.png  key frames with the same overlays + numbers
  step2c_gaze.csv          frame,eye,primary,raw,refined,smoothed,gaze_angle,gaze_mag,conf,status
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
OUTDIR = OUTBASE / "step2c_gaze"
OUTDIR.mkdir(parents=True, exist_ok=True)
OPH, OPW = 240, 320
SECTOR, MINARC = 55, 7
DIAG_EYE = "R"
KEY = [40, 100, 160, 220, 235, 286, 335, 400, 470, 540, 610]
SMOOTH_W = 7          # rolling-median window for the trajectory
FRONTAL_ASP = 0.85    # roundest-disc frames define the primary (straight-ahead) centre
TRAIL = 24            # frames of centre history drawn

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

def sclera_and_dark(bgr, gray):
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); S = hsv[:, :, 1]
    bright = gray >= thr; bs = S[bright]
    st = float(np.clip(cv2.threshold(bs.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0], 25, 90)) if bs.size >= 30 else 60.0
    sclera = (bright & (S <= st)).astype(np.uint8)
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx((gray < thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return thr, sclera, dark

def iris_blob_at_anchor(dark, disc):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    best, bov = None, 0
    for i in range(1, n):
        ov = int(((lab == i) & (disc > 0)).sum())
        if ov > bov: bov, best = ov, (lab == i)
    return best.astype(np.uint8) if (best is not None and bov >= 10) else None

def _runs(mask):
    n = len(mask)
    if not mask.any(): return []
    if mask.all(): return [np.arange(n)]
    s = int(np.where(~mask)[0][0]); order = (np.arange(n)+s) % n; m = mask[order]; runs = []; i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            runs.append(order[i:j]); i = j
        else: i += 1
    return runs

def limbus_pts(iris_blob, sclera, dcx, dcy):
    cnts, _ = cv2.findContours(iris_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.empty((0, 2))
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    if len(c) < MINARC: return np.empty((0, 2))
    sd = cv2.dilate(sclera, np.ones((5, 5), np.uint8)); xs, ys = c[:, 0], c[:, 1]
    ang = np.degrees(np.arctan2(ys-dcy, xs-dcx))
    ml = (np.abs(ang) <= SECTOR) | (np.abs(np.abs(ang)-180) <= SECTOR)
    touch = np.zeros(len(c), bool)
    for k, (x, y) in enumerate(c):
        if ml[k]: touch[k] = bool(sd[max(0, y-2):y+3, max(0, x-2):x+3].any())
    good = ml & touch
    keep = [c[r].astype(np.float64) for r in _runs(good) if len(r) >= MINARC]
    return np.vstack(keep) if keep else np.empty((0, 2))

def fit_centre_fixed(pts, R, c0):
    c = np.asarray(c0, float).copy()
    for _ in range(8):
        d = c - pts; nn = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        votes = pts + R * (d / nn[:, None]); inl = np.abs(nn - R) < max(3.0, 0.20*R)
        c = votes[inl].mean(0) if int(inl.sum()) >= 6 else votes.mean(0)
    return c

def disc_aspect(disc):
    dc, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not dc: return 1.0
    dm = max(dc, key=cv2.contourArea)
    if len(dm) < 5: return 1.0
    (_, _), (a1, a2), _ = cv2.fitEllipse(dm); return min(a1, a2)/max(a1, a2, 1e-6)

# ---- seeds ----
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

# ================= PASS 1 — EllSeg once; collect raw centre, limbus pts, disc aspect, width =================
cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); FPS = cap.get(5) or 30.0
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
per = {ek: {} for ek in eyes}       # per[ek][frame] = dict(win, raw=(x,y) full, pts, asp, width)
widths = {ek: [] for ek in eyes}
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
        st.update(cx=reg["cx"]+x0, cy=reg["cy"]+y0, lost=0)          # tracking follows EllSeg centroid (Step 1)
        thr, sclera, dark = sclera_and_dark(frame[y0:y1, x0:x1], gray[y0:y1, x0:x1])
        blob = iris_blob_at_anchor(dark, reg["disc"])
        pts = limbus_pts(blob, sclera, reg["cx"], reg["cy"]) if blob is not None else np.empty((0, 2))
        # width for R calibration (both medial+lateral extremes present)
        if len(pts) >= 6:
            lx = pts[pts[:, 0] < reg["cx"], 0]; rx = pts[pts[:, 0] > reg["cx"], 0]
            if len(lx) >= 3 and len(rx) >= 3:
                w = float(np.median(rx)-np.median(lx))
                if 12 <= w <= 320: widths[ek].append(w)
        per[ek][fi] = dict(win=(x0, y0, x1, y1), raw=(reg["cx"]+x0, reg["cy"]+y0),
                           pts=pts, asp=disc_aspect(reg["disc"]))
cap.release()

R = {}
for ek in eyes:
    arr = np.array(widths[ek])
    R[ek] = float(np.percentile(arr, 80)/2.0) if arr.size >= 5 else 40.0
    print(f"[R] {ek}: {R[ek]:.1f}px  ({arr.size} two-sided frames)")

# ---- build per-frame iris centre: refined-if-reliable else raw; then forward-fill + smooth ----
def roll_med(a, w=SMOOTH_W):
    n = len(a); out = a.copy(); h = w//2
    for i in range(n):
        seg = a[max(0, i-h):i+h+1]; seg = seg[~np.isnan(seg[:, 0])] if seg.ndim == 2 else seg
        if len(seg): out[i] = np.median(seg, 0)
    return out

gaze = {}   # gaze[ek] -> dict of arrays over 0..NF-1
for ek in eyes:
    raw = np.full((NF, 2), np.nan); ref = np.full((NF, 2), np.nan); asp = np.full(NF, np.nan); reli = np.zeros(NF, bool)
    for f, d in per[ek].items():
        raw[f] = d["raw"]; asp[f] = d["asp"]; x0, y0, _, _ = d["win"]
        if len(d["pts"]) >= MINARC:
            seed = (d["raw"][0]-x0, d["raw"][1]-y0)
            c = fit_centre_fixed(d["pts"], R[ek], np.array(seed, float)); cf = np.array([c[0]+x0, c[1]+y0])
            if np.hypot(cf[0]-d["raw"][0], cf[1]-d["raw"][1]) <= 0.6*R[ek]:
                ref[f] = cf; reli[f] = True
    base = np.where(~np.isnan(ref[:, :1]), ref, raw)                  # refined when reliable, else raw EllSeg
    # forward/back fill gaps
    last = None
    for f in range(NF):
        if np.isnan(base[f, 0]):
            base[f] = last if last is not None else base[f]
        else: last = base[f]
    last = None
    for f in range(NF-1, -1, -1):
        if np.isnan(base[f, 0]): base[f] = last if last is not None else base[f]
        else: last = base[f]
    sm = roll_med(base)                                              # smoothed trajectory
    fr = asp >= FRONTAL_ASP                                          # frontal = roundest disc
    prim = np.nanmedian(sm[fr], 0) if fr.any() else np.nanmedian(sm, 0)
    gv = sm - prim
    gaze[ek] = dict(raw=raw, ref=ref, reli=reli, asp=asp, sm=sm, prim=prim, gv=gv)
    print(f"[gaze] {ek}: primary=({prim[0]:.0f},{prim[1]:.0f})  |gaze| median={np.nanmedian(np.hypot(gv[:,0],gv[:,1])):.0f}px  max={np.nanmax(np.hypot(gv[:,0],gv[:,1])):.0f}px")

def status_of(ek, f):
    g = gaze[ek]; gv = g["gv"][f]; mag = float(np.hypot(gv[0], gv[1]))
    if np.isnan(mag): return "no_data", 0.0, 0.0, 0.0
    ang = math.degrees(math.atan2(gv[1], gv[0]))
    lo, hi = max(0, f-SMOOTH_W), min(NF, f+1)                        # recent trajectory stability
    seg = g["gv"][lo:hi]; seg = seg[np.hypot(seg[:, 0], seg[:, 1]) > 2]
    if len(seg):
        u = seg / (np.hypot(seg[:, 0], seg[:, 1])[:, None]); res = float(np.hypot(*u.mean(0)))
    else: res = 0.0
    conf = min(1.0, mag/(0.35*R[ek])) * res
    if mag < 0.18*R[ek]: st = "near_primary"                        # eye ~straight ahead; direction not needed
    elif res < 0.6: st = "ambiguous_gaze"
    else: st = "gaze_ok"
    return st, ang, mag, conf

# ================= PASS 2 — render gaze-direction video (NO oval) =================
cap = cv2.VideoCapture(str(VIDEO))
vw = cv2.VideoWriter(str(OUTDIR / "step2c_gaze.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
COLST = {"gaze_ok": (0, 220, 0), "near_primary": (200, 200, 200), "ambiguous_gaze": (0, 165, 255), "no_data": (0, 0, 255)}
rows = []; fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    for ek in eyes:
        g = gaze[ek]
        if fi not in per[ek] and g["reli"][fi] == False and np.isnan(g["sm"][fi, 0]): continue
        prim = g["prim"]; sm = g["sm"][fi]
        st, ang, mag, conf = status_of(ek, fi)
        col = COLST.get(st, (255, 255, 255))
        trail = g["sm"][max(0, fi-TRAIL):fi+1]                       # centre history
        for a, b in zip(trail[:-1], trail[1:]):
            if not (np.isnan(a[0]) or np.isnan(b[0])): cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (180, 120, 0), 1)
        cv2.drawMarker(frame, (int(prim[0]), int(prim[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)   # primary (cyan)
        cv2.circle(frame, (int(sm[0]), int(sm[1])), 5, col, -1)      # smoothed current centre
        if mag > 2:                                                 # gaze arrow primary -> current
            cv2.arrowedLine(frame, (int(prim[0]), int(prim[1])), (int(sm[0]), int(sm[1])), (255, 0, 255), 2, tipLength=0.25)
        cv2.putText(frame, f"{ek}:{st}", (int(sm[0])-40, int(sm[1])-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(frame, f"f{fi}  STEP2c-i gaze direction (no oval)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}  STEP2c-i gaze direction (no oval)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    vw.write(frame)
cap.release(); vw.release()

# ---- key-frame table + montage (DIAG_EYE) ----
print(f"\n--- {DIAG_EYE}-eye gaze at key frames (primary=({gaze[DIAG_EYE]['prim'][0]:.0f},{gaze[DIAG_EYE]['prim'][1]:.0f})) ---")
print(f"{'frame':>6} {'raw':>13} {'refined':>13} {'smoothed':>13} {'ang':>6} {'mag':>5} {'conf':>5}  status")
for f in KEY:
    g = gaze[DIAG_EYE]
    if f >= NF: continue
    st, ang, mag, conf = status_of(DIAG_EYE, f)
    rw = g["raw"][f]; rf = g["ref"][f]; sm = g["sm"][f]
    rws = f"({rw[0]:.0f},{rw[1]:.0f})" if not np.isnan(rw[0]) else "   --   "
    rfs = f"({rf[0]:.0f},{rf[1]:.0f})" if not np.isnan(rf[0]) else "   --   "
    sms = f"({sm[0]:.0f},{sm[1]:.0f})" if not np.isnan(sm[0]) else "   --   "
    print(f"{f:>6} {rws:>13} {rfs:>13} {sms:>13} {ang:>6.0f} {mag:>5.0f} {conf:>5.2f}  {st}")

cap = cv2.VideoCapture(str(VIDEO)); PW = 300; panels = []
for f in KEY:
    if f >= NF or f not in per[DIAG_EYE]: continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    if not ok: continue
    x0, y0, x1, y1 = per[DIAG_EYE][f]["win"]; crop = fr[y0:y1, x0:x1].copy()
    g = gaze[DIAG_EYE]; prim = g["prim"]; sm = g["sm"][f]; st, ang, mag, conf = status_of(DIAG_EYE, f)
    col = COLST.get(st, (255, 255, 255))
    def loc(p): return (int(p[0]-x0), int(p[1]-y0))
    trail = g["sm"][max(0, f-TRAIL):f+1]
    for a, b in zip(trail[:-1], trail[1:]):
        if not (np.isnan(a[0]) or np.isnan(b[0])): cv2.line(crop, loc(a), loc(b), (180, 120, 0), 1)
    cv2.drawMarker(crop, loc(prim), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.circle(crop, loc(sm), 5, col, -1)
    if mag > 2: cv2.arrowedLine(crop, loc(prim), loc(sm), (255, 0, 255), 2, tipLength=0.25)
    h = int(PW*crop.shape[0]/crop.shape[1]); crop = cv2.resize(crop, (PW, h))
    bar = np.full((54, PW, 3), 25, np.uint8)
    for i, t in enumerate([f"f{f}  {st}", f"ang={ang:.0f} mag={mag:.0f} conf={conf:.2f}"]):
        cv2.putText(bar, t, (4, 20+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col if i == 0 else (255, 255, 255), 1)
    panels.append(np.vstack([crop, bar]))
cap.release()
if panels:
    hmax = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, hmax-p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels]
    rowsm = [np.hstack(panels[i:i+4]) for i in range(0, len(panels), 4)]
    wmx = max(r.shape[1] for r in rowsm)
    rowsm = [cv2.copyMakeBorder(r, 0, 0, 0, wmx-r.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for r in rowsm]
    cv2.imwrite(str(OUTDIR / "step2c_gaze_keyframes.png"), np.vstack(rowsm))

with open(OUTDIR / "step2c_gaze.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["frame", "eye", "prim_x", "prim_y", "raw_x", "raw_y", "sm_x", "sm_y", "gaze_ang", "gaze_mag", "conf", "status"])
    for ek in eyes:
        g = gaze[ek]
        for f in range(NF):
            if np.isnan(g["sm"][f, 0]): continue
            st, ang, mag, conf = status_of(ek, f)
            w.writerow([f, ek, round(g["prim"][0], 1), round(g["prim"][1], 1), round(g["raw"][f, 0], 1) if not np.isnan(g["raw"][f, 0]) else "",
                        round(g["raw"][f, 1], 1) if not np.isnan(g["raw"][f, 1]) else "", round(g["sm"][f, 0], 1), round(g["sm"][f, 1], 1),
                        round(ang, 1), round(mag, 1), round(conf, 2), st])
print("\nvideo ->", OUTDIR / "step2c_gaze.mp4")
print("keyframes ->", OUTDIR / "step2c_gaze_keyframes.png")
print("csv ->", OUTDIR / "step2c_gaze.csv")
