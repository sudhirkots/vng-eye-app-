"""Raw EllSeg CENTROID trace -> ellseg_centroid.csv (frame, eye, cx, cy, status).

The plain, reliable EllSeg iris/pupil-disc CENTROID per frame -- no oval fit, no arc fit, no rescue logic
that adds jitter. EllSeg's LOCATION is smooth/continuous (only its disc SHAPE flickers), so the centroid is
the cleanest simple eye-position signal. Feeds tools/nystagmus_direction.py. Run with the rit-nets venv.
"""
import sys, os, json, csv
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
OUTDIR = OUTBASE / "step2_fixed_circle"; OUTDIR.mkdir(parents=True, exist_ok=True)
OPH, OPW = 240, 320

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

def sclera_mask(bgr, gray):
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]; bright = gray >= thr; bs = S[bright]
    st = float(np.clip(cv2.threshold(bs.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0], 25, 90)) if bs.size >= 30 else 60.0
    m = (bright & (S <= st)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

def region(bgr_full, gray_full, cx, cy, hw, hh, W, H):
    """EllSeg iris centroid + SCLERA BALANCE (sclera px left vs right of the iris) + disc area, for the
    anatomical (head-motion-invariant) gaze-zone classification."""
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    if disc.sum() < 30: return None
    m = cv2.moments(disc); icx = m["m10"]/m["m00"]
    scl = sclera_mask(bgr_full[y0:y1, x0:x1], gc)
    cols = scl.sum(0)                                         # sclera pixels per column
    xs = np.arange(len(cols))
    nL = int(cols[xs < icx].sum()); nR = int(cols[xs >= icx].sum())   # sclera left vs right of the iris
    return dict(cx=m["m10"]/m["m00"]+x0, cy=m["m01"]/m["m00"]+y0, nL=nL, nR=nR, area=int(disc.sum()))

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok0, frame0 = cap.read()

def auto_seed(fr):
    """No clinician marks -> locate the two eyes as the darkest round iris blobs (close-up two-eye clips)."""
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); gb = cv2.GaussianBlur(g, (0, 0), 2)
    d = cv2.morphologyEx((gb < np.percentile(gb, 8)).astype(np.uint8)*255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stt, cen = cv2.connectedComponentsWithStats(d, 8)
    cand = []
    for i in range(1, n):
        a, w, h = stt[i, cv2.CC_STAT_AREA], stt[i, cv2.CC_STAT_WIDTH], stt[i, cv2.CC_STAT_HEIGHT]
        if a < (W*0.02)**2 or a > (W*0.22)**2 or not (0.4 < w/(h+1e-9) < 2.5) or cen[i][1] > H*0.72: continue
        cand.append((cen[i][0], cen[i][1], max(w, h)))
    best = None
    for i in range(len(cand)):
        for j in range(i+1, len(cand)):
            A, B = cand[i], cand[j]; dy = abs(A[1]-B[1]); dx = abs(A[0]-B[0]); sr = min(A[2], B[2])/max(A[2], B[2])
            if dy < H*0.15 and W*0.12 < dx < W*0.6 and sr > 0.4:
                sc = sr*100 - dy/5.0
                if best is None or sc > best[0]: best = (sc, A, B)
    if best: a, b = sorted([best[1], best[2]], key=lambda c: c[0])          # left-image, right-image
    else: a, b = (W*0.33, H*0.45, W*0.2), (W*0.67, H*0.45, W*0.2)           # fallback default positions
    ow, oh = W*0.22, W*0.17
    return {"R": dict(ow=ow, oh=oh, seed=(a[0], a[1])), "L": dict(ow=ow, oh=oh, seed=(b[0], b[1]))}  # R eye = image-left

def iris_win_at(gray, cx, cy):
    """Auto-size the tracking window from the dark iris blob at a clicked seed (so ONE click is enough)."""
    r = int(W*0.12); x0, y0 = max(0, int(cx-r)), max(0, int(cy-r)); x1, y1 = min(W, int(cx+r)), min(H, int(cy+r))
    p = gray[y0:y1, x0:x1]
    if p.size == 0: return W*0.13
    d = cv2.morphologyEx((p < np.percentile(p, 30)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(d, 8); lc = (cx-x0, cy-y0); best, bd = None, 1e18
    for i in range(1, n):
        dd = (cen[i][0]-lc[0])**2 + (cen[i][1]-lc[1])**2
        if dd < bd and st[i, cv2.CC_STAT_AREA] > 20: bd, best = dd, i
    if best is None: return W*0.13
    rad = 0.5*max(st[best, cv2.CC_STAT_WIDTH], st[best, cv2.CC_STAT_HEIGHT])
    return float(max(W*0.05, rad*2.5))                       # window*0.6 ~ eye half-width

MSF = REPO / "manual_seeds.json"
manual = json.load(open(MSF, encoding="utf-8")) if MSF.exists() else {}
if CLIP in manual:
    g0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY); ent = manual[CLIP]; eyes = {}
    for ek in ("R", "L"):
        if ek in ent:                                        # 1 or 2 eyes -- use the clearer iris if only one
            sx, sy = ent[ek]; ow = iris_win_at(g0, sx, sy); eyes[ek] = dict(ow=ow, oh=ow, seed=(sx, sy))
    print(f"seeds: MANUAL {list(eyes)}")
elif (GT / "meta.json").exists():
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
    print("seeds: clinician orbit marks")
else:
    eyes = auto_seed(frame0); print(f"seeds: AUTO  R={tuple(round(v) for v in eyes['R']['seed'])} L={tuple(round(v) for v in eyes['L']['seed'])}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
MAXF = int(os.environ.get("RIT_MAXF", 100000))
rows = []; found = {ek: 0 for ek in eyes}; fi = -1
while True:
    ok, frame = cap.read()
    if not ok or fi+1 >= MAXF: break
    fi += 1; gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek, e in eyes.items():
        st = state[ek]; grow = 1.0 + min(st["lost"], 6)*0.25
        r = region(frame, gray, st["cx"], st["cy"], e["ow"]*0.6*grow, e["oh"]*0.6*grow, W, H)
        if r is None:
            st["lost"] += 1; rows.append((fi, ek, "", "", "needs_rescue", "", "", "")); continue
        st.update(cx=r["cx"], cy=r["cy"], lost=0); found[ek] += 1
        rows.append((fi, ek, round(r["cx"], 2), round(r["cy"], 2), "ok", r["nL"], r["nR"], r["area"]))
cap.release()
with open(OUTDIR / "ellseg_centroid.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["frame", "eye", "cx", "cy", "status", "scl_left", "scl_right", "disc_area"]); w.writerows(rows)
print("EllSeg centroid found:", found, "of", NF)
print("csv ->", OUTDIR / "ellseg_centroid.csv")
