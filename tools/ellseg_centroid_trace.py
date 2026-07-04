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

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7))
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
rows = []; found = {ek: 0 for ek in eyes}; fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
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
