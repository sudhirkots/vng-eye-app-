"""RIT — mark the VISIBLE iris only, in red. No centre, no circle completion, no geometry.

Dr. K: stop marking the centre / completing the circle. Just mark whatever part of the iris is actually
SEEN. So: EllSeg localizes the iris region (and its centroid keeps the search box on the eye through head
motion — used internally only, never drawn); then we trim that region to the genuinely dark, visible iris
(EllSeg region AND dark pixels) so nothing is painted under the lid; fill it red. That's it.

Rule-based trim = shape (EllSeg region) + darkness (visible). Run with the rit-nets venv python.
"""
import sys, os, json, math
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
OUTDIR = OUTBASE / "RIT_ellseg_iris_paint"
OUTDIR.mkdir(parents=True, exist_ok=True)
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

def visible_iris(gray_full, cx, cy, hw, hh, W, H):
    """EllSeg iris region trimmed to dark visible pixels -> full-frame mask + centroid (internal)."""
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    if disc.sum() < 40: return None
    # darkness trim: within a dilated disc, keep only pixels darker than Otsu (drops lid completion / skin)
    ctx = cv2.dilate(disc, np.ones((9, 9), np.uint8))
    vals = gc[ctx > 0]
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (gc < max(thr, 40)).astype(np.uint8)
    vis = cv2.morphologyEx(disc & dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    vis = cv2.morphologyEx(vis, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(vis, 8)
    if n <= 1: return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    vis = (lab == big).astype(np.uint8)
    ccx, ccy = cent[big]
    full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = vis
    return dict(mask=full, cx=ccx + x0, cy=ccy + y0, area=int(vis.sum()))

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

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); FPS = cap.get(5) or 30.0
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
vw = cv2.VideoWriter(str(OUTDIR / "iris_painted.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
painted = {ek: 0 for ek in eyes}
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek, e in eyes.items():
        st = state[ek]
        grow = 1.0 + min(st["lost"], 6) * 0.25
        det = visible_iris(gray, st["cx"], st["cy"], e["ow"]*0.6*grow, e["oh"]*0.6*grow, W, H)
        if det is not None:
            m = det["mask"] > 0
            frame[m] = (0.40 * frame[m] + np.array([0, 0, 255]) * 0.60).astype(np.uint8)  # red fill
            cv2.drawContours(frame, cv2.findContours(det["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                             -1, (0, 0, 200), 2)                                          # crisp edge
            st.update(cx=det["cx"], cy=det["cy"], lost=0); painted[ek] += 1
        else:
            st["lost"] += 1
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    vw.write(frame)
cap.release(); vw.release()
print("painted frames:", painted, "of", NF)
print("video ->", OUTDIR / "iris_painted.mp4")
