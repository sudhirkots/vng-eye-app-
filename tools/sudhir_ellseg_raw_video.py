"""RAW EllSeg output over the WHOLE clip — no Steps 2-4, no sclera/dark, no final iris. Just what EllSeg
gives: its iris/pupil disc (green fill + outline) and its centroid (yellow dot), on every frame, both eyes.
This is only to judge EllSeg's location / disc size / whether it grabs reflection/canthus/lid.
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
CLIP = "nystagmus at rest to left in right vestibular neuritis"
VIDEO = REPO / "samples" / f"{CLIP}.mp4"
OUTBASE = REPO / "outputs" / f"{CLIP}_tracked"
GT = OUTBASE / "RIT_ground_truth"
OUTDIR = OUTBASE / "sudhirs_iris_sclera_almond_method"
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

def ellseg_region(gray_full, cx, cy, hw, hh, W, H):
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    if disc.sum() < 30: return None
    m = cv2.moments(disc); return dict(win=(x0, y0, x1, y1), disc=disc, cx=m["m10"]/m["m00"]+x0, cy=m["m01"]/m["m00"]+y0)

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
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); FPS = cap.get(5) or 30.0
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
vw = cv2.VideoWriter(str(OUTDIR / "ellseg_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
found = {ek: 0 for ek in eyes}; fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek, e in eyes.items():
        st = state[ek]; grow = 1.0 + min(st["lost"], 6) * 0.25
        reg = ellseg_region(gray, st["cx"], st["cy"], e["ow"]*0.6*grow, e["oh"]*0.6*grow, W, H)
        if reg is None: st["lost"] += 1; continue
        st.update(cx=reg["cx"], cy=reg["cy"], lost=0); found[ek] += 1
        x0, y0, x1, y1 = reg["win"]
        full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = reg["disc"]
        m = full > 0
        frame[m] = (0.5 * frame[m] + np.array([0, 200, 0]) * 0.5).astype(np.uint8)   # RAW EllSeg disc (green)
        cv2.drawContours(frame, cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
        cv2.circle(frame, (int(reg["cx"]), int(reg["cy"])), 6, (0, 255, 255), -1)     # EllSeg centroid (yellow)
    cv2.putText(frame, f"f{fi}  RAW EllSeg disc only", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}  RAW EllSeg disc only", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    vw.write(frame)
cap.release(); vw.release()
print("EllSeg found:", found, "of", NF, "frames")
print("video ->", OUTDIR / "ellseg_raw.mp4")
