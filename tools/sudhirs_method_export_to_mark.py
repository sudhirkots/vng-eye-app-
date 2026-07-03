"""Export the failing side-gaze / half-covered frames of Sudhir's iris-sclera-almond method as clean
per-eye crops for the clinician to mark the TRUE iris. Also saves the auto prediction next to each clean
crop, so after marking we can see which rule was broken. Writes to <method_out>/to_mark/.

id = f{frame:04d}_{eye}.  Clean crop = f0330_R.png (paint the real iris RED on this).
Prediction = f0330_R_prediction.png (what the method marked). meta.json maps crop->source pixels.
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
# reuse the method's exact functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("_m", Path(__file__).resolve().parent / "sudhirs_iris_sclera_almond_method.py")

REPO = Path(__file__).resolve().parent.parent
CLIP = "nystagmus at rest to left in right vestibular neuritis"
VIDEO = REPO / "samples" / f"{CLIP}.mp4"
OUTBASE = REPO / "outputs" / f"{CLIP}_tracked"
GT = OUTBASE / "RIT_ground_truth"
TOMARK = OUTBASE / "sudhirs_iris_sclera_almond_method" / "to_mark"
TOMARK.mkdir(parents=True, exist_ok=True)
OPH, OPW = 240, 320
SCALE = 2
SELECTED = [227, 234, 267, 300, 311, 330, 333, 336, 338, 341, 353, 364, 378]

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
    return dict(win=(x0, y0, x1, y1), disc=disc)

# import mark_iris from the method module (load without running its __main__ loop by exec of just the def)
_src = (Path(__file__).resolve().parent / "sudhirs_iris_sclera_almond_method.py").read_text(encoding="utf-8")
_start = _src.index("def mark_iris(")
_end = _src.index("\nmeta = json.load(")
_ns = {}
exec("import cv2, numpy as np, math\n" + _src[_start:_end], _ns)
mark_iris = _ns["mark_iris"]

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
sel = set(SELECTED); out_meta = {}
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
        iris, _oval = mark_iris(frame[y0:y1, x0:x1], gray[y0:y1, x0:x1], reg["disc"])
        if iris is not None and iris.sum() > 0:
            M = cv2.moments(iris); st.update(cx=M["m10"]/M["m00"]+x0, cy=M["m01"]/M["m00"]+y0, lost=0)
        else:
            st["lost"] += 1
        if fi in sel:
            uid = f"f{fi:04d}_{ek}"
            crop = frame[y0:y1, x0:x1]
            big = cv2.resize(crop, (crop.shape[1]*SCALE, crop.shape[0]*SCALE), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(TOMARK / f"{uid}.png"), big)                      # CLEAN - paint the real iris RED here
            pred = big.copy()
            if iris is not None and iris.sum() > 0:
                irisb = cv2.resize(iris, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                pred[irisb] = (0.35 * pred[irisb] + np.array([0, 0, 255]) * 0.65).astype(np.uint8)
            cv2.imwrite(str(TOMARK / f"{uid}_prediction.png"), pred)          # what the method marked
            out_meta[uid] = {"frame": fi, "eye": ek, "X0": x0, "Y0": y0, "scale": SCALE,
                             "crop_w": big.shape[1], "crop_h": big.shape[0]}
cap.release()
json.dump(out_meta, open(TOMARK / "meta.json", "w"), indent=1)
(TOMARK / "MARKING_INSTRUCTIONS.md").write_text(
    "# Mark the TRUE iris — Sudhir's iris-sclera-almond method correction set\n\n"
    "For each `fNNNN_<eye>.png` (the CLEAN crop), paint the **real iris** in OPAQUE RED — the iris as its\n"
    "true circular/oval/sector shape (complete the circle you see). Save with the SAME filename.\n"
    "`fNNNN_<eye>_prediction.png` shows what the method marked — for your reference only, do not edit it.\n\n"
    f"Frames exported: {SELECTED} (both eyes). meta.json maps each crop back to source-video pixels.\n",
    encoding="utf-8")
# paint tool with the crop list pre-embedded -> auto-loads on open (no "Load images" click)
_tpl = (Path(__file__).resolve().parent / "rit_paint_tool.html").read_text(encoding="utf-8")
_names = ", ".join(json.dumps(f"{u}.png") for u in sorted(out_meta))
(TOMARK / "rit_paint_tool.html").write_text(_tpl.replace("/*__PRELOAD__*/", _names), encoding="utf-8")
print(f"exported {len(out_meta)} crops + auto-loading paint tool to {TOMARK}")
