"""RITnet smoke test on RIT per-eye crops (approach #2 — pretrained eye net).

Runs the pretrained RITnet (OpenEDS, near-IR) UNCHANGED on a few vestibular clinician-marked crops,
using RITnet's own test preprocessing (grayscale -> gamma 0.8 -> CLAHE 1.5/8x8 -> Normalize[.5],[.5]).
Outputs class map: 0=bg 1=sclera 2=iris 3=pupil. Overlays + iris IoU vs the clinician mask.
Exploration only; nothing wired, nothing committed.
"""
import sys, glob, os
import numpy as np
import cv2
import torch

RITNET = r"C:/Users/Dr.Sudhir/rit-nets/RITnet"
sys.path.insert(0, RITNET)
from models import model_dict            # noqa: E402


def get_predictions(output):             # inlined from RITnet utils (avoids its sklearn import)
    bs, c, h, w = output.size()
    _, indices = output.cpu().max(1)
    return indices.view(bs, h, w)

GT = r"C:/Users/Dr.Sudhir/OneDrive/Documents/Neurology Talks/VNG-EYE app/outputs/nystagmus at rest to left in right vestibular neuritis_tracked/RIT_ground_truth"
OUT = r"C:/Users/Dr.Sudhir/rit-nets/_smoke_out"
os.makedirs(OUT, exist_ok=True)
TW, TH = 640, 400                        # RITnet-ish input (letterboxed, no distortion)

# representative frames if present, else first 6 crops
WANT = ["f0187_L", "f0227_L", "f0267_R", "f0311_L", "f0123_R", "f0060_R"]
raws = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(GT, "raw", "*.png"))}
uids = [u for u in WANT if u in raws] or sorted(raws)[:6]

# ---- model
model = model_dict["densenet"]
try:
    sd = torch.load(os.path.join(RITNET, "best_model.pkl"), map_location="cpu", weights_only=True)
except Exception:
    sd = torch.load(os.path.join(RITNET, "best_model.pkl"), map_location="cpu", weights_only=False)
model.load_state_dict(sd)
model.eval()
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)


def letterbox(img, tw, th):
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw), np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas, (ox, oy, nw, nh)


def iou(a, b):
    a = a > 0; b = b > 0
    u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else float("nan")


tiles = []
print(f"RITnet smoke test on {len(uids)} crops: {uids}")
for uid in uids:
    bgr = cv2.imread(raws[uid])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.LUT(gray, table)
    g = clahe.apply(g)
    lb, (ox, oy, nw, nh) = letterbox(g, TW, TH)
    x = torch.from_numpy(((lb.astype(np.float32) / 255.0) - 0.5) / 0.5)[None, None]
    with torch.no_grad():
        out = model(x)
    pred = get_predictions(out)[0].numpy().astype(np.uint8)      # TH x TW, {0..3}
    pred_crop = pred[oy:oy + nh, ox:ox + nw]
    pred_full = cv2.resize(pred_crop, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    iris = (pred_full == 2); sclera = (pred_full == 1); pupil = (pred_full == 3)
    ov = bgr.copy()
    ov[sclera] = (0.5 * ov[sclera] + np.array([200, 90, 0])).astype(np.uint8)     # blue-ish
    ov[iris] = (0.5 * ov[iris] + np.array([0, 0, 200])).astype(np.uint8)          # red
    ov[pupil] = (0.5 * ov[pupil] + np.array([0, 220, 0])).astype(np.uint8)        # green
    # iris IoU vs clinician mask if present
    gm = os.path.join(GT, "masks", uid + "_iris.png")
    txt = uid
    if os.path.exists(gm):
        gt = cv2.imread(gm, 0)
        ii = iou(iris.astype(np.uint8), gt)
        txt = f"{uid}  irisIoU={ii:.2f}"
        print(f"  {uid}: iris px={int(iris.sum())} sclera px={int(sclera.sum())} vs clinician iris IoU={ii:.3f}")
    else:
        print(f"  {uid}: iris px={int(iris.sum())} sclera px={int(sclera.sum())} (no GT)")
    cv2.putText(ov, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(ov, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(OUT, uid + "_ritnet.png"), ov)
    tiles.append(cv2.resize(ov, (520, int(520 * ov.shape[0] / ov.shape[1]))))

# montage (2 cols)
if tiles:
    hmax = max(t.shape[0] for t in tiles); w = 520
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows = []
    for i in range(0, len(tiles), 2):
        row = tiles[i:i + 2]
        while len(row) < 2:
            row.append(np.zeros((hmax, w, 3), np.uint8))
        rows.append(np.hstack([cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(50, 50, 50)) for t in row]))
    cv2.imwrite(os.path.join(OUT, "_montage.png"), np.vstack(rows))
    print("montage ->", os.path.join(OUT, "_montage.png"))
