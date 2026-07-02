"""EllSeg smoke test on RIT per-eye crops (approach #2, second net).

Runs pretrained EllSeg (DenseElNet 'all' model) UNCHANGED, with EllSeg's own preprocessing
(resize width->320, vertical pad to 240, per-image standardisation). Classes: 0=bg 1=iris 2=pupil
(EllSeg predicts the FULL iris even when occluded). Overlay + iris IoU vs the clinician mask.
Exploration only; nothing wired, nothing committed.
"""
import sys, glob, os
import numpy as np
import cv2
import torch

ELL = r"C:/Users/Dr.Sudhir/rit-nets/EllSeg"
sys.path.insert(0, ELL)
# EllSeg predates numpy 2 (uses np.int/np.float/np.bool, removed in numpy 2). opencv 5 forces
# numpy>=2 here, so restore the removed aliases (they were always the plain builtins).
for _a, _t in (("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
from modelSummary import model_dict          # noqa: E402

GT = r"C:/Users/Dr.Sudhir/OneDrive/Documents/Neurology Talks/VNG-EYE app/outputs/nystagmus at rest to left in right vestibular neuritis_tracked/RIT_ground_truth"
OUT = r"C:/Users/Dr.Sudhir/rit-nets/_smoke_ellseg"
os.makedirs(OUT, exist_ok=True)
OPH, OPW = 240, 320

WANT = ["f0187_L", "f0227_L", "f0267_R", "f0311_L", "f0123_R", "f0060_R"]
raws = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(GT, "raw", "*.png"))}
uids = [u for u in WANT if u in raws] or sorted(raws)[:6]

model = model_dict["ritnet_v3"]
netDict = torch.load(os.path.join(ELL, "weights", "all.git_ok"), map_location="cpu", weights_only=False)
model.load_state_dict(netDict["state_dict"], strict=True)
model.eval()


def preprocess(gray):
    h, w = gray.shape
    sc = OPW / w
    nw, nh = int(round(w * sc)), int(round(h * sc))
    img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    if OPH >= nh:                                   # pad (our landscape crops)
        pad = OPH - nh; top = pad // 2
        img = np.pad(img, ((top, pad - top), (0, 0)))
        valid = (top, nh)
    else:                                           # chop
        top = (nh - OPH) // 2
        img = img[top:top + OPH, :]; valid = (0, OPH)
    imgn = (img - img.mean()) / (img.std() + 1e-6)
    return torch.from_numpy(imgn).unsqueeze(0).unsqueeze(0).to(torch.float32), valid, (nw, nh)


def seg_forward(x):
    with torch.no_grad():
        x4, x3, x2, x1, xc = model.enc(x)
        seg = model.dec(x4, x3, x2, x1, xc)
    return seg.max(1)[1][0].numpy().astype(np.uint8)          # HxW class map


def iou(a, b):
    a = a > 0; b = b > 0; u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else float("nan")


tiles = []
print(f"EllSeg smoke test on {len(uids)} crops: {uids}")
for uid in uids:
    bgr = cv2.imread(raws[uid]); H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    x, (vtop, vnh), (nw, nh) = preprocess(gray)
    seg = seg_forward(x)                                       # 240x320
    seg_valid = seg[vtop:vtop + vnh, :]                        # remove vertical pad
    seg_full = cv2.resize(seg_valid, (W, H), interpolation=cv2.INTER_NEAREST)
    iris = (seg_full == 1); pupil = (seg_full == 2)
    ov = bgr.copy()
    ov[iris] = (0.5 * ov[iris] + np.array([0, 0, 200])).astype(np.uint8)     # red iris
    ov[pupil] = (0.5 * ov[pupil] + np.array([0, 220, 0])).astype(np.uint8)   # green pupil
    gm = os.path.join(GT, "masks", uid + "_iris.png")
    txt = uid
    if os.path.exists(gm):
        gt = cv2.imread(gm, 0)
        ii = iou(iris.astype(np.uint8), gt)
        disc = iou((iris | pupil).astype(np.uint8), gt)      # fair: EllSeg iris+pupil = clinician disc
        txt = f"{uid}  irisIoU={ii:.2f} discIoU={disc:.2f}"
        print(f"  {uid}: iris-only IoU={ii:.3f}  iris+pupil(disc) IoU={disc:.3f}")
    else:
        print(f"  {uid}: iris px={int(iris.sum())} pupil px={int(pupil.sum())} (no GT)")
    cv2.putText(ov, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(ov, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    tiles.append(cv2.resize(ov, (520, int(520 * H / W))))

if tiles:
    hmax = max(t.shape[0] for t in tiles); w = 520
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows = [np.hstack([cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(50, 50, 50))
                       for t in (tiles[i:i + 2] + [np.zeros((hmax, w, 3), np.uint8)])[:2]])
            for i in range(0, len(tiles), 2)]
    cv2.imwrite(os.path.join(OUT, "_montage.png"), np.vstack(rows))
    print("montage ->", os.path.join(OUT, "_montage.png"))
