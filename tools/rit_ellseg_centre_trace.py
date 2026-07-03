"""RIT approach #2(a) — EllSeg iris-centre trace, orbit-constrained, across a full clip.

Runs pretrained EllSeg (DenseElNet 'all', zero-shot) on each frame's per-eye orbit crop, takes the
IRIS DISC (EllSeg iris+pupil = the clinician's whole dark disc), RESTRICTS it to inside the marked RIT
Orbit Lock oval (kills the canthal-skin / eyelid bleed that is EllSeg's known failure), and reports the
disc-ellipse CENTRE per frame. Output = the iris-centre TRACE (the clinical signal), not a mask IoU.

Design notes / honesty:
- The orbit oval is treated as STATIC per eye (fit once to the union of the clinician-marked frames'
  `orbit_crop` polygons in meta.json, mapped to source-video pixels). That matches the RIT Orbit Lock model
  (marked on frame 1, held while the iris moves inside). A per-frame tracked orbit would be tighter.
- No fabrication: if the constrained disc is too small / the centre falls outside the oval, the frame is
  marked `uncertain` and the centre is left blank (NaN) rather than guessed. False negatives are safe.
- Nothing here is wired into the clinical RIT path. Exploration only.

Env: the off-OneDrive EllSeg stack (repo + weights + torch venv). ELLSEG_DIR overrides the location;
default is ~/rit-nets/EllSeg so it works on both the home and clinic machines.
Run:  <rit-nets-venv python>  tools/rit_ellseg_centre_trace.py
"""
import sys, os, json, math
from pathlib import Path
import numpy as np
import cv2

# --- EllSeg import (numpy-2 alias shim; EllSeg predates numpy 2) ---
ELL = Path(os.environ.get("ELLSEG_DIR", Path.home() / "rit-nets" / "EllSeg"))
sys.path.insert(0, str(ELL))
for _a, _t in (("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
import torch
from modelSummary import model_dict  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLIP = os.environ.get("RIT_CLIP", "nystagmus at rest to left in right vestibular neuritis")
VIDEO = REPO / "samples" / f"{CLIP}.mp4"
OUTBASE = REPO / "outputs" / f"{CLIP}_tracked"
GT = OUTBASE / "RIT_ground_truth"
OUTDIR = OUTBASE / "RIT_ellseg_centre_trace"
OUTDIR.mkdir(parents=True, exist_ok=True)
OPH, OPW = 240, 320
MARGIN = 0.18            # crop margin around the oval bbox
MIN_DISC_FRAC = 0.004    # min disc area (fraction of oval area) to accept a frame

# --- model ---
model = model_dict["ritnet_v3"]
netDict = torch.load(ELL / "weights" / "all.git_ok", map_location="cpu", weights_only=False)
model.load_state_dict(netDict["state_dict"], strict=True)
model.eval()


def preprocess(gray):
    h, w = gray.shape
    sc = OPW / w
    nw, nh = int(round(w * sc)), int(round(h * sc))
    img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    if OPH >= nh:
        pad = OPH - nh; top = pad // 2
        img = np.pad(img, ((top, pad - top), (0, 0))); valid = (top, nh)
    else:
        top = (nh - OPH) // 2
        img = img[top:top + OPH, :]; valid = (0, OPH)
    imgn = (img - img.mean()) / (img.std() + 1e-6)
    return torch.from_numpy(imgn).unsqueeze(0).unsqueeze(0).to(torch.float32), valid, (nw, nh)


def seg_forward(x):
    with torch.no_grad():
        x4, x3, x2, x1, xc = model.enc(x)
        seg = model.dec(x4, x3, x2, x1, xc)
    return seg.max(1)[1][0].numpy().astype(np.uint8)


# --- static per-eye orbit oval in SOURCE-video pixels, from the clinician-marked meta.json ---
meta = json.load(open(GT / "meta.json", encoding="utf-8"))


def orbit_source_points(entry):
    X0, Y0, sc = entry["X0"], entry["Y0"], entry["scale"]
    return [(x / sc + X0, y / sc + Y0) for x, y in entry["orbit_crop"]]


ovals = {}   # eye -> dict(ellipse=((cx,cy),(MA,ma),ang), bbox=(x0,y0,x1,y1))
for ek in ("L", "R"):
    pts = []
    for v in meta.values():
        if v.get("eye") == ek and "orbit_crop" in v:
            pts.extend(orbit_source_points(v))
    if len(pts) < 5:
        continue
    pts = np.array(pts, np.float32)
    ell = cv2.fitEllipse(pts)                       # ((cx,cy),(MA,ma),angle)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    mw, mh = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    ovals[ek] = dict(ellipse=ell, bbox=(x0 - mw, y0 - mh, x1 + mw, y1 + mh))

print("orbit ovals (source px):")
for ek, o in ovals.items():
    (cx, cy), (MA, ma), ang = o["ellipse"]
    print(f"  {ek}: centre=({cx:.0f},{cy:.0f}) axes=({MA:.0f},{ma:.0f}) angle={ang:.0f}")


def oval_mask(shape, ell):
    m = np.zeros(shape[:2], np.uint8)
    (cx, cy), (MA, ma), ang = ell
    cv2.ellipse(m, (int(cx), int(cy)), (int(MA / 2), int(ma / 2)), ang, 0, 360, 255, -1)
    return m


# --- run over the clip ---
cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
NF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"video {W}x{H} {NF} frames @ {FPS:.1f}fps")

masks = {ek: oval_mask((H, W), o["ellipse"]) for ek, o in ovals.items()}
oval_area = {ek: int((masks[ek] > 0).sum()) for ek in ovals}

vw = cv2.VideoWriter(str(OUTDIR / "centre_trace_overlay.mp4"),
                     cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W // 2, H // 2))
rows = []
fi = -1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fi += 1
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ov = frame.copy()
    rec = {"frame": fi}
    for ek, o in ovals.items():
        (bx0, by0, bx1, by1) = o["bbox"]
        cx0, cy0 = max(0, int(bx0)), max(0, int(by0))
        cx1, cy1 = min(W, int(bx1)), min(H, int(by1))
        gcrop = gray_full[cy0:cy1, cx0:cx1]
        status, ctr, disc_frac = "uncertain", (np.nan, np.nan), 0.0
        if gcrop.size and gcrop.shape[0] > 8 and gcrop.shape[1] > 8:
            x, (vtop, vnh), _ = preprocess(gcrop)
            seg = seg_forward(x)
            seg_valid = seg[vtop:vtop + vnh, :]
            seg_crop = cv2.resize(seg_valid, (gcrop.shape[1], gcrop.shape[0]), interpolation=cv2.INTER_NEAREST)
            disc_full = np.zeros((H, W), np.uint8)
            disc_full[cy0:cy1, cx0:cx1] = ((seg_crop == 1) | (seg_crop == 2)).astype(np.uint8)
            disc_full &= (masks[ek] > 0).astype(np.uint8)      # ORBIT CONSTRAINT
            cnts, _ = cv2.findContours(disc_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                disc_frac = area / max(1, oval_area[ek])
                if disc_frac >= MIN_DISC_FRAC:
                    if len(c) >= 5:
                        (ecx, ecy), _, _ = cv2.fitEllipse(c)
                    else:
                        M = cv2.moments(c); ecx, ecy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    inside = masks[ek][int(np.clip(ecy, 0, H - 1)), int(np.clip(ecx, 0, W - 1))] > 0
                    ctr = (ecx, ecy)
                    status = "ok" if inside else "centre_outside_oval"
                    col = (0, 0, 255) if ek == "R" else (0, 200, 255)
                    ov[disc_full > 0] = (0.55 * ov[disc_full > 0] + np.array([0, 0, 120])).astype(np.uint8)
                    cv2.circle(ov, (int(ecx), int(ecy)), 5, col, -1)
        rec[f"{ek}_x"], rec[f"{ek}_y"] = ctr
        rec[f"{ek}_status"], rec[f"{ek}_disc_frac"] = status, round(disc_frac, 4)
        cv2.ellipse(ov, (int(o["ellipse"][0][0]), int(o["ellipse"][0][1])),
                    (int(o["ellipse"][1][0] / 2), int(o["ellipse"][1][1] / 2)),
                    o["ellipse"][2], 0, 360, (0, 255, 0), 1)
    rows.append(rec)
    cv2.putText(ov, f"f{fi}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    vw.write(cv2.resize(ov, (W // 2, H // 2)))
cap.release(); vw.release()

# --- write trace CSV ---
cols = ["frame"]
for ek in ovals:
    cols += [f"{ek}_x", f"{ek}_y", f"{ek}_status", f"{ek}_disc_frac"]
import csv as _csv
with open(OUTDIR / "centre_trace.csv", "w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in cols})

# --- plots ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for ek in ovals:
    fr = [r["frame"] for r in rows]
    xs = [r[f"{ek}_x"] for r in rows]
    ys = [r[f"{ek}_y"] for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(fr, xs, ".-", ms=2, lw=0.6); ax[0].set_ylabel(f"{ek} iris x (px)")
    ax[1].plot(fr, ys, ".-", ms=2, lw=0.6); ax[1].set_ylabel(f"{ek} iris y (px)")
    ax[1].set_xlabel("frame")
    ax[0].set_title(f"EllSeg orbit-constrained iris-centre trace — {ek} eye — {CLIP}")
    fig.tight_layout(); fig.savefig(OUTDIR / f"centre_trace_{ek}.png", dpi=110); plt.close(fig)

# --- coverage summary + check vs the one clinician gold centre (frame 501, image-side L) ---
def cov(ek):
    n = sum(1 for r in rows if r.get(f"{ek}_status") == "ok")
    return n, len(rows)
summary = {ek: {"ok": cov(ek)[0], "total": cov(ek)[1]} for ek in ovals}
gold = OUTBASE / "manual_iris_annotations.json"
if gold.exists():
    g = json.load(open(gold))
    for a in g.get("annotations", []):
        ek = a["image_side_eye"]; fn = a["frame_number"]
        if ek in ovals and fn < len(rows):
            gx, gy = a["iris_centre"]; pr = rows[fn]
            px, py = pr.get(f"{ek}_x"), pr.get(f"{ek}_y")
            if isinstance(px, float) and not math.isnan(px):
                d = math.hypot(px - gx, py - gy)
                summary.setdefault("gold_check", []).append(
                    {"frame": fn, "eye": ek, "pred": [round(px, 1), round(py, 1)],
                     "clinician": [gx, gy], "dist_px": round(d, 1)})
json.dump(summary, open(OUTDIR / "summary.json", "w"), indent=2)
print("coverage / gold check:", json.dumps(summary, indent=2))
print("outputs ->", OUTDIR)
