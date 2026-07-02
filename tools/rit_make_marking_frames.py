"""Make ~40 per-eye marking frames for ANY clip that has a RIT Orbit Lock, so the clinician can paint
iris(red)/sclera(blue) ground truth (tools/rit_paint_tool.html). Frames are spread across the clip (varied
gaze), with a few drawn from the frames the current model was least sure about (active learning), if an
apply run exists. Crops around the clinician's orbit oval, ×2 upscaled -- same format as the paint workflow.

Usage: py -3.14 tools/rit_make_marking_frames.py "<video stem>" [n_frames=20]
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
stem = sys.argv[1] if len(sys.argv) > 1 else "gaze-evoked nystagmus in pontine glioma"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20               # distinct frames -> 2N per-eye units

OUT = ROOT / "outputs" / f"{stem}_tracked"
video = str(ROOT / "samples" / f"{stem}.mp4")
OLP = OUT / "_rit_orbit_lock_probe" / "orbit_lock_points.csv"
GT = OUT / "RIT_ground_truth"
(GT / "raw").mkdir(parents=True, exist_ok=True); (GT / "to_mark").mkdir(parents=True, exist_ok=True)

orb = defaultdict(dict)
for r in csv.DictReader(open(OLP, encoding="utf-8")):
    orb[int(r["frame"])][r["eye"]] = np.array(json.loads(r["contour_json"]), np.float32)
frames_avail = sorted(orb)
nfr = max(frames_avail)

# spread of frames across the clip (varied gaze); add up to 6 model-uncertain frames if an apply exists
picks = sorted(set(int(round(f)) for f in np.linspace(frames_avail[0], nfr, N)))
apply_pts = OUT / "_rit_learned_apply" / "points.csv"
if apply_pts.exists():
    unc = defaultdict(float)
    for r in csv.DictReader(open(apply_pts, encoding="utf-8")):
        if r["status"] in ("needs_human_review", "learned_uncertain"):
            unc[int(r["frame"])] += 1 - float(r["confidence"] or 0)
    for f in sorted(unc, key=unc.get, reverse=True)[:6]:
        picks.append(f)
picks = sorted(set(f for f in picks if f in orb))[:N + 6]

cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
meta = {}; made = 0
for fnum in picks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum - 1); ok, frame = cap.read()
    if not ok:
        continue
    for ek in ("R", "L"):
        c = orb.get(fnum, {}).get(ek)
        if c is None:
            continue
        x0, y0 = c.min(0); x1, y1 = c.max(0)
        mx = (x1 - x0) * 0.30; my = (y1 - y0) * 0.55
        X0 = max(0, int(x0 - mx)); Y0 = max(0, int(y0 - my))
        X1 = min(W, int(x1 + mx)); Y1 = min(H, int(y1 + my))
        crop = frame[Y0:Y1, X0:X1]
        if crop.size == 0:
            continue
        z = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        uid = f"f{fnum:04d}_{ek}"
        cv2.imwrite(str(GT / "raw" / f"{uid}.png"), z)
        cv2.imwrite(str(GT / "to_mark" / f"{uid}.png"), z)
        meta[uid] = dict(frame=fnum, eye=ek, X0=int(X0), Y0=int(Y0), scale=2.0,
                         crop_w=int(z.shape[1]), crop_h=int(z.shape[0]),
                         orbit_crop=((c - np.array([X0, Y0])) * 2.0).tolist())
        made += 1
cap.release()
json.dump(meta, open(GT / "meta.json", "w", encoding="utf-8"), indent=1)
(GT / "MARKING_INSTRUCTIONS.md").write_text(
    "# RIT ground-truth marking\n\nPaint on each image in `to_mark/` (tools/rit_paint_tool.html):\n"
    "- IRIS   -> pure RED  (255,0,0) = the dark round iris disc / visible sliver\n"
    "- SCLERA -> pure BLUE (0,0,255) = the white/pink between the lids (everything in the opening that is not iris)\n"
    "Together red+blue fill the eye opening; leave lids/lashes/skin blank.\n"
    "Export the .zip, then run the RIT Learning Workbench on this clip.\n", encoding="utf-8")
print(f"made {made} per-eye units from {len(picks)} frames -> {GT}")
print(f"  paint them in tools/rit_paint_tool.html (load from {GT/'to_mark'})")
