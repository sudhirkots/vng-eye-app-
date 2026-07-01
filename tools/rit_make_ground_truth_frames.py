"""Generate clean per-eye crops for MANUAL ground-truth marking (RIT gold-standard test set).

Selects ~40 (frame, eye) units from the vestibular-neuritis clip -- full iris / partial / extreme medial
gaze / lash-brow-shadow / the flagged frames / gap frames / good frames -- and writes, per unit:
  RIT_ground_truth/raw/<id>.png       -- untouched reference crop
  RIT_ground_truth/to_mark/<id>.png   -- clean copy to PAINT the masks on
  RIT_ground_truth/meta.json          -- frame, eye, crop transform (X0,Y0,scale), orbit contour
Plus MARKING_INSTRUCTIONS.md. Nothing is tuned or committed.

Usage: py -3.14 tools/rit_make_ground_truth_frames.py [out_dir]
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
OLP = OUT / "_rit_orbit_lock_probe/orbit_lock_points.csv"
PTS = OUT / "_rit_iris/iris_points.csv"
GT = OUT / "RIT_ground_truth"
(GT / "raw").mkdir(parents=True, exist_ok=True)
(GT / "to_mark").mkdir(parents=True, exist_ok=True)
appr = json.load(open(OUT / "approved_landmarks.json", encoding="utf-8"))
video = appr.get("video")
if not video or not Path(video).exists():
    stem = OUT.name[:-8] if OUT.name.endswith("_tracked") else OUT.name
    video = str(Path("samples") / (stem + ".mp4"))

orb = defaultdict(dict)
for r in csv.DictReader(open(OLP, encoding="utf-8")):
    try:
        orb[int(r["frame"])][r["eye"]] = np.array(json.loads(r["contour_json"]), np.float32)
    except Exception:
        pass

# --- choose units: the flagged frames + a category spread (both eyes) + detected gap frames -----------
flagged = [187, 227, 267, 311, 341, 378, 382, 460]           # requested specific frames
spread = [60, 90, 130, 195, 262, 300, 348, 353, 438, 540]    # full / partial / lash-shadow variety
units = []
for f in flagged + spread:
    for ek in ("L", "R"):
        units.append((f, ek))
# gap frames from the current detector output (frames it failed on)
gaps = []
if PTS.exists():
    for r in csv.DictReader(open(PTS, encoding="utf-8")):
        if r.get("cx", "") == "":
            gaps.append((int(r["frame"]), r["eye"]))
for g in gaps[:: max(1, len(gaps) // 6)][:6]:                 # up to ~6 gap units, spread out
    if g not in units:
        units.append(g)
units = sorted(set(units))

cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
by_frame = defaultdict(list)
for f, ek in units:
    by_frame[f].append(ek)
meta = {}
made = 0
for fnum in sorted(by_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum - 1)                # frame N (1-based overlay) == index N-1
    ok, frame = cap.read()
    if not ok:
        continue
    for ek in by_frame[fnum]:
        c = orb.get(fnum, {}).get(ek)
        if c is None:
            continue
        x0, y0 = c.min(0); x1, y1 = c.max(0); mx = (x1 - x0) * 0.30; my = (y1 - y0) * 0.55
        X0 = max(0, int(x0 - mx)); Y0 = max(0, int(y0 - my))
        X1 = min(W, int(x1 + mx)); Y1 = min(H, int(y1 + my))
        crop = frame[Y0:Y1, X0:X1]
        if crop.size == 0:
            continue
        z = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        uid = f"f{fnum:04d}_{ek}"
        cv2.imwrite(str(GT / "raw" / f"{uid}.png"), z)
        cv2.imwrite(str(GT / "to_mark" / f"{uid}.png"), z)
        cc = ((c - np.array([X0, Y0])) * 2.0).tolist()        # orbit contour in crop coords (reference)
        meta[uid] = dict(frame=fnum, eye=ek, X0=int(X0), Y0=int(Y0), scale=2.0,
                         crop_w=int(z.shape[1]), crop_h=int(z.shape[0]), orbit_crop=cc)
        made += 1
cap.release()
json.dump(meta, open(GT / "meta.json", "w", encoding="utf-8"), indent=1)

INSTR = """# RIT ground-truth marking

Paint THREE masks on each image in `to_mark/` (edit in place, or save over the same name).
Use OPAQUE, pure colours (MS Paint bucket/brush is fine):

- IRIS   -> pure RED   (255, 0, 0)      = the dark iris+pupil disc (the part that is NOT white sclera)
- SCLERA -> pure BLUE  (0, 0, 255)      = all the white/pink between the eyelids (incl. the pink tails)
- ALMOND -> (optional) pure GREEN outline (0,255,0). If you skip it, the almond is taken as iris + sclera.

Rules (same as the method):
- Together, RED + BLUE should fill the whole almond (eye opening) -- nothing left over, nothing outside.
- Do NOT paint eyelid skin, lashes, or brow. Leave them unpainted (they are outside the almond).
- The iris is round; paint the visible dark disc/sliver. The sclera is everything else white/pink.

When done, run:  py -3.14 tools/rit_extract_ground_truth.py
Then:            py -3.14 tools/rit_evaluate_iris_masks.py
"""
(GT / "MARKING_INSTRUCTIONS.md").write_text(INSTR, encoding="utf-8")
print(f"made {made} units -> {GT}\\nflagged+spread+gaps; see MARKING_INSTRUCTIONS.md")
