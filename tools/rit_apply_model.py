"""Apply the trained RIT learned segmenter to ANY video that has a RIT Orbit Lock, and render an overlay.
Cross-video generalization test: the model was trained on the vestibular-neuritis clip; here we run it,
unchanged, inside a clinician-marked orbit lock on a different clip.

Reuses rit_learning_workbench.apply_clip (identical features + overlay), so nothing about the model changes.

Usage: py -3.14 tools/rit_apply_model.py "<video stem>" [model.pkl]
  e.g. py -3.14 tools/rit_apply_model.py "gaze-evoked nystagmus in pontine glioma"
"""
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rit_learning_workbench as wb   # module-level is side-effect-free (only reads files inside main())

ROOT = Path(__file__).resolve().parent.parent
stem = sys.argv[1] if len(sys.argv) > 1 else "gaze-evoked nystagmus in pontine glioma"
model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    ROOT / "outputs/nystagmus at rest to left in right vestibular neuritis_tracked/RIT_learned_segmenter/model.pkl")

OUT = ROOT / "outputs" / f"{stem}_tracked"
video = str(ROOT / "samples" / f"{stem}.mp4")
csv_path = OUT / "_rit_orbit_lock_probe" / "orbit_lock_points.csv"
assert csv_path.exists(), f"no orbit lock for {stem} (run rit_orbit_lock_from_marks.py first)"
assert Path(video).exists(), f"missing video {video}"

print(f"loading model {model_path.name} ({model_path.stat().st_size/1e6:.0f} MB) ...")
clf = joblib.load(model_path)["model"]

orb = defaultdict(dict)
for r in csv.DictReader(open(csv_path, encoding="utf-8")):
    orb[int(r["frame"])][r["eye"]] = np.array(json.loads(r["contour_json"]), np.float32)

run_dir = OUT / "_rit_learned_apply"
(run_dir / "overlays").mkdir(parents=True, exist_ok=True)

t0 = time.time()
rows, preds, dims = wb.classify_status(*wb.apply_clip(clf, video, orb, run_dir))

with open(run_dir / "points.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=["frame", "eye", "cx", "cy", "major", "minor", "angle",
                                       "iris_area", "sclera_area", "confidence", "status"])
    w.writeheader(); w.writerows(rows)

sc = defaultdict(int)
for r in rows:
    sc[r["status"]] += 1
summary = dict(video=stem, frames=int(max(r["frame"] for r in rows)), n_records=len(rows),
               status_counts=dict(sc),
               mean_confidence=round(float(np.mean([r["confidence"] for r in rows if r["confidence"]])), 3),
               model=str(model_path))
json.dump(summary, open(run_dir / "summary.json", "w"), indent=1)
print(f"applied to {summary['frames']} frames in {time.time()-t0:.0f}s")
print("status:", dict(sc), " mean_conf:", summary["mean_confidence"])
print("overlay ->", run_dir / "overlays" / "iris_sclera_overlay.mp4")
