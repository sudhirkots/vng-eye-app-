"""Evaluate the automatic iris detector against Dr. K's ground-truth masks (gold-standard test set).

Reuses the detector's OWN fit_iris (imported from rit_iris.py) so the evaluation tracks the real code.
For each ground-truth unit it runs the detector cold-start (seed = approved iris position) on the full
frame, crops the predicted almond/sclera/iris to the same crop, and reports vs the hand-marked masks:
  - IoU for iris / sclera / almond
  - iris centre error (full-res pixels)
  - iris-outside-almond fraction  (predicted iris on skin/lid/lash -> should be ~0)
  - fill check: (sclera + iris) vs almond IoU, for prediction and for the ground truth itself
Prints a per-unit table + aggregate means, and writes RIT_ground_truth/evaluation.json.

Does NOT tune the detector. Does NOT commit. Usage: py -3.14 tools/rit_evaluate_iris_masks.py [out_dir]
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rit_iris  # noqa: E402  (functions only; its render loop is guarded)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
GT = OUT / "RIT_ground_truth"
MASKS = GT / "masks"
OLP = OUT / "_rit_orbit_lock_probe/orbit_lock_points.csv"
appr = json.load(open(OUT / "approved_landmarks.json", encoding="utf-8"))
video = appr.get("video")
if not video or not Path(video).exists():
    stem = OUT.name[:-8] if OUT.name.endswith("_tracked") else OUT.name
    video = str(Path("samples") / (stem + ".mp4"))
iris0 = {ek: (float(v["x"]), float(v["y"]), float(v["radius"]))
         for ek, v in (appr.get("iris") or {}).items() if v}
meta = json.load(open(GT / "meta.json", encoding="utf-8")) if (GT / "meta.json").exists() else {}

orb = defaultdict(dict)
for r in csv.DictReader(open(OLP, encoding="utf-8")):
    try:
        orb[int(r["frame"])][r["eye"]] = np.array(json.loads(r["contour_json"]), np.float32)
    except Exception:
        pass


def iou(a, b):
    a = a > 0; b = b > 0
    u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else (1.0 if int((a | b).sum()) == 0 else 0.0)


def centroid(m):
    ys, xs = np.where(m > 0)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def predict(frame, ek):
    """Run the real detector cold-start for one eye -> full-frame (almond, sclera, iris) masks + centre."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    c = None
    if fnum in orb and ek in orb[fnum]:
        c = orb[fnum][ek]
    if c is None:
        return None
    held_r = iris0.get(ek, (0, 0, 30))[2]
    om = np.zeros((H, W), np.uint8); cv2.fillPoly(om, [c.astype(np.int32)], 255)
    seed = (iris0[ek][0], iris0[ek][1]) if ek in iris0 else None
    f = rit_iris.fit_iris(frame, gray, om, held_r, seed)
    if f is None:
        return None
    alm = f["almond"]; scl = f["scl"]
    em = np.zeros((H, W), np.uint8)
    cv2.ellipse(em, (int(f["cx"]), int(f["cy"])), (int(f["major"]), int(f["minor"])), f["tilt"], 0, 360, 255, -1)
    iris = cv2.bitwise_and(em, alm)
    return dict(almond=alm, sclera=scl, iris=iris, cx=f["cx"], cy=f["cy"])


units = sorted(p.name[:-9] for p in MASKS.glob("*_iris.png")) if MASKS.exists() else []
if not units:
    print("No ground-truth masks yet. Paint masks in RIT_ground_truth/to_mark/, run",
          "tools/rit_extract_ground_truth.py, then re-run this.")
    sys.exit(0)

cap = cv2.VideoCapture(video)
rows = []
print(f"{'unit':13} {'irisIoU':>7} {'sclIoU':>7} {'almIoU':>7} {'ctrErr':>7} {'outAlm':>7} {'fillP':>6} {'fillGT':>6}")
for uid in units:
    m = meta.get(uid)
    gi = cv2.imread(str(MASKS / f"{uid}_iris.png"), 0)
    gs = cv2.imread(str(MASKS / f"{uid}_sclera.png"), 0)
    ga = cv2.imread(str(MASKS / f"{uid}_almond.png"), 0)
    if m is None or gi is None or gs is None or ga is None:
        continue
    fnum = m["frame"]; ek = m["eye"]; X0, Y0, sc = m["X0"], m["Y0"], m["scale"]
    cw, ch = m["crop_w"], m["crop_h"]; x1 = X0 + int(round(cw / sc)); y1 = Y0 + int(round(ch / sc))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum - 1); ok, frame = cap.read()
    if not ok:
        continue
    pred = predict(frame, ek)
    fill_gt = iou(cv2.bitwise_or(gi, gs), ga)
    if pred is None:                                                # detector GAP on this frame
        rows.append(dict(unit=uid, iris_iou=0.0, sclera_iou=0.0, almond_iou=0.0, ctr_err=None,
                         out_alm=None, fill_pred=None, fill_gt=round(fill_gt, 3), gap=True))
        print(f"{uid:13} {'GAP':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>6} {fill_gt:6.2f}")
        continue

    def crp(mask):
        return cv2.resize(mask[Y0:y1, X0:x1], (cw, ch), interpolation=cv2.INTER_NEAREST)
    pi, ps, pa = crp(pred["iris"]), crp(pred["sclera"]), crp(pred["almond"])
    ii, si, ai = iou(pi, gi), iou(ps, gs), iou(pa, ga)
    pc = ((pred["cx"] - X0) * sc, (pred["cy"] - Y0) * sc)           # pred centre in crop coords
    gc = centroid(gi)
    cerr = (float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])) / sc) if gc else None   # full-res px
    pi_n = int((pi > 0).sum())                                     # predicted-iris pixels on skin/lid/lash
    outalm = (int(((pi > 0) & (ga == 0)).sum()) / pi_n) if pi_n else 0.0   # = outside the (GT) almond
    fill_pred = iou(cv2.bitwise_or(pi, ps), pa)
    rows.append(dict(unit=uid, iris_iou=round(ii, 3), sclera_iou=round(si, 3), almond_iou=round(ai, 3),
                     ctr_err=round(cerr, 1) if cerr is not None else None, out_alm=round(outalm, 3),
                     fill_pred=round(fill_pred, 3), fill_gt=round(fill_gt, 3), gap=False))
    ce = f"{cerr:6.1f}" if cerr is not None else "   -  "
    print(f"{uid:13} {ii:7.2f} {si:7.2f} {ai:7.2f} {ce:>7} {outalm:7.2f} {fill_pred:6.2f} {fill_gt:6.2f}")
cap.release()

ev = [r for r in rows if not r["gap"]]
def mean(key):
    xs = [r[key] for r in ev if r[key] is not None]
    return round(sum(xs) / len(xs), 3) if xs else None
n_out = sum(1 for r in ev if r["out_alm"] is not None and r["out_alm"] > 0.05)
summary = dict(n_units=len(rows), n_gap=sum(1 for r in rows if r["gap"]),
               mean_iris_iou=mean("iris_iou"), mean_sclera_iou=mean("sclera_iou"),
               mean_almond_iou=mean("almond_iou"), mean_ctr_err_px=mean("ctr_err"),
               mean_fill_pred=mean("fill_pred"), mean_fill_gt=mean("fill_gt"),
               n_iris_outside_almond=n_out)
print("\n== AGGREGATE ==")
for k, v in summary.items():
    print(f"  {k}: {v}")
json.dump(dict(summary=summary, per_unit=rows), open(GT / "evaluation.json", "w", encoding="utf-8"), indent=1)
print(f"\nwrote {GT/'evaluation.json'}")
