"""RIT learned pixel segmenter -- candidate replacement for the OpenCV rule detector (rit_iris.py).

Trains a lightweight per-pixel RandomForest (0=outside, 1=iris, 2=sclera) directly from Dr. K's
ground-truth masks (RIT_ground_truth/masks/), restricted to the moving RIT Orbit Lock search region
(meta.json orbit_crop, already in mask-pixel coordinates -- so no extra alignment step is needed).

Evaluated by K-fold cross-validation over WHOLE UNITS (each of the 39 clean units is held out
exactly once and scored on a model that never saw its pixels) -- this is the "held-out frame
validation" the OpenCV baseline (rit_evaluate_iris_masks.py) does not need, since it isn't trained.

Reports the same metrics as the OpenCV baseline for a direct, apples-to-apples comparison:
  iris IoU, sclera IoU, almond IoU, iris-centre error (full-res px), iris-outside-almond fraction.

Excludes incomplete ground truth (f0090_L: sclera only, no iris; f0135_L: unpainted).

Does NOT use U-Net. Does NOT wire into clinical RIT. Does NOT commit.
Usage: py -3.14 tools/rit_learned_segmenter.py [out_dir]
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
GT = OUT / "RIT_ground_truth"
MASKS = GT / "masks"; RAW = GT / "raw"
REV = GT / "learned_segmenter"; (REV / "overlays_worst").mkdir(parents=True, exist_ok=True)
(REV / "overlays_best").mkdir(parents=True, exist_ok=True)

EXCLUDE = {"f0090_L", "f0135_L"}   # incomplete marks: no iris paint / not painted at all
N_ESTIMATORS, MAX_DEPTH, N_FOLDS, SEED = 200, 16, 5, 0
PER_CLASS_CAP = 3000               # max samples of each class drawn per training unit

meta = json.load(open(GT / "meta.json", encoding="utf-8"))
units = sorted(p.name[:-9] for p in MASKS.glob("*_iris.png") if p.name[:-9] not in EXCLUDE)


# ---------- data ----------
def load_unit(uid):
    img = cv2.imread(str(RAW / f"{uid}.png"))
    gi = cv2.imread(str(MASKS / f"{uid}_iris.png"), 0)
    gs = cv2.imread(str(MASKS / f"{uid}_sclera.png"), 0)
    ga = cv2.imread(str(MASKS / f"{uid}_almond.png"), 0)
    poly = np.array(meta[uid]["orbit_crop"], np.float32).astype(np.int32)
    h, w = img.shape[:2]
    roi = np.zeros((h, w), np.uint8); cv2.fillPoly(roi, [poly], 255)
    label = np.zeros((h, w), np.uint8)
    label[gs > 0] = 2
    label[gi > 0] = 1                                  # iris wins on the rare overlap pixel
    return dict(uid=uid, img=img, roi=roi, label=label, iris_gt=gi, sclera_gt=gs, almond_gt=ga,
                scale=float(meta[uid]["scale"]))


def feature_maps(img):
    b, g, r = cv2.split(img.astype(np.float32) / 255.0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, ss, vv = hsv[:, :, 0] / 179.0, hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return np.stack([b, g, r, hh, ss, vv, gray, blur, lap], axis=-1)   # (H,W,9)


FEATURE_NAMES = ["B", "G", "R", "H", "S", "V", "gray", "blur3", "|lap|", "dx", "dy"]


def sample_vectors(u, idxs=None):
    """Feature rows for pixels inside the orbit-lock ROI. idxs=None -> ALL roi pixels (for prediction)."""
    fm = feature_maps(u["img"])
    ys_all, xs_all = np.where(u["roi"] > 0)
    ys, xs = (ys_all, xs_all) if idxs is None else (ys_all[idxs], xs_all[idxs])
    cy, cx = ys_all.mean(), xs_all.mean()
    scale = max(1.0, float(np.hypot(ys_all - cy, xs_all - cx).mean()))
    dy = (ys - cy) / scale; dx = (xs - cx) / scale
    base = fm[ys, xs]
    feats = np.concatenate([base, dx[:, None], dy[:, None]], axis=1).astype(np.float32)
    return feats, ys, xs


def training_rows(u, rng):
    label = u["label"]; roi = u["roi"] > 0
    rows_f, rows_y = [], []
    for cls in (0, 1, 2):
        ys, xs = np.where(roi & (label == cls))
        if len(ys) == 0:
            continue
        if len(ys) > PER_CLASS_CAP:
            pick = rng.choice(len(ys), PER_CLASS_CAP, replace=False)
            ys, xs = ys[pick], xs[pick]
        rows_y.append(np.full(len(ys), cls, np.uint8))
        rows_f.append((ys, xs))
    fm = feature_maps(u["img"])
    ys_all, xs_all = np.where(roi)
    cy, cx = ys_all.mean(), xs_all.mean()
    scale = max(1.0, float(np.hypot(ys_all - cy, xs_all - cx).mean()))
    feats_list = []
    for ys, xs in rows_f:
        dy = (ys - cy) / scale; dx = (xs - cx) / scale
        base = fm[ys, xs]
        feats_list.append(np.concatenate([base, dx[:, None], dy[:, None]], axis=1).astype(np.float32))
    return np.concatenate(feats_list, axis=0), np.concatenate(rows_y, axis=0)


# ---------- metrics (mirrors rit_evaluate_iris_masks.py) ----------
def iou(a, b):
    a = a > 0; b = b > 0
    u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else 1.0


def centroid(m):
    ys, xs = np.where(m > 0)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def eval_unit(u, pred_label):
    pi = (pred_label == 1).astype(np.uint8) * 255
    ps = (pred_label == 2).astype(np.uint8) * 255
    pa = cv2.bitwise_or(pi, ps)
    ii, si, ai = iou(pi, u["iris_gt"]), iou(ps, u["sclera_gt"]), iou(pa, u["almond_gt"])
    pc, gc = centroid(pi), centroid(u["iris_gt"])
    cerr = (float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])) / u["scale"]) if (pc and gc) else None
    pi_n = int((pi > 0).sum())
    outalm = (int(((pi > 0) & (u["almond_gt"] == 0)).sum()) / pi_n) if pi_n else 0.0
    fill_pred = iou(pa, u["almond_gt"])
    return dict(unit=u["uid"], iris_iou=round(ii, 3), sclera_iou=round(si, 3), almond_iou=round(ai, 3),
                ctr_err=round(cerr, 1) if cerr is not None else None, out_alm=round(outalm, 3),
                fill_pred=round(fill_pred, 3)), pi, ps


# ---------- cross-validation ----------
print(f"loading {len(units)} clean units ...")
data = {uid: load_unit(uid) for uid in units}
rng = np.random.default_rng(SEED)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
uarr = np.array(units)

oof_rows = []
oof_pred = {}
for fold, (tr_idx, te_idx) in enumerate(kf.split(uarr)):
    train_units = uarr[tr_idx]; test_units = uarr[te_idx]
    Xs, ys = [], []
    for uid in train_units:
        f, y = training_rows(data[uid], rng)
        Xs.append(f); ys.append(y)
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf.fit(X, y)
    print(f"fold {fold + 1}/{N_FOLDS}: trained on {len(train_units)} units ({len(X)} px), "
          f"testing {len(test_units)} units")
    for uid in test_units:
        u = data[uid]
        feats, ys_px, xs_px = sample_vectors(u)
        pred = clf.predict(feats)
        pred_label = np.zeros(u["roi"].shape, np.uint8)
        pred_label[ys_px, xs_px] = pred
        row, pi, ps = eval_unit(u, pred_label)
        oof_rows.append(row)
        oof_pred[uid] = (pi, ps)

# final model trained on ALL 39 clean units, for future reuse (not used in the metrics above)
Xs, ys = [], []
for uid in units:
    f, y = training_rows(data[uid], rng)
    Xs.append(f); ys.append(y)
X_all = np.concatenate(Xs, axis=0); y_all = np.concatenate(ys, axis=0)
final_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                    class_weight="balanced", random_state=SEED, n_jobs=-1)
final_clf.fit(X_all, y_all)
try:
    import joblib
    joblib.dump(dict(model=final_clf, feature_names=FEATURE_NAMES), REV / "model_rf.joblib")
except Exception as e:
    print("model save skipped:", e)

# ---------- aggregate + compare vs OpenCV baseline ----------
def mean(rows, key):
    xs = [r[key] for r in rows if r[key] is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


summary = dict(n_units=len(oof_rows), n_folds=N_FOLDS,
               mean_iris_iou=mean(oof_rows, "iris_iou"), mean_sclera_iou=mean(oof_rows, "sclera_iou"),
               mean_almond_iou=mean(oof_rows, "almond_iou"), mean_ctr_err_px=mean(oof_rows, "ctr_err"),
               mean_fill_pred=mean(oof_rows, "fill_pred"),
               n_iris_outside_almond=sum(1 for r in oof_rows if r["out_alm"] and r["out_alm"] > 0.05))

baseline_path = GT / "evaluation.json"
baseline = json.load(open(baseline_path, encoding="utf-8")) if baseline_path.exists() else None
baseline_by_unit = {r["unit"]: r for r in (baseline["per_unit"] if baseline else [])}

comparison = []
for r in oof_rows:
    b = baseline_by_unit.get(r["unit"])
    comparison.append(dict(unit=r["unit"],
                            learned_iris_iou=r["iris_iou"], opencv_iris_iou=(None if not b or b.get("gap") else b["iris_iou"]),
                            learned_sclera_iou=r["sclera_iou"], opencv_sclera_iou=(None if not b or b.get("gap") else b["sclera_iou"]),
                            learned_ctr_err=r["ctr_err"], opencv_ctr_err=(None if not b or b.get("gap") else b["ctr_err"]),
                            opencv_gap=bool(b and b.get("gap"))))

print("\n== HELD-OUT (learned model, out-of-fold) ==")
print(f"{'unit':13} {'irisIoU':>7} {'sclIoU':>7} {'almIoU':>7} {'ctrErr':>7} {'outAlm':>7}")
for r in sorted(oof_rows, key=lambda r: r["unit"]):
    ce = f"{r['ctr_err']:7.1f}" if r["ctr_err"] is not None else "   -   "
    print(f"{r['unit']:13} {r['iris_iou']:7.2f} {r['sclera_iou']:7.2f} {r['almond_iou']:7.2f} {ce} {r['out_alm']:7.2f}")

print("\n== AGGREGATE (learned model) ==")
for k, v in summary.items():
    print(f"  {k}: {v}")

if baseline:
    bsum = baseline["summary"]
    print("\n== OpenCV baseline (for reference) ==")
    for k in ("mean_iris_iou", "mean_sclera_iou", "mean_almond_iou", "mean_ctr_err_px", "n_iris_outside_almond"):
        print(f"  {k}: {bsum.get(k)}  (learned: {summary.get(k)})")

# ---------- best / worst 10 ----------
def composite(r):
    return 0.5 * r["iris_iou"] + 0.3 * r["sclera_iou"] + 0.2 * r["almond_iou"]


ranked = sorted(oof_rows, key=composite, reverse=True)
best10 = ranked[:10]; worst10 = ranked[-10:]

print("\n== BEST 10 (learned model) ==")
for r in best10:
    print(f"  {r['unit']:13} score={composite(r):.3f} iris={r['iris_iou']:.2f} scl={r['sclera_iou']:.2f}")
print("\n== WORST 10 (learned model) ==")
for r in worst10:
    print(f"  {r['unit']:13} score={composite(r):.3f} iris={r['iris_iou']:.2f} scl={r['sclera_iou']:.2f}")


def overlay(u, pi, ps):
    ov = u["img"].copy()
    ga = u["almond_gt"]
    cnts, _ = cv2.findContours(ga, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 220, 0), 2)                       # GT almond outline (green)
    ov[ps > 0] = (0.55 * ov[ps > 0] + np.array([200, 90, 0])).astype(np.uint8)   # predicted sclera
    ov[pi > 0] = (0.55 * ov[pi > 0] + np.array([0, 0, 200])).astype(np.uint8)    # predicted iris
    return ov


for tag, rows in (("best", best10), ("worst", worst10)):
    for r in rows:
        u = data[r["unit"]]; pi, ps = oof_pred[r["unit"]]
        ov = overlay(u, pi, ps)
        cv2.putText(ov, f"{r['unit']} iris={r['iris_iou']:.2f} scl={r['sclera_iou']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(ov, f"{r['unit']} iris={r['iris_iou']:.2f} scl={r['sclera_iou']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imwrite(str(REV / f"overlays_{tag}" / f"{r['unit']}.png"), ov)

# ---------- review video: all 39 held-out overlays, sorted by unit ----------
CANVAS = (960, 640)
writer = cv2.VideoWriter(str(REV / "learned_overlay_review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, CANVAS)
row_by_unit = {r["unit"]: r for r in oof_rows}
for uid in sorted(oof_pred):
    u = data[uid]; pi, ps = oof_pred[uid]
    ov = overlay(u, pi, ps)
    r = row_by_unit[uid]
    disp = cv2.resize(ov, CANVAS)
    txt = f"{uid}  iris={r['iris_iou']:.2f} scl={r['sclera_iou']:.2f} alm={r['almond_iou']:.2f} ctrErr={r['ctr_err']}"
    cv2.putText(disp, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(disp, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    for _ in range(2):    # ~2s hold per frame at 1fps writer
        writer.write(disp)
writer.release()

out = dict(summary=summary, per_unit=oof_rows, comparison_vs_opencv=comparison,
           best10=[r["unit"] for r in best10], worst10=[r["unit"] for r in worst10],
           excluded_units=sorted(EXCLUDE), n_training_units_total=len(units),
           model=dict(type="RandomForestClassifier", n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                      features=FEATURE_NAMES, cv="KFold(5, shuffle, seed=0) over whole units"))
json.dump(out, open(REV / "learned_evaluation.json", "w", encoding="utf-8"), indent=1)
print(f"\nwrote {REV/'learned_evaluation.json'}")
print(f"wrote {REV/'learned_overlay_review.mp4'}")
print(f"wrote {REV/'model_rf.joblib'}")
