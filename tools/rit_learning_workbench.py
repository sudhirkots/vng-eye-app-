"""RIT Learning Workbench -- self-testing loop for the learned iris/sclera segmenter.

ACTIVE PATH:  RIT Orbit Lock  +  clinician-marked ground truth  +  lightweight learned segmenter.
RETAINED:     RIT Orbit Lock (moving anatomical search region).
DEPRECATED:   the OpenCV iris/sclera/almond RULE detector (kept only as a historical benchmark;
              its Almond-Sclera-Iris rules are now ANNOTATION rules, not threshold code).

One command runs the whole loop end-to-end and stops only where a human is needed:
  A validate clinician annotations        (auto-continue if valid enough, else stop with a correction list)
  B train a CPU pixel classifier          (ExtraTrees; held-out-FRAME cross-validation)
  C evaluate on held-out marked frames     (iris/sclera/almond IoU, centre err, bleed, overlap; vs OpenCV)
  D apply to the WHOLE clip                (masks, centres, ellipse, confidence, status; overlay mp4)
  E auto-select the worst frames           (self-consistency anomalies) and build a correction set
  (F self-improvement: re-run after corrections are painted back -> run_002 vs run_001)

Each invocation writes a fresh numbered run under RIT_learning_workbench/runs/run_NNN (never overwrites).

Does NOT tune OpenCV. Does NOT patch the old detector. Does NOT detect nystagmus.
Does NOT wire into clinical RIT. Does NOT commit.
Usage: py -3.14 tools/rit_learning_workbench.py [out_dir]
"""
import csv, json, sys, shutil, math, time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------------- paths / config
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
GT = OUT / "RIT_ground_truth"
MASKS = GT / "masks"; RAW = GT / "raw"; TOMARK = GT / "to_mark"
WB = OUT / "RIT_learning_workbench"; RUNS = WB / "runs"
PROMOTED = OUT / "RIT_learned_segmenter"           # top-level "current model" folder
OLP = OUT / "_rit_orbit_lock_probe/orbit_lock_points.csv"
BASELINE = GT / "evaluation.json"                  # deprecated OpenCV rule detector, historical only

MODEL = dict(kind="ExtraTreesClassifier", n_estimators=300, max_depth=18, min_samples_leaf=4,
             class_weight="balanced", random_state=0)
N_FOLDS = 5
PER_CLASS_CAP = 3500          # balanced sampling: max px of each class per training unit
MIN_MARK_PX = 120             # a real iris/sclera mark has at least this many painted px
APPLY_MAXDIM = 150            # orbit bbox is downscaled to <= this (px) for full-clip application
# feature texture scales are set RELATIVE to orbit size, so a model trained on the 2x marked crops
# applies unchanged to the downscaled full-clip crops:
SIG_SMALL, SIG_LARGE = 0.015, 0.05
FEATURE_NAMES = ["B", "G", "R", "H", "S", "V", "L*", "a*", "b*", "gray",
                 "blur_s", "blur_l", "|lap|", "dx", "dy", "radius"]
# held-out per-frame PASS gates (for pass/fail reporting only)
PASS = dict(iris_iou=0.55, sclera_iou=0.45, ctr_err_px=25.0)

ANN = dict(  # ANNOTATION colour rules (pure MS-Paint palette) -- iris RED, sclera BLUE, almond GREEN
    iris=lambda R, G, B: (R > 180) & (G < 70) & (B < 70),
    sclera=lambda R, G, B: (B > 170) & (R < 80) & (G < 100),
    almond=lambda R, G, B: (G > 170) & (R < 80) & (B < 100))


def log(msg):
    print(msg, flush=True)


def next_run_dir():
    RUNS.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name[4:]) for p in RUNS.glob("run_*") if p.name[4:].isdigit()]
    n = (max(existing) + 1) if existing else 1
    d = RUNS / f"run_{n:03d}"
    for sub in ("model", "reports", "overlays", "predictions", "worst_frames", "correction_set"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d, n


# ============================================================================= STEP A  validate
def color_masks(img):
    B, G, R = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    return (ANN["iris"](R, G, B).astype(np.uint8) * 255,
            ANN["sclera"](R, G, B).astype(np.uint8) * 255,
            ANN["almond"](R, G, B).astype(np.uint8) * 255)


def close(mask, k):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(k)) | 1,) * 2))


def validate_and_extract(meta):
    """Read every painted to_mark unit, apply annotation rules, validate, and (re)write binary masks.
    Returns (valid_units, report)."""
    MASKS.mkdir(exist_ok=True)
    units, excluded, malformed, lr = [], [], [], defaultdict(int)
    overlaps = []
    for f in sorted(TOMARK.glob("*.png")):
        uid = f.stem
        m = meta.get(uid)
        img = cv2.imread(str(f))
        red, blue, green = color_masks(img)
        ni, ns = int((red > 0).sum()), int((blue > 0).sum())
        reason = None
        if m is None:
            reason = "no meta entry"
        elif ni < MIN_MARK_PX and ns < MIN_MARK_PX:
            reason = "unpainted (no iris and no sclera)"
        elif ni < MIN_MARK_PX:
            reason = "incomplete (no iris paint)"
        elif ns < MIN_MARK_PX:
            reason = "incomplete (no sclera paint)"
        if reason:
            excluded.append(dict(unit=uid, reason=reason, iris_px=ni, sclera_px=ns))
            continue
        iris = close(red, 7); sclera = close(blue, 7)
        overlap = int(((iris > 0) & (sclera > 0)).sum())
        ov_frac = overlap / max(1, int((iris > 0).sum()))
        if ov_frac > 0.15:
            malformed.append(dict(unit=uid, issue="iris/sclera overlap", overlap_frac=round(ov_frac, 3)))
        sclera[iris > 0] = 0                                   # iris wins the (rare) overlap pixel
        almond_from_green = None
        if int((green > 0).sum()) > 30:
            g = close(green, 9)
            cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                am = np.zeros(green.shape, np.uint8)
                cv2.drawContours(am, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
                almond_from_green = am
        almond = almond_from_green if almond_from_green is not None else \
            close(cv2.bitwise_or(iris, sclera), 11)
        union = cv2.bitwise_or(iris, sclera)
        alm_ok = iou(union, almond) if almond_from_green is not None else 1.0   # union==almond by build
        cv2.imwrite(str(MASKS / f"{uid}_iris.png"), iris)
        cv2.imwrite(str(MASKS / f"{uid}_sclera.png"), sclera)
        cv2.imwrite(str(MASKS / f"{uid}_almond.png"), almond)
        lr[m["eye"]] += 1
        overlaps.append(ov_frac)
        units.append(uid)
    report = dict(
        total_found=len(units) + len(excluded), total_valid=len(units),
        excluded=excluded, malformed=malformed,
        left=lr.get("L", 0), right=lr.get("R", 0),
        max_iris_sclera_overlap_frac=round(max(overlaps), 3) if overlaps else 0.0,
        almond_equals_union="by construction (union of iris+sclera) unless GREEN outline supplied",
        iris_sclera_disjoint=all(o <= 0.15 for o in overlaps))
    return sorted(units), report


# ============================================================================= features
def compute_features(bgr, orbit_mask):
    """Scale-invariant per-pixel features over an image; texture sigmas track orbit size so a model
    trained at one scale transfers to another. Returns (feat HxWxF float32, centroid(cx,cy), rscale)."""
    h, w = bgr.shape[:2]
    ys, xs = np.where(orbit_mask > 0)
    if len(xs) < 50:
        return None
    diam = 2.0 * math.sqrt(len(xs) / math.pi)                 # equivalent-circle diameter (px)
    ss = max(1.0, SIG_SMALL * diam); sl = max(2.0, SIG_LARGE * diam)
    bgrf = bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(bgrf)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, sN, vN = hsv[:, :, 0] / 179.0, hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    L, A, Bc = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur_s = cv2.GaussianBlur(gray, (0, 0), ss)
    blur_l = cv2.GaussianBlur(gray, (0, 0), sl)
    lap = np.abs(cv2.Laplacian(blur_s, cv2.CV_32F, ksize=3))
    cx, cy = xs.mean(), ys.mean()
    rscale = max(1.0, float(np.hypot(ys - cy, xs - cx).mean()))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / rscale; dy = (yy - cy) / rscale
    rad = np.hypot(dx, dy)
    feat = np.stack([b, g, r, hh, sN, vN, L, A, Bc, gray, blur_s, blur_l, lap, dx, dy, rad], axis=-1)
    return feat.astype(np.float32), (cx, cy), rscale


def orbit_from_crop(meta_u, shape):
    poly = np.array(meta_u["orbit_crop"], np.float32).astype(np.int32)
    roi = np.zeros(shape[:2], np.uint8); cv2.fillPoly(roi, [poly], 255)
    return roi


# ============================================================================= metrics
def iou(a, b):
    a = a > 0; b = b > 0
    u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else 1.0


def centroid(m):
    ys, xs = np.where(m > 0)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


# ============================================================================= STEP B/C  train + CV
def load_unit(uid, meta):
    img = cv2.imread(str(RAW / f"{uid}.png"))
    gi = cv2.imread(str(MASKS / f"{uid}_iris.png"), 0)
    gs = cv2.imread(str(MASKS / f"{uid}_sclera.png"), 0)
    ga = cv2.imread(str(MASKS / f"{uid}_almond.png"), 0)
    roi = orbit_from_crop(meta[uid], img.shape)
    label = np.zeros(img.shape[:2], np.uint8)
    label[gs > 0] = 2; label[gi > 0] = 1
    return dict(uid=uid, img=img, roi=roi, label=label, iris_gt=gi, sclera_gt=gs, almond_gt=ga,
                scale=float(meta[uid]["scale"]), eye=meta[uid]["eye"])


def training_rows(u, rng):
    fr = compute_features(u["img"], u["roi"])
    if fr is None:
        return None, None
    feat = fr[0]; roi = u["roi"] > 0; label = u["label"]
    Xs, ys = [], []
    for cls in (0, 1, 2):
        yy, xx = np.where(roi & (label == cls))
        if len(yy) == 0:
            continue
        if len(yy) > PER_CLASS_CAP:
            pick = rng.choice(len(yy), PER_CLASS_CAP, replace=False)
            yy, xx = yy[pick], xx[pick]
        Xs.append(feat[yy, xx]); ys.append(np.full(len(yy), cls, np.uint8))
    return np.concatenate(Xs), np.concatenate(ys)


def predict_unit(clf, u):
    fr = compute_features(u["img"], u["roi"])
    feat = fr[0]; ys, xs = np.where(u["roi"] > 0)
    proba = clf.predict_proba(feat[ys, xs])
    lab = clf.classes_[proba.argmax(1)]
    conf = proba.max(1)
    pl = np.zeros(u["roi"].shape, np.uint8); cf = np.zeros(u["roi"].shape, np.float32)
    pl[ys, xs] = lab; cf[ys, xs] = conf
    return pl, cf


def eval_pred(u, pl):
    pi = (pl == 1).astype(np.uint8) * 255
    ps = (pl == 2).astype(np.uint8) * 255
    pa = cv2.bitwise_or(pi, ps)
    ii, si, ai = iou(pi, u["iris_gt"]), iou(ps, u["sclera_gt"]), iou(pa, u["almond_gt"])
    pc, gc = centroid(pi), centroid(u["iris_gt"])
    cerr = (float(np.hypot(pc[0] - gc[0], pc[1] - gc[1])) / u["scale"]) if (pc and gc) else None
    pin = int((pi > 0).sum())
    out_alm = (int(((pi > 0) & (u["almond_gt"] == 0)).sum()) / pin) if pin else 0.0
    overlap = int(((pi > 0) & (ps > 0)).sum())                 # should be 0 (argmax is exclusive)
    row = dict(unit=u["uid"], eye=u["eye"], iris_iou=round(ii, 3), sclera_iou=round(si, 3),
               almond_iou=round(ai, 3), ctr_err=round(cerr, 1) if cerr is not None else None,
               out_alm=round(out_alm, 3), iris_sclera_overlap_px=overlap)
    row["pass"] = bool(ii >= PASS["iris_iou"] and si >= PASS["sclera_iou"]
                       and (cerr is not None and cerr <= PASS["ctr_err_px"]))
    return row, pi, ps


def cross_validate(units, meta, run_dir):
    log(f"[B] loading {len(units)} clean units ...")
    data = {u: load_unit(u, meta) for u in units}
    rng = np.random.default_rng(0)
    kf = KFold(N_FOLDS, shuffle=True, random_state=0)
    uarr = np.array(units)
    oof, oof_pred, fold_log = [], {}, []
    for k, (tr, te) in enumerate(kf.split(uarr)):
        Xs, ys = [], []
        for uid in uarr[tr]:
            X, y = training_rows(data[uid], rng)
            if X is not None:
                Xs.append(X); ys.append(y)
        X = np.concatenate(Xs); y = np.concatenate(ys)
        clf = ExtraTreesClassifier(n_estimators=MODEL["n_estimators"], max_depth=MODEL["max_depth"],
                                   min_samples_leaf=MODEL["min_samples_leaf"], n_jobs=-1,
                                   class_weight=MODEL["class_weight"], random_state=MODEL["random_state"])
        clf.fit(X, y)
        fold_log.append(dict(fold=k + 1, n_train_units=len(tr), n_px=int(len(X)), n_test_units=len(te)))
        log(f"[B] fold {k + 1}/{N_FOLDS}: {len(tr)} train units ({len(X)} px) -> {len(te)} held-out")
        for uid in uarr[te]:
            pl, _ = predict_unit(clf, data[uid])
            row, pi, ps = eval_pred(data[uid], pl)
            oof.append(row); oof_pred[uid] = (pi, ps)
    return data, oof, oof_pred, fold_log


def fit_final(units, meta, data, run_dir):
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for uid in units:
        X, y = training_rows(data[uid], rng)
        Xs.append(X); ys.append(y)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    clf = ExtraTreesClassifier(n_estimators=MODEL["n_estimators"], max_depth=MODEL["max_depth"],
                               min_samples_leaf=MODEL["min_samples_leaf"], n_jobs=-1,
                               class_weight=MODEL["class_weight"], random_state=MODEL["random_state"])
    clf.fit(X, y)
    import joblib
    joblib.dump(dict(model=clf, feature_names=FEATURE_NAMES), run_dir / "model" / "model.pkl")
    json.dump(dict(features=FEATURE_NAMES, sig_small=SIG_SMALL, sig_large=SIG_LARGE,
                   apply_maxdim=APPLY_MAXDIM, note="texture sigmas are relative to orbit diameter"),
              open(run_dir / "model" / "feature_config.json", "w"), indent=1)
    json.dump(dict(model=MODEL, n_folds=N_FOLDS, per_class_cap=PER_CLASS_CAP,
                   n_train_units=len(units), total_px=int(len(X)), pass_gates=PASS),
              open(run_dir / "model" / "train_config.json", "w"), indent=1)
    fi = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in clf.feature_importances_]))
    return clf, fi


def agg(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows):
    by_eye = {}
    for ek in ("L", "R"):
        er = [r for r in rows if r["eye"] == ek]
        if er:
            by_eye[ek] = dict(n=len(er), iris_iou=agg(er, "iris_iou"), sclera_iou=agg(er, "sclera_iou"),
                              almond_iou=agg(er, "almond_iou"), ctr_err_px=agg(er, "ctr_err"),
                              n_pass=sum(1 for r in er if r["pass"]))
    return dict(n_units=len(rows), mean_iris_iou=agg(rows, "iris_iou"),
                mean_sclera_iou=agg(rows, "sclera_iou"), mean_almond_iou=agg(rows, "almond_iou"),
                mean_ctr_err_px=agg(rows, "ctr_err"), mean_out_alm=agg(rows, "out_alm"),
                n_iris_outside_almond=sum(1 for r in rows if r["out_alm"] and r["out_alm"] > 0.05),
                n_pass=sum(1 for r in rows if r["pass"]), by_eye=by_eye)


# ============================================================================= STEP D  apply clip
def load_orbits():
    orb = defaultdict(dict)
    for r in csv.DictReader(open(OLP, encoding="utf-8")):
        try:
            orb[int(r["frame"])][r["eye"]] = np.array(json.loads(r["contour_json"]), np.float32)
        except Exception:
            pass
    return orb


def apply_clip(clf, video, orb, run_dir):
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dw = min(1.0, 960.0 / W); ow, oh = int(W * dw), int(H * dw)
    writer = cv2.VideoWriter(str(run_dir / "overlays" / "iris_sclera_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    rows = []; preds = {}                                       # preds[(frame,eye)] = dict for step E
    fn = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fn += 1
        disp = cv2.resize(frame, (ow, oh))
        for ek in ("L", "R"):
            c = orb.get(fn, {}).get(ek)
            rec = dict(frame=fn, eye=ek, cx="", cy="", major="", minor="", angle="",
                       iris_area=0, sclera_area=0, confidence=0.0, status="learned_gap")
            if c is None:
                rows.append(rec); continue
            x0, y0 = c.min(0); x1, y1 = c.max(0)
            bx0, by0 = max(0, int(x0) - 4), max(0, int(y0) - 4)
            bx1, by1 = min(W, int(x1) + 4), min(H, int(y1) + 4)
            sub = frame[by0:by1, bx0:bx1]
            if sub.size == 0:
                rows.append(rec); continue
            bw, bh = bx1 - bx0, by1 - by0
            s = min(1.0, APPLY_MAXDIM / max(bw, bh))
            sw, sh = max(8, int(bw * s)), max(8, int(bh * s))
            small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)
            poly_s = ((c - np.array([bx0, by0])) * np.array([sw / bw, sh / bh])).astype(np.int32)
            roi_s = np.zeros((sh, sw), np.uint8); cv2.fillPoly(roi_s, [poly_s], 255)
            fr = compute_features(small, roi_s)
            if fr is None:
                rows.append(rec); continue
            feat = fr[0]; ys, xs = np.where(roi_s > 0)
            proba = clf.predict_proba(feat[ys, xs])
            lab = clf.classes_[proba.argmax(1)]; conf = proba.max(1)
            pl = np.zeros((sh, sw), np.uint8); pl[ys, xs] = lab
            iris_s = (pl == 1).astype(np.uint8) * 255
            scl_s = (pl == 2).astype(np.uint8) * 255
            # keep largest iris blob (denoise) for centre/ellipse
            iris_big = largest_blob(iris_s)
            rec["status"] = "learned_ok"                       # prediction produced; classify_status refines
            rec["confidence"] = round(float(conf.mean()), 3)
            rec["iris_area"] = int((iris_big > 0).sum() / (s * s))
            rec["sclera_area"] = int((scl_s > 0).sum() / (s * s))
            gc = centroid(iris_big)
            if gc is not None:
                fx, fy = bx0 + gc[0] / s, by0 + gc[1] / s
                rec["cx"], rec["cy"] = round(fx, 1), round(fy, 1)
                ell = fit_ellipse(iris_big, bx0, by0, s)
                if ell:
                    rec.update(major=ell[0], minor=ell[1], angle=ell[2])
            # store small masks (in bbox frame) for step E anomaly geometry + overlay
            preds[(fn, ek)] = dict(bbox=(bx0, by0, bx1, by1), s=s, iris=iris_big, sclera=scl_s,
                                   orbit_area=float((roi_s > 0).sum() / (s * s)),
                                   iris_full=(iris_big > 0).sum() / (s * s))
            draw_overlay(disp, dw, bx0, by0, s, iris_big, scl_s, c)
            rows.append(rec)
        cv2.putText(disp, f"frame {fn}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(disp, f"frame {fn}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        writer.write(disp)
    writer.release(); cap.release()
    log(f"[D] applied to {fn} frames in {time.time() - t0:.0f}s")
    return rows, preds, (W, H)


def largest_blob(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8) * 255


def fit_ellipse(mask, bx0, by0, s):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    (_, _), (MA, ma), ang = cv2.fitEllipse(c)
    return [round(MA / s, 1), round(ma / s, 1), round(ang, 1)]


def draw_overlay(disp, dw, bx0, by0, s, iris_s, scl_s, contour):
    h, w = iris_s.shape
    iris_full = cv2.resize(iris_s, (int(w / s), int(h / s)), interpolation=cv2.INTER_NEAREST)
    scl_full = cv2.resize(scl_s, (int(w / s), int(h / s)), interpolation=cv2.INTER_NEAREST)
    y1, x1 = by0 + iris_full.shape[0], bx0 + iris_full.shape[1]
    reg = disp[int(by0 * dw):int(y1 * dw), int(bx0 * dw):int(x1 * dw)]
    if reg.size == 0:
        return
    ir = cv2.resize(iris_full, (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    sr = cv2.resize(scl_full, (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    reg[sr] = (0.5 * reg[sr] + np.array([200, 90, 0])).astype(np.uint8)
    reg[ir] = (0.5 * reg[ir] + np.array([0, 0, 200])).astype(np.uint8)
    cv2.polylines(disp, [(contour * dw).astype(np.int32)], True, (0, 220, 0), 1)


# ============================================================================= STEP E  worst frames
def find_worst(rows, preds, meta_frames, n_want=15):
    by_key = {(r["frame"], r["eye"]): r for r in rows}
    # temporal series per eye
    series = {ek: sorted([r for r in rows if r["eye"] == ek], key=lambda r: r["frame"]) for ek in ("L", "R")}
    iris_area_med = {ek: np.median([r["iris_area"] for r in series[ek] if r["iris_area"] > 0] or [1])
                     for ek in ("L", "R")}
    flagged = []
    for ek in ("L", "R"):
        ser = series[ek]
        for i, r in enumerate(ser):
            key = (r["frame"], ek); reasons = []; score = 0.0
            p = preds.get(key)
            conf = r["confidence"]
            if r["status"] == "learned_gap" or p is None:
                reasons.append("no_prediction"); score += 3
                flagged.append(dict(frame=r["frame"], eye=ek, score=round(score, 2),
                                    reasons=reasons, confidence=conf)); continue
            if conf < 0.55:
                reasons.append(f"low_confidence({conf:.2f})"); score += (0.55 - conf) * 4
            if r["sclera_area"] < 0.05 * p["orbit_area"]:
                reasons.append("sclera_missing"); score += 2
            fr_iris = r["iris_area"] / max(1.0, p["orbit_area"])
            if fr_iris < 0.03:
                reasons.append("iris_too_small"); score += 2
            if fr_iris > 0.75:
                reasons.append("iris_too_large"); score += 1.5
            # iris outside the opening (fragments detached from sclera-bordered opening)
            alm = largest_blob(cv2.morphologyEx(cv2.bitwise_or(p["iris"], p["sclera"]),
                                                cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)))
            almf = alm.copy()
            cnts, _ = cv2.findContours(almf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            almfill = np.zeros_like(almf)
            if cnts:
                cv2.drawContours(almfill, cnts, -1, 255, -1)
            out = int(((p["iris"] > 0) & (almfill == 0)).sum())
            outfrac = out / max(1, int((p["iris"] > 0).sum()))
            if outfrac > 0.10:
                reasons.append(f"iris_outside_almond({outfrac:.2f})"); score += outfrac * 3
            ni, _, _, _ = cv2.connectedComponentsWithStats(p["iris"])
            if ni - 1 >= 3:
                reasons.append("iris_fragmented"); score += 1
            na, _, _, _ = cv2.connectedComponentsWithStats(alm)
            if na - 1 >= 3:
                reasons.append("almond_fragmented"); score += 1
            # temporal centre jump / neighbour disagreement
            if r["cx"] != "" and i > 0 and ser[i - 1]["cx"] != "":
                jump = math.hypot(r["cx"] - ser[i - 1]["cx"], r["cy"] - ser[i - 1]["cy"])
                if jump > 40:
                    reasons.append(f"centre_jump({jump:.0f}px)"); score += min(3, jump / 40)
            if abs(r["iris_area"] - iris_area_med[ek]) > 0.6 * iris_area_med[ek]:
                reasons.append("iris_area_outlier"); score += 1
            if reasons:
                flagged.append(dict(frame=r["frame"], eye=ek, score=round(score, 2),
                                    reasons=reasons, confidence=conf))
    flagged.sort(key=lambda d: d["score"], reverse=True)
    # spread the picks across the clip (avoid 15 adjacent frames)
    picked, seen_frames = [], []
    for d in flagged:
        if all(abs(d["frame"] - sf) >= 8 or e != d["eye"] for sf, e in seen_frames) or len(picked) < 4:
            picked.append(d); seen_frames.append((d["frame"], d["eye"]))
        if len(picked) >= n_want:
            break
    return picked, flagged


def build_correction_set(picked, preds, video, orb, run_dir):
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cset = run_dir / "correction_set"; wf = run_dir / "worst_frames"
    meta_corr = {}
    by_frame = defaultdict(list)
    for d in picked:
        by_frame[d["frame"]].append(d)
    for fnum in sorted(by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum - 1); ok, frame = cap.read()
        if not ok:
            continue
        for d in by_frame[fnum]:
            ek = d["eye"]; c = orb.get(fnum, {}).get(ek)
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
            cv2.imwrite(str(cset / f"{uid}_raw.png"), z)
            cv2.imwrite(str(cset / f"{uid}_to_correct.png"), z)     # clean copy to PAINT on
            # prediction overlay on the crop
            ov = z.copy(); p = preds.get((fnum, ek))
            if p:
                bx0, by0, bx1, by1 = p["bbox"]; s = p["s"]
                for msk, col in ((p["sclera"], (200, 90, 0)), (p["iris"], (0, 0, 200))):
                    full = cv2.resize(msk, (int(msk.shape[1] / s), int(msk.shape[0] / s)),
                                      interpolation=cv2.INTER_NEAREST)
                    canvas = np.zeros((H, W), np.uint8)
                    canvas[by0:by0 + full.shape[0], bx0:bx0 + full.shape[1]] = full
                    sub = canvas[Y0:Y1, X0:X1]
                    subz = cv2.resize(sub, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    ov[subz] = (0.5 * ov[subz] + np.array(col)).astype(np.uint8)
            cv2.imwrite(str(cset / f"{uid}_prediction.png"), ov)
            cv2.imwrite(str(wf / f"{uid}.png"), ov)
            cc = ((c - np.array([X0, Y0])) * 2.0).tolist()
            meta_corr[uid] = dict(frame=fnum, eye=ek, X0=int(X0), Y0=int(Y0), scale=2.0,
                                  crop_w=int(z.shape[1]), crop_h=int(z.shape[0]), orbit_crop=cc,
                                  score=d["score"], reasons=d["reasons"], confidence=d["confidence"])
            json.dump(meta_corr[uid], open(cset / f"{uid}_meta.json", "w"), indent=1)
    cap.release()
    json.dump(meta_corr, open(cset / "correction_meta.json", "w"), indent=1)
    (cset / "README_CORRECTION.md").write_text(CORRECTION_README, encoding="utf-8")
    return meta_corr


CORRECTION_README = """# RIT correction set

These are the frames the workbench is LEAST sure about (worst self-consistency). Paint the same three
colours as before, on each `*_to_correct.png` (open in tools/rit_paint_tool.html, or MS Paint):

- IRIS   -> pure RED  (255,0,0)
- SCLERA -> pure BLUE (0,0,255)
- (optional) ALMOND outline -> pure GREEN (0,255,0)

`*_prediction.png` shows what the current model guessed (red iris / blue sclera) so you can see where it
went wrong. `*_raw.png` is the clean reference.

When done, copy each corrected `*_to_correct.png` into RIT_ground_truth/to_mark/ as `<id>.png`
(e.g. f0123_R.png), add its meta from `correction_meta.json` into RIT_ground_truth/meta.json,
then re-run:  py -3.14 tools/rit_learning_workbench.py   -> it will create run_002 and compare.
"""


# ============================================================================= run comparison (F)
def compare_prev(run_n, this_summary):
    prev = RUNS / f"run_{run_n - 1:03d}" / "reports" / "heldout_summary.json"
    if run_n <= 1 or not prev.exists():
        return None
    p = json.load(open(prev))
    def d(k):
        a, b = this_summary.get(k), p.get(k)
        return None if (a is None or b is None) else round(a - b, 3)
    return dict(prev_run=run_n - 1, delta_iris_iou=d("mean_iris_iou"),
                delta_sclera_iou=d("mean_sclera_iou"), delta_almond_iou=d("mean_almond_iou"),
                delta_ctr_err_px=d("mean_ctr_err_px"), prev=p)


# ============================================================================= main
def main():
    meta = json.load(open(GT / "meta.json", encoding="utf-8"))
    appr = json.load(open(OUT / "approved_landmarks.json", encoding="utf-8"))
    video = appr.get("video")
    if not video or not Path(video).exists():
        stem = OUT.name[:-8] if OUT.name.endswith("_tracked") else OUT.name
        video = str(Path("samples") / (stem + ".mp4"))

    run_dir, run_n = next_run_dir()
    log(f"=== RIT Learning Workbench :: run_{run_n:03d} ===")

    # ---- STEP A
    log("[A] validating clinician annotations ...")
    units, vreport = validate_and_extract(meta)
    json.dump(vreport, open(run_dir / "reports" / "annotation_validation.json", "w"), indent=1)
    log(f"[A] found {vreport['total_found']} units | valid {vreport['total_valid']} "
        f"| L {vreport['left']} R {vreport['right']} | excluded {len(vreport['excluded'])}")
    for e in vreport["excluded"]:
        log(f"      excluded {e['unit']}: {e['reason']}")
    if vreport["total_valid"] < 12 or vreport["total_valid"] < 0.5 * vreport["total_found"]:
        log("[A] STOP: too few valid annotations -- correction needed before training.")
        json.dump(dict(stopped="insufficient_valid_annotations", validation=vreport),
                  open(run_dir / "reports" / "STOP.json", "w"), indent=1)
        return
    if vreport["malformed"]:
        log(f"[A] note: {len(vreport['malformed'])} unit(s) with mask issues (kept, flagged): "
            + ", ".join(m["unit"] for m in vreport["malformed"]))

    # ---- STEP B + C
    data, oof, oof_pred, fold_log = cross_validate(units, meta, run_dir)
    clf, feat_imp = fit_final(units, meta, data, run_dir)
    hsum = summarize(oof)
    json.dump(dict(summary=hsum, per_unit=oof, folds=fold_log, feature_importances=feat_imp),
              open(run_dir / "reports" / "cv_report.json", "w"), indent=1)
    json.dump(hsum, open(run_dir / "reports" / "heldout_summary.json", "w"), indent=1)
    log(f"[C] held-out: iris {hsum['mean_iris_iou']} sclera {hsum['mean_sclera_iou']} "
        f"almond {hsum['mean_almond_iou']} ctrErr {hsum['mean_ctr_err_px']}px "
        f"pass {hsum['n_pass']}/{hsum['n_units']}")

    baseline = json.load(open(BASELINE)) if BASELINE.exists() else None
    bcmp = None
    if baseline:
        bs = baseline["summary"]
        bcmp = {k: dict(opencv=bs.get(k), learned=hsum.get(k))
                for k in ("mean_iris_iou", "mean_sclera_iou", "mean_almond_iou",
                          "mean_ctr_err_px", "n_iris_outside_almond")}
        json.dump(bcmp, open(run_dir / "reports" / "opencv_baseline_comparison.json", "w"), indent=1)

    # best / worst held-out (for the report)
    def comp(r):
        return 0.5 * r["iris_iou"] + 0.3 * r["sclera_iou"] + 0.2 * r["almond_iou"]
    ranked = sorted(oof, key=comp, reverse=True)
    save_heldout_overlays(data, oof_pred, ranked[:10], run_dir / "reports", "best")
    save_heldout_overlays(data, oof_pred, ranked[-10:], run_dir / "reports", "worst")

    # ---- STEP D
    log("[D] applying learned model to the whole clip ...")
    orb = load_orbits()
    rows, preds, (W, H) = classify_status(*apply_clip(clf, video, orb, run_dir))
    with open(run_dir / "predictions" / "points.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["frame", "eye", "cx", "cy", "major", "minor", "angle",
                                           "iris_area", "sclera_area", "confidence", "status"])
        w.writeheader(); w.writerows(rows)
    status_counts = defaultdict(int)
    for r in rows:
        status_counts[r["status"]] += 1
    clip_summary = dict(frames=W and int(max(r["frame"] for r in rows)), n_records=len(rows),
                        status_counts=dict(status_counts),
                        mean_confidence=round(float(np.mean([r["confidence"] for r in rows if r["confidence"]])), 3))
    json.dump(clip_summary, open(run_dir / "predictions" / "summary.json", "w"), indent=1)
    log(f"[D] status: {dict(status_counts)}")

    # ---- STEP E
    log("[E] selecting worst frames for correction ...")
    picked, flagged = find_worst(rows, preds, meta, n_want=15)
    meta_corr = build_correction_set(picked, preds, video, orb, run_dir)
    json.dump(dict(picked=picked, all_flagged=flagged[:80]),
              open(run_dir / "reports" / "worst_frames.json", "w"), indent=1)
    log(f"[E] flagged {len(flagged)} anomalies; prepared {len(meta_corr)} correction units")

    # ---- promote model + run comparison (F)
    PROMOTED.mkdir(exist_ok=True)
    for fn in ("model.pkl", "feature_config.json", "train_config.json"):
        src = run_dir / "model" / fn
        if src.exists():
            shutil.copy2(src, PROMOTED / fn)
    json.dump(dict(latest_run=f"run_{run_n:03d}", heldout=hsum), open(PROMOTED / "LATEST.json", "w"), indent=1)
    delta = compare_prev(run_n, hsum)
    if delta:
        json.dump(delta, open(run_dir / "reports" / "vs_prev_run.json", "w"), indent=1)
        log(f"[F] vs run_{run_n-1:03d}: d-irisIoU {delta['delta_iris_iou']} "
            f"d-sclIoU {delta['delta_sclera_iou']} d-ctrErr {delta['delta_ctr_err_px']}px")

    write_run_report(run_dir, run_n, vreport, hsum, bcmp, feat_imp, clip_summary, picked, meta_corr, delta, video)
    log(f"=== run_{run_n:03d} complete -> {run_dir} ===")


def classify_status(rows, preds, dims):
    """Assign per-record status from confidence + geometry (kept simple + explainable)."""
    for r in rows:
        if r["status"] == "learned_gap":
            continue
        p = preds.get((r["frame"], r["eye"]))
        conf = r["confidence"]
        oa = p["orbit_area"] if p else 1.0
        fr_iris = r["iris_area"] / max(1.0, oa)
        bad = (r["sclera_area"] < 0.04 * oa) or fr_iris < 0.02 or fr_iris > 0.85 or r["iris_area"] == 0
        if bad:
            r["status"] = "needs_human_review"
        elif conf < 0.55:
            r["status"] = "learned_uncertain"
        else:
            r["status"] = "learned_ok"
    return rows, preds, dims


def save_heldout_overlays(data, oof_pred, rows, reports_dir, tag):
    d = reports_dir / f"heldout_{tag}"; d.mkdir(exist_ok=True)
    for r in rows:
        u = data[r["unit"]]; pi, ps = oof_pred[r["unit"]]
        ov = u["img"].copy()
        cnts, _ = cv2.findContours(u["almond_gt"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 220, 0), 2)
        ov[ps > 0] = (0.55 * ov[ps > 0] + np.array([200, 90, 0])).astype(np.uint8)
        ov[pi > 0] = (0.55 * ov[pi > 0] + np.array([0, 0, 200])).astype(np.uint8)
        txt = f"{r['unit']} iris={r['iris_iou']:.2f} scl={r['sclera_iou']:.2f}"
        cv2.putText(ov, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(ov, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imwrite(str(d / f"{r['unit']}.png"), ov)


def write_run_report(run_dir, run_n, vreport, hsum, bcmp, feat_imp, clip_summary, picked, meta_corr, delta, video):
    L = []
    L.append(f"# RIT Learning Workbench -- run_{run_n:03d}\n")
    L.append("Active path: RIT Orbit Lock + clinician ground truth + learned pixel segmenter. "
             "OpenCV rule detector is DEPRECATED (historical benchmark only).\n")
    L.append("## 1. Ground-truth validation")
    L.append(f"- units found: **{vreport['total_found']}**, valid: **{vreport['total_valid']}** "
             f"(L {vreport['left']} / R {vreport['right']})")
    L.append(f"- excluded: {len(vreport['excluded'])}")
    for e in vreport["excluded"]:
        L.append(f"  - `{e['unit']}` -- {e['reason']} (iris {e['iris_px']}px, sclera {e['sclera_px']}px)")
    L.append(f"- iris/sclera disjoint: {vreport['iris_sclera_disjoint']} "
             f"(max overlap {vreport['max_iris_sclera_overlap_frac']}); almond = {vreport['almond_equals_union']}")
    if vreport["malformed"]:
        L.append(f"- malformed flagged: {', '.join(m['unit'] for m in vreport['malformed'])}")
    L.append("\n## 2. Model")
    L.append(f"- **{MODEL['kind']}**, {MODEL['n_estimators']} trees, max_depth {MODEL['max_depth']}, "
             f"class_weight balanced; {N_FOLDS}-fold held-out-FRAME CV")
    L.append(f"- features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    top = sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)[:6]
    L.append(f"- top features: {', '.join(f'{k}={v}' for k, v in top)}")
    L.append("\n## 3. Held-out validation (learned)")
    L.append(f"- iris IoU **{hsum['mean_iris_iou']}**, sclera IoU **{hsum['mean_sclera_iou']}**, "
             f"almond IoU **{hsum['mean_almond_iou']}**")
    L.append(f"- iris centre error **{hsum['mean_ctr_err_px']} px**, iris-outside-almond units "
             f"{hsum['n_iris_outside_almond']}, pass {hsum['n_pass']}/{hsum['n_units']}")
    for ek, d in hsum["by_eye"].items():
        L.append(f"  - {ek}: iris {d['iris_iou']} sclera {d['sclera_iou']} almond {d['almond_iou']} "
                 f"ctrErr {d['ctr_err_px']}px pass {d['n_pass']}/{d['n']}")
    if bcmp:
        L.append("\n## 4. vs OpenCV baseline (historical)")
        L.append("| metric | OpenCV (deprecated) | learned |")
        L.append("|---|---|---|")
        for k, v in bcmp.items():
            L.append(f"| {k} | {v['opencv']} | {v['learned']} |")
    L.append("\n## 5. Full-clip application")
    L.append(f"- records: {clip_summary['n_records']} ({clip_summary.get('frames')} frames x 2 eyes)")
    L.append(f"- status: {clip_summary['status_counts']}")
    L.append(f"- mean confidence: {clip_summary['mean_confidence']}")
    L.append(f"- overlay: `{(run_dir / 'overlays' / 'iris_sclera_overlay.mp4')}`")
    L.append("\n## 6. Worst frames selected for correction")
    for d in picked:
        L.append(f"- f{d['frame']:04d}_{d['eye']} (score {d['score']}, conf {d['confidence']}): "
                 f"{', '.join(d['reasons'])}")
    L.append(f"\n- correction set: `{run_dir / 'correction_set'}` "
             f"({len(meta_corr)} units, paint red/blue/green then re-run for run_{run_n+1:03d})")
    if delta:
        L.append("\n## 7. vs previous run")
        L.append(f"- d iris IoU {delta['delta_iris_iou']}, d sclera IoU {delta['delta_sclera_iou']}, "
                 f"d almond IoU {delta['delta_almond_iou']}, d centre err {delta['delta_ctr_err_px']} px")
    (run_dir / "reports" / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
