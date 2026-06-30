"""RIT Iris-in-Orbit detector PROBE (read-only; no validator/clinical wiring, no commit).

Finds the iris ONLY inside the RIT-Orbit-Lock moving orbit, per Dr. Kothari's hardened rules
(docs/RIT_IRIS_DETECTION_CASES.md):
  1. Hard containment — iris edges/centre must lie inside the moving orbit (polygon passed to fit_limbus).
  2. Dark-only — fit_limbus(bgr=...) sclera-whiteness gate rejects skin/eyelid.
  3. Imagine the circle from the visible arc — free RANSAC fit when the iris is well seen (sets the
     radius), then r_fixed centre-from-arc when it is partial/foreshortened (occlusion-invariant centre).
  4. Never give up — every frame re-seeds from the darkest blob inside the orbit if the last frame lost it.

Reuses src/core/limbus.fit_limbus (the existing arc→circle/ellipse fitter). Reads the validated
orbit_lock_points.csv for the per-frame orbit. Outputs to _rit_iris_in_orbit_probe/.

Usage: python tools/rit_iris_in_orbit_probe.py [out_dir] [n_frames]
"""
import csv, json, sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.limbus import fit_limbus

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 640
REV = OUT / "_rit_iris_in_orbit_probe"
(REV / "representative_frames").mkdir(parents=True, exist_ok=True)
ORB = OUT / "_rit_orbit_lock_probe" / "orbit_lock_points.csv"

appr = json.load(open(OUT / "approved_landmarks.json", encoding="utf-8"))
video = appr.get("video")
if not video or not Path(video).exists():
    stem = OUT.name[:-8] if OUT.name.endswith("_tracked") else OUT.name
    video = str(Path("samples") / (stem + ".mp4"))
iris0 = {ek: (float(v["x"]), float(v["y"]), float(v["radius"]))
         for ek, v in (appr.get("iris") or {}).items() if v}

# per-frame moving orbit contour from the validated orbit lock
orb = {}
for r in csv.DictReader(open(ORB, encoding="utf-8")):
    orb[(int(r["frame"]), r["eye"])] = np.array(json.loads(r["contour_json"]), np.float32)

COV_GOOD = 0.40           # min limbus arc coverage to accept a FREE fit's centre
COV_HOLD = 0.60           # min coverage to TRUST a free fit's RADIUS (update the held iris size)
CONTAIN_TOL = 8.0         # fitted centre may sit at most this far outside the orbit
R_ALPHA = 0.12            # EMA on the running iris radius
R_LO_F, R_HI_F = 0.80, 1.30   # a free fit's radius must stay within this band of the held size,
#                               else it is a shrunk/partial-arc artifact -> use the held radius instead
#                               (rule: the circle must COVER ALL the dark iris, never a sub-part)

def poly_cv(c):
    return c.reshape(-1, 1, 2).astype(np.float32)

def centroid(c):
    return float(c[:, 0].mean()), float(c[:, 1].mean())

def dark_seed(gray, c):
    """Reacquire seed: centroid of the darkest pixels INSIDE the orbit (the iris blob)."""
    H, W = gray.shape
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [c.astype(np.int32)], 255)
    vals = gray[m > 0]
    if vals.size < 30:
        return None
    thr = float(np.percentile(vals, 18))
    ys, xs = np.where((m > 0) & (gray <= thr))
    if len(xs) < 12:
        return None
    return float(xs.mean()), float(ys.mean())


def _kk(k):
    s = max(3, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def fit_iris(gray, bgr, poly, seed, held_r):
    """VERTICAL-ELLIPSE limbus model (Dr. Kothari's hand marks, 2026-06-30). Inside the moving orbit:
    segment the DARK-BROWN/BLACK iris (low value; bright sclera AND brighter skin excluded); a strong
    morphological open detaches the upper-lid lashes; SELECT the dark component that BORDERS the sclera
    (the limbus — lashes/shadow border skin, not sclera) nearest the tracked centre. The iris is then a
    VERTICAL ellipse: major (vertical) axis = the held iris diameter (CONSTANT — the iris does not
    shrink, the occluded part is imagined), minor (horizontal) axis = the VISIBLE dark width (the
    foreshortening), slight tilt from the dark region's own long axis (clamped near vertical). Centre =
    centroid of the dark region, so it always lands INSIDE the dark, never on sclera/skin. This replaces
    the old free fitEllipse-on-blob, which produced horizontal ovals, circles, and edge-pushed centres."""
    H, W = gray.shape
    D = 2.0 * held_r                                           # held iris diameter (vertical major axis)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly.reshape(-1, 2).astype(np.int32)], 255)
    inside = m > 0
    if int(inside.sum()) < 60:
        return None
    if bgr is not None:                                        # sclera = bright AND whitish (low sat)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        Vc = hsv[:, :, 2]; Sc = hsv[:, :, 1]
        sclera = (inside & (Vc > 140) & (Sc < 60)).astype(np.uint8) * 255
        vin = Vc[inside]
        darkthr = float(np.percentile(vin, 35))               # dark-brown/black only (skin is brighter)
        dark = (inside & (Vc < darkthr) & (sclera == 0)).astype(np.uint8) * 255
    else:
        vin = gray[inside]
        sclera = (inside & (gray > float(np.percentile(vin, 75)))).astype(np.uint8) * 255
        darkthr = float(np.percentile(vin, 35))
        dark = (inside & (gray < darkthr)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _kk(held_r * 0.20))   # detach thin lashes
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _kk(held_r * 0.12))  # fill specular holes in the iris
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_OPEN, _kk(held_r * 0.10))  # kill specular specks ON the
    scl_dil = cv2.dilate(sclera, _kk(held_r * 0.18))                       #   iris that fake a limbus
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    amin = 0.02 * np.pi * held_r * held_r
    best = None
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < amin:
            continue
        comp = (lab == i).astype(np.uint8)
        edge = cv2.morphologyEx(comp * 255, cv2.MORPH_GRADIENT, _kk(3))
        touches = int(((edge > 0) & (scl_dil > 0)).sum())     # does this dark region border the sclera?
        if touches < 4:                                       # iris borders sclera; lash/shadow does not
            continue
        d = (cen[i][0] - seed[0]) ** 2 + (cen[i][1] - seed[1]) ** 2
        score = touches - 0.002 * d
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    i = best[1]
    comp = (lab == i).astype(np.uint8)
    ys, xs = np.where(comp > 0)
    cx = float(np.median(xs)); cy = float(np.median(ys))      # CENTRE = dark centroid -> always in the dark
    a = 0.5 * D                                              # a = held vertical semi-axis (major axis)
    tilt = 0.0
    # LIMBUS = the dark-iris boundary that BORDERS the sclera -- the 'solid' convex arc Dr. K draws. Use it
    # ONLY to size the (foreshortened) WIDTH: at the waist, |limbus - centre| = the minor semi-axis b.
    edge = cv2.morphologyEx(comp * 255, cv2.MORPH_GRADIENT, _kk(3))
    ly, lx = np.where((edge > 0) & (scl_dil > 0))
    if lx.size >= 10:
        u = np.sqrt(np.clip(1.0 - ((ly - cy) / a) ** 2, 0.0, 1.0))   # ellipse profile: x-offset = b*u
        uu = u > 0.30                                        # waist-ish points give a stable b = |dx|/u
        if int(uu.sum()) >= 6:
            b = float(np.median(np.abs(lx[uu] - cx) / u[uu]))
        else:
            b = float(np.median(np.abs(lx - cx)))
        width_h = float(np.clip(2.0 * b, 0.16 * D, D))       # thinner the more the eye is turned
    else:                                                    # too little limbus -> waist-width fallback
        band = np.abs(ys - cy) <= 0.20 * D
        xb = xs[band] if int(band.sum()) >= 8 else xs
        width_h = float(np.clip(np.percentile(xb, 95) - np.percentile(xb, 5), 0.16 * D, D))
    height_v = D
    # ellipse tuple = ((cx,cy),(MA=horizontal, ma=vertical), angle) -> drawn as a VERTICAL-major ellipse
    ell = ((cx, cy), (width_h, height_v), tilt)
    roundness = float(width_h / height_v)                     # ~1 at front gaze, small at extreme gaze
    r_out = held_r                                            # iris does NOT shrink -> hold the diameter
    area = float(st[i, cv2.CC_STAT_AREA])
    cov = min(1.0, area / (np.pi * held_r * held_r))
    return SimpleNamespace(cx=cx, cy=cy, r=float(r_out), ellipse=ell,
                           arc_coverage=float(cov), arc_pts=None, n_inliers=int(area),
                           roundness=roundness, comp=comp)

GREEN = (0, 200, 0); YELLOW = (0, 255, 255); AMBER = (0, 165, 255); RED = (0, 0, 255); CYAN = (255, 255, 0)
REP = {187, 188, 192, 195, 227, 262, 267, 269, 311, 341, 348, 350, 353, 378, 379, 382, 393, 438, 460, 463, 469, 475}

cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
ds = min(1.0, 1280.0 / max(1, W)); ow, oh = int(W * ds), int(H * ds)
writer = cv2.VideoWriter(str(REV / "iris_in_orbit_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

run_R = {ek: iris0.get(ek, (0, 0, 30))[2] for ek in ("L", "R")}
last_good = {ek: None for ek in ("L", "R")}
counts = {ek: {"free": 0, "fixed": 0, "gap": 0} for ek in ("L", "R")}
rows = []
fn = 0
while fn < NF:
    ok, frame = cap.read()
    if not ok:
        break
    fn += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    disp = cv2.resize(frame, (ow, oh))
    cv2.putText(disp, f"frame {fn}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(disp, f"frame {fn}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    for ek in ("L", "R"):
        c = orb.get((fn, ek))
        if c is None:
            continue
        poly = poly_cv(c)
        cv2.polylines(disp, [(c * ds).astype(np.int32)], True, GREEN, 1)
        # IRIS = dark region inside the orbit, picked NEAREST the tracked centre (location prior), with
        # the shape covering the dark and the centre at its centroid. Seed = last good iris (it follows
        # gaze), else the orbit centroid (reacquire). Held radius caps the size.
        R = run_R[ek]
        def _inside(pt):
            return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True) >= -CONTAIN_TOL
        seed = last_good[ek] if (last_good[ek] is not None and _inside(last_good[ek])) else centroid(c)
        f = fit_iris(gray, frame, poly, seed, R)
        if f is None or not _inside((f.cx, f.cy)):
            f = fit_iris(gray, frame, poly, centroid(c), R)   # reacquire from the orbit centroid
        if f is not None and _inside((f.cx, f.cy)):
            kind = "free" if (f.roundness >= 0.78 and f.arc_coverage >= 0.45) else "fixed"  # full vs partial
            fit = f
            last_good[ek] = (f.cx, f.cy)
            if kind == "free":                                # well-seen full iris -> update the held size
                run_R[ek] = (1 - R_ALPHA) * run_R[ek] + R_ALPHA * f.r
        else:
            kind, fit = "gap", None
            last_good[ek] = None
        counts[ek][kind] += 1
        if fit is not None:
            last_good[ek] = (fit.cx, fit.cy)
            col = YELLOW if kind == "free" else AMBER
            ctr = (int(fit.cx * ds), int(fit.cy * ds))
            if fit.ellipse is not None:
                (ecx, ecy), (MA, ma), ang = fit.ellipse
                cv2.ellipse(disp, (int(ecx * ds), int(ecy * ds)), (int(MA * ds / 2), int(ma * ds / 2)),
                            float(ang), 0, 360, col, 2)
            else:
                cv2.circle(disp, ctr, max(2, int(fit.r * ds)), col, 2)
            cv2.circle(disp, ctr, 3, RED, -1)
            if fit.arc_pts is not None:
                for p in fit.arc_pts:
                    cv2.circle(disp, (int(p[0] * ds), int(p[1] * ds)), 1, CYAN, -1)
        else:
            last_good[ek] = None    # lost -> next frame re-seeds from the dark blob (reacquire)
        rows.append(dict(frame=fn, eye=ek, kind=kind,
                         cx="" if fit is None else round(fit.cx, 1),
                         cy="" if fit is None else round(fit.cy, 1),
                         r="" if fit is None else round(fit.r, 1),
                         arc_coverage="" if fit is None else round(fit.arc_coverage, 2),
                         is_ellipse=(fit is not None and fit.ellipse is not None)))
        if fn in REP and fit is not None or (fn in REP and fit is None):
            # full-res crop with the fit drawn large
            x0, y0 = c.min(0); x1, y1 = c.max(0); cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(x1 - x0, y1 - y0) / 2 + 1.4 * run_R[ek] + 20; sp = int(2 * half); X0 = cxm - half; Y0 = cym - half
            sq = np.zeros((sp, sp, 3), np.uint8); fx0, fy0 = int(X0), int(Y0)
            sx0, sy0 = max(0, -fx0), max(0, -fy0); cfx0, cfy0 = max(0, fx0), max(0, fy0)
            cw = min(sp - sx0, W - cfx0); chh = min(sp - sy0, H - cfy0)
            if cw > 0 and chh > 0:
                sq[sy0:sy0 + chh, sx0:sx0 + cw] = frame[cfy0:cfy0 + chh, cfx0:cfx0 + cw]
            TW = 460; s = TW / sp; d2 = cv2.resize(sq, (TW, TW))
            Tp = lambda p: ((np.asarray(p, np.float32) - [X0, Y0]) * s)
            cv2.polylines(d2, [Tp(c).astype(np.int32)], True, GREEN, 2)
            if fit is not None:
                col = YELLOW if kind == "free" else AMBER
                if fit.ellipse is not None:
                    (ecx, ecy), (MA, ma), ang = fit.ellipse
                    e0 = Tp([[ecx, ecy]])[0]
                    cv2.ellipse(d2, (int(e0[0]), int(e0[1])), (int(MA * s / 2), int(ma * s / 2)),
                                float(ang), 0, 360, col, 2)
                else:
                    cc = Tp([[fit.cx, fit.cy]])[0]
                    cv2.circle(d2, (int(cc[0]), int(cc[1])), max(2, int(fit.r * s)), col, 2)
                cc = Tp([[fit.cx, fit.cy]])[0]
                cv2.circle(d2, (int(cc[0]), int(cc[1])), 3, RED, -1)
                if fit.arc_pts is not None:
                    for p in fit.arc_pts:
                        q = Tp([p])[0]; cv2.circle(d2, (int(q[0]), int(q[1])), 2, CYAN, -1)
            lab = f"f{fn} {ek}  {kind}" + ("" if fit is None else f"  cov={fit.arc_coverage:.2f}")
            for col2, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(d2, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col2, th)
            cv2.imwrite(str(REV / "representative_frames" / f"f{fn:04d}_{ek}.png"), d2)
    writer.write(disp)
writer.release(); cap.release()

with open(REV / "iris_in_orbit_points.csv", "w", newline="", encoding="utf-8") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=["frame", "eye", "kind", "cx", "cy", "r", "arc_coverage", "is_ellipse"])
    w.writeheader(); w.writerows(rows)
json.dump({"video": video, "frames": fn, "per_eye": counts,
           "rule": "iris fit ONLY inside the moving orbit; free->radius, r_fixed->arc centre; reacquire from dark blob"},
          open(REV / "iris_in_orbit_summary.json", "w", encoding="utf-8"), indent=2)
for ek in ("L", "R"):
    c = counts[ek]; tot = sum(c.values())
    print(f"{ek}: free {c['free']}  fixed {c['fixed']}  gap {c['gap']}  (/{tot})")
print("overlay:", REV / "iris_in_orbit_overlay.mp4")
