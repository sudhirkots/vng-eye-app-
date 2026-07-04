"""RIT iris detector -- implements docs/CLINICAL_NYSTAGMUS_DETECTOR.md.

Per eye, inside the tracked almond (orbit):
  STRIP lashes -> read the smooth shapes.
  SCLERA  = bright/pink uniform region, lid-skin & specks excluded (outside-almond impossible by bound).
  IRIS    = almond - sclera (the dark remainder), the region whose sclera-facing edge is a convex circular
            arc (the limbus).  almond = sclera + iris (exact).
  LIMBUS  = iris<->sclera circular arc, convex toward iris.  Locate the centre by VOTING one known radius
            inward from each sclera-edge-facing-dark point (range of distances for foreshortening), clipped
            INSIDE the almond, biased toward the PREVIOUS frame's iris (resolves the extreme sliver vs
            canthus shadow).  Complete the circle, foreshortened to an OVAL by gaze; clip strictly inside
            the almond.

Renders _rit_iris/iris_overlay.mp4 (scrubbable). Probe-only, NO commit, NO validator wiring.
Usage: py -3.14 tools/rit_iris.py [out_dir] [n_frames]
"""
import csv, json, sys, math
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 640
REV = OUT / "_rit_iris"; REV.mkdir(parents=True, exist_ok=True)
OLP = OUT / "_rit_orbit_lock_probe/orbit_lock_points.csv"
appr = json.load(open(OUT / "approved_landmarks.json", encoding="utf-8"))
video = appr.get("video")
if not video or not Path(video).exists():
    stem = OUT.name[:-8] if OUT.name.endswith("_tracked") else OUT.name
    video = str(Path("samples") / (stem + ".mp4"))
iris0 = {ek: (float(v["x"]), float(v["y"]), float(v["radius"]))
         for ek, v in (appr.get("iris") or {}).items() if v}

orb = defaultdict(dict)
for r0 in csv.DictReader(open(OLP, encoding="utf-8")):
    try:
        orb[int(r0["frame"])][r0["eye"]] = np.array(json.loads(r0["contour_json"]), np.float32)
    except Exception:
        pass


def E(k):
    s = max(3, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def find_sclera(bgr, inside, held_r):
    H, W = inside.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    vin = V[inside]
    if vin.size < 50:
        return np.zeros((H, W), np.uint8)
    vth = max(120.0, float(np.percentile(vin, 60)))
    whitish = (V > vth) & (S < 50)                                # true white sclera (stricter)
    redpink = (V > vth * 0.82) & ((Hh < 8) | (Hh > 170)) & (S < 140)   # scleral PINK (red hue) -> tails
    skin = (Hh >= 8) & (Hh <= 30) & (S > 45)                      # orange/brown lid + cheek skin
    cand = (inside & (whitish | redpink) & (~skin)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, E(held_r * 0.06))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    if n <= 1:
        return np.zeros((H, W), np.uint8)
    core = inside & whitish                                       # a real sclera blob CONTAINS white;
    keep = np.zeros((H, W), np.uint8); bestw = -1; besti = -1     # skin-pink patches do not -> dropped
    for i in range(1, n):
        comp = lab == i
        if int((comp & core).sum()) < 20:
            continue
        if st[i, cv2.CC_STAT_AREA] > bestw:
            bestw = st[i, cv2.CC_STAT_AREA]; besti = i
    if besti < 0:
        return np.zeros((H, W), np.uint8)
    keep[lab == besti] = 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, E(held_r * 0.10))
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(keep))
    cap = 0.18 * np.pi * held_r * held_r
    for i in range(1, nh):
        x, y, w, h = (hst[i, cv2.CC_STAT_LEFT], hst[i, cv2.CC_STAT_TOP],
                      hst[i, cv2.CC_STAT_WIDTH], hst[i, cv2.CC_STAT_HEIGHT])
        if not (x <= 0 or y <= 0 or x + w >= W or y + h >= H) and hst[i, cv2.CC_STAT_AREA] < cap:
            keep[hl == i] = 255
    return keep


DISTS = np.linspace(0.18, 1.0, 9)


def fit_iris(bgr, gray, orbit, held_r, seed):
    H, W = gray.shape
    inside0 = orbit > 0
    if int(inside0.sum()) < 80:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    vth = max(110.0, float(np.percentile(V[inside0], 55)))
    # STEP 2 -- block the ALMOND = iris + sclera TOGETHER = the opening (everything that is NOT lid skin),
    # lashes stripped, holes filled -> one clean almond shape. Bright orange lid skin is excluded (the dark
    # iris is kept: skin must also be bright, so low-value iris is never called skin).
    skin = (Hh >= 8) & (Hh <= 30) & (S > 45) & (V > 0.55 * vth)
    araw = (inside0 & (~skin)).astype(np.uint8) * 255
    araw = cv2.morphologyEx(araw, cv2.MORPH_OPEN, E(held_r * 0.14))      # strip lashes / thin skin bridges
    n, lab, st, _ = cv2.connectedComponentsWithStats(araw)
    if n <= 1:
        return None
    ai = -1                                                              # the almond is the blob that
    if seed is not None:                                                 # CONTAINS the eye centre (iris),
        sx, sy = int(round(seed[0])), int(round(seed[1]))               # not merely the largest
        if 0 <= sx < W and 0 <= sy < H and lab[sy, sx] > 0:
            ai = int(lab[sy, sx])
    if ai < 0:
        oc = np.argwhere(inside0).mean(0)                                # fall back: orbit-centre blob
        ocy, ocx = int(oc[0]), int(oc[1])
        if 0 <= ocx < W and 0 <= ocy < H and lab[ocy, ocx] > 0:
            ai = int(lab[ocy, ocx])
    if ai < 0:
        ai = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    almond = (lab == ai).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(almond, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(almond, cnts, -1, 255, -1)                         # fill -> solid almond
    am = almond > 0
    if int(am.sum()) < 0.15 * int(inside0.sum()):
        return None
    apt = np.column_stack(np.where(almond > 0))[:, ::-1].astype(np.float32)
    ac = apt.mean(0); proj = apt - ac
    half = max(1.0, float(np.percentile(np.hypot(proj[:, 0], proj[:, 1]), 90)))
    # STEP 3 -- inside the almond: sclera (white/pink), then IRIS = ALMOND - SCLERA (the dark remainder)
    scl = find_sclera(bgr, am, held_r)
    if int((scl > 0).sum()) < 0.02 * np.pi * held_r * held_r:
        return None
    iris_reg = (am & (scl == 0)).astype(np.uint8) * 255
    iris_reg = cv2.morphologyEx(iris_reg, cv2.MORPH_OPEN, E(held_r * 0.10))
    scl_dil = cv2.dilate(scl, E(held_r * 0.12))
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(iris_reg)
    amin = 0.03 * np.pi * held_r * held_r
    best = None
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] < amin:
            continue
        comp = lab2 == i
        if int((comp & (scl_dil > 0)).sum()) < 3:                       # the iris borders the sclera (limbus)
            continue
        d = math.hypot(cen2[i][0] - seed[0], cen2[i][1] - seed[1]) if seed is not None else 0.0
        score = st2[i, cv2.CC_STAT_AREA] * math.exp(-(d * d) / (2 * (0.9 * held_r) ** 2))
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    comp = (lab2 == best[1]).astype(np.uint8)
    ys, xs = np.where(comp > 0)
    cx, cy = float(xs.mean()), float(ys.mean())                         # centre of the iris region
    gxv, gyv = cx - ac[0], cy - ac[1]; gmag = math.hypot(gxv, gyv); frac = min(0.93, gmag / half)
    minor = held_r * math.sqrt(max(0.04, 1 - frac * frac)); major = float(held_r)
    tilt = math.degrees(math.atan2(gxv, -gyv)) if gmag > 1 else 0.0
    return dict(cx=cx, cy=cy, major=major, minor=minor, tilt=tilt, scl=scl, almond=almond)


if __name__ == "__main__":
    GREEN = (0, 200, 0); YELLOW = (0, 255, 255); RED = (0, 0, 255)
    REP = {187, 192, 195, 227, 262, 267, 269, 300, 311, 341, 348, 350, 353, 378, 382, 460, 469, 475}
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ds = min(1.0, 1280.0 / max(1, W)); ow, oh = int(W * ds), int(H * ds)
    writer = cv2.VideoWriter(str(REV / "iris_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    last_good = {ek: None for ek in ("L", "R")}; miss = {ek: 0 for ek in ("L", "R")}
    counts = {ek: {"ok": 0, "gap": 0} for ek in ("L", "R")}; rows = []
    (REV / "rep").mkdir(exist_ok=True)
    fn = 0
    while fn < NF:
        ok, frame = cap.read()
        if not ok:
            break
        fn += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        big = frame.copy()
        for ek in ("L", "R"):
            c = orb.get(fn, {}).get(ek)
            if c is None:
                continue
            held_r = iris0.get(ek, (0, 0, 30))[2]
            orbit_mask = np.zeros((H, W), np.uint8); cv2.fillPoly(orbit_mask, [c.astype(np.int32)], 255)
            seed = last_good[ek]
            if seed is None and ek in iris0:
                seed = (iris0[ek][0], iris0[ek][1])
            f = fit_iris(frame, gray, orbit_mask, held_r, seed)
            cv2.polylines(big, [c.astype(np.int32)], True, (80, 80, 80), 1)   # orbit search region (faint)
            if f is None:
                counts[ek]["gap"] += 1; miss[ek] += 1
                if miss[ek] >= 8:
                    last_good[ek] = None
                rows.append(dict(frame=fn, eye=ek, cx="", cy="", minor=""))
                continue
            almond = f["almond"]
            acnts, _ = cv2.findContours(almond, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(big, acnts, -1, GREEN, 2)                        # the ALMOND (iris + sclera)
            big[f["scl"] > 0] = (0.7 * big[f["scl"] > 0] + np.array([90, 70, 0])).astype(np.uint8)
            em = np.zeros((H, W), np.uint8)
            cv2.ellipse(em, (int(f["cx"]), int(f["cy"])), (int(f["major"]), int(f["minor"])),
                        f["tilt"], 0, 360, 255, -1)
            em = cv2.bitwise_and(em, almond)                                  # IRIS clipped inside the almond
            cnts, _ = cv2.findContours(em, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(big, cnts, -1, YELLOW, 2)
            cv2.circle(big, (int(f["cx"]), int(f["cy"])), 4, RED, -1)
            last_good[ek] = (f["cx"], f["cy"]); miss[ek] = 0; counts[ek]["ok"] += 1
            rows.append(dict(frame=fn, eye=ek, cx=round(f["cx"], 1), cy=round(f["cy"], 1), minor=round(f["minor"], 1)))
        disp = cv2.resize(big, (ow, oh))
        cv2.putText(disp, f"frame {fn}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(disp, f"frame {fn}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        writer.write(disp)
        if fn in REP:
            cv2.imwrite(str(REV / "rep" / f"f{fn:04d}.png"), disp)
    writer.release(); cap.release()
    with open(REV / "iris_points.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["frame", "eye", "cx", "cy", "minor"]); w.writeheader(); w.writerows(rows)
    print("L:", counts["L"], " R:", counts["R"], " ->", REV / "iris_overlay.mp4")
