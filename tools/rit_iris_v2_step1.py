"""RIT iris method v2 -- STEP 1 ONLY: find the white sclera inside the almond opening.

Sclera rule (Dr. K): the ONE large, UNIFORMLY white patch. Discard blotchy specks (skin reflections).
Visualise the detected sclera (cyan tint) on the flagged frames so it can be verified BEFORE step 2.
No iris, no ellipse, no commit. Output -> ~/Desktop/RIT_step1/.
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

OUT = Path("outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
OLP = OUT/"_rit_orbit_lock_probe/orbit_lock_points.csv"
appr = json.load(open(OUT/"approved_landmarks.json", encoding="utf-8"))
video = appr.get("video")
if not video or not Path(video).exists():
    video = "samples/nystagmus at rest to left in right vestibular neuritis.mp4"
iris0 = {ek:(float(v["x"]),float(v["y"]),float(v["radius"])) for ek,v in (appr.get("iris") or {}).items() if v}
DST = Path.home()/"OneDrive/Desktop/RIT_step1"; DST.mkdir(parents=True, exist_ok=True)

orb = defaultdict(dict)
for r in csv.DictReader(open(OLP, encoding="utf-8")):
    try: orb[int(r["frame"])][r["eye"]] = np.array(json.loads(r["contour_json"]), np.float32)
    except Exception: pass

def _kk(k):
    s = max(3, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def find_sclera(bgr, inside, held_r):
    """Sclera = EVERYTHING between the lids that is not the dark iris (Dr. K): white OR pink, running into
    both corners as triangular tails, filled solid (vessel spots are still sclera). It is NOT the lid skin
    (orange-brown). So: keep bright pixels that are whitish (low sat) OR pink/red (red hue), EXCLUDE
    orange-brown skin by hue, then fill only the SMALL vessel holes (never the big iris hole)."""
    H, W = inside.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    vin = V[inside]
    if vin.size < 50:
        return np.zeros((H, W), np.uint8)
    vth = max(110.0, float(np.percentile(vin, 55)))
    whitish = (V > vth) & (S < 65)                                  # white sclera
    redpink = (V > vth * 0.80) & ((Hh < 10) | (Hh > 165)) & (S < 175)   # PINK sclera (red hue) -> the tails
    skin = (Hh >= 10) & (Hh <= 32) & (S > 55)                       # orange/brown lid + skin -> excluded
    scl = (inside & (whitish | redpink) & (~skin)).astype(np.uint8) * 255
    scl = cv2.morphologyEx(scl, cv2.MORPH_OPEN, _kk(held_r * 0.08))    # drop thin lash strands / specks
    scl = cv2.morphologyEx(scl, cv2.MORPH_CLOSE, _kk(held_r * 0.18))   # bridge vessel gaps
    n, lab, st, _ = cv2.connectedComponentsWithStats(scl)
    if n <= 1:
        return np.zeros((H, W), np.uint8)
    amax = float(st[1:, cv2.CC_STAT_AREA].max())
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= max(0.05 * np.pi * held_r * held_r, 0.30 * amax):
            keep[lab == i] = 255
    # fill ONLY small enclosed holes (capillary spots); the big IRIS hole is left open (iris is not sclera)
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(keep))
    hole_cap = 0.20 * np.pi * held_r * held_r
    for i in range(1, nh):
        x, y, w, h = (hst[i, cv2.CC_STAT_LEFT], hst[i, cv2.CC_STAT_TOP],
                      hst[i, cv2.CC_STAT_WIDTH], hst[i, cv2.CC_STAT_HEIGHT])
        border = (x <= 0 or y <= 0 or x + w >= W or y + h >= H)
        if (not border) and hst[i, cv2.CC_STAT_AREA] < hole_cap:
            keep[hl == i] = 255
    return keep

FRAMES = [187, 227, 262, 267, 341, 348, 350, 353, 378, 382, 460]
want = {f-1: f for f in FRAMES}
cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
idx = 0
while want:
    ok, frame = cap.read()
    if not ok: break
    if idx in want:
        fn = want.pop(idx)
        for ek in ("L", "R"):
            c = orb.get(fn, {}).get(ek)
            if c is None: continue
            held_r = iris0.get(ek, (0,0,30))[2]
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [c.astype(np.int32)], 255); inside = m > 0
            scl = find_sclera(frame, inside, held_r)
            vis = frame.copy()
            vis[scl > 0] = (0.45*vis[scl > 0] + np.array([255,255,0])*0.55).astype(np.uint8)  # cyan tint
            cv2.polylines(vis, [c.astype(np.int32)], True, (0,200,0), 2)
            x0,y0 = c.min(0); x1,y1 = c.max(0); mx=(x1-x0)*0.30; my=(y1-y0)*0.5
            X0=max(0,int(x0-mx));Y0=max(0,int(y0-my));X1=min(W,int(x1+mx));Y1=min(H,int(y1+my))
            crop = cv2.resize(vis[Y0:Y1, X0:X1], None, fx=2, fy=2)
            cv2.putText(crop, f"f{fn} {ek}  STEP1 sclera (cyan)", (6,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.imwrite(str(DST/f"scl_{fn:04d}_{ek}.png"), crop)
    idx += 1
cap.release()
print("step1 sclera saved ->", DST)
