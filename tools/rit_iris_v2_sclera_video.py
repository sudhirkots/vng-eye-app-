"""RIT iris method v2 -- STEP 1 full-video overlay: mark ONLY the sclera on every frame, for scrubbing.

Sclera (Dr. K): everything between the lids that is not the dark iris -- white OR pink, running into both
corners as triangular tails, filled solid (vessel spots are still sclera); NOT the orange-brown lid skin.
Cyan tint + green almond + frame number. Scrubbable mp4. No iris, no ellipse, no commit.
"""
import csv, json, shutil
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
REV = OUT/"_rit_iris_v2"; REV.mkdir(parents=True, exist_ok=True)
MP4 = REV/"sclera_only_overlay.mp4"

orb = defaultdict(dict)
for r in csv.DictReader(open(OLP, encoding="utf-8")):
    try: orb[int(r["frame"])][r["eye"]] = np.array(json.loads(r["contour_json"]), np.float32)
    except Exception: pass

def _kk(k):
    s = max(3, int(k)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))

def find_sclera(bgr, inside, held_r):
    H, W = inside.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    vin = V[inside]
    if vin.size < 50: return np.zeros((H, W), np.uint8)
    vth = max(110.0, float(np.percentile(vin, 55)))
    whitish = (V > vth) & (S < 65)
    redpink = (V > vth*0.80) & ((Hh < 10) | (Hh > 165)) & (S < 175)
    skin = (Hh >= 10) & (Hh <= 32) & (S > 55)
    scl = (inside & (whitish | redpink) & (~skin)).astype(np.uint8)*255
    scl = cv2.morphologyEx(scl, cv2.MORPH_OPEN, _kk(held_r*0.08))
    scl = cv2.morphologyEx(scl, cv2.MORPH_CLOSE, _kk(held_r*0.18))
    n, lab, st, _ = cv2.connectedComponentsWithStats(scl)
    if n <= 1: return np.zeros((H, W), np.uint8)
    amax = float(st[1:, cv2.CC_STAT_AREA].max())
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= max(0.05*np.pi*held_r*held_r, 0.30*amax):
            keep[lab == i] = 255
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(keep))
    hole_cap = 0.20*np.pi*held_r*held_r
    for i in range(1, nh):
        x,y,w,h = (hst[i,cv2.CC_STAT_LEFT],hst[i,cv2.CC_STAT_TOP],hst[i,cv2.CC_STAT_WIDTH],hst[i,cv2.CC_STAT_HEIGHT])
        if not (x<=0 or y<=0 or x+w>=W or y+h>=H) and hst[i,cv2.CC_STAT_AREA] < hole_cap:
            keep[hl == i] = 255
    return keep

cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
ds = min(1.0, 1280.0/max(1,W)); ow,oh = int(W*ds), int(H*ds)
writer = cv2.VideoWriter(str(MP4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow,oh))
fn = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    fn += 1
    big = frame.copy()
    for ek in ("L","R"):
        c = orb.get(fn,{}).get(ek)
        if c is None: continue
        held_r = iris0.get(ek,(0,0,30))[2]
        m = np.zeros((H,W),np.uint8); cv2.fillPoly(m,[c.astype(np.int32)],255); inside = m>0
        scl = find_sclera(frame, inside, held_r)
        big[scl>0] = (0.45*big[scl>0] + np.array([255,255,0])*0.55).astype(np.uint8)
        cv2.polylines(big, [c.astype(np.int32)], True, (0,200,0), 2)
    disp = cv2.resize(big, (ow,oh))
    cv2.putText(disp, f"frame {fn}  SCLERA only (cyan)", (10,26), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3)
    cv2.putText(disp, f"frame {fn}  SCLERA only (cyan)", (10,26), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
    writer.write(disp)
writer.release(); cap.release()
dst = Path.home()/"OneDrive/Desktop/RIT_sclera_only.mp4"
shutil.copy(str(MP4), str(dst))
print(f"frames {fn} -> {dst}")
