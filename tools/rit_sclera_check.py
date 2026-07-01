"""Emit 10 matched pairs: my MARKED sclera (cyan) + the CLEAN original eye crop, for Dr. K to mark the
correct sclera. Uses the v3 sclera detector. Output -> ~/Desktop/RIT_sclera_check/{marked,clean}/.
"""
import csv, json
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
DST = Path.home()/"OneDrive/Desktop/RIT_sclera_check"
(DST/"marked").mkdir(parents=True, exist_ok=True); (DST/"clean").mkdir(parents=True, exist_ok=True)

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
    Hh = hsv[:,:,0]; S = hsv[:,:,1]; V = hsv[:,:,2]
    vin = V[inside]
    if vin.size < 50: return np.zeros((H,W),np.uint8)
    vth = max(110.0, float(np.percentile(vin,55)))
    whitish = (V > vth) & (S < 65)
    redpink = (V > vth*0.80) & ((Hh < 10)|(Hh > 165)) & (S < 175)
    skin = (Hh >= 10) & (Hh <= 32) & (S > 55)
    scl = (inside & (whitish|redpink) & (~skin)).astype(np.uint8)*255
    scl = cv2.morphologyEx(scl, cv2.MORPH_OPEN, _kk(held_r*0.08))
    scl = cv2.morphologyEx(scl, cv2.MORPH_CLOSE, _kk(held_r*0.18))
    n,lab,st,_ = cv2.connectedComponentsWithStats(scl)
    if n<=1: return np.zeros((H,W),np.uint8)
    amax = float(st[1:,cv2.CC_STAT_AREA].max())
    keep = np.zeros((H,W),np.uint8)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA] >= max(0.05*np.pi*held_r*held_r, 0.30*amax): keep[lab==i]=255
    nh,hl,hst,_ = cv2.connectedComponentsWithStats(cv2.bitwise_not(keep))
    cap = 0.20*np.pi*held_r*held_r
    for i in range(1,nh):
        x,y,w,h = (hst[i,cv2.CC_STAT_LEFT],hst[i,cv2.CC_STAT_TOP],hst[i,cv2.CC_STAT_WIDTH],hst[i,cv2.CC_STAT_HEIGHT])
        if not (x<=0 or y<=0 or x+w>=W or y+h>=H) and hst[i,cv2.CC_STAT_AREA] < cap: keep[hl==i]=255
    return keep

PAIRS = [(60,'L'),(130,'R'),(187,'R'),(227,'R'),(262,'L'),(300,'L'),(341,'L'),(382,'R'),(460,'L'),(540,'R')]
want = {f-1: (f,ek) for f,ek in PAIRS}
cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
idx=0; made=0
while want:
    ok, frame = cap.read()
    if not ok: break
    if idx in want:
        fn, ek = want.pop(idx)
        c = orb.get(fn,{}).get(ek)
        if c is not None:
            held_r = iris0.get(ek,(0,0,30))[2]
            m = np.zeros((H,W),np.uint8); cv2.fillPoly(m,[c.astype(np.int32)],255); inside=m>0
            scl = find_sclera(frame, inside, held_r)
            x0,y0=c.min(0); x1,y1=c.max(0); mx=(x1-x0)*0.30; my=(y1-y0)*0.55
            X0=max(0,int(x0-mx));Y0=max(0,int(y0-my));X1=min(W,int(x1+mx));Y1=min(H,int(y1+my))
            clean = frame[Y0:Y1,X0:X1].copy()
            mk = frame.copy()
            mk[scl>0] = (0.45*mk[scl>0] + np.array([255,255,0])*0.55).astype(np.uint8)
            cv2.polylines(mk,[c.astype(np.int32)],True,(0,200,0),2)
            marked = mk[Y0:Y1,X0:X1]
            clz = cv2.resize(clean,None,fx=2,fy=2); mkz = cv2.resize(marked,None,fx=2,fy=2)
            cv2.rectangle(clz,(0,0),(clz.shape[1],26),(0,0,0),-1)
            cv2.putText(clz,f"f{fn} {ek}  CLEAN - mark the sclera",(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
            cv2.rectangle(mkz,(0,0),(mkz.shape[1],26),(0,0,0),-1)
            cv2.putText(mkz,f"f{fn} {ek}  MY SCLERA (cyan)",(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
            cv2.imwrite(str(DST/"clean"/f"f{fn:04d}_{ek}.png"), clz)
            cv2.imwrite(str(DST/"marked"/f"f{fn:04d}_{ek}.png"), mkz)
            made+=1
    idx+=1
cap.release()
print("pairs made:", made, "->", DST)
