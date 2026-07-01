"""IRIS = ALMOND - SCLERA (Dr. K). The almond (tracked eye opening) is only sclera or iris; remove the
sclera and what remains is the iris. No dark-hunting, no voting. Highlight iris (red), centre (yellow),
and complete a foreshortened oval. In: RIT_sclera_check/clean (blue=Dr K sclera).  Out: RIT_iris_sub.
"""
import csv, json, math
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

OUT = Path("outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
OLP = OUT/"_rit_orbit_lock_probe/orbit_lock_points.csv"
iris0 = {ek: float(v["radius"]) for ek,v in (json.load(open(OUT/"approved_landmarks.json",encoding="utf-8")).get("iris") or {}).items() if v}
SRC = Path.home()/"OneDrive/Desktop/RIT_sclera_check/clean"
DST = Path.home()/"OneDrive/Desktop/RIT_iris_sub"; DST.mkdir(parents=True, exist_ok=True)
def E(k): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))

orb = defaultdict(dict)
for r0 in csv.DictReader(open(OLP, encoding="utf-8")):
    try: orb[int(r0["frame"])][r0["eye"]] = np.array(json.loads(r0["contour_json"]), np.float32)
    except Exception: pass

for f in sorted(SRC.glob("*.png")):
    fn=int(f.name[1:5]); ek='L' if '_L' in f.name else 'R'
    r = 2.0*iris0.get(ek,88.0)
    c = orb.get(fn,{}).get(ek)
    img=cv2.imread(str(f)); H,W=img.shape[:2]; out=img.copy()
    if c is None: cv2.imwrite(str(DST/f.name),out); continue
    x0,y0=c.min(0); x1,y1=c.max(0); mx=(x1-x0)*0.30; my=(y1-y0)*0.55
    X0=max(0,int(x0-mx)); Y0=max(0,int(y0-my))
    cc=((c-np.array([X0,Y0]))*2.0).astype(np.int32)
    almond=np.zeros((H,W),np.uint8); cv2.fillPoly(almond,[cc],255)
    almond=cv2.erode(almond,E(9))                     # pull just inside the lid margins (drop lash fringe)
    ac=cc.mean(0); proj=cc-ac; half=max(1.0,float(np.percentile(np.hypot(proj[:,0],proj[:,1]),90)))
    cv2.polylines(out,[cc],True,(255,180,0),1)
    B,G,R=img[:,:,0].astype(int),img[:,:,1].astype(int),img[:,:,2].astype(int)
    blue=((B>130)&(B-R>50)&(B-G>20)).astype(np.uint8)*255
    # IRIS = almond minus sclera
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); Hh,Sk,Vk=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    skin=((Hh>=8)&(Hh<=30)&(Sk>50)).astype(np.uint8)*255   # lid/brow skin (orange-brown) is NOT the opening
    iris = cv2.bitwise_and(almond, cv2.bitwise_not(cv2.dilate(blue,E(5))))
    iris = cv2.bitwise_and(iris, cv2.bitwise_not(skin))      # opening = sclera or iris only; drop lid skin
    iris = cv2.morphologyEx(iris, cv2.MORPH_OPEN, E(13))      # ignore thin lash strands
    n,lab,st,cen = cv2.connectedComponentsWithStats(iris)
    if n>1:
        i=int(1+np.argmax(st[1:,cv2.CC_STAT_AREA])); m=(lab==i).astype(np.uint8)
        ys,xs=np.where(m>0); cx,cy=float(xs.mean()),float(ys.mean())
        red=out.copy(); red[m>0]=(0,0,255); out=cv2.addWeighted(out,0.6,red,0.4,0)
        gxv,gyv=cx-ac[0],cy-ac[1]; gmag=math.hypot(gxv,gyv); frac=min(0.93,gmag/half)
        minor=r*math.sqrt(max(0.04,1-frac*frac)); major=r
        tilt=math.degrees(math.atan2(gxv,-gyv)) if gmag>1 else 0.0
        cv2.ellipse(out,(int(cx),int(cy)),(int(major),int(minor)),tilt,0,360,(0,255,0),2)
        cv2.circle(out,(int(cx),int(cy)),4,(0,255,255),-1)
        cv2.putText(out,f"iris=almond-sclera  minor={int(minor)} frac={frac:.2f}",(8,46),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
    cv2.imwrite(str(DST/f.name),out)
print("iris = almond - sclera ->",DST)
