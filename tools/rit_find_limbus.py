"""Find the iris from Dr. K's blue sclera marks, using the REAL almond (tracked eye opening = orbit).
(1) Almond = orbit contour (the opening; only sclera or iris live here). (2) VOTE inward at a range of
distances from each sclera-edge-facing-dark point; only votes INSIDE the almond count -> foreshortened
centres allowed, nothing outside the opening. (3) Peak = centre; complete an OVAL foreshortened by gaze.
"""
import csv, json, math
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

OUT = Path("outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
OLP = OUT/"_rit_orbit_lock_probe/orbit_lock_points.csv"
iris0 = {ek: float(v["radius"]) for ek,v in (json.load(open(OUT/"approved_landmarks.json",encoding="utf-8")).get("iris") or {}).items() if v}
SRC = Path.home()/"OneDrive/Desktop/RIT_sclera_check/clean"
DST = Path.home()/"OneDrive/Desktop/RIT_limbus"; DST.mkdir(parents=True, exist_ok=True)
def E(k): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
DISTS = np.linspace(0.25,1.0,7)

orb = defaultdict(dict)
for r0 in csv.DictReader(open(OLP, encoding="utf-8")):
    try: orb[int(r0["frame"])][r0["eye"]] = np.array(json.loads(r0["contour_json"]), np.float32)
    except Exception: pass

for f in sorted(SRC.glob("*.png")):
    fn = int(f.name[1:5]); ek = 'L' if '_L' in f.name else 'R'
    r = 2.0 * iris0.get(ek, 88.0)
    c = orb.get(fn,{}).get(ek)
    img = cv2.imread(str(f)); H,W = img.shape[:2]
    out = img.copy()
    if c is None: cv2.imwrite(str(DST/f.name),out); continue
    # reproduce the crop transform used to make these images, map the orbit into crop coords
    x0,y0=c.min(0); x1,y1=c.max(0); mx=(x1-x0)*0.30; my=(y1-y0)*0.55
    X0=max(0,int(x0-mx)); Y0=max(0,int(y0-my))
    cc = ((c - np.array([X0,Y0]))*2.0).astype(np.int32)
    almond = np.zeros((H,W),np.uint8); cv2.fillPoly(almond,[cc],255)
    ac = cc.mean(0); proj = cc-ac; half = max(1.0,float(np.percentile(np.hypot(proj[:,0],proj[:,1]),90)))
    cv2.polylines(out,[cc],True,(255,180,0),1)
    B,G,R = img[:,:,0].astype(int),img[:,:,1].astype(int),img[:,:,2].astype(int)
    blue = ((B>130)&(B-R>50)&(B-G>20)).astype(np.uint8)*255
    if blue.sum() >= 500:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark = (gray<90).astype(np.uint8)*255
        bluef = cv2.GaussianBlur(blue.astype(np.float32),(0,0),3)
        gx=cv2.Sobel(bluef,cv2.CV_32F,1,0,ksize=5); gy=cv2.Sobel(bluef,cv2.CV_32F,0,1,ksize=5)
        edge = cv2.morphologyEx(blue,cv2.MORPH_GRADIENT,E(5))
        cand = (edge>0)&(cv2.dilate(dark,E(7))>0)
        ys,xs = np.where(cand)
        acc = np.zeros((H,W),np.float32)
        for x,y in zip(xs,ys):
            nx,ny=-gx[y,x],-gy[y,x]; nrm=(nx*nx+ny*ny)**0.5
            if nrm<1e-3: continue
            nx,ny=nx/nrm,ny/nrm
            for dd in DISTS:
                cx=int(round(x+dd*r*nx)); cy=int(round(y+dd*r*ny))
                if 0<=cx<W and 0<=cy<H and almond[cy,cx]>0 and blue[cy,cx]==0: acc[cy,cx]+=1
        acc=cv2.GaussianBlur(acc,(0,0),6)
        if acc.max()>0:
            cy,cx=np.unravel_index(int(acc.argmax()),acc.shape)
            gxv,gyv=cx-ac[0],cy-ac[1]; gmag=math.hypot(gxv,gyv); frac=min(0.93,gmag/half)
            minor=r*math.sqrt(max(0.04,1-frac*frac)); major=r
            tilt=math.degrees(math.atan2(gxv,-gyv)) if gmag>1 else 0.0
            cv2.ellipse(out,(int(cx),int(cy)),(int(major),int(minor)),tilt,0,360,(0,255,0),2)
            cv2.circle(out,(int(cx),int(cy)),4,(0,255,255),-1)
            cv2.putText(out,f"r={int(r)} minor={int(minor)} frac={frac:.2f}",(8,46),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
    cv2.imwrite(str(DST/f.name),out)
print("real-almond vote + oval ->",DST)
