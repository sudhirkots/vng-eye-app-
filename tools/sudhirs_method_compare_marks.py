"""Compare Dr. K's TRUE iris marks (from the paint tool zip) against Sudhir's iris-sclera-almond auto-pick,
per frame, and diagnose which rule broke. Auto mask is recovered from the *_prediction.png overlay vs the
clean crop (no re-run needed). All in the 2x crop frame stored in to_mark/meta.json.
"""
import sys, os, json, math, zipfile, io
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parent.parent
CLIP = "nystagmus at rest to left in right vestibular neuritis"
TOMARK = REPO / "outputs" / f"{CLIP}_tracked" / "sudhirs_iris_sclera_almond_method" / "to_mark"
ZIP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "RIT_masks.zip"
OUT = TOMARK.parent / "compare"; OUT.mkdir(exist_ok=True)

meta = json.load(open(TOMARK / "meta.json", encoding="utf-8"))

# --- user masks from the zip: red = true iris ---
user = {}
with zipfile.ZipFile(ZIP) as z:
    for n in z.namelist():
        if not n.lower().endswith(".png"): continue
        uid = Path(n).name[:-4]
        arr = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        if arr is None: continue
        B, G, R = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
        user[uid] = ((R > 100) & (R - B > 40) & (R - G > 40)).astype(np.uint8)   # red iris only

def auto_from_pred(uid):
    clean = cv2.imread(str(TOMARK / f"{uid}.png")); pred = cv2.imread(str(TOMARK / f"{uid}_prediction.png"))
    if clean is None or pred is None: return None
    cG, pG = clean[:, :, 1].astype(int), pred[:, :, 1].astype(int)
    cR, pR = clean[:, :, 2].astype(int), pred[:, :, 2].astype(int)
    return ((cG - pG > 15) & (pR >= cR - 5)).astype(np.uint8)     # where red overlay was blended in

def iou(a, b):
    a = a > 0; b = b > 0; u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else float("nan")

def centroid(m):
    M = cv2.moments(m)
    return (M["m10"]/M["m00"], M["m01"]/M["m00"]) if M["m00"] > 0 else None

rows = []
for uid in sorted(meta):
    um = user.get(uid); am = auto_from_pred(uid)
    if um is None: continue
    if am is not None and um.shape != am.shape:
        am = cv2.resize(am, (um.shape[1], um.shape[0]), interpolation=cv2.INTER_NEAREST)
    u_area = int(um.sum()); a_area = int(am.sum()) if am is not None else 0
    if u_area < 30:                                   # user left it blank
        rows.append(dict(uid=uid, verdict="not_marked", iou=None)); continue
    ov = iou(um, am) if a_area else 0.0
    uc, ac = centroid(um), (centroid(am) if a_area else None)
    dist = math.hypot(uc[0]-ac[0], uc[1]-ac[1]) if (uc and ac) else None
    inter = int(((um > 0) & (am > 0)).sum()) if a_area else 0
    frac_auto_on_iris = inter / a_area if a_area else 0.0
    # diagnosis of the broken rule
    if a_area == 0:
        v = "MISS — auto marked nothing (no candidate passed 'dark blob abutting sclera')"
    elif ov < 0.10:
        v = "WRONG REGION — auto is off the true iris (picked a dark blob that isn't the iris; sclera-adjacency/convex-arc rule failed)"
    elif frac_auto_on_iris < 0.6:
        v = "BLEED/MISPLACED — much of auto lies off the iris (fit pulled onto lid/lash band beside the limbus)"
    elif ov < 0.5:
        v = "PARTIAL — right region, wrong extent/shape (circle fit under- or over-shot the true iris)"
    else:
        v = "OK — matches your mark"
    rows.append(dict(uid=uid, verdict=v, iou=round(ov, 3), centre_dist_px=(round(dist, 1) if dist else None),
                     frac_auto_on_iris=round(frac_auto_on_iris, 2), user_area=u_area, auto_area=a_area))

# rank worst first (marked frames), then build a montage: green=your iris, red=auto iris
def keyf(r): return (r["iou"] is None, r["iou"] if r["iou"] is not None else 9)
marked = [r for r in rows if r["verdict"] != "not_marked"]
marked.sort(key=keyf)
tiles = []
for r in marked:
    uid = r["uid"]; clean = cv2.imread(str(TOMARK / f"{uid}.png"))
    um = user[uid]; am = auto_from_pred(uid)
    if am is not None and am.shape != um.shape: am = cv2.resize(am, (um.shape[1], um.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = clean.copy()
    if am is not None: cv2.drawContours(vis, cv2.findContours(am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
    cv2.drawContours(vis, cv2.findContours(um, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
    lbl = f"{uid} IoU={r['iou']}"
    cv2.putText(vis, lbl, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4); cv2.putText(vis, lbl, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    tiles.append(cv2.resize(vis, (360, int(360*vis.shape[0]/vis.shape[1]))))
if tiles:
    hmax = max(t.shape[0] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, hmax-t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    grid = [np.hstack((tiles[i:i+4] + [np.zeros_like(tiles[0])]*4)[:4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(OUT / "compare_montage.png"), np.vstack(grid))

json.dump(rows, open(OUT / "compare.json", "w"), indent=1)
marked_iou = [r["iou"] for r in marked if r["iou"] is not None]
print(f"marked {len(marked)} / {len(rows)} frames | median IoU {np.median(marked_iou):.2f}" if marked_iou else "no marks")
for r in marked:
    print(f"  {r['uid']}: IoU={r['iou']}  {r['verdict']}")
print("montage ->", OUT / "compare_montage.png")
