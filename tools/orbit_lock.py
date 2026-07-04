"""ORBIT LOCK (Step 2c-ii) — a head/camera-fixed, eye-local coordinate frame per eye.

The apparent "head motion" in the image-coordinate gaze is largely the CAMERA moving. Fix it by locking to
the eye itself: build an eye-local frame from clinician-approved landmarks (medial/lateral canthus, upper/
lower eyelid margin), track it through the clip, and measure the iris INSIDE this frame.

Landmarks come from the RIT Orbit Lock ground truth already marked (meta.json: eye-opening almond polygons on
~20 frames/eye). Per marked frame we derive 4 landmarks by PCA of the polygon (principal axis ends = canthi;
minor axis ends = lid midpoints), then interpolate them across every frame. EllSeg gives the iris centre.

Diagnostic (confirm the orbit tracks smoothly BEFORE using it for gaze):
  orbit_lock.mp4        overlay: canthus line, lid line, eye-local axes, iris centre, iris-in-orbit (gx,gy)
  orbit_lock_track.png  orbit centre x/y + canthus distance across the clip (should be smooth)
  orbit_lock.csv        frame,eye, canthus/lid landmarks, orbit centre, scale, iris centre, gx, gy
Run with the rit-nets venv python.
"""
import sys, os, json, math, csv
from pathlib import Path
import numpy as np, cv2

ELL = Path(os.environ.get("ELLSEG_DIR", Path.home() / "rit-nets" / "EllSeg"))
sys.path.insert(0, str(ELL))
for _a, _t in (("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import torch
from modelSummary import model_dict

REPO = Path(__file__).resolve().parent.parent
CLIP = os.environ.get("RIT_CLIP", "nystagmus at rest to left in right vestibular neuritis")
VIDEO = REPO / "samples" / f"{CLIP}.mp4"
OUTBASE = REPO / "outputs" / f"{CLIP}_tracked"
GT = OUTBASE / "RIT_ground_truth"
OUTDIR = OUTBASE / "step2c_orbit_lock"
OUTDIR.mkdir(parents=True, exist_ok=True)
OPH, OPW = 240, 320

model = model_dict["ritnet_v3"]
nd = torch.load(ELL / "weights" / "all.git_ok", map_location="cpu", weights_only=False)
model.load_state_dict(nd["state_dict"], strict=True); model.eval()

def preprocess(gray):
    h, w = gray.shape; sc = OPW / w
    nw, nh = int(round(w*sc)), int(round(h*sc))
    img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    if OPH >= nh:
        pad = OPH-nh; top = pad//2; img = np.pad(img, ((top, pad-top), (0, 0))); valid = (top, nh)
    else:
        top = (nh-OPH)//2; img = img[top:top+OPH, :]; valid = (0, OPH)
    imgn = (img-img.mean())/(img.std()+1e-6)
    return torch.from_numpy(imgn).unsqueeze(0).unsqueeze(0).to(torch.float32), valid

def seg_forward(x):
    with torch.no_grad():
        x4, x3, x2, x1, xc = model.enc(x); seg = model.dec(x4, x3, x2, x1, xc)
    return seg.max(1)[1][0].numpy().astype(np.uint8)

def iris_centre(gray_full, cx, cy, hw, hh, W, H):
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    if disc.sum() < 30: return None
    m = cv2.moments(disc); return (m["m10"]/m["m00"]+x0, m["m01"]/m["m00"]+y0)

# ---- clinician marks -> 4 landmarks per marked frame ----
meta = json.load(open(GT / "meta.json", encoding="utf-8"))
def poly_full(e):
    X0, Y0, s = e["X0"], e["Y0"], e["scale"]
    return np.array([[x/s+X0, y/s+Y0] for x, y in e["orbit_crop"]], np.float64)

def landmarks(poly):
    """PCA of the eye-opening polygon: principal-axis extremes = canthi, minor-axis extremes = lid midpoints.
    Returns dict with CL,CR (canthus left/right by image x) and LT,LB (lid top/bottom by image y)."""
    C = poly.mean(0); U, S, Vt = np.linalg.svd(poly - C)
    major, minor = Vt[0], Vt[1]
    pm = (poly - C) @ major; a, b = poly[pm.argmin()], poly[pm.argmax()]
    pn = (poly - C) @ minor; c, d = poly[pn.argmin()], poly[pn.argmax()]
    CL, CR = (a, b) if a[0] <= b[0] else (b, a)
    LT, LB = (c, d) if c[1] <= d[1] else (d, c)
    return CL, CR, LT, LB

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); FPS = cap.get(5) or 30.0
cap.release()

def smoothc(a, w=15):
    if len(a) < 3: return a
    k = np.ones(w)/w; return np.convolve(np.pad(a, w//2, mode="edge"), k, mode="valid")[:len(a)]

# RIGID CANTHUS FRAME: distance + vertical scale FIXED (median of marks); angle smoothed. The MIDPOINT
# (translation) is TEMPLATE-TRACKED by optical flow of the eye-region texture (canthi/lids/skin, iris
# excluded), re-anchored at every clinician mark, carry-forward on low confidence. Canthi are reconstructed
# rigidly = midpoint +- (D0/2)*(cos th, sin th), so the pair can never drift independently.
markinfo = {}
for ek in ("L", "R"):
    ents = sorted([v for v in meta.values() if v.get("eye") == ek and "orbit_crop" in v], key=lambda v: v["frame"])
    if not ents: continue
    fr = []; mid = []; ang = []; dist = []; lh = []
    for v in ents:
        cl, cr, lt, lb = landmarks(poly_full(v)); vec = cr - cl
        fr.append(v["frame"]); mid.append((cl + cr) / 2.0); ang.append(math.atan2(vec[1], vec[0]))
        dist.append(float(np.hypot(*vec))); lh.append(float(np.hypot(*(lb - lt))))
    markinfo[ek] = dict(fr=np.array(fr), mid=np.array(mid), ang=np.array(ang), dist=dist, lh=lh)

orbit = {}
for ek, mi in markinfo.items():
    aa = smoothc(np.interp(np.arange(NF), mi["fr"], np.unwrap(mi["ang"])), 25)
    orbit[ek] = dict(ang=aa, dist=np.full(NF, float(np.median(mi["dist"]))), lh=np.full(NF, float(np.median(mi["lh"]))),
                     D0=float(np.median(mi["dist"])), lh0=float(np.median(mi["lh"])))

# ---- NOSE-BRIDGE ANCHOR: track the stable bridge (shared, rigid) and pin each eye's midpoint to it ----
markC = {}
for ek in orbit:
    ents = sorted([v for v in meta.values() if v.get("eye") == ek and "orbit_crop" in v], key=lambda v: v["frame"])
    markC[ek] = {int(v["frame"]): tuple(np.array(p) for p in landmarks(poly_full(v))[:2]) for v in ents}
theta0 = {ek: float(np.median(markinfo[ek]["ang"])) for ek in orbit}
shared = sorted(set(markC.get("L", {})) & set(markC.get("R", {}))) if len(orbit) == 2 else []
Bmark = {}; offs = {ek: [] for ek in orbit}
for f in shared:
    (CLl, CRl), (CLr, CRr) = markC["L"][f], markC["R"][f]
    midL = (CLl + CRl)/2; midR = (CLr + CRr)/2
    medL = CLl if np.hypot(*(CLl-midR)) < np.hypot(*(CRl-midR)) else CRl      # inner canthus = closest to other eye
    medR = CLr if np.hypot(*(CLr-midL)) < np.hypot(*(CRr-midL)) else CRr
    B = (medL + medR)/2; Bmark[f] = B
    offs["L"].append(midL - B); offs["R"].append(midR - B)                    # fixed bridge->eye-midpoint offset
offset0 = {ek: (np.median(np.array(offs[ek]), 0) if offs[ek] else np.array([0.0, 0.0])) for ek in orbit}
gap = float(np.median([np.hypot(*(offs["L"][i]-offs["R"][i])) for i in range(len(shared))])) if shared else 120.0

# track the nose-bridge point B by optical flow of the stable central patch, re-anchored at shared marks
Btr = np.full((NF, 2), np.nan); Bconf = np.zeros(NF); curB = None; LK = dict(winSize=(21, 21), maxLevel=3)
cap = cv2.VideoCapture(str(VIDEO)); prevg = None; f = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    f += 1; g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if f in Bmark:
        curB = Bmark[f].copy(); Bconf[f] = 1.0
    elif prevg is not None and curB is not None:
        hw, hh = 0.42*gap, 0.75*gap                                          # narrow, tall box on the bridge/glabella
        x0, y0 = max(0, int(curB[0]-hw)), max(0, int(curB[1]-hh)); x1, y1 = min(W, int(curB[0]+hw)), min(H, int(curB[1]+hh))
        if x1-x0 > 10 and y1-y0 > 10:
            mask = np.zeros((H, W), np.uint8); mask[y0:y1, x0:x1] = 255
            p0 = cv2.goodFeaturesToTrack(prevg, maxCorners=100, qualityLevel=0.01, minDistance=4, mask=mask)
            if p0 is not None and len(p0) >= 8:
                p1, stt, _ = cv2.calcOpticalFlowPyrLK(prevg, g, p0, None, **LK)
                good = stt.reshape(-1) == 1
                if int(good.sum()) >= 6:
                    tr = np.median(p1.reshape(-1, 2)[good] - p0.reshape(-1, 2)[good], 0)
                    if np.hypot(*tr) <= 0.30*gap: curB = curB + tr; Bconf[f] = float(good.sum())/len(p0)
    if curB is not None: Btr[f] = curB
    prevg = g
cap.release()
v = np.where(~np.isnan(Btr[:, 0]))[0]
if len(v):
    Btr[:v[0]] = Btr[v[0]]; last = None
    for i in range(NF):
        if np.isnan(Btr[i, 0]): Btr[i] = last if last is not None else Btr[i]
        else: last = Btr[i]
Btr = np.column_stack([smoothc(Btr[:, 0], 5), smoothc(Btr[:, 1], 5)])

# each eye's midpoint = tracked bridge + fixed offset (rotated by the small head angle)
for ek in orbit:
    dth = orbit[ek]["ang"] - theta0[ek]; off = offset0[ek]
    c, s = np.cos(dth), np.sin(dth)
    orbit[ek]["mx"] = Btr[:, 0] + off[0]*c - off[1]*s
    orbit[ek]["my"] = Btr[:, 1] + off[0]*s + off[1]*c
    orbit[ek]["conf"] = Bconf
    print(f"[orbit] {ek}: {len(markinfo[ek]['fr'])} marks; ABSOLUTE D_canthi={orbit[ek]['D0']:.1f}px (no scale); "
          f"angle {math.degrees(orbit[ek]['ang'].min()):.1f}..{math.degrees(orbit[ek]['ang'].max()):.1f}deg; "
          f"bridge-anchored (offset {off[0]:+.0f},{off[1]:+.0f}); bridge flow-tracked {int((Bconf>0).sum())}/{NF}")
BRIDGE = Btr

def frame_axes(ek, f):
    o = orbit[ek]; M = np.array([o["mx"][f], o["my"][f]]); a = o["ang"][f]; d = o["dist"][f]; sy = o["lh"][f]
    u = np.array([math.cos(a), math.sin(a)])                          # canthus (horizontal) unit
    CL = M - (d/2)*u; CR = M + (d/2)*u                                # RIGID pair: always d apart about the midpoint
    yax = np.array([-u[1], u[0]])                                     # vertical = perpendicular
    LT = M - (sy/2)*yax; LB = M + (sy/2)*yax
    return CL, CR, LT, LB, M, u, yax, d, sy

state = {ek: dict(cx=orbit[ek]["mx"][0], cy=orbit[ek]["my"][0]) for ek in orbit}

cap = cv2.VideoCapture(str(VIDEO))
vw = cv2.VideoWriter(str(OUTDIR / "orbit_lock.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
rows = []; track = {ek: dict(ox=[], oy=[], cd=[], ang=[], gx=[], gy=[]) for ek in orbit}
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek in orbit:
        CL, CR, LT, LB, O, xax, yax, sx, sy = frame_axes(ek, fi)
        st = state[ek]
        ic = iris_centre(gray, O[0], O[1], sx*0.55, sy*0.9, W, H)       # search around the ORBIT centre
        gx = gy = float("nan")
        if ic is not None:
            v = np.array([ic[0]-O[0], ic[1]-O[1]])
            gx = float(v @ xax / (sx/2)); gy = float(v @ yax / (sy/2))  # iris-in-orbit (head/camera cancelled)
        # ---- overlay ----
        p = lambda q: (int(q[0]), int(q[1]))
        Bp = BRIDGE[fi]                                                  # nose-bridge anchor
        med = CR if np.hypot(CR[0]-Bp[0], CR[1]-Bp[1]) < np.hypot(CL[0]-Bp[0], CL[1]-Bp[1]) else CL
        cv2.line(frame, p(Bp), p(med), (255, 0, 255), 1)                # bridge -> medial canthus anchor line
        cv2.line(frame, p(CL), p(CR), (255, 255, 0), 2)                 # canthus line (cyan)
        cv2.line(frame, p(LT), p(LB), (0, 220, 0), 2)                   # lid line (green)
        cv2.ellipse(frame, p(O), (int(sx/2), int(sy/2)),
                    math.degrees(math.atan2(xax[1], xax[0])), 0, 360, (200, 200, 200), 1)   # opening
        cv2.arrowedLine(frame, p(O), p(O + xax*sx*0.5), (255, 0, 255), 2, tipLength=0.2)     # x axis
        cv2.arrowedLine(frame, p(O), p(O + yax*sy*0.5), (0, 165, 255), 2, tipLength=0.2)     # y axis
        for q, col in ((CL, (255, 255, 0)), (CR, (255, 255, 0)), (LT, (0, 220, 0)), (LB, (0, 220, 0))):
            cv2.circle(frame, p(q), 4, col, -1)
        if ic is not None:
            cv2.circle(frame, p(ic), 6, (0, 0, 255), -1)                # iris centre (red)
            cv2.putText(frame, f"{ek} gx={gx:+.2f} gy={gy:+.2f}", (p(O)[0]-60, p(LT)[1]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        t = track[ek]; t["ox"].append(O[0]); t["oy"].append(O[1]); t["cd"].append(sx)
        t["ang"].append(math.degrees(math.atan2(xax[1], xax[0]))); t["gx"].append(gx); t["gy"].append(gy)
        rows.append((fi, ek, round(O[0], 1), round(O[1], 1), round(sx, 1), round(sy, 1),
                     round(ic[0], 1) if ic else "", round(ic[1], 1) if ic else "", round(gx, 3) if ic else "", round(gy, 3) if ic else ""))
    cv2.drawMarker(frame, (int(BRIDGE[fi][0]), int(BRIDGE[fi][1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)  # nose bridge
    cv2.putText(frame, f"f{fi}  ORBIT LOCK (eye-local frame)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}  ORBIT LOCK (eye-local frame)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    vw.write(frame)
cap.release(); vw.release()

with open(OUTDIR / "orbit_lock.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["frame", "eye", "O_x", "O_y", "canthus_dist", "lid_height", "iris_x", "iris_y", "gx", "gy"])
    w.writerows(rows)

# ---- track plot: orbit centre + canthus distance (should be smooth) ----
def plot(series, colors, title, path, Hc=300, PWc=1500):
    cv = np.full((Hc, PWc, 3), 255, np.uint8); L, Rm, T, Bm = 60, 20, 30, 24
    allv = [v for s in series.values() for v in s if v == v]
    if not allv: cv2.imwrite(path, cv); return
    ylo, yhi = min(allv), max(allv); yhi = yhi if yhi > ylo else ylo+1
    X = lambda f: int(L + (PWc-L-Rm)*f/max(1, NF-1)); Y = lambda v: int(T + (Hc-T-Bm)*(1-(v-ylo)/(yhi-ylo)))
    cv2.rectangle(cv, (L, T), (PWc-Rm, Hc-Bm), (210, 210, 210), 1)
    for nm, s in series.items():
        col = colors[nm]; prev = None
        for f, v in enumerate(s):
            if v != v: prev = None; continue
            q = (X(f), Y(v))
            if prev: cv2.line(cv, prev, q, col, 1)
            prev = q
    cv2.putText(cv, title, (L, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(cv, f"[{ylo:.0f}..{yhi:.0f}]", (L, Hc-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cv2.imwrite(path, cv)

if orbit:
    COL = {"L": (200, 80, 0), "R": (0, 0, 220)}
    plot({ek: track[ek]["cd"] for ek in orbit}, COL,
         "RIGID CANTHUS FRAME: inter-canthus DISTANCE px across clip (must be ~flat; slow zoom only, no jitter)",
         str(OUTDIR / "orbit_lock_distance.png"))
    plot({ek: track[ek]["ang"] for ek in orbit}, COL,
         "RIGID CANTHUS FRAME: canthus-line ANGLE deg across clip (must be ~flat; small head rotation only)",
         str(OUTDIR / "orbit_lock_angle.png"))
    plot({"O_x": track[list(orbit)[0]]["ox"], "O_y": track[list(orbit)[0]]["oy"]},
         {"O_x": (200, 80, 0), "O_y": (0, 140, 0)},
         "ORBIT midpoint translation O_x/O_y across clip (this is the head/camera motion the frame follows)",
         str(OUTDIR / "orbit_lock_midpoint.png"))
    for ek in orbit:
        cd = np.array(track[ek]["cd"]); an = np.array(track[ek]["ang"])
        print(f"[orbit] {ek}: canthus dist {cd.min():.0f}-{cd.max():.0f}px (spread {100*(cd.max()-cd.min())/np.median(cd):.1f}%); "
              f"angle {an.min():.1f}..{an.max():.1f}deg; per-frame dist jump median {np.median(np.abs(np.diff(cd))):.2f}px")
print("video ->", OUTDIR / "orbit_lock.mp4")
print("track ->", OUTDIR / "orbit_lock_track.png")
print("csv ->", OUTDIR / "orbit_lock.csv")
