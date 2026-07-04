"""ORBIT LOCK (mark-once, rigid-track) — clinician marks landmarks on ONE reference frame; the app tracks the
whole rigid constellation through the clip by optical flow of the STABLE anchors (canthi + nose bridge + extra),
fitting ONE rigid transform per frame (rotation + translation, NO scale) and applying it to every landmark.

Reads outputs/_bridge_marking/orbit_lock_v2.json (from tools/rit_orbit_lock_marker_v2.html). Stable anchors =
medial/lateral canthi + bridge + extra points (NOT the upper/lower lid borders, which move with blinks). The
rigid transform preserves every inter-anchor distance -> D_canthi is absolutely constant by construction.
EllSeg gives the iris centre; gaze = iris position INSIDE the eye-local frame (head/camera cancelled).

Diagnostic:
  orbit_lock_tracked.mp4       overlay: canthi, opening ellipse, bridge, extra anchors, iris centre, (gx,gy)
  orbit_lock_tracked_rigid.png inter-canthus distance + canthus angle over time (must be flat/smooth)
  orbit_lock_tracked.csv       per frame/eye landmark + iris + gx,gy + tracking confidence
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
MARKS = REPO / "outputs" / "_bridge_marking" / "orbit_lock_v2.json"
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
    m = cv2.moments(disc); return np.array([m["m10"]/m["m00"]+x0, m["m01"]/m["m00"]+y0])

# ---- reference marks ----
raw = json.load(open(MARKS, encoding="utf-8"))
rec = next(iter(raw.values())); REF = int(rec.get("frame", 0))
EYES = [ek for ek in ("R", "L") if ek in rec]
land = {}                                   # land[ek] = {medial,lateral,upper,lower} ref points (2,)
for ek in EYES:
    land[ek] = {k: np.array(rec[ek][k], float) for k in ("medial", "lateral", "upper", "lower")}
bridge = np.array(rec["bridge"], float) if rec.get("bridge") else None
extra = [np.array(p, float) for p in rec.get("extra", [])]

# flat landmark list + which are stable ANCHORS (used to fit the rigid transform)
names = []; P0 = []; anchor = []
for ek in EYES:
    for k in ("medial", "lateral", "upper", "lower"):
        names.append(f"{ek}.{k}"); P0.append(land[ek][k]); anchor.append(k in ("medial", "lateral"))
if bridge is not None: names.append("bridge"); P0.append(bridge); anchor.append(True)
for i, p in enumerate(extra): names.append(f"x{i+1}"); P0.append(p); anchor.append(True)
P0 = np.array(P0, np.float32); anchor = np.array(anchor, bool)
D0 = {ek: float(np.hypot(*(land[ek]["lateral"]-land[ek]["medial"]))) for ek in EYES}
print(f"ref frame {REF}; {len(P0)} landmarks ({int(anchor.sum())} anchors); D_canthi " +
      ", ".join(f"{ek}={D0[ek]:.0f}px" for ek in EYES))

# ---- track the rigid constellation forward from the reference frame ----
def rigid_from(a, b):
    """Least-squares rigid (rotation+translation, NO scale) mapping points a->b (Umeyama without scale)."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a-ca).T @ (b-cb); U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
    return R, cb - R @ ca

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); FPS = cap.get(5) or 30.0
allP = np.tile(P0.astype(float), (NF, 1, 1))         # allP[f] = landmark positions at frame f
conf = np.zeros(NF)
cap.set(cv2.CAP_PROP_POS_FRAMES, REF); ok, ref_frame = cap.read()
prevg = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
cur = P0.copy(); allP[REF] = cur; conf[REF] = 1.0
LK = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
cap.set(cv2.CAP_PROP_POS_FRAMES, REF+1); f = REF
while True:
    ok, frame = cap.read()
    if not ok: break
    f += 1; g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a_prev = cur[anchor].reshape(-1, 1, 2).astype(np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prevg, g, a_prev, None, **LK)
    good = (st.reshape(-1) == 1)
    ap = cur[anchor]
    if int(good.sum()) >= 3:
        A, B = ap[good], p1.reshape(-1, 2)[good]
        R, t = rigid_from(A, B)
        pred = (R @ ap[good].T).T + t
        inl = np.hypot(*(pred - B).T) < 6.0                      # RANSAC-ish: reject blink/occluded anchors
        if int(inl.sum()) >= 3:
            R, t = rigid_from(ap[good][inl], B[inl]); conf[f] = float(inl.sum())/len(ap)
        else:
            R, t = np.eye(2), np.zeros(2)                        # carry forward
    else:
        R, t = np.eye(2), np.zeros(2)
    cur = (R @ cur.T).T + t                                      # apply ONE rigid transform to ALL landmarks
    allP[f] = cur; prevg = g
cap.release()
print(f"tracked {NF} frames; mean anchor-inlier confidence {conf[conf>0].mean():.2f}")

def eye_pts(f, ek):
    i = EYES.index(ek)*4; return {k: allP[f, i+j] for j, k in enumerate(("medial", "lateral", "upper", "lower"))}

# ---- render + iris-in-orbit gaze ----
cap = cv2.VideoCapture(str(VIDEO))
vw = cv2.VideoWriter(str(OUTDIR / "orbit_lock_tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
COL = {"R": (255, 210, 25), "L": (60, 150, 255)}
track = {ek: dict(cd=[], ang=[], gx=[], gy=[]) for ek in EYES}; rows = []; fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1; gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pf = lambda q: (int(q[0]), int(q[1]))
    for ek in EYES:
        e = eye_pts(fi, ek); med, lat, up, lo = e["medial"], e["lateral"], e["upper"], e["lower"]
        O = (med + lat)/2; xax = lat - med; d = float(np.hypot(*xax)); xax = xax/(d+1e-9)
        yax = np.array([-xax[1], xax[0]]); sy = abs(float((lo - up) @ yax)) + 1e-6
        ic = iris_centre(gray, O[0], O[1], d*0.6, sy*0.9, W, H)
        gx = gy = float("nan")
        if ic is not None:
            v = ic - O; gx = float(v @ xax/(d/2)); gy = float(v @ yax/(sy/2))
        cv2.line(frame, pf(med), pf(lat), COL[ek], 2)
        cv2.ellipse(frame, pf(O), (int(d/2), int(sy/2)), math.degrees(math.atan2(xax[1], xax[0])), 0, 360, COL[ek], 2)
        for q in (med, lat, up, lo): cv2.circle(frame, pf(q), 5, COL[ek], -1)
        if ic is not None:
            cv2.circle(frame, pf(ic), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"{ek} gx={gx:+.2f} gy={gy:+.2f}", (pf(O)[0]-60, pf(up)[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        track[ek]["cd"].append(d); track[ek]["ang"].append(math.degrees(math.atan2(xax[1], xax[0])))
        track[ek]["gx"].append(gx); track[ek]["gy"].append(gy)
        rows.append((fi, ek, round(O[0], 1), round(O[1], 1), round(d, 1), round(sy, 1),
                     round(ic[0], 1) if ic is not None else "", round(ic[1], 1) if ic is not None else "",
                     round(gx, 3) if ic is not None else "", round(gy, 3) if ic is not None else "", round(conf[fi], 2)))
    # bridge + extra anchors
    bi = len(EYES)*4
    for j in range(bi, len(names)):
        q = allP[fi, j]; cv2.drawMarker(frame, pf(q), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    cv2.putText(frame, f"f{fi}  ORBIT LOCK tracked (mark-once, rigid)  conf={conf[fi]:.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}  ORBIT LOCK tracked (mark-once, rigid)  conf={conf[fi]:.2f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    vw.write(frame)
cap.release(); vw.release()

with open(OUTDIR / "orbit_lock_tracked.csv", "w", newline="", encoding="utf-8") as fp:
    w = csv.writer(fp); w.writerow(["frame", "eye", "O_x", "O_y", "canthus_dist", "lid_height", "iris_x", "iris_y", "gx", "gy", "conf"])
    w.writerows(rows)

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
    cv2.putText(cv, f"[{ylo:.1f}..{yhi:.1f}]", (L, Hc-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cv2.imwrite(path, cv)

CC = {"L": (200, 80, 0), "R": (0, 0, 220)}
plot({ek: track[ek]["cd"] for ek in EYES}, CC,
     "ORBIT LOCK tracked: inter-canthus DISTANCE px (rigid -> should be ~flat)", str(OUTDIR / "orbit_lock_tracked_distance.png"))
plot({ek: track[ek]["ang"] for ek in EYES}, CC,
     "ORBIT LOCK tracked: canthus ANGLE deg (should be smooth)", str(OUTDIR / "orbit_lock_tracked_angle.png"))
for ek in EYES:
    cd = np.array(track[ek]["cd"])
    print(f"[{ek}] canthus dist {cd.min():.0f}-{cd.max():.0f}px (spread {100*(cd.max()-cd.min())/np.median(cd):.1f}%)")
print("video ->", OUTDIR / "orbit_lock_tracked.mp4")
print("plots ->", OUTDIR / "orbit_lock_tracked_distance.png", "&", OUTDIR / "orbit_lock_tracked_angle.png")
print("csv ->", OUTDIR / "orbit_lock_tracked.csv")
