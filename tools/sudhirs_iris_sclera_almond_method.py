"""Sudhir's iris-sclera-almond method — mark the VISIBLE iris by Dr. K's rules
(docs/RIT_IRIS_METHOD.md, 2026-07-03).

EllSeg gives only the approximate location. Then, inside that eye region:
  1. SCLERA = the bright white / pink-reddish patch (one Otsu split of the eye interior separates the
     bright sclera from the dark iris).
  2. IRIS = the uniformly-dark region that ABUTS the sclera along a smooth CONVEX arc (limbus). Lashes are
     thin & ragged -> stripped by a morphological open and by requiring solidity + sclera-contact.
  3. iris = almond - sclera: within the opening the non-sclera dark blob adjacent to the white/pink IS the
     iris. A dark blob NOT touching sclera (lid shadow, brow) is rejected.
  4. Empty almond (no dark blob abutting sclera) -> mark NOTHING.
Dr. K's two rules: (1) the iris shape is ALWAYS a circle/oval; (2) the iris is never bright. So COMPLETE
the oval from the limbus arc and output the oval itself (fit to its boundary with the bright sclera) —
never clip it to dark pixels. Paint it red. Run with the rit-nets venv python.
"""
import sys, os, json, math
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
OUTDIR = OUTBASE / "sudhirs_iris_sclera_almond_method"
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

def ellseg_region(gray_full, cx, cy, hw, hh, W, H):
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    if disc.sum() < 30: return None
    return dict(win=(x0, y0, x1, y1), disc=disc)

def mark_iris(bgr, gray, disc):
    """Rule-based iris mask within an eye crop (Dr. K's two rules). Returns mask (uint8) or None.

    RULE 1 (shape): the iris is ALWAYS a clean circle or oval — never a jagged blob.
    RULE 2 (never bright): the iris can never be bright, so the oval is bounded by the LIMBUS (its
    boundary with the bright sclera). These together mean: complete the oval from the limbus arc and
    RETURN THE OVAL — do NOT clip it to dark pixels (that broke both rules: jagged shape + dropped the
    mid-tone iris periphery, which is not bright).
    """
    Hc, Wc = gray.shape
    ctx = np.full_like(gray, 255)                                  # ALMOND-CONTEXT RULE: analyse the whole eye-opening crop, not the iris disc
    vals = gray[ctx > 0]
    if vals.size < 50: return None, None
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # bright/dark split
    B, G, R = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    # SCLERA = bright whitish OR pink/reddish, inside the eye-interior context
    whitish = gray >= thr
    pinkish = (R - G > 12) & (R > 110) & (gray > 0.6 * thr)
    sclera = ((whitish | pinkish) & (ctx > 0)).astype(np.uint8)
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    if sclera.sum() < 30: return None, None
    # DARK candidates (iris interior), inside context, lashes stripped by open
    dark = ((gray < thr) & (ctx > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))     # strip thin lashes
    sclera_d = cv2.dilate(sclera, np.ones((5, 5), np.uint8))
    def _visible(oval):
        # CONTINUITY RULE: once a valid iris arc is found, continue the smooth circle/oval THROUGH any
        # interruption and keep it — a bright patch inside the completed oval is OVERRIDDEN (treated as iris,
        # not sclera). Bright is only sclera OUTSIDE the oval. So do NOT subtract sclera inside the oval: the
        # iris = the completed oval clipped to the eye OPENING. (Eyelid & inner-canthus left for a later step.)
        opening = cv2.morphologyEx((sclera | dark).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        vis = (oval & opening).astype(np.uint8)
        cs, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: return None
        m = np.zeros_like(vis); cv2.drawContours(m, [max(cs, key=cv2.contourArea)], -1, 1, -1)
        return m if m.sum() > 0 else None
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    disc_area = max(1, int(disc.sum()))
    best, best_score = None, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.03 * disc_area: continue
        comp = (lab == i).astype(np.uint8)
        edge = cv2.morphologyEx(comp, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        contact = int(((edge > 0) & (sclera_d > 0)).sum())        # boundary abutting sclera (the limbus)
        if contact < 8: continue                                   # must abut white/pink sclera
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = (area / hull) if hull > 0 else 0                # convex/round -> high; ragged -> low
        score = contact * (0.4 + solidity)                         # convex arc abutting sclera wins
        if score > best_score: best_score, best = score, comp
    if best is None: return None, None                             # empty almond / no iris abutting sclera
    # OVERRIDING RULE: complete the OVAL from the clear LIMBUS arc (dark<->sclera boundary), not from the dark
    # blob (which leaks toward canthus/lid/lash). Long axis = perpendicular to gaze = full iris width; short
    # axis = along gaze = foreshortened. Then the iris = that completed oval; dark only lives INSIDE it.
    r0 = math.sqrt(disc_area / math.pi)                            # known iris radius (constant)
    edge = cv2.morphologyEx(best, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    limbus = (edge > 0) & (sclera_d > 0)                          # the iris edge that abuts sclera = the limbus
    ys, xs = np.where(limbus)
    if len(xs) >= 12:
        pts = np.column_stack([xs, ys]).astype(np.float64)
        # FIXED-RADIUS fit (kills the centre jitter): the iris radius r0 is KNOWN, so fit only the CENTRE
        # (2 DOF), not centre+radius (3 DOF). Fitting a circle of *free* radius to a short/partial limbus
        # arc is ill-conditioned -- centre and radius trade off, so tiny per-frame edge changes swing the
        # centre. With the radius locked, the centre is well-determined even from a short arc. Fixed-point
        # solve of the known-radius circle: each limbus point implies a centre exactly r0 inward; average
        # the inliers (points ~r0 from the current centre -> rejects lash/canthus edge), repeat to converge.
        Mb = cv2.moments(best)
        c = np.array([Mb["m10"]/Mb["m00"], Mb["m01"]/Mb["m00"]]) if Mb["m00"] else np.array([xs.mean(), ys.mean()])
        n_inl = 0
        for _ in range(6):
            dirs = c - pts; norms = np.hypot(dirs[:, 0], dirs[:, 1]) + 1e-9
            votes = pts + r0 * (dirs / norms[:, None])            # each limbus point's implied centre at radius r0
            inl = np.abs(norms - r0) < max(3.0, 0.18*r0)          # keep points that lie on the r0 circle (the true arc)
            n_inl = int(inl.sum())
            c = votes[inl].mean(0) if n_inl >= 8 else votes.mean(0)
        if n_inl >= max(10, int(0.35*len(pts))):
            cx, cy = float(c[0]), float(c[1])
            alm = ((sclera > 0) | (best > 0)).astype(np.uint8); Mm = cv2.moments(alm)
            acx, acy, alm_r = (Mm["m10"]/Mm["m00"], Mm["m01"]/Mm["m00"], math.sqrt(Mm["m00"]/math.pi)) if Mm["m00"] > 0 else (cx, cy, r0)
            gx, gy = cx - acx, cy - acy; disp = math.hypot(gx, gy)     # gaze = opening centre -> iris
            frac = min(0.9, disp / max(alm_r, 1.0))
            major = 2 * r0                                             # RADIUS LOCKED to the known iris radius
            minor = max(0.30*major, major*math.sqrt(max(0.05, 1 - frac*frac)))     # foreshorten along gaze
            ang = math.degrees(math.atan2(gy, gx)) + 90 if disp > 2 else 0.0       # long axis perpendicular to gaze
            oval = np.zeros_like(best)
            cv2.ellipse(oval, (int(round(cx)), int(round(cy))), (int(major/2), int(minor/2)), ang, 0, 360, 1, -1)
            vis = _visible(oval)                                  # keep only what is SEEN (cut at the lid line)
            return (vis if vis is not None else oval), ((cx, cy), (major, minor), ang)
    M = cv2.moments(best); cx, cy = (M["m10"]/M["m00"], M["m01"]/M["m00"]) if M["m00"] else (Wc/2, Hc/2)
    oval = np.zeros_like(best); cv2.circle(oval, (int(cx), int(cy)), int(r0), 1, -1)   # fallback: circle of known radius
    vis = _visible(oval)
    return (vis if vis is not None else oval), ((cx, cy), (2*r0, 2*r0), 0.0)

meta = json.load(open(GT / "meta.json", encoding="utf-8"))
def src_pts(e):
    X0, Y0, s = e["X0"], e["Y0"], e["scale"]; return np.array([[x/s+X0, y/s+Y0] for x, y in e["orbit_crop"]], np.float32)
eyes = {}
for ek in ("L", "R"):
    ents = sorted([v for v in meta.values() if v.get("eye") == ek and "orbit_crop" in v], key=lambda v: v["frame"])
    if not ents: continue
    ws, hs = [], []
    for v in ents:
        (cx, cy), (a1, a2), ang = cv2.fitEllipse(src_pts(v)); ws.append(max(a1, a2)); hs.append(min(a1, a2))
    (cx0, cy0), _, _ = cv2.fitEllipse(src_pts(ents[0]))
    eyes[ek] = dict(ow=float(np.median(ws)), oh=float(np.median(hs)), seed=(cx0, cy0))
    print(f"{ek}: opening ~{eyes[ek]['ow']:.0f}x{eyes[ek]['oh']:.0f}px seed=({cx0:.0f},{cy0:.0f})")

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); FPS = cap.get(5) or 30.0
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
vw = cv2.VideoWriter(str(OUTDIR / "sudhirs_iris_sclera_almond_method.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
painted = {ek: 0 for ek in eyes}
trace = []                                                    # (frame, eye, cx, cy) iris-centre trace
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek, e in eyes.items():
        st = state[ek]
        grow = 1.0 + min(st["lost"], 6) * 0.25
        reg = ellseg_region(gray, st["cx"], st["cy"], e["ow"]*0.6*grow, e["oh"]*0.6*grow, W, H)
        marked = False
        if reg is not None:
            x0, y0, x1, y1 = reg["win"]
            iris, oval = mark_iris(frame[y0:y1, x0:x1], gray[y0:y1, x0:x1], reg["disc"])
            if iris is not None and oval is not None:
                (ocx, ocy), _, _ = oval
                full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = iris   # only the SEEN iris
                m = full > 0
                frame[m] = (0.5 * frame[m] + np.array([0, 0, 255]) * 0.5).astype(np.uint8)   # marked red
                cv2.drawContours(frame, cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                                 -1, (0, 0, 200), 2)
                fcx, fcy = ocx + x0, ocy + y0                        # centre used to follow the eye (not drawn)
                st.update(cx=fcx, cy=fcy, lost=0); painted[ek] += 1; marked = True
                trace.append((fi, ek, round(fcx, 2), round(fcy, 2)))
        if not marked: st["lost"] += 1
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    vw.write(frame)
cap.release(); vw.release()
import csv as _csv
with open(OUTDIR / "points.csv", "w", newline="", encoding="utf-8") as _fp:
    _w = _csv.writer(_fp); _w.writerow(["frame", "eye", "cx", "cy"]); _w.writerows(trace)
print("painted frames:", painted, "of", NF)
print("video ->", OUTDIR / "sudhirs_iris_sclera_almond_method.mp4")
print("points ->", OUTDIR / "points.csv")
