"""RIT approach #2 — HYBRID: EllSeg localizes, rule-based shape+darkness gives the STABLE centre.

Rationale (Dr. K): EllSeg is excellent at finding WHERE the iris is (robust through head motion) but its
exact centre JITTERS when the iris is half-covered / foreshortened, because the fit uses the whole visible
dark blob (limbus edge AND lid-cut edge). Fix: use EllSeg only for localization + radius; then read
iris/sclera by rule (dark vs bright), isolate the LIMBUS ARC (the iris edge that abuts the white sclera,
NOT the lid-cut edge), and locate the centre by voting one iris-radius inward from that arc — the limbus is
a rigid circle, so its centre does not move as the lid covers more of the iris. (Dr. K's own method:
docs/RIT_IRIS_METHOD.md + tools/rit_find_limbus.py, now anchored by EllSeg instead of painted marks.)

Radius is held ~constant (running median of EllSeg's radius) — the same eye can't change size frame to
frame; that removes the size-flicker that shifts a blob centre. No fabrication: too few limbus votes / bad
size / teleport -> needs_rescue. Rule-based, so it transfers across clips; the 40 marks are the scorecard.
"""
import sys, os, json, math, csv
from pathlib import Path
from collections import deque
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
OUTDIR = OUTBASE / "RIT_ellseg_limbus_hybrid"
OUTDIR.mkdir(parents=True, exist_ok=True)
OPH, OPW = 240, 320
VOTE_DISTS = np.array([0.55, 0.7, 0.85, 1.0])   # fractions of iris radius to vote inward
MIN_LIMBUS = 25                                  # min limbus-arc votes to trust the geometric centre

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

def ellseg_disc(gray_full, cx, cy, hw, hh, W, H):
    """EllSeg iris+pupil disc as a full-frame mask + rough ellipse; localization only."""
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = cv2.morphologyEx(((seg == 1) | (seg == 2)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if len(c) >= 5 and cv2.contourArea(c) > 30]
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    m = np.zeros_like(disc); cv2.drawContours(m, [c], -1, 1, -1)
    full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = m
    (ex, ey), (aw, ah), ang = cv2.fitEllipse(c)
    r_eq = math.sqrt(cv2.contourArea(c) / math.pi)
    return dict(disc=full, cx=ex+x0, cy=ey+y0, r_eq=r_eq, aspect=min(aw, ah)/max(aw, ah), ang=ang,
                win=(x0, y0, x1, y1))

def limbus_centre(gray_full, disc, cx0, cy0, r, win):
    """Vote one radius inward from the iris edge that abuts BRIGHT sclera (the limbus arc)."""
    x0, y0, x1, y1 = win
    g = gray_full[y0:y1, x0:x1].astype(np.float32)
    d = disc[y0:y1, x0:x1]
    if d.sum() < 50: return None, 0
    # brightness threshold: sclera is much brighter than the dark iris interior
    iris_val = float(g[d > 0].mean())
    bright_thr = max(iris_val + 35, float(np.percentile(g, 70)))
    edge = cv2.morphologyEx(d, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.where(edge > 0)
    lcx, lcy = cx0 - x0, cy0 - y0                    # rough centre in crop coords
    acc = np.zeros(g.shape, np.float32); n_lim = 0
    for x, y in zip(xs, ys):
        nx, ny = x - lcx, y - lcy; nrm = math.hypot(nx, ny)
        if nrm < 1e-3: continue
        nx, ny = nx/nrm, ny/nrm                      # outward normal
        sx, sy = int(x + 4*nx), int(y + 4*ny)        # sample just outside the iris edge
        if not (0 <= sx < g.shape[1] and 0 <= sy < g.shape[0]): continue
        if g[sy, sx] < bright_thr: continue          # outside is not sclera -> lid edge, skip
        n_lim += 1
        for dd in VOTE_DISTS:                          # vote inward by ~one iris radius
            vx, vy = int(round(x - dd*r*nx)), int(round(y - dd*r*ny))
            if 0 <= vx < g.shape[1] and 0 <= vy < g.shape[0]: acc[vy, vx] += 1
    if n_lim < MIN_LIMBUS or acc.max() <= 0: return None, n_lim
    acc = cv2.GaussianBlur(acc, (0, 0), 5)
    py, px = np.unravel_index(int(acc.argmax()), acc.shape)
    return (px + x0, py + y0), n_lim

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

rows = []
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0, have=False, rbuf=deque(maxlen=15), abuf=deque(maxlen=9))
         for ek, e in eyes.items()}
vw = cv2.VideoWriter(str(OUTDIR / "hybrid_iris.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rec = {"frame": fi}
    for ek, e in eyes.items():
        st = state[ek]
        grow = 1.0 + min(st["lost"], 6) * 0.25
        det = ellseg_disc(gray, st["cx"], st["cy"], e["ow"]*0.6*grow, e["oh"]*0.6*grow, W, H)
        status, cx, cy, drawn = "needs_rescue", np.nan, np.nan, None
        if det is not None:
            st["rbuf"].append(det["r_eq"]); st["abuf"].append(det["aspect"])
            r = float(np.median(st["rbuf"]))                       # stable radius prior
            lc, n_lim = limbus_centre(gray, det["disc"], det["cx"], det["cy"], r, det["win"])
            src = "limbus" if lc is not None else "ellseg"
            gcx, gcy = lc if lc is not None else (det["cx"], det["cy"])
            jump = math.hypot(gcx - st["cx"], gcy - st["cy"])
            size_ok = 0.4*e["oh"]*0.35 <= r <= 1.6*e["oh"]*0.35
            jump_ok = (not st["have"]) or st["lost"] > 0 or jump <= r*2.0
            if size_ok and jump_ok:
                status, cx, cy = ("ok" if src == "limbus" else "ellseg_fallback"), gcx, gcy
                st.update(cx=gcx, cy=gcy, lost=0, have=True)
                aspect = float(np.median(st["abuf"])); major = r; minor = max(4.0, r*aspect)
                drawn = ((gcx, gcy), (2*major, 2*minor), det["ang"])
            else:
                st["lost"] += 1
        else:
            st["lost"] += 1
        # draw: green completed oval + yellow centre when limbus-locked; blue = ellseg fallback; orange rescue
        if drawn is not None:
            col = (0, 255, 0) if status == "ok" else (255, 150, 0)
            cv2.ellipse(frame, ((float(drawn[0][0]), float(drawn[0][1])),
                                (float(drawn[1][0]), float(drawn[1][1])), float(drawn[2])), col, 4)
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 255), -1)
        elif st["have"]:
            cv2.circle(frame, (int(st["cx"]), int(st["cy"])), 30, (0, 165, 255), 3)
            cv2.putText(frame, "rescue", (int(st["cx"])-45, int(st["cy"])-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        rec[f"{ek}_x"], rec[f"{ek}_y"], rec[f"{ek}_status"] = cx, cy, status
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    vw.write(frame); rows.append(rec)
cap.release(); vw.release()

cols = ["frame"]
for ek in eyes: cols += [f"{ek}_x", f"{ek}_y", f"{ek}_status"]
with open(OUTDIR / "hybrid_trace.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, "") for k in cols})

summary = {}
for ek in eyes:
    okn = sum(1 for r in rows if r[f"{ek}_status"] == "ok")
    fb = sum(1 for r in rows if r[f"{ek}_status"] == "ellseg_fallback")
    summary[ek] = {"limbus_ok": okn, "ellseg_fallback": fb, "needs_rescue": len(rows)-okn-fb, "total": len(rows)}
gold = OUTBASE / "manual_iris_annotations.json"
if gold.exists():
    g = json.load(open(gold))
    for a in g.get("annotations", []):
        ek, fn = a["image_side_eye"], a["frame_number"]
        if ek in eyes and fn < len(rows) and isinstance(rows[fn][f"{ek}_x"], float) and not math.isnan(rows[fn][f"{ek}_x"]):
            gx, gy = a["iris_centre"]; px, py = rows[fn][f"{ek}_x"], rows[fn][f"{ek}_y"]
            summary.setdefault("gold_check", []).append(
                {"frame": fn, "eye": ek, "status": rows[fn][f"{ek}_status"],
                 "pred": [round(px, 1), round(py, 1)], "clinician": [gx, gy], "dist_px": round(math.hypot(px-gx, py-gy), 1)})

# jitter metric: median frame-to-frame centre step during locked runs (lower = smoother)
for ek in eyes:
    steps = []
    for i in range(1, len(rows)):
        a, b = rows[i-1], rows[i]
        if all(isinstance(b[f"{ek}_{k}"], float) and not math.isnan(b[f"{ek}_{k}"]) for k in ("x", "y")) and \
           all(isinstance(a[f"{ek}_{k}"], float) and not math.isnan(a[f"{ek}_{k}"]) for k in ("x", "y")):
            steps.append(math.hypot(b[f"{ek}_x"]-a[f"{ek}_x"], b[f"{ek}_y"]-a[f"{ek}_y"]))
    if steps:
        summary[ek]["median_step_px"] = round(float(np.median(steps)), 2)
        summary[ek]["p90_step_px"] = round(float(np.percentile(steps, 90)), 2)
json.dump(summary, open(OUTDIR / "summary.json", "w"), indent=2)
print("summary:", json.dumps(summary, indent=2))
print("outputs ->", OUTDIR)
