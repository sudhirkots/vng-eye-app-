"""Raw EllSeg CENTROID trace -> ellseg_centroid.csv (frame, eye, cx, cy, status).

The plain, reliable EllSeg iris/pupil-disc CENTROID per frame -- no oval fit, no arc fit, no rescue logic
that adds jitter. EllSeg's LOCATION is smooth/continuous (only its disc SHAPE flickers), so the centroid is
the cleanest simple eye-position signal. Feeds tools/nystagmus_direction.py. Run with the rit-nets venv.
"""
import sys, os, json, csv
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
if not VIDEO.exists():                                       # accept any container (.mpg/.MPG/.wmv/...)
    _c = [p for p in (REPO / "samples").iterdir() if p.is_file() and p.stem == CLIP]
    if _c: VIDEO = _c[0]
OUTBASE = REPO / "outputs" / f"{CLIP}_tracked"
GT = OUTBASE / "RIT_ground_truth"
OUTDIR = OUTBASE / "step2_fixed_circle"; OUTDIR.mkdir(parents=True, exist_ok=True)
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

def sclera_mask(bgr, gray):
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]; bright = gray >= thr; bs = S[bright]
    st = float(np.clip(cv2.threshold(bs.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0], 25, 90)) if bs.size >= 30 else 60.0
    m = (bright & (S <= st)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

def region(bgr_full, gray_full, cx, cy, hw, hh, W, H):
    """EllSeg iris centroid + SCLERA BALANCE (sclera px left vs right of the iris) + disc area, for the
    anatomical (head-motion-invariant) gaze-zone classification."""
    x0, y0 = int(max(0, cx-hw)), int(max(0, cy-hh)); x1, y1 = int(min(W, cx+hw)), int(min(H, cy+hh))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    if disc.sum() < 30: return None
    m = cv2.moments(disc); icx = m["m10"]/m["m00"]
    scl = sclera_mask(bgr_full[y0:y1, x0:x1], gc)
    cols = scl.sum(0)                                         # sclera pixels per column
    xs = np.arange(len(cols))
    nL = int(cols[xs < icx].sum()); nR = int(cols[xs >= icx].sum())   # sclera left vs right of the iris
    return dict(cx=m["m10"]/m["m00"]+x0, cy=m["m01"]/m["m00"]+y0, nL=nL, nR=nR, area=int(disc.sum()))

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7)); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok0, frame0 = cap.read()

def auto_seed(fr):
    """No clinician marks -> locate the two eyes as the darkest round iris blobs (close-up two-eye clips)."""
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); gb = cv2.GaussianBlur(g, (0, 0), 2)
    d = cv2.morphologyEx((gb < np.percentile(gb, 8)).astype(np.uint8)*255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stt, cen = cv2.connectedComponentsWithStats(d, 8)
    cand = []
    for i in range(1, n):
        a, w, h = stt[i, cv2.CC_STAT_AREA], stt[i, cv2.CC_STAT_WIDTH], stt[i, cv2.CC_STAT_HEIGHT]
        if a < (W*0.02)**2 or a > (W*0.22)**2 or not (0.4 < w/(h+1e-9) < 2.5) or cen[i][1] > H*0.72: continue
        cand.append((cen[i][0], cen[i][1], max(w, h)))
    best = None
    for i in range(len(cand)):
        for j in range(i+1, len(cand)):
            A, B = cand[i], cand[j]; dy = abs(A[1]-B[1]); dx = abs(A[0]-B[0]); sr = min(A[2], B[2])/max(A[2], B[2])
            if dy < H*0.15 and W*0.12 < dx < W*0.6 and sr > 0.4:
                sc = sr*100 - dy/5.0
                if best is None or sc > best[0]: best = (sc, A, B)
    if best: a, b = sorted([best[1], best[2]], key=lambda c: c[0])          # left-image, right-image
    else: a, b = (W*0.33, H*0.45, W*0.2), (W*0.67, H*0.45, W*0.2)           # fallback default positions
    ow, oh = W*0.22, W*0.17
    return {"R": dict(ow=ow, oh=oh, seed=(a[0], a[1])), "L": dict(ow=ow, oh=oh, seed=(b[0], b[1]))}  # R eye = image-left

def iris_win_at(gray, cx, cy):
    """Auto-size the tracking window from the dark iris blob at a clicked seed (so ONE click is enough)."""
    r = int(W*0.12); x0, y0 = max(0, int(cx-r)), max(0, int(cy-r)); x1, y1 = min(W, int(cx+r)), min(H, int(cy+r))
    p = gray[y0:y1, x0:x1]
    if p.size == 0: return W*0.13
    d = cv2.morphologyEx((p < np.percentile(p, 30)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(d, 8); lc = (cx-x0, cy-y0); best, bd = None, 1e18
    for i in range(1, n):
        dd = (cen[i][0]-lc[0])**2 + (cen[i][1]-lc[1])**2
        if dd < bd and st[i, cv2.CC_STAT_AREA] > 20: bd, best = dd, i
    if best is None: return W*0.13
    rad = 0.5*max(st[best, cv2.CC_STAT_WIDTH], st[best, cv2.CC_STAT_HEIGHT])
    return float(max(W*0.05, rad*2.5))                       # window*0.6 ~ eye half-width

# ================= SEEDING: MANUAL FIRST (required), with frame-keyed re-marks =================
# CLINICAL WORKFLOW (Dr. K): begin ONLY by manually marking the iris (one or both eyes); then run. If the app
# loses track of the iris along the clip, it STOPS and asks for a re-mark at that frame. A re-mark is just an
# extra circle for that eye tagged with its "frame". The circle is ONLY a starting window for EllSeg (centre +
# radius) -- never an iris outline/measurement, and no OpenCV iris fitting is done from it.
MSF = REPO / "manual_seeds.json"
manual = json.load(open(MSF, encoding="utf-8")) if MSF.exists() else {}
ALLOW_AUTO = os.environ.get("RIT_ALLOW_AUTOSEED") == "1"      # opt-in only (batch convenience) -- not clinical
STOP_ON_LOSS = os.environ.get("RIT_BATCH") != "1"            # RIT_BATCH=1 -> track through losses (old behaviour)
LOSS_K = int(os.environ.get("RIT_LOSS_FRAMES", 12))          # consecutive lost frames = sustained loss -> STOP
g0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

def _marks_from(v):
    """One eye's manual entry -> time-sorted marks {frame,cx,cy,ow,oh,src}. Accepts a single circle dict,
    a LIST of circle dicts (re-marks at different frames), or a legacy [x,y] point."""
    if isinstance(v, dict): raw = [v]
    elif isinstance(v, list) and v and isinstance(v[0], dict): raw = v
    else: raw = [{"cx": v[0], "cy": v[1], "frame": 0, "source": "manual_point_seed"}]
    out = []
    for m in raw:
        rr = m.get("r"); ow = max(W*0.05, rr*2.5) if rr else iris_win_at(g0, m["cx"], m["cy"])
        out.append(dict(frame=int(m.get("frame", 0)), cx=m["cx"], cy=m["cy"], ow=ow, oh=ow,
                        src=m.get("source", "manual_circle_seed")))
    return sorted(out, key=lambda m: m["frame"])

if CLIP in manual:
    ent = manual[CLIP]; eyes = {ek: dict(marks=_marks_from(ent[ek])) for ek in ("R", "L") if ek in ent}
    nmk = sum(len(e["marks"]) for e in eyes.values())
    print(f"seeds: MANUAL {list(eyes)}  ({nmk} mark(s); re-marks applied on track loss)")
elif (GT / "meta.json").exists():
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
        eyes[ek] = dict(marks=[dict(frame=0, cx=cx0, cy=cy0, ow=float(np.median(ws)), oh=float(np.median(hs)), src="clinician_orbit_marks")])
    print("seeds: clinician orbit marks")
elif ALLOW_AUTO:
    a = auto_seed(frame0)
    eyes = {ek: dict(marks=[dict(frame=0, cx=a[ek]["seed"][0], cy=a[ek]["seed"][1], ow=a[ek]["ow"], oh=a[ek]["oh"], src="auto_seed")]) for ek in a}
    print("seeds: AUTO (FALLBACK -- NOT clinician-marked; unreliable) " + ", ".join(f"{ek}={tuple(round(x) for x in a[ek]['seed'])}" for ek in a))
else:
    print(f"\n*** NO MANUAL IRIS SEED for '{CLIP}'. ***")
    print("  The clinical workflow REQUIRES marking the iris first. Open tools/rit_iris_seed_marker.html, mark the")
    print("  iris circle of one or both eyes, add the block to manual_seeds.json, then re-run.")
    print("  (Set RIT_ALLOW_AUTOSEED=1 only for batch convenience -- auto-seed is unreliable, not a clinical read.)")
    sys.exit(2)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
state = {}
for ek, e in eyes.items():
    m0 = e["marks"][0]
    state[ek] = dict(cx=m0["cx"], cy=m0["cy"], ow=m0["ow"], oh=m0["oh"], src=m0["src"],
                     lost=0, consec_bad=0, loss_start=None, midx=0)
MAXF = int(os.environ.get("RIT_MAXF", 100000))
rows = []; found = {ek: 0 for ek in eyes}; fi = -1

def write_csv():
    with open(OUTDIR / "ellseg_centroid.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["frame", "eye", "cx", "cy", "status", "scl_left", "scl_right", "disc_area", "seed_source"]); w.writerows(rows)

while True:
    ok, frame = cap.read()
    if not ok or fi+1 >= MAXF: break
    fi += 1; gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek, e in eyes.items():
        st = state[ek]; marks = e["marks"]
        while st["midx"]+1 < len(marks) and marks[st["midx"]+1]["frame"] <= fi:    # apply clinician re-mark(s)
            st["midx"] += 1; m = marks[st["midx"]]
            st.update(cx=m["cx"], cy=m["cy"], ow=m["ow"], oh=m["oh"], src=m["src"], lost=0, consec_bad=0, loss_start=None)
            print(f"  re-mark applied: {ek} eye re-seeded at frame {fi} (mark frame {m['frame']})")
        grow = 1.0 + min(st["lost"], 6)*0.25
        r = region(frame, gray, st["cx"], st["cy"], st["ow"]*0.6*grow, st["oh"]*0.6*grow, W, H)
        rad = st["ow"]/2.5
        bad = (r is None) or (np.hypot(r["cx"]-st["cx"], r["cy"]-st["cy"]) > 3.0*rad)   # lost, or jumped to wrong spot
        if bad:
            if st["consec_bad"] == 0: st["loss_start"] = fi
            st["lost"] += 1; st["consec_bad"] += 1
            rows.append((fi, ek, "", "", "needs_rescue", "", "", "", st["src"]))
            remark_ahead = any(mm["frame"] >= st["loss_start"] for mm in marks[st["midx"]+1:])
            if STOP_ON_LOSS and st["consec_bad"] >= LOSS_K and not remark_ahead:
                write_csv(); cap.release()
                imgp = OUTDIR / f"loss_frame_{st['loss_start']}.png"; cv2.imwrite(str(imgp), frame)
                print(f"\n*** TRACK LOST -- STOPPED (eye {ek}) ***")
                print(f"  iris lost from frame {st['loss_start']} (confirmed after {LOSS_K} lost frames, at {fi}).")
                print(f"  RE-MARK the {ek} iris around frame {st['loss_start']} in tools/rit_iris_seed_marker.html")
                print(f"  (navigate to that frame, mark the circle), then add it to manual_seeds.json as a re-mark:")
                print(f'     "{CLIP}": {{ "{ek}": [ <existing mark>, {{"cx":_, "cy":_, "r":_, "frame": {st["loss_start"]}, "source": "manual_circle_seed"}} ] }}')
                print(f"  Frame to mark on: {imgp}   Partial trace written ({fi+1} frames). Re-run to resume.")
                sys.exit(3)
        else:
            st.update(cx=r["cx"], cy=r["cy"], lost=0, consec_bad=0, loss_start=None); found[ek] += 1
            rows.append((fi, ek, round(r["cx"], 2), round(r["cy"], 2), "ok", r["nL"], r["nR"], r["area"], st["src"]))
cap.release()
write_csv()
print("EllSeg centroid found:", found, "of", NF)
print("csv ->", OUTDIR / "ellseg_centroid.csv")
