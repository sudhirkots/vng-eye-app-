"""RIT approach #2 — Orbit Lock + EllSeg TOGETHER (moving orbit).

The static orbit failed because the head moves ~300 px (> the eye's own height), so a fixed box loses the
eye and EllSeg segments the eyebrow. Fix: the orbit FOLLOWS the eye. EllSeg is both the iris detector and
the re-anchor — each frame we crop an orbit-sized window around the previous accepted iris centre, run
EllSeg inside it, fit the iris ellipse, and gate the result. On accept the window re-centres on the new
iris (so it tracks head motion); on reject (teleport / implausible size / brow grab) the frame is marked
`needs_rescue` and the window holds, widening to re-acquire. No fabrication from invalid tracking.

Exploration only; nothing wired into the clinical path. Run with the rit-nets venv python.
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
OUTDIR = OUTBASE / "RIT_ellseg_orbit_track"
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

# --- per-eye orbit size + a seed eye location, from the clinician-marked frames ---
meta = json.load(open(GT / "meta.json", encoding="utf-8"))
def src_pts(e):
    X0, Y0, s = e["X0"], e["Y0"], e["scale"]; return np.array([[x/s+X0, y/s+Y0] for x, y in e["orbit_crop"]], np.float32)
eyes = {}
for ek in ("L", "R"):
    entries = sorted([v for v in meta.values() if v.get("eye") == ek and "orbit_crop" in v], key=lambda v: v["frame"])
    if not entries: continue
    ws, hs, seeds = [], [], []
    for v in entries:
        (cx, cy), (a1, a2), ang = cv2.fitEllipse(src_pts(v))
        ws.append(max(a1, a2)); hs.append(min(a1, a2)); seeds.append((v["frame"], cx, cy))
    ow, oh = float(np.median(ws)), float(np.median(hs))       # eye-opening w x h (px)
    f0, sx, sy = seeds[0]
    eyes[ek] = dict(ow=ow, oh=oh, r=oh*0.35, seed=(sx, sy))    # r ~ nominal iris radius from eye height
    print(f"{ek}: eye opening ~{ow:.0f}x{oh:.0f}px, nominal iris r~{eyes[ek]['r']:.0f}, seed=({sx:.0f},{sy:.0f}) @f{f0}")

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
NF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

def detect(gray_full, cx, cy, half_w, half_h):
    """Run EllSeg on a window centred at (cx,cy); return best iris ellipse in full-frame coords or None."""
    x0, y0 = int(max(0, cx-half_w)), int(max(0, cy-half_h))
    x1, y1 = int(min(W, cx+half_w)), int(min(H, cy+half_h))
    gc = gray_full[y0:y1, x0:x1]
    if gc.size == 0 or gc.shape[0] < 12 or gc.shape[1] < 12: return None
    xin, (vtop, vnh) = preprocess(gc)
    seg = seg_forward(xin)[vtop:vtop+vnh, :]
    seg = cv2.resize(seg, (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_NEAREST)
    disc = ((seg == 1) | (seg == 2)).astype(np.uint8)
    disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        if len(c) < 5: continue
        area = cv2.contourArea(c)
        if area < 30: continue
        (ex, ey), (aw, ah), ang = cv2.fitEllipse(c)
        best_c = (area, ((ex+x0, ey+y0), (aw, ah), ang))
        if best is None or area > best[0]: best = best_c
    return best   # (area, ellipse_box) full-frame

rows = []
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0, have=False) for ek, e in eyes.items()}
vw = cv2.VideoWriter(str(OUTDIR / "tracked_iris.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rec = {"frame": fi}
    for ek, e in eyes.items():
        st = state[ek]; R = e["r"]
        # window grows while lost (re-acquire), tight while locked
        grow = 1.0 + min(st["lost"], 6) * 0.25
        det = detect(gray, st["cx"], st["cy"], e["ow"] * 0.6 * grow, e["oh"] * 0.6 * grow)
        status, cx, cy, box = "needs_rescue", np.nan, np.nan, None
        if det is not None:
            area, ebox = det
            (dcx, dcy) = ebox[0]
            nominal = math.pi * R * R
            size_ok = 0.15 * nominal <= area <= 3.0 * nominal
            jump = math.hypot(dcx - st["cx"], dcy - st["cy"])
            jump_ok = (not st["have"]) or st["lost"] > 0 or jump <= R * 2.2
            if size_ok and jump_ok:
                status, cx, cy, box = "ok", dcx, dcy, ebox
                st.update(cx=dcx, cy=dcy, lost=0, have=True)
            else:
                st["lost"] += 1
        else:
            st["lost"] += 1
        # thick iris marker: green fitted ellipse + centre when ok; orange held circle on rescue
        if status == "ok" and box is not None:
            ib = ((float(box[0][0]), float(box[0][1])), (float(box[1][0]), float(box[1][1])), float(box[2]))
            cv2.ellipse(frame, ib, (0, 255, 0), 4)
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 255), -1)
        elif st["have"]:
            cv2.circle(frame, (int(st["cx"]), int(st["cy"])), int(R), (0, 165, 255), 3)
            cv2.putText(frame, "rescue", (int(st["cx"]) - 45, int(st["cy"]) - int(R) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        rec[f"{ek}_x"], rec[f"{ek}_y"], rec[f"{ek}_status"] = cx, cy, status
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
    cv2.putText(frame, f"f{fi}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    vw.write(frame)
    rows.append(rec)
cap.release(); vw.release()

cols = ["frame"]
for ek in eyes: cols += [f"{ek}_x", f"{ek}_y", f"{ek}_status"]
with open(OUTDIR / "orbit_track.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, "") for k in cols})

summary = {}
for ek in eyes:
    okn = sum(1 for r in rows if r[f"{ek}_status"] == "ok")
    summary[ek] = {"ok": okn, "needs_rescue": len(rows) - okn, "total": len(rows)}
# gold check
gold = OUTBASE / "manual_iris_annotations.json"
if gold.exists():
    g = json.load(open(gold))
    for a in g.get("annotations", []):
        ek, fn = a["image_side_eye"], a["frame_number"]
        if ek in eyes and fn < len(rows) and rows[fn][f"{ek}_status"] == "ok":
            gx, gy = a["iris_centre"]; px, py = rows[fn][f"{ek}_x"], rows[fn][f"{ek}_y"]
            summary.setdefault("gold_check", []).append(
                {"frame": fn, "eye": ek, "pred": [round(px, 1), round(py, 1)],
                 "clinician": [gx, gy], "dist_px": round(math.hypot(px-gx, py-gy), 1)})
json.dump(summary, open(OUTDIR / "summary.json", "w"), indent=2)
print("summary:", json.dumps(summary, indent=2))

# --- review montage (same layout as before: fitted iris + centre on the tracked window) ---
cap = cv2.VideoCapture(str(VIDEO))
sample = list(range(0, NF, max(1, NF // 24)))[:24]
byfi = {r["frame"]: r for r in rows}
tiles = {ek: [] for ek in eyes}
TW = 320
for fi in sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, frame = cap.read()
    if not ok: continue
    r = byfi.get(fi, {})
    for ek, e in eyes.items():
        cx, cy = r.get(f"{ek}_x"), r.get(f"{ek}_y"); stt = r.get(f"{ek}_status", "?")
        # crop around tracked centre if known else seed
        ccx = cx if isinstance(cx, float) and not math.isnan(cx) else e["seed"][0]
        ccy = cy if isinstance(cy, float) and not math.isnan(cy) else e["seed"][1]
        hw, hh = int(e["ow"] * 0.7), int(e["oh"] * 0.7)
        x0, y0 = max(0, int(ccx-hw)), max(0, int(ccy-hh)); x1, y1 = min(W, int(ccx+hw)), min(H, int(ccy+hh))
        crop = frame[y0:y1, x0:x1].copy()
        if isinstance(cx, float) and not math.isnan(cx):
            col = (0, 255, 0) if stt == "ok" else (0, 165, 255)
            cv2.circle(crop, (int(cx-x0), int(cy-y0)), int(e["r"]), col, 2)
            cv2.circle(crop, (int(cx-x0), int(cy-y0)), 3, (0, 255, 255), -1)
        if crop.size:
            th = int(TW * crop.shape[0] / crop.shape[1]); t = cv2.resize(crop, (TW, th))
            cv2.putText(t, f"f{fi} {stt}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(t, f"f{fi} {stt}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles[ek].append(t)
cap.release()
for ek in eyes:
    ts = tiles[ek]
    if not ts: continue
    hmax = max(t.shape[0] for t in ts); ts = [cv2.copyMakeBorder(t, 0, hmax-t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in ts]
    rows2 = []
    for i in range(0, len(ts), 6):
        row = ts[i:i+6]
        while len(row) < 6: row.append(np.zeros_like(ts[0]))
        rows2.append(np.hstack(row))
    cv2.imwrite(str(OUTDIR / f"track_review_{ek}.png"), np.vstack(rows2))
print("outputs ->", OUTDIR)
