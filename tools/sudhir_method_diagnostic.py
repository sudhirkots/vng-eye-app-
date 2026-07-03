"""DIAGNOSTIC ONLY (no algorithm change): for frames 330-341, R eye, show four panels per frame —
original / Otsu bright-dark split (Step 2) / sclera mask (Step 3) / dark candidates after lash strip (Step 4).
Masks are computed exactly as tools/sudhirs_iris_sclera_almond_method.py computes them.
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
CLIP = "nystagmus at rest to left in right vestibular neuritis"
VIDEO = REPO / "samples" / f"{CLIP}.mp4"
OUTBASE = REPO / "outputs" / f"{CLIP}_tracked"
GT = OUTBASE / "RIT_ground_truth"
OUTDIR = OUTBASE / "sudhirs_iris_sclera_almond_method"
OPH, OPW = 240, 320
SELECTED = list(range(330, 342))
EYE = "R"

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
    m = cv2.moments(disc); return dict(win=(x0, y0, x1, y1), disc=disc, cx=m["m10"]/m["m00"]+x0, cy=m["m01"]/m["m00"]+y0)

def layers(bgr, gray, disc):
    ctx = np.full_like(gray, 255)   # ALMOND-CONTEXT RULE: whole eye-opening crop, not the iris disc
    vals = gray[ctx > 0]
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    B, G, R = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    whitish = gray >= thr
    pinkish = (R - G > 12) & (R > 110) & (gray > 0.6 * thr)
    sclera = ((whitish | pinkish) & (ctx > 0)).astype(np.uint8)
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    dark = ((gray < thr) & (ctx > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return thr, ctx, sclera, dark

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

cap = cv2.VideoCapture(str(VIDEO))
W = int(cap.get(3)); H = int(cap.get(4)); NF = int(cap.get(7))
state = {ek: dict(cx=e["seed"][0], cy=e["seed"][1], lost=0) for ek, e in eyes.items()}
sel = set(SELECTED); rows = []; PW = 300
hdr = None
fi = -1
while True:
    ok, frame = cap.read()
    if not ok: break
    fi += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ek, e in eyes.items():
        st = state[ek]; grow = 1.0 + min(st["lost"], 6) * 0.25
        reg = ellseg_region(gray, st["cx"], st["cy"], e["ow"]*0.6*grow, e["oh"]*0.6*grow, W, H)
        if reg is None: st["lost"] += 1; continue
        st.update(cx=reg["cx"], cy=reg["cy"], lost=0)
        if ek == EYE and fi in sel:
            x0, y0, x1, y1 = reg["win"]; crop = frame[y0:y1, x0:x1].copy(); g = gray[y0:y1, x0:x1]
            thr, ctx, sclera, dark = layers(crop, g, reg["disc"])
            # panel 2: Otsu split (within ctx: bright=white, dark=dark-gray; outside=black)
            split = np.zeros_like(crop)
            split[(ctx > 0) & (g >= thr)] = (235, 235, 235); split[(ctx > 0) & (g < thr)] = (70, 70, 70)
            # panel 3: sclera overlay (blue); panel 4: dark overlay (red)
            p3 = crop.copy(); p3[sclera > 0] = (0.35*p3[sclera > 0] + np.array([255, 60, 0])*0.65).astype(np.uint8)
            p4 = crop.copy(); p4[dark > 0] = (0.35*p4[dark > 0] + np.array([0, 0, 255])*0.65).astype(np.uint8)
            panels = [crop, split, p3, p4]
            def fit(im): h = int(PW*im.shape[0]/im.shape[1]); return cv2.resize(im, (PW, h))
            panels = [fit(p) for p in panels]
            hmax = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, hmax-p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
            row = np.hstack([cv2.copyMakeBorder(p, 0, 0, 0, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels])
            cv2.putText(row, f"f{fi}  (thr={thr:.0f})", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(row, f"f{fi}  (thr={thr:.0f})", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            rows.append(row)
cap.release()
# header strip
hw = rows[0].shape[1]; head = np.full((30, hw, 3), 20, np.uint8)
for i, lab in enumerate(["ORIGINAL", "OTSU split (Step2)", "SCLERA mask (Step3)", "DARK cand. (Step4)"]):
    cv2.putText(head, lab, (i*(PW+3)+8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
half = len(rows)//2
for name, rr in (("diag_330_335.png", [head]+rows[:half]), ("diag_336_341.png", [head]+rows[half:])):
    w = max(r.shape[1] for r in rr); rr = [cv2.copyMakeBorder(r, 0, 0, 0, w-r.shape[1], cv2.BORDER_CONSTANT) for r in rr]
    cv2.imwrite(str(OUTDIR / name), np.vstack(rr))
print("wrote", OUTDIR / "diag_330_335.png", "and diag_336_341.png")
