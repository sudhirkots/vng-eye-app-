"""[DEPRECATED / NEGATIVE RESULT -- NOT AN ACTIVE PATH.  Kept on purpose so the idea is not re-tried blindly.]

    Head-motion-compensated optical flow was tested (2026-07-05) as a motion-based front end and DISCARDED.
    The clinical principle was correct -- observed eye-region motion MINUS common face/head/camera motion =
    eye-relative motion -- but the implementation failed on this video material:
      - raw flow false-positived on the normal clip,
      - compensated flow did not separate normal from positive,
      - compensation FLIPPED the positive vestibular clip to the WRONG direction,
      - the pontine/uncertain clip did not improve.
    Conclusion: optical flow is too low-SNR for this material as the main front end. Active path stays
    EllSeg centroid -> signal-quality gate -> qualitative nystagmus detector; the real lever is good seeding /
    good centroid localization. DO NOT integrate, tune, or return to optical-flow motion detection unless the
    clinician (Dr. K) explicitly asks. See EYE_VNG_DEVELOPMENT_HISTORY.md sec. 27.

FEASIBILITY PROBE ONLY -- head/camera-motion-COMPENSATED optical flow for nystagmus direction.
NOT part of the app. Does NOT touch ellseg_centroid_trace.py or nystagmus_direction.py. Nothing committed.

Correction (Dr. K): raw screen-coordinate flow is wrong -- the clinician automatically subtracts head/camera
motion (anything common to the whole face) and reads only the EYE-RELATIVE residual. So this probe:
  1. tracks stable peri-orbital/FACE features OUTSIDE the eye boxes  -> global (head/camera) motion,
  2. fits a RANSAC affine to that = the common motion model,
  3. computes dense flow inside the eye box, SUBTRACTS the model's predicted motion  -> residual,
  4. reads slow-drift + fast-correction from the residual (eye-relative) horizontal velocity only.
Key test: does compensation recover the visible nystagmus direction better than the centroid trace, and
does it stop head/camera motion from faking nystagmus (the normal clip)?

Run with the rit-nets venv (cv2). Eye boxes come from manual_seeds.json. Overlay -> outputs/_flow_probe/.
"""
import os, json
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "_flow_probe"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = json.load(open(REPO / "manual_seeds.json", encoding="utf-8"))
MAXF = int(os.environ.get("FLOW_MAXF", 200))
COMPENSATE = os.environ.get("FLOW_COMPENSATE", "1") == "1"    # 1 = subtract head/camera motion (the correction)

CLIPS = [
    ("nystagmus at rest to left in right vestibular neuritis", 30, "POSITIVE  | centroid: nystagmus_likely LEFT 0.63 | raw-flow probe: LEFT 0.56"),
    ("3", 30, "NORMAL    | centroid: no_nystagmus 0.07 | raw-flow probe: LEFT 0.60 (false pos)"),
    ("gaze-evoked nystagmus in pontine glioma", 25, "UNCERTAIN | centroid: uncertain_tracking 0.12 | raw-flow probe: 0.01"),
]

def find_video(stem):
    p = REPO / "samples" / f"{stem}.mp4"
    if p.exists(): return p
    c = [q for q in (REPO / "samples").iterdir() if q.is_file() and q.stem == stem]
    return c[0] if c else p

def gsmooth(x, s=1.0):
    r = max(1, int(3*s)); k = np.exp(-0.5*(np.arange(-r, r+1)/s)**2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")[:len(x)]

def interp_nan(x):
    x = np.asarray(x, float); idx = np.arange(len(x)); g = ~np.isnan(x)
    if g.sum() < 2: return np.nan_to_num(x)
    x = x.copy(); x[~g] = np.interp(idx[~g], idx[g], x[g]); return x

def asymmetry(v):
    floor = max(0.03, 0.5*np.median(np.abs(v)))
    posm, negm = v > floor, v < -floor; npos, nneg = int(posm.sum()), int(negm.sum())
    if npos+nneg < 8: return None
    sp_pos = float(v[posm].mean()) if npos else 0.0; sp_neg = float(-v[negm].mean()) if nneg else 0.0
    time_asym = (nneg-npos)/(npos+nneg); speed_asym = (sp_pos-sp_neg)/(sp_pos+sp_neg+1e-9)
    sk = ((v-v.mean())**3).mean()/((v.std()+1e-9)**3)
    votes = [np.sign(time_asym), np.sign(speed_asym), np.sign(sk)]
    return dict(fast=int(np.sign(sum(votes))) or 1, agree=abs(sum(votes))/3.0,
                strength=(abs(time_asym)+abs(speed_asym)+min(1.0, abs(sk)/2))/3.0)

def windowed(v, fps):
    A = asymmetry(v)
    if A is None: return dict(fast=0, conf=0.0, cons=0.0)
    win, step = int(2*fps), int(fps)
    signs = [w["fast"] for a in range(0, max(1, len(v)-win), step) if (w := asymmetry(v[a:a+win]))]
    signs = np.array(signs) if signs else np.array([A["fast"]])
    cons = float((signs == A["fast"]).mean())
    return dict(fast=A["fast"], conf=A["agree"]*min(1.0, A["strength"]*2.2)*cons, cons=cons)

def jumps(v, fps):
    slow = gsmooth(v, 0.8*fps); resid = v - slow
    mad = 1.4826*np.median(np.abs(resid - np.median(resid))) + 1e-9
    refr = int(0.2*fps); out = []
    for i in np.where(np.abs(resid) > 4.0*mad)[0]:
        if slow[i] != 0 and np.sign(resid[i]) == -np.sign(slow[i]):
            if not out or i - out[-1][0] >= refr: out.append((int(i), int(np.sign(resid[i]))))
    if not out: return 0, 0
    s = [x[1] for x in out]; run = best = 1
    for k in range(1, len(s)):
        run = run+1 if s[k] == s[k-1] else 1; best = max(best, run)
    return len(out), best

def name_dir(f): return "none" if not f else ("LEFT-beating" if f > 0 else "RIGHT-beating")

def global_affine(prev_g, cur_g, boxes):
    """Head/camera motion from FACE features OUTSIDE the eye boxes -> RANSAC partial-affine."""
    p0 = cv2.goodFeaturesToTrack(prev_g, 300, 0.01, 7)
    if p0 is None: return None, 0.0
    keep = [pt for pt in p0 if not any(x0 <= pt[0][0] <= x1 and y0 <= pt[0][1] <= y1
                                       for (x0, y0, x1, y1) in boxes.values())]
    if len(keep) < 8: return None, 0.0
    p0 = np.array(keep, np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None)
    g0, g1 = p0[st.ravel() == 1], p1[st.ravel() == 1]
    if len(g0) < 8: return None, 0.0
    M, _ = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC)
    mag = float(np.median(np.hypot(*(g1 - g0).reshape(-1, 2).T))) if len(g0) else 0.0   # median face-feature px shift
    return M, mag

def eye_residual_vx(prev_crop_g, cur_crop_g, x0, y0, M):
    """Dense flow in the eye crop MINUS the global model's predicted motion = eye-relative horizontal velocity."""
    fl = cv2.calcOpticalFlowFarneback(prev_crop_g, cur_crop_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = cur_crop_g.shape; fx, fy = fl[..., 0].copy(), fl[..., 1].copy()
    if M is not None and COMPENSATE:                          # subtract predicted global flow per pixel
        ys, xs = np.mgrid[0:h, 0:w]; X = xs + x0; Y = ys + y0
        gfx = M[0, 0]*X + M[0, 1]*Y + M[0, 2] - X
        gfy = M[1, 0]*X + M[1, 1]*Y + M[1, 2] - Y
        fx -= gfx; fy -= gfy
    gx = cv2.Sobel(cur_crop_g, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(cur_crop_g, cv2.CV_32F, 0, 1, 3)
    grad = cv2.magnitude(gx, gy)
    my0, my1, mx0, mx1 = int(0.2*h), int(0.8*h), int(0.2*w), int(0.8*w)
    cen = np.zeros((h, w), bool); cen[my0:my1, mx0:mx1] = True
    cm = cen & (grad > np.percentile(grad[cen], 60))
    blink = np.median(np.abs(fy)) > 0.35*w
    vx = float(np.median(fx[cm])) if cm.sum() > 10 else float(np.median(fx[cen]))
    return (np.nan if blink else vx), fl, fx, fy

def draw_quiver(crop, fx, fy, step=12):
    img = crop.copy(); h, w = crop.shape[:2]
    for y in range(step, h-step, step):
        for x in range(step, w-step, step):
            cv2.arrowedLine(img, (x, y), (int(x+fx[y, x]*3), int(y+fy[y, x]*3)), (0, 255, 0), 1, tipLength=0.35)
    return img

def trace_strip(vals, W, H=120):
    s = np.full((H, W, 3), 30, np.uint8); cv2.line(s, (0, H//2), (W, H//2), (90, 90, 90), 1)
    if len(vals) > 1:
        a = np.array(vals[-W:]); m = max(1e-6, np.nanmax(np.abs(a)))
        pts = [(i, int(H//2 - (0 if np.isnan(v) else v)/m*(H//2-8))) for i, v in enumerate(a)]
        for i in range(1, len(pts)): cv2.line(s, pts[i-1], pts[i], (0, 200, 255), 1)
    cv2.putText(s, "eye-RELATIVE horizontal velocity (residual after head-motion removal)", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return s

def run(stem, fps, note):
    vid = find_video(stem); cap = cv2.VideoCapture(str(vid)); W = int(cap.get(3)); H = int(cap.get(4))
    ent = SEEDS.get(stem, {}); eyes = {ek: ent[ek] for ek in ("R", "L") if ek in ent}
    hw, hh = int(0.09*W), int(0.075*W)
    boxes = {ek: (int(max(0, sx-hw)), int(max(0, sy-hh)), int(min(W, sx+hw)), int(min(H, sy+hh)))
             for ek, (sx, sy) in eyes.items()}
    sig = {ek: [] for ek in eyes}; combined = []; gmags = []
    prev_full = None; prev_crop = {ek: None for ek in eyes}; vw = None; fi = -1
    while True:
        ok, fr = cap.read()
        if not ok or fi+1 >= MAXF: break
        fi += 1; full_g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        M, gmag = (None, 0.0) if prev_full is None else global_affine(prev_full, full_g, boxes)
        gmags.append(gmag); panels = []
        for ek in ("R", "L"):
            if ek not in boxes: continue
            x0, y0, x1, y1 = boxes[ek]; crop = fr[y0:y1, x0:x1]; g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if prev_crop[ek] is None or crop.size == 0:
                sig[ek].append(np.nan); prev_crop[ek] = g
                panels.append(cv2.resize(crop if crop.size else np.zeros((10, 10, 3), np.uint8), (300, 240))); continue
            vx, fl, fx, fy = eye_residual_vx(prev_crop[ek], g, x0, y0, M); sig[ek].append(vx); prev_crop[ek] = g
            q = cv2.resize(draw_quiver(crop, fx, fy), (300, 240))
            cv2.putText(q, ek, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            panels.append(q)
        vals = [sig[ek][-1] for ek in eyes if not np.isnan(sig[ek][-1])]
        combined.append(np.mean(vals) if vals else np.nan)
        top = np.hstack(panels) if panels else np.zeros((240, 300, 3), np.uint8)
        strip = trace_strip(combined, top.shape[1])
        cv2.putText(strip, f"head motion: {gmag:4.1f}px  comp={'ON' if COMPENSATE else 'OFF'}",
                    (top.shape[1]-260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 120), 1)
        out = np.vstack([top, strip])
        if vw is None:
            vw = cv2.VideoWriter(str(OUT / f"{stem}_flowcomp.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (out.shape[1], out.shape[0]))
        vw.write(out)
        prev_full = full_g
    cap.release()
    if vw: vw.release()
    per = {ek: windowed(gsmooth(interp_nan(sig[ek]), 1.0), fps) for ek in eyes}
    comb = gsmooth(interp_nan(combined), 1.0); R = windowed(comb, fps); nj, run_len = jumps(comb, fps)
    craw = comb - comb.mean(); ac = np.correlate(craw, craw, "full")[len(craw)-1:]; ac = ac/(ac[0]+1e-9)
    lo, hi = max(1, int(fps/6)), int(fps/1)
    pl = lo + int(np.argmax(ac[lo:hi])) if hi > lo else 0
    print(f"\n==== {stem}  ({fps} fps) ====\n  [{note}]")
    print(f"  eyes: {list(eyes)}   median head motion: {np.median(gmags):.1f}px   overlay: outputs/_flow_probe/{stem}_flowcomp.mp4")
    for ek in eyes:
        print(f"    {ek} eye residual: {name_dir(per[ek]['fast']):13s} conf {per[ek]['conf']:.2f} (cons {per[ek]['cons']:.2f})")
    print(f"  COMPENSATED FLOW: {name_dir(R['fast'])}  conf {R['conf']:.2f} (cons {R['cons']:.2f})")
    print(f"  drift-and-snap: {nj} jump(s), longest same-dir run {run_len}   rhythmicity: {ac[pl]:.2f} @ {fps/pl if pl else 0:.1f} Hz")

if __name__ == "__main__":
    print(f"=== HEAD/CAMERA-MOTION COMPENSATION: {'ON' if COMPENSATE else 'OFF'} ===")
    for stem, fps, note in CLIPS:
        try: run(stem, fps, note)
        except Exception as e:
            import traceback; print(f"\n==== {stem} ERROR: {e}"); traceback.print_exc()
