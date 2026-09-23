"""Kymograph MOTION-STRIP probe -- QUALITATIVE, clinician-style nystagmus screening.

STANDALONE PROBE (2026-07-06, Dr. K's simplified goal). Does NOT import or modify the committed detector
(`tools/ellseg_centroid_trace.py`, `tools/nystagmus_direction.py`). Pure OpenCV + NumPy; no EllSeg/torch.
Nothing is committed by this script.

Goal (Dr. K): imitate the clinician's qualitative viewing, NOT quantitative VNG. Per clip/eye it answers:
  1) is the eye moving or static?  2) is it rhythmic / jerk-like?  3) is there a fast phase + direction
  (left/right/up/down)?  4) or is the signal too poor -> UNCERTAIN.

Principle: APPROXIMATE, ROBUST, CLINICIAN-READABLE beats precise-but-fragile. No peak velocity, no VNG metric,
no iris outline. One clear eye is enough (single-eye supported).

Pipeline (per eye):
  ROI  = manual circle seed (centre+radius) from manual_seeds.json  ->  square crop of side ~2*BOX_K*r
  STAB = subtract head/camera motion: track peri-orbital face features OUTSIDE the eye boxes (sparse
         Lucas-Kanade), accumulate a global translation, and shift the crop to follow the head. (This is
         video stabilization of rigid landmarks -- NOT the discarded dense-flow eye-motion reader; the
         nystagmus is read from the KYMOGRAPH, not from a flow scalar.)
  STRIP = kymograph / motion strip: a fixed horizontal slit through the iris, stacked over time -> the dark
         iris draws a trace whose vertical position = horizontal eye position (a sawtooth for jerk nystagmus).
         A second vertical slit -> vertical motion strip (for up/down-beat).
  TRACE = approximate dark-iris centroid per frame (overlaid on the strips; used for the qualitative call).

Outputs (outputs/_kymograph_probe/<clip>/<eye>/):  stabilized_crop.mp4, h_strip.png, v_strip.png,
panel.png, trace.csv.  A run summary is printed and written to outputs/_kymograph_probe/REPORT.md.

Run:  py -3.14 tools/kymograph_probe.py                 # all seeded clips
      py -3.14 tools/kymograph_probe.py "clip stem" ... # only these
"""
import sys, os, json, csv
from pathlib import Path
import numpy as np, cv2

REPO    = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
SEEDF   = REPO / "manual_seeds.json"
OUTROOT = REPO / "outputs" / "_kymograph_probe"

# ---- tunables (qualitative; deliberately forgiving) -----------------------------------------------
BOX_K      = 1.9      # crop half-size = BOX_K * iris_radius
SLIT_K     = 1.0      # motion-strip slit half-thickness = SLIT_K * r
MAXSEC     = 16.0     # cap analysis length
VIS_MAX    = 240      # max stored patch side (memory / video size)
MOVE_REL   = 0.06     # detrended amplitude (fraction of r) above which the eye is "moving"
VBIAS      = 1.25     # vertical is only chosen if its amplitude beats horizontal by this factor
RHY_LO, RHY_HI = 1.0, 8.0     # nystagmus beat band (Hz)
RHY_RATIO  = 0.12     # min in-band power fraction to call it rhythmic
CONTRAST_MIN = 0.06   # min median iris/surround contrast; below -> poor signal
VALID_MIN  = 0.55     # min fraction of frames with a usable dark iris; below -> poor signal


# ================================ helpers ================================
def resolve(stem):
    """Clip stem -> mp4 in samples/. Prefer .mp4; NEVER .mpg/.wmv (unplayable / not used)."""
    order = [".mp4", ".MP4", ".mov", ".avi", ".mkv"]
    c = [p for p in SAMPLES.iterdir()
         if p.is_file() and p.stem == stem and p.suffix.lower() not in (".mpg", ".wmv")]
    c.sort(key=lambda p: order.index(p.suffix) if p.suffix in order else 99)
    return c[0] if c else None

def rolling_median(a, win):
    win = max(3, int(win) | 1); pad = win // 2
    ap = np.pad(a, (pad, pad), mode="edge")
    return np.array([np.median(ap[i:i + win]) for i in range(len(a))])

def mad_std(x):
    x = x[np.isfinite(x)]
    if x.size < 3: return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))

def dom_freq(sig, fps, lo=RHY_LO, hi=RHY_HI):
    sig = np.asarray(sig, float); sig = sig[np.isfinite(sig)]
    n = len(sig)
    if n < 12: return 0.0, 0.0
    sig = sig - sig.mean(); w = np.hanning(n)
    ps = np.abs(np.fft.rfft(sig * w)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / fps)
    band = (fr >= lo) & (fr <= hi)
    total = ps[1:].sum() + 1e-9
    if not band.any(): return 0.0, 0.0
    bi = np.argmax(np.where(band, ps, 0.0))
    return float(fr[bi]), float(ps[band].sum() / total)

def vskew(sig):
    """Sign points to the FAST phase: slow drift = many small steps, fast beat = few large steps."""
    v = np.diff(np.asarray(sig, float)); v = v[np.isfinite(v)]
    if len(v) < 8: return 0.0
    m, s = v.mean(), v.std() + 1e-9
    return float(np.mean(((v - m) / s) ** 3))

def dark_centroid(prof, r):
    """Robust iris position along a slit profile: locate the darkest point, then take the darkness-weighted
    centroid within +/-~r of it. Smooth and glitch-free (no connected components / no dark-corner capture).
    Returns (position_px, contrast)."""
    p = cv2.GaussianBlur(prof.astype(np.float32).reshape(1, -1), (0, 0), 1.0).ravel()
    i0 = int(np.argmin(p)); w0 = int(max(3, 1.2 * r))
    a, b = max(0, i0 - w0), min(len(p), i0 + w0)
    seg = p[a:b]; wt = np.clip(seg.max() - seg, 0, None)
    x = np.arange(a, b)
    pos = float((x * wt).sum() / (wt.sum() + 1e-9))
    contrast = float((p.max() - p.min()) / (p.mean() + 1e-6))
    return pos, contrast

def feat_mask(shape, centers, box):
    m = np.full(shape, 255, np.uint8)
    b = 5; m[:b] = 0; m[-b:] = 0; m[:, :b] = 0; m[:, -b:] = 0
    for (cx, cy) in centers:
        cv2.circle(m, (int(cx), int(cy)), int(1.4 * box), 0, -1)
    return m

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# ================================ per-clip processing ================================
def process(clip, seeds):
    v = resolve(clip)
    if not v:
        print(f"  [skip] {clip}: no mp4 in samples/"); return None
    eyes = {ek: s for ek, s in seeds.items() if isinstance(s, dict)}
    if not eyes:
        print(f"  [skip] {clip}: no circle seed"); return None
    cap = cv2.VideoCapture(str(v))
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 25.0; NF = int(cap.get(7))
    start = min(int(s.get("frame", 0)) for s in eyes.values())
    end = min(NF, start + int(MAXSEC * fps))
    box = {ek: int(round(BOX_K * s["r"])) for ek, s in eyes.items()}
    ctr0 = {ek: (float(s["cx"]), float(s["cy"])) for ek, s in eyes.items()}
    rad = {ek: float(s["r"]) for ek, s in eyes.items()}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev_gray = prev_pts = None
    Dx = Dy = 0.0; feat_hist = []
    store = {ek: dict(H=[], V=[], px=[], py=[], ct=[], val=[], vis=[]) for ek in eyes}

    fi = start
    while fi < end:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_centers = [(ctr0[ek][0] + Dx, ctr0[ek][1] + Dy) for ek in eyes]
        if prev_gray is None:
            prev_pts = cv2.goodFeaturesToTrack(gray, 250, 0.01, 7,
                                               mask=feat_mask(gray.shape, cur_centers, max(box.values())))
        else:
            if prev_pts is not None and len(prev_pts) >= 6:
                np_, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK)
                if np_ is not None:
                    gn = np_[stt.ravel() == 1]; go = prev_pts[stt.ravel() == 1]
                    if len(gn) >= 6:
                        d = (gn - go).reshape(-1, 2)
                        md = np.median(d, axis=0)
                        Dx += float(md[0]); Dy += float(md[1])
                        prev_pts = gn.reshape(-1, 1, 2)
                    else:
                        prev_pts = None
                else:
                    prev_pts = None
            if prev_pts is None or len(prev_pts) < 25:
                cur_centers = [(ctr0[ek][0] + Dx, ctr0[ek][1] + Dy) for ek in eyes]
                prev_pts = cv2.goodFeaturesToTrack(gray, 250, 0.01, 7,
                                                   mask=feat_mask(gray.shape, cur_centers, max(box.values())))
        feat_hist.append(0 if prev_pts is None else len(prev_pts))
        prev_gray = gray

        for ek, s in eyes.items():
            b = box[ek]; r = rad[ek]
            cen = (ctr0[ek][0] + Dx, ctr0[ek][1] + Dy)
            patch = cv2.getRectSubPix(frame, (2 * b, 2 * b), cen)          # subpixel, border-replicated
            pg = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            cH, cW = pg.shape
            sh = int(min(SLIT_K * r, cH / 2 - 2, cW / 2 - 2))
            cy0, cx0 = cH // 2, cW // 2                                     # fixed geometric slits (generous)
            hprof = pg[cy0 - sh:cy0 + sh, :].mean(0)                        # horizontal slit -> position along x
            vprof = pg[:, cx0 - sh:cx0 + sh].mean(1)                        # vertical slit  -> position along y
            px, cth = dark_centroid(hprof, r)                              # smooth darkness-centroid trace
            py, ctv = dark_centroid(vprof, r)
            ct = max(cth, ctv); val = 1 if ct >= CONTRAST_MIN else 0
            st = store[ek]
            st["H"].append(hprof); st["V"].append(vprof)
            st["px"].append(px); st["py"].append(py); st["ct"].append(ct); st["val"].append(val)
            # visual patch (downscaled) with slit band + darkness-centroid marker
            vis = patch.copy()
            cv2.drawMarker(vis, (int(px), int(py)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.rectangle(vis, (0, cy0 - sh), (cW, cy0 + sh), (60, 200, 60), 1)
            sc = VIS_MAX / max(cW, cH)
            if sc < 1: vis = cv2.resize(vis, (int(cW * sc), int(cH * sc)))
            st["vis"].append(vis)
        fi += 1
    cap.release()

    results = {}
    for ek in eyes:
        results[ek] = analyse_and_render(clip, ek, store[ek], rad[ek], box[ek], fps,
                                         int(np.median(feat_hist) if feat_hist else 0), v.name, W, H)
    return dict(video=v.name, W=W, H=H, fps=fps, eyes=results)


def analyse_and_render(clip, ek, st, r, box, fps, feat_med, vname, W, H):
    n = len(st["px"])
    outdir = OUTROOT / clip / ek; outdir.mkdir(parents=True, exist_ok=True)
    px = np.array(st["px"], float); py = np.array(st["py"], float)
    val = np.array(st["val"], int); ct = np.array(st["ct"], float)
    valid_frac = float(val.mean()) if n else 0.0
    contrast_med = float(np.median(ct)) if n else 0.0

    # interpolate dropouts, detrend slow head residual
    def interp(a):
        a = a.copy(); bad = (val == 0)
        if bad.all(): return a
        idx = np.arange(n); a[bad] = np.interp(idx[bad], idx[~bad], a[~bad]); return a
    pxi, pyi = interp(px), interp(py)
    win = max(5, int(1.5 * fps))
    dh = pxi - rolling_median(pxi, win); dv = pyi - rolling_median(pyi, win)
    amp_h, amp_v = mad_std(dh), mad_std(dv)
    rel_h, rel_v = amp_h / r, amp_v / r

    # axis / rhythm / direction
    if rel_v > VBIAS * rel_h and rel_v > MOVE_REL: axis, sig, rel = "vertical", dv, rel_v
    else: axis, sig, rel = "horizontal", dh, rel_h
    fpk, ratio = dom_freq(sig, fps)
    sk = vskew(sig)
    if axis == "horizontal": direction = "left-beating" if sk > 0 else "right-beating"
    else: direction = "down-beating" if sk > 0 else "up-beating"

    moving = (rel_h > MOVE_REL) or (rel_v > MOVE_REL)
    rhythmic = moving and (RHY_LO <= fpk <= RHY_HI) and (ratio >= RHY_RATIO)
    poor = (valid_frac < VALID_MIN) or (contrast_med < CONTRAST_MIN)

    if poor:
        label = "UNCERTAIN - poor signal (low iris contrast / lost tracking)"; cat = "uncertain"
    elif not moving:
        label = "STATIC / no jerk nystagmus"; cat = "static"
    elif rhythmic:
        label = f"RHYTHMIC jerk-like -> {direction}  [{axis}]"; cat = "nystagmus"
    else:
        label = "UNCERTAIN - irregular motion, not clearly rhythmic"; cat = "uncertain"

    stats = dict(cat=cat, label=label, axis=axis, direction=direction if cat == "nystagmus" else "-",
                 freq=round(fpk, 2), power=round(ratio, 2), rel_h=round(rel_h, 3), rel_v=round(rel_v, 3),
                 valid=round(valid_frac, 2), contrast=round(contrast_med, 3), feats=feat_med, nframes=n)

    # ---- renders ----
    Hs = np.array(st["H"]);  Vs = np.array(st["V"])                     # [n, L]
    _render_strip(outdir / "h_strip.png", Hs, pxi, val, fps, r,
                  f"{clip} [{ek}] - HORIZONTAL motion strip", "image-LEFT  (patient-R gaze)",
                  "image-RIGHT (patient-L gaze)", label)
    _render_strip(outdir / "v_strip.png", Vs, pyi, val, fps, r,
                  f"{clip} [{ek}] - VERTICAL motion strip", "UP", "DOWN", label)
    _render_wave(outdir / "wave.png", dh, dv, val, fps,
                 f"{clip} [{ek}] - detrended eye-position trace (slow drift removed)")
    _panel(outdir / "panel.png", [outdir / "h_strip.png", outdir / "v_strip.png", outdir / "wave.png"],
           clip, ek, stats)
    _write_video(outdir / "stabilized_crop.mp4", st["vis"], fps)
    with open(outdir / "trace.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp); w.writerow(["frame", "px", "py", "valid", "contrast"])
        for i in range(n): w.writerow([i, round(px[i], 2), round(py[i], 2), val[i], round(ct[i], 3)])
    print(f"    {ek}: {label}   (axis {axis} f~{fpk:.1f}Hz pw{ratio:.2f} relH{rel_h:.02f} relV{rel_v:.02f} "
          f"valid{valid_frac:.2f} contr{contrast_med:.02f})")
    return stats


def _colorize(strip):
    lo, hi = np.percentile(strip, 3), np.percentile(strip, 97)
    s = np.clip((strip - lo) / (hi - lo + 1e-6), 0, 1)
    s = s ** 0.85
    return cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_BONE)

def _render_strip(path, prof2d, trace, val, fps, r, title, top_lbl, bot_lbl, label):
    n, L = prof2d.shape
    img = prof2d.T                                                      # rows = position, cols = time
    col = _colorize(img)
    tsc = max(1, int(round(1100 / max(n, 1)))); rsc = max(1, int(round(220 / max(L, 1))))
    col = cv2.resize(col, (n * tsc, L * rsc), interpolation=cv2.INTER_NEAREST)
    # overlay approximate centre trace (red) + dropout ticks
    pts = []
    for i in range(n):
        y = int(np.clip(trace[i], 0, L - 1) * rsc)
        pts.append((i * tsc, y))
        if not val[i]: cv2.line(col, (i * tsc, (L * rsc) - 3), (i * tsc, L * rsc), (0, 180, 255), 1)
    for i in range(1, n):
        if val[i] and val[i - 1]:
            cv2.line(col, pts[i - 1], pts[i], (40, 40, 255), 1, cv2.LINE_AA)
    # time gridlines every second
    for s in range(1, int(n / fps) + 1):
        x = int(s * fps * tsc); cv2.line(col, (x, 0), (x, L * rsc), (90, 90, 90), 1)
    # frame with margins + labels
    pad_l, pad_t, pad_b = 210, 30, 26
    canvas = np.full((pad_t + L * rsc + pad_b, pad_l + n * tsc + 20, 3), 20, np.uint8)
    canvas[pad_t:pad_t + L * rsc, pad_l:pad_l + n * tsc] = col
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (10, 20), f, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, top_lbl, (6, pad_t + 14), f, 0.42, (150, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, bot_lbl, (6, pad_t + L * rsc - 6), f, 0.42, (150, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "time ->", (pad_l, pad_t + L * rsc + 18), f, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (pad_l + 90, pad_t + L * rsc + 18), f, 0.42, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)

def _render_wave(path, dh, dv, val, fps, title):
    """Detrended eye-position trace -- the clinician-recognizable VNG-style waveform (sawtooth = jerk)."""
    n = len(dh); plotW = max(600, min(1100, n * max(1, int(round(1100 / max(n, 1))))))
    plotH = 150; pad_l, pad_t, pad_b = 60, 26, 22
    canvas = np.full((pad_t + plotH + pad_b, pad_l + plotW + 20, 3), 22, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(mad_std(dh), mad_std(dv), 1e-3) * 4.0                    # +/-4 MAD fills half-height
    mid = pad_t + plotH // 2
    cv2.line(canvas, (pad_l, mid), (pad_l + plotW, mid), (70, 70, 70), 1)
    for s in range(1, int(n / fps) + 1):
        x = pad_l + int(s * fps / max(n, 1) * plotW); cv2.line(canvas, (x, pad_t), (x, pad_t + plotH), (45, 45, 45), 1)
    def draw(sig, color):
        pts = []
        for i in range(n):
            x = pad_l + int(i / max(n, 1) * plotW)
            y = int(np.clip(mid - sig[i] / scale * (plotH // 2), pad_t, pad_t + plotH))
            pts.append((x, y))
        for i in range(1, n): cv2.line(canvas, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
    draw(dh, (255, 210, 90)); draw(dv, (90, 210, 255))                   # H = cyan-ish, V = amber
    cv2.putText(canvas, title, (8, 18), f, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "H", (pad_l + plotW - 40, pad_t + 14), f, 0.5, (255, 210, 90), 2, cv2.LINE_AA)
    cv2.putText(canvas, "V", (pad_l + plotW - 20, pad_t + 14), f, 0.5, (90, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "time ->", (pad_l, pad_t + plotH + 16), f, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)

def _panel(path, parts, clip, ek, stats):
    ims = [cv2.imread(str(p)) for p in parts]
    w = max(im.shape[1] for im in ims)
    def padw(im): return cv2.copyMakeBorder(im, 0, 0, 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
    ims = [padw(im) for im in ims]
    head = np.full((30, w, 3), 30, np.uint8)
    cv2.putText(head, f"{clip} [{ek}]  ::  {stats['label']}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([head] + ims))

def _write_video(path, frames, fps):
    if not frames: return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (w, h))
    for fr in frames:
        if fr.shape[:2] != (h, w): fr = cv2.resize(fr, (w, h))
        vw.write(fr)
    vw.release()


# ================================ main ================================
def main():
    seeds = json.load(open(SEEDF, encoding="utf-8"))
    want = sys.argv[1:]
    clips = want if want else [c for c in seeds if resolve(c)]
    OUTROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in clips:
        if clip not in seeds:
            print(f"[skip] {clip}: not in manual_seeds.json"); continue
        print(f"* {clip}")
        res = process(clip, seeds[clip])
        if not res: continue
        for ek, s in res["eyes"].items():
            rows.append((clip, ek, s["cat"], s["label"], s["axis"], s["freq"], s["power"],
                         s["rel_h"], s["rel_v"], s["valid"], s["contrast"]))
    # report
    with open(OUTROOT / "REPORT.md", "w", encoding="utf-8") as fp:
        fp.write("# Kymograph motion-strip probe - run report\n\n")
        fp.write("Qualitative screening (approximate/robust). Not VNG. See panel.png per clip/eye.\n\n")
        fp.write("| clip | eye | category | call | axis | f(Hz) | power | relH | relV | valid | contrast |\n")
        fp.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fp.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"\nreport -> {OUTROOT / 'REPORT.md'}   ({len(rows)} eye-results)")

if __name__ == "__main__":
    main()
