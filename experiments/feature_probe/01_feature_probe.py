"""READ-ONLY feature-feasibility probe for multi-feature iris tracking.

Does NOT modify or import the tracking pipeline's tracker. It only:
  - imports IrisDetector to read MediaPipe's iris centre as an anatomical reference,
  - inside the iris disc, detects Shi-Tomasi corners and tracks them with Lucas-Kanade,
  - estimates the iris centre from the consensus (median translation) of survivors,
  - compares that against a plain single-template NCC tracker (replicated inline),
  - saves overlay frames + prints a structured report.

Nothing is written into the project pipeline; all images go to ./probe_out/ here.
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]  # repo root (portable across machines)
sys.path.insert(0, str(APP))
from src.core.iris_tracking import IrisDetector   # read-only: MediaPipe iris reference

OUT = Path(__file__).resolve().parent / "probe_out"
OUT.mkdir(exist_ok=True)
SAMPLES = APP / "samples"

N_FRAMES = 45            # run >= 30; extra frames to catch a possible blink
SURV_MARKS = (10, 20, 30)
FB_THRESH = 1.0          # forward-backward consistency (px) for a "good" track
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def better_eye_obs(frame, det):
    ok, le, re = det.detect(frame)
    if not ok:
        return None
    cands = [(e) for e in (le, re) if e.detected]
    return cands


def iris_circle(e):
    return float(e.single_x), float(e.single_y), float(e.iris_radius or e.radius or 20.0)


def detect_features(gray, cx, cy, r):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(0.95 * r), 255, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01,
                                  minDistance=3, blockSize=7, mask=mask)
    return pts.reshape(-1, 2).astype(np.float32) if pts is not None else np.empty((0, 2), np.float32)


def consensus_centre(cur, p0, alive, c0):
    """Iris centre under rigid translation: median over survivors of (cur - p0) + c0."""
    if alive.sum() < 3:
        return None
    d = cur[alive] - p0[alive]
    t = np.median(d, axis=0)
    return (float(c0[0] + t[0]), float(c0[1] + t[1]))


def make_template(gray, cx, cy, r):
    half = int(max(8, 1.3 * r))
    x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
    x1, y1 = min(gray.shape[1], int(cx) + half + 1), min(gray.shape[0], int(cy) + half + 1)
    return gray[y0:y1, x0:x1].copy(), (int(cx) - x0, int(cy) - y0)


def template_match(gray, templ, off, prev, r):
    win = int(max(18, 2.5 * r)); ph, pw = templ.shape
    cx, cy = prev
    sx0, sy0 = max(0, int(cx - win)), max(0, int(cy - win))
    sx1, sy1 = min(gray.shape[1], int(cx + win)), min(gray.shape[0], int(cy + win))
    roi = gray[sy0:sy1, sx0:sx1]
    if roi.shape[0] < ph or roi.shape[1] < pw:
        return prev, 0.0
    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
    _, mv, _, ml = cv2.minMaxLoc(res)
    return (sx0 + ml[0] + off[0], sy0 + ml[1] + off[1]), float(mv)


def quadrant_dist(pts, alive, c0):
    s = pts[alive]
    if len(s) == 0:
        return {}
    rel = s - np.array(c0)
    return {"temporal_upper": int(np.sum((rel[:, 0] >= 0) & (rel[:, 1] < 0))),
            "temporal_lower": int(np.sum((rel[:, 0] >= 0) & (rel[:, 1] >= 0))),
            "nasal_upper": int(np.sum((rel[:, 0] < 0) & (rel[:, 1] < 0))),
            "nasal_lower": int(np.sum((rel[:, 0] < 0) & (rel[:, 1] >= 0)))}


def probe(clip, tag):
    det = IrisDetector()
    cap = cv2.VideoCapture(str(clip))
    ok, f0 = cap.read()
    if not ok:
        return {"clip": clip.name, "error": "cannot read"}
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    cands = better_eye_obs(f0, det)
    if not cands:
        return {"clip": clip.name, "error": "no eye detected on init frame"}
    # pick the eye with more initial features
    best = None
    for e in cands:
        cx, cy, r = iris_circle(e)
        p = detect_features(g0, cx, cy, r)
        if best is None or len(p) > len(best[3]):
            best = (cx, cy, r, p)
    cx0, cy0, r0, p0 = best
    n0 = len(p0)
    iris_brightness = float(g0[max(0, int(cy0 - r0)):int(cy0 + r0),
                               max(0, int(cx0 - r0)):int(cx0 + r0)].mean())
    init_init_feats = {"L_or_R_best_eye_features": n0,
                       "all_eyes_features": [len(detect_features(g0, *iris_circle(e))) for e in cands]}

    pts = p0.copy()
    alive = np.ones(n0, bool)
    templ, toff = make_template(g0, cx0, cy0, r0)
    tprev = (cx0, cy0)
    prev_gray = g0
    surv, fb_means, cons, tcens, mpc, ears, motion = {}, [], [], [], [], [], []
    cons.append((cx0, cy0)); tcens.append((cx0, cy0))
    det2 = IrisDetector()
    save_frames = {0}
    for k in range(1, N_FRAMES + 1):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        # --- LK forward + backward ---
        if alive.sum() > 0:
            p1, stf, ef = cv2.calcOpticalFlowPyrLK(prev_gray, g, pts, None, **LK)
            p0b, stb, eb = cv2.calcOpticalFlowPyrLK(g, prev_gray, p1, None, **LK)
            fb = np.linalg.norm(pts - p0b, axis=1)
            good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < FB_THRESH)
            inb = (p1[:, 0] >= 0) & (p1[:, 0] < g.shape[1]) & (p1[:, 1] >= 0) & (p1[:, 1] < g.shape[0])
            newalive = alive & good & inb
            pts = np.where(newalive[:, None], p1, pts)
            c = consensus_centre(pts, p0, newalive, (cx0, cy0))
            if c is not None:                       # outlier gate: drop features far from consensus
                dist = np.linalg.norm(pts - np.array(c), axis=1)
                newalive &= dist < 1.6 * r0
            alive = newalive
            fb_means.append(float(np.mean(fb[alive])) if alive.sum() else float("nan"))
        c = consensus_centre(pts, p0, alive, (cx0, cy0))
        cons.append(c if c else cons[-1])
        # --- single-template tracker (replicated, for comparison) ---
        tprev, tscore = template_match(g, templ, toff, tprev, r0)
        tcens.append(tprev)
        # --- MediaPipe anatomical reference for this eye (nearest iris to init) ---
        cc = better_eye_obs(fr, det2)
        if cc:
            ic = min((iris_circle(e) for e in cc), key=lambda q: (q[0] - cx0) ** 2 + (q[1] - cy0) ** 2)
            mpc.append((ic[0], ic[1]))
            ears.append(min([e.ear for e in cc if e.ear is not None], default=float("nan")))
        else:
            mpc.append(None); ears.append(float("nan"))
        if mpc[-1]:
            motion.append(math.hypot(mpc[-1][0] - cx0, mpc[-1][1] - cy0))
        if k in SURV_MARKS:
            surv[k] = int(alive.sum())
            save_frames.add(k)
        prev_gray = g
    # blink frame = lowest EAR
    valid_ear = [(i + 1, e) for i, e in enumerate(ears) if not math.isnan(e)]
    blink_k = min(valid_ear, key=lambda z: z[1])[0] if valid_ear else None
    if blink_k:
        save_frames.add(blink_k)

    # --- gaps to MediaPipe reference: consensus vs template ---
    def gap(series, idx):
        if idx < len(series) and series[idx] and idx - 1 < len(mpc) and mpc[idx - 1]:
            return math.hypot(series[idx][0] - mpc[idx - 1][0], series[idx][1] - mpc[idx - 1][1])
        return None
    gaps = {k: {"consensus_vs_MP_px": gap(cons, k), "template_vs_MP_px": gap(tcens, k)} for k in SURV_MARKS}

    def jitter(series):
        d = [math.hypot(series[i][0] - series[i - 1][0], series[i][1] - series[i - 1][1])
             for i in range(1, len(series)) if series[i] and series[i - 1]]
        return round(float(np.median(d)), 2) if d else None

    # --- save overlay frames ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    saved = []
    # re-run quickly just to render the chosen frames with the recorded alive/pos is complex;
    # instead re-track lightweight to render. Simpler: re-run storing per-frame snapshots.
    cap.release()
    return {
        "clip": clip.name, "tag": tag, "best_eye_centre": [round(cx0, 1), round(cy0, 1)],
        "iris_radius_px": round(r0, 1), "iris_roi_brightness_0_255": round(iris_brightness, 1),
        "initial_features": init_init_feats,
        "survivors": {**{f"frame_{k}": surv.get(k) for k in SURV_MARKS}},
        "survival_pct": {f"frame_{k}": (round(100 * surv[k] / n0, 1) if (k in surv and n0) else None)
                          for k in SURV_MARKS},
        "mean_fb_error_px": round(float(np.nanmean(fb_means)), 3) if fb_means else None,
        "quadrant_distribution_init": quadrant_dist(p0, np.ones(n0, bool), (cx0, cy0)),
        "consensus_centre_jitter_px": jitter(cons),
        "template_centre_jitter_px": jitter(tcens),
        "gaps_to_MediaPipe": gaps,
        "ear_min": round(min([e for e in ears if not math.isnan(e)], default=float('nan')), 3),
        "ear_baseline": round(float(np.nanmedian(ears)), 3) if ears else None,
        "blink_frame": blink_k,
        "head_motion_px_max": round(max(motion), 1) if motion else None,
        "_render": dict(clip=str(clip), cx0=cx0, cy0=cy0, r0=r0, save_frames=sorted(save_frames)),
    }


def render(meta):
    """Second pass: redo tracking but save overlay PNGs at the chosen frames (green=alive, red=lost)."""
    r = meta["_render"]
    clip, cx0, cy0, r0 = r["clip"], r["cx0"], r["cy0"], r["r0"]
    det = IrisDetector()
    cap = cv2.VideoCapture(clip)
    ok, f0 = cap.read(); g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    p0 = detect_features(g0, cx0, cy0, r0); n0 = len(p0)
    pts = p0.copy(); alive = np.ones(n0, bool); prev = g0
    templ, toff = make_template(g0, cx0, cy0, r0); tprev = (cx0, cy0)
    half = int(2.4 * r0)

    def crop(frame, cc, feats, al, cons, tcen, mp, k):
        x0, y0 = max(0, int(cx0 - half * 1.6)), max(0, int(cy0 - half))
        x1, y1 = int(cx0 + half * 1.6), int(cy0 + half)
        vis = frame.copy()
        for i, p in enumerate(feats):
            col = (0, 200, 0) if al[i] else (0, 0, 255)
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, col, -1)
        if cons: cv2.drawMarker(vis, (int(cons[0]), int(cons[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        if tcen: cv2.drawMarker(vis, (int(tcen[0]), int(tcen[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        if mp: cv2.circle(vis, (int(mp[0]), int(mp[1])), int(r0), (0, 220, 220), 1)
        sub = vis[max(0, y0):y1, max(0, x0):x1]
        sub = cv2.resize(sub, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(sub, f"{Path(clip).stem[:18]} f{k} alive {int(al.sum())}/{n0}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(sub, "green=alive red=lost  cyan+=consensus  magenta x=template  amber o=MP iris",
                    (8, sub.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        out = OUT / f"{meta['tag']}_{Path(clip).stem[:14]}_f{k:02d}.png"
        cv2.imwrite(str(out), sub)
        return out
    det2 = IrisDetector()
    saved = []
    if 0 in r["save_frames"]:
        cc = better_eye_obs(f0, det2); mp = None
        if cc: q = min((iris_circle(e) for e in cc), key=lambda z: (z[0]-cx0)**2+(z[1]-cy0)**2); mp=(q[0],q[1])
        saved.append(str(crop(f0, None, p0, alive, (cx0, cy0), (cx0, cy0), mp, 0)))
    for k in range(1, max(r["save_frames"]) + 1):
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        p1, stf, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **LK)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **LK)
        fb = np.linalg.norm(pts - p0b, axis=1)
        good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < FB_THRESH)
        alive = alive & good
        pts = np.where(alive[:, None], p1, pts)
        cons = consensus_centre(pts, p0, alive, (cx0, cy0))
        if cons:
            dist = np.linalg.norm(pts - np.array(cons), axis=1); alive = alive & (dist < 1.6 * r0)
        tprev, _ = template_match(g, templ, toff, tprev, r0)
        mp = None
        cc = better_eye_obs(fr, det2)
        if cc: q = min((iris_circle(e) for e in cc), key=lambda z: (z[0]-cx0)**2+(z[1]-cy0)**2); mp=(q[0],q[1])
        if k in r["save_frames"]:
            saved.append(str(crop(fr, None, pts, alive, cons, tprev, mp, k)))
        prev = g
    cap.release()
    return saved


def main():
    det = IrisDetector()
    # choose a lighter-iris clip = brightest iris ROI among candidates
    cand = ["1.mp4", "3.mp4", "gaze-evoked nystagmus-1.mp4", "gaze-evoked nystagmus in pontine glioma.mp4"]
    bright = []
    for c in cand:
        p = SAMPLES / c
        if not p.exists(): continue
        cap = cv2.VideoCapture(str(p)); ok, f = cap.read(); cap.release()
        if not ok: continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        cc = better_eye_obs(f, det)
        if not cc: continue
        e = cc[0]; cx, cy, r = iris_circle(e)
        b = float(g[max(0,int(cy-r)):int(cy+r), max(0,int(cx-r)):int(cx+r)].mean())
        bright.append((b, c))
    bright.sort(reverse=True)
    lighter = bright[0][1] if bright else "1.mp4"
    print("iris brightness ranking (brightest first):", [(round(b,1), c) for b, c in bright])
    print("=> lighter-iris clip chosen:", lighter)

    clips = [("fistula", SAMPLES / "fistula nystagmus.mp4"),
             ("ref2", SAMPLES / "2.mp4"),
             ("lighter", SAMPLES / lighter)]
    results = []
    for tag, clip in clips:
        if not clip.exists():
            print("MISSING", clip); continue
        print(f"\n--- probing {tag}: {clip.name} ---")
        m = probe(clip, tag)
        imgs = render(m) if "_render" in m else []
        m["overlay_images"] = imgs
        m.pop("_render", None)
        results.append(m)
        print(json.dumps(m, indent=2))
    (OUT / "probe_report.json").write_text(json.dumps(results, indent=2))
    print("\nSaved report + overlays to", OUT)


if __name__ == "__main__":
    main()
