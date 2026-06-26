"""READ-ONLY feature-feasibility probe v2 — blink-aware.

Fixes v1 confounds: (a) start on a properly DETECTED frame, (b) place the 30-frame survival
window inside a BLINK-FREE stretch (EAR scan), (c) characterise blink separately (reset +
re-detection recovery), (d) report consensus jitter only with >=3 survivors.
Touches no pipeline code; images -> ./probe_out2/ .
"""
import sys, json, math
from pathlib import Path
import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]  # repo root (portable across machines)
sys.path.insert(0, str(APP))
from src.core.iris_tracking import IrisDetector

OUT = Path(__file__).resolve().parent / "probe_out2"; OUT.mkdir(exist_ok=True)
SAMPLES = APP / "samples"
NEED = 35
SCAN = 170
FB_THRESH = 1.0
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def eyes(frame, det):
    ok, le, re = det.detect(frame)
    return [e for e in (le, re) if e.detected] if ok else []


def circ(e):
    return float(e.single_x), float(e.single_y), float(e.iris_radius or e.radius or 20.0)


def feats(gray, cx, cy, r):
    m = np.zeros(gray.shape, np.uint8); cv2.circle(m, (int(cx), int(cy)), int(0.95 * r), 255, -1)
    p = cv2.goodFeaturesToTrack(gray, 80, 0.01, 3, blockSize=7, mask=m)
    return p.reshape(-1, 2).astype(np.float32) if p is not None else np.empty((0, 2), np.float32)


def consensus(cur, p0, alive, c0):
    if alive.sum() < 3:
        return None
    t = np.median(cur[alive] - p0[alive], axis=0)
    return (float(c0[0] + t[0]), float(c0[1] + t[1]))


def mk_templ(gray, cx, cy, r):
    h = int(max(8, 1.3 * r)); x0, y0 = max(0, int(cx) - h), max(0, int(cy) - h)
    x1, y1 = min(gray.shape[1], int(cx) + h + 1), min(gray.shape[0], int(cy) + h + 1)
    return gray[y0:y1, x0:x1].copy(), (int(cx) - x0, int(cy) - y0)


def tmatch(gray, templ, off, prev, r):
    win = int(max(18, 2.5 * r)); ph, pw = templ.shape; cx, cy = prev
    sx0, sy0 = max(0, int(cx - win)), max(0, int(cy - win))
    sx1, sy1 = min(gray.shape[1], int(cx + win)), min(gray.shape[0], int(cy + win))
    roi = gray[sy0:sy1, sx0:sx1]
    if roi.shape[0] < ph or roi.shape[1] < pw:
        return prev
    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED); _, _, _, ml = cv2.minMaxLoc(res)
    return (sx0 + ml[0] + off[0], sy0 + ml[1] + off[1])


def quad(pts, c0):
    if len(pts) == 0:
        return {}
    rel = pts - np.array(c0)
    return {"temporal_upper": int(np.sum((rel[:, 0] >= 0) & (rel[:, 1] < 0))),
            "temporal_lower": int(np.sum((rel[:, 0] >= 0) & (rel[:, 1] >= 0))),
            "nasal_upper": int(np.sum((rel[:, 0] < 0) & (rel[:, 1] < 0))),
            "nasal_lower": int(np.sum((rel[:, 0] < 0) & (rel[:, 1] >= 0)))}


def scan_ear(clip, det):
    """Per-frame EAR + eye-detected over first SCAN frames; returns lists."""
    cap = cv2.VideoCapture(str(clip)); ears, ok_eye = [], []; f = 0
    while f < SCAN:
        ok, fr = cap.read()
        if not ok:
            break
        es = eyes(fr, det)
        if es:
            e = [x.ear for x in es if x.ear is not None]
            ears.append(min(e) if e else float("nan")); ok_eye.append(True)
        else:
            ears.append(float("nan")); ok_eye.append(False)
        f += 1
    cap.release(); return ears, ok_eye


def longest_clean(ears, ok_eye, base):
    good = [ok_eye[i] and not math.isnan(ears[i]) and ears[i] >= 0.80 * base for i in range(len(ears))]
    best = (0, 0); i = 0
    while i < len(good):
        if good[i]:
            j = i
            while j < len(good) and good[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best  # [start, end)


def track_window(clip, det, det2, start, n, tag):
    cap = cv2.VideoCapture(str(clip)); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, f0 = cap.read()
    if not ok:
        return None
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    es = eyes(f0, det)
    if not es:
        cap.release(); return None
    cx0, cy0, r0 = max((circ(e) for e in es), key=lambda q: len(feats(g0, *q)))
    p0 = feats(g0, cx0, cy0, r0); n0 = len(p0)
    bright = float(g0[max(0, int(cy0 - r0)):int(cy0 + r0), max(0, int(cx0 - r0)):int(cx0 + r0)].mean())
    pts = p0.copy(); alive = np.ones(n0, bool)
    templ, toff = mk_templ(g0, cx0, cy0, r0); tprev = (cx0, cy0); prev = g0
    cons_series = [(cx0, cy0)]; tcen = [(cx0, cy0)]; mpref = [(cx0, cy0)]
    surv, fbs, save = {}, [], {0: (f0.copy(), p0.copy(), alive.copy(), (cx0, cy0), (cx0, cy0), (cx0, cy0))}
    for k in range(1, n + 1):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        p1, stf, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **LK)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **LK)
        fb = np.linalg.norm(pts - p0b, axis=1)
        good = (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < FB_THRESH)
        inb = (p1[:, 0] >= 0) & (p1[:, 0] < g.shape[1]) & (p1[:, 1] >= 0) & (p1[:, 1] < g.shape[0])
        alive = alive & good & inb
        pts = np.where(alive[:, None], p1, pts)
        c = consensus(pts, p0, alive, (cx0, cy0))
        if c is not None:
            d = np.linalg.norm(pts - np.array(c), axis=1); alive = alive & (d < 1.6 * r0)
        c = consensus(pts, p0, alive, (cx0, cy0))
        cons_series.append(c if c else None)
        if alive.sum():
            fbs.append(float(np.mean(fb[alive])))
        tprev = tmatch(g, templ, toff, tprev, r0); tcen.append(tprev)
        es2 = eyes(fr, det2)
        mpref.append(min((circ(e)[:2] for e in es2), key=lambda q: (q[0] - cx0) ** 2 + (q[1] - cy0) ** 2)
                     if es2 else None)
        if k in (10, 20, 30):
            surv[k] = int(alive.sum())
            save[k] = (fr.copy(), pts.copy(), alive.copy(), c, tprev, mpref[-1])
        prev = g
    cap.release()

    def gap(series, k):
        if k < len(series) and series[k] and k < len(mpref) and mpref[k]:
            return round(math.hypot(series[k][0] - mpref[k][0], series[k][1] - mpref[k][1]), 1)
        return None

    def jit(series):
        d = [math.hypot(series[i][0] - series[i-1][0], series[i][1] - series[i-1][1])
             for i in range(1, len(series)) if series[i] and series[i-1]]
        return round(float(np.median(d)), 2) if d else None
    # consensus drift vs MP over the window (how far the centre estimate departs from anatomy)
    drift_cons = gap(cons_series, min(n, 30))
    # render
    half = int(2.4 * r0); imgs = []
    for k, (fr, fp, al, c, tc, mp) in sorted(save.items()):
        vis = fr.copy()
        for i, p in enumerate(fp):
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 200, 0) if al[i] else (0, 0, 255), -1)
        if c: cv2.drawMarker(vis, (int(c[0]), int(c[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        if tc: cv2.drawMarker(vis, (int(tc[0]), int(tc[1])), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        if mp: cv2.circle(vis, (int(mp[0]), int(mp[1])), int(r0), (0, 220, 220), 1)
        x0, y0 = max(0, int(cx0 - half*1.6)), max(0, int(cy0 - half))
        sub = vis[y0:int(cy0 + half), x0:int(cx0 + half*1.6)]
        sub = cv2.resize(sub, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(sub, f"{tag} +{k}f  alive {int(al.sum())}/{n0}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = OUT / f"{tag}_win_+{k:02d}.png"; cv2.imwrite(str(out), sub); imgs.append(str(out))
    return {"start_frame": start, "init_features": n0, "iris_radius_px": round(r0, 1),
            "iris_brightness": round(bright, 1),
            "survivors": surv, "survival_pct": {k: round(100*surv[k]/n0, 1) for k in surv if n0},
            "mean_fb_error_px": round(float(np.mean(fbs)), 3) if fbs else None,
            "quadrant_distribution": quad(p0, (cx0, cy0)),
            "consensus_jitter_px": jit([c for c in cons_series if c]),
            "template_jitter_px": jit(tcen),
            "consensus_drift_vs_MP_px_at30": drift_cons,
            "gap_to_MP_at30": {"consensus": gap(cons_series, min(n, 30)),
                               "template": gap(tcen, min(n, 30))},
            "images": imgs}


def blink_recovery(clip, det, det2, blink_k, tag):
    """Track into a blink and re-detect after: pre-count, min during, recovered count."""
    cap = cv2.VideoCapture(str(clip)); start = max(0, blink_k - 6)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start); ok, f0 = cap.read()
    if not ok:
        return None
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); es = eyes(f0, det)
    if not es:
        return None
    cx0, cy0, r0 = max((circ(e) for e in es), key=lambda q: len(feats(g0, *q)))
    p0 = feats(g0, cx0, cy0, r0); pts = p0.copy(); alive = np.ones(len(p0), bool); prev = g0
    pre = len(p0); during_min = pre; recovered = None
    for k in range(1, 14):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        p1, stf, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **LK)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **LK)
        fb = np.linalg.norm(pts - p0b, axis=1)
        alive = alive & (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < FB_THRESH)
        pts = np.where(alive[:, None], p1, pts); during_min = min(during_min, int(alive.sum()))
        es2 = eyes(fr, det2)
        if alive.sum() < 3 and es2:          # blink passed → re-detect
            c2 = max((circ(e) for e in es2), key=lambda q: len(feats(g, *q)))
            recovered = len(feats(g, *c2)); break
        prev = g
    cap.release()
    return {"pre_blink_features": pre, "min_during_blink": during_min, "recovered_after_redetect": recovered}


def main():
    det, det2 = IrisDetector(), IrisDetector()
    # lighter-iris pick (brightest detected iris among candidates)
    cand = ["1.mp4", "3.mp4", "gaze-evoked nystagmus-1.mp4", "gaze-evoked nystagmus in pontine glioma.mp4"]
    br = []
    for c in cand:
        p = SAMPLES / c
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p)); ok, f = cap.read(); cap.release()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); es = eyes(f, det)
        if not es:
            continue
        cx, cy, r = circ(es[0]); br.append((round(float(g[max(0,int(cy-r)):int(cy+r),
                                      max(0,int(cx-r)):int(cx+r)].mean()), 1), c))
    br.sort(reverse=True); lighter = br[0][1] if br else "1.mp4"
    print("brightness:", br, "=> lighter:", lighter)

    clips = [("fistula", SAMPLES / "fistula nystagmus.mp4"),
             ("ref2", SAMPLES / "2.mp4"),
             ("lighter", SAMPLES / lighter)]
    report = []
    for tag, clip in clips:
        if not clip.exists():
            continue
        ears, ok_eye = scan_ear(clip, det)
        base = float(np.nanmedian(ears)) if any(not math.isnan(e) for e in ears) else float("nan")
        s, e = longest_clean(ears, ok_eye, base)
        clean_len = e - s
        n_blink = sum(1 for x in ears if not math.isnan(x) and x < 0.6 * base)
        blink_k = next((i for i, x in enumerate(ears) if not math.isnan(x) and x < 0.6 * base), None)
        print(f"\n=== {tag} {clip.name} === detected={sum(ok_eye)}/{len(ok_eye)}  ear_base={base:.3f}  "
              f"longest_clean_run={clean_len} @frame {s}  blink_frames={n_blink}")
        win = track_window(clip, det, det2, s, NEED, tag) if clean_len >= 10 else None
        rec = blink_recovery(clip, det, det2, blink_k, tag) if blink_k is not None else None
        m = {"clip": clip.name, "tag": tag, "eye_detected_frames": f"{sum(ok_eye)}/{len(ok_eye)}",
             "ear_baseline": round(base, 3), "longest_blinkfree_run_frames": clean_len,
             "clean_window_start": s, "blink_count_in_scan": n_blink,
             "clean_window": win, "blink_recovery": rec}
        report.append(m); print(json.dumps({k: v for k, v in m.items() if k != "clean_window"}, indent=2))
        if win:
            print("  clean-window:", json.dumps({k: v for k, v in win.items() if k != "images"}))
    (OUT / "report2.json").write_text(json.dumps(report, indent=2))
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
