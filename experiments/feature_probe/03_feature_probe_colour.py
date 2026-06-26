"""READ-ONLY probe v3 — iris-appearance / colour generalization analysis.

For EVERY sample clip: measure iris brightness, contrast, texture energy (the mechanism
that actually drives corner yield), feature count + density, and short-window survival.
This characterises feature yield vs iris APPEARANCE across the available gamut so we can
reason about generalization to untested iris colours. Touches no pipeline code.
"""
import sys, math, json
from pathlib import Path
import cv2
import numpy as np

APP = Path(__file__).resolve().parents[2]  # repo root (portable across machines)
sys.path.insert(0, str(APP))
from src.core.iris_tracking import IrisDetector

OUT = Path(__file__).resolve().parent / "probe_out3"; OUT.mkdir(exist_ok=True)
SAMPLES = APP / "samples"
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def eyes(frame, det):
    ok, le, re = det.detect(frame)
    return [e for e in (le, re) if e.detected] if ok else []


def circ(e):
    return float(e.single_x), float(e.single_y), float(e.iris_radius or e.radius or 20.0)


def feats(gray, cx, cy, r):
    m = np.zeros(gray.shape, np.uint8); cv2.circle(m, (int(cx), int(cy)), int(0.95 * r), 255, -1)
    p = cv2.goodFeaturesToTrack(gray, 120, 0.01, 3, blockSize=7, mask=m)
    return p.reshape(-1, 2).astype(np.float32) if p is not None else np.empty((0, 2), np.float32)


def iris_stats(gray, cx, cy, r):
    """brightness (mean), contrast (std), texture energy (var of Laplacian) inside the iris disc."""
    m = np.zeros(gray.shape, np.uint8); cv2.circle(m, (int(cx), int(cy)), int(0.9 * r), 255, -1)
    vals = gray[m > 0]
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    texture = float(lap[m > 0].var()) if vals.size else 0.0
    return (round(float(vals.mean()), 1), round(float(vals.std()), 1), round(texture, 1)) if vals.size else (0, 0, 0)


def first_open_frame(clip, det, scan=70):
    cap = cv2.VideoCapture(str(clip)); best = None; f = 0
    while f < scan:
        ok, fr = cap.read()
        if not ok:
            break
        es = eyes(fr, det)
        if es:
            ear = max([e.ear for e in es if e.ear is not None], default=0.0)
            if best is None or ear > best[0]:
                best = (ear, f, fr.copy(), es)   # keep the detections to avoid stateful re-detect
        f += 1
    cap.release(); return best  # (ear, frame_idx, frame, eyes) or None


def probe_clip(clip, det, det2):
    op = first_open_frame(clip, det)
    if op is None:
        return {"clip": clip.name, "error": "no eye detected in first 70 frames"}
    ear0, fidx, f0, es = op
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    cx0, cy0, r0 = max((circ(e) for e in es), key=lambda q: len(feats(g0, *q)))
    bright, contrast, texture = iris_stats(g0, cx0, cy0, r0)
    p0 = feats(g0, cx0, cy0, r0); n0 = len(p0)
    area = math.pi * r0 * r0
    density = round(1000.0 * n0 / area, 2) if area else 0.0
    # short track (15 frames) from this open frame
    cap = cv2.VideoCapture(str(clip)); cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    cap.read()  # the op frame
    pts = p0.copy(); alive = np.ones(n0, bool); prev = g0; fbs = []; s10 = s15 = None
    for k in range(1, 16):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        p1, stf, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, **LK)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **LK)
        fb = np.linalg.norm(pts - p0b, axis=1)
        alive = alive & (stf.ravel() == 1) & (stb.ravel() == 1) & (fb < 1.0)
        pts = np.where(alive[:, None], p1, pts)
        if alive.sum():
            fbs.append(float(fb[alive].mean()))
        if k == 10:
            s10 = int(alive.sum())
        if k == 15:
            s15 = int(alive.sum())
        prev = g
    cap.release()
    # overlay
    vis = f0.copy()
    for p in p0:
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 220, 0), -1)
    cv2.circle(vis, (int(cx0), int(cy0)), int(r0), (0, 220, 220), 1)
    half = int(2.3 * r0)
    sub = vis[max(0, int(cy0 - half)):int(cy0 + half), max(0, int(cx0 - half * 1.5)):int(cx0 + half * 1.5)]
    if sub.size:
        sub = cv2.resize(sub, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        cv2.putText(sub, f"{clip.stem[:20]}  bright {bright} feat {n0}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imwrite(str(OUT / f"iris_{clip.stem[:18].replace(' ','_')}.png"), sub)
    return {"clip": clip.name, "brightness": bright, "contrast_std": contrast, "texture_varLap": texture,
            "iris_radius_px": round(r0, 1), "init_features": n0, "feature_density_per1000px2": density,
            "surv10": s10, "surv15": s15, "mean_fb_err_px": round(float(np.mean(fbs)), 3) if fbs else None}


def main():
    det, det2 = IrisDetector(), IrisDetector()
    clips = sorted(SAMPLES.glob("*.mp4"))
    rows = []
    for c in clips:
        r = probe_clip(c, det, det2)
        rows.append(r)
        print(json.dumps(r))
    rows_ok = [r for r in rows if "error" not in r]
    rows_ok.sort(key=lambda r: r["brightness"])
    print("\n==== iris appearance vs feature yield (sorted by brightness) ====")
    hdr = ["clip", "bright", "contrast", "texture", "feat", "density", "surv10", "fb_err"]
    print("{:<34}{:>7}{:>9}{:>9}{:>6}{:>9}{:>8}{:>8}".format(*hdr))
    for r in rows_ok:
        print("{:<34}{:>7}{:>9}{:>9}{:>6}{:>9}{:>8}{:>8}".format(
            r["clip"][:33], r["brightness"], r["contrast_std"], r["texture_varLap"],
            r["init_features"], r["feature_density_per1000px2"], str(r["surv10"]), str(r["mean_fb_err_px"])))
    # correlations: does feature count track texture/contrast more than brightness?
    import numpy as _np
    b = _np.array([r["brightness"] for r in rows_ok], float)
    ct = _np.array([r["contrast_std"] for r in rows_ok], float)
    tx = _np.array([r["texture_varLap"] for r in rows_ok], float)
    fc = _np.array([r["init_features"] for r in rows_ok], float)
    def corr(a, c):
        return round(float(_np.corrcoef(a, c)[0, 1]), 2) if len(a) > 2 and a.std() and c.std() else None
    print("\ncorrelation feature_count vs brightness:", corr(b, fc),
          " vs contrast:", corr(ct, fc), " vs texture:", corr(tx, fc))
    (OUT / "report3.json").write_text(json.dumps(rows, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
