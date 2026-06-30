"""RIT Orbit Lock PROBE runner (read-only; no validator/clinical wiring).

Drives src/core/rit_orbit_lock.OrbitLock over a clip, starting from the clinician-approved Stage-0
eye-opening contour, and renders a diagnostic overlay so a clinician can judge whether the moving
orbit stays locked to the real eye opening (vs the frozen static Stage-0 contour).

Outputs to outputs/<clip>_tracked/_rit_orbit_lock_probe/:
  orbit_lock_overlay.mp4, orbit_lock_points.csv, orbit_lock_summary.json, representative_frames/

Usage: python tools/rit_orbit_lock_probe.py [out_dir] [n_frames]
"""
import csv, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.rit_orbit_lock import OrbitLock

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 200
REV = OUT / "_rit_orbit_lock_probe"
(REV / "representative_frames").mkdir(parents=True, exist_ok=True)

appr = json.load(open(OUT / "approved_landmarks.json", encoding="utf-8"))
video = appr.get("video")
if not video or not Path(video).exists():
    stem = OUT.name[:-8] if OUT.name.endswith("_tracked") else OUT.name
    video = str(Path("samples") / (stem + ".mp4"))

contours0, iris0 = {}, {}
for ek in ("L", "R"):
    e = (appr.get("eye_opening_contours") or {}).get(ek)
    pts = e.get("points") if isinstance(e, dict) else e
    if pts:
        contours0[ek] = np.array([[float(p["x"]), float(p["y"])] for p in pts], np.float32)
    iv = (appr.get("iris") or {}).get(ek) or {}
    iris0[ek] = (float(iv.get("x", 0)), float(iv.get("y", 0)), float(iv.get("radius", 30)))

trk = {int(r["frame_number"]): r for r in csv.DictReader(open(OUT / "tracking.csv", encoding="utf-8"))}

def num(r, k):
    v = r.get(k) if r else None
    return None if v in (None, "", "None") else float(v)

# map csv side -> contour key by proximity of median raw candidate to Stage-0 iris
sidemap = {}
for side in ("left", "right"):
    xs = [num(trk[f], f"raw_{side}_iris_center_x") for f in trk]
    xs = [v for v in xs if v is not None]
    if xs and iris0:
        mx = float(np.median(xs))
        sidemap[side] = min(iris0, key=lambda k: abs(iris0[k][0] - mx))
    else:
        sidemap[side] = "L" if side == "left" else "R"
ek_to_side = {v: k for k, v in sidemap.items()}

CYAN = (255, 255, 0); AMBER = (0, 165, 255); WHITE = (255, 255, 255); RED = (0, 0, 255); GREEN = (0, 200, 0)
STAT_COL = {"orbit_locked": GREEN, "orbit_uncertain": AMBER, "orbit_lost": RED}
REP_FRAMES = {"R": [29, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], "L": [8]}

cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
ds = min(1.0, 1280.0 / max(1, W))
ow, oh = int(W * ds), int(H * ds)
writer = cv2.VideoWriter(str(REV / "orbit_lock_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

locks = {ek: OrbitLock(contours0[ek], iris0[ek][2]) for ek in contours0}
counts = {ek: {"orbit_locked": 0, "orbit_uncertain": 0, "orbit_lost": 0} for ek in locks}
rep = {ek: [] for ek in locks}
rows = []
prev_gray = None
pj = lambda c: [[round(float(x), 1), round(float(y), 1)] for x, y in c]

fn = 0
while fn < NF:
    ok, frame = cap.read()           # sequential read (no per-frame seek) — much faster
    if not ok:
        break
    fn += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    disp = cv2.resize(frame, (ow, oh))
    cv2.putText(disp, f"frame {fn}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(disp, f"frame {fn}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1)
    for ek, lock in locks.items():
        side = ek_to_side.get(ek, "left")
        t = trk.get(fn)
        hint = (num(t, f"raw_{side}_iris_center_x"), num(t, f"raw_{side}_iris_center_y")) if t else (None, None)
        if prev_gray is None:
            res = type("R", (), dict(status="orbit_locked", confidence=1.0, n_inliers=0,
                                     scale=1.0, rotation_deg=0.0, contour=lock.contour.copy()))()
        else:
            res = lock.step(prev_gray, gray, iris_hint=hint)
        counts[ek][res.status] += 1
        # overlay (downscaled) — moving orbit + iris only (static Stage-0 oval intentionally NOT drawn)
        mov = (res.contour * ds).astype(np.int32)
        cv2.polylines(disp, [mov], True, STAT_COL[res.status], 2)
        if hint[0] is not None:
            cv2.circle(disp, (int(hint[0] * ds), int(hint[1] * ds)),
                       max(2, int(iris0[ek][2] * ds)), WHITE, 1)
        mcx, mcy = res.contour[:, 0].mean(), res.contour[:, 1].mean()   # no per-eye label drawn
        rows.append(dict(frame=fn, eye=ek, status=res.status, confidence=round(res.confidence, 3),
                         n_inliers=res.n_inliers, scale=round(res.scale, 4),
                         rotation_deg=round(res.rotation_deg, 2),
                         centroid_x=round(float(mcx), 1), centroid_y=round(float(mcy), 1),
                         contour_json=json.dumps(pj(res.contour))))
        # representative full-res crop
        if fn in REP_FRAMES.get(ek, []):
            allc = np.vstack([contours0[ek], res.contour])
            x0, y0 = allc.min(0); x1, y1 = allc.max(0)
            cxm, cym = (x0 + x1) / 2, (y0 + y1) / 2
            half = max(x1 - x0, y1 - y0) / 2 + 1.5 * iris0[ek][2] + 25
            sp = int(round(2 * half)); X0 = cxm - half; Y0 = cym - half
            sq = np.zeros((sp, sp, 3), np.uint8)
            fx0, fy0 = int(round(X0)), int(round(Y0))
            sx0, sy0 = max(0, -fx0), max(0, -fy0); cfx0, cfy0 = max(0, fx0), max(0, fy0)
            cw = min(sp - sx0, W - cfx0); chh = min(sp - sy0, H - cfy0)
            if cw > 0 and chh > 0:
                sq[sy0:sy0 + chh, sx0:sx0 + cw] = frame[cfy0:cfy0 + chh, cfx0:cfx0 + cw]
            TW = 520; s = TW / sp; d2 = cv2.resize(sq, (TW, TW))
            T = lambda c: (np.asarray(c, np.float32) - [X0, Y0]) * s
            cv2.polylines(d2, [T(res.contour).astype(np.int32)], True, STAT_COL[res.status], 2)
            if hint[0] is not None:
                hc = T([hint]).astype(np.int32)[0]
                cv2.circle(d2, tuple(hc), max(3, int(iris0[ek][2] * s)), WHITE, 2)
            for txt2, yy, col in ((f"f{fn} {ek}  RIT-orbit-lock", 22, WHITE),
                                  (f"{res.status}  conf={res.confidence:.2f} inl={res.n_inliers}", 44, STAT_COL[res.status]),
                                  ("green=moving orbit   white=iris cand", TW - 12, WHITE)):
                cv2.putText(d2, txt2, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(d2, txt2, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            pth = REV / "representative_frames" / f"f{fn:04d}_{ek}.png"
            cv2.imwrite(str(pth), d2)
            rep[ek].append({"frame": fn, "status": res.status, "confidence": round(res.confidence, 2),
                            "png": pth.name})
    writer.write(disp)
    prev_gray = gray
writer.release(); cap.release()

with open(REV / "orbit_lock_points.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["frame", "eye", "status", "confidence", "n_inliers", "scale",
                                      "rotation_deg", "centroid_x", "centroid_y", "contour_json"])
    w.writeheader(); w.writerows(rows)

summary = {"video": video, "frames": NF, "side_mapping_csv_to_contour": sidemap,
           "method": "KLT optical flow (eye-region features, iris masked) + RANSAC similarity transform",
           "per_eye_status_counts": counts, "representative_frames": rep}
json.dump(summary, open(REV / "orbit_lock_summary.json", "w", encoding="utf-8"), indent=2)

print("side mapping:", sidemap)
for ek in locks:
    c = counts[ek]; tot = sum(c.values())
    print(f"{ek}: locked {c['orbit_locked']}  uncertain {c['orbit_uncertain']}  lost {c['orbit_lost']}  (/{tot})")
print("R frames 29-43:", [(r["frame"], r["status"], r["confidence"]) for r in rep.get("R", [])])
print("L frame 8:", rep.get("L", []))
print("overlay:", REV / "orbit_lock_overlay.mp4")
