"""RIT orbit-drift DIAGNOSTIC (read-only).

Renders a per-frame review showing, for each eye:
  - the STATIC Stage-0 orbit / eye-opening oval (cyan polygon),
  - the raw iris candidate the V1 tracker proposed (circle; green=RIT-accepted, red=rejected),
  - the CURRENT eye-opening landmarks from face_landmarks.csv (amber dots = where the eye is now),
  - text labels: RIT status + reject reason, ORBIT MODE, ORBIT STATUS (on-eye / likely stale).

It does NOT run the tracker or validator and changes no logic/thresholds. It only reads the existing
guarded tracking.csv + face_landmarks.csv + approved_landmarks.json and visualises the orbit reference,
so a clinician can see whether the iris is correct but the orbit is stale.

Output: outputs/<clip>_tracked/_rit_orbit_drift_review/{f####_<eye>.png, summary.csv}
"""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
REV = OUT / "_rit_orbit_drift_review"
REV.mkdir(exist_ok=True)

appr = json.load(open(OUT / "approved_landmarks.json", encoding="utf-8"))
video = appr.get("video") or json.load(open(OUT / "metadata.json", encoding="utf-8")).get("video")
if not Path(video).exists():
    # fall back to samples/<stem without _tracked>.mp4
    stem = OUT.name[:-8] if OUT.name.endswith("_tracked") else OUT.name
    video = str(Path("samples") / (stem + ".mp4"))

orbit = {}
for ek in ("L", "R"):
    e = (appr.get("eye_opening_contours") or {}).get(ek)
    pts = e.get("points") if isinstance(e, dict) else e
    orbit[ek] = (np.array([[float(p["x"]), float(p["y"])] for p in pts], np.float32).reshape(-1, 1, 2)
                 if pts else None)
iris0 = {ek: (float(v["x"]), float(v["y"]), float(v["radius"]))
         for ek, v in (appr.get("iris") or {}).items() if v}

trk = {int(r["frame_number"]): r for r in csv.DictReader(open(OUT / "tracking.csv", encoding="utf-8"))}
fac = {int(r["frame_number"]): r for r in csv.DictReader(open(OUT / "face_landmarks.csv", encoding="utf-8"))}


def num(r, k):
    v = r.get(k) if r else None
    return None if v in (None, "", "None") else float(v)


def fg(r, n):
    if not r:
        return None
    x = r.get(n + "_x")
    return None if x in (None, "", "None") else (float(x), float(r[n + "_y"]))


# map CSV side (left/right) -> Stage-0 orbit key (L/R) by proximity of the median raw candidate
sidemap = {}
for side in ("left", "right"):
    xs = [num(trk[f], f"raw_{side}_iris_center_x") for f in trk]
    ys = [num(trk[f], f"raw_{side}_iris_center_y") for f in trk]
    xs = [v for v in xs if v is not None]; ys = [v for v in ys if v is not None]
    if not xs:
        sidemap[side] = "L" if side == "left" else "R"; continue
    mx, my = float(np.median(xs)), float(np.median(ys))
    sidemap[side] = min(iris0, key=lambda k: (iris0[k][0] - mx) ** 2 + (iris0[k][1] - my) ** 2) if iris0 else (
        "L" if side == "left" else "R")


def poly_centroid(poly):
    p = poly.reshape(-1, 2)
    return float(p[:, 0].mean()), float(p[:, 1].mean())


def aperture_centroid(fr, ek):
    pts = [fg(fr, n + "_" + ek) for n in ("inner_canthus", "outer_canthus", "upper_margin", "lower_margin")]
    pts = [p for p in pts if p]
    if len(pts) < 2:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def proj(p, a, b):
    if None in (p, a, b):
        return None
    ax, ay = b[0] - a[0], b[1] - a[1]; L2 = ax * ax + ay * ay
    return None if L2 < 1 else ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L2


CYAN = (255, 255, 0); GREEN = (0, 200, 0); RED = (0, 0, 255); AMBER = (0, 165, 255); WHITE = (255, 255, 255)
cap = cv2.VideoCapture(video)
rows_out = []
saved = 0
best = []   # (score, frame, eye, png) for ranking the clearest stale-orbit cases

for fn in range(1, NFRAMES + 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn - 1)
    ok, frame = cap.read()
    if not ok:
        break
    t = trk.get(fn); fr = fac.get(fn)
    if not t:
        continue
    for side in ("left", "right"):
        ek = sidemap[side]
        cand = (num(t, f"raw_{side}_iris_center_x"), num(t, f"raw_{side}_iris_center_y"))
        rad = num(t, f"{side}_iris_radius") or (iris0.get(ek, (0, 0, 30))[2])
        valid = str(t.get(f"iris_valid_{side}")).strip().lower() in ("true", "1")
        reason = t.get(f"drift_reason_{side}", "") or ""
        op = orbit.get(ek)
        # ORBIT STATUS: compare static orbit centroid to the CURRENT eye-opening centroid
        orbit_mode = "static_stage0"
        ac = aperture_centroid(fr, ek)
        orbit_status = "unknown"
        if op is not None and ac is not None:
            ocx, ocy = poly_centroid(op)
            orad = float(np.sqrt(cv2.contourArea(op) / np.pi)) or 1.0
            off = float(np.hypot(ocx - ac[0], ocy - ac[1]))
            orbit_status = "likely_stale" if off > 0.5 * orad else "on_eye"
        # candidate eye-local (is it inside the current opening? -> "good iris" character)
        inner, outer = fg(fr, "inner_canthus_" + ek), fg(fr, "outer_canthus_" + ek)
        up, lo = fg(fr, "upper_margin_" + ek), fg(fr, "lower_margin_" + ek)
        elx = proj(cand, inner, outer) if cand[0] is not None else None
        ely = proj(cand, up, lo) if cand[0] is not None else None
        inside_opening = (elx is not None and ely is not None and -0.25 <= elx <= 1.25 and -0.25 <= ely <= 1.25)

        is_outside_orbit = "outside_orbit" in reason
        png_path = ""
        if is_outside_orbit and cand[0] is not None:
            # crop around the union of orbit + candidate
            xs = [cand[0]]; ys = [cand[1]]
            if op is not None:
                xs += list(op.reshape(-1, 2)[:, 0]); ys += list(op.reshape(-1, 2)[:, 1])
            if ac is not None:
                xs.append(ac[0]); ys.append(ac[1])
            cx0, cy0 = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 2.2 * rad + 30
            side_px = int(round(2 * half)); H, W = frame.shape[:2]
            x0, y0 = cx0 - half, cy0 - half
            sq = np.zeros((side_px, side_px, 3), np.uint8)
            fx0, fy0 = int(round(x0)), int(round(y0))
            sx0, sy0 = max(0, -fx0), max(0, -fy0); cfx0, cfy0 = max(0, fx0), max(0, fy0)
            cw = min(side_px - sx0, W - cfx0); chh = min(side_px - sy0, H - cfy0)
            if cw > 0 and chh > 0:
                sq[sy0:sy0 + chh, sx0:sx0 + cw] = frame[cfy0:cfy0 + chh, cfx0:cfx0 + cw]
            TW = 560; s = TW / side_px; disp = cv2.resize(sq, (TW, TW))
            def TX(p):
                return (int((p[0] - x0) * s), int((p[1] - y0) * s))
            if op is not None:
                cv2.polylines(disp, [np.array([TX(p) for p in op.reshape(-1, 2)], np.int32)], True, CYAN, 2)
                cv2.putText(disp, "Stage-0 orbit (static)", (8, TW - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
            for p, nm in ((inner, "Ic"), (outer, "Oc"), (up, "Up"), (lo, "Lo")):
                if p:
                    cv2.drawMarker(disp, TX(p), AMBER, cv2.MARKER_CROSS, 12, 2)
            cv2.putText(disp, "amber = current eye opening", (8, TW - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1)
            ccol = GREEN if valid else RED
            cv2.circle(disp, TX(cand), max(4, int(rad * s)), ccol, 2)
            cv2.circle(disp, TX(cand), 3, ccol, -1)
            # labels
            def lab(txt, y, col):
                cv2.putText(disp, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(disp, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
            lab(f"f{fn}  {ek} eye   r={rad:.0f}", 22, WHITE)
            lab(f"RIT: {'ACCEPTED' if valid else 'REJECTED'}  {reason}", 44, ccol)
            lab(f"ORBIT MODE: static Stage-0", 66, CYAN)
            lab(f"ORBIT STATUS: {orbit_status}", 88, (0, 0, 255) if orbit_status == 'likely_stale' else GREEN)
            lab(f"cand inside current opening: {'YES' if inside_opening else 'no'}"
                + (f"  (eye-local {elx:.2f},{ely:.2f})" if elx is not None else ""), 110,
                GREEN if inside_opening else AMBER)
            cv2.rectangle(disp, (0, TW - 30), (TW, TW), (30, 30, 30), -1)
            cv2.putText(disp, "CHECK: is iris correct but orbit stale?", (8, TW - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            png = REV / f"f{fn:04d}_{ek}.png"
            cv2.imwrite(str(png), disp); png_path = png.name; saved += 1
            # "best" = clearest stale-orbit: candidate inside opening + plausible radius + orbit stale
            r0 = iris0.get(ek, (0, 0, rad))[2]
            rscore = 1.0 - abs(rad / r0 - 1.0) if r0 else 0.0
            score = (2.0 if inside_opening else 0.0) + (1.0 if orbit_status == "likely_stale" else 0.0) + rscore
            best.append((score, fn, ek, png_path))
        rows_out.append(dict(frame=fn, eye=ek,
                             raw_iris_x="" if cand[0] is None else round(cand[0], 1),
                             raw_iris_y="" if cand[1] is None else round(cand[1], 1),
                             radius=round(rad, 1), rit_valid=valid, rit_reject_reason=reason,
                             orbit_mode=orbit_mode, orbit_status=orbit_status, png_path=png_path))
cap.release()

with open(REV / "summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["frame", "eye", "raw_iris_x", "raw_iris_y", "radius",
                                      "rit_valid", "rit_reject_reason", "orbit_mode", "orbit_status", "png_path"])
    w.writeheader(); w.writerows(rows_out)

# report stats
oo = [r for r in rows_out if "outside_orbit" in (r["rit_reject_reason"] or "")]
stale = sum(1 for r in oo if r["orbit_status"] == "likely_stale")
print(f"side mapping (csv->Stage0): {sidemap}")
print(f"frames scanned: 1..{NFRAMES}   PNGs saved: {saved}")
print(f"outside_orbit (frame,eye) pairs in first {NFRAMES}: {len(oo)}  | orbit likely_stale: {stale}")
print("\nTOP 10 clearest stale-orbit PNGs (candidate inside opening, plausible radius, orbit stale):")
for sc, fn, ek, png in sorted(best, reverse=True)[:10]:
    print(f"   score {sc:.2f}  f{fn:04d} {ek}  -> {png}")
