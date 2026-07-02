"""Turn the clinician's 3-point orbit-lock marks (tools/rit_orbit_lock_marker.html -> orbit_lock.json)
into a per-video RIT Orbit Lock: an oval contour per eye, written as orbit_lock_points.csv for every
frame (static orbit marked on the first frame; the eyelid opening is ~stable while the iris moves inside).

Each eye is 3 clicks: medial canthus, lateral canthus, top. The canthi line is HORIZONTAL; the vertical
(right-angle) semi-axis is the perpendicular height of the top; the bottom is its mirror. The oval is the
ellipse centred at the canthi midpoint, sampled to a polygon. Horizontal/vertical axes are saved too.

No MediaPipe, no Stage-0. Usage: py -3.14 tools/rit_orbit_lock_from_marks.py [orbit_lock.json]
"""
import csv, json, math, sys
from pathlib import Path

MARKS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "orbit_lock.json"
ROOT = Path(__file__).resolve().parent.parent
VIDINFO = json.load(open(ROOT / "outputs/_orbit_lock_marking/videos.json", encoding="utf-8"))
N_CONTOUR = 48

marks = json.load(open(MARKS, encoding="utf-8"))
# keep a copy in the repo next to the frames
json.dump(marks, open(ROOT / "outputs/_orbit_lock_marking/orbit_lock.json", "w"), indent=1)


def oval_params(m):
    (ix, iy), (ox, oy), (tx, ty) = m["inner"], m["outer"], m["top"]
    cx, cy = (ix + ox) / 2, (iy + oy) / 2
    dx, dy = ox - ix, oy - iy
    a = math.hypot(dx, dy) / 2
    theta = math.atan2(dy, dx)                       # horizontal (canthi) direction
    px, py = -math.sin(theta), math.cos(theta)       # perpendicular (vertical) unit
    d = (tx - cx) * px + (ty - cy) * py              # signed perpendicular height of the top
    b = abs(d) or a * 0.5
    return cx, cy, a, b, theta


def contour(cx, cy, a, b, theta, n=N_CONTOUR):
    ct, st = math.cos(theta), math.sin(theta)
    pts = []
    for i in range(n):
        ph = 2 * math.pi * i / n
        ex, ey = a * math.cos(ph), b * math.sin(ph)
        pts.append([round(cx + ex * ct - ey * st, 1), round(cy + ex * st + ey * ct, 1)])
    return pts


done = []
for name, eyes in marks.items():
    stem = name[:-4] if name.lower().endswith(".png") else name
    if stem not in VIDINFO:
        print(f"skip {name}: no video info"); continue
    nfr = VIDINFO[stem]["frames"]
    out = ROOT / "outputs" / f"{stem}_tracked"
    rev = out / "_rit_orbit_lock_probe"; rev.mkdir(parents=True, exist_ok=True)
    conts, axes = {}, {}
    for ek in ("R", "L"):
        if ek not in eyes:
            continue
        cx, cy, a, b, theta = oval_params(eyes[ek])
        conts[ek] = contour(cx, cy, a, b, theta)
        axes[ek] = dict(center=[round(cx, 1), round(cy, 1)], horizontal_deg=round(math.degrees(theta), 2),
                        half_width=round(a, 1), half_height=round(b, 1),
                        medial_canthus=eyes[ek]["inner"], lateral_canthus=eyes[ek]["outer"], top=eyes[ek]["top"])
    with open(rev / "orbit_lock_points.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["frame", "eye", "status", "confidence", "n_inliers", "scale", "rotation_deg",
                    "centroid_x", "centroid_y", "contour_json"])
        for f in range(1, nfr + 1):
            for ek in conts:
                ax = axes[ek]
                w.writerow([f, ek, "orbit_locked", 1.0, 0, 1.0, ax["horizontal_deg"],
                            ax["center"][0], ax["center"][1], json.dumps(conts[ek])])
    json.dump(dict(source="clinician 3-point marks", static=True, frames=nfr, axes=axes),
              open(rev / "orbit_lock_axes.json", "w"), indent=1)
    done.append((stem, nfr, list(conts)))
    print(f"{stem:60} {nfr:5}f  eyes={list(conts)}  -> {rev/'orbit_lock_points.csv'}")

print(f"\nwrote orbit locks for {len(done)} videos.")
