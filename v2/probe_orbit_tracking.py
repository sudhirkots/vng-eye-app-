"""V2 'virtual spectacles' orbit-tracking probe.

Pipeline:
    1. Load the video.
    2. Run MediaPipe (Layer 1) on the first N=200 frames as a Stage-0 suggestion only;
       MediaPipe is closed immediately after.
    3. Open the Stage-0 marker (4 points per eye -> 8 anchors total).
    4. Apply the rigid 'spectacles' tracker (KLT + Umeyama similarity fit) per frame.
    5. Write:
            outputs/v2_orbit_tracking_probe/orbit_stage0_v2.json
            outputs/v2_orbit_tracking_probe/orbit_tracking_overlay.mp4
            outputs/v2_orbit_tracking_probe/orbit_tracking_points.csv
            outputs/v2_orbit_tracking_probe/orbit_tracking_summary.json

Does NOT detect the iris, the limbus, or nystagmus. The clip is loaded, the spectacles
are drawn moving with the face, and a CSV / summary records how often they were
ok / uncertain / lost.

Usage:
    .venv\\Scripts\\python.exe v2\\probe_orbit_tracking.py --video PATH

    --video PATH        path to the input video (required)
    --output-dir DIR    default: outputs/v2_orbit_tracking_probe
    --init-frame N      force Stage 0 on a specific frame (otherwise picks the first
                        MediaPipe-detected frame in the first --mp-scan-frames, or
                        falls back to frame 1)
    --mp-scan-frames N  cap on MediaPipe scan window (default 200)
    --skip-stage0       if a stage0 JSON already exists in the output dir, reuse it
                        without showing the marker (useful to iterate on the tracker
                        without redoing Stage 0)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v2.core.face_locator import FaceLocator                       # noqa: E402
from v2.core.orbit_stage0 import (                                  # noqa: E402
    POINTS, run_orbit_stage0, save_stage0_json,
)
from v2.core.spectacles_tracker import SpectaclesTracker, FrameResult  # noqa: E402


# --------------------------------------------------------------------------- drawing
COL_STAGE0_OVAL = (140, 140, 140)            # dim grey -- the static Stage-0 oval
COL_LIVE_OK = (255, 0, 255)                   # bright magenta -- tracked, ok
COL_LIVE_UNC = (0, 200, 255)                  # amber          -- uncertain
COL_LIVE_LOST = (0, 0, 255)                   # red            -- lost
COL_AXIS_H = (0, 255, 0)                      # green   medial<->lateral
COL_AXIS_V = (255, 100, 0)                    # blue    lower<->upper
COL_KLT_PT = (180, 255, 180)                  # pale green raw-KLT dot
COL_HUD_BG = (0, 0, 0)
COL_HUD_FG = (255, 255, 255)

# Used to pick the more-pessimistic per-eye status for the overall frame counter.
STATUS_RANK = {
    "ok_sclera_iris_locked": 0,
    "corrected_by_sclera_iris": 1,
    "provisional_iris_only": 2,
    "provisional_sclera_only": 2,
    "occluded_blink_hold": 3,
    "uncertain_no_sclera_iris_pair": 4,
    "lost": 5,
}


def _draw_oval_through_four(view, pts, scale, colour, thickness):
    """Approximate ellipse through 4 named anchor points and draw it. pts is the
    {medial, lateral, upper, lower} dict in source-video coords."""
    M = np.array(pts["medial"], float)
    L = np.array(pts["lateral"], float)
    U = np.array(pts["upper"], float)
    D = np.array(pts["lower"], float)
    cx = (M[0] + L[0] + U[0] + D[0]) / 4.0
    cy = (M[1] + L[1] + U[1] + D[1]) / 4.0
    horiz = L - M
    vert = D - U
    a = float(np.linalg.norm(horiz) / 2.0)
    b = float(np.linalg.norm(vert) / 2.0)
    theta = float(np.degrees(np.arctan2(horiz[1], horiz[0])))
    cv2.ellipse(view,
                (int(round(cx * scale)), int(round(cy * scale))),
                (max(2, int(round(a * scale))), max(2, int(round(b * scale)))),
                theta, 0, 360, colour, thickness, cv2.LINE_AA)
    return (cx, cy)


def _draw_axis(view, p_from, p_to, scale, colour):
    cv2.line(view,
             (int(round(p_from[0] * scale)), int(round(p_from[1] * scale))),
             (int(round(p_to[0] * scale)), int(round(p_to[1] * scale))),
             colour, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="V2 virtual-spectacles orbit-tracking probe.")
    ap.add_argument("--video", required=True, help="path to the input video")
    ap.add_argument("--output-dir", default="outputs/v2_orbit_tracking_probe")
    ap.add_argument("--init-frame", type=int, default=0,
                    help="force Stage 0 on this 1-indexed frame; 0 = pick automatically")
    ap.add_argument("--mp-scan-frames", type=int, default=200)
    ap.add_argument("--skip-stage0", action="store_true",
                    help="if an existing stage0 JSON is in the output dir, reuse it")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage0_path = out_dir / "orbit_stage0_v2.json"
    overlay_path = out_dir / "orbit_tracking_overlay.mp4"
    csv_path = out_dir / "orbit_tracking_points.csv"
    summary_path = out_dir / "orbit_tracking_summary.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[v2-orbit] video={video_path.name}  {W}x{H} @ {fps:.2f} fps  total={total_frames}")

    # ---- Stage 0 (load existing or run interactive) -------------------------
    stage0: Optional[Dict] = None
    if args.skip_stage0 and stage0_path.exists():
        try:
            stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
            print(f"[v2-orbit] reusing existing Stage-0 JSON: {stage0_path}")
        except Exception as e:
            print(f"[v2-orbit] failed to load {stage0_path}: {e}")
            stage0 = None

    if stage0 is None:
        # Decide the Stage-0 frame.
        chosen_frame = max(1, int(args.init_frame)) if args.init_frame else 0
        suggested_zones = None
        detected_at = None
        scan_window = 0
        if chosen_frame == 0:
            print(f"[v2-orbit] scanning first {args.mp_scan_frames} frame(s) with "
                  f"MediaPipe for a Stage-0 suggestion...")
            locator = FaceLocator()
            scan_window = int(min(args.mp_scan_frames, total_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for fr in range(1, scan_window + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                loc = locator.locate(frame)
                if loc.mp_detected:
                    chosen_frame = fr
                    suggested_zones = {"L": loc.L_eye_bbox, "R": loc.R_eye_bbox}
                    detected_at = fr
                    print(f"[v2-orbit] MediaPipe detected face at fr={fr}; "
                          f"using as Stage-0 frame.")
                    break
            locator.close()
            if chosen_frame == 0:
                chosen_frame = 1
                print(f"[v2-orbit] MediaPipe found no face in the first {scan_window} "
                      f"frame(s); falling back to frame 1 (no hint zones).")
        else:
            print(f"[v2-orbit] using --init-frame {chosen_frame} (skipped MediaPipe scan).")

        # Grab the Stage-0 frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frame - 1)
        ok, init_frame = cap.read()
        if not ok or init_frame is None:
            raise SystemExit(f"Could not read frame {chosen_frame}.")
        # Run the interactive marker.
        stage0 = run_orbit_stage0(
            frame_bgr=init_frame,
            init_frame_number=chosen_frame,
            video_path=video_path,
            resolution=(W, H),
            fps=fps,
            suggested_eye_zones=suggested_zones,
            detected_at_frame=detected_at,
            scan_window_frames=scan_window,
        )
        save_stage0_json(stage0, stage0_path)

    init_frame_no = int(stage0.get("init_frame_number", 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_no - 1)
    ok, init_frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not re-read Stage-0 frame {init_frame_no}.")
    init_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)

    # ---- Tracker + outputs ---------------------------------------------------
    tracker = SpectaclesTracker(stage0, init_gray, fps=fps)
    writer = cv2.VideoWriter(str(overlay_path),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open overlay writer: {overlay_path}")

    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    cw = csv.writer(csv_f)
    cw.writerow([
        "frame_number", "eye",
        "medial_x", "medial_y", "medial_conf",
        "lateral_x", "lateral_y", "lateral_conf",
        "upper_x", "upper_y", "upper_conf",
        "lower_x", "lower_y", "lower_conf",
        "sclera_area", "sclera_inside_oval",
        "iris_support_present", "iris_support_inside_oval",
        "iris_support_curvature_score",
        "iris_support_radius", "iris_support_centre_x", "iris_support_centre_y",
        "sclera_iris_pair_score",
        "content_score", "geometry_score", "final_score",
        "blink_hold_active", "last_good_frame", "frames_since_last_good_content",
        "orbit_status", "status_reason",
    ])

    # Status -> overlay colour. Tracks the spec exactly.
    STATUS_COL = {
        "ok_sclera_iris_locked":         (255,   0, 255),    # bright magenta
        "corrected_by_sclera_iris":      (  0, 200, 255),    # amber
        "provisional_iris_only":         (255, 200,   0),    # cyan-ish
        "provisional_sclera_only":       (200, 200,   0),    # teal
        "occluded_blink_hold":           (200,  60, 180),    # purple
        "uncertain_no_sclera_iris_pair": (  0, 100, 255),    # orange-red
        "lost":                          (  0,   0, 255),    # red
    }
    COL_PREDICTED = (180, 180, 180)        # dim white -- rigid prediction (pre-correction)
    COL_SCLERA = (  0, 220, 220)            # yellow blob outline
    COL_IRIS_PTS = (255, 200,   0)          # cyan dots = iris-support boundary points
    COL_IRIS_FIT = (255, 255,   0)          # bright cyan = iris-support fitted circle

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    counts = {"L": {}, "R": {}}
    overall_status_counts: Dict[str, int] = {}

    # Stage-0 anchors per eye (for the dim-grey reference oval).
    s0_eyes: Dict[str, Dict[str, np.ndarray]] = {}
    for ek in ("L", "R"):
        block = stage0["eyes"][ek]["oval_points"]
        s0_eyes[ek] = {name: (float(block[name]["x"]), float(block[name]["y"]))
                       for name in POINTS}

    def _bump(d: Dict[str, int], k: str) -> None:
        d[k] = d.get(k, 0) + 1

    scale = 1.0  # we render at source size; the .mp4 is full resolution
    t_start = time.time()
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fr += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result: FrameResult = tracker.step(frame, gray, fr)

        # ---- draw -------------------------------------------------------
        vis = frame.copy()
        # 1. dim grey static Stage-0 ovals
        for ek in ("L", "R"):
            _draw_oval_through_four(vis, s0_eyes[ek], scale, COL_STAGE0_OVAL, 1)
        # 2. per-eye rendering
        for ek in ("L", "R"):
            per = result.per_eye[ek]
            col_live = STATUS_COL.get(per.status, (255, 255, 255))
            # 2a. rigid prediction oval (dim white, before correction)
            _draw_oval_through_four(vis, per.predicted, scale, COL_PREDICTED, 1)
            # 2b. corrected/displayed oval
            _draw_oval_through_four(vis, per.spectacles, scale, col_live, 2)
            # 2c. axes through the corrected oval
            _draw_axis(vis, per.spectacles["medial"], per.spectacles["lateral"],
                        scale, COL_AXIS_H)
            _draw_axis(vis, per.spectacles["lower"], per.spectacles["upper"],
                        scale, COL_AXIS_V)
            # 2d. raw KLT dots (debug)
            for name in POINTS:
                px, py = per.klt_points[name]
                cv2.circle(vis, (int(round(px)), int(round(py))),
                           2, COL_KLT_PT, -1, cv2.LINE_AA)
            # 2e. sclera centroid marker (yellow cross), if sclera present
            if per.sclera_present and per.sclera_centroid is not None:
                sx, sy = per.sclera_centroid
                cv2.drawMarker(vis, (int(round(sx)), int(round(sy))),
                                COL_SCLERA, cv2.MARKER_TILTED_CROSS, 14, 2,
                                cv2.LINE_AA)
            # 2f. provisional iris circle (if present), regardless of status
            if per.iris_support_present and per.iris_support_radius is not None \
                    and per.iris_support_centre is not None:
                cx, cy = per.iris_support_centre
                r = per.iris_support_radius
                cv2.circle(vis, (int(round(cx)), int(round(cy))),
                           max(2, int(round(r))), COL_IRIS_FIT, 1, cv2.LINE_AA)
                cv2.drawMarker(vis, (int(round(cx)), int(round(cy))),
                                COL_IRIS_FIT, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        # 3. HUD strip
        cv2.rectangle(vis, (0, 0), (W, 64), COL_HUD_BG, -1)
        L_per = result.per_eye["L"]
        R_per = result.per_eye["R"]
        line1 = (f"V2 spectacles + sclera/iris  |  fr {fr}/{total_frames}  |  "
                  f"rigid conf={result.confidence:.2f}  s={result.s:.2f}  "
                  f"rot={np.degrees(result.theta_rad):+.1f} deg")
        cv2.putText(vis, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        line2 = (f"L (pt-R): {L_per.status}  [{L_per.status_reason}]   "
                  f"R (pt-L): {R_per.status}  [{R_per.status_reason}]")
        cv2.putText(vis, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 220, 220), 1, cv2.LINE_AA)

        writer.write(vis)

        # ---- csv + counts -----------------------------------------------
        for ek in ("L", "R"):
            per = result.per_eye[ek]
            specs = per.spectacles
            confs = per.klt_weights
            row = [fr, ek]
            for name in POINTS:
                xy = specs[name]
                row.extend([round(xy[0], 2), round(xy[1], 2),
                             round(confs[name], 3)])
            row.append(int(per.sclera_area))
            row.append(round(per.sclera_inside_oval, 3))
            row.append(int(per.iris_support_present))
            row.append(round(per.iris_support_inside_oval, 3))
            row.append(round(per.iris_support_curvature_score, 3))
            row.append(round(per.iris_support_radius, 2) if per.iris_support_radius is not None else "")
            if per.iris_support_centre is not None:
                row.append(round(per.iris_support_centre[0], 2))
                row.append(round(per.iris_support_centre[1], 2))
            else:
                row.extend(["", ""])
            row.append(round(per.sclera_iris_pair_score, 3))
            row.append(round(per.content_score, 3))
            row.append(round(per.geometry_score, 3))
            row.append(round(per.final_score, 3))
            row.append(int(per.blink_hold_active))
            row.append(int(per.last_good_frame))
            row.append(int(per.frames_since_last_good_content))
            row.append(per.status)
            row.append(per.status_reason)
            cw.writerow(row)
            _bump(counts[ek], per.status)
        # combined frame status counter — most-pessimistic of the two eyes.
        frame_status = result.per_eye["L"].status \
            if STATUS_RANK.get(result.per_eye["L"].status, 0) \
               >= STATUS_RANK.get(result.per_eye["R"].status, 0) \
            else result.per_eye["R"].status
        _bump(overall_status_counts, frame_status)
        if fr == 1 or fr % 50 == 0:
            elapsed = time.time() - t_start
            rate = fr / max(elapsed, 1e-6)
            ms_per_frame = (elapsed / max(1, fr)) * 1000.0
            eta = (total_frames - fr) / max(rate, 1e-6)
            print(f"[v2-orbit] fr={fr:5d}/{total_frames}  "
                  f"L={result.per_eye['L'].status[:20]:20s}  "
                  f"R={result.per_eye['R'].status[:20]:20s}  "
                  f"rate={rate:.1f} fps  {ms_per_frame:.0f} ms/frame  "
                  f"ETA={eta:.0f}s")

    cap.release()
    writer.release()
    csv_f.close()
    summary = {
        "schema": "v2_orbit_tracking_summary",
        "schema_version": "2.0",
        "video": str(video_path),
        "stage0_frame": init_frame_no,
        "frames_processed": fr,
        "fps": round(float(fps), 4),
        "blink_tolerance_sec": 0.5,
        "blink_tolerance_frames": max(1, int(round(fps * 0.5))),
        "overall_status_counts": overall_status_counts,
        "L_status_counts": counts["L"],
        "R_status_counts": counts["R"],
        "outputs": {
            "stage0_json": str(stage0_path),
            "overlay": str(overlay_path),
            "csv": str(csv_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("[v2-orbit] SUMMARY:")
    print(f"[v2-orbit]   frames processed     : {fr}")
    print(f"[v2-orbit]   overall              : {overall_status_counts}")
    print(f"[v2-orbit]   L (patient Right)    : {counts['L']}")
    print(f"[v2-orbit]   R (patient Left)     : {counts['R']}")
    print(f"[v2-orbit]   overlay              : {overlay_path}")
    print(f"[v2-orbit]   csv                  : {csv_path}")
    print(f"[v2-orbit]   summary              : {summary_path}")


if __name__ == "__main__":
    main()
