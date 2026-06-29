"""V2 Milestone 0 probe — Layer 1 only.

Loops over a video, runs the V2 face/eye-zone locator (MediaPipe FaceMesh), draws the
face bbox + L/R eye-zone bboxes on each frame, and writes:

    outputs/v2_face_eye_zone_probe/face_eye_zones_overlay.mp4
    outputs/v2_face_eye_zone_probe/face_eye_zones.csv

The probe answers exactly one question (per the user's spec):

    "Does MediaPipe give us a reliable moving search region for the eyes?"

Nothing in this script touches:
  * Stage 0,
  * iris detection,
  * limbus,
  * pupil,
  * nystagmus.

It does not import any V1 code (src/core/* or iris_tracker.py).

Usage
-----
    .venv\\Scripts\\python.exe v2\\probe_face_eye_zones.py --video PATH

    --video PATH   path to the input video (required)
    --output-dir   default: outputs/v2_face_eye_zone_probe
    --max-frames   optional cap on number of frames processed
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

# Make `from v2.core.face_locator import FaceLocator` work whether the probe is run
# from the repo root or from inside v2/.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from v2.core.face_locator import FaceLocator   # noqa: E402  (sys.path tweak above)


# --------------------------------------------------------------------------- drawing
COL_FACE = (0, 200, 255)      # amber  -- face bbox
COL_EYE_L = (0, 255, 0)       # green  -- image-side L eye zone (patient Right)
COL_EYE_R = (255, 100, 0)     # blue   -- image-side R eye zone (patient Left)
COL_HUD_BG = (0, 0, 0)
COL_HUD_FG = (255, 255, 255)
COL_MP_OK = (0, 255, 0)
COL_MP_FAIL = (0, 0, 255)


def _draw_bbox(img, bbox, colour, label):
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
    # Label sits just above the top-left corner, with a dark shadow for legibility.
    label_y = max(20, y0 - 6)
    cv2.putText(img, label, (x0 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (x0 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def _draw_hud(img, fr: int, total: int, mp_ok: bool):
    """Top-left status strip: frame counter + MediaPipe detection state."""
    h, w = img.shape[:2]
    strip_h = 34
    cv2.rectangle(img, (0, 0), (w, strip_h), COL_HUD_BG, -1)
    txt = (f"V2 Milestone-0 probe  |  fr {fr}/{total}  |  "
           f"MediaPipe: {'detected' if mp_ok else 'NOT DETECTED'}")
    colour = COL_MP_OK if mp_ok else COL_MP_FAIL
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                colour, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="V2 Milestone-0 face / eye-zone probe.")
    ap.add_argument("--video", required=True, help="path to the input video")
    ap.add_argument("--output-dir", default="outputs/v2_face_eye_zone_probe",
                    help="directory for the overlay MP4 and CSV (default: %(default)s)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="optional cap on processed frames; 0 = no cap")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / "face_eye_zones_overlay.mp4"
    csv_path = out_dir / "face_eye_zones.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames and args.max_frames < total_frames:
        cap_total = args.max_frames
    else:
        cap_total = total_frames

    writer = cv2.VideoWriter(str(overlay_path),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open overlay writer: {overlay_path}")

    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    cw = csv.writer(csv_f)
    cw.writerow([
        "frame_number", "mp_detected",
        "face_x0", "face_y0", "face_x1", "face_y1",
        "L_eye_x0", "L_eye_y0", "L_eye_x1", "L_eye_y1",
        "R_eye_x0", "R_eye_y0", "R_eye_x1", "R_eye_y1",
    ])

    print(f"[v2-probe] video={video_path.name}  {width}x{height} @ {fps:.2f} fps  "
          f"total_frames={total_frames}")
    print(f"[v2-probe] writing  overlay  -> {overlay_path}")
    print(f"[v2-probe] writing  csv      -> {csv_path}")

    locator = FaceLocator(max_num_faces=1, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5, static_image_mode=False)

    n_detected = 0
    n_failed = 0
    t0 = time.time()
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fr += 1
        if args.max_frames and fr > args.max_frames:
            break
        loc = locator.locate(frame)
        if loc.mp_detected:
            n_detected += 1
        else:
            n_failed += 1
        # ---- draw ----
        vis = frame.copy()
        _draw_bbox(vis, loc.face_bbox, COL_FACE, "face")
        _draw_bbox(vis, loc.L_eye_bbox, COL_EYE_L, "L eye zone (patient Right)")
        _draw_bbox(vis, loc.R_eye_bbox, COL_EYE_R, "R eye zone (patient Left)")
        _draw_hud(vis, fr, cap_total, loc.mp_detected)
        writer.write(vis)
        # ---- csv ----
        def _row_bbox(b):
            return list(b) if b is not None else ["", "", "", ""]
        cw.writerow(
            [fr, int(bool(loc.mp_detected))]
            + _row_bbox(loc.face_bbox)
            + _row_bbox(loc.L_eye_bbox)
            + _row_bbox(loc.R_eye_bbox)
        )
        if fr % 100 == 0 or fr == 1:
            elapsed = time.time() - t0
            rate = fr / max(elapsed, 1e-6)
            print(f"[v2-probe] fr={fr:5d}/{cap_total}  detected_so_far={n_detected:5d}  "
                  f"failed_so_far={n_failed:5d}  rate={rate:.1f} fps")

    cap.release()
    writer.release()
    csv_f.close()
    locator.close()

    n_processed = n_detected + n_failed
    pct = (100.0 * n_detected / n_processed) if n_processed else 0.0
    print()
    print(f"[v2-probe] SUMMARY:")
    print(f"[v2-probe]   frames processed     : {n_processed}")
    print(f"[v2-probe]   MediaPipe detected   : {n_detected} ({pct:.1f} %)")
    print(f"[v2-probe]   MediaPipe failed     : {n_failed}")
    print(f"[v2-probe]   overlay              : {overlay_path}")
    print(f"[v2-probe]   csv                  : {csv_path}")


if __name__ == "__main__":
    main()
