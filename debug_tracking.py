"""Tracking verification / debug mode for the face-based (Mode A) pipeline.

Goal: watch the overlay and immediately know whether tracking is trustworthy.
See VERIFICATION_MODE.md. This does NOT smooth or guess silently — failures stay
visible. Carried-forward and jump-flagged points are shown but clearly marked,
and are NEVER written into the raw iris columns of the CSV.

    python debug_tracking.py <video> [--output-dir outputs] [--max-debug-frames 80]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2

from src.core.tracking import EyeTrackingProcessor

# --- classification thresholds ---
GOOD_CONF = 0.60          # per-eye confidence to count as solidly tracked
DETECT_CONF = 0.30        # below this an eye is treated as not reliably detected
CARRY_MAX_FRAMES = 5      # carry a missing eye's last-known position up to this many frames
JUMP_FRAC = 0.60          # flag a jump larger than this fraction of inter-ocular distance
OVERLAY_MAX_W = 960       # downscale overlay for quick review

GREEN = (0, 200, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _classify(face_ok, mode, left, right, jump_left, jump_right, carried_left, carried_right):
    """Return (status_str, color). left/right = (detected, conf)."""
    if not face_ok or mode in ("none", "rejected_collapse"):
        reason = "collapsed_mesh" if mode == "rejected_collapse" else "no_face"
        return f"lost:{reason}", RED
    ld, lc = left
    rd, rc = right
    if not ld and not rd:
        return "lost:no_iris", RED

    reasons = []
    if carried_left or carried_right:
        reasons.append("carried_forward")
    if (ld != rd) and not (carried_left or carried_right):
        reasons.append("one_eye")
    confs = [c for d, c in (left, right) if d]
    if confs and min(confs) < GOOD_CONF:
        reasons.append("low_conf")
    if jump_left or jump_right:
        reasons.append("jump_flag")

    if reasons:
        return "uncertain:" + ",".join(reasons), YELLOW
    return "good", GREEN


def run_debug(video_path, output_dir="outputs", max_debug_frames=80):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path(output_dir) / video_path.stem
    dbg_dir = out_dir / "debug_frames"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    scale = min(1.0, OVERLAY_MAX_W / width)
    ow, oh = int(width * scale), int(height * scale)
    fs = 0.55                       # font scale (fixed, in overlay pixels)
    line_h = 22                     # text line height in overlay pixels
    marker_r = max(7, int(ow * 0.013))  # iris marker radius in overlay pixels
    writer = cv2.VideoWriter(str(out_dir / "debug_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    csv_f = (out_dir / "tracking_quality.csv").open("w", newline="", encoding="utf-8")
    cw = csv.writer(csv_f)
    cw.writerow(["frame_number", "time_sec", "face_detected",
                 "left_eye_detected", "right_eye_detected",
                 "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
                 "tracking_status", "left_conf", "right_conf"])

    proc = EyeTrackingProcessor()
    last_good = {"left": None, "right": None}   # (x, y, frame)
    last_interocular = None
    counts = {"good": 0, "uncertain": 0, "lost": 0}
    saved = {"uncertain": 0, "lost": 0}
    seen = {"uncertain": 0, "lost": 0}
    stride = max(1, total // max(1, max_debug_frames))
    cur_lost_run = longest_lost_run = 0
    non_good_examples = []
    fr = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr += 1
        t = fr / fps
        out = proc.process_frame(frame)
        face_ok = bool(out.get("face_detected"))
        mode = out.get("detection_mode", "none")

        lx, ly = out.get("left_eye_x"), out.get("left_eye_y")
        rx, ry = out.get("right_eye_x"), out.get("right_eye_y")
        lc, rc = out.get("left_conf", 0.0), out.get("right_conf", 0.0)
        ld = lx is not None and lc >= DETECT_CONF
        rd = rx is not None and rc >= DETECT_CONF

        if ld and rd:
            last_interocular = abs(rx - lx)
        ref = last_interocular

        # jump flags (flag only, never delete) ---------------------------------
        jump_l = jump_r = False
        if ld and last_good["left"] and ref:
            if _dist((lx, ly), last_good["left"][:2]) > JUMP_FRAC * ref:
                jump_l = True
        if rd and last_good["right"] and ref:
            if _dist((rx, ry), last_good["right"][:2]) > JUMP_FRAC * ref:
                jump_r = True

        # carry-forward (overlay only; not written to raw columns) -------------
        carried_l = carried_r = None
        if not ld and last_good["left"] and fr - last_good["left"][2] <= CARRY_MAX_FRAMES:
            carried_l = last_good["left"][:2]
        if not rd and last_good["right"] and fr - last_good["right"][2] <= CARRY_MAX_FRAMES:
            carried_r = last_good["right"][:2]

        status, color = _classify(face_ok, mode, (ld, lc), (rd, rc),
                                   jump_l, jump_r, carried_l, carried_r)
        bucket = status.split(":")[0]
        counts[bucket] += 1

        if bucket == "lost":
            cur_lost_run += 1
            longest_lost_run = max(longest_lost_run, cur_lost_run)
        else:
            cur_lost_run = 0

        if ld:
            last_good["left"] = (lx, ly, fr)
        if rd:
            last_good["right"] = (rx, ry, fr)

        # --- CSV row (raw detections only; blanks when not detected) ----------
        cw.writerow([fr, round(t, 4), face_ok, ld, rd,
                     lx if ld else "", ly if ld else "",
                     rx if rd else "", ry if rd else "",
                     status, round(lc, 3), round(rc, 3)])

        # --- overlay ----------------------------------------------------------
        vis = cv2.resize(frame, (ow, oh))
        cv2.rectangle(vis, (0, 0), (ow - 1, oh - 1), color, 5)
        for det, x, y, conf in ((ld, lx, ly, lc), (rd, rx, ry, rc)):
            if det:
                px, py = int(x * scale), int(y * scale)
                mc = GREEN if conf >= GOOD_CONF else YELLOW
                cv2.circle(vis, (px, py), marker_r, mc, 2)
                cv2.circle(vis, (px, py), 2, mc, -1)            # centre dot
        for c in (carried_l, carried_r):   # carried = hollow yellow + 'C', clearly not a fresh detection
            if c:
                px, py = int(c[0] * scale), int(c[1] * scale)
                cv2.circle(vis, (px, py), marker_r + 2, YELLOW, 1)
                cv2.putText(vis, "C", (px + marker_r, py - marker_r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        lines = [f"frame {fr}  t={t:.2f}s",
                 f"face:{'Y' if face_ok else 'N'}  L:{'Y' if ld else 'N'}({lc:.2f})  R:{'Y' if rd else 'N'}({rc:.2f})",
                 status]
        for i, txt in enumerate(lines):
            y = 20 + i * line_h
            cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3)
            cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1)
        writer.write(vis)

        # --- save sampled debug frames ----------------------------------------
        if bucket in ("uncertain", "lost"):
            seen[bucket] += 1
            if seen[bucket] % stride == 0 and saved[bucket] < max_debug_frames:
                tag = status.replace(":", "_").replace(",", "+")
                cv2.imwrite(str(dbg_dir / f"{fr:06d}_{tag}.png"), vis)
                saved[bucket] += 1
            if len(non_good_examples) < 25:
                non_good_examples.append((fr, round(t, 2), status))

    cap.release()
    writer.release()
    csv_f.close()

    analyzed = max(1, fr)
    good_pct = 100.0 * counts["good"] / analyzed
    summary = {
        "video": str(video_path),
        "resolution": f"{width}x{height}", "fps": round(fps, 2),
        "frames": fr,
        "good": counts["good"], "uncertain": counts["uncertain"], "lost": counts["lost"],
        "good_pct": round(good_pct, 1),
        "uncertain_pct": round(100.0 * counts["uncertain"] / analyzed, 1),
        "lost_pct": round(100.0 * counts["lost"] / analyzed, 1),
        "longest_lost_run_frames": longest_lost_run,
        "longest_lost_run_sec": round(longest_lost_run / fps, 2),
        "target_95pct_met": good_pct >= 95.0,
        "debug_frames_saved": saved,
        "outputs": {
            "tracking_quality_csv": str(out_dir / "tracking_quality.csv"),
            "debug_overlay": str(out_dir / "debug_overlay.mp4"),
            "debug_frames_dir": str(dbg_dir),
        },
    }
    (out_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    if non_good_examples:
        print("\nfirst non-good frames (frame, t, status):")
        for ex in non_good_examples:
            print("  ", ex)
    print(f"\n{'PASS' if summary['target_95pct_met'] else 'BELOW TARGET'}: "
          f"{good_pct:.1f}% good (target >95%).")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Eye-tracking verification / debug mode")
    ap.add_argument("video")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--max-debug-frames", type=int, default=80)
    args = ap.parse_args()
    run_debug(args.video, args.output_dir, args.max_debug_frames)


if __name__ == "__main__":
    main()
