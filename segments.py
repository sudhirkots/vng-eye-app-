"""Segment-based analysis (EYEVNG spec §11 part 2).

Clinical principle: a short TRUSTWORTHY segment beats a long corrupted trace. Bad frames are NOT
rescued — they are excluded, and the trace is never connected across them.

This tool reads a clip's `tracking.csv`, classifies every frame, finds continuous VALID segments that
are long enough and high-confidence, plots each segment separately, and lets the clinician approve or
reject segments. Final analysis uses ONLY approved segments.

Per-frame label (one of):  valid · blink · occluded · drift_suspected · tracking_lost
A segment is a maximal run of `valid` frames (iris attached, no blink/drift/occlusion, frame
confidence ≥ threshold) lasting ≥ the minimum duration.

Usage:
    python segments.py --video samples/2.mp4 [--output-dir outputs] [--eye auto|left|right]
                       [--min-duration 2.0] [--min-confidence 0.4]
    python segments.py --video samples/2.mp4 --approve 1,3      # approve segments 1 and 3
    python segments.py --video samples/2.mp4 --reject 2         # reject segment 2
    python segments.py --video samples/2.mp4 --status           # print the current segment table
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from iris_tracker import plot_trace, GREEN

# CFT state (spec §B) → clinical frame label. Only `valid` frames are analysable.
STATE_LABEL = {
    "TRACKING": "valid", "REPLENISHING_FEATURES": "valid", "INITIALIZING": "valid",
    "PARTIAL_OCCLUSION": "occluded",
    "BLINK": "blink",
    "DRIFT_SUSPECTED": "drift_suspected", "REACQUIRING": "drift_suspected",
    "TRACK_LOST": "tracking_lost",
}
LABELS = ("valid", "blink", "occluded", "drift_suspected", "tracking_lost")


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def classify(rows, eye, min_conf):
    """Per-frame (label, time_sec, eye_local_x, eye_local_y, frame_confidence) for the chosen eye.
    A frame is `valid` only if its state is a valid state, it has an iris position, AND its frame
    confidence clears the bar — otherwise it carries its state's label (blink/occluded/drift/lost)."""
    out = []
    for r in rows:
        t = _f(r.get("time_sec"))
        state = r.get(f"state_{eye}", "")
        validity = r.get(f"validity_{eye}", "")
        fc = _f(r.get(f"frame_confidence_{eye}")) or 0.0
        elx = _f(r.get(f"{eye}_eye_local_x"))
        ely = _f(r.get(f"{eye}_eye_local_y"))
        label = STATE_LABEL.get(state, "tracking_lost")
        if label == "valid" and (validity != "valid" or elx is None or fc < min_conf):
            label = "drift_suspected" if validity != "valid" else "occluded"  # demote low-confidence
        out.append((label, t, elx, ely, fc))
    return out


def find_segments(frames, min_dur):
    """Maximal runs of `valid` frames lasting ≥ min_dur seconds. Never spans an invalid frame."""
    segs = []
    i, n = 0, len(frames)
    while i < n:
        if frames[i][0] != "valid":
            i += 1
            continue
        j = i
        while j < n and frames[j][0] == "valid":
            j += 1
        t0, t1 = frames[i][1], frames[j - 1][1]
        if t0 is not None and t1 is not None and (t1 - t0) >= min_dur:
            confs = [frames[k][4] for k in range(i, j)]
            segs.append({"start_i": i, "end_i": j - 1, "t0": round(t0, 2), "t1": round(t1, 2),
                         "duration": round(t1 - t0, 2), "n_frames": j - i,
                         "mean_confidence": round(float(np.mean(confs)), 3), "status": "pending"})
        i = j
    return segs


def _seg_path(out_dir):
    return out_dir / "segments.json"


def load_segments(out_dir):
    p = _seg_path(out_dir)
    return json.loads(p.read_text()) if p.exists() else None


def save_segments(out_dir, data):
    _seg_path(out_dir).write_text(json.dumps(data, indent=2))


def build(video, output_dir, eye_arg, min_dur, min_conf):
    out_dir = Path(output_dir) / (Path(video).stem + "_tracked")
    rows = list(csv.DictReader((out_dir / "tracking.csv").open()))
    n = len(rows)
    # choose eye: the one with more valid frames (unless forced)
    def n_valid(e):
        return sum(1 for r in rows if STATE_LABEL.get(r.get(f"state_{e}", ""), "") == "valid"
                   and r.get(f"validity_{e}", "") == "valid")
    eye = {"left": "left", "right": "right"}.get(eye_arg) or ("left" if n_valid("left") >= n_valid("right") else "right")

    frames = classify(rows, eye, min_conf)
    counts = {lab: sum(1 for f in frames if f[0] == lab) for lab in LABELS}
    segs = find_segments(frames, min_dur)

    # per-segment eye-local plots (each segment plotted ALONE — no connecting across gaps)
    for idx, s in enumerate(segs, 1):
        sl = frames[s["start_i"]:s["end_i"] + 1]
        ts = [f[1] for f in sl]
        plot_trace(ts, {"eye-local H": ([f[2] for f in sl], GREEN)},
                   f"Segment {idx}: {s['t0']}-{s['t1']}s  H", out_dir / f"segment_{idx}_H.png")
        plot_trace(ts, {"eye-local V": ([f[3] for f in sl], GREEN)},
                   f"Segment {idx}: {s['t0']}-{s['t1']}s  V", out_dir / f"segment_{idx}_V.png")

    data = {"video": str(video), "eye": eye, "n_frames": n, "min_duration": min_dur,
            "min_confidence": min_conf, "frame_label_counts": counts, "segments": segs}
    save_segments(out_dir, data)
    return data, out_dir


def print_table(data):
    print(f"\nClip: {Path(data['video']).name}   eye={data['eye']}   frames={data['n_frames']}")
    print("Frame labels:", "  ".join(f"{k}={v}" for k, v in data["frame_label_counts"].items()))
    segs = data["segments"]
    if not segs:
        print("\nNo valid segments >= the minimum duration. Nothing to analyse (this is the correct,"
              " honest result for a clip with no long clean stretch).")
        return
    print(f"\n{len(segs)} candidate valid segment(s):")
    for idx, s in enumerate(segs, 1):
        mark = {"approved": "[APPROVED]", "rejected": "[rejected]", "pending": "[ pending]"}[s["status"]]
        print(f"  Segment {idx}: {s['t0']:6.2f}-{s['t1']:6.2f} s  "
              f"({s['duration']:.1f}s, {s['n_frames']} frames, conf {s['mean_confidence']:.2f})   {mark}")
    appr = [i + 1 for i, s in enumerate(segs) if s["status"] == "approved"]
    print(f"\nApproved: {appr or 'none yet'}   "
          f"(approve with:  --approve {','.join(str(i+1) for i in range(len(segs)))})")


def set_status(out_dir, idxs, status):
    data = load_segments(out_dir)
    if not data:
        raise SystemExit("No segments.json — run the build first (without --approve/--reject).")
    for i in idxs:
        if 1 <= i <= len(data["segments"]):
            data["segments"][i - 1]["status"] = status
    save_segments(out_dir, data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--eye", choices=["auto", "left", "right"], default="auto")
    ap.add_argument("--min-duration", type=float, default=2.0)
    ap.add_argument("--min-confidence", type=float, default=0.4)
    ap.add_argument("--approve", default="", help="comma-separated segment numbers to approve")
    ap.add_argument("--reject", default="", help="comma-separated segment numbers to reject")
    ap.add_argument("--status", action="store_true", help="print the current segment table only")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) / (Path(args.video).stem + "_tracked")
    if args.status or args.approve or args.reject:
        if args.approve:
            data = set_status(out_dir, [int(x) for x in args.approve.split(",")], "approved")
        if args.reject:
            data = set_status(out_dir, [int(x) for x in args.reject.split(",")], "rejected")
        data = load_segments(out_dir)
        print_table(data)
        return
    data, out_dir = build(args.video, args.output_dir, args.eye, args.min_duration, args.min_confidence)
    print_table(data)
    print(f"\nWrote {out_dir/'segments.json'} + per-segment plots (segment_N_H/V.png).")


if __name__ == "__main__":
    main()
