import csv
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List


def write_measurements_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_number",
                "time_sec",
                "left_eye_x",
                "left_eye_y",
                "right_eye_x",
                "right_eye_y",
                "mean_eye_x",
                "mean_eye_y",
                "face_center_x",
                "face_center_y",
                "head_yaw",
                "head_pitch",
                "head_roll",
                "face_detected",
                "tracking_confidence",
                "left_conf",
                "right_conf",
                "detection_mode",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_quality(rows: List[Dict], fps: float) -> Dict:
    total_frames = len(rows)
    analyzed_frames = sum(1 for row in rows if row.get("face_detected", False))
    lost_frames = total_frames - analyzed_frames
    tracking_success = (analyzed_frames / total_frames * 100.0) if total_frames else 0.0

    # Average confidence over detected frames only; undetected frames are 0 and
    # would otherwise drag the figure down misleadingly.
    detected_conf = [row.get("tracking_confidence", 0.0) for row in rows if row.get("face_detected", False)]
    average_confidence = mean(detected_conf) if detected_conf else 0.0

    mode_counts: Dict[str, int] = {}
    for row in rows:
        mode = row.get("detection_mode", "none")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    warnings = []
    if tracking_success < 90.0:
        warnings.append("tracking success below 90%")
    rejected = mode_counts.get("rejected_collapse", 0)
    if rejected:
        warnings.append(f"{rejected} frame(s) rejected as collapsed mesh (likely eye close-up — needs Mode B)")

    return {
        "total_frames": total_frames,
        "analyzed_frames": analyzed_frames,
        "lost_frames": lost_frames,
        "tracking_success_percent": round(tracking_success, 2),
        "average_confidence": round(average_confidence, 3),
        "detection_mode_counts": mode_counts,
        "warnings": warnings,
        "fps": fps,
    }
