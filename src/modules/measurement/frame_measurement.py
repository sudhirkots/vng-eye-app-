import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.core.head_pose import HeadPoseEstimator
from src.core.tracking import EyeTrackingProcessor


class FrameMeasurementPipeline:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = EyeTrackingProcessor()
        self.head_pose_estimator = HeadPoseEstimator()

    def process_video(self, video_path: str | Path, protocol: str = "custom") -> Dict:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        measurements: List[Dict] = []
        frame_number = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            tracking = self.tracker.process_frame(frame)
            head_pose = self.head_pose_estimator.estimate_pose(tracking.get("landmarks"))
            measurement = {
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
                "time_sec": frame_number / fps if fps else 0.0,
                "left_eye_x": tracking.get("left_eye_x"),
                "left_eye_y": tracking.get("left_eye_y"),
                "right_eye_x": tracking.get("right_eye_x"),
                "right_eye_y": tracking.get("right_eye_y"),
                "mean_eye_x": tracking.get("mean_eye_x"),
                "mean_eye_y": tracking.get("mean_eye_y"),
                "face_center_x": tracking.get("face_center_x"),
                "face_center_y": tracking.get("face_center_y"),
                "tracking_confidence": tracking.get("tracking_confidence", 0.0),
                "face_detected": tracking.get("face_detected", False),
                "head_yaw": head_pose.get("head_yaw", 0.0),
                "head_pitch": head_pose.get("head_pitch", 0.0),
                "head_roll": head_pose.get("head_roll", 0.0),
            }
            measurements.append(measurement)
            frames.append(frame)

        cap.release()

        raw_dir = self.output_dir / "raw_measurements"
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = raw_dir / f"{video_path.stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(measurements[0].keys()) if measurements else [])
            writer.writeheader()
            writer.writerows(measurements)

        return {
            "video_path": str(video_path),
            "video_name": video_path.name,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration_sec": round(frame_count / fps if fps else 0.0, 3),
            "frame_count": frame_count,
            "raw_csv": str(csv_path),
            "measurements": measurements,
            "frames": frames,
            "protocol": protocol,
        }
