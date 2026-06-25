from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.core.head_pose import estimate_head_pose
from src.core.signal_processing import summarize_quality, write_measurements_csv
from src.core.tracking import EyeTrackingProcessor
from src.core.visualization import write_overlay_video


class VideoAnalyzer:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = EyeTrackingProcessor()

    def analyze(self, video_path: str | Path) -> Dict:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0.0

        rows: List[Dict] = []
        overlay_frames: List[np.ndarray] = []
        frame_number = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            tracking = self.tracker.process_frame(frame)
            head_pose = estimate_head_pose(frame, tracking.get("landmarks"))
            measurement = {
                "frame_number": frame_number,
                "time_sec": frame_number / fps,
                "left_eye_x": tracking.get("left_eye_x"),
                "left_eye_y": tracking.get("left_eye_y"),
                "right_eye_x": tracking.get("right_eye_x"),
                "right_eye_y": tracking.get("right_eye_y"),
                "mean_eye_x": tracking.get("mean_eye_x"),
                "mean_eye_y": tracking.get("mean_eye_y"),
                "face_center_x": tracking.get("face_center_x"),
                "face_center_y": tracking.get("face_center_y"),
                "head_yaw": head_pose["head_yaw"],
                "head_pitch": head_pose["head_pitch"],
                "head_roll": head_pose["head_roll"],
                "face_detected": tracking.get("face_detected", False),
                "tracking_confidence": tracking.get("tracking_confidence", 0.0),
                "left_conf": tracking.get("left_conf", 0.0),
                "right_conf": tracking.get("right_conf", 0.0),
                "detection_mode": tracking.get("detection_mode", "none"),
            }
            rows.append(measurement)

            overlay_frame = frame.copy()
            scale = max(1.0, min(width, height) / 480.0)
            mesh_r = max(1, int(2 * scale))
            eye_r = max(4, int(10 * scale))
            eye_thick = max(2, int(3 * scale))
            font_scale = 0.7 * scale
            font_thick = max(2, int(2 * scale))
            if tracking.get("landmarks"):
                for idx, point in enumerate(tracking["landmarks"]):
                    cv2.circle(overlay_frame, tuple(map(int, point)), mesh_r, (0, 255, 0), -1)
            if tracking.get("left_eye_x") is not None:
                cv2.circle(overlay_frame, (int(tracking["left_eye_x"]), int(tracking["left_eye_y"])), eye_r, (255, 0, 0), eye_thick)
            if tracking.get("right_eye_x") is not None:
                cv2.circle(overlay_frame, (int(tracking["right_eye_x"]), int(tracking["right_eye_y"])), eye_r, (0, 0, 255), eye_thick)
            if tracking.get("mean_eye_x") is not None:
                cv2.circle(overlay_frame, (int(tracking["mean_eye_x"]), int(tracking["mean_eye_y"])), eye_r, (0, 255, 255), eye_thick)
            cv2.putText(overlay_frame, f"frame {frame_number}", (int(20 * scale), int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
            cv2.putText(overlay_frame, f"yaw {head_pose['head_yaw']:.1f}", (int(20 * scale), int(80 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
            overlay_frames.append(overlay_frame)

        cap.release()

        write_measurements_csv(rows, self.output_dir / "measurements.csv")
        write_overlay_video(overlay_frames, self.output_dir / "overlay.mp4", fps)

        quality = summarize_quality(rows, fps)
        output = {
            "video_path": str(video_path),
            "filename": video_path.name,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frame_count": frame_count,
            "duration": round(duration, 3),
            "measurements_csv": str(self.output_dir / "measurements.csv"),
            "overlay_video": str(self.output_dir / "overlay.mp4"),
            "quality": quality,
            "rows": rows,
        }
        return output
