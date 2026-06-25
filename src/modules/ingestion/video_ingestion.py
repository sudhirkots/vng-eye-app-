import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2


class VideoIngestion:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_metadata(self, video_path: str | Path, protocol: str = "custom") -> Dict:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0.0

        metadata = {
            "video_filename": video_path.name,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration_sec": round(duration, 3),
            "frame_count": frame_count,
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "protocol": protocol,
            "tracking_success_percent": None,
            "generated_output_files": [],
        }

        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / f"{video_path.stem}_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        cap.release()
        return metadata
