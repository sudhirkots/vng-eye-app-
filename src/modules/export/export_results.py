import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


class ExportResults:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_overlay_video(self, frames: List[np.ndarray], video_name: str, fps: float) -> str:
        overlay_dir = self.output_dir / "overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        output_path = overlay_dir / f"{video_name}_overlay.mp4"
        if not frames:
            return str(output_path)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open writer for {output_path}")
        for frame in frames:
            writer.write(frame)
        writer.release()
        return str(output_path)

    def export_roi_videos(self, roi_frames_left: List[np.ndarray], roi_frames_right: List[np.ndarray], video_name: str, fps: float) -> Dict:
        roi_dir = self.output_dir / "rois"
        roi_dir.mkdir(parents=True, exist_ok=True)
        left_path = roi_dir / f"{video_name}_left_eye_roi.mp4"
        right_path = roi_dir / f"{video_name}_right_eye_roi.mp4"
        left_writer = self._create_writer(left_path, roi_frames_left, fps)
        right_writer = self._create_writer(right_path, roi_frames_right, fps)
        if left_writer:
            left_writer.release()
        if right_writer:
            right_writer.release()
        return {"left_roi_video": str(left_path), "right_roi_video": str(right_path)}

    def _create_writer(self, path: Path, frames: List[np.ndarray], fps: float):
        if not frames:
            return None
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to open writer for {path}")
        for frame in frames:
            writer.write(frame)
        return writer

    def export_roi_coordinates(self, roi_coordinates: List[Dict], video_name: str) -> str:
        roi_dir = self.output_dir / "rois"
        roi_dir.mkdir(parents=True, exist_ok=True)
        coordinates_path = roi_dir / f"{video_name}_roi_coordinates.json"
        with coordinates_path.open("w", encoding="utf-8") as handle:
            json.dump(roi_coordinates, handle, indent=2)
        return str(coordinates_path)
