from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def write_overlay_video(frames: List[np.ndarray], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for {output_path}")

    for frame in frames:
        writer.write(frame)

    writer.release()
