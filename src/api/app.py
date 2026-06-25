import argparse
import json
import sys
from pathlib import Path

from src.modules.analyze_video import VideoAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="EyeVNG Version 1 measurement pipeline")
    parser.add_argument("video", help="Path to uploaded MP4 video")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported results")
    args = parser.parse_args()

    analyzer = VideoAnalyzer(args.output_dir)
    results = analyzer.analyze(args.video)

    print(json.dumps({
        "filename": results["filename"],
        "resolution": results["resolution"],
        "fps": results["fps"],
        "frame_count": results["frame_count"],
        "duration": results["duration"],
        "measurements_csv": results["measurements_csv"],
        "overlay_video": results["overlay_video"],
        "quality": results["quality"],
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
