"""EyeVNG entry point (Stage 0 → tracking; see docs/TRACKING_PHILOSOPHY.md).

    python app.py --video sample.mp4 --propose       # preview proposed landmarks (image)
    python app.py --video sample.mp4 --approve        # Stage 0: confirm/correct, save approved_landmarks.json
    python app.py --video sample.mp4 --auto-approve   # Stage 0 headless (accept proposal)
    python app.py --video sample.mp4                  # track (refuses until approved)
    python app.py --video sample.mp4 --review         # review low-confidence frames
    python app.py --video sample.mp4 --rescue         # clinician teaching mode: scrub + correct iris
    python app.py --video sample.mp4 --annotate-iris  # simple clinician-controlled iris video editor (§19)
"""
import argparse
from pathlib import Path

from iris_tracker import approve, propose_init, review, run


def main():
    ap = argparse.ArgumentParser(description="EyeVNG — iris + facial-landmark tracking")
    ap.add_argument("--video", required=True, help="path to the input video")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--propose", action="store_true", help="preview proposed landmarks (no save)")
    ap.add_argument("--approve", action="store_true", help="Stage 0: interactive confirm/correct → save")
    ap.add_argument("--auto-approve", action="store_true", help="Stage 0 headless: accept the proposal")
    ap.add_argument("--review", action="store_true", help="review low-confidence frames of an existing run")
    ap.add_argument("--rescue", action="store_true",
                    help="Clinician Teaching Mode (§18): scrub the tracked overlay video frame-by-frame "
                         "and mark/correct the iris circle on frames the tracker lost. Saves rescue anchors.")
    ap.add_argument("--rescue-eye", choices=["left", "right"], default="left",
                    help="(--rescue) which anatomical eye to start the scrubber on (default: left)")
    ap.add_argument("--annotate-iris", action="store_true",
                    help="Open the SIMPLE clinician-controlled iris video editor (§19) on the RAW video. "
                         "Independent of the V1 tracker, detector, and rescue scrubber. The clinician owns "
                         "the iris marker; save per-frame annotations to manual_iris_annotations.json. "
                         "First version covers the patient's Right eye only.")
    ap.add_argument("--simple-iris-editor", action="store_true",
                    help="Alias for --annotate-iris.")
    ap.add_argument("--filter", choices=["none", "light", "adaptive"], default="adaptive",
                    help="iris jitter filter (default adaptive 1€)")
    ap.add_argument("--show-raw-trace", action="store_true",
                    help="also draw the raw iris signal faint (overlay + traces)")
    ap.add_argument("--eye", choices=["auto", "left", "right", "both"], default="auto",
                    help="which eye's trace to display (default auto=single best-tracked eye; "
                         "'both' only when the eyes differ, e.g. INO)")
    ap.add_argument("--engine", choices=["v1", "composite", "template", "anatomical"], default="v1",
                    help="tracking engine: v1 (limbus+contour orchestrator, default), "
                         "anatomical (anatomically-constrained iris–sclera boundary "
                         "tracker, freeze-on-fail; see TRACKER_REDESIGN.md), "
                         "composite (CFT only), template (legacy)")
    ap.add_argument("--no-mediapipe", action="store_true",
                    help="run from approved landmarks without MediaPipe proposal/recovery")
    ap.add_argument("--cft-helper", action="store_true",
                    help="(v1 engine) enable the CFT as a motion-prediction helper "
                         "(does not change the clinical centre or validity)")
    ap.add_argument("--overlay-mode", choices=["clinical", "debug"], default="clinical",
                    help="overlay video presentation: clinical (default, minimal Left/Right + "
                         "nystagmus status only) or debug (contour, limbus, fc/ic/rc/cov text, "
                         "trace strip)")
    args = ap.parse_args()
    if args.propose:
        propose_init(args.video, args.output_dir)
    elif args.approve or args.auto_approve:
        approve(args.video, args.output_dir, auto=args.auto_approve)
    elif args.review:
        review(args.video, args.output_dir)
    elif args.rescue:
        # Clinician Teaching Mode (§18). Opens the V1 clinical overlay video for the clip
        # in a scrubber; the clinician corrects the iris circle on whichever frames need it
        # and saves rescue anchors. Does NOT modify tracking output.
        from src.core.iris_rescue import run_rescue_scrubber
        video_path = Path(args.video)
        stem_dir = Path(args.output_dir) / (video_path.stem + "_tracked")
        # Image-side mapping: front-facing video → patient Left = image side R, patient Right = image side L.
        eye_key = "R" if args.rescue_eye == "left" else "L"
        run_rescue_scrubber(stem_dir, video_path, eye_key=eye_key)
    elif args.annotate_iris or args.simple_iris_editor:
        # Simple Clinician-Controlled Iris Video Editor (§19). Opens the RAW video and lets
        # the clinician drag a single iris circle through the video. Independent of the V1
        # tracker, the detector, the rescue scrubber, and the overlay video. The clinician
        # owns the marker; saves to outputs/<clip>_tracked/manual_iris_annotations.json.
        from src.core.simple_iris_editor import run_simple_iris_editor
        run_simple_iris_editor(Path(args.video), Path(args.output_dir))
    else:
        run(args.video, args.output_dir, filter_mode=args.filter, show_raw=args.show_raw_trace,
            eye=args.eye, engine=args.engine, no_mediapipe=args.no_mediapipe,
            cft_helper=args.cft_helper, overlay_mode=args.overlay_mode)


if __name__ == "__main__":
    main()
