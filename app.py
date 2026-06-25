"""EyeVNG entry point (Stage 0 → tracking; see docs/TRACKING_PHILOSOPHY.md).

    python app.py --video sample.mp4 --propose       # preview proposed landmarks (image)
    python app.py --video sample.mp4 --approve        # Stage 0: confirm/correct, save approved_landmarks.json
    python app.py --video sample.mp4 --auto-approve   # Stage 0 headless (accept proposal)
    python app.py --video sample.mp4                  # track (refuses until approved)
    python app.py --video sample.mp4 --review         # review low-confidence frames
"""
import argparse

from pupil_tracker import approve, propose_init, review, run


def main():
    ap = argparse.ArgumentParser(description="EyeVNG — pupil + facial-landmark tracking")
    ap.add_argument("--video", required=True, help="path to the input video")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--propose", action="store_true", help="preview proposed landmarks (no save)")
    ap.add_argument("--approve", action="store_true", help="Stage 0: interactive confirm/correct → save")
    ap.add_argument("--auto-approve", action="store_true", help="Stage 0 headless: accept the proposal")
    ap.add_argument("--review", action="store_true", help="review low-confidence frames of an existing run")
    ap.add_argument("--filter", choices=["none", "light", "adaptive"], default="adaptive",
                    help="pupil jitter filter (default adaptive 1€)")
    ap.add_argument("--show-raw-trace", action="store_true",
                    help="also draw the raw pupil signal faint (overlay + traces)")
    ap.add_argument("--eye", choices=["auto", "left", "right", "both"], default="auto",
                    help="which eye's trace to display (default auto=single best-tracked eye; "
                         "'both' only when the eyes differ, e.g. INO)")
    args = ap.parse_args()
    if args.propose:
        propose_init(args.video, args.output_dir)
    elif args.approve or args.auto_approve:
        approve(args.video, args.output_dir, auto=args.auto_approve)
    elif args.review:
        review(args.video, args.output_dir)
    else:
        run(args.video, args.output_dir, filter_mode=args.filter, show_raw=args.show_raw_trace,
            eye=args.eye)


if __name__ == "__main__":
    main()
