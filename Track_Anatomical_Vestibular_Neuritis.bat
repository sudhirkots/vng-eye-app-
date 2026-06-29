@echo off
REM Anatomically-constrained iris tracker dry run on the vestibular-neuritis clip.
REM Uses --engine anatomical (TRACKER_REDESIGN.md). Stage 0 must already be approved.
REM If you have rescue anchors in teaching_anchors.json they will win absolutely on
REM their frames. Non-OK frames are logged to stdout with status + reason.

cd /d "C:\Users\sudhi\OneDrive\Documents\Neurology Talks\VNG-EYE app"
".venv\Scripts\python.exe" app.py --video "samples\nystagmus at rest to left in right vestibular neuritis.mp4" --engine anatomical --no-mediapipe
pause
