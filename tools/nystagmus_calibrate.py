"""CALIBRATION HARNESS — run the nystagmus-direction pipeline on a labelled set of clips and build a
comparison table. For each clip: auto-seed + EllSeg centroid trace -> windowed slow-phase asymmetry +
sustained sclera-balance gaze zones. Captures the per-zone app finding. The clinician fills the EXPECTED
column from their own reading (write it BEFORE looking at the app result).

VNG split-screen goggle clips are excluded (different layout). Long clips are capped to CAP_S seconds.
Output: outputs/_survey/CALIBRATION.md  (one block per clip; fill in `expected:` and compare).
Run with the rit-nets venv python.
"""
import os, sys, subprocess, re
from pathlib import Path
import cv2

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
CAP_S = 15.0                                          # cap each clip to 15 s for speed

# clips to test (exclude the VNG split-screen goggle clips). expected label = clinician's, filled below if known.
CLIPS = [
    ("nystagmus at rest to left in right vestibular neuritis", "primary left-beating"),
    ("nystagmus at rest to left and head impulse positive in right vestibular neuritis", "primary left-beating (expected)"),
    ("gaze-evoked nystagmus in pontine glioma", "gaze-evoked: L on left gaze, R on right gaze; no vertical"),
    ("gaze-evoked nystagmus-1", "gaze-evoked? (confirm)"),
    ("fistula nystagmus", "right-ear pressure induces nystagmus (Hennebert)"),
    ("3", "?"),
    ("2", "?"),
    ("1", "?"),
]

def fps_of(clip):
    c = cv2.VideoCapture(str(REPO / "samples" / f"{clip}.mp4")); f = c.get(cv2.CAP_PROP_FPS) or 30.0; c.release(); return f

out = ["# CALIBRATION — nystagmus direction detector vs clinician\n",
       "For each clip write your **expected** finding first, then compare to the app.  ",
       "`present/absent · direction · gaze zone · gaze effect · confidence`\n"]
for clip, expected in CLIPS:
    fps = fps_of(clip)
    env = dict(os.environ, RIT_CLIP=clip, RIT_MAXF=str(int(CAP_S*fps)))
    print(f"=== {clip}  ({fps:.0f} fps) ===")
    tr = subprocess.run([PY, str(REPO/"tools"/"ellseg_centroid_trace.py")], env=env, capture_output=True, text=True)
    seedline = next((l for l in tr.stdout.splitlines() if l.startswith("seeds:")), "seeds: ?")
    foundline = next((l for l in tr.stdout.splitlines() if "found:" in l), "")
    csv = REPO/"outputs"/f"{clip}_tracked"/"step2_fixed_circle"/"ellseg_centroid.csv"
    app_lines = ["(trace failed)"]
    if csv.exists():
        de = subprocess.run([PY, str(REPO/"tools"/"nystagmus_direction.py"), str(csv), str(fps)], capture_output=True, text=True)
        app_lines = [l.rstrip() for l in de.stdout.splitlines() if l.strip().startswith(("primary", "left", "right", "extreme", "whole-clip"))]
    block = [f"\n## {clip}  ({fps:.0f} fps)",
             f"- **expected (clinician):** {expected}",
             f"- {seedline}  |  {foundline}",
             "- **app finding:**", "```"] + app_lines + ["```"]
    out += block
    print("\n".join(app_lines) if app_lines else "(no app output)")
(REPO/"outputs"/"_survey"/"CALIBRATION.md").write_text("\n".join(out), encoding="utf-8")
print("\nwrote outputs/_survey/CALIBRATION.md")
