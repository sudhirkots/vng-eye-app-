@echo off
REM Step through the iris-in-orbit overlay and mark problem frames (writes problem_frames.csv).
cd /d "%~dp0"
py -3.14 tools\rit_iris_scrubber.py "outputs\nystagmus at rest to left in right vestibular neuritis_tracked"
