@echo off
REM Step through the orbit-lock overlay and mark problem frames (writes problem_frames.csv).
cd /d "%~dp0"
py -3.14 tools\rit_orbit_lock_scrubber.py "outputs\nystagmus at rest to left in right vestibular neuritis_tracked"
