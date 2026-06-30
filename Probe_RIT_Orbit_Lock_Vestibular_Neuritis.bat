@echo off
REM RIT Orbit Lock probe (diagnostic only; no validator/clinical wiring, no commit).
cd /d "%~dp0"
py -3.14 tools\rit_orbit_lock_probe.py "outputs\nystagmus at rest to left in right vestibular neuritis_tracked" 640
echo.
echo Done. See outputs\...\_rit_orbit_lock_probe\ (overlay mp4 + representative_frames + summary).
pause
