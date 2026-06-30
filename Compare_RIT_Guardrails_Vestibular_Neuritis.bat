@echo off
REM Side-by-side RIT comparison scrubber for the rest-left vestibular-neuritis clip.
REM LEFT panel  = no RIT guardrails (baseline run)
REM RIGHT panel = with RIT guardrails (current clinical output)

setlocal
set "VIDEO=samples\nystagmus at rest to left in right vestibular neuritis.mp4"
set "GUARDED=outputs\nystagmus at rest to left in right vestibular neuritis_tracked"
set "BASELINE=outputs\nystagmus at rest to left in right vestibular neuritis_tracked_no_guardrails"

REM Auto-create the no-guardrails baseline if missing.
if not exist "%BASELINE%\tracking.csv" (
    echo [launcher] Baseline tracking.csv not found — generating it now ^(takes a few minutes^)...
    py -3.14 iris_tracker.py "%VIDEO%" --engine v1 --no-mediapipe --output-dir outputs ^
        --disable-rit-validator --output-suffix "_no_guardrails"
    if errorlevel 1 (
        echo [launcher] Baseline run failed. Aborting.
        exit /b 1
    )
)

py -3.14 tools\rit_compare_scrubber.py "%VIDEO%" ^
    --guarded-dir "%GUARDED%" ^
    --baseline-dir "%BASELINE%" ^
    --eye BOTH

endlocal
