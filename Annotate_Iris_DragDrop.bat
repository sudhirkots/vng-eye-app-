@echo off
REM Drag-and-drop launcher for the Simple Clinician-Controlled Iris Video Editor.
REM Drop a video file onto this .bat (or double-click and pass the path) and the
REM editor opens on that clip. The active eye is patient anatomical Right.

cd /d "C:\Users\sudhi\OneDrive\Documents\Neurology Talks\VNG-EYE app"

if "%~1"=="" (
    echo No video file supplied.
    echo Drag a video onto this .bat, or run:
    echo     Annotate_Iris_DragDrop.bat "path\to\video.mp4"
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py --video "%~1" --annotate-iris
pause
