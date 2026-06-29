@echo off
cd /d "C:\Users\sudhi\OneDrive\Documents\Neurology Talks\VNG-EYE app"
".venv\Scripts\python.exe" app.py --video "samples\nystagmus at rest to left in right vestibular neuritis.mp4" --rescue --rescue-eye left
pause
