# VNG-EYE app — Setup

Run these three commands inside the `VNG-EYE app/` folder on any Windows machine with Python 3.10–3.12 installed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run on a video

```powershell
.\.venv\Scripts\python.exe app.py "C:\path\to\video.mp4" --output-dir outputs
```

Outputs land in `outputs\`:
- `measurements.csv` — per-frame eye xy, head yaw/pitch/roll, confidence
- `overlay.mp4` — video with face mesh + eye centres drawn

## Notes

- The `.venv\` folder is machine-specific (Windows binaries, absolute paths). Don't sync it — recreate it on each computer with the 3 commands above.
- `.venv\` and `outputs\` are already in `.gitignore` and won't follow via Git/OneDrive.
- Pinned versions in `requirements.txt`: mediapipe 0.10.18 (later versions removed the `mp.solutions` API this code uses), opencv-python <4.12 (numpy 1.x compatibility), numpy <2.
- Python 3.13 is not supported by mediapipe 0.10.18; use 3.10, 3.11, or 3.12.
