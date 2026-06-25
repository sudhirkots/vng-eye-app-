# VNG-EYE app — Setup

> ⚠️ **The project folder is in OneDrive, which syncs across machines — but a venv is machine-specific
> (it hard-codes the base-Python path). A `.venv` created on one PC FAILS on another with
> `No Python at 'C:\Users\<other-user>\...python.exe'`. So put the venv OUTSIDE OneDrive, per machine.**

## Per-machine venv (do this once on each computer)

```powershell
# a local, non-synced venv at %USERPROFILE%\eyevng-venv
python -m venv $env:USERPROFILE\eyevng-venv
$env:USERPROFILE\eyevng-venv\Scripts\python.exe -m pip install -r "requirements.txt"
```

Fast alternative if a working `.venv` already synced in from another PC (same Python 3.12, same OS):
create the empty venv as above, then copy `.venv\Lib\site-packages\*` into
`%USERPROFILE%\eyevng-venv\Lib\site-packages\` (reuses the wheels, no re-download).

Then always use `%USERPROFILE%\eyevng-venv\Scripts\python.exe` (NOT `.\.venv\...`). Launching the GUI:
`Start-Process -FilePath <that python> -ArgumentList 'app.py --video "samples/<file>.mp4" --approve'`
— quote the video path (filenames have spaces).

## Legacy (single-machine) setup

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
