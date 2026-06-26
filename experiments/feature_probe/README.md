# Feature-feasibility probe (READ-ONLY experiments)

These scripts answered one question: **is multi-feature iris tracking feasible on ordinary
smartphone video?** They do NOT touch the tracking pipeline — they only import `IrisDetector`
to read MediaPipe's iris as a reference, detect/track features inside the marked iris, and
report. Nothing here is part of the shipping tracker.

## Files
- `01_feature_probe.py` — first pass (fixed window from frame 0). Exposed a confound: an early
  blink wipes the feature set, so a from-frame-0 window measures blinking, not tracking.
- `02_feature_probe_blinkaware.py` — corrected: finds a blink-free window via an EAR scan,
  measures clean 30-frame survival, and characterises blink reset + re-detection recovery.
- `03_feature_probe_colour.py` — iris-appearance / colour-generalization analysis across all
  sample clips (brightness, contrast, texture energy vs feature yield + survival).
- `results/report_blinkaware.json`, `results/report_colour.json` — raw numbers.
- `results/overlays/` — representative frames (green = surviving feature, red = lost,
  cyan + = consensus centre, magenta x = single-template, amber o = MediaPipe iris).

## How to run (per-machine venv; OneDrive `.venv` does NOT work across machines)
```
# from the VNG-EYE app folder, using the per-machine venv (see SETUP.md / vng_env memory):
%USERPROFILE%\eyevng-venv\Scripts\python.exe experiments\feature_probe\02_feature_probe_blinkaware.py
```
`APP` resolves from the file location (`parents[2]`), so it works regardless of the Windows
user folder. Samples are read from `<repo>/samples/`.

## Headline result
Multi-feature tracking is feasible. Feature scarcity (even on dark brown irides) is a non-issue;
the real challenges are motion and blinks, which are handled by re-detection. See
`docs/FEATURE_PROBE_FINDINGS.md` for the evidence and `MULTIFEATURE_TRACKER_DESIGN.md` for the
resulting architecture.
