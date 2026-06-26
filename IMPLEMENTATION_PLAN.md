# EyeVNG Version 1 Implementation Plan

> **⭐ V1 DESIGN UPDATE (2026-06-26): single-eye, EYE-LOCAL tracking.** V1 = track ONE user-selected
> eye via its confirmed pupil + four eye-boundary landmarks (inner/outer canthus, upper/lower margin);
> eye-local coords (0=inner→1=outer, 0=upper→1=lower); **no face/head landmarks**; MediaPipe is
> proposal/fallback only. Canonical spec: **`VISION.md` → "VERSION 1 DESIGN — Single-Eye, Eye-Local
> Tracking".** Binocular / head-impulse / VOR are future versions.

> Tracking design → `docs/TRACKING_PHILOSOPHY.md` (source of truth). The revised staged order below
> supersedes the older phase list further down for anything tracking-related.

## Revised implementation order (current)

**Tracking must not begin until Stage 0 is complete.** Human-confirmed anatomy is the source of truth.

- **Stage 0 — Landmark proposal and approval**
  - Detect / propose landmarks (MediaPipe) on the best frame
  - User review
  - User correction (move / add / delete facial landmarks; move / resize pupil circles)
  - User approval
  - Save **`approved_landmarks.json`**
- **Stage 1 — Continuous tracking** (from `approved_landmarks.json`)
- **Stage 2 — Blink handling**
- **Stage 3 — Reacquisition**
- **Stage 4 — CSV export**
- **Stage 5 — Overlay generation**

All tracking modules start from `approved_landmarks.json`; MediaPipe is fallback (reacquisition) only.

---

## Files to be created

```text
app.py
requirements.txt
src/api/app.py
src/core/tracking.py
src/core/head_pose.py
src/core/visualization.py
src/core/torsional_tracking.py
src/modules/ingestion/video_ingestion.py
src/modules/measurement/frame_measurement.py
src/modules/export/export_results.py
src/modules/quality/tracking_quality.py
src/modules/visualization/plotting.py
src/modules/reporting/reporting.py
src/modules/calibration/calibration.py
outputs/raw_measurements/
outputs/metadata/
outputs/overlay/
outputs/rois/
outputs/plots/
outputs/reports/
```

## Implementation phases

### Phase 1 – Video ingestion
- Accept an MP4 file path.
- Read video metadata with OpenCV.
- Save metadata JSON to outputs/metadata/<video_name>.json.

### Phase 2 – Face and eye tracking
- Process every frame with MediaPipe Face Mesh.
- Detect face landmarks and eye landmarks.
- Estimate iris centres for both eyes.
- Store frame-level measurements including confidence and face detection status.

### Phase 3 – Head pose
- Estimate head yaw, pitch, and roll from facial landmarks.
- Store these values for every frame.

### Phase 4 – Raw data storage
- Save a CSV containing the raw frame-by-frame measurements.
- Never overwrite the raw CSV.

### Phase 5 – Overlay video
- Render all requested tracking points into each frame.
- Write a saved overlay MP4 file.

### Phase 6 – Eye ROI export
- Crop left and right eye regions for every frame.
- Save ROI videos and ROI coordinate records.
- Store ROI coordinates for future torsional analysis.

### Phase 7 – Tracking quality
- Calculate total frames, analysed frames, lost frames, tracking success percentage, and average confidence.
- Generate a JSON report.

### Phase 8 – Visualization
- Create simple plots of horizontal and vertical eye position versus time using the raw measurements.

### Phase 9 – Sample outputs
- Provide example CSV structure, metadata JSON, tracking report, overlay output, and ROI output references.

## Additional requirements
- No smoothing, filtering, or other processing is performed in Version 1.
- The system will maintain raw measurements separately from any future processed outputs.
- Protocol will be stored in metadata and reports.
- Timestamp milliseconds will be recorded for every frame.
- Eye ROI videos will preserve the original resolution as much as practical.
