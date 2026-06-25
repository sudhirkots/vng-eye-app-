# EyeVNG Version 1 Architecture

> **Tracking design → see `docs/TRACKING_PHILOSOPHY.md`** — the source of truth for all
> tracking-related design decisions. The measurement/tracking sections below defer to it.

## 1. Goal
EyeVNG is a measurement-first video analysis platform for vestibular eye movement work. Version 1 focuses on reliable extraction of eye and head movement traces from uploaded smartphone videos without any diagnostic or classification logic.

## 2. Design Principles
- Measurement first, diagnosis later.
- Preserve every raw measurement and never overwrite raw data.
- Keep raw and processed measurements separate by design.
- Make all coordinate systems explicit and documented.
- Support future calibration to convert pixel measurements into degrees.
- Keep head pose estimation as an independent module for future vHIT and VOR work.
- Preserve every frame and export eye ROIs for future torsional tracking.

## 3. Explicit Coordinate Conventions
- Pixel coordinate system:
  - Origin at the top-left corner of the image.
  - x increases to the right.
  - y increases downward.
- Image coordinate convention:
  - The image is treated as a 2D array with rows = y and columns = x.
- All tracking outputs are stored in pixel units unless a calibration module later converts them to degrees.
- Raw measurements should preserve the original coordinate values.
- Processed measurements may later be smoothed or transformed, but should remain separate from raw values.

## 4. Proposed Folder Structure

```text
EyeVNG/
├── app.py
├── requirements.txt
├── README.md
├── SYSTEM_ARCHITECTURE.md
├── src/
│   ├── api/
│   │   └── app.py
│   ├── core/
│   │   ├── tracking.py
│   │   ├── head_pose.py
│   │   ├── signal_processing.py
│   │   ├── visualization.py
│   │   └── torsional_tracking.py
│   ├── modules/
│   │   ├── ingestion/
│   │   ├── measurement/
│   │   ├── export/
│   │   ├── quality/
│   │   ├── visualization/
│   │   ├── calibration/
│   │   └── reporting/
│   └── shared/
├── outputs/
│   └── sample/
├── docs/
│   └── diagrams/
└── tests/
```

## Stage 0 – Landmark Initialization (before any tracking)

Human-confirmed anatomy is the source of truth; tracking must not begin until the user approves the
landmarks (see `docs/TRACKING_PHILOSOPHY.md`). Components:

- **MediaPipe proposal engine** — proposes facial landmarks + pupil circles on the best init frame.
- **Landmark review interface** — the user moves / adds / deletes facial landmarks and moves / resizes
  pupil circles; marks unavailable landmarks.
- **Landmark approval interface** — the user explicitly approves the set; nothing tracks before this.
- **Landmark storage module** — persists the approved set.

**Output: `approved_landmarks.json`.** All subsequent tracking modules MUST start from
`approved_landmarks.json` as the initial reference. MediaPipe is fallback only thereafter.

## 5. Data Flow

```text
Upload MP4
  -> Ingestion and validation
  -> STAGE 0: landmark proposal (MediaPipe, best frame)
       -> user review / correction (move, add, delete, resize)
       -> user APPROVAL  ->  save approved_landmarks.json
  -> Continuous TRACKING from approved_landmarks.json (local search, prior position/size)
  -> MediaPipe reacquisition only on failure / blink / occlusion
  -> Head pose estimation
  -> Raw measurement capture
  -> Quality assessment
  -> ROI export per frame
  -> Overlay video generation
  -> CSV export (raw + future processed columns)
  -> Analysis report generation
```

## 6. Module Responsibilities

### Ingestion
- Accept uploaded MP4 files.
- Validate video metadata such as filename, fps, frame count, resolution, and duration.
- Preserve the original video file.

### Measurement (see `docs/TRACKING_PHILOSOPHY.md` — detect once → confirm → track)
- Propose landmarks once with MediaPipe; the user confirms the anatomy (facial landmarks + pupil
  circles). Confirmed structures become tracking targets.
- Process every frame, but **track** the confirmed structures (local search from the previous
  position/size) rather than re-detecting them independently each frame.
- MediaPipe is a backup: reacquisition after tracking failure, blink, or occlusion only.
- Estimate head pose axes.
- Store raw measurements for every frame; never invent positions (blank during blink/occlusion).

### Quality
- Evaluate tracking confidence per frame.
- Record lost frames and tracking success percentage.
- Flag low-confidence segments.

### Export
- Export raw CSV measurements.
- Export eye ROI images per frame.
- Write overlay video with all requested visual verification elements.

### Visualization
- Render facial landmarks, eye landmarks, iris centres, head pose axes, and frame number.
- Make the overlay video suitable for manual verification.

### Calibration
- Provide a future conversion layer from pixels to degrees.
- Keep calibration separate from the tracking pipeline.

### Reporting
- Generate a human-readable analysis summary with FPS, analyzed frames, tracking success, and output files.

## 7. Raw vs Processed Measurements
- Raw measurements are the direct output from frame-by-frame tracking.
- Processed measurements are optional downstream values such as smoothing or filtering.
- The CSV schema will be structured to allow both raw and processed columns.
- Raw data is never overwritten.

## 8. Overlay Video Requirements
The overlay video will show:
- iris centres
- eye landmarks
- face landmarks
- head pose axes
- frame number

This makes visual verification straightforward for clinicians and developers.

## 9. Eye ROI Export Strategy
- For every frame, save a cropped image for the left eye region and the right eye region.
- Store these in an output directory structured by frame number.
- This will support future torsional analysis and other eye-region based work.

## 10. Future Expansion Path
- Calibration module will later convert pixel coordinates to angular degrees.
- Head pose estimation remains independent so it can support head impulse testing and VOR gain estimation later.
- Torsional tracking will use the saved eye ROIs and the existing measurement pipeline as a foundation.
