# EyeVNG Version 1 Architecture

> **⭐ V1 DESIGN UPDATE (2026-06-27): single-eye, LIMBUS + CONTOUR tracking.** The clinical
> *tracking* uses ONE user-selected eye. The moving object is the estimated full iris circle/ellipse
> fitted from the visible limbus / iris-sclera boundary. The moving reference frame is one manually
> approved and tracked eye-opening contour, not four independently tracked points. MediaPipe is
> proposal/fallback/reacquisition/quality only, never the per-frame signal. Any face/head-landmark
> correction described below is deferred to a future (binocular / head-impulse / VOR) version.

> **⭐ V1 PRIMARY OUTPUT UPDATE (2026-06-27): NYSTAGMUS DETECTION + DIRECTION ARROW.** The
> user-facing clinical output of V1 is NOT a VNG-style position graph. It is a single overlay
> video that answers two questions: *Is nystagmus present?* and *What is the beating direction?*
> (left / right / up / down / oblique; torsional only when iris-texture rotation tracking exists).
> The contour-relative eye-local position trace is retained as an INTERNAL analysis signal +
> debug artefact, not as the primary clinical display. Centre motion can detect horizontal,
> vertical, and oblique nystagmus; torsion requires a separate rotation signal and is reported as
> `not_assessed` until that signal exists.

> **Tracking design → see `docs/TRACKING_PHILOSOPHY.md`** — the source of truth for all
> tracking-related design decisions. The measurement/tracking sections below defer to it.

## 1. Goal
EyeVNG is a video analysis platform for vestibular eye movement work. Version 1's user-facing
clinical output is **nystagmus detection (presence + beating-direction arrow on the overlay
video)** for a single user-selected eye. The contour-relative iris-centre signal feeds the
detector but is not itself the clinical display. V1 does NOT diagnose, classify, or attribute.
Head movement traces and binocular/INO/skew/torsional capabilities are future modules.

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
selected eye's eye-opening contour and iris/limbus boundary (see `docs/TRACKING_PHILOSOPHY.md`).
Components:

- **Optional proposal engine** — may propose the eye-opening contour and iris/limbus boundary on the
  best init frame. MediaPipe may help when available but is not required when approved landmarks
  already exist.
- **Anatomy review interface** — the user corrects the eye-opening contour and iris/limbus
  circle/ellipse; marks unavailable anatomy if needed.
- **Approval interface** — the user explicitly approves the contour and iris boundary; nothing tracks
  before this.
- **Anatomy storage module** — persists the approved set.

**Output: `approved_landmarks.json`.** All subsequent tracking modules MUST start from
`approved_landmarks.json` as the initial reference. MediaPipe is fallback only thereafter.

## 5. Data Flow

```text
Upload MP4
  -> Ingestion and validation
  -> STAGE 0: anatomy proposal/review (optional MediaPipe, best frame)
       -> user review / correction (eye-opening contour + iris/limbus boundary)
       -> user APPROVAL  ->  save approved_landmarks.json
  -> Continuous TRACKING from approved_landmarks.json (limbus tracker + contour tracker)
  -> MediaPipe reacquisition only on failure / blink / occlusion
  -> Raw measurement capture (per frame)
  -> Best-eye selection (anatomical label: Left / Right; image↔patient mapping in metadata)
  -> INTERNAL contour-relative iris-centre signal (analysis input; NOT the clinical display)
  -> NYSTAGMUS DETECTOR
       sliding windows -> candidate fast phases -> require repeated beats
       -> require direction consistency -> require approximate rhythmicity
       -> reject random / isolated movements
       -> output: nystagmus_present + beating_direction + per-segment timings
  -> OVERLAY VIDEO GENERATION (V1 PRIMARY OUTPUT)
       limbus + contour + iris-circle centre + state/confidence
       + when nystagmus is detected: arrow (← → ↑ ↓ ↗ ↖ ↘ ↙) and short label
       + torsion: "Torsion: not assessed" unless a rotation signal exists
  -> Nystagmus detection report (JSON) + CSV (audit trail)
  -> Debug-only trace plots (eye-local h/v, velocity) saved to output directory
```

## 6. Module Responsibilities

### Ingestion
- Accept uploaded MP4 files.
- Validate video metadata such as filename, fps, frame count, resolution, and duration.
- Preserve the original video file.

### Measurement (see `docs/TRACKING_PHILOSOPHY.md` — detect once → confirm → track)
- Propose or load anatomy once; the user confirms the selected eye's eye-opening contour and
  iris/limbus boundary. Confirmed structures become tracking targets.
- Track the limbus / iris-sclera boundary as the primary clinical object and estimate the full iris
  circle/ellipse from the visible arc.
- Emit the limbus-derived iris-circle centre as the clinical centre.
- Track the eye-opening contour as one reference-frame shape. Derive medial/lateral/upper/lower
  values from contour geometry; do not track those as four independent points.
- Use CFT/internal iris features only as optional helpers for prediction, stabilization, weak-fit
  support, or consistency checking.
- MediaPipe is a backup: initialization/reacquisition after tracking failure, blink, or occlusion
  only, and must remain optional.
- Store raw measurements for every frame; never invent positions (blank during blink/occlusion).

### Quality
- Evaluate tracking confidence per frame.
- Record lost frames and tracking success percentage.
- Flag low-confidence segments.
- Drive the best-eye selector (see `iris_tracker.run()` → `eye_selection` metadata block).

### Nystagmus detection (V1 primary clinical module)
- Consume the contour-relative eye-local iris-centre signal of the selected best eye.
- Run in sliding time windows.
- Identify candidate fast phases.
- Require repeated beats, direction consistency, and approximate rhythmicity.
- Reject isolated saccades, smooth pursuit, drift, and random gaze shifts.
- Emit per-window and per-clip results:
  `nystagmus_present`, `beating_direction` ∈ {left, right, up, down, oblique, torsional, none},
  `direction_consistency`, `rhythmicity`, `n_beats`, `mean_beat_rate_hz`, `confidence`,
  `torsional_status` (default `not_assessed` until a rotation signal exists).
- Does NOT diagnose, attribute, or classify central vs peripheral.

### Export
- Export raw CSV measurements (audit trail).
- Export eye ROI images per frame.
- Write the overlay video as the V1 PRIMARY clinical artefact, with anatomy + nystagmus arrow.
- Export the nystagmus detection report as JSON.
- Save debug-only trace plots (eye-local h/v, velocity) to the output directory.

### Visualization
- Render the V1 overlay video:
  - Anatomy layer: eye-opening contour, contour-derived limits, visible limbus arc, estimated full
    iris circle/ellipse, limbus-derived iris-circle centre, state, confidence, frame number.
  - Clinical layer: anatomical eye label ("Left" / "Right"); when nystagmus is detected, a
    directional arrow (← → ↑ ↓ ↗ ↖ ↘ ↙) and a short label ("Left-beating nystagmus", etc.);
    "Torsion: not assessed" until a rotation signal exists.
  - Arrows and labels must NEVER cover the eyes.
- Debug visualisations (eye-local position/velocity plots, debug zoom replay) are kept for
  developer verification and are NOT the primary clinical display.

### Calibration
- Provide a future conversion layer from pixels to degrees.
- Keep calibration separate from the tracking + detection pipeline.

### Reporting
- Generate a human-readable analysis summary with FPS, analysed frames, tracking success, best-eye
  selection record, nystagmus result (presence + beating direction + per-segment timings), and
  output file paths.

## 7. Raw vs Processed Measurements
- Raw measurements are the direct output from frame-by-frame tracking.
- Processed measurements are optional downstream values such as smoothing or filtering.
- The CSV schema will be structured to allow both raw and processed columns.
- Raw data is never overwritten.
- The nystagmus detector runs on the contour-relative (eye-local) signal — itself derived from raw
  iris-circle centre + tracked eye-opening contour. The detector adds per-window classification
  outputs but never modifies the underlying raw values.

## 8. Overlay Video Requirements
The overlay video is the V1 PRIMARY clinical output. It will show:

Anatomy / tracking layer (verification):
- tracked eye-opening contour
- contour-derived medial/lateral/upper/lower limits
- visible limbus arc
- estimated full iris circle/ellipse
- limbus-derived iris-circle centre
- state and confidence
- frame number

Clinical layer (V1 primary):
- anatomical eye label ("Left" / "Right")
- when nystagmus is detected: a directional arrow (← → ↑ ↓ ↗ ↖ ↘ ↙) plus a short label
  ("Left-beating nystagmus", "Up-beating nystagmus", "Oblique nystagmus (up-right)", etc.)
- when no nystagmus is detected: a small "No nystagmus detected" caption
- torsion: "Torsion: not assessed" until a rotation signal exists; never inferred from centre
  motion alone
- arrows/labels in a safe overlay region; the eyes are never covered

This makes both the clinical reading AND the tracking quality verifiable from a single artefact.
Debug-only position / velocity plots, when generated, are kept in the output directory but are
NOT the clinical display.

## 9. Eye ROI Export Strategy
- For every frame, save a cropped image for the left eye region and the right eye region.
- Store these in an output directory structured by frame number.
- This will support future torsional analysis and other eye-region based work.

## 10. Future Expansion Path
- Calibration module will later convert pixel coordinates to angular degrees.
- Head pose estimation remains independent so it can support head impulse testing and VOR gain estimation later.
- Torsional nystagmus detection requires iris-texture rotation tracking inside the iris circle (a
  future module on top of the saved eye ROIs). Until that exists the overlay shows
  "Torsion: not assessed" and the report carries `torsional_status: not_assessed`. Centre motion
  alone CANNOT detect torsion.
- Binocular nystagmus comparison (INO, skew, dysconjugate) reuses the per-eye detector running
  independently on each eye; both-eye mode is opt-in.
