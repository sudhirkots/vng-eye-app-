# EyeVNG – Master Development Notes (Current Consensus)

> # 🔑 CODE WORD: **EYEVNG**
> One of the **two canonical EYEVNG docs** (say "EYEVNG" to pull both up):
> **`VISION.md`** — what we're building, requirements, rules, decisions (this file).
> **`HANDOFF.md`** — current status + running session log.
> (`VISION.md` was renamed from the long `PROJECT_VISION_AND_REQUIREMENTS.md`.)

> **Tracking design → see `docs/TRACKING_PHILOSOPHY.md`.** That document is the source of truth for
> all tracking-related design decisions (detector-vs-tracker, template tracking, blink handling,
> re-acquisition, statuses, confidence, overlay, honesty rules). **Clinical reasoning → see
> `docs/CLINICAL_REQUIREMENTS.md`** (why the software behaves as it does). **What we tried + learned
> (development journal) → `docs/DEVELOPMENT_NOTES.md`.** This file holds the overall vision,
> requirements, and build order.

## ⭐ VERSION 1 DESIGN — Single-Eye, Limbus/Iris-Circle Tracking (DEFINITIVE, Dr. Kothari 2026-06-27)

**Motto: "Detect once. Track forever."** Detection is only initialisation; tracking is the
measurement. The clinical signal is continuous tracking of the estimated iris circle/ellipse within
the tracked eye-opening contour.

**Clinical principle.** We measure movement of the eyeball. The iris is rigidly attached to the
eyeball, so tracking the limbus / iris-sclera boundary is sufficient for horizontal/vertical eye
movement. EyeVNG V1 fits or estimates the full iris circle/ellipse from the visible limbus arc, and
the centre of that estimated circle/ellipse is the clinical iris centre. **We do NOT track the pupil,
detect pupil darkness, or use pupil centre as the clinical point.**

**1. Single eye.** V1 tracks ONE user-selected eye (left or right). The other eye may be detected by
MediaPipe in the background but MUST NOT contribute to, average into, or alter the clinical trace.
Binocular (INO, skew, disconjugate gaze) is a future version.

**2. Initialisation — the clinician approves TWO anatomical objects for the selected eye** (and
nothing else for V1: no pupil, no face/nose/tragus/cheek/head-pose). Tracking must not begin until
approved; these become the permanent anatomical reference for the recording:
1. **Eye-opening contour** — one manually approved contour around the visible palpebral fissure /
   eyelid opening outline.
2. **Iris/limbus boundary** — a visible iris-sclera boundary arc plus estimated full iris
   circle/ellipse covering the whole iris.

The medial canthus region, lateral canthus region, upper boundary, and lower boundary are derived
from the tracked eye-opening contour geometry. They are not four independently tracked point
landmarks.

**3. Eye-local coordinate system (the clinical trace).** Defined by the tracked eye-opening contour;
the measured point is the **limbus-derived iris-circle centre**:
- Horizontal: iris-circle centre relative to the contour-derived medial and lateral extents.
- Vertical: iris-circle centre relative to the contour-derived upper and lower extents.
Movement is WITHIN the eye opening, never within the face or frame. The eye-opening contour moves
with the eye region, so head/camera translation is handled locally without a face model or head pose
at this stage.

**4. Tracking hierarchy.**
1. Limbus / iris-boundary tracker = primary clinical tracker.
2. Eye-opening contour tracker = reference-frame tracker.
3. CFT/internal iris features = optional helper only for motion prediction, search stabilization,
   weak-fit support, and consistency checking.
4. MediaPipe = optional helper for initialization/recovery only, not required.
5. Pupil tracking = not part of V1.

After init, track the SAME iris boundary and SAME eye-opening contour continuously. Each frame starts
from the previous state and searches locally. Never re-detect the iris globally every frame.
**MediaPipe is NOT the tracker** — it is used ONLY for initial proposal/recovery/optional quality
checking and must never adjust the clinical trace during normal tracking.

**5. Eye selection / failure.** The user chooses the eye explicitly (`--eye left|right`). **No
automatic eye switching** — if the selected eye fails, ASK before switching.

**6. Blink handling.** Blink/occlusion frames are invalid → GAPS (never connected, interpolated, or
allowed to look like a fast phase). After a blink, resume from the tracked iris; call MediaPipe only
if tracking has truly failed.

**7. Drift guards (a high template score is NOT proof of attachment).** A frame's iris position is
trusted ONLY if the estimated full iris circle/ellipse remains anatomically attached to the visible
limbus / iris-sclera boundary. Reject / mark `drift_suspected` (not valid eye movement) when the
limbus cannot be fit, visible arc coverage is poor, the circle/ellipse fit is inconsistent with the
visible boundary, temporal continuity fails, or a large excursion cannot be visually verified.
Aperture/contour extent checks are coarse diagnostics, not the dominant validity gate.
**The trace is valid only if the overlay proves the iris circle/ellipse stays anatomically attached.**

**8. Version-1 clinical outputs.**
1. Eye-local horizontal iris-circle-centre position vs time
2. Eye-local vertical iris-circle-centre position vs time
3. Overlay: selected eye, visible eye-opening contour, visible limbus arc, estimated full iris
   circle/ellipse, iris-circle centre, state, confidence
4. Raw CSV (per frame): `frame_number, timestamp_ms, time_sec, selected_eye, iris_center_x_raw,
   iris_center_y_raw, iris_radius_or_axes, eye_local_horizontal, eye_local_vertical,
   iris_tracking_status, iris_tracking_confidence, blink_or_occlusion_status, artifact_type`
5. Tracking quality report (incl. blink statistics)

**Success criterion:** the overlay shows the estimated iris circle/ellipse attached to the visible
limbus throughout, and the contour-relative eye-local trace matches what an experienced vestibular
clinician sees on the video.

**Future modules** (only after V1 is stable): V2 head tracking / head-impulse / VOR; V3 binocular /
disconjugate / INO / skew; V4 torsional (iris-texture rotation, using the already-tracked iris).

---

## VNG Display Rules — LOCKED (Dr. Kothari, 2026-06-25)

How the eye-movement trace is shown on/with the video. These are firm requirements:

1. **Single eye by default.** Conjugate nystagmus on both eyes is redundant and confusing — show ONE
   eye. Show BOTH only when the eyes differ, e.g. **internuclear ophthalmoplegia** (one eye nystagmus,
   the other not). CLI: `--eye auto|left|right|both` (default `auto` = single best-tracked eye). The
   CSV always keeps BOTH eyes (data integrity); this rule is about the *display* only.
2. **Trace synced to the video** (a moving time cursor on the trace tracks playback).
3. **No opaque band over the video.** A dark/translucent panel over the face obscures it — not allowed.
4. **The trace must NEVER cover the eyes.** Because the source clips zoom/pan over the eyes, any
   on-video overlay eventually lands on them. RESOLUTION: render the trace in a **dedicated strip BELOW
   the video** (extend the canvas downward; video pixels untouched, eyes always fully visible).
5. **Signal shown = contour-relative ("eye-in-socket")** — iris-circle centre measured against that
   eye's own tracked eye-opening contour. The contour-derived medial/lateral/upper/lower extents
   define the corrected coordinate system.

Implemented in `iris_tracker.py` (`superimpose_traces`, `run`) + `app.py` (`--eye`). Uncommitted.

## Project Vision

EyeVNG is a smartphone-video-based vestibular eye movement analysis platform.

The long-term goal is to approach the functionality of:

* Video Frenzel
* Standard VNG
* Partial vHIT functionality

using ordinary smartphone videos.

The platform should eventually support:

* Spontaneous nystagmus
* Gaze-evoked nystagmus
* Direction-changing nystagmus
* Positional nystagmus
* Head impulse testing
* VOR estimation
* Torsional eye movement analysis
* Vestibular neuritis patterns
* BPPV patterns
* Ocular flutter
* Opsoclonus

However, diagnosis is NOT the current goal.

---

# Current Development Philosophy

The project must be built from the bottom up.

Order of importance:

1. Reliable eye tracking
2. Reliable eye movement traces
3. Reliable torsional tracking
4. Nystagmus detection
5. Head impulse analysis
6. Diagnostic interpretation

No diagnosis should be attempted until the tracking system is trustworthy.

---

# Recording Assumptions

Typical input:

* Android smartphone
* 1080p
* Usually 30 fps
* 60 fps preferred
* 120 fps for head impulse testing when available

Distance:

* Approximately 30–50 cm

Spectacles:

* Usually absent

Recording types:

* Primary gaze
* Gaze left
* Gaze right
* Gaze up
* Gaze down
* Head impulse
* Dix-Hallpike
* Roll test

---

# Critical Realization

MediaPipe is a detector.

MediaPipe is NOT the final tracker.

MediaPipe should locate:

* face
* eyes
* optional eye-opening contour proposals
* optional iris/limbus boundary proposals

But should not determine eye position independently in every frame.

Reason:

Frame-by-frame detection causes the iris marker to jitter or "dance."

This creates artificial eye movements.

---

# Desired Tracking Architecture

MediaPipe
↓
Initial iris detection
↓
User confirmation
↓
Continuous iris tracking
↓
Blink handling
↓
Reacquisition
↓
MediaPipe fallback only if needed

MediaPipe should become a backup system after initialization.

---

# User Initialization Workflow

After video upload:

1. Find first good frame.
2. Show enlarged eye images.
3. Display detected iris circles.
4. Allow user to edit:

   * iris centre
   * iris size

The user must be able to:

* click and drag iris centre
* resize iris circle
* ensure circle covers the entire iris

Store:

Left eye:

* centre x
* centre y
* radius

Right eye:

* centre x
* centre y
* radius

If one eye is unusable:

* continue using the other eye.

---

# Landmark Review and Approval Workflow

Human-confirmed anatomy is the source of truth; AI landmark detection is only a proposal. **Tracking
must not begin until the user approves the selected eye's eye-opening contour and iris/limbus
boundary.** (Core project requirement — see `docs/TRACKING_PHILOSOPHY.md`.)

Workflow:

```
Video Upload
   ↓
Best Frame Selection
   ↓
Landmark Proposal
   ↓
User Review
   ↓
User Correction
   ↓
Landmark Approval
   ↓
Tracking Begins
```

Requirements:

* the eye-opening contour can be corrected
* contour points can be moved, added, or removed as needed to outline the visible palpebral fissure
* the iris/limbus boundary can be moved or resized
* the estimated full iris circle/ellipse can be reviewed against the visible limbus arc
* tracking starts only after approval

Approved anatomy is saved to `approved_landmarks.json` and becomes the initial reference for all
tracking modules.

**Stage 0 must be an INTERACTIVE confirmation screen (not silent auto-detection).** `--approve` opens
the frame, marks and labels the selected eye's proposed eye-opening contour and iris/limbus boundary,
asks the clinician to confirm or correct them, saves to `approved_landmarks.json`, and does **not**
start tracking. Plain `python app.py --video X` then refuses to run without `approved_landmarks.json`,
loads it, and tracks from the approved anatomy (no per-frame re-detection; reacquisition only on lost
confidence).

**Stage 0 controls (LOCKED conceptually):** editing tools must support correcting the eye-opening
contour and iris/limbus boundary. Existing point/radius controls may be preserved for compatibility,
but V1 architecture treats medial/lateral/upper/lower reference values as contour-derived geometry,
not independently tracked landmarks.

# Eye-Local Reference Requirements

The iris circle is the moving object. The eye-opening contour is the moving reference frame.

* Track the approved eye-opening contour frame-to-frame as one anatomical shape.
* Derive the medial limit, lateral limit, upper limit, and lower limit from the tracked contour
  geometry each frame.
* Track the iris/limbus boundary independently from image evidence; estimate the full iris
  circle/ellipse from the visible limbus arc.
* The clinical point is the limbus-derived iris-circle centre, never a pupil centre and never the CFT
  centre by itself.
* CSV must separate raw iris-circle centre, contour-derived eye-local coordinates, reference status
  such as `reference_uncertain`, tracking confidence, blink/lost/reacquired flags, and drift reasons.
* If iris tracking is good but the contour reference is weak, the raw iris trace may remain valid,
  but the corrected relative clinical trace must be marked `reference_uncertain`.
* Overlay may show the raw iris-circle centre, visible limbus arc, estimated full iris circle/ellipse,
  tracked eye-opening contour, contour-derived reference limits, state, and confidence.

---

# Continuous Tracking Requirements

After confirmation:

Track the iris continuously.

Do NOT redetect independently on every frame.

Instead:

* use previous iris position
* use previous iris size
* search locally
* minimize frame-to-frame jitter

Goal:

The marker should appear attached to the iris.

---

# Blink Handling

When the iris disappears:

Possible causes:

* blink
* eyelid occlusion
* motion blur
* tracking failure

Workflow:

1. Mark frame as:

   blink_or_occluded

2. Do not invent iris position.

3. Attempt automatic reacquisition.

4. Use MediaPipe as fallback.

5. Ask user only if reacquisition fails.

---

# Tracking Status

Possible values:

* initialized
* tracked
* uncertain
* blink_or_occluded
* reacquired
* lost

Store status for every frame.

---

# Tracking Confidence

Store:

tracking_confidence

for every frame.

Confidence should be visible on overlay and exported.

---

# Overlay Video Requirements

Overlay must display:

* tracked eye-opening contour
* contour-derived medial/lateral/upper/lower limits
* visible limbus arc
* estimated full iris circle/ellipse
* limbus-derived iris-circle centre
* tracking status
* tracking confidence
* frame number
* timestamp

Colour coding:

Green:

* confident tracking

Yellow:

* uncertain tracking

Blue:

* blink or occlusion

Red:

* tracking lost

Overlay video is considered one of the most important outputs.

It is the primary quality-control tool.

---

# Raw Data Philosophy

Raw data must never be overwritten.

No hidden smoothing.

No silent interpolation.

No invented iris positions.

If tracking fails:

Tracking should fail honestly.

Raw truth is preferred over artificial correction.

---

# CSV Output Requirements

Store:

frame_number
timestamp_ms
time_sec

left_iris_center_x
left_iris_center_y
left_iris_radius

right_iris_center_x
right_iris_center_y
right_iris_radius

tracking_status_left
tracking_status_right

tracking_confidence_left
tracking_confidence_right

face_detected

head_yaw
head_pitch
head_roll

---

# Immediate Outputs Required

For every video:

1. Overlay video
2. Raw CSV
3. Metadata JSON
4. Horizontal trace
5. Vertical trace
6. Tracking quality report

No diagnosis.

No classification.

---

# Metadata JSON Requirements

Store:

* filename
* resolution
* fps
* duration
* frame count
* protocol
* analysis timestamp
* tracking success %
* output files generated

---

# Protocol System

Every video should be tagged as:

* primary_gaze
* gaze_left
* gaze_right
* gaze_up
* gaze_down
* head_impulse
* dix_hallpike_left
* dix_hallpike_right
* roll_test
* custom

Protocol should be stored in metadata.

---

# Eye ROI Requirements

Future torsional analysis requires preservation of eye detail.

Save:

* left eye ROI
* right eye ROI

Also generate:

* left_eye_roi.mp4
* right_eye_roi.mp4

Store ROI coordinates.

Do not aggressively resize.

Preserve image quality.

---

# Torsional Tracking Philosophy

Iris centre alone cannot detect torsion.

Iris tracking provides:

* horizontal movement
* vertical movement

Torsion requires tracking of:

* iris texture
* iris crypts
* limbus
* conjunctival vessels
* surrounding ocular features

Torsional analysis is NOT being implemented now.

Current requirement:

Preserve sufficient image information so torsional analysis can be added later.

---

# Current Development Stage

Current objective:

Upload Video
↓
Confirm Iris
↓
Track Iris Reliably
↓
Handle Blinks
↓
Generate Overlay
↓
Export Raw CSV
↓
Generate Simple Traces

Nothing beyond this stage should be implemented until tracking quality is clinically believable.

---

# Current Success Criterion

A vestibular neurologist watches the overlay video and says:

"The marker is genuinely attached to the iris and follows it smoothly through the recording."

Only after this is achieved should the project move to:

* VNG metrics
* Nystagmus detection
* Head impulse analysis
* Torsional analysis
* Diagnostic interpretation
