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

## ⭐ VERSION 1 DESIGN — Single-Eye, IRIS Tracking (DEFINITIVE, Dr. Kothari 2026-06-26)

**Motto: "Detect once. Track forever."** Detection is only initialisation; tracking is the
measurement. The clinical signal is continuous IRIS tracking within the eye opening.

**Clinical principle.** We measure movement of the eyeball. The iris is rigidly attached to the
eyeball, so tracking the iris is sufficient for horizontal/vertical eye movement. **We do NOT track
the pupil separately** (the small pupil marker jitters; the iris has a strong, high-contrast boundary
against the sclera and is larger and steadier). The iris centre is the eye-position measurement point.

**1. Single eye.** V1 tracks ONE user-selected eye (left or right). The other eye may be detected by
MediaPipe in the background but MUST NOT contribute to, average into, or alter the clinical trace.
Binocular (INO, skew, disconjugate gaze) is a future version.

**2. Initialisation — the clinician marks exactly FIVE structures for the selected eye** (and nothing
else: no pupil, no face/nose/tragus/cheek/head-pose). Tracking must not begin until approved; these
become the permanent anatomical reference for the recording:
1. Medial (inner) canthus
2. Lateral (outer) canthus
3. Upper eye margin
4. Lower eye margin
5. **Iris boundary** (a circle/ellipse covering the whole visible iris)

**3. Eye-local coordinate system (the clinical trace).** Defined ONLY by the four eye-boundary
landmarks; the measured point is the **iris centre**:
- Horizontal: iris centre on the medial→lateral canthus axis — **0 = inner canthus, 1 = outer canthus**.
- Vertical: iris centre on the upper→lower margin axis — **0 = upper margin, 1 = lower margin**.
Movement is WITHIN the eye opening, never within the face or frame. The four boundary landmarks move
with the eye, so head/camera translation cancels — no face model or head pose at this stage.

**4. Tracking = follow the marked iris as a single physical object.** After init, track the SAME iris
continuously: each frame starts from the **previous** iris position and searches **locally**. Never
re-detect the iris globally every frame. **MediaPipe is NOT the tracker** — it is used ONLY for
(a) initial eye localisation, (b) recovery after complete tracking loss, (c) recovery after a
prolonged blink/occlusion, (d) optional quality checking. MediaPipe must never adjust the iris
position during normal tracking; the clinical trace must not depend on it frame-to-frame.

**5. Eye selection / failure.** The user chooses the eye explicitly (`--eye left|right`). **No
automatic eye switching** — if the selected eye fails, ASK before switching.

**6. Blink handling.** Blink/occlusion frames are invalid → GAPS (never connected, interpolated, or
allowed to look like a fast phase). After a blink, resume from the tracked iris; call MediaPipe only
if tracking has truly failed.

**7. Drift guards (a high template score is NOT proof of attachment).** A frame's iris position is
trusted ONLY if it is anatomically plausible. Reject / mark `drift_suspected` (not valid eye movement)
when: the iris centre leaves the marked aperture box; the iris radius/size changes substantially;
frame-to-frame motion is physiologically implausible; or a large excursion can't be visually verified.
**The trace is valid only if the overlay proves the iris marker stays anatomically attached.**

**8. Version-1 clinical outputs.**
1. Eye-local horizontal iris-centre position vs time
2. Eye-local vertical iris-centre position vs time
3. Overlay: selected eye, the four eye-boundary landmarks, the **tracked iris boundary**, iris centre, status
4. Raw CSV (per frame): `frame_number, timestamp_ms, time_sec, selected_eye, iris_center_x_raw,
   iris_center_y_raw, iris_radius_or_axes, eye_local_horizontal, eye_local_vertical,
   iris_tracking_status, iris_tracking_confidence, blink_or_occlusion_status, artifact_type`
5. Tracking quality report (incl. blink statistics)

**Success criterion:** the overlay shows a marker that stays attached to the iris throughout, and the
eye-local trace matches what an experienced vestibular clinician sees on the video.

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
5. **Signal shown = canthus-relative ("eye-in-socket")** — pupil measured against that eye's own
   inner+outer canthus (cancels head/"hair" movement). This is the `corrected_*` output.

Implemented in `pupil_tracker.py` (`superimpose_traces`, `run`) + `app.py` (`--eye`). Uncommitted.

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
* iris

But should not determine eye position independently in every frame.

Reason:

Frame-by-frame detection causes the pupil marker to jitter or "dance."

This creates artificial eye movements.

---

# Desired Tracking Architecture

MediaPipe
↓
Initial pupil detection
↓
User confirmation
↓
Continuous pupil tracking
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
3. Display detected pupil circles.
4. Allow user to edit:

   * pupil centre
   * pupil size

The user must be able to:

* click and drag pupil centre
* resize pupil circle
* ensure circle covers the entire pupil

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
must not begin until the user approves the proposed facial landmarks and pupil circles.** (Core
project requirement — see `docs/TRACKING_PHILOSOPHY.md`.)

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

* facial landmarks can be moved
* facial landmarks can be deleted
* facial landmarks can be added
* pupil circles can be moved
* pupil circles can be resized
* tracking starts only after approval

Approved landmarks are saved to `approved_landmarks.json` and become the initial reference for all
tracking modules.

**Stage 0 must be an INTERACTIVE confirmation screen (not silent auto-detection).** `--approve` opens
the frame, marks + labels each proposed point (left pupil, right pupil, nose bridge / central nasal
reference, left/right cheek, and any other stable reference), asks the clinician to confirm each,
lets the clinician click the correct location for any wrong point, saves to `approved_landmarks.json`,
and does **not** start tracking. Plain `python app.py --video X` then refuses to run without
`approved_landmarks.json`, loads it, and tracks from the approved points (no per-frame re-detection;
reacquisition only on lost confidence).

**Stage 0 controls (LOCKED):** the **mouse is used only to drag a point** to move it (pupils and
facial landmarks). The **pupil radius is changed only with the `+` / `-` keys**. No edge-drag or
mouse-wheel resize. (`d` toggles a point off/on; Enter approves; q cancels.)

# Head-Motion Compensation Requirements

Facial landmarks move with the head; pupils move within the eyes — track both, but the facial
landmarks are used **only** to correct for head movement, never to decide the pupil location.

* Track the approved facial landmarks frame-to-frame; they define the head/face reference frame.
* Track pupils independently from the eye image; the face tracker may move the eye **search box** but
  must **never** move/overwrite the pupil result. If the pupil can't be found → lost/blink/reacquire,
  never invented from face motion.
* CSV must separate: `raw_left_pupil_x/y`, `raw_right_pupil_x/y`, tracked `face_landmark_x/y`,
  head reference-frame / face transform (if implemented), `corrected_eye_h`, `corrected_eye_v`,
  `tracking_confidence`, blink/lost/reacquired flags.
* Overlay may show: raw tracked pupils, tracked facial landmarks, the head-reference frame, and the
  corrected gaze trace.

---

# Continuous Tracking Requirements

After confirmation:

Track the pupil continuously.

Do NOT redetect independently on every frame.

Instead:

* use previous pupil position
* use previous pupil size
* search locally
* minimize frame-to-frame jitter

Goal:

The marker should appear attached to the pupil.

---

# Blink Handling

When the pupil disappears:

Possible causes:

* blink
* eyelid occlusion
* motion blur
* tracking failure

Workflow:

1. Mark frame as:

   blink_or_occluded

2. Do not invent pupil position.

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

* face landmarks
* eye landmarks
* iris centres
* tracked pupil circle
* pupil centre
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

No invented pupil positions.

If tracking fails:

Tracking should fail honestly.

Raw truth is preferred over artificial correction.

---

# CSV Output Requirements

Store:

frame_number
timestamp_ms
time_sec

left_pupil_x
left_pupil_y
left_pupil_radius

right_pupil_x
right_pupil_y
right_pupil_radius

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

Pupil centre alone cannot detect torsion.

Pupil tracking provides:

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
Confirm Pupil
↓
Track Pupil Reliably
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

"The marker is genuinely attached to the pupil and follows it smoothly through the recording."

Only after this is achieved should the project move to:

* VNG metrics
* Nystagmus detection
* Head impulse analysis
* Torsional analysis
* Diagnostic interpretation
