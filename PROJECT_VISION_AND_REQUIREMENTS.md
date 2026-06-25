# EyeVNG – Master Development Notes (Current Consensus)

> **Tracking design → see `docs/TRACKING_PHILOSOPHY.md`.** That document is the source of truth for
> all tracking-related design decisions (detector-vs-tracker, template tracking, blink handling,
> re-acquisition, statuses, confidence, overlay, honesty rules). **Clinical reasoning → see
> `docs/CLINICAL_REQUIREMENTS.md`** (why the software behaves as it does). **What we tried + learned
> (development journal) → `docs/DEVELOPMENT_NOTES.md`.** This file holds the overall vision,
> requirements, and build order.

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
