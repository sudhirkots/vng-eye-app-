# EYEVNG TRACKING SPECIFICATION

> **Status: DRAFT FOR APPROVAL — 2026-06-26 (V1 primary-output amendment 2026-06-27).**
> Authoritative engineering specification for the EyeVNG Version 1 tracking engine and the V1
> nystagmus detector that consumes it. This is the contract the implementation must satisfy. It is
> not pseudocode and not implementation. Once approved, code is written *from* this document; the
> architecture is frozen (see §J). Supersedes the engine portions of `STABLE_TRACKING.md`;
> consistent with `VISION.md`, `docs/TRACKING_PHILOSOPHY.md`, `MULTIFEATURE_TRACKER_DESIGN.md`, and
> `docs/FEATURE_PROBE_FINDINGS.md`.

> **⭐ V1 PRIMARY OUTPUT (2026-06-27).** The user-facing clinical artefact of V1 is the
> **overlay video carrying a nystagmus arrow + label** when nystagmus is detected, NOT a
> VNG-style position graph. The tracking engine specified in §A–§J produces the per-frame
> measurements; the **nystagmus detector (§K)** consumes the contour-relative iris-centre signal
> from those measurements and emits the clinical result. Position/velocity traces are RETAINED as
> internal analysis input and debug artefacts only.

---

## A. Design philosophy

The tracking engine is governed by these established principles:

1. **One eye (V1).** A single user-selected eye is tracked. The other eye may be detected but must
   never average into, alter, or contribute to the per-eye clinical measurement signal (the
   contour-relative iris-centre series that feeds the V1 nystagmus detector). Binocular work is a
   later version that *reuses* this engine, not a redesign.
2. **Track the limbus / iris boundary, not the pupil.** The iris is rigidly attached to the eyeball,
   so iris motion is eyeball motion. V1 fits or estimates the full iris circle/ellipse from the
   visible limbus / iris-sclera boundary. No pupil concept, pupil centre, pupil darkness, or
   dark-pixel thresholding exists anywhere in the clinical pipeline.
3. **Eye-local coordinate system.** The clinical signal is the centre of the estimated iris
   circle/ellipse expressed relative to the tracked eye-opening contour. The contour's most medial,
   most lateral, highest, and lowest extents derive the coordinate limits. They are not four
   independently tracked point landmarks.
4. **Detect once, track continuously.** Detection initialises the tracker; tracking is the
   measurement. The engine behaves like a clinical video-oculographer: it follows a pattern, it does
   not re-find the eye every frame.
5. **MediaPipe is initialization and recovery only.** It may seed the eye-opening contour and
   iris/limbus boundary at Stage 0 and may assist recovery. It is never required when approved
   landmarks already exist and is never the per-frame clinical signal.
6. **Limbus-first clinical tracking.** The primary clinical object is the estimated iris
   circle/ellipse fitted from the visible limbus arc. The clinical centre comes from this fit.
7. **Eye-opening contour reference.** The reference object is one tracked palpebral fissure contour.
   Medial/lateral/upper/lower values are derived from the contour geometry each frame.
8. **CFT is optional helper only.** Composite/internal iris features may assist prediction, search
   stabilization, weak-fit support, and consistency checking, but the CFT centre is not the primary
   clinical centre.
9. **Drift is detected by loss of anatomical attachment,** primarily limbus unfit, poor visible arc
   coverage, limbus-fit disagreement, temporal discontinuity, or helper-feature inconsistency.
10. **Fast phases must never be mistaken for drift.** A genuine nystagmus fast phase preserves
    anatomical attachment of the iris circle to the visible limbus. Drift loses that attachment or
    violates temporal continuity. The engine separates them by anatomy and coherence, never by motion
    magnitude alone.
11. **Blink frames are invalid, not interpolated.** When the iris is occluded the engine emits no
    position. It never invents, smooths across, or interpolates a gap.

---

## B. Tracker state machine

The engine runs a per-eye state machine. Each frame produces exactly one state. The clinical centre is
the limbus-derived iris-circle/ellipse centre and is emitted only in states marked **VALID OUTPUT**;
all other states emit a gap (centre = None). The contour-relative trace is valid only when the
eye-opening contour reference is also confident; otherwise the raw iris trace may remain valid while
the relative trace is marked `reference_uncertain`.

```
            ┌──────────────┐
            │ INITIALIZING │
            └──────┬───────┘
                   ▼
            ┌──────────────┐   active<target    ┌────────────────────────┐
            │   TRACKING   │ ─────────────────▶ │ REPLENISHING_FEATURES  │
            │ VALID OUTPUT │ ◀───────────────── │  VALID OUTPUT          │
            └──┬───┬───┬───┘   pool restored    └────────────────────────┘
   partial lid │   │   │ composite disagreement
               ▼   │   ▼
   ┌──────────────┐│┌────────────────┐
   │PARTIAL_OCCLU.│││ DRIFT_SUSPECTED │
   │ VALID OUTPUT │││  (gap)          │
   └──────┬───────┘│└───────┬─────────┘
     EAR  │  quorum│        │ persists
     drops│   lost │        ▼
          ▼        ▼   ┌───────────┐      ┌──────────────┐
        ┌─────────┐    │TRACK_LOST │ ───▶ │ REACQUIRING  │ ──▶ TRACKING
        │  BLINK  │ ──────────────────────▶│  (gap)       │
        │  (gap)  │   EAR recovers          └──────────────┘
        └─────────┘
```

### INITIALIZING
- **Purpose:** load the Stage-0 approved eye-opening contour and iris/limbus boundary on the approved
  init frame. Establish the initial limbus model, estimated full iris circle/ellipse, contour model,
  and any optional CFT helper features.
- **Entry:** tracking starts on an approved clip (`approved_landmarks.json` present and approved).
- **Exit:** pool seeded with at least `QUORUM` trust-eligible features → **TRACKING**. If seeding
  fails → **TRACK_LOST**.
- **Outputs:** the approved/fit iris-circle centre (init frame only); contour and optional helper
  state created.
- **Allowed transitions:** → TRACKING, → TRACK_LOST.

### TRACKING  *(VALID OUTPUT)*
- **Purpose:** normal limbus and contour tracking; emit the clinical iris-circle centre and
  contour-relative coordinates.
- **Entry:** from INITIALIZING (seeded), REPLENISHING_FEATURES (restored), REACQUIRING (recovered).
- **Exit:** to PARTIAL_OCCLUSION (reduced visible limbus arc but fit remains usable), BLINK (iris
  boundary unavailable due to lid closure), DRIFT_SUSPECTED (limbus unfit/disagreement/temporal
  discontinuity), REPLENISHING_FEATURES (optional helper pool low), TRACK_LOST (iris boundary cannot
  be recovered and not a blink).
- **Outputs:** iris-circle centre, eye-local (h,v), radius/axes, optional rotation θ, all confidence
  levels, reference status, drift flag = none.
- **Allowed transitions:** → PARTIAL_OCCLUSION, BLINK, DRIFT_SUSPECTED, REPLENISHING_FEATURES, TRACK_LOST.

### PARTIAL_OCCLUSION  *(VALID OUTPUT)*
- **Purpose:** continue tracking while a lid/lash partially covers the iris; fit the visible limbus
  arc and estimate the full iris circle/ellipse.
- **Entry:** from TRACKING when visible arc coverage is reduced but sufficient for a stable
  circle/ellipse estimate. Optional helper-feature loss may support this classification.
- **Exit:** → TRACKING when features recover/replenish and coverage is restored; → BLINK when EAR
  drops below the blink threshold or trusted inliers fall below `QUORUM`.
- **Outputs:** valid iris-circle centre (flagged `partial_occlusion`), reduced frame confidence.
- **Allowed transitions:** → TRACKING, → BLINK.

### BLINK  *(gap — INVALID OUTPUT)*
- **Purpose:** eye closed; emit no position.
- **Entry:** from TRACKING/PARTIAL_OCCLUSION when EAR < blink threshold or the quorum is lost by lid
  closure.
- **Exit:** → REACQUIRING when EAR recovers above the open threshold.
- **Outputs:** centre = None (gap). Pool frozen (birth references preserved; no replenishment).
- **Allowed transitions:** → REACQUIRING.

### REPLENISHING_FEATURES  *(VALID OUTPUT)*
- **Purpose:** top the optional CFT helper pool back up to target by detecting new stable iris
  features inside the current iris circle/ellipse, without interrupting the clinical output.
- **Entry:** from TRACKING when active features < replenish trigger but quorum still holds.
- **Exit:** → TRACKING when the pool is restored to target (or the attempt completes).
- **Outputs:** unchanged valid limbus-derived centre; new helper features added as PROBATION
  (non-voting until trusted).
- **Allowed transitions:** → TRACKING. (May also fall through to BLINK/TRACK_LOST if state changes.)

### DRIFT_SUSPECTED  *(gap — INVALID OUTPUT)*
- **Purpose:** the estimated iris circle/ellipse no longer proves anatomical attachment to the visible
  limbus, or the frame violates continuity/fit criteria (see §F); the frame cannot be trusted.
- **Entry:** from TRACKING/PARTIAL_OCCLUSION when any drift criterion (§F) is met.
- **Exit:** → TRACKING if agreement is restored on the next frame; → REACQUIRING if drift persists for
  `DRIFT_PERSIST` frames; → TRACK_LOST if unrecoverable.
- **Outputs:** centre = None (gap) for the frame; drift flag + reason recorded; raw values still logged.
- **Allowed transitions:** → TRACKING, → REACQUIRING, → TRACK_LOST.

### TRACK_LOST  *(gap — INVALID OUTPUT)*
- **Purpose:** the limbus/iris-boundary model cannot be maintained and it is not attributable to a
  blink (e.g., hand/object occlusion, the eye left the frame, sustained drift).
- **Entry:** from TRACKING/PARTIAL_OCCLUSION/DRIFT_SUSPECTED/INITIALIZING when quorum is lost without
  a blink signature.
- **Exit:** → REACQUIRING (recovery attempt).
- **Outputs:** centre = None (gap).
- **Allowed transitions:** → REACQUIRING.

### REACQUIRING  *(gap — INVALID OUTPUT)*
- **Purpose:** re-establish the eye-opening contour and limbus/iris-boundary model from approved
  anatomy, local image evidence, and optional helpers. MediaPipe may assist if available, but is not
  required when manual approved landmarks are present.
- **Entry:** from BLINK (after reopen), TRACK_LOST, or persistent DRIFT_SUSPECTED.
- **Exit:** → TRACKING on a successful, agreeing re-seed; remains REACQUIRING while attempts fail; →
  TRACK_LOST if recovery is abandoned (timeout).
- **Outputs:** centre = None until reacquired.
- **Allowed transitions:** → TRACKING, → TRACK_LOST.

---

## C. Data structures

### EyeOpeningContour
The tracked reference shape for one selected eye.

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `points` | ordered contour points around the visible palpebral fissure | px | per frame |
| `approved_points` | Stage-0 approved contour on the init frame | px | whole clip |
| `medial_extent`, `lateral_extent` | contour-derived horizontal reference limits | px | per frame |
| `upper_extent`, `lower_extent` | contour-derived vertical reference limits | px | per frame |
| `confidence` | contour attachment/reference confidence | 0..1 | per frame |
| `status` | tracked / uncertain / lost | enum | per frame |

The extents are derived from the contour geometry. They are not independently tracked landmarks.

### LimbusFit
The primary clinical object for one selected eye.

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `visible_arc_points` | visible iris-sclera boundary samples used this frame | px | per frame |
| `ellipse` | estimated full iris circle/ellipse from the visible arc | px | per frame |
| `iris_circle_center` | centre of the estimated full iris circle/ellipse | px | per frame |
| `radius_or_axes` | circle radius or ellipse axes | px | per frame |
| `arc_coverage` | fraction/quality of visible limbus arc | 0..1 | per frame |
| `fit_residual` | visible arc disagreement with the estimated circle/ellipse | px | per frame |
| `confidence` | limbus attachment / fit confidence | 0..1 | per frame |

This is the source of the clinical centre. Pupil centre is not represented.

### IrisFeature
A single optional tracked texture point inside the iris. These features are helpers only; they may
support prediction, stabilization, weak-fit handling, and consistency checking, but do not define the
clinical centre.

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `id` | unique monotonic identifier | int | birth → discard |
| `pos` | current image position (x, y) | px | per frame |
| `birth_pos` | position when created | px | constant for life |
| `birth_offset` | vector from `birth_pos` to the iris centre at birth | px | constant for life |
| `confidence` | reliability estimate (EMA) | 0..1 | per frame |
| `age` | frames survived since birth | frames | per frame |
| `texture` | corner strength at detection (min-eigenvalue) | unitless | constant (or slow update) |
| `fb_error` | latest forward-backward LK error | px | per frame |
| `residual` | distance from the composite-predicted position | px | per frame |
| `state` | PROBATION / TRUSTED / SUSPECT / LOST | enum | per frame |
| `birth_frame`, `last_seen_frame` | bookkeeping | frame index | per frame |

### FeaturePool
The dynamic set of optional CFT helper features for one eye.

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `features` | collection of `IrisFeature` | — | whole clip (per eye) |
| `approved_iris` | Stage-0 iris circle/ellipse from the limbus boundary | px | whole clip |
| `target_size`, `quorum` | desired feature count; minimum trusted inliers to emit a centre | count | constant |
| `counts` | active / trusted / probation / lost tallies | count | per frame |

### Composite
The trusted inlier subset and helper fit it produces this frame (recomputed each frame). Composite
output is a consistency/prediction helper, not the primary clinical centre.

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `voters` | trusted, in-bounds features used this frame | — | per frame |
| `transform` | similarity fit birth→current: translation, rotation θ, scale | px, radians, ratio | per frame |
| `inlier_fraction` | fraction of voters consistent with `transform` | 0..1 | per frame |
| `median_residual` | median voter residual | px | per frame |
| `helper_center` | optional CFT-predicted centre for comparison with the limbus fit | px / None | per frame |

### FrameMeasurement
The per-frame result for one eye.

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `frame_number`, `timestamp_ms`, `time_sec` | timing | int, ms, s | per frame |
| `iris_center` | clinical limbus-derived iris-circle centre, or None | px / None | per frame |
| `raw_iris_center` | raw limbus-derived centre before validity blanking | px / None | per frame |
| `eye_local` | (h, v) relative to the eye-opening contour, or None | 0..1 / None | per frame |
| `iris_radius_or_axes` | estimated iris circle radius or ellipse axes | px | per frame |
| `limbus_confidence`, `arc_coverage`, `fit_residual` | limbus fit quality | 0..1, 0..1, px | per frame |
| `contour_confidence`, `reference_status` | contour reference quality | 0..1, enum | per frame |
| `rotation` | optional helper rotation θ (logged for future torsion only) | radians | per frame |
| `feature_confidence_mean` | mean confidence of trusted features | 0..1 | per frame |
| `composite_confidence` | composite agreement | 0..1 | per frame |
| `frame_confidence` | **principal per-frame quality** | 0..1 | per frame |
| `state` | tracker state (§B) | enum | per frame |
| `drift_flag`, `drift_reason` | drift status + cause | bool, enum | per frame |
| `n_active`, `n_trusted` | pool sizes | count | per frame |
| `ear` | optional eye-aspect-ratio if available; not required without MediaPipe | unitless / None | per frame |

### TrackingStatus
The compact validity/state descriptor attached to each frame: `state` (§B) + `validity`
(valid / gap) + `flags` (partial_occlusion, replenishing, drift, reacquiring).

### EyeMeasurement
The full per-eye time series: the ordered list of `FrameMeasurement` plus per-eye summary statistics
(valid-frame %, drift %, mean frame_confidence, blink events, feature-survival stats). Lifetime: whole clip.

### TrackingReport
The whole-clip report (written to `metadata.json`): clip metadata; the displayed eye; per-eye
`EyeMeasurement` summaries; drift statistics; feature-survival statistics; the frame-confidence
distribution; output file paths; and the exact parameter set used. Lifetime: persisted per run.

---

## D. Per-frame processing pipeline

The processing sequence for each frame (one eye). No implementation detail — sequence only.

1. **Acquire frame.** Read the frame; convert to the working (grayscale) image; record timing.
2. **Predict/search locally.** Use the previous limbus model, previous eye-opening contour, and
   optional CFT helper features to choose local search regions.
3. **Track eye-opening contour.** Update the contour as one shape; derive medial/lateral/upper/lower
   limits from its geometry; compute `contour_confidence` and `reference_status`.
4. **Fit visible limbus arc.** Identify usable visible iris-sclera boundary samples; compute visible
   arc coverage and fit/estimate the full iris circle/ellipse.
5. **Clinical centre.** Emit the centre of the estimated full iris circle/ellipse as the raw clinical
   iris centre. Do not use pupil centre and do not substitute the CFT centre.
6. **Optional helper voting.** Advance and validate internal iris features if available. Use their
   composite motion only for prediction, weak-fit support, and consistency checking against the
   limbus-derived centre.
7. **Confidence calculation.** Compute limbus confidence, contour/reference confidence, optional
   helper confidence, and the principal `frame_confidence` (§E).
8. **Drift detection.** Evaluate limbus attachment, fit quality, visible arc coverage, temporal
   continuity, and helper disagreement (§F). If drift → mark the clinical frame invalid (gap).
9. **Blink / occlusion detection.** Use visible limbus arc loss, frame confidence, helper-feature loss,
   and optional EAR if available; classify TRACKING / PARTIAL_OCCLUSION / BLINK; on blink emit a gap.
10. **Pool maintenance.** Promote/demote/discard optional helper features; if below target and still
   tracking, replenish new features inside the current iris circle/ellipse (PROBATION).
11. **State resolution.** Resolve the frame's state per the state machine (§B).
12. **Output.** Emit the iris-circle centre (or gap), contour-relative coordinates, optional θ, all
    confidence levels, state, reference status, and drift flags; append to the `EyeMeasurement`;
    render the overlay; write the CSV row. If iris tracking is valid but contour tracking is
    uncertain, keep the raw iris centre and mark the relative trace `reference_uncertain`.
13. **Reacquisition.** Only after loss/drift/blink recovery, use approved landmarks and optional
    MediaPipe/helper evidence to re-establish the contour and limbus model. MediaPipe is never a
    normal per-frame clinical anchor.

---

## E. Confidence model

Four nested confidence levels. **`frame_confidence` is the principal quality measure that every future
module (CSV consumers, plots, diagnosis, vHIT, VOR) must read.**

- **`limbus_confidence` [0..1] (per frame):** how strongly the estimated full iris circle/ellipse is
  attached to the visible limbus. Derived from arc coverage, fit residual, boundary contrast, and
  temporal continuity. This is the primary clinical confidence component.
- **`contour_confidence` [0..1] (per frame):** how strongly the tracked eye-opening contour remains
  attached to the visible palpebral fissure. Low contour confidence may mark the relative trace
  `reference_uncertain` even when the raw iris trace remains valid.
- **`feature_confidence` [0..1] (per feature):** how reliable one optional helper feature is. Rises with sustained low
  forward-backward error and low residual vs the composite; falls with disagreement. Interpretation:
  the feature's earned trust as a helper, not as the clinical centre.
- **`composite_confidence` [0..1] (per frame):** how much the trusted features agree this frame.
  Derived from inlier fraction, median residual, and the number of trusted inliers. Interpretation:
  the internal coherence of the helper vote; a consistency input, not the primary validity gate.
- **`frame_confidence` [0..1] (per frame, PRINCIPAL):** the overall trust in this frame's clinical
  measurement. Combines limbus_confidence, visible arc coverage, fit quality, temporal continuity,
  contour/reference confidence, optional composite_confidence, and validity (a gap frame has
  frame_confidence 0). Interpretation: *how much a clinician/algorithm should trust this single data
  point.* Drives shading, quality gating, and the acceptance tests.
- **`tracking_confidence` [0..1] (per clip / rolling):** the aggregate quality of a recording or
  segment — e.g., the proportion of frames whose frame_confidence exceeds a threshold, and the mean
  frame_confidence. Interpretation: is this clip clinically usable, and which segments.

---

## F. Drift model

**Drift is loss of anatomical attachment, detected primarily by the limbus/iris-boundary model -
never by a single correlation/template score and never mainly by an aperture box.** A frame is
`DRIFT_SUSPECTED` (invalid) when any of:

- **Limbus unfit:** the visible limbus / iris-sclera boundary cannot support a circle/ellipse estimate.
- **Poor visible arc coverage:** too little usable boundary remains to prove attachment.
- **Limbus disagreement:** the fitted/estimated circle/ellipse does not match the visible boundary.
- **Temporal discontinuity:** the iris-circle centre, radius, or axes jump in a way not supported by
  the image evidence.
- **Helper disagreement:** optional CFT/internal features strongly disagree with the limbus-derived
  centre or cannot be explained by coherent motion.
- **Feature loss:** optional helper features collapse in a way that supports occlusion/loss.
- **Gross contour/aperture failure:** the iris-circle centre is anatomically impossible relative to
  the eye-opening contour. This is a coarse diagnostic guard, not the dominant validity gate.

**Drift vs genuine fast phase - the central invariant.** A nystagmus fast phase is a large but
anatomically attached motion: the estimated iris circle/ellipse remains fitted to the visible limbus
and moves with temporal continuity. Drift loses that attachment, fit quality, or coherence.
Thresholds must be calibrated so that no anatomically attached coherent fast phase on the reference
clips is ever flagged.

---

## G. Output specification

For each processed frame the engine produces:

### CSV fields (`tracking.csv`)
`frame_number, timestamp_ms, time_sec,
iris_center_x, iris_center_y` (px, blank on a gap),
`raw_iris_center_x, raw_iris_center_y` (px, preserved when available),
`eye_local_x, eye_local_y` (0..1, blank when the relative trace is invalid or `reference_uncertain`),
`iris_radius_or_axes, rotation_theta,
state` (§B), `validity` (valid/gap),
`reference_status, reference_uncertain,
limbus_confidence, arc_coverage, fit_residual, contour_confidence,
`drift_flag, drift_reason,
feature_confidence_mean, composite_confidence, frame_confidence,
n_active, n_trusted, ear`.
Raw (always-detected) values are preserved separately; clinical (valid) values are blank on gaps and
never interpolated.

### Overlay elements (`tracking_overlay.mp4` + debug frames) — V1 PRIMARY clinical artefact

Anatomy / tracking layer (verification):
- The video frame with the eye fully visible. Any debug trace strip, when rendered for
  development, goes **below** the video, never over the eyes.
- The tracked eye-opening contour and its derived medial/lateral/upper/lower extents.
- The visible limbus arc points.
- The estimated full iris circle/ellipse.
- The limbus-derived iris-circle centre marker.
- Optional helper features: **green = trusted/active, amber = probation, red = lost** this frame.
- A state/`drift_reason` banner and the `frame_confidence` value.
- Saved representative debug frames on drift / blink / state changes.

Clinical layer (V1 primary):
- The anatomical eye label ("Left" / "Right") next to the analysed eye.
- When the nystagmus detector (§K) reports nystagmus for the current window: a clear directional
  arrow (← → ↑ ↓ ↗ ↖ ↘ ↙) plus a short label ("Left-beating nystagmus", "Up-beating nystagmus",
  "Oblique nystagmus (up-right)", "Torsional (clockwise)" — torsional only when a rotation signal
  exists; otherwise the overlay shows "Torsion: not assessed").
- When no nystagmus is detected: a small caption such as "No nystagmus detected".
- Arrows and labels MUST be drawn in a region that never covers the eyes.
- A live eye-local trace panel is **debug-only**, opt-in for development; it is not the V1
  clinical display.

### Metadata (`metadata.json` = `TrackingReport`)
Clip info; displayed eye; per-eye summaries; drift statistics (counts by reason); feature-survival
statistics; frame-confidence distribution; the exact parameters used; output paths.

### Quality flags
Per frame: `validity` (valid/gap), `state`, `partial_occlusion`, `replenishing`,
`reference_uncertain`.

### Drift flags
Per frame: `drift_flag` (bool) + `drift_reason` ∈ {limbus_unfit, poor_arc_coverage,
limbus_disagrees, temporal_discontinuity, helper_disagreement, feature_loss, gross_contour_failure}.

### Confidence scores
Per frame: `limbus_confidence`, `contour_confidence`, optional `feature_confidence_mean`,
`composite_confidence`, `frame_confidence` (principal); per clip: `tracking_confidence`.

---

## H. Future compatibility

The engine is designed so later clinical capabilities are *added as consumers of its output*, never a
redesign. The per-frame `FrameMeasurement` already carries the limbus-derived iris-circle centre,
contour-relative eye-local position, optional helper rotation θ, and confidence for each eye.

- **Torsional eye movement:** optional helper features may produce a rotation θ each frame; torsion is
  refined later with an iris-pattern angular check. No V1 clinical-centre change.
- **Head impulse testing (vHIT):** add a head-pose module producing head angular velocity; combine
  with the existing eye velocity (from the eye-local series). The eye signal is already available.
- **VOR gain:** ratio of eye velocity to head velocity over the impulse — both series are produced by
  existing/added modules; the tracker is unchanged.
- **Binocular tracking:** run one independent limbus/contour tracker per eye (the engine is already per-eye); no
  cross-coupling is introduced into either trace.
- **INO (internuclear ophthalmoplegia):** compare the two eyes' eye-local **horizontal** series
  (adduction lag); both eyes are tracked independently and comparably.
- **Skew deviation:** compare the two eyes' eye-local **vertical** series.

All of the above consume `FrameMeasurement`/`EyeMeasurement` and the principal `frame_confidence`; none
requires changing the V1 rule that the clinical centre comes from the estimated iris circle/ellipse.

---

## I. Validation protocol

Every future modification to the tracker is validated against the same artefacts before acceptance:

1. **Overlay video (PRIMARY clinical artefact)** — the estimated iris circle/ellipse must remain
   visibly attached to the real limbus / iris-sclera boundary throughout, AND when the clip
   contains repeated rhythmic jerk nystagmus the overlay must display the correct
   beating-direction arrow within a few beats of onset (and no arrow when the clip contains only
   random saccades, smooth pursuit, drift, or steady gaze).
2. **Eye-local traces** — debug-only horizontal/vertical position plots, with gaps and drift
   ticks. Used to audit the detector input, not as the V1 clinical display.
3. **Known clinical videos** — the reference set: a clean clip (e.g., `2.mp4`) and a hard clip
   (e.g., the fistula clip), plus the nystagmus clips. The nystagmus clips are now the primary
   acceptance set for §K.
4. **Drift statistics** — counts by reason; must not regress (no new false drift on coherent motion).
5. **Reference and helper survival** — contour confidence, limbus fit quality, and optional helper
   pool sizes / survival across the clip.
6. **Frame confidence** — distribution and per-clip `tracking_confidence`; must not regress.
7. **Nystagmus detection accuracy (§K)** — on the reference nystagmus clips the detector must
   report the correct beating direction; on the clean clip it must NOT report nystagmus during
   steady gaze. Must not regress.

**Acceptance rule:** no implementation is accepted unless (a) the overlay remains anatomically
correct — the iris circle/ellipse stays attached to the limbus, the eye-opening contour stays
attached to the palpebral fissure, every drift/blink/reference-uncertain frame is honestly flagged
rather than plotted — AND (b) the nystagmus arrow matches a vestibular clinician's reading on the
reference clips. A fixed verification montage (init, mid, maximum inner excursion, maximum outer
excursion, end, plus a representative nystagmus segment with arrow visible) is the standing
acceptance test.

---

## K. Nystagmus detection (V1 primary clinical output)

The detector consumes the contour-relative iris-centre signal (`eye_local` from
`FrameMeasurement`, §C) of the SELECTED BEST EYE (see best-eye selector in
`iris_tracker.run()`). It emits a per-window classification and a per-clip summary used to drive
the overlay arrow. The detector does NOT alter any tracking output; it only adds analysis fields.

### K.1 Definition (clinical)

Nystagmus, for V1 purposes, is a *repeated, rhythmic, involuntary eye movement pattern with
consistent fast phases in one direction*. Random-looking movements, isolated saccades, gaze
shifts, smooth pursuit, sustained drift, and single beats are NOT nystagmus and must not be
labelled as such.

### K.2 Inputs

- `eye_local` time series (h, v) ∈ [0,1] for the selected eye, with gaps preserved (blink / drift /
  reference_uncertain frames are gaps, not interpolated).
- `frame_confidence` per frame — gates whether a window is analysable.
- `state`, `validity`, `drift_flag` — used to exclude invalid frames from the rhythm analysis.

### K.3 Detection principle

Apply a sliding analysis window over the selected-eye contour-relative signal:

1. **Pre-conditions per window.** Sufficient fraction of valid frames; sufficient mean
   `frame_confidence`; otherwise the window emits `nystagmus_present: false` with reason
   `insufficient_signal`.
2. **Candidate fast phases.** Identify rapid, brief excursions in (h, v) consistent with
   nystagmus fast phases (short rise time, large signed step relative to noise floor, return /
   recovery in the slow phase). Smooth pursuit and gaze shifts are excluded by duration / shape /
   non-repetition.
3. **Repeated beats.** A window must contain **≥ N_beats** candidate fast phases.
4. **Direction consistency.** The fast-phase direction vectors must be aligned (e.g., dominant
   axis identified by their resultant; consistency = fraction of fast phases within a tolerance
   cone around the resultant).
5. **Approximate rhythmicity.** Inter-beat intervals must lie within a tolerance band around their
   median (or, equivalently, the dominant frequency in the fast-phase train must be well-defined).
6. **Rejection.** Failing **any** of (3), (4), (5) → `nystagmus_present: false`. Isolated
   movements, smooth pursuit, drift, and random saccades are rejected by construction.

### K.4 Direction labelling

The window's beating direction is the fast-phase direction (clinical convention). The dominant
axis chosen from the resultant of the fast-phase vectors maps to:

| Resultant direction (eye-local h,v) | Label | Arrow |
|---|---|---|
| dominant −h          | left-beating  | ← |
| dominant +h          | right-beating | → |
| dominant −v          | up-beating    | ↑ |
| dominant +v          | down-beating  | ↓ |
| mixed h&v, +h −v     | oblique up-right | ↗ |
| mixed h&v, −h −v     | oblique up-left  | ↖ |
| mixed h&v, +h +v     | oblique down-right | ↘ |
| mixed h&v, −h +v     | oblique down-left  | ↙ |

Eye-local coordinate convention: `h` increases from medial to lateral extent, `v` increases from
upper to lower extent. The user-facing label is the anatomical direction the eye is beating —
implementations MUST map "image up/down/left/right of the eye-local axes" to the clinical words
("Left-beating", "Up-beating", etc.) using the standard ophthalmic convention. Bare `L` / `R` /
`h` / `v` must never appear in the user-facing overlay.

### K.5 Torsion

Iris-circle-centre motion CANNOT detect torsional nystagmus. Torsion requires iris-texture
rotation tracking inside the iris circle. The detector MUST set
`torsional_status: not_assessed` and the overlay MUST display "Torsion: not assessed" until a
rotation signal is implemented (future module). The detector MUST NOT infer torsion from centre
motion alone.

### K.6 Outputs

Per window:
`window_start_frame`, `window_end_frame`, `nystagmus_present` (bool), `beating_direction` ∈
{left, right, up, down, oblique_up_right, oblique_up_left, oblique_down_right, oblique_down_left,
torsional, none}, `n_beats`, `mean_beat_rate_hz`, `direction_consistency` ∈ [0,1],
`rhythmicity` ∈ [0,1], `confidence` ∈ [0,1], `rejection_reason` (only when `nystagmus_present:
false`: insufficient_signal / too_few_beats / direction_inconsistent / not_rhythmic / random).

Per clip (`nystagmus_report.json`):
`clip_nystagmus_present`, `dominant_beating_direction`, `nystagmus_windows[]`, `analysed_eye`
(anatomical: "Left" / "Right"), `torsional_status: not_assessed`, detector parameters used.

### K.7 Acceptance bar

- On a steady-gaze clip (clean baseline), the detector reports `nystagmus_present: false`.
- On each reference nystagmus clip, the detector reports the correct beating direction during the
  nystagmus segment and stays silent during the steady-gaze segments.
- On the fistula clip, the detector does not invent nystagmus during pure gaze excursions or head
  movement.
- Torsional clips: detector reports `torsional_status: not_assessed`; presence of torsion is
  invisible to V1 (acceptable — it is explicitly out of scope until rotation tracking is added).

### K.8 Parameters

Seed values (`N_beats`, fast-phase amplitude threshold, rhythmicity tolerance, direction tolerance
cone, window length, minimum valid-frame fraction) are calibrated — not redesigned — on the
reference clip set during implementation. They live in a single `NystagmusParams` block and may
be adjusted per protocol (e.g., positional testing vs spontaneous gaze) without changing the
detection structure described above.

---

## J. Development rules

After this specification is approved:

1. **Implementation must follow the specification.** The spec is the contract.
2. **Do not redesign the architecture while writing code.** No silent deviations.
3. **If implementation reveals that an architectural change is required, stop coding and propose the
   change first** — amend this document and obtain approval before continuing.

This document is the master engineering specification for EyeVNG Version 1. Parameter seed values for
limbus fit quality, visible arc coverage, contour confidence, temporal continuity, optional helper
features, blink/occlusion handling, and reacquisition are calibrated - not redesigned - during
implementation. `APERTURE_MARGIN` or equivalent gross-contour checks must remain diagnostic guards,
not the dominant validity gate.
