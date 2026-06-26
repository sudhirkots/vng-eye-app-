# EYEVNG TRACKING SPECIFICATION

> **Status: DRAFT FOR APPROVAL — 2026-06-26.** Authoritative engineering specification for the
> EyeVNG Version 1 tracking engine. This is the contract the implementation must satisfy. It is not
> pseudocode and not implementation. Once approved, code is written *from* this document; the
> architecture is frozen (see §J). Supersedes the engine portions of `STABLE_TRACKING.md`; consistent
> with `VISION.md`, `docs/TRACKING_PHILOSOPHY.md`, `MULTIFEATURE_TRACKER_DESIGN.md`, and
> `docs/FEATURE_PROBE_FINDINGS.md`.

---

## A. Design philosophy

The tracking engine is governed by these established principles:

1. **One eye (V1).** A single user-selected eye is tracked. The other eye may be detected but must
   never average into, alter, or contribute to the clinical trace. Binocular work is a later version
   that *reuses* this engine, not a redesign.
2. **Track the iris, not the pupil.** The iris is rigidly attached to the eyeball, so iris motion is
   eyeball motion. No pupil concept and no dark-pixel thresholding exists anywhere in the pipeline.
3. **Eye-local coordinate system.** The clinical signal is the iris centre expressed within the
   marked eye opening: x = 0 inner canthus → 1 outer canthus; y = 0 upper margin → 1 lower margin.
   Because the four eye-margin landmarks move with the eye, head/camera translation cancels with no
   head-pose model.
4. **Detect once, track continuously.** Detection initialises the tracker; tracking is the
   measurement. The engine behaves like a clinical video-oculographer: it follows a pattern, it does
   not re-find the eye every frame.
5. **MediaPipe is initialization and recovery only.** It seeds the iris/aperture at Stage 0 and
   re-anchors during recovery. It is never the per-frame clinical signal.
6. **Multi-feature iris tracking.** The tracked object is a pool of many anatomical texture features
   inside the iris, not a single template. (Single-template tracking was proven to drift to ~15%
   anatomical validity on the fistula clip.)
7. **Weighted committee model.** The iris centre is the robust weighted consensus of trusted
   features. No single feature can move the centre — a committee, not a dictator.
8. **Drift is detected by committee disagreement,** not by any single correlation/template score.
9. **Fast phases must never be mistaken for drift.** A genuine nystagmus fast phase is *coherent*
   motion — all features move together (high committee agreement). Drift is *incoherent* — features
   diverge (low agreement). The engine separates them by coherence, never by motion magnitude.
10. **Blink frames are invalid, not interpolated.** When the iris is occluded the engine emits no
    position. It never invents, smooths across, or interpolates a gap.

---

## B. Tracker state machine

The engine runs a per-eye state machine. Each frame produces exactly one state. The clinical centre
is emitted only in states marked **VALID OUTPUT**; all other states emit a gap (centre = None).

```
            ┌──────────────┐
            │ INITIALIZING │
            └──────┬───────┘
                   ▼
            ┌──────────────┐   active<target    ┌────────────────────────┐
            │   TRACKING   │ ─────────────────▶ │ REPLENISHING_FEATURES  │
            │ VALID OUTPUT │ ◀───────────────── │  VALID OUTPUT          │
            └──┬───┬───┬───┘   pool restored    └────────────────────────┘
   partial lid │   │   │ committee disagreement
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
- **Purpose:** build the initial feature pool inside the Stage-0 approved iris, on the approved init
  frame. Establish the rigid reference (each feature's offset to the approved iris centre).
- **Entry:** tracking starts on an approved clip (`approved_landmarks.json` present and approved).
- **Exit:** pool seeded with at least `QUORUM` trust-eligible features → **TRACKING**. If seeding
  fails → **TRACK_LOST**.
- **Outputs:** the approved iris centre (init frame only); pool created.
- **Allowed transitions:** → TRACKING, → TRACK_LOST.

### TRACKING  *(VALID OUTPUT)*
- **Purpose:** normal committee tracking; emit the clinical iris centre and eye-local coordinates.
- **Entry:** from INITIALIZING (seeded), REPLENISHING_FEATURES (restored), REACQUIRING (recovered).
- **Exit:** to PARTIAL_OCCLUSION (localized feature loss, quorum holds), BLINK (EAR below threshold
  or quorum lost to lid closure), DRIFT_SUSPECTED (committee disagreement), REPLENISHING_FEATURES
  (active < replenish trigger), TRACK_LOST (quorum lost, not a blink).
- **Outputs:** iris centre, eye-local (h,v), radius, rotation θ, all confidence levels, drift flag = none.
- **Allowed transitions:** → PARTIAL_OCCLUSION, BLINK, DRIFT_SUSPECTED, REPLENISHING_FEATURES, TRACK_LOST.

### PARTIAL_OCCLUSION  *(VALID OUTPUT)*
- **Purpose:** continue tracking while a lid/lash partially covers the iris; the committee runs on the
  still-visible features.
- **Entry:** from TRACKING when feature loss is spatially localized AND `EAR` is reduced but above the
  blink threshold AND trusted inliers ≥ `QUORUM`.
- **Exit:** → TRACKING when features recover/replenish and coverage is restored; → BLINK when EAR
  drops below the blink threshold or trusted inliers fall below `QUORUM`.
- **Outputs:** valid centre (flagged `partial_occlusion`), reduced frame confidence.
- **Allowed transitions:** → TRACKING, → BLINK.

### BLINK  *(gap — INVALID OUTPUT)*
- **Purpose:** eye closed; emit no position.
- **Entry:** from TRACKING/PARTIAL_OCCLUSION when EAR < blink threshold or the quorum is lost by lid
  closure.
- **Exit:** → REACQUIRING when EAR recovers above the open threshold.
- **Outputs:** centre = None (gap). Pool frozen (birth references preserved; no replenishment).
- **Allowed transitions:** → REACQUIRING.

### REPLENISHING_FEATURES  *(VALID OUTPUT)*
- **Purpose:** top the pool back up to target by detecting new corners inside the approved iris
  boundary (mapped to the current frame), without interrupting the clinical output.
- **Entry:** from TRACKING when active features < replenish trigger but quorum still holds.
- **Exit:** → TRACKING when the pool is restored to target (or the attempt completes).
- **Outputs:** unchanged valid centre; new features added as PROBATION (non-voting until trusted).
- **Allowed transitions:** → TRACKING. (May also fall through to BLINK/TRACK_LOST if state changes.)

### DRIFT_SUSPECTED  *(gap — INVALID OUTPUT)*
- **Purpose:** the committee disagrees (see §F); the frame cannot be trusted.
- **Entry:** from TRACKING/PARTIAL_OCCLUSION when any drift criterion (§F) is met.
- **Exit:** → TRACKING if agreement is restored on the next frame; → REACQUIRING if drift persists for
  `DRIFT_PERSIST` frames; → TRACK_LOST if unrecoverable.
- **Outputs:** centre = None (gap) for the frame; drift flag + reason recorded; raw values still logged.
- **Allowed transitions:** → TRACKING, → REACQUIRING, → TRACK_LOST.

### TRACK_LOST  *(gap — INVALID OUTPUT)*
- **Purpose:** the committee cannot be maintained and it is not attributable to a blink (e.g.,
  hand/object occlusion, the eye left the frame, sustained drift).
- **Entry:** from TRACKING/PARTIAL_OCCLUSION/DRIFT_SUSPECTED/INITIALIZING when quorum is lost without
  a blink signature.
- **Exit:** → REACQUIRING (recovery attempt).
- **Outputs:** centre = None (gap).
- **Allowed transitions:** → REACQUIRING.

### REACQUIRING  *(gap — INVALID OUTPUT)*
- **Purpose:** re-establish tracking using MediaPipe iris + the approved iris boundary; re-detect
  features and require their consensus to agree with the anatomy / last trusted estimate before
  resuming.
- **Entry:** from BLINK (after reopen), TRACK_LOST, or persistent DRIFT_SUSPECTED.
- **Exit:** → TRACKING on a successful, agreeing re-seed; remains REACQUIRING while attempts fail; →
  TRACK_LOST if recovery is abandoned (timeout).
- **Outputs:** centre = None until reacquired.
- **Allowed transitions:** → TRACKING, → TRACK_LOST.

---

## C. Data structures

### IrisFeature
A single tracked texture point inside the iris.

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
| `residual` | distance from the committee-predicted position | px | per frame |
| `state` | PROBATION / TRUSTED / SUSPECT / LOST | enum | per frame |
| `birth_frame`, `last_seen_frame` | bookkeeping | frame index | per frame |

### FeaturePool
The dynamic set of features for one eye (the tracker's persistent state).

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `features` | collection of `IrisFeature` | — | whole clip (per eye) |
| `approved_iris` | Stage-0 iris centre + radius (init frame) | px | whole clip |
| `target_size`, `quorum` | desired feature count; minimum trusted inliers to emit a centre | count | constant |
| `counts` | active / trusted / probation / lost tallies | count | per frame |

### Committee
The trusted inlier subset and the fit it produces this frame (recomputed each frame).

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `voters` | trusted, in-bounds features used this frame | — | per frame |
| `transform` | similarity fit birth→current: translation, rotation θ, scale | px, radians, ratio | per frame |
| `inlier_fraction` | fraction of voters consistent with `transform` | 0..1 | per frame |
| `median_residual` | median voter residual | px | per frame |

### FrameMeasurement
The per-frame result for one eye.

| Field | Meaning | Units | Lifetime |
|---|---|---|---|
| `frame_number`, `timestamp_ms`, `time_sec` | timing | int, ms, s | per frame |
| `iris_centre` | clinical iris centre, or None | px / None | per frame |
| `eye_local` | (h, v) within the aperture, or None | 0..1 / None | per frame |
| `iris_radius` | tracked iris radius | px | per frame |
| `rotation` | committee rotation θ (logged for future torsion) | radians | per frame |
| `feature_confidence_mean` | mean confidence of trusted features | 0..1 | per frame |
| `committee_confidence` | committee agreement | 0..1 | per frame |
| `frame_confidence` | **principal per-frame quality** | 0..1 | per frame |
| `state` | tracker state (§B) | enum | per frame |
| `drift_flag`, `drift_reason` | drift status + cause | bool, enum | per frame |
| `n_active`, `n_trusted` | pool sizes | count | per frame |
| `ear` | eye-aspect-ratio (blink signal) | unitless | per frame |

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
2. **Predict features.** Advance every active feature from the previous frame to this frame by optical
   flow.
3. **Validate features.** Forward-backward consistency, in-bounds, and inside-aperture checks; mark
   failures SUSPECT/LOST.
4. **Committee voting.** From the trusted, in-bounds features, fit the robust consensus transform with
   outlier rejection; identify inliers; weight voters by confidence × age × texture.
5. **Consensus centre.** Apply the consensus transform to the approved iris centre to obtain the
   clinical iris centre; obtain the rotation θ.
6. **Confidence calculation.** Update each feature's confidence; compute committee confidence; compute
   the principal `frame_confidence` (§E).
7. **Drift detection.** Evaluate committee-disagreement criteria (§F). If drift → mark the frame
   invalid (gap).
8. **Blink / occlusion detection.** Evaluate EAR and quorum; classify TRACKING / PARTIAL_OCCLUSION /
   BLINK; on blink emit a gap and freeze the pool.
9. **Pool maintenance.** Promote/demote/discard features; if below target and still tracking,
   replenish new features inside the approved iris boundary (PROBATION).
10. **State resolution.** Resolve the frame's state per the state machine (§B).
11. **Output.** Emit the iris centre (or gap), eye-local coordinates, θ, all confidence levels, state,
    and drift flags; append to the `EyeMeasurement`; render the overlay; write the CSV row.
12. **Re-anchor (periodic).** Only when committee agreement is high, slowly reconcile the consensus
    centre with the MediaPipe iris / aperture to bound long-term optical-flow creep (a correction, not
    a per-frame re-find).

---

## E. Confidence model

Four nested confidence levels. **`frame_confidence` is the principal quality measure that every future
module (CSV consumers, plots, diagnosis, vHIT, VOR) must read.**

- **`feature_confidence` [0..1] (per feature):** how reliable one feature is. Rises with sustained low
  forward-backward error and low residual vs the committee; falls with disagreement. Interpretation:
  the feature's earned trust.
- **`committee_confidence` [0..1] (per frame):** how much the trusted features agree this frame.
  Derived from inlier fraction, median residual, and the number of trusted inliers. Interpretation:
  the internal coherence of the vote; the basis of drift detection.
- **`frame_confidence` [0..1] (per frame, PRINCIPAL):** the overall trust in this frame's clinical
  measurement. Combines committee_confidence, the fraction of the iris visible (occlusion), the number
  of trusted voters, and validity (a gap frame has frame_confidence 0). Interpretation: *how much a
  clinician/algorithm should trust this single data point.* Drives shading, quality gating, and the
  acceptance tests.
- **`tracking_confidence` [0..1] (per clip / rolling):** the aggregate quality of a recording or
  segment — e.g., the proportion of frames whose frame_confidence exceeds a threshold, and the mean
  frame_confidence. Interpretation: is this clip clinically usable, and which segments.

---

## F. Drift model

**Drift is loss of anatomical attachment, detected by committee disagreement — never by a single
template/correlation score.** A frame is `DRIFT_SUSPECTED` (invalid) when any of:

- **Committee disagreement:** inlier fraction of trusted voters < `INLIER_MIN`.
- **Feature divergence:** the trusted features cannot be explained by one consensus transform (they
  split into conflicting motions); median residual > `RESID_MAX` (absolute px and as a fraction of
  iris radius).
- **Feature loss:** trusted inliers fall below `QUORUM` for reasons other than a detected blink.
- **Outside eye aperture:** the consensus iris centre leaves the marked aperture box beyond
  `APERTURE_MARGIN`.
- **Low consensus:** the consensus transform is degenerate (too few inliers, implausible scale or
  rotation).

**Drift vs genuine fast phase — the central invariant.** A nystagmus fast phase is a large but
*coherent* motion: every feature translates together, the consensus transform explains them with a
high inlier fraction and low residual → **high committee_confidence → VALID, never drift.** Drift is
*incoherent*: features diverge, the transform fails, inlier fraction collapses → **low
committee_confidence → invalid.** The discriminator is **coherence, not motion magnitude.** Drift
thresholds must be calibrated so that no coherent fast phase on the reference clips is ever flagged.

---

## G. Output specification

For each processed frame the engine produces:

### CSV fields (`tracking.csv`)
`frame_number, timestamp_ms, time_sec,
iris_center_x, iris_center_y` (px, blank on a gap),
`eye_local_x, eye_local_y` (0..1, blank on a gap),
`iris_radius, rotation_theta,
state` (§B), `validity` (valid/gap),
`drift_flag, drift_reason,
feature_confidence_mean, committee_confidence, frame_confidence,
n_active, n_trusted, ear`.
Raw (always-detected) values are preserved separately; clinical (valid) values are blank on gaps and
never interpolated.

### Overlay elements (`tracking_overlay.mp4` + debug frames)
- The video frame with the eye fully visible (trace strip rendered **below** the video, never over the
  eyes).
- Tracked features: **green = trusted/active, amber = probation, red = lost** this frame.
- The consensus iris centre marker and the tracked iris boundary circle.
- The four approved eye-margin landmarks.
- A state/`drift_reason` banner and the `frame_confidence` value.
- The live eye-local trace panel synced to playback.
- Saved representative debug frames on drift / blink / state changes.

### Metadata (`metadata.json` = `TrackingReport`)
Clip info; displayed eye; per-eye summaries; drift statistics (counts by reason); feature-survival
statistics; frame-confidence distribution; the exact parameters used; output paths.

### Quality flags
Per frame: `validity` (valid/gap), `state`, `partial_occlusion`, `replenishing`.

### Drift flags
Per frame: `drift_flag` (bool) + `drift_reason` ∈ {committee_disagreement, feature_divergence,
feature_loss, outside_aperture, low_consensus}.

### Confidence scores
Per frame: `feature_confidence_mean`, `committee_confidence`, `frame_confidence` (principal); per clip:
`tracking_confidence`.

---

## H. Future compatibility

The engine is designed so later clinical capabilities are *added as consumers of its output*, never a
redesign. The per-frame `FrameMeasurement` already carries iris centre, eye-local position, rotation θ,
and confidence for each eye.

- **Torsional eye movement:** the committee already produces a rotation θ each frame; torsion is read
  directly from θ (refined later with an iris-pattern angular check). No tracker change.
- **Head impulse testing (vHIT):** add a head-pose module producing head angular velocity; combine
  with the existing eye velocity (from the eye-local series). The eye signal is already available.
- **VOR gain:** ratio of eye velocity to head velocity over the impulse — both series are produced by
  existing/added modules; the tracker is unchanged.
- **Binocular tracking:** run one independent committee per eye (the engine is already per-eye); no
  cross-coupling is introduced into either trace.
- **INO (internuclear ophthalmoplegia):** compare the two eyes' eye-local **horizontal** series
  (adduction lag); both eyes are tracked independently and comparably.
- **Skew deviation:** compare the two eyes' eye-local **vertical** series.

All of the above consume `FrameMeasurement`/`EyeMeasurement` and the principal `frame_confidence`; none
requires altering the committee tracker.

---

## I. Validation protocol

Every future modification to the tracker is validated against the same artefacts before acceptance:

1. **Overlay video** — the iris marker must remain visibly stuck to the real iris throughout.
2. **Eye-local traces** — the clinical horizontal/vertical traces, with gaps and drift ticks.
3. **Known clinical videos** — the reference set: a clean clip (e.g., `2.mp4`) and a hard clip
   (e.g., the fistula clip), plus the nystagmus clips.
4. **Drift statistics** — counts by reason; must not regress (no new false drift on coherent motion).
5. **Feature survival** — pool sizes / survival across the clip.
6. **Frame confidence** — distribution and per-clip `tracking_confidence`; must not regress.

**Acceptance rule:** no implementation is accepted unless the overlay remains anatomically correct —
the marker stays on the iris and every drift/blink frame is honestly flagged rather than plotted. A
fixed verification montage (init, mid, maximum inner excursion, maximum outer excursion, end) is the
standing acceptance test.

---

## J. Development rules

After this specification is approved:

1. **Implementation must follow the specification.** The spec is the contract.
2. **Do not redesign the architecture while writing code.** No silent deviations.
3. **If implementation reveals that an architectural change is required, stop coding and propose the
   change first** — amend this document and obtain approval before continuing.

This document is the master engineering specification for EyeVNG Version 1. Parameter seed values
(`POOL_TARGET`, `QUORUM`, `TRUST_AGE`, `CONF_FLOOR`, `FB_REJECT`, `RESID_MAX`, `INLIER_MIN`,
`APERTURE_MARGIN`, `DRIFT_PERSIST`, EMA α, blink/open EAR thresholds, re-anchor cadence) are defined in
`MULTIFEATURE_TRACKER_DESIGN.md` §6 and are calibrated — not redesigned — during implementation.
