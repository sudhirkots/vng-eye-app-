# EyeVNG V1 — Design History

> **Historical record only.** This document summarises how the EyeVNG Version 1 tracker reached its
> current design — the decisions taken, the approaches that failed, and the reasoning behind each
> choice. It introduces **no new design ideas**. For the live architecture/contract see
> `EYEVNG_TRACKING_SPECIFICATION.md`; for the running session log see `HANDOFF.md`.

---

## 1. Purpose of the system

EyeVNG extracts an eye-movement (VNG) trace from smartphone / IR nystagmus clips: it follows the eye
through a recording and produces a clinical horizontal/vertical position signal for interpretation
(nystagmus, gaze-evoked nystagmus, fistula, etc.). The guiding clinical principle throughout was that
the tracker must be **visually verifiable** — a clinician should be able to watch the overlay and see
the marker stay on the eye — and that **honest gaps are better than a smooth but wrong trace**.

---

## 2. Major design decisions (chronological)

1. **Detect once, track continuously** — not a per-frame detector. Detection initialises the tracker;
   tracking is the measurement. (Motto: *"Detect once. Track forever."*)
2. **Stage 0 landmark approval** — tracking does not begin until the clinician approves the proposed
   landmarks (saved to `approved_landmarks.json`); the approved set is the reference for tracking.
3. **Eye-local coordinate system** — the clinical signal is the iris position expressed *within* the
   marked eye opening (x: 0 = inner canthus → 1 = outer; y: 0 = upper margin → 1 = lower margin), so
   head/camera translation cancels with no head-pose model.
4. **Track the iris, not the pupil** (see §4).
5. **Single eye for V1** (see §6).
6. **MediaPipe is initialisation / recovery / sanity only** (see §7).
7. **Composite (multi-feature) tracking of the iris** (see §8).
8. **Blinks/occlusion/drift are gaps** — never interpolated, never invented.
9. **The whole iris is tracked as a DISC (limbus boundary)**, with the composite features supplying
   motion only — the final V1 hybrid (see §9).
10. **Segment-based analysis** — only continuous, high-confidence valid segments are analysed; the
    trace is never connected across invalid sections.

---

## 3. Failed approaches (and why each was abandoned)

- **Dark-pupil template / dark-blob centroid.** Drifted catastrophically off the pupil during head
  motion (median ~493 px off on one clip, sliding onto cheek/brow). Low contrast on brown irides made
  the pupil unreliable. Abandoned.
- **Pure optical flow (no anchor).** Temporally smooth but drifted ~187 px off the target over time.
- **MediaPipe iris every frame as the signal.** Stable on the eye but jittered ~5–6 px frame-to-frame;
  any tracker pulled toward MediaPipe each frame inherited that jitter. Marginal in every variant tried.
- **Causal EMA smoothing of the marker.** Lagged and made the marker *slide* behind during head
  movement — worse than raw. Reverted.
- **Temporal / 3-frame-median filtering to remove shimmer.** Erased real nystagmus fast phases (shimmer
  and beats share the same high-frequency band). Conclusion: reduce noise at the source, not by
  temporal filtering.
- **Affine / similarity head-correction from face landmarks.** Removed ~85–93 % of head motion on the
  frames it fitted, but only ~65 % frame coverage under large 3-D head rotation — limited by MediaPipe
  facial-landmark quality. Kept only as a secondary path.
- **Single-template iris tracking.** The chosen object was a single iris template. It side-jumped and
  drifted; on the hard fistula clip it stayed anatomically valid in only ~15 % of frames. This was the
  bottleneck that motivated the composite tracker.
- **Composite-tracker recovery deadlock (a bug, since fixed).** Recovery gated re-seeding on agreement
  with a *stale* frozen centre; on a fast-moving eye it could never catch up and stuck in REACQUIRING
  (~1015/1063 frames on fistula). Fixed by re-anchoring recovery on MediaPipe/anatomy.
- **Free-centre limbus circle fit.** Fitting a circle with a free centre let upper-eyelid edges drag
  the centre up and oversize the circle. Replaced by an arc-based fit that excludes the lid-covered top
  and (under occlusion) holds the radius fixed (see §9).

---

## 4. Why pupil tracking was abandoned

The pupil is small and **low-contrast**, especially against a dark brown iris (measured pupil/iris
contrast ~0.03 on a reference clip). Dark-pupil detection (template match or dark-blob centroid)
drifted badly during motion and produced unstable, sometimes off-eye results. The pupil also adds no
information the iris does not already give for position. The codebase was renamed throughout from
*pupil* to *iris* (`pupil_tracker.py`→`iris_tracker.py`, `PupilDetector`→`IrisDetector`, CSV/JSON keys,
etc.), and all dark-pupil logic was deleted.

---

## 5. Why iris tracking was chosen

- The **iris is rigidly attached to the eyeball**, so iris motion *is* eyeball motion — the correct
  physical quantity for an oculography signal.
- The iris is **larger and more textured** than the pupil, giving many trackable features even in dark
  brown eyes.
- It needs **no dark-pixel thresholding**, removing the pigment/contrast fragility of pupil methods.
- The iris boundary and pattern also carry **rotation (torsion)** information for future versions.

---

## 6. Why single-eye tracking was chosen for V1

- A single user-selected eye is tracked; the other eye may be detected but must **never average into,
  alter, or contribute to** the clinical trace.
- For conjugate nystagmus, plotting both eyes is redundant and cluttered.
- Binocular work (INO adduction lag, skew) is a **later version that reuses this engine per eye**, not
  a redesign — so V1 deliberately keeps one independent, clean per-eye trace.

---

## 7. Why MediaPipe is not the main tracker

- MediaPipe is a **per-frame detector**, not a tracker; its per-frame iris centre **jitters ~5–6 px**,
  and anchoring to it every frame transferred that jitter into the signal.
- MediaPipe Face Mesh **needs a whole face** and fails on the eyes-only close-ups VNG often uses.
- The system's philosophy is *detect once, track continuously*; a per-frame re-find contradicts it.

MediaPipe's role was therefore restricted to: **initialisation** (Stage-0 iris/aperture proposal,
radius sizing), **recovery** (reseed after a blink or complete loss), and an independent **sanity
check** (an off-anchor guard that can invalidate a strayed frame). It does **not** define the per-frame
clinical iris centre, radius, boundary, or confidence. (The one remaining indirect influence — a
periodic creep-correction toward the MediaPipe iris — was removed; the correction now anchors on the
fitted iris boundary instead.) The EAR blink signal is still derived from MediaPipe lid landmarks, but
it gates blink detection only and never positions the iris.

---

## 8. Why Composite Feature Tracking was chosen

The single-template tracker was the proven bottleneck. A read-only feasibility probe showed that even
dark brown irides yield **20–120 trackable texture corners** (feature scarcity is a non-issue), that
sub-pixel optical-flow accuracy is achievable, and that a blink is fully recoverable by re-detection.

The replacement — the **Composite Feature Tracker (CFT)**, originally called the "committee" tracker —
tracks the iris as a **dynamic pool of many weighted features** (each with id, position, confidence,
age, texture, forward-backward error, and trust state). The iris centre is the **robust weighted
consensus** of the trusted features, so **no single feature can move the centre** (a composite, not a
dictator). **Drift is detected by composite disagreement**, not by any single template/correlation
score; a genuine fast phase is coherent (all features move together → still valid), whereas drift is
incoherent (features diverge → invalid). The pool is replenished inside the approved iris after
recovery, and feature rotation is logged for future torsion.

---

## 9. Current final V1 architecture

V1 is a **hybrid** in which the **whole iris is the clinical object, tracked as a disc**:

- **Stage 0** — clinician approves the iris and the four eye-opening landmarks (inner/outer canthus,
  upper/lower margin); saved as the tracking reference.
- **Motion (Composite Feature Tracker)** — each frame, the pool of weighted iris features supplies the
  iris *motion* and the disagreement-based drift signal. Features are a stability aid, not the visible
  output.
- **Boundary (limbus fit)** — each frame, a circle/ellipse is fit to the visible **iris–sclera
  boundary** (radial dark-iris→bright-**sclera** edges, with lid/lash/skin edges rejected by skipping
  the lid-covered top, requiring sustained brightening, and a sclera-whiteness colour gate). The fit is
  seeded by the composite centre and **completes the full circle from whatever arc is visible**.
- **Partial occlusion** — the iris radius is held constant (learned from well-seen frames); when an
  eyelid covers part of the iris, only the centre is solved from the visible arc at that fixed radius,
  so eyelid coverage cannot shrink the disc or shift the centre. Arc coverage drives confidence; too
  small an arc → low confidence / invalid.
- **Emitted clinical signal** — the **fitted iris-disc centre** (not MediaPipe, not the feature dots),
  expressed in eye-local coordinates relative to the four eye-opening landmarks.
- **Validity** — if the boundary cannot be fit, or it disagrees with the feature consensus, or a blink
  is detected, the frame is a **gap** (never rescued or interpolated).
- **Confidence** — per-frame `frame_confidence` integrates composite agreement, occlusion, and
  iris-boundary arc coverage; it is the principal quality metric for downstream modules.
- **Segment-based analysis** — every frame is labelled (valid / blink / occluded / drift_suspected /
  tracking_lost); continuous valid segments of sufficient length and confidence are identified, plotted
  separately (never joined across gaps), and approved or rejected by the clinician; only approved
  segments feed any downstream analysis.
- **MediaPipe** — initialisation, recovery, and sanity only (see §7).

**Acceptance is visual:** the estimated iris circle must remain anatomically attached to the iris
throughout — through saccades, nystagmus, partial eyelid occlusion, and blink recovery — without ever
jumping onto the eyelid, lashes, eyebrow, cheek, or sclera. Statistics are secondary evidence.
