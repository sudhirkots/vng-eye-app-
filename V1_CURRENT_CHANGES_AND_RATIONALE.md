# EyeVNG V1 — Current changes and rationale

Running record of in-flight V1 changes that have not yet landed in the four source-of-truth docs
(`PROJECT_VISION_AND_REQUIREMENTS.md`, `SYSTEM_ARCHITECTURE.md`, `docs/TRACKING_PHILOSOPHY.md`,
`EYEVNG_TRACKING_SPECIFICATION.md`). Update this file alongside the code; once a section here is
mature, fold it into the relevant source-of-truth document and remove it from here.

---

## Detector Correction — False Positive from Velocity Peaks (2026-06-27)

### What happened

* `samples/2_short15.mp4` is the V1 **negative-control / no-nystagmus calibration clip**.
* The V1 nystagmus detector falsely labelled an *Oblique nystagmus (down-left)* event
  (frames 241–358, 6.18–9.17 s, 11 beats, 5.57 Hz, consistency 0.85, confidence 0.73).
* This is a **failed calibration** result: there is no clinical nystagmus in this clip.

### Why the false positive happened

The detector was not seeing true clinical nystagmus. It was counting **small repeated
mathematical velocity peaks in an unreliable contour-relative signal**:

* The horizontal eye-local coordinate was grossly out of range, around `h = 2.34` to `h = 2.72`.
  A healthy contour reference puts the iris-circle centre between `0` and `1` along the
  medial→lateral axis. Values around 2.5 mean the synthesised eye-opening contour (derived
  from the four face landmarks when no contour was approved) is mis-shaped: the iris-circle
  centre is being projected far outside the contour's lateral extent.
* On a near-static signal the velocity median absolute deviation collapses to a tiny number,
  so the velocity-MAD gate (`|v| > 4 × MAD`) drops to its absolute floor and any sub-pixel
  flicker registers as a "fast phase."
* Most of these sub-pixel flickers happened to point in the same diagonal direction (a slow
  drift in image space, projected onto the broken contour axis), so direction consistency
  came out high.
* Inter-flicker intervals were roughly periodic because the refractory-period rule enforces
  uniform spacing on noise — the 5.57 Hz "beat rate" is suspiciously close to `fps / 7`.

### Clinical rule (lock-in)

> **Velocity peaks are not nystagmus.**
>
> Nystagmus is a *visible repeated rhythmic jerk pattern with a consistent fast-phase
> direction*. Random eye movements, isolated saccades, tracking jitter, contour-reference
> error, and small mathematical velocity peaks **must not** be labelled as nystagmus.

### Detector correction (implemented 2026-06-27)

Iris / limbus / eye-opening contour tracking is **unchanged**. Only the detector and the
clinical-output logic were modified:

1. **Input validity gate.** Before analysing a window the detector checks that the
   contour-relative coordinates are anatomically plausible: both `h` and `v` must lie in
   `[-0.5, 1.5]`. If more than `MAX_OUT_OF_RANGE_FRAC` (20 %) of the window's frames fall
   outside this band, the window is rejected as `reference_out_of_range`. This is the change
   that suppresses the negative-control false positive at source.

2. **Minimum fast-phase amplitude floor.** Every velocity peak is checked for actual
   eye-local displacement around the peak frame. Peaks whose displacement is below
   `MIN_FAST_PHASE_AMP_EYELOCAL = 0.05` (5 % of the contour width) are rejected as tracker
   jitter, not clinical beats. When the amplitude filter strips most/all of the velocity
   peaks, the window is rejected with reason `amplitude_below_floor`.

3. **Two-tier output: candidate vs clinical.** Each merged event is at minimum a
   `candidate_event`. To be promoted to `clinically_confirmed = True` and
   `displayed_on_overlay = True` the event must additionally clear:
   * `MIN_CLINICAL_EVENT_DURATION_SEC = 1.5` (sustained over time),
   * `MIN_CLINICAL_EVENT_CONFIDENCE = 0.75`,
   * `MIN_CLINICAL_EVENT_BEATS = 6`.
   Candidates that fail any of these get a `rejection_reason` like
   `not_clinically_confirmed: event_too_short, event_confidence_below_clinical_floor`.

4. **Silent overlay rule.** When no clinically-confirmed event is active, the overlay
   renders **no text and no arrow**. No "No nystagmus detected" caption, no "Doubtful
   nystagmus" label. The only on-video elements are the anatomical `Left`/`Right` labels,
   the iris circle + centre cross, and the cyan eye-opening contour.

5. **Debug candidates preserved.** The full candidate list — passing and failing — is
   written to `nystagmus_events.json` / `nystagmus_events.csv` with the rejection reason
   and per-window scores so the clinician can audit what the detector saw. The CSV per-
   window row still records `rejection_reason` for every window that did not pass.

### Negative-control rule (calibration check)

* `samples/2_short15.mp4` is the current V1 negative-control / no-nystagmus calibration clip.
* The filename is **not hardcoded** into runtime logic. Suppression must come from the rules
  above, not from a filename match.
* The expected results on this clip are:
  * `clinical_nystagmus_detected: false`
  * `clinical_overlay_silent: true`
  * no text and no arrow displayed on the overlay
  * candidate events, if any, recorded only in JSON/CSV with a rejection reason
* If `samples/2_short15.mp4` ever produces a displayed clinical nystagmus label, **the
  detector has failed calibration** and the change that caused it must be reverted or
  corrected before any other detector work proceeds.

### Locked principle

> **Tracking can be sensitive, but clinical labelling must be conservative.**

For V1, false positives are more damaging than missed subtle nystagmus. The detector is
deliberately strict — small subtle nystagmus may be missed until we tune against
known-positive clips.

### Next-step calibration

* Run known-positive nystagmus clips (e.g. spontaneous left-beating from right vestibular
  neuritis, gaze-evoked nystagmus, fistula nystagmus).
* Verify that confirmed clinical labels match the expected anatomical direction.
* Only then consider relaxing any threshold here.
