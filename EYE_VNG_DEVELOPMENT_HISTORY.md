# eye_vng Development History

## Purpose of this document

This document records the development history of the eye_vng software from the beginning of the current design cycle. It is written as a clean chronological narrative, not as a raw implementation log.

The aim is to preserve the reasoning behind the major design decisions: what was tried, what failed, what was simplified, and what the current V1 architecture has become.

---

## ⚠ V1 Stage-0 Reminder (locked 2026-06-27)

> **Before any V1 analysis, Stage 0 must validate both the iris/limbus circle and the
> eye-opening/orbital margin contour. Legacy pupil approvals must force re-approval.**

eye_vng V1 tracks the iris/limbus circle (moving object) within the eye-opening/orbital
margin contour (reference frame). Pupil approval is invalid for V1. If a video's
`approved_landmarks.json` lacks `approval_schema_version: "v1_iris_limbus_and_eye_contour"`
or carries the legacy `pupils` block, the V1 runtime refuses to start the tracker and the
detector does not run. The full rule is in §17.

---

## ⚠ Naming decision (locked 2026-06-29): RIT — Rescue Iris Tracker

> The clinically supervised workflow is now named **RIT: Rescue Iris Tracker**.
> RIT means existing limbus tracking constrained by clinician-marked anatomy, strict
> iris-identification rules, and rescue correction when uncertain.

Full workflow name: **RIT with Anatomical Guardrails**. RIT is composed of:

1. clinician-marked orbit / eye-opening oval,
2. clinician-marked eyeball / sclera oval,
3. clinician-marked full iris / limbus circle,
4. existing limbus-based iris tracking (the engine is reused, not rewritten),
5. strict anatomical guardrails (RIT Guardrails),
6. conservative freeze-on-uncertainty,
7. rescue correction when needed (RIT Rescue).

Component names to use in code and docs: `RIT`, `RIT Stage 0`, `RIT Tracking`,
`RIT Guardrails`, `RIT Rescue`, `RIT with Anatomical Guardrails`. This workflow must **not**
be called "V1-plus-anatomy", "V2 tracker", or "anatomical engine". The two source-of-truth
rule documents for RIT are [IRIS_IDENTIFICATION_RULES_V2.md](IRIS_IDENTIFICATION_RULES_V2.md)
and [ORBIT_AND_IRIS_TRACKING_RULES_V2.md](ORBIT_AND_IRIS_TRACKING_RULES_V2.md).

---

## 1. The original clinical idea

The starting idea was simple and clinically attractive:

> Can an ordinary smartphone video be used to detect eye movements and nystagmus in a way that is useful for neuro-otology?

The early goal was to approximate some of the clinical value of VNG / video Frenzel examination using ordinary recorded videos. The software would analyse the eyes in a video and help answer practical clinical questions:

- Is there nystagmus?
- What is the direction?
- Is it left-beating, right-beating, up-beating, down-beating, oblique, or torsional?
- Can we later analyse gaze-evoked nystagmus, positional nystagmus, BPPV patterns, vestibular neuritis, and head impulse videos?

At this stage, the thinking was still close to traditional VNG: track the eye, generate traces, measure movement, and later interpret the pattern.

---

## 2. First technical approach: MediaPipe

The first implementation direction used MediaPipe because it could detect the face, eyes, and iris landmarks quickly.

The early hope was:

- MediaPipe finds the face.
- MediaPipe finds the eyes.
- MediaPipe finds the iris/pupil.
- The software tracks these landmarks frame by frame.
- The movement is converted into VNG-like traces.

MediaPipe was useful at the beginning because it could provide an initial estimate of where the face and eyes were. It helped propose landmarks and gave a quick way to start the project.

But after looking carefully at the video overlays, a major limitation became clear:

> MediaPipe is a detector, not a reliable clinical tracker.

It re-estimates landmarks frame by frame. The resulting markers can jitter, jump, or “dance” even when the eye is not truly moving. This is acceptable for rough face analysis, but not acceptable when small eye movements are clinically important.

This led to the first major design rule:

> MediaPipe may help with initialization or recovery, but it must not be the frame-by-frame clinical measurement source.

---

## 3. Early pupil-tracking idea

The next natural idea was to track the pupil. In many eye-tracking systems, the pupil is the obvious target: dark, circular, and central.

Initially, the software attempted or considered:

- detecting the dark pupil,
- estimating pupil centre,
- using pupil centre movement as the clinical signal.

However, the pupil quickly became a poor target for this application.

The problems were:

- The pupil may be poorly visible in ordinary videos.
- Lighting changes alter the pupil appearance.
- Dark-pupil thresholding is unreliable.
- Eyelids, reflections, blur, and camera exposure interfere.
- The pupil may be small compared with the iris.
- The clinical question is eye movement, not pupil behaviour.

The important clinical realization was:

> We do not need to track the pupil at all.

The eye movement can be estimated from the iris/limbus, because the iris moves with the eyeball. The pupil is not necessary for V1.

This led to the second major design rule:

> eye_vng V1 does not track the pupil.

The software should not detect pupil darkness, segment the pupil, estimate pupil centre, or use pupil centre as the clinical point.

---

## 4. Shift from pupil to iris / limbus

Once pupil tracking was abandoned, the moving anatomical object became the iris.

More precisely, the clinically meaningful object became:

> the iris circle or ellipse defined by the limbus / iris-sclera boundary.

The limbus is the boundary between the iris and the sclera. It is larger and more stable than the pupil. Even when the eyelid covers part of the iris, the visible limbus arc can still be used to estimate the full iris circle or ellipse.

This gave a cleaner V1 tracking target:

- detect or track the visible limbus / iris-sclera boundary,
- fit or estimate the full iris circle/ellipse,
- use the centre of that estimated circle/ellipse as the clinical eye-position point.

This was a major simplification.

The new rule became:

> Track the iris circle. The centre of that circle is the eye-position point.

The centre of the iris circle is not called the pupil centre. It is better named:

- iris centre,
- limbus centre,
- iris-circle centre,
- clinical eye centre.

---

## 5. The problem with raw image-space movement

Once the iris centre was being tracked, another problem became obvious.

If the software reports only the raw iris centre in the video frame, then the signal includes:

- true eye movement,
- head movement,
- camera movement,
- slight body movement,
- video framing changes.

Clinically, we do not want “where the eye is in the image.” We want:

> where the iris is within the eye opening.

Therefore, the iris circle centre must be measured relative to a local reference frame.

This led to the need for an eye-opening reference.

---

## 6. First reference idea: four eye-opening points

The first reference model used four manually marked points:

- medial canthus,
- lateral canthus,
- upper lid margin,
- lower lid margin.

The idea was:

> Measure iris movement relative to these four points.

This would allow horizontal movement to be measured relative to the medial-lateral axis and vertical movement relative to the upper-lower axis.

However, visual review showed that the four points themselves were difficult to track reliably. They are not clean objects like the iris. They are soft-tissue regions, skin folds, shadows, and deforming lid margins.

The problem was not the clinical idea. The problem was treating the reference as four fragile single pixels.

The lesson was:

> The eye opening is not four independent dots. It is one anatomical shape.

---

## 7. Shift from four points to the eye-opening contour

The reference model was then changed.

Instead of tracking four separate points, the software should track the visible eye-opening contour:

- the palpebral fissure,
- the eyelid-opening outline,
- the almond-shaped contour around the visible eye.

From that contour, the software can derive:

- the most medial extent,
- the most lateral extent,
- the highest extent,
- the lowest extent.

These are not independently tracked points. They are derived from the tracked contour shape.

This became the third major design rule:

> The eye-opening contour is the moving reference frame.

The final V1 tracking model became:

> Iris circle = moving object.  
> Eye-opening contour = moving reference frame.  
> Clinical measurement = iris-circle centre relative to the tracked contour.

This is the core of the current eye_vng V1 architecture.

---

## 8. Composite Feature Tracking: useful, but not primary

During the development, Composite Feature Tracking was also explored.

The idea was to track multiple small features within the iris texture, rather than relying on a single point. This could help stabilize motion estimation and detect when features move coherently.

This was useful as a technical helper, but it created a risk: the software could begin to treat internal feature clusters as the main clinical object.

The clinical correction was:

> Internal iris features may help, but the visible anatomical object is the iris/limbus circle.

Therefore, Composite Feature Tracking was demoted to an optional helper.

The current hierarchy is:

1. Limbus / iris-boundary tracker = primary clinical tracker.
2. Eye-opening contour tracker = reference-frame tracker.
3. Composite/internal iris features = optional helper only.
4. MediaPipe = optional initialization/recovery helper only.
5. Pupil tracking = not part of V1.

---

## 9. No-MediaPipe mode

A practical problem occurred when the runtime environment could not use MediaPipe reliably. The software was blocked because `IrisDetector()` still attempted to instantiate MediaPipe even when manually approved landmarks were already available.

This led to an important architectural correction:

> MediaPipe must be optional.

If the user has already approved the iris and eye-opening reference, the tracker should run without MediaPipe.

No-MediaPipe mode allows the V1 tracker to use:

- approved iris/limbus boundary,
- approved or derived eye-opening contour,
- limbus tracking,
- contour tracking,
- blink/occlusion/drift logic based on the tracker itself.

MediaPipe may remain useful later for:

- initial landmark proposal,
- recovery after tracking loss,
- optional sanity checks.

But it must not be required for V1 clinical tracking.

---

## 10. Best-eye selection

Another practical decision came from visual review of both-eye tracking.

The software may track both eyes internally, but in routine nystagmus analysis, a poorly tracked second eye should not spoil the clinical result.

For most routine tasks, one high-quality eye is enough.

Therefore, V1 now uses automatic best-eye selection by default.

The software compares left and right tracking quality using:

- iris-valid percentage,
- contour-reference-valid percentage,
- clinical-relative-valid percentage,
- frame confidence,
- drift or invalid frame count.

It then selects the better-tracked eye for the main clinical output.

Both-eye data are still preserved for review.

Both-eye analysis remains important for special situations such as:

- internuclear ophthalmoplegia,
- skew deviation,
- ophthalmoplegia,
- dysconjugate gaze,
- strabismus-related questions,
- research/debug analysis.

But the default routine workflow is:

> Track both if available, but clinically use the better eye.

---

## 11. Anatomical eye labels

A nomenclature issue then became clear.

In a front-facing video, the eye on the right side of the image is usually the patient’s anatomical left eye, and the eye on the left side of the image is the patient’s anatomical right eye.

Therefore, the internal image-side labels can be misleading.

The software now separates:

- image-left eye,
- image-right eye,
- patient-left eye,
- patient-right eye.

For a standard front-facing video:

- image-left eye = patient’s right eye,
- image-right eye = patient’s left eye.

The metadata preserves this mapping.

However, for clinical overlay, the display should remain simple:

- show “Left” above the patient’s anatomical left eye,
- show “Right” above the patient’s anatomical right eye.

The overlay should not show long labels like “image right / patient left.” That technical mapping belongs in metadata, not on the clinical video.

---

## 12. From VNG-style graphs to clinical video overlay

The early design still carried the idea of generating VNG-like traces. But as the project matured, the clinical goal became clearer.

This smartphone-based system does not need to reproduce a full laboratory VNG graph.

The practical V1 clinical question is simpler:

> Is nystagmus present?  
> If yes, what is the beating direction?

Therefore, VNG-style traces were demoted.

The software may still compute internal time-series data and save debug traces, but those are no longer the main user-facing output.

The main V1 output should be:

- video overlay,
- clean eye labels,
- iris circle,
- eye-opening contour,
- nystagmus arrow and label only when clinically detected.

This led to the fourth major design rule:

> VNG traces are internal/debug signals, not the main clinical display.

---

## 13. Arrow-based nystagmus display

The current primary output is an overlay that answers the clinical question directly.

If nystagmus is detected, the overlay should show a direction arrow and label:

- left-beating nystagmus,
- right-beating nystagmus,
- up-beating nystagmus,
- down-beating nystagmus,
- oblique nystagmus.

If torsional tracking is not implemented, torsion must not be inferred from centre movement alone. Torsional nystagmus requires iris texture rotation tracking inside the iris circle.

Therefore, for now:

> Torsion is not assessed unless a reliable iris-rotation signal is implemented.

The overlay should be clean:

- Left / Right labels,
- iris circle,
- iris centre,
- eye-opening contour,
- no technical clutter,
- no frame number,
- no confidence values,
- no trace strip,
- no “No nystagmus detected” text.

If there is no nystagmus, the clinical overlay should be silent.

---

## 14. Conservative nystagmus detection

A crucial lesson came from testing a negative-control clip.

The detector falsely labelled oblique/downbeat nystagmus in a video where there was no clinical nystagmus.

The reason was not that the eye clearly jerked. The detector had counted small repeated mathematical velocity peaks in an unreliable contour-relative signal.

In the false-positive segment, the horizontal eye-local coordinate was grossly out of range, around:

```text
h = 2.34 to 2.72
```

This is not anatomically meaningful. The eye-local coordinate should normally be around:

```text
0 to 1
```

or within a conservative tolerance around that range.

Because the relative signal was unreliable, small tracking or contour-reference fluctuations were treated as “fast phases.” Since they happened to point in a similar diagonal direction, the detector labelled them as oblique nystagmus.

Clinically, there were no visible repeated jerks.

This produced an important rule:

> Velocity peaks are not nystagmus.

Nystagmus requires:

- visible repeated rhythmic jerks,
- a consistent fast phase direction,
- enough beats,
- enough amplitude,
- a valid anatomical reference signal.

The detector must therefore be conservative.

The current clinical rule is:

- fewer than 3 rhythmic jerks: no label,
- 3–5 rhythmic jerks: doubtful or candidate only,
- more than 5 clear rhythmic jerks with consistent fast phase: possible nystagmus,
- candidate events are saved for debug review,
- only clinically confirmed events are displayed on the overlay.

If there is no clinically confirmed nystagmus, the overlay should show no nystagmus text at all.

---

## 15. Candidate events versus clinical output

The false-positive episode led to another important separation.

The detector may produce candidate events internally. These can be useful for debugging and calibration.

But candidate events should not automatically appear on the clinical overlay.

The system should separate:

### Debug candidates

These are possible rhythmic-looking signal events.

They may be saved in:

- JSON,
- CSV,
- debug reports.

They may include:

- time window,
- beat count,
- direction estimate,
- confidence,
- rejection reason.

### Clinical output

These are stricter confirmed events.

Only clinical-output events should produce:

- nystagmus label,
- arrow,
- displayed direction.

This separation is essential.

> Tracking can be sensitive, but clinical labelling must be conservative.

---

## 16. Detector Correction After Negative-Control Failure

After §14 and §15 had tightened the detector and split candidate from clinical output, we ran
the conservative detector on `samples/2_short15.mp4`. The result was a documented failure that
forced a further correction.

* `samples/2_short15.mp4` was confirmed clinically as a **negative-control / no-nystagmus clip**.
  There is no clinical nystagmus in this video.
* The earlier detector (the version that ended §15) still **falsely labelled oblique/downbeat
  nystagmus** on this clip — frames 241–358, ~3 s of "Oblique nystagmus (down-left)",
  11 beats, 5.57 Hz, consistency 0.85.
* The false positive happened because the detector counted **small repeated velocity peaks in
  an unreliable contour-relative signal**. It was not seeing visible jerks; it was seeing
  mathematical micro-fluctuations.
* In the false-positive segment, the **horizontal eye-local coordinate was grossly out of
  range, around `h = 2.34` to `2.72`**. A healthy contour-relative signal lives roughly in
  `[0, 1]`. Values around 2.5 mean the synthesised eye-opening contour fallback (used when no
  contour was approved at Stage 0) was mis-shaped, and the iris-circle centre was being
  projected far outside the contour's lateral extent. The contour-relative reference signal
  was simply not anatomically valid in that segment.
* On a near-static signal the velocity median-absolute-deviation collapsed, so the
  velocity-MAD gate fell to its absolute floor and any sub-pixel flicker registered as a
  "fast phase." Most of those flickers happened to point in the same diagonal direction
  (slow image-space drift on a broken axis), so direction consistency came out high, and the
  refractory rule made the inter-flicker intervals look roughly periodic.

### Correction

The correction touched only the detector and the clinical-output logic. Iris, limbus, and
eye-opening contour tracking were not changed.

* **Detector-input validity gate.** Windows whose contour-relative coordinates fall outside
  the acceptable anatomical range (`h` and `v` in `[-0.5, 1.5]`) for too many frames are now
  rejected as `reference_out_of_range`. Out-of-range data does not enter peak picking.
* **Minimum fast-phase amplitude floor.** Velocity peaks whose actual eye-local displacement
  is below `MIN_FAST_PHASE_AMP_EYELOCAL = 0.05` (5 % of the contour width) are rejected as
  tracker jitter rather than counted as beats. Stripped windows are recorded as
  `amplitude_below_floor`.
* **Two-tier output separation made explicit.** The detector now produces:
  * `debug_candidates` — possible rhythmic-looking events, saved to JSON/CSV with their
    rejection reason, not displayed on the clinical overlay,
  * `clinical_output` — only events that additionally clear the clinical-confirmation gates
    (duration, beat count, confidence) become `clinically_confirmed = True` and
    `displayed_on_overlay = True`.
* **Clinical overlay is silent when there is no clinically confirmed nystagmus.** No "No
  nystagmus detected" caption, no "Doubtful nystagmus" text, no arrow. Anatomical Left/Right
  labels, the iris circle, the iris centre, and the eye-opening contour remain — nothing else.

### Rerun result on the negative-control clip

After the correction, running:

```
python app.py --video samples/2_short15.mp4 --no-mediapipe
```

produced:

* `clinical_nystagmus_detected: false`
* `clinical_overlay_silent: true`
* `candidate_events_count: 0`
* all 26 sliding windows rejected as `reference_out_of_range`

The overlay video carried no text and no arrow. The previous false-positive
oblique/downbeat label was fully suppressed at the source.

### Negative-control calibration rule

* `samples/2_short15.mp4` is the current V1 negative-control / no-nystagmus calibration clip.
* The filename is intentionally not hardcoded into runtime logic; suppression comes from the
  validity gate, the amplitude floor, and the clinical-confirmation gates.
* If this clip ever produces a displayed clinical nystagmus label, the detector has failed
  calibration and the offending change must be corrected before any other detector work
  proceeds.

### Principle reinforced

> **Tracking can be sensitive, but clinical labelling must be conservative.**

For V1, false positives are more damaging than missed subtle nystagmus. Subtle real nystagmus
may be missed until the detector is calibrated against known-positive clips, but the overlay
must not invent nystagmus the clinician is not seeing.

---

## 17. Stage 0 Anatomical Approval Requirement

### Legacy Pupil Approval Error — Mandatory Stage 0 Anatomical Approval

> **Principle: V1 tracks the iris/limbus and the eye-opening/orbital margin. Pupil approval
> is invalid for V1.**

Early versions of this software marked the pupil — the small dark disc inside the iris —
because the pupil is what older video-oculographers used as the moving point. V1 no longer
uses pupil tracking. V1 tracks the full iris/limbus (the coloured iris out to the white
sclera) as the moving object, and the eye-opening/orbital margin contour (the palpebral
fissure outline) as the reference frame.

A known-positive nystagmus clip
(`samples/nystagmus at rest to left in right vestibular neuritis.mp4`) failed during V1
testing because its old `approved_landmarks.json` was in the legacy schema:

* `iris: None`
* `pupils` block present, with tiny radii (approximately 8.56 px and 5.0 px)
* no true eye-opening contour

The code silently treated that legacy `pupils` block as if it were an iris approval:

```python
iris_appr = approved.get("iris") or approved.get("pupils") or {}
```

Result:

* The V1 tracker was seeded with **pupil-sized** circles (5–9 px) where it needed
  full **iris/limbus** circles (15–25 px).
* The full iris/limbus was therefore not properly marked.
* The iris circle was not visible or stable on the overlay.
* The tracker failed or became very weak on most frames of the clip.
* The contour-relative iris-centre signal was invalid or absent because the synthesised
  eye-opening contour was driven by four loose face landmarks rather than a real
  palpebral fissure outline.
* The known-positive left-beating vestibular-neuritis clip therefore produced a
  misleading "no clinical nystagmus" result — not because the detector was wrong, but
  because Stage 0 had silently given V1 the wrong anatomy to track.

This must not happen again.

#### New rules (locked)

1. **V1 must never silently use legacy pupil approvals.** Reading `approved.get("pupils")`
   as an iris seed is forbidden in all runtime paths. The runtime gate rejects any approval
   without `approval_schema_version: "v1_iris_limbus_and_eye_contour"`.
2. **V1 must never treat four rough landmarks alone as a proper eye-opening contour.** The
   eye-opening contour helper (`contour_from_approved`) no longer falls back to a synthesised
   ellipse around the four canthi/lid landmarks at runtime; the V1 runtime path requires a
   clinician-approved `eye_opening_contours` polygon.
3. **Stage 0 is the mandatory anatomical gatekeeper.** Before any V1 analysis, Stage 0 must
   validate both the iris/limbus circle and the eye-opening/orbital margin contour.
4. **Runtime must reject incomplete or legacy approvals and ask for re-approval.** A
   misleading "No nystagmus detected" result on invalid tracking is far worse than a
   refusal-to-run; the refusal is honest.

### Assisted Anatomical Approval (clarification 2026-06-27)

Stage 0 should be **assisted but clinician-controlled**. The app proposes iris/limbus and
eye-opening/orbital margin markings, but the clinician can quickly override poor proposals.
In difficult clips, manual correction by the clinician is preferred over over-engineering
automatic frame selection. The priority is a simple, fast, reliable approval step.

We briefly considered adding an automatic init-frame picker that would scan candidate frames
and score MediaPipe's proposals, then offer the highest-scoring frame as default. We did NOT
ship this. On difficult clips (e.g. the spontaneous-nystagmus vestibular-neuritis clip, where
one eye is partially closed and looking sharply downward on most frames) MediaPipe's
proposals are noisy enough that the clinician can pick a workable frame and mark the iris in
under a minute — faster, and more reliably, than any score-and-rank heuristic. Over-
engineering this step would slow the routine workflow without improving its hardest cases.

Stage 0 is **not** intended to be a fully manual drawing process either. The app proposes
the iris/limbus circle and eye-opening/orbital margin contour automatically. The clinician's
role is to verify and correct only what is wrong, then approve. This keeps the workflow
clinically practical while still preventing the old pupil-approval error.

The workflow is:

> **`App proposes → user verifies/corrects → user approves → tracking starts`**

When the user runs `python app.py --video "..." --approve`, the app:

1. Selects the best initialization frame using MediaPipe face/iris detection.
2. **Proposes the full iris/limbus circle** for each detected eye, sized to MediaPipe's
   iris-boundary estimate plus a small safety margin (~10 %) so the initial circle reliably
   covers the coloured iris out to the white sclera — **not the pupil**.
3. **Proposes the iris centre** as the centre of that circle.
4. **Proposes the eye-opening / orbital margin contour** as a smooth 24-vertex polygon
   seeded from the four MediaPipe face landmarks (inner canthus, outer canthus, upper lid
   margin, lower lid margin).
5. **Proposes the usable-eye status** — both eyes default to usable when MediaPipe
   detected them.

The clinician then sees the proposed markings on the Stage-0 frame and corrects only what
is wrong. One Enter approves the entire anatomical setup. q cancels.

**Why this matters.** A blank-canvas tool would force the clinician to draw two anatomical
structures by hand on every approval, which is impractical for routine clinical use. An
auto-propose-then-verify workflow is the way modern medical imaging tools handle anatomical
landmark approval. The app does the laborious initial estimate; the clinician contributes
clinical judgment about whether the estimate is right and adjusts where needed.

**What this does not change.** Even though the app proposes the markings automatically, V1
must still require explicit Stage-0 approval. If the user has not pressed Enter on a
properly-approved iris/limbus + eye-opening contour set, V1 must not run. If the approval
file is legacy or incomplete, V1 must reject it as `invalid_stage0_approval`. The runtime
rules in this section all stand unchanged; the only change is *how* the clinician arrives at
the approved values.

**Important distinction:**

> This is not manual landmarking. This is assisted clinical verification.
> **The app proposes the anatomy. The clinician verifies the anatomy. Only then does V1 track.**

**Windows BAT launcher (2026-06-27).** To support the routine clinical workflow without
asking the clinician to navigate PowerShell, a Windows BAT launcher was added at the project
root: `Approve_Vestibular_Neuritis.bat`. Double-clicking the file opens the Stage-0 approval
GUI for the spontaneous-nystagmus vestibular-neuritis reference clip directly. The launcher
simply `cd /d`s into the project directory and runs `python app.py --video "..." --approve`,
preserving the propose → correct → one-Enter-approve workflow. Per-clip launchers can be
added the same way for any sample the clinician needs to approve frequently.

### Origin of the rule

### Rule

eye_vng V1 requires two approved structures, not one:

1. **The full iris / limbus circle** — the moving clinical object. Approved out to the white
   sclera, not the pupil.
2. **The eye-opening / orbital margin contour** — the local reference frame. Approved as a
   polygon refined to the visible palpebral fissure, not four loose landmarks.

The iris/limbus is the moving object. The eye-opening contour is the local reference frame.
Both must exist; both must be clinician-approved.

V1 must not proceed with:

* pupil approval alone,
* full iris approval alone,
* four rough legacy landmarks alone if a true contour is missing,
* an old `pupils` block from the pre-V1 schema.

### Stage 0 implementation (single session)

`--approve` now opens **one interactive window** where the clinician approves all three
structures in one editing session before pressing Enter:

* `i` — iris/limbus mode: drag the centre, +/- to resize the circle to the limbus boundary.
* `f` — face landmark mode: drag the canthi and lid margins to the visible corners and lids.
* `c` — contour mode: drag vertices of the eye-opening polygon, `a` to add a vertex under the
  cursor, `x` to delete the selected vertex, `e` to switch to the other eye.

When Enter is pressed, Stage 0 validates the result. Each eye marked usable must pass:

* `_validate_iris_seed` — iris radius must be at least 9 px and at least 5 % of inter-canthal
  distance, so a pupil-sized seed is rejected with `iris_radius_too_small_likely_pupil`.
* `_validate_eye_opening_contour` — the contour must have at least 8 vertices, must surround
  the iris seed (the iris centre must lie inside the polygon), and must have a bounding box
  at least 2 × iris radius wide and 1 × iris radius tall.

If either check fails, Stage 0 refuses to save and prints the per-eye failure reasons.

### Approval schema

A passing Stage 0 writes the V1 schema explicitly into `approved_landmarks.json`:

```json
{
  "approval_schema_version": "v1_iris_limbus_and_eye_contour",
  "tracking_target": "iris_limbus",
  "reference_target": "eye_opening_contour",
  "iris_limbus_approved": true,
  "eye_opening_contour_approved": true,
  "legacy_pupil_fallback_used": false,
  "legacy_four_point_contour_fallback_used": false,
  "per_eye_validity": {
    "L": {"usable_for_v1": true,
          "iris_limbus_approved": true,
          "eye_opening_contour_approved": true},
    "R": {"usable_for_v1": true,
          "iris_limbus_approved": true,
          "eye_opening_contour_approved": true}
  },
  "iris": { "L": {...}, "R": {...} },
  "face_landmarks": { ... },
  "eye_opening_contours": { "L": [ {x, y}, ... ], "R": [ {x, y}, ... ] }
}
```

The old `pupils` field is not written by the V1 approver. Old approval files that contain it
are rejected at runtime (see below).

### Runtime gate

The V1 tracker now refuses to start when:

* `approval_schema_version != "v1_iris_limbus_and_eye_contour"`,
* `iris_limbus_approved` is missing or false,
* `eye_opening_contour_approved` is missing or false,
* `eye_opening_contours` is missing,
* an eye is marked `usable_for_v1: false` but is still selected for analysis.

The error message says exactly which check failed and asks the clinician to re-approve.
There is no silent fallback to:

* legacy pupil seeds,
* the old `pupils` schema,
* four-point landmarks treated as a contour.

### Detector protection

When the schema gate rejects an approval, the detector does not run and a tombstone
`nystagmus_events.json` is written with:

```json
{
  "analysis_valid": false,
  "rejection_reason": "invalid_stage0_approval",
  "clinical_nystagmus_detected": false,
  "clinical_overlay_silent": true
}
```

No overlay video is produced for that run; no arrow can possibly be displayed.

### Overlay rule during development

While the V1 tracker is being calibrated, the clinical overlay shows:

* anatomical `Left` / `Right` labels,
* iris / limbus circle (state-coloured),
* iris centre cross,
* eye-opening contour outline (cyan when reference valid, amber when uncertain).

So the clinician can see at a glance whether both the moving object and the reference frame
were tracked. Debug values (frame number, fc/ic/rc/cov, trace strip, "Torsion: not assessed"
footnote) stay off in clinical mode; they remain available in `--overlay-mode debug`.

### Principle reinforced

> **No valid iris/limbus approval plus no valid eye-opening contour approval means no V1
> analysis.**

Stage 0 anatomical approval is non-negotiable for V1. Without both structures, the
contour-relative iris-centre signal is mathematically meaningless and the detector
must not run on it.

And alongside it (clarification 2026-06-27):

> **Stage 0 is assisted anatomical approval: app proposes, clinician verifies, V1 runs
> only after approval.**

And refined (2026-06-27, after the auto-picker was considered and dropped):

> **Stage 0 is app-assisted but clinician-controlled: the app proposes, the clinician can
> override, and V1 runs only after approval.**

### Verification pass (2026-06-27)

After the runtime gate was strengthened, the rule was verified end-to-end:

* `python -m py_compile app.py iris_tracker.py src/core/*.py` — passes cleanly.
* `python -c "import iris_tracker, app, debug_replay; print('imports OK')"` — passes.
* Forbidden runtime read of `approved.get("pupils")` — grep returns no matches anywhere
  in `iris_tracker.py`, `app.py`, `debug_replay.py`, `src/core/`, or `segments.py`. The
  legacy `pupils` schema has no runtime entry point left.
* No runtime caller passes `allow_legacy_fallback=True` to `contour_from_approved` — grep
  confirms only the function's own docstring mentions the keyword. The eye-opening contour
  helper therefore returns `None` when `eye_opening_contours` is missing, which the V1
  runtime gate uses to reject the approval.
* Running V1 on the failing clip's legacy approval —
  `python app.py --video "samples/nystagmus at rest to left in right vestibular neuritis.mp4" --no-mediapipe` —
  exits with code 1, prints the "Invalid Stage 0 approval for eye_vng V1" message with all
  five specific failure reasons (schema version, two flags, contour block, no usable eye),
  and writes a tombstone `nystagmus_events.json` + `metadata.json` recording
  `stage0_valid_for_v1: false`, `analysis_valid: false`, `rejection_reason:
  invalid_stage0_approval`, `clinical_nystagmus_detected: false`, `clinical_overlay_silent:
  true`. The tombstone `note` says explicitly *"Invalid Stage 0 is NOT a negative clinical
  result; it is a failed analysis."* Stale `tracking_overlay.mp4`,
  `tracking_overlay_with_arrow.mp4`, `tracking.csv`, and `nystagmus_events.csv` from any
  previous successful run are deleted by the rejection path so the directory cannot
  contain a misleading overlay alongside the failure tombstone.

### Manual GUI test — pending (2026-06-27)

The single-session Stage-0 editor (`approve_interactive`) opens an OpenCV window with three
editing modes (iris/limbus, face landmarks, eye-opening contour) and a single Enter to
APPROVE. It cannot be exercised from a non-display session. The expected manual procedure is:

1. Run `python app.py --video "..." --approve` on the failing clip.
2. In IRIS mode, mark the full iris/limbus circle out to the white sclera — NOT the pupil.
   On 1080p face crops the iris radius is typically 15–25 px (the legacy pupil approval was
   5–9 px).
3. In FACE LANDMARK mode, refine the four canthi/lid-margin points if MediaPipe's seed is
   off. These landmarks seed the contour starting shape but are not the contour itself.
4. In CONTOUR mode, drag/add/delete vertices until the cyan polygon traces the visible
   palpebral fissure. `e` switches to the other eye's contour.
5. Press Enter. `approve()` validates both structures per eye and writes the V1 schema
   fields (`approval_schema_version`, `tracking_target`, `reference_target`,
   `iris_limbus_approved`, `eye_opening_contour_approved`, `legacy_pupil_fallback_used:
   false`, `legacy_four_point_contour_fallback_used: false`) plus the canonical `eyes`
   block and `eye_opening_contours`. The writer asserts `"pupils" not in data`.
6. Re-run `python app.py --video "..." --no-mediapipe`. The V1 runtime gate now accepts the
   approval, the tracker starts, and the detector either confirms left-beating nystagmus
   or honestly reports its absence. Detector thresholds remain conservative; no tuning
   until at least one known-positive clip is correctly labelled with the proper anatomy.

---

## 18. Clinician Teaching Mode for Difficult Iris Frames

After the V1 schema gate (§17) was satisfied and a proper iris/limbus + eye-opening contour
approval was made for the spontaneous-nystagmus vestibular-neuritis clip, V1 tracking was
re-run. The tracker improved substantially over the legacy-approval run, but on this clip
the iris was still lost for long segments: roughly 41 % of frames on the better eye and
70 % on the other, with `limbus_unfit` and `poor_arc_coverage` as the dominant drift reasons.
Inspecting an overlay frame from extreme eccentric gaze confirmed the failure mode visually:
the fitted iris circle drifted off the iris onto the white sclera. The detector then stayed
honestly silent, but the underlying problem was that the tracker had no clinical signal to
analyse during the segments where nystagmus was most visible.

### What we observed

* In extreme gaze or partial-iris views, the tracker may still lose the iris or place the
  circle wrongly. The visible part of the iris becomes a thin crescent against the white
  sclera, the radial-edge search produces few inliers, and the limbus fit either fails
  outright (`limbus_unfit`) or shrinks toward something pupil-sized or sclera-aligned.
* The clinician may understand the iris better than the current tracker in such frames.
  A human eye can see the dark iris–white sclera arc on one side, recognise the
  curvature, and infer the full circle including the part hidden under the lid. The
  current limbus fitter cannot reliably do that on its own.

### The new direction

A new idea was introduced: **clinician teaching mode** (also called **rescue-anchor mode**).
In this mode, the clinician marks the full imagined iris/limbus circle on difficult frames —
not the pupil, and not just the visible arc. The marking is an act of clinical interpretation:
"this is what the iris really is on this frame, even though most of it is hidden."

These corrections serve two roles at once:

1. **Rescue anchors for the current clip.** When V1 tracking is re-run with rescue anchors
   present, the tracker can use them as strong constraints on nearby frames — preserving the
   expected iris radius, allowing partial-arc fitting around the anchor, penalising circles
   that drift onto the sclera, and interpolating smoothly between corrected anchors where
   appropriate. This rescues the clinical signal during exactly the segments where it was
   being lost.
2. **A curated teaching dataset for future improvement.** Each correction is saved as a
   structured teaching example — the frame image, the corrected iris centre and radius, the
   eye-opening contour at that frame, and the reason for rescue. Over many clips and many
   clinicians these accumulate into a curated expert-corrected dataset.

### Important scope limit

At this stage the system is **not** yet doing automatic retraining of a model. It is saving
expert corrections and reusing them as rescue anchors within the current clip. The teaching
dataset is captured now so the value compounds over time; the formal model-training pipeline
is deliberately deferred until we have enough curated examples and a clear training plan.
Calling this "AI" before that point would overstate what is happening.

### What is being marked

The marked object remains the **full iris/limbus circle**:

* the moving clinical object — the coloured iris out to the white sclera,
* inferred from the visible iris–sclera arc even when most of the iris is hidden,
* **never the pupil**.

This is the same anatomical commitment §17 made for Stage 0. The clinician marks the iris
the way an experienced vestibular neurologist would judge it on the frame, then approves
that interpretation for the tracker to use.

### Principle

> **Teach the system by marking the iris the way the clinician would judge it.**

### Manual testing — teaching mode must work when tracking fails (2026-06-28)

A first end-to-end manual test of the teaching scrubber surfaced a critical usability
problem. When the patient looked to the side (especially during extreme left gaze) only part
of the iris was visible. The tracker — correctly — lost the iris. But the teaching UI also
became unusable on those very frames:

* The active iris marker could sometimes be resized but could not be dragged.
* When the patient looked back toward centre, tracking did not recover, and the teaching UI
  could not place a marker on the visible partial iris.

The root cause was a residual coupling between the teaching marker and the tracker state.
The marker's clickability was implicitly tied to a valid tracker seed for the current frame;
when the tracker had no seed, the marker became ungrabbable. That is the opposite of what
teaching mode should do — teaching mode must work *precisely* when the tracker is wrong or
lost.

The teaching UI was patched accordingly:

* The active iris marker was decoupled from tracker state entirely. It is a pure
  clinician-controlled object held in `state["iris_*"]` and edited only via mouse + keys.
* In iris mode, **click anywhere on the frame** to place the marker. If the click is within
  `iris_r + 8` px of the current centre it begins a drag; otherwise the marker teleports
  there and a drag continues from the click point. The debug log distinguishes
  `drag_start | moved | created` so it is unambiguous what the click did.
* `+` / `-` and the mouse wheel resize the active iris radius. Save with `s`.
* Anchors are saved to `outputs/<clip>_tracked/teaching_anchors.json` (a path at the run
  root, not under `teaching_examples/`); per-frame PNGs go under `teaching_examples/`. The
  legacy `teaching_examples/rescue_anchors.json` is still read on open so prior sessions
  are not lost.
* The anchor record now carries explicit anatomical identity:
  `active_teaching_eye`, `image_side_eye`, `patient_anatomical_eye`,
  `corrected_iris_centre`, `corrected_iris_radius`, `iris_visibility`,
  `correction_reason`, `corrected_by_clinician`.
* For this manual-testing pass the active teaching eye is **forced to patient anatomical
  Right** (image-side L). The `e` eye-switch key is disabled. The top HUD reads
  *ACTIVE TEACHING EYE: Right* in magenta so the clinician can never be in doubt.
* Mode-2 (contour-oval) edits are still possible but optional; this turn focused on iris.

Manual testing also showed the importance of partial-iris clinical reasoning. In extreme
gaze only part of the iris may be visible. The clinician is marking the FULL imagined
iris/limbus circle — inferred from the visible iris–sclera arc — **not the pupil**. The
on-screen instruction line says so explicitly.

> **Teaching mode must work when tracking fails: click to place the iris, resize it, and
> save an anchor.**

### Manual testing — second pass: bad defaults, slow resize, lost saves (2026-06-28)

A second manual-testing pass found that while click / drag / resize / save now technically
worked, the workflow was still much slower and more error-prone than it needed to be. The
specific problems and their fixes:

* **Magenta marker on the nose at startup.** When no tracker iris row existed for the
  active eye on the init frame, the helper `_seed_iris_from_tracking()` was falling through
  to a hard-coded `(0.35 * W, 0.5 * H)` "default" — which on a face crop landed on the
  bridge of the nose. The helper now returns `None` and the renderer hides the marker
  entirely until the clinician clicks. The marker is never initialised at a face-centre
  position.
* **Large cyan oval persisting between the eyes.** The contour-init path had a last-resort
  fallback that drew a wide ellipse around `state["iris_x"], state["iris_y"]` whenever no
  anchor and no Stage-0 oval was available — which, combined with the nose-default marker,
  drew a huge oval across the bridge. The fallback has been deleted. If neither an anchor
  nor a Stage-0 oval exists, no contour is drawn.
* **Default iris radius starting at 80–95 px.** The default-radius rule has been extended:
  priority 1 is now the clinician's *last edited radius in the current session*, so the
  very next click reuses whatever the clinician has just settled on. Priorities 2–5 are
  anchor median, Stage-0 radius, tracker median, fallback. The result is clamped to
  `[10, 45] px` at marker-creation time. Fallback raised from 12 px (too small) to 30 px
  (right size for a 1080p face crop). The clinician can still enlarge above 45 px after
  the marker exists.
* **Resize too slow.** `+` / `-` now step by 3 px (was 1). New keys `*` / `_` step by
  10 px. Mouse wheel also steps by 3 px now (was 1).
* **Many corrections lost — only 1 anchor saved at end.** The previous version only
  persisted to disk on `s`, and only when the editor quit. The new behaviour:
  * Every successful save (manual or auto) writes the full `teaching_anchors.json` to
    disk immediately. Closing the editor unexpectedly no longer loses prior anchors.
  * **Auto-save on leaving a frame**: any frame with an unsaved iris/contour edit is
    auto-saved when the clinician navigates to a different frame, with
    `save_reason: "auto_on_leave"` in the anchor record.
  * Auto-save on quit (`q` / `Esc`): same safety net.
* **Unclear save status.** The top HUD now shows a save-status pill on the right:
  `UNSAVED EDIT — press s to save anchor` (yellow), `SAVED anchor for this frame` (green),
  or `unedited` (grey). A large centred `ANCHOR SAVED — frame N, Right eye` toast
  appears for ~2 seconds after every save. Terminal prints both display and image
  coordinates on every click, drag, resize, and save.
* **Enter does NOT save.** The bottom HUD says so explicitly. Pressing Enter prints a
  one-line nudge to the terminal.
* **Saved coordinates appeared out-of-bounds.** A user-reported save said
  `centre=(542.4, 977.1), radius=80.5` on a 960×540 overlay — image y outside the canvas.
  A `_clamp_to_frame()` guard now applies at save time so iris centre cannot be persisted
  outside the visible frame. The saved record now also carries `image_size_at_save`,
  `display_size_at_save`, and `scale_at_save` so the display↔image transform is auditable.

### Principle

> **Teaching markers must be clinician-controlled, correctly initialized, draggable, and
> explicitly saved.**

### Manual testing — third pass: simplified iris teaching mode (2026-06-28)

After the second-pass patch the marker still drifted into a two-state confusion: a fallback
"hidden" state, an "edited but not anchored" state, and a per-frame seeded state, with
distinct `created` / `drag_start` / `moved` click actions. In practice the simpler mental
model is what the clinician wants:

> **The iris teaching task is simple: one clinician-controlled circle should cover the iris
> wherever the iris is visible.**

The iris-mode click handler was therefore collapsed into two click outcomes only:

* `[iris-select]` — click within `iris_r + 8` px of the existing circle's centre → select
  the same circle and start dragging.
* `[iris-reset]` — click further away → recentre the same circle to the click point and
  start dragging from there.

The marker is now **never created as a second object**. Drag events are logged as
`[iris-move]`. Resize is unchanged at `[iris-resize]`. The "very first click of a
session with no seed at all" case is still handled — the first click is treated as a
reset, and the radius is taken from the default-radius rule (last edited → anchor median
→ Stage-0 → tracker median → 30 px, clamped to 10–45 px).

Equally important, the eye-opening oval is now drawn **only in contour mode**. In iris
mode the canvas shows only:

* the video frame,
* the one magenta/yellow/green iris circle (active eye, state-coloured),
* the `ACTIVE TEACHING EYE: Right` banner + save-status pill,
* HUD text.

No bridge oval, no inactive-eye markers, no second circle. The clinician sees one circle
and drags it onto whatever iris is visible.

### Manual testing — fourth pass: always-visible, parkable markers (2026-06-28)

The "hide the marker until the clinician clicks" rule from the third pass turned out to
be too aggressive. On a tracker-lost segment the clinician would see an empty frame and
have to *guess* where the iris is, then click to spawn a marker. That is harder than the
opposite: a marker that is always present in a neutral position and can simply be dragged
onto the iris.

The teaching UI was therefore patched to make markers **always visible**:

* `marker_visible` is now forced True in iris mode. The state machine no longer hides.
* A new `is_parked` flag is True when the iris marker has no real seed (no anchor, no
  tracker iris, no Stage-0 seed) and is sitting at a neutral visible location instead.
  Priority for the parked position:
  1. centre of the active eye's eye-opening contour if a Stage-0 contour exists,
  2. left third of the image for patient Right (image-side L) / right third for
     patient Left (image-side R),
  3. screen centre (last resort — only if `W` or `H` are missing).
  The parked position is always clamped inside the visible frame so the marker can
  never sit off-screen.
* When parked, the marker draws the label `UNPLACED IRIS - DRAG TO IRIS r=NN` next to
  it (instead of the usual `ACT-Right r=NN`) so the clinician sees that this is a
  draggable default, not a real seed.
* Default parked radius comes from the existing `_default_iris_radius_clamped()` rule
  (last edited → anchor median → Stage-0 → tracker median → 30 px fallback, clamped to
  10–45 px).
* The same idea is applied to the eye-opening oval: when no anchor and no Stage-0
  contour are available, a parked oval (`a ≈ 2.6 × iris_r`, `b ≈ 1.4 × iris_r`) is
  drawn around the iris position with the label `UNPLACED CONTOUR - DRAG/RESIZE TO EYE
  OPENING`. The clinician drags the centre handle and resizes the cardinals to fit.
* Any iris click or contour-handle drag clears the corresponding `is_parked` /
  `oval_is_parked` flag, so the UNPLACED label disappears the moment the clinician
  commits.
* Saved anchor records gain two new fields: `was_parked_before_correction` (whether the
  marker was in parked state when the clinician started editing it) and
  `placed_manually` (whether the clinician moved/resized it before saving). These are
  honest about clinician intent: a "parked → manually placed → saved" anchor is more
  trustworthy than one that was simply auto-saved on frame change without a real edit.

### Principle

> **Teaching markers should always be visible, draggable, and ready to correct.**

### 18a. Tracker-assisted teaching (revival, 2026-06-28)

After the simple §19 editor was built, manual testing on the vestibular-neuritis clip
showed that pure manual annotation loses the benefit of the tracker on the long stretches
where it follows the iris well. The right workflow is not "tracker-only" and not
"manual-only" — it is **tracker-assisted**: the tracker carries the marker through the
easy stretches; the clinician takes over on the failure stretches (medial gaze, partial
limbus, occluded iris); after correction the tracker resumes from the clinician's
position.

Two surgical patches were applied to `src/core/iris_rescue.py` (the §18 scrubber) to
restore that workflow without rewriting it:

* **Patch 1 — smooth drag.** Replaced the blocking `cv2.waitKey(0)` in the scrubber's
  main loop with a non-blocking `cv2.waitKey(16)` (~60 Hz). Previously, with no key
  pressed, the loop never re-rendered during a drag, so the marker visually didn't
  follow the cursor. This is the same bug we hit in the simple editor and the same fix.
* **Patch 2 — clinician-wins propagation window.** When the clinician edits a frame and
  scrubs forward, the next ~30 frames now prefer the clinician's last-edited position
  over fresh tracker output, before the tracker takes back over. The seed-priority
  chain becomes: saved anchor (exact) → saved anchor (±5) → **clinician_propagated (≤30
  frames since last edit)** → tracker → Stage-0 → stale previous_manual → parked. The
  `last_manual_frame_no` is only updated on dirty frames (real edits), so simply
  scrubbing past a frame does not extend the window.

On-eye text labels (`ACT-Right`, `r=…`, `UNPLACED IRIS`) were removed from beside the
iris. Status now lives in the top HUD; only the circle and the centre cross sit on the
eye. The "UNPLACED" hint moves up into the HUD.

Deferred (not yet implemented; documented for future work):

* **Patch 3 — resume V1 tracking from clinician anchors.** When the V1 tracker is
  re-run on a clip that has rescue anchors, those anchors should become local restart
  seeds. The tracker should preserve radius unless strong evidence, search locally, and
  continue forward until the next anchor or next failure. Touches `v1_tracker.py`.
* **Patch 4 — partial-arc limbus fitting.** Allow the tracker to fit a full iris
  circle from a partial visible limbus arc, with sclera-drift penalty and an
  anti-pupil-shrink term. Touches `limbus.py`. This is a non-trivial CV change, not a
  patch, and will be designed separately.

### Principle (tracker-assisted teaching)

> **The tracker does most of the work, the clinician corrects failures, and tracking
> resumes from the correction.**

---

## 19. Simple Clinician-Controlled Iris Video Editor

The §18 rescue scrubber accumulated too much logic over several patches. Even after the
"always visible + parked" rule (§18, fourth pass), the marker was still being influenced by
tracker output (via `tracking.csv`), saved anchor records (via `teaching_anchors.json`),
parked / unparked state, baked overlay colours, contour mode, contour anchors, and the
Stage-0 fallback chain. Manual testing showed that the marker still jumped around when the
clinician wanted it to stay put, and the workflow was confusing in a way that no single
patch could clean up.

The conclusion: **the rescue scrubber should not be turned into a teaching editor.** A new,
much simpler, separate tool was needed.

### The new tool

A new editor was added at [src/core/simple_iris_editor.py](src/core/simple_iris_editor.py)
and wired into `app.py` as:

```
python app.py --video "..." --annotate-iris        # canonical flag
python app.py --video "..." --simple-iris-editor   # alias
```

It opens the **raw video** (not `tracking_overlay.mp4`) in a clean OpenCV window. There is
one magenta iris circle. The clinician drags it onto the iris on each frame, presses `s`
to save, advances frames, and saves per-frame annotations to
`outputs/<clip>_tracked/manual_iris_annotations.json` in source-video coordinates.

The editor is independent of the V1 tracker, the V1 detector, the rescue scrubber, and all
overlay logic. Nothing in `iris_tracker.py`, `nystagmus_detector.py`, `v1_tracker.py`, or
`iris_rescue.py` is imported. The tracker does not own the marker.

### Marker behaviour

* The iris circle is **always visible**.
* The iris circle is **always draggable**.
* The marker is **never overwritten** by tracker state, anchor state, parking state, or
  any other internal state, after the first-frame init.
* First-frame init priority is: (1) existing annotation for frame 1, (2) Stage-0 approved
  iris seed for the active eye, (3) default 35 px circle at the left third of the frame.
* On frame change, the marker **keeps its current position** — it does not snap back. The
  only exception is when the clinician navigates to a frame they previously saved an
  annotation for; in that case the saved annotation is loaded into the working circle.
* Generous hit-test: a click within `max(radius × 1.25, 80 display px)` of the centre
  grabs the circle. A click further away recentres the same circle to the click point.
  No second circle is ever created.
* Resize: `+` / `-` ±3 px, mouse wheel ±3 px per tick. Radius is shown in the HUD only —
  not next to the iris.

### Save format

`outputs/<clip_name>_tracked/manual_iris_annotations.json`:

```json
{
  "video_path": "samples/...mp4",
  "source_resolution": [1920, 1080],
  "fps": 30.0,
  "active_teaching_eye": "Right",
  "image_side_eye": "L",
  "n_annotated_frames": 12,
  "annotations": [
    {
      "frame_number": 123,
      "time_sec": 4.1,
      "patient_anatomical_eye": "Right",
      "image_side_eye": "L",
      "iris_centre": [504.12, 384.55],
      "iris_radius": 35.0,
      "coordinate_space": "source_video",
      "corrected_by_clinician": true,
      "save_reason": "manual",
      "saved_at": "2026-06-28T09:01:14"
    }
  ],
  "updated_at": "2026-06-28T09:01:14"
}
```

Per-frame PNGs are NOT written automatically (the spec says only on request, which is a
future option). Persisted on every save so an unexpected close cannot lose work.

### Controls

| Key | Effect |
|---|---|
| Drag | move iris centre to the cursor |
| Mouse wheel | resize ±3 px |
| `+` / `-` | resize ±3 px |
| `←` / `→` (or `b` / `n`) | previous / next frame |
| `[` / `]` | ±10 frames |
| PgUp / PgDn | ±1 second |
| `g` / `G` | first / last frame |
| Space | play / pause |
| `s` | save annotation for the current frame |
| `a` | toggle auto-save on/off |
| `q` / Esc | quit (auto-save current frame if auto-save is on) |

### Scope for this first version

* Active eye is **patient anatomical Right only** (image-side L). No eye switching.
* No contour editing.
* No detector logic.
* No automatic rescue anchors.

These restrictions are deliberate: this tool is designed to make the simplest case
(one iris, one eye, drag and save) work reliably first. Contour editing and the second
eye can be added once the workflow is comfortable on real clips.

### Principle

> **The clinician owns the iris marker.**

And:

> **First make manual iris annotation easy and reliable. Then use those annotations to
> improve tracking.**

### 19a. Hybrid refinement — tracker suggests, clinician owns

The first version of the simple editor ignored the V1 tracker completely. On clips where
the tracker actually works for most frames, that means the clinician has to drag the
marker even on frames where a click would have done. The fix is **not** to let the tracker
drive the marker — that is exactly the mistake the rescue scrubber made. The fix is a
**hybrid**:

* If `outputs/<clip>_tracked/tracking.csv` exists, the editor loads the per-frame
  iris `(x, y, r)` for the active eye (image-side L → patient Right, columns
  `raw_left_iris_center_x/y` and `left_iris_radius`).
* Those values are drawn as a **faint cyan, thin, non-editable circle**. They are a
  suggestion, not the truth.
* The clinician's magenta circle is still the only editable marker. Mouse, wheel, `+`,
  `-`, arrows — they all act on the clinician marker, never on the suggestion.
* New key `t` = **accept tracker suggestion for the current frame**. It copies the
  suggestion into the clinician marker, marks the frame dirty, and shows a brief HUD
  toast `TRACKER SUGGESTION ACCEPTED — press s to save`. It does **not** auto-save.
* New key `h` = hide/show the tracker suggestion.
* The save record gains a `"source"` field: `"manual"` (default) or
  `"tracker_accepted_by_clinician"` (after `t` and before any further manual touch).
  Any subsequent manual edit on the same frame reverts the source to `"manual"`.
* First-frame init priority is updated to:
  (1) saved annotation for frame 1,
  (2) previous clinician position (n/a on the first frame),
  (3) tracker suggestion for frame 1,
  (4) Stage-0 approved iris seed for the active eye,
  (5) neutral default.
* If `tracking.csv` is absent, the editor still works exactly as before — no suggestion
  circle, the `t` key prints `no tracker suggestion for this frame`, and saves are all
  `source: manual`.

The principle for the hybrid:

> **Tracker suggests; clinician owns the annotation.**

The tracker may help. The tracker must never own, overwrite, or fight the clinician
marker.

---

## 19b. Anatomically-Constrained Iris Tracker (`--engine anatomical`, 2026-06-28)

The V1 default tracker is good at finding a limbus arc but the radial-edge search has
no anatomical containment — it can drift onto cheek, nose, brow, or medial canthus when
the prior centre is bad. The clinician's observation: in Indian eyes the iris is a dark
structure against a bright sclera, and even when the iris is partly covered or at
extreme gaze the visible iris–sclera arc remains a dark curve against sclera. The
tracker should use that arc and infer the full iris circle, but with hard anatomical
limits so it cannot wander onto the face.

This was specified end-to-end in
[TRACKER_REDESIGN.md](TRACKER_REDESIGN.md) and implemented in
[src/core/iris_anatomical_tracker.py](src/core/iris_anatomical_tracker.py) as an opt-in
engine choice (`python app.py --video ... --engine anatomical`). The V1 default
remains the V1Tracker, untouched.

Key design points:

* **Stage 0 is mandatory.** The anatomical engine refuses to construct without a
  Stage-0 eye-opening contour polygon for the active eye. Containment is enforced in
  code, not in spirit.
* **The visible iris–sclera arc must lie inside the Stage-0 contour. The inferred full
  circle does not have to.** This matches the clinician's anatomical rule literally: at
  extreme lateral gaze the unseen part of the iris is allowed to lie outside the
  contour. Implementation: `limbus._radial_edges` gained an optional `polygon=` arg;
  every accepted radial-edge pixel is `cv2.pointPolygonTest`-ed before it enters the
  fit. The circle centre and radius are free.
* **Hard radius band**: r ∈ [0.70, 1.35] × Stage-0 radius. Outside the band → freeze.
* **Confidence** = w_arc · arc_coverage + w_inlfr · inlier_fraction + w_dist ·
  exp(−d_prev/r_prev) + w_rad · exp(−|Δr_prev|/0.15r_prev) + w_stage ·
  exp(−|Δr_st0|/0.20r_st0). Initial weights 0.30/0.25/0.20/0.15/0.10.
* **Three-tier decision**: conf ≥ 0.55 → `tracked` (or `partial_arc_tracked` if arc <
  0.35). 0.30 ≤ conf < 0.55 → `low_conf` (accept but mark). conf < 0.30 → freeze last
  good, emit `frozen_last_good`, do **not** update last-good so the next prior is the
  same last-good (no drift via the prior).
* **Rescue anchors win absolutely.** If `teaching_anchors.json` has an entry for this
  frame, the tracker emits the anchor's centre/radius and that becomes the new
  last-good. This satisfies the "resume tracking from clinician correction" rule.
* **Fixed-radius mode** after 10 consecutive OK frames — `r_fixed = r_prev` makes the
  centre fit occlusion-invariant (cannot shrink to pupil, cannot grow into eyelid).
* **Per-frame debug log** (printed only on non-OK frames to keep stdout sane) carries
  candidate count, centre/radius, arc/inlier/distance/radius-deviation, containment
  fraction, confidence, status, and reject reason.

Files touched:
* [src/core/limbus.py](src/core/limbus.py) — added `polygon=` arg to `_radial_edges`
  and `fit_limbus`. Backward-compatible.
* [src/core/iris_anatomical_tracker.py](src/core/iris_anatomical_tracker.py) — new
  file, the orchestrator.
* [iris_tracker.py](iris_tracker.py) — new `engine == "anatomical"` branch in the
  dispatch. V1 path unchanged.
* [app.py](app.py) — `--engine anatomical` added to the choices.

### Principle (anatomical engine)

> **The iris tracker must fail safely, not wander.**
> **A missing iris is better than a wrong iris on the face.**

### 19b.1 Correction: don't replace the working tracker (2026-06-28)

The first anatomical-engine run on the vestibular-neuritis clip froze at frame 49
and never recovered (`status=frozen_last_good reason=no_limbus_fit_inside_contour`
on every subsequent frame). Two follow-on changes were made:

* A multi-prior "recovery search" was added that tried `last-good`, `Stage-0
  centre`, `contour centroid`, and a 3×3 grid inside the contour. It "worked" in
  the sense that frames stopped being frozen — but on a sequential run it
  found wrong arcs (eyelid, cheek inside the generous contour) because grid
  priors have no temporal locality. **This was the wrong direction.** Clinician
  corrected the direction: *do not replace the working tracker; keep the
  successful local fit logic and add anatomical guardrails*.
* The recovery search was reverted in a follow-up edit on the same day.

The structural bug behind the original frame-49 freeze was elsewhere: **automatic
fixed-radius mode**. After 10 consecutive OK frames the tracker locked the
radius (`r_fixed = prev_r`) to make the centre fit occlusion-invariant. The
problem: any subsequent frame where the true iris radius changed (eyelid
covering it, partial occlusion) failed the fit and the tracker froze
permanently. Fix: `AnatomicalParams.R_FIXED_AFTER_N_OK = 10_000` (effectively
disabled). Radius now stays free, bounded only by the hard ±20 % band.

Hard guardrails added per the corrected clinician spec, all running **before**
the soft confidence score (one good arc-coverage must not override a hard
rule):

* **Radius band:** candidate radius must be within ±20 % of Stage-0 radius
  (`R_BAND_FRAC = 0.20`). Outside the band → `reason=radius_out_of_range`,
  freeze, prior unchanged.
* **Jump cap:** candidate centre must not jump more than 0.35 × Stage-0 radius
  from `last_good` (`MAX_NORMAL_JUMP_FRAC = 0.35`). Even at the peak of a fast
  nystagmus phase the iris does not move that far in one frame, so anything
  larger is a wrong arc (cheek, eyelid, nose) and is rejected. → `reason=
  unstable_jump`, freeze, prior unchanged. A separate `MAX_FAST_PHASE_JUMP_FRAC
  = 0.75` is reserved for a future "we are inside a confirmed fast phase" gate.
* **Containment** (already in place): edge pixels must lie inside the Stage-0
  contour polygon. The contour is a *boundary*, not a permission to search
  everywhere inside it; the local fit pattern keeps the search anchored to the
  previous iris.

Structured rejection-reason debug lines were added per spec:

```text
[iris-track][L] fr=87 reject=unstable_jump d=112 max=32 r=88 arc=0.29
[iris-track][L] fr=91 status=partial_arc_tracked arc=0.32 conf=0.71 d=18 r=89
```

### Principle (corrected, 2026-06-28)

> **Keep the working tracker. Add anatomical guardrails.**

> **The Stage-0 contour restricts the search space, but the previous iris
> position controls the tracking. The tracker must not rediscover the iris from
> scratch every frame.**

### 19b.2 Reset — V1 stays primary, anatomical demoted to experimental (2026-06-28)

A full-clip smoke test of the §19b.1 guardrailed anatomical tracker on the
vestibular-neuritis clip produced 466 / 640 frames frozen (73 %) with frequent
`unstable_jump`, `radius_out_of_range`, `low_confidence`, and
`frozen_last_good` rejections. Even though each individual rejection was for an
anatomically defensible reason, the *cumulative* effect was that the tracker
could not keep up with the real iris through the clip.

The clinician's reset: **we have over-engineered the anatomical tracker; stop
tuning it; return to V1.**

Decisions locked:

* **V1 is the primary tracker** for the clinical workflow — `--engine v1` is
  the CLI default and the `Rescue_*.bat` launcher already uses it.
* **`--engine anatomical` is experimental only.** The anatomical engine code
  stays in the tree for future experimentation but is not part of the
  clinical workflow. Do not run it for clinical analysis until the design has
  been rethought from scratch.
* **A simple post-tracking plausibility filter is added on top of V1.** The
  filter runs after V1 produces a per-frame `FrameMeasurement` and marks the
  frame as failed only when the result is anatomically impossible:
  * centre is more than `PLAUSIBILITY_OUTSIDE_TOL_PX = 20 px` outside the
    Stage-0 eye-opening contour, or
  * radius is `< 50 %` or `> 200 %` of the Stage-0 iris radius.
* The filter **does not touch V1's internal state**. It only writes to the
  outgoing `FrameMeasurement` (`iris_valid = False`, `drift_flag = True`,
  `drift_reason = "plausibility:<reason>"`). V1's `last_good` / prior continue
  normally on the next frame.
* The filter is intentionally lenient. It catches cheek/forehead/nose drift
  and impossible radii. It does NOT enforce the tight jump / radius caps the
  §19b.1 anatomical engine used — those were the source of the over-rejection.

Files touched:
* [iris_tracker.py](iris_tracker.py) — new `_v1_plausibility_check` /
  `_apply_v1_plausibility` helpers in `run()`; called immediately after V1's
  per-frame `fm_L`/`fm_R` for `engine == "v1"` only.

Files **not** touched:
* `src/core/v1_tracker.py` — V1 tracker unchanged.
* `src/core/iris_anatomical_tracker.py` — experimental engine kept as-is.
* Rescue scrubber, simple iris editor, Stage-0 approval — unchanged.

### Principle (reset, 2026-06-28)

> **Return to the working tracker. Add guardrails, not a new tracker.**

> **Do not replace working tracking logic unless necessary. First add simple
> guardrails.**

### 19b.3 V1 radius-priority bug (2026-06-28)

After the §19b.2 reset, the first V1 run on the vestibular-neuritis clip
produced a storm of plausibility warnings dominated by
`radius_too_small(r=17<44)` / `radius_too_small(r=18<45)`. Initial suspicion:
Stage 0 had approved pupil-sized circles. **That suspicion was wrong.**

Direct inspection of
`outputs/<clip>_tracked/approved_landmarks.json`:

* `resolution: 1920x1080`,
* `tracking_target: iris_limbus`,
* `legacy_pupil_fallback_used: False`,
* `iris.L.radius = 90.55`, `iris.R.radius = 87.06` — full limbus values, correct
  for this 1920×1080 face video.

Stage 0 was correct. The bug was in V1 runtime: `_iris_r()` at
[iris_tracker.py:1860](iris_tracker.py#L1860) silently **overrode** the
Stage-0 clinician-approved limbus radius with MediaPipe's iris-landmark radius
whenever MediaPipe was available:

```python
# old (wrong) order
def _iris_r(center, fallback):
    cands = [(o.single_x, o.single_y, o.iris_radius) for o in (le0, re0) ...]
    if not cands or center is None:
        return float(fallback or 12.0)         # Stage-0 only used as fallback
    ...
    return float(min(cands, ...)[2])           # MediaPipe wins
```

MediaPipe's `iris_radius` is the radius of a tight ring inscribed through its
4 iris landmarks; on this clip it lands around 44–45 px, roughly **half** the
true limbus radius. The radius seed for V1Tracker thus came in at ≈ 44 instead
of ≈ 88. V1 then locked onto pupil-sized features in many frames.

#### Fix (2026-06-28)

`_iris_r()` rewritten so **Stage-0 wins absolutely**; MediaPipe is fallback
only:

```python
def _iris_r(center, fallback):
    # 1. Stage-0 clinician-approved limbus radius wins absolutely.
    if fallback and float(fallback) > 0:
        return float(fallback)
    # 2. MediaPipe radius is fallback only.
    cands = [(o.single_x, o.single_y, o.iris_radius) for o in (le0, re0) ...]
    if cands and center is not None:
        cx, cy = center
        return float(min(cands, ...)[2])
    # 3. Last-resort default.
    return 12.0
```

A `[V1-radius]` debug line is printed at init so the priority cannot silently
regress:

```
[V1-radius] L stage0=90.55 used=90.55 source=stage0
[V1-radius] R stage0=87.06 used=87.06 source=stage0
```

If Stage-0 is missing and MediaPipe is used, the source field reads
`mediapipe_fallback`; if Stage-0 was somehow modified, it reads
`stage0_modified` (which should not normally happen).

#### Verification on the vestibular-neuritis clip

Before fix:
* V1 R eye: TRACKING=13, TRACK_LOST=281, valid_frames=158 (24.7 %).
* V1 L eye: TRACKING=46, TRACK_LOST=192, valid_frames=157 (24.6 %).
* Plausibility rejects: 222, dominated by `radius_too_small(r=17..18<44..45)`.

After fix:
* V1 R eye: TRACKING=**255**, TRACK_LOST=129, valid_frames=**393 (61.5 %)**.
* V1 L eye: TRACKING=**181**, TRACK_LOST=105, valid_frames=**269 (42.1 %)**.
* Plausibility rejects: 646, **all `outside_contour`** (no more
  `radius_too_small`). Of those: 59 within 50 px, 113 within 50–100 px, 338
  within 100–200 px, 135 within 200–400 px, 1 ≥ 400 px.

The radius bug is gone — V1 is now properly seeded with the limbus radius and
TRACKING frames roughly quadrupled on R eye. The remaining `outside_contour`
warnings are a separate question (legitimate gaze / head motion vs V1
genuinely drifting onto cheek), to be diagnosed by viewing the overlay video.

### Locked rule (superseded by §19b.4)

> **Stage-0 clinician-approved limbus radius always wins.**
> **MediaPipe radius is fallback only.**

### 19b.4 MediaPipe removed from V1 clinical workflow (2026-06-28)

After the §19b.3 radius-priority fix landed and confirmed that MediaPipe's
iris-landmark radius (~44 px) was structurally wrong for V1 (the clinical
limbus is ~88 px on this clip), the clinician decided to **remove MediaPipe
from V1 entirely** rather than merely demote it to fallback. Reasoning: a
"fallback" with the wrong semantics is a future regression waiting to happen,
and MediaPipe iris landmarks are not the clinical iris/limbus for this
software in any sense.

#### Code changes

* `_iris_r()` was replaced with `_stage0_radius_or_refuse()` in
  [iris_tracker.py](iris_tracker.py). The new function reads the radius
  directly from the Stage-0 `iris.L.radius` / `iris.R.radius` field. There
  is no MediaPipe call. If Stage-0 radius is missing or invalid, V1
  **refuses to run** with a clear error message rather than falling back.
* Init-frame `IrisDetector.detect(init_frame)` and its `le0` / `re0` outputs
  are no longer consulted for radius. The detector is still instantiated for
  optional facial-landmark tracking when MediaPipe is enabled, but the V1
  per-frame loop now runs without it when `--no-mediapipe` is passed.
* `[V1-radius]` debug prints simplified: source is always `stage0`. If
  Stage-0 is missing the runtime prints:
  ```
  [V1-radius] ERROR: missing Stage-0 iris/limbus radius for eye L. Refusing V1 tracking.
  ```
  and raises `SystemExit` (no tombstone written, no nystagmus inference, no
  silent overlay).
* [Track_V1_Vestibular_Neuritis.bat](Track_V1_Vestibular_Neuritis.bat) updated
  to pass `--no-mediapipe` so the per-frame loop runs without MediaPipe in the
  clinical workflow.

#### Verification on the vestibular-neuritis clip (no-mediapipe)

```
[V1-radius] L used=90.55 source=stage0
[V1-radius] R used=87.06 source=stage0
Tracking engine: v1
[v1-plausibility] enabled with 2 eye contour(s); outside_tol=20px, radius_band=[0.50,2.00] x Stage-0
```

* No `[anatomical]` logs.
* No `radius_too_small` rejects (0 / 385).
* All 385 plausibility rejects are `outside_contour` — i.e. legitimate cases
  where V1 placed the iris outside the static Stage-0 contour, almost
  certainly due to head motion in this hand-held clip. Without MediaPipe the
  contour reference no longer moves with the head; the right next step for
  head-motion compensation is rescue anchors (or a future moving-contour
  feature), not MediaPipe.
* V1 state counts (no MediaPipe, Stage-0-only radius):
  * R eye: TRACKING=184, valid_frames=194 (30.4 %).
  * L eye: TRACKING=183, valid_frames=207 (32.4 %).

#### What is removed from the V1 clinical path

* MediaPipe iris radius. Removed.
* MediaPipe iris landmarks for V1 init. Removed.
* Any fallback from Stage-0 iris to MediaPipe iris. Removed.
* Any fallback from iris to pupil. Was already removed in §17.
* Any MediaPipe-based eye selection. Not used in V1 anyway (V1 uses
  best-eye anatomical selection on the limbus fit, not MediaPipe).

#### What still uses MediaPipe (outside V1 clinical tracking)

The codebase retains some MediaPipe code paths for non-V1 use. These are
explicitly NOT part of the V1 clinical workflow:

* Stage 0 **proposal** generation (`propose_init`, `IrisDetector` calls in
  the assisted-approval UI). These run only during Stage 0 to seed the
  clinician's initial circle; the clinician then accepts or overrides.
* Optional face-landmark tracking when MediaPipe is enabled (the runtime's
  `FaceLandmarkTracker`). When `--no-mediapipe` is passed (the clinical
  default in the .bat), this is bypassed and only the Stage-0 face landmarks
  are used.
* Legacy engines (`--engine composite`, `--engine template`, experimental
  `--engine anatomical`). Not part of the V1 clinical path.

### Locked principle

> **No MediaPipe in V1 clinical tracking.**
> **Stage-0 clinician-approved anatomy is the only source of truth.**

> **If Stage-0 anatomy is missing, V1 refuses to run rather than falling
> back to MediaPipe.**

### 19b.5 Clinical clarification — why V1 does not need MediaPipe (2026-06-28)

The §19b.4 decision was made on engineering grounds (MediaPipe's radius is
structurally wrong for V1). The clinician immediately added a **clinical
clarification** that explains why MediaPipe is not just a wrong fallback —
it is not needed at all.

#### Clinical observation

In Indian patients, the iris is usually dark and clearly darker than the
white sclera. Therefore, after the clinician marks the eye and iris in
Stage 0, the software should be able to track the iris/limbus by looking
for:

* a dark iris,
* a bright sclera,
* the iris–sclera boundary,
* a visible curved limbus arc,
* an approximate radius and centre carried forward from Stage 0.

#### Core V1 clinical workflow

1. Stage 0 — clinician marks the full imagined iris/limbus circle.
2. Stage 0 — clinician marks the eye-opening / orbital contour.
3. V1 uses these clinician-approved marks.
4. V1 searches only in or near the approved eye region.
5. V1 looks for the dark iris against white sclera.
6. V1 does not use MediaPipe.
7. V1 does not search the whole face.
8. V1 does not use pupil radius.
9. V1 does not allow cheek / forehead / nose candidates.

#### What V1 may use

* Stage-0 iris / limbus centre.
* Stage-0 iris / limbus radius.
* Stage-0 eye-opening / orbital contour.
* Dark-to-bright iris–sclera contrast.
* Partial-arc limbus fitting.
* Temporal continuity from the previous accepted frame.
* Clinician rescue anchors if available.

#### What V1 must not use

* MediaPipe iris radius.
* MediaPipe iris landmarks.
* MediaPipe pupil / eye landmarks.
* Any fallback from iris to pupil.
* Whole-face search.

#### Partial-hidden iris rule

When the full iris is visible: fit the full dark iris / limbus circle.

When the iris is partly hidden:

* do not require a complete circle,
* use the visible dark iris–sclera arc,
* infer the full circle from that visible arc,
* keep radius close to Stage-0 radius,
* keep centre near the previous accepted centre,
* reject arcs from eyelid shadow, eyebrow, cheek, nose, forehead, or skin
  fold.

(The §19b.1 anatomical engine attempted these constraints but was
over-engineered and demoted to experimental — see §19b.2. The §19b.4
V1-only path now provides the same anatomical constraints through the
simpler combination of Stage-0 anatomy plus the lenient post-tracking
plausibility filter. The partial-hidden iris support comes from the
underlying `fit_limbus` RANSAC arc fitter, which V1 already uses.)

### Locked clinical principle

> **Clinician-approved Stage-0 anatomy is the source of truth.**
> **V1 tracks the dark iris against white sclera within that anatomy.**
> **No MediaPipe in V1 clinical tracking.**

### Note

The tracker remains anatomically constrained even when image evidence is poor. In
Indian eyes the useful signal is usually the dark iris against bright sclera; even when
partly visible, the iris–sclera boundary provides a curved arc from which the full iris
circle can be inferred, provided radius, eye-opening contour, and temporal continuity
are enforced.

---

## 19c. Stage 0 UI cleanup (2026-06-28)

The first attempt to redo Stage 0 on the vestibular-neuritis clip exposed two issues
with the approval UI:

1. **The left eye's contour oval would not move.** Root cause: the contour-mode
   handle search only looked at the *currently selected* eye's oval. A click on the
   other eye's oval found no handle and silently did nothing. The clinician was
   expected to know about a hidden `e` keyboard shortcut to switch which eye was
   active.
2. **The clinician UI showed face landmarks** (canthi + lid margins) as an editable
   mode (`f`). Now that V1 is fixed on iris + eye-opening contour as the only
   clinically meaningful editable anatomy, the face-mode controls are noise.

Two surgical edits to [iris_tracker.py:approve_interactive](iris_tracker.py):

* **FACE mode hidden from the UI.** Tab cycles `IRIS ↔ CONTOUR` only. The face dots
  are not drawn. The face landmarks are still detected by MediaPipe internally and
  still seed the initial eye-opening oval shape, and they are still written into
  `approved_landmarks.json` (so V1 default and any downstream code that consumes
  `face_landmarks` keep working). The anatomical engine does not use them.
* **Click any oval to make it active.** In CONTOUR mode the click first picks the
  eye whose oval centre is closest to the cursor (`_auto_select_contour_eye`), then
  finds the nearest handle of *that* eye. The `e` shortcut still works as a
  fallback.

### Principle (Stage 0 UI)

> **Stage 0 should show only clinically meaningful editable anatomy.**

---

## 20. Current committed checkpoints

The project currently has important Git checkpoints:

```text
453ca58 Update V1 architecture to limbus tracker with eye-opening contour reference
b13c4e0 Implement V1 limbus-contour tracker with best-eye anatomical selection
3225bcf Define arrow-based nystagmus detection as V1 primary output
```

These commits preserve the current architecture and make the project recoverable.

---

## 21. Current V1 architecture in one paragraph

eye_vng V1 tracks the iris/limbus circle as the moving anatomical object and tracks the eye-opening contour as the moving local reference frame. It measures the centre of the estimated iris circle relative to that contour. It uses the better-tracked eye by default, keeps anatomical Left/Right labels on the overlay, runs without mandatory MediaPipe when approved landmarks are available, and does not track the pupil. VNG-style traces are kept only as internal/debug outputs. The main clinical output is a clean video overlay that remains silent when there is no nystagmus and shows a direction arrow only when clinically confirmed rhythmic nystagmus is detected.

---

## 22. Current one-line rule

Track the iris circle.  
Track the eye-opening contour.  
Measure the iris-circle centre within that contour.  
Detect only true rhythmic fast phases.  
Show an arrow only for clinically confirmed nystagmus.

---

## 23. Locked principles (reference appendix, 2026-06-28)

This appendix consolidates the principles scattered through §1–§19 so a future reader
(or future model) can find them in one place. Each principle is a hard rule that has
been arrived at by failure analysis, not by guess. If a future change appears to
violate one of these, that change is suspect.

### 23.1 V1 core principle

V1 does **not** track the pupil. V1 tracks:

1. the full iris / limbus circle, and
2. the eye-opening / orbital margin contour.

The clinical signal is the iris/limbus centre **relative to the eye-opening
contour**. Pupil approval is invalid for V1; legacy fallback from iris to pupil must
never be used. A bad legacy pattern that caused a real false-negative on the known-
positive vestibular-neuritis clip was:

```python
approved.get("iris") or approved.get("pupils") or {}
```

This seeded V1 with pupil radii instead of iris/limbus radii and produced a false
"no nystagmus" result.

> **V1 tracks iris/limbus and eye-opening contour. Pupil is not a V1 target.**

### 23.2 Stage 0 is mandatory

Every video must begin with Stage 0. Stage 0 must approve:

* iris/limbus circle,
* eye-opening / orbital margin contour,
* usable-eye status.

If Stage 0 is missing, incomplete, or legacy-only, V1 tracking must refuse to run
and must not produce a "no nystagmus" result from invalid anatomy. The correct
non-result is a tombstone:

```json
{
  "analysis_valid": false,
  "rejection_reason": "invalid_stage0_approval",
  "clinical_nystagmus_detected": false,
  "clinical_overlay_silent": true
}
```

> **No approved anatomy, no clinical tracking.**

### 23.3 Eye nomenclature

Internal image-side convention (front-facing video → patient appears mirrored):

* image-side `L` = image-left eye = patient anatomical **Right** eye,
* image-side `R` = image-right eye = patient anatomical **Left** eye.

The clinician-facing UI must show patient anatomical Right / Left, not the
image-side labels. For current vestibular-neuritis testing the active teaching eye
is forced to patient anatomical Right, i.e. image-side `L`.

### 23.4 Coordinate-space bug (locked lesson)

The overlay video is **960 × 540** but `tracking.csv` and `approved_landmarks.json`
are in **1920 × 1080** source-video coordinates. The rescue scrubber initially
drew source coordinates directly onto the smaller overlay window, which placed the
iris marker on the cheek/nose and made the radius appear too large.

Locked rule:

* keep internal state in source-video coordinates,
* convert source → display only for rendering,
* convert display → source for mouse clicks,
* use non-blocking redraw so the display follows the mouse during a drag.

> **All stored iris coordinates are source-video coordinates. Display coordinates
> are only for rendering and mouse input.**

### 23.5 Rescue Scrubber lessons

The Rescue Scrubber is conceptually important because it uses tracker output and
lets the clinician correct failures. Manual testing went through four passes
(§18) before settling on the locked behaviour:

* the visible iris marker is **the editable object** — directly draggable,
* render loop uses non-blocking `cv2.waitKey(16)`, never `cv2.waitKey(0)`,
* exactly one active iris circle in iris mode,
* click near the circle = drag the same circle,
* click far away = recentre the same circle,
* `+` / `-` resize the same circle,
* `s` saves the rescue anchor; auto-save fires when leaving an edited frame,
* there is no second marker, no hidden marker,
* no marker stuck on the nose, no large cyan bridge/nose oval in iris mode,
* on-eye text such as `ACT-Right r=…` and the on-eye `UNPLACED IRIS` label are
  removed; status lives in the top HUD.

> **The visible marker must be the editable object.**

### 23.6 Tracker-assisted teaching workflow

The intended Rescue Scrubber workflow is **not** "tracker-only" and **not**
"manual-only" — it is **tracker-assisted**:

* tracker drives the iris marker on frames where tracking is correct,
* clinician drags/recentres and presses `s` to save a rescue anchor on frames where
  tracking is wrong,
* after a correction, that position becomes the local truth and tracking should
  continue from the correction, not snap back to the old failed tracker state.

Patches 1 and 2 (already landed, §18a) were UI / state fixes:

1. non-blocking refresh for smooth drag,
2. short propagation window in which a clinician correction wins over fresh
   tracker output before the tracker takes back over.

Patches 3 and 4 remain larger tracker work and are deferred:

3. restart V1 tracking from saved rescue anchors,
4. improve partial-arc iris/limbus tracking with sclera-drift penalty and
   anti-pupil-shrink terms.

> **Tracker-assisted teaching: the tracker does most of the work, the clinician
> corrects failures, and tracking resumes from the correction.**

### 23.7 Simple Iris Editor

The §19 Simple Iris Editor was created because the rescue scrubber had accumulated
too much logic and was fighting the clinician. The simple editor is a separate,
minimal tool that proves the correct UI interaction model:

* raw video only,
* one clinician-controlled iris circle,
* no baked tracking overlay,
* no cyan tracker circles,
* no detector text,
* no face dots,
* drag moves the circle smoothly,
* `+` / `-` resize,
* `s` saves,
* annotations saved to `manual_iris_annotations.json` in source-video coordinates.

Key bug fixed during construction: `cv2.waitKey(0)` blocked the render loop and
made dragging appear frozen. Fix: `cv2.waitKey(16)` (the same fix later applied to
the rescue scrubber as Patch 1).

> **The clinician owns the iris marker.**

### 23.8 Hybrid simple editor

The simple editor worked mechanically but lost the useful automatic tracker
guidance. The hybrid refinement (§19a) re-introduces the tracker **only** as a
faint, non-editable suggestion:

* clinician marker remains the editable truth,
* tracker is shown only as a faint cyan non-editable circle (read from
  `tracking.csv` if it exists),
* `t` copies the suggestion into the clinician marker for that frame,
* `h` hides / shows the suggestion,
* the tracker never overwrites the clinician marker unless the clinician
  explicitly accepts it,
* the save record gains a `"source"` field: `"manual"` or
  `"tracker_accepted_by_clinician"`.

> **Tracker suggests; clinician owns the annotation.**

### 23.9 Stage 0 UI cleanup

The Stage 0 approval UI should focus on the only anatomy V1 needs:

* iris / limbus circle,
* eye-opening / orbital oval (contour).

The minimal fix (§19c) is:

* hide the FACE mode from Stage 0,
* Tab cycles IRIS ↔ CONTOUR only,
* face landmarks may still be detected internally and saved for legacy / V1
  support, but they are not editable in the UI,
* in CONTOUR mode the click auto-selects whichever eye's oval is nearer, so the
  clinician does not need to press `e`.

> **Stage 0 should show only clinically meaningful editable anatomy.**

### 23.10 Anatomically constrained iris tracker

The iris tracker should **not** be a general circle tracker. It should be an
anatomically constrained iris–sclera boundary tracker. Clinical observation: in
Indian eyes the iris is usually dark against bright sclera; even when partly
covered by eyelid or at extreme gaze the visible iris–sclera boundary remains a
curved dark arc against sclera.

Tracker design:

* search only inside the Stage-0 approved eye-opening contour,
* never search the whole face,
* use the previous-frame centre / radius as a strong prior,
* keep radius close to the Stage-0 and previous-frame radius,
* do not shrink to the pupil,
* detect dark iris adjacent to bright sclera,
* accept partial visible arcs,
* fit the full imagined circle from the visible arc,
* if confidence is low, freeze the last good iris and mark `tracking_failed`,
* do not drift onto cheek, nose, eyebrow, eyelid shadow, or skin.

Important nuance the clinician explicitly insisted on: **the visible iris–sclera
arc must lie inside the Stage-0 eye-opening contour. The inferred full circle
does not have to.** At extreme lateral gaze the unseen part of the iris is allowed
to lie outside the contour, because it is imagined from the visible arc.

> **The iris tracker must fail safely, not wander.**
> **A missing iris is better than a wrong iris on the face.**

### 23.11 Existing limbus tracker — what it already does

On review (2026-06-28), `src/core/limbus.py` already implements much of what the
clinician's anatomical spec asks for:

* dark→bright radial edge detection,
* RANSAC / Kasa circle fitting,
* partial-arc circle fitting,
* radius constraints (`0.4 r_est < r < 1.8 r_est`),
* sclera whiteness gate (rejects iris→skin edges as iris→sclera edges),
* fixed-radius mode for occlusion-invariant centre after stable tracking,
* arc-coverage as a confidence signal.

What was missing:

* anatomical containment using the Stage-0 eye-opening contour as a hard polygon
  gate on edge evidence,
* strict freeze-on-fail behaviour at the orchestrator level,
* per-frame confidence / status reporting,
* rescue-anchor restart support.

Therefore the anatomical tracker **extends** the existing limbus tracker rather
than replacing it.

### 23.12 Anatomical tracker — implementation direction (landed)

The anatomical mode exists alongside V1 default rather than replacing it:

* `--engine anatomical` opt-in CLI choice,
* new file `src/core/iris_anatomical_tracker.py`,
* optional `polygon=` argument added to `limbus._radial_edges` and `limbus.fit_limbus`
  (backward compatible — omitting the arg keeps current behaviour),
* the Stage-0 contour is used as the edge-evidence containment mask,
* rescue anchors from `teaching_anchors.json` are loaded and absolutely win on
  their frames (they become the new "last good" and seed the next frame's search),
* low-confidence frames freeze the last good position and do not update the prior,
* `[anatomical][L]` / `[anatomical][R]` debug lines print only on non-OK frames.

V1 default remains the dispatch default until the anatomical mode is validated on
real clips.

### 23.13 Detector calibration lesson (negative-control)

The negative control clip `samples/2_short15.mp4` clinically has no nystagmus.
Earlier detector iterations produced false positives by labelling tiny repeated
velocity fluctuations as beats. Locked fixes / principles (already implemented):

* separate **debug candidates** from **clinical output**,
* impose a minimum fast-phase amplitude before clinical labelling,
* reject out-of-range eye-local (contour-relative) values,
* keep the clinical overlay silent unless a beat is clinically confirmed.

> **Tracking can be sensitive, but clinical labelling must be conservative.**

### 23.14 Files and working-tree caution

Important source files:

* [iris_tracker.py](iris_tracker.py)
* [src/core/limbus.py](src/core/limbus.py)
* [src/core/v1_tracker.py](src/core/v1_tracker.py)
* [src/core/iris_rescue.py](src/core/iris_rescue.py)
* [src/core/simple_iris_editor.py](src/core/simple_iris_editor.py)
* [src/core/iris_anatomical_tracker.py](src/core/iris_anatomical_tracker.py)
* [src/core/eye_contour.py](src/core/eye_contour.py)
* [app.py](app.py)

Important docs:

* [EYE_VNG_DEVELOPMENT_HISTORY.md](EYE_VNG_DEVELOPMENT_HISTORY.md)
* [TRACKER_REDESIGN.md](TRACKER_REDESIGN.md)
* [V1_CURRENT_CHANGES_AND_RATIONALE.md](V1_CURRENT_CHANGES_AND_RATIONALE.md)
* [HANDOFF.md](HANDOFF.md)

Important launchers (the venv has cv2, the user's bare PowerShell `python` does
not, so always use the `.bat`s or call `.venv\Scripts\python.exe` explicitly):

* [Approve_Vestibular_Neuritis.bat](Approve_Vestibular_Neuritis.bat)
* [Rescue_Vestibular_Neuritis.bat](Rescue_Vestibular_Neuritis.bat)
* [Annotate_Vestibular_Neuritis.bat](Annotate_Vestibular_Neuritis.bat)
* [Annotate_Iris_DragDrop.bat](Annotate_Iris_DragDrop.bat)
* [Track_Anatomical_Vestibular_Neuritis.bat](Track_Anatomical_Vestibular_Neuritis.bat)

Do **not** commit generated outputs:

* `outputs/` (whole tree),
* `tracking_overlay.mp4`,
* preview PNGs,
* teaching-example PNGs,
* per-clip JSON outputs.

Be careful with `landmark_calibration.json` — it may contain local GUI calibration
that should not be committed unless deliberately intended.

### 23.15 Current state / next validation

Before committing any of the recent work, the following must be visually verified
end-to-end:

1. Stage 0 approval allows **both** eye-opening ovals to move (click-to-select).
2. Face dots are hidden in the Stage 0 UI.
3. The anatomical tracker uses the newly drawn Stage-0 contours as containment.
4. When testing pure anatomical tracking, rescue anchors are temporarily moved
   aside (`ren teaching_anchors.json teaching_anchors.json.bak`) — otherwise the
   anchors win absolutely on their frames and mask the tracker's true behaviour.
5. The anatomical tracker does not wander into face / cheek / nose on any frame.
6. On low-confidence medial-gaze frames it freezes or stays near the eye; it
   does not hallucinate a confident circle elsewhere.
7. The Rescue Scrubber still allows smooth drag correction.
8. Saved rescue anchors still write to `teaching_anchors.json`.

Until those eight checks pass on the vestibular-neuritis clip the work is not
ready to commit.

---
