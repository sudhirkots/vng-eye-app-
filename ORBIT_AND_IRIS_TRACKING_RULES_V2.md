# ORBIT_AND_IRIS_TRACKING_RULES_V2

## Purpose

This document defines how EYE_VNG V2 should combine:

1. orbit / eye-opening tracking,
2. sclera recognition,
3. iris / limbus tracking,
4. eye-local coordinate conversion,
5. later nystagmus analysis.

The goal is to prevent the tracker from following face motion, eyebrow hair, eyelid shadows, cheek folds, nose shadows, or random dark structures.

The system must track the **eye as an anatomical unit**, not as isolated image features.

---

## Core Clinical Model

The eye-tracking model is:

```text
face → orbit / eye opening → sclera → iris / limbus → iris centre → eye-local movement
```

The orbit oval defines the **reference frame**.

The sclera–iris complex defines the **eye content**.

The iris centre, measured relative to the moving orbit oval, becomes the future clinical movement signal.

---

## Foundational Rule

```text
The orbit oval is the moving anatomical reference frame.
The iris is tracked only within that moving reference frame.
The sclera validates that the iris candidate is truly inside the eye.
```

---

## V2 Layered Architecture

### Layer 1 — Face / Eye-Zone Suggestion

MediaPipe may be used only as an early helper.

Allowed use:

* find rough face position,
* suggest approximate eye zones,
* suggest a Stage-0 frame for clinician marking.

Not allowed:

* iris centre,
* iris radius,
* limbus,
* pupil,
* clinical tracking,
* nystagmus signal.

Locked rule:

```text
MediaPipe suggests where to mark. The clinician marks anatomy.
```

---

### Layer 2 — Clinician Stage-0 Marking

The clinician marks, for each eye:

1. medial point,
2. lateral point,
3. upper point,
4. lower point,
5. full imagined iris / limbus circle.

The four oval points define the eye-opening reference.

The iris / limbus circle defines the expected full iris model.

Stage-0 anatomy is the clinical source of truth.

---

### Layer 3 — Moving Orbit / Eye-Opening Reference

The orbit oval must move with the eye / face region.

The oval should be tracked as one anatomical object, not as four unrelated points.

The oval may translate, rotate, and mildly scale.

It must not drift independently to:

* eyebrow,
* cheek,
* nose,
* forehead,
* skin fold,
* eyelid shadow.

The moving oval defines the eye-local coordinate system.

---

### Layer 4 — Sclera–Iris Content Validation

The tracker must confirm that the oval still contains the eye.

The valid eye-content complex is:

```text
white sclera + dark circular or arc-like iris / limbus
```

White alone is not enough.

Dark alone is not enough.

The pair is the eye.

---

### Layer 5 — Iris / Limbus Tracking

The iris is tracked as a full circle or partial circular arc.

The system should track the full limbus circle even when only a segment is visible.

The radius must remain close to the Stage-0 approved limbus radius.

The tracker must not shrink to the pupil.

---

### Layer 6 — Eye-Local Coordinate Conversion

The iris centre must be converted from image coordinates into eye-local coordinates.

The local axes are:

```text
horizontal axis = medial → lateral
vertical axis = lower → upper
```

The iris movement must be analysed relative to these axes, not raw image x/y.

This subtracts head translation, head tilt, and camera movement as much as possible.

---

### Layer 7 — Nystagmus Analysis

Nystagmus analysis comes later.

It should use the eye-local iris centre trace.

The detector should look for:

* rhythmic repeated movement,
* slow phase,
* fast phase,
* consistent direction,
* clinically plausible frequency and amplitude.

Random or arrhythmic movement should be ignored.

---

## Orbit Tracking Rules

The orbit oval should be tracked from the clinician-marked Stage-0 oval.

Accept orbit tracking if:

1. the oval remains over the visible eye opening,
2. the oval contains scleral white,
3. the oval contains or is adjacent to the dark iris / limbus,
4. the medial–lateral axis remains aligned with the eye opening,
5. the upper–lower axis remains aligned with the lid opening,
6. the oval moves smoothly frame to frame.

Reject or mark uncertain if:

1. the oval drifts to eyebrow,
2. the oval drifts to cheek,
3. the oval drifts to nose or forehead,
4. the oval contains mostly skin,
5. the oval no longer contains sclera,
6. the oval no longer contains plausible iris / limbus support,
7. motion is too abrupt or anatomically implausible.

If the oval cannot be tracked:

```text
status = orbit_uncertain / orbit_lost
```

Do not hallucinate.

---

## Iris Tracking Rules

The iris candidate must satisfy:

1. inside the moving orbit oval,
2. supported by true scleral white,
3. dark relative to sclera,
4. circular or arc-like,
5. capable of completing to a plausible full limbus circle,
6. radius close to Stage-0 approved radius,
7. centre anatomically plausible within the eye opening,
8. temporally plausible relative to the previous good frame.

If not satisfied:

```text
status = iris_uncertain / iris_lost
```

Freeze last good position rather than jump.

---

## Sclera Rule

Sclera is not just “white pixels.”

Accept as sclera:

* broad white eye-shaped region,
* relatively continuous white patch,
* anatomically inside the eye opening,
* spatially related to the dark iris.

Reject as sclera:

* tiny bright reflections,
* shiny skin highlights,
* patchy glare,
* eyebrow white hair,
* alternating black-white hair texture,
* skin blotches,
* background light spots.

The sclera must help distinguish iris from eyebrow / eyelid / cheek artefacts.

---

## Iris Arc Rule

The iris may be seen as:

1. full circle,
2. near-full circle,
3. half circle,
4. crescent-like visible segment,
5. medial arc,
6. lateral arc,
7. superior arc,
8. inferior arc.

A visible arc is acceptable only if:

1. curvature is compatible with a circle,
2. radius matches Stage-0 limbus radius,
3. completed circle centre is plausible,
4. arc is adjacent to scleral white,
5. arc lies inside the orbit oval.

Reject:

* wiggly dark lines,
* eyebrow hairs,
* lashes,
* lid margin shadow without scleral support,
* skin folds,
* cheek creases,
* random dark blobs.

---

## Blink / Occlusion Rule

During blink:

* sclera may disappear,
* iris may disappear,
* eyelids may cover the eye.

This is not a reason to search the face.

During short occlusion:

```text
hold last good orbit
hold last good iris
status = blink_hold / occluded_hold
```

When sclera–iris content returns near the predicted location:

```text
reacquire
```

If occlusion persists too long:

```text
status = lost / needs_rescue
```

---

## Head Movement Rule

Head movement should not be mistaken for eye movement.

The orbit oval must follow the head / eye region.

The iris movement must then be measured relative to the moving oval.

Raw image movement is not the clinical signal.

Clinical signal:

```text
iris centre relative to eye-local axes
```

Not:

```text
iris centre in raw image pixels
```

---

## Eye-Local Coordinate System

For each frame, define:

```text
O = centre of orbit oval
H-axis = medial → lateral
V-axis = lower → upper
```

Then express iris centre as:

```text
iris_h = projection of iris centre onto H-axis
iris_v = projection of iris centre onto V-axis
```

This allows horizontal and vertical eye movement to be judged relative to the patient’s own eye orientation.

If the head tilts, the eye-local axes tilt with it.

---

## Coupled Orbit–Iris Logic

Orbit and iris should constrain each other.

### Orbit helps iris

The orbit oval tells the iris tracker where it is allowed to search.

The iris must remain anatomically plausible inside the oval.

### Iris helps orbit

The sclera–iris complex helps confirm that the oval is still over the eye.

If the oval drifts away from sclera and iris, the oval is wrong.

### Combined rule

```text
The correct target is not orbit alone and not iris alone.
The correct target is the eye-content complex inside the orbit oval.
```

---

## Candidate Scoring

A candidate orbit / iris state should be scored by:

1. orbit geometry stability,
2. orbit motion smoothness,
3. sclera area inside oval,
4. sclera compactness / continuity,
5. dark iris arc support,
6. iris radius similarity to Stage-0,
7. iris centre plausibility,
8. containment inside oval,
9. temporal continuity,
10. blink / occlusion status.

Accept only if the combined score is anatomically plausible.

---

## Hard Rejection Conditions

Reject immediately if:

1. iris candidate lies outside the moving orbit oval,
2. dark feature has no scleral support,
3. dark feature is hair-like or wiggly,
4. white feature is hair / reflection / glare rather than sclera,
5. completed circle radius is grossly wrong,
6. completed circle centre is anatomically impossible,
7. orbit oval is on eyebrow / cheek / forehead / nose,
8. candidate jumps implausibly from last good state.

---

## Failure Behaviour

On failure:

1. mark uncertain / lost,
2. freeze last good anatomical state,
3. do not search the whole face,
4. do not jump to a new dark object,
5. allow clinician rescue correction.

The system must fail safely.

---

## Rescue Principle

Rescue correction is part of the clinical workflow.

If automatic tracking fails:

1. clinician corrects orbit and/or iris,
2. correction becomes a local anchor,
3. tracking resumes from that corrected anatomy,
4. later analysis records that the segment was rescue-assisted.

The goal is not fully automatic tracking at all costs.

The goal is reliable, clinician-supervised measurement.

---

## Milestone Order

### Milestone 1 — Orbit tracking only

Goal:

```text
Can the clinician-marked orbit oval remain locked to the moving eye region?
```

No iris tracking.

No nystagmus.

---

### Milestone 2 — Orbit + sclera validation

Goal:

```text
Can the moving oval stay over the eye by using sclera as content support?
```

No final iris signal yet.

---

### Milestone 3 — Iris / limbus tracking inside orbit

Goal:

```text
Can the iris circle or arc be tracked inside the moving oval?
```

Still no nystagmus detection.

---

### Milestone 4 — Eye-local movement trace

Goal:

```text
Can iris centre be converted into eye-local horizontal and vertical traces?
```

---

### Milestone 5 — Nystagmus detection

Goal:

```text
Can rhythmic slow-phase / fast-phase movement be detected from the eye-local trace?
```

---

## What Must Not Happen

The tracker must not:

* track the pupil instead of iris,
* use MediaPipe iris radius,
* use MediaPipe iris centre,
* search the whole face,
* jump to eyebrow hair,
* jump to lid shadow,
* jump to cheek crease,
* jump to nose shadow,
* treat raw head movement as eye movement,
* produce “no nystagmus” from invalid tracking,
* hallucinate a confident result when the eye is not visible.

---

## Practical One-Line Rule

```text
Keep the moving orbit oval locked to the eye, identify the dark iris arc only when supported by scleral white inside that oval, complete it to the Stage-0 limbus circle, and analyse the iris centre only in eye-local coordinates.
```

---

## Locked Principles

1. Clinician-approved Stage-0 anatomy is the source of truth.
2. MediaPipe is only a Stage-0 assistant, never a clinical tracker.
3. The orbit oval is the moving reference frame.
4. The sclera–iris complex validates the eye.
5. The iris is a full or partial limbus circle, not a dark blob.
6. False negatives are acceptable; false positives are dangerous.
7. If uncertain, freeze and ask for rescue.
8. Nystagmus must be analysed in eye-local coordinates, not raw image coordinates.

---
