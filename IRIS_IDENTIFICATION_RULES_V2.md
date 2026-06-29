# IRIS_IDENTIFICATION_RULES_V2

## Purpose

This document defines how **EYE_VNG V2** should identify and track the iris.  
Its purpose is to prevent drift onto eyebrows, eyelashes, cheek, nose, forehead, shadows, or reflections.

The tracker must not behave like a general circle detector.  
It must behave like an **anatomically constrained iris–sclera boundary tracker**.

---

## Core Clinical Definition

The iris is defined as:

> **A dark full or partial circular structure, supported by surrounding scleral white, lying inside the Stage-0 approved orbital oval, from which a full limbus circle can be plausibly completed.**

This is the main rule.

The tracker must not ask only:

- “Is this a dark round thing?”

It must ask:

- “Is this a dark circular or arc-like structure,
- inside the approved orbital oval,
- with proper scleral white support,
- and can it plausibly represent the full iris circle?”

---

## Key Clinical Observations

1. In most Indian eyes, the **iris is dark** relative to the **white sclera**.
2. The iris may be seen as:
   - a full circle,
   - a nearly full circle,
   - a large segment,
   - or only a partial arc.
3. Even when partly hidden by eyelid or extreme gaze, the visible iris–sclera boundary often remains visible as a **dark curved arc against white sclera**.
4. The tracker should use that visible arc to infer the **full iris / limbus circle**.
5. The iris should always remain within the clinician-approved **orbital oval / eye opening**.
6. A temporary miss is acceptable. A false lock onto face, hair, eyebrow, or shadow is **not** acceptable.

---

## Primary Design Principle

### Fail safely. Do not wander.

If the tracker is uncertain:

- freeze the last good iris,
- mark tracking uncertain or failed,
- allow rescue / clinician correction,
- **do not jump** to another dark structure.

A false negative is better than a false positive.

---

## Stage-0 Source of Truth

Stage-0 must provide:

1. the approved **orbital oval / eye opening**
2. the four reference points:
   - medial
   - lateral
   - upper
   - lower
3. the approved **iris / limbus circle**
4. the approved **iris radius**

These are the clinical ground truth for initialization.

---

## Search Constraints

The tracker must obey these constraints:

1. **Search only inside the Stage-0 approved orbital oval**, with only a small safety margin.
2. **Never search the whole face** for the iris.
3. **Never allow the iris centre to drift outside a plausible eye region**.
4. **Use the previous valid iris centre and radius as strong priors**.
5. **Keep the radius close to the Stage-0 approved limbus radius**.
6. **Do not shrink to the pupil**.
7. If no plausible candidate is found, **freeze** rather than drift.

---

## Positive Iris Rules

A structure may be accepted as iris only if all or most of the following are satisfied.

### 1. Orbital Containment
The candidate must lie inside the approved orbital oval (or within a very small allowed margin).

### 2. Scleral Support
There must be meaningful surrounding **white scleral support**.

This means:
- the white region is broad and eye-like,
- not patchy,
- not a small reflection,
- not isolated bright noise.

### 3. Dark Iris Contrast
The candidate must contain a **dark brown / dark grey / black region** relative to sclera.

### 4. Circular or Arc-like Geometry
The dark region must behave like:
- a full circle, or
- a circular arc, or
- a segment of a circle.

### 5. Circle Completion
From the visible arc, the system should be able to plausibly infer the **full iris / limbus circle**.

### 6. Radius Plausibility
The completed circle must have a radius close to the approved Stage-0 iris radius, within a reasonable tolerance.

### 7. Position Plausibility
The completed circle must occupy a plausible anatomical position inside the eye opening.

### 8. Temporal Plausibility
The new centre/radius must be reasonably close to the previous valid frame, unless there is a true large eye movement.

---

## Valid Positive Examples

The following should be accepted as valid iris configurations:

1. **Full iris visible**
2. **Large iris segment visible during lateral gaze**
3. **Partial iris arc visible because of eyelid coverage**
4. **Inferior visible arc**
5. **Superior visible arc**
6. **Medial or lateral visible arc**
7. **Any visible dark arc within scleral white that plausibly completes to the full iris circle**

---

## Negative / Reject Rules

The following must be rejected as iris candidates.

### 1. Eyebrow Hair
Dark wiggly lines with no proper scleral support.

### 2. Eyelashes
Fine dark linear structures, often repeated, not circular.

### 3. White Hair / Mixed Hair
Alternating black-and-white streaks; not a broad scleral field.

### 4. Lid Shadow
A shadow edge without true dark-circle / white-sclera geometry.

### 5. Cheek / Nose / Forehead / Skin Fold
Even if there are dark and bright patches, they do not form a sclera-supported circular iris structure.

### 6. Reflections / Glare
Patchy bright highlights do not count as sclera.

### 7. Dark Blob Without Circle Logic
Any dark patch that is not consistent with a full or partial iris circle.

### 8. Candidate Outside Orbital Oval
Anything outside the approved oval must be rejected.

### 9. Implausible Radius
Anything too small (pupil-like) or too large must be rejected.

---

## What Counts as “White”

The tracker must be taught what counts as true scleral white.

### Accept as sclera:
- a broad eye-shaped white region,
- relatively uniform pale area,
- spatially consistent with the eye opening,
- often surrounding a dark iris.

### Reject as non-sclera:
- tiny bright reflections,
- patchy glare,
- shiny skin highlights,
- eyebrow white hair,
- irregular white blotches on skin,
- noise.

The sclera is not just “something white.”  
It is a **broad anatomical white region of the eyeball**.

---

## Circle Completion Rule

When the full iris is not visible, the tracker must infer the full circle from the visible arc.

### Required logic:
1. detect the visible dark arc
2. estimate curvature
3. compare with approved Stage-0 radius
4. infer the full circle centre
5. check whether the completed circle lies plausibly inside the orbital oval
6. confirm surrounding scleral support

If the arc cannot support a plausible completed circle, reject it.

---

## Radius Rule

The system must track the **limbus / full iris**, not the pupil.

### Therefore:
- the approved Stage-0 iris radius is the main radius prior
- the tracker must not collapse inward to the pupil
- moderate variation is acceptable
- large shrinkage is not acceptable unless specifically justified by image scale change

---

## Temporal Continuity Rule

The tracker must prefer continuity.

For each new frame, score candidates by:

1. closeness to previous centre
2. closeness to previous radius
3. closeness to Stage-0 radius
4. arc support length
5. dark-inside / bright-outside contrast
6. containment inside the orbital oval
7. temporal smoothness

If confidence falls below threshold:
- do not drift,
- do not jump,
- freeze last good state.

---

## Blink / Transient Occlusion Rule

During blink or lid closure:

1. the iris and sclera may transiently disappear
2. this should not trigger search over the whole face
3. the tracker should:
   - freeze the last good iris,
   - wait for recovery,
   - resume from the last valid prior

Blink is a temporary absence, not a reason to wander.

---

## Combined Eye Logic

The ideal tracker should use both:

1. **orbital containment**
2. **iris–sclera recognition**

This means the system should track:

- the orbital oval / eye opening, and
- the iris within it,

simultaneously or in a tightly linked way.

The orbital oval constrains where the iris may be.  
The iris–sclera structure confirms what the eye actually is.

---

## Recommended Candidate Scoring

Each iris candidate may be scored using:

- **S1**: dark-inside / bright-outside contrast
- **S2**: visible arc support length
- **S3**: radius similarity to Stage-0 radius
- **S4**: distance from previous centre
- **S5**: containment within the orbital oval
- **S6**: temporal smoothness
- **S7**: scleral support quality
- **S8**: circle / arc plausibility

Only candidates with adequate combined score should be accepted.

---

## Hard Rejection Conditions

Immediately reject if:

1. centre is outside the plausible eye box
2. candidate is outside orbital oval
3. radius is grossly wrong
4. no scleral support exists
5. dark structure is linear / hairy / irregular rather than circular or arc-like
6. candidate is on brow, lash line, cheek, nose, or forehead

---

## Behaviour on Failure

If no acceptable candidate exists:

1. mark `tracking_uncertain` or `tracking_failed`
2. keep the last good iris
3. do not invent a new iris
4. do not jump to the face
5. allow clinician rescue / correction

---

## Summary Rule

### The tracker must recognize the iris as:

- dark,
- circular or arc-like,
- supported by scleral white,
- inside the approved orbital oval,
- and plausibly completable to a full limbus circle.

### It must reject:

- dark hair,
- lid shadow,
- patchy glare,
- skin texture,
- and anything outside the eye opening.

### If uncertain:
- freeze,
- do not drift.

---

## Practical One-Line Rule

> **Track only a dark full or partial circular iris boundary supported by scleral white and contained within the approved orbital oval. If uncertain, freeze rather than wander.**

---

## Change Log Rationale

This document exists because the earlier tracker definition of “what counts as iris” was too loose.

The main failure was not only drift.  
The deeper problem was that the tracker sometimes accepted dark non-iris structures as candidate iris targets.

This specification corrects that by enforcing:

1. scleral support
2. circular / arc-like geometry
3. Stage-0 radius prior
4. orbital containment
5. fail-safe freezing instead of wandering

---