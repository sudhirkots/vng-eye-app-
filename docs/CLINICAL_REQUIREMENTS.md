# EyeVNG Clinical Requirements

*This document captures the clinical reasoning behind EyeVNG design decisions. It is not a software
architecture document — it explains **why** the software behaves the way it does. For tracking design
see `docs/TRACKING_PHILOSOPHY.md`; for requirements/architecture see
`PROJECT_VISION_AND_REQUIREMENTS.md` and `SYSTEM_ARCHITECTURE.md`.*

## Project Goal

EyeVNG is intended to become a smartphone-based vestibular eye movement analysis system.

The objective is to approach the clinical usefulness of:

* Video Frenzel
* VNG
* Partial vHIT functionality

using ordinary smartphone videos.

The software is intended for vestibular and neuro-otological examination.

---

## Clinical Philosophy

Accurate measurement is more important than diagnosis.

Diagnosis should only occur after reliable measurement.

Order of importance:

1. Reliable tracking
2. Reliable traces
3. Reliable metrics
4. Reliable diagnosis

---

## Human Confirmation Principle

Human-confirmed anatomy is the source of truth.

AI landmark detection is only a proposal.

Reason:

Repeated AI landmark detection creates jitter and false movement.

Therefore:

* landmarks should be proposed
* landmarks should be confirmed by the clinician
* tracking should follow confirmed anatomy

---

## Landmark Confirmation Requirements

Before tracking begins:

The clinician should confirm:

* nose bridge
* nose tip
* cheeks
* ears if visible
* eye corners
* pupil circles

Tracking should begin only after landmark approval.

---

## Pupil Versus Iris

The primary measurement target is the pupil.

The pupil should be represented by:

* centre
* radius

The user should be able to:

* move pupil centre
* resize pupil circle

The pupil circle should cover only the dark pupil.

It should not cover the entire iris.

---

## Blink Handling

When the pupil is hidden:

* do not invent coordinates
* mark blink or occlusion
* attempt reacquisition
* request user confirmation if reacquisition fails

Missing data is preferable to false data.

---

## One-Eye Fallback

Tracking should ideally use both eyes.

However:

Analysis should still proceed if only one eye can be tracked reliably.

---

## Raw Data Principle

Raw measurements must always be preserved.

No hidden smoothing.

No hidden interpolation.

No artificial eye movements.

The clinician should always be able to review the raw trace.

---

## Overlay Verification Principle

The overlay video is a primary clinical output.

The clinician must be able to verify:

* what is being tracked
* where the pupil is
* which landmarks are being followed

The overlay acts as a quality-control tool.

---

## Torsional Eye Movement

Horizontal and vertical eye movement can be estimated from pupil tracking.

Torsional eye movement cannot.

Future torsional analysis will require:

* iris texture
* iris crypts
* limbus
* conjunctival vessels
* scleral features

Therefore high-quality eye ROI videos must be preserved.

---

## Head Impulse Philosophy

Current goal:

Positive versus negative head impulse.

Future goal:

Approximate VOR gain estimation.

Head pose tracking should therefore be preserved from the beginning.

---

## Current Version Goal

Current version success criterion:

The clinician watches the overlay and agrees that:

"The tracked pupil marker remains attached to the true pupil and follows it naturally through the recording."

Only after this is achieved should further diagnostic modules be added.
