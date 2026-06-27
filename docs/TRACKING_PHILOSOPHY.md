# EyeVNG Tracking Philosophy (Revised)

> **⭐ V1 DESIGN UPDATE (2026-06-27): single-eye, LIMBUS + CONTOUR tracking.** For Version 1 the
> internal *measurement* is built from ONE user-selected eye. The moving object is the estimated
> full iris circle/ellipse fitted from the visible limbus / iris-sclera boundary. The moving
> reference frame is one manually approved and tracked eye-opening contour, not four independently
> tracked points. MediaPipe is proposal/fallback/reacquisition/quality only — never the per-frame
> measurement. The head-motion-compensation section below is retained only for future
> head-impulse/VOR work where explicitly stated.

> **⭐ V1 USER-FACING OUTPUT (2026-06-27): NYSTAGMUS DETECTION.** The V1 clinical *display* is
> NOT a VNG-style position trace. It is the overlay video answering: *Is nystagmus present?*
> and *What is the beating direction?* (left / right / up / down / oblique; torsional only when a
> rotation signal is implemented). The contour-relative iris-centre signal described in this
> document is the INTERNAL analysis input to that detector — it is the right *measurement* to
> build, but it is no longer the right *display*. Iris-circle centre motion CAN detect horizontal,
> vertical, and oblique nystagmus. It CANNOT detect torsion; torsion requires iris-texture
> rotation tracking inside the iris circle, and until that exists torsion is reported as
> `not_assessed`.

**Status: SOURCE OF TRUTH for all tracking-related design decisions.** Where any other document
(`VISION.md`, `SYSTEM_ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`,
`STABLE_TRACKING.md`, `HANDOFF.md`) disagrees about how anatomy/iris are tracked, this document
wins. Last revised: 2026-06-25.

**The central shift:** the old design assumed *"detect landmarks continuously"*. The revised
philosophy is *"detect once → confirm anatomically → track continuously."*

---

## Core Principle

EyeVNG is not a landmark detection system.

EyeVNG is a landmark tracking system.

Detection and tracking are fundamentally different.

Detection identifies anatomical structures.

Tracking follows already-confirmed anatomical structures through time.

The software should minimise repeated detection and maximise continuous tracking.

---

# MediaPipe Role

MediaPipe is primarily a landmark proposal system.

Its role is:

* locate the face
* locate the eyes
* optionally propose the selected eye-opening contour
* optionally propose the iris/limbus boundary

MediaPipe should NOT be considered the final source of measurements on every frame.

Repeated frame-by-frame detection creates:

* landmark jitter
* iris jitter
* artificial movement
* noisy eye traces

MediaPipe should therefore be used for:

1. Initial landmark proposal
2. Landmark reacquisition after tracking failure
3. Landmark reacquisition after blink or occlusion

MediaPipe should not be the primary measurement source after initialization.

---

# Anatomical Confirmation Workflow

Before analysis begins:

The software should display the first good frame.

The user should confirm important anatomical landmarks.

The user becomes the final authority regarding landmark identity.

---

# Eye-Opening Contour Confirmation

For V1, the reference frame is the visible eye-opening contour: the palpebral fissure / eyelid
opening outline.

The software may propose this contour, or load it from `approved_landmarks.json`.

The user should confirm or correct:

* the selected eye-opening contour
* the visible medial extent of the contour
* the visible lateral extent of the contour
* the visible upper and lower contour boundaries

The medial, lateral, upper, and lower reference values are derived from this contour geometry. They
are not isolated points tracked independently.

If the contour cannot be confidently tracked later, the raw iris trace may remain usable, but the
contour-relative clinical trace must be marked `reference_uncertain`.

---

# Iris Confirmation

The software should display enlarged eye views.

The software may propose an iris location.

The user should:

* confirm the visible limbus / iris-sclera boundary
* review the estimated full iris circle/ellipse inferred from the visible boundary
* correct the centre/radius/axes only as a way of fitting the iris boundary, not as a pupil marker

The confirmed circle should cover the visible iris, out to the limbus.

The circle represents the iris boundary, not the smaller pupil.

Store:

* limbus / iris-boundary representation
* estimated full iris circle/ellipse
* iris-circle centre
* iris radius or axes

for each eye.

If only one eye is visible:

continue with one-eye tracking.

---

# Landmark Review and Approval

**Core principle: human-confirmed anatomy is the source of truth.** AI landmark detection (MediaPipe)
is only a *proposal*. **Tracking must not begin until the user explicitly approves the selected eye's
eye-opening contour and iris/limbus boundary.** This is a permanent project requirement, not a
temporary implementation detail.

**Why confirmed anatomy is preferred over repeated detection:** per-frame detection jitters and can
silently mislabel or drift between structures, creating artificial movement. A human confirming the
anatomy *once* gives a stable, correct reference that tracking then follows. MediaPipe is therefore
used primarily for **initialization** (proposing landmarks) and **re-acquisition** (after loss, blink,
or occlusion) — never as the per-frame source of truth.

Workflow:

1. **Anatomy proposal** — the software may propose the eye-opening contour and iris/limbus boundary
   on the best initial frame.
2. **Anatomy editing** — the user may correct the contour, resize/refit the iris circle/ellipse, and
   mark any not visible as unavailable.
3. **Landmark approval** — the user explicitly approves the set. Nothing is tracked before this.
4. **Approved anatomy becomes tracking targets** — persisted to `approved_landmarks.json` and used as
   the initial reference by every tracking module.
5. **MediaPipe becomes fallback only** — consulted afterwards solely for re-acquisition.

**Stage 0 is an INTERACTIVE confirmation screen, not silent auto-detection.** Running `--approve`
must: open the chosen frame; visibly **mark and label** the selected eye's eye-opening contour,
visible limbus arc, estimated full iris circle/ellipse, and iris-circle centre; ask the user to
confirm or correct them; save the corrected set to `approved_landmarks.json`; and **NOT proceed to
tracking** (approval only saves). The clinician must never have to trust automatic detection blindly.

---

# Iris and Eye-Opening Contour Are Independent

**The iris circle is the moving object. The eye-opening contour is the moving reference frame.** They
are two separate anatomical objects and must be tracked separately.

- The limbus / iris-boundary tracker estimates the visible limbus arc, the full iris circle/ellipse,
  and the iris-circle centre from image evidence. This is the primary clinical tracker.
- The eye-opening contour tracker follows the approved palpebral fissure contour as one shape. It
  defines the reference frame used to express the iris-circle centre in eye-local coordinates.
- CFT/internal iris features may assist prediction, stabilization, weak-fit support, or consistency
  checking, but the CFT centre is not the primary clinical centre.
- MediaPipe may propose or reacquire anatomy, but it is optional and not required when approved
  landmarks already exist.
- Pupil tracking, pupil darkness, and pupil-centre terminology are not part of V1.
- Output **both**: (a) raw iris-circle centre in image space, and (b) contour-relative eye position.
  If the iris is good but the contour is uncertain, keep the raw iris trace and mark the corrected
  relative trace `reference_uncertain`.

**Permanent audit rule:** no face-transform / landmark shift / affine / face-template motion may be
applied directly to iris x/y. Keep raw iris-circle centre and contour-relative clinical coordinates as
separate outputs.

**Drift fix (2026-06-25):** the earlier iris tracker searched a template around the *previous iris
position*; during head movement the iris left the window and the template locked onto a wrong dark
region (lash/brow/cheek) and drifted off the eye (493px median off-iris on the head-movement clip),
while still reporting "tracked". **Fixed:** the search box is now centred every frame on the **robust
eye location** (MediaPipe iris, which follows the head), the iris is found *inside* that box,
and any match outside the iris is rejected — so the marker physically cannot leave the eye. MediaPipe
positions the search box only; it is not forced to be the iris result. Verified: off-iris gap
493px → ~10px median; the frame that used to drift onto the jaw now keeps the marker on the eye.

**Eye-in-head — V1 method: CONTOUR-RELATIVE (2026-06-27).** The reference is the tracked
eye-opening contour. The medial/lateral/upper/lower reference values are derived from the contour's
geometry each frame. The iris-circle centre's displacement within those contour-derived limits is the
`corrected_*_eye_h/v` signal. A whole-face affine transform and isolated canthus points are future or
legacy ideas, not the V1 reference rule.

**Shimmer vs nystagmus (key principle, 2026-06-25).** Measurement-noise "shimmer" and nystagmus fast
phases share the same high-frequency band, so **no temporal filter can remove one without denting the
other**. Therefore: preserving the beats always wins (the filter is beat-preserving/near-passthrough);
reduce noise at the **source** (better iris detection), not by smoothing. The **overlay shows RAW**
iris circle/centre (verification — must sit on the iris, no lag). For V1, the **filtered
contour-relative signal is the INTERNAL input to the nystagmus detector** — it is not the V1
clinical display; the V1 clinical display is the nystagmus arrow + label on the overlay video.
Raw values are always kept in the CSV (audit trail).

**Detector reading vs reader reading.** Random-looking movements and isolated saccades must not
be classified as nystagmus by the detector — V1 requires repeated beats, direction consistency,
and approximate rhythmicity (see `EYEVNG_TRACKING_SPECIFICATION.md` §K). Debug-only position and
velocity plots may still be saved to the output directory for developer verification, but they
are not the primary clinical artefact.

---

# Continuous Tracking

After confirmation:

Track the same anatomical structures continuously.

Track:

* the confirmed eye-opening contour as one shape
* the confirmed limbus / iris-boundary model and estimated full iris circle/ellipse

Do not redetect these structures independently on every frame.

Use:

* previous position
* previous size
* local search

to follow the structure through time.

The objective is anatomical continuity.

---

# Tracking Confidence

Each tracked structure should have:

tracking_confidence

Examples:

* high confidence
* medium confidence
* low confidence

Confidence should be visible to the user.

---

# Tracking Status

Each structure should have:

tracking_status

Possible values:

* initialized
* tracked
* uncertain
* blink_or_occluded
* reacquired
* lost

These values should be exported.

---

# Blink Handling

If the iris becomes invisible:

Possible causes:

* blink
* eyelid occlusion
* motion blur
* tracking failure

The software should:

1. Mark blink_or_occluded
2. Stop reporting iris position
3. Attempt automatic reacquisition
4. Use MediaPipe if needed
5. Ask the user only if reacquisition fails

No artificial iris position should be generated during blinks.

---

# Eye Movement Measurement

For Version 1:

Eye position should be derived from:

the centre of the estimated iris circle/ellipse fitted from the visible limbus / iris-sclera
boundary.

Not from repeated iris detection.

The limbus-derived iris circle is the primary measurement target. The eye-opening contour is the
reference frame for corrected clinical coordinates. The resulting contour-relative iris-centre
signal is the INTERNAL analysis input to the V1 nystagmus detector — it is not the V1 clinical
display. The clinical display is the overlay video carrying the nystagmus arrow and label (see
`PROJECT_VISION_AND_REQUIREMENTS.md` "Nystagmus Detection — V1 PRIMARY OUTPUT").

---

# Torsional Analysis Strategy

The iris alone cannot detect torsional eye movement.

Future torsional analysis will require:

* iris texture
* iris crypts
* limbus
* conjunctival vessels
* scleral features

Therefore:

Eye ROI videos should be preserved at high quality.

Torsional analysis is a future module.

It is not part of Version 1.

---

# Success Criterion

The V1 overlay video must satisfy BOTH of the following from a vestibular neurologist's reading:

Tracking layer:
1. The estimated full iris circle/ellipse remains attached to the visible limbus / iris-sclera
   boundary.
2. The eye-opening contour remains attached to the visible palpebral fissure.
3. The tracker follows anatomy rather than repeatedly rediscovering it.
4. The internal contour-relative iris-centre signal represents real eye movement rather than
   landmark jitter.

Clinical layer (the V1 user-facing output):
5. When the clip contains repeated rhythmic jerk nystagmus, the overlay displays the correct
   beating-direction arrow (← → ↑ ↓ ↗ ↖ ↘ ↙) within a few beats of onset.
6. When the clip contains only random saccades, smooth pursuit, drift, or steady gaze, no arrow
   is shown.
7. Torsion is shown as "not assessed" until a torsional/rotation signal is implemented.

Only after both layers are achieved should EyeVNG move toward:

* torsional nystagmus detection (requires iris-texture rotation tracking inside the iris circle)
* head impulse testing
* VOR estimation
* binocular / INO / skew / dysconjugate analysis
* diagnostic interpretation

---

## Appendix — Evidence behind "detect once → track" (implementation note)

Measured on a real 1080p smartphone clip (`samples/2.mp4`), steady-eye segments, to justify the shift
away from per-frame detection:

| Approach | Median frame-to-frame motion | Verdict |
|---|---|---|
| MediaPipe single iris landmark (468/473) | ~4.8 px | the "dance" — intrinsic, white-noise-like |
| MediaPipe iris ring mean (5 landmarks) | ~4.8 px | no help — the iris estimate jitters as a unit |
| Iris referenced to eye corner | ~5.5 px | no help — the corner landmark jitters more (6.8 px) |
| Optical flow alone | ~3.2 px | smoother, but drifts ~187 px off the iris |
| Template tracking + MediaPipe re-anchor | ~3.3 px, drift-bounded | smooth *and* locked |

The jitter is intrinsic, independent MediaPipe landmark noise — confirming that continuous detection
cannot be made stable by averaging or re-referencing; it must be replaced by confirmed-then-tracked
anatomy.

**Current code vs this philosophy:**
- ✅ Iris boundary, concentric: `refine_iris()` keeps the centre at the iris centre and sets the
  radius to the iris boundary (limbus); `iris_contrast()` reports the iris/sclera contrast (low ⇒
  flagged, e.g. a washed-out or low-contrast iris). (Earlier drafts found the dark *pupil* concentric
  inside the iris; V1 marks the iris boundary directly.)
- ✅ Anatomical-confirmation proposal: Stage 0 must mark the selected eye's iris/limbus boundary and
  eye-opening contour. Any older facial-landmark proposal output is legacy support and must not define
  the V1 clinical reference.
- ✅ **Stage 0 approval gate:** `--approve` (review/correct → APPROVE) writes `approved_landmarks.json`;
  tracking refuses to start without it and builds the tracker from the approved init frame anatomy.
- ⬜ **Eye-opening contour tracking:** implementation must replace the old four-independent-point
  reference with a single tracked contour and derive medial/lateral/upper/lower values from that
  contour.
- ⬜ **Legacy facial landmark tracking:** may remain for future modules or compatibility, but it is not
  the V1 clinical reference frame.
