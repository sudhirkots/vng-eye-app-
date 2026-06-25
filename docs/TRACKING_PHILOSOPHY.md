# EyeVNG Tracking Philosophy (Revised)

**Status: SOURCE OF TRUTH for all tracking-related design decisions.** Where any other document
(`PROJECT_VISION_AND_REQUIREMENTS.md`, `SYSTEM_ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`,
`STABLE_TRACKING.md`, `HANDOFF.md`) disagrees about how anatomy/pupils are tracked, this document
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
* propose iris landmarks
* propose facial landmarks

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

# Facial Landmark Confirmation

The software should propose facial landmarks.

The user should confirm or correct:

* bridge of nose midpoint
* nose tip
* left cheek point
* right cheek point
* left ear or tragus if visible
* right ear or tragus if visible
* left inner canthus
* left outer canthus
* right inner canthus
* right outer canthus

If a landmark is not visible:

mark as unavailable.

After confirmation these landmarks become tracking targets.

---

# Pupil Confirmation

The software should display enlarged eye views.

The software may propose a pupil location.

The user should:

* move the pupil centre
* resize the pupil circle
* confirm the pupil boundary

The confirmed circle should cover the dark pupil only.

The pupil circle should not represent the whole iris.

Store:

* pupil centre
* pupil radius

for each eye.

If only one eye is visible:

continue with one-eye tracking.

---

# Landmark Review and Approval

**Core principle: human-confirmed anatomy is the source of truth.** AI landmark detection (MediaPipe)
is only a *proposal*. **Tracking must not begin until the user explicitly approves the proposed facial
landmarks and pupil circles.** This is a permanent project requirement, not a temporary implementation
detail.

**Why confirmed anatomy is preferred over repeated detection:** per-frame detection jitters and can
silently mislabel or drift between structures, creating artificial movement. A human confirming the
anatomy *once* gives a stable, correct reference that tracking then follows. MediaPipe is therefore
used primarily for **initialization** (proposing landmarks) and **re-acquisition** (after loss, blink,
or occlusion) — never as the per-frame source of truth.

Workflow:

1. **Landmark proposal** — MediaPipe proposes the facial landmarks and pupil circles on the best
   initial frame.
2. **Landmark editing** — the user may move, resize (pupil radius), add, or delete landmarks, and
   mark any not visible as unavailable.
3. **Landmark approval** — the user explicitly approves the set. Nothing is tracked before this.
4. **Approved landmarks become tracking targets** — persisted to `approved_landmarks.json` and used
   as the initial reference by every tracking module.
5. **MediaPipe becomes fallback only** — consulted afterwards solely for re-acquisition.

**Stage 0 is an INTERACTIVE confirmation screen, not silent auto-detection.** Running `--approve`
must: open the chosen frame; visibly **mark and label** every proposed point (left pupil, right pupil,
nose bridge / central nasal reference, left cheek, right cheek, and any other stable facial reference
in use); ask the user to confirm each; let the user **click the correct location** for any wrong
point; save the corrected set to `approved_landmarks.json`; and **NOT proceed to tracking** (approval
only saves). The clinician must never have to trust automatic detection blindly.

---

# Pupils and Face Are Independent — Head-Motion Compensation

**Facial landmarks move with the head. Pupils move within the eyes.** They are two separate coordinate
systems and must be tracked separately.

- The face tracker may define a **moving head/face reference frame** per frame, and may move the eye
  **search ROI** — but it must **never move, drag, shift, overwrite, or infer the pupil result.**
- The pupil centre must always be found from **image evidence inside the eye region**. If it cannot be
  found confidently, mark it `lost`/`blink_or_occluded`/`reacquired` — **never** invent it from face
  movement.
- Output **both**: (a) **raw** pupil coordinates in image space, and (b) **head-corrected** eye
  position relative to the tracked facial reference frame.
- Correct logic: track face landmarks → estimate head reference-frame motion; track pupils → estimate
  pupil centres from image evidence; compute eye position **relative to** the moving face frame. The
  face tracker decides the search box, never the pupil location.

**Permanent audit rule:** no face-transform / landmark shift / affine / face-template motion may be
applied directly to pupil x/y. Keep `raw_*_pupil_x/y` and `corrected_eye_h/v` as separate outputs.

**Drift fix (2026-06-25):** the earlier pupil tracker searched a template around the *previous pupil
position*; during head movement the pupil left the window and the template locked onto a wrong dark
region (lash/brow/cheek) and drifted off the eye (493px median off-iris on the head-movement clip),
while still reporting "tracked". **Fixed:** the search box is now centred every frame on the **robust
eye location** (MediaPipe iris, which follows the head), the dark pupil is found *inside* that box,
and any match outside the iris is rejected — so the marker physically cannot leave the eye. MediaPipe
positions the search box only; it is not forced to be the pupil result. Verified: off-iris gap
493px → ~10px median; the frame that used to drift onto the jaw now keeps the marker on the eye.

**Eye-in-head — preferred method: CANTHUS-RELATIVE (2026-06-25).** The head reference is each eye's
own **medial (inner) and lateral (outer) canthus**: origin = the canthi midpoint, horizontal axis =
inner→outer (forced rightward so the two eyes read conjugately); the pupil's displacement along that
axis is the eye-in-head horizontal position (perpendicular = vertical). Because the canthi move WITH
the head, this **cancels head/"hair" movement locally**, needs only two points per eye, leaves **no
gaps** (canthi are present every frame), and reads conjugately. This is the `corrected_*_eye_h/v`
signal. A whole-face affine transform was tried first but is limited by facial-landmark quality under
3D head rotation (gaps + spikes) — kept only as a fallback. Pair each pupil with its NEAREST canthi.

**Shimmer vs nystagmus (key principle, 2026-06-25).** Measurement-noise "shimmer" and nystagmus fast
phases share the same high-frequency band, so **no temporal filter can remove one without denting the
other**. Therefore: preserving the beats always wins (the filter is beat-preserving/near-passthrough);
reduce noise at the **source** (better pupil detection), not by smoothing. The **overlay shows RAW**
(verification — must sit on the pupil, no lag); the **VNG graph uses the filtered** signal (a small
lag is harmless there). Raw is always kept in the CSV.

**Reading the trace is a scaling problem too.** Real eye movement can be tracked yet invisible in a
graph because the y-scale is dominated by head movement or by a long clip. The cures are the
canthus-relative signal (removes head movement) plus **time-zoom**, and a live **scrolling trace strip
under the video** (synced cursor) — not more filtering.

---

# Continuous Tracking

After confirmation:

Track the same anatomical structures continuously.

Track:

* confirmed facial landmarks
* confirmed pupil circles

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

If the pupil becomes invisible:

Possible causes:

* blink
* eyelid occlusion
* motion blur
* tracking failure

The software should:

1. Mark blink_or_occluded
2. Stop reporting pupil position
3. Attempt automatic reacquisition
4. Use MediaPipe if needed
5. Ask the user only if reacquisition fails

No artificial pupil position should be generated during blinks.

---

# Eye Movement Measurement

For Version 1:

Eye position should be derived from:

confirmed and tracked pupil centres.

Not from repeated iris detection.

The pupil is the primary measurement target.

---

# Torsional Analysis Strategy

The pupil alone cannot detect torsional eye movement.

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

The overlay video should convince a vestibular neurologist that:

1. The pupil marker remains attached to the pupil.
2. Facial landmarks remain attached to the same anatomical structures.
3. The tracker follows anatomy rather than repeatedly rediscovering it.
4. The resulting traces represent real eye movement rather than landmark jitter.

Only after this stage is achieved should EyeVNG move toward:

* VNG trace generation
* nystagmus analysis
* torsional analysis
* head impulse testing
* VOR estimation
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
| Optical flow alone | ~3.2 px | smoother, but drifts ~187 px off the pupil |
| Template tracking + MediaPipe re-anchor | ~3.3 px, drift-bounded | smooth *and* locked |

The jitter is intrinsic, independent MediaPipe landmark noise — confirming that continuous detection
cannot be made stable by averaging or re-referencing; it must be replaced by confirmed-then-tracked
anatomy.

**Current code vs this philosophy:**
- ✅ Pupil = dark pupil, concentric: `refine_pupil()` keeps the centre at the iris centre (pupil is
  concentric) and shrinks only the radius to the dark pupil; `pupil_contrast()` reports how much
  darker the pupil is than the iris (low ⇒ flagged, e.g. a cataractous/whitish pupil, or a dark-brown
  iris where pupil/iris contrast is intrinsically low).
- ✅ Anatomical-confirmation proposal: `pupil_tracker.py --propose` marks pupils (green) + iris (cyan,
  concentric) + all facial landmarks (nose bridge/tip, cheeks, tragus, four canthi; amber) on the init
  frame → `init_proposal.png`, with off-frame landmarks reported unavailable.
- ✅ **Stage 0 approval gate:** `--approve` (review/correct → APPROVE) writes `approved_landmarks.json`;
  tracking refuses to start without it and builds the tracker from the approved init frame + pupils.
- ✅ **Facial landmarks are now TRACKED:** `FaceLandmarkTracker` follows each approved facial landmark
  as a persistent template point (local search + MediaPipe backup), with status/confidence, drawn on
  the overlay (coloured by status) and exported to `face_landmarks.csv`. High-texture points
  (nose bridge/tip, tragus) track rock-solid; low-texture points (cheek, inner canthus) read
  "uncertain" more often — honestly flagged.
- ⬜ **Interactive editing of facial landmarks** (move/add/delete in the approval GUI) is still pending
  — approval currently edits pupil circles only.
