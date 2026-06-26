# EyeVNG — Development Notes & Lessons

Running journal of what we tried, what we changed, and what we learned — so the reasoning isn't lost.
Durable design rules live in `TRACKING_PHILOSOPHY.md`; this is the "how we got there" record.

---

## Session 2026-06-25 — from per-frame detection to a confirmed, head-referenced tracker

The day's arc was: stop trusting MediaPipe every frame → confirm anatomy once → track it → handle
head movement and noise honestly → validate on real nystagmus. Each step below: **what we did →
what we learned**.

### 1. Facial-landmark tracking
- **Did:** added `FaceLandmarkTracker` to follow the confirmed facial landmarks (nose, cheeks, tragus,
  canthi) as persistent points.
- **Learned:** a previous-position template tracker DRIFTS for face points just like for pupils. The
  robust fix is to track each landmark as a **fixed offset from its MediaPipe position** (MediaPipe
  keeps face points locked to anatomy through head motion; the offset preserves the user's correction).

### 2. Revised philosophy: detect once → confirm → track
- **Did:** Dr. Kothari authored `VISION.md` (was `PROJECT_VISION_AND_REQUIREMENTS.md`); we wrote `TRACKING_PHILOSOPHY.md`.
- **Learned (central shift):** EyeVNG is a **tracking** system, not a per-frame detector. MediaPipe is
  a *proposal/anchor*, never the final per-frame measurement — repeated detection creates jitter that
  mimics eye movement.

### 3. Stage 0 — interactive landmark approval
- **Did:** built the approval gate. `--approve` opens the init frame, marks + labels the proposed
  pupils and facial landmarks, the user drags to correct, then approval saves `approved_landmarks.json`.
  Tracking refuses to start without it. CLI unified under `app.py --video X [--approve|...]`.
- **Learned:** human-confirmed anatomy is the source of truth. Controls were **locked**: mouse only
  drags a point; pupil radius changes with `+`/`-` (edge-drag and mouse-wheel were confusing). Also
  fixed a real bug — `approve()` must use a fresh detector for facial landmarks (the init-frame scan
  leaves MediaPipe's tracking state advanced, so re-processing the init frame with it can fail).

### 4. Pupil ≠ iris; concentric
- **Did:** `refine_pupil` finds the **dark pupil** inside the iris (radius from the dark→bright edge),
  keeping the centre **concentric** with the iris.
- **Learned:** the pupil is anatomically concentric with the iris, so the stable centre IS the iris
  centre — only the radius needs the dark-pupil refinement. On brown irises the pupil/iris contrast is
  very low (~0.03), so the radius is only a proposal the clinician adjusts.

### 5. Head-movement DRIFT (the big bug)
- **Did:** the template tracker (search around the *previous pupil*) drifted catastrophically off the
  iris during head movement — median 493 px off on the head-tilt clip, while still reporting "tracked".
- **Learned:** the search anchor was the flaw. **Anchor the eye search box to the robust eye location
  (MediaPipe iris), which follows the head; find the pupil INSIDE the iris; never let it leave.** And
  since the pupil is concentric, the simplest stable, drift-free, no-side-jump centre is just the iris
  centre. Audit confirmed: pupils were always tracked independently of the face — no face→pupil coupling.

### 6. Shimmer vs nystagmus — the filtering lesson (the most important one)
- **Did:** tried EMA smoothing (lagged → markers *slid* off during head movement → reverted), then a
  deadband, then an adaptive **One Euro filter** (median + 1€) with raw/filtered kept separately.
- **Learned, definitively (Dr. K caught it on real nystagmus):** **shimmer (measurement noise) and
  nystagmus fast phases occupy the same high-frequency band — no temporal filter can remove one without
  denting the other.** The 3-frame median erased brief beats. Decisions:
  - **Preserving the nystagmus beats always wins.** The filter is now near-passthrough (beat-preserving).
  - **Reduce noise at the SOURCE** (better pupil detection), not with aggressive temporal filtering.
  - **Overlay shows RAW** (the truth, on the pupil, no lag — for verification); the **VNG graph uses
    filtered** (where a small lag is harmless). Raw is always kept in the CSV.

### 7. Eye-in-head: head-motion compensation
- **Did (first):** an affine transform fit from the facial landmarks back to their init positions,
  applied to the pupil. Tightened it (full affine, curated nose+canthi landmarks, glitch rejection),
  then made it **tiered** — reject only *degenerate* transforms (too few landmarks, NaN, impossible
  scale, or an impossible per-frame jump); keep the rest with a quality flag (good/fair/poor) + residual
  + landmark count; mark poor/degenerate frames with red ticks on the trace.
- **Learned:** the affine is limited by facial-landmark quality under big 3D head rotation — it leaves
  gaps and spikes. Spikes that coincide with the red ticks are **transform** errors, not pupil errors.
- **Did (better — Dr. K's idea):** **canthus-relative "eye-in-socket"** — measure each pupil against
  *its own* medial + lateral canthus (origin = canthi midpoint; horizontal axis = inner→outer, forced
  rightward so L/R read conjugate). This is now the `corrected_*_eye_h/v` signal.
- **Learned:** referencing the pupil to its own eye corners is **simpler, more robust (2 points/eye),
  has no gaps** (canthi are present every frame), and cancels head/"hair" movement locally. It cleanly
  separated the eye movement from head movement where the affine struggled. (Fixed two bugs: pair each
  pupil with its NEAREST canthi; force the axis rightward so conjugate gaze reads parallel, not divergent.)

### 8. Validation on real nystagmus
- **Did:** ran gaze-evoked (pontine glioma), gaze-evoked-1, fistula, and vestibular-neuritis clips.
- **Learned:**
  - Pupil tracking is robust on face/eye-crop clips (≈100% detection) except where the eye is genuinely
    occluded (one neuritis clip had 28% blink/occluded — honest gaps).
  - The **adaptive filter preserves the nystagmus beats** (validated on the sawtooth traces).
  - **Head-correction helps when the head moves and can add noise when it's still** — the tiered quality
    flags tell you which trace to trust (image-space when the head is still, canthus-relative when it moves).
  - A **head-impulse** clip correctly showed the thrusts as degenerate/red-tick regions, not eye movement.

### 9. Visualisation — the eye was tracked but the graph hid it
- **Learned (Dr. K, fistula):** the nystagmus was tracked but invisible in the graph because of
  **scale** — the full 42 s clip + big head movement squashed the small beats. The fixes that revealed
  it: **canthus-relative** (removes the head movement) + **time zoom** (read the beats at a sensible
  scale). The beats were there all along; it was a display problem.
- **Did:** added a **live scrolling trace strip under the overlay video** (8 s window, cursor at "now",
  eye-in-socket horizontal), synced so the marker motion and the graph match frame-by-frame.

---

## Session 2026-06-26 — pivot to single-eye EYE-LOCAL, then to IRIS tracking; the drift wall

The day's arc: build the single-eye eye-local clinical trace → discover the pupil marker jitters →
pivot to tracking the **iris as a physical object** ("detect once, track forever") → hit **template
drift**. Canonical design now in `VISION.md` → "VERSION 1 DESIGN — Single-Eye, IRIS Tracking".

**What we built & learned:**
1. **Eye-local coordinate system** (`eye_local()` in `pupil_tracker.py`): iris/pupil centre projected
   on the inner→outer canthus axis (x: 0 inner→1 outer) and the upper→lower margin axis (y: 0 up→1 lo).
   Replaces the old canthus-midpoint pixel measure. Head/frame-independent; uses NO face/head pose.
2. **Stage 0 = the 4 aperture corners + (now) the iris**, paired to the pupil by PROXIMITY (the pupil
   L/R label and the aperture L/R label use opposite conventions — pair by nearest, never by label).
3. **Axis-scale bug (big one):** a leftover `if ymax-ymin<1: inflate by ±1` guard in `plot_trace`
   silently blew out EVERY normalised (0..1) eye-local plot to −1..2, hiding the signal. Fixed → guard
   only a truly flat trace; robust y-scale now uses median±MAD + [2,98] pct, labels adaptive precision.
4. **Learned landmark calibration** (`landmark_calibration.json`, `update_calibration`/
   `apply_landmark_calibration`): each interactive `--approve` learns Dr. K's placement vs MediaPipe
   (D-normalised, eye-local) and pre-corrects future proposals. Seeded n=1 from fistula: 24.3→8.6 px.
   Big consistent offsets: upper margin ~0.21·D higher, outer canthus ~0.10·D more temporal.
5. **Jitter hunt:** MediaPipe per-frame iris centre jitters ~6 px (steady). Tried, in order — iris
   template-match anchored to MediaPipe (~7% better), dark-disc centroid anchored to MediaPipe (~5%).
   All were still pulled by MediaPipe every frame → marginal. Conclusion: must STOP per-frame MediaPipe.
6. **Pure local iris template tracking** (current `_step_eye`): local match from the previous position,
   MediaPipe only on match failure. The iris template-matches *well* (median score 0.66). **BUT it
   DRIFTS:** on a low-texture brown iris the fixed template matches similar nearby patches and wanders.
   **Proven visually at fistula frame 302 (eye-local x=1.04): the green "iris" circle sat up on the
   EYEBROW while the eye was centred — at HIGH confidence.** So: **template score ≠ attachment.** The
   full-aperture trace swings (−0.21 … 1.04, physically impossible) were drift, not eye movement.

**The wall / key lesson:** jitter-vs-drift tradeoff. MediaPipe-anchored = no drift but jitter; pure
local fixed-template = smooth but drifts. Neither stays *attached* on a low-texture iris.

## NEXT SESSION (2026-06-27) — make the iris marker provably attached
Build BOTH (Dr. K approved direction):
1. **Drift guards** — trust a frame ONLY if anatomically plausible: iris centre inside the aperture
   box; iris radius/size stable; frame-to-frame motion physiologically plausible; large excursions
   need visual verification. Else mark `drift_suspected` (NOT valid). Do not use template score alone
   as confidence. Priority = visual attachment over numeric jitter.
2. **Limbus circle-fit local tracker** (the real fix) — each frame, locally fit the iris/sclera
   boundary circle (the highest-contrast feature; position-specific → can't sit on the brow, which has
   no limbus; boundary-averaged → low jitter). Still local (search around previous centre), still no
   per-frame MediaPipe. The fitted iris circle also feeds future torsion.
   - **Feasibility probed 2026-06-26 (do NOT re-walk):** vanilla `cv2.HoughCircles` is UNRELIABLE on
     this brown-iris clip — strict params find nothing (5/6 frames); loose params find circles but
     9–48 px off the iris. So build a TAILORED limbus detector: take horizontal/directional gradients
     to find the LEFT & RIGHT limbus arcs (the iris/sclera edges lids don't occlude), then robustly fit
     a circle/ellipse (RANSAC) to those arc points; search locally around the previous centre.
3. **Verification harness** (built today, keep using it): render enlarged single-eye frames at
   start / mid / max-outer / max-inner / end with aperture + tracked iris boundary + centre + conf +
   frame number → clinician confirms attachment before any trace is trusted. (`_drift_check.png` /
   `_drift_*.png` in the fistula output dir are today's examples; frame 302 = the drift proof.)
4. Then: rename pupil→iris across CSV/overlay/plots per the VISION CSV spec; size the Stage-0 circle to
   the iris; update the 4 sibling docs; re-mark clips one by one with the 5-point scheme.

## Standing to-dos / open levers
- A `--window t0 t1` zoom and an optional de-trended "beats" view for reading nystagmus.
- Quantify: slow-phase velocity, beat frequency/direction (only after the trace is trusted).
- `head_pose.py` is a broken placeholder (future V2 head module).
- Per-machine local venv at `%USERPROFILE%\eyevng-venv` (OneDrive `.venv` is broken across machines).
