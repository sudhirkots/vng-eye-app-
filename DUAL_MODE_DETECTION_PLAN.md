# Dual-Mode Eye Detection — Implementation Plan (for review)

Status: **PARTIALLY BUILT (Phase 1 done), THEN PAUSED 2026-06-24.** Builds on `SYSTEM_ARCHITECTURE.md`
and the V1 `IMPLEMENTATION_PLAN.md`.

> **Note:** this is the **deferred** IR-Frenzel-goggle path, NOT V1. On IR goggle footage the device
> images the **dark pupil** directly, so "pupil detection" here is the correct anatomical term and is
> intentionally retained — it is unrelated to the V1 iris tracker.

> ⚠️ **SCOPE LOCKED — face-based smartphone primary-gaze videos only (Dr. Kothari, 2026-06-24).**
> The current product is **face-based primary-gaze tracking + verification**. **Mode B / the IR
> Frenzel goggle (eyes-only) path is explicitly DEFERRED to a separate later phase** — this whole
> dual-mode plan is on hold for it. Do not build Mode B, the measurement CSV for analysis, graphs,
> diagnosis, or torsion yet.
>
> **Step 1 first ("Step 1 is not good enough yet"):** make face-based eye/iris tracking **reliable**
> and **visually verifiable** before anything else. Target: a good-quality primary-gaze video tracks
> stably in **>95% of frames**. The verification work is in `VERIFICATION_MODE.md`. This dual-mode/IR
> plan resumes only after the face-based tracker clears the >95% bar and Dr. Kothari asks for it.
>
> **INO clarification (clinical):** eyes can move *disconjugately* in INO — but you can only judge
> that when **both** eyes are visible. If only one eye is visible, assume both move together and use
> the single-eye trace. (So: keep both per-eye traces when both are seen; trust the one when only
> one is seen.)

## 1. Why

Verified on the existing 4K sample (`outputs/measurements.csv`, 828 frames):
- Frames 1–684 (~23 s): MediaPipe Face Mesh tracks the irises well (99% detection, smooth).
- Frames 685–828 (~5 s): the recording becomes an **extreme eye close-up** (no mouth/chin/face
  outline). MediaPipe needs a *whole face* and fails — proven by static-mode re-detection also
  failing on 685/690/760/820 and returning a **collapsed mesh** (bbox 841 px vs healthy ~2760 px)
  on 700. The sibling `eye_movement_app/` is also Face-Mesh-based, so it has the same blind spot.

**Requirement (Dr. Kothari):** VNG sometimes shows the whole face + eyes, and sometimes **only the
eyes — occasionally only one eye**. The pipeline must handle all of these.

### Real eyes-only sample examined — `samples/VNG sample two eyes.mp4`
1426×720, 30 fps, 109 s. **IR grayscale** (channel diff ~1.6 → monochrome goggles). Layout is a
**quad-split export**, not two eyes side-by-side:
- **top-left:** one eye, IR, well-exposed — clear **dark pupil** + bright iris + corneal glints.
- **top-right:** the other eye, IR — **badly underexposed / near-black** in this clip.
- **bottom-centre:** a **colour** picture-in-picture of the room (scene cam) + burned-in captions.

**Mode B validated on this sample:** a dark-blob + circularity detector on the top-left eye region
locked onto the pupil precisely on f400/821/1642 (r≈50, centre on target) and correctly returned
**no pupil on a blink** (f2500, eye closed). So the dark-pupil primitive works; the open design
question is now *eye-region localization and layout handling*, not the pupil detector itself.

## 2. Design — two detection modes with automatic fallback

```
each frame:
  Mode A  MediaPipe Face Mesh  (whole face present)
      └─ accept ONLY if mesh is plausible (bbox large enough) → rejects the collapse
  if Mode A absent/rejected:
  Mode B  Face-independent pupil detector  (eyes-only crop; 1 or 2 eyes)
  record per frame: detection_mode, per-eye confidence, eye_count
```

### Mode A — Face Mesh (mostly exists)
- Keep current iris landmarks 468 (left) / 473 (right).
- **Add a sanity gate:** reject the detection when the landmark bounding box is implausibly small
  relative to the frame (collapse signature: ~22% of frame width vs ~72% when healthy). Start at
  a conservative relative threshold (e.g. mesh bbox width < 0.35 × frame width → reject), tuned on
  the sample. A rejected Mode A frame falls through to Mode B.
- Replace the hardcoded `tracking_confidence = 0.8` with a **real plausibility score** (borrow the
  `_eye_metrics` idea from `eye_movement_app`: iris inside eye box + open-eye aspect ratio).

### Mode B — pupil detector (new; no face needed) — VALIDATED on the IR sample
Pupil-detection core (proven on the sample):
1. Grayscale + light blur + CLAHE (local contrast).
2. Threshold the darkest pixels (dark pupil) → morphological open/close → contours.
3. Filter candidates by area + **circularity** (reject lashes, lid shadows, glints); take the best.
   Optional Hough/min-enclosing-circle refine for a sub-pixel centre and pupil radius.
4. Per-eye confidence from circularity + contrast margin. Blink/closed-eye → no pupil → low conf.

**Eye-region localization — DECIDED: layout-aware (Dr. Kothari confirmed the quad layout is the
fixed standard export).** Crop the two top eye panels (top-left / top-right); ignore the bottom
colour scene PiP and burned-in text entirely. Keep the eye-region step behind a small interface
(`locate_eye_regions(frame) -> [regions]`) so an auto-detect implementation can be swapped in later
if a different device appears — but only layout-aware is built now.

**Single-eye is a first-class case — DECIDED (one eye is often unusable, e.g. this sample's
near-black right panel).** Run the pupil detector independently per eye panel. Each panel
independently yields a pupil-or-nothing; the frame's `eye_count` is 0/1/2. Never require both eyes.

**Combining eyes — conjugacy (Dr. Kothari):** eye movements are normally **conjugate** (both eyes
move together), so a clean trace from *either* eye is a valid gaze measurement. Therefore:
- Always keep **both per-eye traces** as measured. **Never synthesize the missing/closed eye from
  the good one** — fabricating it would hide genuinely disconjugate movement (INO, etc.). A
  measurement-first tool should record only what it sees.
- Add a **`gaze_*` (primary) trace** = mean of both eyes when both are reliable, else the single
  good eye. This is the signal downstream analysis (nystagmus slow-phase, etc.) uses, and it stays
  continuous across stretches where only one eye is visible — justified by conjugacy.
- When both eyes are reliable, the per-eye traces are still kept separate so disconjugacy remains
  visible rather than averaged away.

> Confirmed: footage is **IR grayscale, dark-pupil**. A bright-pupil IR device would invert the
> threshold — would need a flag, but not present here.

## 3. Left/right labelling — must stay consistent across modes
- Mode A labels anatomically (landmark identity); Mode B labels by image position.
- Convention (matches `COORDINATE_SYSTEM.md`, x→right): **image-left blob = patient's right eye**,
  image-right blob = patient's left eye, recorded consistently regardless of mode.
- When only one eye is present, store the point plus an explicit `eye_side_confident = False` so
  downstream code never silently mislabels.

## 4. CSV schema (additive — never overwrite raw, per architecture §7)
Add columns alongside the existing ones:
- `detection_mode` ∈ {`face_mesh`, `pupil`, `none`}
- `eye_count` (0/1/2)
- `left_conf`, `right_conf` (real per-eye confidence)
- `left_pupil_r`, `right_pupil_r` (Mode B radius; blank in Mode A)
- `eye_side_confident` (bool)
- `gaze_x`, `gaze_y` (primary trace: both-eye mean when reliable, else the single good eye)
- `gaze_source` ∈ {`both`, `left`, `right`, `none`} — which eye(s) fed `gaze_*` this frame
Existing columns keep their meaning so old outputs stay readable.

## 5. Borrow from `eye_movement_app` (don't reinvent)
- Plausibility confidence (`_eye_metrics`).
- Separate **processed** columns: rolling-median + Savitzky–Golay smoothing, blink/closed-eye flag.
- Quality warnings incl. "tracking lost ≥0.5 s in a contiguous block" (it would have flagged 685+).

## 6. Overlay changes (verification)
- Colour/label each point by `detection_mode` so collapse vs real pupil is obvious on review.
- In Mode B, draw the detected pupil circle (centre + radius).

## 7. Phasing
1. **✅ DONE — Stop the bleeding:** Mode A sanity gate (`mesh_width_frac` < `min_mesh_frac`=0.50) +
   real per-eye plausibility confidence (`_eye_confidence`, replacing the hardcoded 0.8) +
   `detection_mode`/`left_conf`/`right_conf` CSV columns + quality reporting of rejected frames.
   Verified on the face clip: 684/684 good frames accepted, all 144 collapse-tail frames suppressed
   (coords now `None`, not garbage), zero leakage. Tests in `tests/test_tracking_gate.py` (6 pass,
   dependency-free: `python tests/test_tracking_gate.py`).
2. **Mode B** pupil detector (single + dual eye) as a standalone, unit-tested function.
3. **Wire the fallback** + overlay/CSV updates (per-panel pupil for the quad layout, `gaze_*`).
4. **Validate on the sample:** Mode A owns 1–684; Mode B should recover the IR goggle clip. Ideally
   add an eyes-only integration smoke test.

## 8. Out of scope here (tracked separately)
- `head_pose.py` math is a broken placeholder (pitch pinned ~88°) — fix in its own change.
- Pixel→degree calibration (architecture §6/§10) — future.

## 9. Decisions & remaining questions
- **RESOLVED — Layout:** fixed quad export → layout-aware crop of the two top eye panels.
- **RESOLVED — Second eye:** single-eye tracking is first-class; never require both eyes.
- **RESOLVED — Imaging:** IR grayscale, dark pupil (validated on the sample).
- **RESOLVED — L/R mapping (Dr. Kothari):** the panel on *our* right (top-right, TR) = the
  **patient's LEFT eye**; the panel on our left (top-left, TL) = the **patient's RIGHT eye**
  (standard mirror/face-to-face convention; matches `COORDINATE_SYSTEM.md` image-left = patient's
  right). The CSV still records `eye_panel` (TL/TR) alongside the mapped side for auditability.

All design questions are now resolved — implementation can begin (Phase 1).
