# VNG-EYE app — session handoff

> **SESSION 2026-06-27 (ARCHITECTURE CHANGE — iris DISC as the primary object + segment analysis):**
> Dr. K, watching the overlay: tracking internal feature dots is NOT enough. **V1 must track the WHOLE
> IRIS as a disc** — fit a circle/ellipse to the iris–sclera boundary (limbus) and follow THAT; the
> composite features are a stability aid only, not the visible output. "The iris is a circle; if a lid
> covers part of it, complete the circle from the visible arc. Forget the pupil." **He APPROVED this
> redesign** (so it is within the development rule, not a silent deviation).
> - **NEW `src/core/limbus.py`** — fits the iris boundary by radial dark-iris→bright-**sclera** edge
>   search + RANSAC circle (and an ellipse). Rejects lid/lash/skin edges three ways: skip the
>   lid-covered TOP wedge (keep sides+bottom — the bottom anchors the centre), require sustained
>   brightening, and require the bright side to be **whitish sclera, not reddish lid-skin** (colour
>   gate). Completes the circle from the visible arc. Verified on stills: clean 2.mp4 sits on the iris
>   (~0.86 edge inliers); fistula often returns None (low contrast) → that frame is invalid, excluded.
> - **HYBRID engine (`composite_tracker.py`):** features give MOTION; each frame `fit_limbus` is seeded
>   by the composite centre and pins the disc to the boundary. New drift reasons **`limbus_unfit`** /
>   **`limbus_disagrees`** (boundary can't be fit, or boundary vs feature centre disagree) → drift,
>   frame dropped (never rescued, per Dr. K). Emitted `iris_centre`/`iris_radius` = the limbus disc;
>   eye-local trace now comes from the DISC centre. `frame_confidence` folds in limbus inlier fraction.
>   FrameMeasurement gains `limbus_inlier_fraction`, `limbus_ellipse`.
> - **OVERLAY redesigned (`iris_tracker.py`):** the green iris **disc/ellipse + centre cross + the 4
>   eye-opening landmarks** (cyan: inner/outer canthus, upper/lower margin) are the main object;
>   internal composite features shrink to faint 1-px helpers. (`draw_overlay`/`draw_composite`.)
> - **NEW `segments.py` (spec §11 part 2):** classifies every frame valid/blink/occluded/
>   drift_suspected/tracking_lost, finds continuous VALID segments ≥2 s (high frame-confidence, iris
>   attached, no blink/drift), plots each segment ALONE (never connects across gaps), lists them
>   ("Segment 1: 0.41-20.40 s …"), and supports `--approve a,b` / `--reject c` (persisted in
>   `segments.json`). Final analysis will consume only APPROVED segments. **Nystagmus analysis itself is
>   NOT built yet** — Dr. K: do not proceed to it until the iris boundary is visually accepted.
> - **Backward-compat:** `run()` reads old `"pupils"`-key approvals (radius re-derived from MediaPipe).
> - Validation renders with the disc engine → `outputs/cft_v2/` (2.mp4, fistula, 1.mp4). **Acceptance is
>   VISUAL: the green circle must stay fitted to the whole iris, not dots/lid/cheek.** Uncommitted.

> **SESSION 2026-06-26 (CFT approved + implemented, renamed, debug replay, validated):**
> - **SPEC APPROVED.** `EYEVNG_TRACKING_SPECIFICATION.md` is the frozen contract for the V1 engine.
> - **RENAME (mandated): "Committee Tracker" → "Composite Feature Tracker (CFT)"** throughout — code,
>   comments, spec, design doc. `committee_tracker.py`→**`src/core/composite_tracker.py`**;
>   `CommitteeIrisTracker`→**`CompositeFeatureTracker`**; dataclass `Committee`→`Composite`;
>   `committee_confidence`→`composite_confidence` (incl. CSV cols); drift reason
>   `committee_disagreement`→`composite_disagreement`; `_vote`/`voters`/`vote()`→
>   `_compose`/`contributors`/`contribution()`; trackers dict `committee`→`cft`; `draw_committee`→
>   `draw_composite`; engine flag `--engine committee`→**`--engine composite`** (default). 0 "committee"
>   left in the tree. Imports verified.
> - **Engine bug-fixes (the broken Committee run made 0.3% valid on fistula, stuck ~1015 frames in
>   REACQUIRING):** (1) **recovery deadlock fixed** — recovery now re-anchors on MediaPipe instead of
>   gating the re-seed on agreement with a stale frozen `last_centre` (the deadlock cause); (2)
>   **`FB_REJECT` 1.0→2.5px** so the composite survives fast nystagmus/head sweeps (1px collapsed it
>   below QUORUM in ~3 frames); (3) **age-weight floor 0.3** so a fresh re-seed isn't zero-weighted;
>   (4) **blink reclassified (§8/§D.8)** — BLINK now requires BOTH a reduced/closed EAR **and** the
>   composite actually losing the iris (quorum lost). A low EAR alone (gaze excursion / noisy lid
>   landmarks) no longer fakes a blink. NOTE: this tightens the spec §B BLINK "OR" to an "AND" —
>   treated as §J blink-threshold calibration, flagged for review.
> - **Backward-compat:** `run()` now reads the old approved-landmarks `"pupils"` key as well as `"iris"`
>   (old approvals like 1.mp4 predate the rename); the iris radius is re-derived from MediaPipe anyway.
> - **DEBUG REPLAY built (spec §10): `debug_replay.py`** — re-runs the CFT and writes a 3.5× zoomed
>   `debug_replay.mp4` (HUD: state, feature counts, composite conf, FRAME conf, EAR, frame#, time;
>   features GREEN=trusted / YELLOW=probation / RED=lost). `--live` opens an interactive scrubber
>   (SPACE play/pause, n/→ step, b/← back, q quit) — GUI untested by agent. Stage 0 unchanged (refuses
>   without approval). Render verified headless (PNG + readable mp4).
> - **VALIDATION (engine=composite, reused existing approvals, in `outputs/cft_v1/`):**
>   2.mp4 **89–90%** valid · 1.mp4 (steady-ish face, real head motion) **76–78%** · fistula (worst case)
>   **35–39%** (was 0.3% broken; legacy single-template ~15%). Overlays + eye-local traces produced.
>   **Remaining gap source = composite collapse during fast motion (DRIFT_SUSPECTED + feature_loss),
>   NOT EAR** — the blink fix barely moved the numbers because the leftover blinks coincided with real
>   feature loss. Legacy template still beats CFT on the easy clip (2.mp4 99.8% vs 89%). NOT yet at the
>   >95% / visually-attached bar. Next lever: feature survival / re-detection during fast motion.
> - **Uncommitted.** Visual acceptance (watch the overlays + `debug_replay.mp4`) is the real test.

> **⚠️ TERMINOLOGY UPDATE (2026-06-26): EyeVNG V1 is now an IRIS tracker.** Code was renamed pupil→iris —
> `pupil_tracker.py`→`iris_tracker.py`, `src/core/pupil_tracking.py`→`src/core/iris_tracking.py`,
> `PupilDetector`→`IrisDetector`, `StableTracker`→`IrisTracker`, `refine_pupil`→`refine_iris`,
> `pupil_contrast`→`iris_contrast`; CSV `*_pupil_x/y/radius`→`*_iris_center_x/y` + `*_iris_radius`;
> approved-landmarks JSON key `"pupils"`→`"iris"`. **The dated entries below predate the rename and use
> the old names — preserved as project history.**

> # 🔑 CODE WORD: **EYEVNG**
> One of the **two canonical EYEVNG docs** (say "EYEVNG" to pull both up):
> **`VISION.md`** — what we're building, requirements, rules, decisions.
> **`HANDOFF.md`** — current status + running session log (this file).

> **SESSION 2026-06-25 (env fix + display rules):**
> - **Local Python fixed.** This PC (`C:\Users\Dr.Sudhir`) had no working Python — the `.venv` was
>   built on another machine (`C:\Users\sudhi\...`). Installed **Python 3.12.10** (`winget`,
>   `--scope user`) and repaired `.venv\pyvenv.cfg` to point at it; existing cp312 site-packages
>   (mediapipe 0.10.18, opencv 4.11, numpy 1.26.4, scipy) work. Run with `.\.venv\Scripts\python.exe`.
> - **Neuritis re-tracked.** `nystagmus at rest to left in right vestibular neuritis.mp4` now has a real
>   **canthus-relative** trace (its old folder was the failed affine version, ~empty).
> - **VNG display rules implemented & LOCKED** — see `VISION.md` → "VNG Display Rules". Single eye by
>   default (`--eye auto|left|right|both`, both only for INO); trace synced; **no band over the video**;
>   **trace never over the eyes** → rendered in a **strip BELOW the video** (canvas extended downward,
>   because clips zoom over the eyes). Code: `pupil_tracker.py` (`superimpose_traces`, `run`), `app.py`.
> - **Renamed** `PROJECT_VISION_AND_REQUIREMENTS.md` → `VISION.md`; refs updated. **Code word = EYEVNG.**
> - **Uncommitted.** All the above is in the working tree, not committed.

> **CANONICAL DOCS:** `VISION.md` (what to build + order),
> `docs/TRACKING_PHILOSOPHY.md` (tracking design — source of truth), `docs/CLINICAL_REQUIREMENTS.md`
> (clinical reasoning / why), `SYSTEM_ARCHITECTURE.md` (Stage 0 + data flow), `IMPLEMENTATION_PLAN.md`
> (Stage 0→5 order). This handoff and `STABLE_TRACKING.md`/`VERIFICATION_MODE.md`/
> `DUAL_MODE_DETECTION_PLAN.md` are implementation notes that defer to them.
>
> **CORE REQUIREMENT: Stage 0 — Landmark Review & Approval (IMPLEMENTED 2026-06-25).** Tracking must
> NOT begin until the user **approves** the proposed landmarks; the approved set is saved to
> **`approved_landmarks.json`** and is the initial reference for tracking.
> - `python pupil_tracker.py <video> --approve` — propose → review/correct pupils (edit GUI) → APPROVE
>   → writes `approved_landmarks.json` (pupils + face landmarks + `approved:true` + timestamp).
>   `--auto-approve` = headless (accept proposal, for testing).
> - `python pupil_tracker.py <video>` (track) — **refuses** unless `approved_landmarks.json` exists;
>   builds the template tracker from the approved init frame + pupils.
> - Verified: track-without-approval refused; auto-approve → json; track-from-json runs (jitter ~3.4px).
> **INTERACTIVE Stage 0 approval (2026-06-25):** `approve_interactive()` in `pupil_tracker.py` — full
> frame with proposed pupils (green circles) + facial landmarks (amber dots), labelled; drag to move,
> wheel/`[ ]` resize pupils, `d` toggle off/on, cursor magnifier, Enter=APPROVE. Wired into `--approve`
> (replaces the pupil-only edit GUI); confirms pupils AND facial landmarks → `approved_landmarks.json`.
> **CLI unified:** `python app.py --video X [--propose|--approve|--auto-approve|--review]` (app.py now
> drives the new flow; plain `--video X` tracks, refusing without approval). GUI untested by agent.
> **Stage 0 controls LOCKED (Dr. Kothari):** mouse only DRAGS a point (move); pupil radius changed only
> with `+`/`-` keys (no edge-drag/wheel). `d`=off/on, Enter=approve, q=cancel. Verified working: he
> dragged + resized pupils to r≈16-17 and approved.
>
> **⚠️ DRIFT BUG FOUND (2026-06-25) — fix pending.** The template pupil tracker drifts catastrophically
> OFF the pupil during head movement: on 1.mp4 the tracked pupil sits a **median 493px off the iris**
> (243/264 frames off), sliding onto cheek/brow. Cause is NOT face→pupil coupling (audited clean) —
> it's template drift (small low-contrast brown-pupil template + large search window → false match on
> head motion). MediaPipe iris, by contrast, STAYS on the eye.
>
> **✅ DRIFT FIXED (2026-06-25).** `StableTracker._step_eye` now anchors the pupil search box on the
> MediaPipe iris centre **every frame** (robust to head motion), finds the dark-pupil template INSIDE
> the iris ROI, and **rejects any match outside the iris** (clamp). MediaPipe positions the box only —
> never becomes the pupil result. Verified headless: tracked-vs-iris gap **493px → ~10px median** on
> 1.mp4; frame 64 (used to drift onto the jaw) now keeps the marker on the eye. Tests pass.
> Note: MediaPipe now runs every frame (ROI anchor) — consistent with "landmarks position the search
> box, image finds the pupil." Blink handling is approximate (eye-not-detected → blink_or_occluded;
> pupil-template-weak → uncertain). Head-correction (`corrected_eye_h/v`) still pending.
>
> **REVISED 2026-06-25 (after Dr. K saw side-jumping pupils + drifting face markers on auto-approved
> 1.mp4):** the ROI template still side-jumped within the small low-contrast eye. Decision: since the
> pupil is CONCENTRIC with the iris, `StableTracker._step_eye` now returns the **concentric MediaPipe
> pupil centre** (obs.x/obs.y) directly — stable, on the pupil, no drift, no side-jump; radius = approved
> value. Small residual jitter remains (offer EXPLICIT opt-in smoothing later, never secret).
> `FaceLandmarkTracker` rewritten: tracks each landmark as a fixed **offset from its MediaPipe position**
> (MediaPipe keeps face points locked to anatomy through head motion; offset preserves the Stage-0
> correction) → no more drifting onto forehead/hair. `run()` passes MediaPipe-at-init for the offsets.
> Old template machinery in StableTracker is now unused (dead code, harmless).
> **PROCESS NOTE:** stop auto-approving when showing Dr. K — he must confirm landmarks at Stage 0 first.
>
> **SHIMMER/SMOOTHING (2026-06-25):** Dr. K wants the small residual marker shimmer gone, BUT a causal
> EMA (`SMOOTH_ALPHA`) was tried and **REVERTED** — it lagged and made markers *slide* behind during
> head movement (much worse). Lesson: plain temporal smoothing is wrong here (lag). Overlay/traces are
> back to RAW (tight). `_smooth`/`SMOOTH_ALPHA` left in `pupil_tracker.py` but UNUSED. Next idea for
> de-shimmer WITHOUT lag: a small **deadband** (ignore sub-N-px jitter, snap immediately on real
> movement) — not yet built. CSV was always raw throughout.
>
> **JITTER FILTER BUILT (2026-06-25):** `src/core/filters.py` = 3-frame median + **1€ (One Euro)
> adaptive filter** (Median3, OneEuroFilter, PointFilter; presets light/adaptive; resets on lost/blink
> so it never bridges a gap). Wired into `run()`: per-eye + per-face-landmark filters. Now keeps RAW
> and FILTERED: CSV has `raw_*`/`filtered_*` pupil x/y + radius + status + confidence; overlay draws
> filtered as the main marker + faint raw cross (`--show-raw-trace`); traces plot filtered (main) with
> optional raw; new `trace_raw_vs_filtered.png` verification plot. CLI: `--filter none|light|adaptive`
> (default adaptive), `--show-raw-trace`. Deadband helper now UNUSED (dead code, left in).
> Results: 2.mp4 (steady) jitter left raw 4.86 → filtered 3.22px; 1.mp4 (moving) 10.93 → 8.68 (rest is
> REAL head movement, correctly preserved). 1€ is adaptive: smooths when slow, backs off on fast moves.
> NEXT (phase b): measurement-level centre (intensity-weighted dark-pupil centroid + confidence, fall
> back to MediaPipe). Filter strength tunable in `filters.py _PRESETS` if Dr. K wants more/less.
>
> **DESIGN SPLIT (2026-06-25, after Dr. K saw the filtered marker SLIDE on the overlay):** the filtered
> marker lags during head movement → looked like it slid off the pupil. Decision: **OVERLAY shows the
> RAW marker** (single, locked on the pupil, no lag — it's the verification view) and the **FILTERED
> signal is used only for the VNG trace + filtered CSV columns** (where shimmer matters and small lag
> is harmless). `draw_overlay` is now called with raw `lt/rt/face_tracks`. Raw always kept in CSV.
> **Dr. K accepted the filter for now** (clips are primary-gaze / NO nystagmus, so the trace correctly
> looks smooth/normal). **Validation of "preserves fast eye movements" is DEFERRED until a clip WITH
> nystagmus/saccades is available**, then tune `_PRESETS`. Phase (b) dark-pupil centroid less urgent now.
>
> **HEAD-CORRECTED EYE POSITION BUILT (2026-06-25):** `head_corrected()` in `pupil_tracker.py` fits a
> similarity transform (`cv2.estimateAffinePartial2D`, RANSAC) from the current facial landmarks back
> to their approved init positions, applies it to the RAW pupil, and returns the displacement from the
> init pupil = eye-in-head. Face landmarks position the reference frame only — never the pupil. CSV adds
> `corrected_left/right_eye_h/v` (raw corrected); new traces `trace_corrected_h.png` / `_v.png`
> (filtered, the real VNG signal). Validated on 1.mp4: image-space horizontal swing ~1554px (mostly
> head) → head-corrected ~220px (≈85% head motion removed). Residual + occasional transform glitches
> remain (imperfect landmark stabilisation / poor-frame transforms) — refine later (more/better
> landmarks, reject low-inlier transforms, filter corrected harder).
>
> **HEAD-CORRECTION TIGHTENED (2026-06-25):** switched to **full affine** (`cv2.estimateAffine2D`, 6 DOF)
> + **curated `HEAD_REF_LANDMARKS`** (nose_bridge_mid, nose_tip, 4 canthi — dropped noisy cheeks/tragus)
> + glitch rejection (RANSAC thresh 8, ≥4 inliers, det sanity 0.3–3.0). On 1.mp4: ~93% head removal on
> the frames it fits (1139→80px), spikes nearly gone, BUT only ~65% coverage (141/216) — the rest are
> honest gaps. **Root limit = facial-landmark quality under big 3D head rotation** (MediaPipe landmark
> drifts vs anatomy; FaceLandmarkTracker's fixed-offset model can't hold a tight frame). 1.mp4 is the
> WORST case (huge head motion + hand occlusion + small face). **Expect much better on a real nystagmus
> clip (stiller head).** Dr. K is providing a nystagmus recording next — drop in `samples/`, then test
> the whole chain (filter + head-correction) on real eye movement and tune.
>
> **✅ VALIDATED ON REAL NYSTAGMUS (2026-06-25):** `samples/gaze-evoked nystagmus in pontine glioma.mp4`
> (1080p, 25fps, clinical EYES-region close-up — brows+nose-bridge visible). MediaPipe detected face +
> both irises **100%** of frames (enough facial context, unlike goggle close-ups). Auto-approved + tracked
> 301/303. The horizontal `trace_raw_vs_filtered.png` shows the **gaze-evoked nystagmus beats clearly**
> (sawtooth), and the **adaptive 1€ filter PRESERVES the beats** while removing shimmer (raw jitter 5.97
> → filtered 3.34) — confirming it does NOT flatten fast eye movements. Several more nystagmus clips now
> in `samples/` (fistula, vestibular neuritis, gaze-evoked-1). NOTE: only auto-approved — for clinical
> use Dr. K should confirm pupils at Stage 0. Next ideas: proper Stage-0 confirm; quantify slow-phase
> velocity; test the other clips; head-correction here is limited (only canthi+nose-bridge in frame).
>
> **KEY FINDING (2026-06-25): temporal filtering CANNOT de-shimmer nystagmus.** Dr. K saw the filter
> erasing the nystagmus fast phases. Cause: the 3-frame median treated the brief (~1-2 frame @25fps)
> beat as a spike, and the 1€ filter was too gentle. Shimmer and nystagmus beats occupy the SAME high-
> frequency band → any temporal filter that removes one dents the other. FIX APPLIED: dropped the median
> (PointFilter `use_median=False` default), retuned `_PRESETS` (light 2.5/0.5, adaptive 1.0/0.7,
> dcutoff 2.0) so beats pass — but now shimmer barely drops (5.97→5.56) i.e. near-passthrough. **CONCLUSION:
> reduce noise at the SOURCE (phase b: better pupil detection / dark-pupil centroid), NOT with temporal
> filtering.** Phase (b) is now the priority. Keep raw always; filter stays available but light.
>
> **HEAD-CORRECTION → TIERED QUALITY (2026-06-25):** replaced hard-reject with `HeadStabilizer` class:
> rejects ONLY degenerate transforms (too few landmarks, NaN, impossible scale, egregious per-frame
> scale/rot/translation jump); keeps the rest with a quality flag (good/fair/poor by residual). CSV adds
> `transform_quality`, `transform_residual_px`, `landmark_count_used`; corrected traces are RAW (not
> over-filtered, beats preserved) with RED TICKS at poor/degenerate frames; `transform_quality_counts`
> in metadata. Results: pontine-glioma 262 good/41 degenerate (86% kept; degenerate clusters at clip end
> = transform errors there, beats preserved mid-clip); gaze-evoked-1 (fuller face) 194 good/0 degenerate,
> clean continuous trace showing gaze shifts + horizontal nystagmus beats; vertical = small/noise
> (correct for horizontal GEN). **Nystagmus beats ARE preserved.** Several nystagmus clips in samples/.
>
> **CANTHUS-RELATIVE EYE-IN-SOCKET (2026-06-25, Dr. K's idea — replaces affine for the corrected signal):**
> `canthus_relative(pupil, inner, outer)` measures the pupil vs THAT eye's own medial+lateral canthus
> (origin = canthi midpoint; horizontal axis = inner→outer, forced rightward so L/R read conjugate;
> h/v in px). Per-eye, robust (2 pts), cancels head/"hair" movement locally, NO gaps (canthi available
> every frame). Each pupil paired with its NEAREST canthi (fixes L/R mismatch). This is now the
> `corrected_*_eye_h/v` signal + the corrected traces. Results: gaze-evoked-1 → clean conjugate trace,
> gaze shifts + beats; **fistula → the right-ear-press nystagmus is now VISIBLE** (image-space span
> 1379px [head movement] → canthus span 489px, and the 35-40.5s zoom shows clear conjugate beats that
> were hidden before). HeadStabilizer/affine kept in code but unused for the corrected output.
>
> **LIVE TRACE PANEL UNDER VIDEO (2026-06-25):** `render_trace_panel` draws a scrolling 8s eye-in-socket
> (canthus-relative) horizontal strip beneath the overlay, with a cursor — synced to the video so marker
> motion and the graph match. Overlay video is now (ow × oh+200).

> **Facial-landmark TRACKING done (2026-06-25):** `FaceLandmarkTracker` (in `src/core/pupil_tracking.py`)
> tracks each approved facial landmark as a template point (local search + MediaPipe backup), with
> status/confidence; drawn on the overlay (coloured dots) + exported to `face_landmarks.csv` (per
> landmark x/y/status) + listed in metadata. On 2.mp4: nose bridge/tip & tragus_R track 1787/1787;
> low-texture points (cheek_R, inner_canthus_R) ~half "uncertain" (honest — smooth skin matches weakly).
> **Still pending:** interactive editing of facial landmarks (move/add/delete) in the approval GUI
> (approval currently edits pupil circles only).

Last updated: 2026-06-25 (stable pupil tracking; master requirements doc adopted)

## Where we are

Base commit `1ecb95b` ("WIP: iris-based eye tracking and scaled overlays") on `master`.
**Uncommitted working-tree changes now present** (Phase 1 of dual-mode detection — not yet committed).

## Dual-mode detection (current focus)

Root problem found this session: MediaPipe Face Mesh needs a whole face, so it fails on eyes-only
close-ups (the framing VNG actually uses). On the 4K face clip it tracked frames 1–684 then the mesh
collapsed for 685–828. Real IR goggle footage (`samples/VNG sample two eyes.mp4`) is a fixed
quad-split (two IR eyes top, colour room PiP bottom); a dark-pupil blob detector locks onto the pupil
cleanly there. Full design + decisions in **`DUAL_MODE_DETECTION_PLAN.md`** (read it).

**Phase 1 DONE (uncommitted):** Mode A sanity gate + real confidence + `detection_mode` column.
- `src/core/tracking.py` — `mesh_width_frac` gate (`min_mesh_frac=0.50`) rejects the collapse;
  `_eye_confidence` plausibility replaces the hardcoded 0.8; emits `detection_mode`/`left_conf`/`right_conf`.
- `src/modules/analyze_video.py`, `src/core/signal_processing.py` — new columns wired into the CSV + quality report.
- `tests/test_tracking_gate.py` — 6 dependency-free tests (`python tests/test_tracking_gate.py`).
- Verified: 684/684 good frames accepted, all 144 collapse frames suppressed, zero leakage.

**SCOPE LOCKED (2026-06-24, Dr. Kothari): face-based smartphone primary-gaze videos ONLY for now.**
The IR Frenzel goggle (eyes-only / Mode B) mode is a **separate later phase** — deferred, not in
scope now. "Step 1 is not good enough yet": focus only on **face-based tracking reliability + visual
verification** until a good-quality primary-gaze video tracks stably in **>95% of frames**. Other
downstream features (measurement CSV, graphs, diagnosis, torsion) also deferred.

**Verification/debug mode BUILT (uncommitted):** spec in `VERIFICATION_MODE.md`; tool is
`debug_tracking.py` (`python debug_tracking.py <video> [--output-dir outputs] [--max-debug-frames N]`).
Produces `tracking_quality.csv` (exact columns Dr. K specified + conf), a colour-coded
`debug_overlay.mp4` (green=good / yellow=uncertain / red=lost), saved `debug_frames/`, and
`tracking_summary.json` with %good and pass/fail vs 95%. Carry-forward + jump-detection are
FLAG-ONLY (never delete; nystagmus fast-phases are real jumps). Raw failures stay visible.

**Collapse gate relaxed (`min_mesh_frac` 0.50 → 0.20):** the 0.50 floor (tuned on a 4K clip where the
face filled the frame) wrongly rejected real faces that span only ~0.35–0.47 of the width. An
absolute width gate can't separate a small/distant face from a collapse, so it's now a loose floor;
real face-absence is handled by MediaPipe returning no-face, and dubious frames are surfaced by
confidence + the overlay. (Test `test_gate_decision_at_default_threshold` encodes this.)

**Results on Dr. Kothari's three face clips (`samples/1.mp4 2.mp4 3.mp4`):**
- `1.mp4` (1440p, face fills ~half frame): **100% good** ✓
- `2.mp4` (1080p, ideal framing): **99.8% good** ✓ — the reference example
- `3.mp4`: first half = face (tracks well), **second half zooms to a single-eye close-up** (no face →
  honestly flagged `lost:no_face`, 11 s run). Its 47% loss is the deferred IR/close-up case inside a
  face clip, NOT a face-tracking defect.

**Conclusion: the face-based tracker clears the >95% target on genuine face recordings.** Failures
are only honestly-flagged segments where the camera left the face. Recording guidance: keep the face
in frame for the whole clip (don't zoom to the eye) when using the face-based tool.

**Mode B / dual-mode (PAUSED)** — build the pupil detector etc. only after Step-1 tracking clears
>95%. See `DUAL_MODE_DETECTION_PLAN.md`.

## Stable pupil tracking (CURRENT focus — anti-"dancing", 2026-06-24)

Dr. Kothari: the pupil marker *dances* frame-to-frame even on a steady eye → build a **stable,
time-continuous tracker** (init → confirm → follow the same pupil), not a per-frame detector. Spec in
`STABLE_TRACKING.md`. **Do NOT** work on diagnosis/nystagmus/VNG-classification/BPPV/head-impulse/torsion.

**Root cause (measured on 2.mp4):** the jitter is intrinsic MediaPipe landmark noise (~5px median,
~16px p95), independent per landmark. NOT fixable by iris-ring averaging (1% help) or referencing to
the eye corner (corner jitters 6.75px — more than the iris). Pure optical flow is smoother (3.2px)
but drifts 187px off the pupil. **Winner: optical flow + MediaPipe re-anchor** → 3.4px (30% less),
drift bounded ~7px, 6% re-anchors, and temporally smooth.

**Architecture (per Dr. Kothari, no Kalman, no secret smoothing):** MediaPipe detect → user confirms
the pupil **circle (centre + radius, per eye, one-eye allowed)** → **continuous TEMPLATE tracking**
(normalised cross-correlation in a small window around the previous position, sub-pixel refined);
**MediaPipe is a backup only** (re-detect on low confidence / blink end / loss). Blinks/occlusion:
status `blink_or_occluded`, **coords left blank — never invented**; auto-reacquire when the pupil
returns, else (interactive) ask the user.

**Built (uncommitted):**
- `src/core/pupil_tracking.py` — `PupilDetector` (ring-mean), `select_init_frame` (prefers both eyes,
  falls back to one), `StableTracker` (template match + MediaPipe-backup reacquire + impossible-jump
  gate + `apply_correction`). Statuses: initialized/tracked/uncertain/blink_or_occluded/reacquired/lost.
  Confidence high/medium/low/none from match score (calibrated `high=0.47, med=0.35`).
- `pupil_tracker.py` — CLI. `edit_circles_gui` (drag centre, wheel/[ ] radius, x=disable eye for
  one-eye, Enter accept). Outputs per video: `tracking_overlay.mp4` (tracked **circle** + centre +
  status + frame + time; green=tracked, yellow=uncertain, blue=blink, red=lost; cyan=confirmed init),
  `tracking.csv` (frame_number, timestamp_ms, time_sec, left/right pupil x/y/radius,
  tracking_status_left/right, tracking_confidence_left/right), `trace_horizontal.png` +
  `trace_vertical.png` (raw H/V position vs time, OpenCV-drawn, gaps left blank), `metadata.json`.
- On 2.mp4 (`--auto-confirm`): 1566 tracked / 220 uncertain / 14 pre-init lost; left-eye jitter ~3.3px
  (was ~4.8 single-landmark). Gate tests still pass.

**UNTESTED BY AGENT (need Dr. Kothari's hands — OpenCV windows):** `edit_circles_gui` and `--review`.
Headless `--auto-confirm` path is validated.

**Dark-pupil refinement DONE (2026-06-25):** `refine_pupil()` keeps the centre **concentric with the
iris** (pupil is anatomically centred) and refines **only the radius** to the dark pupil (r≈29–31 on
2.mp4, was 73 = iris). Earlier version wrongly moved the centre off-iris (dark-blob centroid) — fixed.

**Proposal/confirmation step DONE:** `python pupil_tracker.py <video> --propose` →
`init_proposal.png` marks, for the user to confirm/correct: pupils (green) drawn **concentric** with
the iris (cyan) so concentricity is visible; **all facial landmarks** (nose bridge/tip, cheeks,
tragus, four canthi — amber, patient-side by image x; off-frame ones reported unavailable). Pupil
detection uses **darkness contrast** (`pupil_contrast()`), flagged when low (brown-iris or cataract).
`face_landmarks()` added to `PupilDetector`; `EyeObs` now carries `iris_radius` + `pupil_contrast`.
NOTE on 2.mp4: pupil/iris contrast is very low (~0.03) because the iris is dark brown — the radius is
a rough proposal; the user confirms/adjusts. `edit_circles_gui` remains for interactive correction.

**OPEN ISSUES / honest caveats:**
1. **Facial-landmark confirmation & tracking NOT implemented** — the revised philosophy wants confirmed
   + tracked facial landmarks (nose bridge/tip, cheeks, ears/tragus, canthi); code tracks pupils only.
   This is the next alignment item.
2. **Blink detection not yet validated on a real blink clip.** 2.mp4 has ~no blinks (only 2 frames
   flagged). Need a clip with clear blinks; may still need a dedicated blink check (eye-aspect-ratio)
   beyond the template score. **Which sample has blinks?**
3. ~16% of frames read "uncertain" (medium score) with the tighter pupil template — acceptable, tunable.

## Original WIP-commit context (still relevant)

## What changed in the WIP commit

- **src/core/tracking.py** — eye centers now come from MediaPipe iris landmarks **468 (left)** and **473 (right)** instead of averaging eyelid landmarks. Better signal for nystagmus tracking.
- **src/modules/analyze_video.py** — overlay rendering (mesh dots, eye circles, font size, thickness) now scales with video resolution via `scale = max(1.0, min(width, height) / 480.0)`, so annotations stay legible on larger frames.
- **src/core/head_pose.py** — added a module-level `estimate_head_pose(frame, landmarks)` helper backed by a default `HeadPoseEstimator()` instance.
- **requirements.txt** — pinned: `opencv-python>=4.11,<4.12`, `mediapipe==0.10.18`, `numpy>=1.26,<2`.

## What is NOT done yet

- None of this has been run on a sample video. Iris landmarks + scaled overlays are written but unverified.
- No tests, no sample data checked in.
- Untracked siblings next to this folder (not part of this app): `Migraine study app/`, `eye_movement_app/`, `outputs/`, plus a `SETUP.md` in this dir.

## Suggested next step

Run the analyzer on a short sample clip and inspect:
1. CSV output — do `left_eye_x/y` and `right_eye_x/y` track the iris smoothly frame-to-frame?
2. Overlay video — are the iris circles centered on the pupil, and is the text readable at the clip's resolution?

If both look right, the WIP commit can be reworded from "WIP" to a proper feature commit.

## Repo context

- This app lives inside the `Neurology Talks` repo (branch `master`, main branch is `main`).
- VNG-EYE is **not** on GitHub — only the migraine study app is pushed to `github.com/sudhirkots/migraine-study-app`.
- So uncommitted/committed work here only exists on this PC's OneDrive copy.
