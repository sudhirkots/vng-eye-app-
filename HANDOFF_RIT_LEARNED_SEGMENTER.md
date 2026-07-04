# HANDOFF — RIT Learned Segmenter & Learning Workbench

**Date:** 2026-07-01 (updated 2026-07-02)  **Branch:** `feature/cft-implementation`
**Status:** run_001/run_002 complete on the home PC; **run_003 re-run on this PC to verify the metrics
reproduce** (same 39 marks, deterministic). Committed per Dr. K's "save everything".

---

## Active direction

```
RIT Orbit Lock  +  clinician-marked ground truth  +  lightweight learned iris/sclera segmenter
```

- **Retained:** RIT Orbit Lock (moving anatomical search region).
- **Deprecated:** the OpenCV iris/sclera/almond RULE detector (historical benchmark only; its
  Almond–Sclera–Iris rules are now ANNOTATION rules, not threshold code).
- **Not done / forbidden without approval:** nystagmus detection, wiring the learned model into clinical
  RIT, committing.

## The workbench

`tools/rit_learning_workbench.py` — one command runs the whole self-testing loop end-to-end:

```
py tools/rit_learning_workbench.py "outputs/nystagmus at rest to left in right vestibular neuritis_tracked"
```

Steps: **A** validate annotations → **B** train `ExtraTreesClassifier` (5-fold held-out-FRAME CV) →
**C** evaluate (iris/sclera/almond IoU, centre error, bleed) vs the deprecated OpenCV baseline →
**D** apply to the whole clip (masks, centre, ellipse, confidence, status; overlay mp4) →
**E** auto-select worst frames + build a paint-ready correction set → **F** compare vs the previous run.

Each run writes a fresh, never-overwritten `RIT_learning_workbench/runs/run_NNN/` with
`model/ reports/ overlays/ predictions/ worst_frames/ correction_set/`. The current model is promoted to
`RIT_learned_segmenter/` (`model.pkl`, `feature_config.json`, `LATEST.json`).

## Results (run_002; run_001 identical, deterministic)

| metric (held-out frames) | OpenCV (deprecated) | learned |
|---|---|---|
| iris IoU | 0.39 | **0.81** |
| sclera IoU | 0.18 | **0.68** |
| almond IoU | 0.41 | **0.80** |
| iris centre error | 66 px | **10 px** |
| iris-outside-almond | 34/39 | 23/39 |

- Ground truth: 42 units found, **39 valid** (L20/R19); excluded `f0060_L` (no sclera), `f0090_L` (no
  iris), `f0135_L` (unpainted).
- Full-clip apply: 640 frames, **1279 `learned_ok` / 1 `needs_human_review`**.
- Visual: best held-out (f0187_L, IoU 0.91) is clean; **worst = near-closed/blink frames** where sclera
  bleeds onto the eyelid (f0123_R sclera IoU 0.17). That is the primary correction target.

## What Dr. K does next (the loop)

1. Paint the **15 correction frames** in
   `RIT_learning_workbench/runs/run_002/correction_set/*_to_correct.png` (RED iris, BLUE sclera; the
   `*_prediction.png` shows what the model guessed). These are mostly late-clip right-eye centre-jumps /
   small-iris and a couple of blink frames — the true hard cases.
2. Copy each corrected PNG into `RIT_ground_truth/to_mark/<id>.png` and add its entry from
   `correction_set/correction_meta.json` into `RIT_ground_truth/meta.json`.
3. Re-run the workbench → it creates `run_003`, retrains on old + new marks, and reports
   `vs_prev_run.json` (did iris IoU / centre error improve?).
4. Repeat until acceptable. If it plateaus: more marked frames → better features → cascade (almond first,
   then iris/sclera) → small U-Net (U-Net only with approval).

## Environment note (important)

The workbench needs `scikit-learn` + `joblib` (+ `scipy`). Installing them on this PC pulled in
**numpy 2.5**, which **breaks `mediapipe` 0.10.18** (needs numpy<2). The workbench itself only uses
`cv2` + `numpy` + `sklearn` (works fine under numpy 2), but the **main Stage-0 app** (`iris_tracker.py`,
which uses MediaPipe) will not run in this venv until numpy is pinned back or a separate ML venv is used.
On the home PC use `py -3.14` / the dedicated `eyevng-venv`. Recommendation: keep a **separate venv for the
ML workbench** so the clinical Stage-0 path keeps numpy<2 + mediapipe.

## Data-integrity note

The clinician's painted masks live in `Desktop/RIT_paint/` and `RIT_masks (1).zip` (41 masks, 39 complete)
and in `RIT_ground_truth/masks/` (extracted binaries) — both intact. `RIT_ground_truth/to_mark/` had been
overwritten with clean crops by an earlier `rit_make_ground_truth_frames.py` re-run; it was **restored from
the zip** (41 painted masks, verified pixel-aligned to `raw/` + `meta.json`, 0 mismatches) before run_003.
Guard the painted `to_mark/` set from OneDrive overwrite; the extracted `masks/` are the durable ground truth.

## OPEN — clinician recognition interview (pending Dr. K's answers)

Before more feature work, capture *how Dr. K recognizes* the anatomy so the annotation rules + features come
from the clinician, not from guesses. Questions posed (awaiting answers), to fold into `docs/CLINICAL_NYSTAGMUS_DETECTOR.md`:
1. **Sclera vs red conjunctiva** — does injected pink/red conjunctiva count as sclera, or only pure white?
   (Drives the sclera class; the model currently bleeds sclera onto the lid on blink frames.)
2. **Iris vs pupil** — mark the whole dark disc as one (iris), pupil never separate? Track limbus, never pupil?
3. **Cornea & catchlight** — is the bright corneal reflection on the iris painted as iris or excluded? Is
   "iris edge = limbus" sufficient, or use the corneal/limbal edge?
4. **Extreme gaze** — completing the full circle from a thin nasal sliver: arc curvature vs Stage-0 radius vs
   fellow-eye symmetry?
5. **Near-closed / blink** — how much visible arc is "enough to place it" vs "too covered, unusable"?
6. **Eye movement** — visual reference for eye-vs-head (iris centre vs canthi? limbus vs sclera?); and what
   signals slow-phase vs fast-phase direction.

## Environment (this PC)

`scikit-learn 1.9.0` + `joblib` were installed into `.venv` so the workbench runs here. Per the numpy-2 vs
mediapipe caveat above, keep the clinical Stage-0 (MediaPipe) path on a numpy<2 interpreter — or a separate
ML venv for the workbench. The `.venv/pyvenv.cfg` on this PC was repointed to this machine's Python 3.12.
