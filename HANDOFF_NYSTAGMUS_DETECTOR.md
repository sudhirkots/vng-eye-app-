# HANDOFF — Clinical Nystagmus Detector (CURRENT ACTIVE ARCHITECTURE)

**Date:** 2026-07-05  **Branch:** `feature/cft-implementation`  **Status:** active clinical path.
This is the handoff for the architecture the project now runs on. It **supersedes** the precise
iris-tracking / iris-marking handoffs (`HANDOFF_SUDHIRS_IRIS_METHOD.md`, `HANDOFF_RIT_*`), which describe the
retired track. Full narrative in `EYE_VNG_DEVELOPMENT_HISTORY.md` §26; rules in
`docs/CLINICAL_NYSTAGMUS_DETECTOR.md`.

## The active architecture (five load-bearing facts)

1. **EllSeg centroid is the active eye-position signal.** EllSeg is used only as an iris *locator*: per frame
   we take its disc **centroid** (smooth, reliable, conjugate) and ignore its jagged mask/outline.
2. **The app goal is QUALITATIVE nystagmus screening, NOT quantitative VNG.** No velocity, no rate, no degrees,
   no waveform. Present/absent · direction · gaze zone · gaze effect · confidence — that is the whole product.
3. **OpenCV iris/sclera/almond rule detection is DEPRECATED** (fixed-radius circle, limbus-arc RANSAC,
   dark-within-oval, foreshortening). Code stays in history; not on the active path. Perfect iris-outline
   drawing is abandoned — a rough centroid is enough.
4. **The learned red/blue (iris/sclera) segmenter is FALLBACK / RESEARCH ONLY** — usable *if the EllSeg
   centroid fails*, but not the main path, and not to be promoted unless EllSeg proves unreliable.
5. **Signal-quality gating is REQUIRED before any `no_nystagmus` call.** `poor_signal`/`missing_signal` must
   produce `uncertain_tracking`, never `no_nystagmus` (false negatives are dangerous).

## Pipeline (two scripts)

1. **`tools/ellseg_centroid_trace.py`** → `ellseg_centroid.csv` (`frame,eye,cx,cy,status,scl_left,scl_right,
   disc_area`). Seed priority: `manual_seeds.json` (one-click marker `tools/rit_iris_seed_marker.html`) →
   clinician orbit marks / `meta.json` → automatic dark-blob finder. One clear iris suffices (conjugate).
2. **`tools/nystagmus_direction.py`** → the clinical read:
   - **Signal-quality layer** (per-frame → per-window → overall: good/usable/poor/missing) from 7 checks:
     orbit range, plausible disc area, no impossible jumps, stable tracking %, L/R conjugacy (≥0.20),
     plausible sclera balance, not blink/occlusion. Runs FIRST; gates `uncertain_tracking`.
   - **Slow-phase asymmetry** direction (keeps slow drift, removes >3 s head drift; time/speed/skew asymmetry
     + cross-window consistency). Horizontal-dominance bias curbs vertical over-call.
   - **Sclera-balance gaze zones** (head-motion invariant), sustained ~2 s, referenced to median = primary.
   - **Four categories** `nystagmus_likely` / `no_nystagmus` / `insufficient_beats` / `uncertain_tracking`
     (NYST_CONF 0.35, TEND_CONF 0.30), plus evidence components + a clinical pattern summary
     (direction-fixed / gaze-evoked direction-changing / vertical / no nystagmus / uncertain_tracking).

## Definition of nystagmus (the anchor)

Jerk pattern — slow drift one way + brief faster jerk the other — repeating **≥3 times in succession, same
direction.** <3 or inconsistent → not nystagmus. **Jerk only** (pendular out of scope, intrinsically excluded);
negatives read "no jerk nystagmus".

## Calibration (8 clinician-seeded clips, `docs/CALIBRATION_RESULTS.md`)

- 3/3 normals → `no_nystagmus`; 3 positives → `nystagmus_likely` (vestibular *clear* 0.63, gaze-evoked-1
  *clear* 0.60, fistula *probable* 0.41).
- **pontine gaze-evoked** (inter-eye conjugacy 0.12 → `poor_signal`) → `uncertain_tracking` (was a silent
  `no_nystagmus` miss — now an honest "can't tell"). head-impulse remains a `no_nystagmus` miss.

## Reproduce (home PC — rit-nets venv, numpy 2)

```
C:\Users\sudhi\rit-nets\rit-nets-venv\Scripts\python.exe tools\ellseg_centroid_trace.py     # trace (RIT_CLIP env)
...\python.exe tools\nystagmus_direction.py <ellseg_centroid.csv> <fps>                      # clinical read
...\python.exe tools\nystagmus_calibrate.py                                                  # 8-clip calibration
```
Env: EllSeg/nets stack is off-OneDrive & not in git (regenerate per `HANDOFF_RIT_NETS_EVAL.md`); it uses
numpy 2, so keep the clinical MediaPipe path on a numpy<2 interpreter.

## Known limit + next lever

30 fps undersamples the fast phase → **direction only, no beat rate**; gaze-evoked direction-changing is not
reliably separable at 30 fps (detector reports direction-fixed rather than over-call). **Recommended upgrade:
capture at 60/120/240 fps** to count ≥3 beats directly, report rate, and unlock gaze-evoked detection.

## Hard rules still in force

Never claim "no nystagmus" from insufficient/invalid data (→ `uncertain_tracking`); conservative confidence;
EllSeg is a LOCATION prior only; OpenCV iris rules & perfect-outline drawing stay retired; red/blue segmenter
stays fallback/research only; Orbit Lock kept for a possible future head-free input.
