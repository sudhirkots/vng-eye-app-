# Calibration results — nystagmus detector vs clinician (2026-07-04)

First multi-clip calibration of the Clinical Nystagmus Detector (`tools/nystagmus_direction.py` on the EllSeg
centroid trace). 8 clips, each seeded by the clinician with the one-click seed marker (`manual_seeds.json`),
capped to ~15 s. VNG split-screen goggle clips excluded. Reproduce: `python tools/nystagmus_calibrate.py`.

## Scored table (app = primary-zone call; conf = whole-clip confidence)

| clip | expected (clinician) | app finding | conf | verdict |
|---|---|---|---|---|
| nystagmus at rest, R vestibular neuritis | primary left-beating | LEFT-beating | 0.63 | ✅ correct |
| gaze-evoked-1 | gaze-evoked | LEFT-beating | 0.60 | ✅ direction right (zone not split) |
| 1 | **no nystagmus (normal)** | no clear nystagmus | 0.26 | ✅ correct (true neg) |
| 2 | **no nystagmus (normal)** | no clear nystagmus | 0.20 | ✅ correct (true neg) |
| 3 | **no nystagmus (normal)** | no clear nystagmus | 0.07–0.22 | ✅ correct (true neg) |
| rest + head impulse, R vestibular neuritis | primary left-beating | no clear nystagmus | 0.12 | ❌ miss |
| gaze-evoked, pontine glioma | gaze-evoked (L↔left gaze, R↔right gaze), no vertical | no clear nystagmus | 0.22–0.27 | ❌ miss (down-beating FP now gone) |
| fistula | nystagmus on right-ear pressure (Hennebert) | UP-beating (H left 0.41) | 0.51 | ⚠️ vertical over-call |

## Key learnings

1. **True negatives all correct** — the 3 normal clips (1,2,3) all report "no clear nystagmus" (conf ≤ 0.26).
   The app **stays quiet on normal eyes** — the most important safety property.
2. **Confidence separates cleanly** — true nystagmus **0.60–0.63**, normal eyes **≤ 0.26**. A threshold around
   **0.4** classifies all of those correctly (current `CONF_MIN = 0.3`). Suggested bands: clear ≥ 0.55,
   probable 0.35–0.55, unclear < 0.35.
3. **Better (clinician) seeds removed the pontine down-beating false positive** (now under threshold, not a
   confident wrong call).

## Remaining problems (priority order)

1. ~~**Vertical over-calling**~~ **FIXED (2026-07-04)** — added a **horizontal-dominance bias** (`pick_axis`:
   vertical is only called if its confidence beats horizontal by ≥1.5× AND ≥0.50, because horizontal nystagmus
   is far more common and lid/blink noise mimics vertical). fistula "UP-beating 0.51" → **LEFT-beating 0.41**;
   clip-1 whole-clip UP → horizontal; no regression on the 3 true negatives or 2 positives.
2. **Gaze-evoked not safely detectable at 25 fps (ATTEMPTED + REVERTED, 2026-07-05).** Tried softened graded
   gaze zones + a per-zone gaze-evoked prior at low confidence (test left-beating in left gaze, right-beating
   in right gaze). Result on the full set: it **false-positived on normal clip 3** ("GAZE-EVOKED") and **lost
   fistula's correct call**, while STILL not fixing pontine (left gaze read right-beating). **Root cause:** at
   25 fps with short gaze segments the per-zone beat signal is too weak to distinguish real gaze-evoked from a
   normal eye's small wander + noise — so any low-confidence prior flags normals. **The normal controls caught
   it.** Reverted to this baseline. *Gaze-evoked reliable detection needs higher fps (60/120); it cannot be
   squeezed out of 25 fps IR without endangering the negatives.* The gaze DETECTION itself works (pontine:
   early=left gaze, later=right gaze); only the per-zone beat direction is unreliable at this sampling.
3. **Some positives missed** — head-impulse clip said nothing (0.12); the head-impulse thrusts likely disrupt
   the trace.

Baseline after vertical fix: **6/8 clean (3 positives + 3/3 true negatives), 2 misses** (pontine gaze-evoked,
head-impulse). Clean confidence separation (nystagmus 0.41–0.63, normal ≤ 0.26). Next: make gaze-zone
splitting work (for gaze-evoked).

## Update 2026-07-05 — four named categories + signal-quality gate

The output is now the four named categories with a **signal-quality gate** (see
`docs/CLINICAL_NYSTAGMUS_DETECTOR.md`). Re-run of the 8 clips:
- 3 normals → **`no_nystagmus`** (unchanged true negatives); 3 detected positives → **`nystagmus_likely`**
  (vestibular *clear* 0.63, gaze-evoked-1 *clear* 0.60, fistula *probable* 0.41) — unchanged.
- **pontine gaze-evoked** (inter-eye conjugacy 0.12 → `poor_signal`) → **`uncertain_tracking`**, replacing the
  old silent `no_nystagmus`. This **converts a false-negative into an honest "can't tell"** — a safety
  improvement, since the pontine clip genuinely has nystagmus.
- head-impulse remains a **`no_nystagmus`** miss (conjugacy 0.28 passes; asymmetry genuinely weak at 0.12).

Net: no regression on the 6 clean calls; one former miss (pontine) is now correctly flagged uncertain.
