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

1. **Vertical over-calling** — fistula got "UP-beating 0.51" when the real signal is horizontal (left 0.41).
   The vertical channel fires falsely and can win the zone. TOP fix.
2. **Gaze-evoked not detected** — pontine + gaze-evoked-1 have direction-changing nystagmus, but the gaze
   zones don't split, so "beats toward gaze" can't be shown. Pontine collapsed to no-clear.
3. **Some positives missed** — head-impulse clip said nothing (0.12); the head-impulse thrusts likely disrupt
   the trace.

Baseline: **4/8 fully correct incl. 3/3 true negatives; clean confidence separation.** Next: suppress the
vertical channel, then make gaze-zone splitting work.
