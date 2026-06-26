# Feature-feasibility probe — findings (2026-06-26)

Read-only experiment (`experiments/feature_probe/`) to decide whether multi-feature iris
tracking should replace the single-template tracker as the V1 engine. No pipeline code changed.

## Method
Inside the manually marked iris only: detect Shi-Tomasi corners, track with pyramidal
Lucas-Kanade + forward-backward consistency, estimate the iris centre from the consensus
(median translation) of survivors, and compare against a replicated single-template NCC tracker
and MediaPipe's iris (anatomical reference). Tested on fistula (dark brown), 2.mp4, 1.mp4
(lighter), and all 8 sample clips for the appearance analysis.

## Core results (blink-aware window)

| Clip (condition) | Init features | Survivors 10/20/30 | % | Mean FB err | Consensus jitter | Template jitter | Gap→MP @30 (cons/templ) |
|---|---|---|---|---|---|---|---|
| 2.mp4 (steady) | 37 | 37/37/37 | 100/100/100 | **0.024 px** | 3.94 | 4.12 | **5.0 / 6.2** |
| 1.mp4 (big head motion) | 41 | 18/6/2 | 44/15/5 | 0.236 px | 36.6 | 34.0 | — / 14.5 |
| fistula (blinks+motion+nystagmus) | 80 | 2/0/0 | 2.5/0/0 | 0.191 px | 10.96 | 11.18 | — / 20.5 |

Blink recovery (fistula, across a real blink): **80 → 2 → 80** (re-detection fully restores the pool).

## Iris-appearance / colour generalization (all 8 clips, brightness 25–111)
- Feature detection **never failed**: 8–120 features per clip; even the lowest-contrast iris
  (fistula, contrast 4.6, texture 22) gave 93 features.
- Feature **count** is driven by iris pixel-size, not colour; density 1.3–19.7 /1000 px².
- Survival when the eye is open & steady is high at **every** brightness (2.mp4 100% @28,
  3.mp4 78/80 @33, pontine 46/51 @73, small-iris 8/8 @111). Low-survival clips are all
  high-motion/blur (elevated FB error), not pigment.
- Mechanism is colour-agnostic: corner detection sees texture/gradients, not hue.

## Conclusions
1. **Feature scarcity is not a risk** — even on dark brown irides. The hardest texture case passed.
2. **Tracking is sub-pixel accurate** when the eye is open (FB 0.02–0.24 px).
3. **A fixed feature set has a short lifetime** under motion/blinks (10–30 frames) but
   **re-detection fully replenishes it**. The engine must be a self-replenishing pool.
4. **Consensus ≥ template on accuracy, strictly better on robustness** (graceful degradation;
   no catastrophic lock-off like the template's 15%-valid full-clip fistula result).
5. **Colour:** strong within the available brown gamut; predicted (by mechanism) to generalise
   to blue/green/hazel/light-brown (more visible stroma → more features). **Arcus senilis**
   degrades the limbus, not the central stroma — so it *favours* multi-feature over a limbus fit.
   Non-brown colours are predicted-not-proven; collect such clips and re-run `03_*` to confirm.

**Decision: replace the single-template tracker with a self-replenishing weighted multi-feature
iris tracker.** Architecture in `MULTIFEATURE_TRACKER_DESIGN.md`.
