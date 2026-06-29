# EyeVNG V1 — Weighted Multi-Feature Iris Tracker (architecture, no code yet)

> Status: DESIGN, 2026-06-26. Justified by `docs/FEATURE_PROBE_FINDINGS.md`. This replaces the
> single-template `IrisTracker` as the V1 tracking engine. It does NOT change Stage 0, eye-local
> coordinates, or the drift guards — it changes only HOW the iris centre is produced each frame.

## Design principle — a composite, not a dictator
The iris is tracked as a **dynamic pool of weighted anatomical features**, not one template.
Each frame the iris centre is the **robust weighted consensus** of many trusted features. No single
feature can move the centre; drift is detected when the composite *disagrees with itself*, not by a
single correlation score. The tracker behaves like a clinical video-oculographer: it tracks a
pattern, keeps the reliable parts, discards the unreliable, replenishes after disruptions, and never
invents a position it cannot vouch for.

## 1. Data model — the Feature Pool
The pool is the tracker's state: a set of `IrisFeature` records, each with

| Field | Meaning |
|---|---|
| `id` | unique, monotonic — stable identity across frames |
| `pos` | current (x, y) in image pixels |
| `birth_pos` | (x, y) when created; with `birth_offset = approved_iris_centre − birth_pos` |
| `confidence` | EMA in [0,1] from FB error + consensus agreement |
| `age` | frames survived since birth |
| `texture` | corner strength (Shi-Tomasi min-eigenvalue) — intrinsic trackability |
| `fb_error` | latest forward-backward LK error (px) |
| `residual` | distance from where the composite consensus predicts it should be |
| `state` | `PROBATION` → `TRUSTED` → `SUSPECT` → `LOST` |
| `birth_frame`, `last_seen_frame` | bookkeeping |

`TRUSTED` features are the voting composite. `PROBATION` features (newly added) are tracked and
scored but **cannot vote** until they earn trust — so a fresh, unproven feature can never swing the
centre. `LOST` features are removed.

## 2. Per-frame algorithm
```
            ┌─────────────────────────── prev frame pool ───────────────────────────┐
 new frame  │                                                                        │
   │        ▼                                                                        │
   ├─▶ (1) PREDICT      pyramidal Lucas-Kanade fwd for every ACTIVE feature          │
   ├─▶ (2) VALIDATE     fwd-back LK → fb_error; bounds + inside-aperture check        │
   ├─▶ (3) CONSENSUS    weighted RANSAC similarity fit of TRUSTED inliers:            │
   │                    centre = transform(approved_iris_centre); also rotation θ     │
   ├─▶ (4) SCORE        per-feature residual vs consensus → update confidence (EMA)   │
   ├─▶ (5) DRIFT TEST   composite agreement (inlier fraction, median residual)        │
   ├─▶ (6) MAINTAIN     promote / demote / discard; REPLENISH inside iris boundary    │
   └─▶ (7) EMIT         centre (or None), confidence, θ(torsion), validity flag       │
```

**(1) Predict** — `calcOpticalFlowPyrLK` forward (prev→cur) for all active features. Cheap,
real-time, the same operator the probe validated at sub-pixel error.

**(2) Validate** — track back (cur→prev); `fb_error = |start − round-trip|`. Reject features with
`fb_error > FB_REJECT`, out-of-frame, or outside the approved aperture (mapped to the current frame
via the consensus transform). These become `SUSPECT`/`LOST`.

**(3) Consensus (the vote)** — from the `TRUSTED` survivors, fit a **similarity transform**
(translation + rotation + optional uniform scale) `birth_pos → pos` by **weighted RANSAC**. Weight:
```
w_i = conf_i · sat(age_i) · norm(texture_i) · inlier_i
```
(`sat` saturates so an ancient feature cannot dominate forever; `inlier_i` is 0 for this-frame
RANSAC outliers). **Iris centre = transform(approved_iris_centre).** Equivalent view: each feature
"votes" a centre = `pos_i + birth_offset_i`; the output is the weighted robust mean of inlier votes.
A lone outlier gets `inlier_i = 0` → **zero influence**.

**(4) Score** — `residual_i = |pos_i − consensus_predict(birth_pos_i)|`. Update
`confidence_i ← (1−α)·confidence_i + α·g(fb_error_i, residual_i)`. Sustained agreement raises
confidence; disagreement lowers it.

**(5) Drift detection — by composite disagreement, never a single score:**
- `inlier_fraction` of trusted features < `INLIER_MIN`, OR
- `median trusted residual` > `RESID_MAX` (px and as a fraction of iris radius), OR
- transform degenerate (too few inliers, implausible scale/rotation).
Any → **`drift_suspected`** for the frame → centre invalid (a gap, not plotted). This *replaces* the
old single-template-score notion of confidence entirely.

**(6) Maintain the pool:**
- **Promote** `PROBATION → TRUSTED` after `TRUST_AGE` frames of high confidence + inlier status.
- **Demote/discard** features whose confidence < `CONF_FLOOR` or that are outliers for `K_BAD`
  consecutive frames → `LOST`, removed.
- **Replenish** (rule 3): if `active_count < POOL_TARGET`, detect new Shi-Tomasi corners **only inside
  the approved iris boundary** (transformed to the current frame), assign new `id`s, add as
  `PROBATION`. New features are spatially de-duplicated against existing ones.

**(7) Emit** — iris centre (or `None`), a composite-agreement confidence, the rotation θ (logged now,
used for **torsion** in V4), and the validity flag. This feeds the existing `eye_local()` + drift
guards unchanged.

## 3. Occlusion & blink (rules 6, 7)
- **Partial eyelid occlusion:** covered features get high `fb_error` / leave the aperture → dropped.
  The composite continues on the **remaining trusted features** as long as `trusted_inliers ≥ QUORUM`.
  The centre is still emitted. (Probe: features die in the covered region first; the rest carry on.)
- **Blink / full occlusion:** when `trusted_inliers < QUORUM`, declare `blink_or_occluded`, emit
  centre = `None` (never invented), and **freeze** birth references — do not replenish during closure.
- **Recovery:** when the aperture EAR recovers AND newly detected features inside the iris agree with
  the last trusted estimate, **re-seed** the pool (PROBATION), re-anchor, and resume. (Probe proved
  recovery: 80 → 2 → 80 features across a blink.)

## 4. Bounded long-term drift (re-anchor, not re-find)
Optical flow creeps over long runs. A **slow** re-anchor reconciles the consensus centre with the
MediaPipe iris / aperture **only when composite agreement is high** — a correction, never a per-frame
re-detection. This keeps the "track a pattern, don't hunt the eye each frame" behaviour while bounding
creep (the probe's pure-OF drift was the one weakness this closes).

## 5. Trust state machine (per feature)
```
   detect inside iris
        │
        ▼
   ┌──────────┐  TRUST_AGE frames high-conf+inlier   ┌──────────┐
   │ PROBATION │ ───────────────────────────────────▶ │ TRUSTED  │
   └──────────┘                                       └────┬─────┘
        │ low conf / OOB                                   │ residual/fb rising
        ▼                                                  ▼
   ┌──────────┐ ◀──────────── recovers ──────────────  ┌──────────┐
   │   LOST   │ ◀── K_BAD consecutive bad frames ────── │ SUSPECT  │
   └──────────┘                                         └──────────┘
   (only PROBATION/TRUSTED are tracked; only TRUSTED vote)
```

## 6. Parameters (named, tunable; seed values to calibrate on the probe clips)
`POOL_TARGET≈40`, `QUORUM≈5 trusted inliers`, `TRUST_AGE≈8 frames`, `CONF_FLOOR≈0.3`,
`FB_REJECT≈1.0 px`, `RESID_MAX≈0.15·iris_radius`, `INLIER_MIN≈0.6`, `K_BAD≈3`, EMA `α≈0.2`,
replenish trigger at `active < 0.7·POOL_TARGET`, re-anchor cadence ≈ every 15 high-agreement frames.

## 7. Integration (keeps the rest of the pipeline intact)
- Lives in `src/core/iris_tracking.py` as the new engine behind the **same `Track` interface**, so
  `iris_tracker.run()` and `eye_local()` are unchanged.
- **Stage 0** supplies the approved iris boundary (seed region + the rigid `birth_offset` anchor) and
  the eye-margin landmarks (eye-local coords).
- The **drift guards** (`iris_drift_check`) stay as an outer anatomical sanity gate; the composite's
  internal disagreement (step 5) becomes the *primary* drift signal and the guards become
  confirmatory.
- **Torsion (V4):** the rotation θ from the consensus transform is logged now, for free.

## 8. How this satisfies the stated requirements
| Requirement | Mechanism |
|---|---|
| unique ID / position / confidence / age / texture / FB error / status | `IrisFeature` fields (§1) |
| high-confidence features become trusted landmarks | trust state machine, PROBATION→TRUSTED (§5) |
| unstable features discarded | demote/discard on conf floor / K_BAD (§6.6) |
| new features only inside approved iris boundary | replenishment masked to the transformed iris (§6.6) |
| centre from weighted consensus of trusted features | weighted RANSAC similarity vote (§2.3) |
| drift = disagreement, not a single template score | composite agreement metrics (§2.5) |
| continue through partial occlusion/blink on remaining features | quorum-based continuation (§3) |
| replenish after recovery | re-seed on reopen (§3) |
| one bad feature cannot move the centre | RANSAC inlier weighting + quorum (§2.3) |

## 9. Open calibration questions (before/with implementation)
- Tune `RESID_MAX`/`INLIER_MIN` so real nystagmus fast-phases (valid coherent motion of the whole
  composite) are NEVER flagged as drift — fast phases move all features together (high agreement),
  drift fragments them (low agreement). The composite design makes this separable; thresholds set it.
- Confirm `POOL_TARGET`/`QUORUM` hold on the high-motion fistula clip after replenishment is added.
- Validate on non-brown irides + arcus senilis once footage exists (see findings doc).
