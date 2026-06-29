# Anatomically-Constrained Iris Tracker — Design

**Status:** design draft, awaiting clinician approval before code is written.
**Date:** 2026-06-28.
**Supersedes (locally):** the wandering behaviour of the V1 default limbus tracker on
the vestibular-neuritis clip. V1 itself remains the default `--engine` choice; the
anatomical tracker ships as `--engine anatomical` so the two can be A/B compared on
the same clip without regressing V1.

---

## 1. Why this exists

Two hard rules from the clinician:

> **Rule 1.** Every video must start with Stage 0, where the clinician marks the eye
> opening. The iris cannot go outside the eye opening.

> **Rule 2.** The tracker should not be a general circle tracker. It should be an
> anatomically-constrained iris–sclera boundary tracker. A missing iris is better than
> a wrong iris on the face.

The current V1 tracker is good at *finding* a limbus arc (`src/core/limbus.py` does
RANSAC on dark→bright transitions, has a sclera-whiteness gate, supports fixed-radius
occlusion-invariant centre fits). What it does *not* do is enforce **anatomical
containment**: the radial ray search can extend onto cheek/nose/brow when the prior
centre is bad, and the orchestrator (`v1_tracker.py`) currently falls back to other
estimators on failure instead of freezing. This is the wandering you observed.

This redesign closes that gap with the minimum useful change.

---

## 2. The two anatomical objects

* **Stage-0 eye-opening contour** (one polygon per eye, set once at approval time):
  the bony orbital margin / visible eye opening at neutral gaze, drawn by the
  clinician with enough room for extreme lateral gaze (the clinician explicitly chose
  to make it generous for this reason). This polygon is the **anatomical containment
  box** for the iris.
* **Iris–sclera boundary (limbus)** (per frame): the dark-iris/bright-sclera curve
  the tracker fits a circle to.

The contour does **not** track frame-to-frame in this version. It is fixed at Stage 0.
If the head moves the contour will be wrong; for now we accept that and the clinician
rescues. (A future patch can re-project the contour via face landmarks if needed.)

---

## 3. The containment rule

The clinician's exact rule, paraphrased:

> The visible iris–sclera arc must lie inside the contour. The inferred full circle
> does not have to — when the patient looks extremely to the side, the unseen part of
> the iris is allowed to "imaginary-poke" out of the contour.

Operationally:

* The radial-edge search inside `fit_limbus` looks for limbus edge pixels.
* Each candidate edge pixel is accepted only if it lies **inside the Stage-0
  contour polygon**.
* The circle fit (RANSAC, free or radius-fixed) runs on the accepted edge pixels only.
* The fitted **centre** and **radius** are **not** required to lie inside the contour
  — only the input edge pixels are.

Consequence: at extreme lateral gaze the visible crescent is inside the contour and
produces a valid full-circle fit even if the full circle's far edge would be outside
the contour. At neutral gaze the full circle is inside anyway. There is no special
case for "partial cover" — the same code handles it.

If the clinician's contour was too small for the extreme gaze in this clip, the
correct answer is "redraw a bigger contour at Stage 0" — not "tracker invents a
permissive rule." This keeps the responsibility line clean.

---

## 4. Hard preconditions

The anatomical engine **refuses to run** unless:

1. `approved_landmarks.json` exists for this clip with schema
   `v1_iris_limbus_and_eye_contour`.
2. The schema includes a non-empty `eye_opening_contours` polygon for the active eye.
3. The schema includes a Stage-0 `iris` block with `radius` for the active eye.

If any precondition fails, the engine prints a clear message and exits non-zero:

```
[anatomical-engine] Stage 0 missing or incomplete for clip <name>.
Run: python app.py --video "<clip>" --approve
The anatomical engine requires:
  - approved eye-opening contour for the active eye
  - approved iris radius for the active eye
```

This enforces Rule 1 in code, not just in spirit.

---

## 5. Per-frame logic (anatomical engine)

For each frame, for each active eye:

```
1. Compute prior centre c_prev = last good iris centre.
   If no last-good (frame 1 only): c_prev = Stage-0 approved iris centre.

2. Compute prior radius r_prev = last good iris radius.
   If no last-good: r_prev = Stage-0 approved radius.

3. Pull the Stage-0 contour polygon P for this eye.

4. Run fit_limbus(gray, cx=c_prev, cy=c_prev, r_est=r_prev, ...)
   WITH a containment hook: every candidate edge pixel must satisfy
   cv2.pointPolygonTest(P, (x, y), measureDist=False) >= 0.
   (This is the single new gate inside limbus._radial_edges.)

5. The fit returns LimbusFit(cx, cy, r, inlier_fraction, arc_coverage, ...) or None.

6. Compute confidence:
       conf = w_arc   * arc_coverage           # how much of 360° the arc covers
            + w_inlfr * inlier_fraction        # how well points agreed with the circle
            + w_dist  * exp(-||(cx,cy) - c_prev|| / r_prev)   # locality
            + w_rad   * exp(-|r - r_prev|       / (0.15 * r_prev))   # radius stability
            + w_stage * exp(-|r - r_stage0|     / (0.20 * r_stage0)) # Stage-0 stability
       Weights sum to 1. Initial guess:
       w_arc=0.30, w_inlfr=0.25, w_dist=0.20, w_rad=0.15, w_stage=0.10.

7. Decision:
       conf >= CONF_OK       -> accept fit. Write to tracking.csv with confidence=conf.
                                Update last good = (cx, cy, r).
       conf <  CONF_FREEZE   -> reject fit. Freeze last good. Write tracking.csv row
                                with status="STALE" and confidence=conf.
                                Do NOT update last good (so we don't drift even via the
                                prior).
       between -> tentative   accept fit but mark "LOW_CONF" in the CSV; overlay
                                shows the circle in a thinner / amber colour.
   Initial thresholds: CONF_OK = 0.55, CONF_FREEZE = 0.30. Tunable.

8. If a rescue anchor exists at this frame (teaching_anchors.json), it always wins.
   The fit step is skipped, the anchor is the truth for this frame, and "last good"
   is updated from the anchor.
```

---

## 6. What gets written to `tracking.csv`

Existing columns are preserved. New columns added (anatomical engine only):

| column | meaning |
|---|---|
| `<eye>_iris_confidence` | the conf score above, [0, 1] |
| `<eye>_iris_status` | one of `OK`, `LOW_CONF`, `STALE` |
| `<eye>_iris_arc_coverage` | from the LimbusFit (already computed) |

For the V1 default engine these columns are emitted as empty so downstream code stays
backward-compatible.

---

## 7. Overlay behaviour

Per the clinician's earlier choice (freeze + STALE badge):

* `status = OK`: magenta full circle + centre cross (current behaviour).
* `status = LOW_CONF`: thin amber circle + centre cross + `LOW CONF` small text in
  the top HUD only (not on the eye).
* `status = STALE`: dim red circle (frozen at last good) + `TRACKING LOST` text in the
  HUD + a small red dot in the upper-right of the frame.

No on-eye text labels in any mode (consistent with the rescue-scrubber cleanup we
just landed).

---

## 8. Failure modes the design explicitly prevents

* **Drift onto cheek / nose / brow.** Containment gate: edge pixels outside the
  Stage-0 contour cannot enter the fit. The polygon defines the only legal search
  region.
* **Shrink to pupil.** Already covered by `limbus.py`'s `r_fixed` mode + radius band
  `0.4 r_est < r < 1.8 r_est`. Confidence score additionally penalises radius drift
  from `r_prev` and `r_stage0`.
* **Wander on lost frames.** `status = STALE` freezes last good and does NOT update
  the prior, so the next frame's prior is also the last-good (not a drifted one).
* **Eyelid edge mistaken for limbus.** Already mitigated by `exclude_top_deg=55°`
  asymmetric wedge + sclera-whiteness gate in `limbus.py`.
* **No iris this frame → tracker invents one.** Refusal: if no candidate passes
  `CONF_FREEZE`, the frame is STALE and the iris circle does not move.

---

## 9. What this design does NOT do (yet)

* The contour does **not** move with the head. If the head shifts laterally in the
  frame the contour will be off. Acceptable for V1 anatomical because the clinician
  rescues — and the freeze-on-fail rule means a wrong-position contour just produces
  STALE frames, not drifted ones. Future: optionally re-project the contour via the
  outer/inner canthus face landmarks.
* No re-run-from-anchor seeding of the tracker yet. (That was the deferred Patch 3
  from the rescue-scrubber thread.) The anatomical engine can be added on top of
  resume-from-anchor later; the design here is orthogonal.
* No machine-learned iris segmenter. The whole tracker remains classical CV —
  reproducible, debuggable, no GPU.

---

## 10. Files I will touch

| file | change |
|---|---|
| `src/core/limbus.py` | add optional `polygon` arg to `fit_limbus` and `_radial_edges`. When given, edge pixels outside the polygon are rejected. Backward-compatible: omitting `polygon` keeps current behaviour. |
| `src/core/iris_anatomical_tracker.py` | **new file.** The orchestrator: precondition check, per-frame loop in §5, confidence math in §6, freeze-on-fail, CSV columns in §6. |
| `iris_tracker.py` | dispatch `--engine anatomical` to the new module. V1 unchanged. |
| `app.py` | add `"anatomical"` to the `--engine` choices list. |
| `EYE_VNG_DEVELOPMENT_HISTORY.md` | new §22 documenting this design and the "fail safely, not wander" principle. |
| `TRACKER_REDESIGN.md` | this file. |

What I will **not** touch:
* `src/core/v1_tracker.py` (V1 engine stays as-is)
* `src/core/eye_contour.py`
* `src/core/composite_tracker.py`
* The rescue scrubber
* The simple editor
* Stage 0 approval / proposal code

---

## 11. Parameter table (initial values, all tunable)

| name | value | comment |
|---|---|---|
| `CONF_OK` | 0.55 | ≥ this → OK |
| `CONF_FREEZE` | 0.30 | < this → STALE |
| `w_arc` | 0.30 | arc coverage weight |
| `w_inlfr` | 0.25 | inlier fraction weight |
| `w_dist` | 0.20 | locality weight |
| `w_rad` | 0.15 | radius-vs-prev weight |
| `w_stage` | 0.10 | radius-vs-Stage-0 weight |
| radius band | `0.4 r_est < r < 1.8 r_est` | inherited from `limbus.py` |
| `r_fixed` after N OK frames | once 10 consecutive OK frames seen | switch to fixed-radius centre fits to make the centre occlusion-invariant |

---

## 12. Principle

> **Fail safely, not wander. A missing iris is better than a wrong iris on the face.**

---

## 13. Open questions for the clinician before I write code

1. **Are CONF_OK=0.55 / CONF_FREEZE=0.30 reasonable starting points?** I expect to
   tune these on the vestibular-neuritis clip after the first run.
2. **Should `LOW_CONF` frames go into VNG traces, or be skipped like `STALE`?**
   Current draft: included with confidence in the CSV; downstream can filter.
3. **Does the confidence weight bias feel right?** (Arc coverage is the strongest
   signal because partial cover is the failure mode you care most about.)
4. **Anything missing from the failure-modes list in §8?**

---

End of design.
