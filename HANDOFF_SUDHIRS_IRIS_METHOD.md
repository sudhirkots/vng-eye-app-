# HANDOFF — Sudhir's iris–sclera–almond method (EllSeg-localized, rule-based iris marking)

**Date:** 2026-07-03  **Branch:** `feature/cft-implementation`  **Status:** working method + a full
clinician-in-the-loop correction toolchain. Rule-based mask IoU has **plateaued ~0.44–0.48** on the hardest
frames, but the **iris-CENTRE** (the clinical target) is good (~12 px on the worst frames). Nothing wired into
the clinical path; large outputs gitignored.

## What happened this session
1. **Rebuilt the EllSeg stack on the HOME PC** (was clinic-only) — off-OneDrive at `C:\Users\sudhi\rit-nets\`
   (`rit-nets-venv` torch 2.12.1+cpu, cv2 5, numpy 2.5; cloned `RSKothari/EllSeg`, weights `all.git_ok`).
   Verified it reproduces the clinic zero-shot disc IoU (0.75–0.80 on clean vestibular crops).
2. **Approach-#2 exploration on the vestibular clip** (`samples/nystagmus at rest to left in right vestibular
   neuritis.mp4`, 640 frames): centre-trace, moving-orbit tracker, EllSeg+limbus hybrid, red-paint. Findings:
   - EllSeg finds the iris well on open frames; fails at **head motion** (static orbit → jumps to eyebrow) and
     **extreme side gaze** (sliver over-segments). **Moving orbit (window follows the iris) fixes head motion.**
   - Coverage ≠ accuracy; the raw EllSeg centre **jitters** at half-covered gaze.
3. **Named + built "Sudhir's iris–sclera–almond method"** (`tools/sudhirs_iris_sclera_almond_method.py`):
   EllSeg for LOCATION only → find sclera (white/pink) → iris = uniform dark abutting sclera → **complete the
   CIRCLE/OVAL from the convex limbus arc (RANSAC), then select dark within it**. Rules + Dr. K's 3 sketches
   folded into `docs/RIT_IRIS_METHOD.md` (+ `docs/iris_sketches/`). Key rule order (Dr. K): **shape overrides
   darkness** — complete the oval first, then keep the dark inside (cuts at lid line & sclera, ignores lashes);
   long axis ⟂ gaze, short axis foreshortened along gaze.
4. **Correction loop built:** `tools/rit_paint_tool.html` upgraded to (a) **auto-load** the crops in its folder
   (filename list injected at export) and (b) show a **side-by-side reference** (clean crop to paint | auto
   prediction). `tools/sudhirs_method_export_to_mark.py` exports the failing frames + the auto-loading paint
   tool to `outputs/.../sudhirs_iris_sclera_almond_method/to_mark/`. `tools/sudhirs_method_compare_marks.py`
   scores Dr. K's painted masks vs the auto-pick and diagnoses the broken rule per frame.

## Results (25 clinician-marked HARD frames, side-gaze/half-covered)
- Five rule iterations (pure circle → shape-first → limbus-radius → convex-arc RANSAC → dark-within-oval).
- **RANSAC convex-arc killed the catastrophic failures** (no more total misses / wrong-region).
- **dark-within-oval removed the lid/lash bleed.** Remaining error = mask *extent* (dark threshold clips the
  lighter iris periphery), concentrated on the L eye. **Median IoU stuck ~0.44.**
- **CENTRE error (clinical metric): median ~12 px, iris radius ~90 px; L 11.9 / R 11.5; worst 28 px** — and
  these are the *hardest* frames. Confirms the project reframe: **optimise the centre trace, not mask IoU.**
- Overlay video (thick green iris circle + yellow centre dot):
  `outputs/.../sudhirs_iris_sclera_almond_method/sudhirs_iris_sclera_almond_method.mp4`.

## Reproduce (home PC)
```
C:\Users\sudhi\rit-nets\rit-nets-venv\Scripts\python.exe tools\sudhirs_iris_sclera_almond_method.py   # video
...\python.exe tools\sudhirs_method_export_to_mark.py    # export failing frames + paint tool to to_mark/
...\python.exe tools\sudhirs_method_compare_marks.py     # score Downloads/RIT_masks.zip vs auto-pick
```
Env note: the rit-nets stack is off-OneDrive & not in git (regenerate per `HANDOFF_RIT_NETS_EVAL.md`); it uses
numpy 2, so keep the clinical MediaPipe path on a numpy<2 interpreter.

## Next steps (pick)
- **(a) Plot the full-clip iris-CENTRE trace** from this method and judge it as a nystagmus signal (the real
  deliverable; centre is already ~12 px on the worst frames). RECOMMENDED.
- **(b) Fine-tune EllSeg on the 25 gold masks** if tighter *masks* are needed — rules have plateaued.
- **(c) If continuing rules:** the L-eye undershoot is the dark threshold clipping the lighter iris periphery;
  needs a lid-line-only clip (keep full oval on the sclera side) rather than clip-by-darkness everywhere.

## Hard rules still in force
No nystagmus detection; nothing wired into clinical RIT; no MediaPipe in the clinical path; EllSeg stays a
LOCATION prior; conservative (flag `needs_rescue`, never fabricate from invalid tracking).
