# RIT iris — strategy & roadmap (Dr. Kothari, 2026-07-01)

*Why we are doing what we are doing. Read this before changing the iris detector.*

## Where we are
We are solving iris detection with a **classical OpenCV, rule-based detector** (`tools/rit_iris.py`):

```
find white/pink sclera → find dark iris → remove lashes → fit ellipse/circle → keep everything inside the almond
```

This can work, but it is **brittle**: every new frame creates an exception — shadow, eyelash, partial
iris, nasal crowding, reflection, eyelid, skin, low contrast. And so far the detector has been judged the
**weak** way:

```
OpenCV rules → visual overlay → subjective judgement ("does this look right?")
```

That produces the endless cycle of **fix one frame, break another**.

## The change we are making now (NOT abandoning OpenCV)
We are **adding ground truth** so we can judge the detector **objectively**, by measurement instead of by
eye:

```
Dr. K's ground-truth masks → OpenCV rules → measured comparison → improve
```

In plain words: **we make an exam paper for the detector.** Dr. K marks ~30–50 difficult frames —
`this is almond`, `this is sclera`, `this is iris` — and the code must then answer, for every attempt:

- How close did I get to Dr. Kothari's marking? (IoU / overlap for iris, sclera, almond)
- How many pixels did I wrongly include?
- How far is my iris centre from the marked centre? (pixels)
- Did I include eyelid / skin / lash?
- Did iris + sclera fill the almond correctly?

This tells us **objectively** whether the OpenCV method is salvageable — and stops the guessing.

## Decision rule
- **If OpenCV scores well on the marked frames → keep OpenCV** and keep improving it against this fixed
  test set.
- **If OpenCV keeps failing even after being tested against the markings → move to the next approach:**
  a **small segmentation model / semi-automatic segmentation**. Instead of only hand-written thresholds,
  train/fine-tune a small model to learn from Dr. K's marked examples. The task becomes **pixel
  classification**: for each pixel in the eye image, label it `iris` / `sclera` / `eyelid-skin-outside`.

**We do not jump to model training now.** The ground-truth step is done first because it **serves both
paths** — even AI segmentation needs Dr. K's marked examples as training/validation data.

## Roadmap
```
Step 1: Continue OpenCV, but STOP guessing.
Step 2: Create 30–50 gold-standard marked frames (Dr. K marks almond, sclera, iris).
Step 3: Build an evaluator to compare OpenCV output with the markings (measured, not visual).
Step 4: Improve OpenCV against this fixed test set.
Step 5: If OpenCV still fails, use the SAME marked frames to train/guide a segmentation model.
```

**The key point:** we are *not yet* abandoning OpenCV — we are adding ground truth so we can *know*,
objectively, whether OpenCV is good enough. This ends "fix one frame, break another."

## Status (2026-07-01) — Steps 2 & 3 built
- **Step 2 (marking set):** `tools/rit_make_ground_truth_frames.py` → 38 per-eye crops in
  `RIT_ground_truth/to_mark/` covering the flagged frames (187/227/267/311/341/378/382/460), a
  full/partial/lash/extreme spread, and detector gap frames. Marking convention in
  `RIT_ground_truth/MARKING_INSTRUCTIONS.md` (opaque **RED = iris**, **BLUE = sclera**, optional green
  almond outline; else almond = iris ∪ sclera). **Awaiting Dr. K's marks.**
- **Step 3 (evaluator):** `tools/rit_extract_ground_truth.py` (painted PNGs → binary masks + overlays +
  polygons) and `tools/rit_evaluate_iris_masks.py` (imports the detector's own `fit_iris`, runs it
  cold-start on each marked frame, reports iris/sclera/almond IoU, centre error in px, iris-outside-almond
  fraction, and the sclera+iris-fill-almond check; writes `RIT_ground_truth/evaluation.json`).
- The method spec (the OpenCV rules) is in `docs/RIT_IRIS_METHOD.md`.
- **Next:** Dr. K marks the 38 frames → run extract + evaluate → read the numbers → decide Step 4 vs
  Step 5. **No more detector tuning by visual guess.** Nothing committed; probe-only.
