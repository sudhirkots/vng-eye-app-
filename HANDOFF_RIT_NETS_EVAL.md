# HANDOFF — RIT approach #2 (pretrained eye nets): RITnet & EllSeg zero-shot eval

**Date:** 2026-07-02  **Branch:** `feature/cft-implementation`  **Status:** exploration done, EllSeg is
the promising lead; **nothing wired into RIT, nothing about the clinical path changed.**

## Context (where this sits)
Of the four learning approaches (SAM 2 / **pretrained eye nets** / classical geometry / centre-correction
loop), this evaluated **approach #2 — pretrained eye-segmentation networks — zero-shot** on the vestibular
clinician crops. Motivation: the colour-feature learned segmenter (ExtraTrees, iris IoU 0.81 on vestibular)
**does not transfer** to the monochrome pontine clip because position features carry it, not real iris
appearance. Nets learn shape/spatial context, which should transfer better ("learn geometry, not appearance").

## What was tested
Both nets run UNCHANGED (zero-shot, no training on our data), each with its own native preprocessing, on the
same 6 clinician-marked vestibular per-eye crops (full / partial / nasal-sliver / blink). IoU vs the clinician
iris mask.

- **RITnet** (OpenEDS only, 248K params; classes bg/sclera/iris/pupil). Preproc: gray → gamma 0.8 → CLAHE
  1.5/8×8 → Normalize[.5],[.5]; letterboxed to 640×400.
- **EllSeg** "all" model (DenseElNet, trained on 6 datasets; classes bg/iris/pupil, predicts the FULL
  occluded iris + fits pupil/iris ELLIPSES). Preproc: resize width→320, vertical pad→240, per-image standardise.

## Results (iris IoU vs clinician mask)
| frame | RITnet iris | EllSeg iris-only | **EllSeg disc (iris∪pupil)** |
|---|---|---|---|
| f0187_L full | 0.27 | 0.52 | **0.80** |
| f0060_R full | 0.11 | 0.22 | **0.75** |
| f0311_L partial | 0.00 | 0.39 | 0.57 |
| f0227_L partial | 0.00 | 0.20 | 0.43 |
| f0267_R nasal | 0.00 | 0.32 | 0.29 |
| f0123_R blink | 0.05 | 0.03 | 0.16 |

*"disc" = EllSeg iris + pupil (it splits the dark disc into an iris-ring + pupil class, while the clinician
marks the whole disc as iris — so iris-only IoU is unfairly deflated; disc IoU is the fair comparison.)*

Montages saved for review: `outputs/RIT_nets_eval/ritnet_montage.png`, `.../ellseg_montage.png`.

## Findings
1. **RITnet zero-shot FAILS.** Classes land on skin/brow/lid; misses the iris on partial gaze. Trained on
   near-IR VR-headset close-ups → too far from visible-light face video. Not usable here.
2. **EllSeg zero-shot is PROMISING.** On clean, full-iris frames it reaches **disc IoU ~0.75–0.80 — matching
   the trained colour model, with NO training on our data.** It localizes the iris disc correctly and gives
   an iris ellipse (centre) directly = the "optimise the iris-centre trace" reframe.
3. **EllSeg weaknesses:** partial/extreme gaze (0.3–0.6, over-segments onto canthal skin) and blink (0.16,
   hallucinates an eye on skin) — exactly the frames the **orbit lock could constrain**.

## Reproduce (this PC)
Everything for the nets lives **off OneDrive** in `C:\Users\Dr.Sudhir\rit-nets\` (not in git — external repos,
weights, and a venv are large/regenerable):
- `rit-nets-venv\` — Python 3.12 venv: torch 2.12.1+cpu, opencv 5, scikit-image, scikit-learn, matplotlib.
- `RITnet\` (github AayushKrChaudhary/RITnet, `best_model.pkl` in-repo) and `EllSeg\`
  (github RSKothari/EllSeg, weights `weights/all.git_ok` in-repo, ~10 MB).
- Runners (also copied into this repo's `tools/`): `rit_ritnet_smoke.py`, `rit_ellseg_smoke.py`.
- Run: `C:\Users\Dr.Sudhir\rit-nets-venv\Scripts\python.exe tools\rit_ellseg_smoke.py`
- Note: EllSeg predates numpy 2 (`np.int` etc.); the runner shims the removed aliases before importing it.
  Its per-eye crop inputs come from `…vestibular…/RIT_ground_truth/raw/`.

## Recommended next steps (pick one — not started)
- **(a) EllSeg iris-centre trace, orbit-constrained:** take EllSeg's iris ELLIPSE centre, restrict output to
  inside the marked orbit oval (kills the skin bleed), and plot the centre trace across a full clip. This is
  the most direct test of "iris-centre trace, not mask IoU".
- **(b) EllSeg zero-shot on the pontine (monochrome) clip:** does the generalized net survive the domain
  jump where the colour RF collapsed? Directly answers the generalization question.
- **(c) Fine-tune EllSeg on the clinician marks:** small net, CPU-feasible; should lift the partial/blink
  frames and further improve transfer.

## Hard rules still in force
No OpenCV rule tuning; no nystagmus work; no wiring any net into clinical RIT; no MediaPipe in the clinical
path. Orbit-marking method retained as the reference frame. Nothing here is committed beyond the runner
scripts + docs + result montages.
