# EyeVNG / RIT — project instructions for Claude Code

This file is auto-loaded every session on any machine (it lives in the project, synced via OneDrive + git).
It exists so work continues seamlessly across the home PC (`C:\Users\sudhi\…`) and the other PC
(`C:\Users\Dr.Sudhir\…`). The clinician/user is Dr. Sudhir Kothari.

## Session conventions — "Let's begin" / "Let's end"

The SAME two-phrase convention runs on BOTH machines (clinic `C:\Users\Dr.Sudhir\…` and home
`C:\Users\sudhi\…`), sharing the same OneDrive project folder and the GitHub remote. **GitHub is the
durable source of truth; OneDrive is convenience sync.** The user starts a session with "Let's begin" and
ends it with "Let's end", from either machine.

### "Let's begin" (resume) — orient first, do NOT act
1. Sync from GitHub: `git fetch origin`; compare local vs `origin/<branch>`. If behind and the working tree
   is **clean**, `git pull` to take the latest committed state. If the tree is dirty or there's a conflict,
   **STOP and report** — never force.
2. Read/reconcile the saved state:
   - the **latest `HANDOFF_*.md`** by date (currently `HANDOFF_RIT_NETS_EVAL.md`) + `HANDOFF_RIT_LEARNED_SEGMENTER.md`;
   - recent sections of **`EYE_VNG_DEVELOPMENT_HISTORY.md`** and the RIT docs
     (`docs/RIT_STRATEGY_AND_ROADMAP.md`, `docs/RIT_IRIS_METHOD.md`, `IRIS_IDENTIFICATION_RULES_V2.md`,
     `ORBIT_AND_IRIS_TRACKING_RULES_V2.md`);
   - `git log --oneline -10` + `git status --short`;
   - anything freshly modified under `outputs/**/RIT_ground_truth/` (new clinician marks) and
     `RIT_learning_workbench/runs/` (new runs).
3. Give a short accurate summary of where things stand, then **wait for the instruction.**
   ("Let's begin" reconstructs the WORK STATE from these files — not this literal chat transcript.)

### "Let's end" (save + store) — standing approval to COMMIT and PUSH
1. Capture the session's work in files (update the relevant `HANDOFF_*.md` and, if the state changed,
   `EYE_VNG_DEVELOPMENT_HISTORY.md`). Large generated outputs stay out (gitignored).
2. `git add -A`; commit with a clear message summarizing the session; `git push origin <branch>`.
3. Verify the tree is clean and in sync with origin; report the commit hash. If push is rejected (behind),
   pull/rebase and retry, or report honestly.

**Only "Let's end" authorizes committing/pushing.** At all other times, still no commit without an explicit OK.

## Checkpoints — named locked stages (Dr. K, 2026-07-04)

Each time a rule/stage is confirmed good, **lock it as a named checkpoint** (like clinical-trial phases), so
if a later change makes things worse we can return to the last known-good stage without losing the reasoning.
Mechanism = an **annotated git tag** `checkpoint/<short-name>` on the commit that locks the stage, pushed to
GitHub (durable), + a matching commit message `Lock <rule name>`.

- **Do NOT pre-name future checkpoints.** Name each one only when we actually reach and confirm that stage,
  one by one, as the work reveals it.
- Checkpoints locked so far:
  - `checkpoint/ellseg-anchor` — **EllSeg Anchor Locked**: EllSeg locates the iris well enough; do NOT use its
    jagged mask as the final mark (Step 1 = Location of Iris by EllSeg). *(= tag `iris-step1-ellseg-location`.)*
  - `checkpoint/limbus-arc-rescue-gate` — **Step 2a+2b locked** (`tools/step2_fixed_base_circle.py`): fixed
    base radius (no size shimmer) + True Limbus Evidence / Medial-Lateral Limbus Arc / Partial Arc Completion
    (one clean sclera-facing arc places the fixed circle) + Side-Gaze Rescue Gate (a frame is
    `fixed_circle_ok` only if the circle stays near the EllSeg anchor AND overlaps the EllSeg disc AND has a
    valid medial/lateral arc; else `needs_rescue`, carry forward last good). Verified by full-clip rescue
    sweep: rescues are side-gaze/blink/occlusion/drifted-fit, no good open frame withheld. Foreshortening NOT
    yet added (deferred). (Supersedes the never-tagged intermediate "fixed-base-radius" state.)
  - `checkpoint/nystagmus-direction-primary-gaze-zones` — **GOAL REFRAMED to clinical nystagmus-direction
    detection** (not perfect iris geometry). Pipeline: `tools/ellseg_centroid_trace.py` (clean EllSeg centroid
    + per-eye sclera balance) → `tools/nystagmus_direction.py` (windowed slow-phase asymmetry detector +
    Sustained Gaze-Zone classification by sclera-balance/canthus-proximity, head-motion invariant). Correctly
    calls the right-vestibular-neuritis clip: **primary gaze, LEFT-beating (conf 0.34)** at 30 fps. Rules in
    `docs/RIT_IRIS_METHOD.md`: Clinical Nystagmus Direction, Fast-Phase Acceptance, Windowed Slow-Phase
    Asymmetry, Sclera-Balance Gaze Zone, Sustained Gaze Zone. Note: 30 fps undersamples the fast phase (no beat
    rate) — direction only; 60/120 fps recommended for rate. Orbit-lock tools (`tools/orbit_lock*.py`,
    `rit_orbit_lock_marker_v2.html`) built this session but NOT needed for the direction answer.
- To return to a stage: `git fetch origin --tags` then `git checkout checkpoint/<name>`.

## Current direction (RIT = Rescue Iris Tracker)

RIT = **orbit-marking method** (clinician draws the eye-opening oval with 3 clicks/eye on frame 1,
MediaPipe-free — the reference frame) **+ clinician-marked ground truth + a learned iris/sclera model**.
Four candidate learning approaches: **SAM 2**, **pretrained eye nets** (RITnet/EllSeg/DeepVOG),
**classical geometry**, **centre-correction loop**. Two reframes govern all: optimise the **iris-centre
trace** (not mask IoU), and learn **geometry/motion** (not appearance).

**Latest result:** EllSeg (pretrained net) is the lead — zero-shot disc IoU 0.75–0.80 on clean frames; RITnet
fails. Details + reproduction + next steps (a/b/c) in `HANDOFF_RIT_NETS_EVAL.md`. The colour-feature
ExtraTrees model (iris IoU 0.81 vestibular) does NOT transfer to the monochrome pontine clip.

## Hard rules (do not break without explicit approval)

- **No commit/push without the user's explicit OK.**
- No OpenCV iris-rule tuning (that detector is DEPRECATED — its rules are annotation rules now).
- No nystagmus detection until tracking is reliable; never emit "no nystagmus" from invalid tracking.
- No wiring any learned model into the clinical RIT path yet.
- No MediaPipe in the clinical iris path (Stage-0 suggestion only; may be unavailable anyway).
- Conservative: if uncertain, freeze / mark needs_rescue — false negatives ok, false positives dangerous.

## Environment (per machine — the `.venv` shim is unreliable across machines)

- Clinical/Stage-0 app needs **numpy<2 + mediapipe**; the ML workbench + nets need sklearn/torch (numpy 2).
  Keep them in **separate venvs**. Installing sklearn/torch pulls numpy 2, which breaks mediapipe 0.10.18.
- **Home PC:** use `py -3.14` or `C:\Users\sudhi\eyevng-venv`. **This PC:** `.venv` was repointed to
  `C:\Users\Dr.Sudhir\…\Python312`; the eye-nets stack is a separate off-OneDrive venv at
  `C:\Users\Dr.Sudhir\rit-nets\` (machine-specific — regenerate elsewhere per `HANDOFF_RIT_NETS_EVAL.md`).
- Large generated outputs under `outputs/` are gitignored (kept on OneDrive only).

## Naming

Use RIT / RIT Orbit Lock / RIT Guardrails / RIT Stage 0 / RIT Rescue. Do NOT call the workflow
"V1-plus-anatomy", "V2 tracker", or "anatomical engine".
