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
