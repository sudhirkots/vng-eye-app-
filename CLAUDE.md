# Clinical Nystagmus Detector (EyeVNG) — project instructions for Claude Code

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
2. Read/reconcile the saved state (current first):
   - **`CHECKPOINTS_BRIEF_DESCRIPTION.md`** (what each locked checkpoint achieved) and
     **`docs/CLINICAL_NYSTAGMUS_DETECTOR.md`** (the CLINICAL OBJECTIVE + all current rules);
   - `git log --oneline -10` + `git status --short` + `git tag -l "checkpoint/*"`;
   - *legacy/history only if needed:* `HANDOFF_*.md`, `EYE_VNG_DEVELOPMENT_HISTORY.md`,
     `docs/RIT_STRATEGY_AND_ROADMAP.md` (these describe the retired iris-tracking track).
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
  - *(retired 2026-07-04) `checkpoint/limbus-arc-rescue-gate`* — accurate iris marking (fixed-radius circle
    from the sclera-facing limbus + rescue gate, `tools/step2_fixed_base_circle.py`). Tag DELETED: we gave up
    on precise iris tracking/marking — the clinical goal only needs the rough EllSeg centroid. Code stays in
    history; we go straight from `checkpoint/ellseg-anchor` to the nystagmus-direction checkpoint below.
  - `checkpoint/nystagmus-direction-primary-gaze-zones` — **GOAL REFRAMED to clinical nystagmus-direction
    detection** (not perfect iris geometry). Pipeline: `tools/ellseg_centroid_trace.py` (clean EllSeg centroid
    + per-eye sclera balance) → `tools/nystagmus_direction.py` (windowed slow-phase asymmetry detector +
    Sustained Gaze-Zone classification by sclera-balance/canthus-proximity, head-motion invariant). Correctly
    calls the right-vestibular-neuritis clip: **primary gaze, LEFT-beating (conf 0.34)** at 30 fps. Rules in
    `docs/CLINICAL_NYSTAGMUS_DETECTOR.md`: Clinical Nystagmus Direction, Fast-Phase Acceptance, Windowed Slow-Phase
    Asymmetry, Sclera-Balance Gaze Zone, Sustained Gaze Zone. Note: 30 fps undersamples the fast phase (no beat
    rate) — direction only; 60/120 fps recommended for rate. Orbit-lock tools (`tools/orbit_lock*.py`,
    `rit_orbit_lock_marker_v2.html`) built this session but NOT needed for the direction answer.
- To return to a stage: `git fetch origin --tags` then `git checkout checkpoint/<name>`.

## Current direction — QUALITATIVE nystagmus diagnosis (Dr. K, 2026-07-04)

**The goal is a rough, qualitative bedside read — NOT precise iris tracking/measurement.** All we want:
(1) is there nystagmus or not; (2) is it in primary position or only on side gaze (which gaze zone);
(3) direction (left/right/up/down-beating); (4) does it increase or decrease looking to one side (Alexander's
law). No exact velocity/rate/degrees; **no accurate iris outline needed** — the rough EllSeg centroid suffices.

**Approach that works (checkpoint `nystagmus-direction-primary-gaze-zones`):** EllSeg iris CENTROID (clean,
smooth eye-position signal) → windowed **slow-phase asymmetry** direction detector → **sustained sclera-balance
gaze zones** (head-motion invariant). Correctly calls the right-vestibular-neuritis clip as primary-gaze
LEFT-beating at 30 fps. Full objective + rules in `docs/CLINICAL_NYSTAGMUS_DETECTOR.md`; milestones in
`CHECKPOINTS_BRIEF_DESCRIPTION.md`.

**Retired:** precise iris tracking/marking (SAM2 / learned segmenter / limbus-arc circle) — abandoned; only a
rough eye-position signal is needed. 30 fps gives direction only (fast phase undersampled); 60/120 fps for rate.

## Hard rules (do not break without explicit approval)

- **No commit/push without the user's explicit OK.**
- Goal = a QUALITATIVE nystagmus reader (present/absent · direction · gaze zone · gaze effect · confidence).
  NOT precise iris geometry, velocity, rate, or a VNG waveform. Rough EllSeg centroid is enough.
- **Never claim "no nystagmus" from insufficient/invalid data** — report **unclear / insufficient** instead
  (false negatives dangerous). If the eye-position signal is unreliable, say so; don't force a call.
- Conservative on confidence: report clear / probable / unclear honestly; don't overstate.
- EllSeg is used ONLY as a rough iris *locator* (centroid); do not depend on its mask/shape.
- Retired (do not resurrect without a reason): precise iris tracking/marking — SAM2, learned segmenter,
  limbus-arc fixed circle, oval foreshortening. Their code stays in history.
- Orbit Lock is kept as an optional future input (head-free eye position) — do not delete it.

## Environment (per machine — the `.venv` shim is unreliable across machines)

- Clinical/Stage-0 app needs **numpy<2 + mediapipe**; the ML workbench + nets need sklearn/torch (numpy 2).
  Keep them in **separate venvs**. Installing sklearn/torch pulls numpy 2, which breaks mediapipe 0.10.18.
- **Home PC:** use `py -3.14` or `C:\Users\sudhi\eyevng-venv`. **This PC:** `.venv` was repointed to
  `C:\Users\Dr.Sudhir\…\Python312`; the eye-nets stack is a separate off-OneDrive venv at
  `C:\Users\Dr.Sudhir\rit-nets\` (machine-specific — regenerate elsewhere per `HANDOFF_RIT_NETS_EVAL.md`).
- Large generated outputs under `outputs/` are gitignored (kept on OneDrive only).

## Naming

The project is the **Clinical Nystagmus Detector**. The old "RIT / Rescue Iris Tracker" branding is **legacy**
(that was the retired precise iris-tracking approach). Existing `rit_*` filenames, `RIT_ground_truth/` folders,
and handoffs are kept as-is (renaming them would break scripts) but the *goal* is the nystagmus reader, not
iris tracking.

**Orbit Lock is KEPT** — `tools/orbit_lock*.py`, `src/core/rit_orbit_lock.py`,
`tools/rit_orbit_lock_marker_v2.html`. It is not part of the current path but may feed a head-free eye
position in later; do not delete it.
