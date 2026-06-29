# CLINIC HANDOFF — 2026-06-29

> Written when active development was paused for the clinic break.
> This file is intentionally a "what to read on Monday" note.

---

## 1. Current project state

- All recent work is saved in [`C:\Users\sudhi\OneDrive\Documents\Neurology Talks\VNG-EYE app`](.).
- V1 clinical iris tracker is the working baseline. It tracks the iris reasonably on the
  first ~187 frames of the vestibular-neuritis test clip, then starts drifting.
- Two new V2 clinical rule documents have been written and are the source of truth for
  what comes next.
- No tracker logic changes have been made on top of the V1 baseline since the last
  paused decision.
- A previous V2 prototype (orbit + spectacles + sclera/iris content) was built but
  has been **explicitly set aside**. It must not be used as the clinical tracker.
- A previous experimental anatomical engine was built; it is **explicitly set aside**.

---

## 2. What was fixed in this work block

1. **Stage-0 anatomy is the source of truth** for V1 clinical tracking.
2. **MediaPipe was removed from the V1 clinical path.** The V1 runtime previously
   silently overrode the clinician-approved Stage-0 limbus radius with MediaPipe's
   iris-landmark radius (~44 px), which made V1 lock onto pupil-sized structures.
   That bug has been corrected. V1 now reads the iris/limbus radius directly from
   the Stage-0 JSON.
3. **Stage-0 radii were verified** on the vestibular-neuritis clip:
   `iris.L.radius = 90.55 px` and `iris.R.radius = 87.06 px`. Correct full-limbus
   values for a 1920×1080 face video.
4. A **V1 post-tracking plausibility filter** is active: rejects iris circles that
   are clearly outside the Stage-0 oval or whose radius is wildly off Stage-0.
5. A **per-frame orbit-guardrail discrimination** was added on top of the
   plausibility filter to distinguish `tracker_drift_outside_orbit` from
   `good_iris_but_static_orbit_stale` (head moved but iris was still tracked).
6. The Stage-0 approval UI (`approve_interactive` in `iris_tracker.py`) was cleaned
   up: face landmarks hidden, contour mode auto-selects the clicked eye.
7. Two **new locked rule documents** were written and are now the V2 source of truth:
   - [IRIS_IDENTIFICATION_RULES_V2.md](IRIS_IDENTIFICATION_RULES_V2.md)
   - [ORBIT_AND_IRIS_TRACKING_RULES_V2.md](ORBIT_AND_IRIS_TRACKING_RULES_V2.md)
8. A **plan** for the smallest validator change consistent with those rules was
   produced (audit + proposed `validate_iris_candidate` function + 6-section
   report). **No code was written from that plan.**

---

## 3. What was deliberately stopped / paused

- **No new V2 spectacles tracker work.** The V2 spectacles prototype drifts onto
  eyebrows; it is set aside.
- **No anatomical-engine work.** The `--engine anatomical` path was over-engineered
  and is left as experimental only.
- **No MediaPipe in V1 clinical tracking.** MediaPipe may only be a Stage-0
  assistant.
- **No nystagmus detector changes.** Tracker reliability is upstream; no nystagmus
  output should be produced from invalid tracking.
- **No tracker rewrite.** The V1 limbus fitter (`src/core/limbus.py`) is the working
  baseline and must not be replaced.
- **No commits of large generated outputs** under `outputs/`. Only source code,
  rules docs, markdown, BAT launchers, JSON configs, small CSV/review files.

---

## 4. Source-of-truth documents

```text
Current locked direction:
Use the new V2 clinical rules to define iris correctly.

The two source-of-truth rule documents are:
- IRIS_IDENTIFICATION_RULES_V2.md
- ORBIT_AND_IRIS_TRACKING_RULES_V2.md
```

Supporting documents (project history + design notes):

- [EYE_VNG_DEVELOPMENT_HISTORY.md](EYE_VNG_DEVELOPMENT_HISTORY.md) — chronological
  development history, locked principles, decision log.
- [TRACKER_REDESIGN.md](TRACKER_REDESIGN.md) — the anatomical-engine design
  (kept as historical reference; the engine itself is set aside).
- [V1_CURRENT_CHANGES_AND_RATIONALE.md](V1_CURRENT_CHANGES_AND_RATIONALE.md) —
  V1-specific changes recorded during the session.
- [HANDOFF.md](HANDOFF.md) — earlier session handoff (kept for context).

Recovery snapshots (mid-session safety files, kept on purpose):

- `RECOVERY_current_uncommitted_changes.patch`
- `RECOVERY_git_status_before_restore.txt`
- `RECOVERY_untracked_files.txt`

---

## 5. What should be done next

```text
Audit current tracker and propose the smallest iris-candidate validator based on:
scleral support,
dark circular/arc geometry,
Stage-0 limbus radius,
orbital containment,
and conservative freeze-on-uncertainty.
```

More specifically, the agreed first step on resumption is:

1. Add a NEW file `src/core/iris_candidate_validator.py` exposing a single function
   `validate_iris_candidate(...) -> IrisCandidateValidation` that scores any
   iris candidate with:
   - sclera support (broad bright/low-saturation region adjacent to the arc)
   - dark circular / arc-like geometry
   - radius plausibility vs Stage-0
   - containment within the Stage-0 (or moving) orbital oval
   - temporal continuity vs the last good centre
2. Wire it into `iris_tracker.py` after `V1Tracker.step()` returns
   `fm_L`/`fm_R`, BEFORE the existing `_apply_v1_plausibility` call, gated on
   `engine == "v1"`. Replace the existing `_v1_plausibility_check` /
   `_classify_orbit_outside` / `_update_iris_history` helpers with a thin
   wrapper that delegates to the validator.
3. Do NOT touch `src/core/limbus.py`. Do NOT touch `src/core/v1_tracker.py`.
4. Add structured per-frame logs of the form
   `[iris-validator][L] fr=465 reject=no_sclera_support sclera=0.12 arc=0.74
   radius=0.91 containment=0.20 temporal=0.40`.
5. Verify on the vestibular-neuritis clip by observing:
   - frame ~188 → expected `reject=unstable_jump`
   - frames ~275–284 → expected `reject=no_sclera_support` or
     `reject=skin_fold_or_shadow`
   - frame 392 → still rejected (V1 misses the visible iris; validator catches it).

---

## 6. What should NOT be done next

```text
Do not reintroduce MediaPipe into V1 clinical iris tracking.
MediaPipe may only be a Stage-0 assistant.
Stage-0 clinician-approved anatomy is the source of truth.
The tracker must accept only a dark full or partial circular iris/limbus
structure supported by true scleral white inside the approved orbital oval.
If uncertain, freeze rather than wander.
```

And:

```text
Do not implement nystagmus detection until tracking is reliable.
Do not produce "no nystagmus" from invalid tracking.
Do not redesign everything from scratch.
First audit the current tracker and propose the smallest validator
insertion point.
```

---

## 7. Recent important conclusions (in one place)

```text
1. Stage-0 limbus radii were correct: approximately 90.55 px and 87.06 px.
2. V1 previously used MediaPipe iris radius incorrectly; this has been
   corrected/removed.
3. No MediaPipe should be used in V1 clinical tracking.
4. The remaining deeper issue is not only drift; it is that the tracker's
   definition of "iris" was too loose.
5. The next work should add an iris-candidate validator based on scleral
   support + circular/arc geometry + orbital containment + Stage-0 radius
   plausibility.
6. False negatives are acceptable; false positives are dangerous.
```

---

## 8. Important asset files (do not delete)

| asset | path |
|---|---|
| Stage-0 approval for the test clip | [outputs/nystagmus at rest to left in right vestibular neuritis_tracked/approved_landmarks.json](outputs/nystagmus%20at%20rest%20to%20left%20in%20right%20vestibular%20neuritis_tracked/approved_landmarks.json) |
| Clinician rescue anchors (active backup) | `outputs/nystagmus at rest to left in right vestibular neuritis_tracked/teaching_anchors.json.bak` |
| Second rescue-anchor snapshot | `outputs/nystagmus at rest to left in right vestibular neuritis_tracked/teaching_anchors.json.session2.bak` |
| V1 launcher | [Track_V1_Vestibular_Neuritis.bat](Track_V1_Vestibular_Neuritis.bat) |
| Rescue scrubber launcher | [Rescue_Vestibular_Neuritis.bat](Rescue_Vestibular_Neuritis.bat) |
| Stage-0 approval launcher | [Approve_Vestibular_Neuritis.bat](Approve_Vestibular_Neuritis.bat) |
| V2 face-zone probe | [Probe_V2_Face_Eye_Zones.bat](Probe_V2_Face_Eye_Zones.bat) |
| V2 orbit-tracking probe | [Probe_V2_Orbit_Tracking.bat](Probe_V2_Orbit_Tracking.bat) |

---

End of handoff.
