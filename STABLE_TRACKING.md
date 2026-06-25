# Stable Pupil Tracking — objective & spec (Dr. Kothari, 2026-06-24)

**Problem:** the pupil marker dances frame-to-frame even on a steady eye, because each frame is
detected independently from a single MediaPipe landmark (468/473). This noise could later be
mistaken for real eye movement.

**Objective:** a **stable, time-continuous pupil tracker** — initialise once (user-confirmed), then
follow the *same* pupil through time like a human observer. NOT a per-frame fresh detector.

**Success criterion:** watching the overlay, the marker feels *attached to the pupil*, moving with it,
not dancing randomly.

**Do NOT work on** (explicitly out of scope now): diagnosis, nystagmus detection, VNG classification,
vestibular neuritis, BPPV, head-impulse interpretation, torsion, VNG traces.

## Workflow
**Step 1 — user confirmation (init):**
1. Auto-select the best init frame (first/clearest frame with both eyes well detected).
2. Show enlarged images of both eyes + the detected pupil centre.
3. User confirms each centre, or clicks the true centre to correct it.
4. Store confirmed coordinates (→ `metadata.json`).

**Step 2 — tracking mode:**
- Follow the same pupil continuously; use the previous frame position as the prior/search anchor.
- Avoid unnecessary jumps; reject impossible jumps (flag, don't delete).
- Use the MediaPipe iris estimate as a **guide, not absolute truth**; use the **ring mean** (5 iris
  landmarks), not a single landmark.
- If tracking is lost, attempt re-detection and clearly flag it.

**Step 3 — visual debug overlay:** confirmed pupil centre + current tracked centre + status + frame
number + timestamp. Colour: green=confident, yellow=uncertain, red=lost. Show the rejected MediaPipe
point when it differs too much.

**Step 4 — tracking_status per frame (in CSV):** `initialized | tracked | uncertain | lost |
redetected` (with reason suffixes where useful).

**Step 5 — quality review mode:** screen showing original frame + enlarged eye ROI + detected pupil +
tracked pupil; rapid review of low-confidence frames.

## Stability rules
- Stability must come from a **better estimator (iris ring) + temporal continuity + outlier
  rejection** — NOT from secretly smoothing the output trace. No hidden EMA/savgol/Kalman on the
  final trace. Raw failures stay visible.

## metadata.json
initial frame number; detected left/right pupil; user-corrected left/right pupil (= detected if the
user accepted without correction); interocular distance; video/resolution/fps.

## Tools
- `pupil_tracker.py <video> [--output-dir outputs] [--auto-confirm] [--review]`
  - default: interactive confirm (OpenCV window, click to correct). `--auto-confirm`: accept detected
    init without GUI (for batch / headless validation). `--review`: low-confidence review screen.
- Core: `src/core/pupil_tracking.py` (`PupilDetector` ring-mean detection, `select_init_frame`,
  `StableTracker`).
