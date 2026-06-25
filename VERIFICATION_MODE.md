# Tracking Verification / Debug Mode

Goal (Dr. Kothari, 2026-06-24): **I should be able to watch the overlay and immediately know whether
tracking is trustworthy.** Step 1 (tracking) must be reliable before any downstream work (CSV for
analysis, graphs, diagnosis, torsion — all PAUSED). Target: **>95% of frames stably tracked** in a
good-quality primary-gaze video. Targets the **face-based Mode A** pipeline.

## Tool
`python debug_tracking.py <video> [--output-dir outputs] [--max-debug-frames N]`

Outputs (under `<output-dir>/<video-stem>/`):
- `tracking_quality.csv`
- `debug_overlay.mp4` (colour-coded, downscaled for quick review)
- `debug_frames/` — saved uncertain + lost frames (capped/sampled)
- `tracking_summary.json` + console summary (%good / %uncertain / %lost, longest lost run, pass/fail vs 95%)

## 1. Per-frame info (overlay + CSV)
face detected (Y/N), left eye (Y/N), right eye (Y/N), per-iris confidence, frame number, timestamp.

## 2. `tracking_quality.csv` columns (exact)
`frame_number, time_sec, face_detected, left_eye_detected, right_eye_detected,
left_iris_x, left_iris_y, right_iris_x, right_iris_y, tracking_status`
(plus `left_conf`, `right_conf` appended — extra info, never fewer columns than specified).
**Raw only:** when an eye is not detected its iris columns are blank. Carried/estimated positions
are NEVER written into these columns — they live only on the overlay, clearly flagged.

## 3. Overlay colour flags
- **green** = good tracking
- **yellow** = uncertain (low confidence / only one eye / carried-forward / suspicious jump)
- **red** = lost (no face, collapsed mesh, or no iris)

## 4. Frame review mode
Save lost (red) and uncertain (yellow) frames into `outputs/.../debug_frames/`, filename encodes
frame number + status + reason. Capped/sampled to `--max-debug-frames` per category.

## 5. Robustness (each one VISIBLE, never silent)
- **Crop face → then read eyes:** use the face ROI to focus eye/iris reading and to crop saved
  debug frames tightly on the eyes for inspection.
- **Carry-forward on brief failure:** if an eye drops out for a few frames but was just seen,
  show its last-known position on the overlay (distinct dashed marker) + status `carried_forward`.
  Carried values are NOT written to the raw iris columns.
- **Reject impossible jumps — FLAG, don't delete.** A jump larger than a scale-relative threshold
  (fraction of inter-ocular distance) marks the frame `jump_flag` (yellow); the raw coordinate stays
  in the CSV and on the overlay. **⚠️ Nystagmus caveat:** fast phases are real large jumps, so the
  threshold is deliberately high and jumps are only flagged, never removed — otherwise we'd delete
  the signal we want to measure.

## 6. No silent smoothing or guessing
Raw tracking failures must be visible. No interpolation/smoothing in this mode.

## 7. tracking_status taxonomy
`good` | `uncertain:<reasons>` | `lost:<reason>` where reasons ∈
{`low_conf`, `one_eye`, `carried_forward`, `jump_flag`} / {`no_face`, `collapsed_mesh`, `no_iris`}.

## Recording protocol (Dr. Kothari) — improves input quality
Phone horizontal, 1080p or 4K. Distance 30–40 cm. Face fills most of the frame. Bright light from
the front (not behind). No spectacles if possible. Eyelids open, look at the camera. For HIT use
60 or 120 fps.

## Likely causes of imperfect tracking (to investigate against the CSV/overlay)
face too small · eyes poorly lit · head moving too much · eyelids covering iris · motion blur ·
only 30 fps · MediaPipe tracks eyelids but iris centre jumps · glasses/reflection · patient too far ·
camera angle too high/low.
