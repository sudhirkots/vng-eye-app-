# V2 — clinical-model re-derivation of the iris-tracking pipeline

This directory is a clean, **independent** re-derivation of the EyeVNG iris-tracking
pipeline starting from the clinical reasoning model:

1. Find the face.
2. Use the face to estimate approximate eye zones.
3. Identify the eye opening — the white sclera anchors the eye; the iris is the dark
   structure within the white.
4. Treat the eye as a globe; the iris rides on its surface.
5. Model the iris/limbus as a full circle, even when only a partial arc is visible.
6. Track the centre of the completed iris circle.
7. Read the iris centre against the moving oval eye opening.
8. Use four oval points (medial, lateral, upper, lower) as the reference frame.
9. Build eye-local axes (medial→lateral = horizontal; lower→upper = vertical).
10. Analyse movement in eye-local coordinates, not raw image coordinates.
11. Detect only rhythmic slow-drift/fast-jerk nystagmus; ignore random movement.

## Independence rules

* V2 code MUST NOT import V1 modules (`src/core/*`, `iris_tracker.py`, `app.py`, ...).
* V1 code remains the production path; V2 is parallel and prototype-only.
* No file in V2 modifies anything outside `v2/`.

## Layered architecture (target)

| Layer | Purpose | Status |
|---|---|---|
| 1 | MediaPipe face + approximate eye-zone locator (Stage-0 hint ONLY; closed before tracking) | Milestone 0 |
| 2 | Stage-0 clinician-marked **4-point oval per eye** (medial / lateral / upper / lower) | **Milestone 1 (this commit)** |
| 3a | "Virtual spectacles" tracker — rigid (similarity) transform of the 8 Stage-0 anchors per frame | **Milestone 1 (this commit)** |
| 3b | Moving eye-local reference frame (axes / projection) | not started |
| 4 | Iris circle / partial-arc detector (dark iris vs bright sclera) | not started |
| 5 | Eye-local movement trace | not started |
| 6 | Nystagmus rhythm detector | not started |

## Milestone 1 — virtual spectacles

The two clinician-marked orbit ovals are treated as one **rigid pair of spectacles**
mounted on the face. Per frame, the 8 Stage-0 anchor points are tracked by KLT
optical flow; a weighted Umeyama similarity (translation + rotation + uniform scale,
4 DOF) is fit to map the Stage-0 anchors onto the tracked positions. The displayed
ovals come from that fitted transform, so individual KLT noise cannot deform the
spectacles. The clinician principle:

> First stabilise the reference frame. Then track the moving object inside it.

Run it with [`Probe_V2_Orbit_Tracking.bat`](../Probe_V2_Orbit_Tracking.bat). After
Stage-0 marking the probe writes:

```
outputs/v2_orbit_tracking_probe/
├── orbit_stage0_v2.json
├── orbit_tracking_overlay.mp4
├── orbit_tracking_points.csv
└── orbit_tracking_summary.json
```

## What this milestone proves

Milestone 0 answers exactly one question:

> Does MediaPipe give us a reliable moving search region for the eyes?

Run the probe ([probe_face_eye_zones.py](probe_face_eye_zones.py)) on a clip and view
the resulting overlay video to decide. If the eye-zone boxes follow the face through
the clip, V2 Layer 1 is good and we proceed to Layer 2 (Stage 0). If not, we need
another way to locate the moving eye region.

## MediaPipe usage policy in V2

MediaPipe is allowed only for:
* face bounding box,
* approximate image-side L / R eye-zone rectangles,
* frame-to-frame eye-region movement.

MediaPipe is **not** allowed to define:
* iris radius,
* iris centre,
* limbus,
* pupil,
* nystagmus signal.

This is the same policy locked in [`EYE_VNG_DEVELOPMENT_HISTORY.md` §19b.4 / §19b.5](../EYE_VNG_DEVELOPMENT_HISTORY.md):
clinician-approved Stage-0 anatomy is the source of truth for V1; V2 will adopt the
same rule once Layer 2 is built.

## Running Milestone 0

From the repo root, in PowerShell:

```
.\.venv\Scripts\python.exe v2\probe_face_eye_zones.py --video "samples\nystagmus at rest to left in right vestibular neuritis.mp4"
```

Or double-click [`Probe_V2_Face_Eye_Zones.bat`](../Probe_V2_Face_Eye_Zones.bat) (in
the repo root).

Outputs:

```
outputs/v2_face_eye_zone_probe/face_eye_zones_overlay.mp4
outputs/v2_face_eye_zone_probe/face_eye_zones.csv
```
