"""V2 Layer 1 — face + approximate eye-zone localiser.

This module is allowed to use MediaPipe FaceMesh, and ONLY for:
  * face bounding box,
  * approximate image-side L / R eye-zone rectangles,
  * frame-to-frame eye-region movement.

It is NOT allowed to define iris radius, iris centre, limbus, pupil, or any clinical
nystagmus signal. Those are downstream layers in V2 and will not call into this module
for anything anatomical beyond a rough search rectangle.

Why MediaPipe is acceptable here
--------------------------------
The clinical model says we first locate the face and approximate eye zones, then look
inside those zones for the dark iris against the white sclera. MediaPipe FaceMesh is a
reasonable face/eye-region locator. Its iris-landmark output, however, is structurally
the wrong number for V1 limbus tracking — see EYE_VNG_DEVELOPMENT_HISTORY.md §19b.4
("MediaPipe removed from V1 clinical workflow") — so we never read it here.

Public surface
--------------
    FaceLocator().locate(frame_bgr) -> Optional[FaceLocation]

FaceLocation carries the face bbox, the L/R eye-zone bboxes, and a flag for whether
MediaPipe detected the face on this frame. All coordinates are SOURCE-VIDEO pixels.

Image-side convention (locked across this codebase): image-side L = patient anatomical
Right; image-side R = patient anatomical Left.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- constants
# FaceMesh public landmark indices. Source: MediaPipe FaceMesh canonical_face_model.
# These ring indices trace the visible eye opening (upper + lower lid margins +
# medial + lateral canthi). We compute the eye-zone rectangle from the bounding box
# of the ring on each frame, then pad it for safety. The IRIS landmarks (468 / 473)
# are intentionally NOT consumed — V2 must not let MediaPipe define the iris.
LEFT_EYE_RING_IDS = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 246, 161, 160, 159, 158]
RIGHT_EYE_RING_IDS = [263, 249, 390, 373, 374, 380, 381, 382, 255,
                      282, 286, 414, 287, 285, 284, 283]

# Padding fractions added to each eye-ring bounding box. The ring sits tightly on the
# eyelid margins; we pad outward so the V2 search region has a little headroom for
# eyelid motion and head-pose jitter without becoming a face-wide search.
EYE_ZONE_PAD_FRAC_X = 0.30
EYE_ZONE_PAD_FRAC_Y = 0.45


# --------------------------------------------------------------------------- data class
@dataclass
class FaceLocation:
    """Result of locating a face + eye zones on ONE frame. Coordinates are SOURCE-VIDEO
    pixels (x0 <= x1, y0 <= y1). `mp_detected` is True iff MediaPipe FaceMesh produced
    a face on this frame; when False, all bboxes are None."""
    mp_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]] = None     # (x0, y0, x1, y1)
    L_eye_bbox: Optional[Tuple[int, int, int, int]] = None    # image-side L
    R_eye_bbox: Optional[Tuple[int, int, int, int]] = None    # image-side R


# --------------------------------------------------------------------------- helpers
def _ring_bbox(landmarks_xy: np.ndarray, ids: List[int],
               frame_w: int, frame_h: int,
               pad_x_frac: float, pad_y_frac: float) -> Tuple[int, int, int, int]:
    """Return an axis-aligned bbox around the requested landmark IDs, padded outward
    by `pad_*_frac` of the bbox size. Clamped to the frame."""
    pts = landmarks_xy[ids]
    x0 = float(pts[:, 0].min()); x1 = float(pts[:, 0].max())
    y0 = float(pts[:, 1].min()); y1 = float(pts[:, 1].max())
    w = x1 - x0
    h = y1 - y0
    px = pad_x_frac * w
    py = pad_y_frac * h
    x0 = int(round(max(0, x0 - px)))
    y0 = int(round(max(0, y0 - py)))
    x1 = int(round(min(frame_w - 1, x1 + px)))
    y1 = int(round(min(frame_h - 1, y1 + py)))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return (x0, y0, x1, y1)


def _face_bbox_from_all_landmarks(landmarks_xy: np.ndarray,
                                    frame_w: int, frame_h: int,
                                    pad_frac: float = 0.05) -> Tuple[int, int, int, int]:
    """Face bbox = a small pad around the convex extent of ALL FaceMesh landmarks.
    A tiny pad (5 %) is enough — FaceMesh covers the whole face."""
    xs = landmarks_xy[:, 0]
    ys = landmarks_xy[:, 1]
    x0 = float(xs.min()); x1 = float(xs.max())
    y0 = float(ys.min()); y1 = float(ys.max())
    w = x1 - x0
    h = y1 - y0
    x0 = int(round(max(0, x0 - pad_frac * w)))
    y0 = int(round(max(0, y0 - pad_frac * h)))
    x1 = int(round(min(frame_w - 1, x1 + pad_frac * w)))
    y1 = int(round(min(frame_h - 1, y1 + pad_frac * h)))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return (x0, y0, x1, y1)


# --------------------------------------------------------------------------- locator
class FaceLocator:
    """Stateless MediaPipe FaceMesh wrapper. Construct once per process; call
    `locate(frame_bgr)` per frame. Returns a `FaceLocation`.

    The MediaPipe import is performed lazily so that scripts that never construct a
    FaceLocator do not need MediaPipe installed.
    """

    def __init__(self,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(
                "MediaPipe is not installed in this environment. The V2 face_locator "
                "needs `mediapipe` for face/eye-zone localisation. Install it in the "
                "project venv before running the V2 probe."
            ) from e
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
            raise RuntimeError(
                "The installed MediaPipe runtime does not expose `solutions.face_mesh`. "
                "Update MediaPipe in the project venv."
            )
        self._fm = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=False,        # we don't need iris refinement (we never read iris)
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            static_image_mode=bool(static_image_mode),
        )

    def locate(self, frame_bgr) -> FaceLocation:
        """Process one BGR frame and return a FaceLocation. If MediaPipe does not
        find a face this frame, `mp_detected=False` is returned and all bboxes are
        None."""
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceLocation(mp_detected=False)
        h, w = frame_bgr.shape[:2]
        # MediaPipe expects RGB.
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._fm.process(rgb)
        if not res.multi_face_landmarks:
            return FaceLocation(mp_detected=False)
        lm = res.multi_face_landmarks[0].landmark
        # Convert all 468 landmarks to (x, y) pixel coordinates once.
        xy = np.array([[lm[i].x * w, lm[i].y * h] for i in range(len(lm))],
                       dtype=np.float32)
        face_bbox = _face_bbox_from_all_landmarks(xy, w, h)
        L_bbox = _ring_bbox(xy, LEFT_EYE_RING_IDS, w, h,
                             EYE_ZONE_PAD_FRAC_X, EYE_ZONE_PAD_FRAC_Y)
        R_bbox = _ring_bbox(xy, RIGHT_EYE_RING_IDS, w, h,
                             EYE_ZONE_PAD_FRAC_X, EYE_ZONE_PAD_FRAC_Y)
        return FaceLocation(mp_detected=True,
                             face_bbox=face_bbox,
                             L_eye_bbox=L_bbox,
                             R_eye_bbox=R_bbox)

    def close(self) -> None:
        """Release the MediaPipe runtime. Optional — Python GC also handles it."""
        try:
            self._fm.close()
        except Exception:
            pass
