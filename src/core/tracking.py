from typing import Dict

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe iris-centre landmarks (require refine_landmarks=True).
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Eye-contour landmark rings, used only to score detection plausibility.
LEFT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 246, 161, 160, 159, 158]
RIGHT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 255, 282, 286, 414, 287, 285, 284, 283]


def mesh_width_frac(points, width: int) -> float:
    """Fraction of the frame width spanned by the landmark bounding box.

    Used as the collapse-detection signal: a healthy frontal face spans most of
    the width, a collapsed mesh bunches into a small region.
    """
    if not points or width <= 0:
        return 0.0
    xs = [p[0] for p in points]
    return (max(xs) - min(xs)) / float(width)


def _eye_confidence(points, ring, iris_x, iris_y) -> float:
    """Plausibility score in [0, 1] for one eye's mesh detection.

    Derived (NOT a model probability): the iris centre should sit inside the
    eye-contour bounding box, and the eye's height/width aspect should look like
    an open eye. Mirrors the approach in eye_movement_app._eye_metrics.
    """
    pts = np.array([points[i] for i in ring if i < len(points)], dtype=np.float32)
    if len(pts) < 4 or iris_x is None:
        return 0.0
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    eye_w = max(float(x_max - x_min), 1.0)
    eye_h = max(float(y_max - y_min), 1.0)
    inside = (x_min <= iris_x <= x_max) and (y_min <= iris_y <= y_max)
    aspect_score = float(np.clip((eye_h / eye_w - 0.08) / 0.25, 0.0, 1.0))
    return float(np.clip((1.0 if inside else 0.3) * (0.5 + 0.5 * aspect_score), 0.0, 1.0))


class EyeTrackingProcessor:
    """Mode A detector: MediaPipe Face Mesh (requires a substantially whole face).

    A bbox sanity gate rejects the "mesh collapse" failure seen on extreme eye
    close-ups, where MediaPipe returns landmarks bunched onto a tiny region. When
    a frame is rejected (or no face is found), all coordinates stay ``None`` and
    ``detection_mode`` is set so a face-independent detector can take over.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_mesh_frac: float = 0.20,
    ) -> None:
        # Loose floor: reject only a grossly degenerate landmark bbox (spans less
        # than this fraction of the frame width). An absolute width gate cannot
        # distinguish a small/distant-but-valid face from a collapsed mesh, so it
        # is kept permissive (genuine faces in test clips spanned as little as
        # ~0.35); real face-absence is handled by MediaPipe returning no face, and
        # dubious detections are surfaced by the per-eye confidence + the
        # verification overlay rather than hard-rejected here. A stricter,
        # distance-invariant collapse check belongs with the (deferred) IR mode.
        self.min_mesh_frac = min_mesh_frac
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _empty_output(mode: str = "none") -> Dict:
        return {
            "face_detected": False,
            "face_center_x": None,
            "face_center_y": None,
            "left_eye_x": None,
            "left_eye_y": None,
            "right_eye_x": None,
            "right_eye_y": None,
            "mean_eye_x": None,
            "mean_eye_y": None,
            "tracking_confidence": 0.0,
            "left_conf": 0.0,
            "right_conf": 0.0,
            "detection_mode": mode,
            "landmarks": None,
            "iris_centers": None,
        }

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return self._empty_output("none")

        landmark_list = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list]
        if not points:
            return self._empty_output("none")

        # --- sanity gate: reject the collapsed-mesh failure ---
        if mesh_width_frac(points, w) < self.min_mesh_frac:
            return self._empty_output("rejected_collapse")

        output = self._empty_output("face_mesh")

        face_center = np.mean(np.array(points[:20], dtype=np.float32), axis=0)
        output["face_detected"] = True
        output["face_center_x"] = float(face_center[0])
        output["face_center_y"] = float(face_center[1])

        if len(points) > RIGHT_IRIS_CENTER:
            lx, ly = points[LEFT_IRIS_CENTER]
            rx, ry = points[RIGHT_IRIS_CENTER]
            output["left_eye_x"] = float(lx)
            output["left_eye_y"] = float(ly)
            output["right_eye_x"] = float(rx)
            output["right_eye_y"] = float(ry)

        if output["left_eye_x"] is not None and output["right_eye_x"] is not None:
            output["mean_eye_x"] = float((output["left_eye_x"] + output["right_eye_x"]) / 2.0)
            output["mean_eye_y"] = float((output["left_eye_y"] + output["right_eye_y"]) / 2.0)

        output["left_conf"] = _eye_confidence(points, LEFT_EYE_RING, output["left_eye_x"], output["left_eye_y"])
        output["right_conf"] = _eye_confidence(points, RIGHT_EYE_RING, output["right_eye_x"], output["right_eye_y"])
        output["tracking_confidence"] = float((output["left_conf"] + output["right_conf"]) / 2.0)

        output["landmarks"] = points
        output["iris_centers"] = {
            "left": (output["left_eye_x"], output["left_eye_y"]),
            "right": (output["right_eye_x"], output["right_eye_y"]),
        }
        return output
