from typing import Dict, List, Optional

import numpy as np


class HeadPoseEstimator:
    def estimate_pose(self, landmarks: Optional[List[tuple]]) -> Dict:
        pose = {"head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0}
        if not landmarks or len(landmarks) < 3:
            return pose

        points = np.array(landmarks, dtype=np.float32)
        left = points[133] if len(points) > 133 else points[0]
        right = points[362] if len(points) > 362 else points[0]
        nose = points[1] if len(points) > 1 else points[0]
        chin = points[152] if len(points) > 152 else points[0]

        if np.linalg.norm(right - left) > 0:
            pose["head_yaw"] = float(np.degrees(np.arctan2(nose[1] - chin[1], right[0] - left[0])))

        if np.linalg.norm(chin - nose) > 0:
            pose["head_pitch"] = float(np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0])))

        roll_vector = right - left
        if np.linalg.norm(roll_vector) > 0:
            pose["head_roll"] = float(np.degrees(np.arctan2(roll_vector[1], roll_vector[0])))

        return pose


_default_estimator = HeadPoseEstimator()


def estimate_head_pose(frame, landmarks: Optional[List[tuple]]) -> Dict:
    return _default_estimator.estimate_pose(landmarks)
