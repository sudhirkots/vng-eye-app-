"""Stable, time-continuous iris tracking (see STABLE_TRACKING.md).

Architecture: MediaPipe detection -> user confirmation of the iris CIRCLE
(centre + radius, per eye) -> CONTINUOUS TEMPLATE TRACKING, with MediaPipe used
only as a *backup* (re-detection on low confidence, blink end, or loss).
No Kalman filter, no smoothing, and no invented positions during blinks/occlusion
(coordinates are left blank).

Per frame the iris is found by normalised cross-correlation of a confirmed iris
template within a small window around the previous position (sub-pixel refined).
The correlation score is the tracking confidence. Low score = iris not visible
(blink/occlusion) -> stop, do not guess, attempt re-acquisition. One eye may be
disabled (None) for one-eye analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from src.core.tracking import _eye_confidence, LEFT_EYE_RING, RIGHT_EYE_RING

# The V1 engine (EYEVNG_TRACKING_SPECIFICATION.md) — the weighted multi-feature composite iris
# tracker. Re-exported here so it "lives in src/core/iris_tracking.py" per MULTIFEATURE_TRACKER_DESIGN
# §7, while the implementation sits in composite_tracker.py. This replaces the single-template
# IrisTracker (below, retained only for reference) as the per-frame clinical engine.
from src.core.composite_tracker import (  # noqa: F401  (re-export)
    CompositeFeatureTracker, FrameMeasurement, Composite, IrisFeature,
    PROBATION, TRUSTED, SUSPECT, LOST, VALID_STATES, project_eye_local)

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


@dataclass
class EyeObs:
    detected: bool
    x: Optional[float] = None
    y: Optional[float] = None
    conf: float = 0.0
    radius: float = 0.0              # iris (limbus) radius — the marked/tracked iris boundary
    single_x: Optional[float] = None
    single_y: Optional[float] = None
    iris_radius: Optional[float] = None   # MediaPipe iris radius (== radius; kept for template sizing)
    ear: Optional[float] = None     # eye-aspect-ratio (lid gap / eye width); low ⇒ eye closing/blink


# eyelid landmark sets for the eye-aspect-ratio (vertical lid gap / horizontal eye width).
# left eye  (iris 468-472): upper lid 386, lower lid 374, corners 362 (inner) & 263 (outer)
# right eye (iris 473-477): upper lid 159, lower lid 145, corners 133 (inner) & 33  (outer)
LEFT_EAR = (386, 374, 362, 263)
RIGHT_EAR = (159, 145, 133, 33)

# Eye-aperture landmarks: the 4 corners per eye that define the eye-local coordinate box.
#   L eye (iris 468-472): inner canthus 362, outer canthus 263, upper lid 386, lower lid 374
#   R eye (iris 473-477): inner canthus 133, outer canthus 33,  upper lid 159, lower lid 145
# "inner" = nasal corner, "outer" = temporal corner; "upper"/"lower" = lid margins.
APERTURE = {
    "inner_canthus_L": 362, "outer_canthus_L": 263, "upper_margin_L": 386, "lower_margin_L": 374,
    "inner_canthus_R": 133, "outer_canthus_R": 33,  "upper_margin_R": 159, "lower_margin_R": 145,
}


def _ear(pts, up, lo, c1, c2):
    """Eye-aspect-ratio: vertical eyelid gap divided by horizontal eye width. Position-independent
    blink signal — it reflects lid closure, not where the iris is, so flagging blinks with it never
    suppresses real eye movement (nystagmus/saccades)."""
    try:
        v = np.hypot(pts[up][0] - pts[lo][0], pts[up][1] - pts[lo][1])
        hh = np.hypot(pts[c1][0] - pts[c2][0], pts[c1][1] - pts[c2][1])
        return float(v / hh) if hh > 1e-6 else None
    except (IndexError, TypeError):
        return None


def _iris_center(points, idxs):
    pts = np.array([points[i] for i in idxs if i < len(points)], dtype=np.float32)
    if len(pts) < 5:
        return None
    c = pts.mean(axis=0)
    r = float(np.mean([np.hypot(p[0] - c[0], p[1] - c[1]) for p in pts[1:]]))
    return float(c[0]), float(c[1]), r


class IrisDetector:
    """MediaPipe iris detector (ring-mean). Used for init and as the tracking backup."""

    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed. Use --no-mediapipe with approved landmarks.") from exc
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
            raise RuntimeError("This MediaPipe runtime does not expose solutions.face_mesh. "
                               "Use --no-mediapipe with approved landmarks.")
        self.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, frame_bgr) -> Tuple[bool, EyeObs, EyeObs]:
        h, w = frame_bgr.shape[:2]
        res = self.fm.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return False, EyeObs(False), EyeObs(False)
        lm = res.multi_face_landmarks[0].landmark
        pts = [(lm[i].x * w, lm[i].y * h) for i in range(len(lm))]

        def eye(iris_idx, ring, ear_idx):
            ic = _iris_center(pts, iris_idx)
            if ic is None:
                return EyeObs(False)
            ix, iy, ir = ic          # MediaPipe iris: centre + boundary (limbus) radius — the iris proposal
            return EyeObs(True, ix, iy, _eye_confidence(pts, ring, ix, iy), ir,
                          single_x=ix, single_y=iy, iris_radius=ir,
                          ear=_ear(pts, *ear_idx))

        return True, eye(LEFT_IRIS, LEFT_EYE_RING, LEFT_EAR), eye(RIGHT_IRIS, RIGHT_EYE_RING, RIGHT_EAR)

    def face_landmarks(self, frame_bgr):
        """Propose the EYE-APERTURE landmarks for user confirmation — the 4 aperture corners per eye
        that define the eye-local coordinate box (see TRACKING_PHILOSOPHY.md). At the current clinical
        stage NO nose/cheek/tragus/face landmarks are used: the clinical trace is iris motion WITHIN
        the eye opening, which these four landmarks (all moving with the eye) fully define — so head
        translation/camera movement cancels without any face/head reference. Returns {name:(x,y)|None};
        None = landmark outside the frame (unavailable)."""
        h, w = frame_bgr.shape[:2]
        res = self.fm.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        def P(i):
            return (lm[i].x * w, lm[i].y * h)

        def avail(p):
            return None if not (0 <= p[0] < w and 0 <= p[1] < h) else p

        out = {name: P(i) for name, i in APERTURE.items()}
        return {k: avail(v) for k, v in out.items()}


def select_init_frame(detector: IrisDetector, video_path, scan_frames: int = 150,
                      good_conf: float = 0.6):
    """Return (frame_number, frame_bgr, left EyeObs, right EyeObs) for the clearest
    early frame. Prefers both eyes; falls back to the best single-eye frame so that
    one-eye analysis is possible. EyeObs.detected=False marks an unusable eye."""
    cap = cv2.VideoCapture(str(video_path))
    best_both = best_one = None
    fr = 0
    while fr < scan_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fr += 1
        face, le, re = detector.detect(frame)
        if not face:
            continue
        if le.detected and re.detected:
            s = min(le.conf, re.conf)
            if s >= good_conf and (best_both is None or s > best_both[0]):
                best_both = (s, fr, frame.copy(), le, re)
        else:
            for eye in (le, re):
                if eye.detected and eye.conf >= good_conf and (best_one is None or eye.conf > best_one[0]):
                    best_one = (eye.conf, fr, frame.copy(), le, re)
    cap.release()
    chosen = best_both or best_one
    if chosen is None:
        return None
    _, fr, frame, le, re = chosen
    return fr, frame, le, re


@dataclass
class Track:
    status: str                     # initialized|tracked|uncertain|blink_or_occluded|reacquired|lost
    x: Optional[float]              # VALID iris position — None during blink/occlusion/jump (never invented)
    y: Optional[float]
    radius: Optional[float] = None
    confidence: str = "none"        # high|medium|low|none
    score: float = 0.0              # template-match correlation
    raw_x: Optional[float] = None   # ALWAYS the detected centre (preserved even when invalid)
    raw_y: Optional[float] = None
    rejected: bool = False
    artifact_type: str = "none"     # none|blink|occlusion|motion_blur|tracking_jump|uncertain
    ear: Optional[float] = None     # eye-aspect-ratio at this frame


_SEVERITY = {"lost": 5, "blink_or_occluded": 4, "uncertain": 3,
             "reacquired": 2, "tracked": 1, "initialized": 0}


def frame_status(left: Track, right: Track) -> str:
    return max((left.status, right.status), key=lambda s: _SEVERITY.get(s, 3))


def _subpix(res, loc):
    x, y = loc
    h, w = res.shape
    fx, fy = float(x), float(y)
    if 0 < x < w - 1:
        d = res[y, x - 1] - 2 * res[y, x] + res[y, x + 1]
        if abs(d) > 1e-9:
            fx += 0.5 * (res[y, x - 1] - res[y, x + 1]) / d
    if 0 < y < h - 1:
        d = res[y - 1, x] - 2 * res[y, x] + res[y + 1, x]
        if abs(d) > 1e-9:
            fy += 0.5 * (res[y - 1, x] - res[y + 1, x]) / d
    return fx, fy


class IrisTracker:
    """Template-based continuous tracker; MediaPipe is the backup only.

    init_left/init_right are (x, y) or None (disabled eye). left_radius/right_radius
    are the user-confirmed iris radii.
    """

    def __init__(self, init_gray, init_left, init_right, left_radius, right_radius,
                 interocular, high: float = 0.47, med: float = 0.35,
                 max_jump_frac: float = 0.30, reacq_frac: float = 0.25, blink_to_lost: int = 20):
        self.templates = {}
        self.prev = {"L": tuple(init_left) if init_left else None,
                     "R": tuple(init_right) if init_right else None}
        self.last_reliable = dict(self.prev)
        self.radius = {"L": float(left_radius) if init_left else None,
                       "R": float(right_radius) if init_right else None}
        self.enabled = {"L": init_left is not None, "R": init_right is not None}
        self.lost_count = {"L": 0, "R": 0}
        self.high, self.med = high, med
        self.max_jump = max(10.0, max_jump_frac * (interocular or 100.0))
        self.reacq_win = max(30.0, reacq_frac * (interocular or 100.0))
        self.anchor = max(10.0, 0.02 * (interocular or 100.0))
        self.blink_to_lost = blink_to_lost
        self.corrections = []
        # --- blink / artifact detection state (per eye) ---
        self.ear_base = {"L": None, "R": None}   # adaptive open-eye EAR baseline
        self.ear_close_frac = 0.62               # blink when EAR < 62% of the open baseline
        self.ear_alpha = 0.04                    # slow EMA so a brief blink can't pull the baseline down
        # "impossible jump" set FAR above any real saccade/nystagmus beat (those are tens of px), so a
        # genuine fast phase is never mistaken for an artifact — only a teleport (lid/brow grab) trips it.
        self.blink_jump = max(60.0, 0.55 * (interocular or 100.0))
        if init_left:
            self._make_template(init_gray, "L", init_left, left_radius)
        if init_right:
            self._make_template(init_gray, "R", init_right, right_radius)

    # -- template ---------------------------------------------------------------
    def _make_template(self, gray, key, center, radius):
        half = max(8, int(1.3 * (radius or 12)))
        cx, cy = int(round(center[0])), int(round(center[1]))
        h, w = gray.shape
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, cx + half + 1), min(h, cy + half + 1)
        self.templates[key] = {"img": gray[y0:y1, x0:x1].copy(),
                               "cxoff": cx - x0, "cyoff": cy - y0, "half": half}
        self.radius[key] = float(radius)

    def _match(self, gray, key, center, win):
        t = self.templates.get(key)
        if t is None or center is None:
            return 0.0, None
        patch = t["img"]
        ph, pw = patch.shape
        H, W = gray.shape
        cx, cy = center
        sx0, sy0 = max(0, int(cx - win)), max(0, int(cy - win))
        sx1, sy1 = min(W, int(cx + win)), min(H, int(cy + win))
        roi = gray[sy0:sy1, sx0:sx1]
        if roi.shape[0] < ph or roi.shape[1] < pw:
            return 0.0, None
        res = cv2.matchTemplate(roi, patch, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        mx, my = _subpix(res, maxloc)
        return float(maxv), (sx0 + mx + t["cxoff"], sy0 + my + t["cyoff"])

    def _eye_obs(self, mp_detect, key) -> EyeObs:
        res = mp_detect()
        return res[1] if key == "L" else res[2]

    def _step_eye(self, key, gray, mp_detect) -> Track:
        """Track the marked IRIS as a single physical object: each frame searches LOCALLY around the
        previous iris position for the best template match — NO per-frame MediaPipe. MediaPipe is
        consulted ONLY when the local match fails (blink/occlusion/loss) to re-acquire the iris.
        This is 'stop detecting, start tracking': the marker stays attached to the same iris and does
        not jitter back toward a per-frame re-estimate."""
        if not self.enabled[key]:
            return Track("lost", None, None)
        r = self.radius[key] or 12.0
        win = max(18.0, 2.5 * r)                       # local search window (covers fast eye movement)
        prev = self.prev[key]

        # --- NORMAL TRACKING: local template match, no MediaPipe ---
        score, pos = self._match(gray, key, prev, win) if prev is not None else (0.0, None)
        if pos is not None and score >= self.med:
            reacq = self.lost_count[key] > 0
            self.lost_count[key] = 0
            self.prev[key] = pos
            self.last_reliable[key] = pos
            conf = "high" if score >= self.high else "medium"
            status = "reacquired" if reacq else ("tracked" if score >= self.high else "uncertain")
            return Track(status, pos[0], pos[1], r, conf, score,
                         raw_x=pos[0], raw_y=pos[1], artifact_type="none")

        # --- iris not confidently found locally → blink / occlusion / loss. MediaPipe ONLY to reacquire ---
        self.lost_count[key] += 1
        obs = self._eye_obs(mp_detect, key)
        last = self.last_reliable[key]
        if obs.detected and last is not None and np.hypot(obs.x - last[0], obs.y - last[1]) <= self.reacq_win:
            # verify the iris template actually matches near MediaPipe's proposal before trusting it
            s2, p2 = self._match(gray, key, (obs.x, obs.y), win)
            cx, cy = (p2 if (p2 is not None and s2 >= self.med) else (obs.x, obs.y))
            self._make_template(gray, key, (cx, cy), r)   # refresh the template at re-acquisition
            self.prev[key] = (cx, cy)
            self.last_reliable[key] = (cx, cy)
            self.lost_count[key] = 0
            return Track("reacquired", cx, cy, r, "medium", max(score, s2),
                         raw_x=cx, raw_y=cy, artifact_type="none")
        status = "blink_or_occluded" if self.lost_count[key] <= self.blink_to_lost else "lost"
        return Track(status, None, None, None, "none", score,
                     raw_x=(pos[0] if pos else None), raw_y=(pos[1] if pos else None),
                     artifact_type="occlusion")

    def _reacquire(self, key, gray, mp_detect) -> Track:
        obs = self._eye_obs(mp_detect, key)
        last = self.last_reliable[key]
        score, pos = self._match(gray, key, last, self.reacq_win) if last else (0.0, None)

        if (pos and score >= self.high and obs.detected
                and np.hypot(pos[0] - obs.x, pos[1] - obs.y) <= self.anchor
                and last and np.hypot(pos[0] - last[0], pos[1] - last[1]) <= self.reacq_win):
            self.prev[key] = pos
            self.last_reliable[key] = pos
            self.lost_count[key] = 0
            return Track("reacquired", pos[0], pos[1], self.radius[key], "high", score, obs.x, obs.y)

        if obs.detected and (last is None or
                             np.hypot(obs.x - last[0], obs.y - last[1]) <= self.reacq_win):
            self._make_template(gray, key, (obs.x, obs.y), obs.radius or self.radius[key])
            self.prev[key] = (obs.x, obs.y)
            self.last_reliable[key] = (obs.x, obs.y)
            self.lost_count[key] = 0
            return Track("reacquired", obs.x, obs.y, self.radius[key], "medium", score, obs.x, obs.y)

        self.lost_count[key] += 1
        status = "blink_or_occluded" if self.lost_count[key] <= self.blink_to_lost else "lost"
        return Track(status, None, None, None, "none", score,
                     obs.x if obs.detected else None, obs.y if obs.detected else None)

    def apply_correction(self, key, gray, x, y, radius, frame_number):
        """Record a manual iris-circle correction and rebuild the template from it."""
        self.enabled[key] = True
        self.corrections.append({"frame_number": frame_number, "eye": key,
                                 "x": round(x, 2), "y": round(y, 2), "radius": round(radius, 2)})
        self._make_template(gray, key, (x, y), radius)
        self.prev[key] = (x, y)
        self.last_reliable[key] = (x, y)
        self.lost_count[key] = 0

    def step(self, gray, mp_detect: Callable) -> Tuple[Track, Track]:
        return self._step_eye("L", gray, mp_detect), self._step_eye("R", gray, mp_detect)


class FaceLandmarkTracker:
    """Tracks the confirmed facial landmarks robustly, as a fixed OFFSET from their
    MediaPipe positions. MediaPipe re-detects these anatomical points every frame and
    keeps them locked to the anatomy through head movement (unlike a previous-position
    template, which drifts). The user's Stage-0 correction is preserved as the offset
    from the MediaPipe point at the init frame, so the tracked landmark = MediaPipe_now
    + offset. A landmark MediaPipe cannot locate is reported lost, never invented.

    landmarks: confirmed {name: {"x","y"} | (x,y) | None}; init_face_mp: the MediaPipe
    facial landmarks at the init frame {name: (x,y) | None}.
    """

    def __init__(self, landmarks, init_face_mp):
        init_face_mp = init_face_mp or {}
        self.offset, self.enabled = {}, {}
        self._names = list(landmarks.keys())
        for name, pt in landmarks.items():
            p = (pt["x"], pt["y"]) if isinstance(pt, dict) else (tuple(pt) if pt else None)
            self.enabled[name] = p is not None
            mp0 = init_face_mp.get(name)
            self.offset[name] = (p[0] - mp0[0], p[1] - mp0[1]) if (p and mp0) else (0.0, 0.0)

    def names(self):
        return self._names

    def step(self, gray, face_backup):
        cur = face_backup() or {}
        out = {}
        for name in self._names:
            if not self.enabled[name]:
                out[name] = Track("lost", None, None)
                continue
            mp_now = cur.get(name)
            if mp_now is None:
                out[name] = Track("lost", None, None)        # MediaPipe lost it → honest, not invented
            else:
                ox, oy = self.offset[name]
                out[name] = Track("tracked", mp_now[0] + ox, mp_now[1] + oy, confidence="high")
        return out
