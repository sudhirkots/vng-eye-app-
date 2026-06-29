"""Eye-opening contour reference tracker for EyeVNG V1.

The contour is the moving reference frame. It is tracked as one shape from local
features along/near the palpebral fissure, then medial/lateral/upper/lower
reference values are derived from the contour geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class EyeOpeningContour:
    eye: str
    points: np.ndarray
    confidence: float = 1.0
    reference_state: str = "tracked"
    n_features: int = 0
    n_inliers: int = 0
    residual_px: float = 0.0

    @property
    def reference_valid(self) -> bool:
        return self.reference_state == "tracked" and self.confidence >= 0.35

    @property
    def reference_uncertain(self) -> bool:
        return not self.reference_valid

    def extents(self) -> Dict[str, Point]:
        pts = np.asarray(self.points, np.float32)
        left = pts[int(np.argmin(pts[:, 0]))]
        right = pts[int(np.argmax(pts[:, 0]))]
        upper = pts[int(np.argmin(pts[:, 1]))]
        lower = pts[int(np.argmax(pts[:, 1]))]
        return {
            "medial": (float(left[0]), float(left[1])),
            "lateral": (float(right[0]), float(right[1])),
            "upper": (float(upper[0]), float(upper[1])),
            "lower": (float(lower[0]), float(lower[1])),
        }

    def eye_local(self, iris_center: Optional[Point]) -> Tuple[Optional[float], Optional[float]]:
        if iris_center is None or self.reference_uncertain:
            return None, None
        ex = self.extents()
        w = ex["lateral"][0] - ex["medial"][0]
        h = ex["lower"][1] - ex["upper"][1]
        if abs(w) < 1.0 or abs(h) < 1.0:
            return None, None
        return ((iris_center[0] - ex["medial"][0]) / w,
                (iris_center[1] - ex["upper"][1]) / h)


def contour_from_approved(approved: dict, eye: str,
                          iris: Optional[Tuple[float, float, float]] = None,
                          allow_legacy_fallback: bool = False) -> Optional[np.ndarray]:
    """Load the V1 eye-opening / orbital margin contour for ONE eye from the approved file.

    V1 RULE (locked 2026-06-27): there is no silent fallback. The approved file must contain
    a clinician-approved `eye_opening_contours` block. Without that block this function
    returns None; the V1 runtime gate at `iris_tracker.run()` rejects approvals that don't
    have it, so this branch is normally unreachable in production.

    `allow_legacy_fallback=True` is provided ONLY for development/debug tooling (e.g. a
    one-off inspection script). It permits the old fallbacks — singular `eye_opening_contour`
    key, four-landmark ellipse, iris-shaped fallback — and is NEVER used by the V1 runtime.
    """
    contours = approved.get("eye_opening_contours") or {}
    if not isinstance(contours, dict):
        contours = {}
    raw = contours.get(eye)
    if raw:
        pts = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in raw]
        if len(pts) >= 6:
            return np.asarray(pts, np.float32)

    if not allow_legacy_fallback:
        # V1 runtime path: no silent fallback. Caller (run() / V1 schema gate) must reject the
        # approval if we reach here. Returning None forces that to happen at construction time.
        return None

    # ---- legacy fallback path (debug tooling only) ------------------------------
    legacy_contours = approved.get("eye_opening_contour") or {}
    if isinstance(legacy_contours, dict):
        raw = legacy_contours.get(eye)
        if raw:
            pts = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in raw]
            if len(pts) >= 6:
                return np.asarray(pts, np.float32)
    lm = approved.get("face_landmarks") or {}
    names = {
        "inner": f"inner_canthus_{eye}",
        "outer": f"outer_canthus_{eye}",
        "upper": f"upper_margin_{eye}",
        "lower": f"lower_margin_{eye}",
    }
    if not all(lm.get(n) for n in names.values()):
        if iris is None:
            return None
        cx, cy, r = iris
        return _ellipse_contour((cx, cy), 2.8 * r, 1.25 * r, n=48)
    inner = _as_point(lm[names["inner"]])
    outer = _as_point(lm[names["outer"]])
    upper = _as_point(lm[names["upper"]])
    lower = _as_point(lm[names["lower"]])
    centre = ((inner[0] + outer[0] + upper[0] + lower[0]) / 4.0,
              (inner[1] + outer[1] + upper[1] + lower[1]) / 4.0)
    width = max(10.0, float(np.hypot(outer[0] - inner[0], outer[1] - inner[1])))
    height = max(8.0, float(np.hypot(lower[0] - upper[0], lower[1] - upper[1])))
    angle = float(np.arctan2(outer[1] - inner[1], outer[0] - inner[0]))
    return _ellipse_contour(centre, width, height, angle=angle, n=64)


class EyeOpeningContourTracker:
    def __init__(self, eye: str, approved_points: Iterable[Point], init_gray, params=None):
        self.eye = eye
        self.points = np.asarray(list(approved_points), np.float32)
        self.prev_gray = init_gray
        self.state = EyeOpeningContour(eye, self.points.copy(), 1.0, "tracked")
        self.params = params or {
            "band_px": 16,
            "max_features": 80,
            "min_features": 8,
            "max_residual_px": 6.0,
        }
        self.features = self._detect(init_gray, self.points)

    def step(self, gray) -> EyeOpeningContour:
        if self.prev_gray is None or self.features is None or len(self.features) < self.params["min_features"]:
            self.features = self._detect(gray, self.points)
            conf = 0.25 if self.features is None else min(0.45, len(self.features) / self.params["max_features"])
            self.state = EyeOpeningContour(self.eye, self.points.copy(), conf, "uncertain",
                                           0 if self.features is None else len(self.features), 0, 0.0)
            self.prev_gray = gray
            return self.state

        p0 = self.features.reshape(-1, 1, 2).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **_LK)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **_LK)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & np.isfinite(fb) & (fb <= 2.5)
        if ok.sum() < self.params["min_features"]:
            self.features = self._detect(gray, self.points)
            self.state = EyeOpeningContour(self.eye, self.points.copy(), 0.2, "lost",
                                           int(ok.sum()), int(ok.sum()), 0.0)
            self.prev_gray = gray
            return self.state

        old = p0.reshape(-1, 2)[ok]
        new = p1.reshape(-1, 2)[ok]
        disp = new - old
        med = np.median(disp, axis=0)
        resid = np.linalg.norm(disp - med, axis=1)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
        inl = resid <= max(self.params["max_residual_px"], np.median(resid) + 3.0 * mad)
        if inl.sum() < self.params["min_features"]:
            self.features = self._detect(gray, self.points)
            self.state = EyeOpeningContour(self.eye, self.points.copy(), 0.25, "uncertain",
                                           int(ok.sum()), int(inl.sum()), float(np.median(resid)))
            self.prev_gray = gray
            return self.state

        med = np.median(disp[inl], axis=0)
        residual = float(np.median(np.linalg.norm(disp[inl] - med, axis=1)))
        self.points = self.points + med
        kept = new[inl]
        fresh = self._detect(gray, self.points, exclude=kept)
        self.features = _merge_features(kept, fresh, self.params["max_features"])
        inlier_frac = float(inl.sum() / max(1, len(ok)))
        residual_score = max(0.0, 1.0 - residual / self.params["max_residual_px"])
        count_score = min(1.0, len(kept) / (2.0 * self.params["min_features"]))
        conf = float(0.45 * inlier_frac + 0.35 * residual_score + 0.20 * count_score)
        ref_state = "tracked" if conf >= 0.35 else "uncertain"
        self.state = EyeOpeningContour(self.eye, self.points.copy(), conf, ref_state,
                                       int(ok.sum()), int(inl.sum()), residual)
        self.prev_gray = gray
        return self.state

    def _detect(self, gray, points, exclude=None):
        mask = _contour_band_mask(gray.shape, points, self.params["band_px"])
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.params["max_features"], qualityLevel=0.01,
            minDistance=4, mask=mask, blockSize=5, useHarrisDetector=False)
        if corners is None:
            return np.empty((0, 2), np.float32)
        pts = corners.reshape(-1, 2).astype(np.float32)
        if exclude is not None and len(exclude):
            keep = []
            for p in pts:
                d = np.min(np.linalg.norm(exclude - p, axis=1))
                if d >= 4.0:
                    keep.append(p)
            pts = np.asarray(keep, np.float32) if keep else np.empty((0, 2), np.float32)
        return pts


def _as_point(v) -> Point:
    return float(v["x"]), float(v["y"])


def _ellipse_contour(centre, width, height, angle=0.0, n=64):
    ts = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ca, sa = np.cos(angle), np.sin(angle)
    pts = []
    for t in ts:
        x = 0.5 * width * np.cos(t)
        y = 0.5 * height * np.sin(t)
        pts.append((centre[0] + ca * x - sa * y, centre[1] + sa * x + ca * y))
    return np.asarray(pts, np.float32)


def _contour_band_mask(shape, points, band_px):
    mask = np.zeros(shape, np.uint8)
    pts = np.asarray(points, np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=max(3, int(2 * band_px)))
    return mask


def _merge_features(kept, fresh, max_features):
    if fresh is None or len(fresh) == 0:
        return kept[:max_features].astype(np.float32)
    pts = list(kept.astype(np.float32))
    for p in fresh:
        if len(pts) >= max_features:
            break
        if not pts or min(np.linalg.norm(np.asarray(pts) - p, axis=1)) >= 4.0:
            pts.append(p.astype(np.float32))
    return np.asarray(pts, np.float32)


_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
