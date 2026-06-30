"""RIT Orbit Lock — PROBE / prototype (2026-06-30).

Tracks the clinician-marked Stage-0 eye-opening oval frame-to-frame so the anatomical container moves
with the eye region instead of staying frozen at its first-frame pixels (the 'stale orbit' that the
orbit-drift diagnostic confirmed was rejecting good irises). PROBE ONLY: it does NOT feed the iris
validator or any clinical output. One OrbitLock per eye, tracked independently.

Method (deliberately chosen over per-contour-point KLT, which scatters on low-texture lid margins):
track many eye-region features with pyramidal Lucas-Kanade optical flow, fit ONE robust similarity
transform (translation + mild rotation + mild scale via RANSAC) and apply it to the whole contour as
a rigid-ish body. The central iris disc is MASKED OUT when selecting features, because the iris moves
with GAZE (nystagmus fast phases) while the eye OPENING moves with the HEAD — the orbit must follow
the head, not the jerking iris. Per-frame scale/rotation/translation guards reject implausible jumps
(onto brow/cheek/forehead) and flag the frame uncertain/lost instead of applying a bad transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OrbitLockResult:
    status: str            # orbit_locked | orbit_uncertain | orbit_lost
    confidence: float      # inlier fraction, 0..1
    n_inliers: int
    scale: float           # per-frame incremental scale (1.0 = unchanged)
    rotation_deg: float    # per-frame incremental rotation
    contour: np.ndarray    # current contour (N,2) float32


class OrbitLock:
    MIN_FEATURES = 8
    MIN_INLIERS = 6
    SCALE_LO, SCALE_HI = 0.90, 1.11        # per-frame scale must stay near 1
    ROT_MAX_DEG = 8.0                       # per-frame rotation cap
    TRANS_MAX_FRAC = 0.35                   # per-frame centroid move cap, fraction of box diagonal
    LOST_AFTER = 8                          # consecutive bad frames -> orbit_lost

    def __init__(self, contour_xy, iris_radius, margin=0.30):
        self.contour = np.asarray(contour_xy, np.float32).reshape(-1, 2)
        self.iris_r = float(iris_radius or 30.0)
        self.margin = float(margin)
        self.bad = 0

    def centroid(self):
        return float(self.contour[:, 0].mean()), float(self.contour[:, 1].mean())

    def _box(self):
        x0, y0 = self.contour.min(0)
        x1, y1 = self.contour.max(0)
        mx, my = self.margin * (x1 - x0), self.margin * (y1 - y0)
        return x0 - mx, y0 - my, x1 + mx, y1 + my

    def step(self, prev_gray, cur_gray, iris_hint=None) -> OrbitLockResult:
        """Advance the orbit one frame using flow between prev_gray and cur_gray. iris_hint = current
        raw iris centre (px) to mask out gaze-borne features; falls back to the contour centroid."""
        H, W = prev_gray.shape[:2]
        x0, y0, x1, y1 = self._box()
        ix0, iy0 = max(0, int(x0)), max(0, int(y0))
        ix1, iy1 = min(W, int(x1)), min(H, int(y1))
        cx, cy = self.centroid()
        hint = iris_hint if (iris_hint and iris_hint[0] is not None) else (cx, cy)

        def bail():
            self.bad += 1
            st = "orbit_lost" if self.bad >= self.LOST_AFTER else "orbit_uncertain"
            return OrbitLockResult(st, 0.0, 0, 1.0, 0.0, self.contour.copy())

        if ix1 - ix0 < 8 or iy1 - iy0 < 8:
            return bail()

        # features INSIDE the orbit box but OUTSIDE the iris disc (track head/lids, not gaze)
        mask = np.zeros((H, W), np.uint8)
        mask[iy0:iy1, ix0:ix1] = 255
        cv2.circle(mask, (int(hint[0]), int(hint[1])), int(1.15 * self.iris_r), 0, -1)
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.01, minDistance=5, mask=mask)
        if p0 is None or len(p0) < self.MIN_FEATURES:
            return bail()

        p1, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                              winSize=(21, 21), maxLevel=3)
        if p1 is None or stt is None:
            return bail()
        good = stt.reshape(-1) == 1
        a0, a1 = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if len(a0) < self.MIN_INLIERS:
            return bail()

        M, inl = cv2.estimateAffinePartial2D(a0, a1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inl is None:
            return bail()
        n_inl = int(inl.sum())
        conf = n_inl / float(len(a0))
        sc = float(np.hypot(M[0, 0], M[1, 0]))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        nc = cv2.transform(np.array([[[cx, cy]]], np.float32), M).reshape(2)
        trans = float(np.hypot(nc[0] - cx, nc[1] - cy))
        boxdiag = float(np.hypot(x1 - x0, y1 - y0)) or 1.0

        ok = (n_inl >= self.MIN_INLIERS and self.SCALE_LO <= sc <= self.SCALE_HI
              and abs(rot) <= self.ROT_MAX_DEG and trans <= self.TRANS_MAX_FRAC * boxdiag)
        if not ok:
            self.bad += 1
            st = "orbit_lost" if self.bad >= self.LOST_AFTER else "orbit_uncertain"
            return OrbitLockResult(st, conf, n_inl, sc, rot, self.contour.copy())   # keep last good contour

        self.contour = cv2.transform(self.contour.reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.float32)
        self.bad = 0
        return OrbitLockResult("orbit_locked", conf, n_inl, sc, rot, self.contour.copy())
