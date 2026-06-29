"""Anatomically-constrained iris–sclera boundary tracker (EyeVNG `--engine anatomical`).

Per TRACKER_REDESIGN.md (2026-06-28). The clinical observation is that in Indian eyes
the iris is a dark structure against a bright sclera; even at extreme gaze the visible
iris–sclera arc remains a dark curve against sclera. This tracker uses that arc and
infers the full iris circle, subject to hard anatomical constraints.

Principle:
    The iris tracker must fail safely, not wander.
    A missing iris is better than a wrong iris on the face.

Hard rules implemented here:
 1. Stage 0 is mandatory. The orchestrator (iris_tracker.run) already refuses to track
    without `approved_landmarks.json`. This tracker additionally refuses to construct
    without an eye-opening contour polygon for its eye.
 2. Edge evidence must lie inside the Stage-0 eye-opening contour. Enforced by passing
    the polygon to `limbus.fit_limbus(polygon=...)` — every accepted radial-edge pixel
    is inside-polygon-tested.
 3. Radius is bounded around Stage-0 + previous-frame radius (radius band stays close
    to the clinician's approved value).
 4. The previous-frame centre is the only search prior. No whole-face fallback.
 5. Confidence is a weighted sum of: arc coverage, inlier fraction, locality, radius
    stability vs previous, radius stability vs Stage-0.
 6. Confidence < CONF_FREEZE freezes the last good iris and emits status STALE.
    Confidence in [CONF_FREEZE, CONF_OK) accepts the fit but emits status LOW_CONF.
    Confidence >= CONF_OK is status OK.
 7. Rescue anchor for this frame, if present, wins absolutely. Anchor centre/radius
    become the new "last good" and seed the next frame's search.
 8. Per-frame debug log includes candidate count, score components, arc support, radius
    deviation, distance from previous, containment, final confidence, and the reject
    reason if any.

This tracker emits the same `FrameMeasurement` shape as V1Tracker so it drops into the
existing `cft["L"] / cft["R"]` slot in iris_tracker.run().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.composite_tracker import (
    FrameMeasurement, project_eye_local,
    INITIALIZING, TRACKING, PARTIAL_OCCLUSION, TRACK_LOST,
)
from src.core.limbus import fit_limbus, LimbusFit


# --------------------------------------------------------------------------- params
class AnatomicalParams:
    # Confidence thresholds. Values from TRACKER_REDESIGN.md §11.
    CONF_OK = 0.55
    CONF_FREEZE = 0.30

    # Score weights (sum = 1.0).
    W_ARC = 0.30
    W_INLFR = 0.25
    W_DIST = 0.20
    W_RAD = 0.15
    W_STAGE = 0.10

    # Radius band — candidate radius must be within ±R_BAND_FRAC of Stage-0
    # (per clinician spec 2026-06-28: ±20%). Tighter than the 0.4–1.8 r_est band
    # inside the limbus fitter itself.
    R_BAND_FRAC = 0.20
    # Legacy names (kept for any external readers) — derived from R_BAND_FRAC.
    R_MIN_FRAC = 0.80
    R_MAX_FRAC = 1.20

    # Jump caps (per clinician spec 2026-06-28). The candidate centre must not
    # move more than MAX_NORMAL_JUMP_FRAC × Stage-0 radius from last_good. Even
    # at the peak of a fast nystagmus phase the iris does not move >0.75 × r in
    # one frame, so MAX_FAST_PHASE_JUMP_FRAC = 0.75 is the absolute ceiling.
    # The "normal" cap is what step() enforces today; the "fast phase" cap is
    # reserved for a future "I am inside a confirmed fast phase" gate.
    MAX_NORMAL_JUMP_FRAC = 0.35
    MAX_FAST_PHASE_JUMP_FRAC = 0.75

    # FIXED-RADIUS MODE: disabled by default 2026-06-28 because auto-locking the
    # radius after 10 OK frames caused the frame-49 freeze cascade — once
    # r_fixed was on, any partial-occlusion frame produced no valid fit and the
    # tracker froze permanently. The constant is kept (set very high) so the
    # condition `consecutive_ok >= R_FIXED_AFTER_N_OK` is effectively never true.
    R_FIXED_AFTER_N_OK = 10_000

    # Locality decay scale (px) for the W_DIST term: dist_score = exp(-||dc|| / scale).
    # scale = r_prev so a centre move of one iris radius drops dist_score to 1/e.
    # (Defined in code, not a constant — depends on prev radius.)


@dataclass
class _AnatomicalDebug:
    """One frame's debug record. Used both for stdout logs and for return-from-step."""
    frame_no: int
    n_candidates: int
    selected_centre: Optional[Tuple[float, float]]
    selected_radius: Optional[float]
    arc_coverage: float
    inlier_fraction: float
    dist_from_prev: float
    radius_dev_prev: float
    radius_dev_stage0: float
    containment_fraction: float
    confidence: float
    status: str            # one of: tracked, partial_arc_tracked, frozen_last_good, tracking_failed, rescue_anchor
    reject_reason: str     # "" if accepted

    def as_log_line(self, eye: str) -> str:
        c = self.selected_centre
        r = self.selected_radius
        cstr = f"({c[0]:.1f},{c[1]:.1f})" if c is not None else "—"
        rstr = f"{r:.1f}" if r is not None else "—"
        return (f"[anatomical][{eye}] fr={self.frame_no:5d} n_cand={self.n_candidates} "
                f"c={cstr} r={rstr} arc={self.arc_coverage:.2f} inl={self.inlier_fraction:.2f} "
                f"d_prev={self.dist_from_prev:.1f} dr_prev={self.radius_dev_prev:+.1f} "
                f"dr_st0={self.radius_dev_stage0:+.1f} contain={self.containment_fraction:.2f} "
                f"conf={self.confidence:.2f} status={self.status}"
                + (f" reason={self.reject_reason}" if self.reject_reason else ""))


# --------------------------------------------------------------------------- helpers
def _polygon_from_contour(contour_dict_or_polygon) -> Optional[np.ndarray]:
    """Accept either a list of {'x','y'} points (the schema in approved_landmarks.json)
    or a (N,2) array. Return a (N,1,2) float32 polygon suitable for cv2.pointPolygonTest."""
    if contour_dict_or_polygon is None:
        return None
    if isinstance(contour_dict_or_polygon, np.ndarray):
        poly = contour_dict_or_polygon.astype(np.float32)
    else:
        try:
            poly = np.array([[float(p["x"]), float(p["y"])]
                             for p in contour_dict_or_polygon], np.float32)
        except (KeyError, TypeError, ValueError):
            return None
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None
    return poly.reshape(-1, 1, 2)


def _containment_fraction(cx: float, cy: float, r: float,
                           polygon: np.ndarray, n_samples: int = 24) -> float:
    """Fraction of n_samples points on the fitted circle that lie inside the polygon.
    Reported for debug only — the tracker does NOT require the full circle to be
    contained (per the clinician's rule: only the VISIBLE ARC must be inside)."""
    if polygon is None or r <= 0:
        return 0.0
    angs = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = cx + r * np.cos(angs)
    ys = cy + r * np.sin(angs)
    inside = 0
    import cv2 as _cv2
    for x, y in zip(xs, ys):
        if _cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
            inside += 1
    return inside / n_samples


def _confidence(arc_coverage: float, inlier_fraction: float,
                 dist_px: float, r_prev: float, r: float, r_stage0: float) -> Tuple[float, dict]:
    """Combine signals into a single [0,1] confidence + return per-term contributions."""
    P = AnatomicalParams
    # Locality: 1.0 when dist=0, drops to 1/e at dist = r_prev.
    dist_score = float(np.exp(-dist_px / max(1.0, r_prev)))
    # Radius stability vs previous: tight band (1/e at 15% deviation).
    rad_score_prev = float(np.exp(-abs(r - r_prev) / max(1.0, 0.15 * r_prev)))
    # Radius stability vs Stage-0: looser band (1/e at 20%).
    rad_score_st0 = float(np.exp(-abs(r - r_stage0) / max(1.0, 0.20 * r_stage0)))
    score = (P.W_ARC * arc_coverage
             + P.W_INLFR * inlier_fraction
             + P.W_DIST * dist_score
             + P.W_RAD * rad_score_prev
             + P.W_STAGE * rad_score_st0)
    return float(score), {
        "arc": arc_coverage, "inl": inlier_fraction,
        "dist": dist_score, "rad_prev": rad_score_prev, "rad_st0": rad_score_st0,
    }


# --------------------------------------------------------------------------- the tracker
class AnatomicalTracker:
    """Per-eye anatomical iris tracker. One instance tracks ONE eye.

    Constructor signature mirrors V1Tracker so it drops into iris_tracker.run's
    `cft["L"] / cft["R"]` slot. The eye-opening contour polygon for THIS eye must be
    supplied via `set_contour(...)` before the first `step()`. iris_tracker.run does
    this from `approved_landmarks["eye_opening_contours"][eye]["points"]`.

    Rescue anchors (teaching_anchors.json) are loaded via `set_anchors(...)`. If an
    anchor exists at a frame, it absolutely wins for that frame and becomes the new
    last-good.
    """

    def __init__(self, eye: str, approved_centre, approved_radius, init_gray,
                 init_frame_number: int, use_cft_helper: bool = False,
                 params: AnatomicalParams = AnatomicalParams):
        self.eye = eye
        self.P = params
        # Stage-0 approved iris is the reference point for radius stability.
        self.stage0_centre: Tuple[float, float] = (float(approved_centre[0]),
                                                    float(approved_centre[1]))
        self.stage0_radius: float = float(approved_radius)

        self.prev_gray = init_gray
        self.init_frame = int(init_frame_number)

        # last_good_* is the last frame the tracker confidently accepted. STALE frames
        # do NOT update last_good (intentional: prevents drift via the prior).
        self.last_good_centre: Tuple[float, float] = self.stage0_centre
        self.last_good_radius: float = self.stage0_radius
        self.last_good_frame: int = self.init_frame
        self.consecutive_ok = 0
        # 2026-06-28: freeze-recovery state. Without this the tracker froze forever once
        # the iris drifted away from the last-good prior. On every frozen frame we now
        # attempt a recovery search from multiple priors inside the Stage-0 contour
        # (last-good, Stage-0 centre, contour centroid, a sparse grid). Display still
        # stays frozen; only search strategy widens.
        self.frozen_count: int = 0

        self.state = INITIALIZING
        # `helper` exists for shape parity with V1Tracker — never used here.
        self.helper = None

        # Containment polygon for this eye (set later via set_contour).
        self._polygon: Optional[np.ndarray] = None
        # Rescue anchors: {frame_number: {"cx","cy","r"}}.
        self._anchors: Dict[int, Tuple[float, float, float]] = {}

        # Debug records by frame number — useful for tests / overlays.
        self.debug_log: Dict[int, _AnatomicalDebug] = {}

    # ---- configuration the orchestrator pushes in -----------------------------
    def set_contour(self, contour_pts) -> None:
        """Install the Stage-0 eye-opening contour polygon for this eye. Must be called
        before step(). The polygon is used to gate edge-pixel evidence inside limbus.fit."""
        self._polygon = _polygon_from_contour(contour_pts)
        if self._polygon is None:
            raise ValueError(f"AnatomicalTracker[{self.eye}]: contour polygon is missing "
                              "or has fewer than 3 points. Stage 0 must be completed first.")

    def set_anchors(self, anchors: List[dict]) -> None:
        """Install rescue anchors loaded from teaching_anchors.json. Each anchor entry
        looks like {frame_number, image_side_eye, corrected_iris_centre:[x,y], corrected_iris_radius}.
        Anchors for the WRONG eye are skipped."""
        out: Dict[int, Tuple[float, float, float]] = {}
        for a in anchors or []:
            side = a.get("image_side_eye") or a.get("eye_image_side")
            if side != self.eye:
                continue
            try:
                fr = int(a["frame_number"])
                cxy = a.get("corrected_iris_centre") or [a.get("iris_centre_x"),
                                                          a.get("iris_centre_y")]
                cr = float(a.get("corrected_iris_radius", a.get("iris_radius", self.stage0_radius)))
                out[fr] = (float(cxy[0]), float(cxy[1]), cr)
            except (KeyError, TypeError, ValueError):
                continue
        self._anchors = out

    # ---- the per-frame step ---------------------------------------------------
    def step(self, gray, frame_no, timestamp_ms, time_sec, ear, aperture, mp_iris=None,
             bgr=None, reference_state="lost", reference_confidence=0.0) -> FrameMeasurement:
        """Advance one frame. Mirrors V1Tracker.step's signature so iris_tracker.run can
        call it without branching. Most kwargs are accepted for interface parity and not
        used (this tracker doesn't consume MediaPipe or CFT signals)."""
        if self._polygon is None:
            raise RuntimeError(f"AnatomicalTracker[{self.eye}]: set_contour() was not "
                                "called before step(). Stage-0 contour is required.")

        fm = FrameMeasurement(frame_no, int(timestamp_ms), float(time_sec), self.state,
                              ear=ear, reference_state=reference_state,
                              reference_confidence=float(reference_confidence))

        # ---- Rescue anchor wins absolutely ------------------------------------
        if frame_no in self._anchors:
            cx, cy, r = self._anchors[frame_no]
            self._record_accepted(fm, cx, cy, r, arc_coverage=1.0,
                                   inlier_fraction=1.0, status="rescue_anchor")
            self.last_good_centre = (cx, cy)
            self.last_good_radius = r
            self.last_good_frame = frame_no
            self.consecutive_ok = max(self.consecutive_ok + 1, AnatomicalParams.R_FIXED_AFTER_N_OK)
            self.state = TRACKING
            self._debug(frame_no, n_candidates=1, centre=(cx, cy), radius=r,
                         arc_coverage=1.0, inlier_fraction=1.0,
                         dist=0.0, dr_prev=0.0,
                         dr_st0=r - self.stage0_radius,
                         containment=_containment_fraction(cx, cy, r, self._polygon),
                         conf=1.0, status="rescue_anchor", reason="")
            return fm

        # ---- LOCAL fit from previous iris (the working tracker pattern) --------
        # 2026-06-28 (correction): the previous "recovery search" widened to grid
        # priors inside the whole contour, which is exactly the rediscover-from-
        # scratch pattern the clinician spec forbids. Reverted. We now do ONE
        # local fit from the last-good prior; HARD GUARDRAILS (jump cap, radius
        # cap) run BEFORE confidence scoring. If guardrails fail, we freeze.
        # No grid search, no Stage-0 centre fallback, no contour centroid.
        # The clinician's rule: the tracker tracks; it does not rediscover.
        prev_cx, prev_cy = self.last_good_centre
        prev_r = self.last_good_radius

        # FIXED-RADIUS MODE DISABLED 2026-06-28: auto-enabling r_fixed after 10
        # consecutive OK frames was the structural bug behind the frame-49 freeze
        # cascade. Once r_fixed was on, any frame where the true iris radius
        # changed (partial occlusion, blink onset, etc.) produced an unfittable
        # circle and the tracker froze permanently. r_fixed remains None.
        r_fixed = None

        fit: Optional[LimbusFit] = fit_limbus(
            gray, prev_cx, prev_cy, prev_r,
            bgr=bgr, polygon=self._polygon, r_fixed=r_fixed,
        )

        # ---- No fit at all → freeze (display only; prior never poisoned) ------
        if fit is None:
            self._record_frozen(fm)
            self.frozen_count += 1
            self._debug(frame_no, n_candidates=0,
                         centre=None, radius=None,
                         arc_coverage=0.0, inlier_fraction=0.0,
                         dist=0.0, dr_prev=0.0, dr_st0=0.0,
                         containment=0.0, conf=0.0,
                         status="frozen_last_good",
                         reason="no_limbus_fit_inside_contour")
            return fm

        cx, cy, r = float(fit.cx), float(fit.cy), float(fit.r)
        arc = float(getattr(fit, "arc_coverage", 0.0) or 0.0)
        inl = float(getattr(fit, "inlier_fraction", 0.0) or 0.0)
        dist = float(np.hypot(cx - prev_cx, cy - prev_cy))
        dr_prev = r - prev_r
        dr_st0 = r - self.stage0_radius
        contain = _containment_fraction(cx, cy, r, self._polygon)

        # ---- HARD GUARDRAILS (run BEFORE confidence) ---------------------------
        # Per the corrected clinician spec: hard rejects first, then confidence.
        # One good arc-coverage score must NOT override a hard rule.
        P = AnatomicalParams

        # 1. Radius band — candidate radius must be within ±20% of Stage-0.
        r_lo = (1.0 - P.R_BAND_FRAC) * self.stage0_radius
        r_hi = (1.0 + P.R_BAND_FRAC) * self.stage0_radius
        if not (r_lo <= r <= r_hi):
            self._freeze_with_reason(fm, frame_no, cx, cy, r, arc, inl,
                                      dist, dr_prev, dr_st0, contain, conf=0.0,
                                      reason="radius_out_of_range")
            return fm

        # 2. Jump cap — candidate centre must not jump too far from last-good.
        # max_normal_jump = 0.35 × stage0 radius. Above that, the candidate is
        # almost certainly a wrong arc (cheek, eyelid, nose) — even at the peak
        # of a fast phase the iris does not jump >0.75×r in one frame.
        max_jump = P.MAX_NORMAL_JUMP_FRAC * self.stage0_radius
        if dist > max_jump:
            print(f"[iris-track][{self.eye}] fr={frame_no} reject=unstable_jump "
                  f"d={dist:.0f} max={max_jump:.0f} r={r:.0f} arc={arc:.2f}")
            self._freeze_with_reason(fm, frame_no, cx, cy, r, arc, inl,
                                      dist, dr_prev, dr_st0, contain, conf=0.0,
                                      reason="unstable_jump")
            return fm

        # ---- Confidence (only after hard rejects pass) ------------------------
        conf, _terms = _confidence(arc, inl, dist, prev_r, r, self.stage0_radius)

        # ---- Three-tier decision -----------------------------------------------
        if conf >= P.CONF_OK:
            status = "tracked" if arc >= 0.35 else "partial_arc_tracked"
            self._record_accepted(fm, cx, cy, r, arc, inl, status=status)
            self.last_good_centre = (cx, cy)
            self.last_good_radius = r
            self.last_good_frame = frame_no
            self.consecutive_ok += 1
            self.frozen_count = 0
            self.state = TRACKING if arc >= 0.35 else PARTIAL_OCCLUSION
            self._debug(frame_no, n_candidates=1, centre=(cx, cy), radius=r,
                         arc_coverage=arc, inlier_fraction=inl,
                         dist=dist, dr_prev=dr_prev, dr_st0=dr_st0,
                         containment=contain, conf=conf,
                         status=status, reason="")
            if status == "partial_arc_tracked":
                print(f"[iris-track][{self.eye}] fr={frame_no} "
                      f"status=partial_arc_tracked arc={arc:.2f} conf={conf:.2f} "
                      f"d={dist:.0f} r={r:.0f}")
        elif conf >= P.CONF_FREEZE:
            # low_conf — accept (last_good DOES update) but mark for downstream.
            self._record_accepted(fm, cx, cy, r, arc, inl, status="low_conf")
            self.last_good_centre = (cx, cy)
            self.last_good_radius = r
            self.last_good_frame = frame_no
            self.consecutive_ok = 0
            self.frozen_count = 0
            self.state = PARTIAL_OCCLUSION
            self._debug(frame_no, n_candidates=1, centre=(cx, cy), radius=r,
                         arc_coverage=arc, inlier_fraction=inl,
                         dist=dist, dr_prev=dr_prev, dr_st0=dr_st0,
                         containment=contain, conf=conf,
                         status="low_conf", reason="low_confidence")
        else:
            # Below CONF_FREEZE — freeze.
            reason = ("weak_iris_sclera_contrast" if arc < 0.10 else
                       "low_confidence")
            self._freeze_with_reason(fm, frame_no, cx, cy, r, arc, inl,
                                      dist, dr_prev, dr_st0, contain, conf,
                                      reason=reason)
        return fm

    def _freeze_with_reason(self, fm: FrameMeasurement, frame_no: int,
                             cx: float, cy: float, r: float,
                             arc: float, inl: float, dist: float,
                             dr_prev: float, dr_st0: float, contain: float,
                             conf: float, reason: str) -> None:
        """Common freeze path. Does NOT update last_good (so the prior cannot be
        poisoned by a bad candidate). Increments frozen_count. Records debug."""
        self._record_frozen(fm)
        self.frozen_count += 1
        self._debug(frame_no, n_candidates=1, centre=(cx, cy), radius=r,
                     arc_coverage=arc, inlier_fraction=inl,
                     dist=dist, dr_prev=dr_prev, dr_st0=dr_st0,
                     containment=contain, conf=conf,
                     status="frozen_last_good", reason=reason)

    # 2026-06-28 (correction): the multi-prior recovery search was REMOVED. It
    # widened to grid priors inside the contour and was finding eyelid / cheek
    # arcs because it had no temporal locality. Clinician's corrected rule:
    # "tracker tracks; it does not rediscover." Replacement: hard guardrails
    # (jump cap, radius cap) above, with one local fit from last-good. See git
    # history before this date if you need to compare.

    # ---- helpers -------------------------------------------------------------
    def _record_accepted(self, fm: FrameMeasurement, cx: float, cy: float, r: float,
                         arc_coverage: float, inlier_fraction: float, status: str) -> None:
        fm.iris_centre = (cx, cy)
        fm.iris_radius = r
        fm.raw_iris_centre = (cx, cy)
        fm.iris_valid = True
        fm.limbus_inlier_fraction = float(inlier_fraction)
        fm.limbus_arc_coverage = float(arc_coverage)
        # Reuse the existing frame_confidence channel so the rest of the pipeline can
        # tell good from bad without anatomical-engine-specific code.
        fm.frame_confidence = 1.0 if status in ("tracked", "rescue_anchor") else 0.5
        fm.composite_confidence = fm.frame_confidence

    def _record_frozen(self, fm: FrameMeasurement) -> None:
        # On a frozen frame we EMIT the last-good centre so the overlay has something
        # to render (with STALE styling). iris_valid is False so VNG traces drop the
        # frame and downstream segmentation sees a gap, not a drifted point.
        fm.iris_centre = self.last_good_centre
        fm.iris_radius = self.last_good_radius
        fm.raw_iris_centre = self.last_good_centre
        fm.iris_valid = False
        fm.drift_flag = True
        fm.drift_reason = "frozen_last_good"
        fm.frame_confidence = 0.0
        fm.composite_confidence = 0.0
        self.state = TRACK_LOST

    def _debug(self, frame_no, n_candidates, centre, radius, arc_coverage,
                inlier_fraction, dist, dr_prev, dr_st0, containment, conf,
                status, reason) -> None:
        rec = _AnatomicalDebug(
            frame_no=frame_no,
            n_candidates=int(n_candidates),
            selected_centre=tuple(map(float, centre)) if centre is not None else None,
            selected_radius=float(radius) if radius is not None else None,
            arc_coverage=float(arc_coverage),
            inlier_fraction=float(inlier_fraction),
            dist_from_prev=float(dist),
            radius_dev_prev=float(dr_prev),
            radius_dev_stage0=float(dr_st0),
            containment_fraction=float(containment),
            confidence=float(conf),
            status=status,
            reject_reason=reason,
        )
        self.debug_log[frame_no] = rec
        # Print only on status changes / non-OK to keep stdout readable on a 30 s clip.
        if status != "tracked":
            print(rec.as_log_line(self.eye))
