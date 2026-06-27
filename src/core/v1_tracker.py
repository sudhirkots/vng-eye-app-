"""EyeVNG V1 tracking orchestrator — limbus + eye-opening contour.

This is the V1 clinical pipeline per EYEVNG_TRACKING_SPECIFICATION.md §A–F. The
hierarchy is fixed in code:

  1. Limbus / iris-boundary fit  →  PRIMARY clinical tracker. The estimated
     iris circle/ellipse fitted from the visible limbus arc IS the clinical
     object; its centre IS the clinical iris centre.
  2. Eye-opening contour          →  moving reference frame. Owned by the
     orchestrator's caller (an EyeOpeningContourTracker); medial/lateral/upper/
     lower extents are derived from the contour geometry, never tracked as four
     independent points.
  3. CFT / internal iris features →  OPTIONAL helper. Provides a motion prior
     for the limbus search and a consistency check. NEVER the clinical centre
     and NEVER a validity gate — a CFT quorum failure cannot invalidate a
     frame when the limbus fit and contour reference are healthy.
  4. MediaPipe                    →  optional init/recovery helper only.
     Never required, never on the per-frame clinical path.

Pupil tracking, pupil darkness, and pupil-centre terminology do not exist here.

The orchestrator emits the same `FrameMeasurement` dataclass that the rest of
the pipeline (CSV writer, overlay, segments.py) already consumes, so swapping
engines is a one-line change in iris_tracker.run().
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.core.composite_tracker import (
    FrameMeasurement, CompositeFeatureTracker, project_eye_local,
    INITIALIZING, TRACKING, PARTIAL_OCCLUSION, BLINK, DRIFT_SUSPECTED,
    TRACK_LOST, REACQUIRING, VALID_STATES,
)
from src.core.limbus import fit_limbus, _fit_centre_fixed_r, _arc_coverage, LimbusFit


# --------------------------------------------------------------------------- §J parameters
class V1Params:
    # limbus fit gates — drift is loss of anatomical attachment (§F).
    LIMBUS_MIN_FRAC = 0.45            # min RANSAC inlier fraction for a usable fit
    LIMBUS_COV_MIN = 0.18             # arc coverage below this ⇒ poor_arc_coverage
    LIMBUS_COV_GOOD = 0.45            # arc coverage above this ⇒ refresh stable radius
    LIMBUS_R_SMOOTH = 0.20            # EMA on the stable iris radius (well-seen frames only)
    LIMBUS_DISAGREE_RES_FRAC = 0.20   # median fit-residual cap as fraction of iris radius

    # temporal continuity — a jump unsupported by the image is drift (§F).
    CENTRE_JUMP_FRAC = 0.60           # max |Δcentre| per frame as fraction of iris radius
    RADIUS_JUMP_FRAC = 0.30           # max |Δradius| per frame as fraction of iris radius

    DRIFT_PERSIST = 6                 # consecutive drift frames before TRACK_LOST

    # blink classification — REQUIRES both signs (low EAR AND lost iris), per spec rationale.
    EAR_BLINK_FRAC = 0.62
    EAR_OPEN_FRAC = 0.80
    EAR_ALPHA = 0.04

    # contour reference — caller-supplied confidence below this is "uncertain".
    REFERENCE_VALID_CONF = 0.35

    # PARTIAL_OCCLUSION classification — visible arc is shrunk but fit is good.
    PARTIAL_COV = 0.30

    # diagnostic-only flags (never invalidate the frame).
    APERTURE_MARGIN = 0.25            # eye-local box overflow before outside_aperture WARNING


# --------------------------------------------------------------------------- helpers
def _aperture_from_extents(extents):
    """Adapt eye-opening contour extents → the (inner, outer, upper, lower) tuple project_eye_local
    expects. The contour replaces four independently tracked points; medial→inner, lateral→outer."""
    if extents is None:
        return (None, None, None, None)
    return (extents.get("medial"), extents.get("lateral"),
            extents.get("upper"), extents.get("lower"))


# --------------------------------------------------------------------------- the orchestrator
class V1Tracker:
    """Per-eye V1 orchestrator: limbus + contour primary, CFT helper optional.

    One instance tracks ONE eye. Binocular work uses one independent instance per
    eye (spec §H); there is no cross-coupling.

    Construction seeds from the Stage-0 approved iris centre/radius on the
    approved init frame. After construction `step()` advances one frame and
    returns a FrameMeasurement.

    The clinical centre comes from `fit_limbus` on the colour frame; the CFT
    helper (if enabled) only supplies a motion prior for the search seed and
    logs an internal consistency score.
    """

    def __init__(self, eye: str, approved_centre, approved_radius, init_gray,
                 init_frame_number: int, use_cft_helper: bool = False,
                 params: V1Params = V1Params):
        self.eye = eye
        self.P = params
        self.approved_centre = (float(approved_centre[0]), float(approved_centre[1]))
        self.approved_radius = float(approved_radius)

        self.prev_gray = init_gray
        self.init_frame = int(init_frame_number)

        self.last_centre: Tuple[float, float] = self.approved_centre
        self.last_radius: float = self.approved_radius
        self.iris_radius_est: Optional[float] = None    # stable radius from well-seen frames
        self.last_valid_centre: Optional[Tuple[float, float]] = self.approved_centre
        self.last_valid_radius: Optional[float] = self.approved_radius

        self.ear_base: Optional[float] = None
        self.drift_run = 0
        self.state = INITIALIZING

        # Optional CFT helper — gated by use_cft_helper. The helper provides a motion
        # prior for the limbus search and a consistency score; it CANNOT invalidate a
        # frame. Constructed lazily because it does its own seeding work.
        self.helper: Optional[CompositeFeatureTracker] = None
        if use_cft_helper:
            self.helper = CompositeFeatureTracker(eye, self.approved_centre,
                                                  self.approved_radius, init_gray,
                                                  init_frame_number)

    # ---- EAR / blink classification (same logic as composite_tracker, kept local) -----
    def _ear_class(self, ear):
        if ear is None:
            return "unknown"
        if self.ear_base is None:
            self.ear_base = ear
            return "open"
        if ear > self.ear_base:
            self.ear_base = (1 - self.P.EAR_ALPHA) * self.ear_base + self.P.EAR_ALPHA * ear
        if ear < self.P.EAR_BLINK_FRAC * self.ear_base:
            return "closed"
        if ear < self.P.EAR_OPEN_FRAC * self.ear_base:
            return "reduced"
        return "open"

    # ---- limbus refit at a held radius from the visible arc (occlusion-invariant) -----
    def _refit_fixed_r(self, limbus, prior, R) -> Optional[LimbusFit]:
        if limbus.arc_pts is None or len(limbus.arc_pts) < 5:
            return None
        c, inl = _fit_centre_fixed_r(limbus.arc_pts, prior, R, max(1.5, 0.05 * R))
        if inl.sum() < 5:
            return None
        cov2 = _arc_coverage(limbus.arc_pts[inl], c[0], c[1])
        return LimbusFit(c[0], c[1], R, limbus.inlier_fraction, int(inl.sum()),
                         limbus.n_edges, None, arc_coverage=cov2,
                         arc_pts=limbus.arc_pts[inl], radius_fixed=True)

    # ---- main step ---------------------------------------------------------
    def step(self, gray, frame_no, timestamp_ms, time_sec, ear=None,
             aperture=None, extents=None, mp_iris=None, bgr=None,
             reference_state: str = "lost", reference_confidence: float = 0.0
             ) -> FrameMeasurement:
        """Advance one frame and return a FrameMeasurement.

        gray, bgr        : the frame (gray + colour). bgr is required for the limbus
                           sclera-whiteness gate; without it the fit is much weaker.
        ear              : eye-aspect-ratio for blink classification, or None.
        aperture/extents : the eye-opening contour reference. Pass `extents` (the dict
                           from EyeOpeningContour.extents()) — `aperture` is accepted as
                           the equivalent (inner, outer, upper, lower) tuple for callers
                           that already build that form.
        mp_iris          : (cx, cy, r) MediaPipe iris this frame, or None. Used ONLY for
                           recovery / re-seed — NEVER the per-frame clinical centre.
        reference_state, reference_confidence : from the EyeOpeningContour tracker.
        """
        fm = FrameMeasurement(
            frame_no, int(timestamp_ms), float(time_sec), self.state, ear=ear,
            reference_state=reference_state,
            reference_confidence=float(reference_confidence or 0.0),
        )
        ear_class = self._ear_class(ear)

        if aperture is None and extents is not None:
            aperture = _aperture_from_extents(extents)
        aperture = aperture or (None, None, None, None)

        # ---- recovery path (TRACK_LOST) — gap output until we re-seed --------------
        if self.state == TRACK_LOST:
            self._attempt_recovery(gray, frame_no, mp_iris, bgr)
            self.prev_gray = gray
            return self._finalize(fm, reference_confidence)

        # ---- optional CFT helper: provide a motion prior, log consistency ---------
        helper_centre = None
        helper_cc = 0.0
        helper_fc = 0.0
        n_active = n_trusted = 0
        if self.helper is not None:
            try:
                hfm = self.helper.step(gray, frame_no, timestamp_ms, time_sec, ear,
                                       aperture=aperture, mp_iris=mp_iris, bgr=bgr,
                                       reference_state=reference_state,
                                       reference_confidence=reference_confidence)
                # The helper produced its own measurement — we extract only the
                # prediction + scores. Its validity verdict is NOT consulted.
                helper_centre = hfm.iris_centre or hfm.raw_iris_centre
                helper_cc = hfm.composite_confidence
                helper_fc = hfm.feature_confidence_mean
                n_active = hfm.n_active
                n_trusted = hfm.n_trusted
            except Exception:
                helper_centre = None

        # ---- search seed for the limbus fit ---------------------------------------
        # Prior preference: CFT helper prediction (when available and reasonable),
        # else last clinical centre. MediaPipe is NOT used as a per-frame seed here —
        # it is for recovery only.
        seed = self.last_centre
        if helper_centre is not None:
            dx = helper_centre[0] - self.last_centre[0]
            dy = helper_centre[1] - self.last_centre[1]
            if np.hypot(dx, dy) <= self.last_radius:    # plausible per-frame motion
                seed = helper_centre

        # ---- limbus fit — the primary clinical step (§D.4) ------------------------
        limbus = None
        drift_reason = ""
        if bgr is not None:
            r_seed = self.iris_radius_est or self.last_radius or self.approved_radius
            limbus = fit_limbus(gray, seed[0], seed[1], r_seed, bgr=bgr)
            if limbus is None or limbus.inlier_fraction < self.P.LIMBUS_MIN_FRAC:
                drift_reason = "limbus_unfit"
                limbus = None
        else:
            drift_reason = "limbus_unfit"

        if limbus is not None:
            cov = limbus.arc_coverage
            # When the iris is well seen, refresh the stable radius from the free fit.
            if cov >= self.P.LIMBUS_COV_GOOD:
                a = self.P.LIMBUS_R_SMOOTH
                self.iris_radius_est = limbus.r if self.iris_radius_est is None \
                    else (1 - a) * self.iris_radius_est + a * limbus.r
            # When a lid partly covers the iris (low coverage) but we know the radius,
            # refit the centre at a FIXED radius from the visible arc. Eyelid coverage
            # MUST NOT shrink the disc or move the centre.
            elif self.iris_radius_est:
                refit = self._refit_fixed_r(limbus, seed, self.iris_radius_est)
                if refit is not None:
                    limbus = refit
                    cov = limbus.arc_coverage

            # Final coverage + temporal-continuity gates.
            if cov < self.P.LIMBUS_COV_MIN:
                drift_reason = "poor_arc_coverage"
            else:
                dx = limbus.cx - self.last_centre[0]
                dy = limbus.cy - self.last_centre[1]
                r_ref = self.iris_radius_est or self.last_radius or limbus.r
                if np.hypot(dx, dy) > self.P.CENTRE_JUMP_FRAC * r_ref:
                    drift_reason = "temporal_discontinuity"
                elif abs(limbus.r - (self.iris_radius_est or self.last_radius)) > \
                        self.P.RADIUS_JUMP_FRAC * r_ref:
                    drift_reason = "temporal_discontinuity"

        # ---- blink classification — REQUIRES both signs (low EAR AND lost iris) ----
        iris_failed = (drift_reason == "limbus_unfit")
        if iris_failed and ear_class in ("closed", "reduced"):
            self.state = BLINK
            fm.state = BLINK
            self.drift_run = 0
            self.prev_gray = gray
            return self._finalize(fm, reference_confidence, n_active=n_active,
                                  n_trusted=n_trusted, helper_cc=helper_cc,
                                  helper_fc=helper_fc, limbus=None)

        # ---- drift handling — gap output for a drift frame ------------------------
        if drift_reason:
            self.drift_run += 1
            fm.drift_flag = True
            fm.drift_reason = drift_reason
            if self.drift_run >= self.P.DRIFT_PERSIST:
                self.state = TRACK_LOST
            else:
                self.state = DRIFT_SUSPECTED
            fm.state = self.state
            self.prev_gray = gray
            return self._finalize(fm, reference_confidence, n_active=n_active,
                                  n_trusted=n_trusted, helper_cc=helper_cc,
                                  helper_fc=helper_fc, limbus=limbus)

        # ---- VALID frame: TRACKING or PARTIAL_OCCLUSION ----------------------------
        assert limbus is not None
        self.drift_run = 0
        disc_centre = (limbus.cx, limbus.cy)
        self.last_centre = disc_centre
        self.last_radius = self.iris_radius_est or limbus.r
        self.last_valid_centre = disc_centre
        self.last_valid_radius = self.last_radius

        # Helper disagreement is a diagnostic FLAG ONLY — never invalidates the frame
        # when the limbus fit is healthy (the limbus is primary; CFT is a helper).
        if helper_centre is not None and \
                np.hypot(helper_centre[0] - disc_centre[0],
                         helper_centre[1] - disc_centre[1]) > 0.5 * self.last_radius:
            fm.drift_flag = True
            fm.drift_reason = "helper_disagreement"   # flag only — fm.state stays VALID

        cov = limbus.arc_coverage
        if cov < self.P.PARTIAL_COV or ear_class == "reduced":
            self.state = PARTIAL_OCCLUSION
        else:
            self.state = TRACKING
        fm.state = self.state

        # Eye-local from the eye-opening contour (the V1 reference frame).
        el = project_eye_local(disc_centre, *aperture)

        fm.iris_centre = disc_centre
        fm.raw_iris_centre = disc_centre
        fm.iris_radius = self.last_radius
        fm.iris_dx_from_initial = disc_centre[0] - self.approved_centre[0]
        fm.iris_dy_from_initial = disc_centre[1] - self.approved_centre[1]
        fm.iris_valid = fm.validity == "valid"
        fm.reference_valid = (reference_state == "tracked"
                              and reference_confidence >= self.P.REFERENCE_VALID_CONF)
        fm.reference_uncertain = not fm.reference_valid
        fm.clinical_relative_valid = (fm.iris_valid and fm.reference_valid
                                      and el[0] is not None and el[1] is not None)
        fm.eye_local = el if fm.clinical_relative_valid else (None, None)

        fm.limbus_inlier_fraction = limbus.inlier_fraction
        fm.limbus_ellipse = limbus.ellipse
        fm.limbus_arc_coverage = cov
        fm.limbus_arc_pts = limbus.arc_pts
        fm.limbus_radius_fixed = limbus.radius_fixed

        # frame_confidence — the principal per-frame quality (spec §E). Combines limbus
        # fit + coverage with the contour reference confidence. CFT helper agreement
        # only modulates it slightly; never gates validity.
        lq = 0.5 * limbus.inlier_fraction + 0.5 * cov
        occ = 0.7 if self.state == PARTIAL_OCCLUSION else 1.0
        ref_q = max(0.35, min(1.0, fm.reference_confidence)) if fm.reference_valid \
            else max(0.2, fm.reference_confidence)
        helper_q = 0.85 + 0.15 * helper_cc if self.helper is not None else 1.0
        fm.frame_confidence = float(occ * min(1.0, 0.3 + lq) * ref_q * helper_q)
        fm.composite_confidence = helper_cc
        fm.feature_confidence_mean = helper_fc
        fm.n_active = n_active
        fm.n_trusted = n_trusted

        self.prev_gray = gray
        return fm

    # ---- recovery ----------------------------------------------------------
    def _attempt_recovery(self, gray, frame_no, mp_iris, bgr) -> None:
        """Attempt a re-seed from MediaPipe (when available) or the last valid centre.
        A successful re-seed restores TRACKING; nothing else changes state."""
        seed = (mp_iris[0], mp_iris[1]) if mp_iris else self.last_valid_centre
        radius = (mp_iris[2] if mp_iris else None) or self.last_valid_radius \
            or self.approved_radius
        if seed is None or bgr is None:
            return
        limbus = fit_limbus(gray, seed[0], seed[1], float(radius), bgr=bgr)
        if limbus is None or limbus.inlier_fraction < self.P.LIMBUS_MIN_FRAC:
            return
        if limbus.arc_coverage < self.P.LIMBUS_COV_MIN:
            return
        self.last_centre = (limbus.cx, limbus.cy)
        self.last_radius = limbus.r
        self.state = TRACKING
        self.drift_run = 0

    # ---- finalize a gap frame ----------------------------------------------
    def _finalize(self, fm: FrameMeasurement, reference_confidence,
                  n_active=0, n_trusted=0, helper_cc=0.0, helper_fc=0.0,
                  limbus=None) -> FrameMeasurement:
        # Gap states emit no clinical centre. Reference info is still recorded so
        # the CSV/metadata can see why the relative trace is also a gap.
        fm.reference_valid = (fm.reference_state == "tracked"
                              and reference_confidence >= self.P.REFERENCE_VALID_CONF)
        fm.reference_uncertain = not fm.reference_valid
        fm.iris_valid = False
        fm.clinical_relative_valid = False
        fm.eye_local = (None, None)
        fm.iris_radius = self.last_radius
        fm.n_active = n_active
        fm.n_trusted = n_trusted
        fm.composite_confidence = helper_cc
        fm.feature_confidence_mean = helper_fc
        if limbus is not None:
            fm.limbus_inlier_fraction = limbus.inlier_fraction
            fm.limbus_ellipse = limbus.ellipse
            fm.limbus_arc_coverage = limbus.arc_coverage
            fm.limbus_arc_pts = limbus.arc_pts
            fm.limbus_radius_fixed = limbus.radius_fixed
        fm.frame_confidence = 0.0
        return fm
