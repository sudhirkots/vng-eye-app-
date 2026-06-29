"""V2 'virtual spectacles' tracker — rigid orbit-pair + sclera/iris content lock.

Two clinician-marked orbit ovals are treated as a single rigid pair of spectacles
mounted on the face (translation + rotation + uniform scale = 4 DOF, fit by weighted
Umeyama). Then, per frame, a small local translation search around the rigid
prediction picks the candidate oval that best contains the eye-content complex:

    large white sclera region  +  dark curved iris/limbus arc inside or adjacent.

The clinician principle (locked):

    Track the eye-content complex inside the orbit oval:
    sclera plus dark curved iris/limbus.

Per-frame pipeline:

    1. KLT (forward + reverse round-trip error) on the 8 Stage-0 anchor points.
    2. Weighted Umeyama similarity fit Stage-0 -> tracked.
    3. Apply fit to Stage-0 ovals -> RIGID PREDICTION.
    4. For each eye, search a 5x5 translation grid around the rigid prediction (no
       rotation/scale change in the local search; rigid model already handles those).
       For each candidate, score the sclera+iris-support pair via
       v2.core.eye_content_detector.score_oval_content().
    5. final_score = w_geom * geometry_proximity + w_pair * pair_score - penalty.
       Best candidate wins, per eye.
    6. Status taxonomy:
          ok_sclera_iris_locked       -> best candidate IS the rigid prediction AND
                                         pair_score > content threshold
          corrected_by_sclera_iris    -> best candidate is a translation of the rigid
                                         prediction AND pair_score > content threshold
          uncertain_no_iris_support   -> sclera found but no qualifying iris-support
                                         in the search grid
          uncertain_no_sclera         -> no qualifying sclera in any candidate
          occluded_blink_hold         -> content missing this frame; frames_since_last
                                         _good <= int(fps * 0.5); oval is extrapolated
                                         by last_motion_vector
          uncertain_long_occlusion    -> content missing beyond blink tolerance
          lost                        -> rigid sanity caps fail, OR uncertain for too
                                         long in a row

No iris CENTRE is output (this milestone is orbit only). No MediaPipe. No V1 imports.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from v2.core.eye_content_detector import (
    EyeContent, EyeContentParams, OvalGeom, score_oval_content,
)


# --------------------------------------------------------------------------- params
class SpectaclesParams:
    # KLT pyramid window.
    LK_WIN_SIZE = (31, 31)
    LK_MAX_LEVEL = 5
    LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    # Forward-backward error scale (px) for the per-point weight.
    FB_ERR_SCALE_PX = 4.0
    # Geometric KLT-confidence thresholds (mean of 8 KLT point weights).
    CONF_OK = 0.60
    CONF_UNC = 0.30
    # Sanity caps on the similarity transform — outside these → forced `lost`.
    MAX_TRANS_PX = 400.0
    MAX_ROT_DEG = 30.0
    MIN_SCALE = 0.50
    MAX_SCALE = 1.80

    # ---- content-search params (clinician spec) -----------------------------
    # Local translation search around the rigid prediction. 5x5 grid at ±25 px.
    SEARCH_TRANS_PX = (-25, -12, 0, 12, 25)
    # Score weights.
    W_GEOM = 0.30
    W_PAIR = 0.60
    W_PEN = 1.0
    # Pair-score threshold for OK statuses.
    PAIR_OK_THRESHOLD = 0.30
    PAIR_UNCERTAIN_THRESHOLD = 0.10
    # Lost holdoff: this many consecutive 'uncertain' frames -> 'lost'.
    LOST_HOLDOFF_FRAMES = 8
    # ---- blink / transient occlusion handling (clinician decision: 0.5 s) ----
    BLINK_TOLERANCE_SEC = 0.5
    # Motion smoothing for blink prediction: average over last N 'ok' frames.
    MOTION_HISTORY_OK_FRAMES = 5
    # Reacquisition cap: during blink_hold, only accept new content if the new
    # oval centre is within this fraction of the major axis from the predicted
    # oval centre. Prevents 'reacquiring' the wrong eye / drift target.
    REACQUIRE_MAX_DRIFT_FRAC = 1.5


POINTS = ("medial", "lateral", "upper", "lower")     # in canonical order


# --------------------------------------------------------------------------- helpers
def _stage0_anchor_array(stage0: Dict) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Pack the Stage-0 anchor coordinates into a (8, 2) array, in canonical order
    [L.medial, L.lateral, L.upper, L.lower, R.medial, R.lateral, R.upper, R.lower].
    Returns (anchors, names) where names parallel the rows for debug."""
    anchors: List[List[float]] = []
    names: List[Tuple[str, str]] = []
    for ek in ("L", "R"):
        block = (stage0.get("eyes") or {}).get(ek) or {}
        pts = block.get("oval_points") or {}
        for name in POINTS:
            p = pts.get(name) or {}
            anchors.append([float(p.get("x", 0.0)), float(p.get("y", 0.0))])
            names.append((ek, name))
    return np.asarray(anchors, dtype=np.float32), names


def _weighted_similarity_fit(src: np.ndarray, dst: np.ndarray,
                              w: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Closed-form weighted similarity (Umeyama, scale + rotation + translation):

        dst_i  ~=  s * R @ src_i  +  t

    Inputs:
        src : (N, 2) source points (Stage-0 anchors)
        dst : (N, 2) destination points (KLT-tracked positions this frame)
        w   : (N,)   per-point weights, w >= 0

    Returns (s, theta_rad, t)  where  t is shape (2,)  and  R = rot(theta_rad).
    If the total weight is tiny or the source is degenerate the identity is returned.
    """
    if w.sum() < 1e-6:
        return 1.0, 0.0, np.zeros(2, dtype=np.float32)
    W = float(w.sum())
    mu_src = (w[:, None] * src).sum(axis=0) / W
    mu_dst = (w[:, None] * dst).sum(axis=0) / W
    src_c = src - mu_src
    dst_c = dst - mu_dst
    # Weighted covariance.
    Sxy = ((w[:, None] * src_c) * dst_c).sum(axis=0)        # shape (2,) per pair
    # Build the 2x2 cross-covariance H.
    H = np.zeros((2, 2), dtype=np.float64)
    for i in range(src_c.shape[0]):
        H += w[i] * np.outer(src_c[i], dst_c[i])
    # SVD-based rotation (Umeyama 1991).
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return 1.0, 0.0, mu_dst - mu_src
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0 if d > 0 else -1.0])
    R = (Vt.T @ D @ U.T).astype(np.float64)
    # Weighted variance of src for scale.
    var_src = ((w[:, None] * src_c) * src_c).sum() / W
    if var_src < 1e-6:
        s = 1.0
    else:
        s = float((S * np.diag(D)).sum() / max(var_src, 1e-6))
    t = mu_dst.astype(np.float64) - s * (R @ mu_src.astype(np.float64))
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(s), theta, t.astype(np.float32)


def _apply_similarity(pts: np.ndarray, s: float, theta: float,
                       t: np.ndarray) -> np.ndarray:
    """Apply  out = s * R(theta) @ pts.T  +  t  ;  pts is (N, 2)."""
    c, sn = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -sn], [sn, c]], dtype=np.float32)
    out = (s * (pts @ R.T)) + t
    return out.astype(np.float32)


def _is_within_sanity(s: float, theta: float, t: np.ndarray,
                       P: SpectaclesParams) -> bool:
    if not (P.MIN_SCALE <= s <= P.MAX_SCALE):
        return False
    if abs(np.degrees(theta)) > P.MAX_ROT_DEG:
        return False
    if float(np.hypot(t[0], t[1])) > P.MAX_TRANS_PX:
        return False
    return True


# --------------------------------------------------------------------------- result
@dataclass
class PerEyeResult:
    """Per-eye result for one frame.

    `predicted` is the oval BEFORE content correction (just the rigid fit).
    `spectacles` is the displayed oval AFTER content correction. When the rigid
    prediction itself wins the local search, the two are the same.
    """
    status: str = "ok_sclera_iris_locked"
    status_reason: str = ""
    # True when the rigid Umeyama fit was within sanity caps and `predicted` is
    # meaningful. False when the rigid fit was rejected; the probe should NOT draw
    # the predicted oval in that case (it's the catastrophic prediction).
    rigid_sane: bool = True
    predicted: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    spectacles: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    klt_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    klt_weights: Dict[str, float] = field(default_factory=dict)
    # Content scoring
    geometry_score: float = 0.0
    content_score: float = 0.0
    final_score: float = 0.0
    sclera_area: int = 0
    sclera_inside_oval: float = 0.0
    sclera_centroid: Optional[Tuple[float, float]] = None
    sclera_present: bool = False
    iris_support_present: bool = False
    iris_support_inside_oval: float = 0.0
    iris_support_curvature_score: float = 0.0
    iris_support_radius: Optional[float] = None
    iris_support_centre: Optional[Tuple[float, float]] = None
    iris_support_pts: Optional[np.ndarray] = None
    sclera_iris_pair_score: float = 0.0
    # Blink/occlusion bookkeeping (set by the tracker, not the detector)
    blink_hold_active: bool = False
    last_good_frame: int = 0
    frames_since_last_good_content: int = 0


@dataclass
class FrameResult:
    """One per-frame result for the spectacles tracker (both eyes)."""
    frame_no: int
    confidence: float                                 # [0, 1] = mean point weight
    s: float                                          # similarity scale
    theta_rad: float                                  # similarity rotation
    t: Tuple[float, float]                             # similarity translation
    per_eye: Dict[str, PerEyeResult] = field(default_factory=dict)


# --------------------------------------------------------------------------- tracker
class SpectaclesTracker:
    """One instance covers both eyes as one rigid spectacles object.

    Construct with the Stage-0 dict (from `orbit_stage0.run_orbit_stage0`) and an
    init grayscale frame. Call `step(curr_gray, frame_no)` per frame.
    """

    def __init__(self, stage0: Dict, init_gray: np.ndarray, fps: float = 30.0,
                 params: SpectaclesParams = SpectaclesParams,
                 content_params: EyeContentParams = EyeContentParams):
        self.P = params
        self.CP = content_params
        self.stage0_anchors, self.anchor_names = _stage0_anchor_array(stage0)
        # Bail if fewer than 8 anchors (single-eye Stage 0 not supported here).
        if self.stage0_anchors.shape[0] != 8:
            raise ValueError(
                f"SpectaclesTracker needs 8 Stage-0 anchors (4 per eye); "
                f"got {self.stage0_anchors.shape[0]}."
            )
        self.prev_gray: Optional[np.ndarray] = init_gray.copy()
        self.klt_prev: np.ndarray = self.stage0_anchors.copy()
        # Last good rigid transform.
        self.last_good_s: float = 1.0
        self.last_good_theta: float = 0.0
        self.last_good_t: np.ndarray = np.zeros(2, dtype=np.float32)

        # Per-eye state for content lock + blink hold.
        self.fps = float(fps) if fps and fps > 1.0 else 30.0
        self.blink_tolerance_frames = max(1, int(round(self.fps * self.P.BLINK_TOLERANCE_SEC)))
        self.eye_state: Dict[str, Dict] = {}
        for ek in ("L", "R"):
            # last_good_oval: the 4 oval-anchor positions (CORRECTED) from the most
            # recent ok / corrected frame for this eye.
            init_pts = {name: tuple(self.stage0_anchors[i])
                        for i, (eye, name) in enumerate(self.anchor_names)
                        if eye == ek}
            self.eye_state[ek] = {
                "last_good_oval": init_pts,
                "last_good_frame": 0,                                # frame_no
                "centre_history": deque(maxlen=self.P.MOTION_HISTORY_OK_FRAMES + 1),
                "uncertain_run": 0,
                "blink_hold_active": False,
            }
        # Stage-0 iris radii per eye (optional; for the iris-support radius prior).
        self.stage0_iris_radius: Dict[str, Optional[float]] = {"L": None, "R": None}
        eyes_blk = (stage0.get("eyes") or {})
        for ek in ("L", "R"):
            iris_blk = (eyes_blk.get(ek) or {}).get("iris_limbus")
            if iris_blk and "radius" in iris_blk:
                try:
                    self.stage0_iris_radius[ek] = float(iris_blk["radius"])
                except (TypeError, ValueError):
                    pass

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _oval_from_four(pts: Dict[str, Tuple[float, float]]) -> OvalGeom:
        """4 named anchors {medial, lateral, upper, lower} -> OvalGeom."""
        M = np.array(pts["medial"], float)
        L = np.array(pts["lateral"], float)
        U = np.array(pts["upper"], float)
        D = np.array(pts["lower"], float)
        cxy = (M + L + U + D) / 4.0
        a = float(np.linalg.norm(L - M) / 2.0)
        b = float(np.linalg.norm(D - U) / 2.0)
        theta = float(np.degrees(np.arctan2(L[1] - M[1], L[0] - M[0])))
        return OvalGeom(cx=float(cxy[0]), cy=float(cxy[1]),
                         a_half=max(4.0, a), b_half=max(3.0, b),
                         theta_deg=theta)

    @staticmethod
    def _translate(pts: Dict[str, Tuple[float, float]],
                    dx: float, dy: float) -> Dict[str, Tuple[float, float]]:
        return {name: (float(p[0] + dx), float(p[1] + dy))
                for name, p in pts.items()}

    def _motion_vector(self, ek: str) -> Tuple[float, float]:
        """Average per-frame translation of the eye's oval centre over the last
        few 'ok' frames. (0, 0) if history is empty."""
        hist = list(self.eye_state[ek]["centre_history"])
        if len(hist) < 2:
            return 0.0, 0.0
        dx = (hist[-1][0] - hist[0][0]) / max(1, len(hist) - 1)
        dy = (hist[-1][1] - hist[0][1]) / max(1, len(hist) - 1)
        return float(dx), float(dy)

    # ---- public step --------------------------------------------------------
    def step(self, curr_bgr: np.ndarray, curr_gray: np.ndarray,
              frame_no: int) -> FrameResult:
        """Advance one frame. Note: takes BOTH the BGR frame (for content
        detection) and the gray frame (for KLT). The caller should pass both."""
        # ------------------------------------------------ rigid KLT prediction
        p_prev = self.klt_prev.reshape(-1, 1, 2).astype(np.float32)
        p_curr, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, p_prev, None,
            winSize=self.P.LK_WIN_SIZE, maxLevel=self.P.LK_MAX_LEVEL,
            criteria=self.P.LK_CRITERIA,
        )
        p_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray, p_curr, None,
            winSize=self.P.LK_WIN_SIZE, maxLevel=self.P.LK_MAX_LEVEL,
            criteria=self.P.LK_CRITERIA,
        )
        fb_err = np.linalg.norm(
            p_back.reshape(-1, 2) - p_prev.reshape(-1, 2), axis=1
        )
        weights = np.zeros(8, dtype=np.float32)
        for i in range(8):
            ok_fwd = bool(st_fwd[i, 0]) if st_fwd is not None else False
            ok_bwd = bool(st_bwd[i, 0]) if st_bwd is not None else False
            if not (ok_fwd and ok_bwd):
                continue
            err = float(fb_err[i])
            weights[i] = max(0.0, 1.0 - min(1.0, err / max(1e-6, self.P.FB_ERR_SCALE_PX)))
        tracked = p_curr.reshape(-1, 2).astype(np.float32)
        s, theta, t = _weighted_similarity_fit(self.stage0_anchors, tracked, weights)
        rigid_conf = float(weights.mean())
        rigid_sane = _is_within_sanity(s, theta, t, self.P)
        if not rigid_sane:
            # Rigid model went catastrophic — fall back to last good for prediction,
            # but DO mark eyes as lost at the end if no content rescue happens.
            use_s, use_theta, use_t = (self.last_good_s, self.last_good_theta,
                                         self.last_good_t)
        else:
            use_s, use_theta, use_t = s, theta, t
            self.last_good_s = s
            self.last_good_theta = theta
            self.last_good_t = t
        predicted_xy = _apply_similarity(self.stage0_anchors, use_s, use_theta, use_t)

        # Pack rigid prediction into per-eye dicts.
        predicted_per_eye: Dict[str, Dict[str, Tuple[float, float]]] = {"L": {}, "R": {}}
        klt_per_eye: Dict[str, Dict[str, Tuple[float, float]]] = {"L": {}, "R": {}}
        klt_w_per_eye: Dict[str, Dict[str, float]] = {"L": {}, "R": {}}
        for i, (ek, name) in enumerate(self.anchor_names):
            predicted_per_eye[ek][name] = (float(predicted_xy[i, 0]),
                                             float(predicted_xy[i, 1]))
            klt_per_eye[ek][name] = (float(tracked[i, 0]), float(tracked[i, 1]))
            klt_w_per_eye[ek][name] = float(weights[i])

        # ----------------------------- per-eye content search + status decision
        per_eye_result: Dict[str, PerEyeResult] = {}
        for ek in ("L", "R"):
            per_eye_result[ek] = self._decide_eye(
                ek, predicted_per_eye[ek], klt_per_eye[ek], klt_w_per_eye[ek],
                curr_bgr, frame_no, rigid_sane,
            )

        # ------------------------------------------------ advance KLT prior
        # Use the CORRECTED oval anchors (per-eye spectacles) as the next prior.
        next_prior = np.zeros_like(self.klt_prev)
        for i, (ek, name) in enumerate(self.anchor_names):
            xy = per_eye_result[ek].spectacles.get(name,
                                                       predicted_per_eye[ek][name])
            next_prior[i, 0] = xy[0]
            next_prior[i, 1] = xy[1]
        self.klt_prev = next_prior
        self.prev_gray = curr_gray.copy()

        return FrameResult(
            frame_no=frame_no, confidence=rigid_conf,
            s=use_s, theta_rad=use_theta, t=(float(use_t[0]), float(use_t[1])),
            per_eye=per_eye_result,
        )

    # ---- the per-eye decision ----------------------------------------------
    def _decide_eye(self, ek: str,
                     predicted_pts: Dict[str, Tuple[float, float]],
                     klt_pts: Dict[str, Tuple[float, float]],
                     klt_w: Dict[str, float],
                     curr_bgr: np.ndarray, frame_no: int,
                     rigid_sane: bool) -> PerEyeResult:
        """Local content search + status decision for one eye on this frame."""
        P = self.P
        CP = self.CP
        state = self.eye_state[ek]

        # Score the rigid prediction itself first.
        oval_pred = self._oval_from_four(predicted_pts)
        content_pred = score_oval_content(curr_bgr, oval_pred, CP,
                                            stage0_iris_radius=self.stage0_iris_radius[ek])

        # Local translation search around the rigid prediction. 5x5 grid.
        best = {
            "dxdy": (0, 0),
            "oval": oval_pred,
            "content": content_pred,
            "geom_score": 1.0,
            "pair_score": content_pred.pair_score,
            "final_score": 0.0,
        }
        # Geometry score: 1 at centre, fall off linearly to 0.5 at the corner.
        corner = float(np.hypot(P.SEARCH_TRANS_PX[0], P.SEARCH_TRANS_PX[0]))
        for dx in P.SEARCH_TRANS_PX:
            for dy in P.SEARCH_TRANS_PX:
                pts_dxdy = self._translate(predicted_pts, dx, dy)
                oval_c = self._oval_from_four(pts_dxdy)
                content_c = score_oval_content(curr_bgr, oval_c, CP,
                                                 stage0_iris_radius=self.stage0_iris_radius[ek])
                # Penalty if sclera blob is largely OUTSIDE the candidate oval.
                pen = 0.0
                if content_c.sclera_score > 0 and content_c.sclera_inside_oval < 0.50:
                    pen += 0.30
                # Penalty if iris-support blob is largely outside the oval.
                if content_c.iris_support_present and content_c.iris_support_inside_oval < 0.50:
                    pen += 0.20
                # Geometry proximity score.
                d = float(np.hypot(dx, dy))
                geom = 1.0 - 0.5 * (d / max(1.0, corner))
                final = (P.W_GEOM * geom + P.W_PAIR * content_c.pair_score
                          - P.W_PEN * pen)
                if final > best["final_score"]:
                    best.update({
                        "dxdy": (dx, dy),
                        "oval": oval_c,
                        "content": content_c,
                        "geom_score": float(geom),
                        "pair_score": float(content_c.pair_score),
                        "final_score": float(final),
                    })

        chosen_pts = self._translate(predicted_pts, best["dxdy"][0], best["dxdy"][1])
        cnt: EyeContent = best["content"]
        is_pair = (cnt.sclera_score > 0 and cnt.iris_support_present
                    and best["pair_score"] >= P.PAIR_OK_THRESHOLD)
        # 'single-half' bar: in spec each half alone needs ≥0.40 to be provisional.
        SINGLE_BAR = 0.40
        is_iris_only_solid = (cnt.iris_support_present
                                and cnt.iris_support_curvature_score >= SINGLE_BAR
                                and cnt.iris_support_inside_oval >= 0.50)
        is_sclera_only_solid = (cnt.sclera_score >= SINGLE_BAR
                                  and cnt.sclera_inside_oval >= 0.50)

        # Decide status.
        result = PerEyeResult(
            predicted=predicted_pts,
            spectacles=chosen_pts,
            klt_points=klt_pts,
            klt_weights=klt_w,
            geometry_score=best["geom_score"],
            content_score=float(cnt.pair_score),
            final_score=best["final_score"],
            sclera_area=cnt.sclera_area,
            sclera_inside_oval=cnt.sclera_inside_oval,
            sclera_centroid=cnt.sclera_centroid,
            sclera_present=(cnt.sclera_score > 0),
            iris_support_present=cnt.iris_support_present,
            iris_support_inside_oval=cnt.iris_support_inside_oval,
            iris_support_curvature_score=cnt.iris_support_curvature_score,
            iris_support_radius=cnt.iris_support_radius,
            iris_support_centre=cnt.iris_support_centre,
            iris_support_pts=cnt.iris_support_pts,
            sclera_iris_pair_score=float(cnt.pair_score),
            last_good_frame=int(state["last_good_frame"]),
        )

        # If rigid model went catastrophic AND we have no real content rescue:
        if not rigid_sane and not is_pair:
            result.status = "lost"
            result.status_reason = "rigid_transform_out_of_sanity"
            state["uncertain_run"] += 1
            # Freeze: use last_good oval
            result.spectacles = dict(state["last_good_oval"])
            return result

        # Pair found ⇒ ok / corrected. Update last_good and reset uncertain_run.
        if is_pair:
            zero_dx = (best["dxdy"] == (0, 0))
            result.status = ("ok_sclera_iris_locked" if zero_dx
                              else "corrected_by_sclera_iris")
            if best["dxdy"] != (0, 0):
                result.status_reason = (f"translated_by("
                                          f"{best['dxdy'][0]:+d},{best['dxdy'][1]:+d})px")
            else:
                result.status_reason = "rigid_prediction"
            state["last_good_oval"] = dict(chosen_pts)
            state["last_good_frame"] = frame_no
            state["uncertain_run"] = 0
            state["blink_hold_active"] = False
            # Track motion history (oval centre).
            cxy = ((chosen_pts["medial"][0] + chosen_pts["lateral"][0]
                     + chosen_pts["upper"][0] + chosen_pts["lower"][0]) / 4.0,
                    (chosen_pts["medial"][1] + chosen_pts["lateral"][1]
                     + chosen_pts["upper"][1] + chosen_pts["lower"][1]) / 4.0)
            state["centre_history"].append(cxy)
            return result

        # No pair, but one half is solid → provisional. Prefer the detector's
        # explicit rejection reason if one was set (e.g. 'hair_like_white_component'
        # or 'eyebrow_black_white_texture'), which is the eyebrow-drift signal.
        if is_iris_only_solid and not is_sclera_only_solid:
            result.status = "provisional_iris_only"
            result.status_reason = (cnt.rejection_reason
                                      if cnt.rejection_reason
                                      else "iris_found_no_qualifying_sclera")
            state["uncertain_run"] += 1
            state["blink_hold_active"] = False
            return result
        if is_sclera_only_solid and not is_iris_only_solid:
            result.status = "provisional_sclera_only"
            result.status_reason = (cnt.rejection_reason
                                      if cnt.rejection_reason
                                      else "sclera_found_no_qualifying_iris")
            state["uncertain_run"] += 1
            state["blink_hold_active"] = False
            return result

        # Neither qualifying. Branch on blink tolerance.
        frames_since = frame_no - int(state["last_good_frame"])
        result.frames_since_last_good_content = int(frames_since)
        if frames_since <= self.blink_tolerance_frames:
            # Blink hold: extrapolate from last_good using motion vector.
            dx_pf, dy_pf = self._motion_vector(ek)
            steps = max(1, frames_since)
            held = self._translate(state["last_good_oval"],
                                     dx_pf * steps, dy_pf * steps)
            result.status = "occluded_blink_hold"
            result.status_reason = (f"blink_hold({frames_since}/"
                                      f"{self.blink_tolerance_frames})")
            result.spectacles = held
            state["uncertain_run"] += 1
            state["blink_hold_active"] = True
            result.blink_hold_active = True
            return result

        # Beyond blink tolerance: decide between 'uncertain' and 'lost'.
        state["uncertain_run"] += 1
        state["blink_hold_active"] = False
        if state["uncertain_run"] >= P.LOST_HOLDOFF_FRAMES + self.blink_tolerance_frames:
            result.status = "lost"
            # Prefer the explicit content-rejection reason (eyebrow drift etc.)
            # over the generic "no_content_for_N_frames" so the HUD says WHY.
            result.status_reason = (cnt.rejection_reason
                                      if cnt.rejection_reason
                                      else f"no_content_for_{state['uncertain_run']}_frames")
            # Freeze on last_good.
            result.spectacles = dict(state["last_good_oval"])
        else:
            result.status = "uncertain_no_sclera_iris_pair"
            result.status_reason = (cnt.rejection_reason
                                      if cnt.rejection_reason
                                      else "no_sclera_iris_pair_beyond_blink_tolerance")
            # Keep the chosen candidate (best score even though not qualifying).
        return result
