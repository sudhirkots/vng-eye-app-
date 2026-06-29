"""V2 sclera + iris-support content detector.

Pure functions, no I/O. Classical CV only: HSV thresholds + connected components +
Kasa circle fit on dark-component boundaries. No MediaPipe, no V1 imports.

The clinician rule this implements:

    The eye is identified by a LARGE WHITE SCLERA region combined with a DARK CURVED
    iris/limbus arc or circle adjacent to or inside the sclera.

A white area alone is not enough (eyebrow hair, skin highlights, reflections may be
white). A dark area alone is not enough (eyebrow hair, lashes, shadow may be dark).
But the *pair* — large white + dark curved arc inside or adjacent — is the eye.

Public surface:

    OvalGeom(centre, a_half, b_half, theta_deg)
    EyeContent(sclera_score, sclera_area, sclera_inside_oval, sclera_centroid,
               iris_support_present, iris_support_inside_oval,
               iris_support_curvature_score, iris_support_radius, iris_support_centre,
               iris_support_pts, pair_proximity_score, pair_score)
    score_oval_content(frame_bgr, oval, params, stage0_iris_radius=None) -> EyeContent
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np


# --------------------------------------------------------------------------- params
class EyeContentParams:
    # Sclera detection (HSV).
    SCLERA_V_MIN = 160          # bright
    SCLERA_S_MAX = 60           # whitish (low saturation)
    SCLERA_MIN_AREA_FRAC = 0.10  # min area as fraction of oval area
    SCLERA_MAX_ASPECT = 4.0     # reject thin white hairs / lines

    # Iris-support detection (dark blob).
    DARK_V_MAX = 80
    IRIS_MIN_AREA_FRAC = 0.03    # min area as fraction of oval area
    IRIS_MAX_AREA_FRAC = 0.45    # too dark = closed eye / shadow, not iris

    # Curvature score: residual normalised by radius. Lower residual -> higher score.
    CURV_RESIDUAL_SCALE = 0.30   # 30 % residual / radius gives score ~ 1/e
    # Optional radius prior (if Stage-0 iris radius is provided).
    RADIUS_PRIOR_SCALE = 0.30    # 30 % deviation gives score ~ 1/e

    # Pair-proximity: distance from iris-support centre to sclera centroid, as a
    # fraction of the oval's major axis (a_half). Closer = higher score.
    PAIR_PROXIMITY_SCALE = 0.40

    # ------------------- shape rules (eyebrow-rejection, 2026-06-29) -----------
    # The clinician observed that V2 was drifting onto the eyebrow because
    # eyebrow hair creates a black-white texture that the original detector
    # accepted as sclera + iris. These rules add anatomical shape constraints
    # so 'compact smooth white near a round dark blob' qualifies but
    # 'fragmented black-white squiggles' does not.
    #
    # Sclera shape requirements (in addition to V/S/aspect/min-area-frac above).
    # 2026-06-29 (perf + correctness pass): thresholds loosened after the first
    # run showed real sclera being rejected by aggressive shape rules; cost of
    # the fragmentation diagnostic was the main performance killer (per-frame
    # walk over every white component in every search candidate × 25 candidates
    # × 2 eyes), so it is now OFF by default.
    SCLERA_MIN_ABS_AREA_PX = 250      # absolute floor regardless of oval size
    SCLERA_MIN_SOLIDITY = 0.55        # squiggles still <0.55; real sclera ~0.85
    SCLERA_MIN_EXTENT = 0.30          # thin strands still <0.30; real sclera ~0.55
    SCLERA_ENABLE_FRAGMENTATION_CHECK = False
    SCLERA_FRAG_TOTAL_SMALL_FRAC = 0.60   # if enabled: 60 % small/biggest threshold
    SCLERA_SMALL_FRAG_AREA = 30       # px; components smaller than this are 'small'
    # Iris-support (dark blob) shape requirements:
    IRIS_MIN_SOLIDITY = 0.45          # dark eyebrow strand has very low solidity
    # Reject if the dark blob centre is in the TOP fraction of the oval vertical
    # extent (eyebrows tend to sit at the top of the candidate oval).
    IRIS_TOP_BAND_FRAC = 0.30
    # When a qualifying sclera is found, the dark iris-support must be adjacent
    # (already enforced as `adj > area/20` in the detector). When NO qualifying
    # sclera is found, accept the dark blob alone only if its curvature score
    # is solid -- the tracker still uses this as 'provisional_iris_only'.


# --------------------------------------------------------------------------- data
@dataclass
class OvalGeom:
    """Geometry of a candidate oval. theta is degrees, the long axis is the
    medial-lateral direction."""
    cx: float
    cy: float
    a_half: float       # major (medial-lateral) half-length
    b_half: float       # minor (upper-lower) half-length
    theta_deg: float

    @property
    def area(self) -> float:
        return float(np.pi * max(1.0, self.a_half) * max(1.0, self.b_half))


@dataclass
class EyeContent:
    """Per-candidate eye-content scoring result. All scores in [0, 1] except areas
    (pixel counts)."""
    sclera_score: float = 0.0
    sclera_area: int = 0
    sclera_inside_oval: float = 0.0            # fraction in [0, 1]
    sclera_centroid: Optional[Tuple[float, float]] = None

    iris_support_present: bool = False
    iris_support_inside_oval: float = 0.0
    iris_support_curvature_score: float = 0.0
    iris_support_radius: Optional[float] = None
    iris_support_centre: Optional[Tuple[float, float]] = None
    iris_support_pts: Optional[np.ndarray] = None  # (N,2) inlier boundary pts

    pair_proximity_score: float = 0.0
    pair_score: float = 0.0                    # combined sclera * iris * proximity
    # Short explicit reason emitted when a candidate is rejected by the shape
    # validator. Empty string when the content was accepted. The tracker plumbs
    # this into PerEyeResult.status_reason so it shows up in the HUD / CSV.
    rejection_reason: str = ""


# --------------------------------------------------------------------------- helpers
def _oval_mask(shape_hw: Tuple[int, int], oval: OvalGeom) -> np.ndarray:
    """Binary mask of the candidate oval, same shape as the frame."""
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask,
                (int(round(oval.cx)), int(round(oval.cy))),
                (max(1, int(round(oval.a_half))),
                 max(1, int(round(oval.b_half)))),
                float(oval.theta_deg), 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _kasa_circle(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit on (N, 2) points.
    Returns (cx, cy, r, mean_residual_px). If <3 points returns NaNs."""
    if pts is None or pts.shape[0] < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.c_[2 * x, 2 * y, np.ones(len(pts))]
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan"), float("nan")
    cx, cy = float(sol[0]), float(sol[1])
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 1.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = float(np.sqrt(r2))
    residuals = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, float(residuals.mean())


def _binary_aspect_ratio(component_mask: np.ndarray) -> float:
    """Aspect ratio of the smallest enclosing rotated rectangle."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    cnt = max(contours, key=cv2.contourArea)
    if cnt.shape[0] < 5:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            return 1.0
        return float(max(w, h)) / float(max(1, min(w, h)))
    (_, (mw, mh), _) = cv2.minAreaRect(cnt)
    if mw <= 0 or mh <= 0:
        return 1.0
    return float(max(mw, mh)) / float(max(1.0, min(mw, mh)))


def _solidity(component_mask: np.ndarray) -> float:
    """Solidity = area / area_of_convex_hull. Smooth oval/crescent has solidity
    close to 1.0; a fragmented or squiggly shape has solidity well below 1.0
    because the hull encloses the gaps."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return 0.0
    try:
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
    except cv2.error:
        return 0.0
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


def _extent(component_mask: np.ndarray) -> float:
    """Extent = area / bbox_area. Thin elongated lines have low extent because
    most of the bounding box is empty."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(max(1, w) * max(1, h))
    return float(area / bbox_area)


# --------------------------------------------------------------------------- detectors
def detect_sclera_in_oval(frame_bgr: np.ndarray, oval: OvalGeom,
                            P: EyeContentParams,
                            roi_box: Optional[Tuple[int, int, int, int]] = None
                            ) -> Tuple[float, int, float, Optional[Tuple[float, float]],
                                        Optional[np.ndarray], str]:
    """Detect the largest plausible sclera-like white component overlapping the oval.

    Returns (sclera_score, sclera_area_px, sclera_inside_oval_frac,
              sclera_centroid_xy, sclera_component_mask, rejection_reason).
    `sclera_score` is in [0, 1]. `rejection_reason` is "" on success.

    Shape rules added 2026-06-29 to reject eyebrow black-white texture:
        * absolute minimum area (px), not just relative fraction
        * solidity (area / convex_hull_area) - rejects squiggly hair texture
        * extent (area / bbox_area) - rejects thin line-like strands
        * fragmentation - rejects 'many small white pieces' (skin highlights / hair)
    """
    h, w = frame_bgr.shape[:2]
    # Limit work to a slightly expanded ROI around the oval so we don't HSV the
    # whole frame on every candidate.
    if roi_box is None:
        pad = int(round(max(oval.a_half, oval.b_half) * 1.4))
        x0 = max(0, int(round(oval.cx - pad)))
        y0 = max(0, int(round(oval.cy - pad)))
        x1 = min(w, int(round(oval.cx + pad)))
        y1 = min(h, int(round(oval.cy + pad)))
    else:
        x0, y0, x1, y1 = roi_box
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0, 0.0, None, None, "no_sclera_support"

    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] >= P.SCLERA_V_MIN) &
             (hsv[..., 1] <= P.SCLERA_S_MAX)).astype(np.uint8) * 255
    # Connected components in the ROI.
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n_cc <= 1:
        return 0.0, 0, 0.0, None, None, "no_sclera_support"
    # Build the oval mask in the same ROI coords.
    oval_local = OvalGeom(
        cx=oval.cx - x0, cy=oval.cy - y0,
        a_half=oval.a_half, b_half=oval.b_half, theta_deg=oval.theta_deg,
    )
    oval_mask = _oval_mask(roi.shape[:2], oval_local)

    min_area_rel = max(20, int(round(P.SCLERA_MIN_AREA_FRAC * oval.area)))
    min_area = max(min_area_rel, int(P.SCLERA_MIN_ABS_AREA_PX))

    # ---- FAST PASS over component STATS only (no per-component masks).
    # We pick the single biggest component that has the most pixels inside the
    # oval (using the stats table + ROI intersection check on the labels array,
    # not per-component masks). Shape checks (solidity/extent/aspect) are
    # computed ONLY on the winner. This is the perf path: a 5x5 grid * 2 eyes
    # was doing per-component contour+convex-hull work, which was the bottleneck.
    best_idx = -1
    best_total_inside = 0
    best_area = 0
    for idx in range(1, n_cc):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Intersection-pixel count without building a per-component mask:
        # boolean-AND on the labels slice and the oval mask is the same speed
        # as the old mask-then-AND but skips one np.uint8 conversion.
        inside = int(((labels == idx) & (oval_mask > 0)).sum())
        if inside == 0:
            continue
        if inside > best_total_inside:
            best_total_inside = inside
            best_idx = idx
            best_area = area
    if best_idx < 0:
        return 0.0, 0, 0.0, None, None, "no_sclera_support"

    # Build the winner's binary mask ONCE.
    winner_mask = (labels == best_idx).astype(np.uint8) * 255

    # ---- Shape checks on the winner only -------------------------------
    aspect = _binary_aspect_ratio(winner_mask)
    if aspect > P.SCLERA_MAX_ASPECT:
        return 0.0, 0, 0.0, None, None, "hair_like_white_component"
    solidity = _solidity(winner_mask)
    if solidity < P.SCLERA_MIN_SOLIDITY:
        return 0.0, 0, 0.0, None, None, "hair_like_white_component"
    extent = _extent(winner_mask)
    if extent < P.SCLERA_MIN_EXTENT:
        return 0.0, 0, 0.0, None, None, "hair_like_white_component"

    # ---- Optional fragmentation diagnostic (OFF by default for perf) ----
    fragmentation_flag = False
    if P.SCLERA_ENABLE_FRAGMENTATION_CHECK:
        small_frag_total = 0
        for idx in range(1, n_cc):
            if idx == best_idx:
                continue
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= P.SCLERA_SMALL_FRAG_AREA:
                continue
            # Cheap: only count pixels in the bbox slice that are inside the oval.
            x_, y_, w_, h_ = (int(stats[idx, cv2.CC_STAT_LEFT]),
                                int(stats[idx, cv2.CC_STAT_TOP]),
                                int(stats[idx, cv2.CC_STAT_WIDTH]),
                                int(stats[idx, cv2.CC_STAT_HEIGHT]))
            x_e, y_e = x_ + w_, y_ + h_
            sub_lbl = labels[y_:y_e, x_:x_e]
            sub_msk = oval_mask[y_:y_e, x_:x_e]
            inside = int(((sub_lbl == idx) & (sub_msk > 0)).sum())
            small_frag_total += inside
        fragmentation_flag = small_frag_total > (
            P.SCLERA_FRAG_TOTAL_SMALL_FRAC * best_area
        )

    best_inside_frac = best_total_inside / max(1, best_area)
    cx_local, cy_local = centroids[best_idx]
    centroid = (float(cx_local + x0), float(cy_local + y0))
    full_mask = np.zeros((h, w), np.uint8)
    full_mask[y0:y1, x0:x1] = winner_mask

    if fragmentation_flag:
        return (0.5 * (1.0 if best_inside_frac >= 0.50 else best_inside_frac),
                best_area, float(best_inside_frac), centroid, full_mask,
                "fragmented_white_texture")

    sclera_score = (1.0 if (best_area >= min_area and best_inside_frac >= 0.50)
                     else max(0.0, min(1.0, best_inside_frac)))
    return float(sclera_score), best_area, float(best_inside_frac), centroid, full_mask, ""


def detect_iris_support_in_oval(frame_bgr: np.ndarray, oval: OvalGeom,
                                  sclera_mask: Optional[np.ndarray],
                                  P: EyeContentParams,
                                  stage0_iris_radius: Optional[float] = None
                                  ) -> Tuple[bool, float, float, Optional[float],
                                              Optional[Tuple[float, float]],
                                              Optional[np.ndarray], str]:
    """Detect a dark blob inside or adjacent to the sclera and score its curvature.

    Returns (present, inside_oval_frac, curvature_score, fit_radius, fit_centre,
              pts, rejection_reason).

    Shape rules added 2026-06-29 to reject eyebrow texture as iris support:
        * solidity (area / convex_hull_area) -- hairy strands have low solidity
        * top-band rejection -- a dark blob whose centre sits in the top
          IRIS_TOP_BAND_FRAC of the oval is most likely the eyebrow inside a
          too-tall candidate oval, not the iris
        * if a qualifying sclera exists but every dark candidate fails, return
          reason='white_without_round_iris'; if no sclera but a curved dark blob
          passes, return reason='dark_arc_without_sclera' (still a valid signal
          for 'provisional_iris_only' upstream).
    """
    h, w = frame_bgr.shape[:2]
    pad = int(round(max(oval.a_half, oval.b_half) * 1.2))
    x0 = max(0, int(round(oval.cx - pad)))
    y0 = max(0, int(round(oval.cy - pad)))
    x1 = min(w, int(round(oval.cx + pad)))
    y1 = min(h, int(round(oval.cy + pad)))
    if x1 <= x0 or y1 <= y0:
        return False, 0.0, 0.0, None, None, None, "no_iris_support"

    roi = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark = (gray <= P.DARK_V_MAX).astype(np.uint8) * 255

    # Oval mask in ROI coords.
    oval_local = OvalGeom(
        cx=oval.cx - x0, cy=oval.cy - y0,
        a_half=oval.a_half, b_half=oval.b_half, theta_deg=oval.theta_deg,
    )
    oval_mask_local = _oval_mask(roi.shape[:2], oval_local)
    # Restrict dark search to inside the oval (we don't want eyebrows above).
    dark_inside = cv2.bitwise_and(dark, dark, mask=oval_mask_local)

    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_inside, connectivity=8)
    if n_cc <= 1:
        return False, 0.0, 0.0, None, None, None, (
            "white_without_round_iris" if sclera_mask is not None else "no_iris_support"
        )

    min_area = max(20, int(round(P.IRIS_MIN_AREA_FRAC * oval.area)))
    max_area = int(round(P.IRIS_MAX_AREA_FRAC * oval.area))

    # Top-band y threshold in ROI coords: dark blob centre must NOT be in the top
    # IRIS_TOP_BAND_FRAC of the oval's vertical extent (those are eyebrow-prone).
    top_band_y_local = (oval_local.cy - oval_local.b_half) + \
                        P.IRIS_TOP_BAND_FRAC * (2.0 * oval_local.b_half)

    # Optional: dilated sclera mask in ROI coords, for adjacency test.
    sclera_local = None
    if sclera_mask is not None:
        sclera_local = sclera_mask[y0:y1, x0:x1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sclera_dilated = cv2.dilate((sclera_local > 0).astype(np.uint8) * 255, kernel, iterations=2)
    else:
        sclera_dilated = None

    best = None
    saw_top_band_only = False
    saw_low_solidity_only = False
    for idx in range(1, n_cc):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        # ---- Top-band rejection FIRST (cheap, from stats only).
        cy_local_centre = float(centroids[idx, 1])
        if cy_local_centre < top_band_y_local:
            saw_top_band_only = True
            continue
        # Only NOW build the mask, after the cheap stat-only filters pass.
        comp = (labels == idx).astype(np.uint8) * 255
        # ---- Solidity: hair-like dark strands have low solidity.
        sol = _solidity(comp)
        if sol < P.IRIS_MIN_SOLIDITY:
            saw_low_solidity_only = True
            continue
        # Iris support is expected to touch/overlap the sclera. If we have a sclera
        # mask, require the dark component to lie adjacent to it.
        if sclera_dilated is not None:
            adj = int(((comp > 0) & (sclera_dilated > 0)).sum())
            if adj < max(8, area // 20):
                continue
        # Boundary pixels of the dark component, as candidate limbus pixels.
        # Use morphological gradient (dilated XOR original) for speed.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundary = cv2.dilate(comp, kernel) ^ comp
        ys, xs = np.where(boundary > 0)
        if len(xs) < 8:
            continue
        pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        cx, cy, r, mean_resid = _kasa_circle(pts)
        if not np.isfinite(r) or r <= 0:
            continue
        # Curvature score: small residual relative to radius is good.
        norm_resid = mean_resid / max(1.0, r)
        curv = float(np.exp(-norm_resid / max(1e-3, P.CURV_RESIDUAL_SCALE)))
        # Stage-0 radius prior.
        if stage0_iris_radius is not None and stage0_iris_radius > 0:
            r_dev = abs(r - stage0_iris_radius) / max(1.0, stage0_iris_radius)
            r_score = float(np.exp(-r_dev / max(1e-3, P.RADIUS_PRIOR_SCALE)))
            curv = curv * r_score
        # Inside-oval fraction (the dark component should mostly live inside).
        inside = int(((comp > 0) & (oval_mask_local > 0)).sum())
        inside_frac = inside / max(1, area)
        # Compose this candidate's score and keep the best.
        cand_score = float(curv * inside_frac)
        if (best is None) or (cand_score > best[0]):
            # Translate fit centre back to full-frame coords.
            best = (cand_score, curv, inside_frac, float(r),
                    (float(cx + x0), float(cy + y0)),
                    pts + np.array([x0, y0], dtype=np.float32))

    if best is None:
        # Choose the most informative rejection reason.
        if saw_top_band_only and not saw_low_solidity_only:
            reason = "eyebrow_black_white_texture"
        elif saw_low_solidity_only and not saw_top_band_only:
            reason = "hair_like_white_component"   # dark-side hair
        elif sclera_mask is not None:
            reason = "white_without_round_iris"
        else:
            reason = "no_iris_support"
        return False, 0.0, 0.0, None, None, None, reason

    _, curv_best, inside_best, r_best, c_best, pts_best = best
    # If we found iris support but had NO sclera mask available, signal the
    # 'dark arc without sclera' condition so the tracker can downgrade to
    # provisional_iris_only with the right reason text.
    reason_out = "" if sclera_mask is not None else "dark_arc_without_sclera"
    return (True, float(inside_best), float(curv_best), float(r_best), c_best,
            pts_best, reason_out)


# --------------------------------------------------------------------------- top-level
def score_oval_content(frame_bgr: np.ndarray, oval: OvalGeom,
                         params: EyeContentParams = EyeContentParams,
                         stage0_iris_radius: Optional[float] = None
                         ) -> EyeContent:
    """Combine sclera + iris-support detection on a candidate oval. Returns an
    EyeContent. The combined `pair_score` is in [0, 1]; the caller multiplies by
    its own geometry score and adds penalties before deciding."""
    P = params
    result = EyeContent()

    (sclera_score, sclera_area, sclera_inside, sclera_cent, sclera_mask,
     sclera_reason) = detect_sclera_in_oval(frame_bgr, oval, P)
    result.sclera_score = float(sclera_score)
    result.sclera_area = int(sclera_area)
    result.sclera_inside_oval = float(sclera_inside)
    result.sclera_centroid = sclera_cent

    (present, iris_inside, curv, r_fit, c_fit, pts, iris_reason) = \
        detect_iris_support_in_oval(frame_bgr, oval, sclera_mask, P,
                                      stage0_iris_radius=stage0_iris_radius)
    result.iris_support_present = bool(present)
    result.iris_support_inside_oval = float(iris_inside)
    result.iris_support_curvature_score = float(curv)
    result.iris_support_radius = r_fit
    result.iris_support_centre = c_fit
    result.iris_support_pts = pts

    # Compose the rejection_reason for the candidate. Priority order:
    #   1. an explicit sclera rejection always takes precedence (the eyebrow
    #      drift case typically fails the sclera shape rules first),
    #   2. then any iris rejection,
    #   3. else empty (accepted by both).
    if sclera_reason:
        result.rejection_reason = sclera_reason
    elif iris_reason and not present:
        result.rejection_reason = iris_reason
    else:
        result.rejection_reason = ""

    # Pair proximity: distance from iris-support centre to sclera centroid, as a
    # fraction of the oval major axis a_half. Close = high score.
    prox = 0.0
    if sclera_cent is not None and c_fit is not None:
        d = float(np.hypot(c_fit[0] - sclera_cent[0], c_fit[1] - sclera_cent[1]))
        prox = float(np.exp(-(d / max(1.0, oval.a_half)) / max(1e-3, P.PAIR_PROXIMITY_SCALE)))
    result.pair_proximity_score = float(prox)

    # Combined pair score: requires BOTH sclera and iris-support to be present.
    if result.sclera_score > 0.0 and result.iris_support_present:
        result.pair_score = float(result.sclera_score *
                                    result.iris_support_curvature_score *
                                    result.pair_proximity_score)
    else:
        result.pair_score = 0.0
    return result
