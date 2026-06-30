"""Smallest iris-candidate validator (IRIS_IDENTIFICATION_RULES_V2 / ORBIT_AND_IRIS_TRACKING_RULES_V2).

The V1 tracker (`limbus.py` + `v1_tracker.py`) proposes an iris circle each frame. Until now the only
runtime gate was geometric (orbital containment + Stage-0 radius band) — it never looked at the image,
so it could accept a correctly-sized circle sitting on a dark cheek-fold, brow, or lid shadow inside the
oval. This module adds the missing *image-content* gate, enforcing the V2 clinical definition of an iris:

    > a DARK full or partial CIRCULAR / ARC structure, supported by surrounding SCLERAL WHITE,
    > lying inside the Stage-0 approved orbital oval, with a radius close to the Stage-0 limbus radius.

It is deliberately small and conservative. A false negative (freeze / "needs rescue") is acceptable;
a false positive (locking onto skin / brow / shadow) is dangerous. When in doubt it rejects.

This module does NOT track, search, or modify any tracker state. It only *scores and judges* one
candidate `(cx, cy, r)` against one frame and returns an `IrisCandidateValidation`. Wiring lives in
`iris_tracker.py`; `limbus.py` and `v1_tracker.py` are not touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- tunable thresholds
# Seed values, calibrated on the vestibular-neuritis clip. Conservative by design (reject when unsure).
RADIUS_LO = 0.55              # reject if candidate r < this × Stage-0 radius (pupil-sized / collapsed)
RADIUS_HI = 1.70              # reject if candidate r > this × Stage-0 radius (too large)
OUTSIDE_TOL_PX = 22.0         # iris centre may sit at most this far outside the orbital oval

SCLERA_MIN = 0.22             # min white-fraction on the better-supported flank to accept as sclera
SCLERA_V_MIN = 125            # a "white" pixel is brighter than this (0..255 value) ...
SCLERA_S_MAX = 70             # ... and less saturated than this (0..255) — sclera is pale, not skin-toned
CONTRAST_MIN = 10.0           # sclera_mean − iris_mean must exceed this (iris darker than its sclera)

ARC_SAMPLES = 24              # angular samples around the limbus boundary
ARC_GRAD_MIN = 10.0           # a boundary sample counts if (outside − inside) intensity exceeds this
ARC_MIN_FRACTION = 0.17       # reject if fewer than this fraction of boundary samples show dark→bright

TEMPORAL_JUMP_FRAC = 0.55     # one-frame centre jump beyond this × interocular = implausible teleport


@dataclass
class IrisCandidateValidation:
    """Verdict + component scores for one iris candidate (all scores 0..1, higher = better)."""
    accepted: bool
    reject_reason: str          # "" if accepted; else one of the reasons below
    sclera: float               # scleral-white support on the better flank
    arc: float                  # dark-inside / bright-outside circular-arc support length
    radius: float               # radius plausibility vs Stage-0 (1.0 = exact match)
    containment: float          # orbital-oval containment (1.0 = comfortably inside)
    temporal: float             # continuity vs last good centre (1.0 = unchanged)
    # raw diagnostics (not scores)
    iris_mean: float = 0.0
    sclera_mean: float = 0.0
    contrast: float = 0.0

    def log_line(self, eye_key: str, frame_no: int) -> str:
        return (f"[iris-validator][{eye_key}] fr={frame_no} reject={self.reject_reason or 'none'} "
                f"sclera={self.sclera:.2f} arc={self.arc:.2f} radius={self.radius:.2f} "
                f"containment={self.containment:.2f} temporal={self.temporal:.2f}")


# Reject-reason vocabulary (kept stable for logs / verification):
#   radius_implausible   — completed circle radius is far from the Stage-0 limbus radius
#   unstable_jump        — centre teleported from the last good frame (follow nothing; freeze)
#   no_sclera_support    — neither flank has a broad scleral-white field (brow/cheek/skin)
#   skin_fold_or_shadow  — not a dark circular structure with a bright (sclera) surround
#   outside_orbit        — content is iris-like but the centre left the orbital oval (caller decides)


def _disc_mean(gray, cx, cy, rad) -> Optional[float]:
    """Mean intensity inside a filled disc, or None if it falls outside the frame."""
    h, w = gray.shape[:2]
    rad = max(1, int(round(rad)))
    x0, y0 = int(cx - rad), int(cy - rad)
    x1, y1 = int(cx + rad) + 1, int(cy + rad) + 1
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x1), min(h, y1)
    roi = gray[ya:yb, xa:xb]
    if roi.size == 0:
        return None
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.circle(mask, (int(cx - xa), int(cy - ya)), rad, 255, -1)
    vals = roi[mask > 0]
    return float(vals.mean()) if vals.size else None


def _flank_white(hsv, cx, cy, r, sign) -> Tuple[float, float]:
    """Scleral-white assessment of the patch medial (sign=-1) or lateral (sign=+1) to the iris.
    Returns (white_fraction, mean_value). A sclera patch is broad, bright, and pale (low saturation)."""
    h, w = hsv.shape[:2]
    px = cx + sign * 1.30 * r
    bw, bh = max(3, int(0.55 * r)), max(3, int(0.80 * r))
    x0, y0 = int(px - bw / 2), int(cy - bh / 2)
    x1, y1 = x0 + bw, y0 + bh
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if xb - xa < 3 or yb - ya < 3:
        return 0.0, 0.0
    patch = hsv[ya:yb, xa:xb]
    s, v = patch[:, :, 1], patch[:, :, 2]
    white = (v >= SCLERA_V_MIN) & (s <= SCLERA_S_MAX)
    return float(white.mean()), float(v.mean())


def _arc_support(gray, cx, cy, r) -> Tuple[float, float]:
    """Fraction of limbus-boundary samples that show a dark-inside / bright-outside transition
    (iris → sclera), and the mean such gradient. Captures full circles and partial arcs alike;
    hair, lashes, skin folds and shadows do not produce a consistent circular transition."""
    h, w = gray.shape[:2]
    hits, grads = 0, []
    for k in range(ARC_SAMPLES):
        a = 2.0 * np.pi * k / ARC_SAMPLES
        ca, sa = np.cos(a), np.sin(a)
        ix, iy = cx + 0.80 * r * ca, cy + 0.80 * r * sa
        ox, oy = cx + 1.20 * r * ca, cy + 1.20 * r * sa
        if not (0 <= ix < w and 0 <= iy < h and 0 <= ox < w and 0 <= oy < h):
            continue
        inside = float(gray[int(iy), int(ix)])
        outside = float(gray[int(oy), int(ox)])
        g = outside - inside
        if g >= ARC_GRAD_MIN:
            hits += 1
            grads.append(g)
    frac = hits / float(ARC_SAMPLES)
    return frac, (float(np.mean(grads)) if grads else 0.0)


def validate_iris_candidate(gray, bgr, cx, cy, r, *, stage0_radius=0.0, orbit_poly=None,
                            last_centre=None, interocular=None) -> IrisCandidateValidation:
    """Judge one iris candidate against one frame (see module docstring / V2 rules).

    gray         : grayscale frame.
    bgr          : colour frame (for sclera saturation); may be None (whiteness falls back to value).
    cx, cy, r    : the candidate iris circle (px).
    stage0_radius: clinician-approved Stage-0 limbus radius (px); 0 disables the radius check.
    orbit_poly   : approved orbital-oval contour as an (N,1,2) float32 array; None disables containment.
    last_centre  : previous good iris centre (px) or None — for temporal continuity.
    interocular  : inter-eye distance (px) used to scale the temporal jump threshold.

    Reasons are evaluated in priority order so that `outside_orbit` is only ever returned when the
    image content is genuinely iris-like (dark arc + sclera) — letting the caller's stale-oval logic
    distinguish a head-moved frame from true drift onto the face.
    """
    cx, cy, r = float(cx), float(cy), float(r)

    # Contract: caller must supply the grayscale frame. Without it none of the image-content
    # gates (disc mean, arc support, scleral support) can run — refuse rather than guess.
    if gray is None:
        return IrisCandidateValidation(False, "missing_gray_frame", 0.0, 0.0, 0.0, 0.0, 0.0)

    # --- radius plausibility vs Stage-0 (S3) ---
    radius_score = 0.0
    if stage0_radius > 0:
        ratio = r / stage0_radius
        radius_score = max(0.0, 1.0 - abs(ratio - 1.0))
        if ratio < RADIUS_LO or ratio > RADIUS_HI:
            return IrisCandidateValidation(False, "radius_implausible", 0.0, 0.0, radius_score, 0.0, 0.0)

    # --- temporal continuity (S4): reject only egregious teleports (freeze, do not follow) ---
    temporal_score = 1.0
    if last_centre is not None and interocular and interocular > 1:
        jump = float(np.hypot(cx - last_centre[0], cy - last_centre[1]))
        temporal_score = max(0.0, 1.0 - jump / float(interocular))
        if jump > TEMPORAL_JUMP_FRAC * float(interocular):
            return IrisCandidateValidation(False, "unstable_jump", 0.0, 0.0, radius_score, 0.0,
                                           temporal_score)

    # --- image content: iris darkness, scleral support, dark circular arc ---
    iris_mean = _disc_mean(gray, cx, cy, 0.55 * r)
    if iris_mean is None:
        return IrisCandidateValidation(False, "skin_fold_or_shadow", 0.0, 0.0, radius_score, 0.0,
                                       temporal_score)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if bgr is not None else None
    if hsv is not None:
        wl, vl = _flank_white(hsv, cx, cy, r, -1)
        wr, vr = _flank_white(hsv, cx, cy, r, +1)
    else:                                   # grayscale fallback: value only, saturation unknown
        wl = wr = 0.0
        vl = vr = iris_mean
    sclera_white = max(wl, wr)
    sclera_mean = max(vl, vr)
    contrast = sclera_mean - iris_mean

    arc_frac, _arc_grad = _arc_support(gray, cx, cy, r)

    # --- scleral support (S7): at least one broad white flank ---
    if sclera_white < SCLERA_MIN:
        return IrisCandidateValidation(False, "no_sclera_support", sclera_white, arc_frac,
                                       radius_score, 0.0, temporal_score,
                                       iris_mean=iris_mean, sclera_mean=sclera_mean, contrast=contrast)

    # --- dark circular / arc geometry (S1+S2+S8): dark inside, bright (sclera) outside on an arc ---
    if contrast < CONTRAST_MIN or arc_frac < ARC_MIN_FRACTION:
        return IrisCandidateValidation(False, "skin_fold_or_shadow", sclera_white, arc_frac,
                                       radius_score, 0.0, temporal_score,
                                       iris_mean=iris_mean, sclera_mean=sclera_mean, contrast=contrast)

    # --- orbital containment (S5): only judged after content passed ---
    containment_score = 1.0
    if orbit_poly is not None:
        dist = cv2.pointPolygonTest(orbit_poly, (cx, cy), True)   # +inside, −outside (px)
        containment_score = float(np.clip(0.5 + dist / (2.0 * OUTSIDE_TOL_PX), 0.0, 1.0))
        if dist < -OUTSIDE_TOL_PX:
            return IrisCandidateValidation(False, "outside_orbit", sclera_white, arc_frac,
                                           radius_score, containment_score, temporal_score,
                                           iris_mean=iris_mean, sclera_mean=sclera_mean, contrast=contrast)

    return IrisCandidateValidation(True, "", sclera_white, arc_frac, radius_score,
                                   containment_score, temporal_score,
                                   iris_mean=iris_mean, sclera_mean=sclera_mean, contrast=contrast)
