"""Limbus (iris–sclera boundary) fitting — EyeVNG V1.

The clinical object for V1 is the WHOLE IRIS DISC, not scattered internal points. This module fits a
circle/ellipse to the visible iris–sclera boundary (the limbus) so the tracked disc stays
anatomically attached to the real iris.

Method (robust radial-edge + circle fit):
  1. Cast rays outward from a centre estimate over a band of radii around the expected iris radius.
  2. On each ray find the strongest DARK→BRIGHT transition (iris is darker than sclera) — the limbus.
  3. Collect the edge points; reject eyelid/lash outliers with a RANSAC circle fit.
  4. Refine the centre+radius on the inliers (algebraic Kasa fit); optionally fit an ellipse.

The internal composite features estimate motion; this keeps the disc pinned to the anatomy. If the
two disagree the caller flags drift_suspected (spec hybrid C/D).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LimbusFit:
    cx: float
    cy: float
    r: float
    inlier_fraction: float       # fraction of rays whose edge point fit the circle
    n_inliers: int
    n_edges: int                 # edge points found before RANSAC
    ellipse: Optional[tuple] = None   # ((cx,cy),(MA,ma),angle) if an ellipse was fit, else None
    arc_coverage: float = 0.0    # fraction of the 360° circle the visible inlier arc spans (0..1)
    arc_pts: Optional[np.ndarray] = None   # the visible inlier edge points (for the overlay)
    radius_fixed: bool = False   # True if the radius was held constant (occlusion-invariant centre)


def _kasa_circle(pts):
    """Algebraic (Kasa) circle fit: returns (cx, cy, r) minimising algebraic distance."""
    x = pts[:, 0]; y = pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(pts))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(0.0, sol[2] + cx * cx + cy * cy)))
    return float(cx), float(cy), r


def _whiteness(bgr_patch):
    """Sclera-likeness of a small BGR patch: bright (high value) AND whitish (low saturation). Lid
    skin is reddish (R≫B → high saturation) so it scores low; the white sclera scores high."""
    p = bgr_patch.reshape(-1, 3).astype(np.float32)
    mx = p.max(1); mn = p.min(1)
    val = mx.mean() / 255.0
    sat = ((mx - mn) / (mx + 1e-3)).mean()
    return val, sat


def _radial_edges(gray, cx, cy, r_est, n_rays, r_lo_f, r_hi_f, min_grad, exclude_top_deg,
                  exclude_bottom_deg=15.0, sclera_step=14.0, bgr=None, max_sat=0.45, min_val=0.32):
    """Find one limbus edge candidate per ray: the DARK→BRIGHT transition where the iris meets the
    sclera. Guards that reject eyelid / lash / lid-skin edges:
      • ASYMMETRIC lid masking — skip a wide wedge at the TOP (±exclude_top_deg around 12 o'clock,
        where the upper lid covers the iris and the detected 'edge' is really the lid margin, biasing
        the circle up) and a small wedge at the very BOTTOM (lower lashes). The visible side+bottom
        arc still fully determines the circle, which is then completed through the occluded top;
      • sclera brightening — beyond the edge the profile must brighten by ≥ `sclera_step` and stay up;
      • sclera WHITENESS (when `bgr` given) — the outer side must be whitish (low saturation, bright),
        separating a true iris→sclera edge from iris→upper-lid-skin (skin is reddish)."""
    h, w = gray.shape
    r_lo, r_hi = r_lo_f * r_est, r_hi_f * r_est
    rr = np.arange(r_lo, r_hi, 0.6)
    pts, grads = [], []
    g = cv2.GaussianBlur(gray, (0, 0), 1.4)
    for k in range(n_rays):
        ang = 2 * np.pi * k / n_rays
        # image y is DOWN: straight up (12 o'clock) = 270°, straight down (6 o'clock) = 90°.
        d_top = abs(((np.degrees(ang) - 270 + 180) % 360) - 180)
        d_bot = abs(((np.degrees(ang) - 90 + 180) % 360) - 180)
        if d_top < exclude_top_deg or d_bot < exclude_bottom_deg:
            continue
        ca, sa = np.cos(ang), np.sin(ang)
        xs = cx + rr * ca; ys = cy + rr * sa
        inb = (xs >= 1) & (xs < w - 1) & (ys >= 1) & (ys < h - 1)
        if inb.sum() < 8:
            continue
        xs, ys = xs[inb], ys[inb]
        vals = g[ys.astype(int), xs.astype(int)].astype(np.float32)
        deriv = np.gradient(vals)                        # + = brightening outward (iris→sclera)
        best_j, best_d = -1, min_grad
        for j in range(2, len(vals) - 4):
            if deriv[j] < best_d:
                continue
            inner = vals[max(0, j - 5):j].mean()         # iris side (darker)
            outer = vals[j + 1:j + 6].mean()             # sclera side (brighter, sustained)
            if outer - inner < sclera_step or outer < inner:
                continue
            if bgr is not None:                          # whiteness gate on the sclera side
                oxs = xs[j + 1:j + 6].astype(int); oys = ys[j + 1:j + 6].astype(int)
                val, sat = _whiteness(bgr[oys, oxs])
                if sat > max_sat or val < min_val:
                    continue
            best_j, best_d = j, deriv[j]
        if best_j >= 0:
            pts.append((xs[best_j], ys[best_j])); grads.append(float(best_d))
    return np.array(pts, np.float32), np.array(grads, np.float32)


def _arc_coverage(pts, cx, cy, n_bins=24):
    """Fraction of the full 360° circle that the edge points span (occupied angular bins / all bins).
    A nearly-full iris → ~1.0; a thin sliver under heavy lid cover → near 0. Drives confidence."""
    if len(pts) == 0:
        return 0.0
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    return len(set(bins.tolist())) / n_bins


def _fit_centre_fixed_r(pts, prior, R, tol):
    """Fit the iris CENTRE with the radius held at R, from the visible arc (occlusion-invariant).
    Each edge point lies on the iris circle, so it places the centre at distance R inward along its
    radial direction (estimated from the prior centre). The robust median of those votes is the
    centre. Because R is fixed, a partial bottom/side arc cannot shrink the radius OR pull the centre
    — the whole point: eyelid coverage must not move the iris centre."""
    dirs = pts - np.asarray(prior, float)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    u = dirs / np.maximum(norms, 1e-6)
    votes = pts - R * u                       # each edge's vote for the centre
    c = np.median(votes, axis=0)
    for _ in range(2):                        # 2 robust refinements
        d = np.abs(np.linalg.norm(pts - c, axis=1) - R)
        inl = d < tol
        if inl.sum() < 4:
            break
        c = np.median(votes[inl], axis=0)
    d = np.abs(np.linalg.norm(pts - c, axis=1) - R)
    inl = d < tol
    return (float(c[0]), float(c[1])), inl


def fit_limbus(gray, cx, cy, r_est, n_rays=120, r_lo_f=0.6, r_hi_f=1.45,
               min_grad=1.5, exclude_top_deg=55.0, ransac_iters=200, tol_f=0.05,
               try_ellipse=True, bgr=None, r_fixed=None) -> Optional[LimbusFit]:
    """Fit the iris–sclera boundary near (cx,cy) with expected radius r_est. Pass `bgr` (the colour
    frame) for the sclera-whiteness gate. Pass `r_fixed` to hold the iris radius CONSTANT and fit only
    the centre from the visible arc (occlusion-invariant — eyelid coverage cannot move the centre or
    shrink the disc); the radius is established by an earlier free fit when the iris is well seen.
    Returns a LimbusFit (with arc_coverage + the visible arc points), or None if too few edges."""
    pts, grads = _radial_edges(gray, cx, cy, r_est, n_rays, r_lo_f, r_hi_f, min_grad, exclude_top_deg,
                               bgr=bgr)
    n_edges = len(pts)
    if n_edges < 6:
        return None
    tol = max(1.5, tol_f * (r_fixed or r_est))

    if r_fixed is not None:
        # FIXED-radius centre fit from the visible arc (occlusion-invariant clinical centre).
        R = float(r_fixed)
        (cx2, cy2), inl = _fit_centre_fixed_r(pts, (cx, cy), R, tol)
        if inl.sum() < 5:
            return None
        r2 = R
    else:
        # FREE RANSAC circle fit — establishes the iris radius when the iris is well seen.
        best_inl, best = None, None
        rng = np.random.default_rng(0)
        idx_all = np.arange(n_edges)
        for _ in range(ransac_iters):
            s = rng.choice(idx_all, 3, replace=False)
            try:
                ccx, ccy, cr = _kasa_circle(pts[s])
            except Exception:
                continue
            if not (0.4 * r_est < cr < 1.8 * r_est):
                continue
            d = np.abs(np.hypot(pts[:, 0] - ccx, pts[:, 1] - ccy) - cr)
            inl = d < tol
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl, best = inl, (ccx, ccy, cr)
        if best is None or best_inl.sum() < 6:
            return None
        inl = best_inl
        cx2, cy2, r2 = _kasa_circle(pts[inl])            # refine on inliers
        if not (0.4 * r_est < r2 < 1.8 * r_est):
            return None

    inl_pts = pts[inl]
    n_in = int(inl.sum())
    frac = n_in / max(1, n_edges)
    coverage = _arc_coverage(inl_pts, cx2, cy2)

    ellipse = None
    if try_ellipse and r_fixed is None and len(inl_pts) >= 6:
        try:
            ellipse = cv2.fitEllipse(inl_pts.astype(np.float32))
            (ecx, ecy), (MA, ma), _ = ellipse
            if not (0.5 * r2 < MA / 2 < 2.0 * r2 and 0.5 * r2 < ma / 2 < 2.0 * r2):
                ellipse = None
        except Exception:
            ellipse = None

    return LimbusFit(cx2, cy2, r2, float(frac), n_in, n_edges, ellipse,
                     arc_coverage=float(coverage), arc_pts=inl_pts, radius_fixed=(r_fixed is not None))
