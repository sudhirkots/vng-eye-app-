"""Weighted multi-feature composite iris tracker — EyeVNG V1 engine.

This module implements the engine specified in EYEVNG_TRACKING_SPECIFICATION.md
(the approved contract). It is written *from* that document; section references
below (§A..§J) point at the spec. No architecture is introduced here that is not
in the spec; parameter SEED values (§J) live in the `Params` block and are meant to
be calibrated — not redesigned — on the reference clips.

The engine tracks ONE eye as a dynamic pool of weighted anatomical texture features
inside the iris. Each frame the iris centre is the robust weighted consensus of the
trusted features (a composite, not a dictator). Drift is detected by composite
disagreement, never by a single template/correlation score. Blink/occlusion/drift
frames emit a gap (centre = None) and are never interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.core.limbus import fit_limbus, _fit_centre_fixed_r

# --------------------------------------------------------------------------- §B states
# Tracker state machine (spec §B). VALID OUTPUT states emit a clinical centre; all
# others emit a gap (centre = None).
INITIALIZING = "INITIALIZING"
TRACKING = "TRACKING"
PARTIAL_OCCLUSION = "PARTIAL_OCCLUSION"
BLINK = "BLINK"
REPLENISHING_FEATURES = "REPLENISHING_FEATURES"
DRIFT_SUSPECTED = "DRIFT_SUSPECTED"
TRACK_LOST = "TRACK_LOST"
REACQUIRING = "REACQUIRING"

VALID_STATES = frozenset({TRACKING, PARTIAL_OCCLUSION, REPLENISHING_FEATURES, INITIALIZING})

# per-feature trust states (spec §C / design §5)
PROBATION = "PROBATION"
TRUSTED = "TRUSTED"
SUSPECT = "SUSPECT"
LOST = "LOST"

# drift reasons (spec §F / §G) — the only permitted values
DRIFT_REASONS = ("composite_disagreement", "feature_divergence", "feature_loss",
                 "outside_aperture", "low_consensus",
                 "limbus_unfit",        # the iris–sclera boundary could not be confidently fit
                 "limbus_disagrees")    # the fitted iris disc and the feature consensus disagree


# --------------------------------------------------------------------------- §J parameters
# Seed values. Those marked (§6) are taken from MULTIFEATURE_TRACKER_DESIGN.md §6;
# the remainder are calibratable seeds chosen here (the spec leaves their numeric
# values to calibration, not redesign — §J).
DRIFT_REASONS = (
    "composite_disagreement",
    "feature_divergence",
    "feature_loss",
    "outside_aperture",
    "low_consensus",
    "limbus_unfit",
    "limbus_disagrees",
)


class Params:
    POOL_TARGET = 40                 # (§6) desired feature count
    QUORUM = 5                       # (§6) minimum trusted inliers to emit a centre
    TRUST_AGE = 8                    # (§6) frames of high-conf+inlier before PROBATION→TRUSTED
    CONF_FLOOR = 0.30                # (§6) discard below this confidence
    FB_REJECT = 2.5                  # (§6) px forward-backward LK error reject threshold. 1.0 was too
                                     # tight: during a fast nystagmus/head sweep most features briefly
                                     # exceed 1px FB error and the composite collapses below QUORUM
                                     # within ~3 frames (measured on the fistula clip). Loosened so the
                                     # composite survives the fast motion it exists to measure.
    RESID_MAX_FRAC = 0.15            # (§6) median residual cap as a fraction of iris radius
    RESID_MAX_ABS = 6.0              # absolute px floor for the residual cap (seed)
    INLIER_MIN = 0.60                # (§6) minimum trusted inlier fraction
    K_BAD = 3                        # (§6) consecutive bad frames → LOST
    EMA_ALPHA = 0.20                 # (§6) confidence EMA rate
    REPLENISH_FRAC = 0.70            # (§6) replenish when active < this × POOL_TARGET
    REANCHOR_CADENCE = 15            # (§6) re-anchor every N high-agreement frames

    DRIFT_PERSIST = 6                # (seed §B) drift frames before → REACQUIRING
    APERTURE_MARGIN = 0.25           # (seed §F) eye-local box overflow allowed before outside_aperture
    REACQUIRE_TIMEOUT = 45           # (seed §B) frames of failed recovery before → TRACK_LOST

    # limbus (iris-boundary) gate — the iris DISC is the clinical object (hybrid A+B). The composite
    # features supply motion; the limbus fit pins the disc to the real boundary each frame.
    LIMBUS_MIN_FRAC = 0.45           # min fraction of edge points agreeing with the fitted circle
    LIMBUS_DISAGREE_FRAC = 0.45      # max |limbus centre − feature centre| as a fraction of iris r
    LIMBUS_R_SMOOTH = 0.20           # EMA on the iris radius (tracks slow zoom; only from clear frames)
    LIMBUS_COV_GOOD = 0.45           # arc coverage above which the iris is "well seen": trust the FREE
                                     # fit + refresh the stable radius from it. NB the top-lid wedge is
                                     # always excluded, so a FULLY-open iris caps at ~0.55-0.6 coverage.
    LIMBUS_COV_MIN = 0.18            # arc coverage below which too little iris is visible → invalid

    # blink / aperture (EAR) thresholds — adaptive baseline, matching the existing
    # tracker's tuning (iris_tracking.IrisTracker): blink at 62% of the open baseline.
    EAR_BLINK_FRAC = 0.62            # below this × baseline ⇒ closed (BLINK)
    EAR_OPEN_FRAC = 0.80             # below this × baseline (but ≥ blink) ⇒ reduced (partial)
    EAR_ALPHA = 0.04                 # slow baseline EMA so a brief blink can't pull it down

    # feature detection (Shi-Tomasi inside the iris)
    DETECT_QUALITY = 0.01            # goodFeaturesToTrack qualityLevel
    DETECT_MIN_DIST = 3.0            # px minimum spacing between features
    DETECT_BLOCK = 5                 # corner block size
    IRIS_FILL_FRAC = 0.92            # detect/keep features within this × iris radius (stay off the limbus)

    # pyramidal Lucas-Kanade
    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# --------------------------------------------------------------------------- §C data structures
@dataclass
class IrisFeature:
    """A single tracked texture point inside the iris (spec §C)."""
    id: int
    pos: Tuple[float, float]
    birth_pos: Tuple[float, float]
    birth_offset: Tuple[float, float]      # birth_pos → approved iris centre at birth
    texture: float
    birth_frame: int
    confidence: float = 0.5
    age: int = 0
    fb_error: float = 0.0
    residual: float = 0.0
    state: str = PROBATION
    last_seen_frame: int = 0
    bad_streak: int = 0                     # internal: consecutive bad frames (for K_BAD)

    def contribution(self) -> Tuple[float, float]:
        """This feature's vote for the iris centre = current pos + birth offset."""
        return (self.pos[0] + self.birth_offset[0], self.pos[1] + self.birth_offset[1])


@dataclass
class Composite:
    """The trusted inlier subset and the fit it produced this frame (spec §C)."""
    contributors: List[IrisFeature] = field(default_factory=list)
    transform: Optional[np.ndarray] = None        # 2x3 similarity birth→current
    inlier_fraction: float = 0.0
    median_residual: float = 0.0
    n_inliers: int = 0
    rotation: float = 0.0
    scale: float = 1.0
    centre: Optional[Tuple[float, float]] = None


@dataclass
class FrameMeasurement:
    """The per-frame result for one eye (spec §C / §G)."""
    frame_number: int
    timestamp_ms: int
    time_sec: float
    state: str
    iris_centre: Optional[Tuple[float, float]] = None    # clinical centre, or None on a gap
    eye_local: Tuple[Optional[float], Optional[float]] = (None, None)
    iris_radius: Optional[float] = None
    rotation: float = 0.0
    feature_confidence_mean: float = 0.0
    composite_confidence: float = 0.0
    frame_confidence: float = 0.0
    drift_flag: bool = False
    drift_reason: str = ""
    n_active: int = 0
    n_trusted: int = 0
    ear: Optional[float] = None
    limbus_inlier_fraction: float = 0.0          # quality of the iris-boundary fit this frame
    limbus_ellipse: Optional[tuple] = None       # fitted iris ellipse for the overlay, or None
    limbus_arc_coverage: float = 0.0             # fraction of the iris circle actually visible (0..1)
    limbus_arc_pts: Optional[object] = None      # the visible iris-boundary arc points (for overlay)
    limbus_radius_fixed: bool = False            # centre fit at a held radius (partial-occlusion mode)
    raw_iris_centre: Optional[Tuple[float, float]] = None
    iris_dx_from_initial: Optional[float] = None
    iris_dy_from_initial: Optional[float] = None
    iris_valid: bool = False
    reference_valid: bool = False
    clinical_relative_valid: bool = False
    reference_uncertain: bool = True
    reference_state: str = "lost"
    reference_confidence: float = 0.0

    @property
    def validity(self) -> str:
        return "valid" if (self.state in VALID_STATES and self.iris_centre is not None) else "gap"

    @property
    def flags(self) -> List[str]:
        out = []
        if self.state == PARTIAL_OCCLUSION:
            out.append("partial_occlusion")
        if self.state == REPLENISHING_FEATURES:
            out.append("replenishing")
        if self.drift_flag:
            out.append("drift")
        if self.state == REACQUIRING:
            out.append("reacquiring")
        return out


# --------------------------------------------------------------------------- helpers
def project_eye_local(iris, inner, outer, upper, lower):
    """Iris position in EYE-LOCAL aperture coordinates (spec §A.3). Uses only this eye's
    four aperture corners, so it is independent of head/camera translation.
        x: 0 = inner canthus → 1 = outer canthus
        y: 0 = upper margin  → 1 = lower margin
    Returns (x, y); either is None if its landmark pair is missing/degenerate or the iris
    is invalid. Identical projection to iris_tracker.eye_local (kept in lock-step)."""
    if not iris or iris[0] is None:
        return None, None

    def proj(a, b):
        if a is None or b is None:
            return None
        ax, ay = b[0] - a[0], b[1] - a[1]
        L2 = ax * ax + ay * ay
        if L2 < 1.0:
            return None
        return ((iris[0] - a[0]) * ax + (iris[1] - a[1]) * ay) / L2

    return proj(inner, outer), proj(upper, lower)


def _weighted_similarity(src, dst, w):
    """Weighted similarity (uniform scale + rotation + translation) fit src→dst minimizing
    Σ w_i |s·R·src_i + t − dst_i|² (closed-form weighted Umeyama). Returns a 2×3 matrix."""
    w = np.asarray(w, float)
    sw = w.sum()
    if sw < 1e-9 or len(src) < 2:
        return None
    w = w / sw
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    sc = src - mu_s
    dc = dst - mu_d
    Sigma = (w[:, None] * dc).T @ sc            # 2×2 weighted covariance
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_s = float((w * (sc ** 2).sum(1)).sum())
    scale = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-9 else 1.0
    t = mu_d - scale * (R @ mu_s)
    M = np.zeros((2, 3), float)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def _apply(M, pt):
    return (float(M[0, 0] * pt[0] + M[0, 1] * pt[1] + M[0, 2]),
            float(M[1, 0] * pt[0] + M[1, 1] * pt[1] + M[1, 2]))


def _circle_mask(shape, centre, radius, exclude=None, exclude_r=4.0):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(round(centre[0])), int(round(centre[1]))), int(max(2, radius)), 255, -1)
    if exclude:
        for ex in exclude:
            cv2.circle(mask, (int(round(ex[0])), int(round(ex[1]))), int(exclude_r), 0, -1)
    return mask


# --------------------------------------------------------------------------- the engine (§B, §D)
class CompositeFeatureTracker:
    """Per-eye weighted multi-feature composite tracker (spec §B–§F). One instance tracks
    one eye; binocular work runs one independent instance per eye (spec §H), with no
    cross-coupling.

    Construction seeds the pool inside the Stage-0 approved iris on the approved init frame
    (INITIALIZING → TRACKING). Thereafter `step()` advances one frame and returns a
    `FrameMeasurement`.
    """

    def __init__(self, eye: str, approved_centre, approved_radius, init_gray, init_frame_number,
                 params: Params = Params):
        self.eye = eye
        self.P = params
        self.approved_centre = (float(approved_centre[0]), float(approved_centre[1]))
        self.approved_radius = float(approved_radius)
        self.resid_max = max(self.P.RESID_MAX_ABS, self.P.RESID_MAX_FRAC * self.approved_radius)

        self.pool: List[IrisFeature] = []
        self._next_id = 0
        self.prev_gray = init_gray
        self.init_frame = int(init_frame_number)

        self.last_transform: Optional[np.ndarray] = None    # last good similarity (birth→cur)
        self.last_centre = self.approved_centre             # last trusted iris centre (px)
        self.last_radius = self.approved_radius
        self.iris_radius_est: Optional[float] = None        # stable iris radius from well-seen frames;
                                                            # held constant when a lid partly covers the iris

        self.ear_base: Optional[float] = None
        self.drift_run = 0                                   # consecutive DRIFT_SUSPECTED frames
        self.reacq_run = 0                                   # consecutive REACQUIRING frames
        self.high_agree = 0                                  # high-agreement frames since re-anchor
        self.state = INITIALIZING

        # seed the rigid reference on the approved init frame
        n = self._seed(init_gray, self.last_centre, self.approved_radius, self.init_frame,
                       as_trusted=True)
        self.state = TRACKING if n >= self.P.QUORUM else TRACK_LOST

    # ---- feature pool primitives -------------------------------------------------
    def _detect(self, gray, centre, radius, exclude):
        """Shi-Tomasi corners with min-eigenvalue texture, inside the iris boundary only."""
        r = self.P.IRIS_FILL_FRAC * radius
        mask = _circle_mask(gray.shape, centre, r, exclude, self.P.DETECT_MIN_DIST)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=2 * self.P.POOL_TARGET, qualityLevel=self.P.DETECT_QUALITY,
            minDistance=self.P.DETECT_MIN_DIST, mask=mask, blockSize=self.P.DETECT_BLOCK,
            useHarrisDetector=False)
        if corners is None:
            return []
        eig = cv2.cornerMinEigenVal(gray, self.P.DETECT_BLOCK)
        out = []
        for c in corners.reshape(-1, 2):
            x, y = float(c[0]), float(c[1])
            tex = float(eig[int(round(y)), int(round(x))])
            out.append((x, y, tex))
        return out

    def _add_feature(self, x, y, tex, frame_no, centre, as_trusted=False):
        f = IrisFeature(
            id=self._next_id, pos=(x, y), birth_pos=(x, y),
            birth_offset=(centre[0] - x, centre[1] - y), texture=tex,
            birth_frame=frame_no, last_seen_frame=frame_no,
            confidence=0.6 if as_trusted else 0.45,
            state=TRUSTED if as_trusted else PROBATION)
        self._next_id += 1
        self.pool.append(f)
        return f

    def _seed(self, gray, centre, radius, frame_no, as_trusted):
        self.pool = []
        feats = self._detect(gray, centre, radius, exclude=None)
        for (x, y, tex) in feats[:self.P.POOL_TARGET]:
            self._add_feature(x, y, tex, frame_no, centre, as_trusted=as_trusted)
        return sum(1 for f in self.pool if f.state == TRUSTED) if as_trusted else len(self.pool)

    def _replenish(self, gray, centre, radius, frame_no):
        existing = [f.pos for f in self.pool if f.state in (PROBATION, TRUSTED)]
        need = self.P.POOL_TARGET - len(existing)
        if need <= 0:
            return
        feats = self._detect(gray, centre, radius, exclude=existing)
        for (x, y, tex) in feats[:need]:
            self._add_feature(x, y, tex, frame_no, centre, as_trusted=False)   # PROBATION (non-voting)

    # ---- per-frame stages --------------------------------------------------------
    def _predict_validate(self, gray, frame_no):
        """(§D.2–3) Advance every active feature by optical flow; forward-backward + bounds check."""
        active = [f for f in self.pool if f.state in (PROBATION, TRUSTED)]
        if not active or self.prev_gray is None:
            return active
        p0 = np.array([f.pos for f in active], np.float32).reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.P.LK)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self.P.LK)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        h, w = gray.shape
        for f, np_, ok1, ok2, e in zip(active, p1.reshape(-1, 2), st1.ravel(), st2.ravel(), fb):
            inb = (0 <= np_[0] < w and 0 <= np_[1] < h)
            if ok1 and ok2 and np.isfinite(e) and e <= self.P.FB_REJECT and inb:
                f.pos = (float(np_[0]), float(np_[1]))
                f.fb_error = float(e)
                f.last_seen_frame = frame_no
            else:
                f.fb_error = float(e) if np.isfinite(e) else 1e3
                if f.state == TRUSTED:
                    f.state = SUSPECT
                f.bad_streak += 1
        return [f for f in self.pool if f.state in (PROBATION, TRUSTED, SUSPECT)]

    def _compose(self) -> Composite:
        """(§D.4–5) Robust weighted consensus similarity fit of the trusted features; the iris
        centre is transform(approved_centre)."""
        com = Composite()
        trusted = [f for f in self.pool if f.state == TRUSTED]
        com.contributors = trusted
        if len(trusted) < 2:
            return com
        src = np.array([f.birth_pos for f in trusted], np.float32)
        dst = np.array([f.pos for f in trusted], np.float32)
        thr = max(2.0, 0.5 * self.resid_max)
        M, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr,
            maxIters=2000, confidence=0.99, refineIters=10)
        if M is None:
            return com
        inl = inl.ravel().astype(bool) if inl is not None else np.ones(len(trusted), bool)
        n_in = int(inl.sum())
        if n_in >= 2:
            # weighted refine over RANSAC inliers: w = conf · sat(age) · norm(texture)
            idx = np.where(inl)[0]
            tex = np.array([trusted[i].texture for i in idx])
            tex = tex / (tex.max() + 1e-9)
            # age contributes a floor of 0.3 (not 0): a fresh re-seed has age 0, and a pure age factor
            # zeroed every weight → degenerate weighted fit right after recovery. Keep a baseline.
            w = np.array([
                trusted[i].confidence * (0.3 + 0.7 * min(1.0, trusted[i].age / self.P.TRUST_AGE))
                * (0.3 + 0.7 * tx)
                for i, tx in zip(idx, tex)])
            Mw = _weighted_similarity(src[idx], dst[idx], w)
            if Mw is not None:
                M = Mw
        # residuals of all contributors vs the fit
        proj = (M[:, :2] @ src.T).T + M[:, 2]
        res = np.linalg.norm(proj - dst, axis=1)
        for f, r in zip(trusted, res):
            f.residual = float(r)
        for f in self.pool:
            if f.state in (PROBATION, SUSPECT):
                px, py = _apply(M, f.birth_pos)
                f.residual = float(np.hypot(px - f.pos[0], py - f.pos[1]))
        com.transform = M
        com.n_inliers = n_in
        com.inlier_fraction = n_in / len(trusted)
        com.median_residual = float(np.median(res[inl])) if n_in else float(np.median(res))
        com.scale = float(np.hypot(M[0, 0], M[1, 0]))
        com.rotation = float(np.arctan2(M[1, 0], M[0, 0]))
        com.centre = _apply(M, self.approved_centre)
        return com

    def _score(self, com: Composite):
        """(§D.6) Update each feature's confidence (EMA) from fb_error and residual vs consensus."""
        a = self.P.EMA_ALPHA
        for f in self.pool:
            if f.state not in (PROBATION, TRUSTED):
                continue
            g_fb = max(0.0, 1.0 - f.fb_error / self.P.FB_REJECT)
            g_res = max(0.0, 1.0 - f.residual / self.resid_max)
            g = 0.5 * g_fb + 0.5 * g_res
            f.confidence = (1 - a) * f.confidence + a * g
            f.age += 1

    def _composite_confidence(self, com: Composite) -> float:
        if com.transform is None or com.n_inliers < 2:
            return 0.0
        f_inlier = com.inlier_fraction
        f_resid = max(0.0, 1.0 - min(1.0, com.median_residual / self.resid_max))
        f_count = min(1.0, com.n_inliers / (2.0 * self.P.QUORUM))
        return float(f_inlier * f_resid * (0.5 + 0.5 * f_count))

    def _drift_reason(self, com: Composite, eye_local, ear_class) -> str:
        """(§F) Drift by composite disagreement (never a single score). Returns a reason in
        DRIFT_REASONS or "" if the frame is anatomically trustworthy."""
        if com.transform is None or com.n_inliers < 2:
            return "low_consensus"
        if com.n_inliers < self.P.QUORUM and ear_class != "closed":
            return "feature_loss"
        if com.inlier_fraction < self.P.INLIER_MIN:
            return "composite_disagreement"
        if com.median_residual > self.resid_max:
            return "feature_divergence"
        if not (0.5 < com.scale < 2.0) or abs(com.rotation) > np.radians(45):
            return "low_consensus"
        return ""

    def _maintain(self, gray, com: Composite, frame_no, allow_replenish):
        """(§D.9) Promote / demote / discard; replenish inside the iris boundary if low."""
        inlier_ids = set()
        if com.transform is not None:
            for f in self.pool:
                if f.state in (TRUSTED, PROBATION, SUSPECT) \
                        and f.fb_error <= self.P.FB_REJECT \
                        and f.residual <= max(2.0, 0.5 * self.resid_max):
                    inlier_ids.add(f.id)
        for f in list(self.pool):
            if f.state == LOST:
                self.pool.remove(f)
                continue
            good = (f.fb_error <= self.P.FB_REJECT and f.id in inlier_ids)
            if good:
                f.bad_streak = 0
            elif f.state in (TRUSTED, SUSPECT, PROBATION):
                f.bad_streak += 0  # already incremented on LK failure; residual-only badness below
                if f.id not in inlier_ids and f.state in (TRUSTED, PROBATION):
                    f.bad_streak += 1
            # promote PROBATION → TRUSTED
            if f.state == PROBATION and f.age >= self.P.TRUST_AGE \
                    and f.confidence >= self.P.CONF_FLOOR and f.id in inlier_ids:
                f.state = TRUSTED
            # recover SUSPECT → TRUSTED if it agrees again
            elif f.state == SUSPECT and f.id in inlier_ids and f.confidence >= self.P.CONF_FLOOR:
                f.state = TRUSTED
                f.bad_streak = 0
            # demote / discard
            if f.confidence < self.P.CONF_FLOOR or f.bad_streak >= self.P.K_BAD:
                f.state = LOST
                self.pool.remove(f)
        active = sum(1 for f in self.pool if f.state in (PROBATION, TRUSTED))
        replenished = False
        if allow_replenish and active < self.P.REPLENISH_FRAC * self.P.POOL_TARGET:
            centre = com.centre or self.last_centre
            self._replenish(gray, centre, self.last_radius, frame_no)
            replenished = True
        return replenished

    def _reanchor(self, com: Composite, anchor):
        """(§D.12) Slowly reconcile the feature consensus toward the per-frame IMAGE IRIS BOUNDARY (the
        limbus disc centre) — a bounded correction against optical-flow creep, every REANCHOR_CADENCE
        high-agreement frames. NOTE: the anchor is the limbus centre, NOT MediaPipe — MediaPipe never
        influences the per-frame clinical centre (MediaPipe audit rule: init / recovery / sanity only)."""
        if anchor is None or com.centre is None:
            return
        self.high_agree += 1
        if self.high_agree < self.P.REANCHOR_CADENCE:
            return
        self.high_agree = 0
        dx, dy = anchor[0] - com.centre[0], anchor[1] - com.centre[1]
        # only correct small, plausible creep (well within one iris radius)
        if np.hypot(dx, dy) > 0.5 * self.approved_radius:
            return
        k = 0.10                                   # conservative fractional pull
        for f in self.pool:
            f.birth_offset = (f.birth_offset[0] + k * dx, f.birth_offset[1] + k * dy)

    def _ear_class(self, ear) -> str:
        """Adaptive blink classification: open / reduced / closed / unknown."""
        if ear is None:
            return "unknown"
        if self.ear_base is None:
            self.ear_base = ear
        if ear >= self.P.EAR_OPEN_FRAC * self.ear_base:
            self.ear_base = (1 - self.P.EAR_ALPHA) * self.ear_base + self.P.EAR_ALPHA * ear
        if ear < self.P.EAR_BLINK_FRAC * self.ear_base:
            return "closed"
        if ear < self.P.EAR_OPEN_FRAC * self.ear_base:
            return "reduced"
        return "open"

    # ---- public step -------------------------------------------------------------
    def step(self, gray, frame_no, timestamp_ms, time_sec, ear, aperture, mp_iris=None,
             bgr=None, reference_state="lost", reference_confidence=0.0) -> FrameMeasurement:
        """Advance one frame and return a FrameMeasurement (spec §C/§D/§G).

        gray       : grayscale frame.
        ear        : eye-aspect-ratio (blink signal) for this eye, or None.
        aperture   : (inner, outer, upper, lower) corner points (px) or None each — for eye-local.
        mp_iris    : (cx, cy, r) MediaPipe iris this frame, or None — used ONLY for re-anchor/recovery.
        bgr        : the colour frame — enables the limbus (iris-boundary) fit + sclera whiteness gate.
                     The fitted iris DISC is the clinical object; the features supply motion only.
        """
        fm = FrameMeasurement(frame_no, int(timestamp_ms), float(time_sec), self.state, ear=ear,
                              reference_state=reference_state,
                              reference_confidence=float(reference_confidence or 0.0))
        ear_class = self._ear_class(ear)

        # ---- recovery states: BLINK / TRACK_LOST / REACQUIRING (gap output) -------
        if self.state in (BLINK, TRACK_LOST, REACQUIRING):
            self._handle_recovery(gray, frame_no, ear_class, mp_iris, aperture, fm)
            self.prev_gray = gray
            return self._finish(fm, n_active=self._n_active(), n_trusted=self._n_trusted(),
                                composite_confidence=0.0, feature_conf=self._feature_conf_mean())

        # ---- normal tracking path (§D.2–12) ---------------------------------------
        self._predict_validate(gray, frame_no)
        com = self._compose()

        # CFT is a helper: it predicts where to look for the limbus. The clinical centre is the
        # centre of the estimated full iris circle/ellipse fitted from the visible limbus.
        centre = com.centre
        limbus_prior = centre or self.last_centre
        el = project_eye_local(limbus_prior, *(aperture or (None, None, None, None))) if limbus_prior else (None, None)

        self._score(com)
        cc = self._composite_confidence(com)
        cft_reason = self._drift_reason(com, el, ear_class)
        reason = ""

        # ---- blink / occlusion classification (§D.8) ------------------------------
        # A blink HIDES the iris, so a true blink shows BOTH signs at once: the eyelid closing lowers
        # the EAR, AND the composite loses the iris features it was tracking (quorum lost). We require
        # BOTH. A low EAR *alone* is NOT a blink — a large gaze excursion (the eye driven hard to one
        # side, exactly what nystagmus produces) and noisy MediaPipe lid landmarks both drop the EAR
        # while the iris stays fully visible and the composite stays coherent; those frames must keep
        # tracking, not freeze as a fake blink. (The earlier EAR-alone test produced 163 false blinks
        # on the clean 2.mp4 and stalled the fistula clip on its gaze excursions.) If the eye truly
        # shut but optical flow rode the closing lid and held quorum, the consensus centre leaves the
        # iris and is caught by the drift guard (§F) / anatomical guards below — never invented.
        quorum_ok = com.n_inliers >= self.P.QUORUM
        iris_lost = not quorum_ok and limbus_prior is None
        if iris_lost and ear_class in ("closed", "reduced"):
            self.state = BLINK
            self.drift_run = 0
            fm.state = BLINK
            self.prev_gray = gray
            return self._finish(fm, self._n_active(), self._n_trusted(), cc, self._feature_conf_mean())

        # ---- limbus (iris-boundary) fit — the clinical iris DISC (hybrid A+B) ------
        # The composite features above provide MOTION; here we fit the iris–sclera boundary near that
        # centre so the emitted disc is pinned to the real iris each frame (not to internal dots, the
        # lid or the cheek). The fitted circle is the clinical object; if it can't be fit, or it
        # disagrees with the feature consensus, the frame is drift_suspected and dropped (spec hybrid
        # C/D — never rescued). The fit is seeded by the composite centre, NOT MediaPipe.
        limbus = None
        if limbus_prior is not None and bgr is not None:
            r_seed = self.iris_radius_est or self.last_radius or self.approved_radius
            limbus = fit_limbus(gray, limbus_prior[0], limbus_prior[1], r_seed, bgr=bgr)
            if limbus is None or limbus.inlier_fraction < self.P.LIMBUS_MIN_FRAC:
                reason = "limbus_unfit"
            else:
                cov = limbus.arc_coverage
                # When the iris is WELL SEEN, trust the free fit and refresh the stable radius from it.
                if cov >= self.P.LIMBUS_COV_GOOD:
                    a = self.P.LIMBUS_R_SMOOTH
                    self.iris_radius_est = limbus.r if self.iris_radius_est is None \
                        else (1 - a) * self.iris_radius_est + a * limbus.r
                # When a lid PARTLY covers the iris (low coverage) but we know the radius, refit the
                # centre at that FIXED radius from the visible arc — eyelid coverage must NOT shrink the
                # disc or move the centre. Reuses the arc points already found (no second edge search).
                elif self.iris_radius_est and limbus.arc_pts is not None and len(limbus.arc_pts) >= 5:
                    R = self.iris_radius_est
                    c, inl = _fit_centre_fixed_r(limbus.arc_pts, limbus_prior, R, max(1.5, 0.05 * R))
                    if inl.sum() >= 5:
                        from src.core.limbus import LimbusFit, _arc_coverage
                        cov2 = _arc_coverage(limbus.arc_pts[inl], c[0], c[1])
                        limbus = LimbusFit(c[0], c[1], R, limbus.inlier_fraction, int(inl.sum()),
                                           limbus.n_edges, None, arc_coverage=cov2,
                                           arc_pts=limbus.arc_pts[inl], radius_fixed=True)
                        cov = cov2
                # Too little of the iris visible to trust, or the disc disagrees with the features.
                if cov < self.P.LIMBUS_COV_MIN:
                    reason = "limbus_unfit"
                elif centre is not None and np.hypot(limbus.cx - centre[0], limbus.cy - centre[1]) \
                        > self.P.LIMBUS_DISAGREE_FRAC * limbus.r:
                    reason = "limbus_disagrees"
                elif cft_reason in ("feature_loss", "low_consensus", "composite_disagreement", "feature_divergence") \
                        and limbus is None:
                    reason = cft_reason
        else:
            reason = cft_reason or "limbus_unfit"

        # ---- drift (§F) -----------------------------------------------------------
        if reason:
            self.drift_run += 1
            fm.drift_flag = True
            fm.drift_reason = reason
            if self.drift_run >= self.P.DRIFT_PERSIST:
                self.state = REACQUIRING
                self.reacq_run = 0
            else:
                self.state = DRIFT_SUSPECTED
            fm.state = self.state
            self.high_agree = 0
            self.prev_gray = gray
            return self._finish(fm, self._n_active(), self._n_trusted(), cc, self._feature_conf_mean(),
                                rotation=com.rotation, radius=self.last_radius)

        # ---- VALID frame: TRACKING / PARTIAL_OCCLUSION / REPLENISHING -------------
        self.drift_run = 0
        self.last_transform = com.transform
        self.last_centre = (limbus.cx, limbus.cy) if limbus is not None else limbus_prior
        # the emitted IRIS is the fitted limbus disc (clinical object); the feature centre supplied
        # motion + the drift cross-check. Radius tracks the (occlusion-invariant) iris radius estimate.
        disc_centre = (limbus.cx, limbus.cy)
        self.last_radius = self.iris_radius_est or limbus.r
        el = project_eye_local(disc_centre, *(aperture or (None, None, None, None)))
        # maintain the pool (replenish only while a healthy composite holds)
        replenished = self._maintain(gray, com, frame_no, allow_replenish=True)
        # Re-anchor the feature pool on the IMAGE IRIS BOUNDARY (the limbus disc), NOT MediaPipe, to
        # bound optical-flow creep. MediaPipe never touches the per-frame clinical centre (audit rule).
        self._reanchor(com, disc_centre)

        active = self._n_active()
        if ear_class == "reduced" and quorum_ok:
            self.state = PARTIAL_OCCLUSION
        elif replenished or active < self.P.REPLENISH_FRAC * self.P.POOL_TARGET:
            self.state = REPLENISHING_FEATURES
        else:
            self.state = TRACKING

        fm.state = self.state
        fm.iris_centre = disc_centre
        fm.raw_iris_centre = disc_centre
        fm.iris_dx_from_initial = disc_centre[0] - self.approved_centre[0]
        fm.iris_dy_from_initial = disc_centre[1] - self.approved_centre[1]
        fm.iris_valid = fm.validity == "valid"
        fm.reference_valid = reference_state == "tracked" and reference_confidence >= 0.35
        fm.reference_uncertain = not fm.reference_valid
        fm.clinical_relative_valid = fm.iris_valid and fm.reference_valid and el[0] is not None and el[1] is not None
        if not fm.clinical_relative_valid:
            el = (None, None)
        fm.eye_local = el
        fm.iris_radius = self.last_radius
        fm.rotation = com.rotation
        if limbus is not None:
            fm.limbus_inlier_fraction = limbus.inlier_fraction
            fm.limbus_ellipse = limbus.ellipse
            fm.limbus_arc_coverage = limbus.arc_coverage
            fm.limbus_arc_pts = limbus.arc_pts
            fm.limbus_radius_fixed = limbus.radius_fixed
        occ = 0.7 if self.state == PARTIAL_OCCLUSION else 1.0
        # Frame confidence integrates composite agreement, occlusion, AND how much of the iris boundary
        # was actually visible+fitted (arc coverage). A thin sliver of iris → low confidence.
        lq = 0.5 * limbus.inlier_fraction + 0.5 * limbus.arc_coverage
        helper = 0.5 + 0.5 * cc
        ref_q = max(0.35, min(1.0, fm.reference_confidence)) if fm.reference_valid else max(0.2, fm.reference_confidence)
        fm.frame_confidence = float(occ * min(1.0, 0.3 + lq) * helper * ref_q)
        self.prev_gray = gray
        return self._finish(fm, active, self._n_trusted(), cc, self._feature_conf_mean(),
                            keep_conf=True)

    # ---- recovery handling -------------------------------------------------------
    def _handle_recovery(self, gray, frame_no, ear_class, mp_iris, aperture, fm: FrameMeasurement):
        if self.state == BLINK:
            if ear_class == "open":
                self.state = REACQUIRING
                self.reacq_run = 0
            else:
                fm.state = BLINK
                return
        # TRACK_LOST and REACQUIRING both attempt a re-seed
        if self.state == TRACK_LOST:
            self.state = REACQUIRING
            self.reacq_run = 0

        self.reacq_run += 1
        centre = (mp_iris[0], mp_iris[1]) if mp_iris else self.last_centre
        radius = mp_iris[2] if mp_iris else self.last_radius
        # MediaPipe is the recovery anchor (spec §A.5 / §B REACQUIRING): when it is available we trust
        # it to reposition the search box and re-seed there, EVEN after a large eye/head excursion. We
        # do NOT gate the re-seed on agreement with the frozen `last_centre`: that deadlocks on a
        # fast-moving eye, because `last_centre` can only be refreshed by a successful re-seed, so once
        # MediaPipe drifts >1.5·r away it can never catch up and the tracker is stuck in REACQUIRING
        # forever (measured: ~1015/1063 frames on the fistula clip). Anchoring on MediaPipe IS the
        # "agree with the anatomy" requirement. When MediaPipe is absent we fall back to last_centre.
        feats = self._detect(gray, centre, radius, exclude=None)
        if len(feats) >= self.P.QUORUM:
            self.approved_centre = (float(centre[0]), float(centre[1]))   # re-anchor the rigid reference
            self.last_centre = self.approved_centre
            self.last_radius = float(radius)
            self._seed(gray, self.last_centre, self.last_radius, frame_no, as_trusted=True)
            self.state = TRACKING
            self.drift_run = 0
            fm.state = REACQUIRING        # this frame is still a gap; next frame emits
            return
        if self.reacq_run >= self.P.REACQUIRE_TIMEOUT:
            self.state = TRACK_LOST
        fm.state = self.state

    # ---- bookkeeping -------------------------------------------------------------
    def _n_active(self):
        return sum(1 for f in self.pool if f.state in (PROBATION, TRUSTED))

    def _n_trusted(self):
        return sum(1 for f in self.pool if f.state == TRUSTED)

    def _feature_conf_mean(self):
        cs = [f.confidence for f in self.pool if f.state == TRUSTED]
        return float(np.mean(cs)) if cs else 0.0

    def _finish(self, fm: FrameMeasurement, n_active, n_trusted, composite_confidence,
                feature_conf, rotation=0.0, radius=None, keep_conf=False):
        fm.n_active = n_active
        fm.n_trusted = n_trusted
        fm.composite_confidence = composite_confidence
        fm.feature_confidence_mean = feature_conf
        if not keep_conf:
            fm.rotation = rotation if rotation else fm.rotation
            fm.iris_radius = radius if radius is not None else fm.iris_radius
            if fm.validity == "gap":
                fm.frame_confidence = 0.0
        return fm
