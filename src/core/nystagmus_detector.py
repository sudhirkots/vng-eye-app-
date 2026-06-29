"""EyeVNG V1 nystagmus detector (EYEVNG_TRACKING_SPECIFICATION.md §K).

Consumes the contour-relative iris-centre signal (`eye_local`) of the SELECTED BEST EYE produced
by the V1 tracker, and emits per-window classifications + a per-clip summary used to drive the
overlay arrow. The detector does NOT touch any tracking output; it only adds analysis fields.

Hierarchy (spec §K):
  inputs (§K.2)     : eye_local (h,v), frame_confidence, validity per frame
  detection (§K.3)  : sliding windows → candidate fast phases → require repeated beats,
                      direction consistency, approximate rhythmicity → reject otherwise
  labelling (§K.4)  : resultant fast-phase direction → clinical anatomical label (Left/Right/
                      Up/Down/Oblique). The eye-local axes are MIRRORED for the patient's left
                      eye vs the patient's right eye; this module owns that mapping so callers
                      never see bare h/v.
  torsion (§K.5)    : torsional_status = "not_assessed" — centre motion cannot detect torsion.

No pupil tracking. No MediaPipe dependency. Pure numpy on the per-frame measurement series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- §K.4 labels
# eye_local convention (from project_eye_local + EyeOpeningContour.extents):
#   h:  0 = medial extent  → 1 = lateral extent
#   v:  0 = upper extent   → 1 = lower extent
# The PATIENT's left vs right eye flips the relationship between "lateral" and the patient's
# anatomical side (a front-facing video mirrors the patient). So image-side L (= patient's
# RIGHT eye) lateral is the patient's RIGHT side; image-side R (= patient's LEFT eye) lateral
# is the patient's LEFT side. Vertical does not flip — `v↑` is always toward the chin.
#
# The map below is the §K.4 contract: a (sign_h, sign_v) pair on the eye-local resultant maps
# to the clinical anatomical direction word + the arrow glyph. Bare h/v/L/R never escape.
_DIRECTION_TABLE = {
    # (image_side, sign_h, sign_v) -> (label, arrow, beating_direction_key)
    # `beating_direction_key` is the §K.6 enum value (snake_case, anatomical).
    # image_side 'L' = patient RIGHT eye; image_side 'R' = patient LEFT eye.
    ("L", -1,  0): ("Left-beating nystagmus",  "<-",  "left"),
    ("L", +1,  0): ("Right-beating nystagmus", "->",  "right"),
    ("R", -1,  0): ("Right-beating nystagmus", "->",  "right"),
    ("R", +1,  0): ("Left-beating nystagmus",  "<-",  "left"),
    ("L",  0, -1): ("Up-beating nystagmus",    "^",   "up"),
    ("R",  0, -1): ("Up-beating nystagmus",    "^",   "up"),
    ("L",  0, +1): ("Down-beating nystagmus",  "v",   "down"),
    ("R",  0, +1): ("Down-beating nystagmus",  "v",   "down"),
    # oblique combinations (signs of (h, v)):
    ("L", -1, -1): ("Oblique nystagmus (up-left)",    "\\^", "oblique_up_left"),
    ("L", +1, -1): ("Oblique nystagmus (up-right)",   "/^",  "oblique_up_right"),
    ("L", -1, +1): ("Oblique nystagmus (down-left)",  "/v",  "oblique_down_left"),
    ("L", +1, +1): ("Oblique nystagmus (down-right)", "\\v", "oblique_down_right"),
    ("R", -1, -1): ("Oblique nystagmus (up-right)",   "/^",  "oblique_up_right"),
    ("R", +1, -1): ("Oblique nystagmus (up-left)",    "\\^", "oblique_up_left"),
    ("R", -1, +1): ("Oblique nystagmus (down-right)", "\\v", "oblique_down_right"),
    ("R", +1, +1): ("Oblique nystagmus (down-left)",  "/v",  "oblique_down_left"),
}


# --------------------------------------------------------------------------- §K.8 parameters
class NystagmusParams:
    """Seed values per spec §K.8 — calibrated (not redesigned) on the reference clip set.

    DESIGN PRINCIPLE: tracking can be sensitive, *diagnosis must be conservative*. For V1, false
    positives are more damaging than missed subtle nystagmus. These thresholds are deliberately
    strict — they require a sustained rhythmic train of fast phases all going the same way before
    the detector will label a window positive. One or two jerks are NEVER nystagmus.

    NEGATIVE-CONTROL CALIBRATION CLIP (V1, 2026-06-27): `samples/2_short15.mp4` is the current
    no-nystagmus reference. If this clip produces a DISPLAYED clinical nystagmus label, the
    detector has failed calibration. The clip name is intentionally NOT hardcoded into runtime
    logic; suppression must come from the rules below (input validity gate, amplitude floor,
    clinical-confirmation gate), not from a filename match. See `V1_CURRENT_CHANGES_AND_RATIONALE.md`.

    The detector reads the V1 tracker output stream, so all thresholds reference the eye_local
    units (0..1 across the eye-opening contour) and frames-per-second from the clip."""
    # §K.3.1 pre-conditions
    WINDOW_SEC = 2.5                  # sliding window length (s) — longer window favours a true
                                       # repeating beat train over an isolated fast saccade.
    STEP_SEC = 0.5                    # window step (s); 0.5 s gives 4 overlapping windows / s
    MIN_VALID_FRAC = 0.70             # fraction of frames in window that must be valid
    MIN_MEAN_FRAME_CONF = 0.40        # mean frame_confidence over valid frames
    # ---- INPUT VALIDITY GATE (added 2026-06-27) ------------------------------
    # The eye-local coordinate must be anatomically plausible. A healthy contour reference puts
    # the iris-circle centre roughly between 0 and 1 along medial→lateral and upper→lower; we
    # allow a small tolerance for fast-phase excursion outside the extent points. Values well
    # outside this band mean the contour reference is mis-shaped or off (e.g. the synthetic
    # ellipse fallback when no eye-opening contour was approved), and the resulting velocity
    # peaks are mathematical artefacts, NOT clinical fast phases. We therefore reject any
    # window where most frames fall outside this band.
    EYELOCAL_RANGE_MIN = -0.5
    EYELOCAL_RANGE_MAX = 1.5
    MAX_OUT_OF_RANGE_FRAC = 0.20      # >20% out-of-range frames → reference_out_of_range
    # §K.3.2 candidate fast phases
    FAST_K_MAD = 4.0                  # |velocity| > FAST_K_MAD × MAD(velocity in window)
    FAST_ABS_FLOOR = 0.30             # |velocity| floor (eye_local units/s) — kills jitter
    MAX_FAST_DUR_FRAMES = 4           # a fast phase peak must be brief (<= ~80 ms at 50 fps)
    MIN_REFRACTORY_FRAMES = 3         # minimum gap between adjacent fast phases
    # ---- MINIMUM FAST-PHASE AMPLITUDE (added 2026-06-27) ---------------------
    # The biggest source of false positives on negative-control clips: a "fast phase" that is
    # really just a sub-pixel velocity flicker. A true nystagmus fast phase displaces the iris
    # by a visible amount of the contour width — typically 10–20%. We require AT LEAST 5%, so
    # mathematical micro-peaks in a near-static signal are rejected even when they pass the
    # velocity-MAD gate. This is the single most important rule for noise rejection.
    MIN_FAST_PHASE_AMP_EYELOCAL = 0.05
    # §K.3.3 repeated beats — V1 clinical floor: NEVER label nystagmus on fewer than 5 beats.
    # Below 5 beats in a window the detector emits "insufficient_rhythmic_beats" (negative). The
    # "preferably 6 if uncertain" rule is enforced by the confidence floor (see below): a 5-beat
    # window that has weak consistency or rhythmicity will fall under MIN_WINDOW_CONFIDENCE and
    # be rejected anyway, so in practice 6+ rhythmic beats are required for the positive label.
    MIN_BEATS_PER_WINDOW = 5
    # §K.3.4 direction consistency — most fast phases must point the same way.
    MIN_DIRECTION_CONSISTENCY = 0.80  # resultant magnitude / N
    OBLIQUE_RATIO = 0.55              # |minor| / |major| above this → oblique; otherwise pure axis
    # §K.3.5 rhythmicity — intervals must not be random.
    MAX_RHYTHM_MAD_RATIO = 0.35       # MAD(IBI) / median(IBI) <= this
    # Confidence floor for emitting a positive at the per-window tier. The CANDIDATE tier (saved
    # to JSON/CSV for debug) uses this threshold. The CLINICAL tier (shown on the overlay) uses
    # the stricter MIN_EVENT_* gates below.
    MIN_WINDOW_CONFIDENCE = 0.55
    # ---- CLINICAL CONFIRMATION GATE (added 2026-06-27) ----------------------
    # A candidate window passing the per-window gates above is recorded as a debug candidate.
    # Before it is DISPLAYED on the overlay it must also clear these event-level gates:
    #   * the event must last at least MIN_CLINICAL_EVENT_DURATION_SEC of overlapping windows,
    #   * the event's mean confidence must clear MIN_CLINICAL_EVENT_CONFIDENCE,
    #   * the event must contain at least MIN_CLINICAL_EVENT_BEATS confirmed beats.
    # These extra hurdles are the difference between "the detector saw something" (candidate)
    # and "we will display this to the clinician" (confirmed). Tracking can be sensitive,
    # CLINICAL LABELLING must be conservative.
    MIN_CLINICAL_EVENT_DURATION_SEC = 1.5
    MIN_CLINICAL_EVENT_CONFIDENCE = 0.75
    MIN_CLINICAL_EVENT_BEATS = 6


# --------------------------------------------------------------------------- data classes
@dataclass
class NystagmusWindow:
    """One sliding-window classification (spec §K.6 per-window record)."""
    window_start_frame: int
    window_end_frame: int
    window_start_sec: float
    window_end_sec: float
    nystagmus_present: bool
    beating_direction: str            # left / right / up / down / oblique_* / torsional / none
    label: str                        # user-facing string e.g. "Left-beating nystagmus"
    arrow: str                        # ASCII glyph used by the overlay (<- -> ^ v /^ \^ /v \v)
    n_beats: int = 0
    mean_beat_rate_hz: float = 0.0
    direction_consistency: float = 0.0
    rhythmicity: float = 0.0
    confidence: float = 0.0
    rejection_reason: str = ""        # only when nystagmus_present is False
    # diagnostic only — the resultant unit vector in eye_local coords (debug; not displayed)
    resultant_h: float = 0.0
    resultant_v: float = 0.0


@dataclass
class NystagmusEvent:
    """A maximal run of consecutive windows agreeing on the same beating direction.

    Two-tier output (added 2026-06-27 after the negative-control false positive):
      * Every event from window merging is a `candidate_event`.
      * Only those that ALSO clear the §K.8 MIN_CLINICAL_EVENT_* gates become a `clinical_event`
        (`clinically_confirmed=True`, `displayed_on_overlay=True`).

    Debug consumers (JSON/CSV) see all candidates; the clinical overlay shows only confirmed
    events. `rejection_reason` records WHY a candidate was not clinically confirmed."""
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    beating_direction: str
    label: str
    arrow: str
    n_windows: int = 1
    n_beats_total: int = 0
    mean_beat_rate_hz: float = 0.0
    mean_direction_consistency: float = 0.0
    mean_rhythmicity: float = 0.0
    mean_confidence: float = 0.0
    # ---- clinical confirmation tier --------------------------------------
    candidate_event: bool = True                  # every merged event is at least a candidate
    clinically_confirmed: bool = False            # cleared MIN_CLINICAL_EVENT_* gates
    displayed_on_overlay: bool = False            # always == clinically_confirmed in V1
    rejection_reason: str = ""                    # filled when not clinically_confirmed
    duration_sec: float = 0.0                     # convenience, end_sec − start_sec


@dataclass
class NystagmusReport:
    """Per-clip nystagmus_report.json payload (spec §K.6, two-tier extension 2026-06-27)."""
    analysed_eye: str                              # "Left" / "Right" / "None"
    image_side_internal: Optional[str]             # "L" / "R" / None (audit only)
    # ---- CLINICAL tier (displayed) ---------------------------------------
    clinical_nystagmus_detected: bool = False     # at least one event passed clinical gates
    clinical_overlay_silent: bool = True          # convenience: NOT clinical_nystagmus_detected
    dominant_beating_direction: str = "none"      # spec §K.6 enum (clinical level only)
    dominant_label: str = "No nystagmus detected" # user-facing
    dominant_arrow: str = ""
    # ---- CANDIDATE tier (debug, NOT displayed) ---------------------------
    candidate_events_count: int = 0
    clinical_events_count: int = 0
    candidate_events: List[NystagmusEvent] = field(default_factory=list)
    # ---- shared --------------------------------------------------------
    torsional_status: str = "not_assessed"        # V1 always
    n_windows: int = 0
    n_candidate_windows: int = 0                  # per-window passes (renamed from positive)
    parameters: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers
def _params_dict(P) -> dict:
    return {k: getattr(P, k) for k in dir(P) if not k.startswith("_") and k.upper() == k}


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))))


def _label_from_resultant(resultant: Tuple[float, float], image_side: str,
                          oblique_ratio: float) -> Tuple[str, str, str]:
    """Map a resultant eye-local fast-phase vector to (label, arrow, beating_direction_key)."""
    rh, rv = float(resultant[0]), float(resultant[1])
    ah, av = abs(rh), abs(rv)
    if ah < 1e-9 and av < 1e-9:
        return "Indeterminate beating", "?", "none"
    sgn_h, sgn_v = (1 if rh > 0 else -1) if ah > 0 else 0, (1 if rv > 0 else -1) if av > 0 else 0
    # decide whether the resultant is dominated by one axis or is oblique
    major = max(ah, av)
    minor = min(ah, av)
    if minor / max(major, 1e-9) < oblique_ratio:
        # dominant axis only — zero the minor sign
        if ah >= av:
            sgn_v = 0
        else:
            sgn_h = 0
    key = (image_side, sgn_h, sgn_v)
    if key in _DIRECTION_TABLE:
        return _DIRECTION_TABLE[key]
    return "Indeterminate beating", "?", "none"


# --------------------------------------------------------------------------- the detector
class NystagmusDetector:
    """V1 nystagmus event detector (spec §K).

    Inputs are the per-frame V1 tracker output for ONE eye: the eye_local (h, v) series, plus
    frame_confidence and validity. The detector produces:
      * a list of per-window classifications (`NystagmusWindow`),
      * a list of merged events (`NystagmusEvent`) for the overlay/report,
      * a clip-level summary (`NystagmusReport`).

    The detector NEVER modifies the input series and never invents data on gaps.
    """

    def __init__(self, image_side: str, fps: float, params: NystagmusParams = NystagmusParams):
        if image_side not in ("L", "R"):
            raise ValueError(f"image_side must be 'L' or 'R', got {image_side!r}")
        self.image_side = image_side
        self.fps = float(fps)
        self.P = params
        # cached anatomical name of the analysed eye — for the report header.
        # 'L' = image_left_eye = patient RIGHT; 'R' = image_right_eye = patient LEFT.
        self.analysed_eye = "Right" if image_side == "L" else "Left"

    # ---- per-window analysis ----------------------------------------------
    def _analyse_window(self, h: np.ndarray, v: np.ndarray, valid: np.ndarray,
                        fconf: np.ndarray, start_frame: int) -> NystagmusWindow:
        """Run the §K.3 pipeline on one window and return a NystagmusWindow."""
        P = self.P
        n = len(h)
        end_frame = start_frame + n - 1
        t0 = start_frame / self.fps
        t1 = end_frame / self.fps

        def _negative(reason, n_beats=0, rate=0.0, cons=0.0, rhy=0.0, conf=0.0,
                      rh=0.0, rv=0.0):
            return NystagmusWindow(
                window_start_frame=start_frame, window_end_frame=end_frame,
                window_start_sec=round(t0, 3), window_end_sec=round(t1, 3),
                nystagmus_present=False, beating_direction="none",
                label="No nystagmus detected", arrow="",
                n_beats=n_beats, mean_beat_rate_hz=round(rate, 2),
                direction_consistency=round(cons, 3), rhythmicity=round(rhy, 3),
                confidence=round(conf, 3), rejection_reason=reason,
                resultant_h=round(rh, 4), resultant_v=round(rv, 4))

        # ---- §K.3.1 pre-conditions ----------------------------------------
        valid_frac = float(valid.mean()) if n else 0.0
        if valid_frac < P.MIN_VALID_FRAC:
            return _negative("insufficient_signal")
        fconf_v = fconf[valid]
        mean_fc = float(fconf_v.mean()) if fconf_v.size else 0.0
        if mean_fc < P.MIN_MEAN_FRAME_CONF:
            return _negative("insufficient_signal")

        # ---- INPUT VALIDITY GATE (added 2026-06-27) -----------------------
        # Reject windows where the contour-relative coordinates are anatomically implausible.
        # This catches the negative-control false-positive class where the synthesised contour
        # is mis-shaped and h drifts to ~2.3-2.7 — the resulting velocity peaks are mathematical
        # artefacts, not clinical fast phases. We tally the fraction of in-range valid frames
        # using BOTH axes; if the majority of the window is out-of-range, the reference is unfit.
        in_range = (valid
                    & (h >= P.EYELOCAL_RANGE_MIN) & (h <= P.EYELOCAL_RANGE_MAX)
                    & (v >= P.EYELOCAL_RANGE_MIN) & (v <= P.EYELOCAL_RANGE_MAX)
                    & np.isfinite(h) & np.isfinite(v))
        out_frac = 1.0 - float(in_range.mean()) if n else 0.0
        if out_frac > P.MAX_OUT_OF_RANGE_FRAC:
            return _negative("reference_out_of_range")

        # ---- §K.3.2 candidate fast phases ---------------------------------
        # velocity between *consecutive in-range* frames only (we never bridge a gap)
        idx = np.where(in_range)[0]
        if idx.size < 4:
            return _negative("insufficient_signal")
        # build (frame_index_in_window, dh, dv) for adjacent valid pairs
        pairs = []
        for j in range(1, idx.size):
            i0, i1 = idx[j - 1], idx[j]
            if (i1 - i0) > P.MAX_FAST_DUR_FRAMES:
                continue   # too long a gap → cannot identify a brief fast phase across it
            dh = (h[i1] - h[i0]) * (self.fps / (i1 - i0))
            dv = (v[i1] - v[i0]) * (self.fps / (i1 - i0))
            pairs.append((i1, float(dh), float(dv)))
        if len(pairs) < 4:
            return _negative("insufficient_signal")

        frames = np.asarray([p[0] for p in pairs], int)
        dh = np.asarray([p[1] for p in pairs], float)
        dv = np.asarray([p[2] for p in pairs], float)
        speed = np.hypot(dh, dv)
        mad = _mad(speed)
        thresh = max(P.FAST_ABS_FLOOR, P.FAST_K_MAD * mad)

        # peak detection on the speed series — a fast phase candidate is a frame where the
        # speed is above threshold AND a local maximum (so a long ramp counts once).
        velocity_peaks = []
        last_kept = -10_000
        for k in range(len(speed)):
            if speed[k] < thresh:
                continue
            if k > 0 and speed[k] < speed[k - 1]:
                continue
            if k + 1 < len(speed) and speed[k] < speed[k + 1]:
                continue
            if frames[k] - last_kept < P.MIN_REFRACTORY_FRAMES:
                if speed[k] > speed[velocity_peaks[-1]]:
                    velocity_peaks[-1] = k
                    last_kept = frames[k]
                continue
            velocity_peaks.append(k)
            last_kept = frames[k]

        # ---- MINIMUM FAST-PHASE AMPLITUDE (added 2026-06-27) ---------------
        # A true nystagmus fast phase produces a visible displacement, not just a velocity
        # flicker. For each velocity peak, measure the actual eye-local displacement around
        # the peak frame (over a brief MAX_FAST_DUR_FRAMES interval) and reject peaks whose
        # displacement is below the floor — these are tracker jitter, not clinical beats.
        candidate_idx = []
        n_velocity_peaks = len(velocity_peaks)
        half = max(1, P.MAX_FAST_DUR_FRAMES // 2)
        for k in velocity_peaks:
            if P.MIN_FAST_PHASE_AMP_EYELOCAL > 0:
                fr_centre = frames[k]
                lo = max(0, fr_centre - 1 - half)
                hi = min(len(h) - 1, fr_centre - 1 + half)
                if lo >= hi:
                    continue
                dx = float(h[hi] - h[lo])
                dy = float(v[hi] - v[lo])
                amp = float(np.hypot(dx, dy))
                if amp < P.MIN_FAST_PHASE_AMP_EYELOCAL:
                    continue
            candidate_idx.append(k)

        n_beats = len(candidate_idx)

        # ---- §K.3.3 repeated beats ----------------------------------------
        # V1 clinical floor (`MIN_BEATS_PER_WINDOW`): below this we never label nystagmus.
        # Distinct rejection reasons make the audit trail obvious; the overlay still presents
        # all of these as "No nystagmus detected" (the detailed reason lives in JSON/CSV only).
        if n_beats < P.MIN_BEATS_PER_WINDOW:
            # If the amplitude filter stripped most/all of the velocity peaks, name that
            # specifically — the negative-control class is "lots of peaks, none big enough."
            stripped_by_amp = (P.MIN_FAST_PHASE_AMP_EYELOCAL > 0
                               and n_velocity_peaks >= P.MIN_BEATS_PER_WINDOW)
            reason = "amplitude_below_floor" if stripped_by_amp else "insufficient_rhythmic_beats"
            return _negative(reason, n_beats=n_beats)

        beat_h = dh[candidate_idx]
        beat_v = dv[candidate_idx]
        beat_frames = frames[candidate_idx]

        # ---- §K.3.4 direction consistency ---------------------------------
        # unit vectors per beat; resultant magnitude / N is the consistency metric
        amp = np.hypot(beat_h, beat_v)
        amp = np.where(amp > 1e-9, amp, 1e-9)
        uh = beat_h / amp
        uv = beat_v / amp
        rh = float(uh.mean())
        rv = float(uv.mean())
        consistency = float(np.hypot(rh, rv))

        if consistency < P.MIN_DIRECTION_CONSISTENCY:
            return _negative("direction_inconsistent", n_beats=n_beats, cons=consistency,
                             rh=rh, rv=rv)

        # ---- §K.3.5 rhythmicity -------------------------------------------
        ibis = np.diff(beat_frames).astype(float) / self.fps
        if ibis.size == 0:
            return _negative("not_rhythmic", n_beats=n_beats, cons=consistency, rh=rh, rv=rv)
        med = float(np.median(ibis))
        mad_ibi = _mad(ibis)
        rhy_score = 1.0 - min(1.0, (mad_ibi / max(med, 1e-3)) / max(P.MAX_RHYTHM_MAD_RATIO, 1e-6))
        rhy_score = max(0.0, rhy_score)
        rate_hz = 1.0 / med if med > 1e-3 else 0.0

        if mad_ibi / max(med, 1e-3) > P.MAX_RHYTHM_MAD_RATIO:
            return _negative("not_rhythmic", n_beats=n_beats, rate=rate_hz, cons=consistency,
                             rhy=rhy_score, rh=rh, rv=rv)

        # ---- §K.6 confidence (geometric mean of evidence) ------------------
        beat_score = min(1.0, n_beats / 5.0)
        conf_inputs = [consistency, rhy_score, beat_score, max(0.0, min(1.0, mean_fc))]
        # geometric mean is robust to one weak link (a single low score caps the result)
        confidence = float(np.exp(np.mean(np.log(np.clip(conf_inputs, 1e-3, 1.0)))))

        if confidence < P.MIN_WINDOW_CONFIDENCE:
            return _negative("random", n_beats=n_beats, rate=rate_hz, cons=consistency,
                             rhy=rhy_score, conf=confidence, rh=rh, rv=rv)

        label, arrow, key = _label_from_resultant((rh, rv), self.image_side, P.OBLIQUE_RATIO)
        return NystagmusWindow(
            window_start_frame=start_frame, window_end_frame=end_frame,
            window_start_sec=round(t0, 3), window_end_sec=round(t1, 3),
            nystagmus_present=True, beating_direction=key, label=label, arrow=arrow,
            n_beats=n_beats, mean_beat_rate_hz=round(rate_hz, 2),
            direction_consistency=round(consistency, 3),
            rhythmicity=round(rhy_score, 3), confidence=round(confidence, 3),
            rejection_reason="", resultant_h=round(rh, 4), resultant_v=round(rv, 4))

    # ---- main entry point --------------------------------------------------
    def analyse(self, h_series, v_series, frame_confidence, valid_mask,
                start_frame_offset: int = 1) -> Tuple[List[NystagmusWindow], List[NystagmusEvent], NystagmusReport]:
        """Run the §K detector over a complete per-frame stream and return
        (windows, merged_events, report).

        `start_frame_offset` is the frame number that index 0 in the input series corresponds to
        (the V1 tracker writes frame_number starting at 1, so the default is 1).
        """
        h = np.asarray([np.nan if x is None else float(x) for x in h_series], float)
        v = np.asarray([np.nan if x is None else float(x) for x in v_series], float)
        fc = np.asarray([0.0 if x is None else float(x) for x in frame_confidence], float)
        # valid_mask says clinical_relative_valid was true that frame; we additionally require a
        # finite h and v (gaps inside the V1 tracker leave eye_local as None).
        if valid_mask is None:
            v_arr = np.isfinite(h) & np.isfinite(v)
        else:
            v_arr = np.asarray([bool(x) for x in valid_mask], bool) & np.isfinite(h) & np.isfinite(v)

        n_total = h.size
        win = max(2, int(round(self.P.WINDOW_SEC * self.fps)))
        step = max(1, int(round(self.P.STEP_SEC * self.fps)))

        windows: List[NystagmusWindow] = []
        i = 0
        while i + win <= n_total:
            sl = slice(i, i + win)
            w = self._analyse_window(
                h[sl].copy(), v[sl].copy(),
                v_arr[sl].copy(), fc[sl].copy(),
                start_frame=i + start_frame_offset)
            windows.append(w)
            i += step
        # tail window (so the last <step frames are still considered) — only if there is data
        if n_total >= win and (n_total - win) % step != 0:
            i = n_total - win
            sl = slice(i, i + win)
            w = self._analyse_window(
                h[sl].copy(), v[sl].copy(),
                v_arr[sl].copy(), fc[sl].copy(),
                start_frame=i + start_frame_offset)
            windows.append(w)

        events = self._merge_windows(windows)
        self._classify_clinical(events)
        report = self._build_report(windows, events)
        return windows, events, report

    # ---- clinical confirmation tier ---------------------------------------
    def _classify_clinical(self, events: List["NystagmusEvent"]) -> None:
        """Promote each candidate event to a CLINICAL event only when it clears the §K.8
        MIN_CLINICAL_EVENT_* gates (duration, confidence, beat count). Events that do not
        clear the gates remain candidates with a `rejection_reason` filled in — they are
        recorded in JSON/CSV but NEVER displayed on the overlay. This is the "tracking can be
        sensitive, clinical labelling must be conservative" tier separation."""
        P = self.P
        for e in events:
            e.duration_sec = round(max(0.0, e.end_sec - e.start_sec), 3)
            reason_bits = []
            if e.duration_sec < P.MIN_CLINICAL_EVENT_DURATION_SEC:
                reason_bits.append("event_too_short")
            if e.mean_confidence < P.MIN_CLINICAL_EVENT_CONFIDENCE:
                reason_bits.append("event_confidence_below_clinical_floor")
            if e.n_beats_total < P.MIN_CLINICAL_EVENT_BEATS:
                reason_bits.append("event_beats_below_clinical_floor")
            if reason_bits:
                e.clinically_confirmed = False
                e.displayed_on_overlay = False
                e.rejection_reason = "not_clinically_confirmed: " + ", ".join(reason_bits)
            else:
                e.clinically_confirmed = True
                e.displayed_on_overlay = True
                e.rejection_reason = ""

    # ---- merging windows → events -----------------------------------------
    @staticmethod
    def _merge_windows(windows: Sequence[NystagmusWindow]) -> List[NystagmusEvent]:
        """A nystagmus EVENT is a maximal run of consecutive positive windows that AGREE on the
        beating direction. The overlay shows the event's arrow continuously for its duration."""
        events: List[NystagmusEvent] = []
        cur = None
        for w in windows:
            if not w.nystagmus_present:
                cur = None
                continue
            if cur is not None and cur.beating_direction == w.beating_direction \
                    and w.window_start_frame <= cur.end_frame + 1:
                # extend the existing event
                cur.end_frame = max(cur.end_frame, w.window_end_frame)
                cur.end_sec = max(cur.end_sec, w.window_end_sec)
                cur.n_windows += 1
                cur.n_beats_total += w.n_beats
                cur.mean_beat_rate_hz = (cur.mean_beat_rate_hz * (cur.n_windows - 1)
                                         + w.mean_beat_rate_hz) / cur.n_windows
                cur.mean_direction_consistency = (
                    cur.mean_direction_consistency * (cur.n_windows - 1)
                    + w.direction_consistency) / cur.n_windows
                cur.mean_rhythmicity = (cur.mean_rhythmicity * (cur.n_windows - 1)
                                        + w.rhythmicity) / cur.n_windows
                cur.mean_confidence = (cur.mean_confidence * (cur.n_windows - 1)
                                       + w.confidence) / cur.n_windows
            else:
                cur = NystagmusEvent(
                    start_frame=w.window_start_frame, end_frame=w.window_end_frame,
                    start_sec=w.window_start_sec, end_sec=w.window_end_sec,
                    beating_direction=w.beating_direction, label=w.label, arrow=w.arrow,
                    n_beats_total=w.n_beats, mean_beat_rate_hz=w.mean_beat_rate_hz,
                    mean_direction_consistency=w.direction_consistency,
                    mean_rhythmicity=w.rhythmicity, mean_confidence=w.confidence)
                events.append(cur)
        # round the rolling means once at the end
        for e in events:
            e.mean_beat_rate_hz = round(e.mean_beat_rate_hz, 2)
            e.mean_direction_consistency = round(e.mean_direction_consistency, 3)
            e.mean_rhythmicity = round(e.mean_rhythmicity, 3)
            e.mean_confidence = round(e.mean_confidence, 3)
        return events

    # ---- per-clip report ---------------------------------------------------
    def _build_report(self, windows: Sequence[NystagmusWindow],
                      events: Sequence[NystagmusEvent]) -> NystagmusReport:
        n_candidate = sum(1 for w in windows if w.nystagmus_present)
        clinical = [e for e in events if e.clinically_confirmed]
        # dominant direction is taken from the CLINICAL tier only — candidates never set the
        # displayed label. If no event passes the clinical gates, the report says no nystagmus.
        dom_key, dom_label, dom_arrow = "none", "No nystagmus detected", ""
        if clinical:
            score = lambda e: e.n_beats_total * max(0.001, e.mean_confidence)
            best = max(clinical, key=score)
            dom_key, dom_label, dom_arrow = best.beating_direction, best.label, best.arrow
        return NystagmusReport(
            analysed_eye=self.analysed_eye,
            image_side_internal=self.image_side,
            clinical_nystagmus_detected=bool(clinical),
            clinical_overlay_silent=not bool(clinical),
            dominant_beating_direction=dom_key,
            dominant_label=dom_label,
            dominant_arrow=dom_arrow,
            candidate_events_count=len(events),
            clinical_events_count=len(clinical),
            candidate_events=list(events),
            torsional_status="not_assessed",
            n_windows=len(windows),
            n_candidate_windows=n_candidate,
            parameters=_params_dict(self.P),
        )


# --------------------------------------------------------------------------- frame lookup
def event_for_frame(events: Iterable[NystagmusEvent], frame_no: int
                    ) -> Optional[NystagmusEvent]:
    """Return the CLINICALLY CONFIRMED NystagmusEvent (if any) active at `frame_no`. Used by the
    overlay to decide whether to draw the arrow + label this frame.

    NEGATIVE-CONTROL RULE (added 2026-06-27): only events with `clinically_confirmed=True`
    drive the overlay. Candidate events that did not clear the §K.8 MIN_CLINICAL_EVENT_* gates
    are visible in JSON/CSV but the overlay stays silent. Specifically, if no clinical event
    is active this returns None, and the caller MUST draw nothing (no arrow, no caption)."""
    for e in events:
        if not getattr(e, "displayed_on_overlay", True):
            continue
        if e.start_frame <= frame_no <= e.end_frame:
            return e
    return None
