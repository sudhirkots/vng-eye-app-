"""Stable iris tracking pipeline (see STABLE_TRACKING.md).

    python iris_tracker.py <video> [--output-dir outputs] [--auto-confirm] [--review]

Default: interactive iris-CIRCLE editor (drag centre, resize radius, per eye),
then continuous template tracking with MediaPipe as backup only.
--auto-confirm : accept the auto-detected circles without a GUI (batch / headless).
--review       : open the low-confidence review screen for an existing run.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np

from dataclasses import replace

from src.core.filters import PointFilter
from src.core.eye_contour import EyeOpeningContourTracker, contour_from_approved
from src.core.iris_tracking import (APERTURE, CompositeFeatureTracker, EyeObs, FaceLandmarkTracker, IrisDetector,
                                     IrisTracker, PROBATION, TRUSTED, Track, frame_status,
                                     select_init_frame)
from src.core.nystagmus_detector import (NystagmusDetector, NystagmusParams,
                                          event_for_frame)
from src.core.iris_candidate_validator import validate_iris_candidate
from src.core.v1_tracker import V1Tracker

APERTURE_NAMES = tuple(APERTURE.keys())   # the 8 eye-aperture landmark names (4 per eye)

# Explicit display de-shimmer via a DEADBAND: hold the marker while it moves less than
# DEADBAND_PX (kills steady-eye jitter) but snap to it immediately on any larger, real movement
# (no lag — unlike smoothing). Display/traces only; the CSV keeps the RAW values.
DEADBAND_PX = 6.0

# ---- IRIS DRIFT GUARDS (anatomical; pure geometry, NO pixel thresholding) ----
# The tracked iris must stay anatomically attached. A frame failing an INVALIDATING guard is marked
# drift_suspected → its eye-local value is dropped (a gap; never plotted as false eye movement).
DRIFT_APERTURE_MARGIN = 0.25   # iris centre may sit at most this far outside the [0,1] aperture box
DRIFT_ANCHOR_FRAC = 0.90       # tracked centre must stay within this × the MediaPipe iris radius of
                               # the independent MediaPipe iris centre (which stays on the real eye)
DRIFT_EXCURSION = 0.60         # 1-frame eye-local jump above this = "review": flagged for visual
                               # confirmation but STILL PLOTTED (a real fast-phase is a valid big jump)


def _deadband(state, key, x, y, db=DEADBAND_PX):
    """Hold the previous displayed position if the new one is within db px; else snap. Resets
    on a lost frame so it never bridges a gap. No temporal lag on real movement."""
    if x is None:
        state.pop(key, None)
        return None, None
    if key in state:
        px, py = state[key]
        if (x - px) ** 2 + (y - py) ** 2 < db * db:
            return px, py              # within deadband → hold (removes shimmer)
    state[key] = (x, y)
    return x, y                        # real movement → snap immediately (no lag)

CYAN = (255, 255, 0)
GREEN = (0, 200, 0)
YELLOW = (0, 215, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
AMBER = (0, 165, 255)
STATUS_COLOR = {"initialized": CYAN, "tracked": GREEN, "reacquired": GREEN,
                "uncertain": YELLOW, "blink_or_occluded": BLUE, "lost": RED}
OVERLAY_MAX_W = 960

# ---- Eye nomenclature (image side vs anatomical patient side) ----------------------
# Internally, the trackers use the shorthand "L" and "R" — these refer to the LEFT and
# RIGHT SIDE OF THE IMAGE FRAME (image_left_eye / image_right_eye). In a standard
# front-facing video the patient is facing the camera, so the SIDES SWAP:
#   image_left_eye  → patient_right_eye
#   image_right_eye → patient_left_eye
# Eye-side labels are anatomical ONLY after applying this image-to-patient mapping.
# All clinical reporting (metadata, overlays, traces) MUST emit both labels rather than
# bare "L"/"R" so a reader can never confuse image side with anatomical side.
EYE_LABEL_CONVENTION = "front_facing_camera"
IMAGE_TO_PATIENT_EYE = {"L": "patient_right_eye", "R": "patient_left_eye"}
IMAGE_EYE_NAME = {"L": "image_left_eye", "R": "image_right_eye"}
PATIENT_EYE_SHORT = {"L": "patient R", "R": "patient L"}    # for compact overlay banners
EYE_LABEL_NOTE = ("Eye side labels are anatomical only after applying the "
                  "image-to-patient mapping. In a front-facing video the patient's "
                  "left eye appears on the RIGHT side of the image, and vice versa.")


def image_to_patient(image_eye):
    """Map an internal image-side shorthand ('L'/'R'/'both'/None) to a patient-side
    anatomical label. Returns None for 'no eye'; 'both' stays 'both'."""
    if image_eye is None:
        return None
    if image_eye == "both":
        return "both"
    return IMAGE_TO_PATIENT_EYE.get(image_eye, image_eye)


def eye_label_pair(image_eye):
    """Compact dual label for overlays and trace titles: 'image L / patient R'.
    Kept for any caller that explicitly wants the dual label. The clinical default
    is `patient_eye_label()` below — a simple anatomical 'Left' / 'Right'."""
    if image_eye in (None, "both"):
        return "both eyes" if image_eye == "both" else "no eye"
    side = "L" if image_eye == "L" else "R"
    patient = "R" if image_eye == "L" else "L"
    return f"image {side} / patient {patient}"


# Clinical-facing label for one image side: the patient's anatomical eye on that side
# of the image. In a front-facing video the patient is MIRRORED, so:
#   image_left_eye  (L) → 'Right'  (patient's right eye on the left of the image)
#   image_right_eye (R) → 'Left'   (patient's left eye on the right of the image)
# These are the ONLY labels shown to the user on the overlay/trace/HUD; ambiguous bare
# 'L' / 'R' and verbose 'image X / patient Y' strings must not appear in user-facing UI.
PATIENT_EYE_DISPLAY = {"L": "Right", "R": "Left"}


def patient_eye_label(image_eye):
    """User-facing anatomical label for an image-side shorthand. Returns 'Left' / 'Right'
    for a single eye, 'Both' for both-eye mode, or 'None' if nothing was selected."""
    if image_eye is None:
        return "None"
    if image_eye == "both":
        return "Both"
    return PATIENT_EYE_DISPLAY.get(image_eye, image_eye)

# Map the composite engine's state machine (spec §B) onto the existing display/status vocabulary so
# the overlay, traces, review and stats keep working. VALID OUTPUT states carry a position; gap states
# (BLINK/DRIFT_SUSPECTED/TRACK_LOST/REACQUIRING) carry x=None and are never plotted as eye movement.
_STATE_DISPLAY = {
    "INITIALIZING": "initialized", "TRACKING": "tracked",
    "REPLENISHING_FEATURES": "tracked", "PARTIAL_OCCLUSION": "uncertain",
    "BLINK": "blink_or_occluded", "DRIFT_SUSPECTED": "uncertain",
    "TRACK_LOST": "lost", "REACQUIRING": "blink_or_occluded",
}
_STATE_ARTIFACT = {"BLINK": "blink", "PARTIAL_OCCLUSION": "occlusion", "TRACK_LOST": "occlusion",
                   "DRIFT_SUSPECTED": "tracking_jump", "REACQUIRING": "occlusion"}


def fm_to_track(fm) -> Track:
    """Adapt a composite FrameMeasurement (spec §C) to the orchestrator's Track. The clinical centre
    is None on any gap state, so eye-local + the traces stay gaps with no interpolation."""
    if fm is None:
        return Track("lost", None, None)
    status = _STATE_DISPLAY.get(fm.state, "uncertain")
    valid = fm.validity == "valid"
    conf = ("high" if fm.frame_confidence >= 0.6 else "medium" if fm.frame_confidence >= 0.3
            else "low" if valid else "none")
    cx = fm.iris_centre[0] if fm.iris_centre else None
    cy = fm.iris_centre[1] if fm.iris_centre else None
    return Track(status, cx, cy, fm.iris_radius, conf, fm.frame_confidence,
                 raw_x=cx, raw_y=cy, artifact_type=_STATE_ARTIFACT.get(fm.state, "none"),
                 ear=fm.ear)


# ----------------------------------------------------------------------------- circle editor
def edit_circles_gui(frame, le, re):
    """Let the user drag the centre and resize the radius of each iris circle.
    Controls: drag = move centre, mouse wheel or [ ] = radius -/+, r = reset,
    x = disable this eye (one-eye analysis), Enter = accept, q = cancel.
    Returns {"L": (x,y,r) or None, "R": (x,y,r) or None}."""
    ZOOM = 4
    state, detected = {}, {}
    for key, e in (("L", le), ("R", re)):
        if e.detected:
            state[key] = [e.x, e.y, max(6.0, e.radius)]
            detected[key] = (e.x, e.y, max(6.0, e.radius))
        else:
            state[key] = None
            detected[key] = None

    drag = {"key": None}

    def make_cb(key, x0, y0):
        def cb(event, x, y, flags, param):
            if state[key] is None:
                return
            ix, iy = x0 + x / ZOOM, y0 + y / ZOOM
            if event == cv2.EVENT_LBUTTONDOWN:
                drag["key"] = key
                state[key][0], state[key][1] = ix, iy
            elif event == cv2.EVENT_MOUSEMOVE and drag["key"] == key:
                state[key][0], state[key][1] = ix, iy
            elif event == cv2.EVENT_LBUTTONUP:
                drag["key"] = None
            elif event == cv2.EVENT_MOUSEWHEEL:
                state[key][2] = max(3.0, state[key][2] + (1 if cv2.getMouseWheelDelta(flags) > 0 else -1))
        return cb

    print("Edit iris circles: drag=centre, wheel/[ ]=radius, r=reset, x=disable eye, Enter=accept, q=cancel.")
    while True:
        active = None
        for key, e in (("L", le), ("R", re)):
            if state[key] is None and detected[key] is None:
                continue
            cx, cy, r = (state[key] if state[key] else detected[key])
            crop, x0, y0 = _eye_crop(frame, cx, cy)
            if crop.size == 0:
                continue
            big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            col = GREEN if state[key] else RED
            px, py = int((cx - x0) * ZOOM), int((cy - y0) * ZOOM)
            cv2.circle(big, (px, py), int(r * ZOOM), col, 2)
            cv2.drawMarker(big, (px, py), col, cv2.MARKER_CROSS, 14, 1)
            cv2.putText(big, f"{key} {'OFF' if state[key] is None else f'r={r:.0f}'}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            win = f"{key} eye"
            cv2.imshow(win, big)
            cv2.setMouseCallback(win, make_cb(key, x0, y0))
            active = key if active is None else active
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32):
            break
        if k == ord("q"):
            cv2.destroyAllWindows(); raise SystemExit("Cancelled.")
        if k == ord("r"):
            for key in ("L", "R"):
                state[key] = list(detected[key]) if detected[key] else None
        if k in (ord("["), ord("]")) and active and state[active]:
            state[active][2] = max(3.0, state[active][2] + (1 if k == ord("]") else -1))
        if k == ord("x") and active:
            state[active] = None
    cv2.destroyAllWindows()
    return {key: (tuple(state[key]) if state[key] else None) for key in ("L", "R")}


def _eye_crop(frame, cx, cy, half=70):
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    return frame[y0:y1, x0:x1], x0, y0


FACE_CODE = {"nose_bridge_mid": "Nb", "nose_tip": "Nt", "cheek_R": "ChR", "cheek_L": "ChL",
             "tragus_R": "TrR", "tragus_L": "TrL", "outer_canthus_R": "OcR", "outer_canthus_L": "OcL",
             "inner_canthus_R": "IcR", "inner_canthus_L": "IcL",
             # eye-aperture margins (V1 eye-local landmarks)
             "upper_margin_R": "UpR", "upper_margin_L": "UpL",
             "lower_margin_R": "LoR", "lower_margin_L": "LoL"}


def propose_init(video_path, output_dir="outputs"):
    """Mark what the software thinks is the iris AND the facial landmarks, and
    save a review image so the user can confirm/correct before tracking (anatomical
    confirmation, step 1). The iris boundary (MediaPipe limbus) is drawn for the user
    to adjust in Stage 0."""
    video_path = Path(video_path)
    detector = IrisDetector()
    init = select_init_frame(detector, video_path)
    if init is None:
        raise SystemExit("No frame with a confidently detected eye — cannot initialise.")
    fr, frame, le, re = init
    # fresh detector: select_init_frame advanced this detector's MediaPipe tracking state, so
    # re-processing the (earlier) init frame with it can fail — a clean instance detects reliably.
    faces = IrisDetector().face_landmarks(frame) or {}
    out_dir = Path(output_dir) / (video_path.stem + "_tracked")
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = frame.copy()
    # iris boundary (green) + centre dot
    for e in (le, re):
        if e.detected:
            c = (int(e.x), int(e.y))
            cv2.circle(vis, c, int(e.radius), GREEN, 2)
            cv2.circle(vis, c, 2, GREEN, -1)
    # facial landmarks (amber dots + short codes); unavailable ones are skipped
    unavailable = []
    for name, pt in faces.items():
        if pt is None:
            unavailable.append(name)
            continue
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(vis, p, 4, AMBER, -1)
        cv2.putText(vis, FACE_CODE.get(name, name), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1)

    s = 1100 / frame.shape[1]
    full = cv2.resize(vis, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
    cv2.putText(full, f"Init frame {fr} - confirm iris (green) + face landmarks (amber)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    crops = []
    for key, e in (("L", le), ("R", re)):
        if not e.detected:
            continue
        crop, x0, y0 = _eye_crop(frame, e.x, e.y, half=70)
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cx, cy = int((e.x - x0) * 4), int((e.y - y0) * 4)
        cv2.circle(big, (cx, cy), int(e.radius * 4), GREEN, 2)           # iris boundary
        cv2.circle(big, (cx, cy), 3, GREEN, -1)
        cv2.putText(big, f"{key} iris r={e.radius:.0f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        crops.append(big)

    def pad_w(img, width):
        if img.shape[1] >= width:
            return img
        out = np.zeros((img.shape[0], width, 3), np.uint8)
        out[:, :img.shape[1]] = img
        return out

    panels = [full]
    if crops:
        hmax = max(c.shape[0] for c in crops)
        row = np.hstack([pad_w(cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 10, cv2.BORDER_CONSTANT),
                               c.shape[1] + 10) for c in crops])
        panels.append(row)
    width = max(p.shape[1] for p in panels)
    composed = np.vstack([pad_w(p, width) for p in panels])
    path = out_dir / "init_proposal.png"
    cv2.imwrite(str(path), composed)

    print(f"Init frame {fr}: iris boundary proposal (green) — confirm/adjust in Stage 0")
    for key, e in (("L", le), ("R", re)):
        print(f"  iris {key}: " + ("not detected" if not e.detected else
              f"centre=({e.x:.0f},{e.y:.0f}) iris_r={e.radius:.0f}"))
    print("  face landmarks: " + ", ".join(f"{FACE_CODE[k]}" for k, v in faces.items() if v is not None))
    if unavailable:
        print("  unavailable (off-frame): " + ", ".join(FACE_CODE.get(u, u) for u in unavailable))
    print(f"Saved proposal image: {path}")
    return path


# ----------------------------------------------------------------------------- Stage 0: approval
def _approved_path(output_dir, video_path):
    return Path(output_dir) / (Path(video_path).stem + "_tracked") / "approved_landmarks.json"


def load_approved(output_dir, video_path):
    p = _approved_path(output_dir, video_path)
    return json.loads(p.read_text()) if p.exists() else None


def _seed_contour_polygon(face_pts, eye_key, n=24):
    """Seed an eye-opening contour polygon for ONE eye from the four approved face landmarks
    (inner_canthus, outer_canthus, upper_margin, lower_margin). The seed is a smooth ellipse
    through the four points so the clinician usually just confirms or refines a few vertices.
    NOTE: this seed is a STARTING POINT for clinician approval, NOT a silent fallback. If
    the clinician approves the seed unchanged, it is still a clinician-approved contour.
    Returns a numpy array of (x,y) points or None if any of the four landmarks is missing."""
    nm = {t: f"{t}_{eye_key}" for t in
          ("inner_canthus", "outer_canthus", "upper_margin", "lower_margin")}
    if not all(face_pts.get(nm[t]) for t in nm):
        return None
    inner = np.asarray(face_pts[nm["inner_canthus"]], float)
    outer = np.asarray(face_pts[nm["outer_canthus"]], float)
    upper = np.asarray(face_pts[nm["upper_margin"]], float)
    lower = np.asarray(face_pts[nm["lower_margin"]], float)
    centre = (inner + outer + upper + lower) / 4.0
    width = max(10.0, float(np.hypot(*(outer - inner))))
    height = max(8.0, float(np.hypot(*(lower - upper))))
    angle = float(np.arctan2(outer[1] - inner[1], outer[0] - inner[0]))
    ca, sa = np.cos(angle), np.sin(angle)
    ts = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = []
    for t in ts:
        x = 0.5 * width * np.cos(t)
        y = 0.5 * height * np.sin(t)
        pts.append((centre[0] + ca * x - sa * y, centre[1] + sa * x + ca * y))
    return np.asarray(pts, np.float32)


def _validate_iris_seed(centre_xy, radius, interocular_px=None):
    """Sanity-check the approved iris seed. Returns (ok, reason).

    'pupil-sized' detection: a real iris radius is typically 8-14% of inter-canthal distance
    (clinical anatomy) and rarely below ~10 px on a 1080p face crop. The legacy 'pupils'
    schema produced radii around 3-8 px on the same clips, which is too small for an iris
    and indicates the clinician marked the pupil, not the limbus."""
    if centre_xy is None or radius is None:
        return False, "missing"
    if radius < 9.0:
        return False, "iris_radius_too_small_likely_pupil"
    if interocular_px and interocular_px > 50 and radius < 0.05 * interocular_px:
        return False, "iris_radius_too_small_for_anatomy"
    return True, ""


def _validate_eye_opening_contour(contour_pts, iris_centre, iris_radius):
    """Sanity-check the approved eye-opening contour for one eye.
      * must have enough points (>= 8) to count as a real contour, not four landmarks,
      * must SURROUND the iris seed (the iris centre must lie inside the polygon),
      * must be reasonably large vs the iris radius (the palpebral fissure cannot be smaller
        than the iris it shows).
    Returns (ok, reason)."""
    if contour_pts is None or len(contour_pts) < 8:
        return False, "contour_missing_or_too_few_points"
    if iris_centre is None or iris_radius is None:
        return False, "no_iris_seed_to_validate_against"
    pts = np.asarray(contour_pts, np.float32)
    # bbox span check
    span_x = float(pts[:, 0].max() - pts[:, 0].min())
    span_y = float(pts[:, 1].max() - pts[:, 1].min())
    if span_x < 2.0 * iris_radius or span_y < 1.0 * iris_radius:
        return False, "contour_too_small_for_iris"
    # iris centre must be inside the contour polygon
    inside = cv2.pointPolygonTest(pts.reshape(-1, 1, 2),
                                  (float(iris_centre[0]), float(iris_centre[1])), False)
    if inside < 0:
        return False, "iris_centre_outside_contour"
    return True, ""


def approve_interactive(frame, le, re, faces):
    """V1 Stage-0 ASSISTED ANATOMICAL APPROVAL screen (locked 2026-06-27).

    Workflow: app PROPOSES → clinician VERIFIES & CORRECTS → clinician APPROVES.

    This is NOT a blank manual drawing tool. The app proposes all three anatomical structures
    automatically; the clinician's role is to verify the proposals and correct only what is
    wrong, then press ONE Enter to APPROVE everything.

    Proposals shown on entry:
      1. Full iris / limbus circle per eye, sized to MediaPipe's iris-boundary estimate plus a
         small safety margin so the initial circle reliably covers the coloured iris out to the
         white sclera. NOT the pupil.
      2. Iris centre (centre of that circle).
      3. Eye-opening / orbital margin contour, seeded as a smooth ellipse through the four face
         landmarks (canthi + lid margins). The clinician refines it to the visible palpebral
         fissure if the seed is off.
      4. Usable-eye status — both eyes default to usable when MediaPipe detected them.

    Editing modes (cycle with Tab, also pickable with i / f / c):
      * 'i' — IRIS / LIMBUS mode: drag the centre; +/- to resize the limbus circle; d off/on.
      * 'f' — FACE LANDMARK mode: drag a canthus/lid-margin; d off/on.
      * 'c' — CONTOUR mode: drag vertices; a=add vertex under cursor; x=delete vertex;
              e=switch to the other eye's contour.
    Other keys: m = magnifier on/off; Enter = APPROVE everything; q = cancel.

    Returns (iris_dict, faces_dict, contours_dict).
    """
    H, W = frame.shape[:2]
    scale = min(1.0, 1200.0 / W)
    DW, DH = int(W * scale), int(H * scale)

    # ---- IRIS items: ONE PROPOSED CIRCLE PER EYE -------------------------------
    # The proposal is the FULL iris/limbus circle, NOT the small dark pupil. We start from
    # MediaPipe's iris-boundary estimate (already the limbus radius) and enlarge it by a
    # small safety margin so the initial display reliably covers the visible iris out to
    # the white sclera. The clinician verifies and adjusts; they do not draw from scratch.
    IRIS_PROPOSAL_RADIUS_MARGIN = 1.10   # +10% to be more likely to start on the limbus, not inside it
    iris_items = []
    for key, e, dx in (("L", le, 0.35), ("R", re, 0.65)):
        if e.detected:
            proposed_r = max(8.0, e.radius * IRIS_PROPOSAL_RADIUS_MARGIN)
            iris_items.append({"name": key, "x": e.x, "y": e.y,
                                "r": proposed_r, "on": True})
        else:
            iris_items.append({"name": key, "x": W * dx, "y": H * 0.5,
                                "r": 15.0, "on": False})
    # ---- FACE landmark items ---------------------------------------------------
    face_items = []
    for name, pt in faces.items():
        on = pt is not None
        face_items.append({"name": name,
                            "x": pt[0] if on else W * 0.5, "y": pt[1] if on else H * 0.5,
                            "on": on})
    # ---- EYE-OPENING CONTOUR: one EDITABLE OVAL per usable eye -----------------
    # Each contour is a parameterised ellipse: centre (cx, cy), half-width a (along the
    # inner-outer canthus axis), half-height b (perpendicular), rotation angle theta.
    # The clinician edits ONE oval per eye via four handles (medial, lateral, upper, lower)
    # + the centre + rotation, not a 24-vertex polygon. On APPROVE the oval is sampled at
    # 24 points and written into approved_landmarks.json as the eye_opening_contours
    # polygon — so the runtime, validator, and tracker stay unchanged.
    def _faces_dict():
        return {it["name"]: (it["x"], it["y"]) for it in face_items if it["on"]}
    contours = {}    # key -> {"cx","cy","a","b","theta"}  half-axes and rotation in radians
    for it in iris_items:
        if not it["on"]:
            continue
        fd = _faces_dict()
        nm = {t: f"{t}_{it['name']}" for t in
              ("inner_canthus", "outer_canthus", "upper_margin", "lower_margin")}
        if all(fd.get(nm[t]) for t in nm):
            inner = np.asarray(fd[nm["inner_canthus"]], float)
            outer = np.asarray(fd[nm["outer_canthus"]], float)
            upper = np.asarray(fd[nm["upper_margin"]], float)
            lower = np.asarray(fd[nm["lower_margin"]], float)
            centre = (inner + outer + upper + lower) / 4.0
            a = max(10.0, 0.5 * float(np.hypot(*(outer - inner))))
            b = max(8.0, 0.5 * float(np.hypot(*(lower - upper))))
            theta = float(np.arctan2(outer[1] - inner[1], outer[0] - inner[0]))
        else:
            # Fallback: seed an oval around the iris itself, slightly wider than tall.
            centre = np.array([it["x"], it["y"]], float)
            a = 2.6 * it["r"]
            b = 1.4 * it["r"]
            theta = 0.0
        contours[it["name"]] = {"cx": float(centre[0]), "cy": float(centre[1]),
                                  "a": float(a), "b": float(b), "theta": float(theta)}

    def _ellipse_handle_positions(e):
        """Return the four cardinal handle positions of an ellipse e in IMAGE coords:
        medial (−a), lateral (+a), upper (−b), lower (+b)."""
        cx, cy, a, b, th = e["cx"], e["cy"], e["a"], e["b"], e["theta"]
        ca, sa = np.cos(th), np.sin(th)
        return {
            "medial":  (cx - a * ca, cy - a * sa),
            "lateral": (cx + a * ca, cy + a * sa),
            "upper":   (cx - b * (-sa), cy - b * ca),    # perpendicular axis: rotate +90°
            "lower":   (cx + b * (-sa), cy + b * ca),
        }

    def _ellipse_to_polygon(e, n=24):
        """Sample an ellipse to n polygon points (the format the runtime expects)."""
        ts = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ca, sa = np.cos(e["theta"]), np.sin(e["theta"])
        pts = []
        for t in ts:
            x = e["a"] * np.cos(t)
            y = e["b"] * np.sin(t)
            pts.append((e["cx"] + ca * x - sa * y,
                        e["cy"] + sa * x + ca * y))
        return np.asarray(pts, np.float32)

    MODE_IRIS, MODE_FACE, MODE_CONTOUR = "iris", "face", "contour"
    # contour_handle ∈ {"centre","medial","lateral","upper","lower"} — names the dragged handle.
    st_ = {"mode": MODE_IRIS, "iris_sel": 0, "face_sel": 0,
           "contour_eye": next(iter(contours), "L"),
           "contour_handle": "centre",
           # Magnifier inset is OFF by default — clinicians find it distracting on the
           # large overview; the editor handles + label text are sized for the unzoomed view.
           # Toggle on/off at any time with the `m` key.
           "drag": False, "mx": 0, "my": 0, "mag": False}

    def _nearest_iris(ix, iy):
        best = None
        for i, it in enumerate(iris_items):
            d = np.hypot(it["x"] - ix, it["y"] - iy)
            if best is None or d < best[0]:
                best = (d, i)
        return best[1] if best and best[0] < 45 / scale else None

    def _nearest_face(ix, iy):
        best = None
        for i, it in enumerate(face_items):
            d = np.hypot(it["x"] - ix, it["y"] - iy)
            if best is None or d < best[0]:
                best = (d, i)
        return best[1] if best and best[0] < 45 / scale else None

    def _nearest_handle(ix, iy):
        """Find the nearest editable point (centre or one of the four cardinal handles) of
        the CURRENTLY-SELECTED eye's ellipse. Returns its name or None if too far."""
        key = st_["contour_eye"]
        if key not in contours:
            return None
        e = contours[key]
        handles = _ellipse_handle_positions(e)
        handles["centre"] = (e["cx"], e["cy"])
        best = None
        for name, (hx, hy) in handles.items():
            d = np.hypot(hx - ix, hy - iy)
            if best is None or d < best[0]:
                best = (d, name)
        return best[1] if best and best[0] < 40 / scale else None

    def _auto_select_contour_eye(ix, iy):
        """2026-06-28: in CONTOUR mode, click anywhere near an oval to make THAT oval
        active. Removes the 'why can't I move the left eye' bug: the user no longer
        needs to remember the `e` shortcut. The nearest eye (by centre distance) wins."""
        best = None
        for key, e in contours.items():
            d = np.hypot(e["cx"] - ix, e["cy"] - iy)
            if best is None or d < best[0]:
                best = (d, key)
        if best is not None:
            st_["contour_eye"] = best[1]
            st_["contour_handle"] = "centre"

    def _drag_handle(key, handle, ix, iy):
        """Update ellipse parameters when the named handle is dragged to (ix, iy)."""
        e = contours[key]
        if handle == "centre":
            e["cx"], e["cy"] = float(ix), float(iy)
            return
        # vector from centre to cursor (in image coords); decompose along (theta) and the
        # perpendicular axis so dragging a side handle only resizes that side.
        dx, dy = float(ix - e["cx"]), float(iy - e["cy"])
        ca, sa = np.cos(e["theta"]), np.sin(e["theta"])
        proj_long = dx * ca + dy * sa             # signed projection along the long axis
        proj_perp = -dx * sa + dy * ca            # signed projection along the perpendicular
        if handle == "medial":
            e["a"] = max(8.0, -proj_long)         # medial handle is on the -a side
        elif handle == "lateral":
            e["a"] = max(8.0,  proj_long)         # lateral handle is on the +a side
        elif handle == "upper":
            e["b"] = max(6.0, -proj_perp)
        elif handle == "lower":
            e["b"] = max(6.0,  proj_perp)

    def on_mouse(ev, x, y, flags, _):
        st_["mx"], st_["my"] = x, y
        ix, iy = x / scale, y / scale
        if ev == cv2.EVENT_LBUTTONDOWN:
            if st_["mode"] == MODE_IRIS:
                i = _nearest_iris(ix, iy)
                if i is not None:
                    st_["iris_sel"], st_["drag"] = i, True
                    iris_items[i]["on"] = True
            elif st_["mode"] == MODE_FACE:
                i = _nearest_face(ix, iy)
                if i is not None:
                    st_["face_sel"], st_["drag"] = i, True
                    face_items[i]["on"] = True
            elif st_["mode"] == MODE_CONTOUR:
                # Auto-switch active eye to whichever oval's centre is closer to the
                # click BEFORE looking for a handle. This fixes the 'left eye contour
                # won't move' bug — previously _nearest_handle only searched the
                # currently-selected eye, so a click on the OTHER eye's oval did
                # nothing.
                _auto_select_contour_eye(ix, iy)
                h = _nearest_handle(ix, iy)
                if h is not None:
                    st_["contour_handle"], st_["drag"] = h, True
        elif ev == cv2.EVENT_MOUSEMOVE and st_["drag"]:
            if st_["mode"] == MODE_IRIS:
                iris_items[st_["iris_sel"]].update(x=ix, y=iy)
            elif st_["mode"] == MODE_FACE:
                face_items[st_["face_sel"]].update(x=ix, y=iy)
            elif st_["mode"] == MODE_CONTOUR:
                key = st_["contour_eye"]
                if key in contours:
                    _drag_handle(key, st_["contour_handle"], ix, iy)
        elif ev == cv2.EVENT_LBUTTONUP:
            st_["drag"] = False

    win = "Stage 0 - V1 anatomical approval"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    # 2026-06-28: clinician decision — Stage 0 is iris + orbital oval only. The face
    # landmarks still get detected by MediaPipe and still seed the initial oval, but
    # they are no longer a clinician-editable mode. Tab now cycles IRIS<->CONTOUR.
    HUD = ("VERIFY & CORRECT - modes: [i]ris/limbus  [c]ontour-oval  (Tab cycles)    "
           "i: drag centre, +/- resize, d off/on   "
           "c: click ANY eye to select it, drag centre or cardinal handles, "
           "+/- resize active axis, r/R rotate   "
           "m magnifier   Enter = APPROVE everything   q = cancel")
    while True:
        disp = cv2.resize(frame, (DW, DH))
        # IRIS circles — always drawn; selected one cyan when in IRIS mode
        for i, it in enumerate(iris_items):
            sel = (i == st_["iris_sel"] and st_["mode"] == MODE_IRIS)
            col = (0, 255, 255) if sel else ((0, 200, 0) if it["on"] else (130, 130, 130))
            p = (int(it["x"] * scale), int(it["y"] * scale))
            cv2.circle(disp, p, max(3, int((it["r"] or 10) * scale)), col, 2, cv2.LINE_AA)
            cv2.circle(disp, p, 2, col, -1)
            lbl = f"{it['name']}-iris r{int(it['r'])}{'' if it['on'] else ' OFF'}"
            cv2.putText(disp, lbl, (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        # FACE landmarks are NOT drawn in the UI any more (Stage 0 is iris + orbital
        # oval only). The face_items still exist so the contour seeding code (above)
        # works and so faces_out below saves into approved_landmarks.json for the V1
        # default engine. They are simply invisible to the clinician.
        # EYE-OPENING CONTOURS — drawn as an OVAL per eye using cv2.ellipse.
        # In contour mode the active eye also shows 5 draggable handles (centre + 4 cardinal).
        for key, e in contours.items():
            is_active = (st_["mode"] == MODE_CONTOUR and key == st_["contour_eye"])
            col = (255, 255, 0) if is_active else \
                  ((180, 180, 0) if st_["mode"] == MODE_CONTOUR else (140, 140, 0))
            cv2.ellipse(disp,
                        (int(e["cx"] * scale), int(e["cy"] * scale)),
                        (max(2, int(e["a"] * scale)), max(2, int(e["b"] * scale))),
                        float(np.degrees(e["theta"])), 0, 360, col,
                        2 if is_active else 1, cv2.LINE_AA)
            if st_["mode"] == MODE_CONTOUR:
                handles = _ellipse_handle_positions(e)
                handles["centre"] = (e["cx"], e["cy"])
                # Cardinal handles are square; centre is a diamond. Selected handle is cyan.
                for name, (hx, hy) in handles.items():
                    pp = (int(hx * scale), int(hy * scale))
                    sel = (is_active and name == st_["contour_handle"])
                    hcol = (0, 255, 255) if sel else col
                    if name == "centre":
                        cv2.drawMarker(disp, pp, hcol, cv2.MARKER_DIAMOND,
                                       12 if sel else 9, 2, cv2.LINE_AA)
                    else:
                        sz = 5 if sel else 4
                        cv2.rectangle(disp, (pp[0] - sz, pp[1] - sz),
                                      (pp[0] + sz, pp[1] + sz), hcol, -1)
                    if is_active:
                        cv2.putText(disp, name[:3], (pp[0] + 7, pp[1] - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, hcol, 1, cv2.LINE_AA)
        # MAGNIFIER inset
        ix, iy = int(st_["mx"] / scale), int(st_["my"] / scale)
        Z, rad = 4, 36
        x0, y0 = max(0, ix - rad), max(0, iy - rad)
        x1, y1 = min(W, ix + rad), min(H, iy + rad)
        crop = frame[y0:y1, x0:x1]
        if st_["mag"] and crop.size:
            mag = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(mag, (int((ix - x0) * Z), int((iy - y0) * Z)),
                           (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
            mh, mw = mag.shape[:2]
            if mw <= DW and mh <= DH:
                cv2.rectangle(mag, (0, 0), (mw - 1, mh - 1), (0, 0, 255), 1)
                mx0 = 0 if st_["mx"] > DW / 2 else DW - mw
                my0 = (DH - mh) if st_["my"] < DH / 2 else 0
                disp[my0:my0 + mh, mx0:mx0 + mw] = mag
        # MODE banner + instruction line
        # On-screen wording reflects the propose-verify-approve workflow: the app HAS proposed
        # the anatomy; the clinician VERIFIES (and corrects only what's wrong) before APPROVE.
        mode_lbl = {
            MODE_IRIS:    "VERIFY: proposed full iris / limbus circle. It should cover the "
                          "coloured iris out to the white sclera. Do not mark the pupil.",
            MODE_FACE:    "",   # face mode is hidden from the UI as of 2026-06-28
            MODE_CONTOUR: f"VERIFY: proposed eye-opening / orbital margin OVAL "
                          f"({st_['contour_eye']} eye, click the other eye's oval to switch). "
                          "Drag centre or any cardinal handle to reshape.",
        }[st_["mode"]]
        cv2.rectangle(disp, (0, 0), (DW, 28), (0, 0, 0), -1)
        cv2.putText(disp, mode_lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, HUD, (8, DH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):
            break
        if k == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled — nothing approved.")
        # mode switch — FACE mode is now hidden from the UI per 2026-06-28 decision
        # (Stage 0 is iris + orbital oval only). `f` is a no-op so accidental presses
        # are harmless; Tab cycles only IRIS <-> CONTOUR.
        if k == ord("i"): st_["mode"] = MODE_IRIS
        elif k == ord("c"): st_["mode"] = MODE_CONTOUR
        elif k == 9:  # Tab cycles mode
            order = [MODE_IRIS, MODE_CONTOUR]
            st_["mode"] = order[(order.index(st_["mode"]) + 1) % 2] \
                if st_["mode"] in order else MODE_IRIS
        # mode-specific actions
        if st_["mode"] == MODE_IRIS:
            if k in (ord("-"), ord("+"), ord("=")):
                it = iris_items[st_["iris_sel"]]
                it["r"] = max(3.0, it["r"] + (1 if k in (ord("+"), ord("=")) else -1))
            elif k == ord("d"):
                iris_items[st_["iris_sel"]]["on"] = not iris_items[st_["iris_sel"]]["on"]
        elif st_["mode"] == MODE_FACE:
            if k == ord("d"):
                face_items[st_["face_sel"]]["on"] = not face_items[st_["face_sel"]]["on"]
        elif st_["mode"] == MODE_CONTOUR:
            key = st_["contour_eye"]
            if key in contours:
                e_ = contours[key]
                # ±/[/] adjust the active half-axis without mouse — fine tuning.
                if k in (ord("+"), ord("=")):
                    if st_["contour_handle"] in ("medial", "lateral"):
                        e_["a"] = e_["a"] + 1.0
                    elif st_["contour_handle"] in ("upper", "lower"):
                        e_["b"] = e_["b"] + 1.0
                    else:
                        e_["a"] += 1.0
                        e_["b"] += 1.0
                elif k == ord("-"):
                    if st_["contour_handle"] in ("medial", "lateral"):
                        e_["a"] = max(8.0, e_["a"] - 1.0)
                    elif st_["contour_handle"] in ("upper", "lower"):
                        e_["b"] = max(6.0, e_["b"] - 1.0)
                    else:
                        e_["a"] = max(8.0, e_["a"] - 1.0)
                        e_["b"] = max(6.0, e_["b"] - 1.0)
                # Rotate the oval: r counter-clockwise, R clockwise. Step ~1.5°.
                elif k == ord("r"):
                    e_["theta"] -= np.deg2rad(1.5)
                elif k == ord("R"):
                    e_["theta"] += np.deg2rad(1.5)
            if k == ord("e") or k == ord("E"):
                # switch which eye's oval is being edited.
                eyes = [k for k in ("L", "R") if k in contours]
                if len(eyes) > 1:
                    cur = st_["contour_eye"]
                    st_["contour_eye"] = eyes[(eyes.index(cur) + 1) % len(eyes)]
                    st_["contour_handle"] = "centre"
        if k == ord("m"):
            st_["mag"] = not st_["mag"]
    cv2.destroyAllWindows()
    iris_out = {it["name"]: ((it["x"], it["y"], it["r"]) if it["on"] else None)
                for it in iris_items}
    faces_out = {it["name"]: ((it["x"], it["y"]) if it["on"] else None)
                 for it in face_items}
    # SAMPLE each approved ellipse to a 24-vertex polygon so the runtime/validator/JSON
    # schema stay unchanged. The clinician edited an oval; the rest of the system sees a
    # polygon, exactly as before.
    contours_out = {k: _ellipse_to_polygon(e, n=24)
                    for k, e in contours.items() if iris_out.get(k) is not None}
    return iris_out, faces_out, contours_out


def approve(video_path, output_dir="outputs", auto=False):
    """STAGE 0 — propose landmarks, let the user review/correct, then APPROVE and
    save approved_landmarks.json. Tracking refuses to run until this exists.
    auto=True accepts the proposal without a GUI (headless).

    Workflow (locked 2026-06-27): **app-assisted but clinician-controlled**.
    The app proposes iris/limbus and eye-opening-contour markings using MediaPipe; the
    clinician verifies and can override poor proposals (drag, resize, redraw vertices) within
    the single editor window; one Enter approves the entire setup. There is NO automatic
    frame picker — the clinician chooses to accept whatever default init frame the detector
    selects, and overrides anything that looks wrong directly in the editor.
    """
    video_path = Path(video_path)
    detector = IrisDetector()
    init = select_init_frame(detector, video_path)
    if init is None:
        raise SystemExit("No frame with a confidently detected eye — cannot initialise.")
    fr, frame, le, re = init
    # fresh detector: select_init_frame advanced this detector's MediaPipe tracking state, so
    # re-processing the (earlier) init frame with it can fail — a clean instance detects reliably.
    faces_raw = IrisDetector().face_landmarks(frame) or {}
    faces = apply_landmark_calibration(faces_raw)   # pre-correct toward the clinician's learned placement
    detected = {"L": (le.x, le.y, le.radius) if le.detected else None,
                "R": (re.x, re.y, re.radius) if re.detected else None}

    if auto:
        confirmed = detected
        faces_out = faces
        # Seed contour from the four face landmarks; the clinician sees no GUI so this is a
        # legacy four-point fallback explicitly. Recorded in approved_landmarks.json below.
        contours_out = {}
        for key in ("L", "R"):
            if confirmed.get(key) is None:
                continue
            seed = _seed_contour_polygon(faces_out, key, n=24)
            if seed is not None:
                contours_out[key] = seed
        print("--auto-approve: accepting the proposed iris + face landmarks without review.")
    else:
        print("Stage 0: V1 ASSISTED ANATOMICAL APPROVAL — propose → verify → approve.")
        print("  The app has proposed:")
        print("    1. the full iris / limbus circle (coloured iris out to the white sclera)")
        print("    2. the iris centre")
        print("    3. the eye-opening / orbital margin contour")
        print("    4. usable-eye status")
        print("  Your role: verify and correct only what is wrong, then press Enter to APPROVE.")
        print("  Do NOT mark the pupil. Approve everything in ONE Enter; q to cancel.")
        confirmed, faces_out, contours_out = approve_interactive(frame, le, re, faces)
        # LEARN: fold the clinician's corrections (vs the RAW MediaPipe proposal) into the calibration
        learned = update_calibration(faces_out, faces_raw)
        if learned:
            print(f"Learned landmark calibration (from {learned['n_samples']} clip(s)): "
                  f"{json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in learned['offsets'].items()})}")
    if confirmed["L"] is None and confirmed["R"] is None:
        raise SystemExit("No usable eye approved.")

    # ---- V1 STAGE 0 VALIDATION ---------------------------------------------------
    # Each "usable" eye must have BOTH a credible iris/limbus seed AND a credible eye-opening
    # contour. Fail loudly here so the clinician can correct, rather than letting an invalid
    # approval reach the tracker and the detector.
    interocular_px = None
    if confirmed.get("L") and confirmed.get("R"):
        interocular_px = float(abs(confirmed["R"][0] - confirmed["L"][0]))
    per_eye_validity = {}
    failures = []
    legacy_pupil_fallback_used = False
    legacy_four_point_contour_fallback_used = False
    for key in ("L", "R"):
        iris_tuple = confirmed.get(key)
        if iris_tuple is None:
            per_eye_validity[key] = {"usable_for_v1": False,
                                      "iris_limbus_approved": False,
                                      "eye_opening_contour_approved": False,
                                      "reason": "eye_marked_unavailable"}
            continue
        ok_iris, why_iris = _validate_iris_seed((iris_tuple[0], iris_tuple[1]),
                                                 iris_tuple[2], interocular_px)
        if not ok_iris and why_iris == "iris_radius_too_small_likely_pupil":
            legacy_pupil_fallback_used = True
        contour_pts = contours_out.get(key)
        ok_contour, why_contour = _validate_eye_opening_contour(
            contour_pts, (iris_tuple[0], iris_tuple[1]), iris_tuple[2])
        if contour_pts is None or len(contour_pts) < 8:
            legacy_four_point_contour_fallback_used = True
        per_eye_validity[key] = {
            "usable_for_v1": (ok_iris and ok_contour),
            "iris_limbus_approved": ok_iris,
            "eye_opening_contour_approved": ok_contour,
            "iris_validation_reason": why_iris,
            "contour_validation_reason": why_contour,
        }
        if not ok_iris:
            failures.append(f"  - {key}: iris/limbus invalid ({why_iris})")
        if not ok_contour:
            failures.append(f"  - {key}: eye-opening contour invalid ({why_contour})")

    if failures:
        raise SystemExit(
            "Invalid Stage 0 approval for eye_vng V1. V1 requires both full iris/limbus approval "
            "AND eye-opening contour approval. Please rerun --approve and mark the iris/limbus "
            "circle (full coloured iris out to the white sclera, NOT the pupil) and the "
            "eye-opening contour (visible palpebral fissure). Failures:\n" + "\n".join(failures))

    cap = cv2.VideoCapture(str(video_path))
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30.0
    cap.release()
    out_dir = Path(output_dir) / (video_path.stem + "_tracked")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "video": str(video_path), "resolution": f"{w}x{h}", "fps": round(fps, 2),
        "init_frame_number": fr, "approved": True,
        "approved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        # ---- V1 APPROVAL SCHEMA (added 2026-06-27) -------------------------------
        "approval_schema_version": "v1_iris_limbus_and_eye_contour",
        "tracking_target": "iris_limbus",
        "reference_target": "eye_opening_contour",
        "iris_limbus_approved": all(per_eye_validity[k]["iris_limbus_approved"]
                                     for k in per_eye_validity
                                     if per_eye_validity[k]["usable_for_v1"]),
        "eye_opening_contour_approved": all(per_eye_validity[k]["eye_opening_contour_approved"]
                                             for k in per_eye_validity
                                             if per_eye_validity[k]["usable_for_v1"]),
        "legacy_pupil_fallback_used": legacy_pupil_fallback_used,
        "legacy_four_point_contour_fallback_used": legacy_four_point_contour_fallback_used,
        # ---- per-eye validity: NEW "eyes" block is the canonical V1 schema (matches the
        # locked spec). "per_eye_validity" is retained as an alias for downstream tooling.
        "eyes": {k: {"usable_for_v1": v["usable_for_v1"],
                      "iris_limbus_approved": v["iris_limbus_approved"],
                      "eye_opening_contour_approved": v["eye_opening_contour_approved"]}
                 for k, v in per_eye_validity.items()},
        "per_eye_validity": per_eye_validity,
        # ---- approved geometry --------------------------------------------------
        "iris": {k: ({"x": round(v[0], 2), "y": round(v[1], 2), "radius": round(v[2], 2)} if v else None)
                   for k, v in confirmed.items()},
        "face_landmarks": {name: ({"x": round(p[0], 2), "y": round(p[1], 2)} if p else None)
                           for name, p in faces_out.items()},
        "eye_opening_contours": {
            k: ([{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}
                 for p in pts])
            for k, pts in contours_out.items()
        },
    }
    # V1 NEVER writes a "pupils" key. The presence of "pupils" anywhere downstream is the
    # symptom of a legacy approval; the V1 runtime gate refuses such files.
    assert "pupils" not in data, "V1 approval must not contain 'pupils' — that is the legacy schema"
    path = out_dir / "approved_landmarks.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"APPROVED — saved {path}")
    print(f"  schema: {data['approval_schema_version']}")
    print(f"  iris_limbus_approved: {data['iris_limbus_approved']}  "
          f"eye_opening_contour_approved: {data['eye_opening_contour_approved']}")
    print(f"  eyes: {data['eyes']}")
    print(f'Now run tracking:  python app.py --video "{video_path}"')
    return path


# ----------------------------------------------------------------------------- overlay & plots
def draw_overlay(frame, scale, lt: Track, rt: Track, fstatus, fr, t, face_tracks=None,
                 raw_iris=None, contours=None):
    """lt/rt carry the FILTERED iris centres (main markers). raw_iris = [(x,y),...]
    are drawn as small faint markers so the raw vs filtered difference is visible."""
    ow, oh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
    vis = cv2.resize(frame, (ow, oh))
    cv2.rectangle(vis, (0, 0), (ow - 1, oh - 1), STATUS_COLOR.get(fstatus, YELLOW), 5)
    # facial landmarks: the 4 EYE-OPENING landmarks (canthi + lid margins) are part of the clinical
    # frame — drawn PROMINENT (cyan squares + label); other tracked points stay small/faint.
    APER = ("inner_canthus", "outer_canthus", "upper_margin", "lower_margin")
    for name, tr in (face_tracks or {}).items():
        if tr.x is None:
            continue
        p = (int(tr.x * scale), int(tr.y * scale))
        if any(name.startswith(a) for a in APER):
            cv2.rectangle(vis, (p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4), CYAN, -1)
            short = {"inner_canthus": "in", "outer_canthus": "out",
                     "upper_margin": "up", "lower_margin": "lo"}[name.rsplit("_", 1)[0]]
            cv2.putText(vis, short, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1, cv2.LINE_AA)
        else:
            cv2.circle(vis, p, 2, (120, 120, 120), -1)
    # faint RAW iris markers (so you can see what the filter is removing)
    for rp in (raw_iris or []):
        if rp and rp[0] is not None:
            cv2.drawMarker(vis, (int(rp[0] * scale), int(rp[1] * scale)), (170, 170, 170),
                           cv2.MARKER_CROSS, 9, 1)
    for key, contour in (contours or {}).items():
        if contour is None:
            continue
        pts = np.asarray(contour.points, np.int32)
        pts_s = np.round(pts * scale).astype(np.int32).reshape(-1, 1, 2)
        col = CYAN if contour.reference_valid else AMBER
        cv2.polylines(vis, [pts_s], isClosed=True, color=col, thickness=2, lineType=cv2.LINE_AA)
        ex = contour.extents()
        labels = {"medial": "M", "lateral": "L", "upper": "U", "lower": "D"}
        for name, p0 in ex.items():
            p = (int(p0[0] * scale), int(p0[1] * scale))
            cv2.drawMarker(vis, p, col, cv2.MARKER_DIAMOND, 12, 1)
            cv2.putText(vis, labels[name], (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    for tr in (lt, rt):
        col = STATUS_COLOR.get(tr.status, YELLOW)
        if tr.x is not None:
            p = (int(tr.x * scale), int(tr.y * scale))
            rad = int((tr.radius or 10) * scale)
            cv2.circle(vis, p, max(3, rad), col, 3, cv2.LINE_AA)   # tracked iris DISC (main object)
            cv2.drawMarker(vis, p, col, cv2.MARKER_CROSS, 16, 2)   # iris-disc centre
        elif tr.raw_x is not None and tr.status in ("blink_or_occluded", "uncertain"):
            # INVALID frame (blink/occlusion/jump): show where MediaPipe thought the iris was, as a
            # BLUE cross — explicitly NOT a confident green circle. Position is not reported as valid.
            p = (int(tr.raw_x * scale), int(tr.raw_y * scale))
            cv2.drawMarker(vis, p, BLUE, cv2.MARKER_TILTED_CROSS, 14, 2)
            if tr.artifact_type and tr.artifact_type != "none":
                cv2.putText(vis, tr.artifact_type, (p[0] + 10, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1)
        if tr.rejected and tr.raw_x is not None:
            cv2.drawMarker(vis, (int(tr.raw_x * scale), int(tr.raw_y * scale)), RED,
                           cv2.MARKER_TILTED_CROSS, 12, 2)
    lines = [f"frame {fr}  t={t:.2f}s", f"L:{lt.status}  R:{rt.status}", fstatus]
    for i, txt in enumerate(lines):
        y = 20 + i * 22
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, STATUS_COLOR.get(fstatus, YELLOW), 1)
    return vis


def draw_clinical_overlay(frame, scale, lt: Track, rt: Track,
                          selected_image_eye=None, contours=None):
    """V1 CLINICAL overlay mode — minimal display per the V1 UI rules.

    Shows ONLY:
      * the video frame,
      * the eye-opening contour as a thin cyan polyline (verification — we need to see that the
        reference frame is correctly tracked; M/L/U/D markers and labels are NOT drawn),
      * a thin state-coloured iris-circle outline + small centre cross per eye,
      * the anatomical eye word ("Left" / "Right") above the tracked iris.

    Does NOT show: M/L/U/D extents, frame number, timestamp, tracking-state banner, per-eye
    fc/ic/rc/cov text, raw markers, helper features, debug trace strip, torsion footnote.

    Technical/debug overlay (everything that this function omits) is available via
    overlay_mode "debug" in run().
    """
    ow, oh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
    vis = cv2.resize(frame, (ow, oh))

    def _dim(col):
        """Faint, low-contrast version of a colour — used for the non-selected eye when
        selected_image_eye is set. Keeps the eye visible in the overlay (for verification)
        without the bright cyan ring drawing the eye toward the wrong place."""
        return (max(0, col[0] // 3), max(0, col[1] // 3), max(0, col[2] // 3))

    # Eye-opening contour — cyan when the reference is valid, amber when uncertain. Drawn as a
    # thin polyline only; no extent diamonds, no labels.
    for key, contour in (contours or {}).items():
        if contour is None:
            continue
        is_active = (selected_image_eye in (None, "both") or key == selected_image_eye)
        pts = np.asarray(contour.points, np.int32)
        pts_s = np.round(pts * scale).astype(np.int32).reshape(-1, 1, 2)
        col = CYAN if contour.reference_valid else AMBER
        if not is_active:
            col = _dim(col)
        cv2.polylines(vis, [pts_s], isClosed=True, color=col, thickness=1, lineType=cv2.LINE_AA)

    for image_side, tr in (("L", lt), ("R", rt)):
        if tr.x is None:
            continue
        is_active = (selected_image_eye in (None, "both") or image_side == selected_image_eye)
        col = STATUS_COLOR.get(tr.status, YELLOW)
        if not is_active:
            col = _dim(col)
        p = (int(tr.x * scale), int(tr.y * scale))
        rad = int((tr.radius or 10) * scale)
        # Iris circle outline (state-coloured) + small centre cross — the tracked clinical object.
        # Selected eye gets a thicker outline; non-selected eye uses a thin 1px outline.
        cv2.circle(vis, p, max(3, rad), col, 2 if is_active else 1, cv2.LINE_AA)
        cv2.drawMarker(vis, p, col, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        # Anatomical label ABOVE the iris. In a front-facing video image_side 'L' = anatomical
        # 'Right' and image_side 'R' = anatomical 'Left'; mapping lives in patient_eye_label().
        # The label is drawn brighter on the active eye, dimmer on the inactive eye.
        word = patient_eye_label(image_side)              # 'Right' / 'Left'
        pad = max(14, rad + 10)
        tx, ty = p[0] - 18, max(20, p[1] - pad)
        text_col = (255, 255, 255) if is_active else (140, 140, 140)
        cv2.putText(vis, word, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, word, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_col, 1, cv2.LINE_AA)
    return vis


def draw_composite(vis, scale, cft, fm_L, fm_R):
    """Render the V1 clinical anatomy: visible LIMBUS ARC + estimated full iris CIRCLE/ELLIPSE +
    iris-circle centre. Optional CFT helper features (if the engine carries a `.pool`) are drawn as
    very faint dots — never the primary visual; the limbus arc and ellipse are the clinical view."""
    for key, fm in (("L", fm_L), ("R", fm_R)):
        trk = cft.get(key)
        if trk is None:
            continue
        # Optional CFT helper features (V1Tracker exposes .helper when use_cft_helper=True;
        # CompositeFeatureTracker exposes .pool directly). Draw as 1px faint dots so they are
        # clearly subordinate to the limbus arc/ellipse.
        pool = getattr(trk, "pool", None)
        if pool is None and getattr(trk, "helper", None) is not None:
            pool = trk.helper.pool
        if pool is not None:
            for f in pool:
                col = (0, 120, 0) if f.state == TRUSTED else \
                      ((0, 120, 150) if f.state == PROBATION else (0, 0, 130))
                cv2.circle(vis, (int(f.pos[0] * scale), int(f.pos[1] * scale)), 1, col, -1)
        if fm is not None:
            valid = fm.validity == "valid"
            # VISIBLE limbus arc — the iris-sclera boundary actually detected this frame (bright yellow).
            # The green circle/disc (drawn by draw_overlay) is the ESTIMATED FULL iris completed from it.
            pts = getattr(fm, "limbus_arc_pts", None)
            if pts is not None:
                for px, py in pts:
                    cv2.circle(vis, (int(px * scale), int(py * scale)), 2, (0, 220, 255), -1)
            # fitted iris ELLIPSE (green when valid, well-seen) — the estimated full disc
            if valid and fm.limbus_ellipse is not None:
                (ecx, ecy), (MA, ma), ang = fm.limbus_ellipse
                cv2.ellipse(vis, (int(ecx * scale), int(ecy * scale)),
                            (int(MA / 2 * scale), int(ma / 2 * scale)), ang, 0, 360, GREEN, 2, cv2.LINE_AA)
            go = fm.iris_centre if fm.iris_centre else trk.last_centre
            org = (int(go[0] * scale) - 30, int(go[1] * scale) - int((trk.last_radius + 14) * scale))
            occ = "*" if getattr(fm, "limbus_radius_fixed", False) else ""   # * = radius held (occluded)
            # Clinical label: the patient's anatomical eye on this side of the image.
            # The image-to-patient mapping is recorded in metadata; user-facing UI shows
            # only the simple anatomical word 'Left' or 'Right'.
            eye_lbl = patient_eye_label(key)
            txt = (f"{eye_lbl}: {fm.state.lower()} fc={fm.frame_confidence:.2f} "
                   f"ic={getattr(fm, 'limbus_inlier_fraction', 0):.2f} "
                   f"rc={getattr(fm, 'reference_confidence', 0):.2f} "
                   f"cov={getattr(fm, 'limbus_arc_coverage', 0):.2f}{occ}")
            cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# Arrow vectors per spec §K.4 glyph (image x↑ right, image y↑ down).
# The detector emits ASCII glyphs because cv2.putText cannot render Unicode arrows on Windows
# without a custom font; the renderer here turns each glyph into a drawn arrow.
_ARROW_VECTORS = {
    "<-":   (-1.0, 0.0),
    "->":   (+1.0, 0.0),
    "^":    (0.0, -1.0),
    "v":    (0.0, +1.0),
    "/^":   (+0.70, -0.70),
    "\\^":  (-0.70, -0.70),
    "/v":   (-0.70, +0.70),   # down-LEFT (looks like / going down)
    "\\v":  (+0.70, +0.70),   # down-RIGHT
}


def draw_nystagmus_arrow(vis, label: str, glyph: str, n_beats: int = 0,
                         rate_hz: float = 0.0, confidence: float = 0.0,
                         analysed_eye: str = ""):
    """Draw the V1 clinical nystagmus arrow + label band on the overlay. Spec §K.4 / §G.

    Renders into the TOP-LEFT corner of the frame, in a region that does NOT cover the eyes
    (the source clips zoom over the face; the top-left is consistently free of facial anatomy
    in the V1 reference clips). The arrow is drawn as a thick coloured shaft + filled arrow head,
    sized relative to the band, so the direction reads at a glance from across a clinic room."""
    if not label or glyph not in _ARROW_VECTORS:
        return vis
    h, w = vis.shape[:2]
    band_w = min(380, max(220, int(0.34 * w)))
    band_h = 86
    pad = 12
    x0, y0 = pad, pad
    x1, y1 = x0 + band_w, y0 + band_h

    # translucent dark panel for legibility, then a bright clinical-red border so the user
    # cannot miss that the detector flagged something.
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 40, 220), 2, cv2.LINE_AA)

    # ---- arrow ---------------------------------------------------------------
    ax_cx, ax_cy = x0 + 42, (y0 + y1) // 2
    arrow_len = 56
    dx, dy = _ARROW_VECTORS[glyph]
    tip = (int(ax_cx + dx * arrow_len / 2), int(ax_cy + dy * arrow_len / 2))
    tail = (int(ax_cx - dx * arrow_len / 2), int(ax_cy - dy * arrow_len / 2))
    cv2.arrowedLine(vis, tail, tip, (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.45)
    cv2.arrowedLine(vis, tail, tip, (40, 40, 220), 3, cv2.LINE_AA, tipLength=0.45)

    # ---- text ---------------------------------------------------------------
    title = label
    tx = x0 + 90
    cv2.putText(vis, title, (tx, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, title, (tx, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    sub = []
    if analysed_eye:
        sub.append(f"eye: {analysed_eye}")
    if n_beats:
        sub.append(f"{n_beats} beats")
    if rate_hz:
        sub.append(f"{rate_hz:.1f} Hz")
    if confidence:
        sub.append(f"conf {confidence:.2f}")
    if sub:
        sub_txt = "   ".join(sub)
        cv2.putText(vis, sub_txt, (tx, y0 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, sub_txt, (tx, y0 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    # torsion note — always shown on the band (spec §K.5)
    cv2.putText(vis, "Torsion: not assessed", (tx, y0 + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Torsion: not assessed", (tx, y0 + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    return vis


def _short_direction_label(beating_direction: str) -> str:
    """One-line clinical status text for the clinical overlay. The user-facing words are restricted
    to the V1 set: Left-beating / Right-beating / Up-beating / Down-beating / Oblique."""
    if not beating_direction or beating_direction == "none":
        return "No nystagmus detected"
    mapping = {
        "left":  "Nystagmus detected: Left-beating",
        "right": "Nystagmus detected: Right-beating",
        "up":    "Nystagmus detected: Up-beating",
        "down":  "Nystagmus detected: Down-beating",
    }
    if beating_direction in mapping:
        return mapping[beating_direction]
    if beating_direction.startswith("oblique"):
        return "Nystagmus detected: Oblique"
    return "No nystagmus detected"


def _safe_text_region(vis, iris_pts):
    """Pick a placement strategy for the clinical status text that stays well clear of the eyes.
    Returns ('top'|'bottom', y_band_centre). Strategy: if the average iris-y is in the LOWER half
    of the frame, place text in the FOREHEAD region (top). Otherwise place text BELOW the cheek
    region (bottom). The chosen y is centred in a band that is at least 60 px away from the iris
    line — so the text cannot reach the eyes."""
    h, w = vis.shape[:2]
    ys = [int(round(p[1])) for p in iris_pts if p is not None and p[1] is not None]
    if not ys:
        return "top", max(34, int(0.10 * h))
    eye_y = int(np.mean(ys))
    # if eyes are in the bottom half → forehead (top band); else lower face (bottom band)
    if eye_y >= h // 2:
        return "top", max(34, int(0.10 * h))
    # bottom band — guarantee a 60 px buffer below the lowest eye
    bottom_y = min(h - 30, max(eye_y + 60, int(0.82 * h)))
    return "bottom", bottom_y


def draw_clinical_status_band(vis, status_text: str, glyph: str = "",
                              iris_pts=None):
    """Render the V1 clinical status text + optional arrow in a SAFE region of the frame
    (forehead or below the lower face), NEVER on or near the eyes.

    The band carries:
      * the single clinical status line (no debug text, no frame/time, no torsion footnote)
      * an arrow shaped to match the detected beating direction (or no arrow if 'No nystagmus
        detected'). For 'Oblique' the arrow points at 45° in the direction the resultant
        favoured, taken from the glyph passed in.
    """
    h, w = vis.shape[:2]
    side, y_centre = _safe_text_region(vis, iris_pts or [])
    band_h = 78
    if side == "top":
        y0 = max(8, y_centre - band_h // 2)
    else:
        y0 = min(h - band_h - 8, y_centre - band_h // 2)
    y1 = y0 + band_h
    band_w = min(720, max(360, int(0.62 * w)))
    x0 = (w - band_w) // 2
    x1 = x0 + band_w

    # translucent dark panel for legibility; bright clinical-red border when nystagmus detected
    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
    border_col = (40, 40, 220) if status_text.startswith("Nystagmus detected") else (160, 160, 160)
    cv2.rectangle(vis, (x0, y0), (x1, y1), border_col, 2, cv2.LINE_AA)

    # ---- arrow (only when nystagmus is detected) -----------------------------
    text_x_start = x0 + 24
    if glyph in _ARROW_VECTORS and status_text.startswith("Nystagmus detected"):
        ax_cx, ax_cy = x0 + 50, (y0 + y1) // 2
        L = 56
        dx, dy = _ARROW_VECTORS[glyph]
        tip = (int(ax_cx + dx * L / 2), int(ax_cy + dy * L / 2))
        tail = (int(ax_cx - dx * L / 2), int(ax_cy - dy * L / 2))
        cv2.arrowedLine(vis, tail, tip, (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.45)
        cv2.arrowedLine(vis, tail, tip, border_col, 3, cv2.LINE_AA, tipLength=0.45)
        text_x_start = x0 + 104

    # ---- single status line --------------------------------------------------
    ty = y0 + band_h // 2 + 8
    cv2.putText(vis, status_text, (text_x_start, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, status_text, (text_x_start, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_no_nystagmus_caption(vis, analysed_eye: str = ""):
    """Small unobtrusive caption when the detector saw no nystagmus this frame (spec §G).
    Drawn in the TOP-LEFT corner so it never covers the eyes."""
    h, w = vis.shape[:2]
    txt = "No nystagmus detected"
    pad = 12
    cv2.putText(vis, txt, (pad, pad + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, txt, (pad, pad + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, "Torsion: not assessed", (pad, pad + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "Torsion: not assessed", (pad, pad + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    if analysed_eye:
        cv2.putText(vis, f"eye: {analysed_eye}", (pad, pad + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"eye: {analysed_eye}", (pad, pad + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    return vis


def plot_trace(times, series, title, path, w=1100, h=420, flags=None, zero_each=False,
               robust=True, shade=None, scale_exclude=None):
    """Simple OpenCV line plot (no matplotlib dependency). series = {label:(vals,color)};
    vals may contain None (gaps left blank — blink/lost are not interpolated). flags = optional
    list of bools per time index; True draws a red tick at the bottom (e.g. poor-quality frame).

    robust=True clips the y-range to percentiles so a SINGLE glitch/blink frame can't squash the
    whole trace (the real signal then fills the plot; glitches clip off-edge and are red-ticked).
    zero_each=True de-means each series by its own median, so eyes at very different absolute x
    overlay around 0 and small nystagmus beats become visible instead of a flat 1300 px scale."""
    img = 255 * np.ones((h, w, 3), np.uint8)
    ml, mr, mt, mb = 70, 20, 40, 50
    pw, ph = w - ml - mr, h - mt - mb
    if zero_each:
        dm = {}
        for label, (vals, color) in series.items():
            ok = [v for v in vals if v is not None]
            med = float(np.median(ok)) if ok else 0.0
            dm[label] = ([(v - med) if v is not None else None for v in vals], color)
        series = dm
    # y-scaling values: optionally EXCLUDE flagged frames (e.g. degenerate-aperture) so a minority of
    # bad frames can't blow out the scale and squash the real signal. They're still plotted (clamped).
    if scale_exclude:
        def _keep(i):
            return not (i < len(scale_exclude) and scale_exclude[i])
        allv = [v for vals, _ in series.values() for i, v in enumerate(vals) if v is not None and _keep(i)]
        if len(allv) < 10:
            allv = [v for vals, _ in series.values() for v in vals if v is not None]
    else:
        allv = [v for vals, _ in series.values() for v in vals if v is not None]
    if not allv or not times:
        cv2.imwrite(str(path), img); return
    if robust and len(allv) >= 20:
        a = np.array(allv, float)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med))) or 1e-6
        inl = a[np.abs(a - med) <= 8.0 * mad]          # drop wild outliers from the SCALE only
        base = inl if len(inl) >= 10 else a
        lo, hi = (float(x) for x in np.percentile(base, [2, 98]))
        ymin, ymax = (lo, hi) if hi - lo > 1e-6 else (float(a.min()), float(a.max()))
    else:
        ymin, ymax = min(allv), max(allv)
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
    ymin -= pad; ymax += pad
    if ymax - ymin < 1e-6:           # only guard a truly FLAT trace (avoid div-by-zero), not a small range
        ymax, ymin = ymax + 0.5, ymin - 0.5
    tmax = max(times) or 1.0
    def X(t): return int(ml + pw * t / tmax)
    def Y(v): return int(min(mt + ph, max(mt, mt + ph * (1 - (v - ymin) / (ymax - ymin)))))
    if shade:                                   # light bands over blink/occlusion periods (behind trace)
        for tt, s in zip(times, shade):
            if s:
                x = X(tt)
                cv2.line(img, (x, mt), (x, mt + ph), (225, 225, 225), 1)
    if flags:
        for t, fl in zip(times, flags):
            if fl:
                cv2.line(img, (X(t), mt + ph), (X(t), mt + ph - 10), (0, 0, 255), 1)
    cv2.rectangle(img, (ml, mt), (ml + pw, mt + ph), (0, 0, 0), 1)
    cv2.putText(img, title, (ml, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    yfmt = "%.2f" if (ymax - ymin) < 10 else "%.0f"   # 0..1 eye-local needs decimals; px traces don't
    cv2.putText(img, yfmt % ymax, (8, mt + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, yfmt % ymin, (8, mt + ph), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"{tmax:.1f}s", (ml + pw - 40, mt + ph + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    leg = ml + 10
    for label, (vals, color) in series.items():
        prev = None
        for t, v in zip(times, vals):
            if v is None:
                prev = None; continue
            p = (X(t), Y(v))
            if prev is not None:
                cv2.line(img, prev, p, color, 1)
            prev = p
        cv2.putText(img, label, (leg, mt + ph + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        leg += 110
    cv2.imwrite(str(path), img)


def render_trace_panel(times, lx, rx, t_now, width, height=220, window=8.0):
    """A scrolling VNG-style trace strip for the last `window` seconds up to t_now, auto-scaled
    to the visible data, with a red cursor at the current time. Plots horizontal eye position
    (L green / R blue). Drawn each frame from data collected so far (causal — no future data)."""
    img = 255 * np.ones((height, width, 3), np.uint8)
    t0 = max(0.0, t_now - window)
    vis_vals = [v for i in range(len(times)) if times[i] >= t0
                for v in (lx[i], rx[i]) if v is not None]
    mt, mb = 18, 8
    if vis_vals:
        ymin, ymax = min(vis_vals), max(vis_vals)
        if ymax - ymin < 5:
            ymax, ymin = ymax + 5, ymin - 5
        pad = 0.1 * (ymax - ymin)
        ymin -= pad; ymax += pad

        def X(t): return int((t - t0) / max(window, 1e-6) * (width - 1))
        def Y(v): return int(mt + (height - mt - mb) * (1 - (v - ymin) / (ymax - ymin)))
        for vals, color in ((lx, GREEN), (rx, BLUE)):
            prev = None
            for i in range(len(times)):
                if times[i] < t0:
                    continue
                v = vals[i]
                if v is None:
                    prev = None; continue
                p = (X(times[i]), Y(v))
                if prev is not None:
                    cv2.line(img, prev, p, color, 1)
                prev = p
        cv2.line(img, (X(t_now), 0), (X(t_now), height), (0, 0, 255), 1)   # cursor
    cv2.line(img, (0, 0), (width - 1, 0), (0, 0, 0), 1)
    cv2.putText(img, f"eye-in-socket: iris vs canthi, horizontal  (L green / R blue)  last {window:.0f}s",
                (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return img


def superimpose_traces(vis, times, channels, t_now, y0, window=8.0):
    """Draw scrolling VNG trace lines on the frame BELOW the eyes (band starts at `y0`) — NO opaque
    background band, and never over the eyes. `channels` = list of (label, lines) where
    lines = [(vals, color), ...]; each channel occupies an equal horizontal band from `y0` to the
    bottom, auto-scaled to its own visible data, sharing one red time cursor. Lines get a thin dark
    outline for contrast against the video instead of a panel. Causal — only data up to t_now."""
    h, w = vis.shape[:2]
    band_total = h - y0
    t0 = max(0.0, t_now - window)

    def X(t):
        return int((t - t0) / max(window, 1e-6) * (w - 1))

    def label_text(s, org):                  # outlined text, readable over any background, no band
        cv2.putText(vis, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    bh = band_total // max(len(channels), 1)
    for ci, (label, lines) in enumerate(channels):
        by0, by1 = y0 + ci * bh, y0 + (ci + 1) * bh
        mt, mb = 16, 8
        vis_vals = [vals[i] for (vals, _) in lines for i in range(len(times))
                    if times[i] >= t0 and vals[i] is not None]
        if vis_vals:
            ymin, ymax = min(vis_vals), max(vis_vals)
            if ymax - ymin < 5:
                ymax, ymin = ymax + 5, ymin - 5
            pad = 0.12 * (ymax - ymin)
            ymin -= pad; ymax += pad

            def Y(v, by0=by0, by1=by1, ymin=ymin, ymax=ymax):
                return int(by0 + mt + (by1 - by0 - mt - mb) * (1 - (v - ymin) / (ymax - ymin)))
            for vals, color in lines:
                prev = None
                for i in range(len(times)):
                    if times[i] < t0:
                        continue
                    v = vals[i]
                    if v is None:
                        prev = None; continue
                    p = (X(times[i]), Y(v))
                    if prev is not None:
                        cv2.line(vis, prev, p, (0, 0, 0), 3, cv2.LINE_AA)   # dark outline
                        cv2.line(vis, prev, p, color, 1, cv2.LINE_AA)       # coloured trace
                    prev = p
        label_text(label, (6, by0 + 14))
    xc = X(t_now)
    cv2.line(vis, (xc, y0), (xc, h), (0, 0, 255), 1, cv2.LINE_AA)           # shared time cursor
    return vis


# ----------------------------------------------------------------------------- main run
HEAD_REF_LANDMARKS = ("nose_bridge_mid", "nose_tip", "outer_canthus_L", "outer_canthus_R",
                      "inner_canthus_L", "inner_canthus_R")


class HeadStabilizer:
    """Eye-in-head: remove head motion via an affine transform fit from the head-reference
    landmarks (nose + eye corners) back to their init positions; the iris expressed in that
    frame is the eye movement. TIERED quality, not hard rejection — only DEGENERATE transforms
    are rejected (too few landmarks, NaN, impossible scale, or an impossible per-frame jump in
    scale/rotation/translation that no real head can make). Everything else is kept with a
    quality flag (good/fair/poor) + residual + landmark count, so the trace stays continuous and
    spikes can be attributed to transform quality vs iris-tracking error."""

    def __init__(self, init_face, init_iris, frame_w):
        self.init_face = init_face            # {name: (x, y)}
        self.init_iris = init_iris          # {"L": (x, y), "R": (x, y)}
        self.frame_w = float(frame_w)
        self.prev = None                      # (scale, rot, tx, ty) of last accepted transform

    def step(self, cur_face):
        """Return (M, quality, residual, n_used) or None if the transform is DEGENERATE."""
        names = [n for n in HEAD_REF_LANDMARKS if self.init_face.get(n) and cur_face.get(n)]
        if len(names) < 3:                    # too few landmarks to fit
            return None
        src = np.array([cur_face[n] for n in names], dtype=np.float32)
        dst = np.array([self.init_face[n] for n in names], dtype=np.float32)
        m, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
        if m is None or not np.all(np.isfinite(m)):     # numerical failure / NaN
            return None
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        sx, sy = float(np.hypot(a, c)), float(np.hypot(b, d))
        rot = float(np.arctan2(c, a))
        tx, ty = float(m[0, 2]), float(m[1, 2])
        if not (0.4 < sx < 2.5 and 0.4 < sy < 2.5):     # impossible scale (truly degenerate)
            return None
        if self.prev is not None:                        # only an EGREGIOUS per-frame jump → glitch
            ps, pr, ptx, pty = self.prev
            drot = abs((rot - pr + np.pi) % (2 * np.pi) - np.pi)
            if abs(sx - ps) > 0.5 or drot > np.radians(45) or np.hypot(tx - ptx, ty - pty) > 0.30 * self.frame_w:
                return None
        inl = inl.ravel().astype(bool) if inl is not None else np.ones(len(names), bool)
        proj = (m[:, :2] @ src.T).T + m[:, 2]
        err = np.hypot(proj[:, 0] - dst[:, 0], proj[:, 1] - dst[:, 1])
        n_used = int(inl.sum())
        residual = float(err[inl].mean()) if n_used else float(err.mean())
        quality = "good" if residual < 4.0 else ("fair" if residual < 12.0 else "poor")
        self.prev = (sx, rot, tx, ty)
        return m, quality, residual, n_used

    def correct(self, m, iris_xy, eye):
        ip = self.init_iris[eye]
        if m is None or iris_xy[0] is None or ip[0] is None:
            return None, None
        sx = m[0, 0] * iris_xy[0] + m[0, 1] * iris_xy[1] + m[0, 2]
        sy = m[1, 0] * iris_xy[0] + m[1, 1] * iris_xy[1] + m[1, 2]
        return float(sx - ip[0]), float(sy - ip[1])


def canthus_relative(iris, inner, outer):
    """Eye-in-socket position: the iris expressed relative to this eye's own medial (inner) and
    lateral (outer) canthus. Origin = the canthi midpoint; horizontal axis = inner→outer corner
    (so it follows head tilt); h = displacement along that axis, v = perpendicular, in pixels.
    Because the canthi move WITH the head, this cancels head/"hair" movement locally and leaves only
    the iris's movement within the eye — the nystagmus. Returns (h, v) or (None, None)."""
    if iris[0] is None or inner is None or outer is None:
        return None, None
    inner = np.array(inner, float); outer = np.array(outer, float); p = np.array(iris, float)
    axis = outer - inner
    length = float(np.hypot(axis[0], axis[1]))
    if length < 1.0:
        return None, None
    ux = axis / length
    if ux[0] < 0:                            # force the axis rightward so both eyes read conjugately
        ux = -ux
    uy = np.array([-ux[1], ux[0]])          # perpendicular (vertical-in-eye)
    vec = p - (inner + outer) / 2.0
    return float(np.dot(vec, ux)), float(np.dot(vec, uy))


def eye_local(iris, inner, outer, upper, lower):
    """Iris position in EYE-LOCAL (eye-aperture) coordinates — the current clinical trace. Uses only
    this eye's own four aperture corners, so it is inherently independent of head/camera movement (no
    face/head landmarks needed). Convention:
        x: 0 = inner canthus  →  1 = outer canthus   (projection on the inner→outer axis)
        y: 0 = upper margin   →  1 = lower margin     (projection on the upper→lower axis)
    0.5 ≈ centred. Values may run slightly outside [0,1] at gaze extremes. Returns (x, y); either is
    None if its landmark pair is missing/degenerate or the iris is invalid (blink)."""
    if not iris or iris[0] is None:
        return None, None

    def proj(a, b):                          # normalised projection of iris onto axis a→b (a=0, b=1)
        if a is None or b is None:
            return None
        ax, ay = b[0] - a[0], b[1] - a[1]
        L2 = ax * ax + ay * ay
        if L2 < 1.0:
            return None
        return ((iris[0] - a[0]) * ax + (iris[1] - a[1]) * ay) / L2

    return proj(inner, outer), proj(upper, lower)


def iris_drift_check(hv, prev_hv, tracked_xy, mp_xy, mp_iris_r):
    """Anatomical drift guard for the V1 iris trace — pure geometry, no pixel thresholding.

    Decides whether the tracked iris is still physically attached to the eye this frame:
      • outside_aperture — the iris CENTRE left the marked eye opening (eye-local beyond the box)
      • off_anchor       — the tracked centre is too far from the INDEPENDENT MediaPipe iris centre
                           (MediaPipe iris stays on the real eye; a drifted template won't), scaled by
                           the current MediaPipe iris radius so it is zoom-invariant — this also guards
                           iris SIZE: a jump onto a wrong-size/wrong-place region fails it
      • review_excursion — a large 1-frame eye-local jump that is OTHERWISE anatomically valid; a real
                           fast-phase looks like this, so it is FLAGGED for visual confirmation but NOT
                           discarded (we never delete genuine eye movement)

    Returns (state, invalid): state ∈ {"","outside_aperture","off_anchor","review_excursion"};
    invalid=True only for the first two (caller drops the frame's eye-local value → a gap)."""
    h, v = hv
    if h is None or v is None:
        return "", False                      # already a gap (blink / occlusion / degenerate aperture)
    m = DRIFT_APERTURE_MARGIN
    if not (-m <= h <= 1.0 + m) or not (-m <= v <= 1.0 + m):
        return "outside_aperture", False
    if False and mp_xy is not None and mp_iris_r and tracked_xy[0] is not None:
        gap = float(np.hypot(tracked_xy[0] - mp_xy[0], tracked_xy[1] - mp_xy[1]))
        if gap > DRIFT_ANCHOR_FRAC * mp_iris_r:
            return "off_anchor", True
    if prev_hv is not None:
        ph, pv = prev_hv                       # either axis may be None (degenerate aperture last frame)
        if (ph is not None and abs(h - ph) > DRIFT_EXCURSION) or \
           (pv is not None and abs(v - pv) > DRIFT_EXCURSION):
            return "review_excursion", False  # large but valid → flag for review, keep plotting
    return "", False


def eye_velocity(times, v, fast_k=3.0):
    """Frame-to-frame velocity of an eye-local series (per second), with gaps preserved (None where
    either endpoint is invalid — blink). Returns (vel, fast) aligned to times[1:]: `fast` flags
    fast-phase candidates (|velocity| > fast_k × median |velocity|), so a jerk-nystagmus reset shows
    as a marked spike opposite the slow drift. vel is in eye-local units/s (1.0 = inner→outer span/s)."""
    vel = []
    for i in range(1, len(v)):
        dt = times[i] - times[i - 1]
        if v[i] is not None and v[i - 1] is not None and dt > 0:
            vel.append((v[i] - v[i - 1]) / dt)
        else:
            vel.append(None)
    mags = [abs(x) for x in vel if x is not None]
    thr = fast_k * float(np.median(mags)) if mags else 0.0
    fast = [x is not None and thr > 0 and abs(x) > thr for x in vel]
    return vel, fast


# ----------------------------------------------------------------------------- learned landmark calibration
# Learn the clinician's systematic placement of the eye-aperture landmarks RELATIVE to MediaPipe's
# proposal, so future proposals start closer to where the clinician actually marks (e.g. the upper
# margin sits well above MediaPipe's lid landmark). Offsets are stored per landmark TYPE in each eye's
# OWN frame and normalised by inter-canthal distance, so they transfer across clips, zoom, and side.
CALIB_PATH = Path(__file__).resolve().parent / "landmark_calibration.json"
APERTURE_TYPES = ("inner_canthus", "outer_canthus", "upper_margin", "lower_margin")


def _eye_frame(inner, outer, upper, lower):
    """Eye-local basis: D = inter-canthal distance, u = inner→outer (unit), w = upper→lower (unit)."""
    I, O = np.array(inner, float), np.array(outer, float)
    D = float(np.hypot(*(O - I)))
    if D < 1.0:
        return None
    u = (O - I) / D
    a = np.array(lower, float) - np.array(upper, float)
    na = float(np.hypot(*a))
    w = a / na if na > 1.0 else np.array([-u[1], u[0]])
    return D, u, w


def load_calibration(calib_path=CALIB_PATH):
    try:
        return json.loads(Path(calib_path).read_text())
    except Exception:
        return {"n_samples": 0, "offsets": {}}


def _clip_offsets(user, mp):
    """Per-landmark-type (du along inner→outer, dw along upper→lower), D-normalised, averaged L+R."""
    samples = {t: [] for t in APERTURE_TYPES}
    for eye in ("L", "R"):
        nm = {t: f"{t}_{eye}" for t in APERTURE_TYPES}
        if not all(user.get(nm[t]) and mp.get(nm[t]) for t in APERTURE_TYPES):
            continue
        fr = _eye_frame(user[nm["inner_canthus"]], user[nm["outer_canthus"]],
                        user[nm["upper_margin"]], user[nm["lower_margin"]])
        if fr is None:
            continue
        D, u, w = fr
        for t in APERTURE_TYPES:
            corr = np.array(user[nm[t]], float) - np.array(mp[nm[t]], float)
            samples[t].append((float(np.dot(corr, u) / D), float(np.dot(corr, w) / D)))
    return {t: (float(np.mean([s[0] for s in v])), float(np.mean([s[1] for s in v])))
            for t, v in samples.items() if v}


def update_calibration(user, mp, calib_path=CALIB_PATH):
    """Fold this clip's clinician-vs-MediaPipe offsets into the running calibration (a simple mean)."""
    this = _clip_offsets(user, mp)
    if not this:
        return None
    calib = load_calibration(calib_path)
    n = int(calib.get("n_samples", 0))
    offs = dict(calib.get("offsets", {}))
    for t, (du, dw) in this.items():
        old = offs.get(t)
        offs[t] = ({"du": (old["du"] * n + du) / (n + 1), "dw": (old["dw"] * n + dw) / (n + 1)}
                   if old else {"du": du, "dw": dw})
    out = {"n_samples": n + 1, "offsets": offs}
    Path(calib_path).write_text(json.dumps(out, indent=2))
    return out


def apply_landmark_calibration(proposal, calib=None):
    """Shift a fresh MediaPipe aperture proposal toward the clinician's learned placement."""
    calib = calib if calib is not None else load_calibration()
    offs = calib.get("offsets") or {}
    if not offs or not proposal:
        return proposal
    out = dict(proposal)
    for eye in ("L", "R"):
        nm = {t: f"{t}_{eye}" for t in APERTURE_TYPES}
        if not all(proposal.get(nm[t]) for t in APERTURE_TYPES):
            continue
        fr = _eye_frame(proposal[nm["inner_canthus"]], proposal[nm["outer_canthus"]],
                        proposal[nm["upper_margin"]], proposal[nm["lower_margin"]])
        if fr is None:
            continue
        D, u, w = fr
        for t in APERTURE_TYPES:
            o = offs.get(t)
            if o:
                p = np.array(proposal[nm[t]], float) + D * (o["du"] * u + o["dw"] * w)
                out[nm[t]] = (float(p[0]), float(p[1]))
    return out


def run(video_path, output_dir="outputs", max_debug_frames=80, filter_mode="adaptive", show_raw=True,
        eye="auto", engine="v1", no_mediapipe=False, cft_helper=False,
        overlay_mode="clinical", disable_rit_validator=False, output_suffix=""):
    video_path = Path(video_path)
    out_dir = Path(output_dir) / (video_path.stem + "_tracked" + (output_suffix or ""))
    if disable_rit_validator:
        print("[RIT] validator DISABLED for this run (comparison/baseline mode). "
              "raw_* and valid_* will only differ by upstream V1 limbus checks.")

    # --- STAGE 0 GATE: tracking must not begin until landmarks are approved ---
    approved = load_approved(output_dir, video_path)
    if not approved or not approved.get("approved"):
        raise SystemExit(
            'Stage 0 not complete — landmarks not approved. Approve first:\n'
            f'    python app.py --video "{video_path}" --approve')

    # ---- V1 SCHEMA GATE (locked 2026-06-27) ------------------------------------
    # Locked V1 rule:
    #   No valid iris/limbus approval + no valid eye-opening/orbital margin contour approval
    #   = no V1 analysis.
    #
    # Required top-level fields:
    #   * approval_schema_version == "v1_iris_limbus_and_eye_contour"
    #   * tracking_target == "iris_limbus"
    #   * reference_target == "eye_opening_contour"
    #   * iris_limbus_approved == True
    #   * eye_opening_contour_approved == True
    #   * legacy_pupil_fallback_used == False
    #   * legacy_four_point_contour_fallback_used == False
    #   * eye_opening_contours dict (with at least one usable eye's polygon)
    #
    # Required per-eye fields under "eyes" (preferred) or legacy "per_eye_validity":
    #   * usable_for_v1, iris_limbus_approved, eye_opening_contour_approved
    #
    # Failure here writes a TOMBSTONE nystagmus_events.json with analysis_valid=false /
    # invalid_stage0_approval. No tracker is constructed. No detector runs. No overlay video
    # is produced. The clinician sees the rejection reason and is asked to re-approve.
    schema = approved.get("approval_schema_version")
    eyes_block = approved.get("eyes") or approved.get("per_eye_validity") or {}
    reasons = []
    if schema != "v1_iris_limbus_and_eye_contour":
        reasons.append(f"approval_schema_version is {schema!r} (expected "
                       f"'v1_iris_limbus_and_eye_contour')")
    if approved.get("tracking_target") not in (None, "iris_limbus"):
        reasons.append(f"tracking_target is {approved.get('tracking_target')!r} "
                       f"(expected 'iris_limbus')")
    if approved.get("reference_target") not in (None, "eye_opening_contour"):
        reasons.append(f"reference_target is {approved.get('reference_target')!r} "
                       f"(expected 'eye_opening_contour')")
    if not approved.get("iris_limbus_approved"):
        reasons.append("iris_limbus_approved is missing or false")
    if not approved.get("eye_opening_contour_approved"):
        reasons.append("eye_opening_contour_approved is missing or false")
    if approved.get("legacy_pupil_fallback_used"):
        reasons.append("legacy_pupil_fallback_used is true (pupil approval is not valid for V1)")
    if approved.get("legacy_four_point_contour_fallback_used"):
        reasons.append("legacy_four_point_contour_fallback_used is true "
                       "(four face landmarks are not a valid contour for V1)")
    if not approved.get("eye_opening_contours"):
        reasons.append("eye_opening_contours block is missing")
    # at least one eye must be explicitly usable_for_v1 with both flags
    any_eye_usable_for_v1 = False
    for ek in ("L", "R"):
        e = eyes_block.get(ek)
        if isinstance(e, dict) and e.get("usable_for_v1") and \
                e.get("iris_limbus_approved") and e.get("eye_opening_contour_approved"):
            any_eye_usable_for_v1 = True
            break
    if not any_eye_usable_for_v1:
        reasons.append("no eye has usable_for_v1 + iris_limbus_approved + "
                       "eye_opening_contour_approved all true")

    if reasons:
        _write_invalid_stage0_outputs(out_dir, approved, reasons=reasons)
        raise SystemExit(
            "Invalid Stage 0 approval for eye_vng V1. V1 requires full iris/limbus approval "
            "and eye-opening/orbital margin contour approval. This approval file appears to "
            "be legacy or incomplete. Please rerun --approve and mark the full iris/limbus "
            "circle and eye-opening contour.\n"
            "Failures:\n" + "\n".join("  - " + r for r in reasons) + "\n"
            f'Approve first:  python app.py --video "{video_path}" --approve')

    init_fr = int(approved["init_frame_number"])
    # Schema gate above forbids legacy fields, so reading the V1 "iris" block is now the only path.
    iris_appr = approved.get("iris") or {}
    Lp, Rp = iris_appr.get("L"), iris_appr.get("R")
    L = (Lp["x"], Lp["y"], Lp["radius"]) if Lp else None
    R = (Rp["x"], Rp["y"], Rp["radius"]) if Rp else None
    if not L and not R:
        raise SystemExit("Approved landmarks contain no usable eye.")
    # Per-eye usable_for_v1 — if an eye is marked unusable for V1, skip it even if a seed
    # exists. Both flags (iris_limbus_approved AND eye_opening_contour_approved) must hold.
    for ek in ("L", "R"):
        e = eyes_block.get(ek)
        if not isinstance(e, dict):
            continue
        if not (e.get("usable_for_v1") and e.get("iris_limbus_approved")
                and e.get("eye_opening_contour_approved")):
            if ek == "L":
                L = None
            else:
                R = None
    if not L and not R:
        raise SystemExit("Approved landmarks contain no usable eye after V1 validity check.")
    confirmed = {"L": L, "R": R}
    interocular = abs(R[0] - L[0]) if (L and R) else (L or R)[2] * 6.0
    print(f"Tracking from approved_landmarks.json (init frame {init_fr}).")

    detector = None if no_mediapipe else IrisDetector()
    dbg_dir = out_dir / "debug_frames"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # build the template tracker from the APPROVED init frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_fr - 1)
    ok, init_frame = cap.read()
    init_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY) if ok else np.zeros((height, width), np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    contour_trackers = {}
    contour_states = {}
    for side, iris in (("L", L), ("R", R)):
        if iris is None:
            continue
        pts = contour_from_approved(approved, side, iris)
        if pts is not None and len(pts) >= 6:
            contour_trackers[side] = EyeOpeningContourTracker(side, pts, init_gray)
            contour_states[side] = contour_trackers[side].state
    # IRIS RADIUS — STAGE-0 ONLY (clinician decision 2026-06-28).
    # MediaPipe has been REMOVED from the V1 clinical tracking path. MediaPipe's
    # iris-landmark radius (~44 px on this 1920x1080 clip) is not the clinical
    # limbus radius (~88 px); using it seeded V1 with the wrong radius and made
    # the tracker lock onto pupil-sized structures. V1 now reads the iris/limbus
    # radius DIRECTLY from the Stage-0 clinician-approved JSON; no detector call,
    # no fallback to MediaPipe. If Stage-0 radius is missing or invalid, V1
    # refuses to run.
    def _stage0_radius_or_refuse(stage0_iris, eye_label: str) -> float:
        if not stage0_iris:
            print(f"[V1-radius] ERROR: missing Stage-0 iris/limbus radius for eye "
                  f"{eye_label}. Refusing V1 tracking.")
            raise SystemExit(
                f"V1 refuses to run: Stage-0 iris/limbus radius for eye {eye_label} "
                "is missing. Re-approve Stage 0 before tracking."
            )
        r = float(stage0_iris[2] or 0.0)
        if not (r > 0):
            print(f"[V1-radius] ERROR: invalid Stage-0 iris/limbus radius (r={r}) "
                  f"for eye {eye_label}. Refusing V1 tracking.")
            raise SystemExit(
                f"V1 refuses to run: Stage-0 iris/limbus radius for eye {eye_label} "
                "is invalid. Re-approve Stage 0 before tracking."
            )
        return r
    lr = _stage0_radius_or_refuse(L, "L") if L else 0.0
    rr = _stage0_radius_or_refuse(R, "R") if R else 0.0
    if L:
        print(f"[V1-radius] L used={lr:.2f} source=stage0")
    if R:
        print(f"[V1-radius] R used={rr:.2f} source=stage0")
    # ENGINE SELECTION (V1 hierarchy: limbus = primary, contour = reference, CFT = optional helper).
    # Default = "v1" — the V1Tracker orchestrator (limbus fit + eye-opening contour, CFT only when
    # cft_helper=True). "composite" = the standalone CompositeFeatureTracker (CFT-first; kept for
    # comparison only). "template" = the LEGACY single-template IrisTracker.
    cft, tracker = {}, None
    if engine == "template":
        tracker = IrisTracker(init_gray, (L[0], L[1]) if L else None, (R[0], R[1]) if R else None,
                              lr, rr, interocular)
    elif engine == "composite":
        if L:
            cft["L"] = CompositeFeatureTracker("L", (L[0], L[1]), lr, init_gray, init_fr)
        if R:
            cft["R"] = CompositeFeatureTracker("R", (R[0], R[1]), rr, init_gray, init_fr)
    elif engine == "anatomical":
        # Anatomically-constrained iris–sclera boundary tracker (TRACKER_REDESIGN.md).
        # Stage-0 contour polygons are the containment box; rescue anchors from
        # teaching_anchors.json (if present) become absolute truth on their frames.
        from src.core.iris_anatomical_tracker import AnatomicalTracker
        approved_contours = (approved.get("eye_opening_contours") or {})
        # Rescue anchors are loaded once; the tracker filters by image_side_eye.
        try:
            import json as _json
            ta_path = out_dir / "teaching_anchors.json"
            anchors_blob = _json.loads(ta_path.read_text(encoding="utf-8")) if ta_path.exists() else {}
            anchors_list = anchors_blob.get("anchors", []) if isinstance(anchors_blob, dict) else []
        except Exception:
            anchors_list = []
        if anchors_list:
            print(f"[anatomical] loaded {len(anchors_list)} rescue anchors from teaching_anchors.json")

        def _build(eye_key, seed_xy, seed_r):
            tr = AnatomicalTracker(eye_key, seed_xy, seed_r, init_gray, init_fr)
            # The Stage-0 schema stores the contour as a LIST of {x,y} points directly
            # under each eye key (verified 2026-06-28 on the approved vestibular-neuritis
            # clip). Some older approvals may instead store {"points": [...]} — accept both.
            entry = approved_contours.get(eye_key)
            if isinstance(entry, dict):
                pts = entry.get("points")
            elif isinstance(entry, list):
                pts = entry
            else:
                pts = None
            if not pts:
                raise SystemExit(
                    f"[anatomical-engine] Stage 0 missing eye-opening contour for eye {eye_key}.\n"
                    f"Run: python app.py --video \"{video_path}\" --approve\n"
                    "The anatomical engine requires an approved eye-opening contour per eye."
                )
            tr.set_contour(pts)
            tr.set_anchors(anchors_list)
            return tr
        if L:
            cft["L"] = _build("L", (L[0], L[1]), lr)
        if R:
            cft["R"] = _build("R", (R[0], R[1]), rr)
    else:  # engine == "v1"
        if L:
            cft["L"] = V1Tracker("L", (L[0], L[1]), lr, init_gray, init_fr,
                                  use_cft_helper=cft_helper)
        if R:
            cft["R"] = V1Tracker("R", (R[0], R[1]), rr, init_gray, init_fr,
                                  use_cft_helper=cft_helper)
    print(f"Tracking engine: {engine}{' (+CFT helper)' if (engine == 'v1' and cft_helper) else ''}")

    # ---- V1 post-tracking plausibility filter (clinician reset 2026-06-28) ----
    # Keeps the working V1 tracker as-is and adds only minimal anatomical
    # guardrails on its per-frame output. Lenient by design: rejects only when a
    # result is anatomically impossible (centre clearly outside the Stage-0
    # contour, OR radius wildly off Stage-0). Marks the FrameMeasurement as
    # failed; does NOT touch V1's internal last_good / prior. The earlier
    # over-engineered anatomical engine is now opt-in (--engine anatomical) and
    # not part of the clinical workflow.
    plausibility_polys: dict = {}
    plausibility_stage0_r: dict = {}
    PLAUSIBILITY_OUTSIDE_TOL_PX = 20.0   # centre allowed up to 20 px outside contour
    PLAUSIBILITY_R_LO = 0.50             # reject if r < 50 % of Stage-0
    PLAUSIBILITY_R_HI = 2.00             # reject if r > 200 % of Stage-0
    # ------------ orbit guardrail discrimination (clinician spec 2026-06-29) ----
    # When the iris is outside the static Stage-0 oval, distinguish between:
    #   * tracker drift onto cheek/eyebrow/nose (clinically invalid), and
    #   * the iris is still being tracked correctly but the head has moved, so the
    #     static Stage-0 oval is anatomically stale relative to the current head
    #     pose (clinically VALID; we just record that the oval is stale).
    # Heuristic:
    #   1. Maintain a short history of recent VALID iris centres per eye.
    #   2. When a frame is outside the contour, compute the expected position by
    #      extrapolating the median per-frame iris motion from the history.
    #   3. If the current centre is within MOTION_EXPECTED_TOL_PX of that
    #      expected position AND the radius is still within the radius band,
    #      the iris is moving smoothly with the head -> 'good_iris_but_static_orbit_stale'.
    #   4. Otherwise -> 'tracker_drift_outside_orbit'.
    ORBIT_MOTION_HISTORY = 5
    ORBIT_MOTION_EXPECTED_TOL_PX = 30.0
    iris_history: dict = {"L": [], "R": []}    # [(frame_no, cx, cy, r), ...]
    if engine == "v1":
        _v1_contours_src = (approved.get("eye_opening_contours") or {})
        _v1_iris_src = (approved.get("iris") or {})
        for _ek in ("L", "R"):
            _entry = _v1_contours_src.get(_ek)
            _pts = _entry.get("points") if isinstance(_entry, dict) else (_entry if isinstance(_entry, list) else None)
            if _pts:
                try:
                    plausibility_polys[_ek] = np.array(
                        [[float(p["x"]), float(p["y"])] for p in _pts],
                        np.float32).reshape(-1, 1, 2)
                except Exception:
                    plausibility_polys[_ek] = None
            _iris_ent = _v1_iris_src.get(_ek) or {}
            try:
                plausibility_stage0_r[_ek] = float(_iris_ent.get("radius") or 0.0)
            except Exception:
                plausibility_stage0_r[_ek] = 0.0
        _n_polys = sum(1 for v in plausibility_polys.values() if v is not None)
        print(f"[v1-plausibility] enabled with {_n_polys} eye contour(s); "
              f"outside_tol={PLAUSIBILITY_OUTSIDE_TOL_PX:.0f}px, "
              f"radius_band=[{PLAUSIBILITY_R_LO:.2f},{PLAUSIBILITY_R_HI:.2f}] x Stage-0")

    def _classify_orbit_outside(fm, eye_key: str, frame_no: int) -> str:
        """When the iris centre is outside the static Stage-0 contour, decide
        whether this is tracker drift or just a stale static contour because the
        head moved. Returns one of:
            'tracker_drift_outside_orbit'
            'good_iris_but_static_orbit_stale'
        """
        hist = iris_history.get(eye_key) or []
        if len(hist) < 2:
            # No motion baseline yet: be conservative and call it drift.
            return "tracker_drift_outside_orbit"
        cx, cy = float(fm.iris_centre[0]), float(fm.iris_centre[1])
        r = float(fm.iris_radius)
        # Radius must still be plausible to claim 'good iris'.
        r_st0 = plausibility_stage0_r.get(eye_key, 0.0)
        if r_st0 > 0 and not (PLAUSIBILITY_R_LO * r_st0 <= r <= PLAUSIBILITY_R_HI * r_st0):
            return "tracker_drift_outside_orbit"
        # Per-frame median motion from the history (excluding very stale samples).
        dxs, dys = [], []
        for i in range(1, len(hist)):
            (f0, x0, y0, _), (f1, x1, y1, _) = hist[i - 1], hist[i]
            df = max(1, f1 - f0)
            dxs.append((x1 - x0) / df)
            dys.append((y1 - y0) / df)
        mdx = float(np.median(dxs)) if dxs else 0.0
        mdy = float(np.median(dys)) if dys else 0.0
        # Project last valid centre forward to current frame.
        last_f, last_x, last_y, _ = hist[-1]
        steps = frame_no - last_f
        exp_x = last_x + mdx * steps
        exp_y = last_y + mdy * steps
        dist_to_expected = float(np.hypot(cx - exp_x, cy - exp_y))
        if dist_to_expected <= ORBIT_MOTION_EXPECTED_TOL_PX:
            return "good_iris_but_static_orbit_stale"
        return "tracker_drift_outside_orbit"

    def _update_iris_history(fm, eye_key: str, frame_no: int) -> None:
        """Append the current (valid) iris centre to the per-eye history buffer."""
        if fm.iris_centre is None or fm.iris_radius is None:
            return
        if not fm.iris_valid:
            return
        cx, cy = float(fm.iris_centre[0]), float(fm.iris_centre[1])
        r = float(fm.iris_radius)
        buf = iris_history.setdefault(eye_key, [])
        buf.append((int(frame_no), cx, cy, r))
        if len(buf) > ORBIT_MOTION_HISTORY:
            del buf[0]

    def _apply_v1_plausibility(fm, eye_key: str, frame_no: int, gray=None, bgr=None):
        """Validate the V1 iris candidate against the V2 clinical definition (dark circular/arc iris +
        scleral white support + Stage-0 radius + orbital containment) via
        `iris_candidate_validator.validate_iris_candidate`. On failure, mark the FM as a gap
        (iris_valid False, drift flag + reason) and log; V1's internal state is never touched.

        Preserves the 2026-06-29 orbit-guardrail discrimination: when the SOLE failure is leaving the
        orbital oval but the image content is still iris-like (dark arc + sclera all passed → reason
        'outside_orbit'), defer to `_classify_orbit_outside` so a head-moved frame (static oval gone
        stale) stays clinically VALID, while genuine drift onto face/cheek/brow is rejected.

        Conservative by design: a false negative (freeze / needs-rescue) is acceptable; a false
        positive (locking onto skin/brow/shadow) is dangerous.
        """
        if fm.iris_centre is None or fm.iris_radius is None:
            return
        # RIT precondition: Stage-0 clinician-approved limbus radius is mandatory. Without it the
        # validator's radius gate silently disables — refuse to run the frame instead.
        stage0_r = plausibility_stage0_r.get(eye_key, 0.0)
        if stage0_r <= 0:
            print(f"[iris-validator][{eye_key}] fr={frame_no} reject=no_stage0_radius")
            fm.iris_valid = False
            fm.drift_flag = True
            fm.drift_reason = "iris_validator:no_stage0_radius"
            return
        cx, cy = float(fm.iris_centre[0]), float(fm.iris_centre[1])
        r = float(fm.iris_radius)
        hist = iris_history.get(eye_key) or []
        last_centre = (hist[-1][1], hist[-1][2]) if hist else None
        res = validate_iris_candidate(
            gray, bgr, cx, cy, r,
            stage0_radius=stage0_r,
            orbit_poly=plausibility_polys.get(eye_key),
            last_centre=last_centre, interocular=interocular)

        if res.accepted:
            _update_iris_history(fm, eye_key, frame_no)
            if frame_no % 90 == 0:                 # periodic heartbeat so the log shows liveness
                print(res.log_line(eye_key, frame_no))
            return

        # Content is iris-like but the centre left the oval → distinguish head-moved-stale-oval from
        # true drift (keeps the clinically-valid stale-oval frames; rejects real face drift).
        if res.reject_reason == "outside_orbit":
            cls = _classify_orbit_outside(fm, eye_key, frame_no)
            if cls == "good_iris_but_static_orbit_stale":
                fm.drift_reason = "orbit_status:good_iris_but_static_orbit_stale|iris_validator:outside_orbit"
                _update_iris_history(fm, eye_key, frame_no)
                print(res.log_line(eye_key, frame_no) + " orbit_status=good_iris_but_static_orbit_stale")
                return

        # Genuine reject: freeze this frame (mark invalid). iris_centre/iris_radius are left as V1
        # produced them so the overlay can dim/redden a "needs rescue" marker; downstream consumers
        # gate on iris_valid + drift_flag.
        print(res.log_line(eye_key, frame_no))
        fm.iris_valid = False
        fm.drift_flag = True
        fm.drift_reason = f"iris_validator:{res.reject_reason}"

    face_appr = approved.get("face_landmarks") or {}
    init_face_mp = (detector.face_landmarks(init_frame) or {}) if (detector and face_appr and ok) else {}
    face_tracker = FaceLandmarkTracker(face_appr, init_face_mp) if (face_appr and detector) else None
    face_names = list(face_appr.keys()) if (face_appr and no_mediapipe) else (face_tracker.names() if face_tracker else [])

    # jitter filters: 3-frame median + 1€ (adaptive). Reset on lost/blink (never smooth a gap).
    pf = {"L": PointFilter(filter_mode, fps), "R": PointFilter(filter_mode, fps)}
    ff = {n: PointFilter(filter_mode, fps) for n in face_names}
    print(f"Filter: {filter_mode}")
    scale = min(1.0, OVERLAY_MAX_W / width)
    ow, oh = int(width * scale), int(height * scale)
    # CLINICAL overlay mode (default): the output canvas IS the video — minimal labels only,
    # nothing else. DEBUG overlay mode: extend the canvas downward with a dedicated trace strip
    # below the video (eyes never covered) and render contour/limbus/state banner on the video.
    if overlay_mode == "debug":
        band_h = max(170, int(0.34 * oh))
        out_h = oh + band_h
    else:
        band_h = 0
        out_h = oh
    writer = cv2.VideoWriter(str(out_dir / "tracking_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, out_h))
    print(f"Overlay mode: {overlay_mode}")
    # iris (x,y) per frame in OVERLAY-PIXEL coords — used by the clinical pass-2 annotator to
    # choose forehead vs lower-face placement so the status band never sits on the eyes.
    iris_xy_per_frame_overlay = []

    csv_f = (out_dir / "tracking.csv").open("w", newline="", encoding="utf-8")
    cw = csv.writer(csv_f)
    cw.writerow(["frame_number", "timestamp_ms", "time_sec",
                 # raw_* = always the detected centre (preserved, incl. blinks); valid_* = blank during
                 # blink/occlusion/jump (never invented). Interpret eye movement from valid_*, not raw_*.
                 "raw_left_iris_center_x", "raw_left_iris_center_y", "raw_right_iris_center_x", "raw_right_iris_center_y",
                 "valid_left_iris_center_x", "valid_left_iris_center_y", "valid_right_iris_center_x", "valid_right_iris_center_y",
                 "filtered_left_iris_center_x", "filtered_left_iris_center_y",
                 "filtered_right_iris_center_x", "filtered_right_iris_center_y",
                 # EYE-LOCAL (clinical) coords: x 0=inner→1=outer canthus, y 0=upper→1=lower margin
                 "left_eye_local_x", "left_eye_local_y",
                 "right_eye_local_x", "right_eye_local_y",
                 "left_iris_dx_from_initial", "left_iris_dy_from_initial",
                 "right_iris_dx_from_initial", "right_iris_dy_from_initial",
                 "iris_valid_left", "iris_valid_right",
                 "reference_valid_left", "reference_valid_right",
                 "clinical_relative_valid_left", "clinical_relative_valid_right",
                 "reference_uncertain_left", "reference_uncertain_right",
                 "reference_state_left", "reference_state_right",
                 "reference_confidence_left", "reference_confidence_right",
                 "aperture_quality", "transform_residual_px", "landmark_count_used",
                 "left_iris_radius", "right_iris_radius",
                 "tracking_status_left", "tracking_status_right",
                 "artifact_type_left", "artifact_type_right",
                 "ear_left", "ear_right",
                 "confidence_left", "confidence_right",
                 # drift guard: "" = iris anatomically attached; outside_aperture/off_anchor = invalid
                 # (eye-local dropped, not plotted); review_excursion = large but valid, flagged to review
                 "iris_flag_left", "iris_flag_right",
                 # ---- composite engine (EYEVNG_TRACKING_SPECIFICATION.md §G), per eye ----
                 "state_left", "state_right", "validity_left", "validity_right",
                 "drift_flag_left", "drift_flag_right", "drift_reason_left", "drift_reason_right",
                 "rotation_theta_left", "rotation_theta_right",
                 "feature_confidence_mean_left", "feature_confidence_mean_right",
                 "composite_confidence_left", "composite_confidence_right",
                 "frame_confidence_left", "frame_confidence_right",
                 "n_active_left", "n_active_right", "n_trusted_left", "n_trusted_right"])

    face_csv_f = fcw = None
    if face_tracker:
        face_csv_f = (out_dir / "face_landmarks.csv").open("w", newline="", encoding="utf-8")
        fcw = csv.writer(face_csv_f)
        hdr = ["frame_number", "time_sec"]
        for n in face_names:
            hdr += [f"{n}_x", f"{n}_y", f"{n}_status"]
        fcw.writerow(hdr)

    # head-motion correction reference (the user-approved init landmarks/iris)
    init_face_pts = {n: (v["x"], v["y"]) for n, v in face_appr.items() if v}
    init_iris = {"L": (L[0], L[1]) if L else (None, None), "R": (R[0], R[1]) if R else (None, None)}
    stabilizer = HeadStabilizer(init_face_pts, init_iris, width)

    times = []
    rlx, rly, rrx, rry = [], [], [], []      # RAW left/right x/y (image space)
    flx, fly, frx, fry = [], [], [], []      # FILTERED left/right x/y (image space)
    cor_lh, cor_lv, cor_rh, cor_rv = [], [], [], []   # RAW head-corrected eye-in-head (not over-filtered)
    art_L, art_R = [], []                    # per-frame artifact_type per eye (blink/occlusion/jump/none)
    stat_L, stat_R = [], []                  # per-frame tracking status per eye (for reacquisition stats)
    poor_flags = []                          # True where transform quality is poor/degenerate
    q_counts = {}                            # transform quality tally
    iris_flag_L, iris_flag_R = [], []        # per-frame drift-guard state per eye ("" if attached)
    fm_ser = {"L": [], "R": []}              # per-frame composite FrameMeasurement per eye (None pre-init)
    reference_uncertain_counts = {"L": {}, "R": {}}
    prev_valid = {"L": None, "R": None}      # last anatomically-valid eye-local (h,v) per eye
    flag_counts = {}                         # drift-guard state tally (off_anchor/outside_aperture/review_excursion)
    saved_drift = 0                          # debug frames saved for drift-suspected frames
    tracked_series = []
    counts = {}
    # DISPLAY eye selection: show ONE eye by default (conjugate nystagmus on both eyes is redundant
    # and cluttered); use --eye both only when the eyes differ, e.g. INO. The CSV always keeps BOTH.
    show_both = (eye == "both")
    forced = {"left": "L", "right": "R"}.get(eye)
    if forced and not confirmed.get(forced):
        forced = "L" if confirmed.get("L") else "R"     # asked-for eye wasn't approved → fall back
    chosen_eye = forced                                  # for "auto": locked after a short warmup
    valid_L = valid_R = 0
    seen_bad = saved_bad = 0
    stride = max(1, total // max(1, max_debug_frames))
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr += 1
        t = fr / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cache = {}

        def mp_detect():
            if detector is None:
                return False, EyeObs(False), EyeObs(False)
            if "r" not in cache:
                cache["r"] = detector.detect(frame)
            return cache["r"]

        face_tracks = {}
        if no_mediapipe and face_appr and fr >= init_fr:
            for name, pt in face_appr.items():
                face_tracks[name] = Track("tracked", pt["x"], pt["y"], confidence="high") if pt else Track("lost", None, None)
        elif face_tracker and fr >= init_fr:
            def face_backup():
                if "f" not in cache:
                    cache["f"] = detector.face_landmarks(frame) or {}
                return cache["f"]
            face_tracks = face_tracker.step(gray, face_backup)

        def _pt(name):
            tr = face_tracks.get(name)
            return (tr.x, tr.y) if tr and tr.x is not None else None

        def _aperture_by_eye(s):
            cs = contour_states.get(s)
            if cs is not None:
                ex = cs.extents()
                return ex["medial"], ex["lateral"], ex["upper"], ex["lower"]
            return (None, None, None, None)

        def _reference_for(s):
            cs = contour_states.get(s)
            if cs is None:
                return "lost", 0.0
            return cs.reference_state, cs.confidence

        fm_L = fm_R = None
        if fr < init_fr:
            lt = Track("lost", None, None)
            rt = Track("lost", None, None)
        elif fr == init_fr:
            lt = Track("initialized", L[0], L[1], L[2], "high") if L else Track("lost", None, None)
            rt = Track("initialized", R[0], R[1], R[2], "high") if R else Track("lost", None, None)
        elif engine == "template":
            if no_mediapipe:
                raise SystemExit("--no-mediapipe requires --engine v1 (or composite).")
            lt, rt = tracker.step(gray, mp_detect)        # LEGACY single-template engine (fallback)
        else:
            # Drive each eye's per-eye engine (V1 orchestrator by default; CFT-only when
            # engine="composite"). The eye-opening contour is the V1 reference frame; medial/
            # lateral/upper/lower are derived from the tracked contour, not from four
            # independently tracked points. MediaPipe (ear + iris) — when present — feeds
            # blink classification and recovery ONLY (never the per-frame clinical centre).
            _f, le_mp, re_mp = mp_detect()
            tms = int(round(t * 1000))

            def _mpiris(o):
                return (o.single_x, o.single_y, o.iris_radius) if (o.detected and o.iris_radius) else None
            for side, ctr in contour_trackers.items():
                contour_states[side] = ctr.step(gray) if fr > init_fr else ctr.state
            if "L" in cft:
                ref_state, ref_conf = _reference_for("L")
                fm_L = cft["L"].step(gray, fr, tms, t, le_mp.ear if le_mp.detected else None,
                                      aperture=_aperture_by_eye("L"), mp_iris=_mpiris(le_mp), bgr=frame,
                                      reference_state=ref_state, reference_confidence=ref_conf)
                # V1 iris-candidate validator — only fires for engine=="v1"; needs the frame
                # (gray + colour) to judge dark-arc + scleral-white content.
                if engine == "v1" and not disable_rit_validator:
                    _apply_v1_plausibility(fm_L, "L", fr, gray, frame)
                lt = fm_to_track(fm_L)
            else:
                lt = Track("lost", None, None)
            if "R" in cft:
                ref_state, ref_conf = _reference_for("R")
                fm_R = cft["R"].step(gray, fr, tms, t, re_mp.ear if re_mp.detected else None,
                                      aperture=_aperture_by_eye("R"), mp_iris=_mpiris(re_mp), bgr=frame,
                                      reference_state=ref_state, reference_confidence=ref_conf)
                if engine == "v1" and not disable_rit_validator:
                    _apply_v1_plausibility(fm_R, "R", fr, gray, frame)
                rt = fm_to_track(fm_R)
            else:
                rt = Track("lost", None, None)
        fstatus = frame_status(lt, rt)
        counts[fstatus] = counts.get(fstatus, 0) + 1

        # facial landmarks first — they define the head reference frame for correction
        cur_face = {n: (ftr.x, ftr.y) for n, ftr in face_tracks.items() if ftr.x is not None}

        # FILTERED image-space iris (for the image-space trace); raw kept separately.
        flx_, fly_ = pf["L"](lt.x, lt.y, t)
        frx_, fry_ = pf["R"](rt.x, rt.y, t)
        # HEAD-CORRECTED eye-in-head from the RAW iris. Tiered transform quality: only degenerate
        # transforms are rejected (gap); imperfect ones are kept + flagged. Corrected trace is NOT
        # over-filtered (raw corrected) so nystagmus beats are preserved.
        def _pt(name):
            tr = face_tracks.get(name)
            return (tr.x, tr.y) if tr and tr.x is not None else None

        # EYE-LOCAL coordinates — THE clinical trace: iris position WITHIN this eye's aperture box
        # (inner/outer canthus + upper/lower margin). Head/frame independent; uses NO face/head pose.
        # Built from the VALID iris (lt.x/y) so blink frames stay gaps. x: 0=inner→1=outer canthus;
        # y: 0=upper→1=lower margin. (cor_* arrays now hold eye-local x/y, not the old canthus pixels.)
        def _aperture_for(iris):
            # Pair the iris with its OWN eye's aperture by proximity — robust to whatever L/R
            # convention the iris vs the landmark naming happen to use (they can differ).
            if iris[0] is None:
                return (None, None, None, None)
            best = None
            for s in ("L", "R"):
                ic, oc = _pt("inner_canthus_" + s), _pt("outer_canthus_" + s)
                if ic and oc:
                    mx, my = (ic[0] + oc[0]) / 2, (ic[1] + oc[1]) / 2
                    dd = (mx - iris[0]) ** 2 + (my - iris[1]) ** 2
                    if best is None or dd < best[0]:
                        best = (dd, s)
            if best is None:
                return (None, None, None, None)
            s = best[1]
            return (_pt("inner_canthus_" + s), _pt("outer_canthus_" + s),
                    _pt("upper_margin_" + s), _pt("lower_margin_" + s))

        clh, clv = fm_L.eye_local if fm_L is not None else (None, None)
        crh, crv = fm_R.eye_local if fm_R is not None else (None, None)

        def _san(v):     # eye-local is ~0..1; anything well outside means a bad/mis-paired landmark
            return v if (v is not None and -0.5 <= v <= 1.5) else None
        clh, clv, crh, crv = _san(clh), _san(clv), _san(crh), _san(crv)
        n_used = sum((contour_states.get(s).n_inliers if contour_states.get(s) else 0) for s in ("L", "R"))
        ref_valid_any = any((contour_states.get(s) and contour_states[s].reference_valid) for s in ("L", "R"))
        quality = "good" if ((fm_L is None or fm_L.reference_valid) and (fm_R is None or fm_R.reference_valid)) else \
                  ("fair" if ref_valid_any else "degenerate")
        residual = None
        q_counts[quality] = q_counts.get(quality, 0) + 1
        poor_flags.append(quality == "degenerate")

        # ---- IRIS DRIFT GUARDS — geometry only; MediaPipe iris = independent anatomical anchor ----
        # An invalidating guard (outside_aperture / off_anchor) drops the eye-local value → a gap, so a
        # drifted marker is NEVER plotted as false eye movement. review_excursion keeps plotting (a real
        # fast-phase is a valid big jump) but flags the frame for visual confirmation.
        st_L = "reference_uncertain" if (fm_L is not None and fm_L.reference_uncertain and fm_L.iris_valid) else ""
        st_R = "reference_uncertain" if (fm_R is not None and fm_R.reference_uncertain and fm_R.iris_valid) else ""
        if clh is not None:                        # advance the anchor for the next excursion check
            prev_valid["L"] = (clh, clv)
        if crh is not None:
            prev_valid["R"] = (crh, crv)
        iris_flag_L.append(st_L); iris_flag_R.append(st_R)
        if st_L:
            flag_counts[st_L] = flag_counts.get(st_L, 0) + 1
        if st_R:
            flag_counts[st_R] = flag_counts.get(st_R, 0) + 1
        for side, fm in (("L", fm_L), ("R", fm_R)):
            if fm is not None and fm.reference_uncertain:
                reason = fm.reference_state or "uncertain"
                reference_uncertain_counts[side][reason] = reference_uncertain_counts[side].get(reason, 0) + 1

        # raw_* = always-detected centre (preserved); valid (lt.x/y) is None during blink/occlusion/jump.
        l_raw = (lt.raw_x if lt.raw_x is not None else lt.x, lt.raw_y if lt.raw_y is not None else lt.y)
        r_raw = (rt.raw_x if rt.raw_x is not None else rt.x, rt.raw_y if rt.raw_y is not None else rt.y)
        times.append(round(t, 4))
        rlx.append(l_raw[0]); rly.append(l_raw[1]); rrx.append(r_raw[0]); rry.append(r_raw[1])
        flx.append(flx_); fly.append(fly_); frx.append(frx_); fry.append(fry_)
        cor_lh.append(clh); cor_lv.append(clv); cor_rh.append(crh); cor_rv.append(crv)
        art_L.append(lt.artifact_type); art_R.append(rt.artifact_type)
        stat_L.append(lt.status); stat_R.append(rt.status)
        fm_ser["L"].append(fm_L); fm_ser["R"].append(fm_R)
        tracked_series.append((flx_, fly_) if flx_ is not None else None)

        cl, crr = _fm_cols(fm_L), _fm_cols(fm_R)
        composite_cols = [v for pair in zip(cl, crr) for v in pair]   # interleave L,R per field
        def _fm_attr(fm, name):
            return getattr(fm, name) if fm is not None else None
        def _fm_bool(fm, name):
            return int(bool(getattr(fm, name))) if fm is not None else ""
        cw.writerow([fr, int(round(t * 1000)), round(t, 4),
                     _r(l_raw[0]), _r(l_raw[1]), _r(r_raw[0]), _r(r_raw[1]),
                     _r(lt.x), _r(lt.y), _r(rt.x), _r(rt.y),
                     _r(flx_), _r(fly_), _r(frx_), _r(fry_),
                     _r(clh), _r(clv), _r(crh), _r(crv),
                     _r(_fm_attr(fm_L, "iris_dx_from_initial")), _r(_fm_attr(fm_L, "iris_dy_from_initial")),
                     _r(_fm_attr(fm_R, "iris_dx_from_initial")), _r(_fm_attr(fm_R, "iris_dy_from_initial")),
                     _fm_bool(fm_L, "iris_valid"), _fm_bool(fm_R, "iris_valid"),
                     _fm_bool(fm_L, "reference_valid"), _fm_bool(fm_R, "reference_valid"),
                     _fm_bool(fm_L, "clinical_relative_valid"), _fm_bool(fm_R, "clinical_relative_valid"),
                     _fm_bool(fm_L, "reference_uncertain"), _fm_bool(fm_R, "reference_uncertain"),
                     _fm_attr(fm_L, "reference_state") or "", _fm_attr(fm_R, "reference_state") or "",
                     _r(_fm_attr(fm_L, "reference_confidence")), _r(_fm_attr(fm_R, "reference_confidence")),
                     quality, _r(residual), n_used,
                     _r(lt.radius), _r(rt.radius),
                     lt.status, rt.status, lt.artifact_type, rt.artifact_type,
                     _r(lt.ear), _r(rt.ear), lt.confidence, rt.confidence,
                     st_L, st_R] + composite_cols)

        if fcw:
            row = [fr, round(t, 4)]
            for n in face_names:
                ftr = face_tracks.get(n)
                if ftr and ftr.x is not None:
                    row += [_r(ftr.x), _r(ftr.y), ftr.status]
                else:
                    row += ["", "", ftr.status if ftr else "lost"]
            fcw.writerow(row)

        # OVERLAY = RAW detected positions (verification: the marker must sit ON the iris with no
        # lag). The FILTERED signal is used only for the VNG trace + CSV, where shimmer matters and a
        # small filter lag is harmless. This keeps the overlay honest and never sliding.
        if clh is not None:
            valid_L += 1
        if crh is not None:
            valid_R += 1
        if not show_both and chosen_eye is None and max(valid_L, valid_R) >= 25:
            chosen_eye = "L" if valid_L >= valid_R else "R"   # lock to the better-tracked eye

        if overlay_mode == "debug":
            vis = draw_overlay(frame, scale, lt, rt, fstatus, fr, t, None,
                               contours=contour_states)
            # debug-only: composite features + per-eye state + frame_confidence banner.
            draw_composite(vis, scale, cft or {}, fm_L, fm_R)
            if show_both:
                # Clinical labels: anatomical patient eye on each side of the image
                # (front-facing video → image L = patient Right, image R = patient Left).
                chans = [("Contour-local H  (iris-circle centre / eye-opening contour)   "
                          "green = Right   blue = Left",
                          [(cor_lh, GREEN), (cor_rh, BLUE)]),
                         ("Contour-local V  (iris-circle centre / eye-opening contour)   "
                          "green = Right   blue = Left",
                          [(cor_lv, GREEN), (cor_rv, BLUE)])]
            else:
                e = chosen_eye or ("R" if valid_R >= valid_L else "L")
                col = GREEN if e == "L" else BLUE
                side = patient_eye_label(e)                      # 'Left' / 'Right'
                chans = [(f"Contour-local H  (iris-circle centre / contour)   Selected eye: {side}",
                          [(cor_lh if e == "L" else cor_rh, col)]),
                         (f"Contour-local V  (iris-circle centre / contour)   Selected eye: {side}",
                          [(cor_lv if e == "L" else cor_rv, col)])]
            canvas = np.zeros((out_h, ow, 3), np.uint8)
            canvas[:oh] = vis
            cv2.line(canvas, (0, oh), (ow, oh), (60, 60, 60), 1)
            superimpose_traces(canvas, times, chans, t, oh)
            writer.write(canvas)
        else:
            # CLINICAL overlay mode (default): video + thin eye-opening contour outline +
            # state-coloured iris-circle + small centre cross + 'Left'/'Right' anatomical labels.
            # Everything else (M/L/U/D extents, frame number, timestamp, state banner, fc/ic/rc/cov,
            # trace strip, "Torsion: not assessed" footnote) is excluded by design — debug mode only.
            # Pass `chosen_eye` so the better-tracked eye is drawn brightly and the other eye
            # is dimmed. During warm-up (before chosen_eye locks) both eyes look equal — that
            # only lasts a fraction of a second on most clips. `forced` (from --eye L|R) wins.
            sel_for_render = forced or chosen_eye
            canvas = draw_clinical_overlay(frame, scale, lt, rt,
                                           selected_image_eye=sel_for_render,
                                           contours=contour_states)
            writer.write(canvas)
        # Cache the tracked iris (x,y) in OVERLAY-pixel coords for the clinical pass-2 annotator.
        pts_this_frame = []
        for tr in (lt, rt):
            if tr.x is not None:
                pts_this_frame.append((int(tr.x * scale), int(tr.y * scale)))
        iris_xy_per_frame_overlay.append(pts_this_frame)
        # Debug frame saves use the same canvas the overlay writer received this iteration.
        if fstatus in ("uncertain", "blink_or_occluded", "lost") and fr >= init_fr:
            seen_bad += 1
            if seen_bad % stride == 0 and saved_bad < max_debug_frames:
                cv2.imwrite(str(dbg_dir / f"{fr:06d}_{fstatus}.png"), canvas)
                saved_bad += 1
        # save a debug frame whenever a drift guard fired (visual proof of where the marker sat)
        drift_reason = st_L or st_R
        if drift_reason and fr >= init_fr and saved_drift < max_debug_frames:
            cv2.imwrite(str(dbg_dir / f"{fr:06d}_{drift_reason}.png"), canvas)
            saved_drift += 1
    cap.release(); writer.release(); csv_f.close()
    if face_csv_f:
        face_csv_f.close()

    # ---- BEST-EYE SELECTION ---------------------------------------------------------
    # V1 clinical rule: the main clinical trace uses ONE eye — the better-tracked one. Both eyes are
    # always preserved in the CSV; this only chooses which eye drives the CLINICAL traces, the velocity
    # plot, and the report. `--eye left|right` forces a side; `--eye both` keeps both eyes on the
    # clinical trace (intended for INO / skew / dysconjugate / strabismus / research use). `--eye auto`
    # picks the eye with the higher composite quality score, using the same metrics the V1 spec defines.
    def composite_stats(fms, side):
        """Per-eye V1 summary: valid-frame %, drift counts by reason, state distribution,
        mean frame_confidence, and feature-survival stats. Reusable for both metadata and selection."""
        fms = [m for m in fms if m is not None]
        if not fms:
            return None
        n = len(fms)
        valid = [m for m in fms if m.validity == "valid"]
        ref_valid = [m for m in fms if getattr(m, "reference_valid", False)]
        rel_valid = [m for m in fms if getattr(m, "clinical_relative_valid", False)]
        drift = {}
        for m in fms:
            if m.drift_flag and m.drift_reason:
                drift[m.drift_reason] = drift.get(m.drift_reason, 0) + 1
        states = {}
        for m in fms:
            states[m.state] = states.get(m.state, 0) + 1
        fc = [m.frame_confidence for m in valid]
        return {
            "tracked_frames": n,
            "valid_frames": len(valid),
            "pct_valid": round(100.0 * len(valid) / n, 1),
            "reference_valid_frames": len(ref_valid),
            "pct_reference_valid": round(100.0 * len(ref_valid) / n, 1),
            "clinical_relative_valid_frames": len(rel_valid),
            "pct_clinical_relative_valid": round(100.0 * len(rel_valid) / n, 1),
            "reference_uncertain_by_reason": reference_uncertain_counts.get(side, {}),
            "drift_frames": sum(1 for m in fms if m.drift_flag),
            "drift_by_reason": drift,
            "state_counts": states,
            "mean_frame_confidence": round(float(np.mean(fc)), 3) if fc else 0.0,
            "tracking_confidence": round(sum(c >= 0.5 for c in fc) / max(1, len(fc)), 3) if fc else 0.0,
            "mean_n_trusted": round(float(np.mean([m.n_trusted for m in valid])), 1) if valid else 0,
            "mean_n_active": round(float(np.mean([m.n_active for m in valid])), 1) if valid else 0,
        }

    eye_stats = {"L": composite_stats(fm_ser["L"], "L"),
                 "R": composite_stats(fm_ser["R"], "R")}

    def _quality_score(s):
        """Composite quality score (0..1) for best-eye selection. Weights chosen so a high
        clinical-relative-valid % dominates; mean frame_confidence and the iris-/reference-valid
        rates are tie-breakers; drift fraction is subtracted (any drift hurts). All inputs come
        from V1 engine_stats so the score is symmetric across eyes."""
        if s is None or s["tracked_frames"] == 0:
            return -1.0
        n = max(1, s["tracked_frames"])
        iris_pct = s["pct_valid"] / 100.0
        ref_pct = s["pct_reference_valid"] / 100.0
        rel_pct = s["pct_clinical_relative_valid"] / 100.0
        mfc = float(s["mean_frame_confidence"])
        drift_frac = s["drift_frames"] / n
        # 0.40 relative-valid + 0.25 mean-fc + 0.15 iris-valid + 0.10 ref-valid − 0.25·drift
        return 0.40 * rel_pct + 0.25 * mfc + 0.15 * iris_pct + 0.10 * ref_pct - 0.25 * drift_frac

    sel_scores = {k: _quality_score(eye_stats[k]) for k in ("L", "R")}
    have = {k for k, v in eye_stats.items() if v is not None}

    # Decision: respect the user override; otherwise pick the better eye automatically.
    # `selected_image_eye` is the image-side shorthand (L/R/both/None) used by the trace +
    # display code; `selected_patient_eye` is the anatomical label for clinical reporting.
    if eye == "both":
        selected_image_eye = "both"
        selection_mode = "user_override_both"
        selection_reason = ("Both-eye mode requested by --eye both. The clinical trace shows both "
                           "eyes (intended for INO / skew / dysconjugate / strabismus / research "
                           "use). CSV always retains both eyes regardless.")
    elif forced and forced in have:
        selected_image_eye = forced
        selection_mode = "user_override"
        selection_reason = (
            f"Eye locked by --eye {eye} → {IMAGE_EYE_NAME[forced]} "
            f"({IMAGE_TO_PATIENT_EYE[forced]}). "
            f"Quality scores: {IMAGE_EYE_NAME['L']} {sel_scores['L']:.3f}, "
            f"{IMAGE_EYE_NAME['R']} {sel_scores['R']:.3f}.")
    elif len(have) == 0:
        selected_image_eye = None
        selection_mode = "no_eye_available"
        selection_reason = "No eye had any tracked frames. Clinical trace not produced."
    elif len(have) == 1:
        selected_image_eye = next(iter(have))
        selection_mode = "single_eye_only"
        other = "R" if selected_image_eye == "L" else "L"
        selection_reason = (
            f"Only {IMAGE_EYE_NAME[selected_image_eye]} "
            f"({IMAGE_TO_PATIENT_EYE[selected_image_eye]}) was approved/tracked; "
            f"{IMAGE_EYE_NAME[other]} ({IMAGE_TO_PATIENT_EYE[other]}) had no measurement frames.")
    else:
        # Both eyes have measurements → pick the higher quality score.
        if sel_scores["L"] >= sel_scores["R"]:
            selected_image_eye, other = "L", "R"
        else:
            selected_image_eye, other = "R", "L"
        selection_mode = "auto_best_eye"
        sL, sR = eye_stats["L"], eye_stats["R"]
        selection_reason = (
            f"Auto-selected {IMAGE_EYE_NAME[selected_image_eye]} "
            f"({IMAGE_TO_PATIENT_EYE[selected_image_eye]}) as the better-tracked eye "
            f"(score {sel_scores[selected_image_eye]:.3f} vs {sel_scores[other]:.3f}). "
            f"{IMAGE_EYE_NAME['L']} ({IMAGE_TO_PATIENT_EYE['L']}): "
            f"iris_valid {sL['pct_valid']}%, ref_valid {sL['pct_reference_valid']}%, "
            f"rel_valid {sL['pct_clinical_relative_valid']}%, mean_fc {sL['mean_frame_confidence']}, "
            f"drift {sL['drift_frames']}. "
            f"{IMAGE_EYE_NAME['R']} ({IMAGE_TO_PATIENT_EYE['R']}): "
            f"iris_valid {sR['pct_valid']}%, ref_valid {sR['pct_reference_valid']}%, "
            f"rel_valid {sR['pct_clinical_relative_valid']}%, mean_fc {sR['mean_frame_confidence']}, "
            f"drift {sR['drift_frames']}.")

    selected_patient_eye = image_to_patient(selected_image_eye)
    # `selected_eye` retains the internal shorthand used by the trace/plot code below.
    selected_eye = selected_image_eye

    print(f"Selected eye: {patient_eye_label(selected_image_eye)}  [{selection_mode}]. "
          f"Scores: Right {sel_scores['L']:.3f}, Left {sel_scores['R']:.3f}.")

    # ---- V1 NYSTAGMUS DETECTION + ARROW OVERLAY (EYEVNG_TRACKING_SPECIFICATION.md §K) ----
    # Run the detector on the SELECTED anatomical eye's contour-relative iris-centre series.
    # The detector is a pure consumer of FrameMeasurement values; it never modifies tracking
    # output. Outputs: nystagmus_events.json (per-clip report), nystagmus_events.csv (per-window
    # audit trail). The overlay video is then RE-RENDERED in a second pass to add the arrow band
    # during detected nystagmus windows (spec §G clinical layer).
    nystagmus_report = None
    nystagmus_events = []
    if selected_image_eye in ("L", "R") and fm_ser.get(selected_image_eye):
        h_eye = [m.eye_local[0] if m is not None else None for m in fm_ser[selected_image_eye]]
        v_eye = [m.eye_local[1] if m is not None else None for m in fm_ser[selected_image_eye]]
        fc_eye = [m.frame_confidence if m is not None else 0.0
                  for m in fm_ser[selected_image_eye]]
        valid_eye = [bool(m and m.clinical_relative_valid) for m in fm_ser[selected_image_eye]]
        # series index 0 corresponds to fr=1 (the V1 tracker writes one entry per frame_no >=1)
        det = NystagmusDetector(selected_image_eye, fps)
        det_windows, det_events, nystagmus_report = det.analyse(
            h_eye, v_eye, fc_eye, valid_eye, start_frame_offset=1)
        nystagmus_events = det_events
        print(f"Nystagmus (clinical): detected={nystagmus_report.clinical_nystagmus_detected} "
              f"silent={nystagmus_report.clinical_overlay_silent} "
              f"direction={nystagmus_report.dominant_beating_direction} "
              f"({nystagmus_report.dominant_label})  "
              f"clinical_events={nystagmus_report.clinical_events_count}  "
              f"candidate_events={nystagmus_report.candidate_events_count}  "
              f"torsion={nystagmus_report.torsional_status}")
        _write_nystagmus_outputs(out_dir, nystagmus_report, det_windows, det_events,
                                  selected_patient_eye)
        # Second pass: overlay the arrow on top of the pass-1 overlay. Clinical mode renders the
        # minimal V1 status band on the forehead/lower face; debug mode keeps the verbose band.
        if overlay_mode == "debug":
            _annotate_overlay_with_arrows(out_dir / "tracking_overlay.mp4", det_events,
                                           analysed_eye=patient_eye_label(selected_image_eye))
        else:
            _annotate_clinical_overlay_with_arrows(out_dir / "tracking_overlay.mp4",
                                                    det_events,
                                                    iris_xy_per_frame_overlay)
    elif selected_image_eye == "both":
        # Both-eye mode (INO / skew / dysconjugate / research) — V1 detector is single-eye by
        # spec §K and is not run here. The overlay still carries the anatomy tracking layer.
        print("Nystagmus: both-eye mode requested; single-eye detector not run (spec §K).")
    else:
        print("Nystagmus: no eye selected; detector not run.")

    # blink/occlusion shading per eye (light bands behind the trace; valid_* is None there so the
    # line already breaks — the band makes the excluded period explicit).
    blink_L = [a in ("blink", "occlusion") for a in art_L]
    blink_R = [a in ("blink", "occlusion") for a in art_R]
    blink_any = [a or b for a, b in zip(blink_L, blink_R)]

    # ---- RAW IMAGE-SPACE traces — DEBUG ONLY (include head movement; NOT the clinical trace) ----
    GRAY = (170, 170, 170)
    h_series = {"L filt": (flx, GREEN), "R filt": (frx, BLUE)}
    v_series = {"L filt": (fly, GREEN), "R filt": (fry, BLUE)}
    if show_raw:
        h_series = {"L raw": (rlx, GRAY), "R raw": (rrx, GRAY), **h_series}
        v_series = {"L raw": (rly, GRAY), "R raw": (rry, GRAY), **v_series}
    plot_trace(times, h_series, "Raw image-space iris X (de-meaned) - NOT CLINICAL (includes head movement)",
               out_dir / "trace_image_x.png", zero_each=True, shade=blink_any)
    plot_trace(times, v_series, "Raw image-space iris Y (de-meaned) - NOT CLINICAL (includes head movement)",
               out_dir / "trace_image_y.png", zero_each=True, shade=blink_any)

    # ---- EYE-LOCAL traces — THE CLINICAL VNG TRACE: iris position within the eye aperture ----
    # single eye by default; both only for --eye both. Blink frames are GAPS + shaded; red ticks =
    # degenerate aperture OR a drift-guard hit (drift-suspected frames are already dropped to gaps).
    drift_bool_L = [bool(x) for x in iris_flag_L]
    drift_bool_R = [bool(x) for x in iris_flag_R]
    if selected_eye == "both":
        eh = {"L": (cor_lh, GREEN), "R": (cor_rh, BLUE)}
        ev = {"L": (cor_lv, GREEN), "R": (cor_rv, BLUE)}
        shade_eye, vel_src, vcol = blink_any, cor_lh, GREEN
        drift_disp = [a or b for a, b in zip(drift_bool_L, drift_bool_R)]
    else:
        # selected_eye drives the CLINICAL trace; if no eye was selected (no measurements),
        # fall back to whichever side has any data so the plotting code stays safe.
        e = selected_eye or ("L" if cor_lh else "R")
        nm, vcol = ("L", GREEN) if e == "L" else ("R", BLUE)
        eh = {nm: (cor_lh if e == "L" else cor_rh, vcol)}
        ev = {nm: (cor_lv if e == "L" else cor_rv, vcol)}
        shade_eye, vel_src = (blink_L if e == "L" else blink_R), (cor_lh if e == "L" else cor_rh)
        drift_disp = drift_bool_L if e == "L" else drift_bool_R
    eye_flags = [p or d for p, d in zip(poor_flags, drift_disp)]
    # Plot titles: clinical anatomical label ('Left' / 'Right' / 'Both'). The technical
    # image↔patient mapping lives in metadata for anyone who needs to verify the convention.
    trace_eye_label = patient_eye_label(selected_eye if selected_eye else e)
    plot_trace(times, eh,
               "Contour-local HORIZONTAL  (iris-circle centre / eye-opening contour)"
               f"   Selected eye: {trace_eye_label}",
               out_dir / "trace_eyelocal_h.png", flags=eye_flags, shade=shade_eye, scale_exclude=eye_flags)
    plot_trace(times, ev,
               "Contour-local VERTICAL  (iris-circle centre / eye-opening contour)"
               f"   Selected eye: {trace_eye_label}",
               out_dir / "trace_eyelocal_v.png", flags=eye_flags, shade=shade_eye, scale_exclude=eye_flags)

    # ---- EYE-LOCAL horizontal VELOCITY — reveals slow drift vs fast reset (jerk-nystagmus sawtooth) ----
    vel, fast = eye_velocity(times, vel_src)
    plot_trace(times[1:], {"d(eye-local x)/dt per s": (vel, vcol)},
               "Eye-local horizontal VELOCITY (units/s)  -  red ticks = fast-phase candidates"
               f"   Selected eye: {trace_eye_label}",
               out_dir / "trace_eyelocal_velocity_h.png", flags=fast, shade=shade_eye[1:])

    def blink_stats(arts, statuses):
        """Blink/occlusion summary for one eye: invalid-frame count & %, number of blink EVENTS
        (contiguous runs) with durations in frames and seconds, and reacquisition events."""
        invalid = [a in ("blink", "occlusion") for a in arts]
        runs, c = [], 0
        for v in invalid:
            if v:
                c += 1
            elif c:
                runs.append(c); c = 0
        if c:
            runs.append(c)
        n = max(1, len(arts))
        return {
            "invalid_frames": sum(invalid),
            "pct_invalid": round(100.0 * sum(invalid) / n, 1),
            "blink_events": len(runs),
            "mean_blink_frames": round(st.mean(runs), 1) if runs else 0,
            "mean_blink_ms": round(1000.0 * st.mean(runs) / fps, 0) if runs else 0,
            "longest_blink_frames": max(runs) if runs else 0,
            "tracking_jump_frames": sum(a == "tracking_jump" for a in arts),
            "reacquisition_events": sum(s == "reacquired" for s in statuses),
        }

    def jitter(xs, ys):
        pts = [(xs[i], ys[i]) if xs[i] is not None else None for i in range(len(xs))]
        d = [np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
             for i in range(1, len(pts)) if pts[i] and pts[i - 1]]
        return round(st.median(d), 2) if d else None

    metadata = {
        "video": str(video_path), "resolution": f"{width}x{height}", "fps": round(fps, 2),
        "frames": fr, "initial_frame_number": init_fr,
        "approved_landmarks": str(out_dir / "approved_landmarks.json"),
        "confirmed_left": L, "confirmed_right": R,
        "interocular_px": round(interocular, 1),
        "engine": ({"v1": "v1_limbus_contour", "composite": "composite_feature_tracker",
                    "template": "template_legacy"}.get(engine, engine)),
        "cft_helper_enabled": bool(cft_helper) if engine == "v1" else False,
        "mediapipe_enabled": not no_mediapipe,
        "overlay_mode": overlay_mode,
        # ---- V1 STAGE 0 VALIDITY (runtime fields, added 2026-06-27) ------------
        # These fields make explicit that the V1 tracker started from a valid V1 Stage 0
        # approval: full iris/limbus circle as the tracking seed, eye-opening contour as the
        # reference seed, and NO legacy pupil / four-landmark fallback in use. If Stage 0 had
        # been invalid, run() would have aborted before reaching this metadata write and the
        # tombstone payload from _write_invalid_stage0_outputs would be present instead.
        "stage0_valid_for_v1": True,
        "tracking_seed_source": "approved_iris_limbus",
        "tracking_seed_target": "iris_limbus",
        "reference_seed_source": "approved_eye_opening_contour",
        "reference_seed_target": "eye_opening_contour",
        "legacy_pupil_fallback_used": bool(approved.get("legacy_pupil_fallback_used")),
        "legacy_four_point_contour_fallback_used":
            bool(approved.get("legacy_four_point_contour_fallback_used")),
        "approval_schema_version": approved.get("approval_schema_version"),
        "analysis_valid": True,
        "manual_corrections": (tracker.corrections if tracker else []),
        "engine_stats_left": eye_stats["L"],
        "engine_stats_right": eye_stats["R"],
        "eye_label_convention": EYE_LABEL_CONVENTION,
        "image_to_patient_eye_mapping": IMAGE_TO_PATIENT_EYE,
        "eye_label_note": EYE_LABEL_NOTE,
        "eye_selection": {
            # Anatomical reporting (preferred). image_*_eye = side of the image frame;
            # patient_*_eye = anatomical patient side after applying the mapping above.
            "selected_image_eye": (None if selected_image_eye is None else
                                   ("both" if selected_image_eye == "both"
                                    else IMAGE_EYE_NAME[selected_image_eye])),
            "selected_patient_eye": selected_patient_eye,
            "selection_mode": selection_mode,                # auto_best_eye / user_override /
                                                              # user_override_both / single_eye_only /
                                                              # no_eye_available
            "selection_reason": selection_reason,
            "both_eye_mode_requested": (eye == "both"),
            "user_eye_request": eye,                         # what --eye was set to
            "eye_label_convention": EYE_LABEL_CONVENTION,
            "image_to_patient_eye_mapping": IMAGE_TO_PATIENT_EYE,
            "note": EYE_LABEL_NOTE,
            # Per-side metrics — keyed by anatomical name so the reader can never confuse sides.
            "quality_score": {
                "image_left_eye":  round(sel_scores["L"], 4),   # = patient_right_eye
                "image_right_eye": round(sel_scores["R"], 4),   # = patient_left_eye
                "patient_right_eye": round(sel_scores["L"], 4),
                "patient_left_eye":  round(sel_scores["R"], 4),
            },
            "metrics": {
                "image_left_eye":  eye_stats["L"],              # = patient_right_eye
                "image_right_eye": eye_stats["R"],              # = patient_left_eye
                "patient_right_eye": eye_stats["L"],
                "patient_left_eye":  eye_stats["R"],
            },
            # Internal shorthand — retained for backward compatibility ONLY. Do NOT use
            # bare L/R in clinical narrative; always use the anatomical labels above.
            "internal_image_side_shorthand": selected_image_eye,
        },
        "status_counts": counts,
        "blink_left": blink_stats(art_L, stat_L),
        "blink_right": blink_stats(art_R, stat_R),
        "filter": filter_mode,
        "display_eye": (eye if (show_both or forced) else (selected_eye or "auto")),
        "jitter_left_raw_px": jitter(rlx, rly),
        "jitter_left_filtered_px": jitter(flx, fly),
        "aperture_quality_counts": q_counts,
        "iris_drift_guards": {
            "thresholds": {"aperture_margin": DRIFT_APERTURE_MARGIN, "anchor_frac": DRIFT_ANCHOR_FRAC,
                           "excursion": DRIFT_EXCURSION},
            "flag_counts": flag_counts,
            "invalidated_left": sum(f == "off_anchor" for f in iris_flag_L),
            "invalidated_right": sum(f == "off_anchor" for f in iris_flag_R),
            "outside_aperture_warning_left": sum(f == "outside_aperture" for f in iris_flag_L),
            "outside_aperture_warning_right": sum(f == "outside_aperture" for f in iris_flag_R),
            "review_left": sum(f == "review_excursion" for f in iris_flag_L),
            "review_right": sum(f == "review_excursion" for f in iris_flag_R),
        },
        "eye_opening_contour_reference": {
            "left_initialized": "L" in contour_trackers,
            "right_initialized": "R" in contour_trackers,
            "uncertain_counts": reference_uncertain_counts,
        },
        "nystagmus": ({
            "analysed_eye": nystagmus_report.analysed_eye,
            "clinical_nystagmus_detected": nystagmus_report.clinical_nystagmus_detected,
            "clinical_overlay_silent": nystagmus_report.clinical_overlay_silent,
            "dominant_beating_direction": nystagmus_report.dominant_beating_direction,
            "dominant_label": nystagmus_report.dominant_label,
            "torsional_status": nystagmus_report.torsional_status,
            "candidate_events_count": nystagmus_report.candidate_events_count,
            "clinical_events_count": nystagmus_report.clinical_events_count,
            "n_windows": nystagmus_report.n_windows,
            "n_candidate_windows": nystagmus_report.n_candidate_windows,
        } if nystagmus_report is not None else None),
        "outputs": {# anatomy + tracking layer only (verification artefact)
                    "overlay": str(out_dir / "tracking_overlay.mp4"),
                    # V1 PRIMARY clinical artefact: anatomy + tracking + nystagmus arrow/label
                    # (only written when the detector ran; spec §G clinical layer)
                    "overlay_with_arrow": (str(out_dir / "tracking_overlay_with_arrow.mp4")
                                           if nystagmus_report is not None else None),
                    "csv": str(out_dir / "tracking.csv"),
                    "aperture_landmarks_csv": (str(out_dir / "face_landmarks.csv") if face_tracker else None),
                    # V1 PRIMARY clinical report (spec §K.6):
                    "nystagmus_events_json": (str(out_dir / "nystagmus_events.json")
                                              if nystagmus_report is not None else None),
                    "nystagmus_events_csv":  (str(out_dir / "nystagmus_events.csv")
                                              if nystagmus_report is not None else None),
                    # debug-only contour-relative position/velocity plots (audit only):
                    "trace_eyelocal_h_debug": str(out_dir / "trace_eyelocal_h.png"),
                    "trace_eyelocal_v_debug": str(out_dir / "trace_eyelocal_v.png"),
                    "trace_eyelocal_velocity_h_debug": str(out_dir / "trace_eyelocal_velocity_h.png"),
                    # debug only (image-space, includes head movement):
                    "trace_image_x_debug": str(out_dir / "trace_image_x.png"),
                    "trace_image_y_debug": str(out_dir / "trace_image_y.png")},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


def _r(v):
    return round(v, 2) if v is not None else ""


def _fm_cols(fm):
    """The composite §G columns for one eye, in header order (blank if the eye is disabled/pre-init)."""
    if fm is None:
        return ["", "", "", "", "", "", "", "", "", ""]
    return [fm.state, fm.validity, int(fm.drift_flag), fm.drift_reason,
            _r(fm.rotation), _r(fm.feature_confidence_mean), _r(fm.composite_confidence),
            _r(fm.frame_confidence), fm.n_active, fm.n_trusted]


# ----------------------------------------------------------------------------- §K outputs
def _write_invalid_stage0_outputs(out_dir, approved, reasons=None):
    """V1 detector protection — when Stage 0 is invalid, write a TOMBSTONE
    `nystagmus_events.json` AND a `metadata.json` recording the failure, so any downstream
    reader sees an explicit failure rather than the absence of a report or a misleading
    'no nystagmus detected' from invalid tracking. No tracker is constructed. No detector
    runs. No overlay video is produced. No arrow can possibly be displayed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    approved = approved or {}
    rejection_details = list(reasons or [])
    base_payload = {
        "stage0_valid_for_v1": False,
        "analysis_valid": False,
        "rejection_reason": "invalid_stage0_approval",
        "clinical_nystagmus_detected": False,
        "clinical_overlay_silent": True,
        "approval_schema_version": approved.get("approval_schema_version"),
        "tracking_target": approved.get("tracking_target"),
        "reference_target": approved.get("reference_target"),
        "iris_limbus_approved": approved.get("iris_limbus_approved"),
        "eye_opening_contour_approved": approved.get("eye_opening_contour_approved"),
        "legacy_pupil_fallback_used": approved.get("legacy_pupil_fallback_used"),
        "legacy_four_point_contour_fallback_used":
            approved.get("legacy_four_point_contour_fallback_used"),
        "rejection_details": rejection_details,
        "note": ("eye_vng V1 requires both full iris/limbus approval and "
                 "eye-opening/orbital margin contour approval. Run --approve again and mark "
                 "both structures. Invalid Stage 0 is NOT a negative clinical result; it is a "
                 "failed analysis."),
    }
    try:
        (out_dir / "nystagmus_events.json").write_text(json.dumps(base_payload, indent=2),
                                                       encoding="utf-8")
        # Also drop a minimal metadata.json so any consumer querying metadata sees the failure
        # rather than reading a stale metadata.json from a previous (successful) run.
        (out_dir / "metadata.json").write_text(json.dumps({
            **base_payload,
            "video_stem": out_dir.name,
        }, indent=2), encoding="utf-8")
        # Delete stale clinical artefacts from a previous (valid) run so the directory cannot
        # contain a misleading overlay video or per-window CSV alongside the invalid-analysis
        # tombstone. Anything else in the directory (debug frames, etc.) is left alone.
        for stale in ("tracking_overlay.mp4", "tracking_overlay_with_arrow.mp4",
                       "nystagmus_events.csv", "tracking.csv"):
            p = out_dir / stale
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _write_nystagmus_outputs(out_dir, report, windows, events, analysed_eye):
    """Persist the V1 nystagmus detection results (spec §K.6).
      * nystagmus_events.json — per-clip report + merged events (primary clinical report)
      * nystagmus_events.csv  — per-window audit trail (one row per sliding window)
    The on-overlay arrow is drawn in a separate second-pass step."""
    payload = {
        "analysed_eye": report.analysed_eye,
        "image_side_internal": report.image_side_internal,
        # CLINICAL tier — the only fields that drive the overlay
        "clinical_nystagmus_detected": report.clinical_nystagmus_detected,
        "clinical_overlay_silent": report.clinical_overlay_silent,
        "dominant_beating_direction": report.dominant_beating_direction,
        "dominant_label": report.dominant_label,
        "dominant_arrow": report.dominant_arrow,
        "clinical_events_count": report.clinical_events_count,
        # CANDIDATE tier — debug only; never displayed
        "candidate_events_count": report.candidate_events_count,
        "candidate_events": [
            {
                "candidate_event": e.candidate_event,
                "clinically_confirmed": e.clinically_confirmed,
                "displayed_on_overlay": e.displayed_on_overlay,
                "rejection_reason": e.rejection_reason,
                "not_clinically_confirmed": (not e.clinically_confirmed),
                "time_window": {
                    "start_frame": e.start_frame, "end_frame": e.end_frame,
                    "start_sec": e.start_sec, "end_sec": e.end_sec,
                    "duration_sec": e.duration_sec,
                },
                "direction_estimate": e.beating_direction,
                "label_if_confirmed": e.label,
                "arrow_if_confirmed": e.arrow,
                "beat_count": e.n_beats_total,
                "n_windows": e.n_windows,
                "mean_beat_rate_hz": e.mean_beat_rate_hz,
                "mean_direction_consistency": e.mean_direction_consistency,
                "mean_rhythmicity": e.mean_rhythmicity,
                "confidence": e.mean_confidence,
            } for e in events],
        # shared
        "torsional_status": report.torsional_status,
        "n_windows": report.n_windows,
        "n_candidate_windows": report.n_candidate_windows,
        "parameters": report.parameters,
        "torsion_note": ("Torsion cannot be assessed from iris-circle centre motion alone. "
                         "It requires iris-texture rotation tracking inside the iris circle, "
                         "which is a future module. Reported as not_assessed."),
        "calibration_note": ("samples/2_short15.mp4 is the current V1 negative-control / "
                              "no-nystagmus calibration clip. If this clip produces a DISPLAYED "
                              "clinical nystagmus label, the detector has failed calibration. "
                              "Filename is not hardcoded into runtime logic; the suppression "
                              "comes from the validity gate, amplitude floor, and clinical "
                              "confirmation gates."),
    }
    (out_dir / "nystagmus_events.json").write_text(json.dumps(payload, indent=2),
                                                   encoding="utf-8")
    with (out_dir / "nystagmus_events.csv").open("w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["window_start_frame", "window_end_frame", "window_start_sec",
                     "window_end_sec", "nystagmus_present", "beating_direction", "label",
                     "n_beats", "mean_beat_rate_hz", "direction_consistency", "rhythmicity",
                     "confidence", "rejection_reason"])
        for w in windows:
            cw.writerow([w.window_start_frame, w.window_end_frame,
                         w.window_start_sec, w.window_end_sec,
                         int(w.nystagmus_present), w.beating_direction, w.label,
                         w.n_beats, w.mean_beat_rate_hz, w.direction_consistency,
                         w.rhythmicity, w.confidence, w.rejection_reason])


def _annotate_clinical_overlay_with_arrows(overlay_path, events, iris_xy_per_frame):
    """V1 CLINICAL pass-2 annotator. Reads the anatomy-only clinical overlay produced in pass 1
    and writes a second file alongside it with the V1 clinical status band added every frame:
        * "Nystagmus detected: <Direction>" + arrow during a detected event
        * "No nystagmus detected" otherwise
    Placement chosen per-frame from the tracked iris position so the band stays on the forehead
    or the lower face — never on the eyes.

    `iris_xy_per_frame` is a list (length = total frames written by pass 1) of [(x,y), ...]
    points in the SAME pixel space as the overlay frames; used to decide top vs bottom band.

    Returns the path to the annotated video, or None if the source overlay is missing.
    """
    overlay_path = Path(overlay_path)
    if not overlay_path.exists():
        return None
    cap = cv2.VideoCapture(str(overlay_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = overlay_path.with_name(overlay_path.stem + "_with_arrow.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr += 1
        # NEGATIVE-CONTROL RULE (added 2026-06-27): the clinical overlay is SILENT unless a
        # clinically-confirmed event is active. event_for_frame() already filters out unconfirmed
        # candidates, so when no event is active here we draw NOTHING — no arrow, no "No
        # nystagmus detected" caption, no doubtful-nystagmus text.
        ev = event_for_frame(events, fr)
        if ev is not None:
            status = _short_direction_label(ev.beating_direction)
            glyph = ev.arrow
            iris_pts = iris_xy_per_frame[fr - 1] if 0 <= fr - 1 < len(iris_xy_per_frame) else []
            draw_clinical_status_band(frame, status, glyph, iris_pts=iris_pts)
        writer.write(frame)
    cap.release()
    writer.release()
    return out_path


def _annotate_overlay_with_arrows(overlay_path, events, analysed_eye=""):
    """Render a SECOND overlay video — `tracking_overlay_with_arrow.mp4` — that adds the V1
    nystagmus arrow + label on top of the anatomy-tracking overlay produced in the main pass.

    Two artefacts are kept by design:
      * `tracking_overlay.mp4`             — anatomy + tracking layer only (verification)
      * `tracking_overlay_with_arrow.mp4`  — anatomy + tracking + nystagmus arrow/label
                                              (V1 PRIMARY clinical artefact; spec §G)

    The arrow band is drawn ONLY during detected nystagmus events; outside those frames a
    small "No nystagmus detected" caption is drawn instead. Both labels live in the top-left
    corner — chosen because the V1 reference clips zoom over the face and the top-left is
    consistently free of facial anatomy.

    Returns the path to the new annotated video, or None if the source overlay was missing.
    """
    overlay_path = Path(overlay_path)
    if not overlay_path.exists():
        return None
    cap = cv2.VideoCapture(str(overlay_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = overlay_path.with_name(overlay_path.stem + "_with_arrow.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr += 1
        ev = event_for_frame(events, fr)
        if ev is not None:
            draw_nystagmus_arrow(frame, ev.label, ev.arrow,
                                 n_beats=ev.n_beats_total,
                                 rate_hz=ev.mean_beat_rate_hz,
                                 confidence=ev.mean_confidence,
                                 analysed_eye=analysed_eye)
        else:
            draw_no_nystagmus_caption(frame, analysed_eye=analysed_eye)
        writer.write(frame)
    cap.release()
    writer.release()
    return out_path


# ----------------------------------------------------------------------------- review mode
def review(video_path, output_dir="outputs"):
    out_dir = Path(output_dir) / (Path(video_path).stem + "_tracked")
    rows = list(csv.DictReader((out_dir / "tracking.csv").open()))
    bad = [r for r in rows if "uncertain" in (r["tracking_status_left"], r["tracking_status_right"])
           or "lost" in (r["tracking_status_left"], r["tracking_status_right"])
           or "blink_or_occluded" in (r["tracking_status_left"], r["tracking_status_right"])]
    if not bad:
        print("No low-confidence frames to review."); return
    cap = cv2.VideoCapture(str(video_path))
    i = 0
    print(f"{len(bad)} low-confidence frames. n=next, p=prev, q=quit.")
    while True:
        r = bad[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_number"]) - 1)
        ok, frame = cap.read()
        if ok:
            s = min(1.0, 960 / frame.shape[1])
            disp = cv2.resize(frame, None, fx=s, fy=s)
            cv2.putText(disp, f"frame {r['frame_number']} L:{r['tracking_status_left']} "
                              f"R:{r['tracking_status_right']} ({i+1}/{len(bad)})",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            cv2.imshow("review (n/p/q)", disp)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        i = (i + 1) % len(bad) if k == ord("n") else (i - 1) % len(bad) if k == ord("p") else i
    cap.release(); cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Stable iris tracking")
    ap.add_argument("video")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--propose", action="store_true",
                    help="Stage 0 preview: mark proposed iris + face landmarks on the init frame")
    ap.add_argument("--approve", action="store_true",
                    help="Stage 0: review/correct then approve landmarks (saves approved_landmarks.json)")
    ap.add_argument("--auto-approve", action="store_true",
                    help="Stage 0 headless: approve the proposal without a GUI")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--filter", choices=["none", "light", "adaptive"], default="adaptive",
                    help="iris jitter filter (default adaptive 1€)")
    ap.add_argument("--show-raw-trace", action="store_true", help="also draw the raw signal faint")
    ap.add_argument("--engine", choices=["v1", "composite", "template"], default="v1",
                    help="tracking engine: v1 (limbus+contour orchestrator, default), "
                         "composite (CFT only), template (legacy)")
    ap.add_argument("--no-mediapipe", action="store_true",
                    help="run from approved landmarks without MediaPipe proposal/recovery")
    ap.add_argument("--cft-helper", action="store_true",
                    help="(v1 engine) enable the CFT as a motion-prediction helper "
                         "(does not change the clinical centre or validity)")
    ap.add_argument("--overlay-mode", choices=["clinical", "debug"], default="clinical",
                    help="overlay video presentation: clinical (default, minimal) or debug")
    ap.add_argument("--disable-rit-validator", action="store_true",
                    help="(v1 engine) skip the RIT iris-candidate validator — produces a "
                         "no-guardrails baseline CSV/overlay for comparison. Do NOT use clinically.")
    ap.add_argument("--output-suffix", default="",
                    help="appended to the _tracked folder name (e.g. '_no_guardrails') so a "
                         "comparison run does not overwrite the guarded output")
    args = ap.parse_args()
    if args.propose:
        propose_init(args.video, args.output_dir)
    elif args.approve or args.auto_approve:
        approve(args.video, args.output_dir, auto=args.auto_approve)
    elif args.review:
        review(args.video, args.output_dir)
    else:
        run(args.video, args.output_dir, filter_mode=args.filter, show_raw=args.show_raw_trace,
            engine=args.engine, no_mediapipe=args.no_mediapipe, cft_helper=args.cft_helper,
            overlay_mode=args.overlay_mode, disable_rit_validator=args.disable_rit_validator,
            output_suffix=args.output_suffix)


if __name__ == "__main__":
    main()
