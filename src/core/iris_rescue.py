"""V1 Clinician Teaching Mode — Iris + Contour Rescue Scrubber (EYE_VNG §18).

Workflow:
  1. The clinician runs V1 tracking on a clip (`python app.py --video ... --no-mediapipe`).
  2. The clinician runs this rescue tool on the same clip:
         `python app.py --video ... --rescue`
  3. A scrubber window opens on `tracking_overlay.mp4` (the V1 clinical overlay produced by
     the tracker). The clinician scrubs through every frame.
  4. On any frame where the tracker has lost or mis-placed the iris circle OR the
     eye-opening contour is wrong, the clinician switches mode and drags the marker(s) back
     to where they really should be.
       'i' — IRIS mode: drag iris centre, mouse wheel / +/- to resize.
       'c' — CONTOUR mode: drag the centre or any of the four cardinal handles of the
             eye-opening oval (same Stage 0 UX). +/- resize the active axis; r/R rotate.
       'e' — switch between Left and Right anatomical eyes.
       's' — save anchor for this frame (iris + contour).
       'n' / 'p' — next / previous tracker-lost frame.
       'q' / Esc — quit.
  5. Each saved anchor records the corrected iris circle, the corrected oval (sampled to a
     24-vertex polygon for compatibility with the V1 schema), the raw frame as a PNG, and
     metadata (eye, time, reason). All saved to
     `outputs/<clip>_tracked/teaching_examples/`.

IMPORTANT SCOPE LIMITS:
  - This tool does NOT retrain a model; it captures expert corrections for later reuse.
  - It does NOT modify tracker code; it only writes JSON + PNG files.
  - It does NOT replace Stage 0; it complements Stage 0 by handling failures of the
    per-frame tracker.

The iris object marked here is the FULL iris/limbus circle, NOT the pupil. The contour is
the visible eye-opening / orbital margin. See §18 of `EYE_VNG_DEVELOPMENT_HISTORY.md`.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- helpers
def _load_tracking_csv(csv_path: Path):
    """Read iris centre + radius per frame from V1's tracking.csv. Returns a list of
    {frame_number, time_sec, L:(x,y,r)|None, R:(x,y,r)|None}. Used to initialise the iris
    seed on each frame so the clinician usually drags from a sensible starting circle."""
    if not csv_path.exists():
        return []
    out = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                fr = int(row["frame_number"])
            except Exception:
                continue

            def f3(x, y, r):
                vx = row.get(x, "").strip()
                vy = row.get(y, "").strip()
                vr = row.get(r, "").strip()
                if vx and vy and vr:
                    try:
                        return float(vx), float(vy), float(vr)
                    except ValueError:
                        return None
                return None

            L = f3("raw_left_iris_center_x", "raw_left_iris_center_y", "left_iris_radius")
            R = f3("raw_right_iris_center_x", "raw_right_iris_center_y", "right_iris_radius")
            tsec = float(row.get("time_sec") or 0.0)
            out.append({"frame_number": fr, "time_sec": tsec, "L": L, "R": R})
    return out


def _seed_iris_from_tracking(tracking_rows, frame_no: int, eye_key: str,
                             frame_w: int, frame_h: int, fallback_r: float = 18.0,
                             approved_iris=None):
    """Look up the best available iris seed for (frame_no, eye_key). Priority:
       1. The tracker's iris circle for THIS frame from tracking.csv.
       2. The most recent earlier frame's iris circle.
       3. The nearest later frame's iris circle.
       4. The Stage-0 approved iris seed.
       5. **None** — caller MUST treat this as "hide the marker" rather than drop a default
          circle somewhere arbitrary on the face (the old behaviour returned the centre of
          the frame at 30 %/70 % which on a face crop landed on the nose).
    """
    # 1. exact frame
    idx_at = None
    for i, row in enumerate(tracking_rows):
        if row["frame_number"] == frame_no:
            idx_at = i
            seed = row.get(eye_key)
            if seed is not None:
                return seed
            break
    # 2a. most recent earlier frame with a valid iris for this eye
    if idx_at is not None:
        for i in range(idx_at - 1, -1, -1):
            seed = tracking_rows[i].get(eye_key)
            if seed is not None:
                return seed
    # 2b. nearest later frame with a valid iris for this eye
    if idx_at is not None:
        for i in range(idx_at + 1, len(tracking_rows)):
            seed = tracking_rows[i].get(eye_key)
            if seed is not None:
                return seed
    # 3. Stage-0 approved seed (always on the eye on the init frame)
    if approved_iris is not None:
        return approved_iris
    # 4. Nothing valid — return None so the caller hides the marker. NEVER place a default
    # circle on the face: it has caused the "marker on the nose" bug.
    return None


def _eye_label(eye_key: str) -> str:
    """Image-side shorthand → anatomical label for the user-facing UI.
    In a front-facing video: image L = patient Right; image R = patient Left."""
    return "Right" if eye_key == "L" else "Left"


def _ellipse_from_polygon(pts):
    """Best-fit ellipse params (cx, cy, a, b, theta) from a 2D polygon. Used to convert the
    Stage-0 contour polygon into an editable oval. Returns None if the fit can't be made."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 5:
        return None
    try:
        (cx, cy), (MA, ma), ang = cv2.fitEllipse(pts)
        return {"cx": float(cx), "cy": float(cy),
                "a": float(max(8.0, MA / 2.0)),
                "b": float(max(6.0, ma / 2.0)),
                "theta": float(np.deg2rad(ang))}
    except cv2.error:
        return None


def _ellipse_handle_positions(e):
    """Cardinal handles of an ellipse e in IMAGE coords (used to draw and pick handles)."""
    cx, cy, a, b, th = e["cx"], e["cy"], e["a"], e["b"], e["theta"]
    ca, sa = np.cos(th), np.sin(th)
    return {
        "medial":  (cx - a * ca, cy - a * sa),
        "lateral": (cx + a * ca, cy + a * sa),
        "upper":   (cx + b * sa, cy - b * ca),     # perpendicular axis (+90°)
        "lower":   (cx - b * sa, cy + b * ca),
    }


def _ellipse_to_polygon(e, n=24):
    """Sample an ellipse to n polygon points (the format the V1 schema expects)."""
    ts = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ca, sa = np.cos(e["theta"]), np.sin(e["theta"])
    pts = []
    for t in ts:
        x = e["a"] * np.cos(t)
        y = e["b"] * np.sin(t)
        pts.append((e["cx"] + ca * x - sa * y, e["cy"] + sa * x + ca * y))
    return np.asarray(pts, np.float32)


def _load_stage0_contour(approved_path: Path, eye_key: str):
    """Load the Stage-0 approved eye-opening contour for `eye_key` ('L' or 'R') and convert it
    into an editable ellipse. Returns None if no contour was approved for that eye."""
    if not approved_path.exists():
        return None
    try:
        approved = json.loads(approved_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    contours = approved.get("eye_opening_contours") or {}
    if not isinstance(contours, dict):
        return None
    pts = contours.get(eye_key)
    if not pts:
        return None
    pts_arr = np.asarray([[p["x"], p["y"]] if isinstance(p, dict) else [p[0], p[1]]
                          for p in pts], np.float32)
    return _ellipse_from_polygon(pts_arr)


# --------------------------------------------------------------------------- the scrubber
def run_rescue_scrubber(video_stem_dir: Path, original_video: Path,
                         eye_key: str = "L") -> Dict:
    # ---- TEACHING-MODE LOCK (2026-06-28) ---------------------------------------------
    # For this manual-testing pass the active teaching eye is FORCED to patient anatomical
    # Right (image-side L). The clinician requested no eye-switching during this test so
    # that all UI feedback unambiguously refers to a single eye. The `e` key is therefore
    # also disabled below. To re-enable both eyes, remove these two lines and restore the
    # `e` keymap.
    eye_key = "L"   # image side L == patient anatomical Right
    """Open the V1 clinical overlay video for `video_stem_dir` (folder under outputs/)
    in a scrubber. Let the clinician scrub, drag the iris circle, and save rescue anchors.

    Returns a summary dict of what was saved (so the caller can plug it into metadata.json).
    """
    overlay_path = video_stem_dir / "tracking_overlay.mp4"
    tracking_csv = video_stem_dir / "tracking.csv"
    if not overlay_path.exists():
        raise SystemExit(
            f"No clinical overlay found at {overlay_path}. "
            "Run V1 tracking first: python app.py --video \"...\" --no-mediapipe")

    # ---- output paths (LOCKED 2026-06-28 per teaching-mode spec) ------------------
    # `teaching_anchors.json` lives DIRECTLY under outputs/<clip>_tracked/ (NOT under
    # teaching_examples/); per-frame PNG assets live under teaching_examples/. The legacy
    # path `teaching_examples/rescue_anchors.json` is read-only for migration so any older
    # session's anchors are picked up.
    examples_dir = video_stem_dir / "teaching_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    anchors_index = video_stem_dir / "teaching_anchors.json"
    legacy_index = examples_dir / "rescue_anchors.json"
    existing_anchors: Dict[int, Dict] = {}
    for src in (anchors_index, legacy_index):
        if src.exists():
            try:
                for a in json.loads(src.read_text(encoding="utf-8")).get("anchors", []):
                    existing_anchors[int(a["frame_number"])] = a
            except Exception:
                pass

    cap = cv2.VideoCapture(str(overlay_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {overlay_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    OVERLAY_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    OVERLAY_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Also open the RAW video so we save the un-annotated frame as the teaching example image
    # (the overlay-annotated frame is good for visual review but a clean frame is the asset
    # a future training pass would want).
    raw_cap = cv2.VideoCapture(str(original_video))
    if raw_cap.isOpened():
        SOURCE_W = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        SOURCE_H = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        print(f"Note: cannot open raw video at {original_video}; will save the overlay frame instead.")
        raw_cap = None
        SOURCE_W, SOURCE_H = OVERLAY_W, OVERLAY_H

    # SCRUBBER COORDINATE SPACE
    # ------------------------------------------------------------------------------
    # All anchor records, tracking.csv values, approved_landmarks.json values, and the
    # `state["iris_*"]` / `state["oval"]` values live in **SOURCE-VIDEO coordinate
    # space (SOURCE_W × SOURCE_H)**. The overlay video may be at a lower resolution
    # (e.g. 1920x1080 source → 960x540 overlay), and the scrubber's display window
    # may scale further. We convert ONLY at the drawing/click boundary:
    #
    #   image_coord = display_coord * (SOURCE_W / overlay_W) / state["scale"]
    #   display_coord = image_coord * (overlay_w / SOURCE_W) * state["scale"]
    #
    # Without this you get the bug: a tracker.csv x=504 in 1920x1080 space gets drawn
    # at pixel 504 of a 960x540 overlay (which is the bridge of the nose), and a user
    # click at display y=977 on a 540-tall overlay gets saved as image y=977 (off the
    # bottom of a 1080-tall source frame).
    OVERLAY_TO_SOURCE = float(SOURCE_W) / float(OVERLAY_W)
    print(f"  source video: {SOURCE_W}x{SOURCE_H}  overlay: {OVERLAY_W}x{OVERLAY_H}  "
          f"overlay->source scale: {OVERLAY_TO_SOURCE:.3f}")
    # alias for code that pre-dates the split
    W, H = SOURCE_W, SOURCE_H

    tracking_rows = _load_tracking_csv(tracking_csv)
    # Stage-0 approved contour, as an EDITABLE oval per eye. Used as the seed in CONTOUR mode.
    approved_path = video_stem_dir / "approved_landmarks.json"
    stage0_oval = {
        "L": _load_stage0_contour(approved_path, "L"),
        "R": _load_stage0_contour(approved_path, "R"),
    }
    # Stage-0 approved iris (x, y, r) per eye. Used as the iris seed fallback when no tracker
    # iris exists for this frame and there's no prior frame to inherit from — keeps the marker
    # on the eye instead of falling back to "centre of frame" on the forehead.
    approved_iris = {"L": None, "R": None}
    try:
        if approved_path.exists():
            _appr = json.loads(approved_path.read_text(encoding="utf-8"))
            for k in ("L", "R"):
                v = (_appr.get("iris") or {}).get(k)
                if v:
                    approved_iris[k] = (float(v["x"]), float(v["y"]), float(v["radius"]))
    except Exception:
        pass
    print(f"Rescue scrubber: {total} frames @ {fps:.1f} fps, eye={_eye_label(eye_key)} "
          f"(image side {eye_key})")
    if stage0_oval.get(eye_key):
        print(f"  Loaded Stage-0 eye-opening contour for {_eye_label(eye_key)} eye.")
    if existing_anchors:
        print(f"Loaded {len(existing_anchors)} existing rescue anchors from {anchors_index}")

    win = "V1 Clinician Teaching Mode — Iris Rescue Anchors"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    MODE_IRIS, MODE_CONTOUR = "iris", "contour"
    state = {
        "mode": MODE_IRIS,
        "frame_no": 1, "play": False, "drag": False,
        "scale": 1.0,                        # display scale (frame is shown at native size if it fits)
        "iris_x": 0.0, "iris_y": 0.0, "iris_r": 20.0,   # current iris circle in IMAGE coords
        # marker_visible was previously used to hide the marker when no seed existed. As of
        # the "always visible + parked" rule (2026-06-28) the marker is forced True always.
        # See is_parked below for the "no real seed yet, please drag me" state.
        # turns True the first moment a valid seed is found (anchor/tracker/Stage-0) or
        # the moment the clinician clicks anywhere in iris mode. This replaces the old
        # "always draw a default circle somewhere" behaviour that produced the magenta
        # marker on the nose.
        "marker_visible": True,              # forced True under the "always visible" rule
        "iris_source": "parked",             # debug label: anchor/anchor_nearby/tracker/stage0/parked
        # is_parked == True when the iris circle is at a neutral "drag me to the iris"
        # position (no anchor / no tracker / no Stage-0 seed). The renderer shows an
        # "UNPLACED IRIS — DRAG TO IRIS" label next to it; the first click on the canvas
        # moves it to where the clinician clicked.
        "is_parked": True,
        # current eye-opening oval being edited (cx, cy, a, b, theta). Re-seeded each frame from
        # an existing anchor if there is one, else from the Stage-0 oval, else parked.
        "oval": None,
        "oval_is_parked": True,              # same idea for the contour: parked vs anchored
        "contour_handle": "centre",          # which handle was last clicked
        "iris_dirty": False,                 # iris circle has been edited but not yet saved
        "contour_dirty": False,              # contour oval has been edited but not yet saved
        "eye_key": eye_key,
        "current_frame_overlay": None,
        "current_frame_raw": None,
        # transient on-screen "ANCHOR SAVED" toast — frame_no when shown, used for 2-sec timeout.
        "toast_text": "", "toast_frame_no": -1,
        # last radius the clinician explicitly edited (via click-create OR drag-resize) —
        # priority #1 for new markers per spec'd default-radius rule (2026-06-28). Persists
        # only for this session; not saved to disk.
        "last_edited_radius": None,
        # The clinician's last MANUALLY placed (x, y, r) carries forward across frames as the
        # iris-init priority #4. Updated on every drag/resize/save in iris mode. Distinct
        # from `last_edited_radius` (which is just the radius) — this is the FULL marker
        # state so a clinician advancing one frame at a time keeps the marker on the eye.
        "last_manual_xyr": None,
        # PATCH 2 (2026-06-28): track which frame the last clinician edit was on, so the
        # seed-priority chain on subsequent frames can keep the clinician's correction
        # winning over a still-failing tracker for a short propagation window. Without
        # this the tracker output on frame N+1 immediately overwrites the clinician's
        # frame N correction, which is the "fight the clinician" failure.
        "last_manual_frame_no": -1,
    }
    # Frames the clinician overrides win on the next ~CLINICIAN_PROPAGATION_FRAMES frames
    # over fresh tracker output (which is presumed still-failing in that region). Reset
    # every time the clinician edits again. Anchor saves are independent (they win via
    # priority steps 1 + 2 already).
    CLINICIAN_PROPAGATION_FRAMES = 30

    # -------- DEFAULT-RADIUS RULE (spec'd 2026-06-28, REVISED 2026-06-28) ---------
    # Priority for the default radius of a NEWLY CREATED marker:
    #   1. last radius the clinician edited in this session (state["last_edited_radius"]),
    #   2. median radius of saved anchors for this clip and active eye,
    #   3. Stage-0 approved iris radius for this eye,
    #   4. median recent tracker iris radius for this eye,
    #   5. a conservative fallback (30 px — was 12 px which was much too small).
    # The result is CLAMPED to [DEFAULT_IRIS_MIN, DEFAULT_IRIS_MAX_CREATE] = [10, 45].
    # The clinician can still enlarge above 45 with + after creation; the clamp only
    # applies at marker-creation time.
    DEFAULT_IRIS_FALLBACK = 30.0
    DEFAULT_IRIS_MIN = 10.0
    DEFAULT_IRIS_MAX_CREATE = 45.0
    def _default_iris_radius():
        key = state["eye_key"]
        # 1. last clinician-edited radius in this session
        if state.get("last_edited_radius") is not None:
            return ("session_last", float(state["last_edited_radius"]))
        # 2. median of anchor radii for this eye
        rs = [a.get("corrected_iris_radius", a.get("iris_radius"))
              for a in existing_anchors.values()
              if a.get("image_side_eye", a.get("eye_image_side")) == key
                  and a.get("corrected_iris_radius", a.get("iris_radius"))]
        if rs:
            return ("anchor_median", float(np.median(rs)))
        # 3. Stage-0 approved iris radius
        appr = approved_iris.get(key)
        if appr is not None:
            return ("stage0", float(appr[2]))
        # 4. median tracker iris radius across the clip
        trs = [row[key][2] for row in tracking_rows
               if row.get(key) is not None]
        if trs:
            return ("tracker", float(np.median(trs)))
        # 5. conservative fallback
        return ("fallback", DEFAULT_IRIS_FALLBACK)

    def _default_iris_radius_clamped():
        src, r = _default_iris_radius()
        r = max(DEFAULT_IRIS_MIN, min(DEFAULT_IRIS_MAX_CREATE, r))
        return src, r

    # -------- PARKED-POSITION RULE (2026-06-28) ---------------------------------
    # When there is no valid seed (no anchor, no tracker, no Stage-0) the iris marker is
    # parked in a NEUTRAL VISIBLE location so the clinician can simply drag it onto the
    # iris. Priority for the parked position:
    #   1. centre of the active eye's eye-opening contour if one is available,
    #   2. centre-LEFT third of the image for patient Right (image-side L), centre-right
    #      third for patient Left (image-side R) — i.e. roughly where each eye sits,
    #   3. screen centre (last resort).
    # The marker is ALWAYS clamped to the inside of the frame so it can never park
    # off-screen.
    def _parked_iris_xy(key: str):
        # 1. centre of the eye-opening contour for this eye, if known.
        oval = stage0_oval.get(key)
        if oval is not None:
            return float(oval["cx"]), float(oval["cy"]), "contour_centre"
        # 2. left-third / right-third heuristic in source-video pixels.
        if key == "L":   # image side L == patient Right == LEFT side of the image
            return float(W) * 0.30, float(H) * 0.50, "image_left_third"
        else:
            return float(W) * 0.70, float(H) * 0.50, "image_right_third"
        # (we never reach a "screen centre" path because the heuristic always returns;
        # if W/H were both 0 the entire pipeline would already have failed elsewhere.)

    def _parked_oval(key: str, iris_x: float, iris_y: float, iris_r: float):
        """Default editable oval used when there is no Stage-0 contour and no anchor.
        Sized as a wide ellipse centred on the parked iris position, so a clinician can
        drag/resize it to the visible eye opening."""
        return {
            "cx": float(iris_x), "cy": float(iris_y),
            "a": float(max(20.0, 2.6 * iris_r)),
            "b": float(max(12.0, 1.4 * iris_r)),
            "theta": 0.0,
        }

    def _read_frame(fr: int):
        """Seek to frame fr (1-indexed) in both captures and load the iris seed."""
        fr = max(1, min(total, fr))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr - 1)
        ok, overlay = cap.read()
        if not ok:
            return False
        state["current_frame_overlay"] = overlay
        if raw_cap is not None:
            raw_cap.set(cv2.CAP_PROP_POS_FRAMES, fr - 1)
            ok2, raw = raw_cap.read()
            state["current_frame_raw"] = raw if ok2 else overlay
        else:
            state["current_frame_raw"] = overlay
        state["frame_no"] = fr
        # ---- INIT THE ACTIVE-EYE IRIS MARKER PER SPEC'D RULE ----------------------
        # Priority (LOCKED 2026-06-28):
        #   1. saved anchor for this frame and eye          -> show
        #   2. nearby (±5 frame) saved anchor for this eye  -> show
        #   3. tracker iris for this frame                  -> show
        #   4. nearest tracker iris for this eye            -> show
        #   5. Stage-0 approved iris seed                   -> show
        #   6. NONE                                          -> HIDE until the clinician clicks.
        # The marker is NEVER placed at the centre of the face. The pink "marker on the
        # nose" bug was caused by a 30/70 + vertical-centre fallback in the old code; that
        # fallback has been deleted.
        anchor = existing_anchors.get(fr)
        seed_xyr = None
        seed_source = "none"
        # 1. exact-frame anchor for this eye
        if anchor and anchor.get("image_side_eye", anchor.get("eye_image_side")) == state["eye_key"]:
            cx = float(anchor.get("corrected_iris_centre", [0, 0])[0] if isinstance(
                anchor.get("corrected_iris_centre"), list) else anchor.get("iris_centre_x", 0))
            cy = float(anchor.get("corrected_iris_centre", [0, 0])[1] if isinstance(
                anchor.get("corrected_iris_centre"), list) else anchor.get("iris_centre_y", 0))
            cr = float(anchor.get("corrected_iris_radius", anchor.get("iris_radius", 12)))
            seed_xyr = (cx, cy, cr)
            seed_source = "anchor"
        # 2. nearby anchor (±5 frames)
        if seed_xyr is None:
            for delta in range(1, 6):
                for ofr in (fr - delta, fr + delta):
                    a = existing_anchors.get(ofr)
                    if a and a.get("image_side_eye", a.get("eye_image_side")) == state["eye_key"]:
                        if isinstance(a.get("corrected_iris_centre"), list):
                            cx, cy = a["corrected_iris_centre"]
                        else:
                            cx, cy = a.get("iris_centre_x", 0), a.get("iris_centre_y", 0)
                        cr = a.get("corrected_iris_radius", a.get("iris_radius", 12))
                        seed_xyr = (float(cx), float(cy), float(cr))
                        seed_source = "anchor_nearby"
                        break
                if seed_xyr is not None:
                    break
        # PATCH 2 (2026-06-28): if the clinician just edited a nearby frame, that edit
        # WINS over fresh tracker output for the next CLINICIAN_PROPAGATION_FRAMES.
        # Reason: when the tracker has lost the iris (e.g. medial gaze, partial limbus),
        # tracker output on frame N+1 is still wrong; carrying the clinician's frame N
        # correction forward is the right behaviour for that local window. Outside the
        # window the tracker takes back over, which is also correct because by then it
        # has likely re-acquired.
        last_manual_fr = state.get("last_manual_frame_no", -1)
        last_manual = state.get("last_manual_xyr")
        if (seed_xyr is None and last_manual is not None and last_manual_fr >= 1
                and (fr - last_manual_fr) <= CLINICIAN_PROPAGATION_FRAMES
                and (fr - last_manual_fr) >= 0):
            seed_xyr = (float(last_manual[0]), float(last_manual[1]), float(last_manual[2]))
            seed_source = f"clinician_propagated(+{fr - last_manual_fr})"
        # 3 + 5 — tracker exact, tracker nearest, Stage-0 — handled by the helper.
        # (5 is "Stage-0 iris seed" per the spec, returned as the last entry from
        # _seed_iris_from_tracking when no tracker iris is available.)
        if seed_xyr is None:
            s = _seed_iris_from_tracking(tracking_rows, fr, state["eye_key"], W, H,
                                          fallback_r=12.0,
                                          approved_iris=approved_iris.get(state["eye_key"]))
            if s is not None:
                seed_xyr = (float(s[0]), float(s[1]), float(s[2]))
                # The helper returns approved_iris last; distinguish "tracker" from "stage0"
                # by checking whether the approved iris equals it.
                appr = approved_iris.get(state["eye_key"])
                seed_source = "stage0" if (appr is not None and
                                            abs(s[0] - appr[0]) < 0.5 and
                                            abs(s[1] - appr[1]) < 0.5) else "tracker"
        # 4 — older previous manual (outside the recency window): last-resort, before
        # parking.
        if seed_xyr is None and state.get("last_manual_xyr") is not None:
            mxyr = state["last_manual_xyr"]
            seed_xyr = (float(mxyr[0]), float(mxyr[1]), float(mxyr[2]))
            seed_source = "previous_manual_stale"
        # If we found a real seed, use it. Otherwise PARK the marker in a neutral visible
        # position so the clinician can drag it onto the visible iris. The marker is never
        # hidden under the "always visible + parked" rule (2026-06-28).
        if seed_xyr is not None:
            cx, cy = float(seed_xyr[0]), float(seed_xyr[1])
            cr = float(seed_xyr[2])
            state["is_parked"] = False
            state["iris_source"] = seed_source
        else:
            px, py, psrc = _parked_iris_xy(state["eye_key"])
            _src, cr = _default_iris_radius_clamped()
            cx, cy = px, py
            state["is_parked"] = True
            state["iris_source"] = f"parked:{psrc}"
        # Clamp to the visible source frame so the marker can never appear off-screen.
        cx = max(cr + 1.0, min(float(W) - cr - 1.0, cx))
        cy = max(cr + 1.0, min(float(H) - cr - 1.0, cy))
        state["iris_x"], state["iris_y"], state["iris_r"] = cx, cy, cr
        state["marker_visible"] = True            # forced True always
        state["iris_dirty"] = False

        # ---- INIT THE EYE-OPENING OVAL PER "ALWAYS VISIBLE + PARKED" RULE -------
        # Priority: saved anchor's contour > Stage-0 oval > PARKED oval centred on the
        # current iris position (so it lands roughly where an eye opening would be).
        oval = None
        oval_parked = False
        if anchor and anchor.get("eye_opening_contour"):
            try:
                pts = np.asarray([(p["x"], p["y"]) for p in anchor["eye_opening_contour"]], np.float32)
                oval = _ellipse_from_polygon(pts)
                # Sanity-discard a saved oval if it's wildly out of proportion vs the iris.
                if oval is not None and not (
                        5 <= oval["a"] <= 8 * max(state["iris_r"], 1.0)
                        and 5 <= oval["b"] <= 6 * max(state["iris_r"], 1.0)):
                    print(f"  [warn] anchor's contour for frame {fr} is out of proportion vs the iris; "
                          f"falling back to Stage-0 seed (a={oval['a']:.0f}, b={oval['b']:.0f}, "
                          f"iris_r={state['iris_r']:.0f})")
                    oval = None
            except Exception:
                oval = None
        if oval is None and stage0_oval.get(state["eye_key"]) is not None:
            oval = dict(stage0_oval[state["eye_key"]])
        if oval is None:
            # Parked oval — wide ellipse around the parked/current iris position. Clipped
            # to a moderate size so it doesn't span the bridge of the nose like the old bug.
            oval = _parked_oval(state["eye_key"], state["iris_x"],
                                  state["iris_y"], state["iris_r"])
            oval_parked = True
        state["oval"] = oval
        state["oval_is_parked"] = oval_parked
        state["contour_dirty"] = False
        state["contour_handle"] = "centre"
        return True

    _read_frame(1)

    # -------- SAVE HELPERS (spec'd 2026-06-28) ----------------------------------
    # `_save_current_anchor(reason)` writes one anchor record into `existing_anchors`
    # AND persists `teaching_anchors.json` AND saves the raw frame PNG. Used by both
    # the explicit `s` keypress and by the auto-save-on-leave-frame path.
    # `_persist_anchors_index()` writes the JSON file alone.
    def _persist_anchors_index():
        anchors_list = sorted(existing_anchors.values(), key=lambda a: a["frame_number"])
        payload = {
            "clinician_teaching_mode_used": True,
            "active_teaching_eye": _eye_label(state["eye_key"]),
            "rescue_anchor_count": len(anchors_list),
            "teaching_examples_saved": len(anchors_list),
            "teaching_examples_dir": str(examples_dir),
            "teaching_example_paths": [str(examples_dir / a["saved_frame_image"])
                                         for a in anchors_list],
            "tracking_rerun_with_rescue_anchors": False,
            "approved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "anchors": anchors_list,
        }
        anchors_index.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _clamp_to_frame(x, y, r):
        """Clamp (x, y, r) so the saved iris circle is anatomically plausible — fully
        inside the video frame and at least DEFAULT_IRIS_MIN px. Guards against the bug
        observed in manual testing where the saved record carried coords outside the
        visible canvas (image y ≈ 977 on a 540-tall overlay)."""
        cx = max(0.0, min(float(W) - 1.0, float(x)))
        cy = max(0.0, min(float(H) - 1.0, float(y)))
        cr = max(DEFAULT_IRIS_MIN, float(r))
        return cx, cy, cr

    def _save_current_anchor(reason: str = "manual") -> bool:
        """Save an anchor for the CURRENT frame using the current state. Returns True
        on success, False when nothing meaningful to save (e.g. marker hidden and never
        touched). Used by both `s` and the auto-save-on-leave path."""
        if not state["marker_visible"]:
            return False
        if not (state["iris_dirty"] or state["contour_dirty"]
                or state["frame_no"] in existing_anchors):
            # Nothing has changed — don't bother re-saving.
            return False
        fr_no = state["frame_no"]
        contour_polygon = None
        if state["contour_dirty"] or (
                existing_anchors.get(fr_no)
                and existing_anchors[fr_no].get("eye_opening_contour")):
            if state["oval"] is not None:
                pts = _ellipse_to_polygon(state["oval"], n=24)
                contour_polygon = [{"x": round(float(p[0]), 2),
                                     "y": round(float(p[1]), 2)} for p in pts]
        # Clamp to the visible frame so a stale carry-over y can't be persisted out-of-bounds.
        cx_save, cy_save, cr_save = _clamp_to_frame(state["iris_x"], state["iris_y"], state["iris_r"])
        anchor_record = {
            "active_teaching_eye": _eye_label(state["eye_key"]),
            "image_side_eye": state["eye_key"],
            "patient_anatomical_eye": _eye_label(state["eye_key"]),
            "frame_number": fr_no,
            "time_sec": round(fr_no / max(1e-6, fps), 4),
            "corrected_iris_centre": [round(cx_save, 2), round(cy_save, 2)],
            "corrected_iris_radius": round(cr_save, 2),
            "image_size_at_save": [int(W), int(H)],
            "display_size_at_save": [int(W * state["scale"]),
                                       int(H * state["scale"])],
            "scale_at_save": round(float(state["scale"]), 4),
            "iris_visibility": "partial",
            "correction_reason": "tracker_lost_or_wrong",
            "corrected_by_clinician": True,
            # The iris/contour markers begin in a "parked" state when no real seed is
            # available; the clinician then drags them onto the visible anatomy. Recording
            # whether the marker was parked before the edit + whether it was placed
            # manually makes the audit trail honest about clinician intent.
            "was_parked_before_correction": bool(state.get("iris_source", "")
                                                  .startswith("parked")),
            "placed_manually": bool(state["iris_dirty"]
                                     or state["contour_dirty"]),
            "save_reason": reason,            # "manual" or "auto_on_leave"
            "iris_source_at_init": state.get("iris_source", "unknown"),
            "eye_opening_contour": contour_polygon,
            "iris_edited": bool(state["iris_dirty"]
                                 or (existing_anchors.get(fr_no) is not None)),
            "contour_edited": bool(state["contour_dirty"]
                                    or contour_polygon is not None),
            "approved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "saved_frame_image": f"frame_{fr_no:06d}_{state['eye_key']}.png",
        }
        existing_anchors[fr_no] = anchor_record
        img_to_save = state["current_frame_raw"] if state["current_frame_raw"] is not None \
            else state["current_frame_overlay"]
        cv2.imwrite(str(examples_dir / anchor_record["saved_frame_image"]), img_to_save)
        _persist_anchors_index()
        s = state["scale"]
        dx_d, dy_d = state["iris_x"] * s, state["iris_y"] * s
        print(f"  [anchor-saved] frame={fr_no} eye=Right image_side=L "
              f"display=({dx_d:.0f},{dy_d:.0f}) image=({state['iris_x']:.1f},{state['iris_y']:.1f}) "
              f"radius={state['iris_r']:.1f} reason={reason} path={anchors_index}")
        state["iris_dirty"] = False
        state["contour_dirty"] = False
        # toast on screen for ~2 seconds (frames; we just record frame_no)
        state["toast_text"] = f"ANCHOR SAVED — frame {fr_no}, Right eye"
        state["toast_frame_no"] = fr_no
        return True

    def _read_frame_with_autosave(new_fr: int):
        """Wrapper that auto-saves any unsaved edit on the CURRENT frame before reading
        the new one. This is the spec'd 'auto-save on leave' safety net so the clinician
        cannot accidentally lose corrections by scrubbing onward.

        Also snapshots the clinician's current manual marker position into
        `state["last_manual_xyr"]` so the next frame's seed-priority step 4 can carry
        the manual marker forward when neither anchor nor tracker has a seed.
        """
        if new_fr != state["frame_no"]:
            was_dirty = state["iris_dirty"] or state["contour_dirty"]
            if was_dirty and state["marker_visible"]:
                _save_current_anchor(reason="auto_on_leave")
            # Snapshot the current manual marker before leaving — used as priority 4 in
            # _read_frame() on the next frame. Always carry it forward, not just on
            # dirty frames: tracker-lost stretches benefit from continuity even when
            # the clinician hasn't moved the marker on this frame.
            state["last_manual_xyr"] = (
                float(state["iris_x"]), float(state["iris_y"]), float(state["iris_r"]))
            # PATCH 2 (2026-06-28): record the frame number ONLY when the clinician
            # actually edited this frame. That timestamp is what gates the recency-
            # based "clinician propagation wins over tracker" rule in _read_frame.
            # On unedited scrubbing visits we deliberately leave last_manual_frame_no
            # alone so the tracker is allowed to take back over after the window.
            if was_dirty:
                state["last_manual_frame_no"] = state["frame_no"]
        _read_frame(new_fr)

    # --- mouse: behaviour depends on mode ---
    #   IRIS mode    — click + drag moves the iris centre; wheel resizes.
    #   CONTOUR mode — click the nearest oval handle (centre or one of four cardinals),
    #                  drag to reshape; wheel resizes the iris (not the contour, since the
    #                  iris is the more frequent edit).
    def _nearest_contour_handle(ix, iy):
        e = state["oval"]
        if e is None:
            return None
        handles = _ellipse_handle_positions(e)
        handles["centre"] = (e["cx"], e["cy"])
        best = None
        for name, (hx, hy) in handles.items():
            d = float(np.hypot(hx - ix, hy - iy))
            if best is None or d < best[0]:
                best = (d, name)
        return best[1] if best and best[0] < 60.0 / max(state["scale"], 1e-6) else None

    def _drag_contour_handle(handle, ix, iy):
        e = state["oval"]
        if e is None:
            return
        if handle == "centre":
            e["cx"], e["cy"] = float(ix), float(iy)
        else:
            dx, dy = float(ix - e["cx"]), float(iy - e["cy"])
            ca, sa = np.cos(e["theta"]), np.sin(e["theta"])
            proj_long = dx * ca + dy * sa
            proj_perp = -dx * sa + dy * ca
            if handle == "medial":
                e["a"] = max(8.0, -proj_long)
            elif handle == "lateral":
                e["a"] = max(8.0, proj_long)
            elif handle == "upper":
                e["b"] = max(6.0, -proj_perp)
            elif handle == "lower":
                e["b"] = max(6.0, proj_perp)
        state["contour_dirty"] = True

    # --- iris-mode click behaviour (SIMPLIFIED 2026-06-28).
    # There is ONE active iris circle. The click handler classifies every click into:
    #   * [iris-select] — click within (iris_r + 8) px of the existing circle's centre →
    #                    select the same circle and start dragging it.
    #   * [iris-reset]  — click further away → recentre the same circle to the click point
    #                    and start dragging from there.
    # The marker is NEVER created as a second object. The marker is ALWAYS visible in iris
    # mode (state["marker_visible"] is forced True). If there is genuinely no prior centre
    # (very first click of a session, no seed at all), the first click acts as a reset and
    # the radius is taken from the default-radius rule (clamped 10–45 px).
    def _iris_click_action(ix, iy):
        grab_radius = max(8.0, state["iris_r"] + 8.0)
        d = float(np.hypot(state["iris_x"] - ix, state["iris_y"] - iy))
        return "select" if d <= grab_radius else "reset"

    def on_mouse(ev, x, y, flags, _):
        ix = x / state["scale"]
        iy = y / state["scale"]
        if ev == cv2.EVENT_LBUTTONDOWN:
            if state["mode"] == MODE_IRIS:
                # The marker is ALWAYS visible (parked or seeded). Every click is either
                # `select` (within iris_r + 8 px) or `reset` (further away). Clicking
                # anywhere "unparks" the marker because the clinician has now committed
                # to a position.
                action = _iris_click_action(ix, iy)
                state["drag"] = True
                state["iris_x"], state["iris_y"] = ix, iy
                state["iris_dirty"] = True
                state["last_edited_radius"] = float(state["iris_r"])
                state["is_parked"] = False
                tag = "iris-reset" if action == "reset" else "iris-select"
                print(f"  [{tag}] frame={state['frame_no']} eye=Right image_side=L "
                      f"display=({x},{y}) image=({ix:.0f},{iy:.0f}) "
                      f"radius={state['iris_r']:.1f}")
            else:
                h = _nearest_contour_handle(ix, iy)
                if h is not None:
                    state["contour_handle"], state["drag"] = h, True
                    _drag_contour_handle(h, ix, iy)
                    state["oval_is_parked"] = False
                    print(f"  [contour-click] handle={h} frame={state['frame_no']}")
                else:
                    print(f"  [contour-click] no handle near click; nearest is too far away")
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            if state["mode"] == MODE_IRIS:
                state["iris_x"], state["iris_y"] = ix, iy
                state["iris_dirty"] = True
                print(f"  [iris-move] frame={state['frame_no']} eye=Right image_side=L "
                      f"display=({x},{y}) image=({ix:.0f},{iy:.0f})")
            else:
                _drag_contour_handle(state["contour_handle"], ix, iy)
        elif ev == cv2.EVENT_LBUTTONUP:
            state["drag"] = False
        elif ev == cv2.EVENT_MOUSEWHEEL:
            # Faster resize — 3 px per wheel tick (was 1).
            delta = 3 if (flags >> 16) > 0 else -3
            state["iris_r"] = max(DEFAULT_IRIS_MIN, state["iris_r"] + delta)
            state["iris_dirty"] = True
            state["last_edited_radius"] = float(state["iris_r"])
            print(f"  [iris-resize] frame={state['frame_no']} eye=Right image_side=L "
                  f"radius={state['iris_r']:.1f}")

    cv2.setMouseCallback(win, on_mouse)

    saved_this_session = 0
    while True:
        overlay = state["current_frame_overlay"]
        if overlay is None:
            break
        # Compose display: tracker overlay frame + clinician's iris circle (cyan).
        # The iris circle the clinician is currently editing is drawn on TOP of the overlay
        # so they can see how it relates to the tracker's circle (which is already drawn on
        # the overlay).
        view = overlay.copy()
        # Fit the frame to a reasonable display size (max 1280 wide).
        scale = min(1.0, 1280.0 / W)
        state["scale"] = scale
        DW, DH = int(W * scale), int(H * scale)
        if scale != 1.0:
            view = cv2.resize(view, (DW, DH))
        # NB: the "Right" / "Left" anatomical labels and the cyan iris circles + ovals you
        # already see on the eyes are baked into tracking_overlay.mp4 by the V1 tracker —
        # they are NOT drawn by the scrubber. So we don't need to draw a faint inactive-eye
        # marker too; that just adds visual noise (it produced a "ghost" iris on the
        # bridge of the nose when the tracker's CSV had no iris row for the inactive eye).
        # The HUD bar above the frame tells the clinician which eye is currently active.

        # ---- ACTIVE eye: solid editable markers -----------------------------
        # The clinician's iris circle is drawn distinct from the tracker's baked-in cyan
        # circles: bright MAGENTA when unedited (so it doesn't blend with anything),
        # bright YELLOW the moment the clinician drags it, bright GREEN once saved as an
        # anchor. The label "ACT" + eye letter is drawn next to it so the clinician can
        # always find it on the frame.
        #
        # CRITICAL: only draw the marker when state["marker_visible"] is True. If the
        # ALWAYS draw the iris marker (parked or seeded). marker_visible is forced True
        # under the "always visible + parked" rule (2026-06-28). If is_parked is True the
        # marker is shown with a dashed-look + UNPLACED label so the clinician sees that
        # they should drag it onto the visible iris.
        anchor = existing_anchors.get(state["frame_no"])
        if state["marker_visible"]:
            if anchor:
                iris_col = (0, 255, 0)        # SAVED — bright green
            elif state["iris_dirty"]:
                iris_col = (0, 255, 255)      # EDITED — bright yellow
            else:
                iris_col = (255, 0, 255)      # ready / parked — bright magenta
            cx, cy = int(state["iris_x"] * scale), int(state["iris_y"] * scale)
            cr = max(3, int(state["iris_r"] * scale))
            cv2.circle(view, (cx, cy), cr, iris_col, 3, cv2.LINE_AA)
            cv2.drawMarker(view, (cx, cy), iris_col, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            # 2026-06-28: on-eye text labels (ACT-Right, r=..., UNPLACED) were removed
            # per the cleaner-overlay rule. Status now lives in the top HUD. Only the
            # circle and the centre cross sit on the eye.

        # Draw the eye-opening oval ONLY in contour mode (per the simplified iris-mode rule:
        # in iris mode there is just the iris circle, the eye label, and the save status).
        if state["mode"] == MODE_CONTOUR and state["oval"] is not None:
            e = state["oval"]
            if anchor and anchor.get("eye_opening_contour"):
                contour_col = (0, 200, 0)         # already saved
            elif state["contour_dirty"]:
                contour_col = (255, 255, 0)       # unsaved edit
            else:
                contour_col = (220, 220, 0)       # active contour mode
            cv2.ellipse(view,
                        (int(e["cx"] * scale), int(e["cy"] * scale)),
                        (max(2, int(e["a"] * scale)), max(2, int(e["b"] * scale))),
                        float(np.degrees(e["theta"])), 0, 360, contour_col,
                        2, cv2.LINE_AA)
            # Handles: diamond for centre, squares for the four cardinals; the last-clicked
            # one is highlighted cyan so the clinician knows what +/- will resize.
            handles = _ellipse_handle_positions(e)
            handles["centre"] = (e["cx"], e["cy"])
            for name, (hx, hy) in handles.items():
                pp = (int(hx * scale), int(hy * scale))
                sel = (name == state["contour_handle"])
                hcol = (0, 255, 255) if sel else contour_col
                if name == "centre":
                    cv2.drawMarker(view, pp, hcol, cv2.MARKER_DIAMOND,
                                   12 if sel else 9, 2, cv2.LINE_AA)
                else:
                    sz = 5 if sel else 4
                    cv2.rectangle(view, (pp[0] - sz, pp[1] - sz),
                                  (pp[0] + sz, pp[1] + sz), hcol, -1)
            # UNPLACED label for parked contours so the clinician sees the contour is a
            # "drag me to the eye opening" default, not a real Stage-0 or saved seed.
            if state["oval_is_parked"]:
                label = "UNPLACED CONTOUR - DRAG/RESIZE TO EYE OPENING"
                lx, ly = int(e["cx"] * scale) - 80, int(e["cy"] * scale) - int(e["b"] * scale) - 8
                cv2.putText(view, label, (max(8, lx), max(20, ly)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, label, (max(8, lx), max(20, ly)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_col, 1, cv2.LINE_AA)

        # HUD strip on top — TWO LINES.
        # Line 1: active teaching eye banner (LOCKED). Line 2: frame/mode/save status.
        cv2.rectangle(view, (0, 0), (DW, 56), (0, 0, 0), -1)
        cv2.putText(view, "ACTIVE TEACHING EYE: Right",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2, cv2.LINE_AA)
        t_sec = state["frame_no"] / max(1e-6, fps)
        # Save status: SAVED if there's an anchor AND no further edits, UNSAVED EDIT if
        # iris_dirty or contour_dirty, unedited otherwise.
        if state["iris_dirty"] or state["contour_dirty"]:
            sav_lbl = "UNSAVED EDIT — press s to save anchor"
            sav_col = (0, 255, 255)   # bright yellow
        elif anchor:
            sav_lbl = "SAVED anchor for this frame"
            sav_col = (0, 255, 0)     # green
        else:
            sav_lbl = "unedited"
            sav_col = (200, 200, 200)
        mode_lbl = "IRIS edit" if state["mode"] == MODE_IRIS else "CONTOUR oval edit"
        parked_hint = "  [marker UNPLACED — drag onto iris]" if state["is_parked"] else ""
        # Frame counter
        cv2.putText(view,
                    f"frame {state['frame_no']}/{total}  t={t_sec:5.2f}s  "
                    f"mode: {mode_lbl}{parked_hint}   "
                    f"saved this session: {saved_this_session}  total anchors: {len(existing_anchors)}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Save-status pill on the right side of the top bar (drawn in its own colour)
        (tw, th), _bl = cv2.getTextSize(sav_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(view, sav_lbl, (DW - tw - 14, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, sav_lbl, (DW - tw - 14, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, sav_col, 1, cv2.LINE_AA)

        # Centred "ANCHOR SAVED" toast — shown for ~2 seconds after a save
        # (we approximate "2 seconds" by ~2 * fps frame ticks).
        if state["toast_text"] and state["toast_frame_no"] >= 0:
            ticks_since_save = state["frame_no"] - state["toast_frame_no"]
            if -2 <= ticks_since_save <= max(3, int(2 * fps)):
                toast = state["toast_text"]
                (tw, th), _bl = cv2.getTextSize(toast, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                pad = 18
                tx, ty = (DW - tw) // 2, 100
                cv2.rectangle(view, (tx - pad, ty - th - pad),
                              (tx + tw + pad, ty + pad), (0, 60, 0), -1)
                cv2.rectangle(view, (tx - pad, ty - th - pad),
                              (tx + tw + pad, ty + pad), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(view, toast, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(view, toast, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                state["toast_text"] = ""

        # HUD at bottom — TWO LINES.
        cv2.rectangle(view, (0, DH - 56), (DW, DH), (0, 0, 0), -1)
        cv2.putText(view,
                    "i: click anywhere to place/move iris | drag = move | "
                    "+/- = resize 3px | */_ = resize 10px | s = SAVE anchor",
                    (10, DH - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(view,
                    "[i]ris  [c]ontour-oval  Tab cycles   "
                    "<- -> 1 frame  [ ] = 10  { } = 60  g/G start/end  Space play   "
                    "Enter does NOT save — use s   x delete anchor   q quit",
                    (10, DH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow(win, view)
        # PATCH 1 (2026-06-28): non-blocking refresh so mouse drag is smooth. Previously
        # `cv2.waitKey(0)` blocked the render loop until a key was pressed; during a drag
        # the screen never re-rendered and the marker appeared "stuck". The simple iris
        # editor proved a short ~16 ms wait gives smooth ~60 Hz drag without busy-spinning.
        wait_ms = int(1000 / max(fps, 1.0)) if state["play"] else 16
        k = cv2.waitKey(wait_ms) & 0xFF

        if k == ord("q") or k == 27:        # q or Esc
            # Auto-save any unsaved edit on the current frame before quitting.
            if (state["iris_dirty"] or state["contour_dirty"]) and state["marker_visible"]:
                _save_current_anchor(reason="auto_on_quit")
            break
        elif k == ord(" "):                 # space — play/pause
            state["play"] = not state["play"]
        elif k == 13:                       # Enter — explicit no-op with a one-line nudge
            print("  [enter] Enter does NOT save. Press 's' to save the anchor.")
        # ---- mode + eye switching ----
        elif k == ord("i"):
            state["mode"] = MODE_IRIS
            print(f"  [mode] IRIS (frame {state['frame_no']})")
        elif k == ord("c"):
            state["mode"] = MODE_CONTOUR
            print(f"  [mode] CONTOUR (frame {state['frame_no']})")
        elif k == 9:                        # Tab cycles mode
            state["mode"] = MODE_CONTOUR if state["mode"] == MODE_IRIS else MODE_IRIS
            print(f"  [mode] {state['mode'].upper()} (Tab)")
        elif k == ord("e"):
            # Eye-switch disabled while teaching mode is LOCKED to patient Right (image L).
            print("  [eye] switch disabled — teaching mode locked to patient Right "
                  "for this manual test")
        # ---- navigation (auto-saves any unsaved edit on the CURRENT frame) ----
        elif k in (0x52, 83) or k == ord("n"):
            state["play"] = False
            _read_frame_with_autosave(state["frame_no"] + 1)
        elif k in (0x50, 81) or k == ord("b"):
            state["play"] = False
            _read_frame_with_autosave(state["frame_no"] - 1)
        elif k == ord("]"):
            _read_frame_with_autosave(state["frame_no"] + 10); state["play"] = False
        elif k == ord("["):
            _read_frame_with_autosave(state["frame_no"] - 10); state["play"] = False
        elif k == ord("}") or k == ord(">"):
            _read_frame_with_autosave(state["frame_no"] + 60); state["play"] = False
        elif k == ord("{") or k == ord("<"):
            _read_frame_with_autosave(state["frame_no"] - 60); state["play"] = False
        elif k == ord("g"):
            _read_frame_with_autosave(1)
        elif k == ord("G"):
            _read_frame_with_autosave(total)
        # ---- edits (mode-specific) ----
        # IRIS radius keys:  +/=  -> +3 px (was +1)         (single-key step)
        #                    -    -> -3 px (was -1)
        #                    *    -> +10 px (no conflict)   (large step)
        #                    _    -> -10 px (no conflict)
        # CONTOUR oval keys behave as before (per-handle ±1 px).
        elif k in (ord("+"), ord("=")):
            if state["mode"] == MODE_IRIS:
                state["iris_r"] = min(500.0, state["iris_r"] + 3.0)
                state["iris_dirty"] = True
                state["last_edited_radius"] = float(state["iris_r"])
                print(f"  [iris-resize] frame={state['frame_no']} eye=Right image_side=L "
                      f"radius={state['iris_r']:.1f} step=+3")
            elif state["oval"] is not None:
                h = state["contour_handle"]
                e = state["oval"]
                if h in ("medial", "lateral"):
                    e["a"] = e["a"] + 1.0
                elif h in ("upper", "lower"):
                    e["b"] = e["b"] + 1.0
                else:
                    e["a"] = e["a"] + 1.0
                    e["b"] = e["b"] + 1.0
                state["contour_dirty"] = True
        elif k == ord("-"):
            if state["mode"] == MODE_IRIS:
                state["iris_r"] = max(DEFAULT_IRIS_MIN, state["iris_r"] - 3.0)
                state["iris_dirty"] = True
                state["last_edited_radius"] = float(state["iris_r"])
                print(f"  [iris-resize] frame={state['frame_no']} eye=Right image_side=L "
                      f"radius={state['iris_r']:.1f} step=-3")
            elif state["oval"] is not None:
                h = state["contour_handle"]
                e = state["oval"]
                if h in ("medial", "lateral"):
                    e["a"] = max(8.0, e["a"] - 1.0)
                elif h in ("upper", "lower"):
                    e["b"] = max(6.0, e["b"] - 1.0)
                else:
                    e["a"] = max(8.0, e["a"] - 1.0)
                    e["b"] = max(6.0, e["b"] - 1.0)
                state["contour_dirty"] = True
        elif k == ord("*") and state["mode"] == MODE_IRIS:
            state["iris_r"] = min(500.0, state["iris_r"] + 10.0)
            state["iris_dirty"] = True
            state["last_edited_radius"] = float(state["iris_r"])
            print(f"  [iris-resize] frame={state['frame_no']} eye=Right image_side=L "
                  f"radius={state['iris_r']:.1f} step=+10")
        elif k == ord("_") and state["mode"] == MODE_IRIS:
            state["iris_r"] = max(DEFAULT_IRIS_MIN, state["iris_r"] - 10.0)
            state["iris_dirty"] = True
            state["last_edited_radius"] = float(state["iris_r"])
            print(f"  [iris-resize] frame={state['frame_no']} eye=Right image_side=L "
                  f"radius={state['iris_r']:.1f} step=-10")
        elif k == ord("r") and state["mode"] == MODE_CONTOUR and state["oval"] is not None:
            state["oval"]["theta"] -= np.deg2rad(1.5)
            state["contour_dirty"] = True
        elif k == ord("R") and state["mode"] == MODE_CONTOUR and state["oval"] is not None:
            state["oval"]["theta"] += np.deg2rad(1.5)
            state["contour_dirty"] = True
        # ---- save / delete ----
        elif k == ord("s"):
            # Spec'd 2026-06-28: `s` is the ONLY guaranteed save key. Enter does NOT save
            # (auto-save on frame change is the safety net for forgotten saves).
            if _save_current_anchor(reason="manual"):
                saved_this_session += 1
            else:
                print(f"  [no-op] s pressed but nothing to save on frame {state['frame_no']} "
                      f"(marker_visible={state['marker_visible']}, "
                      f"iris_dirty={state['iris_dirty']}, contour_dirty={state['contour_dirty']})")
        elif k == ord("x"):
            fr_no = state["frame_no"]
            if fr_no in existing_anchors:
                rec = existing_anchors.pop(fr_no)
                img_path = examples_dir / rec.get("saved_frame_image", "")
                try:
                    if img_path.exists():
                        img_path.unlink()
                except OSError:
                    pass
                state["iris_dirty"] = False
                state["contour_dirty"] = False
                print(f"  deleted anchor for frame {fr_no}")
        elif state["play"]:
            if state["frame_no"] >= total:
                state["play"] = False
            else:
                _read_frame_with_autosave(state["frame_no"] + 1)

    cap.release()
    if raw_cap is not None:
        raw_cap.release()
    cv2.destroyAllWindows()

    # ------ write the rescue_anchors index ------
    anchors_list = sorted(existing_anchors.values(), key=lambda a: a["frame_number"])
    summary = {
        "clinician_teaching_mode_used": True,
        "rescue_anchor_count": len(anchors_list),
        "teaching_examples_saved": len(anchors_list),
        "teaching_examples_dir": str(examples_dir),
        "teaching_example_paths": [str(examples_dir / a["saved_frame_image"]) for a in anchors_list],
        "tracking_rerun_with_rescue_anchors": False,
        "approved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "anchors": anchors_list,
    }
    anchors_index.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {anchors_index}  ({len(anchors_list)} anchors)")
    return summary
