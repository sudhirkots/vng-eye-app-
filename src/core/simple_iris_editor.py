"""Simple Clinician-Controlled Iris Video Editor (EYE_VNG §19) — hybrid design.

A clean, minimal manual iris annotation tool. NOT the V1 tracker. NOT the detector. NOT the
rescue scrubber. The clinician owns the iris marker, drags it through the video, and saves
per-frame annotations to JSON.

Hybrid principle: **Tracker suggests; clinician owns the annotation.**
  * If V1's tracking.csv exists, this editor reads the per-frame iris (x, y, r) for the
    active eye and draws it as a faint cyan NON-EDITABLE circle.
  * The clinician's magenta circle is the editable truth. The tracker never silently
    overwrites or fights it.
  * `t` copies the tracker suggestion into the clinician marker for the current frame
    (clinician must still press `s` to save).
  * `h` toggles the tracker suggestion's visibility.
  * Saved records carry `"source": "manual"` or `"source": "tracker_accepted_by_clinician"`
    so downstream analysis can distinguish provenance.

Workflow:
  1. The clinician launches `python app.py --video "..." --annotate-iris`.
  2. The editor opens on the RAW video (NOT tracking_overlay.mp4).
  3. The clinician sees one magenta iris circle on the active eye (patient anatomical
     Right for this first version). Drag it onto the iris. Press s to save. Press the
     right arrow to advance one frame. The circle stays where the clinician put it.
  4. Annotations are saved to:
        outputs/<clip_name>_tracked/manual_iris_annotations.json
     in the SOURCE-VIDEO coordinate system.

Design notes:
  * The clinician owns the marker. The circle never jumps back to tracker output, anchor
    output, parking state, or any V1 internal state. After init, no internal logic moves
    the circle except the clinician's mouse.
  * One circle. Always visible. Always draggable. Anywhere in or near it is a grab. Click
    far away recentres the same circle — there is no concept of "second marker".
  * Coordinates are in SOURCE-VIDEO PIXELS throughout. The display is scaled for comfort
    only; mouse coordinates are translated back to source pixels at the boundary.
  * No detector, no nystagmus, no contour editing, no eye switching, no rescue UI, no
    auto-save unless the clinician toggles it on.
"""
from __future__ import annotations

import csv as _csv
import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- constants
ACTIVE_IMAGE_SIDE = "L"                 # image side L == patient anatomical Right
ACTIVE_PATIENT_EYE = "Right"
DISPLAY_MAX_W = 1280                     # window will fit within this width
DEFAULT_RADIUS_SOURCE = 35.0             # source-video px, sensible iris default
MIN_RADIUS = 8.0
RESIZE_STEP = 3.0
GENEROUS_GRAB_PX = 80.0                  # generous hit-test in DISPLAY pixels
RADIUS_GRAB_FACTOR = 1.25                # alternative grab: radius * 1.25

COL_MARKER = (255, 0, 255)               # bright magenta — editable marker
COL_DRAG = (0, 255, 255)                 # bright yellow — while dragging
COL_SAVED_FLASH = (0, 255, 0)            # bright green — brief flash on save
COL_SUGGESTION = (200, 180, 80)          # faint cyan — tracker's non-editable suggestion
COL_HUD_BG = (0, 0, 0)
COL_HUD_FG = (255, 255, 255)
TOAST_FRAMES = 30                        # ~0.5s at 60Hz HUD toast for `t` acceptance


# --------------------------------------------------------------------------- helpers
def _load_existing_annotations(path: Path) -> Dict[int, Dict]:
    """Read any prior manual_iris_annotations.json and return a dict keyed by frame."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[int, Dict] = {}
    for rec in payload.get("annotations", []):
        try:
            out[int(rec["frame_number"])] = rec
        except Exception:
            continue
    return out


def _stage0_initial_iris(approved_path: Path, image_side: str) -> Optional[Tuple[float, float, float]]:
    """Optionally read the Stage-0 approved iris for the active eye to seed the FIRST
    frame only. After the first frame, the clinician's marker carries forward."""
    if not approved_path.exists():
        return None
    try:
        approved = json.loads(approved_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    iris = (approved.get("iris") or {}).get(image_side)
    if not iris:
        return None
    try:
        return float(iris["x"]), float(iris["y"]), float(iris["radius"])
    except Exception:
        return None


def _load_tracker_suggestions(csv_path: Path, image_side: str) -> Dict[int, Tuple[float, float, float]]:
    """Read tracker iris (cx, cy, r) per frame from V1's tracking.csv for the active eye.
    Returns a dict {frame_number: (x, y, r)} in SOURCE-VIDEO coordinates.

    These values are used ONLY as a faint cyan suggestion overlay (non-editable). The
    clinician's marker is never silently overwritten with them. The clinician can press
    `t` to accept the suggestion for the current frame.

    The active eye for this version is patient anatomical Right (image_side == 'L'), so we
    read the 'left_*' columns of tracking.csv ('left' there means image-side L)."""
    if not csv_path.exists():
        return {}
    if image_side == "L":
        cx_col, cy_col, r_col = "raw_left_iris_center_x", "raw_left_iris_center_y", "left_iris_radius"
    else:
        cx_col, cy_col, r_col = "raw_right_iris_center_x", "raw_right_iris_center_y", "right_iris_radius"
    out: Dict[int, Tuple[float, float, float]] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                vx = (row.get(cx_col) or "").strip()
                vy = (row.get(cy_col) or "").strip()
                vr = (row.get(r_col) or "").strip()
                if not (vx and vy and vr):
                    continue
                try:
                    fr = int(row["frame_number"])
                    out[fr] = (float(vx), float(vy), float(vr))
                except (ValueError, KeyError):
                    continue
    except OSError:
        return {}
    return out


# --------------------------------------------------------------------------- main
def run_simple_iris_editor(video_path: Path, output_dir: Path) -> None:
    """Open the simple iris editor on the RAW video. The editor opens a single OpenCV
    window, the clinician scrubs/drags/saves, and on quit the editor writes
    `manual_iris_annotations.json` into `outputs/<clip>_tracked/`. Nothing else."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(output_dir) / (video_path.stem + "_tracked")
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "manual_iris_annotations.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    SRC_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    SRC_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # display scale: fit the frame inside DISPLAY_MAX_W while preserving aspect ratio.
    scale = min(1.0, DISPLAY_MAX_W / max(1, SRC_W))
    DW, DH = int(SRC_W * scale), int(SRC_H * scale)

    # Existing annotations from prior sessions are READ (so the clinician sees them
    # marked as "saved") but the clinician's working circle is NEVER auto-loaded from
    # them on later frames — that would be the tracker-owns-marker problem. The
    # working circle only loads from an annotation when the clinician explicitly
    # navigates to that frame for the first time in the session (treated as "go to
    # an annotated frame, copy the saved marker into the working circle").
    annotations: Dict[int, Dict] = _load_existing_annotations(annotations_path)

    # Tracker suggestions: read once from V1's tracking.csv. Used ONLY as a faint
    # cyan non-editable overlay and as one of the init fallbacks. Empty dict if the
    # tracker has not been run for this clip.
    tracker_suggestions: Dict[int, Tuple[float, float, float]] = _load_tracker_suggestions(
        out_dir / "tracking.csv", ACTIVE_IMAGE_SIDE
    )

    # First-frame init priority (used ONLY at the very start of the session):
    #   1. saved annotation for frame 1, if any
    #   2. previous clinician position (n/a on the very first frame)
    #   3. tracker suggestion for frame 1, if any
    #   4. Stage-0 approved iris seed for the active eye
    #   5. neutral default (centred-ish, default radius)
    init_x = SRC_W * 0.30
    init_y = SRC_H * 0.50
    init_r = DEFAULT_RADIUS_SOURCE
    if 1 in annotations:
        rec = annotations[1]
        try:
            init_x, init_y = rec["iris_centre"]
            init_r = float(rec["iris_radius"])
        except Exception:
            pass
    elif 1 in tracker_suggestions:
        init_x, init_y, init_r = tracker_suggestions[1]
    else:
        s0 = _stage0_initial_iris(out_dir / "approved_landmarks.json", ACTIVE_IMAGE_SIDE)
        if s0 is not None:
            init_x, init_y, init_r = s0

    # ---- state -----------------------------------------------------------------
    state: Dict = {
        "frame_no": 1,
        "frame_bgr": None,           # current raw video frame (SOURCE resolution)
        "iris_x": float(init_x),
        "iris_y": float(init_y),
        "iris_r": float(init_r),
        "dragging": False,
        "auto_save": False,
        "dirty": False,               # circle moved/resized since last save on this frame
        "play": False,
        "save_flash_until_frame_no": -1,   # for the brief green "ANCHOR SAVED" flash
        "last_visited_frame": -1,     # detect first-time visit of a frame this session
        "show_suggestion": True,      # hybrid: render faint cyan tracker suggestion
        "last_save_source": "manual", # "manual" or "tracker_accepted_by_clinician"
        "toast_text": "",             # transient HUD message (e.g. "tracker accepted")
        "toast_frames_left": 0,
    }

    def _clamp_inside_frame():
        r = state["iris_r"]
        state["iris_x"] = float(max(r + 1.0, min(SRC_W - r - 1.0, state["iris_x"])))
        state["iris_y"] = float(max(r + 1.0, min(SRC_H - r - 1.0, state["iris_y"])))

    def _read_frame(fr: int):
        """Seek and read raw frame `fr` (1-indexed). The iris circle is NEVER overwritten
        on frame change — it keeps the clinician's last position. The only exception is
        when the clinician navigates to a frame they previously SAVED an annotation for:
        in that case the saved annotation is copied into the working circle so it appears
        in its saved place. This is the spec'd "start from previous annotated circle" rule."""
        fr = max(1, min(total, fr))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr - 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        state["frame_bgr"] = frame
        # If we have an annotation for this frame AND this is the first time we're
        # visiting it in this session, snap the working circle to the saved one.
        if fr in annotations and state["last_visited_frame"] != fr:
            try:
                cx, cy = annotations[fr]["iris_centre"]
                cr = float(annotations[fr]["iris_radius"])
                state["iris_x"], state["iris_y"], state["iris_r"] = float(cx), float(cy), float(cr)
            except Exception:
                pass
        state["frame_no"] = fr
        state["last_visited_frame"] = fr
        state["dirty"] = False
        _clamp_inside_frame()
        return True

    def _save_current_frame(reason: str = "manual") -> None:
        # `source` distinguishes a hand-placed marker from one the clinician copied
        # from the tracker suggestion via the `t` key. The clinician still owns it
        # either way — the difference is only in provenance.
        source = state.get("last_save_source", "manual")
        rec = {
            "frame_number": state["frame_no"],
            "time_sec": round(state["frame_no"] / max(1e-6, fps), 4),
            "patient_anatomical_eye": ACTIVE_PATIENT_EYE,
            "image_side_eye": ACTIVE_IMAGE_SIDE,
            "iris_centre": [round(state["iris_x"], 2), round(state["iris_y"], 2)],
            "iris_radius": round(state["iris_r"], 2),
            "coordinate_space": "source_video",
            "corrected_by_clinician": True,
            "source": source,
            "save_reason": reason,
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        annotations[state["frame_no"]] = rec
        # Persist to disk on every save so an unexpected close cannot lose work.
        payload = {
            "video_path": str(video_path),
            "source_resolution": [SRC_W, SRC_H],
            "fps": round(float(fps), 4),
            "active_teaching_eye": ACTIVE_PATIENT_EYE,
            "image_side_eye": ACTIVE_IMAGE_SIDE,
            "n_annotated_frames": len(annotations),
            "annotations": [annotations[k] for k in sorted(annotations)],
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        annotations_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        state["dirty"] = False
        state["save_flash_until_frame_no"] = state["frame_no"]
        # Reset source after the save records it — any further edit on the same frame
        # is, by definition, a manual touch.
        state["last_save_source"] = "manual"
        print(f"[anchor-saved] frame={state['frame_no']} eye=Right image_side=L "
              f"image=({state['iris_x']:.1f},{state['iris_y']:.1f}) "
              f"radius={state['iris_r']:.1f} source={source} reason={reason} "
              f"path={annotations_path}")

    def _go_to_frame(target: int):
        """Move to `target` frame. Auto-save the current frame first if auto_save is on
        and the circle has been edited since the last save."""
        if target != state["frame_no"] and state["dirty"] and state["auto_save"]:
            _save_current_frame(reason="auto_on_leave")
        _read_frame(target)

    # ---- mouse handler ---------------------------------------------------------
    def _hit_test(src_x: float, src_y: float) -> bool:
        """Generous hit: source-pixel distance ≤ max(radius * 1.25, 80 px / scale)."""
        d = float(np.hypot(state["iris_x"] - src_x, state["iris_y"] - src_y))
        threshold = max(state["iris_r"] * RADIUS_GRAB_FACTOR,
                         GENEROUS_GRAB_PX / max(scale, 1e-6))
        return d <= threshold

    def on_mouse(ev, x, y, flags, _):
        # Convert display → source pixels.
        sx = x / scale
        sy = y / scale
        if ev == cv2.EVENT_LBUTTONDOWN:
            hit = _hit_test(sx, sy)
            action = "select" if hit else "reset"
            # In BOTH cases the working circle moves to (sx, sy) so the clinician can
            # drag smoothly from the click point. There is no second circle.
            state["iris_x"], state["iris_y"] = sx, sy
            state["dragging"] = True
            state["dirty"] = True
            state["last_save_source"] = "manual"
            _clamp_inside_frame()
            print(f"  [iris-{action}] frame={state['frame_no']} "
                  f"display=({x},{y}) image=({sx:.0f},{sy:.0f}) "
                  f"radius={state['iris_r']:.1f}")
        elif ev == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["iris_x"], state["iris_y"] = sx, sy
            state["dirty"] = True
            state["last_save_source"] = "manual"
            _clamp_inside_frame()
        elif ev == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif ev == cv2.EVENT_MOUSEWHEEL:
            # Wheel up → grow, wheel down → shrink. RESIZE_STEP px per tick.
            delta = RESIZE_STEP if (flags >> 16) > 0 else -RESIZE_STEP
            state["iris_r"] = float(max(MIN_RADIUS, state["iris_r"] + delta))
            state["dirty"] = True
            state["last_save_source"] = "manual"
            _clamp_inside_frame()

    win = "Iris Editor — clinician-controlled"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DW, DH + 56)         # +56 px for HUD strip
    cv2.setMouseCallback(win, on_mouse)

    print(f"Simple iris editor: {video_path.name}  source={SRC_W}x{SRC_H} @ {fps:.1f} fps  "
          f"display={DW}x{DH}  scale={scale:.3f}")
    print(f"  active eye: {ACTIVE_PATIENT_EYE} (image side {ACTIVE_IMAGE_SIDE})")
    print(f"  annotations file: {annotations_path}")
    print(f"  existing annotations on disk: {len(annotations)}")
    print(f"  tracker suggestions loaded: {len(tracker_suggestions)} "
          f"({'tracking.csv found' if tracker_suggestions else 'tracking.csv absent or empty'})")
    print("  controls: drag = move, wheel = resize, +/- = resize 3 px,")
    print("            ← / → = ±1 frame, [ / ] = ±10, PgUp/PgDn = ±1 sec,")
    print("            t = accept tracker suggestion (then `s` to save),")
    print("            h = hide/show tracker suggestion,")
    print("            Space = play/pause, s = save, a = toggle auto-save, q = quit.")

    _read_frame(1)

    while True:
        if state["frame_bgr"] is None:
            break

        # Render: raw frame → display-scaled → draw circle on top.
        view = cv2.resize(state["frame_bgr"], (DW, DH)) if scale != 1.0 \
            else state["frame_bgr"].copy()

        # Faint cyan tracker SUGGESTION (non-editable). Drawn first so the magenta
        # clinician marker stays on top.
        sugg = tracker_suggestions.get(state["frame_no"]) if state["show_suggestion"] else None
        if sugg is not None:
            sx_ = int(round(sugg[0] * scale))
            sy_ = int(round(sugg[1] * scale))
            sr_ = max(2, int(round(sugg[2] * scale)))
            cv2.circle(view, (sx_, sy_), sr_, COL_SUGGESTION, 1, cv2.LINE_AA)

        col = COL_DRAG if state["dragging"] else COL_MARKER
        # Brief save-confirmation flash — green for one frame after save.
        flashed = (state["save_flash_until_frame_no"] == state["frame_no"])
        if flashed:
            col = COL_SAVED_FLASH
            # Clear after this draw so the next frame is back to magenta.
            state["save_flash_until_frame_no"] = -1
        cx = int(round(state["iris_x"] * scale))
        cy = int(round(state["iris_y"] * scale))
        cr = max(2, int(round(state["iris_r"] * scale)))
        cv2.circle(view, (cx, cy), cr, col, 3, cv2.LINE_AA)
        cv2.drawMarker(view, (cx, cy), col, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        # Deliberately no `ACT-Right r=...` text near the iris — kept off the eye.

        # HUD strip below the video.
        canvas = np.zeros((DH + 56, DW, 3), np.uint8)
        canvas[:DH] = view
        cv2.rectangle(canvas, (0, DH), (DW, DH + 56), COL_HUD_BG, -1)
        saved_lbl = "saved" if (state["frame_no"] in annotations and not state["dirty"]) \
            else ("UNSAVED" if state["dirty"] else "no annotation")
        autosave_lbl = "AUTOSAVE ON" if state["auto_save"] else "autosave off"
        line1 = (f"Iris editor | Right eye | frame {state['frame_no']}/{total}  "
                 f"t={state['frame_no']/max(1e-6, fps):5.2f}s  "
                 f"radius={state['iris_r']:.1f}  ({saved_lbl})  [{autosave_lbl}]")
        cv2.putText(canvas, line1, (8, DH + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_HUD_FG, 1, cv2.LINE_AA)
        sugg_lbl = "tracker ON" if state["show_suggestion"] else "tracker hidden"
        if not tracker_suggestions:
            sugg_lbl = "no tracker.csv"
        line2 = ("drag move | wheel/+- resize | <- -> frame | [ ] +-10 | PgUp/Dn sec | "
                 f"Space play | s save | t accept tracker | h toggle ({sugg_lbl}) | a auto | q quit")
        cv2.putText(canvas, line2, (8, DH + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_HUD_FG, 1, cv2.LINE_AA)
        # Transient HUD toast (e.g. "TRACKER SUGGESTION ACCEPTED — press s to save").
        if state["toast_frames_left"] > 0 and state["toast_text"]:
            (tw, th), _ = cv2.getTextSize(state["toast_text"],
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tx, ty = 12, 30
            cv2.rectangle(canvas, (tx - 6, ty - th - 8),
                          (tx + tw + 6, ty + 8), (0, 0, 0), -1)
            cv2.putText(canvas, state["toast_text"], (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_SUGGESTION, 2, cv2.LINE_AA)
            state["toast_frames_left"] -= 1

        cv2.imshow(win, canvas)
        # CRITICAL: do NOT use waitKey(0) — that blocks the render loop until a key is
        # pressed, which means a mouse drag updates state["iris_*"] but the screen never
        # redraws to show the marker moving. Use a short non-blocking wait (~16 ms ≈ 60 Hz)
        # so the loop continuously re-renders during a drag.
        wait_ms = int(1000 / max(fps, 1.0)) if state["play"] else 16
        k = cv2.waitKey(wait_ms) & 0xFF

        if k == ord("q") or k == 27:
            # Auto-save the current frame on quit if dirty AND auto-save is on.
            if state["dirty"] and state["auto_save"]:
                _save_current_frame(reason="auto_on_quit")
            break
        elif k == ord(" "):
            state["play"] = not state["play"]
        # Navigation: arrows / Shift-arrows / PageUp / PageDown.
        elif k in (83, 0x52) or k == ord("n"):      # right arrow / n
            state["play"] = False
            _go_to_frame(state["frame_no"] + 1)
        elif k in (81, 0x50) or k == ord("b"):      # left arrow / b
            state["play"] = False
            _go_to_frame(state["frame_no"] - 1)
        elif k == ord("]"):
            state["play"] = False
            _go_to_frame(state["frame_no"] + 10)
        elif k == ord("["):
            state["play"] = False
            _go_to_frame(state["frame_no"] - 10)
        elif k == 86:                                # PageDown
            state["play"] = False
            _go_to_frame(state["frame_no"] + int(round(fps)))
        elif k == 85:                                # PageUp
            state["play"] = False
            _go_to_frame(state["frame_no"] - int(round(fps)))
        elif k == ord("g"):
            _go_to_frame(1)
        elif k == ord("G"):
            _go_to_frame(total)
        # Resize.
        elif k in (ord("+"), ord("=")):
            state["iris_r"] = float(state["iris_r"] + RESIZE_STEP)
            state["dirty"] = True
            state["last_save_source"] = "manual"
            _clamp_inside_frame()
        elif k == ord("-"):
            state["iris_r"] = float(max(MIN_RADIUS, state["iris_r"] - RESIZE_STEP))
            state["dirty"] = True
            state["last_save_source"] = "manual"
            _clamp_inside_frame()
        # Accept tracker suggestion for current frame (clinician still must press `s`
        # to save). Does NOT auto-save; the clinician confirms.
        elif k == ord("t"):
            sugg_now = tracker_suggestions.get(state["frame_no"])
            if sugg_now is None:
                state["toast_text"] = "no tracker suggestion for this frame"
                state["toast_frames_left"] = TOAST_FRAMES
                print(f"  [tracker-accept] frame={state['frame_no']} no_suggestion")
            else:
                state["iris_x"], state["iris_y"], state["iris_r"] = (
                    float(sugg_now[0]), float(sugg_now[1]), float(sugg_now[2]))
                state["dirty"] = True
                state["last_save_source"] = "tracker_accepted_by_clinician"
                _clamp_inside_frame()
                state["toast_text"] = "TRACKER SUGGESTION ACCEPTED - press s to save"
                state["toast_frames_left"] = TOAST_FRAMES
                print(f"  [tracker-accept] frame={state['frame_no']} "
                      f"image=({state['iris_x']:.1f},{state['iris_y']:.1f}) "
                      f"radius={state['iris_r']:.1f}")
        elif k == ord("h"):
            state["show_suggestion"] = not state["show_suggestion"]
            print(f"  [tracker-suggestion] "
                  f"{'visible' if state['show_suggestion'] else 'hidden'}")
        # Save / auto-save.
        elif k == ord("s"):
            _save_current_frame(reason="manual")
        elif k == ord("a"):
            state["auto_save"] = not state["auto_save"]
            print(f"  [auto-save] {'ON' if state['auto_save'] else 'off'}")
        # Autoplay advance.
        elif state["play"]:
            if state["frame_no"] >= total:
                state["play"] = False
            else:
                _go_to_frame(state["frame_no"] + 1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Simple iris editor closed. {len(annotations)} annotated frames in "
          f"{annotations_path}.")
