"""V2 Layer 2 — orbit Stage-0 marker.

The clinician marks four points per eye on a single frame:

    medial, lateral, upper, lower.

That's all this module does. NO MediaPipe import, no iris work, no tracking. The result
is saved as `outputs/v2_orbit_tracking_probe/orbit_stage0_v2.json`.

If the caller passes pre-computed eye-zone rectangles from MediaPipe (Layer 1), those
are drawn as faint hints only — not editable, not saved as anatomy.

Image-side convention (locked codebase-wide):
    image-side L = patient anatomical Right
    image-side R = patient anatomical Left
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- constants
POINTS = ("medial", "lateral", "upper", "lower")
POINT_LABEL = {"medial": "M", "lateral": "L", "upper": "U", "lower": "D"}
POINT_COLOUR = {
    "medial":  (255, 100, 255),    # magenta
    "lateral": (100, 255, 255),    # yellow
    "upper":   (255, 200, 100),    # cyan
    "lower":   (100, 200, 255),    # orange
}
COL_HINT = (180, 140, 60)          # faint blue — MediaPipe suggestion rectangles
COL_OVAL_LIVE = (255, 255, 0)      # bright yellow — live oval (once all 4 placed)
COL_HUD_BG = (0, 0, 0)
COL_HUD_FG = (255, 255, 255)

DISPLAY_MAX_W = 1280               # window fits inside this width


# --------------------------------------------------------------------------- helpers
def _fit_ellipse_through_four(points: Dict[str, Tuple[float, float]]
                                ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """Approximate ellipse from 4 named anchor points.
    Centre = midpoint of medial-lateral and upper-lower.
    Long axis = medial-lateral.
    Short axis = upper-lower projected onto the perpendicular.
    Returns ((cx, cy), (a_half, b_half), theta_deg) suitable for cv2.ellipse.
    None if any point is missing."""
    if not all(p in points for p in POINTS):
        return None
    M = np.array(points["medial"], float)
    L = np.array(points["lateral"], float)
    U = np.array(points["upper"], float)
    D = np.array(points["lower"], float)
    centre = (M + L + U + D) / 4.0
    horiz = L - M
    a = float(np.linalg.norm(horiz) / 2.0)
    vert = D - U
    b = float(np.linalg.norm(vert) / 2.0)
    if a < 4.0:
        a = 4.0
    if b < 3.0:
        b = 3.0
    theta = float(np.degrees(np.arctan2(horiz[1], horiz[0])))
    return ((float(centre[0]), float(centre[1])), (a, b), theta)


def _nearest_point(points: Dict[str, Tuple[float, float]], x: float, y: float,
                    grab_px: float) -> Optional[str]:
    best = None
    for name, (px, py) in points.items():
        d = float(np.hypot(px - x, py - y))
        if best is None or d < best[0]:
            best = (d, name)
    if best and best[0] <= grab_px:
        return best[1]
    return None


# --------------------------------------------------------------------------- runner
def run_orbit_stage0(
    frame_bgr: np.ndarray,
    init_frame_number: int,
    video_path: Path,
    resolution: Tuple[int, int],
    fps: float,
    suggested_eye_zones: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    detected_at_frame: Optional[int] = None,
    scan_window_frames: int = 0,
) -> Dict:
    """Open the marking window on the given frame and return the saved JSON dict.

    Returns the dict; the caller writes it to disk (so this function stays I/O-light
    and easy to test). Raises SystemExit if the user quits without saving.
    """
    H, W = frame_bgr.shape[:2]
    scale = min(1.0, DISPLAY_MAX_W / max(1, W))
    DW, DH = int(W * scale), int(H * scale)

    # Per-eye state: {eye_key: {"point_name": (x, y), ...}}
    marks: Dict[str, Dict[str, Tuple[float, float]]] = {"L": {}, "R": {}}

    state = {
        "mode": "medial",     # one of POINTS
        "eye": "L",           # L = patient Right (per convention)
        "dragging": False,
        "drag_point": None,
    }

    win = "V2 Stage-0 orbit marker — mark 4 points per eye"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DW, DH + 60)

    def on_mouse(ev, x, y, flags, _):
        sx, sy = x / scale, y / scale
        if ev == cv2.EVENT_LBUTTONDOWN:
            existing = marks[state["eye"]]
            hit = _nearest_point(existing, sx, sy, grab_px=30.0 / max(scale, 1e-6))
            if hit is not None:
                state["mode"] = hit
                state["dragging"] = True
                state["drag_point"] = hit
                existing[hit] = (sx, sy)
            else:
                existing[state["mode"]] = (sx, sy)
                state["dragging"] = True
                state["drag_point"] = state["mode"]
            print(f"  [stage0] eye={state['eye']} point={state['mode']} "
                  f"image=({sx:.0f},{sy:.0f})")
        elif ev == cv2.EVENT_MOUSEMOVE and state["dragging"] and state["drag_point"]:
            marks[state["eye"]][state["drag_point"]] = (sx, sy)
        elif ev == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            state["drag_point"] = None

    cv2.setMouseCallback(win, on_mouse)
    print("V2 Stage-0 orbit marker:")
    print(f"  video      : {video_path.name}")
    print(f"  init frame : {init_frame_number}")
    print(f"  resolution : {W}x{H} @ {fps:.2f} fps")
    print(f"  image-side L = patient Right, image-side R = patient Left")
    print(f"  CONTROLS:")
    print(f"    1/2/3/4 = mode medial / lateral / upper / lower")
    print(f"    Tab     = cycle modes")
    print(f"    e       = switch eye (L <-> R)")
    print(f"    r       = reset marks for current eye")
    print(f"    Enter   = save and exit (all 8 points required)")
    print(f"    q / Esc = quit without saving")
    if suggested_eye_zones:
        print(f"  MediaPipe HINT zones (faint blue rectangles): "
              f"L={suggested_eye_zones.get('L')} R={suggested_eye_zones.get('R')}")
    else:
        print(f"  MediaPipe found no face in the first {scan_window_frames} frames; "
              "no hint zones drawn.")

    while True:
        view = cv2.resize(frame_bgr, (DW, DH)) if scale != 1.0 else frame_bgr.copy()
        # Faint MediaPipe hint rectangles (if any).
        if suggested_eye_zones:
            for ek in ("L", "R"):
                bb = suggested_eye_zones.get(ek)
                if bb is None:
                    continue
                x0, y0, x1, y1 = bb
                p0 = (int(x0 * scale), int(y0 * scale))
                p1 = (int(x1 * scale), int(y1 * scale))
                cv2.rectangle(view, p0, p1, COL_HINT, 1, cv2.LINE_AA)
                lbl = f"hint {ek}"
                cv2.putText(view, lbl, (p0[0] + 4, max(15, p0[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_HINT, 1, cv2.LINE_AA)
        # Marks per eye.
        for ek, pts in marks.items():
            for name, (px, py) in pts.items():
                col = POINT_COLOUR[name]
                p = (int(px * scale), int(py * scale))
                cv2.circle(view, p, 5, col, -1, cv2.LINE_AA)
                cv2.circle(view, p, 7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(view, POINT_LABEL[name], (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, POINT_LABEL[name], (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
            # Live oval if all 4 present.
            e = _fit_ellipse_through_four(pts)
            if e is not None:
                (cx, cy), (a, b), theta = e
                cv2.ellipse(view,
                            (int(cx * scale), int(cy * scale)),
                            (max(2, int(a * scale)), max(2, int(b * scale))),
                            theta, 0, 360,
                            COL_OVAL_LIVE if ek == state["eye"] else (140, 140, 0),
                            2 if ek == state["eye"] else 1, cv2.LINE_AA)
        # HUD.
        canvas = np.zeros((DH + 60, DW, 3), np.uint8)
        canvas[:DH] = view
        cv2.rectangle(canvas, (0, DH), (DW, DH + 60), COL_HUD_BG, -1)
        n_L = len(marks["L"]); n_R = len(marks["R"])
        line1 = (f"Active eye: {state['eye']} ({'patient Right' if state['eye']=='L' else 'patient Left'})  "
                 f"Mode: {state['mode'].upper()}  "
                 f"L marks: {n_L}/4   R marks: {n_R}/4")
        line2 = ("1/2/3/4 = mode  Tab cycles  e switch eye  r reset eye  "
                 "Enter = SAVE (need 4+4)  q quit")
        cv2.putText(canvas, line1, (8, DH + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COL_HUD_FG, 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (8, DH + 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, COL_HUD_FG, 1, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(16) & 0xFF

        if k == ord("q") or k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Stage 0 cancelled — nothing saved.")
        if k == 13 or k == 10:           # Enter
            if len(marks["L"]) == 4 and len(marks["R"]) == 4:
                break
            print(f"  [stage0] cannot save yet: L has {len(marks['L'])}/4, "
                  f"R has {len(marks['R'])}/4.")
        elif k == ord("1"):
            state["mode"] = "medial"
        elif k == ord("2"):
            state["mode"] = "lateral"
        elif k == ord("3"):
            state["mode"] = "upper"
        elif k == ord("4"):
            state["mode"] = "lower"
        elif k == 9:                     # Tab
            idx = POINTS.index(state["mode"])
            state["mode"] = POINTS[(idx + 1) % len(POINTS)]
        elif k == ord("e"):
            state["eye"] = "R" if state["eye"] == "L" else "L"
            print(f"  [stage0] active eye = {state['eye']}")
        elif k == ord("r"):
            marks[state["eye"]] = {}
            print(f"  [stage0] reset eye {state['eye']}")

    cv2.destroyAllWindows()

    def _eye_block(ek: str) -> Dict:
        return {
            "patient_eye": "Right" if ek == "L" else "Left",
            "approved": True,
            "oval_points": {
                name: {"x": round(float(marks[ek][name][0]), 2),
                        "y": round(float(marks[ek][name][1]), 2)}
                for name in POINTS
            },
        }

    out = {
        "schema": "v2_orbit_stage0",
        "schema_version": "1.0",
        "video": str(video_path),
        "resolution": [resolution[0], resolution[1]],
        "fps": round(float(fps), 4),
        "init_frame_number": int(init_frame_number),
        "approved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "image_side_convention": "image_L_equals_patient_Right",
        "eyes": {ek: _eye_block(ek) for ek in ("L", "R")},
        "mediapipe": {
            "used_for": "stage0_suggestion_only",
            "not_used_for": [
                "orbit_tracking",
                "iris_radius",
                "iris_centre",
                "limbus",
                "pupil",
                "clinical_tracking",
                "nystagmus",
            ],
            "suggested_eye_zones": (
                {ek: list(bb) for ek, bb in suggested_eye_zones.items()}
                if suggested_eye_zones else None
            ),
            "scan_window_frames": int(scan_window_frames),
            "detected_at_frame": int(detected_at_frame) if detected_at_frame else None,
        },
    }
    return out


def save_stage0_json(stage0: Dict, path: Path) -> None:
    """Convenience: write a Stage-0 dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stage0, indent=2), encoding="utf-8")
    print(f"[stage0] saved {path}")
