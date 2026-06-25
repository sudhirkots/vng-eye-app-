"""Stable pupil tracking pipeline (see STABLE_TRACKING.md).

    python pupil_tracker.py <video> [--output-dir outputs] [--auto-confirm] [--review]

Default: interactive pupil-CIRCLE editor (drag centre, resize radius, per eye),
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
from src.core.pupil_tracking import (FaceLandmarkTracker, PupilDetector, StableTracker, Track,
                                     frame_status, select_init_frame)

# Explicit display de-shimmer via a DEADBAND: hold the marker while it moves less than
# DEADBAND_PX (kills steady-eye jitter) but snap to it immediately on any larger, real movement
# (no lag — unlike smoothing). Display/traces only; the CSV keeps the RAW values.
DEADBAND_PX = 6.0


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


# ----------------------------------------------------------------------------- circle editor
def edit_circles_gui(frame, le, re):
    """Let the user drag the centre and resize the radius of each pupil circle.
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

    print("Edit pupil circles: drag=centre, wheel/[ ]=radius, r=reset, x=disable eye, Enter=accept, q=cancel.")
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
             "inner_canthus_R": "IcR", "inner_canthus_L": "IcL"}


def propose_init(video_path, output_dir="outputs"):
    """Mark what the software thinks are the pupils AND the facial landmarks, and
    save a review image so the user can confirm/correct before tracking (anatomical
    confirmation, step 1). Pupil and iris are drawn as concentric circles so the user
    can verify concentricity; pupil darkness contrast is reported."""
    video_path = Path(video_path)
    detector = PupilDetector()
    init = select_init_frame(detector, video_path)
    if init is None:
        raise SystemExit("No frame with a confidently detected eye — cannot initialise.")
    fr, frame, le, re = init
    # fresh detector: select_init_frame advanced this detector's MediaPipe tracking state, so
    # re-processing the (earlier) init frame with it can fail — a clean instance detects reliably.
    faces = PupilDetector().face_landmarks(frame) or {}
    out_dir = Path(output_dir) / (video_path.stem + "_tracked")
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = frame.copy()
    # concentric iris (cyan, thin) + pupil (green) so concentricity is visible
    for e in (le, re):
        if e.detected:
            c = (int(e.x), int(e.y))
            if e.iris_radius:
                cv2.circle(vis, c, int(e.iris_radius), CYAN, 1)
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
    cv2.putText(full, f"Init frame {fr} - confirm pupils (green) + face landmarks (amber)",
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
        if e.iris_radius:
            cv2.circle(big, (cx, cy), int(e.iris_radius * 4), CYAN, 1)   # iris (concentric)
        cv2.circle(big, (cx, cy), int(e.radius * 4), GREEN, 2)           # pupil
        cv2.circle(big, (cx, cy), 3, GREEN, -1)
        cv2.putText(big, f"{key} pupil r={e.radius:.0f} contrast={e.pupil_contrast:.2f}",
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

    print(f"Init frame {fr}: (concentricity = pupil should sit at the iris centre; contrast = dark pupil)")
    for key, e in (("L", le), ("R", re)):
        print(f"  pupil {key}: " + ("not detected" if not e.detected else
              f"centre=({e.x:.0f},{e.y:.0f}) pupil_r={e.radius:.0f} iris_r={e.iris_radius:.0f} "
              f"contrast={e.pupil_contrast:.2f}" + ("  [LOW CONTRAST — check]" if e.pupil_contrast < 0.08 else "")))
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


def approve_interactive(frame, le, re, faces):
    """Interactive Stage 0 confirmation screen. Shows the frame with the proposed
    pupils (green circles) and facial landmarks (amber dots), labelled. The user
    drags any point to the correct spot, resizes a pupil with the wheel or [ ],
    toggles a point off/on with 'd' (unavailable), then presses Enter to APPROVE.
    A magnifier follows the cursor for precise placement. Returns (pupils, faces)."""
    H, W = frame.shape[:2]
    scale = min(1.0, 1200.0 / W)
    DW, DH = int(W * scale), int(H * scale)

    items = []
    for key, e, dx in (("L", le, 0.35), ("R", re, 0.65)):
        if e.detected:
            items.append({"kind": "pupil", "name": key, "x": e.x, "y": e.y, "r": max(4.0, e.radius), "on": True})
        else:
            items.append({"kind": "pupil", "name": key, "x": W * dx, "y": H * 0.5, "r": 15.0, "on": False})
    for name, pt in faces.items():
        on = pt is not None
        items.append({"kind": "face", "name": name,
                      "x": pt[0] if on else W * 0.5, "y": pt[1] if on else H * 0.5, "r": None, "on": on})

    st_ = {"sel": 0, "drag": False, "mx": 0, "my": 0}

    def nearest_center(ix, iy):
        best = None
        for i, it in enumerate(items):
            d = np.hypot(it["x"] - ix, it["y"] - iy)
            if best is None or d < best[0]:
                best = (d, i)
        return best[1] if best and best[0] < 45 / scale else None

    def on_mouse(ev, x, y, flags, _):
        # mouse ONLY moves a point (drag); radius is changed with + / - keys
        st_["mx"], st_["my"] = x, y
        ix, iy = x / scale, y / scale
        if ev == cv2.EVENT_LBUTTONDOWN:
            i = nearest_center(ix, iy)
            if i is not None:
                st_["sel"], st_["drag"] = i, True
                items[i]["on"] = True
        elif ev == cv2.EVENT_MOUSEMOVE and st_["drag"]:
            items[st_["sel"]].update(x=ix, y=iy)
        elif ev == cv2.EVENT_LBUTTONUP:
            st_["drag"] = False

    win = "Stage 0 - confirm landmarks"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = cv2.resize(frame, (DW, DH))
        for i, it in enumerate(items):
            sel = (i == st_["sel"])
            col = (0, 255, 255) if sel else ((0, 200, 0) if it["kind"] == "pupil" else (0, 165, 255))
            if not it["on"]:
                col = (130, 130, 130)
            p = (int(it["x"] * scale), int(it["y"] * scale))
            lbl = (f"{it['name']} r{int(it['r'])}" if it["kind"] == "pupil"
                   else FACE_CODE.get(it["name"], it["name"]))
            if not it["on"]:
                lbl += " OFF"
            if it["kind"] == "pupil":
                cv2.circle(disp, p, max(3, int((it["r"] or 10) * scale)), col, 2)
                cv2.circle(disp, p, 2, col, -1)
            else:
                cv2.circle(disp, p, 4, col, -1)
            cv2.putText(disp, lbl, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        # magnifier inset (top-right) around the cursor for precise placement
        ix, iy = int(st_["mx"] / scale), int(st_["my"] / scale)
        Z, rad = 4, 40
        x0, y0 = max(0, ix - rad), max(0, iy - rad)
        x1, y1 = min(W, ix + rad), min(H, iy + rad)
        crop = frame[y0:y1, x0:x1]
        if crop.size:
            mag = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(mag, (int((ix - x0) * Z), int((iy - y0) * Z)), (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
            mh, mw = mag.shape[:2]
            if mw <= DW and mh <= DH:
                cv2.rectangle(mag, (0, 0), (mw - 1, mh - 1), (0, 0, 255), 1)
                disp[0:mh, DW - mw:DW] = mag
        sel_it = items[st_["sel"]]
        sel_lbl = sel_it["name"] if sel_it["kind"] == "pupil" else FACE_CODE.get(sel_it["name"], sel_it["name"])
        cv2.putText(disp, f"selected: {sel_lbl}   drag = move pupil   + / - = pupil size"
                    "   d = off/on   Enter = APPROVE   q = cancel",
                    (8, DH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):
            break
        if k == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled — nothing approved.")
        if k in (ord("-"), ord("+"), ord("=")):
            it = items[st_["sel"]]
            if it["kind"] == "pupil":
                it["r"] = max(3.0, it["r"] + (1 if k in (ord("+"), ord("=")) else -1))
        if k == ord("d"):
            items[st_["sel"]]["on"] = not items[st_["sel"]]["on"]
    cv2.destroyAllWindows()
    pupils = {it["name"]: ((it["x"], it["y"], it["r"]) if it["on"] else None)
              for it in items if it["kind"] == "pupil"}
    faces_out = {it["name"]: ((it["x"], it["y"]) if it["on"] else None)
                 for it in items if it["kind"] == "face"}
    return pupils, faces_out


def approve(video_path, output_dir="outputs", auto=False):
    """STAGE 0 — propose landmarks, let the user review/correct, then APPROVE and
    save approved_landmarks.json. Tracking refuses to run until this exists.
    auto=True accepts the proposal without a GUI (headless)."""
    video_path = Path(video_path)
    detector = PupilDetector()
    init = select_init_frame(detector, video_path)
    if init is None:
        raise SystemExit("No frame with a confidently detected eye — cannot initialise.")
    fr, frame, le, re = init
    # fresh detector: select_init_frame advanced this detector's MediaPipe tracking state, so
    # re-processing the (earlier) init frame with it can fail — a clean instance detects reliably.
    faces = PupilDetector().face_landmarks(frame) or {}
    detected = {"L": (le.x, le.y, le.radius) if le.detected else None,
                "R": (re.x, re.y, re.radius) if re.detected else None}

    if auto:
        confirmed = detected
        faces_out = faces
        print("--auto-approve: accepting the proposed pupils + face landmarks without review.")
    else:
        print("Stage 0: drag points to correct them, wheel/[ ] resize pupils, d=off/on, Enter=APPROVE.")
        confirmed, faces_out = approve_interactive(frame, le, re, faces)
    if confirmed["L"] is None and confirmed["R"] is None:
        raise SystemExit("No usable eye approved.")

    cap = cv2.VideoCapture(str(video_path))
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 30.0
    cap.release()
    out_dir = Path(output_dir) / (video_path.stem + "_tracked")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "video": str(video_path), "resolution": f"{w}x{h}", "fps": round(fps, 2),
        "init_frame_number": fr, "approved": True,
        "approved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "pupils": {k: ({"x": round(v[0], 2), "y": round(v[1], 2), "radius": round(v[2], 2)} if v else None)
                   for k, v in confirmed.items()},
        "face_landmarks": {name: ({"x": round(p[0], 2), "y": round(p[1], 2)} if p else None)
                           for name, p in faces_out.items()},
    }
    path = out_dir / "approved_landmarks.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"APPROVED — saved {path}")
    print(f'Now run tracking:  python app.py --video "{video_path}"')
    return path


# ----------------------------------------------------------------------------- overlay & plots
def draw_overlay(frame, scale, lt: Track, rt: Track, fstatus, fr, t, face_tracks=None, raw_pupils=None):
    """lt/rt carry the FILTERED pupil centres (main markers). raw_pupils = [(x,y),...]
    are drawn as small faint markers so the raw vs filtered difference is visible."""
    ow, oh = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
    vis = cv2.resize(frame, (ow, oh))
    cv2.rectangle(vis, (0, 0), (ow - 1, oh - 1), STATUS_COLOR.get(fstatus, YELLOW), 5)
    # tracked facial landmarks (small dots, coloured by their own status)
    for tr in (face_tracks or {}).values():
        if tr.x is not None:
            cv2.circle(vis, (int(tr.x * scale), int(tr.y * scale)), 3,
                       STATUS_COLOR.get(tr.status, AMBER), -1)
    # faint RAW pupil markers (so you can see what the filter is removing)
    for rp in (raw_pupils or []):
        if rp and rp[0] is not None:
            cv2.drawMarker(vis, (int(rp[0] * scale), int(rp[1] * scale)), (170, 170, 170),
                           cv2.MARKER_CROSS, 9, 1)
    for tr in (lt, rt):
        col = STATUS_COLOR.get(tr.status, YELLOW)
        if tr.x is not None:
            p = (int(tr.x * scale), int(tr.y * scale))
            rad = int((tr.radius or 10) * scale)
            cv2.circle(vis, p, max(3, rad), col, 2)      # tracked pupil CIRCLE
            cv2.circle(vis, p, 2, col, -1)               # centre point
        if tr.rejected and tr.raw_x is not None:
            cv2.drawMarker(vis, (int(tr.raw_x * scale), int(tr.raw_y * scale)), RED,
                           cv2.MARKER_TILTED_CROSS, 12, 2)
    lines = [f"frame {fr}  t={t:.2f}s", f"L:{lt.status}  R:{rt.status}", fstatus]
    for i, txt in enumerate(lines):
        y = 20 + i * 22
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, STATUS_COLOR.get(fstatus, YELLOW), 1)
    return vis


def plot_trace(times, series, title, path, w=1100, h=420, flags=None):
    """Simple OpenCV line plot (no matplotlib dependency). series = {label:(vals,color)};
    vals may contain None (gaps left blank — blink/lost are not interpolated). flags = optional
    list of bools per time index; True draws a red tick at the bottom (e.g. poor-quality frame)."""
    img = 255 * np.ones((h, w, 3), np.uint8)
    ml, mr, mt, mb = 70, 20, 40, 50
    pw, ph = w - ml - mr, h - mt - mb
    allv = [v for vals, _ in series.values() for v in vals if v is not None]
    if not allv or not times:
        cv2.imwrite(str(path), img); return
    ymin, ymax = min(allv), max(allv)
    if ymax - ymin < 1:
        ymax, ymin = ymax + 1, ymin - 1
    tmax = max(times) or 1.0
    def X(t): return int(ml + pw * t / tmax)
    def Y(v): return int(mt + ph * (1 - (v - ymin) / (ymax - ymin)))
    if flags:
        for t, fl in zip(times, flags):
            if fl:
                cv2.line(img, (X(t), mt + ph), (X(t), mt + ph - 10), (0, 0, 255), 1)
    cv2.rectangle(img, (ml, mt), (ml + pw, mt + ph), (0, 0, 0), 1)
    cv2.putText(img, title, (ml, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"{ymax:.0f}", (8, mt + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"{ymin:.0f}", (8, mt + ph), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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
    cv2.putText(img, f"eye-in-socket: pupil vs canthi, horizontal  (L green / R blue)  last {window:.0f}s",
                (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return img


# ----------------------------------------------------------------------------- main run
HEAD_REF_LANDMARKS = ("nose_bridge_mid", "nose_tip", "outer_canthus_L", "outer_canthus_R",
                      "inner_canthus_L", "inner_canthus_R")


class HeadStabilizer:
    """Eye-in-head: remove head motion via an affine transform fit from the head-reference
    landmarks (nose + eye corners) back to their init positions; the pupil expressed in that
    frame is the eye movement. TIERED quality, not hard rejection — only DEGENERATE transforms
    are rejected (too few landmarks, NaN, impossible scale, or an impossible per-frame jump in
    scale/rotation/translation that no real head can make). Everything else is kept with a
    quality flag (good/fair/poor) + residual + landmark count, so the trace stays continuous and
    spikes can be attributed to transform quality vs pupil-tracking error."""

    def __init__(self, init_face, init_pupil, frame_w):
        self.init_face = init_face            # {name: (x, y)}
        self.init_pupil = init_pupil          # {"L": (x, y), "R": (x, y)}
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

    def correct(self, m, pupil_xy, eye):
        ip = self.init_pupil[eye]
        if m is None or pupil_xy[0] is None or ip[0] is None:
            return None, None
        sx = m[0, 0] * pupil_xy[0] + m[0, 1] * pupil_xy[1] + m[0, 2]
        sy = m[1, 0] * pupil_xy[0] + m[1, 1] * pupil_xy[1] + m[1, 2]
        return float(sx - ip[0]), float(sy - ip[1])


def canthus_relative(pupil, inner, outer):
    """Eye-in-socket position: the pupil expressed relative to this eye's own medial (inner) and
    lateral (outer) canthus. Origin = the canthi midpoint; horizontal axis = inner→outer corner
    (so it follows head tilt); h = displacement along that axis, v = perpendicular, in pixels.
    Because the canthi move WITH the head, this cancels head/"hair" movement locally and leaves only
    the pupil's movement within the eye — the nystagmus. Returns (h, v) or (None, None)."""
    if pupil[0] is None or inner is None or outer is None:
        return None, None
    inner = np.array(inner, float); outer = np.array(outer, float); p = np.array(pupil, float)
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


def run(video_path, output_dir="outputs", max_debug_frames=80, filter_mode="adaptive", show_raw=True):
    video_path = Path(video_path)
    out_dir = Path(output_dir) / (video_path.stem + "_tracked")

    # --- STAGE 0 GATE: tracking must not begin until landmarks are approved ---
    approved = load_approved(output_dir, video_path)
    if not approved or not approved.get("approved"):
        raise SystemExit(
            'Stage 0 not complete — landmarks not approved. Approve first:\n'
            f'    python app.py --video "{video_path}" --approve')
    init_fr = int(approved["init_frame_number"])
    Lp, Rp = approved["pupils"]["L"], approved["pupils"]["R"]
    L = (Lp["x"], Lp["y"], Lp["radius"]) if Lp else None
    R = (Rp["x"], Rp["y"], Rp["radius"]) if Rp else None
    if not L and not R:
        raise SystemExit("Approved landmarks contain no usable eye.")
    confirmed = {"L": L, "R": R}
    interocular = abs(R[0] - L[0]) if (L and R) else (L or R)[2] * 6.0
    print(f"Tracking from approved_landmarks.json (init frame {init_fr}).")

    detector = PupilDetector()
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
    tracker = StableTracker(init_gray, (L[0], L[1]) if L else None, (R[0], R[1]) if R else None,
                            L[2] if L else 0.0, R[2] if R else 0.0, interocular)
    face_appr = approved.get("face_landmarks") or {}
    init_face_mp = (detector.face_landmarks(init_frame) or {}) if (face_appr and ok) else {}
    face_tracker = FaceLandmarkTracker(face_appr, init_face_mp) if face_appr else None
    face_names = face_tracker.names() if face_tracker else []

    # jitter filters: 3-frame median + 1€ (adaptive). Reset on lost/blink (never smooth a gap).
    pf = {"L": PointFilter(filter_mode, fps), "R": PointFilter(filter_mode, fps)}
    ff = {n: PointFilter(filter_mode, fps) for n in face_names}
    print(f"Filter: {filter_mode}")
    scale = min(1.0, OVERLAY_MAX_W / width)
    ow, oh = int(width * scale), int(height * scale)
    panel_h = 200                            # scrolling eye-in-socket trace strip under the video
    writer = cv2.VideoWriter(str(out_dir / "tracking_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh + panel_h))

    csv_f = (out_dir / "tracking.csv").open("w", newline="", encoding="utf-8")
    cw = csv.writer(csv_f)
    cw.writerow(["frame_number", "timestamp_ms", "time_sec",
                 "raw_left_pupil_x", "raw_left_pupil_y", "raw_right_pupil_x", "raw_right_pupil_y",
                 "filtered_left_pupil_x", "filtered_left_pupil_y",
                 "filtered_right_pupil_x", "filtered_right_pupil_y",
                 "corrected_left_eye_h", "corrected_left_eye_v",
                 "corrected_right_eye_h", "corrected_right_eye_v",
                 "transform_quality", "transform_residual_px", "landmark_count_used",
                 "left_pupil_radius", "right_pupil_radius",
                 "tracking_status_left", "tracking_status_right",
                 "confidence_left", "confidence_right"])

    face_csv_f = fcw = None
    if face_tracker:
        face_csv_f = (out_dir / "face_landmarks.csv").open("w", newline="", encoding="utf-8")
        fcw = csv.writer(face_csv_f)
        hdr = ["frame_number", "time_sec"]
        for n in face_names:
            hdr += [f"{n}_x", f"{n}_y", f"{n}_status"]
        fcw.writerow(hdr)

    # head-motion correction reference (the user-approved init landmarks/pupils)
    init_face_pts = {n: (v["x"], v["y"]) for n, v in face_appr.items() if v}
    init_pupil = {"L": (L[0], L[1]) if L else (None, None), "R": (R[0], R[1]) if R else (None, None)}
    stabilizer = HeadStabilizer(init_face_pts, init_pupil, width)

    times = []
    rlx, rly, rrx, rry = [], [], [], []      # RAW left/right x/y (image space)
    flx, fly, frx, fry = [], [], [], []      # FILTERED left/right x/y (image space)
    cor_lh, cor_lv, cor_rh, cor_rv = [], [], [], []   # RAW head-corrected eye-in-head (not over-filtered)
    poor_flags = []                          # True where transform quality is poor/degenerate
    q_counts = {}                            # transform quality tally
    tracked_series = []
    counts = {}
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
            if "r" not in cache:
                cache["r"] = detector.detect(frame)
            return cache["r"]

        if fr < init_fr:
            lt = Track("lost", None, None)
            rt = Track("lost", None, None)
        elif fr == init_fr:
            lt = Track("initialized", L[0], L[1], L[2], "high") if L else Track("lost", None, None)
            rt = Track("initialized", R[0], R[1], R[2], "high") if R else Track("lost", None, None)
        else:
            lt, rt = tracker.step(gray, mp_detect)
        fstatus = frame_status(lt, rt)
        counts[fstatus] = counts.get(fstatus, 0) + 1

        # facial landmarks first — they define the head reference frame for correction
        face_tracks = {}
        if face_tracker and fr >= init_fr:
            def face_backup():
                if "f" not in cache:
                    cache["f"] = detector.face_landmarks(frame) or {}
                return cache["f"]
            face_tracks = face_tracker.step(gray, face_backup)
        cur_face = {n: (ftr.x, ftr.y) for n, ftr in face_tracks.items() if ftr.x is not None}

        # FILTERED image-space pupil (for the image-space trace); raw kept separately.
        flx_, fly_ = pf["L"](lt.x, lt.y, t)
        frx_, fry_ = pf["R"](rt.x, rt.y, t)
        # HEAD-CORRECTED eye-in-head from the RAW pupil. Tiered transform quality: only degenerate
        # transforms are rejected (gap); imperfect ones are kept + flagged. Corrected trace is NOT
        # over-filtered (raw corrected) so nystagmus beats are preserved.
        def _pt(name):
            tr = face_tracks.get(name)
            return (tr.x, tr.y) if tr and tr.x is not None else None

        def _near(pupil, kind):     # pair the pupil with ITS OWN eye's canthus (nearest one)
            cands = [c for c in (_pt(kind + "_canthus_L"), _pt(kind + "_canthus_R")) if c]
            if not cands or pupil[0] is None:
                return None
            return min(cands, key=lambda c: (c[0] - pupil[0]) ** 2 + (c[1] - pupil[1]) ** 2)

        clh, clv = canthus_relative((lt.x, lt.y), _near((lt.x, lt.y), "inner"), _near((lt.x, lt.y), "outer"))
        crh, crv = canthus_relative((rt.x, rt.y), _near((rt.x, rt.y), "inner"), _near((rt.x, rt.y), "outer"))
        n_used = sum(_pt(n) is not None for n in ("inner_canthus_L", "outer_canthus_L",
                                                  "inner_canthus_R", "outer_canthus_R"))
        quality = "good" if (clh is not None and crh is not None) else \
                  ("fair" if (clh is not None or crh is not None) else "degenerate")
        residual = None
        q_counts[quality] = q_counts.get(quality, 0) + 1
        poor_flags.append(quality == "degenerate")

        times.append(round(t, 4))
        rlx.append(lt.x); rly.append(lt.y); rrx.append(rt.x); rry.append(rt.y)
        flx.append(flx_); fly.append(fly_); frx.append(frx_); fry.append(fry_)
        cor_lh.append(clh); cor_lv.append(clv); cor_rh.append(crh); cor_rv.append(crv)
        tracked_series.append((flx_, fly_) if flx_ is not None else None)

        cw.writerow([fr, int(round(t * 1000)), round(t, 4),
                     _r(lt.x), _r(lt.y), _r(rt.x), _r(rt.y),
                     _r(flx_), _r(fly_), _r(frx_), _r(fry_),
                     _r(clh), _r(clv), _r(crh), _r(crv),
                     quality, _r(residual), n_used,
                     _r(lt.radius), _r(rt.radius),
                     lt.status, rt.status, lt.confidence, rt.confidence])

        if fcw:
            row = [fr, round(t, 4)]
            for n in face_names:
                ftr = face_tracks.get(n)
                if ftr and ftr.x is not None:
                    row += [_r(ftr.x), _r(ftr.y), ftr.status]
                else:
                    row += ["", "", ftr.status if ftr else "lost"]
            fcw.writerow(row)

        # OVERLAY = RAW detected positions (verification: the marker must sit ON the pupil with no
        # lag). The FILTERED signal is used only for the VNG trace + CSV, where shimmer matters and a
        # small filter lag is harmless. This keeps the overlay honest and never sliding.
        vis = draw_overlay(frame, scale, lt, rt, fstatus, fr, t, face_tracks)
        panel = render_trace_panel(times, cor_lh, cor_rh, t, ow, height=panel_h)
        writer.write(np.vstack([vis, panel]))
        if fstatus in ("uncertain", "blink_or_occluded", "lost") and fr >= init_fr:
            seen_bad += 1
            if seen_bad % stride == 0 and saved_bad < max_debug_frames:
                cv2.imwrite(str(dbg_dir / f"{fr:06d}_{fstatus}.png"), vis)
                saved_bad += 1
    cap.release(); writer.release(); csv_f.close()
    if face_csv_f:
        face_csv_f.close()

    # H/V traces: FILTERED as the main line; raw faint in the background if --show-raw-trace
    GRAY = (170, 170, 170)
    h_series = {"L filt": (flx, GREEN), "R filt": (frx, BLUE)}
    v_series = {"L filt": (fly, GREEN), "R filt": (fry, BLUE)}
    if show_raw:
        h_series = {"L raw": (rlx, GRAY), "R raw": (rrx, GRAY), **h_series}
        v_series = {"L raw": (rly, GRAY), "R raw": (rry, GRAY), **v_series}
    plot_trace(times, h_series, "Horizontal pupil position (px) vs time", out_dir / "trace_horizontal.png")
    plot_trace(times, v_series, "Vertical pupil position (px) vs time", out_dir / "trace_vertical.png")
    # verification: raw vs filtered, left-eye horizontal (confirm shimmer down, fast moves preserved)
    plot_trace(times, {"raw": (rlx, GRAY), "filtered": (flx, GREEN)},
               f"Left pupil horizontal - RAW vs FILTERED ({filter_mode})",
               out_dir / "trace_raw_vs_filtered.png")
    # HEAD-CORRECTED eye-in-head traces (raw corrected — not over-filtered; red ticks = poor/degenerate)
    plot_trace(times, {"L": (cor_lh, GREEN), "R": (cor_rh, BLUE)},
               "Eye-in-socket (pupil vs canthi) - HORIZONTAL (px) vs time",
               out_dir / "trace_corrected_h.png", flags=poor_flags)
    plot_trace(times, {"L": (cor_lv, GREEN), "R": (cor_rv, BLUE)},
               "Eye-in-socket (pupil vs canthi) - VERTICAL (px) vs time",
               out_dir / "trace_corrected_v.png", flags=poor_flags)

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
        "manual_corrections": tracker.corrections,
        "status_counts": counts,
        "filter": filter_mode,
        "jitter_left_raw_px": jitter(rlx, rly),
        "jitter_left_filtered_px": jitter(flx, fly),
        "transform_quality_counts": q_counts,
        "face_landmarks_tracked": face_names,
        "outputs": {"overlay": str(out_dir / "tracking_overlay.mp4"),
                    "csv": str(out_dir / "tracking.csv"),
                    "face_landmarks_csv": (str(out_dir / "face_landmarks.csv") if face_tracker else None),
                    "trace_horizontal": str(out_dir / "trace_horizontal.png"),
                    "trace_vertical": str(out_dir / "trace_vertical.png"),
                    "trace_raw_vs_filtered": str(out_dir / "trace_raw_vs_filtered.png"),
                    "trace_corrected_h": str(out_dir / "trace_corrected_h.png"),
                    "trace_corrected_v": str(out_dir / "trace_corrected_v.png")},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return metadata


def _r(v):
    return round(v, 2) if v is not None else ""


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
    ap = argparse.ArgumentParser(description="Stable pupil tracking")
    ap.add_argument("video")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--propose", action="store_true",
                    help="Stage 0 preview: mark proposed pupils + face landmarks on the init frame")
    ap.add_argument("--approve", action="store_true",
                    help="Stage 0: review/correct then approve landmarks (saves approved_landmarks.json)")
    ap.add_argument("--auto-approve", action="store_true",
                    help="Stage 0 headless: approve the proposal without a GUI")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--filter", choices=["none", "light", "adaptive"], default="adaptive",
                    help="pupil jitter filter (default adaptive 1€)")
    ap.add_argument("--show-raw-trace", action="store_true", help="also draw the raw signal faint")
    args = ap.parse_args()
    if args.propose:
        propose_init(args.video, args.output_dir)
    elif args.approve or args.auto_approve:
        approve(args.video, args.output_dir, auto=args.auto_approve)
    elif args.review:
        review(args.video, args.output_dir)
    else:
        run(args.video, args.output_dir, filter_mode=args.filter, show_raw=args.show_raw_trace)


if __name__ == "__main__":
    main()
