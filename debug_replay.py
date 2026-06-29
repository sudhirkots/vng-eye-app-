"""Debug → Tracking Replay (EYEVNG V1).

The primary visual debugging tool for the V1 tracker. It re-runs the V1 orchestrator on a clip and
shows a zoomed view of the tracked eye with the clinical anatomy drawn prominently:

  * VISIBLE LIMBUS ARC                — the iris-sclera boundary actually detected this frame
  * ESTIMATED FULL IRIS CIRCLE/ELLIPSE — the disc completed from the visible arc (clinical object)
  * IRIS-CIRCLE CENTRE                — the clinical iris centre (cross marker)
  * EYE-OPENING CONTOUR               — the moving reference frame, with derived M/L/U/D extents

Optional CFT helper features, when present, are drawn as very faint dots — clearly subordinate.

Usage:
    python debug_replay.py --video samples/2.mp4 [--output-dir outputs] [--eye auto|left|right]
                           [--zoom 3.5] [--live] [--start N] [--end N] [--no-mediapipe]
                           [--cft-helper]

Stage 0 is unchanged: the replay refuses to run unless `approved_landmarks.json` already exists for
the clip (the same approval the tracker uses). The replay never re-approves anything.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from iris_tracker import load_approved
from src.core.composite_tracker import TRUSTED, PROBATION, SUSPECT, LOST
from src.core.eye_contour import EyeOpeningContourTracker, contour_from_approved
from src.core.v1_tracker import V1Tracker

GREEN = (0, 200, 0)
YELLOW = (0, 210, 230)
RED = (40, 40, 230)
CYAN = (220, 220, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FEATURE_COLOR = {TRUSTED: (0, 130, 0), PROBATION: (0, 130, 160), SUSPECT: (0, 0, 130),
                 LOST: (0, 0, 130)}

# State → HUD colour (valid states greenish, gaps red/blue).
STATE_COLOR = {
    "TRACKING": GREEN, "REPLENISHING_FEATURES": GREEN, "PARTIAL_OCCLUSION": YELLOW,
    "INITIALIZING": CYAN, "BLINK": ORANGE, "DRIFT_SUSPECTED": ORANGE,
    "TRACK_LOST": RED, "REACQUIRING": ORANGE,
}


def _text(img, s, org, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_frame(frame, trk, fm, contour_state, eye_label, zoom, disp=560, hud_h=152):
    """Build one zoomed replay frame: a `disp`×`disp` zoomed crop centred on the eye + a HUD strip.
    The view is dominated by the limbus arc, the fitted iris ellipse, and the eye-opening contour
    (the V1 clinical anatomy). CFT helper features, if any, are drawn as 1px faint dots."""
    h, w = frame.shape[:2]
    centre = fm.iris_centre if (fm and fm.iris_centre) else trk.last_centre
    radius = (fm.iris_radius if (fm and fm.iris_radius) else trk.last_radius) or 30.0
    cx, cy = float(centre[0]), float(centre[1])

    half = disp / (2.0 * zoom)
    x0, y0 = cx - half, cy - half

    sx0, sy0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
    sx1, sy1 = int(min(w, np.ceil(cx + half))), int(min(h, np.ceil(cy + half)))
    crop = frame[sy0:sy1, sx0:sx1]
    canvas = np.zeros((disp, disp, 3), np.uint8)
    if crop.size:
        z = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        ox, oy = int(round((sx0 - x0) * zoom)), int(round((sy0 - y0) * zoom))
        zh, zw = z.shape[:2]
        dx0, dy0 = max(0, ox), max(0, oy)
        zx0, zy0 = max(0, -ox), max(0, -oy)
        ww, hh = min(disp - dx0, zw - zx0), min(disp - dy0, zh - zy0)
        if ww > 0 and hh > 0:
            canvas[dy0:dy0 + hh, dx0:dx0 + ww] = z[zy0:zy0 + hh, zx0:zx0 + ww]

    def to_disp(px, py):
        return int(round((px - x0) * zoom)), int(round((py - y0) * zoom))

    valid = fm is not None and fm.validity == "valid"

    # ---- eye-opening contour (the V1 reference frame) --------------------------------
    if contour_state is not None:
        col = CYAN if contour_state.reference_valid else ORANGE
        pts = np.asarray(contour_state.points, np.float32)
        last = None
        for px, py in pts:
            dxp, dyp = to_disp(px, py)
            if last is not None:
                cv2.line(canvas, last, (dxp, dyp), col, 1, cv2.LINE_AA)
            last = (dxp, dyp)
        if last is not None:
            first = to_disp(pts[0, 0], pts[0, 1])
            cv2.line(canvas, last, first, col, 1, cv2.LINE_AA)
        # derived extents — diamond markers (NOT independently tracked points).
        ex = contour_state.extents()
        labels = {"medial": "M", "lateral": "L", "upper": "U", "lower": "D"}
        for name, p0 in ex.items():
            p = to_disp(p0[0], p0[1])
            cv2.drawMarker(canvas, p, col, cv2.MARKER_DIAMOND, 10, 1, cv2.LINE_AA)
            _text(canvas, labels[name], (p[0] + 4, p[1] - 4), col, 0.4)

    # ---- visible LIMBUS ARC — the iris-sclera boundary actually detected --------------
    arc_pts = getattr(fm, "limbus_arc_pts", None) if fm is not None else None
    if arc_pts is not None:
        for px, py in arc_pts:
            cv2.circle(canvas, to_disp(px, py), 2, (0, 220, 255), -1, cv2.LINE_AA)

    # ---- estimated full IRIS CIRCLE / ELLIPSE — the clinical object ------------------
    icx, icy = to_disp(cx, cy)
    if fm is not None and fm.limbus_ellipse is not None and valid:
        (ecx, ecy), (MA, ma), ang = fm.limbus_ellipse
        ec = to_disp(ecx, ecy)
        cv2.ellipse(canvas, ec, (int(MA / 2 * zoom), int(ma / 2 * zoom)), ang, 0, 360,
                    GREEN, 2, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (icx, icy), int(radius * zoom),
                   GREEN if valid else (90, 90, 90), 2, cv2.LINE_AA)

    # ---- optional CFT helper features — FAINT 1px dots (V1Tracker.helper.pool) --------
    helper = getattr(trk, "helper", None)
    n_by = {TRUSTED: 0, PROBATION: 0, SUSPECT: 0, LOST: 0}
    if helper is not None:
        for f in helper.pool:
            n_by[f.state] = n_by.get(f.state, 0) + 1
            dxp, dyp = to_disp(f.pos[0], f.pos[1])
            if 0 <= dxp < disp and 0 <= dyp < disp:
                cv2.circle(canvas, (dxp, dyp), 1, FEATURE_COLOR.get(f.state, RED), -1)

    # ---- iris-circle CENTRE — the clinical iris centre marker -------------------------
    cc_col = GREEN if valid else RED
    cv2.drawMarker(canvas, (icx, icy), cc_col, cv2.MARKER_CROSS, 18, 2)

    # ---- HUD strip below ----
    out = np.zeros((disp + hud_h, disp, 3), np.uint8)
    out[:disp] = canvas
    st = fm.state if fm else "INITIALIZING"
    _text(out, f"Eye: {eye_label}   frame {fm.frame_number if fm else 0:>5}   "
               f"t={fm.time_sec if fm else 0:6.2f}s",
          (10, disp + 22), WHITE, 0.52)
    _text(out, f"state: {st}", (10, disp + 48), STATE_COLOR.get(st, WHITE), 0.6, 2)
    if fm and fm.drift_reason:
        _text(out, f"drift: {fm.drift_reason}", (270, disp + 48), ORANGE, 0.5)
    lc = getattr(fm, "limbus_inlier_fraction", 0.0) if fm else 0.0
    cov = getattr(fm, "limbus_arc_coverage", 0.0) if fm else 0.0
    occ = "*" if (fm and getattr(fm, "limbus_radius_fixed", False)) else ""
    _text(out, f"limbus inlier {lc:4.2f}   arc cov {cov:4.2f}{occ}",
          (10, disp + 74), GREEN if lc >= 0.45 else YELLOW, 0.5)
    rc = getattr(fm, "reference_confidence", 0.0) if fm else 0.0
    rs = getattr(fm, "reference_state", "lost") if fm else "lost"
    ref_col = GREEN if (fm and fm.reference_valid) else (YELLOW if rc > 0.2 else RED)
    _text(out, f"contour ref: {rs}  conf {rc:4.2f}", (10, disp + 98), ref_col, 0.5)
    fconf = fm.frame_confidence if fm else 0.0
    _text(out, f"FRAME conf {fconf:4.2f}", (10, disp + 124),
          GREEN if fconf >= .5 else (YELLOW if fconf >= .2 else RED), 0.55, 2)
    ear = fm.ear if (fm and fm.ear is not None) else None
    _text(out, f"EAR {ear:4.2f}" if ear is not None else "EAR  -", (240, disp + 124), WHITE, 0.5)
    if helper is not None:
        _text(out, f"CFT helper  T{n_by[TRUSTED]:2d} P{n_by[PROBATION]:2d} "
                   f"L{n_by[SUSPECT] + n_by[LOST]:2d}",
              (360, disp + 124), WHITE, 0.42)
    return out


def pick_eye(approved, want):
    """Pick the approved eye for the debug replay.

    V1 RULE: only the V1 'iris' block is read here. The legacy `pupils` block is no longer
    accepted as a silent fallback — it seeds pupil-sized radii (5-9 px) where V1 needs full
    iris/limbus radii (15-25 px), which broke a known-positive nystagmus clip in testing.
    Legacy approvals must be re-approved through `python app.py --video ... --approve`.
    """
    if approved.get("approval_schema_version") != "v1_iris_limbus_and_eye_contour":
        raise SystemExit(
            "Invalid Stage 0 approval for eye_vng V1. V1 requires the V1 approval schema. "
            "Please rerun --approve and mark the full iris/limbus circle and the "
            "eye-opening contour.")
    iris = approved.get("iris") or {}
    have = {k: v for k, v in iris.items() if v}
    if want == "left" and "L" in have:
        return "L", have["L"]
    if want == "right" and "R" in have:
        return "R", have["R"]
    for k in ("L", "R"):
        if k in have:
            return k, have[k]
    raise SystemExit("No approved iris in approved_landmarks.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--eye", choices=["auto", "left", "right"], default="auto")
    ap.add_argument("--zoom", type=float, default=3.5)
    ap.add_argument("--live", action="store_true",
                    help="open an interactive window (else write mp4 only)")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--no-mediapipe", action="store_true",
                    help="run from approved landmarks without MediaPipe (recovery only)")
    ap.add_argument("--cft-helper", action="store_true",
                    help="enable the CFT helper (faint dots in the view)")
    args = ap.parse_args()

    video = Path(args.video)
    approved = load_approved(args.output_dir, video)
    if not approved or not approved.get("approved"):
        raise SystemExit(f"No approved_landmarks.json for {video.name} — run Stage 0 approval first.")
    eye_key, iris = pick_eye(approved, args.eye)
    # Clinical anatomical label only ('Left' / 'Right'). The image→patient mapping is
    # recorded in metadata; the HUD shows the patient's anatomical eye directly.
    # Front-facing video: eye_key 'L' (left of image) is the patient's RIGHT eye, and
    # eye_key 'R' (right of image) is the patient's LEFT eye.
    eye_label = "Right" if eye_key == "L" else "Left"

    det = None
    if not args.no_mediapipe:
        # Lazy import — MediaPipe is optional. Tolerate runtimes that don't expose face_mesh.
        try:
            from src.core.iris_tracking import IrisDetector
            det = IrisDetector()
        except Exception as exc:
            print(f"MediaPipe unavailable ({exc}); continuing without it (recovery only).")
            det = None

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    init_fr = int(approved["init_frame_number"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_fr - 1)
    ok, init_frame = cap.read()
    if not ok:
        raise SystemExit("Could not read init frame from video.")
    init_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    trk = V1Tracker(eye_key, (iris["x"], iris["y"]), iris["radius"], init_gray, init_fr,
                    use_cft_helper=args.cft_helper)

    # Eye-opening contour reference — same source as iris_tracker.run() uses.
    contour_pts = contour_from_approved(approved, eye_key,
                                        (iris["x"], iris["y"], iris["radius"]))
    contour_trk = (EyeOpeningContourTracker(eye_key, contour_pts, init_gray)
                   if (contour_pts is not None and len(contour_pts) >= 6) else None)
    contour_state = contour_trk.state if contour_trk is not None else None

    out_dir = Path(args.output_dir) / (video.stem + "_tracked")
    out_dir.mkdir(parents=True, exist_ok=True)
    disp = 560
    hud_h = 152
    writer = cv2.VideoWriter(str(out_dir / "debug_replay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (disp, disp + hud_h))

    print(f"Debug replay (V1): {video.name}  eye={eye_label}  zoom={args.zoom}x  "
          f"mp={'on' if det else 'off'}  helper={'on' if args.cft_helper else 'off'}  "
          f"-> {out_dir/'debug_replay.mp4'}")
    frames_disp = []
    fr = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr += 1
        if fr < init_fr or fr < args.start:
            continue
        if args.end and fr > args.end:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if det is not None:
            _, le, re = det.detect(frame)
            o = le if eye_key == "L" else re
            mp = (o.single_x, o.single_y, o.iris_radius) if (o.detected and o.iris_radius) else None
            ear = o.ear if o.detected else None
        else:
            mp = None
            ear = None
        if contour_trk is not None:
            contour_state = contour_trk.step(gray) if fr > init_fr else contour_trk.state
            ref_state = contour_state.reference_state
            ref_conf = contour_state.confidence
            ex = contour_state.extents()
            aperture = (ex["medial"], ex["lateral"], ex["upper"], ex["lower"])
        else:
            ref_state, ref_conf, aperture = "lost", 0.0, (None, None, None, None)
        fm = trk.step(gray, fr, int(fr / fps * 1000), fr / fps, ear,
                      aperture=aperture, mp_iris=mp, bgr=frame,
                      reference_state=ref_state, reference_confidence=ref_conf)
        disp_frame = render_frame(frame, trk, fm, contour_state, eye_label, args.zoom, disp, hud_h)
        writer.write(disp_frame)
        if args.live:
            frames_disp.append(disp_frame)
    writer.release()
    cap.release()
    print(f"Wrote {out_dir/'debug_replay.mp4'}  ({fr} frames)")

    if args.live and frames_disp:
        i, playing = 0, True
        while True:
            cv2.imshow("V1 Tracking Replay", frames_disp[i])
            k = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord(" "):
                playing = not playing
            elif k in (ord("n"), 83):
                playing = False; i = min(len(frames_disp) - 1, i + 1)
            elif k in (ord("b"), 81):
                playing = False; i = max(0, i - 1)
            elif playing:
                i += 1
                if i >= len(frames_disp):
                    i = 0
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
