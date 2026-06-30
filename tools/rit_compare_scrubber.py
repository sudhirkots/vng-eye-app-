"""Side-by-side RIT comparison scrubber.

LEFT panel  : tracker output WITHOUT RIT guardrails (raw V1 limbus candidate).
RIGHT panel : tracker output WITH RIT guardrails (post-validator).

Both panels show the same source frame. The point is to visually classify each rejected
frame into one of:

    A. RIT correctly rejected wrong target
    B. RIT wrongly rejected good iris because static orbit was stale
    C. RIT wrongly rejected good iris because sclera rule was too strict
    D. RIT wrongly rejected good iris because arc rule was too strict
    E. The existing tracker itself was already wrong

This is a *review tool only* — no tracker logic runs here, both CSVs are pre-computed:
    - guarded:   outputs/<stem>_tracked/tracking.csv
    - baseline:  outputs/<stem>_tracked_no_guardrails/tracking.csv  (run with --disable-rit-validator)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------- styling
COLOR_OK        = (60, 220, 60)        # green: accepted iris
COLOR_BAD       = (60, 60, 230)        # red:   rejected candidate
COLOR_STALE     = (40, 180, 240)       # amber: good iris but static orbit stale
COLOR_ORBIT     = (255, 200, 80)       # cyan-blue: Stage-0 eye-opening oval
COLOR_STAGE0_R  = (200, 200, 200)      # grey: Stage-0 limbus reference circle
COLOR_RAW       = (240, 200, 60)       # warm yellow: raw (pre-RIT) candidate
COLOR_TEXT      = (255, 255, 255)
COLOR_TEXT_BG   = (0, 0, 0)
FONT            = cv2.FONT_HERSHEY_SIMPLEX


def _draw_text(img, text, org, scale=0.55, color=COLOR_TEXT, thickness=1):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + bl + 1), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _safe_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_csv(path: Path) -> dict[int, dict]:
    """Return {frame_number: row_dict}. Frame numbers are 1-based per the writer."""
    out: dict[int, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fr = int(row["frame_number"])
            except (KeyError, ValueError):
                continue
            out[fr] = row
    return out


def _load_stage0(approved_path: Path) -> dict:
    """Pull the bits we want to overlay on the RIGHT panel."""
    if not approved_path.exists():
        return {"iris": {}, "contours": {}}
    d = json.loads(approved_path.read_text(encoding="utf-8"))
    iris = d.get("iris") or {}
    contours = d.get("eye_opening_contours") or {}
    out_contours = {}
    for ek, pts in contours.items():
        if not isinstance(pts, list) or not pts:
            continue
        arr = np.array([[p["x"], p["y"]] for p in pts if "x" in p and "y" in p],
                       dtype=np.int32)
        if arr.size:
            out_contours[ek] = arr
    return {"iris": iris, "contours": out_contours}


# ----------------------------------------------------------------- panel drawing
def _draw_raw_panel(frame, baseline_row, guarded_row, eye_focus: str, frame_no: int):
    """LEFT panel — tracker output WITHOUT RIT guardrails.

    Preferred source is the baseline CSV (--disable-rit-validator); we fall back to the
    `raw_*` columns of the guarded CSV, which are populated even on rejected frames.
    """
    img = frame.copy()
    _draw_text(img, "WITHOUT RIT GUARDRAILS", (12, 28), scale=0.7, color=(220, 220, 255), thickness=2)
    _draw_text(img, f"frame {frame_no}", (12, 56), scale=0.55)
    if eye_focus != "BOTH":
        _draw_text(img, f"eye focus: {eye_focus}", (12, 80), scale=0.55)

    row = baseline_row or guarded_row
    if row is None:
        _draw_text(img, "(no row)", (12, 110), color=(160, 160, 160))
        return img

    for ek_short, prefix in (("L", "left"), ("R", "right")):
        if eye_focus not in ("BOTH", ek_short):
            continue
        # If a baseline row exists, prefer its accepted-centre columns; otherwise read
        # the guarded CSV's raw_* columns (which are always populated, even on rejects).
        if baseline_row is not None:
            x = _safe_float(baseline_row.get(f"valid_{prefix}_iris_center_x"))
            y = _safe_float(baseline_row.get(f"valid_{prefix}_iris_center_y"))
            r = _safe_float(baseline_row.get(f"{prefix}_iris_radius"))
            label_src = "baseline"
        else:
            x = y = r = None
            label_src = "raw_*"
        if x is None or y is None:
            x = _safe_float((guarded_row or {}).get(f"raw_{prefix}_iris_center_x"))
            y = _safe_float((guarded_row or {}).get(f"raw_{prefix}_iris_center_y"))
            r = _safe_float((guarded_row or {}).get(f"{prefix}_iris_radius"))
            label_src = "raw_*"
        if x is None or y is None:
            continue
        rr = int(round(r)) if r and r > 0 else 6
        cv2.circle(img, (int(round(x)), int(round(y))), rr, COLOR_RAW, 2)
        cv2.drawMarker(img, (int(round(x)), int(round(y))), COLOR_RAW,
                       cv2.MARKER_CROSS, 14, 2)
        line = f"{ek_short}: x={x:.1f} y={y:.1f} r={(r or 0):.1f} ({label_src})"
        _draw_text(img, line, (12, 110 + 24 * (0 if ek_short == "L" else 1)), scale=0.5)
    return img


def _draw_guarded_panel(frame, guarded_row, stage0, eye_focus: str, frame_no: int):
    """RIGHT panel — tracker output WITH RIT guardrails."""
    img = frame.copy()
    _draw_text(img, "WITH RIT GUARDRAILS", (12, 28), scale=0.7, color=(220, 255, 220), thickness=2)
    _draw_text(img, f"frame {frame_no}", (12, 56), scale=0.55)
    if eye_focus != "BOTH":
        _draw_text(img, f"eye focus: {eye_focus}", (12, 80), scale=0.55)

    # Stage-0 reference overlays (oval + limbus seed).
    for ek_short, contour in (stage0.get("contours") or {}).items():
        if eye_focus not in ("BOTH", ek_short):
            continue
        cv2.polylines(img, [contour], True, COLOR_ORBIT, 1, cv2.LINE_AA)
    for ek_short, ic in (stage0.get("iris") or {}).items():
        if eye_focus not in ("BOTH", ek_short):
            continue
        cx = int(round(float(ic.get("x", 0))))
        cy = int(round(float(ic.get("y", 0))))
        rr = int(round(float(ic.get("radius", 0))))
        if rr > 0:
            cv2.circle(img, (cx, cy), rr, COLOR_STAGE0_R, 1, cv2.LINE_AA)

    if guarded_row is None:
        _draw_text(img, "(no row)", (12, 110), color=(160, 160, 160))
        return img

    y_text = 110
    for ek_short, prefix in (("L", "left"), ("R", "right")):
        if eye_focus not in ("BOTH", ek_short):
            continue
        valid_flag = (guarded_row.get(f"iris_valid_{prefix}", "") or "").strip().lower() in ("true", "1")
        drift_reason = (guarded_row.get(f"drift_reason_{prefix}") or "").strip()
        # Where the tracker wanted to put the iris (raw_*) — drawn either green (accepted)
        # or red (rejected) on the same panel so the user sees the candidate location.
        rx = _safe_float(guarded_row.get(f"raw_{prefix}_iris_center_x"))
        ry = _safe_float(guarded_row.get(f"raw_{prefix}_iris_center_y"))
        r  = _safe_float(guarded_row.get(f"{prefix}_iris_radius"))
        stale = "good_iris_but_static_orbit_stale" in drift_reason

        if rx is not None and ry is not None:
            color = COLOR_OK if valid_flag else (COLOR_STALE if stale else COLOR_BAD)
            rr = int(round(r)) if r and r > 0 else 6
            cv2.circle(img, (int(round(rx)), int(round(ry))), rr, color, 2)
            cv2.drawMarker(img, (int(round(rx)), int(round(ry))), color,
                           cv2.MARKER_CROSS, 14, 2)
            if not valid_flag and not stale:
                # Draw a clear strikethrough on rejected candidate.
                cv2.line(img,
                         (int(round(rx - rr)), int(round(ry - rr))),
                         (int(round(rx + rr)), int(round(ry + rr))),
                         color, 2)
                cv2.line(img,
                         (int(round(rx - rr)), int(round(ry + rr))),
                         (int(round(rx + rr)), int(round(ry - rr))),
                         color, 2)

        if valid_flag:
            verdict = "ACCEPTED"
            color = COLOR_OK
        elif stale:
            verdict = "GOOD IRIS BUT STATIC ORBIT STALE"
            color = COLOR_STALE
        else:
            reason = drift_reason or "rejected"
            # Strip the source prefix to keep the chip short.
            short = reason.split(":", 1)[-1] if ":" in reason else reason
            verdict = f"REJECTED: {short}"
            color = COLOR_BAD
        _draw_text(img, f"{ek_short}: {verdict}", (12, y_text), scale=0.55, color=color, thickness=2)
        y_text += 26
        # Stage-0 references (text)
        ic = (stage0.get("iris") or {}).get(ek_short, {})
        if ic:
            _draw_text(img, f"   stage0 r={float(ic.get('radius', 0)):.1f} px", (12, y_text), scale=0.45)
            y_text += 20
        _draw_text(img, f"   iris_valid={valid_flag}  drift_reason={drift_reason or '-'}",
                   (12, y_text), scale=0.45)
        y_text += 22
        if r and r > 0:
            _draw_text(img, f"   raw_r={r:.1f}", (12, y_text), scale=0.45)
            y_text += 22
    return img


def _stack_panels(left_img, right_img):
    """Horizontal stack with a thin separator."""
    h = max(left_img.shape[0], right_img.shape[0])
    if left_img.shape[0] != h:
        left_img = cv2.copyMakeBorder(left_img, 0, h - left_img.shape[0], 0, 0,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if right_img.shape[0] != h:
        right_img = cv2.copyMakeBorder(right_img, 0, h - right_img.shape[0], 0, 0,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
    sep = np.full((h, 4, 3), 80, np.uint8)
    return np.hstack([left_img, sep, right_img])


def _is_rejected(guarded_row, eye_focus: str) -> bool:
    """Frame counts as 'rejected' if the focused eye(s) failed iris_valid."""
    if guarded_row is None:
        return False
    eyes = ("L", "R") if eye_focus == "BOTH" else (eye_focus,)
    for ek in eyes:
        prefix = "left" if ek == "L" else "right"
        v = (guarded_row.get(f"iris_valid_{prefix}", "") or "").strip().lower()
        if v not in ("true", "1"):
            return True
    return False


def _short_reason(guarded_row, eye_focus: str) -> str:
    if guarded_row is None:
        return "norow"
    ek = "L" if eye_focus in ("BOTH", "L") else "R"
    prefix = "left" if ek == "L" else "right"
    reason = (guarded_row.get(f"drift_reason_{prefix}") or "").strip()
    if not reason:
        return "ok"
    return reason.replace(":", "_").replace("|", "__").replace(" ", "_")[:80]


# ----------------------------------------------------------------- main loop
def run_scrubber(video_path: Path, guarded_dir: Path, baseline_dir: Path,
                 start_frame: int | None, eye_focus: str):
    guarded_csv  = guarded_dir / "tracking.csv"
    baseline_csv = baseline_dir / "tracking.csv"
    approved     = guarded_dir / "approved_landmarks.json"

    if not video_path.exists():
        sys.exit(f"video not found: {video_path}")
    if not guarded_csv.exists():
        sys.exit(f"guarded tracking.csv not found: {guarded_csv}")
    baseline_rows: dict[int, dict] = {}
    if baseline_csv.exists():
        baseline_rows = _load_csv(baseline_csv)
    else:
        print(f"[warn] baseline CSV missing ({baseline_csv}); LEFT panel will fall back to raw_* columns.")

    guarded_rows = _load_csv(guarded_csv)
    stage0       = _load_stage0(approved)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sorted_frames = sorted(guarded_rows.keys())
    rejected_frames = [fr for fr in sorted_frames if _is_rejected(guarded_rows.get(fr), eye_focus)]

    if start_frame is None:
        cur = rejected_frames[0] if rejected_frames else (sorted_frames[0] if sorted_frames else 1)
    else:
        cur = max(1, min(total, start_frame))

    save_dir = guarded_dir / "_rit_compare_scrubber"
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_csv = save_dir / "saved_frames.csv"
    if not saved_csv.exists():
        saved_csv.write_text(
            "frame,eye,left_raw_x,left_raw_y,left_raw_radius,right_valid,right_reject_reason,png_path\n",
            encoding="utf-8")

    win = "RIT compare scrubber  [arrows/AD next-prev  PgUp/Dn jump10  N/B next-prev rejected  Home/End  S save  Q quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _show(frame_no: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 1))
        ok, frame = cap.read()
        if not ok or frame is None:
            blank = np.zeros((400, 800, 3), np.uint8)
            _draw_text(blank, f"failed to read frame {frame_no}", (20, 200), color=(0, 0, 255))
            cv2.imshow(win, blank)
            return None
        left  = _draw_raw_panel(frame, baseline_rows.get(frame_no),
                                guarded_rows.get(frame_no), eye_focus, frame_no)
        right = _draw_guarded_panel(frame, guarded_rows.get(frame_no),
                                    stage0, eye_focus, frame_no)
        combined = _stack_panels(left, right)
        # Footer with reject totals + position.
        footer = np.zeros((40, combined.shape[1], 3), np.uint8)
        _draw_text(footer,
                   f"frame {frame_no} / {total}    rejected (focus={eye_focus}): "
                   f"{len(rejected_frames)} / {len(sorted_frames)}",
                   (12, 26), scale=0.55)
        cv2.imshow(win, np.vstack([combined, footer]))
        return combined

    print("Controls: ←/→ or A/D  step 1  |  PgUp/PgDn jump 10  |  Home/End  |  "
          "N/B next/prev rejected  |  S save  |  Q/Esc quit")
    print(f"Total frames {total};  rejected (focus={eye_focus}): "
          f"{len(rejected_frames)} / {len(sorted_frames)};  starting at frame {cur}")

    while True:
        combined = _show(cur)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), ord("Q"), 27):                  # Q / Esc
            break
        elif key in (ord("d"), ord("D"), 2555904, 65363):    # right arrow / D
            cur = min(total, cur + 1)
        elif key in (ord("a"), ord("A"), 2424832, 65361):    # left arrow / A
            cur = max(1, cur - 1)
        elif key in (2228224, 65366):                        # PageDown
            cur = min(total, cur + 10)
        elif key in (2162688, 65365):                        # PageUp
            cur = max(1, cur - 10)
        elif key in (2359296, 65360):                        # Home
            cur = sorted_frames[0] if sorted_frames else 1
        elif key in (2293760, 65367):                        # End
            cur = sorted_frames[-1] if sorted_frames else total
        elif key in (ord("n"), ord("N")):                    # next rejected
            nxt = next((f for f in rejected_frames if f > cur), None)
            if nxt is not None:
                cur = nxt
            else:
                print("[scrubber] no further rejected frames")
        elif key in (ord("b"), ord("B")):                    # previous rejected
            prv = next((f for f in reversed(rejected_frames) if f < cur), None)
            if prv is not None:
                cur = prv
            else:
                print("[scrubber] no earlier rejected frames")
        elif key in (ord("s"), ord("S")):                    # save PNG
            if combined is None:
                continue
            reason = _short_reason(guarded_rows.get(cur), eye_focus)
            eye_tag = eye_focus
            fname = f"fr{cur:04d}_{eye_tag}_compare_{reason}.png"
            path = save_dir / fname
            cv2.imwrite(str(path), combined)
            row = guarded_rows.get(cur) or {}
            lx = row.get("raw_left_iris_center_x", "")
            ly = row.get("raw_left_iris_center_y", "")
            lr = row.get("left_iris_radius", "")
            rv = row.get("iris_valid_right", "")
            rr = row.get("drift_reason_right", "")
            with saved_csv.open("a", encoding="utf-8") as f:
                f.write(f"{cur},{eye_tag},{lx},{ly},{lr},{rv},{rr},{path.as_posix()}\n")
            print(f"[scrubber] saved {path}")
        elif key in (ord("g"), ord("G")):                    # goto frame
            try:
                target = int(input("Go to frame: ").strip())
                cur = max(1, min(total, target))
            except (ValueError, EOFError):
                print("[scrubber] invalid frame number")

    cap.release()
    cv2.destroyAllWindows()


def _resolve_dirs(video_path: Path, output_root: Path, guarded: Path | None,
                  baseline: Path | None):
    stem = video_path.stem
    g = guarded if guarded is not None else output_root / f"{stem}_tracked"
    b = baseline if baseline is not None else output_root / f"{stem}_tracked_no_guardrails"
    return g, b


def main():
    ap = argparse.ArgumentParser(description="Side-by-side RIT comparison scrubber")
    ap.add_argument("video", type=Path, help="source video")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--guarded-dir", type=Path, default=None,
                    help="folder with the WITH-RIT tracking.csv (default <output>/<stem>_tracked)")
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="folder with the WITHOUT-RIT tracking.csv "
                         "(default <output>/<stem>_tracked_no_guardrails)")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="initial frame (default = first rejected frame for the focus eye)")
    ap.add_argument("--eye", choices=["L", "R", "BOTH"], default="BOTH",
                    help="which eye to evaluate / focus on")
    args = ap.parse_args()

    guarded, baseline = _resolve_dirs(args.video, args.output_dir, args.guarded_dir, args.baseline_dir)
    run_scrubber(args.video, guarded, baseline, args.start_frame, args.eye)


if __name__ == "__main__":
    main()
