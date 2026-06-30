"""RIT Iris-in-Orbit scrubber — step through iris_in_orbit_overlay.mp4 frame-by-frame and MARK the
frames where the iris fit is wrong. Marked frame numbers are written to
_rit_iris_in_orbit_probe/problem_frames.csv (which Claude then reads).

Controls:
  D / Right arrow ............ +1 frame
  A / Left arrow ............. -1 frame
  ] ......................... +10 frames      [ ......... -10 frames
  . ......................... +1   , ......... -1  (same as D/A)
  trackbar (top) ............ jump anywhere
  SPACE or M ................ MARK / unmark the current frame as a PROBLEM
  S ......................... save now (also auto-saved on quit)
  Q / Esc ................... quit (saves)

The frame number shown matches the "frame N" burned into the overlay (1-based).
"""
import csv, sys
from pathlib import Path

import cv2

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
REV = OUT / "_rit_iris_in_orbit_probe"
MP4 = REV / "iris_in_orbit_overlay.mp4"
PF = REV / "problem_frames.csv"

cap = cv2.VideoCapture(str(MP4))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

marks = set()
if PF.exists():
    for r in csv.DictReader(open(PF, encoding="utf-8")):
        try:
            marks.add(int(r["frame"]))
        except Exception:
            pass

WIN = "RIT Iris scrubber  (D/A step, ]/[ x10, SPACE=mark, Q=quit)"

_cache = {}
def get(i):
    if i not in _cache:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        _cache[i] = fr if ok else None
        if len(_cache) > 60:
            _cache.pop(next(iter(_cache)))
    return _cache.get(i)

def save():
    with open(PF, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["frame"])
        for fn in sorted(marks):
            w.writerow([fn])

# Create the window and PUMP the event loop with a real frame BEFORE adding the
# trackbar -- cv2 4.13 (Windows) raises "NULL window" on getTrackbarPos if the
# window's message loop hasn't run an imshow first.
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 760)
_first = get(0)
if _first is not None:
    cv2.imshow(WIN, _first)
cv2.waitKey(60)
cv2.createTrackbar("frame", WIN, 0, max(0, N - 1), lambda v: None)
cv2.waitKey(60)
_has_tb = True

def tb_get():
    global _has_tb
    if not _has_tb:
        return None
    try:
        return cv2.getTrackbarPos("frame", WIN)
    except cv2.error:
        _has_tb = False
        return None

def tb_set(v):
    if not _has_tb:
        return
    try:
        cv2.setTrackbarPos("frame", WIN, v)
    except cv2.error:
        pass

i = 0
while True:
    tb = tb_get()
    if tb is not None and tb != i:
        i = tb
    fr = get(i)
    if fr is not None:
        disp = fr.copy()
        fnum = i + 1                      # burned-in overlay is 1-based
        marked = fnum in marks
        h = disp.shape[0]
        bar = f"[{fnum}/{N}]   {'>>> MARKED PROBLEM <<<' if marked else 'ok'}   total marks: {len(marks)}"
        cv2.rectangle(disp, (0, h - 30), (disp.shape[1], h), (0, 0, 0), -1)
        cv2.putText(disp, bar, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if marked else (0, 255, 0), 2)
        cv2.imshow(WIN, disp)
    k = cv2.waitKeyEx(30)
    if k == -1:
        continue
    kc = k & 0xFF
    if kc in (ord('d'), ord('.')) or k in (2555904, 65363):       # right
        i = min(N - 1, i + 1)
    elif kc in (ord('a'), ord(',')) or k in (2424832, 65361):     # left
        i = max(0, i - 1)
    elif kc == ord(']'):
        i = min(N - 1, i + 10)
    elif kc == ord('['):
        i = max(0, i - 10)
    elif kc in (32, ord('m')):                                    # SPACE / M = toggle mark
        fnum = i + 1
        marks.discard(fnum) if fnum in marks else marks.add(fnum)
        save()
    elif kc == ord('s'):
        save()
    elif kc in (ord('q'), 27):
        break
    tb_set(i)

save()
cap.release(); cv2.destroyAllWindows()
print(f"saved {len(marks)} problem frames -> {PF}")
print("frames:", sorted(marks))
