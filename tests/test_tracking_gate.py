"""Dependency-free tests for the Mode A collapse gate and plausibility score.

These avoid MediaPipe and any video file so they run fast anywhere:

    python tests/test_tracking_gate.py

(They also work under pytest if it is installed.)

Empirical thresholds being guarded (measured on samples/ + the face clip):
  - good frontal frames: mesh width fraction >= ~0.58
  - collapsed-mesh tail: mesh width fraction <= ~0.47
  - default gate min_mesh_frac = 0.50 sits cleanly between them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.tracking import mesh_width_frac, _eye_confidence  # noqa: E402

WIDTH = 3840


def _rect_ring(cx, cy, half_w, half_h):
    """Minimal eye-contour ring (4 corners) centred on (cx, cy)."""
    return [
        (cx - half_w, cy - half_h),
        (cx + half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        (cx - half_w, cy + half_h),
    ]


def test_mesh_width_frac_healthy_vs_collapsed():
    healthy = [(int(0.10 * WIDTH), 100), (int(0.82 * WIDTH), 900)]  # spans 0.72
    collapsed = [(int(0.40 * WIDTH), 100), (int(0.62 * WIDTH), 300)]  # spans 0.22
    assert mesh_width_frac(healthy, WIDTH) > 0.58
    assert mesh_width_frac(collapsed, WIDTH) < 0.47


def test_gate_decision_at_default_threshold():
    # Default floor is loose (0.20): a small/distant but valid face must pass,
    # only a grossly degenerate mesh is rejected. (Tuned after a 0.50 floor wrongly
    # rejected real faces that filled only ~0.35-0.47 of the frame width.)
    min_frac = 0.20
    small_face = mesh_width_frac([(1340, 0), (2685, 0)], WIDTH)    # ~0.35 -> accept
    degenerate = mesh_width_frac([(1850, 0), (2150, 0)], WIDTH)    # ~0.08 -> reject
    assert small_face >= min_frac
    assert degenerate < min_frac


def test_mesh_width_frac_degenerate_inputs():
    assert mesh_width_frac([], WIDTH) == 0.0
    assert mesh_width_frac([(10, 10)], 0) == 0.0


def test_eye_confidence_iris_inside_open_eye_is_high():
    ring = _rect_ring(1000, 500, half_w=120, half_h=45)  # aspect ~0.375 -> open
    points = {i: (0, 0) for i in range(478)}
    # place the ring landmarks at known indices and the iris centre inside
    for idx, pt in zip([33, 133, 159, 145], ring):
        points[idx] = pt
    conf = _eye_confidence(points, [33, 133, 159, 145], 1000, 500)
    assert conf > 0.8, conf


def test_eye_confidence_iris_outside_is_penalised():
    ring = _rect_ring(1000, 500, half_w=120, half_h=45)
    points = {i: (0, 0) for i in range(478)}
    for idx, pt in zip([33, 133, 159, 145], ring):
        points[idx] = pt
    inside = _eye_confidence(points, [33, 133, 159, 145], 1000, 500)
    outside = _eye_confidence(points, [33, 133, 159, 145], 1500, 500)  # far outside box
    assert outside < inside


def test_eye_confidence_missing_iris_is_zero():
    assert _eye_confidence([(0, 0)] * 4, [0, 1, 2, 3], None, None) == 0.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'OK' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
