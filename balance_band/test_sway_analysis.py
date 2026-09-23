"""Tests for the leg balance band analysis (synthetic data, no hardware needed).

    python balance_band/test_sway_analysis.py      (also runs under pytest)
"""
import csv, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sway_analysis as S  # noqa: E402


def _session(pattern):
    d = Path(tempfile.mkdtemp(prefix=f"bal_{pattern}_"))
    S.write_demo(pattern, d)
    return d


def _static_trial(path, ap_deg=0.0, ml_deg=0.0, fs=100.0, dur=20.0, fwd_axis=0, left_axis=1):
    """Constant tilt, to check the sign conventions and axis remapping."""
    n = int(fs * dur); ap, ml = np.radians(ap_deg), np.radians(ml_deg)
    body = np.array([-np.sin(ap), -np.sin(ml), np.sqrt(1 - np.sin(ap)**2 - np.sin(ml)**2)])
    acc = np.zeros(3); acc[fwd_axis] = body[0]; acc[left_axis] = body[1]; acc[3 - fwd_axis - left_axis] = body[2]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
        for i in range(n):
            w.writerow([i / fs, *acc, 0, 0, 0])


def test_sign_conventions():
    d = Path(tempfile.mkdtemp()); _static_trial(d / "t.csv", ap_deg=5, ml_deg=-3)
    ap, ml, *_ = S.sway_angles(S.load_trial(d / "t.csv"), S.body_matrix("+x", "+y"), np.eye(3))
    assert abs(ap[-1] - 5) < 0.1 and abs(ml[-1] + 3) < 0.1        # forward +, left +, right lean -


def test_axis_remap():
    # band strapped with sensor +z facing forward and +x toward the person's left
    d = Path(tempfile.mkdtemp()); _static_trial(d / "t.csv", ap_deg=4, fwd_axis=2, left_axis=0)
    ap, ml, *_ = S.sway_angles(S.load_trial(d / "t.csv"), S.body_matrix("+z", "+x"), np.eye(3))
    assert abs(ap[-1] - 4) < 0.1 and abs(ml[-1]) < 0.1


def test_lowpass_removes_noise_keeps_sway():
    fs = 100.0; t = np.arange(2000) / fs
    sway, noise = np.sin(2 * np.pi * 0.3 * t), 0.5 * np.sin(2 * np.pi * 5.0 * t)
    y = S.lowpass(sway + noise, fs, S.SWAY_LP_HZ)
    mid = slice(200, -200)
    assert np.max(np.abs(y[mid] - sway[mid])) < 0.1       # 0.3 Hz sway kept, no time shift; 5 Hz noise gone


def test_limits_of_stability_measured():
    r = S.analyse_session(_session("normal"))
    lim = r["los"]["limits_deg"]
    assert abs(lim["forward"] - 7) < 1 and abs(lim["backward"] - 5) < 1
    assert abs(lim["left"] - 5) < 1 and abs(lim["right"] - 5) < 1


def test_normal():
    r = S.analyse_session(_session("normal"))
    assert r["overall"].startswith("NO MARKED SENSORY DEPENDENCE")
    assert all(r["trials"][c]["status"] == "complete" for c in S.CONDITIONS)


def test_vision_dependent():
    assert S.analyse_session(_session("vision"))["overall"] == "VISION-DEPENDENT"


def test_somatosensory_dependent():
    assert S.analyse_session(_session("somatosensory"))["overall"] == "SOMATOSENSORY-DEPENDENT"


def test_fails_on_vestibular_alone():
    r = S.analyse_session(_session("vestibular"))
    assert r["overall"] == "FAILS ON VESTIBULAR INPUT ALONE"
    assert r["trials"]["ec_foam"]["fall_direction"] == "to the RIGHT"
    assert r["trials"]["ec_foam"]["los_used_pct"] == 100


def test_reduced_backward_limit_and_fall():
    r = S.analyse_session(_session("backward"))
    assert any("BACKWARD limit" in f for f in r["findings"])
    assert r["trials"]["ec_foam"]["fall_direction"] == "backward"


def test_waist_is_default_site():
    r = S.analyse_session(_session("normal"))
    assert r["site"] == "waist" and any(c.startswith("Waist band") for c in r["caveats"])


def test_shin_site_warns_about_hip_strategy():
    r = S.analyse_session(_session("normal"), site="shin")
    assert r["site"] == "shin" and any("hips" in c for c in r["caveats"] if c.startswith("Shin band"))


def test_unknown_site_rejected():
    try:
        S.analyse_session(_session("normal"), site="head")
    except ValueError:
        return
    raise AssertionError("site='head' should be rejected")


def test_band_orientation_found_automatically():
    # the same person, band strapped on straight vs at a random angle -> the same result
    d_straight, d_tilted = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    S.write_demo("backward", d_straight); S.write_demo("backward", d_tilted, tilted=True)
    r1, r2 = S.analyse_session(d_straight), S.analyse_session(d_tilted)
    assert r2["axes"]["method"].startswith("found automatically")
    for d in S.DIRS:
        assert abs(r1["los"]["limits_deg"][d] - r2["los"]["limits_deg"][d]) < 0.2
    assert r1["overall"] == r2["overall"]
    assert r2["trials"]["ec_foam"]["fall_direction"] == "backward"


def test_no_phase_labels_means_assumed_axes_with_warning():
    d = _session("normal")
    rows = list(csv.DictReader(open(d / "los.csv")))
    with open(d / "los.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "phase"]); w.writeheader()
        for r in rows:
            r.pop("phase"); w.writerow(r)
    r = S.analyse_session(d)
    assert r["axes"]["method"].startswith("ASSUMED")
    assert any("orientation NOT known" in c for c in r["caveats"])


def test_wrong_lean_order_is_flagged():
    d = _session("normal")
    rows = list(csv.DictReader(open(d / "los.csv")))
    swap = {"left": "right", "right": "left"}                      # person leaned right when told left
    for r in rows:
        r["phase"] = swap.get(r["phase"], r["phase"])
    with open(d / "los.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    r = S.analyse_session(d)
    assert any("did not point where expected" in c for c in r["caveats"])


def test_missing_everything_is_insufficient():
    r = S.analyse_session(Path(tempfile.mkdtemp()))
    assert r["overall"].startswith("INSUFFICIENT")


def test_missing_reference_is_insufficient_not_normal():
    d = _session("normal"); (d / "eo_firm.csv").unlink()
    assert S.analyse_session(d)["overall"].startswith("INSUFFICIENT")


def test_missing_condition_is_unclear_not_normal():
    d = _session("normal"); (d / "ec_foam.csv").unlink()
    assert S.analyse_session(d)["overall"].startswith("UNCLEAR")


def test_no_los_still_runs_with_caveat():
    d = _session("vision"); (d / "los.csv").unlink()
    r = S.analyse_session(d)
    assert r["overall"] == "VISION-DEPENDENT" and any("limits-of-stability" in c for c in r["caveats"])


def test_low_sample_rate_is_insufficient():
    d = Path(tempfile.mkdtemp()); _static_trial(d / "eo_firm.csv", fs=10.0)
    r = S.analyse_session(d)
    assert r["trials"]["eo_firm"]["status"] == "insufficient" and r["overall"].startswith("INSUFFICIENT")


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f(); print("ok ", f.__name__)
    print(f"{len(fns)} passed")
