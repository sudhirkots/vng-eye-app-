"""Tests for the recorder server (no browser, no band).

    python balance_band/recorder/test_server.py      (also runs under pytest)
"""
import csv, json, sys, tempfile, threading, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server as SV  # noqa: E402
import sway_analysis as SA  # noqa: E402


def _start():
    d = Path(tempfile.mkdtemp())
    httpd = SV.serve(d, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}", d


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _rows(csv_path):
    rows = []
    for r in csv.DictReader(open(csv_path)):
        rows.append([r["t"], r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"], r.get("fall") or 0, r.get("phase") or ""])
    return rows


def test_full_session_via_api():
    httpd, url, root = _start()
    try:
        demo = Path(tempfile.mkdtemp()); SA.write_demo("vestibular", demo, tilted=True)
        code, j = _post(url + "/api/session", {"person": "PT 001", "site": "waist"})
        assert code == 200 and j["session"].startswith("PT-001_")
        sid = j["session"]
        for step in SV.STEPS:
            code, j = _post(url + "/api/save", {"session": sid, "step": step, "rows": _rows(demo / f"{step}.csv")})
            assert code == 200 and j["status"][step]
        code, j = _post(url + "/api/analyse", {"session": sid})
        assert code == 200 and "FAILS ON VESTIBULAR INPUT ALONE" in j["report"]
        assert "found automatically" in j["report"]
        assert (root / sid / "report.txt").exists() and (root / sid / "result.json").exists()
    finally:
        httpd.shutdown()


def test_rejects_bad_input():
    httpd, url, root = _start()
    try:
        assert _post(url + "/api/session", {"person": "../.."})[0] == 400          # no usable ID
        assert _post(url + "/api/session", {"person": "A1", "site": "head"})[0] == 400
        assert _post(url + "/api/save", {"session": "../../etc", "step": "los", "rows": [[0] * 9]})[0] == 400
        sid = _post(url + "/api/session", {"person": "A1"})[1]["session"]
        assert _post(url + "/api/save", {"session": sid, "step": "../x", "rows": [[0] * 9]})[0] == 400
        assert _post(url + "/api/save", {"session": sid, "step": "los", "rows": []})[0] == 400
        assert [p.name for p in root.iterdir()] == [sid]                           # nothing written elsewhere
    finally:
        httpd.shutdown()


def test_serves_page_and_config():
    httpd, url, _ = _start()
    try:
        with urllib.request.urlopen(url + "/") as r:
            assert b"Balance Band Recorder" in r.read()
        with urllib.request.urlopen(url + "/api/config") as r:
            cfg = json.loads(r.read())
            assert cfg["los_script"][1] == ["forward", 6] and cfg["duration_s"] == 20.0
        try:
            urllib.request.urlopen(url + "/../server.py")
            raise AssertionError("should not serve files outside static/")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f(); print("ok ", f.__name__)
    print(f"{len(fns)} passed")
