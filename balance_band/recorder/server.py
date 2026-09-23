"""Balance Band recorder — local server (Dr. K, 2026-09-23).

Serves the recording page to Chrome, saves each step's recording as CSV, and runs the analysis.

    python balance_band/recorder/server.py            then open  http://localhost:8765  in Chrome
    python balance_band/recorder/server.py --data-dir D:/BalanceBand/sessions --port 8765

Only this computer can reach it (bound to 127.0.0.1). Recordings go to the data folder, by default
~/BalanceBand/sessions — outside the code folder, so patient data never ends up in git.
Use an ID code for the person, not their name.
"""
import argparse, csv, datetime as dt, json, re, sys, threading, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sway_analysis as SA  # noqa: E402

STATIC = HERE / "static"
STEPS = ("los",) + SA.CONDITIONS
COLUMNS = ["t", "ax", "ay", "az", "gx", "gy", "gz", "fall", "phase"]
MAX_BODY = 50 * 1024 * 1024
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8"}


def safe_id(s, maxlen=40):
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(s or "")).strip("-")[:maxlen]
    if not s:
        raise ValueError("an ID made of letters, digits, - or _ is required")
    return s


class Store:
    def __init__(self, data_dir):
        self.root = Path(data_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, session):
        f = (self.root / safe_id(session, 80)).resolve()
        if f.parent != self.root or not f.is_dir():
            raise ValueError("unknown session")
        return f

    def new_session(self, person, site):
        if site not in SA.SITES:
            raise ValueError(f"site must be one of {SA.SITES}")
        sid = f"{safe_id(person)}_{dt.datetime.now():%Y-%m-%d_%H%M%S}"
        f = self.root / sid
        f.mkdir()
        (f / "session.json").write_text(json.dumps(
            {"person": safe_id(person), "site": site, "started": dt.datetime.now().isoformat(timespec="seconds")},
            indent=2), encoding="utf-8")
        return sid

    def save_step(self, session, step, rows):
        if step not in STEPS:
            raise ValueError(f"step must be one of {STEPS}")
        if not rows:
            raise ValueError("no samples")
        f = self.folder(session)
        with open(f / f"{step}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(COLUMNS)
            for r in rows:
                t, ax, ay, az, gx, gy, gz = (float(v) for v in r[:7])
                fall = 1 if int(r[7]) else 0
                phase = safe_id(r[8], 12) if len(r) > 8 and r[8] else ""
                w.writerow([f"{t:.4f}", f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}",
                            f"{gx:.3f}", f"{gy:.3f}", f"{gz:.3f}", fall, phase])
        return len(rows)

    def status(self, session):
        f = self.folder(session)
        return {s: (f / f"{s}.csv").exists() for s in STEPS}

    def analyse(self, session):
        f = self.folder(session)
        meta = json.loads((f / "session.json").read_text(encoding="utf-8"))
        res = SA.analyse_session(f, site=meta.get("site", "waist"))
        text = SA.report_text(res)
        (f / "report.txt").write_text(text, encoding="utf-8")
        (f / "result.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        return {"report": text, "folder": str(f)}


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/config":
                return self._send(200, {"los_script": SA.LOS_SCRIPT, "conditions": list(SA.CONDITIONS),
                                        "duration_s": SA.PLANNED_DURATION_S, "sites": list(SA.SITES),
                                        "data_dir": str(store.root)})
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            name = "recorder.html" if path == "/" else path.lstrip("/")
            f = (STATIC / name).resolve()
            if f.parent != STATIC or not f.is_file() or f.suffix not in TYPES:
                return self._send(404, {"error": "not found"})
            self._send(200, f.read_bytes(), TYPES[f.suffix])

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._send(413, {"error": "too large"})
                body = json.loads(self.rfile.read(n) or b"{}")
                path = self.path.split("?")[0]
                if path == "/api/session":
                    return self._send(200, {"session": store.new_session(body.get("person"),
                                                                         body.get("site", "waist"))})
                if path == "/api/save":
                    k = store.save_step(body.get("session"), body.get("step"), body.get("rows"))
                    return self._send(200, {"saved": k, "status": store.status(body.get("session"))})
                if path == "/api/analyse":
                    return self._send(200, store.analyse(body.get("session")))
                return self._send(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
    return H


def serve(data_dir, port=8765, open_browser=False):
    store = Store(data_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    url = f"http://localhost:{httpd.server_address[1]}"
    print(f"Balance Band recorder: open {url} in Chrome.  Data folder: {store.root}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    return httpd


def main(argv=None):
    p = argparse.ArgumentParser(description="Balance Band recorder (local server)")
    p.add_argument("--data-dir", default=str(Path.home() / "BalanceBand" / "sessions"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    httpd = serve(a.data_dir, a.port, open_browser=not a.no_browser)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
