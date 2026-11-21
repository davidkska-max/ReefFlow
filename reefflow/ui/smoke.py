"""Manual smoke test: start the UI server headless, fetch the main page and a few APIs, stop it.

    python -m reefflow.ui.smoke [--workspace DIR]

Exit code 0 when everything answered. Uses a temporary workspace by default.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import http.client
import time

from .launcher import free_port


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace")
    a = ap.parse_args(argv)
    ws = a.workspace or tempfile.mkdtemp(prefix="reefflow_smoke_")
    port = free_port(preferred=0)
    proc = subprocess.Popen([sys.executable, "-m", "reefflow.ui", "--workspace", ws, "--port", str(port), "--no-browser"])
    base = f"http://127.0.0.1:{port}"

    def get(path, timeout=30):
        # plain http.client: no proxy, and no SSL context (urllib's opener builds one, which crashes
        # Python on machines whose SSLKEYLOGFILE points at a network-monitor pipe)
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            c.request("GET", path)
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}")
            return body
        finally:
            c.close()
    ok = True
    try:
        for _ in range(150):
            try:
                get("/api/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        checks = [("/", lambda b: b"importmap" in b and b"ReefFlow" in b),
                  ("/api/health", lambda b: json.loads(b)["ok"]),
                  ("/static/vendor/three/three.module.js", lambda b: b"REVISION" in b),
                  ("/static/js/main.js", lambda b: b"class App" in b),
                  ("/api/library", lambda b: "pumps" in json.loads(b)),
                  ("/api/projects", lambda b: "projects" in json.loads(b))]
        for path, test in checks:
            try:
                body = get(path)
                good = bool(test(body))
            except Exception as e:                       # noqa: BLE001
                good, body = False, str(e).encode()
            ok &= good
            print(f"{'OK  ' if good else 'FAIL'} GET {path} ({len(body)} bytes)")
    finally:
        if sys.platform == "win32":      # the venv launcher spawns the real interpreter: stop the whole tree
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
        proc.wait(10)
    print("smoke test", "passed" if ok else "FAILED", f"(server was {base}, workspace {ws})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
