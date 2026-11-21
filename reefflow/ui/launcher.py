"""Start the ReefFlow UI: uvicorn on 127.0.0.1 (a free port) and open the browser.

    python -m reefflow.ui [--workspace DIR] [--port N] [--host 127.0.0.1] [--no-browser]

``main(argv)`` is the console entry point (use it for a ``reefflow ui`` subcommand);
``serve(workspace, port, host, open_browser)`` is the programmatic one.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _answers(port: int) -> bool:
    """Does something already accept connections on this port on this computer?
    (Windows lets 127.0.0.1:N and 0.0.0.0:N be bound by two programs at once, so a
    bind test alone can pick a port another ReefFlow window is serving.)"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def free_port(host: str = "127.0.0.1", preferred: int = 8765) -> int:
    for port in (preferred, 0):
        if port and _answers(port):
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                s.bind((host, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free TCP port")


def _open_when_ready(url: str, host: str, port: int):
    for _ in range(200):
        try:
            with socket.create_connection((host, port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    webbrowser.open(url)


def serve(workspace=None, port: int | None = None, host: str = "127.0.0.1", open_browser: bool = True,
          log_level: str = "warning"):
    import reefflow  # noqa: F401  (Taichi offline setting before anything imports it)
    import uvicorn
    from .server import create_app
    ws = Path(workspace or os.environ.get("REEFFLOW_WORKSPACE") or Path.cwd() / "reefflow_workspace").resolve()
    port = port or free_port(host)
    app = create_app(ws)
    url = f"http://{host}:{port}/"
    print(f"ReefFlow UI: {url}   (workspace: {ws})")
    print("Keep this window open while you use the UI; press Ctrl+C to stop.")
    if open_browser:
        threading.Thread(target=_open_when_ready, args=(url, host, port), daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="reefflow ui", description="ReefFlow browser UI (local, offline)")
    ap.add_argument("--workspace", help="folder for projects and runs (default ./reefflow_workspace)")
    ap.add_argument("--port", type=int, help="TCP port (default: 8765 or any free port)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (default 127.0.0.1 = this computer only)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    a = ap.parse_args(argv)
    serve(a.workspace, a.port, a.host, not a.no_browser)


if __name__ == "__main__":
    main()
