"""Phone photo link: take the reconstruction photos with a phone, straight into the Photos tab.

The main UI listens on 127.0.0.1 only. "Use my phone" (Photos tab) starts a second, small
HTTP server on the home network for one project (DECISIONS D-835):

* it serves one mobile page (static/phone.html) and three calls, all under a random
  128-bit token in the URL (shown as a QR code): ``GET /p/<token>``,
  ``GET /p/<token>/info``, ``POST /p/<token>/photo``, ``DELETE /p/<token>/photo/<vid>``;
* requests from addresses outside private / loopback ranges are refused;
* it can only add photos to that project's photo session (and delete the ones it added);
* it stops after 60 minutes without a request, when "Stop" is pressed, or with the app.

Plain HTTP: phone browsers allow the camera through ``<input type=file capture>`` without
HTTPS, and a self-signed certificate would only add a scary warning on the phone.
"""
from __future__ import annotations

import ipaddress
import os
import secrets
import socket
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from . import photos as PH
from .workspace import check_id

STATIC = Path(__file__).resolve().parent / "static"
IDLE_LIMIT_S = 3600
MAX_PHOTO_BYTES = 40_000_000
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

# the guided shot list (kind = reefflow.photo.tankref.VIEW_KINDS)
SHOTS = [
    {"id": "front", "kind": "front", "title": "Front", "need": "required",
     "how": "Stand in front of the tank, centred, about 1–2 m back. Hold the phone upright and square to the glass."},
    {"id": "top", "kind": "top", "title": "Top, straight down", "need": "required",
     "how": "From above with the lid off, phone flat and level, looking straight down. Use a step stool if needed."},
    {"id": "left", "kind": "left", "title": "Left side", "need": "recommended",
     "how": "Through the left end glass, centred and square to it."},
    {"id": "right", "kind": "right", "title": "Right side", "need": "recommended",
     "how": "Through the right end glass, centred and square to it."},
    {"id": "front_high", "kind": "front", "title": "Front, from higher up", "need": "recommended",
     "how": "Through the front glass again, but from about 30° above (raise the phone, tilt it down). "
            "This view sees between the rocks and stops the 3-D rock coming out too fat."},
    {"id": "back", "kind": "back", "title": "Back", "need": "optional",
     "how": "Only if the back glass is clear and you can reach behind the tank."},
]
SHOT_IDS = {s["id"]: s for s in SHOTS}


def lan_addresses() -> list[str]:
    """Private IPv4 addresses of this computer, best guess for the home network first."""
    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))          # no packet is sent; picks the default-route interface
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    out = []
    for ip in found:
        a = ipaddress.ip_address(ip)
        if a.is_private and not a.is_loopback and not a.is_link_local and ip not in out:
            out.append(ip)

    def rank(ip):
        return 0 if ip.startswith("192.168.") else 1 if ip.startswith("10.") else 2
    first = out[:1]
    return first + sorted(out[1:], key=rank)


def _client_ok(host: str | None) -> bool:
    try:
        a = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return a.is_private or a.is_loopback


class PhoneLink:
    """At most one live link (single-user app)."""

    def __init__(self, ws):
        self.ws = ws
        self.lock = threading.RLock()
        self.server = None
        self.thread = None
        self.state = None

    # ---------------------------------------------------------------- lifecycle
    def start(self, pid: str, port: int | None = None, host: str | None = None) -> dict:
        import uvicorn
        from .launcher import free_port
        check_id(pid, "project id")
        host = host or os.environ.get("REEFFLOW_PHONE_HOST", "0.0.0.0")    # tests bind 127.0.0.1
        raw = self.ws.read_raw(pid)                       # 404s on unknown projects
        with self.lock:
            if self.state and self.state["pid"] == pid and self.server and not self.server.should_exit:
                self.state["last"] = time.time()
                return self.public()
            self.stop()
            ips = lan_addresses()
            port = port or free_port(host, 8770)
            token = secrets.token_urlsafe(16)
            self.state = {"pid": pid, "project": raw.get("name") or pid, "token": token, "port": port,
                          "ips": ips, "started": time.time(), "last": time.time(), "uploads": {}, "events": []}
            app = self._app(token)
            cfg = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off", access_log=False)
            self.server = uvicorn.Server(cfg)
            self.server.install_signal_handlers = lambda: None   # not the main thread
            self.thread = threading.Thread(target=self.server.run, name="phone-link", daemon=True)
            self.thread.start()
            for _ in range(100):
                if self.server.started:
                    break
                time.sleep(0.05)
            if not self.server.started:
                self.stop()
                raise RuntimeError(f"could not open port {port} for the phone link")
            threading.Thread(target=self._watchdog, args=(token,), daemon=True).start()
            return self.public()

    def stop(self):
        with self.lock:
            if self.server:
                self.server.should_exit = True
            th = self.thread
            self.server = self.thread = None
            self.state = None
        if th:
            th.join(timeout=3)

    def _watchdog(self, token):
        while True:
            time.sleep(5)
            with self.lock:
                st = self.state
                if not st or st["token"] != token:
                    return
                if time.time() - st["last"] > IDLE_LIMIT_S:
                    self.stop()
                    return

    def public(self) -> dict:
        with self.lock:
            st = self.state
            if not st:
                return {"active": False}
            urls = [f"http://{ip}:{st['port']}/p/{st['token']}" for ip in st["ips"]]
            return {"active": True, "project_id": st["pid"], "project": st["project"], "urls": urls,
                    "url": urls[0] if urls else None, "port": st["port"],
                    "qr_svg": qr_svg(urls[0]) if urls else None,
                    "received": list(st["events"]), "idle_limit_s": IDLE_LIMIT_S,
                    "seconds_left": max(0, int(IDLE_LIMIT_S - (time.time() - st["last"]))),
                    "warning": None if urls else "This computer has no home-network (private) address: connect it to the same Wi-Fi or router as the phone."}

    # ---------------------------------------------------------------- the phone-side app
    def _session(self, pid):
        from .. import project as PR
        d = PH.session_dir(self.ws.project_dir(pid))
        raw = self.ws.read_raw(pid)
        P = PR.parse({**raw, "rockscape": [], "pumps": [], "overflow": None}, self.ws.project_path(pid))
        PH.sync_tank(d, P.tank)
        return d, P.tank, raw

    def _info(self) -> dict:
        st = self.state
        d, tank, raw = self._session(st["pid"])
        views = PH.load(d)["views"]
        shots = []
        for s in SHOTS:
            mine = [vid for vid, card in st["uploads"].items() if card == s["id"] and vid in views]
            shots.append({**s, "photo": mine[-1] if mine else None})
        kinds = sorted({v["kind"] for v in views.values()})
        return {"project": st["project"], "units": (raw.get("units") or "US"),
                "tank": {"length": tank.length, "width": tank.width, "height": tank.height},
                "shots": shots, "kinds_on_computer": kinds, "photos_on_computer": len(views),
                "seconds_left": max(0, int(IDLE_LIMIT_S - (time.time() - st["last"])))}

    def _app(self, token: str) -> FastAPI:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        link = self

        @app.exception_handler(Exception)
        async def _any(request: Request, exc: Exception):
            code = 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 400 if isinstance(exc, ValueError) else 500
            return JSONResponse({"detail": str(exc).strip("'\"") or type(exc).__name__}, status_code=code)

        @app.middleware("http")
        async def guard(request: Request, call_next):
            if not _client_ok(request.client.host if request.client else None):
                return JSONResponse({"detail": "only devices on the home network may connect"}, status_code=403)
            parts = request.url.path.strip("/").split("/")
            if len(parts) < 2 or parts[0] != "p" or not secrets.compare_digest(parts[1], token):
                return JSONResponse({"detail": "not found (the link may have expired: scan the QR code again)"}, status_code=404)
            with link.lock:
                if not link.state or link.state["token"] != token:
                    return JSONResponse({"detail": "this phone link was stopped"}, status_code=410)
                link.state["last"] = time.time()
            return await call_next(request)

        @app.get("/p/{tok}", response_class=HTMLResponse)
        def page(tok: str):
            return HTMLResponse((STATIC / "phone.html").read_text(encoding="utf-8"),
                                headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})

        @app.get("/p/{tok}/info")
        async def info(tok: str):
            return await run_in_threadpool(link._info)

        @app.post("/p/{tok}/photo")
        async def photo(tok: str, file: UploadFile = File(...), shot: str = Form(...)):
            s = SHOT_IDS.get(shot)
            if not s:
                raise HTTPException(400, f"unknown shot {shot!r}")
            name = Path(file.filename or "photo.jpg").name
            if not name.lower().endswith(IMG_EXT):
                name = Path(name).stem + ".jpg"            # the page sends JPEG when it re-encodes
            data = await file.read()
            if len(data) > MAX_PHOTO_BYTES:
                raise HTTPException(413, "photo larger than 40 MB")
            pid = link.state["pid"]
            try:
                d, _, _ = await run_in_threadpool(link._session, pid)
                v = await run_in_threadpool(PH.add_view, d, s["kind"], name, data)
            except ValueError as e:
                raise HTTPException(400, str(e))
            with link.lock:
                old = [vid for vid, card in link.state["uploads"].items() if card == shot]
                link.state["uploads"][v["id"]] = shot
                link.state["events"].append({"time": time.time(), "view": v["id"], "shot": shot, "title": s["title"],
                                             "size": [v["width"], v["height"]]})
            for vid in old:                                # a retake replaces the earlier photo of the card
                try:
                    PH.delete_view(d, vid)
                except Exception:                          # noqa: BLE001
                    pass
                with link.lock:
                    link.state["uploads"].pop(vid, None)
            return {"view": {"id": v["id"], "width": v["width"], "height": v["height"],
                             "focal_from_exif": bool(v.get("exif_f35_mm"))}, "info": await run_in_threadpool(link._info)}

        @app.delete("/p/{tok}/photo/{vid}")
        async def delete(tok: str, vid: str):
            check_id(vid, "photo id")
            with link.lock:
                if vid not in link.state["uploads"]:
                    raise HTTPException(403, "the phone can only delete photos it sent itself")
                link.state["uploads"].pop(vid)
            d, _, _ = await run_in_threadpool(link._session, link.state["pid"])
            try:
                PH.delete_view(d, vid)
            except Exception:                              # noqa: BLE001
                pass
            return await run_in_threadpool(link._info)

        return app


def qr_svg(text: str) -> str:
    import segno
    q = segno.make(text, error="m")
    return q.svg_inline(scale=6, border=2, dark="#102a33", omitsize=True)


def register(app, ws):
    """Routes of the main (local-only) app that control the link."""
    from fastapi import Body
    link = PhoneLink(ws)
    app.state.phone = link

    @app.post("/api/phone/start")
    async def phone_start(body: dict = Body(...)):
        pid = body.get("project")
        if not pid:
            raise HTTPException(400, "body must be {project: <id>}")
        try:
            return await run_in_threadpool(link.start, pid)
        except RuntimeError as e:
            raise HTTPException(500, str(e))

    @app.get("/api/phone/status")
    def phone_status():
        return link.public()

    @app.post("/api/phone/stop")
    async def phone_stop():
        await run_in_threadpool(link.stop)
        return {"active": False}

    return link
