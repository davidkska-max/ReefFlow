"""Phone photo link (reefflow.ui.phone, DECISIONS D-835): a second, token-protected server
that only adds photos to one project's photo session."""
from __future__ import annotations

import os

import numpy as np
import pytest

import reefflow  # noqa: F401

# httpx builds an SSL context even for http://; with SSLKEYLOGFILE set, OpenSSL on Windows aborts
# ("OPENSSL_Applink"). The phone link itself never uses TLS.
os.environ.pop("SSLKEYLOGFILE", None)
httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from reefflow.ui import phone as PHN  # noqa: E402
from reefflow.ui.server import create_app  # noqa: E402


def _jpeg(w=640, h=480, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture()
def linked(tmp_path, monkeypatch):
    monkeypatch.setenv("REEFFLOW_PHONE_HOST", "127.0.0.1")        # no LAN listener (and no firewall prompt) in tests
    app = create_app(tmp_path / "ws", start_worker=False)
    c = TestClient(app)
    pid = c.post("/api/projects", json={"name": "phone", "template": "demo"}).json()["id"]
    st = c.post("/api/phone/start", json={"project": pid}).json()
    tok = app.state.phone.state["token"]
    base = f"http://127.0.0.1:{st['port']}/p/{tok}"
    yield c, app, pid, st, base
    app.state.phone.stop()


def test_link_flow(linked):
    c, app, pid, st, base = linked
    assert st["active"] and st["project_id"] == pid
    assert st["qr_svg"] is None or st["qr_svg"].startswith("<svg")
    page = httpx.get(base)
    assert page.status_code == 200 and "ReefFlow" in page.text
    info = httpx.get(base + "/info").json()
    assert [s["id"] for s in info["shots"]][:2] == ["front", "top"] and info["photos_on_computer"] == 0
    r = httpx.post(base + "/photo", data={"shot": "front"}, files={"file": ("IMG_1.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    r = httpx.post(base + "/photo", data={"shot": "front_high"}, files={"file": ("IMG_2.jpg", _jpeg(seed=1), "image/jpeg")})
    assert r.json()["view"]["id"] == "front-2"                     # the high front shot is a second front view
    views = c.get(f"/api/projects/{pid}/photos").json()["views"]
    assert sorted(v["id"] for v in views) == ["front", "front-2"]
    # a retake replaces the card's earlier photo
    r = httpx.post(base + "/photo", data={"shot": "front"}, files={"file": ("IMG_3.jpg", _jpeg(seed=2), "image/jpeg")})
    assert r.status_code == 200
    ids = sorted(v["id"] for v in c.get(f"/api/projects/{pid}/photos").json()["views"])
    assert len(ids) == 2 and "front-2" in ids
    front_cards = [s for s in r.json()["info"]["shots"] if s["id"] == "front"]
    assert front_cards[0]["photo"] == r.json()["view"]["id"]
    assert len(c.get("/api/phone/status").json()["received"]) == 3
    # the phone may delete only what it sent itself
    up = c.post(f"/api/projects/{pid}/photos/views", files={"file": ("top.jpg", _jpeg(seed=3), "image/jpeg")},
                data={"kind": "top"}).json()["view"]["id"]
    assert httpx.delete(f"{base}/photo/{up}").status_code == 403
    assert httpx.delete(f"{base}/photo/front-2").status_code == 200
    # wrong token, unknown shot, not an image
    assert httpx.get(base[:-3] + "xyz/info").status_code == 404
    assert httpx.post(base + "/photo", data={"shot": "under"}, files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).status_code == 400
    assert httpx.post(base + "/photo", data={"shot": "top"}, files={"file": ("a.jpg", b"nope", "image/jpeg")}).status_code == 400
    # stop: the port closes
    c.post("/api/phone/stop")
    assert not c.get("/api/phone/status").json()["active"]
    with pytest.raises(httpx.HTTPError):
        httpx.get(base + "/info", timeout=1.0)


def test_client_filter_and_addresses():
    assert PHN._client_ok("192.168.1.20") and PHN._client_ok("10.0.0.5") and PHN._client_ok("127.0.0.1")
    assert not PHN._client_ok("8.8.8.8") and not PHN._client_ok(None)
    for ip in PHN.lan_addresses():
        assert not ip.startswith("127.")
    assert PHN.qr_svg("http://192.168.1.5:8770/p/abc").startswith("<svg viewBox")
