"""Phone / 3-D scan clean-up (reefflow.scan, DECISIONS D-831 to D-834) and its endpoints.

The scans are synthetic (reefflow.scan.synthetic_scan): a ~26 cm rock with a 50 mm
tunnel on a table, open at the bottom, with holes, a loose piece and vertex noise,
in the glTF axis convention (+Y up, metres), like a phone app export."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import reefflow  # noqa: F401
from reefflow import scan as S
from reefflow.geometry import voxelize_mesh


def _inside(v, f, p):
    return bool(voxelize_mesh(v, f, np.asarray(p, float), 0.001, (1, 1, 1))[0, 0, 0])


def _iou(res, truth, cut, h=0.002):
    sh = np.asarray(res["report"]["offset_m"])
    lo, hi = np.array([-0.18, -0.14, cut]), np.array([0.18, 0.14, 0.22])
    ax = [lo[a] + h * (np.arange(int((hi[a] - lo[a]) / h)) + 0.5) for a in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    A = truth["sdf"](P) < 0
    B = voxelize_mesh(res["vertices"] - sh, res["faces"], lo + 0.5 * h, h, [len(a) for a in ax]).reshape(-1)
    return (A & B).sum() / (A | B).sum()


@pytest.mark.parametrize("name,kw,ext", [
    ("glb_y_up", {}, ".glb"),
    ("inside_out", {"invert": True}, ".glb"),
    ("stl_z_up", {"y_up": False}, ".stl"),
    ("obj_z_up_wrong_convention", {"y_up": False}, ".obj"),
    ("no_table", {"table": False}, ".glb"),
    ("big_hole_100mm", {"hole_radius": 0.05, "holes": 1}, ".glb"),
])
def test_clean_synthetic_scan(name, kw, ext):
    v, f, truth = S.synthetic_scan(**kw)
    res = S.clean(v, f, ext)
    rep = res["report"]
    assert rep["unit"] == "m" and rep["watertight"]
    assert rep["up"] == ("+y" if kw.get("y_up", True) else "+z")
    assert rep["normals_flipped"] == bool(kw.get("invert"))
    cut = rep["ground_cut"]["z"] if rep["ground_cut"] else 0.0
    if kw.get("table", True):
        assert rep["ground_cut"] is not None and 0.002 < rep["ground_cut"]["height"] < 0.008
    tv = S.truth_volume(truth, cut)
    assert abs(rep["volume_m3"] / tv - 1) < 0.025, rep["volume_m3"] / tv
    assert _iou(res, truth, cut) > 0.96
    sh = np.asarray(rep["offset_m"])
    tp, tax, _ = truth["tunnel"]
    for s in (-0.06, 0.0, 0.06):                     # the 50 mm tunnel stays open end to end
        assert not _inside(res["vertices"], res["faces"], tp + s * tax + sh)
    assert not _inside(res["vertices"], res["faces"], truth["floater"] + sh)   # loose piece removed
    assert abs(res["vertices"][:, 2].min()) < 1e-9                              # base on z = 0


def test_units_scale_and_preview():
    v, f, truth = S.synthetic_scan(floater=False)
    a = S.analyze(v * 1000, f, ".obj")
    assert a["unit_guessed"] == "mm" and a["up"] == "+y"
    r1 = S.clean(v, f, ".glb")["report"]
    r2 = S.clean(v * 100, f, ".glb", unit="cm", scale=2.0)["report"]          # cm file, rock twice as big
    assert abs(r2["volume_m3"] / (8 * r1["volume_m3"]) - 1) < 0.03
    pv, pf = S.decimate_preview(v, f, max_faces=20000)
    assert len(pf) <= 20000 and len(pv) < len(v)


def test_manual_cut():
    v, f, _ = S.synthetic_scan()
    r_auto = S.clean(v, f, ".glb")["report"]
    r_hi = S.clean(v, f, ".glb", ground=0.03)["report"]                      # cut 30 mm up
    assert r_hi["size_m"][2] < r_auto["size_m"][2] - 0.02
    assert r_hi["volume_m3"] < r_auto["volume_m3"]
    with pytest.raises(ValueError):
        S.clean(v, f, ".glb", ground=5.0)                                     # above the whole scan


# ---------------------------------------------------------------------- endpoints
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from reefflow.ui.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path / "ws", start_worker=False))


def _glb_bytes(v, f):
    import trimesh
    return trimesh.Trimesh(v, f, process=False).export(file_type="glb")


def test_scan_endpoints(client):
    c = client
    pid = c.post("/api/projects", json={"name": "scan", "template": "blank"}).json()["id"]
    v, f, _ = S.synthetic_scan()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:                                      # app-style ZIP export
        z.writestr("rock/texture.jpg", b"not a mesh")
        z.writestr("rock/scan.glb", _glb_bytes(v, f))
    r = c.post(f"/api/projects/{pid}/scans", files={"file": ("rock_scan.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 200, r.text
    sid = r.json()["scan"]["id"]
    info = r.json()["info"]
    assert info["up"] == "+y" and info["ground"] is not None
    r = c.post(f"/api/projects/{pid}/scans/{sid}/preview", json={"up": "+z"})
    assert r.status_code == 200
    meta = json.loads(r.headers["X-Meta"])
    assert meta["up"] == "+z" and meta["up_reason"] == "set by you"
    nv, nf = np.frombuffer(r.content[:8], "<u4")
    assert len(r.content) == 8 + 12 * int(nv) + 12 * int(nf)
    r = c.post(f"/api/projects/{pid}/scans/{sid}/clean", json={"up": "auto", "ground": "auto"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["report"]["watertight"] and j["element"]["params"]["file"].startswith("meshes/")
    # the cleaned STL imports like any mesh element
    r = c.post(f"/api/projects/{pid}/element_mesh", json={"element": j["element"]})
    assert r.status_code == 200
    bad = c.post(f"/api/projects/{pid}/scans", files={"file": ("x.usdz", b"abc", "model/vnd.usdz+zip")})
    assert bad.status_code == 400 and "USDZ" in bad.json()["detail"]
    junk = c.post(f"/api/projects/{pid}/scans", files={"file": ("x.obj", b"this is not an obj", "text/plain")})
    assert junk.status_code == 400
    assert c.delete(f"/api/projects/{pid}/scans/{sid}").status_code == 200
    assert c.post(f"/api/projects/{pid}/scans/{sid}/preview", json={}).status_code == 404
