"""Rock sculpt tool, server side (reefflow.ui.sculpt + routes).

The browser does the brushing (static/js/sculpt.js); here we check the grid the
server hands it, the wire format, drafts/resume, and that applying a sculpt gives
the project a rockscape the CFD tank model voxelizes to the same rock volume.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import reefflow  # noqa: F401  (Taichi offline setting first)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from reefflow import project as PR  # noqa: E402
from reefflow import tank as TK  # noqa: E402
from reefflow.ui import sculpt as SC  # noqa: E402
from reefflow.ui.server import create_app  # noqa: E402

TANK = PR.Tank(length=0.6, width=0.3, height=0.35, glass=0.006, bottom_glass=0.006, water_gap=0.025,
               sand_depth=0.03, external=(0.612, 0.312, 0.356))


def test_grid_layout():
    g = SC.grid_for(TANK, 0.004)
    nx, ny, nz = g["shape"]
    x = g["origin"][0] + g["dx"] * np.arange(nx)
    z = g["origin"][2] + g["dx"] * np.arange(nz)
    assert nx == 150 and ny == 75
    assert x[0] > 0 and x[-1] < TANK.length and abs(x[0] - (TANK.length - x[-1])) < 1e-12   # centred
    ks = g["k_sand"]
    assert z[ks] > TANK.sand_depth > z[ks - 1] and ks == 2                                   # 2 buried layers
    assert z[-1] + g["dx"] >= TANK.height
    with pytest.raises(ValueError):
        SC.grid_for(TANK, 0.0005)


def test_wire_roundtrip_and_bury():
    g = SC.grid_for(TANK, 0.006)
    rng = np.random.default_rng(3)
    D = rng.integers(0, 256, g["shape"], dtype=np.uint8)
    assert np.array_equal(SC.from_wire(SC.to_wire(D), g["shape"]), D)
    # x is fastest on the wire: byte 1 is node (1, 0, 0)
    assert SC.to_wire(D)[1] == D[1, 0, 0] and SC.to_wire(D)[g["shape"][0]] == D[0, 1, 0]
    B = SC.bury(D.copy(), g["k_sand"])
    assert np.array_equal(B[:, :, 0], B[:, :, g["k_sand"]])
    with pytest.raises(ValueError):
        SC.from_wire(b"\x00" * 10, g["shape"])


def test_sphere_volume_and_mesh():
    """A 50 mm sphere: supersampled start field and its mesh reproduce the volume."""
    import trimesh
    s = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    s.apply_translation([0.3, 0.15, 0.17])
    g = SC.grid_for(TANK, 0.004)
    D = SC.field_from_meshes([(s.vertices, s.faces)], g)
    V = s.volume
    vox = (D >= SC.ISO).sum() * g["dx"] ** 3
    frac = D.astype(float).sum() / 255 * g["dx"] ** 3
    assert ((D > 0) & (D < 255)).any()                                  # partial cells at the surface
    v, f, m = SC.to_mesh(D, g)
    assert m.is_watertight and m.volume > 0
    print(f"sphere {V * 1e6:.1f} cm3: nodes {vox * 1e6:.1f}, fraction sum {frac * 1e6:.1f}, mesh {m.volume * 1e6:.1f}")
    assert abs(frac / V - 1) < 0.005
    assert abs(vox / V - 1) < 0.03
    assert abs(m.volume / V - 1) < 0.03


def test_empty_grid_cannot_be_applied():
    g = SC.grid_for(TANK, 0.006)
    with pytest.raises(ValueError, match="no rock"):
        SC.to_mesh(np.zeros(g["shape"], np.uint8), g)


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "ws", start_worker=False)
    with TestClient(app) as c:
        c.ws = app.state.ws
        yield c


def _start(client, pid, **body):
    r = client.post(f"/api/projects/{pid}/sculpt/start", json=body)
    assert r.status_code == 200, r.text
    meta = json.loads(r.headers["X-Meta"])
    return meta, SC.from_wire(r.content, meta["grid"]["shape"])


def test_sculpt_session_end_to_end(client):
    pid = client.post("/api/projects", json={"name": "s", "template": "demo"}).json()["id"]
    pdir = client.ws.project_dir(pid)
    meta, D0 = _start(client, pid, dx=0.005)
    g = meta["grid"]
    assert meta["source"].startswith("voxelized") and not meta["draft"]
    v0 = (D0[:, :, g["k_sand"]:] >= SC.ISO).sum() * g["dx"] ** 3
    assert abs(meta["rock_volume_m3"] - v0) < 1e-12
    # the demo rock: compare with the CFD tank model's own voxelization at the same cell size
    P = PR.load(pdir / "project.yaml")
    m0 = TK.build(P, 0.005)
    print(f"demo rock: sculpt grid {v0 * 1e3:.3f} L, CFD model {m0.info['rock_volume_m3'] * 1e3:.3f} L")
    assert abs(v0 / m0.info["rock_volume_m3"] - 1) < 0.03

    # "brush": open a 30 mm cave through the left island (what a Push stroke does)
    X = g["origin"][0] + g["dx"] * np.arange(g["shape"][0])
    Y = g["origin"][1] + g["dx"] * np.arange(g["shape"][1])
    Z = g["origin"][2] + g["dx"] * np.arange(g["shape"][2])
    I, J, K = np.meshgrid(X, Y, Z, indexing="ij", sparse=True)
    cave = (I - 0.30) ** 2 + (J - 0.20) ** 2 + (K - 0.11) ** 2 < 0.03 ** 2
    D1 = D0.copy()
    D1[cave] = 0
    removed = ((D0 >= SC.ISO) & cave).sum() * g["dx"] ** 3
    assert removed > 50e-6                                             # the cave really cuts rock

    # draft -> resumed exactly
    r = client.put(f"/api/projects/{pid}/sculpt/draft", params={"dx": g["dx"], "base_sig": meta["base_sig"]},
                   content=SC.to_wire(D1))
    assert r.status_code == 200, r.text
    meta2, D2 = _start(client, pid)
    assert meta2["source"] == "resumed" and meta2["draft"] and np.array_equal(D2, D1)

    # apply -> project rockscape is the sculpt STL; the CFD model sees the cave
    r = client.post(f"/api/projects/{pid}/sculpt/apply", params={"dx": g["dx"]}, content=SC.to_wire(D1))
    assert r.status_code == 200, r.text
    j = r.json()
    rep = j["report"]
    assert rep["watertight"] and j["project"]["validation"]["ok"], j["project"]["validation"]
    raw = client.ws.read_raw(pid)
    assert raw["rockscape"][0]["file"] == SC.SCULPT_STL and (pdir / SC.SCULPT_STL).is_file()
    els = raw["designer"]["elements"]
    assert len(els) == 1 and els[0]["type"] == "mesh" and not j["project"]["designer_stale"]
    m1 = TK.build(PR.load(pdir / "project.yaml"), 0.005)
    v1 = (D1[:, :, g["k_sand"]:] >= SC.ISO).sum() * g["dx"] ** 3
    print(f"after the cave: sculpt grid {v1 * 1e3:.3f} L, report {rep['volume_voxel_m3'] * 1e3:.3f} L, "
          f"CFD model {m1.info['rock_volume_m3'] * 1e3:.3f} L (cave removed {removed * 1e6:.1f} cm3)")
    assert abs(rep["volume_voxel_m3"] - v1) < 1e-12
    assert abs(m1.info["rock_volume_m3"] / v1 - 1) < 0.03
    assert m1.info["rock_volume_m3"] < m0.info["rock_volume_m3"] - 0.5 * removed

    # sculpting again resumes the applied grid losslessly (no re-voxelization)
    meta3, D3 = _start(client, pid)
    assert meta3["source"] == "resumed" and not meta3["draft"] and np.array_equal(D3, D1)
    # 'start over' re-voxelizes the applied STL: same rock within a few per cent
    meta4, D4 = _start(client, pid, fresh=True)
    v4 = (D4[:, :, g["k_sand"]:] >= SC.ISO).sum() * g["dx"] ** 3
    assert meta4["source"].startswith("voxelized") and abs(v4 / v1 - 1) < 0.02
    # a different resolution never resumes
    meta5, _ = _start(client, pid, dx=0.006)
    assert meta5["source"].startswith("voxelized") and meta5["grid"]["dx"] == 0.006


def test_changed_rockscape_is_not_resumed(client):
    pid = client.post("/api/projects", json={"name": "s2", "template": "demo"}).json()["id"]
    meta, D = _start(client, pid, dx=0.006)
    client.put(f"/api/projects/{pid}/sculpt/draft", params={"dx": 0.006, "base_sig": meta["base_sig"]},
               content=SC.to_wire(np.zeros_like(D)))
    raw = client.ws.read_raw(pid)
    raw["rockscape"][0]["translate"] = ["10 mm", "0 mm", "0 mm"]            # the rock moved since the draft
    client.ws.write_raw(pid, raw)
    meta2, D2 = _start(client, pid, dx=0.006)
    assert meta2["source"].startswith("voxelized") and D2.any()


def test_bad_uploads_rejected(client):
    pid = client.post("/api/projects", json={"name": "s3", "template": "demo"}).json()["id"]
    r = client.post(f"/api/projects/{pid}/sculpt/apply", params={"dx": 0.006}, content=b"\x00" * 100)
    assert r.status_code == 400 and "expected" in r.json()["detail"]
    g = SC.grid_for(PR.load(client.ws.project_dir(pid) / "project.yaml").tank, 0.006)
    r = client.post(f"/api/projects/{pid}/sculpt/apply", params={"dx": 0.006}, content=bytes(np.prod(g["shape"])))
    assert r.status_code == 400 and "no rock" in r.json()["detail"]
