"""Phase 5 browser UI: FastAPI endpoints (TestClient), rockscape union, results slicing.

Fast tests need no GPU: the results endpoints are exercised on a synthetic run
directory built with the core API (tank.build on a coarse grid + an analytic
velocity field, then the core analyze()). The end-to-end GPU run through the job
worker is marked slow.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest

import reefflow  # noqa: F401  (Taichi offline setting first)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from reefflow import project as PR  # noqa: E402
from reefflow import tank as TK  # noqa: E402
from reefflow.ui import editor_model as EM  # noqa: E402
from reefflow.ui import rockgen as RG  # noqa: E402
from reefflow.ui.server import STATIC, create_app  # noqa: E402

TINY_YAML = """reefflow_project: 1
name: tiny
display_units: US
tank:
  external: {length: 24 in, width: 12 in, height: 14 in}
  glass_thickness: 1/4 in
  water_level_below_rim: 1 in
  sand_depth: 1 in
rockscape: []
overflow: {corner: back-right, width: 4 in, depth: 4 in, crest_below_surface: 1/2 in}
pumps:
  - {name: PH, archetype: narrow_jet_propeller, rated_flow: 800 GPH, position: [2 in, 6 in, 9 in], aim: [1, 0, 0]}
  - {name: ret, archetype: return_nozzle, rated_flow: 300 GPH, position: [18 in, 9 in, 11 in], aim: [-1, -0.5, -0.2]}
simulation: {preset: preview, spinup: 1 s, average: 1 s}
"""
ROCK = {"id": "r1", "type": "rock", "name": "rock", "position": [0.30, 0.15, 0.0254], "rotation": [0, 0, 20],
        "scale": [1, 1, 1], "params": {"size": [0.15, 0.12, 0.12], "seed": 4, "lobes": 2, "roughness": 0.2, "bury": 0.15}}


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "ws", start_worker=False)
    with TestClient(app) as c:
        c.ws = app.state.ws
        c.jobs = app.state.jobs
        yield c


def _tiny(client) -> str:
    pid = client.post("/api/projects", json={"name": "tiny", "template": "blank"}).json()["id"]
    r = client.put(f"/api/projects/{pid}/yaml", json={"text": TINY_YAML})
    assert r.status_code == 200 and r.json()["validation"]["ok"], r.text
    return pid


def _build_rock(client, pid, elements, res=0.005):
    ed = client.get(f"/api/projects/{pid}").json()["editor"]
    ed["designer"]["elements"] = elements
    return client.post(f"/api/projects/{pid}/rockscape/build", json={"editor": ed, "resolution": res})


# ----------------------------------------------------------------------
# static files, launcher
# ----------------------------------------------------------------------
def test_static_files_present():
    for f in ("index.html", "css/app.css", "js/main.js", "js/util.js", "js/viewport.js", "js/designer.js",
              "js/equipment.js", "js/run.js", "js/results.js",
              "vendor/three/three.module.js", "vendor/three/LICENSE", "vendor/three/VERSION.txt",
              "vendor/three/addons/controls/OrbitControls.js", "vendor/three/addons/controls/TransformControls.js",
              "vendor/three/addons/loaders/PLYLoader.js", "vendor/three/addons/loaders/STLLoader.js"):
        assert (STATIC / f).is_file(), f
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    import re
    ext = [u for u in re.findall(r'(?:src|href)="([^"]+)"', html) + re.findall(r'"(https?://[^"]+)"', html)
           if u.startswith(("http:", "https:", "//"))]
    assert not ext, f"external assets referenced (must be vendored for offline use): {ext}"
    for js in (STATIC / "js").glob("*.js"):
        txt = js.read_text(encoding="utf-8")
        assert "cdn." not in txt and "unpkg" not in txt, js.name


def test_index_and_static_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "importmap" in r.text
    assert client.get("/static/vendor/three/three.module.js").status_code == 200
    assert client.get("/api/health").json()["ok"]


def test_free_port():
    from reefflow.ui.launcher import free_port
    p = free_port()
    assert 0 < p < 65536


def test_library(client):
    lib = client.get("/api/library").json()
    assert {"narrow_jet_propeller", "return_nozzle"} <= set(lib["pumps"])
    assert lib["pumps"]["narrow_jet_propeller"]["outlet"]["diameter"] == pytest.approx(0.032)
    assert set(lib["presets"]) == {"preview", "standard", "fine"}
    assert "SPS" in lib["coral_profiles"] and lib["units"]["in"][1] == pytest.approx(0.0254)
    assert lib["fixtures"]
    cm = client.get("/api/colormaps").json()
    assert len(cm["viridis"]) == 256 * 3


# ----------------------------------------------------------------------
# projects
# ----------------------------------------------------------------------
def test_project_crud_and_roundtrip(client):
    pid = client.post("/api/projects", json={"name": "Demo 75", "template": "demo"}).json()["id"]
    assert any(p["id"] == pid for p in client.get("/api/projects").json()["projects"])
    p = client.get(f"/api/projects/{pid}").json()
    assert p["validation"]["ok"], p["validation"]
    ed = p["editor"]
    assert ed["tank"]["length"] == pytest.approx(48 * 0.0254)
    assert len(ed["pumps"]) == 3 and ed["designer"]["elements"][0]["type"] == "mesh"
    assert not p["designer_stale"]
    # save unchanged editor model -> same physics
    p2 = client.put(f"/api/projects/{pid}", json={"editor": ed}).json()
    assert p2["validation"]["ok"]
    P1 = PR.load(client.ws.project_path(pid))
    raw0 = PR.load(Path(reefflow.__file__).parents[1] / "examples" / "demo_75gal" / "demo_75gal.yaml")
    assert P1.tank.length == pytest.approx(raw0.tank.length)
    for a, b in zip(P1.pumps, raw0.pumps):
        assert a.delivered_flow == pytest.approx(b.delivered_flow, rel=1e-6)
        assert np.allclose(a.position, b.position, atol=1e-6) and np.allclose(a.aim, b.aim, atol=1e-3)
    # copy + delete
    cid = client.post(f"/api/projects/{pid}/copy", json={"name": "copy"}).json()["id"]
    assert client.get(f"/api/projects/{cid}").status_code == 200
    assert client.delete(f"/api/projects/{cid}").json()["ok"]
    assert client.get(f"/api/projects/{cid}").status_code == 404
    assert client.get("/api/projects/..%2Fetc").status_code in (400, 404)


def test_units_si_roundtrip(client):
    pid = client.post("/api/projects", json={"name": "si", "template": "demo"}).json()["id"]
    ed = client.get(f"/api/projects/{pid}").json()["editor"]
    ed["display_units"] = "SI"
    p = client.put(f"/api/projects/{pid}", json={"editor": ed}).json()
    assert "1219.2 mm" in p["yaml"] and "L/h" in p["yaml"] and " GPH" not in p["yaml"]
    assert p["editor"]["tank"]["length"] == pytest.approx(ed["tank"]["length"])
    assert p["editor"]["pumps"][1]["rated_flow"] == pytest.approx(ed["pumps"][1]["rated_flow"])


def test_validation_errors(client):
    pid = _tiny(client)
    ed = client.get(f"/api/projects/{pid}").json()["editor"]
    bad = json.loads(json.dumps(ed))
    bad["pumps"][0]["position"][2] = 2.0                      # above the water
    v = client.post(f"/api/projects/{pid}/validate", json={"editor": bad}).json()
    assert not v["ok"] and "not inside the water volume" in v["errors"][0]
    bad = json.loads(json.dumps(ed))
    bad["tank"]["height"] = 0.03                              # no water column
    assert not client.post(f"/api/projects/{pid}/validate", json={"editor": bad}).json()["ok"]
    bad = json.loads(json.dumps(ed))
    bad["overflow"] = None                                    # return pump needs an overflow
    v = client.post(f"/api/projects/{pid}/validate", json={"editor": bad}).json()
    assert any("overflow" in e for e in v["errors"])
    # saving an invalid project is allowed (work in progress) but reported, and runs are refused
    s = client.put(f"/api/projects/{pid}", json={"editor": bad}).json()
    assert not s["validation"]["ok"]
    r = client.post(f"/api/projects/{pid}/runs", json={"preset": "preview"})
    assert r.status_code == 400 and "errors" in r.json()["detail"]
    assert client.put(f"/api/projects/{pid}/yaml", json={"text": "tank: [unclosed"}).status_code == 400


def test_lighting_adapter_roundtrip(client):
    pid = client.post("/api/projects", json={"name": "lights", "template": "demo"}).json()["id"]
    ed = client.get(f"/api/projects/{pid}").json()["editor"]
    fx = ed["lighting"]["fixtures"]
    assert len(fx) == 2 and fx[0]["height"] == pytest.approx(8 * 0.0254)
    fx[0]["tilt_x"] = 10.0
    fx[1]["intensity"] = 60.0
    p = client.put(f"/api/projects/{pid}", json={"editor": ed}).json()
    raw = client.ws.read_raw(pid)
    assert raw["lighting"]["fixtures"][0]["tilt_x"] == 10.0 and raw["lighting"]["fixtures"][1]["intensity"] == 60.0
    assert raw["lighting"]["fixtures"][0]["channels"] == {"white": 80}           # extra keys preserved
    pp = raw["lighting"].get("photoperiod") or {}
    assert True not in pp and False not in pp                                     # YAML 1.1 on/off keys fixed
    assert p["validation"]["ok"], p["validation"]["errors"]
    from reefflow.ui.lighting_adapter import beam_axis
    assert np.allclose(beam_axis(0, 0), [0, 0, -1])
    assert beam_axis(30, 0)[1] > 0 and beam_axis(0, 30)[0] > 0                    # Phase-4 sign convention


# ----------------------------------------------------------------------
# rockscape designer
# ----------------------------------------------------------------------
@pytest.mark.parametrize("etype", ["rock", "slab", "pillar", "arch", "bridge"])
def test_element_mesh_endpoint(client, etype):
    pid = _tiny(client)
    r = client.post(f"/api/projects/{pid}/element_mesh", json={"element": {"type": etype, "params": {"seed": 3}}})
    assert r.status_code == 200
    nv, nf = np.frombuffer(r.content[:8], "<u4")
    assert nv > 100 and nf > 100 and len(r.content) == 8 + 12 * nv + 12 * nf


def test_rock_union_watertight(client):
    import trimesh
    pid = _tiny(client)
    slab = {"id": "s1", "type": "slab", "name": "shelf", "position": [0.36, 0.15, 0.12], "rotation": [0, 10, 0],
            "scale": [1, 1, 1], "params": {"size": [0.12, 0.08, 0.02], "seed": 2, "roughness": 0.1}}
    r = _build_rock(client, pid, [ROCK, slab])
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["watertight"] and rep["volume_m3"] > 5e-4
    proj = r.json()["project"]
    assert proj["editor"]["rockscape"][0]["file"] == "rockscape.stl"
    assert proj["validation"]["ok"] and not proj["designer_stale"]
    stl = client.get(f"/api/projects/{pid}/rockscape.stl")
    assert stl.status_code == 200
    m = trimesh.load(io.BytesIO(stl.content), file_type="stl")
    assert m.is_watertight and abs(m.volume) * 1e-9 == pytest.approx(rep["volume_m3"], rel=0.02)   # STL in mm
    # the core tank model voxelizes it
    P = PR.load(client.ws.project_path(pid))
    model = TK.build(P, 0.01)
    assert model.info["rock_nodes"] > 100
    # moving an element without rebuilding blocks runs
    ed = proj["editor"]
    ed["designer"]["elements"][0]["position"][0] += 0.05
    s = client.put(f"/api/projects/{pid}", json={"editor": ed}).json()
    assert s["designer_stale"]
    r = client.post(f"/api/projects/{pid}/runs", json={"preset": "preview"})
    assert r.status_code == 409


def test_union_issue_flags(tmp_path):
    P = PR.parse(__import__("yaml").safe_load(TINY_YAML), tmp_path / "p.yaml")
    t = P.tank
    touching = {**ROCK, "id": "g", "position": [0.06, 0.15, t.sand_depth]}          # 150 mm rock at x = 60 mm
    floating = {"id": "f", "type": "slab", "position": [0.40, 0.15, 0.20], "rotation": [0, 0, 0], "scale": [1, 1, 1],
                "params": {"size": [0.06, 0.06, 0.02]}}
    res = RG.union([touching, floating], t, tmp_path, h=0.005)
    kinds = {i["kind"]: i for i in res["issues"]}
    assert "glass_contact" in kinds and kinds["glass_contact"]["elements"] == ["g"]
    assert "floating" in kinds and kinds["floating"]["elements"] == ["f"]
    over = {"id": "o", "type": "bridge", "position": [0.30, 0.15, t.sand_depth], "rotation": [0, 0, 0], "scale": [1, 1, 1],
            "params": {"span": 0.04, "height": 0.10}}
    arm = {"id": "a", "type": "slab", "position": [0.47, 0.15, t.sand_depth + 0.07], "rotation": [0, 0, 0], "scale": [1, 1, 1],
           "params": {"size": [0.30, 0.06, 0.04]}}                                # cantilever off a narrow bridge
    res = RG.union([over, arm], t, tmp_path, h=0.005)
    assert any(i["kind"] == "overhang" for i in res["issues"]), res["issues"]


def test_mesh_import_and_union(client, tmp_path):
    from reefflow import geometry as G
    pid = _tiny(client)
    v, f = G.sdf_to_mesh(lambda P: np.linalg.norm(P, axis=-1) - 0.04, [-0.05] * 3, [0.05] * 3, 0.004)
    G.save_stl(tmp_path / "ball.stl", v, f, unit="mm")
    r = client.post(f"/api/projects/{pid}/meshes", files={"file": ("ball.stl", (tmp_path / "ball.stl").read_bytes())},
                    data={"unit": "mm"})
    assert r.status_code == 200, r.text
    el = r.json()["element"]
    assert el["params"]["file"].startswith("meshes/") and r.json()["report"]["watertight"]
    el.update(id="m1", position=[0.25, 0.15, 0.06], rotation=[0, 0, 0], scale=[1, 1, 1])
    r = _build_rock(client, pid, [el])
    assert r.status_code == 200 and r.json()["report"]["watertight"]
    assert r.json()["report"]["volume_m3"] == pytest.approx(4 / 3 * np.pi * 0.04 ** 3, rel=0.08)
    bad = client.post(f"/api/projects/{pid}/meshes", files={"file": ("x.txt", b"hello")}, data={"unit": "mm"})
    assert bad.status_code == 400


def test_estimate_endpoint(client):
    pid = _tiny(client)
    r = client.post(f"/api/projects/{pid}/estimate", json={"preset": "preview"})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["estimate"]["cells"] == int(np.prod(e["info"]["shape"])) and e["estimate"]["steps"] > 0
    assert e["dx_m"] == pytest.approx(0.01)
    assert any("momentum flux" in n for n in e["notes"])                         # equivalent-outlet notes
    r = client.post(f"/api/projects/{pid}/estimate", json={"dx": 0.02})
    assert r.json()["dx_m"] == pytest.approx(0.02)


def test_import_project_yaml(client):
    r = client.post("/api/projects/import", files={"project_file": ("tiny.yaml", TINY_YAML.encode())})
    assert r.status_code == 200 and r.json()["validation"]["ok"]


# ----------------------------------------------------------------------
# jobs (no GPU)
# ----------------------------------------------------------------------
def test_job_queue_cancel_and_failure(client, tmp_path):
    pid = _tiny(client)
    rid, d = client.ws.new_run(pid, "x")
    job = client.jobs.submit("run", pid, rid, d, {"project_file": str(tmp_path / "missing.yaml")})
    assert client.get(f"/api/jobs/{job.id}").json()["status"] == "queued"
    assert client.post(f"/api/jobs/{job.id}/cancel").json()["status"] == "cancelled"
    # a failing job (missing project file) reports the error instead of crashing anything
    rid2, d2 = client.ws.new_run(pid, "y")
    job2 = client.jobs.submit("run", pid, rid2, d2, {"project_file": str(tmp_path / "missing.yaml")})
    client.jobs.run_now(job2)
    j = client.get(f"/api/jobs/{job2.id}").json()
    assert j["status"] == "failed" and "missing.yaml" in j["error"]
    ev = client.get(f"/api/jobs/{job2.id}/events").text
    assert "event: end" in ev and '"failed"' in ev
    runs = {r["id"]: r for r in client.get(f"/api/projects/{pid}/runs").json()["runs"]}
    assert runs[rid]["status"] == "cancelled" and runs[rid2]["status"] == "failed"


# ----------------------------------------------------------------------
# results on a synthetic run directory (core API, no GPU)
# ----------------------------------------------------------------------
def _synthetic_run(client):
    pid = _tiny(client)
    assert _build_rock(client, pid, [ROCK]).status_code == 200
    rid, d = client.ws.new_run(pid, "synthetic")
    pf = client.ws.snapshot_project(pid, d)
    P = PR.load(pf)
    m = TK.build(P, 0.02)
    X, Y, Z = np.meshgrid(*[m.node_coords(a) for a in range(3)], indexing="ij")
    L, W = P.tank.length, P.tank.width
    u = 0.1 * np.stack([-np.sin(np.pi * X / L) * np.cos(np.pi * Y / W),
                        np.cos(np.pi * X / L) * np.sin(np.pi * Y / W) * L / W, 0 * Z], -1)
    u[~m.fluid_mask] = 0
    speed = np.linalg.norm(u, axis=-1) + 0.01
    tke = 0.05 * speed ** 2
    rey = np.zeros(u.shape[:3] + (6,))
    rey[..., :3] = (2 / 3 * tke)[..., None]
    np.savez_compressed(d / "fields.npz", mean_u=u.astype(np.float32), mean_speed=speed.astype(np.float32),
                        tke=tke.astype(np.float32), reynolds=rey.astype(np.float32), tags=m.tags,
                        flags=m.domain.flags, origin=m.origin, dx=m.dx)
    info = {"project": P.name, "project_file": str(pf), "dx_m": 0.02, "metrics_stat_uncertainty": {},
            "sim": {k: v for k, v in P.sim.items() if not k.endswith(("_s", "_ms", "_m"))}}
    (d / "run.json").write_text(json.dumps(info), encoding="utf-8")
    from reefflow.analysis.analyze import analyze
    analyze(d, progress=lambda s: None)
    return pid, rid, m, speed


def test_results_endpoints_synthetic(client):
    pid, rid, m, speed = _synthetic_run(client)
    s = client.get(f"/api/runs/{pid}/{rid}").json()
    assert s["layers"]["speed"] and s["layers"]["dead"] and s["layers"]["age"] and not s["layers"]["par"]
    assert s["grid"]["shape"] == list(m.shape) and set(s["profiles"]) >= {"SPS", "LPS"}
    assert "metrics" in s["analysis"]
    # plan slice through k = 5 matches the field (NaN in solids)
    z = m.origin[2] + 5 * m.dx
    r = client.get(f"/api/runs/{pid}/{rid}/slice", params={"field": "speed", "axis": "z", "pos": z})
    meta = json.loads(r.headers["X-Meta"])
    data = np.frombuffer(r.content, "<f4").reshape(meta["shape"])
    assert meta["index"] == 5 and meta["shape"] == [m.shape[1], m.shape[0]]
    expect = np.where(m.fluid_mask[:, :, 5], speed[:, :, 5], np.nan).T
    assert np.allclose(data, expect, equal_nan=True, atol=1e-6)
    assert meta["max"] == pytest.approx(np.nanmax(expect), rel=1e-5)
    for field, axis in (("age", "y"), ("dead", "x"), ("ti", "z"), ("umag", "y")):
        r = client.get(f"/api/runs/{pid}/{rid}/slice", params={"field": field, "axis": axis, "pos": 0.1})
        assert r.status_code == 200, (field, r.text)
        meta = json.loads(r.headers["X-Meta"])
        assert len(r.content) == 4 * meta["shape"][0] * meta["shape"][1]
    assert client.get(f"/api/runs/{pid}/{rid}/slice", params={"field": "par"}).status_code == 404
    assert client.get(f"/api/runs/{pid}/{rid}/slice", params={"field": "nope"}).status_code == 400
    v = client.get(f"/api/runs/{pid}/{rid}/vectors", params={"axis": "z", "pos": z})
    n = json.loads(v.headers["X-Meta"])["n"]
    assert n > 0 and len(v.content) == 16 * n
    sh = client.get(f"/api/runs/{pid}/{rid}/sand_shear")
    assert json.loads(sh.headers["X-Meta"])["shape"] == [m.shape[1], m.shape[0]]
    sc = client.get(f"/api/runs/{pid}/{rid}/scores", params={"profile": "SPS"})
    assert json.loads(sc.headers["X-Meta"])["n"] > 0
    rm = client.get(f"/api/runs/{pid}/{rid}/rock_mesh")
    assert np.frombuffer(rm.content[:8], "<u4")[0] > 100
    assert client.get(f"/api/runs/{pid}/{rid}/file/report.html").status_code == 200
    assert client.get(f"/api/runs/{pid}/{rid}/file/rock_score_SPS.ply").status_code == 200
    assert client.get(f"/api/runs/{pid}/{rid}/file/..%2Frun.json").status_code in (400, 404)
    assert client.get(f"/api/runs/{pid}/{rid}/project").json()["editor"]["name"] == "tiny"
    # the run snapshot keeps the run loadable after the project changes
    ed = client.get(f"/api/projects/{pid}").json()["editor"]
    ed["pumps"][0]["position"][0] += 0.05
    client.put(f"/api/projects/{pid}", json={"editor": ed})
    from reefflow.analysis.load import load_run
    load_run(client.ws.run_dir(pid, rid))
    # comparison mode (run vs itself is a valid smoke test: zero deltas)
    c = client.post("/api/compare", json={"a": [pid, rid], "b": [pid, rid]})
    assert c.status_code == 200, c.text
    assert all(abs(row["delta"]) < 1e-9 for row in c.json()["result"]["rows"])
    assert client.get(c.json()["html"]).status_code == 200 and client.get(c.json()["image"]).status_code == 200
    # PAR hook: a par.npz in the run directory enables the PAR layer
    par = np.where(m.fluid_mask, 300.0, np.nan).astype(np.float32)
    np.savez_compressed(client.ws.run_dir(pid, rid) / "par.npz", meter=par, normal=par, label=np.array("test"))
    assert client.get(f"/api/runs/{pid}/{rid}").json()["layers"]["par"]
    r = client.get(f"/api/runs/{pid}/{rid}/slice", params={"field": "par", "axis": "z", "pos": z})
    assert json.loads(r.headers["X-Meta"])["max"] == pytest.approx(300.0)
    assert client.delete(f"/api/runs/{pid}/{rid}").json()["ok"]


# ----------------------------------------------------------------------
# end to end through the job worker (GPU / Taichi; tiny case)
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_run_job_end_to_end(client):
    pid = _tiny(client)
    assert _build_rock(client, pid, [ROCK]).status_code == 200
    r = client.post(f"/api/projects/{pid}/runs", json={"preset": "preview", "dx": 0.02, "spinup": 0.3, "average": 0.3})
    assert r.status_code == 200, r.text
    job = client.jobs.get(r.json()["job"]["id"])
    client.jobs.run_now(job)
    j = client.get(f"/api/jobs/{job.id}").json()
    assert j["status"] == "done", (j["error"], j["traceback"])
    assert any("[analyze] done" in ln for ln in j["lines"])
    s = client.get(f"/api/runs/{pid}/{r.json()['run_id']}").json()
    assert s["layers"]["speed"] and s["layers"]["coral"] and s["status"]["status"] == "analysed"
    r = client.get(f"/api/runs/{pid}/{r.json()['run_id']}/slice", params={"field": "speed", "axis": "y", "pos": 0.15})
    assert np.nanmax(np.frombuffer(r.content, "<f4")) > 0
