"""Photos tab (photo reconstruction, spec Feature 1): endpoints end to end on small synthetic
photos of the demo rock rendered with reefflow.photo.synthetic (no real photos needed)."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

import reefflow  # noqa: F401

pytest.importorskip("fastapi")
cv2 = pytest.importorskip("cv2")
from fastapi.testclient import TestClient  # noqa: E402

from reefflow import geometry as G  # noqa: E402
from reefflow import project as PR  # noqa: E402
from reefflow.photo import TankGeometry, View, mask_from_png, mask_iou, mask_to_png, recommended_points  # noqa: E402
from reefflow.photo import synthetic as SY  # noqa: E402
from reefflow.ui.server import create_app  # noqa: E402

SIZE = (640, 480)
F_PX = 480.0


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """Demo tank + rock, three synthetic photos (top, front, left), exact clicks, true masks."""
    from reefflow import demo
    d = tmp_path_factory.mktemp("scene")
    p = demo.write_demo(d)
    tank = TankGeometry.from_project(p)
    v, f = G.load_mesh(d / "rock.stl", unit="mm")
    out = {}
    for k, cam in SY.default_cameras(tank, size=SIZE, f_px=F_PX).items():
        view = View(k, cam, tank)
        img, gt = SY.render_view(view, v, f, seed=7)
        ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out[k] = {"png": png.tobytes(), "gt": gt, "clicks": SY.true_clicks(view, [q["name"] for q in recommended_points(k)])}
    return {"views": out, "tank": tank, "rock": (v, f)}


def _client(ws):
    app = create_app(ws, start_worker=False)
    c = TestClient(app)
    c.app_ = app
    return c


def _project_with_photos(c, scene, masks="truth"):
    pid = c.post("/api/projects", json={"name": "photo", "template": "demo"}).json()["id"]
    base = f"/api/projects/{pid}/photos"
    for k, s in scene["views"].items():
        r = c.post(f"{base}/views", files={"file": (f"{k}.png", s["png"])}, data={"kind": k})
        assert r.status_code == 200, r.text
        r = c.put(f"{base}/views/{k}/clicks", json={"clicks": s["clicks"]})
        assert r.status_code == 200, r.text
        if masks == "truth":
            c.put(f"{base}/views/{k}/mask", files={"file": ("m.png", mask_to_png(s["gt"]))})
    return pid, base


def test_capabilities_and_empty_session(tmp_path):
    c = _client(tmp_path / "ws")
    caps = c.get("/api/photo/capabilities").json()
    assert caps["grabcut"] and "grabcut" in caps["segmentation"]
    pid = c.post("/api/projects", json={"name": "p", "template": "demo"}).json()["id"]
    st = c.get(f"/api/projects/{pid}/photos").json()
    assert st["views"] == [] and st["recon"] is None and st["tank"]["length"] == pytest.approx(47 * 0.0254)
    assert [p["name"] for p in st["recommended"]["top"]][:4] == ["top_FL", "top_FR", "top_BR", "top_BL"]
    assert [p["name"] for p in st["recommended"]["front"]] == ["top_FL", "top_FR", "base_FL", "base_FR", "water_BL",
                                                               "water_BR", "seam_BL", "seam_BR"]
    assert st["grid_estimate"]["voxels"] > 1e6


def test_upload_and_calibration(tmp_path, scene):
    c = _client(tmp_path / "ws")
    pid = c.post("/api/projects", json={"name": "p", "template": "demo"}).json()["id"]
    base = f"/api/projects/{pid}/photos"
    s = scene["views"]["front"]
    v = c.post(f"{base}/views", files={"file": ("front.png", s["png"])}, data={"kind": "front"}).json()["view"]
    assert (v["id"], v["width"], v["height"]) == ("front", *SIZE)
    assert c.get(f"{base}/views/front/image").headers["content-type"] == "image/jpeg"
    # second photo of the same side gets its own id (e.g. an oblique view)
    v2 = c.post(f"{base}/views", files={"file": ("f2.png", s["png"])}, data={"kind": "front"}).json()["view"]
    assert v2["id"] == "front-2"
    assert c.delete(f"{base}/views/front-2").status_code == 200
    # too few points -> message, no calibration
    few = dict(list(s["clicks"].items())[:3])
    r = c.put(f"{base}/views/front/clicks", json={"clicks": few}).json()
    assert r["view"]["calibration"] is None and "more point" in r["view"]["calib_error"]
    # coplanar set (rim corners only) with free focal length is refused with an explanation
    rim = {k: u for k, u in s["clicks"].items() if k.startswith("rim")}
    rim.update({k: u for k, u in s["clicks"].items() if k.startswith("water")})
    r = c.put(f"{base}/views/front/clicks", json={"clicks": {k: rim[k] for k in rim if k.startswith(("rim", "water"))}}).json()
    if r["view"]["calibration"] is None:
        assert r["view"]["calib_error"]
    # full set -> sub-pixel residuals and an overlay of all reference points and edges
    r = c.put(f"{base}/views/front/clicks", json={"clicks": s["clicks"]}).json()
    cal = r["view"]["calibration"]
    assert cal["rms_px"] < 0.05 and cal["focal_px"] == pytest.approx(F_PX, rel=1e-3)
    assert len(r["overlay"]["points"]) == 24 and len(r["overlay"]["edges"]) >= 8      # 16 inside + 8 outside corners
    assert {p["name"] for p in cal["seams"]} == {k for k in s["clicks"] if k.startswith("seam_")}
    assert all(p["residual_px"] < 0.05 for p in cal["seams"])
    for n, uv in s["clicks"].items():
        if not n.startswith("seam_"):
            assert np.allclose(r["overlay"]["points"][n], uv, atol=0.05)
    # known focal length mode
    r = c.put(f"{base}/views/front/clicks", json={"focal": {"mode": "px", "value": F_PX}}).json()
    assert r["view"]["calibration"]["param_std"].get("f_px") is None and r["view"]["focal"]["mode"] == "px"
    assert c.put(f"{base}/views/front/clicks", json={"clicks": {"nope_XX": [1, 2]}}).status_code == 400
    assert c.post(f"{base}/views", files={"file": ("x.txt", b"hi")}, data={"kind": "front"}).status_code == 400
    assert c.post(f"{base}/views", files={"file": ("x.png", b"not a png")}, data={"kind": "front"}).status_code == 400
    assert c.post(f"{base}/views", files={"file": ("x.png", s["png"])}, data={"kind": "diagonal"}).status_code == 400


def test_segmentation_and_mask_edits(tmp_path, scene):
    c = _client(tmp_path / "ws")
    pid, base = _project_with_photos(c, scene, masks=None)
    r = c.post(f"{base}/segment", json={})
    assert r.status_code == 200, r.text
    assert set(r.json()["result"]) == {"top", "front", "left"}
    for k in ("top", "front"):
        m = mask_from_png(c.get(f"{base}/views/{k}/mask.png").content)
        assert mask_iou(m, scene["views"][k]["gt"]) > 0.8, k
    v0 = c.get(base).json()["views"][1]
    assert v0["mask_method"] in ("grabcut", "mobile_sam", "sam_vit_b") and not v0["mask_edited"]
    ops = [{"op": "brush", "stroke": [[20, 20], [60, 40]], "radius": 6, "mode": "add"},
           {"op": "polygon", "vertices": [[300, 10], [340, 10], [320, 40]], "mode": "add"}]
    v = c.post(f"{base}/views/front/mask_ops", json={"ops": ops}).json()["view"]
    m = mask_from_png(c.get(f"{base}/views/front/mask.png").content)
    assert m[30, 40] and m[20, 320] and v["mask_edited"]
    v = c.post(f"{base}/views/front/mask_ops", json={"ops": [{"op": "remove_small", "min_area": 2000},
                                                            {"op": "fill_holes"}, {"op": "smooth", "radius": 1}]})
    assert v.status_code == 200
    assert not mask_from_png(c.get(f"{base}/views/front/mask.png").content)[30, 40]
    assert c.post(f"{base}/views/front/mask_undo").status_code == 200
    assert mask_from_png(c.get(f"{base}/views/front/mask.png").content)[30, 40]
    # 'suggest all' keeps masks the user edited
    r = c.post(f"{base}/segment", json={}).json()["result"]
    assert "front" not in r
    assert c.post(f"{base}/views/front/mask_ops", json={"ops": [{"op": "explode"}]}).status_code == 400
    bad = np.zeros((10, 10), bool)
    assert c.put(f"{base}/views/front/mask", files={"file": ("m.png", mask_to_png(bad))}).status_code == 400


def test_carve_edit_compare_use(tmp_path, scene):
    import trimesh
    ws = tmp_path / "ws"
    c = _client(ws)
    pid, base = _project_with_photos(c, scene)
    jobs = c.app_.state.photo_jobs
    st = c.put(f"{base}/settings", json={"dx": 0.01}).json()
    assert st["settings"]["dx"] == 0.01 and set(st["ready_views"]) == {"top", "front", "left"}
    assert c.put(f"{base}/settings", json={"dx": 0.5}).status_code == 400
    j = c.post(f"{base}/carve").json()["job"]
    assert j["id"].startswith("p")
    assert c.post(f"{base}/carve").status_code == 409                      # one carving job per project
    job = jobs.get(j["id"])
    jobs.run_now(job)
    assert job.status == "done", (job.error, job.lines[-5:])
    ev = c.get(f"/api/jobs/{job.id}/events").text
    assert "event: end" in ev and '"done"' in ev
    st = c.get(base).json()
    rec = st["recon"]
    assert not rec["stale"] and rec["stats"]["occupied"] > 1000 and rec["stats"]["views"] == ["top", "front", "left"]
    # the visual hull contains the true rock above the sand: volume is an upper bound
    v, f = scene["rock"]
    from reefflow.photo import VoxelGrid
    grid = VoxelGrid.for_tank(scene["tank"], 0.01)
    truth_v = G.voxelize_mesh(v, f, grid.origin, grid.dx, grid.shape).sum() * grid.dx ** 3
    vol0 = rec["volume"]["volume_m3"]
    assert vol0 > truth_v and vol0 < 3 * truth_v
    assert rec["volume"]["bulk_density_kg_m3"] == 1400.0
    cmp = rec["comparison"]["views"]
    assert all(r["silhouette_iou"] > 0.75 for r in cmp.values()), cmp
    assert c.get(f"{base}/recon/compare/front").status_code == 200
    mesh = c.get(f"{base}/recon/mesh").content
    assert np.frombuffer(mesh[:8], "<u4")[0] > 100
    # edits: carve a tunnel through the left island, then a sphere; volumes drop
    tunnel = {"op": "carve", "shape": {"type": "capsule", "points": [[0.33, 0.05, 0.10], [0.33, 0.40, 0.10]], "radius": 0.03}}
    r = c.put(f"{base}/edits", json={"edits": [tunnel]}).json()
    assert r["volume"]["volume_m3"] < vol0 and r["volume"]["volume_before_edits_m3"] == pytest.approx(vol0)
    arch = {"op": "carve", "shape": {"type": "arch", "base_center": [0.8, 0.22, 0.0508], "axis": [1, 0], "span": 0.06,
                                     "rise": 0.05, "length": 0.2}}
    r2 = c.put(f"{base}/edits", json={"edits": [tunnel, arch]}).json()
    assert r2["volume"]["volume_m3"] < r["volume"]["volume_m3"]
    assert c.put(f"{base}/edits", json={"edits": [{"op": "carve", "shape": {"type": "blob"}}]}).status_code == 400
    bad_arch = {**arch, "shape": {**arch["shape"], "rise": 0.01}}
    assert c.put(f"{base}/edits", json={"edits": [bad_arch]}).status_code == 400
    vol = c.post(f"{base}/volume", json={"bulk_density": 1600, "solid_fraction": 0.8}).json()
    assert vol["weight_kg"] == pytest.approx(r2["volume"]["volume_m3"] * 0.8 * 1600)
    assert c.post(f"{base}/volume", json={"bulk_density": 5}).status_code == 400
    assert c.post(f"{base}/compare").json()["edits"] == 2
    stl = c.get(f"{base}/recon.stl")
    m = trimesh.load(io.BytesIO(stl.content), file_type="stl")
    assert m.is_watertight
    # use as rockscape -> the project runs with it
    u = c.post(f"{base}/use").json()
    assert u["export"]["mesh"]["watertight"]
    p = c.get(f"/api/projects/{pid}").json()
    assert p["validation"]["ok"] and not p["designer_stale"]
    assert p["editor"]["rockscape"][0]["file"] == "rockscape_photo.stl"
    assert p["editor"]["designer"]["elements"][0]["params"]["file"] == "rockscape_photo.stl"
    model = __import__("reefflow.tank", fromlist=["build"]).build(PR.load(c.app_.state.ws.project_path(pid)), 0.01)
    assert model.info["rock_nodes"] > 500
    # changing a mask makes the reconstruction stale
    c.post(f"{base}/views/top/mask_ops", json={"ops": [{"op": "brush", "stroke": [[5, 5], [6, 6]], "radius": 3}]})
    assert c.get(base).json()["recon"]["stale"]
    # everything survives a server restart
    c2 = _client(ws)
    st2 = c2.get(base).json()
    assert [v["id"] for v in st2["views"]] == ["top", "front", "left"]
    assert all(v["calibration"] and v["mask"] for v in st2["views"])
    assert len(st2["edits"]) == 2 and st2["recon"]["stats"]["occupied"] == rec["stats"]["occupied"]
    assert st2["used_as_rockscape"]["file"] == "rockscape_photo.stl"


def test_carve_cancel_and_tank_change(tmp_path, scene):
    from reefflow.ui import photos as PH
    c = _client(tmp_path / "ws")
    pid, base = _project_with_photos(c, scene)
    d = PH.session_dir(c.app_.state.ws.project_dir(pid))
    c.put(f"{base}/settings", json={"dx": 0.02})
    assert PH.carve_job(d, progress=lambda s: None, should_stop=lambda: True) is False
    assert not (d / "recon" / "occupancy.npz").exists()
    assert c.put(f"{base}/edits", json={"edits": []}).status_code == 404          # nothing carved yet
    # changing the tank keeps the clicks and calibrates again with the new size (D-841);
    # the clicks were made on a tank 1 in shorter, so the fit error grows
    rms0 = {v["id"]: v["calibration"]["rms_px"] for v in c.get(base).json()["views"]}
    ed = c.get(f"/api/projects/{pid}").json()["editor"]
    ed["tank"]["length"] += 0.0254
    c.put(f"/api/projects/{pid}", json={"editor": ed})
    st = c.get(base).json()
    assert st["notice"] and all(v["clicks"] and v["calibration"] is not None for v in st["views"])
    assert all(v["calibration"]["rms_px"] > rms0[v["id"]] for v in st["views"])
    assert abs(st["tank"]["length"] - (ed["tank"]["length"] - 2 * ed["tank"]["glass"])) < 1e-9
    r = c.put(f"{base}/views/top/clicks", json={}).json()          # re-solve with the stored clicks
    assert r["view"]["calibration"] is not None


def test_sensitivity_style_and_size_ok(tmp_path, scene):
    """D-840 sensitivity slider (cached base, monotonic), D-841 tank style round trip,
    size confirmation silences the size check."""
    c = _client(tmp_path)
    pid, base = _project_with_photos(c, scene, masks=None)
    # no suggestion cached yet -> 409, the UI then suggests again
    assert c.post(f"{base}/views/front/sensitivity", json={"sensitivity": 70}).status_code == 409
    r = c.post(f"{base}/segment", json={"views": ["front"], "sensitivity": 50})
    assert r.status_code == 200, r.text
    fr = {}
    for sv in (0, 50, 100):
        r = c.post(f"{base}/views/front/sensitivity", json={"sensitivity": sv})
        assert r.status_code == 200, r.text
        assert r.json()["view"]["mask_sensitivity"] == sv
        fr[sv] = mask_from_png(c.get(f"{base}/views/front/mask.png").content)
    assert not (fr[0] & ~fr[50]).any() and not (fr[50] & ~fr[100]).any()     # only grows
    assert fr[0].sum() < fr[50].sum() < fr[100].sum()
    gt = scene["views"]["front"]["gt"]
    assert mask_iou(fr[50], gt) >= 0.97
    # tank style is stored in the project file and comes back in the editor model
    ed = c.get(f"/api/projects/{pid}").json()["editor"]
    assert ed["tank"]["style"] is None
    ed["tank"]["style"] = "rimless_top"
    assert c.put(f"/api/projects/{pid}", json={"editor": ed}).status_code == 200
    assert c.get(f"/api/projects/{pid}").json()["editor"]["tank"]["style"] == "rimless_top"
    ed["tank"]["style"] = "fancy"                      # unknown styles are dropped, not saved
    c.put(f"/api/projects/{pid}", json={"editor": ed})
    assert c.get(f"/api/projects/{pid}").json()["editor"]["tank"]["style"] is None
    # wrong height -> size check; confirming the size silences it until the size changes
    ed = c.get(f"/api/projects/{pid}").json()["editor"]
    ed["tank"]["height"] += 2 * 0.0254
    c.put(f"/api/projects/{pid}", json={"editor": ed})
    v = next(x for x in c.get(base).json()["views"] if x["id"] == "front")
    assert v["calibration"]["size_check"]["suggest"] and v["calibration"]["size_check"]["height_changed"]
    r = c.put(f"{base}/views/front/clicks", json={"size_ok": True}).json()
    assert "size_check" not in r["view"]["calibration"]
