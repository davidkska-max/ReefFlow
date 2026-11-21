"""Phase 6 end-to-end: synthetic photos of the demo rockscape -> reconstruct() with
automatic GrabCut segmentation. ~30-60 s, so marked slow (pytest -m slow)."""
import numpy as np
import pytest

import reefflow  # noqa: F401
from reefflow import demo, geometry as G
from reefflow.photo import TankGeometry, View, compare_occupancy, reconstruct, recommended_points
from reefflow.photo import synthetic as SY


@pytest.mark.slow
def test_reconstruct_demo_rockscape_auto_segmentation(tmp_path):
    proj = demo.write_demo(tmp_path)
    tank = TankGeometry.from_project(proj)
    v, f = G.load_mesh(tmp_path / "rock.stl", unit="mm")
    rng = np.random.default_rng(0)
    photos = []
    for k, cam in SY.default_cameras(tank).items():
        view = View(k, cam, tank)
        img, _ = SY.render_view(view, v, f, seed=11)
        clicks = SY.true_clicks(view, [p["name"] for p in recommended_points(k)])
        clicks = {n: (np.asarray(u) + rng.normal(0, 2.0, 2)).tolist() for n, u in clicks.items()}
        photos.append({"name": k, "kind": k, "image": img, "clicks": clicks})
    out = reconstruct(tank, photos, dx=0.01, out_dir=tmp_path / "recon", segmentation="grabcut")
    grid = out["grid"]
    truth = G.voxelize_mesh(v, f, grid.origin, grid.dx, grid.shape)
    m = compare_occupancy(out["occupancy"], truth)
    assert m["recall"] > 0.95 and m["iou"] > 0.6, m
    rep = out["report"]
    assert all(rep["views"][k]["calibration"]["rms_px"] < 5 for k in rep["views"])
    assert all(rep["views"][k]["segmentation"]["method"] == "grabcut" for k in rep["views"])
    assert (tmp_path / "recon" / "rock_photo.stl").exists()
    # the exported STL can be referenced by a project file like any rock STL
    vv, ff = G.load_mesh(tmp_path / "recon" / "rock_photo.stl", unit="mm")
    assert G.mesh_report(vv, ff)["watertight"]
