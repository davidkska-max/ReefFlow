"""Phase 6 (photo reconstruction) unit tests -- fast, offline, no model weights."""
import numpy as np
import pytest

import reefflow  # noqa: F401  (offline Taichi guard, project convention)
from reefflow.photo import camera as CM
from reefflow.photo import carve as CV
from reefflow.photo import edit as ED
from reefflow.photo import models as MD
from reefflow.photo import segment as SG
from reefflow.photo import synthetic as SY
from reefflow.photo.calibration import CalibrationError, calibrate_view, focal_px_from_35mm
from reefflow.photo.tankref import TankGeometry, recommended_points

IN = 0.0254


@pytest.fixture(scope="module")
def tank():
    return TankGeometry(47 * IN, 17 * IN, 20.5 * IN, 0.5 * IN, 19.5 * IN, 2 * IN)


@pytest.fixture(scope="module")
def views(tank):
    cams = SY.default_cameras(tank, size=(800, 600), f_px=600.0)
    L, W = tank.length, tank.width
    zc = 0.25
    cams["back"] = CM.Camera.look_at((L / 2, W + 1.4, zc), (L / 2, W / 2, zc - 0.02), 600, 800, 600)
    cams["right"] = CM.Camera.look_at((L + 0.8, W / 2, zc), (L / 2, W / 2, zc - 0.02), 600, 800, 600)
    return {k: CM.View(k, c, tank) for k, c in cams.items()}


def _interior_points(tank, n=500, seed=0):
    r = np.random.default_rng(seed)
    return np.column_stack([r.uniform(0.02, tank.length - 0.02, n), r.uniform(0.02, tank.width - 0.02, n),
                            r.uniform(tank.sand_depth, tank.water_surface - 0.005, n)])


# ----------------------------------------------------------------------
# refraction / projection
# ----------------------------------------------------------------------
def test_solve_tangential_single_interface_snell():
    # camera 1 m above water, point 0.4 m deep, 0.5 m sideways: Snell at the surface
    a, b, h, n = 1.0, 0.4, 0.5, 1.34
    s = CM.solve_tangential(np.array([h]), np.array([[a], [b]]), np.array([[1.0], [n]]))[0]
    th1, th2 = np.arcsin(s), np.arcsin(s / n)
    assert abs(a * np.tan(th1) + b * np.tan(th2) - h) < 1e-12
    assert abs(np.sin(th1) - n * np.sin(th2)) < 1e-14


def test_paraxial_apparent_depth(tank):
    """Near the axis a submerged point appears at depth d/n (classic apparent depth)."""
    C = np.array([0.5, 0.2, tank.water_surface + 1.0])
    cam = CM.Camera.look_at(C, C - [0, 0, 1], 1000, 1001, 1001, up=(0, 1, 0))
    v = CM.View("top", cam, tank)
    d = 0.3
    P = np.array([[0.5 + 0.002, 0.2, tank.water_surface - d]])
    uv_r, ok = v.project(P)
    virt = np.array([[0.5 + 0.002, 0.2, tank.water_surface - d / tank.n_water]])
    uv_v, _ = cam.project_points_pinhole(virt)
    assert ok[0]
    assert np.allclose(uv_r, uv_v, atol=1e-3)


@pytest.mark.parametrize("kind", ["top", "front", "back", "left", "right"])
def test_project_backproject_roundtrip(tank, views, kind):
    v = views[kind]
    P = _interior_points(tank)
    uv, ok = v.project(P)
    assert ok.mean() > 0.9
    r = v.back_project(uv[ok])
    assert np.all(r["valid"])
    d = CM.ray_point_distance(r["entry"], r["dir"], P[ok])
    assert d.max() < 1e-9
    # Snell invariant across the last interface: n_air sin(theta_air) = n_inner sin(theta_inner)
    n = v.normal
    s_air = np.linalg.norm(np.cross(r["dir_air"], n), axis=1)
    s_in = np.linalg.norm(np.cross(r["dir"], n), axis=1) * r["n_inner"]
    assert np.allclose(s_air, s_in, atol=1e-12)


def test_refraction_changes_projection_by_tens_of_pixels(tank, views):
    P = _interior_points(tank, 200)
    for k in ("top", "front", "left"):
        uv_r, ok = views[k].project(P)
        uv_p, _ = views[k].camera.project_points_pinhole(P)
        assert np.median(np.linalg.norm(uv_r - uv_p, axis=1)[ok]) > 3.0


def test_validity_flags(tank, views):
    # side view: submerged point whose straight-layer ray would enter the pane above the water line
    hi_cam = CM.Camera.look_at((tank.length / 2, -1.0, tank.height + 0.6), (tank.length / 2, tank.width, 0.1),
                               600, 800, 600)
    v = CM.View("front", hi_cam, tank)
    _, ok = v.project(np.array([[tank.length / 2, tank.width - 0.01, tank.water_surface - 0.01]]))
    assert not ok[0]
    # top view: point outside the tank opening is not seen through the surface
    _, ok = views["top"].project(np.array([[-0.05, 0.2, 0.2]]))
    assert not ok[0]
    # camera inside the tank is rejected
    bad = CM.View("top", CM.Camera.look_at((0.5, 0.2, 0.3), (0.5, 0.2, 0.0), 600, 800, 600, up=(0, 1, 0)), tank)
    assert not bad.camera_ok()


def test_view_serialization_roundtrip(tank, views):
    v = views["front"]
    v2 = CM.View.from_dict(v.to_dict(), tank)
    P = _interior_points(tank, 50)
    assert np.allclose(v.project(P)[0], v2.project(P)[0], atol=1e-9)


def test_radial_distortion_roundtrip():
    cam = CM.Camera.look_at((0, -1, 0.2), (0, 0, 0.2), 900, 1200, 900, k1=-0.08)
    uv = np.array([[10.0, 20.0], [600, 450], [1150, 870]])
    d = cam.pixel_dirs(uv)
    uv2, _ = cam.project_dirs(d)
    assert np.allclose(uv, uv2, atol=1e-6)


# ----------------------------------------------------------------------
# calibration (PnP + refraction)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["top", "front", "left", "back", "right"])
def test_pnp_recovers_known_camera(tank, views, kind):
    v = views[kind]
    names = [p["name"] for p in recommended_points(kind)]
    clicks = SY.true_clicks(v, names)
    assert len(clicks) == 8
    cal = calibrate_view(kind, (800, 600), tank, clicks)
    assert cal.rms_px < 1e-6
    assert np.linalg.norm(cal.view.camera.centre - v.camera.centre) < 1e-6
    assert abs(cal.view.camera.f / v.camera.f - 1) < 1e-7
    assert np.allclose(cal.view.camera.R, v.camera.R, atol=1e-8)


def test_pnp_noise_error_consistent_with_reported_std(tank, views):
    v = views["front"]
    names = [p["name"] for p in recommended_points("front")]
    exact = SY.true_clicks(v, names)
    r = np.random.default_rng(11)
    z = []
    for _ in range(12):
        clicks = {n: (np.asarray(u) + r.normal(0, 1.0, 2)).tolist() for n, u in exact.items()}
        cal = calibrate_view("front", (800, 600), tank, clicks)
        z.append((cal.view.camera.centre - v.camera.centre) * 1e3 / np.asarray(cal.centre_std_mm))
        assert cal.rms_px < 3.0
    z = np.asarray(z)
    # normalized errors should be O(1): RMS between 0.3 and 3 (loose, 12 trials)
    assert 0.3 < np.sqrt(np.mean(z ** 2)) < 3.0


def test_pnp_refuses_coplanar_points_without_focal(tank, views):
    v = views["front"]
    clicks = SY.true_clicks(v, ["rim_FL", "rim_FR", "bottom_FL", "bottom_FR"])   # all on the y = 0 plane
    with pytest.raises(CalibrationError, match="one plane"):
        calibrate_view("front", (800, 600), tank, clicks)
    cal = calibrate_view("front", (800, 600), tank, clicks, f_px=600.0)          # known f: solvable
    assert np.linalg.norm(cal.view.camera.centre - v.camera.centre) < 1e-5


def test_pnp_needs_four_points(tank, views):
    clicks = SY.true_clicks(views["top"], ["rim_FL", "rim_FR", "sand_BL"])
    with pytest.raises(CalibrationError):
        calibrate_view("top", (800, 600), tank, clicks)


def test_pnp_with_edge_click(tank, views):
    v = views["top"]
    names = [p["name"] for p in recommended_points("top")]
    clicks = SY.true_clicks(v, names)
    A, B = tank.point("sand_BL"), tank.point("rim_BL")
    uv, _ = v.project((A + 0.37 * (B - A))[None])
    cal = calibrate_view("top", (800, 600), tank, clicks, edges=[{"a": "sand_BL", "b": "rim_BL", "uv": uv[0].tolist()}])
    assert cal.rms_px < 0.05            # edge residual: polyline discretisation only
    assert len(cal.edge_residual_px) == 1


def test_focal_from_35mm():
    assert abs(focal_px_from_35mm(26.0, 4032, 3024) - 26 * 5040 / 43.267) < 1e-9


# ----------------------------------------------------------------------
# carving
# ----------------------------------------------------------------------
def _sphere_scene(tank, centre, r):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=4, radius=r)
    return np.asarray(m.vertices) + centre, np.asarray(m.faces)


@pytest.fixture(scope="module")
def hires_views(tank):
    return {k: CM.View(k, c, tank) for k, c in SY.default_cameras(tank, size=(1600, 1200), f_px=1200.0).items()}


def test_carve_sphere_from_three_views(tank, hires_views):
    views = hires_views
    c = np.array([0.6, 0.22, 0.2]); r = 0.05
    vv, ff = _sphere_scene(tank, c, r)
    ks = ("top", "front", "left")
    masks = [SY.render_mask(views[k], vv, ff, clip=False) for k in ks]
    grid = CV.VoxelGrid.from_bounds(c - 0.09, c + 0.09, 0.004)
    res = CV.carve_visual_hull([views[k] for k in ks], masks, grid, tolerance_px=0)
    from reefflow import geometry as G
    truth = G.voxelize_mesh(vv, ff, grid.origin, grid.dx, grid.shape)
    m = CV.compare_occupancy(res.occupancy, truth)
    ratio = res.occupancy.sum() / truth.sum()
    # orthographic 3-view hull of a sphere = Steinmetz tricylinder, 1.119 x sphere;
    # perspective cameras at 1-1.5 m and the 4 mm staircase add a little (measured 1.15)
    assert m["recall"] > 0.995
    assert 1.08 < ratio < 1.22, ratio
    assert res.stats["unconstrained"] == 0


def test_carve_box_from_three_views(tank, hires_views):
    views = hires_views
    lo, hi = np.array([0.5, 0.15, 0.08]), np.array([0.62, 0.25, 0.16])
    import trimesh
    m = trimesh.creation.box(bounds=[lo, hi])
    masks = [SY.render_mask(views[k], m.vertices, m.faces, clip=False) for k in ("top", "front", "left")]
    grid = CV.VoxelGrid.from_bounds(lo - 0.04, hi + 0.04, 0.004)
    res = CV.carve_visual_hull([views[k] for k in ("top", "front", "left")], masks, grid, tolerance_px=0)
    truth = ED.region(grid, {"type": "box", "lo": lo, "hi": hi})
    met = CV.compare_occupancy(res.occupancy, truth)
    # perspective silhouettes include side faces -> small wedges beyond the box (measured IoU 0.89)
    assert met["recall"] > 0.99 and met["iou"] > 0.85
    # the 1 px default tolerance only adds material
    res1 = CV.carve_visual_hull([views[k] for k in ("top", "front", "left")], masks, grid, tolerance_px=1)
    assert (res1.occupancy | res.occupancy).sum() == res1.occupancy.sum()


def test_carve_out_of_view_and_ignore(tank, views):
    grid = CV.VoxelGrid.for_tank(tank, 0.02)
    narrow = CM.View("top", CM.Camera.look_at((0.3, 0.2, 1.5), (0.3, 0.2, 0.0), 300, 200, 200, up=(0, 1, 0)), tank)
    empty = np.zeros((200, 200), bool)
    res = CV.carve_visual_hull([narrow], [empty], grid, min_views=0)
    seen = res.seen_count > 0
    assert seen.any() and (~seen).any()
    assert not res.occupancy[seen].any()            # seen + not rock -> carved
    assert res.occupancy[~seen].all()               # outside the field of view -> untouched
    res2 = CV.carve_visual_hull([narrow], [empty], grid, ignore=[~empty], min_views=0)
    assert res2.occupancy.all() and res2.seen_count.max() == 0
    res3 = CV.carve_visual_hull([narrow], [empty], grid, min_views=1)
    assert res3.stats["unconstrained"] == int((~seen).sum())


def test_mesh_export_roundtrip(tmp_path):
    from reefflow import geometry as G
    grid = CV.VoxelGrid.from_bounds((0.1, 0.1, 0.05), (0.3, 0.25, 0.2), 0.005)
    occ = ED.region(grid, {"type": "sphere", "center": (0.2, 0.17, 0.08), "radius": 0.06})
    occ = ED.carve_tunnel(occ, grid, [(0.12, 0.17, 0.08), (0.28, 0.17, 0.08)], 0.015)
    ex = CV.export_stl(occ, grid, tmp_path / "r.stl")
    assert ex["mesh"]["watertight"] and ex["rockscape_entry"]["file"] == "r.stl"
    v, f = G.load_mesh(tmp_path / "r.stl", unit="mm")
    back = G.voxelize_mesh(v, f, grid.origin, grid.dx, grid.shape)
    assert np.array_equal(back, occ)
    assert SY.signed_volume(v, f) > 0             # outward normals
    assert v[:, 2].min() < grid.origin[2] - grid.dx / 2   # base buried one cell


def test_volume_and_weight():
    w = CV.rock_weight(0.010, bulk_density=1400.0)
    assert abs(w["weight_kg"] - 14.0) < 1e-12
    assert abs(w["weight_lb"] - 14.0 / 0.45359237) < 1e-9
    assert abs(w["volume_L"] - 10.0) < 1e-12 and abs(w["volume_gal"] - 10.0 / 3.785411784) < 1e-9
    assert np.allclose(w["weight_range_kg"], [11.0, 18.0])
    assert abs(CV.rock_weight(0.010, solid_fraction=0.5)["weight_kg"] - 7.0) < 1e-12
    with pytest.raises(ValueError):
        CV.rock_weight(0.01, solid_fraction=0)
    grid = CV.VoxelGrid.from_bounds((0, 0, 0), (0.1, 0.1, 0.1), 0.01)
    occ = np.zeros(grid.shape, bool); occ[:5] = True
    rep = CV.volume_report(occ, grid)
    assert rep["voxels"] == 500 and abs(rep["volume_m3"] - 5e-4) < 1e-15


def test_compare_occupancy():
    a = np.zeros((4, 4, 4), bool); b = a.copy()
    a[:2] = True; b[1:3] = True
    m = CV.compare_occupancy(a, b)
    assert abs(m["iou"] - 1 / 3) < 1e-12 and m["recall"] == 0.5 and m["volume_error"] == 0.0


# ----------------------------------------------------------------------
# occupancy editing
# ----------------------------------------------------------------------
@pytest.fixture
def fine_grid():
    return CV.VoxelGrid.from_bounds((0, 0, 0), (0.2, 0.2, 0.2), 0.002)


def test_edit_shapes_volumes(fine_grid):
    g = fine_grid
    full = np.ones(g.shape, bool)
    V = lambda o: (~o).sum() * g.cell_volume
    r = 0.04
    assert abs(V(ED.carve_sphere(full, g, (0.1, 0.1, 0.1), r)) / (4 / 3 * np.pi * r ** 3) - 1) < 0.03
    pts = [(0.05, 0.1, 0.1), (0.15, 0.1, 0.1)]
    cap = np.pi * 0.02 ** 2 * 0.1 + 4 / 3 * np.pi * 0.02 ** 3
    assert abs(V(ED.carve_tunnel(full, g, pts, 0.02)) / cap - 1) < 0.03
    assert abs(V(ED.carve_box(full, g, (0.05, 0.05, 0.05), (0.1, 0.12, 0.09))) / (0.05 * 0.07 * 0.04) - 1) < 0.1
    span, rise, ln = 0.06, 0.08, 0.1
    arch = (span * (rise - span / 2) + np.pi * (span / 2) ** 2 / 2) * ln
    assert abs(V(ED.carve_arch(full, g, (0.1, 0.1, 0.02), (1, 1), span, rise, ln)) / arch - 1) < 0.05
    with pytest.raises(ValueError):
        ED.carve_arch(full, g, (0.1, 0.1, 0.02), (1, 0), 0.1, 0.02, 0.1)


def test_edit_is_pure_and_replayable(fine_grid):
    g = fine_grid
    occ = np.zeros(g.shape, bool)
    edits = [{"op": "add", "shape": {"type": "obox", "center": [0.1, 0.1, 0.1], "size": [0.1, 0.05, 0.05],
                                     "rotate_z_deg": 30}},
             {"op": "carve", "shape": {"type": "capsule", "points": [[0.05, 0.1, 0.1], [0.15, 0.1, 0.1]],
                                       "radius": 0.01}},
             {"op": "add", "shape": {"type": "sphere", "center": [0.1, 0.1, 0.1], "radius": 0.005}}]
    out = ED.apply_edits(occ, g, edits)
    assert not occ.any()                                  # input untouched
    seq = occ
    for e in edits:
        seq = ED.apply_edit(seq, g, e)
    assert np.array_equal(out, seq)
    i = np.round((np.array([0.1, 0.1, 0.1]) - g.origin) / g.dx).astype(int)
    assert out[tuple(i)]                                  # re-added centre
    reg = ED.region(g, {"type": "sphere", "center": [0.1, 0.1, 0.1], "radius": 0.02})
    assert not ED.mark_open(out, reg)[reg].any()
    assert ED.add_material(occ, g, {"type": "box", "lo": [0, 0, 0], "hi": [0.01, 0.01, 0.01]}).sum() > 0


# ----------------------------------------------------------------------
# masks / segmentation
# ----------------------------------------------------------------------
def test_mask_edit_ops():
    m = np.zeros((100, 120), bool)
    a = SG.brush(m, [[20, 50], [80, 50]], 5, "add")
    assert not m.any() and a[50, 20] and a[50, 80] and a[55, 50] and not a[57, 50]
    b = SG.brush(a, [[50, 50]], 3, "subtract")
    assert not b[50, 50] and a[50, 50]
    p = SG.polygon(m, [[10, 10], [60, 10], [60, 40], [10, 40]])
    assert abs(p.sum() - 51 * 31) <= 2 * (51 + 31)
    ring = SG.polygon(p, [[20, 20], [40, 20], [40, 30], [20, 30]], "subtract")
    assert SG.fill_holes(ring).sum() == p.sum()
    assert SG.fill_holes(ring, max_area=10).sum() == ring.sum()
    spk = p.copy(); spk[90, 110] = True
    assert SG.remove_small(spk, 5).sum() == p.sum()
    assert np.array_equal(SG.mask_from_png(SG.mask_to_png(ring)), ring)
    assert SG.smooth(spk, 2)[90, 110] == False  # noqa: E712
    assert abs(SG.mask_iou(p, p) - 1) < 1e-12
    with pytest.raises(ValueError):
        SG.brush(m, [[1, 1]], 2, "paint")


def test_grabcut_simple_scene():
    r = np.random.default_rng(0)
    img = np.empty((120, 160, 3)); img[:] = (40, 70, 120)
    yy, xx = np.mgrid[0:120, 0:160]
    blob = (xx - 80) ** 2 / 30 ** 2 + (yy - 60) ** 2 / 22 ** 2 < 1
    img[blob] = (150, 100, 80)
    img = np.clip(img + r.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    out = SG.segment_view(img, method="grabcut", rect=[30, 25, 130, 95])
    assert out["method"] == "grabcut"
    assert SG.mask_iou(out["mask"], blob) > 0.95


def test_segment_falls_back_to_grabcut_when_sam_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("REEFFLOW_MODEL_DIR", str(tmp_path))
    img = np.zeros((60, 80, 3), np.uint8); img[20:40, 30:50] = 200
    out = SG.segment_view(img, method="sam", rect=[20, 10, 60, 50], cleanup_min_area=0)
    assert out["method"] == "grabcut"
    assert any("not installed" in n for n in out["notes"])


def test_auto_init_labels_geometry(tank, views):
    lab = SG.auto_init_labels(views["front"])
    reg = SG.view_region_masks(views["front"])
    assert (lab[~reg["tank"]] == SG.GC_BGD).all()
    assert (lab[reg["rockable"]] == SG.GC_PR_FGD).all()
    assert reg["rockable"].sum() < reg["tank"].sum()
    top = SG.view_region_masks(views["top"])
    assert (top["bg_sample"] & top["rockable"]).any()     # sand strip next to the glass


# ----------------------------------------------------------------------
# models (optional; never downloads in tests)
# ----------------------------------------------------------------------
def test_model_registry_license_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("REEFFLOW_MODEL_DIR", str(tmp_path))
    with pytest.raises(PermissionError, match="commercial"):
        MD.download_model("depth_anything_v2_large", accept_license=True)
    with pytest.raises(PermissionError, match="accept_license"):
        MD.download_model("mobile_sam")
    for name, s in MD.MODELS.items():
        if s["commercial_use"]:
            assert s["license"] in MD.LICENSE_ALLOWLIST and s["url"]
    rows = {r["name"]: r for r in MD.model_status()}
    assert rows["mobile_sam"]["status"].startswith("not installed")
    assert rows["depth_anything_v2_base"]["status"] == "excluded (license)"
    assert MD.capabilities()["segmentation"] == ["grabcut"]
    with pytest.raises(MD.ModelNotInstalled):
        MD.estimate_depth(np.zeros((4, 4, 3)))
    with pytest.raises(NotImplementedError):
        MD.photogrammetry_reconstruct([])
    ok, why = MD.SamSegmenter("mobile_sam").available()
    assert not ok and "not installed" in why


# ----------------------------------------------------------------------
# tank reference geometry
# ----------------------------------------------------------------------
def test_tank_geometry_from_demo_project(tmp_path):
    from reefflow import demo
    t = TankGeometry.from_project(demo.write_demo(tmp_path))
    assert abs(t.length - 47 * IN) < 1e-12 and abs(t.height - 20.5 * IN) < 1e-12
    assert abs(t.water_surface - 19.5 * IN) < 1e-12 and abs(t.glass - 0.5 * IN) < 1e-12
    assert np.allclose(t.point("sand_BR"), [47 * IN, 17 * IN, 2 * IN])
    t2 = TankGeometry.from_dims(48, 18, 21, 0.5, 1, 2, unit="in", internal=False)
    assert abs(t2.length - t.length) < 1e-12 and abs(t2.water_surface - t.water_surface) < 1e-12
    for kind in ("top", "front", "back", "left", "right"):
        P = np.array([q for p in recommended_points(kind)
                      for q in ([t.point(p["name"])] if p["type"] == "point" else t.seam(p["name"]))])
        sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
        assert sv[2] > 1e-2 * sv[0]                       # never coplanar (PLAN C9)
    with pytest.raises(KeyError):
        t.point("rim_XX")


# ----------------------------------------------------------------------
# pipeline (fast: sphere scene, user-supplied masks)
# ----------------------------------------------------------------------
def test_reconstruct_pipeline_with_masks(tank, views, tmp_path):
    c = np.array([0.6, 0.22, tank.sand_depth + 0.03]); r = 0.06
    vv, ff = _sphere_scene(tank, c, r)
    photos = []
    for k in ("top", "front", "left"):
        v = views[k]
        names = [p["name"] for p in recommended_points(k)]
        img = np.zeros((600, 800, 3), np.uint8)
        photos.append({"name": k, "kind": k, "image": img, "clicks": SY.true_clicks(v, names),
                       "mask": SY.render_mask(v, vv, ff)})
    out = __import__("reefflow.photo", fromlist=["reconstruct"]).reconstruct(
        tank, photos, dx=0.01, out_dir=tmp_path, edits=[{"op": "carve", "shape": {"type": "sphere",
                                                        "center": c.tolist(), "radius": 0.01}}])
    rep = out["report"]
    assert rep["views"]["top"]["calibration"]["rms_px"] < 1e-5
    assert rep["views"]["front"]["segmentation"]["method"] == "user mask"
    assert (tmp_path / "rock_photo.stl").exists() and (tmp_path / "compare_front.png").exists()
    assert (tmp_path / "reconstruction.json").exists()
    v_true = 0.5 * 4 / 3 * np.pi * r ** 3 * (1 + 3 * 0.5 * 0.03 / r - 0.5 * (0.03 / r) ** 3)  # cap above sand
    assert 0.8 < rep["volume"]["volume_m3"] / v_true < 1.6
    assert rep["views"]["top"]["silhouette_iou"] > 0.8
