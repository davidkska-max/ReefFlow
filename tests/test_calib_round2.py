"""Calibration round 2 (D-836 to D-839): outside corners and seams, tank-size check,
focal prior, and the texture cue for the rock-mask suggestion."""
import numpy as np
import pytest

from reefflow.photo import View, calibrate_view, check_tank_size, recommended_points
from reefflow.photo import segment as SG
from reefflow.photo import synthetic as SY
from reefflow.photo.tankref import TankGeometry, all_names, describe

IN = 0.0254


@pytest.fixture(scope="module")
def tank():
    return TankGeometry.from_dims(48, 18, 21, 0.5, 1, 2, unit="in", internal=False)


def _views(tank, size=(800, 600), f=600.0):
    from reefflow.photo.camera import Camera
    cams = SY.default_cameras(tank, size=size, f_px=f)
    L, W, zc = tank.length, tank.width, 0.25
    cams["back"] = Camera.look_at((L / 2, W + 1.4, zc), (L / 2, W / 2, zc - 0.02), f, *size)
    cams["right"] = Camera.look_at((L + 0.8, W / 2, zc), (L / 2, W / 2, zc - 0.02), f, *size)
    return {k: View(k, c, tank) for k, c in cams.items()}


def test_names_and_outer_corners(tank):
    g = tank.glass
    assert np.allclose(tank.point("top_FL"), [-g, -g, tank.height])
    assert np.allclose(tank.point("base_BR"), [tank.length + g, tank.width + g, -g])
    A, B = tank.seam("seam_BL_2")
    assert np.allclose(A, [0, tank.width, 0]) and np.allclose(B, [0, tank.width, tank.height])
    for bad in ("seam_XX", "seam_BL_3", "top_XX"):
        with pytest.raises(KeyError):
            (tank.seam if bad.startswith("seam") else tank.point)(bad)
    assert len(all_names()) == 6 * 4 + 8
    assert describe("seam_BR")["type"] == "seam" and "back right" in describe("seam_BR")["hint"]
    # the outer box is exactly the entered external size
    assert tank.point("top_FR")[0] - tank.point("top_FL")[0] == pytest.approx(48 * IN)
    assert tank.point("top_FL")[2] - tank.point("base_FL")[2] == pytest.approx(21 * IN)


@pytest.mark.parametrize("kind", ["front", "left", "back", "right", "top"])
def test_outer_scheme_recovers_camera(tank, kind):
    v = _views(tank)[kind]
    names = [p["name"] for p in recommended_points(kind)]
    clicks = SY.true_clicks(v, names)
    assert len(clicks) == 8
    cal = calibrate_view(kind, (800, 600), tank, clicks)
    assert cal.rms_px < 1e-5
    assert np.linalg.norm(cal.view.camera.centre - v.camera.centre) < 1e-5
    assert abs(cal.view.camera.f / v.camera.f - 1) < 1e-6
    if kind != "top":
        assert cal.edge_names == [n for n in names if n.startswith("seam_")]


def test_inner_scheme_still_works(tank):
    v = _views(tank)["front"]
    clicks = SY.true_clicks(v, [p["name"] for p in recommended_points("front", scheme="inner")])
    cal = calibrate_view("front", (800, 600), tank, clicks)
    assert cal.rms_px < 1e-6


def test_four_front_corners_and_seams_only(tank):
    """The four outside corners are coplanar; the back seams alone must fix the focal length."""
    v = _views(tank)["front"]
    names = ["top_FL", "top_FR", "base_FL", "base_FR", "seam_BL", "seam_BR", "seam_BL_2", "seam_BR_2"]
    cal = calibrate_view("front", (800, 600), tank, SY.true_clicks(v, names))
    assert cal.rms_px < 1e-4 and abs(cal.view.camera.f / 600.0 - 1) < 1e-4


def test_focal_prior_is_weak(tank):
    """A prior 30 % off the true focal length moves an 8-click solution by < 1 %."""
    v = _views(tank)["front"]
    rng = np.random.default_rng(5)
    clicks = {n: (np.asarray(u) + rng.normal(0, 0.7, 2)).tolist()
              for n, u in SY.true_clicks(v, [p["name"] for p in recommended_points("front")]).items()}
    free = calibrate_view("front", (800, 600), tank, clicks)
    pri = calibrate_view("front", (800, 600), tank, clicks, f_prior=(600.0 * 1.3, 0.25))
    assert abs(pri.view.camera.f / free.view.camera.f - 1) < 0.01


def test_size_check_finds_wrong_height(tank):
    """Photo of a 23-in-tall tank calibrated with 21 in: the check reports ~23 in."""
    real = TankGeometry.from_dims(48, 18, 23, 0.5, 1, 2, unit="in", internal=False)
    v = _views(real)["front"]
    rng = np.random.default_rng(2)
    clicks = {n: (np.asarray(u) + rng.normal(0, 1.0, 2)).tolist()
              for n, u in SY.true_clicks(v, [p["name"] for p in recommended_points("front")]).items()}
    cal = calibrate_view("front", (800, 600), tank, clicks)
    assert cal.rms_px > 4
    sc = check_tank_size(cal, tank)
    assert sc["suggest"] and sc["height_changed"]
    assert sc["external_m"][2] / IN == pytest.approx(23, abs=0.3)
    assert sc["rms_px"] < 2.0
    ok = check_tank_size(calibrate_view("front", (800, 600), real, clicks), real)
    assert not ok["suggest"]


# ----------------------------------------------------------------------
# texture cue (D-839)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from reefflow.photo import validation as VA
    tank, rv, rf, grid, truth, true_views = VA.setup(tmp_path_factory.mktemp("rt2"))
    return tank, rv, rf, true_views


def test_texture_cue_beats_colour_unguided(scene):
    tank, rv, rf, views = scene
    out = {}
    for k in ("front", "left"):
        img, gt = SY.render_view(views[k], rv, rf, seed=7)
        out[k] = {cue: SG.mask_iou(SG.segment_view(img, views[k], method="grabcut", cue=cue)["mask"], gt)
                  for cue in ("texture", "colour")}
    assert min(o["texture"] for o in out.values()) >= 0.97, out
    assert out["left"]["texture"] > out["left"]["colour"] + 0.2, out      # colour collapses here (0.51)


def test_texture_cue_ignores_hanging_equipment(scene):
    """A rough object that does not reach the sand (a heater, a powerhead) is left out."""
    tank, rv, rf, views = scene
    v = views["front"]
    img, gt = SY.render_view(v, rv, rf, seed=7)
    # a rough 'powerhead' high on the back glass, far from the rock
    P = np.array([[0.12, tank.width - 0.01, tank.water_surface - 0.08]])
    (u, w), = v.project(P)[0]
    rng = np.random.default_rng(0)
    u0, w0 = int(u), int(w)
    patch = rng.integers(20, 235, (36, 46, 3)).astype(np.uint8)
    img = img.copy()
    img[w0:w0 + 36, u0:u0 + 46] = patch
    gear = np.zeros(gt.shape, bool); gear[w0:w0 + 36, u0:u0 + 46] = True
    assert not (gear & gt).any()
    m = SG.segment_view(img, v, method="grabcut", cue="texture")["mask"]
    assert np.count_nonzero(m & gear) < 0.02 * gear.sum()
    assert SG.mask_iou(m, gt) >= 0.97


# ----------------------------------------------------------------------
# mask round 4 (D-842): bubbles, rubble on the sand, bounded outline, open arches
# ----------------------------------------------------------------------
def test_speck_filter_calms_bubbly_water_but_not_rock():
    """Micro-bubbles (bright 2-3 px specks on dark water) used to score as rough as rock."""
    rng = np.random.default_rng(1)
    img = np.full((1500, 2000, 3), 40, np.uint8)                       # dark open water ...
    vv, uu = rng.integers(0, 1500, 6000), rng.integers(0, 1000, 6000)  # ... full of bubbles (left half)
    for dv in (0, 1):
        for du in (0, 1):
            img[np.clip(vv + dv, 0, 1499), np.clip(uu + du, 0, 999)] = 200
    # porous rock on the right: blotches ~12 px across
    blot = rng.integers(50, 130, (125, 84)).astype(np.uint8)
    img[:, 1000:] = np.repeat(np.repeat(blot, 12, 0), 12, 1)[:1500, :1000, None]
    on, off = SG.texture_energy(img), SG.texture_energy(img, speck_px=0)
    water, rock = np.s_[200:1300, 100:900], np.s_[200:1300, 1100:1900]
    assert np.median(off[water]) > 0.8 * np.median(off[rock])          # without it: water ~ rock
    assert np.median(on[water]) < 0.35 * np.median(on[rock])           # with it: water much smoother
    assert np.median(on[rock]) > 0.8 * np.median(off[rock])            # rock texture kept


def test_darker_than_sand_finds_rubble():
    rng = np.random.default_rng(2)
    img = np.zeros((300, 800, 3), np.uint8)
    img[...] = (150 + rng.normal(0, 8, (300, 800))).clip(0, 255).astype(np.uint8)[..., None]
    img[100:160, 300:420] = 105                                         # darker rubble
    sh = np.zeros((300, 800), bool); sh[80:300] = True
    d = SG.darker_than_sand(img, sh)
    assert d[110:150, 320:400].mean() > 0.95
    assert d[200:280].mean() < 0.02 and not d[:80].any()


def test_outline_stays_near_rock_and_openings_stay_open(scene):
    """GrabCut may only move the outline near the seeds (no tank-wide flood), and the
    sensitivity grow does not close an arch in the base mask."""
    tank, rv, rf, views = scene
    img, gt = SY.render_view(views["front"], rv, rf, seed=7)
    ti = SG.texture_init(views["front"], img)
    lab, cand = ti["labels"], ti["edge"]["cand"]
    # outside the band around the seeds, the candidate region is definite background
    assert np.count_nonzero(cand & (lab == SG.GC_BGD)) > 0.2 * np.count_nonzero(cand)
    # an arch: a 'hole' in a mask stays out at the highest sensitivity
    base = np.zeros(gt.shape, bool); base[150:400, 150:500] = True
    hole = np.zeros_like(base); hole[230:300, 260:380] = True          # 8400 px > 0.1 % of the photo
    base &= ~hole
    edge = {"tex": np.ones(gt.shape, np.float32), "thr": 0.1, "D": np.full(gt.shape, 50, np.float32),
            "sand_hit": np.zeros_like(base), "cand": np.ones_like(base), "floor": base.copy()}
    m = SG.finish_texture_mask(base, edge, 100)
    assert m.sum() > base.sum() and not (m & hole).any()
