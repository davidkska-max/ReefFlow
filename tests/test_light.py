"""Phase 4 (lighting) unit and integration tests. Fast by default; GPU/CPU via backend_name."""
import math

import numpy as np
import pytest

from reefflow import project as PR
from reefflow.light import config as CF
from reefflow.light import lunits as LU
from reefflow.light.analysis import hotspots_and_shadows, uniformity
from reefflow.light.calibration import _scale_fit
from reefflow.light.fixtures import Distribution, beam_angle_to_m, load_library, spectrum_bins
from reefflow.light.sdf import SDF, closest_point_on_triangle, sphere_sdf


# ---------------------------------------------------------------- units / library
def test_light_units():
    assert LU.parse("180 umol/s", "ppf") == 180
    assert LU.parse("90 deg", "angle") == pytest.approx(math.pi / 2)
    assert LU.parse(45, "angle") == pytest.approx(math.pi / 4)          # bare angle = degrees
    assert LU.parse("0.1 1/cm", "attenuation") == pytest.approx(10.0)
    assert LU.parse("8 in", "length") == pytest.approx(0.2032)
    assert LU.parse_clock("10:30") == 37800
    with pytest.raises(ValueError):
        LU.parse("90 W", "ppf")


def test_library_and_distributions():
    lib = load_library()
    for k in ("led_puck", "led_bar", "t5_fixture", "metal_halide"):
        assert k in lib["archetypes"]
    assert beam_angle_to_m(math.radians(120)) == pytest.approx(1.0)
    assert beam_angle_to_m(math.radians(90)) == pytest.approx(2.0)
    for m in (1.0, 2.0, 4.82):
        d = Distribution("cosine_power", m=m)
        mu = d.inverse_cdf(200001)
        assert mu.mean() == pytest.approx((m + 1) / (m + 2), abs=2e-4)      # E[mu] for I ~ cos^m
        assert math.degrees(d.beam_angle) == pytest.approx(math.degrees(2 * math.acos(0.5 ** (1 / m))))
    iso = Distribution.parse({"type": "isotropic"}).inverse_cdf(100001)
    assert iso.mean() == pytest.approx(0.0, abs=1e-6)
    # tabulated cos^2 == analytic cos^2
    th = np.linspace(0, 90, 181)
    tab = Distribution.parse({"type": "table", "angles": th.tolist(), "values": (np.cos(np.radians(th)) ** 2).tolist()})
    assert np.max(np.abs(tab.inverse_cdf(4001) - Distribution("cosine_power", m=2).inverse_cdf(4001))) < 2e-3
    # cutoff: no emission beyond it
    cut = Distribution.parse({"type": "lambertian", "cutoff": "60 deg"}).inverse_cdf(1001)
    assert cut.min() == pytest.approx(0.5, abs=1e-9)
    s = spectrum_bins(lib["spectra"]["cool_white"])
    assert s.sum() == pytest.approx(1.0) and len(s) == 30


# ---------------------------------------------------------------- config
def _demo():
    from reefflow.light.demo import demo_project
    return demo_project()


def test_demo_lighting_parses():
    p = _demo()
    cfg = CF.lighting_from_project(p)
    assert len(cfg.fixtures) == 2
    f = cfg.fixtures[0]
    # 90 W x 2.0 umol/J, white channel (share 0.30) at 80 % -> 180 x (1 - 0.3 x 0.2)
    assert f.ppf == pytest.approx(180 * (1 - 0.3 * 0.2))
    assert f.centre[2] == pytest.approx(p.tank.water_surface + 8 * 0.0254)
    assert np.allclose(f.axis, [0, 0, -1])
    assert cfg.walls["back"] == "black"
    assert cfg.photoperiod.full_intensity_seconds == pytest.approx(9 * 3600, rel=1e-6)
    assert cfg.map_depths[0] == pytest.approx(0.075)


def test_tilt_convention():
    p = _demo()
    lt = dict(p.raw["lighting"])
    lt["fixtures"] = [dict(lt["fixtures"][0], tilt_x=30, name="a"), dict(lt["fixtures"][1], tilt_y=30, name="b")]
    cfg = CF.lighting_from_project(p, overrides=lt)
    assert cfg.fixtures[0].axis[1] > 0.49          # + tilt_x aims toward the back (+y)
    assert cfg.fixtures[1].axis[0] > 0.49          # + tilt_y aims toward +x


def test_photoperiod_points_and_wrap():
    pp = CF.Photoperiod.parse({"points": [["08:00", 0], ["09:00", 100], ["21:00", 100], ["22:00", 0]]})
    assert pp.full_intensity_seconds == pytest.approx(13 * 3600, rel=1e-5)
    over = CF.Photoperiod.parse({"on": "18:00", "off": "02:00", "ramp": "30 min", "peak": 50})
    assert over.full_intensity_seconds == pytest.approx(0.5 * 7.5 * 3600, rel=1e-5)
    assert over(0.0) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [{"glass": {"walls": {"back": "mirror"}}}, {"albedo": {"sand": 1.5}},
                                 {"fixtures": [{"archetype": "led_puck", "position": [0.1, 0.1]}]},
                                 {"fixtures": [{"archetype": "nope", "position": [0.1, 0.1], "height": 0.2}]}])
def test_config_errors(bad):
    p = _demo()
    lt = {**p.raw["lighting"], **bad}
    with pytest.raises(ValueError):
        CF.parse_lighting(lt, p.tank)


def test_grid_matches_tank_build():
    from reefflow import tank as TK
    from reefflow.light.tracer import grid_for_tank
    p = _demo()
    m = TK.build(p, 0.010)
    g = grid_for_tank(p.tank, 0.010)
    assert tuple(g.shape) == tuple(m.shape)
    assert np.allclose(g.origin, m.origin)


# ---------------------------------------------------------------- geometry
def test_closest_point_on_triangle_bruteforce():
    rng = np.random.default_rng(0)
    T = rng.normal(size=(200, 3, 3))
    P = rng.normal(size=(200, 3)) * 2
    cp = closest_point_on_triangle(P, T[:, 0], T[:, 1], T[:, 2])
    u, v = np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, 1, 301))
    ok = (u + v) <= 1
    u, v = u[ok], v[ok]
    for i in range(0, 200, 20):
        Q = T[i, 0] + u[:, None] * (T[i, 1] - T[i, 0]) + v[:, None] * (T[i, 2] - T[i, 0])
        dmin = np.min(np.linalg.norm(Q - P[i], axis=1))
        assert np.linalg.norm(cp[i] - P[i]) <= dmin + 1e-9
        assert np.linalg.norm(cp[i] - P[i]) >= dmin - 0.01


def test_sdf_from_mesh_sphere():
    from reefflow import geometry as G
    c, R, h = np.array([0.1, 0.1, 0.1]), 0.04, 0.0025
    v, f = G.sdf_to_mesh(sphere_sdf(c, R), c - 1.3 * R, c + 1.3 * R, 0.0015)
    s = SDF.from_mesh(v, f, h)
    rng = np.random.default_rng(1)
    d = rng.normal(size=(2000, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = R + rng.uniform(-2 * h, 2 * h, 2000)
    P = c + d * r[:, None]
    err = s.sample(P) - (r - R)
    assert np.max(np.abs(err)) < 0.25 * h            # trilinear + mesh facetting
    pt = s.patches(c - 0.06, 0.005, (25, 25, 25))
    assert pt["area"].sum() == pytest.approx(4 * math.pi * R ** 2, rel=0.01)


def test_sdf_from_mask_sphere():
    c, R, dx = np.array([0.1, 0.1, 0.1]), 0.04, 0.005
    o = np.zeros(3)
    X = o[0] + dx * np.arange(41)
    P = np.stack(np.meshgrid(X, X, X, indexing="ij"), -1)
    s = SDF.from_mask(sphere_sdf(c, R)(P) < 0, o, dx, smooth=1.0)
    d = np.array([[1, 0, 0], [0, 1, 0], [0.577, 0.577, 0.577]])
    assert np.all(np.abs(s.sample(c + R * d)) < 0.55 * dx)        # voxel surface within dx/2 (A-106)
    n = s.gradient(c + R * d)
    assert np.all(np.sum(n * d, 1) > math.cos(math.radians(10)))


# ---------------------------------------------------------------- analysis / calibration
def test_uniformity_and_regions():
    a = np.full((20, 10), 100.0)
    a[2:5, 2:5] = 300.0
    a[15:18, 6:9] = 10.0
    a[0, 0] = np.nan
    u = uniformity(a)
    assert u["max"] == 300 and u["min"] == 10 and u["n"] == 199
    hs = hotspots_and_shadows(a, np.arange(20) * 0.01, np.arange(10) * 0.01)
    assert len(hs["hot_spots"]) == 1 and hs["hot_spots"][0]["cells"] == 9
    assert len(hs["shadows"]) == 1 and hs["shadows"][0]["centroid_m"] == pytest.approx([0.16, 0.07])


def test_scale_fit_closed_form():
    p = np.array([100.0, 200.0, 50.0])
    s, r = _scale_fit(p, 0.8 * p)
    assert s == pytest.approx(0.8) and np.allclose(r, 0)


# ---------------------------------------------------------------- tracer (backend)
def test_trace_deterministic_and_closed(backend_name):
    from reefflow.light.tracer import build_scene, grid_for_tank, trace
    p = _demo()
    cfg = CF.lighting_from_project(p)
    sc = build_scene(cfg, p.tank, grid=grid_for_tank(p.tank, 0.01))
    a = trace(sc, 1 << 20, seed=5)
    b = trace(sc, 1 << 20, seed=5)
    # same seed -> same photons; sums agree to fp32 round-off (GPU atomic order is not fixed)
    assert np.allclose(a.sand_w, b.sand_w, rtol=1e-5, atol=1e-6)
    c = trace(sc, 1 << 20, seed=6)
    assert np.abs(a.sand_w - c.sand_w).max() > 1.0
    assert abs(a.energy_balance()["closure_residual"]) < 1e-6


def test_demo_run_and_cfd_interface(backend_name):
    from reefflow import tank as TK
    from reefflow.light import par_on_cfd_grid, run_lighting
    p = _demo()
    m = TK.build(p, 0.010)
    res = run_lighting(p, model=m, rays=4_000_000)
    par = par_on_cfd_grid(m, res)
    assert par.meter.shape == m.shape and par.normal.shape == m.shape
    sa = par.surface_adjacent
    assert sa.sum() > 1000
    assert np.isfinite(par.meter[sa]).mean() > 0.99
    assert np.isfinite(par.normal[sa]).mean() > 0.99
    assert np.all(np.isnan(par.meter[~m.fluid_mask]))
    assert np.isnan(par.normal[m.fluid_mask & ~sa]).all()
    sand = par.adjacent_tag == TK.TAGS["sand"]
    assert 200 < np.nanmean(par.meter[sand]) < 600                 # demo: ~380 at 3e7 rays
    assert np.allclose(par.dli_meter, par.meter * res.cfg.photoperiod.dli_factor, equal_nan=True)
    assert "relative/estimated" in par.label
    s = res.summary()
    assert abs(s["energy_balance"]["closure_residual"]) < 1e-6
    with pytest.raises(ValueError):
        par_on_cfd_grid(TK.build(p, 0.02), res)


def test_comparison_and_sweep(backend_name):
    from reefflow.light import compare, run_lighting, sweep_height
    from reefflow.light.run import prepare_geometry, with_lighting
    p = _demo()
    cfg = CF.lighting_from_project(p)
    geo = prepare_geometry(p, cfg, dx=0.01)
    a = run_lighting(p, geometry=geo, rays=2_000_000)
    lt = dict(p.raw["lighting"])
    lt["fixtures"] = [dict(f, intensity=50) for f in lt["fixtures"]]
    b = run_lighting(with_lighting(p, lt), geometry=geo, rays=2_000_000)
    c = compare(a, b)
    # same photons (same seed), half the output -> exactly half
    lit = a.plan > 0
    assert np.nanmax(np.abs(b.plan[lit] / a.plan[lit] - 0.5)) < 1e-9
    assert c["plan_stats"][0]["diff"]["mean"] == pytest.approx(-0.5 * c["plan_stats"][0]["a"]["mean"])
    sw = sweep_height(p, [0.1, 0.3], target="depth:0.15", rays=1_000_000, geometry=geo, target_mean=300)
    assert len(sw["rows"]) == 2 and sw["most_uniform_height_m"] in (0.1, 0.3)
    # higher fixture -> more uniform at depth (wider footprint)
    assert sw["rows"][1]["cv"] < sw["rows"][0]["cv"]


def test_roughness_and_noise_target(backend_name):
    from reefflow import tank as TK
    from reefflow.light import roughness_from_cfd, run_lighting
    from reefflow.light.run import prepare_geometry
    p = _demo()
    cfg = CF.lighting_from_project(p)
    geo = prepare_geometry(p, cfg, dx=0.01)
    # noise target: tracing stops once the average-lit patch reaches it (well before max_rays)
    r = run_lighting(p, geometry=geo, target_noise=0.08)
    assert r.tallies.photons < cfg.max_rays
    assert r.noise_summary()["average_lit_patch_rel"] <= 0.08
    # CFD-driven roughness map: shape / value
    m = TK.build(p, 0.01)
    rmap = roughness_from_cfd(m, np.full(m.shape, 0.1), slope_per_ms=2.0)      # sigma = 0.2
    assert rmap[0].shape == m.shape[:2] and np.allclose(rmap[0], 0.2)
    r2 = run_lighting(p, geometry=geo, rays=1_000_000, roughness_map=rmap)
    assert abs(r2.tallies.energy_balance()["closure_residual"]) < 1e-6


def test_rough_surface_beam_spread(backend_name):
    """Collimated beam through a rough surface (RMS slope sigma): per-axis angular spread of the
    refracted rays is (1 - 1/n) sigma/sqrt(2) for small slopes, so the spot at depth D has RMS
    width sqrt((1 - 1/n)^2 sigma^2/2 D^2 + w^2/12) (w = beam width)."""
    from reefflow.light.tracer import build_scene, trace
    from reefflow.light.verification import _box_tank, _cfg
    L, D, sig, w = 2.0, 1.0, 0.2, 0.01
    tank = _box_tank(L, L, 0.0, D)
    cfg = _cfg([{"name": "b", "archetype": "collimated_beam", "position": [1.0, 1.0], "height": 0.05}], tank,
               surface={"roughness": sig}, bounces=False, albedo={"sand": 0, "rock": 0}, water={"kd": 0},
               map_depths=[0.5], map_resolution="2 mm")
    sc = build_scene(cfg, tank)
    tal = trace(sc, 2_000_000, seed=3)
    E = tal.plan_w[0]
    x = (np.arange(E.shape[0]) + 0.5) * 0.002
    px = E.sum(1) / E.sum()
    mx = np.sum(px * x)
    rms = math.sqrt(np.sum(px * (x - mx) ** 2))
    pred = math.sqrt(((1 - 1 / 1.34) * sig / math.sqrt(2) * 0.5) ** 2 + w ** 2 / 12)
    assert mx == pytest.approx(1.0, abs=0.001)
    assert rms == pytest.approx(pred, rel=0.08), (rms, pred)
