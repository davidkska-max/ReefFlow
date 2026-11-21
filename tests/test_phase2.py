"""Phase 2: units parsing, pumps, geometry/voxelization, project file, tank model, VTK, GCI."""
import math
from pathlib import Path

import numpy as np
import pytest

from reefflow import units as U
from reefflow import pumps as PM
from reefflow import geometry as G
from reefflow import sources as S

IN = 0.0254


# ---------------------------------------------------------------- units
def test_parse_quantities():
    assert U.parse("48 in", "length") == pytest.approx(48 * IN)
    assert U.parse("1/2 in", "length") == pytest.approx(0.0127)
    assert U.parse("2500 GPH", "flow") == pytest.approx(2500 * 3.785411784e-3 / 3600)
    assert U.parse("10 ft", "head") == pytest.approx(3.048)
    assert U.parse("60 s", "time") == 60.0
    assert U.parse(0.3, "length") == 0.3
    with pytest.raises(ValueError):
        U.parse("48 GPH", "length")
    with pytest.raises(ValueError):
        U.parse("48in", "length")


# ---------------------------------------------------------------- pumps
def test_return_pump_curve_and_momentum_flux():
    lib = PM.load_library()
    p = PM.build_pump({"name": "r", "archetype": "return_nozzle", "rated_flow": "800 GPH",
                       "static_head": "4 ft", "fittings_allowance": "1 ft",
                       "position": ["10 in", "5 in", "10 in"], "aim": [1, 0, 0]}, lib)
    q = U.parse("800 GPH", "flow") * math.sqrt(1 - 5 / 10)
    assert p.delivered_flow == pytest.approx(q)
    A = math.pi * 0.019 ** 2 / 4
    assert p.momentum_flux == pytest.approx(1023 * q * q / A)
    with pytest.raises(ValueError):
        PM.return_pump_flow(1.0, 3.0, 3.0)


def test_powerhead_derating_and_override():
    base = {"name": "p", "archetype": "wide_jet_propeller", "position": [0.1, 0.1, 0.1], "aim": [0, 1, 0]}
    p = PM.build_pump(dict(base, rated_flow="2000 GPH", derating=0.5))
    assert p.delivered_flow == pytest.approx(0.5 * U.parse("2000 GPH", "flow"))
    p2 = PM.build_pump(dict(base, delivered_flow="1000 L/h"))
    assert p2.delivered_flow == pytest.approx(1 / 3600) and p2.flow_basis == "user override"
    assert abs(p.up @ p.aim) < 1e-12


def test_equivalent_outlet_keeps_momentum_flux():
    o = PM.Outlet("circle", diameter=0.019)
    J = 1.2
    o2, notes = PM.equivalent_outlet(o, J, dx=0.01, min_cells=4, velocity_cap=1.0)
    assert o2.diameter >= 0.04 and notes
    assert math.sqrt(J / (1023 * o2.area)) <= 1.0 + 1e-9
    o3, n3 = PM.equivalent_outlet(PM.Outlet("circle", diameter=0.06), 0.1, dx=0.005, velocity_cap=1.0)
    assert o3.diameter == 0.06 and not n3
    r, _ = PM.equivalent_outlet(PM.Outlet("rect", width=0.23, height=0.03), 0.5, dx=0.01, min_cells=4)
    assert r.height == pytest.approx(0.04) and r.width == pytest.approx(0.23)


def test_modes():
    m = PM.Mode.parse({"type": "sine", "period": "10 s", "min": 0.2, "max": 1.0})
    a, b = 0.2, 1.0
    mid, amp = (a + b) / 2, (b - a) / 2
    assert m.mean_s2() == pytest.approx(mid ** 2 + amp ** 2 / 2, rel=1e-6)
    assert float(m.s(2.5)) == pytest.approx(1.0)
    p = PM.Mode.parse({"type": "pulse", "period": "4 s", "min": 0.3, "max": 1.0, "duty": 0.5, "ramp": 0.1})
    s = p.s(np.linspace(0, 4, 4001))
    assert s.min() == pytest.approx(0.3) and s.max() == pytest.approx(1.0)
    assert float(p.s(1.0)) == pytest.approx(1.0) and float(p.s(3.0)) == pytest.approx(0.3)
    q = PM.Mode.parse({"type": "pulse", "period": "4 s", "min": 0.3, "max": 1.0, "phase": 0.5})
    assert float(q.s(3.0)) == pytest.approx(1.0)          # alternating partner
    assert PM.Mode.parse(None).mean_s2() == 1.0


def test_ou_process_statistics():
    rng = np.random.default_rng(0)
    ou = PM.OUProcess(4000, tcorr=1.0, rng=rng)
    x0 = ou.x.copy()
    x1 = ou.advance(0.5).copy()
    assert np.var(x1) == pytest.approx(1.0, abs=0.06)
    assert np.corrcoef(x0, x1)[0, 1] == pytest.approx(math.exp(-0.5), abs=0.03)


def test_mode_shapes_unit_variance():
    t = np.linspace(-1, 1, 41)
    T1, T2 = np.meshgrid(t, t)
    m = T1 ** 2 + T2 ** 2 <= 1
    phi = PM.mode_shapes(T1[m], T2[m])
    assert phi.shape[1] == S.N_MODES
    assert np.mean(np.sum(phi ** 2, axis=1)) == pytest.approx(1.0)


# ---------------------------------------------------------------- geometry
def _sphere(r=0.1, c=(0.25, 0.25, 0.25)):
    import trimesh
    s = trimesh.creation.icosphere(subdivisions=5, radius=r)
    return s.vertices + np.array(c), s.faces, abs(s.volume)


def test_voxelize_sphere_volume():
    v, f, vol = _sphere()
    dx = 0.01
    m = G.voxelize_mesh(v, f, origin=(0.0, 0.0, 0.0), dx=dx, shape=(50, 50, 50))
    assert m.sum() * dx ** 3 == pytest.approx(vol, rel=0.02)          # r = 10 cells
    assert m[25, 25, 25] and not m[25, 25, 36] and m[25, 25, 34]


def test_voxelize_box_exact():
    import trimesh
    b = trimesh.creation.box(extents=(0.1, 0.06, 0.04))
    v = b.vertices + np.array([0.105, 0.105, 0.105])      # box spans 0.055..0.155 etc.
    m = G.voxelize_mesh(v, b.faces, origin=(0.0, 0.0, 0.0), dx=0.01, shape=(30, 30, 30))
    # node centres at 0.06..0.15 (10), 0.08..0.13 (6), 0.09..0.12 (4)
    assert m.sum() == 10 * 6 * 4


def test_region_mask_rect_orientation():
    P = np.stack(np.meshgrid(*[np.linspace(-0.1, 0.1, 41)] * 3, indexing="ij"), axis=-1)
    m = G.region_mask(P, (0, 0, 0), aim=(1, 0, 0), up=(0, 0, 1), shape="rect", s0=0.0, s1=0.02,
                      width=0.1, height=0.02)
    pts = P[m]
    assert pts[:, 1].max() == pytest.approx(0.05) and pts[:, 2].max() == pytest.approx(0.01)
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() < 0.02


# ---------------------------------------------------------------- project / tank
@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from reefflow.demo import write_demo
    return write_demo(tmp_path_factory.mktemp("demo"))


def test_demo_project_dimensions(demo):
    from reefflow import project
    P = project.load(demo)
    t = P.tank
    assert (t.length, t.width, t.height) == pytest.approx((47 * IN, 17 * IN, 20.5 * IN))
    assert t.water_column == pytest.approx(17.5 * IN)
    r = [p for p in P.pumps if p.kind == "return"][0]
    assert U.from_si(r.delivered_flow, "GPH") == pytest.approx(800 * math.sqrt(0.5))
    assert P.sim["lattice"] == "D3Q19" and P.sim["smagorinsky"] == 0.10


def test_project_validation(demo):
    from reefflow import project
    import yaml
    raw = yaml.safe_load(Path(demo).read_text())
    bad = dict(raw, overflow=None)
    with pytest.raises(ValueError, match="overflow"):
        project.parse(bad, demo)
    raw2 = yaml.safe_load(Path(demo).read_text())
    raw2["pumps"][1]["position"] = ["60 in", "5 in", "10 in"]
    with pytest.raises(ValueError, match="not inside"):
        project.parse(raw2, demo)


def test_tank_model_sources_and_volume(demo):
    from reefflow import project, tank
    from reefflow.domain import SOURCE, SOLID, FREESLIP
    P = project.load(demo)
    m = tank.build(P, 0.02)
    lu = m.lu
    tab = m.spec.table
    for pr in m.pumps:
        rows = tab[tab[:, S.S_PUMP] == pr.index]
        if pr.pump.kind == "powerhead":
            F = rows[:, S.S_F0:S.S_F0 + 3].sum(axis=0) * lu.C_J
            np.testing.assert_allclose(F, pr.pump.momentum_flux * pr.pump.aim, rtol=1e-12)
            assert np.all(rows[:, S.S_MASS] == 0)
        else:
            Q = rows[:, S.S_MASS].sum() * lu.C_Q
            assert Q == pytest.approx(pr.pump.delivered_flow, rel=1e-12)
            mom = (rows[:, S.S_MASS:S.S_MASS + 1] * rows[:, S.S_U0:S.S_U0 + 3]).sum(axis=0) * lu.C_J
            np.testing.assert_allclose(mom, pr.pump.momentum_flux * pr.pump.aim, rtol=1e-12)
    # overflow sink balances the return exactly
    assert tab[:, S.S_MASS].sum() == pytest.approx(0.0, abs=1e-15)
    f = m.domain.flags
    assert np.all(f[:, :, 0] == SOLID) and np.all(f[1:-1, 1:-1, -1] == FREESLIP)
    assert (f == SOURCE).sum() == len(tab)
    # model water volume: gross minus rock and overflow box, within whole-cell rounding
    gross = P.tank.length * P.tank.width * P.tank.water_column
    assert 0.80 * gross < m.info["water_volume_model_m3"] < 0.97 * gross
    st = m.pump_state(1.0, 0.01)
    assert st.shape == (len(P.pumps) + 1, S.P_NCOL)
    assert st[m.overflow_index, S.P_SM] == pytest.approx(1.0)


def test_tank_short_run_conserves_mass(demo, backend_name, tmp_path):
    from reefflow import project, tank
    from reefflow.run import run
    P = project.load(demo).with_overrides(sample_every=10, batches=2)
    m = tank.build(P, 0.02)
    r = run(m, tmp_path, spinup=0.3, average=0.3, progress=lambda *a: None)
    assert abs(r["mass_rel_drift"]) < 1e-5
    assert r["u_max_overall_lattice"] < 0.15
    assert (tmp_path / "tank.vti").exists() and (tmp_path / "slices.png").exists()


# ---------------------------------------------------------------- io / analysis
def test_vti_roundtrip(tmp_path):
    from reefflow.io_vtk import write_vti, read_vti
    a = np.random.default_rng(1).random((5, 4, 3)).astype(np.float32)
    v = np.random.default_rng(2).random((5, 4, 3, 3)).astype(np.float32)
    t = np.arange(60, dtype=np.uint8).reshape(5, 4, 3)
    write_vti(tmp_path / "x.vti", {"a": a, "v": v, "t": t}, (0.1, 0.2, 0.3), 0.01)
    back = read_vti(tmp_path / "x.vti")
    assert np.array_equal(back["a"], a) and np.array_equal(back["v"], v) and np.array_equal(back["t"], t)


def test_celik_recovers_order():
    from reefflow.gridstudy import celik
    h = (0.003, 0.005, 0.010)
    f = [2.0 + 30.0 * hh ** 1.5 for hh in h]
    c = celik(f[0], f[1], f[2], *h)
    assert c["kind"] == "monotonic"
    assert c["p"] == pytest.approx(1.5, rel=1e-6)
    assert c["f_ext"] == pytest.approx(2.0, rel=1e-9)
