"""SOURCE nodes: exact mass and momentum bookkeeping of pump/return/overflow terms."""
import numpy as np
import pytest

from reefflow import sources as S
from reefflow.domain import Domain, SOLID, SOURCE, FLUID_LIKE


def _box(shape=(48, 24, 24)):
    d = Domain(shape)
    for f in ("x-", "x+", "y-", "y+", "z-", "z+"):
        d.set_face(f, SOLID)
    return d


def _spec(d, m0=2e-3, u0=(0.08, 0.0, 0.0), f0=(0.0, 0.0, 2e-4)):
    """Two pumps: pump 0 = return (mass source at x=10 + sink at x=38, balanced);
    pump 1 = powerhead force at x=24. All at least 10 cells from walls, so for
    n < 7 steps no population has reached a wall or another source."""
    src = np.array([(10, 12, 12), (10, 12, 13), (10, 13, 12), (10, 13, 13)])
    snk = np.array([(38, 12, 12), (38, 12, 13), (38, 13, 12), (38, 13, 13), (38, 11, 12)])
    ph = np.array([(24, 12, 12), (24, 12, 13)])
    cells = np.vstack([src, snk, ph])
    tab = np.zeros((len(cells), S.S_NCOL))
    tab[:4, S.S_PUMP] = 0
    tab[:4, S.S_MASS] = m0 / 4
    tab[:4, S.S_U0:S.S_U0 + 3] = u0
    tab[4:9, S.S_PUMP] = 0
    tab[4:9, S.S_MASS] = -m0 / 5
    tab[9:, S.S_PUMP] = 1
    tab[9:, S.S_F0:S.S_F0 + 3] = np.array(f0) / 2
    for c in cells:
        d.flags[tuple(c)] = SOURCE
    d._final = False
    return S.SourceSpec(cells, tab, n_pumps=2)


@pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
def test_source_mass_and_momentum_exact(backend_name, lattice):
    from reefflow.solver import LBMSolver
    d = _box()
    spec = _spec(d)
    s = LBMSolver(d, lattice=lattice, collision="rr", nu=0.01, smagorinsky=0.1, sources=spec)
    lat = s.lat
    fl = np.isin(d.flags, FLUID_LIKE)
    m0 = s.total_mass()
    expect_rate = spec.momentum_rate() + spec.force_total()
    for n in range(1, 7):
        s.step(1)
        g = s.f.to_numpy().astype(np.float64)
        mom = np.tensordot(lat.c.T.astype(np.float64), g, axes=(1, 0))[:, fl].sum(axis=1)
        # momentum grows by exactly sum(m0 u0) + sum(F) per step until disturbances reach the sink
        np.testing.assert_allclose(mom, n * expect_rate, rtol=2e-4, atol=1e-9)
    # balanced source/sink: total mass constant (fp32 round-off only)
    s.step(2000)
    assert abs(s.total_mass() / m0 - 1.0) < 2e-6
    _, u = s.macros()
    assert np.all(np.isfinite(u))


def test_sink_alone_removes_exact_mass(backend_name):
    from reefflow.solver import LBMSolver
    d = _box((24, 16, 16))
    cells = np.array([(12, 8, 8), (12, 8, 9)])
    tab = np.zeros((2, S.S_NCOL))
    tab[:, S.S_MASS] = -1e-3
    for c in cells:
        d.flags[tuple(c)] = SOURCE
    s = LBMSolver(d, lattice="D3Q19", collision="rr", nu=0.02, sources=S.SourceSpec(cells, tab, 1))
    m0 = s.total_mass()
    s.step(100)
    assert s.total_mass() == pytest.approx(m0 - 100 * 2e-3, rel=1e-6)


def test_pump_state_scales_force(backend_name):
    """s_F scales the force; the turbulence terms average out (zero-mean amplitudes)."""
    from reefflow.solver import LBMSolver
    d = _box((32, 16, 16))
    cells = np.array([(16, 8, 8)])
    tab = np.zeros((1, S.S_NCOL))
    tab[0, S.S_F0] = 1e-4
    tab[0, S.S_PHI0] = 1.0
    d.flags[16, 8, 8] = SOURCE
    spec = S.SourceSpec(cells, tab, 1)
    s = LBMSolver(d, lattice="D3Q19", collision="rr", nu=0.02, sources=spec)
    st = spec.default_state()
    st[0, S.P_SF] = 0.25
    st[0, S.P_I] = 0.2
    st[0, S.P_A0] = 0.5          # delta = 0.5 -> axial factor 1.1
    st[0, S.P_LAT + 1] = 1.0     # lateral: I |f0| in +y
    s.set_pump_state(st)
    s.step(3)
    g = s.f.to_numpy().astype(np.float64)
    fl = np.isin(d.flags, FLUID_LIKE)
    mom = np.tensordot(s.lat.c.T.astype(np.float64), g, axes=(1, 0))[:, fl].sum(axis=1)
    np.testing.assert_allclose(mom, 3 * 0.25 * np.array([1.1e-4, 0.2e-4, 0.0]), rtol=1e-4, atol=1e-10)


def test_source_spec_validation():
    with pytest.raises(ValueError):
        S.SourceSpec(np.array([(1, 1, 1), (1, 1, 1)]), np.zeros((2, S.S_NCOL)), 1)
    tab = np.zeros((1, S.S_NCOL)); tab[0, S.S_PUMP] = 3
    with pytest.raises(ValueError):
        S.SourceSpec(np.array([(1, 1, 1)]), tab, 2)


@pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
def test_compute_graph_stepping_is_bit_identical(backend_name, lattice):
    """Batched stepping via a Taichi compute graph (D-701) must reproduce the plain loop exactly."""
    from reefflow.solver import LBMSolver
    out = []
    for use_graph in (False, True):
        d = _box()
        spec = _spec(d)
        s = LBMSolver(d, lattice=lattice, collision="rr", nu=0.01, smagorinsky=0.1, sources=spec)
        s.use_graph = use_graph
        s.step(10)
        s.step(7)            # odd count: buffer parity must be tracked
        assert s.use_graph == use_graph       # graph path did not silently fall back
        out.append((s.f.to_numpy(), s.u.to_numpy(), s.time_step))
    assert out[0][2] == out[1][2] == 17
    assert np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1])
