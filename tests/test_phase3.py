"""Phase 3: water age, wall shear, dead zones, coral scoring (analytic / exact checks)."""
import math

import numpy as np
import pytest

from reefflow.analysis import age as AGE


def _duct(n=40, U=0.05, dx=0.01):
    """Straight 1-cell duct along x: inflow source at the first cell, sink at the last."""
    fluid = np.zeros((n + 2, 3, 3), bool)
    fluid[1:-1, 1, 1] = True
    u = np.zeros(fluid.shape + (3,))
    u[1:-1, 1, 1, 0] = U
    Q = U * dx * dx
    cells = np.array([(1, 1, 1), (n, 1, 1)])
    return fluid, u, Q, cells


def test_age_plug_flow_exact():
    """No diffusion: upwind FV gives a_i = (i + 1) dx / U, the travel time to the cell's exit face."""
    n, U, dx = 40, 0.05, 0.01
    fluid, u, Q, cells = _duct(n, U, dx)
    out = AGE.solve_age_arrays(fluid, u, np.zeros(fluid.shape), dx, cells, [Q, -Q], tol=1e-12)
    a = out["age"][1:-1, 1, 1]
    np.testing.assert_allclose(a, (np.arange(n) + 1) * dx / U, rtol=1e-8)
    assert out["report"]["outlet_age_over_tau"] == pytest.approx(1.0, rel=1e-8)


def test_age_outlet_identity_with_diffusion_and_divergent_field():
    """sum(outflow x age) = V for any field: projection + conservative FV make it exact,
    even for a rough, non-solenoidal input velocity and strong diffusion."""
    rng = np.random.default_rng(3)
    fluid = np.zeros((14, 12, 10), bool)
    fluid[1:-1, 1:-1, 1:-1] = True
    fluid[5:8, 4:7, 1:5] = False          # an obstacle
    u = rng.normal(0, 0.03, fluid.shape + (3,)) * fluid[..., None]
    D = np.where(fluid, 10 ** rng.uniform(-5, -3, fluid.shape), 0.0)
    Q = 2e-5
    cells = np.array([(1, 1, 1), (1, 2, 1), (12, 10, 8), (11, 10, 8), (12, 9, 8)])
    q = [Q / 2, Q / 2, -Q / 3, -Q / 3, -Q / 3]
    out = AGE.solve_age_arrays(fluid, u, D, 0.01, cells, q, tol=1e-11)
    r = out["report"]
    assert r["outlet_age_over_tau"] == pytest.approx(1.0, rel=1e-6)
    assert r["projection"]["rel_divergence_after"] < 1e-8
    assert np.all(out["age"][fluid] > 0)


def test_age_well_mixed_limit():
    """Strong diffusion (D L / Q ~ 10^4) -> nearly uniform age equal to V/Q (perfectly mixed tank)."""
    fluid = np.zeros((10, 10, 10), bool)
    fluid[1:-1, 1:-1, 1:-1] = True
    u = np.zeros(fluid.shape + (3,))
    D = np.where(fluid, 1e-2, 0.0)
    Q = 1e-7
    out = AGE.solve_age_arrays(fluid, u, D, 0.01, np.array([(1, 1, 1), (8, 8, 8)]), [Q, -Q], tol=1e-12)
    a = out["age"][fluid]
    tau = out["report"]["tau_nominal_s"]
    assert out["report"]["outlet_age_over_tau"] == pytest.approx(1.0, rel=1e-6)
    assert a.mean() / tau == pytest.approx(1.0, rel=2e-3) and a.std() / tau < 2e-3


def test_eddy_viscosity_recovers_boussinesq():
    """If R_ij = 2/3 k delta_ij - 2 nu_t S_ij exactly, the fit returns nu_t."""
    n = 12
    x = np.arange(n) * 0.01
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.stack([3.0 * Y, 0 * X, 0 * X], axis=-1)            # simple shear, S_xy = 1.5
    S = AGE.strain_rate(u, 0.01)
    nu_t = 2e-4
    R = np.zeros(u.shape[:3] + (6,))
    R[..., 0:3] = 0.01
    R -= 2 * nu_t * S
    fluid = np.ones(u.shape[:3], bool)
    tot, res, sgs = AGE.eddy_viscosity(u, R, fluid, 0.01, cs=0.0)
    assert res[3:-3, 3:-3, 3:-3] == pytest.approx(nu_t, rel=1e-6)


# ---------------------------------------------------------------- wall function
def test_spalding_recovers_log_law():
    """In the log layer (y+ = 80) Spalding's law matches u+ = ln(y+)/0.41 + 5.2 to within 2 % in u_tau
    (measured 1.2 %: Spalding's series still differs slightly from the pure log law at y+ = 80)."""
    from reefflow.analysis import wallshear as WS
    ut, y, nu = 0.005, 0.015, 0.94e-6
    yp = y * ut / nu
    u = ut * (np.log(yp) / 0.41 + 5.2)
    assert WS.friction_velocity(np.array([u]), y, nu)[0] == pytest.approx(ut, rel=0.02)
    # viscous sublayer: u+ = y+
    y2 = 2e-4
    ut2 = 0.003
    u2 = ut2 * (y2 * ut2 / nu)
    assert WS.friction_velocity(np.array([u2]), y2, nu)[0] == pytest.approx(ut2, rel=0.02)


def test_coral_trapezoid_and_ramps():
    from reefflow.analysis.coral import _trap, _ramp
    x = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6])
    s = _trap(x, 0.05, 0.10, 0.30, 0.45)
    np.testing.assert_allclose(s, [0, 0, 1, 1, 1, 0, 0])
    assert _ramp(0.075, 0.05, 0.10) == pytest.approx(0.5)
    assert _ramp(0.1, 0.15, 0.0) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory, backend_name):
    from reefflow.demo import write_demo
    from reefflow import project, tank
    from reefflow.run import run
    d = tmp_path_factory.mktemp("p3")
    pf = write_demo(d / "demo")
    P = project.load(pf).with_overrides(batches=2)
    m = tank.build(P, 0.02)
    run(m, d / "run", spinup=0.5, average=0.5, progress=lambda *a: None)
    return d / "run"


def test_analyze_end_to_end(tiny_run):
    from reefflow.analysis.analyze import analyze
    o = analyze(tiny_run, progress=lambda *a: None)
    assert o["age"]["outlet_age_over_tau"] == pytest.approx(1.0, rel=1e-6)
    assert 0 <= o["dead_zones"]["dead_fraction"] <= 1
    assert set(o["coral"]["profiles"]) >= {"SPS", "LPS", "soft"}
    assert (tiny_run / "report.html").stat().st_size > 50_000      # figures embedded


def test_compare_identical_runs(tiny_run, tmp_path):
    from reefflow.analysis.compare import compare
    r = compare(tiny_run, tiny_run, tmp_path)
    assert all(abs(row["delta"]) < 1e-12 for row in r["rows"])
    assert (tmp_path / "comparison.html").exists()
