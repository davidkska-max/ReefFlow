"""Kernel-level invariants: conservation, equilibrium stationarity, forcing, BC plumbing."""
import numpy as np
import pytest

from reefflow.domain import Domain, SOLID, FREESLIP, INLET, OUTLET
from reefflow.solver import LBMSolver

COMBOS = [("D3Q19", "bgk", None), ("D3Q19", "rr", None), ("D3Q19", "rr", 0.17),
          ("D3Q27", "bgk", None), ("D3Q27", "rr", 0.17)]
IDS = [f"{a}-{b}-{'les' if c else 'dns'}" for a, b, c in COMBOS]


def _periodic(shape=(12, 10, 8)):
    return Domain(shape, periodic=(True, True, True))


def _random_state(shape, amp=0.03, seed=0):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.01 * rng.standard_normal(shape)
    u = amp * rng.standard_normal(shape + (3,))
    return rho.astype(np.float32), u.astype(np.float32)


def _mass_momentum(s):
    f = s.populations()
    c = s.lat.c.astype(float)
    return f.sum(), np.einsum("qijk,qa->a", f, c)


@pytest.mark.parametrize("lat,col,cs", COMBOS, ids=IDS)
def test_conservation_periodic(backend_name, lat, col, cs):
    d = _periodic()
    s = LBMSolver(d, lattice=lat, collision=col, nu=0.02, smagorinsky=cs)
    rho, u = _random_state(d.shape)
    s.initialize(rho=rho, u=u)
    m0, p0 = _mass_momentum(s)
    s.step(200)
    m1, p1 = _mass_momentum(s)
    assert abs(m1 / m0 - 1) < 2e-7
    assert np.allclose(p1, p0, atol=2e-5 * m0 * 1e-2)


@pytest.mark.parametrize("lat,col,cs", COMBOS, ids=IDS)
def test_uniform_flow_is_stationary(backend_name, lat, col, cs):
    d = _periodic()
    s = LBMSolver(d, lattice=lat, collision=col, nu=0.01, smagorinsky=cs)
    u = np.zeros(d.shape + (3,), np.float32); u[...] = (0.05, -0.03, 0.02)
    s.initialize(u=u)
    s.step(100)
    rho, uu = s.macros()
    assert np.max(np.abs(uu - u)) < 1e-6
    assert np.max(np.abs(rho - 1)) < 1e-6


@pytest.mark.parametrize("lat,col,cs", COMBOS, ids=IDS)
def test_body_force_momentum_rate(backend_name, lat, col, cs):
    """In a periodic box total momentum must grow by exactly sum(F) per step."""
    d = _periodic()
    g = (2e-5, -1e-5, 5e-6)
    s = LBMSolver(d, lattice=lat, collision=col, nu=0.05, smagorinsky=cs, body_force=g)
    n = 50
    s.step(n)
    _, p = _mass_momentum(s)          # post-collision populations: momentum = n * sum(F)
    ncell = np.prod(d.shape)
    assert np.allclose(p, n * ncell * np.array(g), rtol=2e-5)
    _, u = s.macros()                  # state velocity = (n - 1/2) g (Guo half-step)
    assert np.allclose(u.mean(axis=(0, 1, 2)), (n - 0.5) * np.array(g), rtol=1e-4)


def test_viscous_decay_of_shear_wave(backend_name):
    """u_x = A sin(2 pi y / Ly) decays as exp(-nu k^2 t): checks the viscosity mapping."""
    ny = 64
    for lat, col in (("D3Q19", "bgk"), ("D3Q19", "rr"), ("D3Q27", "rr")):
        d = Domain((4, ny, 1), periodic=(True, True, True))
        nu = 0.05
        s = LBMSolver(d, lattice=lat, collision=col, nu=nu)
        y = np.arange(ny)
        u = np.zeros(d.shape + (3,), np.float32)
        u[:, :, :, 0] = (0.01 * np.sin(2 * np.pi * y / ny))[None, :, None]
        s.initialize(u=u)
        n = 2000
        s.step(n)
        _, uu = s.macros()
        k = 2 * np.pi / ny
        amp = 2 * np.mean(uu[0, :, 0, 0] * np.sin(k * y))
        nu_eff = -np.log(amp / 0.01) / (k * k * n)
        assert nu_eff == pytest.approx(nu, rel=5e-3), (lat, col, nu_eff)


@pytest.mark.parametrize("lat,col", [("D3Q19", "rr"), ("D3Q27", "rr"), ("D3Q19", "bgk")])
def test_smagorinsky_eddy_viscosity(backend_name, lat, col):
    """Shear wave u = A sin(ky) with Smagorinsky nu_t = Cs^2 |du/dy|.

    Projecting the eddy-viscous stress divergence on sin(ky) gives
        dA/dt = -k^2 nu0 A - (8 / 3 pi) Cs^2 k^3 A^2,
    i.e. an effective eddy viscosity nu_t,eff = (8 / 3 pi) Cs^2 k A.
    Measured extra decay (LES run vs no-model run) must match within 10 %
    (the single-mode estimate ignores transfer to the 3k harmonic).
    """
    ny, A0, n, cs = 32, 0.05, 500, 0.17
    k = 2 * np.pi / ny
    y = np.arange(ny)
    amp = {}
    for model in (None, cs):
        d = Domain((4, ny, 1), periodic=(True, True, True))
        s = LBMSolver(d, lattice=lat, collision=col, nu=1e-3 if col == "bgk" else 1e-4, smagorinsky=model)
        u = np.zeros(d.shape + (3,), np.float32)
        u[:, :, :, 0] = (A0 * np.sin(k * y))[None, :, None]
        s.initialize(u=u)
        s.step(n)
        _, uu = s.macros()
        amp[model] = 2 * np.mean(uu[0, :, 0, 0] * np.sin(k * y))
    nut = np.log(amp[None] / amp[cs]) / (k * k * n)
    expect = 8 / (3 * np.pi) * cs**2 * k * np.sqrt(amp[None] * amp[cs])
    assert nut == pytest.approx(expect, rel=0.10)


def test_sponge_is_extra_viscosity(backend_name):
    """A uniform sponge of strength nu_s must make a shear wave decay at nu0 + nu_s."""
    ny, nu0, nu_s, n = 64, 0.02, 0.03, 2000
    d = Domain((4, ny, 1), periodic=(True, True, True))
    s = LBMSolver(d, lattice="D3Q19", collision="rr", nu=nu0, sponge_nu=np.full(d.shape, nu_s))
    y = np.arange(ny)
    k = 2 * np.pi / ny
    u = np.zeros(d.shape + (3,), np.float32)
    u[:, :, :, 0] = (0.01 * np.sin(k * y))[None, :, None]
    s.initialize(u=u)
    s.step(n)
    _, uu = s.macros()
    amp = 2 * np.mean(uu[0, :, 0, 0] * np.sin(k * y))
    assert -np.log(amp / 0.01) / (k * k * n) == pytest.approx(nu0 + nu_s, rel=5e-3)


def test_moving_wall_couette(backend_name):
    """Plane Couette: linear profile between a fixed and a moving halfway wall."""
    N, U = 16, 0.05
    d = Domain((2, N + 2, 1), periodic=(True, False, True))
    d.set_face("y-", SOLID)
    d.set_face("y+", SOLID, velocity=(U, 0, 0))
    s = LBMSolver(d, lattice="D3Q19", collision="bgk", nu=0.1)
    s.step(20000)
    _, u = s.macros()
    y = np.arange(1, N + 1) - 0.5
    assert np.allclose(u[0, 1:N + 1, 0, 0], U * y / N, atol=1e-6)


def test_open_boundaries_plug_flow(backend_name):
    """Velocity inlet + pressure outlet + free-slip sides must carry a uniform plug flow."""
    U = 0.04
    d = Domain((40, 12, 1), periodic=(False, False, True))
    d.set_face("y-", FREESLIP); d.set_face("y+", FREESLIP)
    d.set_face("x-", INLET, velocity=(U, 0, 0)); d.set_face("x+", OUTLET, rho=1.0)
    u0 = np.zeros(d.shape + (3,), np.float32); u0[..., 0] = U
    for col in ("bgk", "rr"):
        s = LBMSolver(d, lattice="D3Q19", collision=col, nu=0.05)
        s.initialize(u=u0)        # start from the plug state: the BCs must preserve it
        s.step(4000)
        rho, u = s.macros()
        fl = d.flags == 0
        assert np.allclose(u[fl][:, 0], U, atol=2e-5), col
        assert np.max(np.abs(u[fl][:, 1])) < 1e-5
        assert np.max(np.abs(rho[fl] - 1)) < 1e-4


def test_domain_rejects_bad_open_boundary():
    d = Domain((6, 6, 6))
    d.set_face("x-", INLET, velocity=(0.01, 0, 0))
    d.set_face("x+", SOLID)
    d.flags[1, :, :] = SOLID   # inlet now faces a wall
    with pytest.raises(ValueError):
        d.finalize()


def test_cpu_backend_matches(tmp_path):
    """Run a short Poiseuille case on the CPU backend in a subprocess; must match the analytical slip law."""
    import subprocess, sys, json, os
    code = (
        "import json; from reefflow import backend; backend.init('cpu');"
        "from reefflow.verification.poiseuille import run_case;"
        "r=run_case(12,'D3Q19','bgk',0.8,'solid');"
        "print(json.dumps({'slip':r['slip_mean'],'theory':r['slip_theory_bgk']}))"
    )
    env = dict(os.environ, REEFFLOW_ARCH="cpu")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["slip"] == pytest.approx(r["theory"], rel=0.05)
