"""Phase-1 verification gate.

Fast tests (default) run reduced versions of each case. The full cases that
produce the reported plots and numbers are marked ``slow``:

    pytest -m slow            # full verification (tens of minutes on GPU)
    python -m reefflow.verification.run_all
"""
import numpy as np
import pytest

from reefflow.verification import cavity, cylinder, jet, mass_box, poiseuille


# ---------------------------------------------------------------- fast
@pytest.mark.parametrize("lat,col", [("D3Q19", "bgk"), ("D3Q19", "rr"), ("D3Q27", "rr")])
def test_poiseuille_second_order(backend_name, lat, col):
    Ns = (8, 16, 32)
    errs = [poiseuille.run_case(N, lat, col, 0.8, "solid")["err_l2"] for N in Ns]
    order = -np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    assert 1.85 < order < 2.15, (errs, order)


def test_poiseuille_freeslip_second_order(backend_name):
    Ns = (4, 8, 16)            # H = 2N <= 32: below the fp32 floor (see poiseuille.py)
    errs = [poiseuille.run_case(N, "D3Q19", "bgk", 0.8, "freeslip")["err_l2"] for N in Ns]
    order = -np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    assert 1.85 < order < 2.15, (errs, order)


@pytest.mark.parametrize("lat", ["D3Q19", "D3Q27"])
def test_poiseuille_exact_at_magic_tau(backend_name, lat):
    # analytically exact; fp32 round-off floor is ~1e-6 - 2e-5 here (fp64 gives 3e-9)
    r = poiseuille.run_case(16, lat, "bgk", poiseuille.MAGIC_TAU, "solid")
    assert r["err_l2"] < 5e-5


def test_poiseuille_bgk_slip_matches_theory(backend_name):
    r = poiseuille.run_case(24, "D3Q19", "bgk", 1.2, "solid")
    assert r["slip_mean"] == pytest.approx(r["slip_theory_bgk"], rel=0.01)


def test_cavity_re100_coarse(backend_name):
    # coarse grid (N = 64): looser than the 0.02 full-resolution acceptance
    r = cavity.run_case(100, 64, "D3Q19", "rr")
    assert r["max_dev"] < 0.03, r


def test_mass_box_small(backend_name):
    r = mass_box.run_case("D3Q19", shape=(64, 32, 32), F0=4e-4, n_steps=6000, spinup=2000,
                          planes=(24, 40))
    assert r["mass_max_rel_drift"] < 1e-5
    assert r["momentum_injection_max_rel_err"] < 1e-4
    for v in r["flux"].values():
        assert abs(v["mean_net_over_fwd"]) < 5e-3


def test_jet_analysis_recovers_synthetic_jet():
    """The post-processing alone (no solver): a synthetic self-similar jet with known
    B, S, x0 on the verification grid must be recovered exactly."""
    D, Uj, nx, ny, cy = 10, 0.08, 301, 161, 80.0
    x = np.arange(nx)[:, None, None]
    J, K = np.meshgrid(np.arange(ny), np.arange(ny), indexing="ij")
    r = np.sqrt((J - cy) ** 2 + (K - cy) ** 2)[None]
    xe = np.maximum(x - 4 * D, 3 * D)
    U = Uj * np.minimum(1, 5.8 * D / xe) * np.exp(-np.log(2) * (r / (0.094 * xe)) ** 2)
    res = jet.analyse(U, np.zeros_like(U), D, Uj, cy, cy)
    assert res["B"] == pytest.approx(5.8, rel=1e-3)
    assert res["S"] == pytest.approx(0.094, rel=1e-3)
    assert res["x0_over_D"] == pytest.approx(4.0, abs=0.01)


# ---------------------------------------------------------------- slow (full verification)
@pytest.mark.slow
def test_full_poiseuille(backend_name):
    """Scheme order verified strictly in fp64; fp32 (production) must agree with fp64
    within 25 % wherever H <= 32 (beyond that the documented fp32 floor dominates)."""
    from reefflow.verification.run_all import run_poiseuille_f64
    f32 = poiseuille.run()
    run_poiseuille_f64()
    import json
    from reefflow.verification.common import RESULTS
    f64 = json.loads((RESULTS / "01_poiseuille" / "results_f64.json").read_text())
    for k, c in f64["cases"].items():
        assert 1.98 < c["order_l2"] < 2.02, k
    for v in f64["magic_tau"].values():
        assert max(v["err_l2"]) < 1e-8
    for k, c in f32["cases"].items():
        if k in f64["cases"]:
            for h, e32, e64 in zip(c["H"], c["err_l2"], f64["cases"][k]["err_l2"]):
                if h <= 32:
                    assert abs(e32 / e64 - 1) < 0.25, (k, h, e32, e64)
    for v in f32["magic_tau"].values():
        assert max(v["err_l2"]) < 5e-5


@pytest.mark.slow
def test_full_cavity(backend_name):
    out = cavity.run()
    assert all(c["pass"] for c in out["cases"]), [(c["Re"], c["max_dev"]) for c in out["cases"]]


@pytest.mark.slow
def test_full_cylinder(backend_name):
    out = cylinder.run()
    assert all(c["pass"] for c in out["cases"]), [c["St"] for c in out["cases"]]


@pytest.mark.slow
def test_full_jet(backend_name):
    out = jet.run()
    assert all(c["pass"] for c in out["cases"]), [(c["B"], c["S"]) for c in out["cases"]]


@pytest.mark.slow
def test_full_mass_box(backend_name):
    out = mass_box.run()
    assert all(c["pass"] for c in out["cases"])


@pytest.mark.slow
def test_pump_sources_deliver_momentum_flux(backend_name):
    """Verification 7: powerhead (force) and return (mass source) jets carry J = rho Q^2 / A."""
    from reefflow.verification import powerhead
    out = powerhead.run(plot=False, cases=["powerhead", "return"], save=False)
    for c in out["cases"]:
        assert c["pass"], (c["tag"], c["J_ratio_window"])
