"""Phase 4 verification gate (photometric). Fast tests run reduced versions of each case
with the same pre-declared acceptance logic; the full cases that write the reported
JSON/PNG to verification/results/09_light are marked slow:

    pytest -m slow tests/test_light_verification.py
    python -m reefflow.light.verification
"""
import pytest

from reefflow.light import verification as V


def test_inverse_square_fast(backend_name):
    # full ray count: the fixed-probe exponent's noise at 1e7 rays is ~ its 0.015 tolerance
    r = V.case_inverse_square(rays=5e7, save=False)
    assert r["pass"], r


def test_lambertian_fast(backend_name):
    r = V.case_lambertian(rays=1e7, save=False)
    assert r["pass"], r


def test_fresnel_fast(backend_name):
    r = V.case_fresnel(rays=1e6, save=False)
    assert r["pass"], r


def test_beer_lambert_fast(backend_name):
    r = V.case_beer_lambert(rays=1e6, rays_diffuse=1e7, save=False)
    assert r["pass"], r


def test_convergence_fast(backend_name):
    r = V.case_convergence(exponents=(19, 20, 21, 22), seeds=4, save=False)
    assert r["pass"], r


def test_sphere_fast(backend_name):
    r = V.case_sphere(rays=5e6, save=False)
    assert r["pass"], r


def test_energy_fast(backend_name):
    r = V.case_energy(rays=1e6, save=False)
    assert r["pass"], r


@pytest.mark.slow
def test_calibration_round_trip(backend_name):
    r = V.case_calibration(save=False)
    assert r["pass"], r


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(V.CASES))
def test_full_case(backend_name, case):
    r = V.CASES[case]()
    assert r["pass"], r
