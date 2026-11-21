import itertools

import numpy as np
import pytest

from reefflow import lattice as L

LATS = [L.D3Q19, L.D3Q27]


@pytest.mark.parametrize("lat", LATS, ids=lambda l: l.name)
def test_weights_and_isotropy(lat):
    c = lat.c.astype(float)
    w = lat.w
    d = np.eye(3)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(np.einsum("q,qa->a", w, c), 0)
    assert np.allclose(np.einsum("q,qa,qb->ab", w, c, c), d / 3)
    assert np.allclose(np.einsum("q,qa,qb,qc->abc", w, c, c, c), 0)
    m4 = np.einsum("q,qa,qb,qc,qd->abcd", w, c, c, c, c)
    iso = (np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d)
           + np.einsum("ad,bc->abcd", d, d)) / 9
    assert np.allclose(m4, iso)


@pytest.mark.parametrize("lat", LATS, ids=lambda l: l.name)
def test_opposite_and_mirror(lat):
    c = lat.c
    assert np.array_equal(c[lat.opp], -c)
    for a in range(3):
        m = c.copy(); m[:, a] *= -1
        assert np.array_equal(c[lat.mirror[a]], m)


@pytest.mark.parametrize("lat", LATS, ids=lambda l: l.name)
def test_third_order_basis_orthogonal(lat):
    B = lat.basis_vectors()
    G = np.array([[np.sum(lat.w * a * b) for b in B] for a in B])
    assert np.allclose(G, np.diag(np.diag(G)), atol=1e-15)
    # orthogonal to lower-order Hermite polynomials as well
    c = lat.c.astype(float)
    for b in B:
        assert abs(np.sum(lat.w * b)) < 1e-15
        for a in range(3):
            assert abs(np.sum(lat.w * b * c[:, a])) < 1e-15
        for a, bb in L.PAIRS:
            assert abs(np.sum(lat.w * b * (c[:, a] * c[:, bb] - (a == bb) / 3))) < 1e-15


@pytest.mark.parametrize("lat", LATS, ids=lambda l: l.name)
def test_equilibrium_moments(lat):
    rho, u = 1.03, np.array([0.06, -0.04, 0.025])
    c = lat.c.astype(float)
    for order in (2, 3):
        f = L.equilibrium(lat, rho, u, order)
        assert f.sum() == pytest.approx(rho, rel=1e-14)
        assert np.allclose(f @ c, rho * u, atol=1e-15)
        Pi = np.einsum("q,qa,qb->ab", f, c, c)
        assert np.allclose(Pi, rho / 3 * np.eye(3) + rho * np.outer(u, u), atol=1e-15)
    f3 = L.equilibrium(lat, rho, u, 3)
    for name, (a, b, g) in L.TRIPLES.items():
        H = L._h3(c, a, b, g)
        expect = rho * u[a] * u[b] * u[g]
        if lat.name == "D3Q19" and name == "xyz":
            expect = 0.0   # not representable on D3Q19
        assert np.sum(f3 * H) == pytest.approx(expect, abs=1e-15)


def test_d3q19_basis_norms():
    # (H_xxy + H_yzz) has lattice norm 2/27, (H_xxy - H_yzz) has 6/27 (Jacob et al. 2018 factors 3:1)
    assert np.allclose(L.D3Q19.N3 * 27, [2, 6, 2, 6, 2, 6])
    assert np.allclose(L.D3Q27.N3 * 27, [2, 2, 2, 2, 2, 2, 1])
