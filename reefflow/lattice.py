"""Lattice definitions (D3Q19, D3Q27) and Hermite-projection coefficient tables.

Everything here is plain NumPy/Python and is evaluated once; the Taichi kernels
consume the tables as compile-time constants.

Notation (lattice units, dx = dt = 1):
    c_s^2 = 1/3
    H2_ab(c)  = c_a c_b - c_s^2 delta_ab
    H3_abc(c) = c_a c_b c_c - c_s^2 (c_a delta_bc + c_b delta_ac + c_c delta_ab)

A population expanded on the lattice's orthogonal Hermite basis reads
    f_q = w_q [ a0 + c_q.a1 / c_s^2 + H2_q : a2 / (2 c_s^4) + sum_k B_k(q) A_k / N_k ]
where B_k are the third-order basis vectors the lattice can represent, A_k the
matching third-order moments and N_k = sum_q w_q B_k(q)^2 their lattice norms.
For D3Q27 the basis is the seven individual H3 polynomials that contain no
cubed component (xxy, xxz, xyy, yyz, xzz, yzz, xyz).  For D3Q19, H_xyz vanishes
on every velocity and H_xxy is not orthogonal to H_yzz, so the representable
basis is the pairs (H_xxy +/- H_yzz) and cyclic permutations
(cf. Malaspinas 2015, arXiv:1505.06900; Jacob et al. 2018, J. Turbul. 19:1051).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

CS2 = 1.0 / 3.0
CS4 = CS2 * CS2

# Symmetric second-order tensor components, in storage order.
PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
# Third-order components (no repeated-thrice index; those vanish on |c_a|<=1 lattices).
TRIPLES = {
    "xxy": (0, 0, 1), "xxz": (0, 0, 2), "xyy": (0, 1, 1), "yyz": (1, 1, 2),
    "xzz": (0, 2, 2), "yzz": (1, 2, 2), "xyz": (0, 1, 2),
}


def _h2(c: np.ndarray, a: int, b: int) -> np.ndarray:
    return c[:, a] * c[:, b] - CS2 * (a == b)


def _h3(c: np.ndarray, a: int, b: int, g: int) -> np.ndarray:
    return (c[:, a] * c[:, b] * c[:, g]
            - CS2 * (c[:, a] * (b == g) + c[:, b] * (a == g) + c[:, g] * (a == b)))


@dataclass(frozen=True)
class Lattice:
    name: str
    c: np.ndarray            # (Q, 3) int
    w: np.ndarray            # (Q,) float
    basis3: tuple            # tuple of tuples ((coef, triple_name), ...)
    Q: int = field(init=False)
    opp: np.ndarray = field(init=False)
    mirror: np.ndarray = field(init=False)   # (3, Q): index of velocity with axis component flipped
    K1: np.ndarray = field(init=False)       # (Q, 3)
    K2: np.ndarray = field(init=False)       # (Q, 6)
    K3: np.ndarray = field(init=False)       # (Q, n3)
    N3: np.ndarray = field(init=False)       # (n3,)

    def __post_init__(self):
        c = self.c.astype(np.int64)
        w = self.w.astype(np.float64)
        Q = len(w)
        object.__setattr__(self, "Q", Q)
        index = {tuple(v): q for q, v in enumerate(c)}
        object.__setattr__(self, "opp", np.array([index[tuple(-v)] for v in c]))
        mir = np.zeros((3, Q), dtype=np.int64)
        for a in range(3):
            for q, v in enumerate(c):
                m = v.copy(); m[a] = -m[a]
                mir[a, q] = index[tuple(m)]
        object.__setattr__(self, "mirror", mir)

        cf = c.astype(np.float64)
        K1 = w[:, None] * cf / CS2
        K2 = np.zeros((Q, 6))
        for p, (a, b) in enumerate(PAIRS):
            mult = 1.0 if a == b else 2.0       # off-diagonal appears twice in H2:a2
            K2[:, p] = w * _h2(cf, a, b) * mult / (2.0 * CS4)
        B = self.basis_vectors()
        N3 = np.array([np.sum(w * b * b) for b in B])
        K3 = np.stack([w * b / n for b, n in zip(B, N3)], axis=1) if B else np.zeros((Q, 0))
        object.__setattr__(self, "K1", K1)
        object.__setattr__(self, "K2", K2)
        object.__setattr__(self, "K3", K3)
        object.__setattr__(self, "N3", N3)

    # ------------------------------------------------------------------
    def basis_vectors(self) -> list[np.ndarray]:
        cf = self.c.astype(np.float64)
        out = []
        for combo in self.basis3:
            v = np.zeros(len(cf))
            for coef, name in combo:
                v += coef * _h3(cf, *TRIPLES[name])
            out.append(v)
        return out

    @property
    def cs2(self) -> float:
        return CS2


def _d3q19() -> Lattice:
    c = [(0, 0, 0)]
    c += [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    c += [(1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
          (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
          (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
    c = np.array(c, dtype=np.int32)
    n2 = np.sum(c * c, axis=1)
    w = np.select([n2 == 0, n2 == 1, n2 == 2], [1 / 3, 1 / 18, 1 / 36])
    basis = (
        ((1, "xxy"), (1, "yzz")), ((1, "xxy"), (-1, "yzz")),
        ((1, "xzz"), (1, "xyy")), ((1, "xzz"), (-1, "xyy")),
        ((1, "yyz"), (1, "xxz")), ((1, "yyz"), (-1, "xxz")),
    )
    return Lattice("D3Q19", c, w, basis)


def _d3q27() -> Lattice:
    c = sorted(itertools.product((0, 1, -1), repeat=3), key=lambda v: (sum(x * x for x in v), [-x for x in v]))
    c = np.array(c, dtype=np.int32)
    n2 = np.sum(c * c, axis=1)
    w = np.select([n2 == 0, n2 == 1, n2 == 2, n2 == 3], [8 / 27, 2 / 27, 1 / 54, 1 / 216])
    basis = tuple(((1, name),) for name in ("xxy", "xxz", "xyy", "yyz", "xzz", "yzz", "xyz"))
    return Lattice("D3Q27", c, w, basis)


D3Q19 = _d3q19()
D3Q27 = _d3q27()
LATTICES = {"D3Q19": D3Q19, "D3Q27": D3Q27}


def get(name: str) -> Lattice:
    try:
        return LATTICES[name.upper()]
    except KeyError:
        raise ValueError(f"Unknown lattice {name!r}; choose from {sorted(LATTICES)}") from None


# ----------------------------------------------------------------------
# Reference (NumPy) implementations of the equilibrium and the moments used
# by the kernels.  The tests compare the GPU kernels against these.
# ----------------------------------------------------------------------
def a3_equilibrium(rho: float, u: np.ndarray, name: str) -> float:
    a, b, g = TRIPLES[name]
    return rho * u[a] * u[b] * u[g]


def a3_recursive(u: np.ndarray, a2: np.ndarray, name: str) -> float:
    """Recursive third-order non-equilibrium moment (Malaspinas 2015, eq. 31).

    a2 is the full 3x3 symmetric non-equilibrium second moment.
    """
    a, b, g = TRIPLES[name]
    return u[a] * a2[b, g] + u[b] * a2[a, g] + u[g] * a2[a, b]


def combo_value(lat: Lattice, k: int, fn) -> float:
    return sum(coef * fn(name) for coef, name in lat.basis3[k])


def equilibrium(lat: Lattice, rho: float, u, order: int = 2) -> np.ndarray:
    """Reference equilibrium (full, not shifted)."""
    u = np.asarray(u, dtype=np.float64)
    feq = lat.w * rho
    feq = feq + lat.K1 @ (rho * u)
    a2 = np.array([rho * u[a] * u[b] for a, b in PAIRS])
    feq = feq + lat.K2 @ a2
    if order >= 3:
        A = np.array([combo_value(lat, k, lambda n: a3_equilibrium(rho, u, n)) for k in range(len(lat.basis3))])
        feq = feq + lat.K3 @ A
    return feq
