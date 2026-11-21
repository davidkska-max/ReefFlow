"""Running time statistics of the velocity field, and run monitoring.

Per FLUID/SOURCE node we accumulate sum(u) (3), sum(u_i u_j) (6) and sum(|u|) (1).
Device partial sums are in the solver's float type and are folded into fp64
host arrays every ``fold_every`` samples, which bounds accumulation error.

Derived quantities:
    mean velocity            U_i = <u_i>
    Reynolds stresses        R_ij = <u_i u_j> - U_i U_j
    turbulent kinetic energy k = R_ii / 2
    turbulence intensity     I = sqrt(2k/3) / |U|      (or relative to <|u|>)
    mean speed               <|u|>   (differs from |U| in unsteady flow)
"""
import numpy as np
import taichi as ti

from .domain import FLUID, SOURCE
from .kernels import array_types, float_type, I3

NSTAT = 10
_cache = {}


def _kernels(fp):
    if fp in _cache:
        return _cache[fp]
    F4, F3, V3, F2 = array_types(fp)
    F1 = ti.types.ndarray(dtype=fp, ndim=1)

    @ti.kernel
    def accumulate(u: V3, flags: I3, acc: F4):
        for i, j, k in ti.ndrange(flags.shape[0], flags.shape[1], flags.shape[2]):
            fl = flags[i, j, k]
            if fl == FLUID or fl == SOURCE:
                v = u[i, j, k]
                acc[0, i, j, k] += v[0]
                acc[1, i, j, k] += v[1]
                acc[2, i, j, k] += v[2]
                acc[3, i, j, k] += v[0] * v[0]
                acc[4, i, j, k] += v[1] * v[1]
                acc[5, i, j, k] += v[2] * v[2]
                acc[6, i, j, k] += v[0] * v[1]
                acc[7, i, j, k] += v[0] * v[2]
                acc[8, i, j, k] += v[1] * v[2]
                acc[9, i, j, k] += v.norm()

    @ti.kernel
    def summary(u: V3, rho: F3, flags: I3, out: F1):
        """out = [max |u|, sum |u|^2, sum rho, n_nonfinite] over FLUID/SOURCE nodes."""
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        out[3] = 0.0
        for i, j, k in ti.ndrange(flags.shape[0], flags.shape[1], flags.shape[2]):
            fl = flags[i, j, k]
            if fl == FLUID or fl == SOURCE:
                v = u[i, j, k]
                s2 = v.dot(v)
                if not (s2 >= 0.0 and s2 < 1e30):       # NaN or inf
                    ti.atomic_add(out[3], 1.0)
                else:
                    ti.atomic_max(out[0], ti.sqrt(s2))
                    ti.atomic_add(out[1], s2)
                    ti.atomic_add(out[2], rho[i, j, k])

    @ti.kernel
    def probe(u: V3, idx: ti.types.ndarray(dtype=ti.i32, ndim=2), out: F2):
        for n in range(idx.shape[0]):
            v = u[idx[n, 0], idx[n, 1], idx[n, 2]]
            out[n, 0] = v[0]
            out[n, 1] = v[1]
            out[n, 2] = v[2]

    _cache[fp] = (accumulate, summary, F1, probe)
    return _cache[fp]


class RunningStats:
    def __init__(self, solver, fold_every: int = 200):
        self.solver = solver
        self.fold_every = fold_every
        self._acc_k = _kernels(solver.fp)[0]
        self.acc = ti.ndarray(solver.fp, shape=(NSTAT,) + solver.shape)
        self.acc.fill(0.0)
        self.host = np.zeros((NSTAT,) + solver.shape, dtype=np.float64)
        self.n = 0
        self._n_dev = 0

    def sample(self):
        """Accumulate the solver's current u (call right after step(..., store_last=True))."""
        self._acc_k(self.solver.u, self.solver.flags, self.acc)
        self._n_dev += 1
        if self._n_dev >= self.fold_every:
            self.fold()

    def fold(self):
        if self._n_dev:
            self.host += self.acc.to_numpy()
            self.acc.fill(0.0)
            self.n += self._n_dev
            self._n_dev = 0

    def snapshot(self):
        """(n, copy of the host sums) - used to form statistics over sub-windows."""
        self.fold()
        return self.n, self.host.copy()

    @staticmethod
    def derive(sums: np.ndarray, n: int) -> dict:
        if n <= 0:
            raise RuntimeError("no samples")
        m = sums / n
        U = np.moveaxis(m[0:3], 0, -1)
        R = np.stack([m[3] - m[0] ** 2, m[4] - m[1] ** 2, m[5] - m[2] ** 2,
                      m[6] - m[0] * m[1], m[7] - m[0] * m[2], m[8] - m[1] * m[2]], axis=-1)
        k = 0.5 * (R[..., 0] + R[..., 1] + R[..., 2])
        return {"n": n, "mean_u": U, "reynolds": R, "tke": k, "mean_speed": m[9]}

    def result(self) -> dict:
        self.fold()
        return self.derive(self.host, self.n)


class Monitor:
    """Cheap whole-field summary of the solver's stored (rho, u)."""

    def __init__(self, solver):
        self.solver = solver
        _, self._k, _, self._probe = _kernels(solver.fp)
        self.out = ti.ndarray(solver.fp, shape=(4,))
        self.n_fluid = None
        self._pidx = None

    def set_probes(self, idx):
        idx = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        self._pidx = ti.ndarray(ti.i32, shape=idx.shape)
        self._pidx.from_numpy(idx)
        self._pout = ti.ndarray(self.solver.fp, shape=idx.shape)

    def probes(self) -> np.ndarray:
        """Stored velocity (lattice units) at the probe nodes, shape (n, 3)."""
        self._probe(self.solver.u, self._pidx, self._pout)
        return self._pout.to_numpy().astype(np.float64)

    def __call__(self) -> dict:
        s = self.solver
        if self.n_fluid is None:
            from .domain import FLUID_LIKE
            self.n_fluid = s.domain.count(FLUID_LIKE)
        self._k(s.u, s.rho, s.flags, self.out)
        o = self.out.to_numpy().astype(np.float64)
        return {"u_max": o[0], "ke_mean": 0.5 * o[1] / self.n_fluid, "rho_mean": o[2] / self.n_fluid,
                "nonfinite": int(o[3])}
