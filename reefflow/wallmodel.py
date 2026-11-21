"""Wall model for under-resolved walls (DECISIONS D-702).

Problem (verification 6): with plain halfway bounce-back the near-wall layer is
about one cell thick, so near-surface speeds and dead-zone volume change
strongly with the grid (Standard -> Fine: dead volume -32 %, speed 15 mm above
the sand +23 %).

Model (eddy-viscosity wall model, as used in wall-modelled LBM-LES): at every
fluid node next to a wall, with tangential speed u_t at wall distance y = dx/2
(halfway bounce-back), Spalding's law of the wall gives the friction velocity
u_tau (see analysis/wallshear.py). Bounce-back transmits a wall shear stress of
about rho nu_node u_t / y, so setting the node's TOTAL viscosity to

    nu_wall = u_tau^2 y / u_t      (not below the molecular value)

makes the wall carry the log-law shear stress rho u_tau^2. The total matters:
the first verification showed the Smagorinsky eddy viscosity at wall nodes is far
larger than nu_wall (no near-wall damping), so the under-resolved wall exerts
too MUCH shear and slows the near-wall flow. The model therefore overrides the
node's relaxation time, tau = 1/2 + 3 nu_wall, replacing the Smagorinsky value
there (encoded as a negative entry in the solver's per-node relaxation array),
updated from the instantaneous velocity every few steps.

Everything is in lattice units (dx = 1): y = 1/2, nu = lattice viscosity.
"""
import math

import numpy as np
import taichi as ti

from .kernels import array_types

KAPPA, B_LOG = 0.41, 5.2
_cache = {}


def wall_nodes(flags, solid_value=1, fluid_values=(0, 5)):
    """(nodes (n,3) int32, normals (n,3) float) of fluid nodes with a face-neighbour
    solid node; normal = unit sum of the directions from the solids into the fluid."""
    fl = np.isin(flags, fluid_values)
    so = flags == solid_value
    acc = np.zeros(flags.shape + (3,), np.float64)
    for a in range(3):
        for s in (1, -1):
            nb_solid = np.roll(so, -s, axis=a)          # neighbour at +s along axis a is solid
            # the wall lies at +s: inward normal points -s
            if s == 1:
                nb_solid[(slice(None),) * a + (-1,)] = False
            else:
                nb_solid[(slice(None),) * a + (0,)] = False
            acc[..., a] -= s * nb_solid
    has = fl & (np.abs(acc).sum(axis=-1) > 0)
    nodes = np.argwhere(has).astype(np.int32)
    n = acc[has]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return nodes, n


def _kernel(fp):
    if fp in _cache:
        return _cache[fp]
    F4, F3, V3, F2 = array_types(fp)
    I2 = ti.types.ndarray(dtype=ti.i32, ndim=2)
    E = math.exp(-KAPPA * B_LOG)

    @ti.func
    def spalding(up):
        ku = ti.min(KAPPA * up, 50.0)
        return up + E * (ti.exp(ku) - 1.0 - ku - ku * ku / 2.0 - ku * ku * ku / 6.0)

    @ti.kernel
    def update(u: V3, nodes: I2, normals: F2, sp: F3, nu0: fp, y: fp):
        for n in range(nodes.shape[0]):
            i, j, k = nodes[n, 0], nodes[n, 1], nodes[n, 2]
            v = u[i, j, k]
            nv = ti.Vector([normals[n, 0], normals[n, 1], normals[n, 2]])
            vt = v - v.dot(nv) * nv
            ut = vt.norm()
            extra = 0.0
            if ut > 1e-8:
                # bisection in log(u_tau) of  y u_tau / nu0 = spalding(ut / u_tau)
                lo = 1e-9
                hi = ut
                if y * hi / nu0 < 1.0:
                    hi = ti.sqrt(nu0 * ut / y) * 1.01
                for _ in range(40):
                    mid = ti.sqrt(lo * hi)
                    if y * mid / nu0 - spalding(ut / mid) > 0.0:
                        hi = mid
                    else:
                        lo = mid
                utau = ti.sqrt(lo * hi)
                nu_w = ti.max(utau * utau * y / ut, nu0)
                extra = -(0.5 + 3.0 * nu_w)        # negative = override the total relaxation time
            sp[i, j, k] = extra

    _cache[fp] = update
    return update


class WallModel:
    """Holds wall-node data on the device and updates the solver's extra-relaxation array."""

    def __init__(self, solver, nodes, normals):
        self.solver = solver
        if solver.sponge.shape != solver.shape:
            raise ValueError("solver must be created with a full-size sponge array (sponge_nu=zeros)")
        self.n = len(nodes)
        self.nodes = ti.ndarray(ti.i32, shape=(max(self.n, 1), 3))
        self.normals = ti.ndarray(solver.fp, shape=(max(self.n, 1), 3))
        if self.n:
            self.nodes.from_numpy(np.ascontiguousarray(nodes, dtype=np.int32))
            self.normals.from_numpy(np.ascontiguousarray(normals, dtype=solver.np_fp))
        self._k = _kernel(solver.fp)

    def update(self):
        """Call right after a step that stored the velocity (solver.step(..., store_last=True))."""
        if self.n:
            self._k(self.solver.u, self.nodes, self.normals, self.solver.sponge, self.solver.nu, 0.5)

    def extra_nu(self) -> np.ndarray:
        """Current added wall viscosity at the wall nodes (lattice units)."""
        sp = self.solver.sponge.to_numpy()
        nd = self.nodes.to_numpy()[: self.n]
        v = sp[nd[:, 0], nd[:, 1], nd[:, 2]]
        return np.where(v < 0, (-v - 0.5) / 3.0, 0.0)
