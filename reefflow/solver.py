"""Lattice Boltzmann solver (lattice units throughout).

Algorithm per time step
-----------------------
1. ``step`` kernel: for every FLUID node, *pull* populations from upstream
   neighbours (halfway bounce-back for SOLID neighbours, with the Ladd correction
   for moving walls; specular reflection for FREESLIP neighbours), compute moments,
   collide, write post-collision populations.
2. ``open_bc`` kernel: rebuild INLET/OUTLET ghost nodes from their inward neighbour
   by regularized non-equilibrium extrapolation (Guo, Zheng & Shi 2002; Latt et
   al. 2008): f_b = f_eq(rho_b, u_b) + Proj[f_nb - f_eq(rho_nb, u_nb)].

Storage
-------
Populations are stored shifted, g_q = f_q - w_q (Skordos 1993), so fp32
round-off scales with the perturbation, not with rho ~ 1.

Collision operators
-------------------
'bgk' : single relaxation time, 2nd-order equilibrium, Guo et al. (2002) forcing.
'rr'  : recursive regularized (Malaspinas 2015): 3rd-order equilibrium (the
        Hermite terms the lattice can represent); non-equilibrium part
        projected on the 2nd-order Hermite basis and the 3rd-order part
        rebuilt recursively, a3_neq = u a2_neq (symmetrised). Guo forcing.

Both use u = (sum f c + F/2)/rho and a2_neq = Pi - Pi_eq + (uF + Fu)/2, so the
post-collision state is  f* = f_eq + (1 - omega) f_neq' + F_q/2  with
F_q = w [c.F/cs^2 + H2:(uF+Fu)/(2 cs^4)].

Source nodes
------------
SOURCE nodes carry pump forces, mass sources (return nozzle) and mass sinks
(overflow); see sources.py. The force enters the Guo forcing like a body force;
mass is added/removed after collision.

LES
---
Smagorinsky, tau = (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |a2_neq| / rho)) / 2
(Hou et al. 1996): nu_t = (Cs dx)^2 |S|, |S| = sqrt(2 S:S), evaluated locally
from the non-equilibrium stress.
"""
import numpy as np
import taichi as ti

from . import kernels as K
from . import lattice as lat_mod
from .domain import FLUID, SOLID, FREESLIP, INLET, OUTLET, SOURCE, FLUID_LIKE, Domain
from . import sources as S

CS2 = 1.0 / 3.0


class LBMSolver:
    def __init__(self, domain: Domain, *, lattice: str = "D3Q19", collision: str = "rr",
                 nu: float, smagorinsky=None, body_force=(0.0, 0.0, 0.0),
                 force_field: bool = False, sponge_nu=None, sources: "S.SourceSpec | None" = None):
        """sponge_nu: optional (nx, ny, nz) array of extra lattice viscosity (sponge
        layers next to open boundaries, to damp eddies before they reach them).
        sources: SourceSpec for the domain's SOURCE nodes (required iff there are any)."""
        if not domain._final:
            domain.finalize()
        if collision not in ("bgk", "rr"):
            raise ValueError("collision must be 'bgk' or 'rr'")
        if nu <= 0:
            raise ValueError("lattice viscosity must be positive")
        self.domain = domain
        self.lat = lat_mod.get(lattice)
        self.Q = self.lat.Q
        self.collision = collision
        self.shape = domain.shape
        self.nx, self.ny, self.nz = self.shape
        self.smagorinsky = smagorinsky
        self.use_force_field = force_field
        self.open_bc = domain.has_open_boundaries
        self.k = K.get(self.lat.name, collision, smagorinsky is not None, force_field,
                       domain.has_moving_walls, domain.count(FREESLIP) > 0, domain.periodic,
                       sponge=sponge_nu is not None, sources=sources is not None)

        s = self.shape
        fp = self.fp = K.float_type()
        self.np_fp = np.float64 if fp == ti.f64 else np.float32
        self.f0 = ti.ndarray(fp, shape=(self.Q,) + s)
        self.f1 = ti.ndarray(fp, shape=(self.Q,) + s)
        self.flags = ti.ndarray(ti.i8, shape=s)
        self.aux = ti.ndarray(ti.i8, shape=s)
        self.rho = ti.ndarray(fp, shape=s)
        self.u = ti.Vector.ndarray(3, fp, shape=s)
        self.force = ti.Vector.ndarray(3, fp, shape=s if force_field else (1, 1, 1))
        self._line = ti.ndarray(fp, shape=(self.nx, self.ny))
        self.sponge = ti.ndarray(fp, shape=s if sponge_nu is not None else (1, 1, 1))
        if sponge_nu is not None:
            sn = np.asarray(sponge_nu, dtype=np.float64)
            if sn.shape != s or np.any(sn < 0):
                raise ValueError("sponge_nu must be a non-negative array of the domain shape")
            self.sponge.from_numpy(np.ascontiguousarray(sn / CS2, dtype=self.np_fp))

        n_src_nodes = domain.count(SOURCE)
        if (sources is None) != (n_src_nodes == 0):
            raise ValueError("SOURCE nodes and the sources argument must come together")
        self.sources = sources
        self.sidx = ti.ndarray(ti.i32, shape=s if sources is not None else (1, 1, 1))
        if sources is not None:
            sidx = np.full(s, -1, dtype=np.int32)
            c = sources.cells
            sidx[c[:, 0], c[:, 1], c[:, 2]] = np.arange(len(c), dtype=np.int32)
            if not np.array_equal(sidx >= 0, domain.flags == SOURCE):
                raise ValueError("source table cells do not match the SOURCE nodes of the domain")
            self.sidx.from_numpy(sidx)
            self.stab = ti.ndarray(fp, shape=sources.table.shape)
            self.stab.from_numpy(np.ascontiguousarray(sources.table, dtype=self.np_fp))
            self.pst = ti.ndarray(fp, shape=(sources.n_pumps, S.P_NCOL))
            self.set_pump_state(sources.default_state())
        else:
            self.stab = ti.ndarray(fp, shape=(1, S.S_NCOL))
            self.pst = ti.ndarray(fp, shape=(1, S.P_NCOL))
            self.stab.fill(0.0)
            self.pst.fill(0.0)

        self.flags.from_numpy(domain.flags)
        self.aux.from_numpy(domain.aux)
        self.set_viscosity(nu)
        self.cs2 = float(smagorinsky or 0.0) ** 2
        self.g = tuple(float(v) for v in body_force)
        self.time_step = 0
        self._cur = 0
        self.initialize()

    # ------------------------------------------------------------------
    def set_viscosity(self, nu: float):
        self.nu = float(nu)
        self.tau0 = 0.5 + self.nu / CS2

    def set_body_force(self, g):
        self.g = tuple(float(v) for v in g)

    def set_pump_state(self, state: np.ndarray):
        """Pump-state table, shape (n_pumps, sources.P_NCOL); see sources.py."""
        if self.sources is None:
            raise RuntimeError("solver has no sources")
        st = np.asarray(state, dtype=np.float64)
        if st.shape != (self.sources.n_pumps, S.P_NCOL):
            raise ValueError(f"pump state must have shape {(self.sources.n_pumps, S.P_NCOL)}")
        self.pst.from_numpy(np.ascontiguousarray(st, dtype=self.np_fp))

    def set_force_field(self, F: np.ndarray):
        if not self.use_force_field:
            raise RuntimeError("solver created without force_field=True")
        self.force.from_numpy(np.ascontiguousarray(F, dtype=self.np_fp))

    def set_boundary_velocity(self, u_bc: np.ndarray):
        """Overwrite the prescribed velocity on INLET (and SOLID) nodes; array shape (nx,ny,nz,3)."""
        if not hasattr(self, "_tmp_v"):
            self._tmp_v = ti.Vector.ndarray(3, self.fp, shape=self.shape)
        self._tmp_v.from_numpy(np.ascontiguousarray(u_bc, dtype=self.np_fp))
        self.k.set_bc_velocity(self.flags, self.u, self._tmp_v)

    @property
    def f(self):
        return self.f0 if self._cur == 0 else self.f1

    # ------------------------------------------------------------------
    def initialize(self, rho=None, u=None):
        """Populations := equilibrium(rho, u). Defaults: rho = 1, u = 0 in the fluid.

        Boundary nodes take only the quantity they prescribe: INLET (and moving SOLID)
        nodes their velocity, OUTLET nodes their density; the other quantity comes
        from the supplied initial field, so a consistent initial state (e.g. a
        uniform stream) starts without a spurious pressure pulse.
        """
        d = self.domain
        rho_np = np.ones(d.shape) if rho is None else np.asarray(rho, np.float64).copy()
        u_np = d.u_bc.astype(np.float64) if u is None else np.asarray(u, np.float64).copy()
        uset = np.isin(d.flags, (INLET, SOLID))
        rset = np.isin(d.flags, (OUTLET, SOLID))
        u_np[uset] = d.u_bc[uset]
        rho_np[rset] = d.rho_bc[rset]
        self.rho.from_numpy(np.ascontiguousarray(rho_np, dtype=self.np_fp))
        self.u.from_numpy(np.ascontiguousarray(u_np, dtype=self.np_fp))
        self.k.init_eq(self.f0, self.flags, self.rho, self.u)
        self.k.init_eq(self.f1, self.flags, self.rho, self.u)
        self._cur = 0
        self.time_step = 0

    # Batched stepping with Taichi compute graphs (DECISIONS D-701): one pre-recorded
    # launch sequence of n steps instead of n Python-side kernel calls. Results are
    # bit-identical to the plain loop (tests/test_kernels.py).
    use_graph = True
    _GRAPH_MIN = 4

    def _build_graph(self, n):
        A = ti.graph.ArgKind
        fp = self.fp
        nd = lambda name, dt, ndim: ti.graph.Arg(A.NDARRAY, name, dtype=dt, ndim=ndim)  # noqa: E731
        v3 = ti.types.vector(3, fp)
        a = {"f0": nd("f0", fp, 4), "f1": nd("f1", fp, 4), "flags": nd("flags", ti.i8, 3),
             "aux": nd("aux", ti.i8, 3), "rho": nd("rho", fp, 3), "u": nd("u", v3, 3),
             "force": nd("force", v3, 3), "sp": nd("sp", fp, 3), "sidx": nd("sidx", ti.i32, 3),
             "stab": nd("stab", fp, 2), "pst": nd("pst", fp, 2)}
        sc = {nm: ti.graph.Arg(A.SCALAR, nm, dtype=fp) for nm in ("tau0", "cs2", "gx", "gy", "gz")}
        zero = ti.graph.Arg(A.SCALAR, "zero", dtype=ti.i32)
        one = ti.graph.Arg(A.SCALAR, "one", dtype=ti.i32)
        gb = ti.graph.GraphBuilder()
        for i in range(n):
            src, dst = (a["f0"], a["f1"]) if i % 2 == 0 else (a["f1"], a["f0"])
            gb.dispatch(self.k.step, src, dst, a["flags"], a["aux"], a["rho"], a["u"], a["force"], a["sp"],
                        a["sidx"], a["stab"], a["pst"], sc["tau0"], sc["cs2"], sc["gx"], sc["gy"], sc["gz"],
                        one if i == n - 1 else zero)
            if self.open_bc:
                gb.dispatch(self.k.open_bc, dst, a["flags"], a["aux"], a["rho"], a["u"])
        return gb.compile()

    def _step_graph(self, n) -> bool:
        if not hasattr(self, "_graphs"):
            self._graphs = {}
        try:
            if n not in self._graphs:
                self._graphs[n] = self._build_graph(n)
            gx, gy, gz = self.g
            f_src, f_dst = (self.f0, self.f1) if self._cur == 0 else (self.f1, self.f0)
            self._graphs[n].run({"f0": f_src, "f1": f_dst, "flags": self.flags, "aux": self.aux, "rho": self.rho,
                                 "u": self.u, "force": self.force, "sp": self.sponge, "sidx": self.sidx,
                                 "stab": self.stab, "pst": self.pst, "tau0": self.tau0, "cs2": self.cs2,
                                 "gx": gx, "gy": gy, "gz": gz, "zero": 0, "one": 1})
        except Exception:                      # backend without graph support: plain loop from now on
            self.use_graph = False
            return False
        self._cur ^= n % 2
        self.time_step += n
        return True

    def step(self, n: int = 1, store_last: bool = True):
        """Advance n steps; macroscopic fields (rho, u) are written on the last one."""
        if self.use_graph and store_last and n >= self._GRAPH_MIN:
            # graphs of at most 16 steps (each stores rho, u at its end); remainder by the loop
            while n >= self._GRAPH_MIN and self.use_graph:
                c = min(n, 16)
                if not self._step_graph(c):
                    break
                n -= c
            if n == 0:
                return
        k = self.k
        gx, gy, gz = self.g
        for s in range(n):
            store = 1 if (store_last and s == n - 1) else 0
            src, dst = (self.f0, self.f1) if self._cur == 0 else (self.f1, self.f0)
            k.step(src, dst, self.flags, self.aux, self.rho, self.u, self.force, self.sponge,
                   self.sidx, self.stab, self.pst, self.tau0, self.cs2, gx, gy, gz, store)
            if self.open_bc:
                k.open_bc(dst, self.flags, self.aux, self.rho, self.u)
            self._cur ^= 1
            self.time_step += 1

    def update_macros(self):
        gx, gy, gz = self.g
        self.k.macros(self.f, self.flags, self.rho, self.u, self.force, self.sidx, self.stab, self.pst,
                      gx, gy, gz)

    def macros(self):
        """(rho, u) NumPy arrays of the current state. Non-fluid entries hold boundary data."""
        self.update_macros()
        return self.rho.to_numpy(), self.u.to_numpy()

    def total_mass(self) -> float:
        """Sum of rho over FLUID and SOURCE nodes (lattice units), final summation in float64."""
        self.k.mass_lines(self.f, self.flags, self._line)
        return float(self.domain.count(FLUID_LIKE) + np.sum(self._line.to_numpy().astype(np.float64)))

    def populations(self) -> np.ndarray:
        """Full (unshifted) populations, shape (Q, nx, ny, nz), float64."""
        return self.f.to_numpy().astype(np.float64) + self.lat.w[:, None, None, None]

    def memory_bytes(self) -> int:
        n = self.nx * self.ny * self.nz
        return n * (2 * self.Q * 4 + 2 + 4 + 12 + (12 if self.use_force_field else 0)
                    + (4 if self.sources is not None else 0))
