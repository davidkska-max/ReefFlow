"""Host side of the photon tracer: scene assembly, batched kernel launches, tallies.

Photons are traced in batches (default 2^22) on the current Taichi backend.
Device tallies are fp32 and zeroed every batch; the host accumulates them in
fp64, so no fp32 accumulator ever sums more than one batch (2^22 < 2^24 keeps
per-batch counts exact in fp32).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .. import backend
from .config import LightingConfig
from .fixtures import N_TABLE
from .sdf import SDF


# ----------------------------------------------------------------------
# tally grid
# ----------------------------------------------------------------------
@dataclass
class Grid:
    """Node grid for the stack / rock / sand tallies: node (i, j, k) at origin + dx (i, j, k)."""
    origin: np.ndarray
    dx: float
    shape: tuple

    def coords(self, axis: int) -> np.ndarray:
        return self.origin[axis] + self.dx * np.arange(self.shape[axis])


def grid_for_tank(tank, dx: float) -> Grid:
    """The node grid ``reefflow.tank.build`` would use for this tank and cell size
    (same layout rule, so lighting results line up with a later CFD model)."""
    nxf = max(int(round(tank.length / dx)), 4)
    nyf = max(int(round(tank.width / dx)), 4)
    nzf = max(int(round(tank.water_column / dx)), 4)
    origin = np.array([0.5 * (tank.length - nxf * dx) - 0.5 * dx,
                       0.5 * (tank.width - nyf * dx) - 0.5 * dx, tank.sand_depth - 0.5 * dx])
    return Grid(origin, float(dx), (nxf + 2, nyf + 2, nzf + 2))


def grid_from_model(model) -> Grid:
    return Grid(np.asarray(model.origin, float), float(model.dx), tuple(int(s) for s in model.shape))


# ----------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------
@dataclass
class Scene:
    L: float
    W: float
    zs: float
    zsand: float
    rim: float
    glass: float
    n_water: float
    n_glass: float
    albedo_sand: float
    albedo_rock: float
    albedo_coralline: float
    roughness: float
    walls: list                      # codes front, back, left, right
    bounces: bool
    has_water: bool
    air_floor: float
    emit: np.ndarray                 # (ne, 16)
    emit_ppf: np.ndarray             # (ne,) umol/s
    profiles: np.ndarray             # (nprof, N_TABLE)
    spectra: np.ndarray              # (ne, nbin) fractions
    kd: np.ndarray                   # (nbin,)
    sdf: SDF | None = None
    grid: Grid | None = None
    stack: bool = True
    plan_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    plan_grid: tuple = (0.0, 0.0, 0.01, 1, 1)   # x0, y0, pixel, nx, ny
    probes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    rough_map: np.ndarray | None = None          # (nx, ny) RMS slope
    rough_grid: tuple = (0.0, 0.0, 1.0)          # x0, y0, dx
    max_events: int = 200

    @property
    def total_ppf(self) -> float:
        return float(self.emit_ppf.sum())


def build_scene(cfg: LightingConfig, tank=None, *, grid: Grid | None = None, sdf: SDF | None = None,
                plan: bool = True, probes=None, has_water: bool = True, air_floor: float = -10.0,
                fixtures=None, box=None, rough_map=None, rough_grid=None, stack: bool | None = None) -> Scene:
    """Assemble a Scene. ``tank`` is a reefflow.project.Tank; ``box`` = (L, W, z_sand, z_s, rim)
    replaces it for synthetic scenes (verification, open water)."""
    fx = cfg.fixtures if fixtures is None else fixtures
    if box is None:
        L, W, zsand, zs, rim, g = (tank.length, tank.width, tank.sand_depth, tank.water_surface, tank.height,
                                   tank.glass)
    else:
        L, W, zsand, zs, rim = box[:5]
        g = box[5] if len(box) > 5 else 0.0127
    rows, ppf, prof_list, spec = [], [], [], []
    prof_key = {}
    for fi, f in enumerate(fx):
        key = (f.distribution.kind, round(f.distribution.m, 9), round(f.distribution.cutoff, 9),
               None if f.distribution.table_theta is None else tuple(f.distribution.table_value))
        if key not in prof_key:
            prof_key[key] = len(prof_list)
            prof_list.append(f.distribution.inverse_cdf(N_TABLE))
        r, p = f.emitter_rows(prof_key[key], fi)
        rows.append(r); ppf.append(p)
        sp = f.spectrum(cfg.spectra) if cfg.spectral else np.ones(1)
        spec.append(np.repeat(sp[None, :], len(r), 0))
    emit = np.vstack(rows); ppf = np.concatenate(ppf)
    if ppf.sum() <= 0:
        raise ValueError("lighting: fixtures emit no light (all intensities 0?)")
    pg = (0.0, 0.0, cfg.map_resolution, 1, 1)
    pz = np.zeros(0)
    if plan and has_water:
        res = cfg.map_resolution
        pg = (0.0, 0.0, res, max(int(math.ceil(L / res)), 1), max(int(math.ceil(W / res)), 1))
        pz = np.array([zs - d for d in cfg.map_depths], float)
    return Scene(L=L, W=W, zs=zs, zsand=zsand, rim=rim, glass=g, n_water=cfg.n_water, n_glass=cfg.n_glass,
                 albedo_sand=cfg.albedo_sand, albedo_rock=cfg.albedo_rock, albedo_coralline=cfg.coralline_albedo,
                 roughness=cfg.roughness, walls=cfg.wall_codes(), bounces=cfg.bounces, has_water=has_water,
                 air_floor=air_floor, emit=emit, emit_ppf=ppf, profiles=np.array(prof_list),
                 spectra=np.vstack(spec), kd=cfg.kd_bins(), sdf=sdf, grid=grid,
                 stack=(grid is not None) if stack is None else stack, plan_z=pz, plan_grid=pg,
                 probes=np.zeros((0, 4)) if probes is None else np.asarray(probes, float).reshape(-1, 4),
                 rough_map=rough_map, rough_grid=rough_grid or (0.0, 0.0, 1.0), max_events=cfg.max_events)


# ----------------------------------------------------------------------
# tallies
# ----------------------------------------------------------------------
@dataclass
class Tallies:
    """Sums of photon weight (w) and squared weight (w2) over all traced photons.
    Multiply weights by ``ppf_per_photon`` for umol/s."""
    photons: int
    ppf_per_photon: float
    stack_w: np.ndarray | None
    stack_w2: np.ndarray | None          # not tallied (D-410): use 1/sqrt(stack_w)
    rock_w: np.ndarray | None
    rock_w2: np.ndarray | None
    sand_w: np.ndarray | None
    sand_w2: np.ndarray | None
    plan_w: np.ndarray
    plan_w2: np.ndarray
    probe_w: np.ndarray
    probe_w2: np.ndarray
    counters: dict
    batch_probe_w: list
    seconds: float
    arch: str
    max_cell_weight_per_batch: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def rays_per_second(self) -> float:
        return self.photons / self.seconds if self.seconds > 0 else float("nan")

    def energy_balance(self) -> dict:
        """Fractions of the emitted photon weight ending in each sink, and the closure residual."""
        c = self.counters
        e = c["emitted"]
        sinks = ["missed_tank", "surface_reflected", "absorbed_water", "absorbed_sand", "absorbed_rock",
                 "lost_through_glass", "absorbed_black_wall", "absorbed_coralline", "escaped_surface",
                 "terminated_event_cap", "reached_floor_air", "lost_other", "emitted_upward"]
        fr = {k: c[k] / e for k in sinks}
        fr["closure_residual"] = 1.0 - sum(fr.values())
        return fr


def _rel_from_sums(w, w2):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, np.sqrt(w2) / w, np.inf)


def trace(scene: Scene, rays: int, seed: int = 1, batch: int = 1 << 22, target_noise: float | None = None,
          max_rays: int | None = None, noise_fn=None, progress=None) -> Tallies:
    """Trace ``rays`` photons (or, with ``target_noise``, until ``noise_fn(tallies)`` <= target,
    capped at ``max_rays``). Returns fp64 host tallies."""
    import taichi as ti
    from . import kernels as K
    if backend.current() is None:
        backend.init()
    if backend.precision() != "f32":
        raise RuntimeError("the light tracer kernels are fp32; initialise the backend with precision='f32'")
    s = scene
    P = np.zeros(K.NPF, np.float32)
    IP = np.zeros(K.NPI, np.int32)
    P[K.P_L], P[K.P_W], P[K.P_ZS], P[K.P_ZSAND], P[K.P_RIM], P[K.P_G] = s.L, s.W, s.zs, s.zsand, s.rim, s.glass
    P[K.P_NW], P[K.P_NG] = s.n_water, s.n_glass
    P[K.P_ALB_SAND], P[K.P_ALB_ROCK], P[K.P_ALB_COR] = s.albedo_sand, s.albedo_rock, s.albedo_coralline
    P[K.P_ROUGH], P[K.P_ZFLOOR] = s.roughness, s.air_floor
    ne = len(s.emit)
    IP[K.I_NEMIT], IP[K.I_NTAB], IP[K.I_NBIN] = ne, s.profiles.shape[1], len(s.kd)
    IP[K.I_WALL0:K.I_WALL0 + 4] = s.walls
    IP[K.I_MAXEV], IP[K.I_HASWATER], IP[K.I_BOUNCE] = s.max_events, int(s.has_water), int(s.bounces)
    # rock
    if s.sdf is not None and s.has_water:
        sdf_arr = np.ascontiguousarray(s.sdf.phi, np.float32)
        P[K.P_SDF_H] = s.sdf.h
        P[K.P_SDF_OX:K.P_SDF_OZ + 1] = s.sdf.lo
        IP[K.I_HASROCK] = 1
    else:
        sdf_arr = np.ones((2, 2, 2), np.float32)
        P[K.P_SDF_H] = 1e-3
    IP[K.I_SNX:K.I_SNZ + 1] = sdf_arr.shape
    # node grid
    if s.grid is not None:
        gshape = tuple(s.grid.shape)
        P[K.P_GOX:K.P_GOZ + 1] = s.grid.origin
        P[K.P_GDX] = s.grid.dx
    else:
        gshape = (1, 1, 1)
        P[K.P_GDX] = 1.0
    IP[K.I_GNX:K.I_GNZ + 1] = gshape
    IP[K.I_STACK] = int(s.stack and s.grid is not None)
    # plan planes
    x0, y0, pdx, pnx, pny = s.plan_grid
    P[K.P_PX0], P[K.P_PY0], P[K.P_PDX] = x0, y0, pdx
    npl = len(s.plan_z)
    IP[K.I_NPLAN], IP[K.I_PNX], IP[K.I_PNY] = npl, pnx, pny
    IP[K.I_NPROBE] = len(s.probes)
    if s.rough_map is not None:
        IP[K.I_ROUGHMAP] = 1
        IP[K.I_RNX], IP[K.I_RNY] = s.rough_map.shape
        P[K.P_RX0], P[K.P_RY0], P[K.P_RDX] = s.rough_grid
        rough = np.ascontiguousarray(s.rough_map, np.float32)
    else:
        rough = np.zeros((1, 1), np.float32)
        IP[K.I_RNX] = IP[K.I_RNY] = 1
        P[K.P_RDX] = 1.0
    cdf = np.cumsum(s.emit_ppf) / s.emit_ppf.sum()
    cdf[-1] = 1.0
    scdf = np.cumsum(s.spectra, 1)
    scdf = scdf / scdf[:, -1:]

    def nd(a, dt=ti.f32):
        x = ti.ndarray(dt, a.shape)
        x.from_numpy(np.ascontiguousarray(a, np.float32 if dt == ti.f32 else np.int32))
        return x

    # Accumulator precision (DECISIONS D-409). Grid tallies (stack / rock / sand) are fp32: a
    # cell sees ~1e2-1e3 photons per batch, so the systematic rounding of repeated near-equal
    # additions is <= n x 2^-24 ~ 1e-5 (checked per batch below), and fp64 atomics cost ~2x
    # run time. Plan maps, probes and counters can be hot (a collimated beam puts every photon
    # in one cell), so they are fp64 where the backend has fp64 atomics (CUDA, CPU).
    hot_dt = ti.f64 if backend.current() in ("cuda", "cpu") else ti.f32

    def zeros(shape, dt=ti.f32):
        x = ti.ndarray(dt, shape)
        x.fill(0.0)
        return x

    d_P, d_IP = nd(P), nd(IP, ti.i32)
    d_emit, d_cdf = nd(s.emit), nd(cdf)
    d_prof, d_scdf, d_kd = nd(s.profiles), nd(scdf), nd(np.asarray(s.kd, float))
    d_rough, d_sdf = nd(rough), nd(sdf_arr)
    d_pz = nd(s.plan_z if npl else np.zeros(1))
    d_probes = nd(s.probes if len(s.probes) else np.zeros((1, 4)))
    has_grid = s.grid is not None
    shapes = {"sw": gshape if (has_grid and IP[K.I_STACK]) else (1, 1, 1), "rw": gshape if has_grid else (1, 1, 1),
              "aw": gshape[:2] if has_grid else (1, 1), "pw": (max(npl, 1), pnx, pny),
              "qw": (max(len(s.probes), 1),), "C": (K.REPLICAS, K.NC)}
    dts = {"sw": ti.f32, "rw": ti.f32, "aw": ti.f32, "pw": hot_dt, "qw": hot_dt}
    dev = {k: (zeros(v, dts[k]), zeros(v, dts[k])) for k, v in shapes.items() if k != "C"}
    d_C = zeros(shapes["C"], hot_dt)
    max_cell_batch = 0.0
    acc = {k: (np.zeros(v), np.zeros(v)) for k, v in shapes.items() if k != "C"}
    cacc = np.zeros(K.NC)
    batch_probe = []
    max_rays = max_rays or rays
    goal = rays if target_noise is None else max_rays
    done = 0
    b = 0
    t0 = time.perf_counter()
    while done < goal:
        n = int(min(batch, goal - done))
        for k in dev:
            dev[k][0].fill(0.0); dev[k][1].fill(0.0)
        d_C.fill(0.0)
        seed_b = (int(seed) * 0x9E3779B1 + b * 0x85EBCA77 + 0x165667B1) & 0xFFFFFFFF
        K.trace(n, seed_b, d_P, d_IP, d_emit, d_cdf, d_prof, d_scdf, d_kd, d_rough, d_sdf,
                dev["sw"][0], dev["rw"][0], dev["rw"][1], dev["aw"][0], dev["aw"][1],
                d_pz, dev["pw"][0], dev["pw"][1], d_probes, dev["qw"][0], dev["qw"][1], d_C)
        for k in dev:
            a = dev[k][0].to_numpy()
            if dts[k] == ti.f32 and a.size > 1:
                max_cell_batch = max(max_cell_batch, float(a.max()))
            acc[k][0][...] += a
            acc[k][1][...] += dev[k][1].to_numpy()
        cacc += d_C.to_numpy().astype(np.float64).sum(0)
        batch_probe.append(dev["qw"][0].to_numpy().astype(np.float64)[:len(s.probes)].copy())
        done += n
        b += 1
        if progress:
            progress(done, goal)
        if target_noise is not None and done >= min(rays, goal) and noise_fn is not None:
            tal = _make(acc, cacc, done, s, batch_probe, time.perf_counter() - t0, npl, has_grid, IP[K.I_STACK])
            if noise_fn(tal) <= target_noise:
                break
    ti.sync()
    tal = _make(acc, cacc, done, s, batch_probe, time.perf_counter() - t0, npl, has_grid, IP[K.I_STACK])
    tal.max_cell_weight_per_batch = max_cell_batch
    if max_cell_batch > 2 ** 15:
        tal.warnings.append(f"a grid cell received {max_cell_batch:.0f} photons in one batch: fp32 accumulation "
                            f"error up to ~{max_cell_batch * 2 ** -24:.1e} relative; use a smaller batch")
    return tal


def _make(acc, cacc, done, s, batch_probe, secs, npl, has_grid, stack) -> Tallies:
    from . import kernels as K
    counters = {name: float(cacc[i]) for i, name in enumerate(K.COUNTER_NAMES)}
    g = has_grid
    return Tallies(photons=int(done), ppf_per_photon=s.total_ppf / done,
                   stack_w=acc["sw"][0] if (g and stack) else None, stack_w2=None,
                   rock_w=acc["rw"][0] if g else None, rock_w2=acc["rw"][1] if g else None,
                   sand_w=acc["aw"][0] if g else None, sand_w2=acc["aw"][1] if g else None,
                   plan_w=acc["pw"][0][:npl], plan_w2=acc["pw"][1][:npl],
                   probe_w=acc["qw"][0][:len(s.probes)], probe_w2=acc["qw"][1][:len(s.probes)],
                   counters=counters, batch_probe_w=batch_probe, seconds=secs, arch=backend.current())
