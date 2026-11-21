"""Top-level lighting run: project (+ optional CFD TankModel) -> LightResult.

    from reefflow import project
    from reefflow.light import run_lighting, par_on_cfd_grid
    proj = project.load("tank.yaml")
    res = run_lighting(proj)                       # own node grid (project preset dx)
    res = run_lighting(proj, model=tank_model)     # on the CFD grid -> par_on_cfd_grid(model, res)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .. import backend
from .config import LightingConfig, lighting_from_project
from .results import LightResult, noise_function
from .sdf import SDF
from .tracer import Grid, build_scene, grid_for_tank, grid_from_model, trace


@dataclass
class Geometry:
    """Rock SDF, tally grid and rock patches for one tank: independent of the fixtures, so
    it is built once and reused by sweeps, comparisons and calibration."""
    grid: Grid
    sdf: SDF | None
    patches: dict | None
    source: str


def rock_sdf(project, cfg: LightingConfig, model=None) -> SDF | None:
    """Rock SDF from the project's STL meshes (preferred) or the CFD voxel grid."""
    from .. import geometry as G
    t = project.tank
    mode = cfg.rock_geometry
    if mode == "auto":
        mode = "stl" if project.rocks else ("voxels" if model is not None else "none")
    if mode == "none":
        return None
    if mode == "stl":
        if not project.rocks:
            raise ValueError("lighting.rock_geometry = stl but the project has no rock meshes")
        vs, fs, off = [], [], 0
        for r in project.rocks:
            v, f = G.load_mesh(r.file, r.unit, r.translate, r.rotate_z, r.scale)
            vs.append(v); fs.append(f + off); off += len(v)
        h = cfg.sdf_resolution
        lo = np.array([0.0, 0.0, t.sand_depth - 3 * h])
        hi = np.array([t.length, t.width, t.water_surface + 3 * h])
        return SDF.from_mesh(np.vstack(vs), np.vstack(fs), h, lo, hi, source="stl")
    if mode == "voxels":
        if model is None:
            raise ValueError("lighting.rock_geometry = voxels needs the CFD TankModel")
        from ..tank import TAGS
        m = model.tags == TAGS["rock"]
        # rock continues into the sand layer and the glass layer (no spurious faces there)
        m[:, :, 0] |= m[:, :, 1]
        m[0] |= m[1]; m[-1] |= m[-2]; m[:, 0] |= m[:, 1]; m[:, -1] |= m[:, -2]
        return SDF.from_mask(m, model.origin, model.dx, smooth=1.0, source="voxels")
    raise ValueError(f"unknown rock_geometry {mode!r}")


def prepare_geometry(project, cfg: LightingConfig, model=None, dx: float | None = None) -> Geometry:
    grid = grid_from_model(model) if model is not None else grid_for_tank(project.tank, dx or project.dx())
    sdf = rock_sdf(project, cfg, model)
    patches = None
    if sdf is not None:
        patches = sdf.patches(grid.origin, grid.dx, grid.shape, zmin=project.tank.sand_depth,
                              zmax=project.tank.water_surface)
    return Geometry(grid, sdf, patches, sdf.source if sdf is not None else "none")


def run_lighting(project, model=None, cfg: LightingConfig | None = None, *, geometry: Geometry | None = None,
                 dx: float | None = None, rays: int | None = None, target_noise: float | None = None,
                 seed: int | None = None, arch: str | None = None, calibrate: bool = True,
                 roughness_map=None, progress=None) -> LightResult:
    """Trace the project's lighting. With ``model`` (reefflow.tank.TankModel) the tallies use
    the CFD node grid, so ``par_on_cfd_grid(model, result)`` can be called.

    Calibration (``lighting.calibration``) is fitted first when present and ``calibrate``:
    a fitted beam angle replaces the fixtures' distribution; the fitted scale multiplies
    every result, which is then labelled 'calibrated'.

    Ray count: an explicit ``rays`` traces exactly that many photons (unless ``target_noise`` is
    also given); otherwise ``lighting.rays`` is the minimum and tracing continues until the
    average-lit surface patch reaches ``lighting.target_noise`` (default 2 %), capped at
    ``lighting.max_rays``. ``roughness_map`` = (sigma (nx, ny), (x0, y0, dx)) replaces the
    uniform surface roughness (see ``roughness_from_cfd``)."""
    if backend.current() is None:
        backend.init(arch or "auto")
    cfg = cfg or lighting_from_project(project)
    geo = geometry or prepare_geometry(project, cfg, model, dx)
    scale, cal = 1.0, None
    notes = []
    if calibrate and cfg.calibration:
        from .calibration import calibrate as fit_cal
        cal = fit_cal(project, cfg, geo)
        if cal.get("ok"):
            scale = cal["scale"]
            if cal.get("beam_angle_deg") is not None:
                cfg = cal["cfg"]
                notes.append(f"beam angle calibrated to {cal['beam_angle_deg']:.1f} deg")
        cal = {k: v for k, v in cal.items() if k != "cfg"}
    rm, rg = (None, None) if roughness_map is None else roughness_map
    scene = build_scene(cfg, project.tank, grid=geo.grid, sdf=geo.sdf, rough_map=rm, rough_grid=rg)
    tn = target_noise if (target_noise is not None or rays is not None) else cfg.target_noise
    nrays = cfg.rays if rays is None else int(rays)
    fn = noise_function(geo.grid, geo.patches) if tn is not None else None
    tal = trace(scene, nrays, seed=cfg.seed if seed is None else seed, batch=cfg.batch, target_noise=tn,
                max_rays=max(cfg.max_rays, nrays), noise_fn=fn, progress=progress)
    notes.append(f"{tal.photons:.3g} photons traced")
    notes.extend(tal.warnings)
    if geo.source == "voxels":
        notes.append("rock geometry from the CFD voxel grid (smoothed SDF): normals on sloped rock are "
                     "approximate; prefer the STL source")
    return LightResult(cfg=cfg, tank=project.tank, grid=geo.grid, tallies=tal, sdf=geo.sdf, patches=geo.patches,
                       scale=scale, calibration=cal, notes=notes)


def with_lighting(project, lighting: dict):
    """Copy of a Project with its raw 'lighting' section replaced (sweeps / comparisons)."""
    from .. import project as PR
    raw = copy.deepcopy(project.raw)
    raw["lighting"] = copy.deepcopy(lighting)
    return PR.parse(raw, project.path)


def roughness_from_cfd(model, mean_speed, slope_per_ms: float, sigma0: float = 0.0, sigma_max: float = 0.3):
    """Water-surface roughness map from the CFD's time-mean speed in the top fluid layer
    (the optional 'surface agitation driven by near-surface velocity' hook of the spec).

    sigma = min(sigma0 + slope_per_ms * |U_top|, sigma_max)   (RMS facet slope)

    HEURISTIC: no validated relation between current speed under a rigid-lid LES and the
    agitation of a tank surface exists (Cox-Munk is for wind waves). ``slope_per_ms`` has no
    default on purpose; it must come from the user (e.g. matched to photos of the caustic
    pattern). ``mean_speed`` is an (nx, ny, nz) array on ``model``'s grid (m/s).
    Returns (sigma (nx, ny), (x0, y0, dx)) for run_lighting(roughness_map=...)."""
    U = np.asarray(mean_speed, float)
    if U.shape != tuple(model.shape):
        raise ValueError("mean_speed must be on the TankModel grid")
    top = U[:, :, -2]                                  # first fluid layer under the lid
    sig = np.clip(sigma0 + slope_per_ms * np.nan_to_num(top), 0.0, sigma_max)
    return sig, (float(model.origin[0] - 0.5 * model.dx), float(model.origin[1] - 0.5 * model.dx), float(model.dx))
