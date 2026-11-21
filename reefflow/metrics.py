"""Scalar flow metrics of a time-averaged tank field (Phase 2 subset).

Phase 3 adds water age, wall shear and coral scores. The metrics here are the
ones the grid-convergence study (verification test 6) reports. All inputs are
physical (SI); masks come from the tank model.

Definitions
    speed            <|u|>  time-averaged speed (what a coral or detritus feels;
                     non-zero in fluctuating flow even where the mean velocity is ~0)
    |U|              magnitude of the time-averaged velocity
    sigma            sqrt(2k/3), RMS velocity fluctuation per component
    dead fraction    share of water volume with <|u|> below the threshold
                     (ASSUMPTIONS A-210; |U| version reported alongside)
"""
from __future__ import annotations

import numpy as np

from .tank import TAGS

# metrics compared in the grid study: name -> (label, unit, scale to display)
KEY_METRICS = {
    "mean_speed": ("volume-mean speed <|u|>", "cm/s", 100.0),
    "median_speed": ("median speed", "cm/s", 100.0),
    "dead_fraction": ("dead-zone volume (<|u|> below threshold)", "%", 100.0),
    "mean_sigma": ("volume-mean RMS fluctuation sqrt(2k/3)", "cm/s", 100.0),
    "rock_top_speed": ("speed 15 mm above rock tops", "cm/s", 100.0),
    "sand_speed": ("speed 15 mm above the sand", "cm/s", 100.0),
    "plane_25": ("plan-plane mean speed at 25 % of the water column", "cm/s", 100.0),
    "plane_50": ("plan-plane mean speed at 50 % of the water column", "cm/s", 100.0),
    "plane_75": ("plan-plane mean speed at 75 % of the water column", "cm/s", 100.0),
    "high_fraction": ("volume with <|u|> > 30 cm/s", "%", 100.0),
}
SURFACE_HEIGHT = 0.015   # m: near-surface speeds are sampled at this fixed physical height


def masks(model) -> dict:
    tags = model.tags
    fluid = model.fluid_mask
    below = np.zeros_like(fluid)
    below[:, :, 1:] = tags[:, :, :-1] == TAGS["rock"]
    return {"fluid": fluid, "rock_top_nodes": np.argwhere(fluid & below)}


def _interp_up(speed, fluid, base_nodes, h, dx):
    """Speed at height h above the surface under each base node (the first fluid node
    above a solid; the surface lies dx/2 below it), by linear interpolation along +z.
    Same physical height on every grid. Returns the values where both nodes are fluid."""
    nf = h / dx - 0.5
    n0 = int(np.floor(nf))
    w = nf - n0
    i, j, k = base_nodes[:, 0], base_nodes[:, 1], base_nodes[:, 2] + n0
    nz = speed.shape[2]
    ok = k + 1 < nz
    i, j, k = i[ok], j[ok], k[ok]
    ok = fluid[i, j, k] & fluid[i, j, k + 1]
    i, j, k = i[ok], j[ok], k[ok]
    return (1 - w) * speed[i, j, k] + w * speed[i, j, k + 1]


def _plane(speed, fluid, h, dx):
    """Mean speed on the horizontal plane at height h above the sand (interpolated)."""
    nf = h / dx - 0.5
    k0 = 1 + int(np.floor(nf))
    w = nf - np.floor(nf)
    m = fluid[:, :, k0] & fluid[:, :, k0 + 1]
    return float(((1 - w) * speed[:, :, k0] + w * speed[:, :, k0 + 1])[m].mean())


def compute(model, mean_u: np.ndarray, speed: np.ndarray, tke: np.ndarray | None,
            dead_speed: float, M: dict | None = None) -> dict:
    M = M or masks(model)
    fl = M["fluid"]
    sp = speed[fl]
    Umag = np.linalg.norm(mean_u, axis=-1)[fl]
    out = {
        "water_volume_m3": float(fl.sum() * model.dx ** 3),
        "mean_speed": float(sp.mean()),
        "median_speed": float(np.median(sp)),
        "p10_speed": float(np.percentile(sp, 10)),
        "p90_speed": float(np.percentile(sp, 90)),
        "mean_velocity_magnitude": float(Umag.mean()),
        "dead_speed_threshold": float(dead_speed),
        "dead_fraction": float(np.mean(sp < dead_speed)),
        "dead_fraction_meanvel": float(np.mean(Umag < dead_speed)),
        "high_fraction": float(np.mean(sp > 0.30)),
    }
    dx = model.dx
    rt = _interp_up(speed, fl, M["rock_top_nodes"], SURFACE_HEIGHT, dx)
    out["rock_top_speed"] = float(rt.mean()) if len(rt) else float("nan")
    nx, ny, _ = speed.shape
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    sand_base = np.stack([I.ravel(), J.ravel(), np.ones(I.size, dtype=int)], axis=1)
    out["sand_speed"] = float(_interp_up(speed, fl, sand_base, SURFACE_HEIGHT, dx).mean())
    H = model.project.tank.water_column          # true water column, same on every grid
    for f in (25, 50, 75):
        out[f"plane_{f}"] = _plane(speed, fl, f / 100.0 * H, dx)
    if tke is not None:
        k = np.clip(tke[fl], 0, None)
        out["mean_tke"] = float(k.mean())
        out["mean_sigma"] = float(np.sqrt(2.0 / 3.0 * k).mean())
        out["global_turbulence_intensity"] = float(np.sqrt(2.0 / 3.0 * k.mean()) / sp.mean())
    return out
