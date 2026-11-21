"""Coral placement scores on exposed rock surfaces (flow now; light when a PAR
field is supplied, Phase 4).

Surface points: fluid nodes face-adjacent to rock whose outward normal (gradient
of a smoothed rock indicator) is not pointing down, n_z > -0.25: rock tops and
upper sides, excluding overhang ceilings (spec item 4).

Flow is sampled 15 mm out from the rock surface along the normal (trilinear
interpolation). The spec says "one cell off the surface"; a fixed physical
distance is used instead so the score does not change with the grid (one cell is
10, 5 or 3 mm depending on the preset; the near-wall speed changes with it,
verification 6). Logged as D-305.

Sub-scores in [0, 1] per profile (data/coral_profiles.yaml):
  speed   trapezoid on <|u|>: 0 at 0.5 lo, 1 on [lo, hi], 0 at 1.5 hi
  turb    modifier 0.6 + 0.4 clip((TI - ti0)/(ti1 - ti0)), TI = sigma/<|u|>
  jet     1 - penalty; penalty = ramp(speed: jet_speed -> 1.5 jet_speed) x ramp(TI: jet_ti -> 0)
  flow    = speed x turb x jet
  light   trapezoid on meter-equivalent PAR: 0 at 0.7 lo, 1 on [lo, hi], 0 at par_limit
  combined = flow x light   (flow only, flagged, when no PAR is available)
"""
from __future__ import annotations

from importlib import resources

import numpy as np
import yaml
from scipy import ndimage

SAMPLE_DISTANCE = 0.015
NZ_MIN = -0.25


def load_profiles(project=None) -> dict:
    with resources.files("reefflow.data").joinpath("coral_profiles.yaml").open("r", encoding="utf-8") as fh:
        prof = yaml.safe_load(fh)
    if project is not None:
        extra = ((project.raw.get("criteria") or {}).get("coral_profiles")) or {}
        for name, p in extra.items():
            prof[name] = {**prof.get(name, {}), **p}
    return prof


def _ramp(x, a, b):
    """0 at a, 1 at b (works for a > b too)."""
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _trap(x, z0, lo, hi, z1):
    return np.minimum(_ramp(x, z0, lo), _ramp(x, z1, hi))


def surface_points(model):
    """(nodes (n,3), outward unit normals (n,3), sample points in index coordinates (n,3))."""
    from ..tank import TAGS
    rock = (model.tags == TAGS["rock"]).astype(float)
    fl = model.fluid_mask
    adj = np.zeros_like(fl)
    for a in range(3):
        for s in (1, -1):
            adj |= np.roll(model.tags == TAGS["rock"], s, axis=a)
    cand = fl & adj
    phi = ndimage.gaussian_filter(rock, 1.0)
    g = np.stack(np.gradient(phi), axis=-1)
    nodes = np.argwhere(cand)
    n = -g[nodes[:, 0], nodes[:, 1], nodes[:, 2]]
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-6
    nodes, n = nodes[ok], n[ok] / norm[ok, None]
    keep = n[:, 2] > NZ_MIN
    nodes, n = nodes[keep], n[keep]
    d = SAMPLE_DISTANCE / model.dx - 0.5          # node lies ~dx/2 off the surface
    pts = nodes + n * d
    return nodes, n, pts


def score(run, profiles: dict, par=None, sample=None) -> dict:
    """Per-profile scores at the rock surface points. par: optional (nx,ny,nz)
    meter-equivalent PAR array (NaN where not evaluated)."""
    m = run.model
    nodes, normals, pts = sample or surface_points(m)
    coords = pts.T
    sp = ndimage.map_coordinates(run.speed, coords, order=1, mode="nearest")
    sg = ndimage.map_coordinates(run.sigma, coords, order=1, mode="nearest")
    ti = sg / np.maximum(sp, 1e-4)
    par_v = None
    if par is not None:
        par_v = par[nodes[:, 0], nodes[:, 1], nodes[:, 2]]
    out = {"nodes": nodes, "normals": normals, "speed": sp, "sigma": sg, "ti": ti, "par": par_v, "profiles": {}}
    for name, p in profiles.items():
        lo, hi = p["speed_ideal"]
        s_speed = _trap(sp, 0.5 * lo, lo, hi, 1.5 * hi)
        t0, t1 = p["turbulence"]
        s_turb = 0.6 + 0.4 * _ramp(ti, t0, t1)
        pen = _ramp(sp, p["jet_speed"], 1.5 * p["jet_speed"]) * _ramp(ti, p["jet_ti"], 0.0)
        s_jet = 1.0 - pen
        flow = s_speed * s_turb * s_jet
        if par_v is not None and np.any(np.isfinite(par_v)):
            plo, phi_ = p["par_ideal"]
            light = _trap(np.nan_to_num(par_v, nan=0.0), 0.7 * plo, plo, phi_, p["par_limit"])
            light = np.where(np.isfinite(par_v), light, np.nan)
            combined = flow * light
        else:
            light = None
            combined = flow
        out["profiles"][name] = {"speed": s_speed, "turb": s_turb, "jet": s_jet, "flow": flow,
                                 "light": light, "combined": combined}
    return out


def best_spots(model, sc: dict, profile: str, n: int = 10, min_sep: float = 0.05) -> list:
    """Greedy non-maximum suppression on the combined score (minimum spacing min_sep)."""
    P = sc["profiles"][profile]
    s = np.nan_to_num(P["combined"], nan=-1.0)
    order = np.argsort(s)[::-1]
    X, Y, Z = (model.node_coords(a) for a in range(3))
    xyz = np.stack([X[sc["nodes"][:, 0]], Y[sc["nodes"][:, 1]], Z[sc["nodes"][:, 2]]], axis=1)
    picked = []
    for i in order:
        if s[i] <= 0 or len(picked) >= n:
            break
        if all(np.linalg.norm(xyz[i] - xyz[j]) >= min_sep for j in picked):
            picked.append(i)
    out = []
    for i in picked:
        d = {"position_m": xyz[i].tolist(), "score": float(s[i]), "flow_score": float(P["flow"][i]),
             "speed_ms": float(sc["speed"][i]), "turbulence_intensity": float(sc["ti"][i]),
             "normal_z": float(sc["normals"][i, 2])}
        if P["light"] is not None:
            d["light_score"] = float(P["light"][i])
            d["par"] = float(sc["par"][i])
        out.append(d)
    return out


def summary(sc: dict) -> dict:
    res = {"surface_points": int(len(sc["nodes"])),
           "speed_median_ms": float(np.median(sc["speed"])), "ti_median": float(np.median(sc["ti"])),
           "light_available": sc["par"] is not None, "profiles": {}}
    for name, P in sc["profiles"].items():
        c = np.nan_to_num(P["combined"], nan=0.0)
        res["profiles"][name] = {"mean_score": float(c.mean()), "fraction_score_ge_0.7": float(np.mean(c >= 0.7)),
                                 "fraction_flow_ge_0.7": float(np.mean(P["flow"] >= 0.7)),
                                 "fraction_jet_penalised": float(np.mean(P["jet"] < 0.99))}
    return res


def rock_mesh_with_scores(model, sc: dict, profile: str, path):
    """Marching-cubes rock surface coloured by the nearest surface-point score -> PLY
    (vertex colours; open in ParaView/MeshLab). Undersides (no score) are grey."""
    import matplotlib.cm as cm
    import trimesh
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    from ..tank import TAGS
    rock = ndimage.gaussian_filter((model.tags == TAGS["rock"]).astype(float), 0.7)
    rock[:, :, 0] = 0.0
    v, f, _, _ = marching_cubes(rock, 0.5)
    xyz = model.origin + v * model.dx
    tree = cKDTree(sc["nodes"].astype(float))
    d, i = tree.query(v, distance_upper_bound=2.0)
    s = np.full(len(v), np.nan)
    ok = np.isfinite(d)
    s[ok] = np.nan_to_num(sc["profiles"][profile]["combined"], nan=0.0)[i[ok]]
    col = (cm.viridis(np.nan_to_num(s, nan=0.0)) * 255).astype(np.uint8)
    col[~ok] = (150, 150, 150, 255)
    trimesh.Trimesh(xyz, f[:, ::-1], vertex_colors=col, process=False).export(str(path))
    return path
