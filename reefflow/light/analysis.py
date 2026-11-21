"""Uniformity statistics, hot-spot / shadow lists, comparison and sweep tools."""
from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage


def uniformity(a) -> dict:
    """min / mean / max / CV (std/mean) / min-to-mean of the finite values of an array."""
    v = np.asarray(a, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    m = float(v.mean())
    return {"n": int(len(v)), "min": float(v.min()), "mean": m, "max": float(v.max()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "cv": float(v.std() / m) if m > 0 else float("nan"),
            "min_over_mean": float(v.min() / m) if m > 0 else float("nan")}


def regions(map2d, x, y, mask, value_name="par", top: int = 10) -> list:
    """Connected regions of ``mask`` on a 2-D map: area, centroid, peak/mean value."""
    lab, n = ndimage.label(mask)
    out = []
    if n == 0:
        return out
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    X, Y = np.meshgrid(x, y, indexing="ij")
    for r in range(1, n + 1):
        m = lab == r
        v = map2d[m]
        out.append({"area_m2": float(m.sum() * dx * dy), "centroid_m": [float(X[m].mean()), float(Y[m].mean())],
                    f"mean_{value_name}": float(np.nanmean(v)), f"max_{value_name}": float(np.nanmax(v)),
                    f"min_{value_name}": float(np.nanmin(v)), "cells": int(m.sum())})
    out.sort(key=lambda d: -d["area_m2"])
    return out[:top]


def hotspots_and_shadows(map2d, x, y, high: float | None = None, low: float | None = None,
                         min_cells: int = 4) -> dict:
    """Hot spots (> high, default 1.3 x mean) and shadows (< low, default 0.5 x mean)."""
    a = np.asarray(map2d, float)
    fin = np.isfinite(a)
    m = float(np.nanmean(a)) if fin.any() else float("nan")
    hi = 1.3 * m if high is None else high
    lo = 0.5 * m if low is None else low
    hot = [r for r in regions(a, x, y, fin & (a > hi)) if r["cells"] >= min_cells]
    shadow = [r for r in regions(a, x, y, fin & (a < lo)) if r["cells"] >= min_cells]
    return {"threshold_high": hi, "threshold_low": lo, "hot_spots": hot, "shadows": shadow}


def compare(a, b) -> dict:
    """Difference maps (b - a) and stats for two LightResults on the same grid."""
    if a.grid.shape != b.grid.shape or abs(a.grid.dx - b.grid.dx) > 1e-12:
        raise ValueError("results are on different grids")
    out = {"label_a": a.label, "label_b": b.label}
    with np.errstate(invalid="ignore", divide="ignore"):
        if a.plan is not None and b.plan is not None:
            out["plan_diff"] = b.plan - a.plan
            out["plan_ratio"] = b.plan / a.plan
            out["plan_stats"] = [{"depth_m": d, "a": uniformity(a.plan[l]), "b": uniformity(b.plan[l]),
                                  "diff": uniformity(b.plan[l] - a.plan[l])} for l, d in enumerate(a.cfg.map_depths)]
        if a.sand is not None:
            out["sand_diff"] = b.sand - a.sand
            out["sand_stats"] = {"a": uniformity(a.sand), "b": uniformity(b.sand), "diff": uniformity(b.sand - a.sand)}
        if a.rock_meter is not None:
            ta, tb = a.rock_top_mask(), b.rock_top_mask()
            out["rock_meter_diff"] = b.rock_meter - a.rock_meter
            out["rock_top_stats"] = {"a": uniformity(np.where(ta, a.rock_meter, np.nan)),
                                     "b": uniformity(np.where(tb, b.rock_meter, np.nan))}
    return out


def target_values(res, target: str):
    """Values a sweep optimises: 'sand', 'rock_tops' (meter-equivalent), or 'depth:<m>' (plan map)."""
    if target == "sand":
        return res.sand
    if target == "rock_tops":
        return np.where(res.rock_top_mask(), res.rock_meter, np.nan)
    if target.startswith("depth:"):
        d = float(target.split(":")[1])
        l = int(np.argmin(np.abs(np.asarray(res.cfg.map_depths) - d)))
        return res.plan[l]
    raise ValueError(f"unknown sweep target {target!r}")


def sweep_height(project, heights, target: str = "rock_tops", rays: int = 8_000_000, target_mean: float | None = None,
                 model=None, geometry=None, fixtures=None, seed: int = 1, progress=None) -> dict:
    """Trace the project's lighting with every fixture (or those named in ``fixtures``) at each
    height above the water (m) and report uniformity at ``target``.

    The most uniform height minimises the coefficient of variation (CV). PAR is linear in
    fixture output, so for each height the intensity scale that brings the mean to
    ``target_mean`` (umol m^-2 s^-1) is also reported (only meaningful if <= 1 at 100 %)."""
    from .config import lighting_from_project
    from .run import prepare_geometry, run_lighting
    base = copy.deepcopy(project.raw["lighting"])
    rows = []
    geo = geometry
    for h in heights:
        lt = copy.deepcopy(base)
        for f in lt["fixtures"]:
            if fixtures is None or f.get("name") in fixtures:
                f["height"] = float(h)
        lt["calibration"] = None
        lt["map_depths"] = base.get("map_depths")
        cfg = lighting_from_project(project, overrides=lt)
        if geo is None:
            geo = prepare_geometry(project, cfg, model)
        res = run_lighting(project, cfg=cfg, geometry=geo, rays=rays, seed=seed, calibrate=False)
        u = uniformity(target_values(res, target))
        row = {"height_m": float(h), "height_in": float(h) / 0.0254, **u}
        if target_mean is not None and u.get("mean"):
            row["intensity_scale_for_target_mean"] = target_mean / u["mean"]
        rows.append(row)
        if progress:
            progress(len(rows), len(heights))
    best = min(rows, key=lambda r: r.get("cv", np.inf))
    hs = [r["height_m"] for r in rows]
    edge = best["height_m"] in (min(hs), max(hs))
    out = {"target": target, "rows": rows, "most_uniform_height_m": best["height_m"],
           "most_uniform_height_in": best["height_in"], "best": best, "best_at_range_edge": bool(edge)}
    if edge:
        out["note"] = ("the most uniform height is at the edge of the swept range: there is no interior "
                       "optimum in this range (extend the range, or the trend is monotonic)")
    return out
