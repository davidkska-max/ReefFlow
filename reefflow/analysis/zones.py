"""Dead zones: connected regions of water whose time-averaged speed <|u|> is below
the threshold (ASSUMPTIONS A-210), ranked by volume with locations.

A zone can be a thin network threading through the tank, so besides the centroid
and bounding box each zone reports its 'core': the point deepest inside it
(largest distance to faster water), which is where to look first."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dead_zones(run, threshold: float, age=None, n_max: int = 10, min_volume_m3: float = 1e-6) -> dict:
    m = run.model
    fl = m.fluid_mask
    dead = fl & (run.speed < threshold)
    lab, n = ndimage.label(dead)        # face connectivity
    zones = []
    if n:
        vol = np.bincount(lab.ravel())[1:] * m.dx ** 3
        order = np.argsort(vol)[::-1]
        X, Y, Z = (m.node_coords(a) for a in range(3))
        objs = ndimage.find_objects(lab)
        depth = ndimage.distance_transform_edt(dead)    # distance to the nearest non-dead node
        for z in order[:n_max]:
            if vol[z] < min_volume_m3:
                break
            sl = objs[z]
            mask = lab[sl] == z + 1
            ii, jj, kk = np.nonzero(mask)
            ii += sl[0].start; jj += sl[1].start; kk += sl[2].start
            zz = {"volume_L": float(vol[z] * 1e3), "volume_fraction": float(vol[z] / (fl.sum() * m.dx ** 3)),
                  "centroid_m": [float(X[ii].mean()), float(Y[jj].mean()), float(Z[kk].mean())],
                  "bbox_m": [[float(X[ii].min()), float(Y[jj].min()), float(Z[kk].min())],
                             [float(X[ii].max()), float(Y[jj].max()), float(Z[kk].max())]],
                  "core_m": None, "core_depth_m": None,
                  "mean_speed_ms": float(run.speed[ii, jj, kk].mean()),
                  "touches": _touches(m, ii, jj, kk)}
            c = int(np.argmax(depth[ii, jj, kk]))
            zz["core_m"] = [float(X[ii[c]]), float(Y[jj[c]]), float(Z[kk[c]])]   # deepest point of the zone
            zz["core_depth_m"] = float(depth[ii[c], jj[c], kk[c]] * m.dx)
            if age is not None:
                zz["mean_age_s"] = float(np.nanmean(age[ii, jj, kk]))
            zones.append(zz)
    return {"mask": dead, "zones": zones, "n_zones": int(n),
            "dead_fraction": float(dead.sum() / fl.sum()), "threshold_ms": threshold}


def _touches(model, ii, jj, kk):
    """Which surfaces a zone lies against (sand / rock / glass / overflow)."""
    from ..tank import TAGS
    t = model.tags
    names = {TAGS["sand"]: "sand", TAGS["rock"]: "rock", TAGS["glass"]: "glass", TAGS["overflow"]: "overflow"}
    found = set()
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        tt = t[np.clip(ii + d[0], 0, t.shape[0] - 1), np.clip(jj + d[1], 0, t.shape[1] - 1),
               np.clip(kk + d[2], 0, t.shape[2] - 1)]
        for v, nme in names.items():
            if np.any(tt == v):
                found.add(nme)
    return sorted(found)


def derating_sensitivity(run, threshold: float, factors=(0.8, 1.0, 1.2)) -> list:
    """Dead-zone fraction if all pump flows were scaled by f (PLAN C4). At these
    Reynolds numbers mean speeds scale ~linearly with source velocity, so the field
    is rescaled rather than rerun; checked once by an actual rerun (D-306)."""
    fl = run.model.fluid_mask
    sp = run.speed[fl]
    return [{"flow_factor": f, "dead_fraction": float(np.mean(sp * f < threshold))} for f in factors]
