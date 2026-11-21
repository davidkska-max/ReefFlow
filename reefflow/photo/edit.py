"""Editing tools for the reconstructed occupancy grid (pure functions).

Silhouettes cannot see caves, arches, overhangs or gaps between stacked rocks;
the visual hull fills them (spec Feature 1 item 5). These tools let the user
correct that. Every function returns a NEW array; inputs are not modified.

All coordinates are tank coordinates in metres. A UI can store the edits as a
JSON list and replay them with :func:`apply_edits`, so the cleaned rockscape is
reproducible from (photos, calibration, masks, edits)::

    [{"op": "carve", "shape": {"type": "capsule", "points": [[x,y,z], ...], "radius": r}},
     {"op": "carve", "shape": {"type": "arch", "base_center": [x,y,z], "axis": [ax, ay],
                               "span": w, "rise": h, "length": l}},
     {"op": "carve", "shape": {"type": "sphere", "center": [x,y,z], "radius": r}},
     {"op": "carve", "shape": {"type": "box", "lo": [x,y,z], "hi": [x,y,z]}},
     {"op": "carve", "shape": {"type": "obox", "center": [..], "size": [..], "rotate_z_deg": a}},
     {"op": "add",   "shape": {...same shapes...}}]

"carve" = mark open (water); "add" = add material.
"""
from __future__ import annotations

import numpy as np

from .carve import VoxelGrid


def _sub(grid: VoxelGrid, lo, hi):
    """Index slices of nodes whose centres may lie in [lo, hi], and their centres."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    i0 = np.clip(np.floor((lo - grid.origin) / grid.dx).astype(int), 0, np.array(grid.shape))
    i1 = np.clip(np.ceil((hi - grid.origin) / grid.dx).astype(int) + 1, 0, np.array(grid.shape))
    sl = tuple(slice(a, b) for a, b in zip(i0, i1))
    ax = [grid.origin[a] + grid.dx * np.arange(i0[a], i1[a]) for a in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
    return sl, P


def region(grid: VoxelGrid, shape: dict) -> np.ndarray:
    """Boolean grid of node centres inside a shape (see module doc for shape dicts)."""
    t = shape["type"]
    out = np.zeros(grid.shape, bool)
    if t == "sphere":
        c = np.asarray(shape["center"], float); r = float(shape["radius"])
        sl, P = _sub(grid, c - r, c + r)
        out[sl] = np.sum((P - c) ** 2, axis=-1) <= r * r
    elif t == "box":
        lo, hi = np.asarray(shape["lo"], float), np.asarray(shape["hi"], float)
        sl, P = _sub(grid, lo, hi)
        out[sl] = np.all((P >= lo) & (P <= hi), axis=-1)
    elif t == "obox":
        c = np.asarray(shape["center"], float); s = np.asarray(shape["size"], float) / 2
        a = np.radians(float(shape.get("rotate_z_deg", 0.0)))
        rad = np.linalg.norm(s)
        sl, P = _sub(grid, c - rad, c + rad)
        d = P - c
        lx = d[..., 0] * np.cos(a) + d[..., 1] * np.sin(a)
        ly = -d[..., 0] * np.sin(a) + d[..., 1] * np.cos(a)
        out[sl] = (np.abs(lx) <= s[0]) & (np.abs(ly) <= s[1]) & (np.abs(d[..., 2]) <= s[2])
    elif t == "capsule":
        pts = np.atleast_2d(np.asarray(shape["points"], float)); r = float(shape["radius"])
        if len(pts) == 1:
            return region(grid, {"type": "sphere", "center": pts[0], "radius": r})
        for a, b in zip(pts[:-1], pts[1:]):
            sl, P = _sub(grid, np.minimum(a, b) - r, np.maximum(a, b) + r)
            ab = b - a
            tt = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-18), 0, 1)
            near = np.sum((P - a - tt[..., None] * ab) ** 2, axis=-1) <= r * r
            out[sl] |= near
    elif t == "arch":
        # opening under an arch: rectangle (base .. rise - span/2) topped by a
        # semicircle of radius span/2, extruded `length` along the horizontal axis
        b = np.asarray(shape["base_center"], float)
        ax = np.asarray(list(shape["axis"])[:2] + [0.0], float)
        ax /= np.linalg.norm(ax)
        lat = np.array([-ax[1], ax[0], 0.0])
        w2, rise, ln = float(shape["span"]) / 2, float(shape["rise"]), float(shape["length"])
        if rise < w2:
            raise ValueError("arch rise must be >= span/2 (semicircular top)")
        rad = np.sqrt(w2 ** 2 + rise ** 2 + (ln / 2) ** 2)
        sl, P = _sub(grid, b - rad, b + rad)
        d = P - b
        s, l, z = d @ ax, d @ lat, d[..., 2]
        zc = rise - w2
        inside = (np.abs(s) <= ln / 2) & (np.abs(l) <= w2) & (z >= 0) & (
            (z <= zc) | (l ** 2 + (z - zc) ** 2 <= w2 ** 2))
        out[sl] = inside
    else:
        raise ValueError(f"unknown shape type {t!r}")
    return out


def apply_edit(occ, grid: VoxelGrid, edit: dict) -> np.ndarray:
    reg = region(grid, edit["shape"])
    if edit["op"] in ("carve", "open"):
        return np.asarray(occ, bool) & ~reg
    if edit["op"] == "add":
        return np.asarray(occ, bool) | reg
    raise ValueError("edit op must be 'carve' or 'add'")


def apply_edits(occ, grid: VoxelGrid, edits: list) -> np.ndarray:
    out = np.asarray(occ, bool).copy()
    for e in edits:
        out = apply_edit(out, grid, e)
    return out


# convenience wrappers -------------------------------------------------------
def carve_tunnel(occ, grid, points, radius):
    """Remove material within ``radius`` of a 3D polyline (a tunnel / cave)."""
    return apply_edit(occ, grid, {"op": "carve", "shape": {"type": "capsule", "points": points, "radius": radius}})


def carve_arch(occ, grid, base_center, axis, span, rise, length):
    """Open the passage under an arch (flat floor at base_center z, semicircular top)."""
    return apply_edit(occ, grid, {"op": "carve", "shape": {"type": "arch", "base_center": base_center,
                                                           "axis": axis, "span": span, "rise": rise,
                                                           "length": length}})


def carve_sphere(occ, grid, center, radius):
    return apply_edit(occ, grid, {"op": "carve", "shape": {"type": "sphere", "center": center, "radius": radius}})


def carve_box(occ, grid, lo, hi):
    return apply_edit(occ, grid, {"op": "carve", "shape": {"type": "box", "lo": lo, "hi": hi}})


def mark_open(occ, region_mask):
    """Mark an arbitrary boolean region (same shape as occ) as open water."""
    return np.asarray(occ, bool) & ~np.asarray(region_mask, bool)


def add_material(occ, grid, shape: dict):
    return apply_edit(occ, grid, {"op": "add", "shape": shape})
