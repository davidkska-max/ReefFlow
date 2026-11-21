"""Geometry: mesh import, voxelization, procedural rock, oriented-region masks.

Coordinates are tank coordinates in metres: origin at the inside bottom-left-front
corner of the glass, x along the length (left to right seen from the front),
y from front glass to back glass, z up from the inside of the bottom glass.

Voxelization rule: a node is inside a closed mesh if its *centre* is inside
(ray-parity along +z, one ray per (x, y) node column). The staircase surface
then lies within dx/2 of the true surface (ASSUMPTIONS A-106).
"""
from __future__ import annotations

import numpy as np

from . import units as U


# ----------------------------------------------------------------------
# meshes
# ----------------------------------------------------------------------
def load_mesh(path, unit: str = "mm", translate=(0.0, 0.0, 0.0), rotate_z_deg: float = 0.0,
              scale: float = 1.0):
    """Load an STL/OBJ/PLY mesh (trimesh) and return (vertices [m], faces).

    The file's numbers are in ``unit``; then scale, rotation about +z (degrees,
    about the mesh's own origin) and translation (m) are applied in that order."""
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=True)
    v = np.asarray(m.vertices, dtype=np.float64) * U.to_si(1.0, unit) * float(scale)
    if rotate_z_deg:
        a = np.radians(rotate_z_deg)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        v = v @ R.T
    v = v + np.asarray(translate, dtype=np.float64)
    return v, np.asarray(m.faces, dtype=np.int64)


def mesh_report(vertices, faces) -> dict:
    import trimesh
    m = trimesh.Trimesh(vertices, faces, process=False)
    return {"triangles": int(len(faces)), "watertight": bool(m.is_watertight),
            "volume_m3": float(abs(m.volume)) if m.is_watertight else None,
            "bounds_m": m.bounds.tolist()}


def save_stl(path, vertices, faces, unit: str = "mm"):
    import trimesh
    trimesh.Trimesh(np.asarray(vertices) / U.to_si(1.0, unit), faces, process=False).export(str(path))


def voxelize_mesh(vertices, faces, origin, dx: float, shape) -> np.ndarray:
    """Boolean (nx, ny, nz) mask of node centres inside the closed mesh.

    Node (i, j, k) has centre origin + (i, j, k) * dx. Ray parity along +z: every
    triangle crossing a column's vertical line toggles inside/outside above the
    crossing. Column positions get a tiny irrational offset so rays never pass
    exactly through vertices or edges."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    nx, ny, nz = (int(s) for s in shape)
    o = np.asarray(origin, dtype=np.float64)
    eps = dx * np.array([1.234567e-6, 2.345671e-6])
    # triangle vertices in node-index units (x, y), keep z physical
    T = v[f]                                             # (M, 3, 3)
    X = (T[:, :, 0] - o[0] - eps[0]) / dx
    Y = (T[:, :, 1] - o[1] - eps[1]) / dx
    Z = T[:, :, 2]
    i0 = np.clip(np.ceil(X.min(1)).astype(np.int64), 0, nx)
    i1 = np.clip(np.floor(X.max(1)).astype(np.int64), -1, nx - 1)
    j0 = np.clip(np.ceil(Y.min(1)).astype(np.int64), 0, ny)
    j1 = np.clip(np.floor(Y.max(1)).astype(np.int64), -1, ny - 1)
    ni = np.maximum(i1 - i0 + 1, 0)
    nj = np.maximum(j1 - j0 + 1, 0)
    cnt = ni * nj
    keep = cnt > 0
    toggles = np.zeros((nx, ny, nz + 1), dtype=np.int32)
    tri = np.nonzero(keep)[0]
    # process in chunks to bound memory
    chunk = 200_000
    for s in range(0, len(tri), chunk):
        t = tri[s:s + chunk]
        c = cnt[t]
        rep = np.repeat(np.arange(len(t)), c)
        off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        tt = t[rep]
        ii = i0[tt] + off // nj[tt]
        jj = j0[tt] + off % nj[tt]
        x1, x2, x3 = X[tt, 0], X[tt, 1], X[tt, 2]
        y1, y2, y3 = Y[tt, 0], Y[tt, 1], Y[tt, 2]
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        ok = np.abs(det) > 1e-14
        det = np.where(ok, det, 1.0)
        l1 = ((y2 - y3) * (ii - x3) + (x3 - x2) * (jj - y3)) / det
        l2 = ((y3 - y1) * (ii - x3) + (x1 - x3) * (jj - y3)) / det
        l3 = 1.0 - l1 - l2
        hit = ok & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        zc = l1 * Z[tt, 0] + l2 * Z[tt, 1] + l3 * Z[tt, 2]
        # first node centre above the crossing
        kk = np.ceil((zc - o[2]) / dx).astype(np.int64)
        kk = np.clip(kk, 0, nz)
        np.add.at(toggles, (ii[hit], jj[hit], kk[hit]), 1)
    inside = (np.cumsum(toggles, axis=2)[:, :, :nz] % 2) == 1
    return inside


# ----------------------------------------------------------------------
# procedural rock (demo rockscape, and later the Phase-5 designer)
# ----------------------------------------------------------------------
def _value_noise(p: np.ndarray, rng_seed: int, octaves: int = 4, base_freq: float = 1.0) -> np.ndarray:
    """Smooth fractal noise in [-1, 1] approx. at points p (..., 3), via random sinusoids."""
    rng = np.random.default_rng(rng_seed)
    out = np.zeros(p.shape[:-1])
    amp, freq, norm = 1.0, base_freq, 0.0
    for _ in range(octaves):
        for _ in range(6):
            k = rng.standard_normal(3)
            k = k / np.linalg.norm(k) * freq * 2 * np.pi
            out += amp * np.sin(p @ k + rng.uniform(0, 2 * np.pi)) / np.sqrt(6)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return out / norm


def rock_sdf(points: np.ndarray, blobs: list[dict], seed: int, roughness: float = 0.25,
             feature: float = 0.08, blend: float = 0.03) -> np.ndarray:
    """Signed distance-like field (< 0 inside) of noise-deformed ellipsoid blobs.

    blobs: [{'center': (x,y,z), 'radii': (a,b,c)}] in metres; smooth union with
    blend radius ``blend``; surface displaced by fractal noise of amplitude
    ``roughness`` x (smallest radius) and feature size ``feature`` (m)."""
    d = None
    for b in blobs:
        c = np.asarray(b["center"], float)
        r = np.asarray(b["radii"], float)
        q = (points - c) / r
        e = (np.linalg.norm(q, axis=-1) - 1.0) * r.min()
        d = e if d is None else _smin(d, e, blend)
    rmin = min(min(b["radii"]) for b in blobs)
    return d + roughness * rmin * _value_noise(points, seed, base_freq=1.0 / feature)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def sdf_to_mesh(sdf_fn, lo, hi, h: float):
    """Marching cubes (scikit-image) of sdf_fn over the box [lo, hi] at spacing h."""
    from skimage.measure import marching_cubes
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[a] + h * np.arange(n[a]) for a in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    F = sdf_fn(P)
    # close the surface at the box faces
    F[0, :, :] = F[-1, :, :] = F[:, 0, :] = F[:, -1, :] = F[:, :, 0] = F[:, :, -1] = 1.0
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(h, h, h))
    return verts + lo, faces.astype(np.int64)   # outward normals for an SDF that is negative inside (checked: positive signed volume)


# ----------------------------------------------------------------------
# oriented regions (pump outlets, housings)
# ----------------------------------------------------------------------
def frame(aim, up):
    """Orthonormal (aim, lateral, up) frame; lateral = up x aim."""
    a = np.asarray(aim, float); a /= np.linalg.norm(a)
    u = np.asarray(up, float); u = u - (u @ a) * a; u /= np.linalg.norm(u)
    l = np.cross(u, a)
    return a, l, u


def local_coords(points, origin, aim, up):
    """(s, t1, t2): distance along aim, along lateral, along up."""
    a, l, u = frame(aim, up)
    d = points - np.asarray(origin, float)
    return d @ a, d @ l, d @ u


def region_mask(points, origin, aim, up, shape: str, s0: float, s1: float,
                diameter=0.0, width=0.0, height=0.0):
    """Points with s0 <= s < s1 along aim and inside the circle/rect cross-section."""
    s, t1, t2 = local_coords(points, origin, aim, up)
    along = (s >= s0) & (s < s1)
    if shape == "circle":
        return along & (t1 ** 2 + t2 ** 2 <= (diameter / 2) ** 2)
    return along & (np.abs(t1) <= width / 2) & (np.abs(t2) <= height / 2)
