"""Rock geometry for the light tracer: a signed-distance field (SDF) on a regular grid.

phi < 0 inside rock, > 0 in water, |phi| ~ distance to the rock surface (m).
The tracer interpolates phi trilinearly, finds ray-surface hits by sphere
tracing + bisection on the interpolated field, and takes the surface normal
from the SDF gradient (smooth normals; stair-stepped voxel faces would give
wrong cosine factors on sloped rock, PLAN section 3).

Builders
    from_mesh      STL/OBJ meshes (preferred): exact point-to-triangle distances in a
                   narrow band of +-2.5 grid cells around the surface, sign from
                   ray-parity voxelization, conservative (under-estimated) distances
                   outside the band.
    from_mask      the CFD voxel grid (TankModel.tags == rock): Euclidean distance
                   transform of the voxel centres, Gaussian-smoothed to round the
                   stair steps (DECISIONS D-405).
    from_function  analytic fields (verification).

Surface patches: the zero level set is triangulated (marching cubes), triangles
are subdivided, and their areas are binned into the cells of the tally grid
(nearest node), giving each rock patch its true area, mean normal and centroid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class SDF:
    origin: np.ndarray               # coordinates of grid node (0, 0, 0), m
    h: float                         # grid spacing, m
    phi: np.ndarray                  # (nx, ny, nz) float32
    source: str = ""

    # -- geometry ---------------------------------------------------------
    @property
    def shape(self):
        return self.phi.shape

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.origin, float)

    @property
    def hi(self) -> np.ndarray:
        return self.lo + self.h * (np.array(self.phi.shape) - 1)

    def sample(self, points) -> np.ndarray:
        """Trilinear phi at points (..., 3); outside the grid box: distance to the box + h
        (a positive lower bound, as the box encloses all rock)."""
        p = np.asarray(points, float)
        sh = p.shape[:-1]
        p = p.reshape(-1, 3)
        idx = ((p - self.lo) / self.h).T
        val = ndimage.map_coordinates(self.phi, idx, order=1, mode="nearest")
        d = np.linalg.norm(np.maximum(np.maximum(self.lo - p, p - self.hi), 0.0), axis=1)
        val = np.where(d > 0, d + self.h, val)
        return val.reshape(sh)

    def gradient(self, points, delta=None) -> np.ndarray:
        """Unit outward normals (central differences of the trilinear field)."""
        p = np.asarray(points, float)
        e = 0.5 * self.h if delta is None else delta
        g = np.stack([self.sample(p + e * np.eye(3)[a]) - self.sample(p - e * np.eye(3)[a]) for a in range(3)], -1)
        n = np.linalg.norm(g, axis=-1, keepdims=True)
        return g / np.where(n > 0, n, 1.0)

    # -- surface -----------------------------------------------------------
    def surface_mesh(self):
        """Marching-cubes triangulation of phi = 0: (verts (m), faces, outward)."""
        from skimage.measure import marching_cubes
        P = self.phi.astype(np.float32).copy()
        if P.min() >= 0 or P.max() <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
        for sl in (np.s_[0, :, :], np.s_[-1, :, :], np.s_[:, 0, :], np.s_[:, -1, :], np.s_[:, :, 0], np.s_[:, :, -1]):
            P[sl] = np.maximum(P[sl], self.h)          # close the surface at the grid boundary
        v, f, _, _ = marching_cubes(P, level=0.0, spacing=(self.h,) * 3)
        v = v + self.lo
        f = f.astype(np.int64)
        # orient outward (normal along +grad phi)
        if len(f):
            s = f[:: max(1, len(f) // 2000)]
            nrm = np.cross(v[s[:, 1]] - v[s[:, 0]], v[s[:, 2]] - v[s[:, 0]])
            if np.mean(np.sum(nrm * self.gradient(v[s].mean(1)), -1)) < 0:
                f = f[:, ::-1]
        return v, f

    def patches(self, origin, dx, shape, zmin=-np.inf, zmax=np.inf, subdiv: int = 2):
        """Bin the rock surface into the cells of a node grid (origin, dx, shape).

        Returns dict: area (nx,ny,nz), normal (nx,ny,nz,3) area-weighted unit mean normal,
        centroid (nx,ny,nz,3), mesh (verts, faces, face_cell (flat index or -1)), all
        restricted to the part of the surface with zmin < z < zmax (the water column)."""
        v, f = self.surface_mesh()
        shape = tuple(int(s) for s in shape)
        area = np.zeros(shape)
        nsum = np.zeros(shape + (3,))
        csum = np.zeros(shape + (3,))
        face_cell = np.full(len(f), -1, np.int64)
        if len(f) == 0:
            return {"area": area, "normal": nsum, "centroid": csum, "mesh": (v, f, face_cell)}
        A, B, C = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        # subdivide each triangle into n^2 sub-triangles (n = 2^subdiv) via barycentric grid
        n = 2 ** subdiv
        bary = []
        for i in range(n):
            for j in range(n - i):
                bary.append(((i + 1 / 3) / n, (j + 1 / 3) / n))          # upright
                if i + j < n - 1:
                    bary.append(((i + 2 / 3) / n, (j + 2 / 3) / n))      # inverted
        bary = np.array(bary)                                            # (n^2, 2)
        cross = np.cross(B - A, C - A)
        tri_area = 0.5 * np.linalg.norm(cross, axis=1)
        tri_n = cross / np.maximum(2 * tri_area[:, None], 1e-30)
        o = np.asarray(origin, float)
        flat_shape = np.prod(shape)
        for s0 in range(0, len(f), 200_000):
            sl = slice(s0, s0 + 200_000)
            Pc = (A[sl, None, :] + bary[None, :, 0:1] * (B - A)[sl, None, :]
                  + bary[None, :, 1:2] * (C - A)[sl, None, :])                  # (m, n^2, 3)
            sub_a = np.repeat(tri_area[sl, None] / len(bary), len(bary), 1)
            ijk = np.floor((Pc - o) / dx + 0.5).astype(np.int64)
            ok = np.all((ijk >= 0) & (ijk < np.array(shape)), -1) & (Pc[..., 2] > zmin) & (Pc[..., 2] < zmax)
            flat = np.ravel_multi_index(tuple(np.clip(ijk, 0, np.array(shape) - 1).transpose(2, 0, 1)), shape)
            w = np.where(ok, sub_a, 0.0)
            area += np.bincount(flat.ravel(), w.ravel(), flat_shape).reshape(shape)
            for a in range(3):
                nsum[..., a] += np.bincount(flat.ravel(), (w * tri_n[sl, None, a]).ravel(), flat_shape).reshape(shape)
                csum[..., a] += np.bincount(flat.ravel(), (w * Pc[..., a]).ravel(), flat_shape).reshape(shape)
            cen = (A[sl] + B[sl] + C[sl]) / 3
            cij = np.floor((cen - o) / dx + 0.5).astype(np.int64)
            cok = np.all((cij >= 0) & (cij < np.array(shape)), -1) & (cen[:, 2] > zmin) & (cen[:, 2] < zmax)
            fc = np.ravel_multi_index(tuple(np.clip(cij, 0, np.array(shape) - 1).T), shape)
            face_cell[sl] = np.where(cok, fc, -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            nn = np.linalg.norm(nsum, axis=-1, keepdims=True)
            normal = np.where(nn > 0, nsum / np.where(nn > 0, nn, 1), 0.0)
            centroid = np.where(area[..., None] > 0, csum / np.where(area[..., None] > 0, area[..., None], 1), 0.0)
        return {"area": area, "normal": normal, "centroid": centroid, "mesh": (v, f, face_cell)}

    # -- builders -----------------------------------------------------------
    @classmethod
    def from_function(cls, fn, lo, hi, h: float, source: str = "function") -> "SDF":
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        n = np.ceil((hi - lo) / h).astype(int) + 1
        axes = [lo[a] + h * np.arange(n[a]) for a in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
        return cls(lo, float(h), fn(P).astype(np.float32), source)

    @classmethod
    def from_mask(cls, mask, origin, dx: float, smooth: float = 1.0, source: str = "voxels") -> "SDF":
        """SDF of a voxel occupancy grid (node centres). The surface lies half-way between
        solid and fluid node centres; ``smooth`` (cells) is the Gaussian sigma that rounds
        the staircase."""
        m = np.asarray(mask, bool)
        if not m.any():
            return cls(np.asarray(origin, float), float(dx), np.full(m.shape, 10 * dx, np.float32), source)
        d_out = ndimage.distance_transform_edt(~m) * dx
        d_in = ndimage.distance_transform_edt(m) * dx
        phi = np.where(m, -(d_in - 0.5 * dx), d_out - 0.5 * dx)
        if smooth > 0:
            phi = ndimage.gaussian_filter(phi, smooth, mode="nearest")
        return cls(np.asarray(origin, float), float(dx), phi.astype(np.float32), source)

    @classmethod
    def from_mesh(cls, vertices, faces, h: float, lo=None, hi=None, band: float = 2.5,
                  source: str = "mesh") -> "SDF":
        """Exact narrow-band SDF of a closed triangle mesh on a grid of spacing h over [lo, hi]."""
        from scipy.spatial import cKDTree
        import trimesh
        from .. import geometry as G
        v = np.asarray(vertices, float); f = np.asarray(faces, np.int64)
        vlo, vhi = v.min(0), v.max(0)
        lo = vlo - 3 * h if lo is None else np.maximum(np.asarray(lo, float), vlo - 3 * h)
        hi = vhi + 3 * h if hi is None else np.minimum(np.asarray(hi, float), vhi + 3 * h)
        n = np.ceil((hi - lo) / h).astype(int) + 1
        inside = G.voxelize_mesh(v, f, lo, h, tuple(n))
        d_out = ndimage.distance_transform_edt(~inside) * h
        d_in = ndimage.distance_transform_edt(inside) * h
        phi0 = np.where(inside, -(d_in - 0.5 * h), d_out - 0.5 * h)
        sign = np.where(inside, -1.0, 1.0)
        # conservative far field (sphere-tracing step must not exceed the true distance)
        phi = sign * np.maximum(np.abs(phi0) - h, 1.5 * h)
        sel = np.nonzero(np.abs(phi0) < band * h)
        if len(sel[0]):
            vs, fs = trimesh.remesh.subdivide_to_size(v, f, max_edge=h)
            T = vs[fs]
            tree = cKDTree(T.mean(1))
            P = lo + h * np.stack(sel, -1)
            k = min(12, len(fs))
            best = np.full(len(P), np.inf)
            for s0 in range(0, len(P), 100_000):
                Ps = P[s0:s0 + 100_000]
                _, nb = tree.query(Ps, k=k)
                nb = nb.reshape(len(Ps), k)
                for j in range(k):
                    t = T[nb[:, j]]
                    cp = closest_point_on_triangle(Ps, t[:, 0], t[:, 1], t[:, 2])
                    best[s0:s0 + 100_000] = np.minimum(best[s0:s0 + 100_000], np.linalg.norm(Ps - cp, axis=1))
            phi[sel] = sign[sel] * best
        return cls(lo, float(h), phi.astype(np.float32), source)


def closest_point_on_triangle(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (N, 3). Ericson,
    Real-Time Collision Detection (2005), sec. 5.1.5, vectorised."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap); d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp); d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp); d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def sdiv(x, y):
        return x / np.where(np.abs(y) > 1e-300, y, 1e-300)

    den = va + vb + vc
    v_in = sdiv(vb, den); w_in = sdiv(vc, den)
    res = a + ab * v_in[:, None] + ac * w_in[:, None]                       # face interior
    w_bc = sdiv(d4 - d3, (d4 - d3) + (d5 - d6))
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    res = np.where(m[:, None], b + (c - b) * w_bc[:, None], res)             # edge BC
    w_ac = sdiv(d2, d2 - d6)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    res = np.where(m[:, None], a + ac * w_ac[:, None], res)                  # edge AC
    m = (d6 >= 0) & (d5 <= d6)
    res = np.where(m[:, None], c, res)                                       # vertex C
    v_ab = sdiv(d1, d1 - d3)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    res = np.where(m[:, None], a + ab * v_ab[:, None], res)                  # edge AB
    m = (d3 >= 0) & (d4 <= d3)
    res = np.where(m[:, None], b, res)                                       # vertex B
    m = (d1 <= 0) & (d2 <= 0)
    res = np.where(m[:, None], a, res)                                       # vertex A
    return res


def sphere_sdf(centre, radius):
    c = np.asarray(centre, float)
    return lambda P: np.linalg.norm(P - c, axis=-1) - radius
