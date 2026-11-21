"""Visual hull (space carving) on a voxel grid, mesh export, volume and weight.

Grid convention matches :func:`reefflow.geometry.voxelize_mesh`: node (i, j, k)
has centre ``origin + (i, j, k) * dx`` and is occupied when its centre is inside
the rock.

Carving rule (DECISIONS D-605)
------------------------------
A voxel is removed if *any* view that sees it projects its centre (through the
refractive interface) onto a non-rock pixel. "Sees it" means: valid light path
(camera.View.project), inside the image, and not on an ``ignore`` pixel. Masks
are first dilated by ``tolerance_px`` to absorb calibration error (a dilation
can only add volume; it never removes true rock). Voxels seen by fewer than
``min_views`` views are left unoccupied and counted as "unconstrained".

Overfill (spec limitation): the visual hull is the largest shape consistent
with the silhouettes; caves, arches, overhangs and gaps between rocks hidden
from every view are filled. See ``edit.py`` for the correction tools.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import View
from .tankref import TankGeometry

US_GAL = 3.785411784e-3
LB = 0.45359237

# Dry porous aragonite reef rock: mineral density 2.93 kg/L, porosity ~40-60 %
# -> bulk 1.2-1.8 kg/L. Default 1.4 kg/L. ASSUMPTION A-606, user-editable.
DEFAULT_BULK_DENSITY = 1400.0          # kg/m3
DEFAULT_DENSITY_RANGE = (1100.0, 1800.0)


@dataclass
class VoxelGrid:
    origin: np.ndarray                 # centre of node (0, 0, 0), m
    dx: float
    shape: tuple

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float)
        self.shape = tuple(int(s) for s in self.shape)

    @classmethod
    def from_bounds(cls, lo, hi, dx: float) -> "VoxelGrid":
        """Cells of size dx tiling [lo, hi] (node centres at lo + dx/2 + i dx)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        n = np.maximum(np.floor((hi - lo) / dx + 1e-9).astype(int), 1)
        return cls(lo + dx / 2, dx, tuple(n))

    @classmethod
    def for_tank(cls, tank: TankGeometry, dx: float, z_top: float | None = None) -> "VoxelGrid":
        """Water column above the sand: x in [0, L], y in [0, W], z in [sand, water surface]."""
        return cls.from_bounds((0, 0, tank.sand_depth),
                               (tank.length, tank.width, tank.water_surface if z_top is None else z_top), dx)

    def axes(self):
        return [self.origin[a] + self.dx * np.arange(self.shape[a]) for a in range(3)]

    def centres(self, k0: int = 0, k1: int | None = None) -> np.ndarray:
        """Node centres of z-layers k0..k1-1 as (n, 3), C order (i, j, k)."""
        ax = self.axes()
        k1 = self.shape[2] if k1 is None else k1
        X, Y, Z = np.meshgrid(ax[0], ax[1], ax[2][k0:k1], indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    @property
    def cell_volume(self) -> float:
        return self.dx ** 3

    def to_dict(self) -> dict:
        return {"origin_m": self.origin.tolist(), "dx_m": self.dx, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, d) -> "VoxelGrid":
        return cls(d["origin_m"], d["dx_m"], d["shape"])


@dataclass
class CarveResult:
    occupancy: np.ndarray              # bool (nx, ny, nz)
    seen_count: np.ndarray             # uint8 number of views that constrain each voxel
    grid: VoxelGrid
    stats: dict


def _dilate(mask, r):
    import cv2
    m = np.asarray(mask, bool).astype(np.uint8)
    if r <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m, k).astype(bool)


def carve_visual_hull(views, masks, grid: VoxelGrid, ignore=None, tolerance_px: int = 1,
                      min_views: int = 1, max_points: int = 1_500_000) -> CarveResult:
    """Space carving with calibrated silhouettes.

    views: list of camera.View; masks: list of bool HxW (True = rock);
    ignore: optional list (same length) of bool HxW or None."""
    if len(views) != len(masks):
        raise ValueError("one mask per view")
    ignore = list(ignore) if ignore is not None else [None] * len(views)
    nx, ny, nz = grid.shape
    removed = np.zeros(grid.shape, bool)
    seen = np.zeros(grid.shape, np.uint8)
    dm = [_dilate(m, tolerance_px) for m in masks]
    for v, m in zip(views, masks):
        if m.shape != (v.camera.height, v.camera.width):
            raise ValueError(f"mask shape {m.shape} != image size of the {v.kind} view")
    per_view = [{"kind": v.kind, "seen": 0, "carved": 0} for v in views]
    layers = max(1, max_points // (nx * ny))
    for k0 in range(0, nz, layers):
        k1 = min(nz, k0 + layers)
        P = grid.centres(k0, k1)
        rem = np.zeros(len(P), bool)
        cnt = np.zeros(len(P), np.uint8)
        for iv, (v, m, ig) in enumerate(zip(views, dm, ignore)):
            uv, ok = v.project(P)
            ok &= v.camera.in_image(uv)
            ui = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, v.camera.width - 1)
            vi = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, v.camera.height - 1)
            if ig is not None:
                ok &= ~np.asarray(ig, bool)[vi, ui]
            out = ok & ~m[vi, ui]
            per_view[iv]["seen"] += int(ok.sum())
            per_view[iv]["carved"] += int(out.sum())
            rem |= out
            cnt += ok.astype(np.uint8)
        removed[:, :, k0:k1] = rem.reshape(nx, ny, k1 - k0)
        seen[:, :, k0:k1] = cnt.reshape(nx, ny, k1 - k0)
    occ = ~removed & (seen >= min_views)
    stats = {"voxels": int(np.prod(grid.shape)), "occupied": int(occ.sum()),
             "unconstrained": int((seen < min_views).sum()),
             "seen_by_n_views": {int(n): int((seen == n).sum()) for n in range(len(views) + 1)},
             "per_view": per_view, "tolerance_px": tolerance_px, "min_views": min_views}
    return CarveResult(occ, seen, grid, stats)


# ----------------------------------------------------------------------
# cleanup, mesh, export
# ----------------------------------------------------------------------
def remove_small_components(occ, min_voxels: int = 8) -> np.ndarray:
    """Drop floating occupied islands smaller than min_voxels (26-connectivity)."""
    from scipy import ndimage
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.asarray(occ, bool).copy()
    size = np.bincount(lab.ravel())
    keep = size >= min_voxels
    keep[0] = False
    return keep[lab]


def occupancy_to_mesh(occ, grid: VoxelGrid, extend_below: int = 1, smooth_sigma: float = 0.0):
    """Watertight triangle mesh (vertices [m], faces) of the occupied voxels.

    Marching cubes at level 0.5 on the zero-padded occupancy: without smoothing,
    re-voxelizing the mesh on the same grid (centre-inside rule) reproduces the
    occupancy exactly. ``extend_below`` copies the bottom layer that many cells
    down so the rock base is buried in the sand (like the demo rock), which makes
    the tank model's sand-surface layer solid under the rock as well."""
    from skimage.measure import marching_cubes
    a = np.asarray(occ, bool)
    if not a.any():
        raise ValueError("occupancy is empty")
    if extend_below > 0:
        a = np.concatenate([np.repeat(a[:, :, :1], extend_below, axis=2), a], axis=2)
    F = np.pad(a.astype(np.float32), 1)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        F = gaussian_filter(F, smooth_sigma)
        F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = 0, 0, 0, 0, 0, 0
    verts, faces, _, _ = marching_cubes(F, level=0.5, spacing=(grid.dx,) * 3)
    off = grid.origin - grid.dx * np.array([1.0, 1.0, 1.0 + extend_below])
    from .synthetic import orient_outward
    return orient_outward(verts + off, faces.astype(np.int64))  # outward normals (positive volume)


def export_stl(occ, grid: VoxelGrid, path, unit: str = "mm", **mesh_kw) -> dict:
    """Write the reconstruction as STL; returns the project-file rockscape entry
    plus a mesh report. The STL is in tank coordinates (no translation needed)."""
    from pathlib import Path

    from .. import geometry as G
    v, f = occupancy_to_mesh(occ, grid, **mesh_kw)
    G.save_stl(path, v, f, unit=unit)
    rep = G.mesh_report(v, f)
    return {"rockscape_entry": {"file": Path(path).name, "unit": unit, "translate": ["0 mm", "0 mm", "0 mm"]},
            "mesh": rep}


# ----------------------------------------------------------------------
# volume / weight
# ----------------------------------------------------------------------
def rock_weight(volume_m3: float, bulk_density: float = DEFAULT_BULK_DENSITY,
                density_range=DEFAULT_DENSITY_RANGE, solid_fraction: float = 1.0) -> dict:
    """Dry-rock weight estimate = volume x solid_fraction x bulk density.

    bulk_density [kg/m3]: dry porous rock including its internal pores
    (default 1400, A-606). solid_fraction: share of the reconstructed envelope
    that is rock (1.0 = treat the cleaned reconstruction as solid; lower it for
    stacked rubble with gaps the silhouettes could not see)."""
    if bulk_density <= 0 or not (0 < solid_fraction <= 1):
        raise ValueError("bulk_density > 0 and 0 < solid_fraction <= 1 required")
    m = volume_m3 * solid_fraction * bulk_density
    lo, hi = (volume_m3 * solid_fraction * d for d in density_range)
    return {"volume_m3": volume_m3, "volume_L": volume_m3 * 1e3, "volume_gal": volume_m3 / US_GAL,
            "bulk_density_kg_m3": bulk_density, "solid_fraction": solid_fraction,
            "weight_kg": m, "weight_lb": m / LB,
            "weight_range_kg": [lo, hi], "weight_range_lb": [lo / LB, hi / LB],
            "density_range_kg_m3": list(density_range),
            "note": "dry-rock weight estimate; bulk density and solid fraction are assumptions (A-606). "
                    "Silhouette reconstruction overfills hidden cavities, so this tends to be an upper bound."}


def volume_report(occ, grid: VoxelGrid, **kw) -> dict:
    """Occupied volume (voxel count x dx^3) and weight estimate."""
    V = float(np.count_nonzero(occ)) * grid.cell_volume
    return rock_weight(V, **kw) | {"voxels": int(np.count_nonzero(occ)), "dx_m": grid.dx}


def compare_occupancy(rec, truth) -> dict:
    """IoU, precision, recall and volume error of a reconstruction vs a reference."""
    rec, truth = np.asarray(rec, bool), np.asarray(truth, bool)
    tp = np.count_nonzero(rec & truth)
    fp = np.count_nonzero(rec & ~truth)
    fn = np.count_nonzero(~rec & truth)
    nt = np.count_nonzero(truth)
    return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(nt, 1),
            "volume_error": (np.count_nonzero(rec) - nt) / max(nt, 1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}
