"""End-to-end photo reconstruction: one call a backend (CLI / FastAPI) can make.

Input (JSON-able except images/masks, which may be file paths or arrays)::

    tank = TankGeometry(...)            # or TankGeometry.from_project("tank.yaml")
    photos = [
      {"name": "top",   "kind": "top",   "image": "top.jpg",
       "clicks": {"rim_FL": [u, v], ...}, "custom": [], "edges": [],
       "f_px": None,                     # optional known focal length (px)
       "mask": None,                     # optional user-edited mask (PNG path / bool array)
       "ignore": None},                  # optional 'no information' mask
      {"name": "front", "kind": "front", ...},
    ]
    report = reconstruct(tank, photos, dx=0.005, out_dir="recon/")

Output: dict with a JSON-able ``report`` (calibration residuals, segmentation
method per view, carving statistics, volume and weight, files written) plus the
arrays ``occupancy``, ``grid``, ``views``, ``masks`` for further editing.
Files in ``out_dir``: rock_photo.stl (tank coordinates, mm), mask_<name>.png,
compare_<name>.png, reconstruction.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import segment as SG
from .calibration import calibrate_view
from .carve import (VoxelGrid, carve_visual_hull, export_stl, remove_small_components, volume_report)
from .compare import side_by_side
from .edit import apply_edits
from .tankref import TankGeometry


def _img(x):
    return SG.load_image(x) if isinstance(x, (str, Path)) else np.asarray(x)


def _mask(x, shape):
    if x is None:
        return None
    if isinstance(x, (str, Path)):
        m = SG.mask_from_png(Path(x).read_bytes())
    elif isinstance(x, (bytes, bytearray)):
        m = SG.mask_from_png(bytes(x))
    else:
        m = np.asarray(x, bool)
    if m.shape != tuple(shape[:2]):
        raise ValueError(f"mask shape {m.shape} does not match image {shape[:2]}")
    return m


def reconstruct(tank: TankGeometry, photos: list, dx: float = 0.005, out_dir=None, tolerance_px: int = 1,
                min_component_voxels: int = 8, edits: list | None = None, segmentation: str = "auto",
                bulk_density: float | None = None, solid_fraction: float = 1.0,
                refraction: bool = True) -> dict:
    names = [p.get("name", p["kind"]) for p in photos]
    if len(set(names)) != len(names):
        raise ValueError("photo names must be unique")
    images = {n: _img(p["image"]) for n, p in zip(names, photos)}
    report = {"tank": tank.to_dict(), "dx_m": dx, "views": {}}
    views = {}
    for n, p in zip(names, photos):
        H, W = images[n].shape[:2]
        cal = calibrate_view(p["kind"], (W, H), tank, p.get("clicks"), p.get("custom"), p.get("edges"),
                             f_px=p.get("f_px"), refraction=refraction)
        views[n] = cal.view
        report["views"][n] = {"calibration": cal.to_dict()}
    masks = {n: _mask(p.get("mask"), images[n].shape) for n, p in zip(names, photos)}
    ignore = {n: _mask(p.get("ignore"), images[n].shape) for n, p in zip(names, photos)}
    todo = [n for n in names if masks[n] is None]
    if todo:
        auto = SG.auto_segment_views({n: images[n] for n in names}, views, method=segmentation)
        for n in todo:
            masks[n] = auto[n]["mask"]
            report["views"][n]["segmentation"] = {"method": auto[n]["method"], "notes": auto[n]["notes"]}
    for n in names:
        report["views"][n].setdefault("segmentation", {"method": "user mask", "notes": []})
    grid = VoxelGrid.for_tank(tank, dx)
    res = carve_visual_hull([views[n] for n in names], [masks[n] for n in names], grid,
                            ignore=[ignore[n] for n in names], tolerance_px=tolerance_px)
    occ = res.occupancy
    if min_component_voxels:
        occ = remove_small_components(occ, min_component_voxels)
    if edits:
        occ = apply_edits(occ, grid, edits)
    report["carving"] = res.stats
    kw = {} if bulk_density is None else {"bulk_density": bulk_density}
    report["volume"] = volume_report(occ, grid, solid_fraction=solid_fraction, **kw)
    report["edits"] = edits or []
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        if occ.any():
            ex = export_stl(occ, grid, out / "rock_photo.stl")
            report["stl"] = ex
            files["stl"] = str(out / "rock_photo.stl")
        for n in names:
            (out / f"mask_{n}.png").write_bytes(SG.mask_to_png(masks[n]))
            sbs = side_by_side(images[n], views[n], occ, grid, masks[n], label=n)
            SG.save_image(out / f"compare_{n}.png", sbs["image"])
            report["views"][n]["silhouette_iou"] = sbs["silhouette_iou"]
            files[f"mask_{n}"] = str(out / f"mask_{n}.png")
            files[f"compare_{n}"] = str(out / f"compare_{n}.png")
        report["files"] = files
        (out / "reconstruction.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return {"report": report, "occupancy": occ, "grid": grid, "views": views, "masks": masks}
