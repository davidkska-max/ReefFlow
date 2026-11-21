"""Reconstruction vs photo: reprojected silhouette, overlay and side-by-side image.

Colour code of the overlay (spec Feature 1 item 5):
    green  reconstruction and rock mask agree
    red    reconstruction only  (material the photo does not show: overfill, or
           a mask/calibration problem in this view)
    blue   rock mask only       (rock the reconstruction lacks: another view's
           mask or calibration removed it)
"""
from __future__ import annotations

import numpy as np

from .camera import View
from .carve import VoxelGrid


def surface_voxels(occ) -> np.ndarray:
    """Occupied voxels with at least one empty 6-neighbour (or on the grid border)."""
    a = np.pad(np.asarray(occ, bool), 1)
    inner = (a[2:, 1:-1, 1:-1] & a[:-2, 1:-1, 1:-1] & a[1:-1, 2:, 1:-1] & a[1:-1, :-2, 1:-1]
             & a[1:-1, 1:-1, 2:] & a[1:-1, 1:-1, :-2])
    return np.asarray(occ, bool) & ~inner


def reproject_silhouette(view: View, occ, grid: VoxelGrid, splat: float = 0.6) -> np.ndarray:
    """Bool image of the reconstruction as this camera sees it (through the interface).

    Surface voxel centres are projected and splatted as disks of radius
    ``splat x dx`` times the local image scale (px per m at that voxel)."""
    import cv2
    cam = view.camera
    idx = np.argwhere(surface_voxels(occ))
    out = np.zeros((cam.height, cam.width), np.uint8)
    if len(idx) == 0:
        return out.astype(bool)
    P = grid.origin + idx * grid.dx
    uv, ok = view.project(P)
    scale = np.zeros(len(P))
    for e in np.eye(3):
        uv2, _ = view.project(P + e * grid.dx)
        scale = np.maximum(scale, np.linalg.norm(uv2 - uv, axis=1))
    ok &= cam.in_image(uv, margin=-50)
    r = np.clip(np.rint(splat * scale[ok]).astype(int), 0, 60)
    ui = np.rint(uv[ok, 0]).astype(int)
    vi = np.rint(uv[ok, 1]).astype(int)
    inb = (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
    for rr in np.unique(r):
        sel = (r == rr) & inb
        if not sel.any():
            continue
        layer = np.zeros_like(out)
        layer[vi[sel], ui[sel]] = 1
        if rr > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
            layer = cv2.dilate(layer, k)
        out |= layer
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k).astype(bool)


def overlay(image, rec_sil, mask, alpha: float = 0.45) -> np.ndarray:
    img = np.asarray(image, np.uint8)[..., :3].astype(np.float32)
    col = np.zeros_like(img)
    both = rec_sil & mask
    col[both] = (40, 200, 60)
    col[rec_sil & ~mask] = (230, 40, 40)
    col[~rec_sil & mask] = (40, 90, 240)
    any_ = rec_sil | mask
    img[any_] = (1 - alpha) * img[any_] + alpha * col[any_]
    return np.clip(img, 0, 255).astype(np.uint8)


def _contour(img, mask, color, thick=2):
    import cv2
    cs, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, cs, -1, color, thick)
    return out


def side_by_side(image, view: View, occ, grid: VoxelGrid, mask, label: str | None = None) -> dict:
    """Left: photo with the rock-mask outline. Right: reconstruction overlay.

    Returns {'image': RGB uint8 (H x 2W x 3), 'silhouette_iou': float,
    'overfill_px': int, 'underfill_px': int}."""
    import cv2
    mask = np.asarray(mask, bool)
    sil = reproject_silhouette(view, occ, grid)
    left = _contour(np.asarray(image, np.uint8)[..., :3].copy(), mask, (255, 255, 0))
    right = overlay(image, sil, mask)
    out = np.hstack([left, right])
    fs = max(0.5, image.shape[1] / 1600)
    band = int(40 * fs)
    out[:band] = (out[:band] * 0.35).astype(np.uint8)
    for x0, txt in ((10, f"{label or view.kind}: photo + rock mask (yellow)"),
                    (image.shape[1] + 10, "reconstruction: green = agree, red = recon only, blue = mask only")):
        cv2.putText(out, txt, (x0, int(28 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    u = np.count_nonzero(sil | mask)
    return {"image": out, "silhouette_iou": float(np.count_nonzero(sil & mask) / u) if u else 1.0,
            "overfill_px": int(np.count_nonzero(sil & ~mask)), "underfill_px": int(np.count_nonzero(mask & ~sil))}
