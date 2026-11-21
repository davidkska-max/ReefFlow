"""Synthetic tank photos of a known rockscape, rendered through the refraction model.

Used for the Phase-6 validation and the tests. Not a photorealistic renderer:
it produces images with the colour structure that matters for segmentation
(textured brown/purple rock over beige sand and a dark-blue back wall, water
tint growing with path length, sensor noise and slight blur) and an exact
ground-truth rock mask.

Rendering
---------
* Background: every pixel is back-projected through the interface
  (camera.View.back_project) and intersected with the inner tank box; the hit
  surface (sand, back/side wall, underside of the water surface, sand face
  behind the front pane, frame, room) sets the colour.
* Rock: the mesh is clipped at the sand surface; vertices are forward-projected
  exactly (refraction included) and triangles are filled in image space with a
  painter's algorithm (far to near). Triangles are ~4 mm, so straight image
  edges between exactly projected vertices deviate from the true (slightly
  curved) refracted edges by far less than a pixel.
* Ground-truth mask: union of the projected rock triangles.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera, View
from .tankref import TankGeometry

SAND_RGB = np.array([205.0, 190.0, 158.0])
BACK_RGB = np.array([22.0, 42.0, 72.0])
SIDE_RGB = np.array([38.0, 66.0, 92.0])
SURF_RGB = np.array([150.0, 190.0, 210.0])
AIR_RGB = np.array([178.0, 182.0, 188.0])
ROOM_RGB = np.array([88.0, 88.0, 94.0])
FRAME_RGB = np.array([14.0, 14.0, 14.0])
WATER_RGB = np.array([30.0, 95.0, 140.0])
ROCK_RGB = (np.array([168.0, 134.0, 102.0]), np.array([150.0, 88.0, 138.0]), np.array([96.0, 74.0, 56.0]))
LIGHT = np.array([0.25, -0.35, 0.90]) / np.linalg.norm([0.25, -0.35, 0.90])
WATER_TINT_LENGTH = 1.5        # m: e-folding length of the blue tint (cosmetic)


def default_cameras(tank: TankGeometry, size=(1600, 1200), f_px: float = 1200.0) -> dict:
    """Realistic hand-held positions: top (above the open top, pumps off), front
    (crouching in front of the tank), left side. Slight offsets / tilt / roll."""
    W, H = size
    L, Wd = tank.length, tank.width
    zc = 0.5 * (tank.sand_depth + tank.water_surface)
    return {
        "top": Camera.look_at((L / 2 + 0.03, Wd / 2 - 0.06, tank.height + 1.05),
                              (L / 2 - 0.01, Wd / 2 + 0.02, tank.sand_depth), f_px, W, H, up=(0, 1, 0),
                              roll_deg=1.5),
        "front": Camera.look_at((L / 2 + 0.06, -1.45, zc + 0.03), (L / 2, Wd / 2, zc - 0.02), f_px, W, H,
                                roll_deg=-1.0),
        "left": Camera.look_at((-0.80, Wd / 2 + 0.03, zc + 0.04), (L / 2, Wd / 2, zc - 0.03), f_px, W, H,
                               roll_deg=0.8),
    }


def _noise2(p, seed, feature):
    from ..geometry import _value_noise
    return _value_noise(p, seed, octaves=3, base_freq=1.0 / feature)


def _background(view: View, rng, chunk=400_000):
    cam, T = view.camera, view.tank
    W, H = cam.width, cam.height
    vv, uu = np.mgrid[0:H, 0:W]
    uv = np.column_stack([uu.ravel(), vv.ravel()]).astype(float)
    out = np.zeros((len(uv), 3))
    for s in range(0, len(uv), chunk):
        sl = slice(s, s + chunk)
        r = view.back_project(uv[sl])
        E, d = r["entry"], r["dir"]
        col = np.tile(ROOM_RGB, (len(E), 1))
        ok_dir = (r["dir_air"] @ view.normal) > 0
        if view.kind == "top":
            inside = (E[:, 0] >= 0) & (E[:, 0] <= T.length) & (E[:, 1] >= 0) & (E[:, 1] <= T.width) & ok_dir
            rim = ((E[:, 0] >= -T.glass - 0.02) & (E[:, 0] <= T.length + T.glass + 0.02)
                   & (E[:, 1] >= -T.glass - 0.02) & (E[:, 1] <= T.width + T.glass + 0.02) & ~inside & ok_dir)
            col[rim] = FRAME_RGB
            water = inside
        else:
            lat = E[:, 0] if view.kind in ("front", "back") else E[:, 1]
            span = T.length if view.kind in ("front", "back") else T.width
            pane = (lat >= 0) & (lat <= span) & (E[:, 2] >= 0) & (E[:, 2] <= T.height) & ok_dir
            frame = ((lat >= -0.025) & (lat <= span + 0.025) & (E[:, 2] >= -0.03) & (E[:, 2] <= T.height + 0.02)
                     & ~pane & ok_dir)
            col[frame] = FRAME_RGB
            sandface = pane & (E[:, 2] < T.sand_depth)
            if sandface.any():
                n = _noise2(E[sandface], 5, 0.01)
                col[sandface] = SAND_RGB * (0.80 + 0.10 * n[:, None])
            air = pane & (E[:, 2] > T.water_surface)
            col[air] = AIR_RGB
            water = pane & ~sandface & ~air
        if water.any():
            o, dd = E[water], d[water]
            lo = np.array([0.0, 0.0, T.sand_depth]); hi = np.array([T.length, T.width, T.water_surface])
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / np.where(np.abs(dd) < 1e-12, 1e-12, dd)
                t1, t2 = (lo - o) * inv, (hi - o) * inv
            tmax = np.maximum(t1, t2)
            tmax = np.where(tmax > 1e-9, tmax, np.inf)
            ax = np.argmin(tmax, axis=1)
            t = tmax[np.arange(len(o)), ax]
            X = o + t[:, None] * dd
            pos = dd[np.arange(len(o)), ax] > 0
            c = np.zeros((len(o), 3))
            sand = (ax == 2) & ~pos
            surf = (ax == 2) & pos
            back = (ax == 1)
            side = (ax == 0)
            if sand.any():
                n1 = _noise2(X[sand], 7, 0.05)
                grain = rng.normal(0, 1, sand.sum())
                c[sand] = SAND_RGB * (0.88 + 0.08 * n1[:, None] + 0.05 * grain[:, None])
            c[surf] = SURF_RGB
            zf = ((X[:, 2] - T.sand_depth) / (T.water_surface - T.sand_depth))[:, None]
            c[back] = BACK_RGB * (0.8 + 0.4 * zf[back])
            c[side] = SIDE_RGB * (0.8 + 0.4 * zf[side])
            a = np.exp(-t / WATER_TINT_LENGTH)[:, None]
            col[water] = c * a + WATER_RGB * (1 - a)
        out[sl] = col
    return out.reshape(H, W, 3)


def signed_volume(vertices, faces) -> float:
    T3 = np.asarray(vertices, float)[np.asarray(faces)]
    return float(np.einsum("ij,ij->i", T3[:, 0], np.cross(T3[:, 1], T3[:, 2])).sum() / 6.0)


def orient_outward(vertices, faces):
    """Flip the winding of a closed mesh if its signed volume is negative."""
    f = np.asarray(faces)
    return np.asarray(vertices, float), (f[:, ::-1].copy() if signed_volume(vertices, f) < 0 else f)


def clip_mesh_z(vertices, faces, z0: float):
    """Keep the part of a triangle mesh with z >= z0 (open at the cut; orientation kept).
    Output vertices are per-face (unshared)."""
    V = np.asarray(vertices, float)[np.asarray(faces)]          # (M, 3, 3)
    above = V[:, :, 2] >= z0
    nab = above.sum(1)
    tris = [V[nab == 3]]

    def cut(a, b):                       # point on segment a-b at z = z0
        t = (z0 - a[:, 2]) / (b[:, 2] - a[:, 2])
        return a + t[:, None] * (b - a)

    for count in (1, 2):
        sel = nab == count
        if not sel.any():
            continue
        Vs, ab = V[sel], above[sel]
        lone = np.argmax(ab if count == 1 else ~ab, axis=1)     # the vertex alone on its side
        idx = (lone[:, None] + np.arange(3)) % 3                # roll, keeping cyclic order
        R = np.take_along_axis(Vs, idx[:, :, None], axis=1)
        A, B, C = R[:, 0], R[:, 1], R[:, 2]
        pab, pac = cut(A, B), cut(A, C)
        if count == 1:                    # A above
            tris.append(np.stack([A, pab, pac], 1))
        else:                             # A below; B, C above: quad pab, B, C, pac
            tris.append(np.stack([pab, B, C], 1))
            tris.append(np.stack([pab, C, pac], 1))
    T3 = np.concatenate(tris, axis=0)
    return T3.reshape(-1, 3), np.arange(len(T3) * 3).reshape(-1, 3)


def clip_mesh_at_sand(vertices, faces, tank: TankGeometry):
    return clip_mesh_z(vertices, faces, tank.sand_depth)


def render_view(view: View, vertices, faces, seed: int = 0, noise_sigma: float = 3.0,
                blur_sigma: float = 0.8, clip: bool = True):
    """Render one synthetic photo. Returns (RGB uint8 image, ground-truth bool mask)."""
    import cv2
    rng = np.random.default_rng(seed)
    img = _background(view, rng)
    cam = view.camera
    vertices, faces = orient_outward(vertices, faces)
    if clip:
        vertices, faces = clip_mesh_at_sand(vertices, faces, view.tank)
    uv, ok, info = view.project(vertices, return_info=True)
    T3 = vertices[faces]
    fok = ok[faces].all(1)
    cen = T3.mean(1)
    nrm = np.cross(T3[:, 1] - T3[:, 0], T3[:, 2] - T3[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-18)
    # back-face cull with the refracted viewing direction at the vertices
    vdir = info["dir_inner"][faces].mean(1)
    front = (nrm * vdir).sum(1) < 0
    sel = np.nonzero(fok & front)[0]
    dist = np.linalg.norm(cen[sel] - cam.centre, axis=1)
    sel = sel[np.argsort(-dist)]
    n1 = _noise2(cen[sel], seed + 101, 0.04)
    n2 = _noise2(cen[sel], seed + 202, 0.015)
    w_pur = np.clip(0.5 + 0.9 * n1, 0, 1)[:, None]
    w_dark = np.clip(0.4 - 1.2 * n2, 0, 1)[:, None]
    base = (ROCK_RGB[0] * (1 - w_pur) + ROCK_RGB[1] * w_pur) * (1 - 0.6 * w_dark) + ROCK_RGB[2] * 0.6 * w_dark
    lam = np.clip(nrm[sel] @ LIGHT, 0, 1)[:, None]
    col = base * (0.32 + 0.75 * lam)
    wp = info["water_path"][faces[sel]].mean(1)
    a = np.exp(-wp / WATER_TINT_LENGTH)[:, None]
    col = col * a + WATER_RGB * (1 - a)
    col = col + rng.normal(0, 6, col.shape)          # per-facet texture
    canvas = np.ascontiguousarray(img[..., ::-1].astype(np.uint8))   # BGR for cv2
    mask = np.zeros((cam.height, cam.width), np.uint8)
    pts = np.round(uv[faces[sel]] * 16).astype(np.int32)
    colb = np.clip(col[:, ::-1], 0, 255)
    for p, c in zip(pts, colb):
        cv2.fillConvexPoly(canvas, p, (float(c[0]), float(c[1]), float(c[2])), lineType=cv2.LINE_8, shift=4)
        cv2.fillConvexPoly(mask, p, 1, lineType=cv2.LINE_8, shift=4)
    out = canvas[..., ::-1].astype(np.float32)
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    out += rng.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), mask.astype(bool)


def true_clicks(view: View, names) -> dict:
    """Exact image positions of named reference points (no click noise). A seam
    (seam_<corner>) is clicked at 30 % of the water depth, seam_<corner>_2 at 75 %."""
    from .tankref import is_seam
    P = []
    for n in names:
        if is_seam(n):
            A, B = view.tank.seam(n)
            P.append([A[0], A[1], (0.75 if n.endswith("_2") else 0.3) * view.tank.water_surface])
        else:
            P.append(view.tank.point(n))
    P = np.array(P, float)
    uv, ok = view.project(P)
    inimg = view.camera.in_image(uv)
    return {n: u.tolist() for n, u, o, i in zip(names, uv, ok, inimg) if o and i}


def render_mask(view: View, vertices, faces, clip: bool = True) -> np.ndarray:
    """Ground-truth silhouette only (no shading/background); fast, for tests."""
    import cv2
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if clip:
        vertices, faces = clip_mesh_at_sand(vertices, faces, view.tank)
    uv, ok = view.project(vertices)
    sel = ok[faces].all(1)
    mask = np.zeros((view.camera.height, view.camera.width), np.uint8)
    for p in np.round(uv[faces[sel]] * 16).astype(np.int32):
        cv2.fillConvexPoly(mask, p, 1, lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)
