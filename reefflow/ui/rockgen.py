"""Rockscape designer geometry (server side; the browser only displays these meshes).

Elements
--------
Every element has a type, parameters (metres / degrees) and a placement:
``position`` (m, tank coordinates of the element's local origin), ``rotation``
[rx, ry, rz] (deg, applied as R = Rx Ry Rz, i.e. three.js Euler order 'XYZ') and
``scale`` [sx, sy, sz]. world = position + R (scale * local). The local origin is
at the element's base (z = 0 = the surface it stands on); rocks extend a little
below it (``bury``) so they sit *in* the sand like a real aquascape.

    rock     noise-deformed blob (reefflow.geometry.rock_sdf): size [x, y, z], seed, lobes,
             roughness, bury
    slab     rounded flat plate (shelf): size [x, y, thickness], seed, roughness
    pillar   tapered column: radius_bottom, radius_top, height, seed, roughness
    arch     tube along a half-ellipse with legs: span, height, thickness, seed, roughness
    bridge   two legs and a deck: span, height, deck_thickness, width, leg_radius, seed, roughness
    mesh     imported STL/OBJ/PLY: file (relative to the project), unit

Each element is a signed-distance-like field (< 0 inside). The rockscape is their
union (min) sampled on one grid of spacing ``resolution`` inside the tank, closed
at the glass and bottom, and meshed with marching cubes -> one watertight STL.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import geometry as G
from .. import units as U

TYPES = ("rock", "slab", "pillar", "arch", "bridge", "mesh")

DEFAULTS = {
    "rock": {"size": [0.20, 0.15, 0.14], "seed": 1, "lobes": 3, "roughness": 0.25, "bury": 0.15},
    "slab": {"size": [0.25, 0.15, 0.025], "seed": 1, "roughness": 0.15},
    "pillar": {"radius_bottom": 0.04, "radius_top": 0.03, "height": 0.20, "seed": 1, "roughness": 0.2},
    "arch": {"span": 0.25, "height": 0.18, "thickness": 0.05, "seed": 1, "roughness": 0.2},
    "bridge": {"span": 0.30, "height": 0.16, "deck_thickness": 0.035, "width": 0.10, "leg_radius": 0.035,
               "seed": 1, "roughness": 0.2},
    "mesh": {"file": None, "unit": "mm"},
}


def params_of(el: dict) -> dict:
    t = el["type"]
    if t not in TYPES:
        raise ValueError(f"unknown element type {t!r}")
    return {**DEFAULTS[t], **(el.get("params") or {})}


# ----------------------------------------------------------------------
# transforms
# ----------------------------------------------------------------------
def rotation_matrix(rot_deg) -> np.ndarray:
    a, b, c = (math.radians(float(v)) for v in rot_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def placement(el: dict):
    pos = np.asarray(el.get("position", (0, 0, 0)), float)
    R = rotation_matrix(el.get("rotation", (0, 0, 0)))
    s = np.asarray(el.get("scale", (1, 1, 1)), float)
    if np.any(s <= 0):
        raise ValueError(f"element {el.get('id')}: scale must be positive")
    return pos, R, s


def to_world(local: np.ndarray, el: dict) -> np.ndarray:
    pos, R, s = placement(el)
    return (local * s) @ R.T + pos


def to_local(world: np.ndarray, el: dict) -> np.ndarray:
    pos, R, s = placement(el)
    return ((world - pos) @ R) / s


# ----------------------------------------------------------------------
# primitive fields (local coordinates, metres)
# ----------------------------------------------------------------------
def _noise(P, seed, amp, feature):
    if amp <= 0:
        return 0.0
    return amp * G._value_noise(P, int(seed) + 7919, octaves=3, base_freq=1.0 / max(feature, 1e-3))


def _rock_blobs(p) -> list:
    sx, sy, sz = (float(v) for v in p["size"])
    bury = float(p.get("bury", 0.15))
    rng = np.random.default_rng(int(p["seed"]))
    r0 = np.array([sx, sy, sz]) / 2 * 0.82
    c0 = np.array([0.0, 0.0, sz / 2 - bury * sz])
    blobs = [{"center": tuple(c0), "radii": tuple(r0)}]
    for _ in range(max(int(p.get("lobes", 3)), 1) - 1):
        k = rng.uniform(0.45, 0.68)
        r = r0 * k * rng.uniform(0.85, 1.15, 3)
        off = np.array([rng.uniform(-0.5, 0.5) * (sx / 2 - r[0]), rng.uniform(-0.5, 0.5) * (sy / 2 - r[1]),
                        rng.uniform(0.0, 0.35) * sz])
        c = c0 + off
        c[2] = min(c[2], sz - bury * sz - r[2])            # keep inside the nominal height
        blobs.append({"center": tuple(c), "radii": tuple(r)})
    return blobs


def _sd_box(P, half, r=0.0):
    q = np.abs(P) - (np.asarray(half) - r)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0) - r


def _sd_polyline_tube(P, pts, radius):
    """Distance from P (..., 3) to a polyline (M, 3) minus radius."""
    flat = P.reshape(-1, 3)
    best = np.full(len(flat), np.inf)
    a, b = pts[:-1], pts[1:]
    for s in range(0, len(flat), 60000):
        X = flat[s:s + 60000, None, :]
        ab = (b - a)[None]
        t = np.clip(np.sum((X - a[None]) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0, 1)
        d = np.linalg.norm(X - (a[None] + t[..., None] * ab), axis=-1)
        best[s:s + 60000] = d.min(axis=1)
    return best.reshape(P.shape[:-1]) - radius


def _smin(a, b, k):
    return G._smin(a, b, k)


def local_field(el: dict, P: np.ndarray, base_dir: Path | None = None, h: float = 0.004):
    """Signed field (< 0 inside, ~metres in local units) of element ``el`` at local points P (..., 3)."""
    t = el["type"]
    p = params_of(el)
    if t == "rock":
        blobs = _rock_blobs(p)
        rmin = min(min(b["radii"]) for b in blobs)
        feature = float(np.clip(0.6 * rmin, 0.015, 0.08))
        d = G.rock_sdf(P, blobs, int(p["seed"]), roughness=float(p["roughness"]), feature=feature,
                       blend=0.3 * rmin)
        return d
    if t == "slab":
        sx, sy, th = (float(v) for v in p["size"])
        d = _sd_box(P - np.array([0, 0, th / 2]), (sx / 2, sy / 2, th / 2), r=min(0.3 * th, 0.4 * min(sx, sy) / 2))
        return d + _noise(P, p["seed"], float(p["roughness"]) * th * 0.6, max(0.6 * th, 0.02))
    if t == "pillar":
        r0, r1, H = float(p["radius_bottom"]), float(p["radius_top"]), float(p["height"])
        z = P[..., 2]
        r = r0 + (r1 - r0) * np.clip(z / max(H, 1e-6), 0, 1)
        d = np.maximum(np.linalg.norm(P[..., :2], axis=-1) - r, np.maximum(-z - 0.1 * r0, z - H))
        return d + _noise(P, p["seed"], float(p["roughness"]) * min(r0, r1) * 0.6, max(min(r0, r1), 0.015))
    if t == "arch":
        span, H, th = float(p["span"]), float(p["height"]), float(p["thickness"])
        r = th / 2
        a, b = span / 2, max(H - r, r)
        th_ = np.linspace(0, math.pi, 72)
        curve = np.stack([a * np.cos(th_), np.zeros_like(th_), b * np.sin(th_)], 1)
        legs_r = np.array([[a, 0, -0.5 * r], [a, 0, 0]])
        legs_l = np.array([[-a, 0, 0], [-a, 0, -0.5 * r]])
        poly = np.vstack([legs_r, curve[1:-1], legs_l])
        d = _sd_polyline_tube(P, poly, r)
        d = np.maximum(d, -P[..., 2] - 0.5 * r)
        return d + _noise(P, p["seed"], float(p["roughness"]) * r * 0.6, max(r, 0.015))
    if t == "bridge":
        span, H = float(p["span"]), float(p["height"])
        dt, w, lr = float(p["deck_thickness"]), float(p["width"]), float(p["leg_radius"])
        z = P[..., 2]
        legs = []
        for sx in (-span / 2, span / 2):
            q = P - np.array([sx, 0, 0])
            legs.append(np.maximum(np.linalg.norm(q[..., :2], axis=-1) - lr, np.maximum(-z - 0.1 * lr, z - (H - dt / 2))))
        deck = _sd_box(P - np.array([0, 0, H - dt / 2]), (span / 2 + lr, w / 2, dt / 2), r=0.3 * dt)
        k = 0.4 * min(lr, dt)
        d = _smin(_smin(legs[0], legs[1], k), deck, k)
        return d + _noise(P, p["seed"], float(p["roughness"]) * min(lr, dt) * 0.6, max(min(lr, dt), 0.015))
    if t == "mesh":
        raise RuntimeError("mesh elements are evaluated with mesh_field()")
    raise ValueError(f"unknown element type {t!r}")


def local_bounds(el: dict, base_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Conservative local bounding box (lo, hi) of the element's solid."""
    t = el["type"]
    p = params_of(el)
    if t == "rock":
        s = np.asarray(p["size"], float)
        m = 0.6 * float(p["roughness"]) * s.min() / 2 + 0.01
        bury = float(p.get("bury", 0.15)) * s[2]
        return np.array([-s[0] / 2 - m, -s[1] / 2 - m, -bury - m]), np.array([s[0] / 2 + m, s[1] / 2 + m, s[2] - bury + m])
    if t == "slab":
        s = np.asarray(p["size"], float)
        m = 0.6 * float(p["roughness"]) * s[2] + 0.005
        return np.array([-s[0] / 2 - m, -s[1] / 2 - m, -m]), np.array([s[0] / 2 + m, s[1] / 2 + m, s[2] + m])
    if t == "pillar":
        r = max(float(p["radius_bottom"]), float(p["radius_top"]))
        m = 0.6 * float(p["roughness"]) * r + 0.005
        return np.array([-r - m, -r - m, -0.1 * r - m]), np.array([r + m, r + m, float(p["height"]) + m])
    if t == "arch":
        r = float(p["thickness"]) / 2
        a = float(p["span"]) / 2
        m = 0.6 * float(p["roughness"]) * r + 0.005
        return np.array([-a - r - m, -r - m, -0.5 * r - m]), np.array([a + r + m, r + m, float(p["height"]) + m])
    if t == "bridge":
        lr, w = float(p["leg_radius"]), float(p["width"])
        a = float(p["span"]) / 2 + lr
        m = 0.6 * float(p["roughness"]) * lr + 0.005
        hw = max(w / 2, lr)
        return np.array([-a - m, -hw - m, -0.1 * lr - m]), np.array([a + m, hw + m, float(p["height"]) + m])
    if t == "mesh":
        v, _ = mesh_local(el, base_dir)
        return v.min(0), v.max(0)
    raise ValueError(t)


def resolve_file(el: dict, base_dir: Path | None) -> Path:
    f = params_of(el).get("file")
    if not f:
        raise ValueError(f"mesh element {el.get('id')}: no file")
    path = Path(f)
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError("mesh element needs the project directory")
        path = (Path(base_dir) / path).resolve()
        if Path(base_dir).resolve() not in path.parents:
            raise ValueError(f"mesh file {f!r} is outside the project directory")
    if not path.is_file():
        raise FileNotFoundError(f"mesh file {f!r} not found")
    return path


_MESH_CACHE: dict = {}


def mesh_local(el: dict, base_dir: Path | None):
    """(vertices in metres in the file's own frame, faces) of a mesh element (cached)."""
    path = resolve_file(el, base_dir)
    unit = params_of(el).get("unit", "mm")
    key = (str(path), path.stat().st_mtime, unit)
    if key not in _MESH_CACHE:
        v, f = G.load_mesh(path, unit=unit)
        if len(_MESH_CACHE) > 16:
            _MESH_CACHE.clear()
        _MESH_CACHE[key] = (v, f)
    return _MESH_CACHE[key]


# ----------------------------------------------------------------------
# single-element preview mesh (local coordinates)
# ----------------------------------------------------------------------
def element_mesh(el: dict, base_dir: Path | None = None, h: float | None = None):
    """(vertices (n,3) float32, faces (m,3) uint32) in local coordinates, for display."""
    if el["type"] == "mesh":
        v, f = mesh_local(el, base_dir)
        return v.astype(np.float32), f.astype(np.uint32)
    lo, hi = local_bounds(el)
    ext = hi - lo
    if h is None:
        h = float(np.clip(ext.min() / 20.0, 0.002, 0.006))
        n = np.prod(np.ceil(ext / h) + 3)
        if n > 2.5e6:
            h *= (n / 2.5e6) ** (1 / 3)
    v, f = G.sdf_to_mesh(lambda P: local_field(el, P), lo - 2 * h, hi + 2 * h, h)
    return v.astype(np.float32), f.astype(np.uint32)


def pack_mesh(v: np.ndarray, f: np.ndarray) -> bytes:
    """Binary mesh for the browser: uint32 nv, uint32 nf, float32 xyz * nv, uint32 abc * nf."""
    v = np.ascontiguousarray(v, dtype="<f4")
    f = np.ascontiguousarray(f, dtype="<u4")
    return np.array([len(v), len(f)], dtype="<u4").tobytes() + v.tobytes() + f.tobytes()


# ----------------------------------------------------------------------
# union -> watertight STL, plus geometry checks
# ----------------------------------------------------------------------
def _mesh_field(el, base_dir, Pw, h):
    """Signed distance (approx., metres) of a closed mesh element at world points Pw (a regular block)."""
    from scipy import ndimage
    v, f = mesh_local(el, base_dir)
    vw = to_world(v, el)
    shape = Pw.shape[:3]
    origin = Pw[0, 0, 0]
    inside = G.voxelize_mesh(vw, f, origin, h, shape)
    if not inside.any():
        return np.full(shape, np.inf)
    d_out = ndimage.distance_transform_edt(~inside) * h
    d_in = ndimage.distance_transform_edt(inside) * h
    return np.where(inside, -d_in + 0.5 * h, d_out - 0.5 * h)


def union(elements: list, tank, base_dir: Path | None, h: float = 0.004, progress=None) -> dict:
    """Union of all elements inside the tank -> {'vertices','faces' (world, m), 'field_info', 'issues', ...}.

    ``tank`` is a reefflow.project.Tank (internal dimensions)."""
    from skimage.measure import marching_cubes
    if not elements:
        raise ValueError("no rock elements to combine")
    h = float(h)
    if not 0.0015 <= h <= 0.02:
        raise ValueError("union resolution must be between 1.5 mm and 20 mm")
    # world bounding box of all elements, clipped to the tank interior
    lo_all, hi_all, boxes = np.full(3, np.inf), np.full(3, -np.inf), []
    for el in elements:
        lo, hi = local_bounds(el, base_dir)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = to_world(corners, el)
        boxes.append((w.min(0), w.max(0)))
        lo_all, hi_all = np.minimum(lo_all, w.min(0)), np.maximum(hi_all, w.max(0))
    tlo = np.array([0.0, 0.0, 0.0])
    thi = np.array([tank.length, tank.width, tank.height])
    lo_g = np.maximum(lo_all, tlo)
    hi_g = np.minimum(hi_all, thi)
    if np.any(hi_g <= lo_g):
        raise ValueError("all rock elements lie outside the tank")
    # grid: node centres from lo_g - h (one closed layer each side)
    n = np.ceil((hi_g - lo_g) / h).astype(int) + 3
    if np.prod(n) > 60e6:
        raise ValueError(f"union grid {tuple(n)} too large at {h * 1e3:.1f} mm; choose a coarser resolution")
    origin = lo_g - h
    axes = [origin[a] + h * np.arange(n[a]) for a in range(3)]
    F = np.full(tuple(n), np.inf, dtype=np.float32)
    owner = []
    for idx, (el, (blo, bhi)) in enumerate(zip(elements, boxes)):
        i0 = np.clip(np.floor((blo - origin) / h).astype(int) - 1, 0, n - 1)
        i1 = np.clip(np.ceil((bhi - origin) / h).astype(int) + 2, 0, n)
        if np.any(i1 <= i0):
            owner.append(None)
            continue
        sl = tuple(slice(i0[a], i1[a]) for a in range(3))
        P = np.stack(np.meshgrid(*[axes[a][sl[a]] for a in range(3)], indexing="ij"), axis=-1)
        if el["type"] == "mesh":
            d = _mesh_field(el, base_dir, P, h)
        else:
            _, _, s = placement(el)
            d = local_field(el, to_local(P, el), base_dir, h) * float(s.min())
        F[sl] = np.minimum(F[sl], d.astype(np.float32))
        owner.append((sl, d < 0))
        if progress:
            progress(f"element {idx + 1}/{len(elements)}")
    # clip to the tank interior (glass walls, bottom glass, rim)
    X, Y, Z = np.meshgrid(*axes, indexing="ij", sparse=True)
    outside = (X < 0) | (X > tank.length) | (Y < 0) | (Y > tank.width) | (Z < 0) | (Z > tank.height)
    F = np.where(outside, np.maximum(F, h), F)
    F[0, :, :] = F[-1, :, :] = F[:, 0, :] = F[:, -1, :] = F[:, :, 0] = F[:, :, -1] = h
    F = np.where(np.isfinite(F), F, 10 * h).astype(np.float32)
    occ = F < 0
    if not occ.any():
        raise ValueError("the elements contain no solid inside the tank")
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(h, h, h))
    verts = verts + origin
    faces = faces[:, ::-1].astype(np.int64)
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=True)
    m.remove_unreferenced_vertices()
    if m.is_watertight and m.volume < 0:
        m.invert()
    issues = check(occ, origin, h, tank, elements, owner)
    return {"vertices": np.asarray(m.vertices), "faces": np.asarray(m.faces), "watertight": bool(m.is_watertight),
            "volume_m3": float(abs(m.volume)) if m.is_watertight else float(occ.sum() * h ** 3),
            "volume_voxel_m3": float(occ.sum() * h ** 3), "triangles": int(len(m.faces)),
            "bounds_m": m.bounds.tolist(), "grid": list(map(int, n)), "resolution_m": h, "issues": issues}


def check(occ, origin, h, tank, elements, owner) -> list:
    """Informational geometry flags: rock touching / close to the glass, floating rock,
    centre of mass outside the sand footprint (unsupported overhang), rock above the water."""
    from scipy import ndimage
    issues = []
    n = occ.shape
    axes = [origin[a] + h * np.arange(n[a]) for a in range(3)]
    ids = [str(e.get("id")) for e in elements]

    def elems_in(mask):
        out = []
        for eid, ow in zip(ids, owner):
            if ow is None:
                continue
            sl, inside = ow
            if np.any(mask[sl] & inside):
                out.append(eid)
        return out

    # ---- glass proximity ---------------------------------------------------------
    walls = {"left (x = 0)": (0, 0.0, -1), "right (x = L)": (0, tank.length, 1),
             "front (y = 0)": (1, 0.0, -1), "back (y = W)": (1, tank.width, 1)}
    for wname, (a, pos, sgn) in walls.items():
        coords = axes[a]
        proj = occ.any(axis=tuple(i for i in range(3) if i != a))
        if not proj.any():
            continue
        c = coords[proj]
        gap = (pos - c.max()) if sgn > 0 else (c.min() - pos)
        gap = max(gap - 0.5 * h, 0.0)
        if gap <= h:
            sel = np.zeros_like(occ)
            k = np.nonzero(proj)[0][-1] if sgn > 0 else np.nonzero(proj)[0][0]
            idx = [slice(None)] * 3
            idx[a] = k
            sel[tuple(idx)] = occ[tuple(idx)]
            issues.append({"kind": "glass_contact", "severity": "warning", "wall": wname, "gap_m": gap,
                           "elements": elems_in(sel),
                           "message": f"rock touches the {wname} glass: no flow path behind it, and no room "
                                      "for cleaning"})
        elif gap < 0.025:
            issues.append({"kind": "glass_close", "severity": "info", "wall": wname, "gap_m": gap, "elements": [],
                           "message": f"rock is {gap * 1e3:.0f} mm from the {wname} glass (tight for a cleaning "
                                      "magnet; little flow behind it)"})
    # ---- above the water --------------------------------------------------------
    zs = axes[2][occ.any(axis=(0, 1))]
    if len(zs) and zs.max() > tank.water_surface:
        issues.append({"kind": "above_water", "severity": "info", "elements": [],
                       "height_m": float(zs.max() - tank.water_surface),
                       "message": f"rock rises {1e3 * (zs.max() - tank.water_surface):.0f} mm above the water "
                                  "surface (the flow model ignores the part above the surface)"})
    # ---- support: connected rock groups --------------------------------------------
    lab, nlab = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    sand_top = tank.sand_depth
    kz = np.nonzero(axes[2] <= sand_top + 1.01 * h)[0]
    for comp in range(1, nlab + 1):
        cm = lab == comp
        cnt = int(cm.sum())
        if cnt * h ** 3 < 1e-6:                           # < 1 cm^3 debris
            continue
        I, J, K = np.nonzero(cm)
        com = np.array([axes[0][I].mean(), axes[1][J].mean(), axes[2][K].mean()])
        els = elems_in(cm)
        base = cm[:, :, kz].any(axis=2) if len(kz) else np.zeros(cm.shape[:2], bool)
        if not base.any():
            issues.append({"kind": "floating", "severity": "warning", "elements": els,
                           "centre_of_mass_m": com.tolist(), "volume_m3": cnt * h ** 3,
                           "message": f"a rock group of {cnt * h ** 3 * 1e3:.2f} L does not reach the sand "
                                      "(floating; needs support or glue)"})
            continue
        bi, bj = np.nonzero(base)
        pts = np.stack([axes[0][bi], axes[1][bj]], 1)
        dist = _outside_hull_distance(pts, com[:2], h)
        if dist > 0.0:
            issues.append({"kind": "overhang", "severity": "info", "elements": els,
                           "centre_of_mass_m": com.tolist(), "outside_m": dist, "volume_m3": cnt * h ** 3,
                           "message": f"centre of mass of a {cnt * h ** 3 * 1e3:.2f} L rock group lies "
                                      f"{dist * 1e3:.0f} mm outside its footprint on the sand: unsupported "
                                      "overhang (tipping risk unless pinned/glued; informational)"})
    return issues


def _outside_hull_distance(pts: np.ndarray, p: np.ndarray, h: float) -> float:
    """Distance (m) of point p outside the convex hull of pts (footprint cells padded by h/2); 0 inside."""
    from scipy.spatial import ConvexHull
    off = 0.5 * h * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    P = (pts[:, None, :] + off[None]).reshape(-1, 2)
    try:
        hull = ConvexHull(P)
    except Exception:
        c = P.mean(0)
        return max(float(np.linalg.norm(p - c)) - h, 0.0)
    # hull.equations: n . x + d <= 0 inside
    s = hull.equations[:, :2] @ p + hull.equations[:, 2]
    if np.all(s <= 1e-12):
        return 0.0
    V = P[hull.vertices]
    best = np.inf
    for a, b in zip(V, np.roll(V, -1, axis=0)):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def save_union_stl(path, result: dict):
    G.save_stl(path, result["vertices"], result["faces"], unit="mm")


def quick_flags(elements: list, tank, base_dir=None) -> list:
    """Cheap per-element flags from world bounding boxes (live feedback while editing)."""
    out = []
    for el in elements:
        try:
            lo, hi = local_bounds(el, base_dir)
        except Exception as e:                              # noqa: BLE001
            out.append({"element": el.get("id"), "message": str(e)})
            continue
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = to_world(corners, el)
        if w[:, 0].min() < 0 or w[:, 0].max() > tank.length or w[:, 1].min() < 0 or w[:, 1].max() > tank.width:
            out.append({"element": el.get("id"), "message": "extends to/through the glass"})
    return out


def length_si(v) -> float:
    return U.parse(v, "length")
