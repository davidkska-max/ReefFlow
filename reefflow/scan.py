"""Phone / 3-D scan clean-up: turn a raw scan of dry rock into a watertight rock mesh.

Phone scanning apps (LiDAR or photogrammetry: Polycam, Scaniverse, 3d Scanner App,
KIRI, ...) export OBJ, GLB/glTF, PLY or STL. A raw scan usually

* is in metres with +Y up (the glTF/OBJ convention; ARKit/ARCore scans are
  gravity-aligned, so "up" really is up),
* includes the table or floor the rock sat on, and stray bits of the room,
* is open at the bottom (the table hid it) and has small holes where the camera
  could not see,
* can be the wrong size (photogrammetry without LiDAR has no absolute scale).

``clean()`` fixes all four, in this order (DECISIONS D-831 to D-834):

1. units x scale (unit guessed from the size when not given);
2. up axis: the axis whose lowest level holds a large flat, upward-facing area
   (the table), else the file format's convention;
3. ground cut: everything below the table top + a small clearance is removed;
4. solid model on a voxel grid. Inside/outside is decided by casting 26 rays per
   voxel and looking at which side of the first surface each ray hits (the scan's
   face normals point outward). That works with holes and an open bottom, where
   ray parity (used for closed meshes, geometry.voxelize_mesh) does not. Near the
   surface the signed distance is the point-to-plane distance to the scan, so the
   result is not stair-stepped;
5. small loose pieces are dropped, sealed internal voids are filled, the base is
   cut flat at the ground plane, and marching cubes gives a closed mesh.

The output sits with its base on z = 0, centred on x = y = 0, in metres.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import units as U

SCAN_EXT = (".stl", ".obj", ".ply", ".glb", ".gltf")
UNITS = ("m", "cm", "mm", "in")
# rotation that takes the named axis of the file to +z (x' = R @ x)
UP_ROT = {
    "+z": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "-z": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    "+y": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),      # glTF: +z (towards the viewer) -> -y (towards the front glass)
    "-y": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "+x": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "-x": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
}
CONVENTION_UP = {".glb": "+y", ".gltf": "+y", ".obj": "+y", ".stl": "+z", ".ply": "+z"}
MAX_VOXELS = 7_000_000
MIN_GROUND_FRACTION = 0.04          # the table must be at least 4 % of the scanned surface
DIRS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_raw(path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices in file units, faces) of any scan format trimesh reads (scenes are flattened)."""
    import trimesh
    path = Path(path)
    if path.suffix.lower() not in SCAN_EXT:
        raise ValueError(f"scan files must be one of {', '.join(SCAN_EXT)}")
    m = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh) or len(getattr(m, "faces", ())) == 0:
        raise ValueError("the file holds no triangle mesh (a point cloud cannot be used: export a mesh from the scan app)")
    m.merge_vertices()
    v = np.asarray(m.vertices, np.float64)
    f = np.asarray(m.faces, np.int64)
    if not np.isfinite(v).all():
        raise ValueError("the mesh has invalid (NaN or infinite) vertex coordinates")
    return v, f


def guess_unit(extent: float) -> str:
    """Unit of a rock scan from its largest dimension in file units. Phone apps write metres."""
    if extent <= 3.0:
        return "m"            # 3 m of rock is the most anyone scans; 3 mm is not a rock
    if extent <= 30.0:
        return "cm"
    return "mm"


def face_geometry(v, f):
    T = v[f]
    c = T.mean(axis=1)
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    a2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(a2, 1e-30)[:, None]
    return c, n, 0.5 * a2


# ----------------------------------------------------------------------
# orientation and ground
# ----------------------------------------------------------------------
def find_ground(c, n, area, extent: float) -> dict:
    """Largest flat, horizontal area at the bottom of the scan: the table or floor."""
    total = float(area.sum())
    tol = max(0.002, 0.004 * extent)
    up = np.abs(n[:, 2]) > 0.966          # horizontal within 15 degrees (either winding: normals may be inverted)
    out = {"found": False, "z": None, "sigma": 0.0, "area_fraction": 0.0, "below_fraction": 0.0, "tol": tol}
    if total <= 0 or area[up].sum() < MIN_GROUND_FRACTION * total:
        return out
    z, w = c[up, 2], area[up]
    lo = float(c[:, 2].min())
    edges = np.arange(lo - tol, lo + 0.5 * extent + 2 * tol, tol)
    if len(edges) < 2:
        return out
    hist, _ = np.histogram(z, bins=edges, weights=w)
    # a plane can straddle two bins: score adjacent pairs
    pair = hist[:-1] + hist[1:] if len(hist) > 1 else hist
    k = int(np.argmax(pair))
    z0 = float(edges[k + 1])
    for _ in range(3):                                            # refine on the inliers
        inl = np.abs(z - z0) < 1.5 * tol
        if not inl.any():
            break
        z0 = float(np.average(z[inl], weights=w[inl]))
    inl = np.abs(z - z0) < 1.5 * tol
    sigma = float(np.sqrt(np.average((z[inl] - z0) ** 2, weights=w[inl]))) if inl.any() else 0.0
    frac = float(w[inl].sum() / total)
    below = float(area[c[:, 2] < z0 - 3 * max(tol, sigma)].sum() / total)
    out.update(z=z0, sigma=sigma, area_fraction=frac, below_fraction=below,
               found=bool(frac >= MIN_GROUND_FRACTION and below <= 0.03))
    return out


def orient(v, up: str) -> np.ndarray:
    if up not in UP_ROT:
        raise ValueError(f"up axis must be one of {', '.join(UP_ROT)}")
    return v @ np.asarray(UP_ROT[up], np.float64).T


def analyze(v_file, f, ext: str = ".obj", unit: str = "auto", scale: float = 1.0, up: str = "auto") -> dict:
    """Units, up axis and ground of a raw scan (nothing is changed). Lengths in metres."""
    extent_file = float(np.ptp(v_file, axis=0).max())
    guessed = guess_unit(extent_file)
    unit_used = guessed if unit in (None, "", "auto") else unit
    if unit_used not in UNITS:
        raise ValueError(f"unit must be auto or one of {', '.join(UNITS)}")
    if not 0.01 <= float(scale) <= 100:
        raise ValueError("scale must be between 0.01 and 100")
    s = U.to_si(1.0, unit_used) * float(scale)
    v = v_file * s
    extent = extent_file * s
    conv = CONVENTION_UP.get(ext.lower(), "+z")
    cands = {}
    for a in UP_ROT:
        vo = orient(v, a)
        c, n, area = face_geometry(vo, f)
        cands[a] = find_ground(c, n, area, extent)
    if up in (None, "", "auto"):
        ok = [a for a in UP_ROT if cands[a]["found"]]
        if ok:
            best = max(ok, key=lambda a: (cands[a]["area_fraction"] + (0.02 if a == conv else 0.0)))
            reason = (f"found a flat surface under the scan (the table or floor, "
                      f"{cands[best]['area_fraction'] * 100:.0f} % of the scanned area)")
        else:
            best = conv
            reason = f"no table found under the scan; using the {ext.lstrip('.').upper()} convention ({conv} up)"
        up_used = best
    else:
        if up not in UP_ROT:
            raise ValueError(f"up axis must be auto or one of {', '.join(UP_ROT)}")
        up_used, reason = up, "set by you"
    g = cands[up_used]
    vo = orient(v, up_used)
    lo, hi = vo.min(axis=0), vo.max(axis=0)
    ground = None
    if g["found"]:
        clear = max(0.003, 3 * g["sigma"], g["tol"])
        ground = {"z": g["z"], "cut": g["z"] + clear, "height": g["z"] + clear - float(lo[2]),
                  "clearance": clear, "area_fraction": g["area_fraction"], "sigma": g["sigma"]}
    return {"unit": unit_used, "unit_guessed": guessed, "unit_was_guessed": unit in (None, "", "auto"),
            "scale": float(scale), "extent_file_units": extent_file, "triangles": int(len(f)),
            "up": up_used, "up_reason": reason, "up_convention": conv,
            "ground": ground, "bounds_m": [lo.tolist(), hi.tolist()],
            "candidates": {a: {"found": cands[a]["found"], "area_fraction": cands[a]["area_fraction"]} for a in UP_ROT}}


def cut_height(info: dict, ground) -> float | None:
    """Absolute z (oriented frame, m) of the ground cut. ground: 'auto' | 'none' | height above the lowest point (m)."""
    lo_z = info["bounds_m"][0][2]
    if ground in (None, "auto"):
        return info["ground"]["cut"] if info["ground"] else None
    if ground == "none":
        return None
    h = float(ground)
    if h < 0:
        raise ValueError("the ground cut height cannot be negative")
    return lo_z + h


# ----------------------------------------------------------------------
# surface samples on a voxel grid
# ----------------------------------------------------------------------
def _surface_voxels(v, f, n_face, origin, dx, shape, z_cut):
    """Per voxel crossed by the surface: mean surface point and summed outward normal.

    Each triangle is sampled on a barycentric lattice with spacing < dx / 2.5, so
    the marked voxels form a closed sheet (no diagonal gaps for the rays)."""
    nx, ny, nz = shape
    h = dx / 2.5
    T = v[f]
    L = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
    ns = np.maximum(np.ceil(L / h).astype(np.int64), 1)
    acc = []
    for nn in np.unique(ns):
        sel = np.nonzero(ns == nn)[0]
        a, b = np.meshgrid(np.arange(nn + 1), np.arange(nn + 1), indexing="ij")
        keep = (a + b) <= nn
        A = (a[keep] / nn)[None, :, None]
        B = (b[keep] / nn)[None, :, None]
        per = max(1, 3_000_000 // A.shape[1])
        for s in range(0, len(sel), per):
            t = sel[s:s + per]
            P0, P1, P2 = T[t, 0][:, None], T[t, 1][:, None], T[t, 2][:, None]
            P = P0 + A * (P1 - P0) + B * (P2 - P0)               # (m, k, 3)
            N = np.broadcast_to(n_face[t][:, None, :], P.shape)
            P = P.reshape(-1, 3)
            N = N.reshape(-1, 3)
            if z_cut is not None:
                ok = P[:, 2] >= z_cut
                P, N = P[ok], N[ok]
            idx = np.floor((P - (np.asarray(origin) - 0.5 * dx)) / dx).astype(np.int64)
            ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
            idx, P, N = idx[ok], P[ok], N[ok]
            lin = idx[:, 0] + nx * (idx[:, 1] + ny * idx[:, 2])
            acc.append(_reduce(lin, P, N))
    if not acc:
        return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3))
    lin = np.concatenate([a[0] for a in acc])
    cnt = np.concatenate([a[1] for a in acc])
    Ps = np.concatenate([a[2] for a in acc])
    Ns = np.concatenate([a[3] for a in acc])
    u, inv = np.unique(lin, return_inverse=True)
    c = np.bincount(inv, weights=cnt)
    P = np.stack([np.bincount(inv, weights=Ps[:, i], minlength=len(u)) for i in range(3)], 1) / c[:, None]
    N = np.stack([np.bincount(inv, weights=Ns[:, i], minlength=len(u)) for i in range(3)], 1)
    return u, P, N


def _reduce(lin, P, N):
    u, inv = np.unique(lin, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    Ps = np.stack([np.bincount(inv, weights=P[:, i], minlength=len(u)) for i in range(3)], 1)
    Ns = np.stack([np.bincount(inv, weights=N[:, i], minlength=len(u)) for i in range(3)], 1)
    return u, cnt, Ps, Ns


def _shift2(X, s1, s2, fill):
    """out[i, j] = X[i + s1, j + s2] (fill outside)."""
    out = np.full_like(X, fill)
    n1, n2 = X.shape
    a0, a1 = max(0, -s1), min(n1, n1 - s1)
    b0, b1 = max(0, -s2), min(n2, n2 - s2)
    out[a0:a1, b0:b1] = X[a0 + s1:a1 + s1, b0 + s2:b1 + s2]
    return out


def _sweep(val, d, bottom_open):
    """For every voxel, the code of the first surface voxel met along direction d (excluding itself).

    Codes: 1 = surface seen from behind (ray leaves the rock: inside vote), -1 = seen
    from the front (outside vote), 2 = grazing (no vote), 0 = the ray left the grid
    (outside vote), 3 = it left through the cut base (no vote)."""
    ax = 2 if d[2] else (1 if d[1] else 0)
    others = [a for a in range(3) if a != ax]
    s1, s2 = d[others[0]], d[others[1]]
    step = d[ax]
    V = np.moveaxis(val, ax, -1)
    F = np.empty_like(val)
    Ft = np.moveaxis(F, ax, -1)
    n = V.shape[-1]
    rng = range(n - 1, -1, -1) if step > 0 else range(n)
    for k in rng:
        kn = k + step
        if 0 <= kn < n:
            X = np.where(V[..., kn] != 0, V[..., kn], Ft[..., kn])
            Ft[..., k] = _shift2(X, s1, s2, 0)
        else:
            Ft[..., k] = 3 if (bottom_open and ax == 2 and step < 0) else 0
    return F


def _vote(shape, lin, N, bottom_open):
    """Inside (True) / outside of every voxel by 26-direction ray voting on oriented surface voxels."""
    vin = np.zeros(shape, np.int8)
    vout = np.zeros(shape, np.int8)
    Nn = N / np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
    flat_shape = int(np.prod(shape))
    for d in DIRS:
        dd = np.asarray(d, np.float64) / np.linalg.norm(d)
        dot = Nn @ dd
        code = np.where(dot > 0.15, 1, np.where(dot < -0.15, -1, 2)).astype(np.int8)
        val = np.zeros(flat_shape, np.int8)
        val[lin] = code
        val = val.reshape(shape, order="F")
        F = _sweep(val, d, bottom_open)
        vin += (F == 1)
        vout += (F == 0) | (F == -1)
    return vin > vout


# ----------------------------------------------------------------------
# clean-up
# ----------------------------------------------------------------------
def auto_dx(extent: float) -> float:
    return float(np.clip(extent / 150.0, 0.0015, 0.004))


def clean(v_file, f, ext: str = ".obj", unit: str = "auto", scale: float = 1.0, up: str = "auto",
          ground="auto", dx: float | None = None, keep: str = "pieces", progress=None) -> dict:
    """Raw scan -> watertight rock mesh. Returns {'vertices' (m), 'faces', 'report'}.

    ground: 'auto' (cut at the detected table), 'none', or the cut height above the
    scan's lowest point (m). keep: 'main' (largest piece only) or 'pieces' (every
    piece of at least 5 % of the largest one's volume)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    import trimesh

    def prog(x, msg):
        if progress:
            progress(x, msg)

    info = analyze(v_file, f, ext, unit, scale, up)
    s = U.to_si(1.0, info["unit"]) * float(scale)
    v = orient(v_file * s, info["up"])
    z_cut = cut_height(info, ground)
    c, n_face, area = face_geometry(v, f)
    total_area = float(area.sum())
    if z_cut is not None:
        keepf = (v[f][:, :, 2] > z_cut).any(axis=1)
        removed_area = float(area[~keepf].sum())
        f2, nf2, c2, a2 = f[keepf], n_face[keepf], c[keepf], area[keepf]
    else:
        removed_area = 0.0
        f2, nf2, c2, a2 = f, n_face, c, area
    if not len(f2):
        raise ValueError("nothing is left above the ground cut: lower the cut height")
    # global orientation: the scan's normals should point away from its middle
    ctr = np.average(c2, axis=0, weights=a2)
    flipped = bool(np.sum(a2 * np.einsum("ij,ij->i", c2 - ctr, nf2)) < 0)
    if flipped:
        nf2 = -nf2
    used = np.unique(f2)
    lo = v[used].min(axis=0)
    hi = v[used].max(axis=0)
    if z_cut is not None:
        lo[2] = z_cut
    extent = float((hi - lo).max())
    dx = float(dx) if dx else auto_dx(extent)
    if not 0.001 <= dx <= 0.01:
        raise ValueError("detail (voxel size) must be between 1 and 10 mm")
    pad = 3
    while True:
        shape = np.ceil((hi - lo) / dx).astype(int) + 2 * pad
        if z_cut is not None:
            shape[2] -= pad
        if np.prod(shape) <= MAX_VOXELS:
            break
        dx *= 1.15
    origin = lo - pad * dx + 0.5 * dx
    if z_cut is not None:
        origin[2] = z_cut + 0.5 * dx
    shape = tuple(int(x) for x in shape)
    prog(0.1, "sampling the scan surface")
    lin, Pm, Ns = _surface_voxels(v, f2, nf2, origin, dx, shape, z_cut)
    if len(lin) < 8:
        raise ValueError("the scan is too small for this detail setting")
    prog(0.35, "deciding inside / outside (26-ray voting)")
    inside = _vote(shape, lin, Ns, bottom_open=z_cut is not None)
    # signed distance near the surface: point-to-plane distance to the scan
    prog(0.6, "signed distance near the surface")
    shell = np.zeros(int(np.prod(shape)), bool)
    shell[lin] = True
    shell = shell.reshape(shape, order="F")
    band = ndimage.binary_dilation(shell, structure=np.ones((3, 3, 3), bool), iterations=2)
    sdf = np.where(inside, -3.0 * dx, 3.0 * dx).astype(np.float32)
    bi = np.argwhere(band)
    X = origin + bi * dx
    Nn = Ns / np.maximum(np.linalg.norm(Ns, axis=1), 1e-30)[:, None]
    tree = cKDTree(Pm)
    dist, q = tree.query(X, k=1, workers=-1)
    r = X - Pm[q]
    dp = np.einsum("ij,ij->i", r, Nn[q])
    sign_vote = np.where(inside[band], -1.0, 1.0)
    along = np.abs(dp) >= 0.5 * dist
    val = np.where(along, dp, sign_vote * dist)
    sdf[band] = np.clip(val, -3 * dx, 3 * dx)
    # pieces and voids
    prog(0.75, "removing loose pieces, filling sealed voids")
    solid = sdf < 0
    lab, nlab = ndimage.label(solid, structure=np.ones((3, 3, 3), bool))
    if nlab == 0:
        raise ValueError("no solid rock found in the scan (is it a single open sheet? try another up axis)")
    sizes = np.bincount(lab.ravel())[1:]
    big = int(np.argmax(sizes)) + 1
    min_keep = sizes.max() if keep == "main" else 0.05 * sizes.max()
    keep_ids = np.nonzero(sizes >= min_keep)[0] + 1
    drop = solid & ~np.isin(lab, keep_ids)
    dropped = np.setdiff1d(np.arange(1, nlab + 1), keep_ids)
    removed_pieces = int((sizes[dropped - 1] * dx ** 3 >= 1e-6).sum())     # pieces of 1 cm3 or more
    removed_piece_volume = float(drop.sum()) * dx ** 3
    sdf[drop] = np.abs(sdf[drop]) + 0.5 * dx
    solid = sdf < 0
    outside = ~solid
    olab, _ = ndimage.label(outside)
    border = np.zeros(shape, bool)
    border[0], border[-1], border[:, 0], border[:, -1], border[:, :, -1] = True, True, True, True, True
    if z_cut is None:
        border[:, :, 0] = True
    open_ids = np.unique(olab[border & outside])
    sealed = outside & ~np.isin(olab, open_ids)
    filled_volume = float(sealed.sum()) * dx ** 3
    sdf[sealed] = -np.abs(sdf[sealed]) - 0.5 * dx
    # flat base at the cut, closed box, marching cubes
    prog(0.85, "building the watertight mesh")
    F = np.pad(sdf, 1, constant_values=3 * dx)
    o = origin - dx
    if z_cut is not None:
        zc = o[2] + dx * np.arange(F.shape[2])
        F = np.maximum(F, (z_cut - zc)[None, None, :].astype(np.float32))
        F[:, :, 0] = 0.5 * dx                     # the layer under the base is the half-space distance: base exactly at the cut
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(dx, dx, dx))
    verts = verts + o
    m = trimesh.Trimesh(verts, faces.astype(np.int64), process=True)
    m.remove_unreferenced_vertices()
    if m.is_watertight and m.volume < 0:
        m.invert()
    vv = np.asarray(m.vertices).copy()
    base = z_cut if z_cut is not None else float(vv[:, 2].min())
    shift = np.array([-(vv[:, 0].min() + vv[:, 0].max()) / 2, -(vv[:, 1].min() + vv[:, 1].max()) / 2, -base])
    vv += shift
    m = trimesh.Trimesh(vv, np.asarray(m.faces), process=False)
    # open edges of the raw scan above the cut (holes that were closed)
    e = np.sort(np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]]), axis=1)
    eu, cnt = np.unique(e, axis=0, return_counts=True)
    be = eu[cnt == 1]
    open_len = float(np.linalg.norm(v[be[:, 0]] - v[be[:, 1]], axis=1).sum()) if len(be) else 0.0
    size = (vv.max(axis=0) - vv.min(axis=0)).tolist()
    rep = {
        **{k: info[k] for k in ("unit", "unit_guessed", "unit_was_guessed", "scale", "up", "up_reason")},
        "ground_cut": None if z_cut is None else {"z": z_cut, "height": z_cut - info["bounds_m"][0][2],
                                                  "removed_area_fraction": removed_area / max(total_area, 1e-30)},
        "ground_detected": info["ground"],
        "normals_flipped": flipped,
        "raw_triangles": int(len(f)), "raw_open_edge_length_m": open_len,
        "dx": dx, "grid": list(shape),
        "removed_pieces": removed_pieces, "removed_piece_volume_m3": removed_piece_volume,
        "filled_void_volume_m3": filled_volume,
        "triangles": int(len(m.faces)), "watertight": bool(m.is_watertight),
        "volume_m3": float(m.volume) if m.is_watertight else None,
        "size_m": size,
        "offset_m": shift.tolist(),
        "footprint_m2": float(_footprint(vv, m.faces)) if z_cut is not None else None,
    }
    prog(1.0, "done")
    return {"vertices": np.asarray(m.vertices), "faces": np.asarray(m.faces, np.int64), "report": rep}


def _footprint(v, f):
    """Area of the flat base (faces in the z = 0 plane, facing down)."""
    c, n, a = face_geometry(np.asarray(v), np.asarray(f))
    sel = (np.abs(c[:, 2]) < 1e-6) & (n[:, 2] < -0.99)
    return a[sel].sum()


# ----------------------------------------------------------------------
# preview
# ----------------------------------------------------------------------
def decimate_preview(v, f, max_faces: int = 150_000):
    """Vertex-clustering decimation for display only (keeps the shape, not the topology)."""
    if len(f) <= max_faces:
        return v, f
    ext = float(np.ptp(v, axis=0).max())
    cell = ext / 120.0
    for _ in range(12):
        q = np.floor((v - v.min(axis=0)) / cell).astype(np.int64)
        key = q[:, 0] + 100_000 * (q[:, 1] + 100_000 * q[:, 2])
        u, inv = np.unique(key, return_inverse=True)
        ff = inv[f]
        ok = (ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])
        ff = ff[ok]
        if len(ff) <= max_faces:
            cnt = np.bincount(inv)
            vv = np.stack([np.bincount(inv, weights=v[:, i]) for i in range(3)], 1) / cnt[:, None]
            return vv, ff
        cell *= 1.3
    cnt = np.bincount(inv)
    vv = np.stack([np.bincount(inv, weights=v[:, i]) for i in range(3)], 1) / cnt[:, None]
    return vv, ff


# ----------------------------------------------------------------------
# synthetic scan (tests, demos)
# ----------------------------------------------------------------------
def synthetic_scan(seed: int = 3, holes: int = 2, table: bool = True, floater: bool = True, noise: float = 0.0003,
                   y_up: bool = True, invert: bool = False, tunnel: bool = True, h: float = 0.0025,
                   hole_radius: float = 0.018):
    """A fake phone scan of a ~26 cm rock with a tunnel, standing on a table. Returns
    (vertices in metres in the file frame, faces, truth) with truth = {'sdf': callable on
    z-up points, 'tunnel': (point, axis), 'floater': point}.

    Like a real scan: the bottom is open where the rock touches the table, there are
    holes, a loose piece, vertex noise, and (y_up) the glTF axis convention."""
    from skimage.measure import marching_cubes
    from .geometry import rock_sdf
    rng = np.random.default_rng(seed)
    blobs = [{"center": (0.0, 0.0, 0.07), "radii": (0.13, 0.09, 0.10)},
             {"center": (0.06, 0.02, 0.13), "radii": (0.07, 0.06, 0.06)}]
    t_pt, t_ax, t_r = np.array([0.0, 0.0, 0.07]), np.array([0.0, 1.0, 0.0]), 0.025

    def sdf(P):
        d = rock_sdf(P, blobs, seed, roughness=0.12, feature=0.05, blend=0.03)
        d = np.maximum(d, -P[..., 2])                       # flat base on the table (z = 0)
        if tunnel:
            r = P - t_pt
            radial = np.linalg.norm(r - (r @ t_ax)[..., None] * t_ax, axis=-1)
            d = np.maximum(d, t_r - radial)
        return d

    lo, hi = np.array([-0.18, -0.14, -0.02]), np.array([0.18, 0.14, 0.22])
    nn = np.ceil((hi - lo) / h).astype(int) + 1
    ax = [lo[a] + h * np.arange(nn[a]) for a in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
    Fv = sdf(P)
    Fv[0], Fv[-1], Fv[:, 0], Fv[:, -1], Fv[:, :, -1] = 1, 1, 1, 1, 1
    vr, fr, _, _ = marching_cubes(Fv, 0.0, spacing=(h, h, h))
    vr = vr + lo
    c, n, _ = face_geometry(vr, fr)
    keep = ~((c[:, 2] < 0.0015) & (n[:, 2] < -0.9))          # the table hides the base
    for _ in range(holes):                                    # camera blind spots
        ci = rng.integers(len(c))
        while c[ci, 2] < 0.04:
            ci = rng.integers(len(c))
        keep &= np.linalg.norm(c - c[ci], axis=1) > hole_radius
    parts_v, parts_f = [vr], [fr[keep]]
    off = len(vr)
    if table:
        g = np.linspace(-0.3, 0.3, 61)
        X, Y = np.meshgrid(g, g, indexing="ij")
        tv = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        idx = np.arange(X.size).reshape(X.shape)
        a, b, cc, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        tf = np.concatenate([np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)])
        tc = tv[tf].mean(axis=1)
        inside_fp = sdf(np.column_stack([tc[:, :2], np.full(len(tc), 0.002)])) < 0
        parts_v.append(tv)
        parts_f.append(tf[~inside_fp] + off)
        off += len(tv)
    fl = np.array([0.24, 0.05, 0.09])
    if floater:
        import trimesh
        s = trimesh.creation.icosphere(subdivisions=2, radius=0.012)
        parts_v.append(np.asarray(s.vertices) + fl)
        parts_f.append(np.asarray(s.faces) + off)
    v = np.concatenate(parts_v)
    f = np.concatenate(parts_f)
    v = v + rng.normal(0, noise, v.shape)
    if invert:
        f = f[:, ::-1]
    if y_up:                                                  # z-up -> glTF y-up: inverse of UP_ROT['+y']
        v = v @ np.asarray(UP_ROT["+y"], np.float64)
    return v, f, {"sdf": sdf, "tunnel": (t_pt, t_ax, t_r), "floater": fl, "h": h}


def truth_volume(truth: dict, z_cut: float, h: float = 0.001) -> float:
    """Volume (m3) of the synthetic rock above z_cut, on a fine grid."""
    lo, hi = np.array([-0.18, -0.14, z_cut]), np.array([0.18, 0.14, 0.22])
    tot = 0.0
    xs = lo[0] + h * (np.arange(int((hi[0] - lo[0]) / h)) + 0.5)
    ys = lo[1] + h * (np.arange(int((hi[1] - lo[1]) / h)) + 0.5)
    zs = lo[2] + h * (np.arange(int((hi[2] - lo[2]) / h)) + 0.5)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    for x in xs:
        P = np.stack([np.full(Y.shape, x), Y, Z], -1)
        tot += float((truth["sdf"](P) < 0).sum())
    return tot * h ** 3
