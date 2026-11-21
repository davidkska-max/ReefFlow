"""Rock sculpting: the rockscape as a density grid that the browser edits with brushes.

The browser holds a uint8 density per grid node (0 = water, 255 = rock; the rock
surface is the 127.5 level) and edits it with push / pull / smooth / flatten
brushes in real time (static/js/sculpt.js). The server only

* builds the starting grid from the project's current rockscape meshes,
* stores drafts (projects/<pid>/sculpt/field.npy + state.json), and
* turns the grid into the project's rockscape (marching cubes -> STL), the same
  way "Use as rockscape" does for the photo reconstruction (DECISIONS D-820).

Grid layout
-----------
Node (i, j, k) has centre ``origin + (i, j, k) * dx``. x and y tile the tank's
inside [0, L] x [0, W]; z runs from 1.5 cells below the sand surface to the rim.
The two node layers below the sand always copy the first layer above it, so rock
that stands on the sand is buried in it (as the CFD tank model expects).
Wire format to the browser: raw bytes in x-fastest order, i.e. index
``i + nx * (j + ny * k)``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .. import geometry as G
from .. import units as U

SCULPT_DIR = "sculpt"
SCULPT_STL = "rockscape_sculpt.stl"
ISO = 127.5
SUPERSAMPLE = 3                     # odd: a cell is never exactly half rock (no ties at the 127.5 level)
DX_CHOICES = (0.003, 0.004, 0.005, 0.006)
MAX_NODES = 16_000_000


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------
def grid_for(tank, dx: float) -> dict:
    """Grid metadata for a reefflow.project.Tank at node spacing dx (m)."""
    dx = float(dx)
    if not 0.002 <= dx <= 0.01:
        raise ValueError("sculpt resolution must be between 2 mm and 10 mm")
    nx = max(int(round(tank.length / dx)), 4)
    ny = max(int(round(tank.width / dx)), 4)
    z0 = tank.sand_depth - 1.5 * dx
    nz = max(int(np.ceil((tank.height - z0) / dx)), 4)
    if nx * ny * nz > MAX_NODES:
        raise ValueError(f"sculpt grid {nx} x {ny} x {nz} is too large at {dx * 1e3:.0f} mm; choose a coarser resolution")
    # nodes centred in x and y (whole cells, like the CFD grid)
    origin = [0.5 * (tank.length - (nx - 1) * dx), 0.5 * (tank.width - (ny - 1) * dx), z0]
    k_sand = int(np.searchsorted(z0 + dx * np.arange(nz), tank.sand_depth))   # first node above the sand
    return {"dx": dx, "origin": [float(c) for c in origin], "shape": [nx, ny, nz], "k_sand": k_sand,
            "sand_z": float(tank.sand_depth), "water_z": float(tank.water_surface),
            "tank": [float(tank.length), float(tank.width), float(tank.height)]}


def bury(D: np.ndarray, k_sand: int) -> np.ndarray:
    """Layers below the sand copy the first layer above it (rock is buried, never floats)."""
    if k_sand > 0:
        D[:, :, :k_sand] = D[:, :, k_sand:k_sand + 1]
    return D


def rock_meshes(raw: dict, base_dir: Path) -> list:
    """(vertices, faces) of every rockscape mesh in the project, in tank coordinates."""
    out = []
    for r in raw.get("rockscape") or []:
        f = Path(r["file"])
        f = f if f.is_absolute() else Path(base_dir) / f
        if not f.is_file():
            raise FileNotFoundError(f"rock mesh {f.name} not found")
        tr = [U.parse(c, "length") for c in r.get("translate", (0, 0, 0))]
        out.append(G.load_mesh(f, r.get("unit", "mm"), tr, float(r.get("rotate_z", 0.0)), float(r.get("scale", 1.0))))
    return out


def field_from_meshes(meshes: list, grid: dict, progress=None) -> np.ndarray:
    """uint8 density (nx, ny, nz): the rock volume fraction of each node's cell, from a
    3 x 3 x 3 supersampled voxelization (sub-cell accuracy, so the starting surface is
    not stair-stepped)."""
    nx, ny, nz = grid["shape"]
    dx = grid["dx"]
    s = SUPERSAMPLE
    h = dx / s
    o = np.asarray(grid["origin"]) - 0.5 * dx + 0.5 * h        # first sub-node centre
    frac = np.zeros((nx, ny, nz), np.float32)
    if not meshes:
        return frac.astype(np.uint8)
    slab = max(1, 2_000_000 // (ny * nz * s * s * s))          # coarse x-cells per slab (bounds memory)
    for i0 in range(0, nx, slab):
        i1 = min(nx, i0 + slab)
        shape = ((i1 - i0) * s, ny * s, nz * s)
        oo = o + np.array([i0 * dx, 0.0, 0.0])
        inside = np.zeros(shape, bool)
        for v, f in meshes:
            inside |= G.voxelize_mesh(v, f, oo, h, shape)
        frac[i0:i1] = inside.reshape(i1 - i0, s, ny, s, nz, s).mean(axis=(1, 3, 5))
        if progress:
            progress(i1 / nx)
    D = np.rint(frac * 255).astype(np.uint8)
    return bury(D, grid["k_sand"])


def to_wire(D: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(D, np.uint8).transpose(2, 1, 0)).tobytes()


def from_wire(buf: bytes, shape) -> np.ndarray:
    nx, ny, nz = (int(c) for c in shape)
    a = np.frombuffer(buf, np.uint8)
    if a.size != nx * ny * nz:
        raise ValueError(f"sculpt field has {a.size} values, expected {nx * ny * nz} ({nx} x {ny} x {nz})")
    return np.ascontiguousarray(a.reshape(nz, ny, nx).transpose(2, 1, 0))


# ----------------------------------------------------------------------
# storage (projects/<pid>/sculpt/)
# ----------------------------------------------------------------------
def rockscape_signature(raw: dict, base_dir: Path) -> str:
    """Identifies the project's current rockscape: entries + mesh file sizes and times."""
    items = []
    for r in raw.get("rockscape") or []:
        f = Path(r["file"])
        f = f if f.is_absolute() else Path(base_dir) / f
        st = f.stat() if f.is_file() else None
        items.append([r.get("file"), r.get("unit"), r.get("translate"), r.get("rotate_z"), r.get("scale"),
                      st.st_size if st else None, int(st.st_mtime_ns) if st else None])
    return hashlib.sha1(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _paths(pdir: Path):
    d = Path(pdir) / SCULPT_DIR
    return d, d / "field.npy", d / "state.json"


def load_state(pdir: Path) -> dict | None:
    _, fpath, spath = _paths(pdir)
    if not (fpath.is_file() and spath.is_file()):
        return None
    try:
        return json.loads(spath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_field(pdir: Path, D: np.ndarray, grid: dict, base_sig: str, **extra) -> dict:
    d, fpath, spath = _paths(pdir)
    d.mkdir(exist_ok=True)
    tmp = fpath.with_suffix(".tmp.npy")
    np.save(tmp, np.asarray(D, np.uint8))
    tmp.replace(fpath)
    st = {"grid": grid, "base_sig": base_sig, "time": time.time(), **extra}
    spath.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return st


def load_field(pdir: Path) -> np.ndarray:
    _, fpath, _ = _paths(pdir)
    return np.load(fpath)


def start(raw: dict, pdir: Path, tank, dx: float | None = None, fresh: bool = False) -> tuple[np.ndarray, dict]:
    """Starting grid for a sculpt session -> (D, info).

    Resumes the stored grid when it was made from (or applied as) the current
    rockscape and the resolution matches; otherwise voxelizes the rockscape."""
    sig = rockscape_signature(raw, pdir)
    st = load_state(pdir)
    if st and not fresh and st.get("base_sig") == sig and (dx is None or abs(st["grid"]["dx"] - dx) < 1e-9):
        g = grid_for(tank, st["grid"]["dx"])
        if g["shape"] == st["grid"]["shape"] and np.allclose(g["origin"], st["grid"]["origin"]):
            D = load_field(pdir)
            return D, {"grid": g, "source": "resumed", "draft": bool(st.get("draft")),
                       "saved": st.get("time"), "base_sig": sig}
    g = grid_for(tank, dx or (st["grid"]["dx"] if st else 0.004))
    t0 = time.time()
    meshes = rock_meshes(raw, pdir)
    D = field_from_meshes(meshes, g)
    src = f"voxelized from the rockscape ({len(meshes)} mesh{'es' if len(meshes) != 1 else ''})" if meshes else "empty"
    return D, {"grid": g, "source": src, "draft": False, "seconds": time.time() - t0, "base_sig": sig}


# ----------------------------------------------------------------------
# grid -> mesh -> project rockscape
# ----------------------------------------------------------------------
def to_mesh(D: np.ndarray, grid: dict):
    """Closed, outward-oriented triangle mesh (vertices in m, faces) of the 127.5 level."""
    from skimage.measure import marching_cubes
    import trimesh
    a = bury(np.array(D, np.uint8, copy=True), grid["k_sand"]).astype(np.float32)
    if not (a >= ISO).any():
        raise ValueError("the sculpt grid contains no rock")
    F = np.pad(a, 1)                                     # zero layer all round: closed surface
    dx = grid["dx"]
    verts, faces, _, _ = marching_cubes(F, level=ISO, spacing=(dx, dx, dx))
    verts = verts + np.asarray(grid["origin"]) - dx
    m = trimesh.Trimesh(verts, faces.astype(np.int64), process=True)
    m.remove_unreferenced_vertices()
    if m.is_watertight and m.volume < 0:
        m.invert()
    return np.asarray(m.vertices), np.asarray(m.faces), m


def report(D: np.ndarray, grid: dict, tank, mesh=None) -> dict:
    from . import rockgen as RG
    occ = np.asarray(D) >= ISO
    dx = grid["dx"]
    k = grid["k_sand"]
    above = occ[:, :, k:]
    rep = {"resolution_m": dx, "grid": list(grid["shape"]),
           "volume_voxel_m3": float(above.sum() * dx ** 3)}
    if mesh is not None:
        rep.update(triangles=int(len(mesh.faces)), watertight=bool(mesh.is_watertight),
                   volume_m3=float(abs(mesh.volume)) if mesh.is_watertight else rep["volume_voxel_m3"])
        rep["note"] = ("volume_m3 is the mesh volume including the part buried 1.5 cells into the sand; "
                       "volume_voxel_m3 counts rock nodes above the sand surface")
    rep["issues"] = RG.check(occ, np.asarray(grid["origin"]), dx, tank, [], [])
    return rep


def apply(D: np.ndarray, grid: dict, tank, pdir: Path, raw: dict) -> tuple[dict, dict]:
    """Write the sculpt as SCULPT_STL and make it the project's rockscape.

    Returns (new raw project dict, report). The designer then holds the sculpt as one
    'mesh' element (already built), like a photo reconstruction."""
    import copy

    from . import editor_model as EM
    v, f, m = to_mesh(D, grid)
    tmp = Path(pdir) / SCULPT_STL.replace(".stl", ".tmp.stl")      # the writer picks the format from the extension
    G.save_stl(tmp, v, f, unit="mm")
    tmp.replace(Path(pdir) / SCULPT_STL)
    raw = copy.deepcopy(raw)
    raw["rockscape"] = [{"file": SCULPT_STL, "unit": "mm", "translate": ["0 mm", "0 mm", "0 mm"]}]
    els = [{"id": "sculpt", "type": "mesh", "name": "sculpted rock", "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "params": {"file": SCULPT_STL, "unit": "mm"}}]
    dz = copy.deepcopy(raw.get("designer") or {})
    dz.update(elements=els, rockscape_file=None, built_hash=EM.elements_hash(els))
    dz.setdefault("resolution", 0.004)
    dz.setdefault("rock_density", 1600.0)
    raw["designer"] = dz
    rep = report(D, grid, tank, m)
    return raw, rep
