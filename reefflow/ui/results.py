"""Results access for the viewer: run summaries and 2-D slices served from the run files.

Only slices (2-D float32) and small tables go to the browser; the 3-D arrays stay on
the server (a Fine run has ~9 M cells). Slices come straight from fields.npz /
analysis.npz / par.npz (no tank-model rebuild needed: those files carry tags,
flags, origin and dx).
"""
from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..domain import FLUID, SOURCE
from ..metrics import KEY_METRICS
from ..tank import TAGS
from . import lighting_adapter as LA

FIELDS = {
    # name: (label, SI unit, description)
    "speed": ("time-averaged speed <|u|>", "m/s", "mean of the instantaneous speed"),
    "umag": ("|mean velocity|", "m/s", "magnitude of the time-averaged velocity"),
    "sigma": ("RMS fluctuation sqrt(2k/3)", "m/s", "per-component velocity fluctuation"),
    "ti": ("turbulence intensity", "-", "sigma / <|u|>"),
    "age": ("water age", "s", "mean time since the water entered from the return (steady solve)"),
    "dead": ("dead zone", "-", "1 where <|u|> is below the dead-zone threshold"),
    "par": ("PAR (meter-equivalent)", "umol/m2/s", "photosynthetically active radiation (Phase 4)"),
}


class _Cache:
    """Small LRU of decompressed arrays keyed by (file, mtime, key)."""

    def __init__(self, max_items: int = 12):
        self.d: OrderedDict = OrderedDict()
        self.max = max_items
        self.lock = threading.Lock()

    def get(self, path: Path, key: str):
        st = path.stat()
        k = (str(path), st.st_mtime, key)
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        with np.load(path, allow_pickle=False) as z:
            if key not in z.files:
                raise KeyError(f"{key} not in {path.name}")
            a = z[key]
        with self.lock:
            self.d[k] = a
            while len(self.d) > self.max:
                self.d.popitem(last=False)
        return a


CACHE = _Cache()


def grid_info(run_dir: Path) -> dict:
    f = run_dir / "fields.npz"
    origin = CACHE.get(f, "origin")
    dx = float(CACHE.get(f, "dx"))
    shape = CACHE.get(f, "tags").shape
    return {"origin_m": [float(v) for v in origin], "dx_m": dx, "shape": list(shape)}


def fluid_mask(run_dir: Path) -> np.ndarray:
    fl = CACHE.get(run_dir / "fields.npz", "flags")
    return (fl == FLUID) | (fl == SOURCE)


def _plane(run_dir: Path, name: str, a: int, k: int) -> np.ndarray:
    """Plane k normal to axis a of field ``name`` (float32, NaN outside the water)."""
    f = run_dir / "fields.npz"

    def take(path, key):
        return np.take(CACHE.get(path, key), k, axis=a)

    fl = np.take(fluid_mask(run_dir), k, axis=a)
    if name == "speed":
        v = take(f, "mean_speed")
    elif name == "umag":
        v = np.linalg.norm(take(f, "mean_u"), axis=-1)
    elif name in ("sigma", "ti"):
        v = np.sqrt(2.0 / 3.0 * np.clip(take(f, "tke"), 0, None))
        if name == "ti":
            v = v / np.maximum(take(f, "mean_speed"), 1e-4)
    elif name == "age":
        if CACHE.get(run_dir / "analysis.npz", "age").size == 0:
            raise KeyError("water age not available (no return pump / overflow)")
        v = take(run_dir / "analysis.npz", "age")
        v = np.where(v < 0, np.nan, v)
    elif name == "dead":
        v = take(run_dir / "analysis.npz", "dead").astype(np.float32)
    elif name == "par":
        p = LA.load_par(run_dir)
        if p is None:
            raise KeyError("no PAR results for this run (run with lighting, Phase 4)")
        v = np.take(p["meter"], k, axis=a)
    else:
        raise KeyError(f"unknown field {name!r}")
    return np.where(fl, v, np.nan).astype(np.float32)


def slice_field(run_dir: Path, name: str, axis: str, pos: float) -> dict:
    """2-D slice at tank coordinate ``pos`` (m) normal to ``axis`` ('x' | 'y' | 'z').

    Returns {'data': float32 (rows, cols), 'extent': [u0, u1, v0, v1] (cell-centre coordinates of the
    first/last column and row), 'index', 'pos', 'axes': (u, v)}. Rows run along v, columns along u:
    z -> (u, v) = (x, y); y -> (x, z); x -> (y, z)."""
    a = "xyz".index(axis)
    g = grid_info(run_dir)
    o, dx, n = g["origin_m"], g["dx_m"], g["shape"]
    k = int(round((pos - o[a]) / dx))
    k = int(np.clip(k, 1, n[a] - 2))
    sl = _plane(run_dir, name, a, k)                   # 2-D, axes = the other two in order
    other = [i for i in range(3) if i != a]
    data = np.ascontiguousarray(sl.T)                  # rows = second remaining axis
    u, v = other
    ext = [o[u], o[u] + dx * (n[u] - 1), o[v], o[v] + dx * (n[v] - 1)]
    return {"data": data, "extent": ext, "index": k, "pos": o[a] + k * dx, "axes": ("xyz"[u], "xyz"[v]),
            "dx": dx}


def slice_vectors(run_dir: Path, axis: str, pos: float, stride: int | None = None) -> dict:
    """In-plane mean-velocity components on a slice, sub-sampled every ``stride`` cells."""
    a = "xyz".index(axis)
    g = grid_info(run_dir)
    o, dx, n = g["origin_m"], g["dx_m"], g["shape"]
    k = int(np.clip(int(round((pos - o[a]) / dx)), 1, n[a] - 2))
    U = CACHE.get(run_dir / "fields.npz", "mean_u")
    fl = fluid_mask(run_dir)
    other = [i for i in range(3) if i != a]
    sl = np.take(U, k, axis=a)                         # (n_u, n_v, 3)
    m = np.take(fl, k, axis=a)
    if stride is None:
        stride = max(1, int(round(max(n[other[0]], n[other[1]]) / 48)))
    pts, vec = [], []
    for i in range(stride // 2, sl.shape[0], stride):
        for j in range(stride // 2, sl.shape[1], stride):
            if m[i, j]:
                pts.append((o[other[0]] + i * dx, o[other[1]] + j * dx))
                vec.append((sl[i, j, other[0]], sl[i, j, other[1]]))
    return {"points": np.asarray(pts, np.float32).reshape(-1, 2), "vectors": np.asarray(vec, np.float32).reshape(-1, 2),
            "index": k, "pos": o[a] + k * dx, "axes": ("xyz"[other[0]], "xyz"[other[1]]), "stride": stride}


def velocity_volume(run_dir: Path, max_nodes: int = 600_000) -> dict:
    """Time-averaged velocity on the whole grid for the particle animation (D-849), every
    ``stride``-th node so at most ``max_nodes`` go to the browser; NaN at rock / outside nodes."""
    g = grid_info(run_dir)
    n = g["shape"]
    stride = max(1, math.ceil((n[0] * n[1] * n[2] / max_nodes) ** (1.0 / 3.0)))
    U = CACHE.get(run_dir / "fields.npz", "mean_u")[::stride, ::stride, ::stride]
    m = fluid_mask(run_dir)[::stride, ::stride, ::stride]
    out = np.where(m[..., None], U, np.nan).astype(np.float32)
    return {"U": out, "shape": list(out.shape[:3]), "origin": g["origin_m"], "dx": g["dx_m"] * stride, "stride": stride}


def sand_shear(run_dir: Path) -> dict:
    """Bed shear stress map on the sand (Pa; NaN where not sampled), rows = y, cols = x."""
    a = CACHE.get(run_dir / "analysis.npz", "sand_tau").astype(np.float32)
    a = np.where(a < 0, np.nan, a)
    g = grid_info(run_dir)
    o, dx, n = g["origin_m"], g["dx_m"], g["shape"]
    return {"data": np.ascontiguousarray(a.T), "extent": [o[0], o[0] + dx * (n[0] - 1), o[1], o[1] + dx * (n[1] - 1)],
            "z": o[2] + 0.5 * dx}


def surface_scores(run_dir: Path, profile: str) -> dict:
    """Coral score points on the rock (positions m, score 0-1, speed m/s, TI)."""
    z = run_dir / "analysis.npz"
    nodes = CACHE.get(z, "surf_nodes")
    g = grid_info(run_dir)
    xyz = np.asarray(g["origin_m"]) + nodes * g["dx_m"]
    return {"points": xyz.astype(np.float32), "score": CACHE.get(z, f"score_{profile}").astype(np.float32),
            "speed": CACHE.get(z, "surf_speed").astype(np.float32), "ti": CACHE.get(z, "surf_ti").astype(np.float32)}


def summary(run_dir: Path) -> dict:
    """Everything the results panel shows, small enough for one JSON response."""
    out = {"files": sorted(p.name for p in run_dir.iterdir() if p.is_file())}
    rj = run_dir / "run.json"
    if rj.is_file():
        info = json.loads(rj.read_text(encoding="utf-8"))
        out["run"] = {k: info.get(k) for k in (
            "project", "backend", "precision", "lattice", "collision", "smagorinsky", "dx_m", "spinup_s", "average_s",
            "steps", "wall_time_s", "mlups", "mass_rel_drift", "u_max_overall_ms", "u_max_overall_lattice",
            "fraction_time_above_u_lat_0.15", "ke_avg_first_half", "ke_avg_second_half", "notes", "metrics",
            "metrics_stat_uncertainty")}
        mod = info.get("model") or {}
        out["run"]["pumps"] = mod.get("pumps")
        out["run"]["overflow"] = mod.get("overflow")
        out["run"]["rock_volume_m3"] = mod.get("rock_volume_m3")
        out["run"]["water_volume_model_m3"] = mod.get("water_volume_model_m3")
    aj = run_dir / "analysis.json"
    if aj.is_file():
        a = json.loads(aj.read_text(encoding="utf-8"))
        a.pop("derating_sensitivity", None)
        out["analysis"] = a
    if (run_dir / "fields.npz").is_file():
        out["grid"] = grid_info(run_dir)
    layers = {"speed": (run_dir / "fields.npz").is_file()}
    layers["umag"] = layers["sigma"] = layers["ti"] = layers["vectors"] = layers["speed"]
    an = (run_dir / "analysis.npz").is_file()
    layers["dead"] = an
    layers["sand_shear"] = an
    layers["coral"] = an
    try:
        layers["age"] = an and CACHE.get(run_dir / "analysis.npz", "age").size > 0
    except Exception:
        layers["age"] = False
    layers["par"] = LA.par_available(run_dir)
    out["layers"] = layers
    out["par_label"] = (LA.load_par(run_dir) or {}).get("label") if layers["par"] else None
    out["profiles"] = sorted(p.name[len("rock_score_"):-4] for p in run_dir.glob("rock_score_*.ply"))
    out["key_metrics"] = {k: {"label": v[0], "unit": v[1], "scale": v[2]} for k, v in KEY_METRICS.items()}
    out["fields"] = {k: {"label": v[0], "unit": v[1], "description": v[2]} for k, v in FIELDS.items()}
    out["tags"] = TAGS
    return out
