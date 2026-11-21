"""Project file: one human-readable YAML per tank, so every run is reproducible.

Quantities are written as "<number> <unit>" strings ("48 in", "2500 GPH",
"60 s") or as plain numbers in SI. Internally everything is SI.

Top-level sections
    name, display_units (US | SI)
    tank         external size, glass thickness, water level, sand depth
    rockscape    list of mesh files (STL/OBJ/PLY) with unit and placement
    overflow     overflow box on any glass wall (mass sink over its weir crest)
    pumps        list of pumps (archetype from the pump library + overrides)
    fluid        density, kinematic viscosity
    simulation   lattice, collision, LES constant, resolution preset, run times, ...
    criteria     metric thresholds (Phase 3 extends this)

See ``reefflow demo`` for a complete, commented example.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import units as U
from .pumps import Pump, build_pump, load_library

FORMAT_VERSION = 1

PRESETS = {"preview": 0.010, "standard": 0.005, "fine": 0.003}   # cell size, m (spec)

SIM_DEFAULTS = {
    "lattice": "D3Q19",              # D-120 (user-approved)
    "collision": "rr",               # recursive regularized (D-004)
    "smagorinsky": 0.10,             # D-118 (user-approved)
    "preset": "standard",
    "dx": None,                      # overrides the preset when given
    "spinup": "60 s",
    "average": "120 s",
    "u_lattice_max": 0.1,            # lattice-velocity target at the estimated peak speed
    "min_source_cells": 4,           # PLAN C3
    "source_min_size": None,         # fixed physical minimum outlet size (grid study)
    "source_velocity_cap": "1.0 m/s",  # D-203
    "source_length": "20 mm",        # axial length of a source region
    "sample_every": 10,              # steps between statistics samples
    "batches": 4,                    # averaging window split for the statistical-uncertainty estimate
    "seed": 1,
    "wall_model": False,             # D-702: EXPERIMENTAL log-law wall model (failed verification; keep off)
}
FLUID_DEFAULTS = {"density": "1023 kg/m3", "viscosity": 0.94e-6}   # A-101, A-102
CRITERIA_DEFAULTS = {"dead_zone_speed": "2.5 cm/s"}                # A-210 (starting assumption)


@dataclass
class Tank:
    length: float                    # internal, m
    width: float
    height: float                    # internal height above the bottom glass
    glass: float
    bottom_glass: float
    water_gap: float                 # rim to water surface
    sand_depth: float
    external: tuple

    @property
    def water_surface(self) -> float:
        return self.height - self.water_gap

    @property
    def water_column(self) -> float:
        return self.water_surface - self.sand_depth

    def summary(self) -> dict:
        return {
            "external_in": [U.from_si(v, "in") for v in self.external],
            "internal_m": [self.length, self.width, self.height],
            "internal_in": [U.from_si(v, "in") for v in (self.length, self.width, self.height)],
            "glass_in": U.from_si(self.glass, "in"), "bottom_glass_in": U.from_si(self.bottom_glass, "in"),
            "water_surface_above_bottom_m": self.water_surface, "sand_depth_m": self.sand_depth,
            "water_column_m": self.water_column,
            "gross_volume_gal": U.from_si(self.length * self.width * self.height, "gal"),
            "water_volume_above_sand_gal": U.from_si(self.length * self.width * self.water_column, "gal"),
        }


@dataclass
class RockItem:
    file: Path
    unit: str = "mm"
    translate: tuple = (0.0, 0.0, 0.0)
    rotate_z: float = 0.0
    scale: float = 1.0


CORNERS = ("back-right", "back-left", "front-right", "front-left")
WALLS = ("back", "front", "left", "right")


@dataclass
class Overflow:
    """Overflow box against one glass wall (D-801).

    Two project-file forms:
      * wall form: ``wall`` (back | front | left | right), ``center`` (box centre along
        the wall, from the left glass for back/front walls or from the front glass for
        left/right walls; default mid-wall), ``width`` (along the wall), ``depth`` (out
        from the wall). width = wall length is a coast-to-coast overflow.
      * corner form (Phase 2, still read): ``corner`` with ``width`` along x and
        ``depth`` along y. Its footprint is computed exactly as before.
    """
    corner: str | None               # corner form only
    size_x: float                    # footprint extent along x
    size_y: float                    # footprint extent along y
    crest_below_surface: float
    wall: str = "back"
    center: float | None = None      # along the wall (m); None = mid-wall
    width: float = 0.0               # along the wall
    depth: float = 0.0               # out from the wall
    capacity: float | None = None    # rated drain capacity (m^3/s), D-846: a check, not a flow limit
    product: str | None = None       # catalogue id when the box is a real product (D-844)

    def wall_length(self, tank: Tank) -> float:
        return tank.length if self.wall in ("back", "front") else tank.width

    def box(self, tank: Tank):
        """(x0, x1, y0, y1) footprint in tank coordinates."""
        if self.corner is not None:
            right = "right" in self.corner
            back = "back" in self.corner
            x0, x1 = (tank.length - self.size_x, tank.length) if right else (0.0, self.size_x)
            y0, y1 = (tank.width - self.size_y, tank.width) if back else (0.0, self.size_y)
            return x0, x1, y0, y1
        Lw = self.wall_length(tank)
        c = Lw / 2 if self.center is None else self.center
        a0, a1 = max(c - self.width / 2, 0.0), min(c + self.width / 2, Lw)
        if self.wall == "back":
            return a0, a1, tank.width - self.depth, tank.width
        if self.wall == "front":
            return a0, a1, 0.0, self.depth
        if self.wall == "left":
            return 0.0, self.depth, a0, a1
        return tank.length - self.depth, tank.length, a0, a1

    def describe(self, tank: Tank) -> str:
        if self.corner is not None:
            return f"{self.corner} corner"
        Lw = self.wall_length(tank)
        c = Lw / 2 if self.center is None else self.center
        if self.width >= Lw - 1e-6:
            return f"{self.wall} wall, coast to coast"
        return f"{self.wall} wall, centre {c * 1e3:.0f} mm along the wall"


def parse_overflow(o: dict, tank: Tank) -> Overflow:
    """Overflow from its project-file entry (either form), checked against the tank."""
    crest = _q(o, "crest_below_surface", "length", "1/2 in")
    if crest < 0:
        raise ValueError("overflow crest_below_surface must be >= 0")
    if "wall" not in o:
        corner = o.get("corner", "back-right")
        if corner not in CORNERS:
            raise ValueError(f"overflow corner {corner!r} not supported (use one of {', '.join(CORNERS)}, "
                             "or the wall/center form)")
        sx, sy = _q(o, "width", "length"), _q(o, "depth", "length")
        if not (0 < sx <= tank.length + 1e-9 and 0 < sy < tank.width):
            raise ValueError(f"overflow {sx * 1e3:.0f} x {sy * 1e3:.0f} mm does not fit in the tank "
                             f"({tank.length * 1e3:.0f} x {tank.width * 1e3:.0f} mm inside)")
        wall = "back" if "back" in corner else "front"
        Lw = tank.length
        c = Lw - sx / 2 if "right" in corner else sx / 2
        return Overflow(corner, sx, sy, crest, wall=wall, center=c, width=sx, depth=sy, **_ov_extra(o))
    wall = o["wall"]
    if wall not in WALLS:
        raise ValueError(f"overflow wall {wall!r} not supported (use one of {', '.join(WALLS)})")
    w, d = _q(o, "width", "length"), _q(o, "depth", "length")
    along_x = wall in ("back", "front")
    Lw, Ld = (tank.length, tank.width) if along_x else (tank.width, tank.length)
    tol = 1e-6
    if w <= 0 or d <= 0:
        raise ValueError("overflow width and depth must be > 0")
    if w > Lw + tol:
        raise ValueError(f"overflow width {w * 1e3:.1f} mm is longer than the {wall} glass "
                         f"({Lw * 1e3:.1f} mm inside); use width = {Lw * 1e3:.1f} mm for coast to coast")
    w = min(w, Lw)
    if d >= Ld:
        raise ValueError(f"overflow depth {d * 1e3:.1f} mm reaches the opposite glass ({Ld * 1e3:.1f} mm)")
    c = Lw / 2 if o.get("center") is None else U.parse(o["center"], "length")
    lo, hi = w / 2, Lw - w / 2
    if c < lo - tol or c > hi + tol:
        ref = "left" if along_x else "front"
        raise ValueError(f"overflow centre {c * 1e3:.1f} mm puts the box outside the tank: for a "
                         f"{w * 1e3:.1f} mm wide box on the {wall} glass the centre must be "
                         f"{lo * 1e3:.1f}-{hi * 1e3:.1f} mm from the {ref} glass")
    c = min(max(c, lo), hi)
    sx, sy = (w, d) if along_x else (d, w)
    return Overflow(None, sx, sy, crest, wall=wall, center=c, width=w, depth=d, **_ov_extra(o))


def load_overflow_catalog() -> dict:
    """Overflow products and generic drain capacities (D-844, D-846):
    {"products": {id: entry}, "drain_capacity": [...]}. Missing file -> empty."""
    from importlib import resources
    try:
        with resources.files("reefflow.data").joinpath("overflow_products.yaml").open("r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        d = {}
    return {"products": d.get("products") or {}, "drain_capacity": d.get("drain_capacity") or []}


def _ov_extra(o: dict) -> dict:
    cap = U.parse(o["capacity"], "flow") if o.get("capacity") is not None else None
    if cap is not None and cap <= 0:
        raise ValueError("overflow capacity must be > 0 (leave it out if unknown)")
    return {"capacity": cap, "product": o.get("product")}


def return_check(p: "Project") -> list[str]:
    """Warn when a return nozzle's exit velocity head U^2/2g is larger than the pump's fittings
    allowance (D-846). The pump-curve flow assumes the exit head is part of that allowance, so the
    real flow would then be lower than the one used. A check only: the flow is not changed."""
    out = []
    for pp in p.pumps:
        if pp.kind != "return" or pp.fittings_allowance is None or pp.flow_basis == "user override":
            continue
        for k, s in enumerate(pp.nozzle_summaries()):
            h = s["exit_velocity_head_m"]
            if h > pp.fittings_allowance + 1e-9:
                which = f" nozzle {k + 1}" if len(pp.nozzles) > 1 else ""
                out.append(f"return {pp.name!r}{which}: water leaves the nozzle at {s['velocity_ms']:.1f} m/s, an exit head of "
                           f"{U.from_si(h, 'ft'):.1f} ft ({h:.2f} m), more than the whole fittings allowance "
                           f"{U.from_si(pp.fittings_allowance, 'ft'):.1f} ft. The real flow would be lower than "
                           f"{U.from_si(pp.delivered_flow, 'GPH'):.0f} GPH: use a bigger nozzle or more nozzles, raise the "
                           "fittings allowance, or enter your measured flow as the delivered flow override.")
    return out


def overflow_check(p: "Project") -> list[str]:
    """Warnings when the return flow exceeds the overflow's rated drain capacity (D-846).
    The flow model does not limit the overflow; this only flags a setup that would flood."""
    o = p.overflow
    if o is None or not o.capacity:
        return []
    q = sum(pp.delivered_flow for pp in p.pumps if pp.kind == "return")
    if q <= o.capacity:
        return []
    return [f"return flow {U.from_si(q, 'GPH'):.0f} GPH ({U.from_si(q, 'L/h'):.0f} L/h) is more than the overflow's "
            f"rated capacity {U.from_si(o.capacity, 'GPH'):.0f} GPH ({U.from_si(o.capacity, 'L/h'):.0f} L/h): "
            "in the real tank it would overflow or the siphon would break. Turn the return pump down "
            "(Delivered flow override) or check the overflow's rating."]


@dataclass
class Project:
    path: Path | None
    name: str
    display_units: str
    tank: Tank
    rocks: list
    overflow: Overflow | None
    pumps: list
    fluid: dict
    sim: dict
    criteria: dict
    raw: dict = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return self.path.parent if self.path else Path.cwd()

    def dx(self, preset: str | None = None) -> float:
        if preset is None and self.sim.get("dx") is not None:
            return U.parse(self.sim["dx"], "length")
        p = preset or self.sim["preset"]
        if p not in PRESETS:
            raise ValueError(f"unknown preset {p!r}; choose from {sorted(PRESETS)}")
        return PRESETS[p]

    def with_overrides(self, **sim) -> "Project":
        """Copy with simulation settings replaced (used by the CLI and the grid study)."""
        raw = copy.deepcopy(self.raw)
        rs = raw.setdefault("simulation", {})
        for k, v in sim.items():
            if v is not None:
                rs[k] = v
            elif k == "dx":              # dx=None explicitly: use the preset
                rs.pop("dx", None)
        return parse(raw, self.path)


def _q(d, key, kind, default=None):
    v = d.get(key, default)
    if v is None:
        raise ValueError(f"missing required field {key!r}")
    return U.parse(v, kind)


def parse(raw: dict, path: Path | None = None) -> Project:
    if not isinstance(raw, dict):
        raise ValueError("project file must be a YAML mapping")
    raw = copy.deepcopy(raw)
    ver = raw.get("reefflow_project", FORMAT_VERSION)
    if ver != FORMAT_VERSION:
        raise ValueError(f"unsupported project format version {ver}")
    t = raw["tank"]
    ext = t["external"]
    L, W, H = (_q(ext, k, "length") for k in ("length", "width", "height"))
    if t.get("style") not in (None, "rimmed", "rimless_top", "rimless"):
        raise ValueError("tank style must be rimmed, rimless_top or rimless")
    g = _q(t, "glass_thickness", "length", "1/2 in")             # D-007
    gb = _q(t, "bottom_glass_thickness", "length", t.get("glass_thickness", "1/2 in"))
    tank = Tank(length=L - 2 * g, width=W - 2 * g, height=H - gb, glass=g, bottom_glass=gb,
                water_gap=_q(t, "water_level_below_rim", "length", "1 in"),
                sand_depth=_q(t, "sand_depth", "length", "0 in"), external=(L, W, H))
    if min(tank.length, tank.width, tank.water_column) <= 0:
        raise ValueError("tank dimensions leave no water column")

    base = Path(path).parent if path else Path.cwd()
    rocks = []
    for r in raw.get("rockscape") or []:
        f = Path(r["file"])
        rocks.append(RockItem(file=f if f.is_absolute() else base / f, unit=r.get("unit", "mm"),
                              translate=tuple(U.parse(v, "length") for v in r.get("translate", (0, 0, 0))),
                              rotate_z=float(r.get("rotate_z", 0.0)), scale=float(r.get("scale", 1.0))))

    ov = parse_overflow(raw["overflow"], tank) if raw.get("overflow") else None

    lib = load_library()
    pumps: list[Pump] = [build_pump(p, lib) for p in raw.get("pumps") or []]
    names = [p.name for p in pumps]
    if len(set(names)) != len(names):
        raise ValueError("pump names must be unique")
    if any(p.kind == "return" for p in pumps) and ov is None:
        raise ValueError("a return pump needs an overflow (where the water leaves the tank)")

    fluid_raw = {**FLUID_DEFAULTS, **(raw.get("fluid") or {})}
    fluid = {"density": U.parse(fluid_raw["density"], "density"), "viscosity": float(fluid_raw["viscosity"])}
    sim = {**SIM_DEFAULTS, **(raw.get("simulation") or {})}
    for k in ("spinup", "average"):
        sim[k + "_s"] = U.parse(sim[k], "time")
    sim["source_velocity_cap_ms"] = (None if sim["source_velocity_cap"] in (None, "none")
                                     else U.parse(sim["source_velocity_cap"], "velocity"))
    sim["source_length_m"] = U.parse(sim["source_length"], "length")
    sim["source_min_size_m"] = (None if sim.get("source_min_size") is None
                                else U.parse(sim["source_min_size"], "length"))
    if sim["lattice"].upper() not in ("D3Q19", "D3Q27"):
        raise ValueError("simulation.lattice must be D3Q19 or D3Q27")
    crit_raw = {**CRITERIA_DEFAULTS, **(raw.get("criteria") or {})}
    criteria = {"dead_zone_speed": U.parse(crit_raw["dead_zone_speed"], "velocity")}
    du = raw.get("display_units", "US")
    if du not in ("US", "SI"):
        raise ValueError("display_units must be US or SI")
    proj = Project(path=Path(path) if path else None, name=raw.get("name", "untitled"), display_units=du,
                   tank=tank, rocks=rocks, overflow=ov, pumps=pumps, fluid=fluid, sim=sim,
                   criteria=criteria, raw=raw)
    _check_pumps(proj)
    return proj


def _check_pumps(p: Project):
    t = p.tank
    for pump in p.pumps:
        for k, nz in enumerate(pump.nozzles):
            x, y, z = nz.position
            if not (0 < x < t.length and 0 < y < t.width and t.sand_depth < z < t.water_surface):
                which = f" nozzle {k + 1}" if len(pump.nozzles) > 1 else ""
                raise ValueError(f"pump {pump.name!r}{which}: outlet position {np.round(nz.position, 4).tolist()} m "
                                 "is not inside the water volume")


def load(path) -> Project:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return parse(raw, path)


def dump(raw: dict, path):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True)
