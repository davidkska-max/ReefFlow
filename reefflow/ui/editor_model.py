"""Project raw dict (YAML) <-> the browser editor's model.

The editor model holds every quantity as a plain SI number (m, m^3/s, s, m/s); the
browser converts to and from the user's display units for showing and typing. When
the model is written back to YAML, quantities are written as "<number> <unit>"
strings in the project's display units (in / GPH / ft for US; mm / L/h / m for SI),
so the file stays readable in the user's own units. Keys the editor does not know
about are carried through unchanged ('extra').

The UI-owned ``designer:`` section (rock elements of the rockscape designer, in
metres / degrees) is ignored by the core parser (DECISIONS draft D-503).
"""
from __future__ import annotations

import copy
import hashlib
import json

from .. import project as PR
from .. import units as U
from ..pumps import load_library
from . import lighting_adapter as LA

UNITS = {"US": {"length": "in", "flow": "GPH", "head": "ft", "velocity": "cm/s"},
         "SI": {"length": "mm", "flow": "L/h", "head": "m", "velocity": "cm/s"}}
PUMP_KEYS = ("name", "archetype", "rated_flow", "derating", "position", "aim", "mode", "turbulence",
             "static_head", "fittings_allowance", "delivered_flow", "nozzles")
NOZZLE_KEYS = ("position", "aim", "diameter", "share")


def fmt(value: float, unit: str, sig: int = 6) -> str:
    v = U.from_si(float(value), unit)
    s = f"{v:.{sig}g}"
    if "e" in s:
        s = f"{v:.6f}".rstrip("0").rstrip(".")
    return f"{s} {unit}"


def _q(v, kind, default=None):
    if v is None:
        return default
    return U.parse(v, kind)


def _vec(v, kind="length"):
    if isinstance(v, dict):
        v = [v["x"], v["y"], v["z"]]
    return [U.parse(c, kind) for c in v]


# ----------------------------------------------------------------------
# raw -> editor
# ----------------------------------------------------------------------
# Tank style (D-841): which edges have a frame. Only changes what the photo calibration
# asks you to click (frame corner or glass corner), not the simulated water volume.
TANK_STYLES = ("rimmed", "rimless_top", "rimless")


def to_editor(raw: dict) -> dict:
    raw = copy.deepcopy(raw or {})
    t = raw.get("tank") or {}
    ext = t.get("external") or {}
    g = _q(t.get("glass_thickness"), "length", 0.0127)
    tank = {"length": _q(ext.get("length"), "length", 1.2192), "width": _q(ext.get("width"), "length", 0.4572),
            "height": _q(ext.get("height"), "length", 0.5334), "glass": g,
            "bottom_glass": _q(t.get("bottom_glass_thickness"), "length", g),
            "water_gap": _q(t.get("water_level_below_rim"), "length", 0.0254),
            "sand_depth": _q(t.get("sand_depth"), "length", 0.0),
            "style": t.get("style") if t.get("style") in TANK_STYLES else None}
    ov = overflow_to_editor(raw.get("overflow"), tank) if raw.get("overflow") else None
    lib = load_library()
    pumps = []
    for i, p in enumerate(raw.get("pumps") or []):
        arch = p.get("archetype", "narrow_jet_propeller")
        a = {**lib.get(arch, {}), **p}
        mode = a.get("mode") or {"type": "constant"}
        m = {"type": mode.get("type", "constant"), "period": _q(mode.get("period"), "time", 10.0),
             "min": float(mode.get("min", 0.3)), "max": float(mode.get("max", 1.0)),
             "duty": float(mode.get("duty", 0.5)), "ramp": float(mode.get("ramp", 0.1)),
             "phase": float(mode.get("phase", 0.0))}
        if p.get("nozzles"):
            nozzles = [{"position": _vec(n.get("position", [0.1, 0.1, 0.2])),
                        "aim": [float(c) for c in (n.get("aim") or [1, 0, 0])],
                        "diameter": _q(n.get("diameter"), "length", None),
                        "share": None if n.get("share") is None else float(n["share"]),
                        "extra": {k: v for k, v in n.items() if k not in NOZZLE_KEYS}}
                       for n in p["nozzles"]]
        else:
            nozzles = [{"position": _vec(p.get("position", [0.1, 0.1, 0.2])),
                        "aim": [float(c) for c in (p.get("aim") or [1, 0, 0])],
                        "diameter": None, "share": None, "extra": {}}]
        pumps.append({
            "name": p.get("name") or f"pump-{i + 1}", "archetype": arch, "kind": a.get("kind", "powerhead"),
            "rated_flow": _q(a.get("rated_flow"), "flow", 0.0),
            "derating": float(a.get("derating", 0.7)),
            "position": nozzles[0]["position"], "aim": nozzles[0]["aim"],
            "nozzles": nozzles,
            "mode": m, "turbulence": float(a.get("turbulence", 0.05)),
            "static_head": _q(a.get("static_head"), "head", 0.0),
            "fittings_allowance": _q(a.get("fittings_allowance"), "head", 0.0),
            "delivered_flow": _q(p.get("delivered_flow"), "flow", None),
            "extra": {k: v for k, v in p.items() if k not in PUMP_KEYS},
        })
    rocks = []
    for r in raw.get("rockscape") or []:
        rocks.append({"file": str(r["file"]), "unit": r.get("unit", "mm"),
                      "translate": _vec(r.get("translate", [0, 0, 0])),
                      "rotate_z": float(r.get("rotate_z", 0.0)), "scale": float(r.get("scale", 1.0))})
    sim = {**PR.SIM_DEFAULTS, **(raw.get("simulation") or {})}
    crit = raw.get("criteria") or {}
    designer = copy.deepcopy(raw.get("designer")) if raw.get("designer") else None
    if designer is None:
        # existing rock meshes become movable 'mesh' elements; the rockscape list already equals them
        els = [{"id": f"m{i + 1}", "type": "mesh", "name": r["file"].split("/")[-1],
                "position": list(r["translate"]), "rotation": [0.0, 0.0, r["rotate_z"]],
                "scale": [r["scale"]] * 3, "params": {"file": r["file"], "unit": r["unit"]}}
               for i, r in enumerate(rocks)]
        designer = {"elements": els, "resolution": 0.004, "rock_density": 1600.0,
                    "rockscape_file": None, "built_hash": elements_hash(els)}
    ed = {
        "name": raw.get("name", "untitled"), "display_units": raw.get("display_units", "US"),
        "tank": tank, "overflow": ov, "pumps": pumps, "rockscape": rocks,
        "lighting": LA.editor_from_raw(raw.get("lighting")),
        "designer": designer,
        "simulation": {"preset": sim.get("preset", "standard"), "lattice": str(sim.get("lattice", "D3Q19")).upper(),
                       "spinup": _q(sim.get("spinup"), "time", 60.0), "average": _q(sim.get("average"), "time", 120.0),
                       "dx": _q(sim.get("dx"), "length", None)},
        "criteria": {"dead_zone_speed": _q(crit.get("dead_zone_speed"), "velocity", 0.025)},
    }
    return ed


# ----------------------------------------------------------------------
# editor -> raw
# ----------------------------------------------------------------------
def from_editor(ed: dict, old_raw: dict | None = None) -> dict:
    raw = copy.deepcopy(old_raw or {})
    du = ed.get("display_units", raw.get("display_units", "US"))
    if du not in UNITS:
        raise ValueError("display_units must be US or SI")
    u = UNITS[du]

    def L(v):
        return fmt(v, u["length"])

    raw["reefflow_project"] = raw.get("reefflow_project", PR.FORMAT_VERSION)
    raw["name"] = ed.get("name") or "untitled"
    raw["display_units"] = du
    t = ed["tank"]
    tank = dict(raw.get("tank") or {})
    tank["external"] = {"length": L(t["length"]), "width": L(t["width"]), "height": L(t["height"])}
    tank["glass_thickness"] = L(t["glass"])
    if abs(t["bottom_glass"] - t["glass"]) > 1e-9:
        tank["bottom_glass_thickness"] = L(t["bottom_glass"])
    else:
        tank.pop("bottom_glass_thickness", None)
    tank["water_level_below_rim"] = L(t["water_gap"])
    tank["sand_depth"] = L(t["sand_depth"])
    if t.get("style") in TANK_STYLES:
        tank["style"] = t["style"]
    else:
        tank.pop("style", None)
    raw["tank"] = tank

    raw["rockscape"] = [
        {"file": r["file"], "unit": r.get("unit", "mm"), "translate": [L(c) for c in r.get("translate", (0, 0, 0))],
         **({"rotate_z": round(float(r["rotate_z"]), 4)} if r.get("rotate_z") else {}),
         **({"scale": float(r["scale"])} if r.get("scale", 1.0) != 1.0 else {})}
        for r in ed.get("rockscape") or []]

    o = ed.get("overflow")
    if o:
        raw["overflow"] = {"wall": o["wall"], "center": L(o["center"]), "width": L(o["width"]),
                           "depth": L(o["depth"]), "crest_below_surface": L(o["crest_below_surface"])}
        if o.get("product"):
            raw["overflow"]["product"] = o["product"]
        if o.get("capacity"):
            raw["overflow"]["capacity"] = fmt(o["capacity"], u["flow"])
    else:
        raw.pop("overflow", None)

    lib = load_library()
    pumps = []
    for p in ed.get("pumps") or []:
        kind = lib.get(p["archetype"], {}).get("kind", p.get("kind", "powerhead"))
        d = dict(copy.deepcopy(p.get("extra") or {}))
        d.update({"name": p["name"], "archetype": p["archetype"], "rated_flow": fmt(p["rated_flow"], u["flow"])})
        if kind == "powerhead":
            d["derating"] = round(float(p.get("derating", 0.7)), 4)
        else:
            d["static_head"] = fmt(p.get("static_head", 0.0), u["head"])
            d["fittings_allowance"] = fmt(p.get("fittings_allowance", 0.0), u["head"])
        if p.get("delivered_flow"):
            d["delivered_flow"] = fmt(p["delivered_flow"], u["flow"])
        nz = p.get("nozzles") or []
        d.pop("nozzles", None)
        d.pop("position", None)
        d.pop("aim", None)
        # one nozzle without overrides is written in the Phase-2 form (old files round-trip unchanged);
        # then the pump's position/aim are authoritative, otherwise the nozzle list is
        simple = len(nz) <= 1 and not any(n.get("diameter") or n.get("share") or n.get("extra") for n in nz)
        if kind != "return" or simple:
            src = p                                          # single outlet: the pump's own position/aim
            d["position"] = [L(c) for c in src["position"]]
            d["aim"] = [round(float(c), 4) for c in src["aim"]]
        else:
            d["nozzles"] = []
            for n in nz:
                e = dict(copy.deepcopy(n.get("extra") or {}))
                e["position"] = [L(c) for c in n["position"]]
                e["aim"] = [round(float(c), 4) for c in n["aim"]]
                if n.get("diameter"):
                    e["diameter"] = L(n["diameter"])
                if n.get("share") is not None:
                    e["share"] = round(float(n["share"]), 4)
                d["nozzles"].append(e)
        m = p.get("mode") or {"type": "constant"}
        if m.get("type", "constant") == "constant":
            d["mode"] = {"type": "constant"}
        else:
            d["mode"] = {"type": m["type"], "period": fmt(m["period"], "s"), "min": float(m["min"]),
                         "max": float(m["max"]), "phase": float(m.get("phase", 0.0))}
            if m["type"] == "pulse":
                d["mode"].update(duty=float(m.get("duty", 0.5)), ramp=float(m.get("ramp", 0.1)))
        d["turbulence"] = float(p.get("turbulence", 0.05))
        pumps.append(d)
    raw["pumps"] = pumps

    lt = LA.raw_from_editor(ed.get("lighting"), L)
    if lt:
        raw["lighting"] = lt
    else:
        raw.pop("lighting", None)

    s = ed.get("simulation") or {}
    sim = dict(raw.get("simulation") or {})
    for k in ("preset", "lattice"):
        if s.get(k):
            sim[k] = s[k]
    for k in ("spinup", "average"):
        if s.get(k) is not None:
            sim[k] = fmt(s[k], "s")
    if s.get("dx"):
        sim["dx"] = fmt(s["dx"], "mm")
    else:
        sim.pop("dx", None)
    raw["simulation"] = sim
    c = ed.get("criteria") or {}
    crit = dict(raw.get("criteria") or {})
    if c.get("dead_zone_speed") is not None:
        crit["dead_zone_speed"] = fmt(c["dead_zone_speed"], "cm/s")
    raw["criteria"] = crit

    dz = ed.get("designer")
    if dz is not None:
        raw["designer"] = clean_designer(dz)
    # keep the conventional key order of the demo file
    order = ["reefflow_project", "name", "display_units", "tank", "rockscape", "overflow", "pumps", "lighting",
             "fluid", "simulation", "criteria", "designer"]
    return {k: raw[k] for k in order if k in raw} | {k: v for k, v in raw.items() if k not in order}


def overflow_to_editor(o: dict, tank: dict) -> dict:
    """Overflow entry (corner or wall form) -> editor dict in the wall form (SI).

    tank: editor tank dict (external sizes + glass), used to place a corner box."""
    L_in = tank["length"] - 2 * tank["glass"]
    W_in = tank["width"] - 2 * tank["glass"]
    crest = _q(o.get("crest_below_surface"), "length", 0.0127)
    w, d = _q(o.get("width"), "length", 0.15), _q(o.get("depth"), "length", 0.15)
    if "wall" in o:
        wall = o["wall"]
        Lw = L_in if wall in ("back", "front") else W_in
        c = _q(o.get("center"), "length", Lw / 2)
    else:
        corner = o.get("corner", "back-right")
        wall = "back" if "back" in corner else "front"         # corner form: width along x, depth along y
        c = L_in - w / 2 if "right" in corner else w / 2
    return {"wall": wall, "center": c, "width": w, "depth": d, "crest_below_surface": crest,
            "product": o.get("product"), "capacity": _q(o.get("capacity"), "flow", None)}


def clean_designer(dz: dict) -> dict:
    els = []
    for e in dz.get("elements") or []:
        els.append({"id": str(e["id"]), "type": e["type"], "name": e.get("name", e["type"]),
                    "position": [round(float(c), 6) for c in e["position"]],
                    "rotation": [round(float(c), 4) for c in e.get("rotation", (0, 0, 0))],
                    "scale": [round(float(c), 5) for c in e.get("scale", (1, 1, 1))],
                    "params": _round(e.get("params") or {})})
    return {"elements": els, "resolution": float(dz.get("resolution", 0.004)),
            "rock_density": float(dz.get("rock_density", 1600.0)),
            "rockscape_file": dz.get("rockscape_file"), "built_hash": dz.get("built_hash")}


def _round(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    return v


def _canon(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return str(v)


def elements_hash(elements: list) -> str:
    """Stable hash of the designer elements (geometry only), to know if the rockscape STL is stale."""
    geo = [{k: e.get(k) for k in ("type", "position", "rotation", "scale", "params")} for e in elements or []]
    return hashlib.sha1(json.dumps(_canon(geo), sort_keys=True).encode()).hexdigest()[:16]


def designer_stale(raw: dict) -> bool:
    dz = raw.get("designer")
    if not dz:
        return False
    return elements_hash(dz.get("elements")) != dz.get("built_hash")
