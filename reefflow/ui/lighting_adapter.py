"""The ONE place where the UI touches the Phase-4 lighting schema and API.

Phase 4 (``reefflow.light``) defines the project-file ``lighting:`` section (see
``reefflow/light/config.py``). Fixture mounting fields used by the UI:

    lighting:
      fixtures:
        - name, archetype           # archetype from reefflow/light/data/fixture_library.yaml
          position: [x, y]          # fixture centre over the tank (tank coordinates)
          height                    # emitting face above the water surface
          tilt_x, tilt_y, rotate_z  # degrees (tilt_x + aims toward the back glass, tilt_y + toward +x)
          intensity                 # master %
          ...                       # any other key is preserved untouched ('extra')
      ...                           # photoperiod, water, glass, ... preserved untouched

Fixture rotation (Phase 4 ``build_fixture``): R = Ry(-tilt_y) Rx(tilt_x) Rz(rotate_z), beam
axis R (0, 0, -1). The browser reproduces this for the beam-cone display.

PAR results: the worker can run the Phase-4 tracer on the CFD grid after a flow run
(``compute_par``) and stores ``par.npz`` (keys ``meter``, ``normal``, ``label``) in the
run directory; ``analyze(run_dir, par=meter)`` then includes the light sub-score.
The results viewer shows a PAR slice layer whenever ``par.npz`` exists (``load_par``).

If ``reefflow.light`` cannot be imported, everything here degrades to a minimal
schema {fixtures: [{name, archetype, position, height, tilt_x, tilt_y, intensity}]}
with basic checks, and the PAR layer stays unavailable.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import yaml

from .. import units as U

_LIB_FILE = Path(__file__).resolve().parents[1] / "light" / "data" / "fixture_library.yaml"
_PROD_FILE = _LIB_FILE.with_name("fixture_products.yaml")
MOUNT_KEYS = ("name", "archetype", "position", "height", "tilt_x", "tilt_y", "rotate_z", "intensity")
PAR_FILE = "par.npz"


def _num(v, default=0.0) -> float:
    """'90 deg' / '2.0 umol/J' / 90 -> 90.0 (leading number of a quantity string)."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).split()[0])
    except (ValueError, IndexError):
        return default


def _len(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        return U.parse(v, "length")
    except ValueError:
        return default


def available() -> bool:
    """True when the Phase-4 package is importable (imports Taichi, does not initialise it)."""
    try:
        import reefflow.light.config  # noqa: F401
        return True
    except Exception:
        return False


def fixture_library() -> dict:
    """Archetypes for the placement UI: {name: {description, footprint_m, beam_angle_deg, ppf_umol_s,
    channels}}. Read straight from the YAML so the server does not need Taichi for it."""
    if not _LIB_FILE.is_file():
        return {"generic_led": {"description": "Generic LED fixture (Phase-4 library not installed)",
                                "footprint_m": [0.2, 0.2], "beam_angle_deg": 90.0, "ppf_umol_s": None,
                                "channels": []}}
    lib = yaml.safe_load(_LIB_FILE.read_text(encoding="utf-8")) or {}
    arches = dict(lib.get("archetypes") or {})
    if _PROD_FILE.is_file():
        for k, v in (yaml.safe_load(_PROD_FILE.read_text(encoding="utf-8")) or {}).items():
            arches[k] = {**v, "product": True}
    out = {}
    for name, a in arches.items():
        fp = a.get("footprint") or {}
        dist = a.get("distribution") or {}
        beam = None
        if dist.get("type") == "beam_angle":
            beam = _num(dist.get("beam_angle"))
        elif dist.get("type") in (None, "lambertian"):
            beam = 120.0                                  # FWHM of a Lambertian emitter
        if a.get("ppf") is not None:
            ppf = _num(a["ppf"])
        elif a.get("power") is not None and a.get("efficacy") is not None:
            ppf = _num(a["power"]) * _num(a["efficacy"])
        else:
            ppf = None
        out[name] = {"description": " ".join(str(a.get("description", "")).split()),
                     "footprint_m": [_len(fp.get("length")), _len(fp.get("width"))],
                     "beam_angle_deg": beam, "distribution": dist.get("type", "lambertian"),
                     "ppf_umol_s": ppf, "channels": sorted((a.get("channels") or {}).keys()),
                     "verification_only": name in ("isotropic_point", "lambertian_panel", "collimated_beam")}
        if a.get("product"):
            out[name].update(_product_meta(a))
            out[name]["power_w"] = _num(a.get("power")) if a.get("power") is not None else None
            out[name]["ppf_basis"] = a.get("ppf_basis")
            out[name]["par_data"] = a.get("par_data") or []
    return out


PRODUCT_META = ("brand", "model", "status", "checked", "sources", "notes")


def _product_meta(a: dict) -> dict:
    """Catalogue fields shown in the UI for a real product (D-844)."""
    return {"product": True, **{k: a.get(k) for k in PRODUCT_META}}


def demo_lighting() -> dict | None:
    """Phase-4 demo lighting for the 75-gallon demo (two LED pucks 8 in above the water)."""
    try:
        from reefflow.light.demo import DEMO_LIGHTING
        return copy.deepcopy(DEMO_LIGHTING)
    except Exception:
        return {"fixtures": [
            {"name": "LED-left", "archetype": "led_puck", "position": ["11.75 in", "8.5 in"], "height": "8 in"},
            {"name": "LED-right", "archetype": "led_puck", "position": ["35.25 in", "8.5 in"], "height": "8 in"}]}


# ----------------------------------------------------------------------
# project raw <-> editor (SI numbers)
# ----------------------------------------------------------------------
def _fix_bool_keys(d):
    """YAML 1.1 reads unquoted ``on:`` / ``off:`` keys as booleans True / False (the Phase-4 demo
    photoperiod has them). Restore the intended string keys so they survive JSON and are
    written back quoted ('on': ...)."""
    if isinstance(d, dict):
        return {("on" if k is True else "off" if k is False else k): _fix_bool_keys(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_fix_bool_keys(v) for v in d]
    return d


def editor_from_raw(section: dict | None) -> dict:
    section = _fix_bool_keys(copy.deepcopy(section or {}))
    fixtures = []
    for i, f in enumerate(section.pop("fixtures", None) or []):
        pos = f.get("position") or [0, 0]
        fixtures.append({
            "name": f.get("name") or f"light-{i + 1}", "archetype": f.get("archetype", "led_puck"),
            "position": [_len(pos[0]), _len(pos[1]) if len(pos) > 1 else 0.0],
            "height": _len(f.get("height"), 0.2),
            "tilt_x": _num(f.get("tilt_x")), "tilt_y": _num(f.get("tilt_y")), "rotate_z": _num(f.get("rotate_z")),
            "intensity": _num(f.get("intensity"), 100.0),
            "extra": {k: v for k, v in f.items() if k not in MOUNT_KEYS},
        })
    return {"fixtures": fixtures, "extra": section}


def raw_from_editor(ed: dict | None, fmt_len) -> dict | None:
    """Editor lighting -> raw section (None when there are no fixtures: Phase 4 requires >= 1)."""
    if not ed or not ed.get("fixtures"):
        return None
    sec = copy.deepcopy(ed.get("extra") or {})
    fx = []
    for f in ed["fixtures"]:
        d = dict(copy.deepcopy(f.get("extra") or {}))
        d.update({"name": f["name"], "archetype": f["archetype"],
                  "position": [fmt_len(f["position"][0]), fmt_len(f["position"][1])],
                  "height": fmt_len(f["height"])})
        for k in ("tilt_x", "tilt_y", "rotate_z"):
            if abs(float(f.get(k) or 0.0)) > 1e-9:
                d[k] = round(float(f[k]), 3)
        if abs(float(f.get("intensity", 100.0)) - 100.0) > 1e-9:
            d["intensity"] = round(float(f["intensity"]), 2)
        fx.append(d)
    out = {"fixtures": fx}
    out.update({k: v for k, v in sec.items() if k != "fixtures"})
    return out


def validate(raw: dict, tank) -> list[str]:
    """Errors in the project's lighting section ([] when fine or absent)."""
    sec = raw.get("lighting")
    if not sec:
        return []
    try:
        from reefflow.light.config import parse_lighting
    except Exception:
        parse_lighting = None
    if parse_lighting is not None:
        try:
            parse_lighting(sec, tank)
            return []
        except Exception as e:                              # noqa: BLE001 (show the user any schema error)
            return [f"lighting: {e}"]
    errs = []
    for f in sec.get("fixtures") or []:
        try:
            if _len(f.get("height"), -1) <= 0:
                errs.append(f"lighting: fixture {f.get('name')!r} needs a height above the water")
        except Exception as e:                              # noqa: BLE001
            errs.append(f"lighting: {e}")
    return errs


def beam_axis(tilt_x_deg: float, tilt_y_deg: float, rotate_z_deg: float = 0.0) -> list:
    """Beam axis R (0,0,-1) with R = Ry(-tilt_y) Rx(tilt_x) Rz(rotate_z) (Phase-4 convention)."""
    def rot(axis, a):
        c, s = math.cos(a), math.sin(a)
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
    R = rot("y", -math.radians(tilt_y_deg)) @ rot("x", math.radians(tilt_x_deg)) @ rot("z", math.radians(rotate_z_deg))
    return (R @ np.array([0.0, 0.0, -1.0])).tolist()


# ----------------------------------------------------------------------
# PAR results (run directory)
# ----------------------------------------------------------------------
def par_available(run_dir) -> bool:
    return (Path(run_dir) / PAR_FILE).is_file()


def load_par(run_dir) -> dict | None:
    """{'meter': (nx,ny,nz) PAR or NaN, 'normal': ..., 'label': str} or None."""
    f = Path(run_dir) / PAR_FILE
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=False)
    return {"meter": z["meter"], "normal": z["normal"] if "normal" in z.files else None,
            "label": str(z["label"]) if "label" in z.files else "PAR"}


def compute_par(project, model, run_dir, progress=print, rays: float | None = None):
    """Trace the project's lighting on the CFD grid of ``model`` (Phase 4), write par.npz and
    return the meter-equivalent PAR array for ``analyze(par=...)``. None when the project has
    no lighting section. Runs in the worker process (needs an initialised Taichi backend)."""
    if not project.raw.get("lighting"):
        return None
    from reefflow.light import par_on_cfd_grid, run_lighting
    progress("[light] tracing PAR on the CFD grid (Phase 4)")
    last = [0.0]

    def _p(done, goal=None, *_):
        import time
        if goal and time.monotonic() - last[0] > 10:
            last[0] = time.monotonic()
            progress(f"[light] {done:.3g} / {goal:.3g} rays ({100 * done / goal:.0f} %)")

    res = run_lighting(project, model=model, rays=int(rays) if rays else None, progress=_p)
    cp = par_on_cfd_grid(model, res)
    np.savez_compressed(Path(run_dir) / PAR_FILE, meter=cp.meter.astype(np.float32),
                        normal=cp.normal.astype(np.float32), label=np.array(cp.label))
    progress(f"[light] PAR done ({cp.label})")
    return cp.meter
