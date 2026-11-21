"""Demo lighting for the 75-gallon reference tank (spec 'Reference / demo case'):
two generic LED fixtures mounted ~8 in above the water.

``DEMO_LIGHTING`` (dict) / ``DEMO_LIGHTING_YAML`` (same content, commented) are meant to be
merged into the demo project as its ``lighting:`` section. ``run_demo()`` traces it on
the demo tank and writes maps and a summary to verification/results/09_light/demo.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

DEMO_LIGHTING_YAML = """\
lighting:
  fixtures:                          # two generic ~90 W LED pucks, 90 deg optics
    - name: LED-left
      archetype: led_puck
      position: [11.75 in, 8.5 in]   # quarter points of the 47 in internal length, mid-width
      height: 8 in                   # emitting face above the water surface (spec demo)
      channels: {white: 80}          # % per channel; the others stay at 100 %
    - name: LED-right
      archetype: led_puck
      position: [35.25 in, 8.5 in]
      height: 8 in
      channels: {white: 80}
  photoperiod: {"on": "10:00", "off": "20:00", ramp: 60 min, peak: 100}   # 9 h full-intensity equivalent
  water: {kd: 0.08 1/m, n: 1.34}     # clear reef water (approximation, PLAN C8)
  surface: {roughness: 0.0}          # flat surface (spec default)
  glass:
    n: 1.5
    walls: {front: clear, back: black, left: clear, right: clear}
  albedo: {sand: 0.6, rock: 0.15}    # aragonite sand / coralline-covered live rock (starting values)
  bounces: true
  rock_geometry: auto                # STL when available (it is: rock.stl)
  rays: 2.0e7                        # minimum photons ...
  target_noise: 0.02                 # ... then trace until an average-lit 5 mm patch has 2 % noise
  map_spacing: 75 mm                 # plan maps every 75 mm (~3 in) below the surface
  map_resolution: 5 mm
"""

DEMO_LIGHTING = yaml.safe_load(DEMO_LIGHTING_YAML)["lighting"]


def demo_project(path=None):
    """The demo project (examples/demo_75gal) with DEMO_LIGHTING merged in (unless it already
    has a lighting section)."""
    from .. import project as PR
    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path else root / "examples" / "demo_75gal" / "demo_75gal.yaml"
    proj = PR.load(p)
    if "lighting" not in proj.raw:
        raw = copy.deepcopy(proj.raw)
        raw["lighting"] = copy.deepcopy(DEMO_LIGHTING)
        proj = PR.parse(raw, p)
    return proj


def run_demo(rays: float | None = None, dx: float | None = None, out: str | None = None, arch: str | None = None):
    """Trace the demo tank, save plots + summary JSON. Returns (result, summary)."""
    from .. import backend
    from ..verification.common import outdir, save_json
    from . import plots
    from .analysis import hotspots_and_shadows, uniformity
    from .run import run_lighting
    if backend.current() is None:
        backend.init(arch or "auto")
    proj = demo_project()
    res = run_lighting(proj, dx=dx, rays=rays)
    name = out or "09_light/demo"
    d = outdir(name)
    plots.plan_maps(res, d / "plan_par.png")
    plots.section(res, d / "front_section.png")
    plots.surface_maps(res, d / "rock_sand_par.png")
    plots.surface_maps(res, d / "rock_sand_dli.png", dli=True)
    plots.rock_3d(res, d / "rock_3d_par.png")
    summ = res.summary()
    summ["depth_bands"] = depth_band_table(res)
    if res.sand is not None:
        summ["sand_hotspots_shadows"] = hotspots_and_shadows(res.sand[1:-1, 1:-1], res.grid.coords(0)[1:-1],
                                                             res.grid.coords(1)[1:-1])
    summ["config"] = res.cfg.summary()
    save_json(name, summ, "summary.json")
    return res, summ


def depth_band_table(res, band: float = 0.075) -> list:
    """Rock-top (upward-facing patches) PAR statistics per depth band below the surface, plus
    the plan-map statistics at each map depth and the sand."""
    import numpy as np
    from .analysis import uniformity
    rows = []
    zs = res.tank.water_surface
    if res.rock_normal is not None:
        top = res.rock_top_mask()
        cz = res.patches["centroid"][..., 2]
        depth = zs - cz
        nb = int(np.ceil(res.tank.water_column / band))
        for b in range(nb):
            m = top & (depth >= b * band) & (depth < (b + 1) * band)
            if m.sum() == 0:
                continue
            rows.append({"what": "rock tops", "depth_from_m": b * band, "depth_to_m": (b + 1) * band,
                         "meter": uniformity(np.where(m, res.rock_meter, np.nan)),
                         "normal": uniformity(np.where(m, res.rock_normal, np.nan)),
                         "area_m2": float(res.patches["area"][m].sum())})
    if res.plan is not None:
        for l, d in enumerate(res.cfg.map_depths):
            rows.append({"what": "plan map", "depth_m": d, "meter": uniformity(res.plan[l])})
    if res.sand is not None:
        rows.append({"what": "sand", "depth_m": res.tank.water_column, "meter": uniformity(res.sand)})
    return rows
