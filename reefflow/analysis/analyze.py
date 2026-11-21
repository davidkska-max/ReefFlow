"""Run every Phase-3 analysis on a finished run and write analysis.json,
analysis.npz, figures and report.html into the run directory."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import metrics as MET
from . import age as AGE
from . import coral as CORAL
from . import figures as FIG
from . import wallshear as WS
from . import zones as ZONES
from .load import load_run


def _criteria(project):
    """Metric thresholds (user-configurable under criteria:; defaults in ASSUMPTIONS)."""
    c = (project.raw.get("criteria") or {})
    return {"dead_zone_speed": project.criteria["dead_zone_speed"],
            "detritus_tau_crit": float(c.get("detritus_shear_Pa", 0.02)),
            "sand_move_tau": float(c.get("sand_move_shear_Pa", 0.5))}


def analyze(run_dir, par=None, progress=print, light=True) -> dict:
    """light=False skips the PAR ray trace even when the project has a lighting section."""
    t0 = time.perf_counter()
    run = load_run(run_dir)
    m = run.model
    P = m.project
    crit = _criteria(P)
    out = {"run": str(run.path), "project": P.name, "dx_m": m.dx, "criteria": crit}
    progress(f"[analyze] {run.path}: metrics")
    out["metrics"] = MET.compute(m, run.mean_u, run.speed, run.tke, crit["dead_zone_speed"])
    out["metrics_stat_uncertainty"] = run.info.get("metrics_stat_uncertainty", {})

    age = None
    if m.overflow_index is not None and run.reynolds is not None:
        progress("[analyze] water age (steady mean-field solve)")
        nut, nres, nsgs = AGE.eddy_viscosity(run.mean_u, run.reynolds, m.fluid_mask, m.dx,
                                             cs=float(P.sim["smagorinsky"] or 0.0))
        a = AGE.solve_age(m, run.mean_u, nut)
        age = a["age"]
        out["age"] = a["report"]
        out["age"]["nu_t_resolved_median"] = float(np.median(nres[m.fluid_mask]))
        out["age"]["nu_t_sgs_median"] = float(np.median(nsgs[m.fluid_mask]))
    else:
        out["age"] = None

    progress("[analyze] dead zones")
    dz = ZONES.dead_zones(run, crit["dead_zone_speed"], age)
    out["dead_zones"] = {k: v for k, v in dz.items() if k != "mask"}
    out["derating_sensitivity"] = ZONES.derating_sensitivity(run, crit["dead_zone_speed"])

    progress("[analyze] wall shear / detritus")
    ws = WS.horizontal_shear(run, rho=P.fluid["density"], tau_crit=crit["detritus_tau_crit"],
                             tau_sand=crit["sand_move_tau"])
    out["wall_shear"] = ws["report"]

    d = Path(run_dir)
    par_arrays = {}
    out["light"] = None
    if par is None and P.raw.get("lighting") and light:
        progress("[analyze] lighting: PAR ray trace on the CFD grid (Phase 4)")
        from ..light import plots as LP
        from ..light.results import par_on_cfd_grid
        from ..light.run import run_lighting
        lres = run_lighting(P, model=m)
        cp = par_on_cfd_grid(m, lres)
        par = cp.meter
        out["light"] = lres.summary()
        out["light"]["label"] = cp.label
        out["light"]["calibrated"] = bool(cp.calibrated)
        for name, fn in (("par_plan.png", lambda p: LP.plan_maps(lres, p)),
                         ("par_section.png", lambda p: LP.section(lres, p)),
                         ("par_surfaces.png", lambda p: LP.surface_maps(lres, p)),
                         ("dli_surfaces.png", lambda p: LP.surface_maps(lres, p, dli=True))):
            try:
                fn(d / name)
            except Exception as e:                            # optional output
                progress(f"[analyze] {name}: {e}")
        par_arrays = {"par_meter": np.nan_to_num(cp.meter, nan=-1).astype(np.float32),
                      "par_normal": np.nan_to_num(cp.normal, nan=-1).astype(np.float32)}

    progress("[analyze] coral scores")
    profiles = CORAL.load_profiles(P)
    sc = CORAL.score(run, profiles, par=par)
    out["coral"] = CORAL.summary(sc)
    out["coral"]["best_spots"] = {name: CORAL.best_spots(m, sc, name) for name in profiles}
    out["coral"]["profiles_used"] = profiles

    np.savez_compressed(d / "analysis.npz", **par_arrays, age=np.nan_to_num(age, nan=-1).astype(np.float32) if age is not None
                        else np.zeros(0), dead=dz["mask"], sand_tau=np.nan_to_num(ws["sand_tau_map"], nan=-1),
                        surf_nodes=sc["nodes"], surf_speed=sc["speed"], surf_ti=sc["ti"],
                        **{f"score_{n}": np.nan_to_num(v["combined"], nan=0) for n, v in sc["profiles"].items()})
    _export_3d(m, d, age, out.get("age"), dz["mask"], progress)
    progress("[analyze] figures")
    figs = FIG.all_figures(run, d, age=age, dead=dz, ws=ws, sc=sc, out=out)
    for name in profiles:
        try:
            CORAL.rock_mesh_with_scores(m, sc, name, d / f"rock_score_{name}.ply")
        except Exception as e:                              # optional output
            progress(f"[analyze] rock mesh {name}: {e}")
    out["figures"] = figs
    out["analysis_time_s"] = time.perf_counter() - t0
    with open(d / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=_json)
    from .report import write_report
    write_report(run, out, d / "report.html")
    progress(f"[analyze] done in {out['analysis_time_s']:.0f} s -> {d / 'report.html'}")
    return out


def _export_3d(m, d, age, age_rep, dead, progress):
    """analysis.vti (ParaView: water age / (V/Q), dead mask) and dead_zones.ply (isosurface)."""
    from ..io_vtk import write_vti
    arrays = {"dead_zone": dead.astype(np.uint8), "tag": m.tags}
    if age is not None:
        arrays["age_over_tau"] = np.nan_to_num(age / age_rep["tau_nominal_s"], nan=0.0).astype(np.float32)
    write_vti(d / "analysis.vti", arrays, m.origin, m.dx)
    if dead.any():
        try:
            import trimesh
            from scipy import ndimage
            from skimage.measure import marching_cubes
            f = ndimage.gaussian_filter(dead.astype(float), 0.6)
            v, fc, _, _ = marching_cubes(f, 0.5)
            trimesh.Trimesh(m.origin + v * m.dx, fc[:, ::-1], process=False).export(str(d / "dead_zones.ply"))
        except Exception as e:                                   # optional output
            progress(f"[analyze] dead-zone isosurface skipped: {e}")


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
