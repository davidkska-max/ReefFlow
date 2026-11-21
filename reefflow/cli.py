"""Command-line interface.

    reefflow ui [--workspace DIR]              start the browser interface
    reefflow demo  DIR                         write the 75-gallon demo project
    reefflow info  PROJECT [--preset P]        build the voxel model, print pumps and estimate
    reefflow run   PROJECT [--preset P] [--lattice L] [--out DIR] [--spinup S] [--average S]
    reefflow analyze RUN_DIR                    Phase-3 metrics, figures and report.html
    reefflow compare RUN_A RUN_B                difference maps and metric deltas
    reefflow light PROJECT [--rays N]           PAR / DLI ray trace of the project's lighting
    reefflow grid-study PROJECT [--lattices D3Q19 D3Q27] [--presets preview standard fine]

(``python -m reefflow ...`` works the same.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt_pump(pi: dict) -> str:
    return (f"  {pi['name']:<10} {pi['archetype']:<22} rated {pi['rated_flow_GPH']:7.0f} GPH -> delivered "
            f"{pi['delivered_flow_GPH']:7.0f} GPH ({pi['delivered_flow_Lh']:6.0f} L/h; {pi['flow_basis']})\n"
            f"  {'':<10} outlet {pi['outlet']}, U = {pi['nozzle_velocity_ms']:.2f} m/s, "
            f"J = rho Q^2/A = {pi['momentum_flux_N']:.3f} N; modelled as {pi['outlet_model']} "
            f"({pi['source_nodes']} nodes, U_eq = {pi['equivalent_velocity_ms']:.2f} m/s), mode {pi['mode']}")


def cmd_demo(a):
    from .demo import write_demo
    p = write_demo(a.dir)
    print(f"wrote {p} and {p.parent / 'rock.stl'}")


def _model(a):
    from . import project, tank
    P = project.load(a.project)
    over = {k: getattr(a, k, None) for k in ("lattice",)}
    if getattr(a, "preset", None):
        over["preset"] = a.preset
        over["dx"] = None
    P = P.with_overrides(**over)
    return P, tank.build(P, P.dx())


def cmd_info(a):
    from . import backend
    from .run import estimate
    backend.init(a.arch)
    P, m = _model(a)
    t = P.tank.summary()
    print(f"{P.name}: internal {t['internal_in'][0]:.2f} x {t['internal_in'][1]:.2f} x "
          f"{t['internal_in'][2]:.2f} in, water above sand {t['water_volume_above_sand_gal']:.1f} gal (gross)")
    i = m.info
    print(f"grid {i['shape']} at dx = {m.dx * 1e3:g} mm; water in model {i['water_volume_model_gal']:.1f} gal; "
          f"rock {i['rock_volume_m3'] * 1e3:.1f} L above the sand")
    for pi in i["pumps"]:
        print(_fmt_pump(pi))
    e = estimate(m)
    print(f"estimate: {e['steps']} steps of {e['dt_s'] * 1e3:.3f} ms, GPU memory {e['gpu_memory_GB']:.2f} GB, "
          f"host {e['host_memory_GB']:.2f} GB, ~{e['runtime_s_est'] / 60:.0f} min on {e['backend']}")
    for n in m.notes:
        print("note:", n)
    if a.plot:
        from .plots import geometry_figure
        geometry_figure(m, a.plot)
        print("geometry plot:", a.plot)


def cmd_run(a):
    from . import backend
    from .run import run
    backend.init(a.arch)
    P, m = _model(a)
    out = a.out or (Path("runs") / f"{Path(a.project).stem}_{P.sim['preset']}_{P.sim['lattice']}")
    r = run(m, out, spinup=a.spinup, average=a.average)
    met, unc = r["metrics"], r["metrics_stat_uncertainty"]
    print(f"done in {r['wall_time_s'] / 60:.1f} min ({r['mlups']:.0f} MLUPS); results in {out}")
    for k in ("mean_speed", "median_speed", "dead_fraction", "mean_sigma", "rock_top_speed"):
        print(f"  {k:<16} {met[k]:.4g} +/- {unc.get(k, float('nan')):.2g}")
    if not a.no_analyze:
        from .analysis.analyze import analyze
        analyze(out)


def cmd_analyze(a):
    from .analysis.analyze import analyze
    analyze(a.run_dir)


def cmd_compare(a):
    from .analysis.compare import compare
    r = compare(a.run_a, a.run_b, a.out, a.label_a, a.label_b)
    for row in r["rows"]:
        sig = "" if row["significant"] is None else ("  *" if row["significant"] else "")
        print(f"  {row['label']:<52} {row['A']:8.2f} -> {row['B']:8.2f} {row['unit']:<5} ({row['delta']:+.2f}){sig}")
    print("(* = change larger than 2 sigma statistical noise)")


def cmd_light(a):
    """PAR ray trace of the project's lighting: plan maps, rock/sand PAR and DLI, summary.json."""
    import json
    from . import backend
    from .light import plots as LP
    from .light.run import run_lighting
    backend.init(a.arch)
    from . import project
    P = project.load(a.project)
    if not P.raw.get("lighting"):
        raise SystemExit("the project has no lighting: section")
    out = Path(a.out or (Path("runs") / f"{Path(a.project).stem}_light"))
    out.mkdir(parents=True, exist_ok=True)
    res = run_lighting(P, rays=a.rays)
    LP.plan_maps(res, out / "par_plan.png")
    LP.section(res, out / "par_section.png")
    LP.surface_maps(res, out / "par_surfaces.png")
    LP.surface_maps(res, out / "dli_surfaces.png", dli=True)
    summ = res.summary()
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    print(f"{summ['label']}: {summ['rays']:.3g} photons in {summ['seconds']:.0f} s; results in {out}")
    for key in ("rock_tops_meter", "sand"):
        if key in summ:
            u = summ[key]
            print(f"  {key:<16} PAR min {u['min']:.0f}  mean {u['mean']:.0f}  max {u['max']:.0f}  CV {u['cv']:.2f}")


def cmd_ui(a):
    from .ui.launcher import main as ui_main
    argv = []
    if a.workspace:
        argv += ["--workspace", a.workspace]
    if a.port:
        argv += ["--port", str(a.port)]
    if a.no_browser:
        argv += ["--no-browser"]
    ui_main(argv)


def cmd_grid(a):
    from . import backend
    from .gridstudy import grid_study
    backend.init(a.arch)
    grid_study(a.project, lattices=a.lattices, presets=a.presets, out=a.out)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="reefflow", description="ReefFlow reef-tank CFD")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arch", default="auto", help="auto | cuda | vulkan | cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demo"); s.add_argument("dir"); s.set_defaults(fn=cmd_demo)
    for name, fn in (("info", cmd_info), ("run", cmd_run)):
        s = sub.add_parser(name, parents=[common])
        s.add_argument("project")
        s.add_argument("--preset", choices=["preview", "standard", "fine"])
        s.add_argument("--lattice", choices=["D3Q19", "D3Q27"])
        if name == "info":
            s.add_argument("--plot", help="write a geometry PNG here")
        else:
            s.add_argument("--out")
            s.add_argument("--spinup", type=float, help="seconds (overrides the project)")
            s.add_argument("--average", type=float, help="seconds (overrides the project)")
            s.add_argument("--no-analyze", action="store_true", help="skip the Phase-3 analysis and report")
        s.set_defaults(fn=fn)
    s = sub.add_parser("analyze", help="dead zones, water age, sand shear, coral scores, report.html")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("compare", help="difference maps and metric deltas between two analysed runs")
    s.add_argument("run_a"); s.add_argument("run_b")
    s.add_argument("--out"); s.add_argument("--label-a", default="A"); s.add_argument("--label-b", default="B")
    s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("ui", help="start the browser interface (local only)")
    s.add_argument("--workspace"); s.add_argument("--port", type=int); s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_ui)
    s = sub.add_parser("light", parents=[common], help="PAR / DLI ray trace of the project's lighting")
    s.add_argument("project"); s.add_argument("--out"); s.add_argument("--rays", type=float)
    s.set_defaults(fn=cmd_light)
    s = sub.add_parser("grid-study", parents=[common])
    s.add_argument("project")
    s.add_argument("--lattices", nargs="+", default=["D3Q19", "D3Q27"])
    s.add_argument("--presets", nargs="+", default=["preview", "standard", "fine"])
    s.add_argument("--out", default="verification/results/06_grid")
    s.set_defaults(fn=cmd_grid)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
