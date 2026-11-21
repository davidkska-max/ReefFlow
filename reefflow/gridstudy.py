"""Verification test 6: grid-convergence study on the reference tank.

The demo tank is run at Preview (10 mm), Standard (5 mm) and Fine (3 mm), on
each lattice, with an IDENTICAL physical model on every grid: equivalent pump
outlets use one fixed minimum size (4 Preview cells = 40 mm) instead of
4 cells of each grid, and the same seed, run times and settings.

Reported for each key metric (metrics.py):
* the value on each grid with its statistical uncertainty (batch means over the
  averaging window, 1 sigma);
* % change Preview -> Standard and Standard -> Fine;
* Richardson extrapolation and the fine-grid convergence index GCI_fine (Roache
  1998; procedure of Celik et al. 2008, J. Fluids Eng. 130:078001, which allows
  the unequal refinement ratios 2 and 5/3 used here), with the observed order.
  Where the changes are not monotonic or are smaller than the statistical
  noise, the GCI is flagged as not meaningful rather than reported.

Pre-declared reading (no hard pass/fail: the spec asks for the % change): a
metric is 'grid-converged for practical use' if Standard -> Fine changes it by
less than 10 % (or by less than 2 sigma of statistical noise).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import metrics as MET
from . import project as PR
from . import tank as TK
from .run import run as run_tank

PRESET_ORDER = ["preview", "standard", "fine"]


def celik(f1, f2, f3, h1, h2, h3, Fs=1.25):
    """Celik et al. (2008). 1 = fine. Returns dict (p, f_ext, gci_fine, kind)."""
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return {"kind": "no change", "p": None, "f_ext": f1, "gci_fine": 0.0}
    s = math.copysign(1.0, e32 / e21)
    p = 2.0
    ok = True
    for _ in range(200):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
            p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        except (ValueError, ZeroDivisionError):
            ok = False
            break
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    kind = "monotonic" if s > 0 else "oscillatory"
    if not ok or not np.isfinite(p):
        return {"kind": kind, "p": None, "f_ext": None, "gci_fine": None}
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea21 = abs((f1 - f2) / f1) if f1 != 0 else float("inf")
    gci = Fs * ea21 / (r21 ** p - 1)
    return {"kind": kind, "p": p, "f_ext": f_ext, "gci_fine": gci}


def _profiles(model, speed):
    """Mean speed along three lines (physical coordinates, NaN in solids)."""
    fl = model.fluid_mask
    sp = np.where(fl, speed, np.nan)
    X, Y, Z = (model.node_coords(a) for a in range(3))
    nx, ny, nz = model.shape
    j = ny // 2
    k = 1 + (nz - 3) // 2
    i = nx // 2
    return {"x_line": (X[1:-1], sp[1:-1, j, k]),          # along the length, mid-width, mid-depth
            "z_line": (Z[1:-1], sp[i, j, 1:-1]),          # vertical, tank centre
            "y_line": (Y[1:-1], sp[i, 1:-1, k])}          # across the width, mid-length, mid-depth


def grid_study(project_file, lattices=("D3Q19", "D3Q27"), presets=PRESET_ORDER,
               out="verification/results/06_grid", progress=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = PR.load(project_file)
    presets = [p for p in PRESET_ORDER if p in presets]
    dx_coarse = max(PR.PRESETS[p] for p in presets)
    smin = base.sim["min_source_cells"] * dx_coarse
    results = {}
    for lat in lattices:
        for pre in presets:
            tag = f"{lat}_{pre}"
            rdir = out / tag
            rj = rdir / "run.json"
            if rj.exists():
                progress(f"[grid] {tag}: using existing result {rj}")
                results[tag] = json.loads(rj.read_text())
                continue
            P = base.with_overrides(lattice=lat, preset=pre, dx=None, source_min_size=f"{smin * 1000:g} mm")
            model = TK.build(P, P.dx())
            progress(f"[grid] {tag}: starting")
            r = run_tank(model, rdir, progress=progress)
            r.pop("_fields", None)
            results[tag] = r
    rep = analyse(out, lattices, presets)
    progress((out / "grid_study.md").read_text(encoding="utf-8"))
    return rep


def _recompute_metrics(out, res):
    """Recompute every metric from the saved time-averaged fields with the current
    definitions (metrics.py), so all grids are compared on identical definitions even
    if a definition changed after a run. Statistical uncertainty comes from the run's
    batch means; where the run used a different definition, its relative uncertainty
    is applied to the recomputed value (flagged in 'uncertainty_scaled')."""
    base = None
    for (lat, pre), r in res.items():
        f = out / f"{lat}_{pre}" / "fields.npz"
        if not f.exists():
            continue
        if base is None:
            base = PR.load(r["project_file"])
        P = base.with_overrides(preset=pre, dx=None)
        model = TK.build(P, P.dx())
        data = np.load(f)
        new = MET.compute(model, data["mean_u"].astype(np.float64), data["mean_speed"].astype(np.float64),
                          data["tke"].astype(np.float64), r["metrics"]["dead_speed_threshold"])
        old, unc = r["metrics"], r["metrics_stat_uncertainty"]
        scaled = []
        for k, v in new.items():
            if k in unc and old.get(k) not in (None, 0) and abs(old[k] - v) > 1e-9 * max(abs(v), 1e-12):
                unc[k] = unc[k] / abs(old[k]) * abs(v)
                scaled.append(k)
        r["metrics_as_run"] = old
        r["metrics"] = new
        r["uncertainty_scaled"] = scaled


def analyse(out, lattices, presets):
    out = Path(out)
    res = {}
    for lat in lattices:
        for pre in presets:
            p = out / f"{lat}_{pre}" / "run.json"
            if p.exists():
                res[(lat, pre)] = json.loads(p.read_text())
    _recompute_metrics(out, res)
    table = {}
    for lat in lattices:
        rs = [res.get((lat, p)) for p in presets]
        if any(r is None for r in rs):
            continue
        h = [r["dx_m"] for r in rs]
        for key in MET.KEY_METRICS:
            v = [r["metrics"][key] for r in rs]
            u = [r["metrics_stat_uncertainty"].get(key, float("nan")) for r in rs]
            row = {"values": v, "stat_unc": u, "dx": h}
            if len(v) == 3:
                f3, f2, f1 = v
                row["change_coarse_mid_pct"] = 100 * (f2 / f3 - 1) if f3 else None
                row["change_mid_fine_pct"] = 100 * (f1 / f2 - 1) if f2 else None
                c = celik(f1, f2, f3, h[2], h[1], h[0])
                noise = 2 * math.hypot(u[1], u[2])
                c["below_noise"] = bool(abs(f1 - f2) < noise)
                row["celik"] = c
                row["converged_practical"] = bool(abs(row["change_mid_fine_pct"] or 0) < 10 or c["below_noise"])
            table[f"{lat}:{key}"] = row
    rep = {"runs": {f"{k[0]}_{k[1]}": {"dx_m": r["dx_m"], "wall_time_s": r["wall_time_s"], "mlups": r["mlups"],
                                       "steps": r["steps"], "mass_rel_drift": r["mass_rel_drift"],
                                       "u_max_overall_lattice": r["u_max_overall_lattice"],
                                       "ke_avg_first_half": r["ke_avg_first_half"],
                                       "ke_avg_second_half": r["ke_avg_second_half"]}
                    for k, r in res.items()},
           "table": table}
    (out / "grid_study.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    _plots(out, lattices, presets, res, table)
    _markdown(out, lattices, presets, res, table)
    return rep


def _plots(out, lattices, presets, res, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(MET.KEY_METRICS)
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
    for ax, key in zip(axes.flat, keys):
        lab, unit, sc = MET.KEY_METRICS[key]
        for lat, mk in zip(lattices, ("o-", "s--")):
            row = table.get(f"{lat}:{key}")
            if not row:
                continue
            dxmm = np.array(row["dx"]) * 1e3
            ax.errorbar(dxmm, np.array(row["values"]) * sc, yerr=2 * np.array(row["stat_unc"]) * sc,
                        fmt=mk, capsize=3, label=lat)
            c = row.get("celik", {})
            if c.get("f_ext") is not None and c.get("kind") == "monotonic" and not c.get("below_noise"):
                ax.plot([0], [c["f_ext"] * sc], "x", color=ax.lines[-1].get_color())
        ax.set_xlim(left=-0.5)
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("dx [mm]", fontsize=8)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Grid study: metric vs cell size (error bars: 2 sigma statistical; x at dx = 0: Richardson "
                 "extrapolation where monotonic)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "grid_metrics.png", dpi=110)
    plt.close(fig)

    # profiles
    base = None
    fig, axes = plt.subplots(len(lattices), 3, figsize=(17, 3.6 * len(lattices)), squeeze=False)
    for li, lat in enumerate(lattices):
        for pre in presets:
            f = out / f"{lat}_{pre}" / "fields.npz"
            if not f.exists():
                continue
            if base is None:
                rj = res[(lat, pre)]
                base = PR.load(rj["project_file"])
            P = base.with_overrides(preset=pre, dx=None)
            model = TK.build(P, P.dx())
            data = np.load(f)
            pr = _profiles(model, data["mean_speed"])
            for ax, (name, (c, v)) in zip(axes[li], pr.items()):
                ax.plot(c * 100, v * 100, label=f"{pre} ({model.dx * 1e3:g} mm)", lw=1.2)
                ax.set_title(f"{lat}: {name.replace('_', ' ')}", fontsize=9)
                ax.set_xlabel({"x_line": "x [cm] (mid-width, mid-depth)", "z_line": "z [cm] (tank centre)",
                               "y_line": "y [cm] (mid-length, mid-depth)"}[name], fontsize=8)
                ax.set_ylabel("<|u|> [cm/s]", fontsize=8)
        axes[li][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "grid_profiles.png", dpi=110)
    plt.close(fig)


def _markdown(out, lattices, presets, res, table):
    L = ["# Verification 6: grid-convergence study (demo 75-gal tank)", ""]
    L += ["| run | dx | steps | wall time | MLUPS | mass drift | max lattice speed | KE 1st/2nd half of averaging |",
          "|---|---|---|---|---|---|---|---|"]
    for (lat, pre), r in res.items():
        L.append(f"| {lat} {pre} | {r['dx_m'] * 1e3:g} mm | {r['steps']} | {r['wall_time_s'] / 60:.1f} min | "
                 f"{r['mlups']:.0f} | {r['mass_rel_drift']:.1e} | {r['u_max_overall_lattice']:.3f} | "
                 f"{r['ke_avg_first_half'] * 1e4:.1f} / {r['ke_avg_second_half'] * 1e4:.1f} cm²/s² |")
    for lat in lattices:
        rows = [(k.split(":")[1], v) for k, v in table.items() if k.startswith(lat + ":")]
        if not rows:
            continue
        L += ["", f"### {lat}", "",
              "| metric | Preview 10 mm | Standard 5 mm | Fine 3 mm | Δ P→S | Δ S→F | order p | GCI_fine | reading |",
              "|---|---|---|---|---|---|---|---|---|"]
        for key, row in rows:
            lab, unit, sc = MET.KEY_METRICS[key]
            vals = " | ".join(f"{v * sc:.2f} ± {2 * u * sc:.2f}" for v, u in zip(row["values"], row["stat_unc"]))
            c = row.get("celik", {})
            if c.get("below_noise"):
                gci, pp = "S→F change < 2σ noise", "—"
            elif c.get("gci_fine") is None:
                gci, pp = f"n/a ({c.get('kind')})", "—"
            else:
                gci, pp = f"{100 * c['gci_fine']:.1f} %", f"{c['p']:.2f} ({c['kind']})"
            L.append(f"| {lab} [{unit}] | {vals} | {row.get('change_coarse_mid_pct', float('nan')):+.1f} % | "
                     f"{row.get('change_mid_fine_pct', float('nan')):+.1f} % | {pp} | {gci} | "
                     f"{'converged' if row.get('converged_practical') else 'NOT converged'} |")
    L += ["", "Values are ± 2σ statistical (batch means). 'converged' = Standard→Fine change < 10 % or "
          "below 2σ statistical noise (pre-declared in gridstudy.py).", "",
          "![metrics](grid_metrics.png)", "", "![profiles](grid_profiles.png)", ""]
    (out / "grid_study.md").write_text("\n".join(L), encoding="utf-8")
