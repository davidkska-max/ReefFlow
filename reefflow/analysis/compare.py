"""Comparison mode: two runs (e.g. a pump moved or re-aimed) -> difference maps and
metric deltas (spec 'Outputs' item 7; the main design workflow).

Both runs should be analysed first (analysis.json). If the grids differ, run B's
fields are interpolated (trilinear, physical coordinates) onto run A's nodes.
Differences are shown as B - A.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..metrics import KEY_METRICS  # noqa: E402
from ..plots import DISP  # noqa: E402
from .load import load_run  # noqa: E402


def _on_grid_of(a, b, field):
    """Field of run b at the nodes of run a."""
    if a.model.shape == b.model.shape and np.allclose(a.model.origin, b.model.origin):
        return field
    from scipy.interpolate import RegularGridInterpolator
    axes_b = [b.model.node_coords(i) for i in range(3)]
    f = np.where(b.model.fluid_mask[..., None] if field.ndim == 4 else b.model.fluid_mask, field, np.nan)
    it = RegularGridInterpolator(axes_b, f, bounds_error=False, fill_value=np.nan)
    A = np.stack(np.meshgrid(*[a.model.node_coords(i) for i in range(3)], indexing="ij"), axis=-1)
    return it(A.reshape(-1, 3)).reshape(a.model.shape + field.shape[3:])


def compare(run_a, run_b, out_dir=None, label_a="A", label_b="B") -> dict:
    A, B = load_run(run_a), load_run(run_b)
    out = Path(out_dir or Path(run_b) / f"compare_vs_{Path(run_a).name}")
    out.mkdir(parents=True, exist_ok=True)
    ja = json.loads((Path(run_a) / "analysis.json").read_text()) if (Path(run_a) / "analysis.json").exists() else None
    jb = json.loads((Path(run_b) / "analysis.json").read_text()) if (Path(run_b) / "analysis.json").exists() else None
    from ..metrics import compute as metrics_compute       # same (current) definitions for both runs
    thr0 = A.model.project.criteria["dead_zone_speed"]
    ma = metrics_compute(A.model, A.mean_u, A.speed, A.tke, thr0)
    mb = metrics_compute(B.model, B.mean_u, B.speed, B.tke, thr0)
    ua = A.info.get("metrics_stat_uncertainty", {})
    ub = B.info.get("metrics_stat_uncertainty", {})
    rows = []
    for k, (lab, unit, sc) in KEY_METRICS.items():
        noise = 2 * np.hypot(ua.get(k, 0.0), ub.get(k, 0.0))
        d = mb[k] - ma[k]
        rows.append({"metric": k, "label": lab, "unit": unit, "A": ma[k] * sc, "B": mb[k] * sc, "delta": d * sc,
                     "delta_pct": 100 * d / ma[k] if ma[k] else None, "noise_2sigma": noise * sc,
                     "significant": bool(abs(d) > noise) if noise else None})
    extra = {}
    if ja and jb:
        if ja.get("age") and jb.get("age"):
            extra["mean_age_over_tau"] = (ja["age"]["mean_age_over_tau"], jb["age"]["mean_age_over_tau"])
        extra["sand_settling_fraction"] = (ja["wall_shear"]["sand_area_settling_fraction"],
                                           jb["wall_shear"]["sand_area_settling_fraction"])
        for name in ja["coral"]["profiles"]:
            if name in jb["coral"]["profiles"]:
                extra[f"coral_{name}_mean_score"] = (ja["coral"]["profiles"][name]["mean_score"],
                                                    jb["coral"]["profiles"][name]["mean_score"])
    sb = _on_grid_of(A, B, B.speed)
    dsp = np.where(A.model.fluid_mask, sb - A.speed, np.nan)
    thr = A.model.project.criteria["dead_zone_speed"]
    dead_a = A.model.fluid_mask & (A.speed < thr)
    dead_b = A.model.fluid_mask & (np.nan_to_num(sb, nan=1.0) < thr)
    _diff_figure(A, dsp, dead_a, dead_b, out / "difference.png", label_a, label_b)
    res = {"run_a": str(run_a), "run_b": str(run_b), "rows": rows, "extra": extra,
           "dead_gained_fraction": float((dead_b & ~dead_a).sum() / A.model.fluid_mask.sum()),
           "dead_removed_fraction": float((dead_a & ~dead_b).sum() / A.model.fluid_mask.sum()),
           "speed_change_mean_abs_ms": float(np.nanmean(np.abs(dsp)))}
    (out / "comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    _html(res, out, A, B, label_a, label_b)
    return res


def _diff_figure(A, dsp, dead_a, dead_b, path, la, lb):
    m = A.model
    (lu, lf), (vu, vf) = DISP[m.project.display_units]["len"], DISP[m.project.display_units]["vel"]
    X, Y, Z = (m.node_coords(i) * lf for i in range(3))
    nz = m.shape[2]
    v = np.nanpercentile(np.abs(dsp), 99) * vf
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for ax, f in zip(axes.flat[:3], (20, 50, 85)):
        k = 1 + int(round((nz - 3) * f / 100))
        im = ax.imshow(dsp[:, :, k].T * vf, origin="lower", extent=[X[0], X[-1], Y[0], Y[-1]], cmap="RdBu_r",
                       vmin=-v, vmax=v, aspect="equal")
        ax.set_title(f"speed change {lb} - {la}, plan at {f} % depth [{vu}]", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    j = m.shape[1] // 2
    im = axes[1, 1].imshow(dsp[:, j, :].T * vf, origin="lower", extent=[X[0], X[-1], Z[0], Z[-1]], cmap="RdBu_r",
                           vmin=-v, vmax=v, aspect="equal")
    axes[1, 1].set_title("speed change, front elevation mid-width", fontsize=9)
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    gained = (dead_b & ~dead_a).sum(axis=2)
    removed = (dead_a & ~dead_b).sum(axis=2)
    nfl = np.maximum(m.fluid_mask.sum(axis=2), 1)
    for ax, g, t, cmap in ((axes[2, 0], gained, f"dead water GAINED in {lb} (fraction of column)", "Reds"),
                           (axes[2, 1], removed, f"dead water REMOVED in {lb} (fraction of column)", "Greens")):
        im = ax.imshow(np.where(m.fluid_mask.any(axis=2), g / nfl, np.nan).T, origin="lower",
                       extent=[X[0], X[-1], Y[0], Y[-1]], cmap=cmap, vmin=0, vmax=0.5, aspect="equal")
        ax.set_title(t, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def _html(res, out, A, B, la, lb):
    img = base64.b64encode((out / "difference.png").read_bytes()).decode("ascii")
    H = ["<!doctype html><html><head><meta charset='utf-8'><title>ReefFlow comparison</title>"
         "<style>body{font:14px system-ui,Arial;max-width:1100px;margin:auto;padding:16px}"
         "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px 6px}"
         "img{max-width:100%}.sig{font-weight:bold}</style></head><body>",
         f"<h1>Comparison: {html.escape(lb)} vs {html.escape(la)}</h1>",
         f"<p>{la} = {html.escape(res['run_a'])}<br>{lb} = {html.escape(res['run_b'])}</p>",
         "<table><tr><th>Metric</th><th>" + la + "</th><th>" + lb + "</th><th>Change</th><th>2σ noise</th>"
         "<th>Beyond noise?</th></tr>"]
    for r in res["rows"]:
        pct = f" ({r['delta_pct']:+.1f} %)" if r["delta_pct"] is not None else ""
        sig = "yes" if r["significant"] else ("no" if r["significant"] is not None else "")
        H.append(f"<tr><td>{r['label']} [{r['unit']}]</td><td>{r['A']:.2f}</td><td>{r['B']:.2f}</td>"
                 f"<td class='{'sig' if r['significant'] else ''}'>{r['delta']:+.2f}{pct}</td>"
                 f"<td>{r['noise_2sigma']:.2f}</td><td>{sig}</td></tr>")
    for k, (a, b) in res["extra"].items():
        H.append(f"<tr><td>{k}</td><td>{a:.3f}</td><td>{b:.3f}</td><td>{b - a:+.3f}</td><td></td><td></td></tr>")
    H.append(f"</table><p>Dead water gained: {100 * res['dead_gained_fraction']:.1f} % of the volume; removed: "
             f"{100 * res['dead_removed_fraction']:.1f} %.</p><img src='data:image/png;base64,{img}'></body></html>")
    (out / "comparison.html").write_text("\n".join(H), encoding="utf-8")
