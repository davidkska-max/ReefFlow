"""Phase-3 figures: water age, dead zones, sand shear / detritus, coral scores."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..plots import DISP  # noqa: E402
from ..tank import TAGS  # noqa: E402


def _u(model):
    d = DISP[model.project.display_units]
    return d["len"], d["vel"]


def _extent_plan(model, lf):
    X, Y = model.node_coords(0) * lf, model.node_coords(1) * lf
    return [X[0], X[-1], Y[0], Y[-1]]


def _rock_outline(ax, model, lf):
    rock = (model.tags == TAGS["rock"]).any(axis=2).astype(float)
    X, Y = model.node_coords(0) * lf, model.node_coords(1) * lf
    ax.contour(X, Y, rock.T, levels=[0.5], colors="k", linewidths=0.6)


def age_figure(run, age, path, tau):
    m = run.model
    (lu, lf), _ = _u(m)
    nz = m.shape[2]
    a = age / tau
    vmin, vmax = np.nanpercentile(a, 1), np.nanpercentile(a, 99.5)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for ax, f in zip(axes.flat[:3], (20, 50, 85)):
        k = 1 + int(round((nz - 3) * f / 100))
        im = ax.imshow(a[:, :, k].T, origin="lower", extent=_extent_plan(m, lf), vmin=vmin, vmax=vmax,
                       cmap="magma", aspect="equal")
        ax.set_title(f"plan, {f} % of depth", fontsize=9)
    j = m.shape[1] // 2
    X, Z = m.node_coords(0) * lf, m.node_coords(2) * lf
    im = axes[1, 1].imshow(a[:, j, :].T, origin="lower", extent=[X[0], X[-1], Z[0], Z[-1]], vmin=vmin, vmax=vmax,
                           cmap="magma", aspect="equal")
    axes[1, 1].set_title("front elevation, mid-width", fontsize=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.7)
    cb.set_label("water age / (V/Q)  (1 = perfectly mixed)")
    fig.suptitle(f"Mean water age, V/Q = {tau / 60:.1f} min (return-pump turnover)", fontsize=11)
    fig.savefig(path, dpi=105)
    plt.close(fig)


def dead_figure(run, dz, path):
    m = run.model
    (lu, lf), (vu, vf) = _u(m)
    dead = dz["mask"]
    fl = m.fluid_mask
    frac = dead.sum(axis=2) / np.maximum(fl.sum(axis=2), 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    im = ax[0].imshow(np.where(fl.any(axis=2), frac, np.nan).T, origin="lower", extent=_extent_plan(m, lf),
                      cmap="Reds", vmin=0, vmax=1, aspect="equal")
    _rock_outline(ax[0], m, lf)
    for n, z in enumerate(dz["zones"][:5]):
        c = np.array(z["core_m"]) * lf
        ax[0].plot(c[0], c[1], "o", mfc="none", mec="b")
        ax[0].annotate(str(n + 1), (c[0], c[1]), color="b", fontsize=8, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(im, ax=ax[0], shrink=0.75, label="fraction of water column dead")
    ax[0].set_title(f"Dead water (<|u|> < {dz['threshold_ms'] * vf:.1f} {vu}): plan projection; "
                    "blue = zone cores (ranked)", fontsize=9)
    side = dead.sum(axis=1) / np.maximum(fl.sum(axis=1), 1)
    X, Z = m.node_coords(0) * lf, m.node_coords(2) * lf
    im = ax[1].imshow(np.where(fl.any(axis=1), side, np.nan).T, origin="lower", extent=[X[0], X[-1], Z[0], Z[-1]],
                      cmap="Reds", vmin=0, vmax=1, aspect="equal")
    ax[1].set_title("front projection (fraction of tank depth front-to-back that is dead)", fontsize=9)
    fig.colorbar(im, ax=ax[1], shrink=0.75)
    for a_ in ax:
        a_.set_xlabel(f"x [{lu}]")
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def shear_figure(run, ws, path):
    m = run.model
    (lu, lf), _ = _u(m)
    tau = ws["sand_tau_map"]
    r = ws["report"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    im = ax[0].imshow(np.log10(np.clip(tau, 1e-4, None)).T, origin="lower", extent=_extent_plan(m, lf),
                      cmap="viridis", vmin=-3.5, vmax=0, aspect="equal")
    X, Y = m.node_coords(0) * lf, m.node_coords(1) * lf
    ax[0].contour(X, Y, np.nan_to_num(tau, nan=1).T, levels=[r["tau_crit_Pa"]], colors="w", linewidths=0.8)
    _rock_outline(ax[0], m, lf)
    fig.colorbar(im, ax=ax[0], shrink=0.75, label="log10 wall shear [Pa]")
    ax[0].set_title(f"Sand-bed wall shear. White line: {r['tau_crit_Pa']} Pa detritus threshold\n"
                    "(detritus likely to settle wherever shear is below it)", fontsize=9)
    vals = tau[np.isfinite(tau)]
    ax[1].hist(np.log10(np.clip(vals, 1e-4, None)), bins=60, color="tab:blue", alpha=0.8, label="sand")
    if len(ws["rock_top_tau"]):
        ax[1].hist(np.log10(np.clip(ws["rock_top_tau"], 1e-4, None)), bins=60, color="tab:brown", alpha=0.6,
                   label="rock tops")
    for v, lab in ((r["tau_crit_Pa"], "detritus threshold"), (r["tau_sand_move_Pa"], "sand moves")):
        ax[1].axvline(np.log10(v), color="k", ls="--", lw=0.8)
        ax[1].text(np.log10(v), ax[1].get_ylim()[1] * 0.9, " " + lab, fontsize=7)
    ax[1].set_xlabel("log10 wall shear [Pa]"); ax[1].set_ylabel("surface cells")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def coral_figure(run, sc, out, path):
    m = run.model
    (lu, lf), _ = _u(m)
    names = list(sc["profiles"])
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.6 * len(names)), squeeze=False)
    X, Y, Z = (m.node_coords(a) for a in range(3))
    nodes = sc["nodes"]
    for ax, name in zip(axes[:, 0], names):
        s = np.nan_to_num(sc["profiles"][name]["combined"], nan=0.0)
        # plan view: highest-scoring surface point in each column (top-most surface wins ties)
        order = np.lexsort((nodes[:, 2], s))
        grid = np.full(m.shape[:2], np.nan)
        grid[nodes[order, 0], nodes[order, 1]] = s[order]
        im = ax.imshow(grid.T, origin="lower", extent=_extent_plan(m, lf), cmap="viridis", vmin=0, vmax=1,
                       aspect="equal", interpolation="nearest")
        _rock_outline(ax, m, lf)
        for n, b in enumerate(out["coral"]["best_spots"][name][:10]):
            p = np.array(b["position_m"]) * lf
            ax.plot(p[0], p[1], "*", color="r", ms=8, mec="k", mew=0.4)
            ax.annotate(str(n + 1), (p[0], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points", color="r")
        lt = "flow x light" if out["coral"]["light_available"] else "flow only (no PAR yet)"
        ax.set_title(f"{name}: placement score ({lt}); stars = ranked best spots", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def all_figures(run, d, *, age, dead, ws, sc, out) -> dict:
    figs = {"slices": "slices.png", "history": "history.png", "geometry": "geometry.png"}
    if age is not None:
        age_figure(run, age, d / "age.png", out["age"]["tau_nominal_s"])
        figs["age"] = "age.png"
    dead_figure(run, dead, d / "dead_zones.png"); figs["dead_zones"] = "dead_zones.png"
    shear_figure(run, ws, d / "sand_shear.png"); figs["sand_shear"] = "sand_shear.png"
    coral_figure(run, sc, out, d / "coral_scores.png"); figs["coral"] = "coral_scores.png"
    return figs
