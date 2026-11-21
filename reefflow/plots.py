"""Standard result figures: plan / front / side slices of time-averaged speed with
in-plane vectors, and run histories. Display units follow the project setting."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .tank import TAGS  # noqa: E402

DISP = {"US": {"len": ("in", 1 / 0.0254), "vel": ("in/s", 1 / 0.0254)},
        "SI": {"len": ("mm", 1000.0), "vel": ("cm/s", 100.0)}}


def _units(model):
    return DISP[model.project.display_units]


def _slice(a, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    return a[tuple(sl)]


def slice_figure(model, speed, mean_u, path, title="Time-averaged speed <|u|>", vmax=None):
    """3 x 3 panel figure: 4 plan slices, 2 front elevations, 3 side elevations."""
    (lu, lf), (vu, vf) = _units(model)["len"], _units(model)["vel"]
    fl = model.fluid_mask
    tags = model.tags
    X, Y, Z = (model.node_coords(a) * lf for a in range(3))
    nx, ny, nz = model.shape
    spd = np.where(fl, speed * vf, np.nan)
    vmax = vmax or float(np.nanpercentile(spd, 99.5))
    plans = [(f"plan, {f} % of depth", 2, 1 + int(round((nz - 3) * f / 100))) for f in (15, 40, 65, 90)]
    fronts = [(f"front elevation, y = {Y[j]:.1f} {lu}", 1, j) for j in (int(ny * 0.33), int(ny * 0.66))]
    sides = [(f"side elevation, x = {X[i]:.1f} {lu}", 0, i) for i in (int(nx * 0.28), int(nx * 0.5), int(nx * 0.68))]
    panels = plans + fronts + sides
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1.05, 1.05])
    axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6]), fig.add_subplot(gs[1, 0:3]),
            fig.add_subplot(gs[1, 3:6]), fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6]),
            fig.add_subplot(gs[3, 0:2]), fig.add_subplot(gs[3, 2:4]), fig.add_subplot(gs[3, 4:6])]
    im = None
    for ax, (name, axis, idx) in zip(axes, panels):
        s = _slice(spd, axis, idx)
        t = _slice(tags, axis, idx)
        u = _slice(mean_u, axis, idx)
        if axis == 2:
            h, v, comps = X, Y, (0, 1)
        elif axis == 1:
            h, v, comps = X, Z, (0, 2)
        else:
            h, v, comps = Y, Z, (1, 2)
        solid = np.where(np.isnan(s), 1.0, np.nan)
        ext = [h[0], h[-1], v[0], v[-1]]
        im = ax.imshow(s.T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        ax.imshow(solid.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6,
                  interpolation="nearest", aspect="equal")
        src = np.isin(t, (TAGS["source"], TAGS["sink"]))
        if src.any():
            ii, jj = np.nonzero(src)
            ax.plot(h[ii], v[jj], "s", ms=2, color="tab:red", alpha=0.6)
        st = max(1, int(round(min(len(h), len(v)) / 14)))
        H, V = np.meshgrid(h[::st], v[::st], indexing="ij")
        uu = np.where(np.isnan(s), np.nan, u[..., comps[0]])[::st, ::st]
        vv = np.where(np.isnan(s), np.nan, u[..., comps[1]])[::st, ::st]
        ax.quiver(H, V, uu, vv, color="w", scale=None, width=0.0025, alpha=0.85)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label(f"<|u|> [{vu}]")
    fig.suptitle(f"{title} - {model.project.name}, dx = {model.dx * 1e3:.0f} mm. Grey: solid; red: "
                 "pump/overflow source nodes; arrows: in-plane mean velocity", fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def history_figure(hist: dict, path, t_spinup: float, probe_names):
    t = np.asarray(hist["t"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(t, np.asarray(hist["ke"]) * 1e4)
    ax[0].set_ylabel("volume-mean KE per mass [cm²/s²]")
    ax[1].plot(t, hist["u_max"])
    ax[1].set_ylabel("max |u| [m/s]")
    P = np.asarray(hist["probes"])
    for n, name in enumerate(probe_names):
        ax[2].plot(t, P[:, n] * 100, lw=0.7, label=name)
    ax[2].set_ylabel("probe speed [cm/s]")
    ax[2].legend(fontsize=6, ncol=2)
    for a in ax:
        a.axvline(t_spinup, color="k", ls="--", lw=0.8)
        a.set_xlabel("time [s]")
    ax[0].set_title("spin-up | averaging (dashed line)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def geometry_figure(model, path):
    """Plan and elevation views of the voxel model (tags)."""
    (lu, lf) = _units(model)["len"]
    X, Y, Z = (model.node_coords(a) * lf for a in range(3))
    tags = model.tags.astype(float)
    rock = (model.tags == TAGS["rock"])
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    h = rock[1:-1, 1:-1, 1:-1].sum(axis=2) * model.dx * lf
    im = ax[0].imshow(np.where(h > 0, h, np.nan).T, origin="lower", extent=[X[1], X[-2], Y[1], Y[-2]],
                      cmap="copper", aspect="equal")
    fig.colorbar(im, ax=ax[0], shrink=0.7, label=f"rock height [{lu}]")
    for name, c in (("overflow", "tab:blue"), ("source", "tab:red"), ("sink", "tab:cyan"),
                    ("housing", "k")):
        m = (model.tags == {"overflow": 4, "source": 7, "sink": 8, "housing": 5}[name]).any(axis=2)
        ii, jj = np.nonzero(m)
        ax[0].plot(X[ii], Y[jj], "s", ms=1.5, color=c, label=name)
    ax[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=4)
    ax[0].set_title("plan: rock height, overflow, pumps")
    side = np.where(model.tags > 0, tags, np.nan).max(axis=1)
    im = ax[1].imshow((model.tags == TAGS["rock"]).any(axis=1).T, origin="lower",
                      extent=[X[0], X[-1], Z[0], Z[-1]], cmap="Greys", aspect="equal")
    ax[1].set_title("front elevation: rock silhouette")
    for a in ax:
        a.set_xlabel(f"x [{lu}]")
    ax[0].set_ylabel(f"y [{lu}]"); ax[1].set_ylabel(f"z [{lu}]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
