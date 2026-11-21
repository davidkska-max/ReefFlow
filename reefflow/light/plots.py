"""Matplotlib (Agg) figures for lighting results: plan PAR maps, front section,
rock / sand PAR, 3-D rock view, difference maps."""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEQ = "YlGn"          # sequential, single hue family, light = low PAR
DIV = "RdBu_r"        # diverging, neutral midpoint (differences)
IN = 0.0254


def _title(res, what):
    return f"{what}\n{res.label}"


def plan_maps(res, path, dli: bool = False):
    if res.plan is None:
        return None
    n = len(res.cfg.map_depths)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    f = res.cfg.photoperiod.dli_factor if dli else 1.0
    data = res.plan * f
    vmax = np.nanpercentile(data, 99.5)
    fig, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 2.6 * rows + 0.6), squeeze=False)
    ext = [0, res.plan_x[-1] + res.plan_x[0], 0, res.plan_y[-1] + res.plan_y[0]]
    for l, d in enumerate(res.cfg.map_depths):
        ax = axs[l // cols][l % cols]
        im = ax.imshow(data[l].T, origin="lower", extent=[e / IN for e in ext], cmap=SEQ, vmin=0, vmax=vmax,
                       aspect="equal")
        if res.sdf is not None:
            X, Y = np.meshgrid(res.plan_x, res.plan_y, indexing="ij")
            phi = res.sdf.sample(np.stack([X, Y, np.full_like(X, res.tank.water_surface - d)], -1))
            if (phi < 0).any():
                ax.contour(X / IN, Y / IN, phi, levels=[0], colors="0.25", linewidths=0.8)
        v = data[l][np.isfinite(data[l])]
        ax.set_title(f"depth {d / IN:.1f} in ({d * 1e3:.0f} mm): min {v.min():.0f} / mean {v.mean():.0f} / "
                     f"max {v.max():.0f}", fontsize=8)
        ax.set_xlabel("x (in, from left glass)"); ax.set_ylabel("y (in, from front)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="DLI (mol m$^{-2}$ d$^{-1}$)" if dli else "PAR (µmol m$^{-2}$ s$^{-1}$)")
    for l in range(n, rows * cols):
        axs[l // cols][l % cols].axis("off")
    fig.suptitle(_title(res, "Meter-equivalent PAR, plan view (upward-facing cosine sensor)"), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def section(res, path, y=None):
    if res.meter is None:
        return None
    E, x, z, yu = res.section(y)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(E.T, origin="lower", cmap=SEQ, vmin=0, vmax=np.nanpercentile(E, 99.5),
                   extent=[(x[0] - res.grid.dx / 2) / IN, (x[-1] + res.grid.dx / 2) / IN,
                           (z[0] - res.grid.dx / 2) / IN, (z[-1] + res.grid.dx / 2) / IN], aspect="equal")
    if res.sdf is not None:
        X, Z = np.meshgrid(x, z, indexing="ij")
        phi = res.sdf.sample(np.stack([X, np.full_like(X, yu), Z], -1))
        if (phi < 0).any():
            ax.contourf(X / IN, Z / IN, phi, levels=[-1, 0], colors=["0.55"])
    ax.axhline(res.tank.water_surface / IN, color="0.3", lw=0.8, ls="--")
    ax.axhline(res.tank.sand_depth / IN, color="0.3", lw=0.8)
    ax.set_xlabel("x (in)"); ax.set_ylabel("z (in, above bottom glass)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="PAR (µmol m$^{-2}$ s$^{-1}$)")
    ax.set_title(_title(res, f"Front-elevation section at y = {yu / IN:.1f} in (meter-equivalent PAR; rock grey)"),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _top_surface(res, which):
    """Plan view of the top-most lit surface per (x, y) column: rock patch (meter or normal
    PAR) where rock is exposed from above, sand elsewhere."""
    g = res.grid
    out = np.where(np.isfinite(res.sand), res.sand, np.nan) if res.sand is not None else np.full(g.shape[:2], np.nan)
    if res.rock_normal is not None:
        val = res.rock_meter if which == "meter" else res.rock_normal
        ok = res.rock_valid & (res.patches["normal"][..., 2] > 0.0) & np.isfinite(val)
        kk = np.where(ok, np.arange(g.shape[2])[None, None, :], -1)
        ktop = kk.max(2)
        i, j = np.nonzero(ktop >= 0)
        out = out.copy()
        out[i, j] = val[i, j, ktop[i, j]]
    return out


def surface_maps(res, path, dli: bool = False):
    """Sand + rock-top PAR (plan view of the top-most surface): meter-equivalent and surface-normal."""
    g = res.grid
    f = res.cfg.photoperiod.dli_factor if dli else 1.0
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.4))
    ext = [(g.coords(0)[0] - g.dx / 2) / IN, (g.coords(0)[-1] + g.dx / 2) / IN,
           (g.coords(1)[0] - g.dx / 2) / IN, (g.coords(1)[-1] + g.dx / 2) / IN]
    for ax, which, ttl in ((axs[0], "meter", "meter-equivalent (horizontal sensor at the surface)"),
                           (axs[1], "normal", "surface-normal irradiance (on the actual rock face)")):
        A = _top_surface(res, which) * f
        im = ax.imshow(A.T, origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=np.nanpercentile(A, 99.5),
                       aspect="equal")
        if res.rock_valid is not None:
            rockcol = (res.patches["area"] > 0).any(2)
            ax.contour(g.coords(0) / IN, g.coords(1) / IN, rockcol.T.astype(float), levels=[0.5], colors="0.3",
                       linewidths=0.6)
        ax.set_title(f"Top surface (rock tops + sand): {ttl}", fontsize=8)
        ax.set_xlabel("x (in)"); ax.set_ylabel("y (in)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85,
                     label="DLI (mol m$^{-2}$ d$^{-1}$)" if dli else "PAR (µmol m$^{-2}$ s$^{-1}$)")
    fig.suptitle(_title(res, "DLI on rock and sand" if dli else "PAR on rock and sand"), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def rock_3d(res, path, which: str = "normal", max_faces: int = 40000):
    """3-D view of the rock surface coloured by PAR (surface-normal by default). Renders a
    coarser marching-cubes mesh of the SDF; each face takes the area-weighted mean patch value
    of the 3x3x3 patch neighbourhood of its centroid."""
    if res.patches is None or res.sdf is None:
        return None
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy import ndimage
    from skimage.measure import marching_cubes
    sdf, g = res.sdf, res.grid
    k = res.tallies.ppf_per_photon * res.scale
    A = res.patches["area"]
    box = np.ones((3, 3, 3))
    if which == "normal":
        num = ndimage.convolve(res.tallies.rock_w * k, box, mode="constant")
    else:
        num = ndimage.convolve(np.nan_to_num(res.rock_meter) * A, box, mode="constant")
    den = ndimage.convolve(A, box, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(den > 0.1 * g.dx ** 2, num / den, np.nan)
    v = f = None
    for step in (1, 2, 3, 4, 6, 8):
        P = sdf.phi[::step, ::step, ::step].copy()
        for sl in (np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1], np.s_[:, :, 0], np.s_[:, :, -1]):
            P[sl] = np.maximum(P[sl], sdf.h)
        if P.min() >= 0:
            return None
        v, f, _, _ = marching_cubes(P, 0.0, spacing=(sdf.h * step,) * 3)
        v = v + sdf.lo
        if len(f) <= max_faces:
            break
    cen = v[f].mean(1)
    keep = (cen[:, 2] > res.tank.sand_depth) & (cen[:, 2] < res.tank.water_surface)
    f, cen = f[keep], cen[keep]
    ijk = np.clip(np.floor((cen - g.origin) / g.dx + 0.5).astype(int), 0, np.array(g.shape) - 1)
    c = smooth[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    cmap = plt.get_cmap(SEQ)
    vmax = np.nanpercentile(c, 99) if np.isfinite(c).any() else 1.0
    col = cmap(np.clip(np.nan_to_num(c, nan=0.0) / vmax, 0, 1))
    col[~np.isfinite(c)] = (0.75, 0.75, 0.75, 1.0)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(v[f] / IN, facecolors=col, edgecolor="none"))
    t = res.tank
    ax.set_xlim(0, t.length / IN); ax.set_ylim(0, t.width / IN); ax.set_zlim(t.sand_depth / IN, t.water_surface / IN)
    ax.set_box_aspect((t.length, t.width, t.water_column))
    ax.view_init(elev=28, azim=-62)
    ax.set_xlabel("x (in)"); ax.set_ylabel("y (in)"); ax.set_zlabel("z (in)")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
    ax.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, shrink=0.6, label="PAR (µmol m$^{-2}$ s$^{-1}$)")
    ax.set_title(_title(res, f"Rock surface PAR ({'surface-normal' if which == 'normal' else 'meter-equivalent'}; "
                             "seen from front-left, grey = not evaluated)"), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def difference_map(a, b, path, depth_index: int = 0, names=("A", "B")):
    """Plan-map difference B - A at one depth (diverging colours, zero = white)."""
    D = b.plan[depth_index] - a.plan[depth_index]
    lim = np.nanpercentile(np.abs(D), 99)
    fig, axs = plt.subplots(1, 3, figsize=(15, 2.8))
    ext = [0, a.plan_x[-1] / IN, 0, a.plan_y[-1] / IN]
    vmax = np.nanpercentile(np.concatenate([a.plan[depth_index].ravel(), b.plan[depth_index].ravel()]), 99.5)
    for ax, M, t in ((axs[0], a.plan[depth_index], names[0]), (axs[1], b.plan[depth_index], names[1])):
        im = ax.imshow(M.T, origin="lower", extent=ext, cmap=SEQ, vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(f"{t}: mean {np.nanmean(M):.0f}", fontsize=8); ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="PAR")
    im = axs[2].imshow(D.T, origin="lower", extent=ext, cmap=DIV, vmin=-lim, vmax=lim, aspect="equal")
    axs[2].set_title(f"{names[1]} - {names[0]}: mean {np.nanmean(D):+.0f}", fontsize=8); axs[2].grid(False)
    fig.colorbar(im, ax=axs[2], shrink=0.8, label="ΔPAR")
    fig.suptitle(f"Plan PAR at depth {a.cfg.map_depths[depth_index] * 1e3:.0f} mm", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
