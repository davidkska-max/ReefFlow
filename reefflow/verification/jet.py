"""Verification 4 - turbulent round free jet: centreline decay and spreading rate.

Self-similar far field of a round jet (Pope 2000, Sec. 5.1):
    U_j / U_c(x) = (x - x0) / (B D)          centreline decay, U_c ~ 1/x
    r_1/2(x)     = S (x - x0)                 half-velocity radius, linear spreading
Experimental spread (Pope 2000, Table 5.1):
    Wygnanski & Fiedler 1969   B = 5.0   S = 0.086
    Panchapakesan & Lumley 93  B = 6.06  S = 0.096
    Hussein et al. 1994 (LDA)  B = 5.8   S = 0.094
    Hussein et al. 1994 (HW)   B = 5.9   S = 0.102
Pre-declared acceptance (literature range widened by ~10 % for an LES with
D = 10 cells): 4.5 <= B <= 6.7 and 0.077 <= S <= 0.112, fitted over
15 <= x/D <= 25.

Set-up (lattice units): nozzle D = 10 cells at the centre of the x = 0 plane
(velocity inlet: smoothed top-hat U_j = 0.08 with ~5 % shear-layer perturbations
on a few azimuthal modes + random noise to trigger transition); the rest of
the x = 0 plane is a no-slip velocity boundary (jet issuing from a wall).
Lateral and outlet faces are constant-pressure boundaries (free entrainment).
Domain 30 D x 16 D x 16 D. Re_D = U_j D / nu = 10^4 (molecular); LES:
recursive-regularized + Smagorinsky Cs = 0.10 (production default, D-118). Statistics after a spin-up of
~2.5 flow-through times.

Sponge layers (added viscosity, quadratic ramp to nu = 0.05) over 26 < x/D < 30
and the outer 1.5 D of each lateral face damp eddies before they reach the
constant-pressure boundaries. Without them a slow spurious circulation built
up through the lateral boundaries (inflow there grew to ~12 % of U_j, versus
~0.4 % required by entrainment) and one run diverged at step 80 000 (DECISIONS
D-115). Both zones lie outside the fitting window.
"""
import math

import numpy as np
import taichi as ti

from ..domain import Domain, INLET, OUTLET
from ..kernels import V3
from ..solver import LBMSolver
from ..stats import RunningStats
from .common import Timer, outdir, plt, save_json

LIT = [("Wygnanski & Fiedler 1969", 5.0, 0.086), ("Panchapakesan & Lumley 1993", 6.06, 0.096),
       ("Hussein et al. 1994 LDA", 5.8, 0.094), ("Hussein et al. 1994 HW", 5.9, 0.102)]
B_RANGE = (4.5, 6.7)
S_RANGE = (0.077, 0.112)

F1 = ti.types.ndarray(dtype=ti.f32, ndim=1)


@ti.kernel
def _inflow(u: V3, t: ti.f32, Uj: ti.f32, R: ti.f32, cy: ti.f32, cz: ti.f32, eps: ti.f32,
            freq: F1, phase: F1):
    ny, nz = u.shape[1], u.shape[2]
    for j, k in ti.ndrange(ny, nz):
        dy = ti.cast(j, ti.f32) - cy
        dz = ti.cast(k, ti.f32) - cz
        r = ti.sqrt(dy * dy + dz * dz)
        phi = ti.atan2(dz, dy)
        base = Uj * 0.5 * (1.0 - ti.tanh((r - R) / 1.0))
        shear = ti.exp(-((r - R) / 1.5) ** 2)
        pert = 0.0
        for m in ti.static(range(4)):
            pert += ti.sin(2.0 * math.pi * freq[m] * t + m * phi + phase[m])
        ux = base + eps * Uj * shear * (0.5 * pert + (ti.random() - 0.5))
        ur = eps * Uj * shear * (ti.random() - 0.5)
        ut = eps * Uj * shear * (ti.random() - 0.5)
        if base < 1e-4 * Uj:
            ux = 0.0
            ur = 0.0
            ut = 0.0
        c = ti.cos(phi)
        sn = ti.sin(phi)
        u[0, j, k] = ti.Vector([ux, ur * c - ut * sn, ur * sn + ut * c])


def sponge_profile(shape, D, cy, cz, Lx, Lyz, nu_max=0.05, x_start=26.0, lat_width=1.5):
    """Extra viscosity: quadratic ramp from 0 to nu_max over x/D in [x_start, Lx] (outlet)
    and over the outer lat_width*D of each lateral face. Zero inside the measurement
    region (x/D <= 25, |y|,|z| <= Lyz/2 - lat_width)."""
    nx, ny, nz = shape
    x = np.arange(nx) / D
    ax = np.clip((x - x_start) / (Lx - x_start), 0, 1) ** 2
    half = Lyz / 2.0
    y = np.abs(np.arange(ny) - cy) / D
    z = np.abs(np.arange(nz) - cz) / D
    ay = np.clip((y - (half - lat_width)) / lat_width, 0, 1) ** 2
    az = np.clip((z - (half - lat_width)) / lat_width, 0, 1) ** 2
    a = np.maximum(ax[:, None, None], np.maximum(ay[None, :, None], az[None, None, :]))
    return nu_max * a


def run_case(lattice="D3Q19", D=10, Uj=0.08, Re=1e4, cs=0.10, Lx=30, Lyz=16, eps=0.05,
             spinup_ft=2.5, average_steps=75_000, sample_every=25, seed=1, sponge_nu=0.05,
             fit=(15.0, 25.0), label=None):
    nx = Lx * D + 1
    ny = nz = Lyz * D + 1
    nu = Uj * D / Re
    d = Domain((nx, ny, nz))
    for f in ("y-", "y+", "z-", "z+", "x+"):
        d.set_face(f, OUTLET, rho=1.0)
    d.set_face("x-", INLET, velocity=(0.0, 0.0, 0.0))
    cy = cz = (ny - 1) / 2.0
    sp = sponge_profile((nx, ny, nz), D, cy, cz, Lx, Lyz, nu_max=sponge_nu, x_start=Lx - 4.0)
    s = LBMSolver(d, lattice=lattice, collision="rr", nu=nu, smagorinsky=cs, sponge_nu=sp)
    rng = np.random.default_rng(seed)
    freq = ti.ndarray(ti.f32, 4)
    freq.from_numpy((rng.uniform(0.3, 0.7, 4) * Uj / D).astype(np.float32))
    phase = ti.ndarray(ti.f32, 4)
    phase.from_numpy(rng.uniform(0, 2 * np.pi, 4).astype(np.float32))

    # flow-through time estimate: integral of dx / U_c with U_c = U_j B D / x
    ft = (nx ** 2) / (2 * 5.8 * D * Uj)
    spinup = int(spinup_ft * ft)
    stats = RunningStats(s)
    diag = []
    n_total = spinup + average_steps
    while s.time_step < n_total:
        _inflow(s.u, float(s.time_step), Uj, D / 2.0, cy, cz, eps, freq, phase)
        s.step(1, store_last=(s.time_step + 1) % sample_every == 0)
        if s.time_step >= spinup and s.time_step % sample_every == 0:
            stats.sample()
        if s.time_step % 5000 == 0:
            uu = s.u.to_numpy()
            if not np.all(np.isfinite(uu)):
                raise FloatingPointError(f"jet diverged at step {s.time_step}")
            lat_in = max(uu[1:-1, 1, :, 1].max(), (-uu[1:-1, -2, :, 1]).max(),
                         uu[1:-1, :, 1, 2].max(), (-uu[1:-1, :, -2, 2]).max())
            diag.append((s.time_step, float(lat_in), float((-uu[-2, 1:-1, 1:-1, 0]).max())))
            print(f"  {lattice} step {s.time_step}: max lateral inflow {lat_in / Uj:.4f} Uj, "
                  f"max outlet backflow {diag[-1][2] / Uj:.4f} Uj", flush=True)
    res = stats.result()
    out = _analyse(res, D, Uj, cy, cz, nx, lattice, cs, s.time_step, stats.n, spinup, fit,
                   label or f"{lattice}_Cs{cs}")
    out.update(inflow_eps=eps, Lx_over_D=Lx)
    out["sponge_nu_max"] = sponge_nu
    out["boundary_diag"] = [(t, a / Uj, b / Uj) for t, a, b in diag]
    return out


def radial_profile(Ui, r, rmax):
    """Azimuthal average of a cross-plane field over 0.5-cell radial bins.
    Bins containing no lattice node are dropped (they are not zero-velocity points)."""
    rb = np.arange(0.0, rmax + 0.5, 0.5)
    idx = np.digitize(r.ravel(), rb) - 1
    ok = (idx >= 0) & (idx < len(rb) - 1)
    sums = np.bincount(idx[ok], weights=Ui.ravel()[ok], minlength=len(rb) - 1)
    rsum = np.bincount(idx[ok], weights=r.ravel()[ok], minlength=len(rb) - 1)
    cnt = np.bincount(idx[ok], minlength=len(rb) - 1)
    keep = cnt > 0
    return rsum[keep] / cnt[keep], sums[keep] / cnt[keep]


def half_radius(rc, pr, Uc):
    """First radius where the azimuthally averaged profile falls to Uc/2 (linear interpolation)."""
    target = 0.5 * Uc
    below = np.where(pr < target)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    b = below[0]
    return rc[b - 1] + (target - pr[b - 1]) * (rc[b] - rc[b - 1]) / (pr[b] - pr[b - 1])


def analyse(U, Rxx, D, Uj, cy, cz, fit_range=(15.0, 25.0)):
    """Centreline decay, half-width and momentum flux from the mean axial velocity U(x,y,z)
    and the axial Reynolds stress Rxx. Returns a dict (private arrays prefixed by '_')."""
    nx, ny, nz = U.shape
    jc, kc = int(round(cy)), int(round(cz))
    Uc = U[:, jc, kc].astype(float)
    J, K = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    r = np.sqrt((J - cy) ** 2 + (K - cz) ** 2)
    rmax = min(cy, cz) - 2
    half = np.full(nx, np.nan)
    mom = np.full(nx, np.nan)
    prof = {}
    inside = r <= rmax
    for i in range(1, nx - 1):
        rc, pr = radial_profile(U[i], r, rmax)
        prof[i] = (rc, pr)
        half[i] = half_radius(rc, pr, Uc[i])
        mom[i] = np.sum((U[i] ** 2 + Rxx[i])[inside])
    J0 = mom[1]                                    # momentum flux of the actual inflow profile
    J0_tophat = Uj ** 2 * math.pi * (D / 2) ** 2
    xd = np.arange(nx) / D
    core = np.where(Uc[1:] < 0.95 * Uc[1])[0]
    x_core = float(xd[core[0] + 1]) if len(core) else float("nan")
    fit = (xd >= fit_range[0]) & (xd <= fit_range[1]) & np.isfinite(half)
    a1, b1 = np.polyfit(xd[fit], Uj / Uc[fit], 1)       # Uj/Uc = (x/D - x0/D) / B
    a2, b2 = np.polyfit(xd[fit], half[fit] / D, 1)       # r_1/2 / D = S (x/D - x0'/D)
    B, x0 = 1.0 / a1, -b1 / a1
    S, x0s = a2, -b2 / a2
    return {
        "B": float(B), "x0_over_D": float(x0), "S": float(S), "x0_spread_over_D": float(x0s),
        "potential_core_over_D": x_core, "J0_over_tophat": float(J0 / J0_tophat),
        "momentum_flux_ratio_range_x5_25": [float(np.nanmin(mom[5 * D:25 * D] / J0)),
                                            float(np.nanmax(mom[5 * D:25 * D] / J0))],
        "pass": bool(B_RANGE[0] <= B <= B_RANGE[1] and S_RANGE[0] <= S <= S_RANGE[1]),
        "_xd": xd, "_Uc": Uc, "_half": half, "_mom": mom / J0, "_prof": prof,
    }


def _analyse(res, D, Uj, cy, cz, nx, lattice, cs, steps, n_samples, spinup, fit=(15.0, 25.0), tag=None):
    U = res["mean_u"][..., 0]
    Rxx = res["reynolds"][..., 0]
    out = {"lattice": lattice, "Cs": cs, "D_cells": D, "U_j": Uj, "steps": int(steps),
           "spinup_steps": spinup, "n_samples": int(n_samples)}
    out.update(analyse(U, Rxx, D, Uj, cy, cz, fit))
    out["fit_range"] = list(fit)
    tag = tag or f"{lattice}_Cs{cs}"
    np.savez_compressed(outdir("04_jet") / f"mean_{tag}.npz", U=U.astype(np.float32),
                        Rxx=Rxx.astype(np.float32), D=D, Uj=Uj, cy=cy, cz=cz)
    return out


def run(lattices=("D3Q19", "D3Q27"), plot=True, tag="", **kw):
    out = {"cases": [], "B_range": B_RANGE, "S_range": S_RANGE, "literature": LIT, "tag": tag}
    with Timer() as tm:
        for lat in lattices:
            with Timer() as t1:
                c = run_case(lat, label=f"{lat}{tag}" if tag else None, **kw)
            c["runtime_s"] = t1.dt
            out["cases"].append(c)
    out["runtime_s"] = tm.dt
    if plot:
        _plot(out, tag)
    for c in out["cases"]:
        for k in [k for k in c if k.startswith("_")]:
            c.pop(k)
    save_json("04_jet", out, filename=f"results{tag}.json")
    return out


def _plot(out, tag=""):
    d = outdir("04_jet")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for c in out["cases"]:
        xd, Uc = c["_xd"], c["_Uc"]
        lab = f"{c['lattice']} Cs={c['Cs']}: B={c['B']:.2f}, S={c['S']:.3f}"
        axes[0].plot(xd, c["U_j"] / Uc, lw=1, label=lab)
        axes[1].plot(xd, c["_half"] / c["D_cells"], lw=1, label=lab)
        axes[2].plot(xd, c["_mom"], lw=1, label=f"{c['lattice']} Cs={c['Cs']}")
        f0, f1 = c.get("fit_range", (15, 25))
        for xi, ls in ((f0, ":"), (0.5 * (f0 + f1), "--"), (f1, "-")):
            i = int(xi * c["D_cells"])
            rc, pr = c["_prof"][i]
            eta = rc / ((xi - c["x0_over_D"]) * c["D_cells"])
            axes[3].plot(eta, pr / c["_Uc"][i], ls, lw=0.9, label=f"{c['lattice']} Cs={c['Cs']} x/D={xi}")
    xx = np.linspace(0, 30, 50)
    for name, B, S in LIT[1:3]:
        axes[0].plot(xx, (xx - 4) / B, "k:", lw=0.7)
        axes[1].plot(xx, S * (xx - 4), "k:", lw=0.7)
    axes[0].set_title("Centreline decay  U_j/U_c  (dotted: B=6.06, 5.8)")
    axes[0].set_xlabel("x / D"); axes[0].set_ylim(0, 6)
    axes[1].set_title("Half-velocity radius r_1/2 / D (dotted: S=0.096, 0.094)")
    axes[1].set_xlabel("x / D"); axes[1].set_ylim(0, 3)
    axes[2].set_title("Momentum flux  J(x)/J_0"); axes[2].set_xlabel("x / D"); axes[2].set_ylim(0, 1.5)
    eta = np.linspace(0, 0.25, 100)
    axes[3].plot(eta, np.exp(-np.log(2) * (eta / 0.094) ** 2), "k", lw=1.2, label="Gaussian fit, S=0.094")
    axes[3].set_title("Self-similar profile"); axes[3].set_xlabel("r / (x - x0)"); axes[3].set_ylabel("U / U_c")
    for ax in axes[:2]:
        f0, f1 = out["cases"][0].get("fit_range", (15, 25))
        ax.axvspan(f0, f1, color="0.9", zorder=0)
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(d / f"jet{tag}.png"); plt.close(fig)


if __name__ == "__main__":
    from .. import backend
    backend.init()
    for c in run()["cases"]:
        print(f"{c['lattice']}: B={c['B']:.3f} x0/D={c['x0_over_D']:.2f} S={c['S']:.4f} "
              f"J/J0 in {c['momentum_flux_ratio_range_x5_25']} pass={c['pass']} ({c['runtime_s']:.0f}s)")
