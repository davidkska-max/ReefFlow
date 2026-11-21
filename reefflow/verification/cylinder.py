"""Verification 3 - 2D flow past a circular cylinder at Re = 100: Strouhal number.

Set-up (lattice units): cylinder D cells (nodes with centre inside the circle are
solid; halfway bounce-back => staircase surface), single cell thick, periodic in z.
Domain 32 D x 25 D (blockage D/H = 4 %), cylinder centre 8 D from the inlet.
Inlet: uniform velocity U; outlet: constant pressure; sides: free-slip.
Shedding is triggered by a small antisymmetric initial perturbation.

St = f D / U from the transverse velocity at a probe 2 D downstream of the
cylinder centre (period from interpolated upward zero crossings over the last
~30 shedding cycles).

Reference: unconfined 2D Re = 100, St = 0.164-0.167 (Williamson 1996 fit gives
0.164; 2D simulations 0.164-0.166). Spec acceptance band: 0.16 <= St <= 0.17.
"""
import numpy as np

from ..domain import Domain, SOLID, FREESLIP, INLET, OUTLET
from ..solver import LBMSolver
from .common import Timer, outdir, plt, save_json


def run_case(lattice="D3Q19", collision="bgk", D=20, Re=100.0, U=0.05, n_periods=70, sample_every=20):
    nx, ny = 32 * D, 25 * D
    nu = U * D / Re
    d = Domain((nx, ny, 1), periodic=(False, False, True))
    d.set_face("y-", FREESLIP)
    d.set_face("y+", FREESLIP)
    d.set_face("x-", INLET, velocity=(U, 0.0, 0.0))
    d.set_face("x+", OUTLET, rho=1.0)
    xc, yc = 8 * D, ny / 2.0 - 0.5
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    disk = (I - xc) ** 2 + (J - yc) ** 2 < (D / 2.0) ** 2
    d.set_mask(disk[:, :, None], SOLID)
    D_eff = 2.0 * np.sqrt(disk.sum() / np.pi)

    s = LBMSolver(d, lattice=lattice, collision=collision, nu=nu)
    u0 = np.zeros((nx, ny, 1, 3), np.float32)
    u0[..., 0] = U
    u0[..., 1] = 0.02 * U * np.sin(np.pi * (I - xc) / (4 * D))[:, :, None] * (I > xc)[:, :, None]
    u0[disk] = 0.0
    s.initialize(u=u0)

    T_est = D / (0.165 * U)
    n_steps = int(n_periods * T_est)
    probe = (int(round(xc + 2 * D)), int(round(yc)))
    t, v = [], []
    while s.time_step < n_steps:
        s.step(sample_every)
        v.append(float(s.u[probe[0], probe[1], 0][1]))
        t.append(s.time_step)
    t = np.array(t, float)
    v = np.array(v)
    if not np.all(np.isfinite(v)):
        raise FloatingPointError("cylinder run diverged")

    # analyse the last ~30 periods
    keep = t > t[-1] - 30 * T_est
    tt, vv = t[keep], v[keep] - v[keep].mean()
    up = np.where((vv[:-1] < 0) & (vv[1:] >= 0))[0]
    tc = tt[up] - vv[up] * (tt[up + 1] - tt[up]) / (vv[up + 1] - vv[up])
    periods = np.diff(tc)
    T = periods.mean()
    St = D / (T * U)
    return {
        "lattice": lattice, "collision": collision, "D": D, "D_eff": float(D_eff), "Re": Re, "U": U,
        "tau": 0.5 + 3 * nu, "steps": int(s.time_step), "n_cycles": int(len(periods)),
        "period_steps": float(T), "period_rel_std": float(periods.std() / T),
        "St": float(St), "St_Deff": float(D_eff / (T * U)),
        "v_amplitude_over_U": float(0.5 * (vv.max() - vv.min()) / U),
        "pass": bool(0.16 <= St <= 0.17),
        "_t": t, "_v": v,
    }


def run(combos=None, plot=True):
    combos = combos or [("D3Q19", "bgk"), ("D3Q19", "rr"), ("D3Q27", "rr")]
    out = {"cases": []}
    with Timer() as tm:
        for lat, col in combos:
            out["cases"].append(run_case(lat, col))
    out["runtime_s"] = tm.dt
    if plot:
        d = outdir("03_cylinder")
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for c in out["cases"]:
            ax.plot(c["_t"] * c["U"] / c["D"], c["_v"] / c["U"], lw=0.8,
                    label=f"{c['lattice']} {c['collision']}: St = {c['St']:.4f}")
        ax.set_xlabel("t U / D"); ax.set_ylabel("v / U at probe (x = 2D behind centre)")
        ax.set_title("Cylinder Re = 100: transverse velocity at wake probe")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(d / "cylinder.png"); plt.close(fig)
    for c in out["cases"]:
        c.pop("_t"); c.pop("_v")
    save_json("03_cylinder", out)
    return out


if __name__ == "__main__":
    from .. import backend
    backend.init()
    for c in run()["cases"]:
        print(f"{c['lattice']} {c['collision']}: St={c['St']:.4f} (D_eff {c['St_Deff']:.4f}) "
              f"cycles={c['n_cycles']} pass={c['pass']}")
