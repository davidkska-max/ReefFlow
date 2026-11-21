"""Verification 2 - lid-driven square cavity, Re = 100 and 1000, vs Ghia et al. (1982).

Set-up (lattice units): N x N fluid nodes, halfway bounce-back walls, lid moving
at U (Ladd moving-wall correction), single cell thick and periodic in z (the 3D
lattices then reduce exactly to D2Q9 dynamics). nu = U N / Re.
Run to steady state: max |du| over 1000 steps < 1e-6 U.

Pre-declared acceptance: max |u_LBM - u_Ghia| / U over all 2 x 15 interior
Ghia points <= 0.02 at Re = 100 and <= 0.03 at Re = 1000 (Ghia's own values differ
from spectral reference solutions by ~1% at Re = 1000; Botella & Peyret 1998).
"""
import numpy as np

from ..domain import Domain, SOLID
from ..solver import LBMSolver
from . import ghia
from .common import Timer, outdir, plt, save_json

TOL = {100: 0.02, 1000: 0.03}


def run_case(Re: int, N: int, lattice="D3Q19", collision="bgk", U=0.1, max_steps=2_000_000):
    nu = U * N / Re
    d = Domain((N + 2, N + 2, 1), periodic=(False, False, True))
    for f in ("x-", "x+", "y-"):
        d.set_face(f, SOLID)
    d.set_face("y+", SOLID, velocity=(U, 0.0, 0.0))
    s = LBMSolver(d, lattice=lattice, collision=collision, nu=nu)
    check = 1000
    prev = None
    res_hist = []
    while s.time_step < max_steps:
        s.step(check)
        _, u = s.macros()
        uf = u[1:N + 1, 1:N + 1, 0, :2]
        if prev is not None:
            r = float(np.max(np.abs(uf - prev)) / U)
            res_hist.append((s.time_step, r))
            if r < 1e-6:
                break
        prev = uf.copy()
    if not np.all(np.isfinite(uf)):
        raise FloatingPointError("cavity run diverged")

    # node centres at (i - 0.5)/N; centre line x = 0.5 lies between columns N/2 and N/2+1
    pos = (np.arange(1, N + 1) - 0.5) / N
    if N % 2 == 0:
        ucol = 0.5 * (uf[N // 2 - 1, :, 0] + uf[N // 2, :, 0])
        vrow = 0.5 * (uf[:, N // 2 - 1, 1] + uf[:, N // 2, 1])
    else:
        ucol = uf[N // 2, :, 0]
        vrow = uf[:, N // 2, 1]
    yy = np.concatenate([[0.0], pos, [1.0]])
    uu = np.concatenate([[0.0], ucol / U, [1.0]])
    vv = np.concatenate([[0.0], vrow / U, [0.0]])
    u_at = np.interp(ghia.Y, yy, uu)
    v_at = np.interp(ghia.X, yy, vv)
    inner = slice(1, -1)
    du = u_at[inner] - ghia.U[Re][inner]
    dv = v_at[inner] - ghia.V[Re][inner]
    dev = np.concatenate([du, dv])
    return {
        "Re": Re, "N": N, "lattice": lattice, "collision": collision, "U_lattice": U,
        "tau": 0.5 + 3 * nu, "steps": s.time_step, "final_residual": res_hist[-1][1],
        "max_dev": float(np.max(np.abs(dev))), "rms_dev": float(np.sqrt(np.mean(dev**2))),
        "max_dev_u": float(np.max(np.abs(du))), "max_dev_v": float(np.max(np.abs(dv))),
        "pass": bool(np.max(np.abs(dev)) <= TOL[Re]), "tolerance": TOL[Re],
        "_profile": {"pos": yy, "u": uu, "v": vv},
    }


def run(cases=None, plot=True):
    cases = cases or [(100, 128, "D3Q19", "bgk"), (100, 128, "D3Q19", "rr"), (100, 128, "D3Q27", "rr"),
                      (1000, 256, "D3Q19", "bgk"), (1000, 256, "D3Q19", "rr"), (1000, 256, "D3Q27", "rr")]
    out = {"cases": []}
    with Timer() as t:
        for Re, N, lat, col in cases:
            out["cases"].append(run_case(Re, N, lat, col))
    out["runtime_s"] = t.dt
    if plot:
        _plot(out)
    for c in out["cases"]:
        c.pop("_profile")
    save_json("02_cavity", out)
    return out


def _plot(out):
    d = outdir("02_cavity")
    Res = sorted({c["Re"] for c in out["cases"]})
    fig, axes = plt.subplots(1, len(Res), figsize=(5.5 * len(Res), 4.6))
    axes = np.atleast_1d(axes)
    for ax, Re in zip(axes, Res):
        for c in [c for c in out["cases"] if c["Re"] == Re]:
            p = c["_profile"]
            lab = f"{c['lattice']} {c['collision']} N={c['N']} (max dev {c['max_dev']:.3f})"
            ax.plot(p["u"], p["pos"], "-", lw=1, label="u(y) " + lab)
            ax.plot(p["pos"], p["v"], "--", lw=1, label="v(x) " + lab)
        ax.plot(ghia.U[Re], ghia.Y, "ks", ms=4, mfc="none", label="Ghia et al. u(y)")
        ax.plot(ghia.X, ghia.V[Re], "k^", ms=4, mfc="none", label="Ghia et al. v(x)")
        ax.set_title(f"Lid-driven cavity Re = {Re}")
        ax.set_xlabel("u/U  (or x for v-profile)"); ax.set_ylabel("y  (or v/U)")
        ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(d / "cavity.png")
    plt.close(fig)


if __name__ == "__main__":
    from .. import backend
    backend.init()
    res = run()
    for c in res["cases"]:
        print(f"Re={c['Re']:5d} N={c['N']} {c['lattice']} {c['collision']}: max dev {c['max_dev']:.4f} "
              f"rms {c['rms_dev']:.4f} steps {c['steps']} pass={c['pass']}")
