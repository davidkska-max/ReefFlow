"""Verification 1 - force-driven plane Poiseuille flow.

Set-up (lattice units): channel of N fluid nodes between halfway bounce-back
walls (wall planes at y = 0 and y = N), periodic in x and z, uniform body force
g in x. Analytical steady solution, y measured from the lower wall plane:

    no-slip / no-slip   u(y) = g/(2 nu) * y (N - y)
    no-slip / free-slip u(y) = g/(2 nu) * y (2N - y)        (free-slip plane at y = N)

Diffusive scaling: tau fixed, u_max fixed => g = 8 nu u_max / N^2 (no-slip case).
The only error source is the wall closure, which for BGK is a uniform slip
(He et al. 1997)
    u_s = g (16 (tau - 1/2)^2 - 3) / (24 nu)
so the relative error falls as N^-2 and vanishes at tau = 1/2 + sqrt(3/16).

fp32 round-off floor: in a steady, force-driven case the per-step force
increment on a population (~ w 3 g) is only ~10^3 ulps of the population value,
and identical rounding repeats every step, biasing the steady state by
~ ulp * H^2 / nu. That grows as H^2 while the discretisation error falls as
H^-2; at tau = 0.8 they cross near H = 64. The observed order is therefore fitted
over H <= 32 in fp32, and the full sweep is repeated in fp64 (CPU backend) to
show the clean asymptotics. See ASSUMPTIONS A-109.
"""
import numpy as np

from ..domain import Domain, SOLID, FREESLIP
from ..solver import LBMSolver
from .common import Timer, fit_order, outdir, plt, save_json

MAGIC_TAU = 0.5 + np.sqrt(3.0 / 16.0)


def run_case(N: int, lattice="D3Q19", collision="bgk", tau=0.8, top="solid", u_max=0.05,
             max_diffusion_times=40.0):
    nu = (tau - 0.5) / 3.0
    H = N if top == "solid" else 2 * N
    g = 8.0 * nu * u_max / H**2
    d = Domain((2, N + 2, 1), periodic=(True, False, True))
    d.set_face("y-", SOLID)
    d.set_face("y+", SOLID if top == "solid" else FREESLIP)
    s = LBMSolver(d, lattice=lattice, collision=collision, nu=nu, body_force=(g, 0, 0))

    y = np.arange(1, N + 1) - 0.5
    ua = g / (2 * nu) * y * (H - y)
    chunk = max(200, int(0.5 * H * H / nu))
    prev = None
    steps = 0
    converged = False
    while steps < max_diffusion_times * H * H / nu:
        s.step(chunk)
        steps += chunk
        _, u = s.macros()
        un = u[0, 1:N + 1, 0, 0].astype(np.float64)
        # 1e-6 u_max is ~13 fp32 ulps at u_max = 0.05; a tighter test can stall on ulp flicker.
        # Each chunk damps the slowest transient mode by exp(-pi^2/2) = 0.007, so the residual
        # transient left in the result is < 1e-8 u_max.
        if prev is not None and np.max(np.abs(un - prev)) < 1e-6 * u_max:
            converged = True
            break
        prev = un
    err = un - ua
    return {
        "N": N, "lattice": lattice, "collision": collision, "tau": tau, "top": top,
        "steps": steps, "converged": converged,
        "err_l2": float(np.linalg.norm(err) / np.linalg.norm(ua)),
        "err_max": float(np.max(np.abs(err)) / np.max(ua)),
        "slip_mean": float(np.mean(err)),
        "slip_theory_bgk": float(g * (16 * (tau - 0.5) ** 2 - 3) / (24 * nu)) if top == "solid" else None,
        "y": y, "u": un, "u_exact": ua,
    }


FIT_H_MAX = 32   # fit the order only where the fp32 floor is negligible (see module doc)


def run(Ns=(8, 16, 32, 64), combos=None, tau=0.8, plot=True):
    from .. import backend
    prec = backend.precision()
    combos = combos or [("D3Q19", "bgk"), ("D3Q19", "rr"), ("D3Q27", "bgk"), ("D3Q27", "rr")]
    out = {"tau": tau, "precision": prec, "cases": {}, "magic_tau": {}}
    with Timer() as t:
        for top in ("solid", "freeslip"):
            Nt = Ns if top == "solid" else tuple(n // 2 for n in Ns)   # same H range for both
            for lat, col in combos:
                key = f"{lat}-{col}-{top}"
                rows = [run_case(N, lat, col, tau, top) for N in Nt]
                H = [N if top == "solid" else 2 * N for N in Nt]
                fit = [i for i, h in enumerate(H) if h <= FIT_H_MAX or prec == "f64"]
                errs = [r["err_l2"] for r in rows]
                out["cases"][key] = {
                    "N": list(Nt), "H": H, "err_l2": errs,
                    "err_max": [r["err_max"] for r in rows],
                    "order_l2": fit_order([Nt[i] for i in fit], [errs[i] for i in fit]),
                    "order_fit_N": [Nt[i] for i in fit],
                    "converged": all(r["converged"] for r in rows),
                    "slip_mean": [r["slip_mean"] for r in rows],
                    "slip_theory_bgk": [r["slip_theory_bgk"] for r in rows],
                    "_profile": rows[1],
                }
        # BGK exactness at the "magic" tau
        for lat in ("D3Q19", "D3Q27"):
            out["magic_tau"][lat] = {"tau": MAGIC_TAU, "N": [8, 16, 24, 32],
                                     "err_l2": [run_case(N, lat, "bgk", MAGIC_TAU, "solid")["err_l2"]
                                                for N in (8, 16, 24, 32)]}
    out["runtime_s"] = t.dt

    if plot:
        _plot(out)
    for c in out["cases"].values():
        c.pop("_profile", None)
    save_json("01_poiseuille", out, filename=f"results_{prec}.json")
    return out


def _plot(out):
    d = outdir("01_poiseuille")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    ax = axes[0]
    for top, ls in (("solid", "-"), ("freeslip", "--")):
        r = out["cases"][f"D3Q19-bgk-{top}"]["_profile"]
        yy = np.linspace(0, r["N"], 200)
        H = r["N"] if top == "solid" else 2 * r["N"]
        ue = r["u_exact"][0] / (r["y"][0] * (H - r["y"][0])) * yy * (H - yy)
        ax.plot(ue / ue.max(), yy / r["N"], "k" + ls, lw=1, label=f"analytical ({top} top)")
        ax.plot(r["u"] / ue.max(), r["y"] / r["N"], "o", ms=4, mfc="none", label=f"LBM N={r['N']} ({top})")
    ax.set_xlabel("u / u_max"); ax.set_ylabel("y / N"); ax.set_title("Profiles (D3Q19, BGK)")
    ax.legend(fontsize=7)

    for ax, top in zip(axes[1:], ("solid", "freeslip")):
        for key, c in out["cases"].items():
            if not key.endswith(top):
                continue
            ax.loglog(c["H"], c["err_l2"], "o-", ms=4, label=f"{key.rsplit('-', 1)[0]}  p={c['order_l2']:.2f}")
        c0 = out["cases"][f"D3Q19-bgk-{top}"]
        H = np.array(c0["H"], float)
        ax.loglog(H, c0["err_l2"][0] * (H / H[0]) ** -2, "k:", lw=1, label="slope 2")
        if out["precision"] == "f32":
            ax.axvspan(FIT_H_MAX * 1.2, H.max() * 1.3, color="0.92", zorder=0)
            ax.text(FIT_H_MAX * 1.3, ax.get_ylim()[1] * 0.5, "fp32 floor region", fontsize=7)
        ax.set_xlabel("H (equivalent full channel width, nodes)"); ax.set_ylabel("relative L2 error")
        ax.set_title(f"{top} top wall, tau={out['tau']}, {out['precision']}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / f"poiseuille_{out['precision']}.png")
    plt.close(fig)


if __name__ == "__main__":
    from .. import backend
    backend.init()
    res = run()
    for k, c in res["cases"].items():
        print(f"{k:28s} order={c['order_l2']:.3f}  err(N)={['%.2e' % e for e in c['err_l2']]}")
    print("magic tau:", {k: ['%.1e' % e for e in v["err_l2"]] for k, v in res["magic_tau"].items()})
