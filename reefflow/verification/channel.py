"""Verification 8 (Phase 7) - wall model: turbulent channel flow at Re_tau = 2000
on a deliberately coarse grid.

Set-up (lattice units): plane channel between no-slip walls at z = 0 and z = 2h,
periodic in x and y; domain 2 pi h x pi h x 2h with h = 16 cells (a near-wall
resolution comparable to the tank grids: first node at y+ = 62). Driven by a
constant body force g = u_tau^2 / h, so in the statistically steady state the
wall shear stress is exactly rho u_tau^2 by the momentum balance; what the wall
treatment controls is the flow rate the channel carries for that shear.
u_tau = 0.004 (lattice), nu = u_tau h / Re_tau. Production LES settings
(D3Q19, RR, C_s = 0.10). Initial state: log-law profile + random perturbations.
Spin-up 40 h/u_tau, averaging 40 h/u_tau.

Reference: bulk velocity from Dean's (1978) correlation Re_tau = 0.09 Re_b^0.88
(Re_b on the full height 2h): U_b / u_tau = 21.7 at Re_tau = 2000. The log law
U+ = ln(y+)/0.41 + 5.2 is shown for the profile.

Pre-declared acceptance (wall model ON): |U_b/u_tau - 21.7| / 21.7 <= 7 %, and the
mean profile within 10 % of the log law for 0.1 <= y/h <= 0.5 (outer log region).
Wall model OFF is run for comparison (no acceptance; expected to carry too much flow
because the under-resolved bounce-back wall exerts too little shear at a given
velocity, i.e. the channel must speed up until the shear balances the force).
"""
import math

import numpy as np

from ..domain import Domain, SOLID
from ..solver import LBMSolver
from ..wallmodel import WallModel, wall_nodes
from .common import Timer, outdir, plt, save_json

RE_TAU = 2000.0
UB_REF = 21.7
ACCEPT_UB = 0.07
ACCEPT_PROFILE = 0.10


def dean_ub(re_tau):
    """U_b / u_tau from Dean (1978): Re_tau = 0.09 Re_b^0.88, Re_b = U_b 2h / nu."""
    re_b = (re_tau / 0.09) ** (1 / 0.88)
    return re_b / (2 * re_tau)


def run_case(wall_model=True, h=16, utau=0.004, cs=0.10, lattice="D3Q19", t_spin=40, t_avg=40, seed=0, pert=0.1,
             re_tau=RE_TAU):
    nx, ny, nz = int(round(2 * math.pi * h)), int(round(math.pi * h)), 2 * h + 2
    d = Domain((nx, ny, nz), periodic=(True, True, False))
    d.set_face("z-", SOLID)
    d.set_face("z+", SOLID)
    nu = utau * h / re_tau
    ub_ref = dean_ub(re_tau)
    g = utau ** 2 / h
    s = LBMSolver(d, lattice=lattice, collision="rr", nu=nu, smagorinsky=cs, body_force=(g, 0, 0),
                  sponge_nu=np.zeros((nx, ny, nz)))
    wm = None
    if wall_model:
        nodes, normals = wall_nodes(d.flags)
        wm = WallModel(s, nodes, normals)
    # initial log-law profile with perturbations; wall at z = 0.5 and z = 2h + 0.5 (halfway)
    z = np.arange(nz) - 0.5
    yw = np.minimum(z, 2 * h - z)
    yp = np.clip(yw, 1e-3, None) * utau / nu
    up = np.where(yp < 11, yp, np.log(yp) / 0.41 + 5.2)
    rng = np.random.default_rng(seed)
    u0 = np.zeros((nx, ny, nz, 3))
    u0[..., 0] = (up * utau)[None, None, :]
    u0 += pert * utau * up.max() * rng.standard_normal(u0.shape) * (yw > 0)[None, None, :, None]
    u0[:, :, [0, -1]] = 0.0
    s.initialize(u=u0)
    T = h / utau
    n_spin, n_avg, every = int(t_spin * T), int(t_avg * T), 20
    acc = np.zeros(nz)
    nsamp = 0
    step = 0
    hist = []
    while step < n_spin + n_avg:
        s.step(every)
        step += every
        if wm is not None:
            wm.update()
        if step % 400 == 0 or step > n_spin:
            u = s.u.to_numpy()
            prof = u[..., 0].mean(axis=(0, 1))
            if step % 400 == 0:
                hist.append((step / T, float(prof[1:-1].mean() / utau)))
            if step > n_spin:
                acc += prof
                nsamp += 1
            if not np.all(np.isfinite(prof)):
                raise FloatingPointError("channel run diverged")
    U = acc / nsamp
    Ub = float(U[1:-1].mean() / utau)
    zc = z[1:h + 1]                                   # lower half, wall distance y = z
    Up = 0.5 * (U[1:h + 1] + U[-2:-h - 2:-1]) / utau  # average both halves
    ypl = zc * utau / nu
    loglaw = np.log(ypl) / 0.41 + 5.2
    sel = (zc / h >= 0.1) & (zc / h <= 0.5)
    prof_err = float(np.max(np.abs(Up[sel] / loglaw[sel] - 1)))
    out = {"wall_model": wall_model, "h": h, "Re_tau": re_tau, "utau": utau, "nu": nu, "y1_plus": float(ypl[0]),
           "Ub_over_utau": Ub, "Ub_ref": ub_ref, "Ub_rel_err": Ub / ub_ref - 1, "profile_max_rel_err_outer": prof_err,
           "steps": step, "_yplus": ypl.tolist(), "_Uplus": Up.tolist(), "_hist": hist}
    if wm is not None:
        out["wall_nu_extra_over_nu_mean"] = float(wm.extra_nu().mean() / nu)
    out["pass"] = bool(abs(Ub / ub_ref - 1) <= ACCEPT_UB and prof_err <= ACCEPT_PROFILE) if wall_model else None
    return out


def run(plot=True, **kw):
    res = {"cases": []}
    with Timer() as tm:
        for wmod in (True, False):
            res["cases"].append(run_case(wall_model=wmod, **kw))
    res["runtime_s"] = tm.dt
    if plot:
        d = outdir("08_channel")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        yy = np.logspace(0.5, math.log10(RE_TAU), 100)
        ax[0].semilogx(yy, np.log(yy) / 0.41 + 5.2, "k--", label="log law")
        for c in res["cases"]:
            lab = "wall model" if c["wall_model"] else "bounce-back only"
            ax[0].semilogx(c["_yplus"], c["_Uplus"], "o-", ms=3, label=f"{lab}: U_b/u_tau = {c['Ub_over_utau']:.1f}")
            t, ub = zip(*c["_hist"])
            ax[1].plot(t, ub, label=lab)
        ax[1].axhline(UB_REF, color="k", ls="--", label=f"Dean (1978): {UB_REF}")
        ax[0].set_xlabel("y+"); ax[0].set_ylabel("U+"); ax[0].legend(fontsize=8)
        ax[0].set_title(f"Channel Re_tau = {RE_TAU:.0f}, h = 16 cells, first node y+ = {res['cases'][0]['y1_plus']:.0f}",
                        fontsize=9)
        ax[1].set_xlabel("t u_tau / h"); ax[1].set_ylabel("U_b / u_tau"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "channel.png", dpi=110)
        plt.close(fig)
    for c in res["cases"]:
        for k in [k for k in c if k.startswith("_")]:
            c.pop(k)
    save_json("08_channel", res)
    return res


if __name__ == "__main__":
    import sys
    from .. import backend
    sys.stdout.reconfigure(encoding="utf-8")
    backend.init()
    for c in run()["cases"]:
        print(c)
