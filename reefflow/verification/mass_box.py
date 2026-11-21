"""Verification 5 - closed box with a momentum-source jet: mass conservation.

A powerhead is modelled as a momentum source (body force over a small
actuator volume), not as a mass inlet, so in a closed box:
  (a) total mass must be constant (bounce-back walls are exactly mass
      conserving; only fp32 round-off remains);
  (b) the time-averaged net volume flux through any cross-section must vanish
      (whatever the jet pushes forward must return);
  (c) before any wall reaction arrives, total momentum must grow at exactly
      sum(F) per step (checks the forcing scheme).

Set-up (lattice units): 128 x 64 x 64 box, all walls no-slip. Actuator:
cylinder of radius 4 cells, 4 cells long, axis along x at mid-height, force
density F0 in +x. Production configuration: recursive-regularized collision +
Smagorinsky (Cs = 0.17), molecular nu = 1e-4.
"""
import numpy as np

from ..domain import Domain, SOLID
from ..solver import LBMSolver
from .common import Timer, outdir, plt, save_json


def run_case(lattice="D3Q19", shape=(128, 64, 64), F0=3e-4, nu=1e-4, cs=0.17, n_steps=40_000,
             spinup=10_000, sample_every=50, planes=(40, 64, 96)):
    nx, ny, nz = shape
    d = Domain(shape)
    for f in ("x-", "x+", "y-", "y+", "z-", "z+"):
        d.set_face(f, SOLID)
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    yc, zc = (ny - 1) / 2, (nz - 1) / 2
    act = ((J - yc) ** 2 + (K - zc) ** 2 <= 16.0) & (I >= 16) & (I < 20)
    F = np.zeros(shape + (3,), np.float32)
    F[act, 0] = F0
    s = LBMSolver(d, lattice=lattice, collision="rr", nu=nu, smagorinsky=cs, force_field=True)
    s.set_force_field(F)
    F_total = float(F[..., 0].sum())

    # (c) momentum injection before wall reaction (acoustic signal: c_s = 0.577 cell/step)
    w = s.lat.w
    cx = s.lat.c[:, 0].astype(np.float64)
    mom = []
    for n in range(1, 11):
        s.step(1)
        g = s.f.to_numpy().astype(np.float64)
        mom.append(np.tensordot(cx, g, axes=(0, 0))[d.flags == 0].sum())
    mom = np.array(mom)
    mom_err = float(np.max(np.abs(mom / (np.arange(1, 11) * F_total) - 1.0)))

    m0 = s.total_mass()
    t_hist, m_hist = [], []
    net = {p: [] for p in planes}
    fwd = {p: [] for p in planes}
    while s.time_step < n_steps:
        s.step(sample_every)
        if s.time_step % 1000 < sample_every:
            t_hist.append(s.time_step)
            m_hist.append(s.total_mass())
        if s.time_step >= spinup:
            rho = s.rho.to_numpy(); u = s.u.to_numpy()
            for p in planes:
                fl = d.flags[p] == 0
                q = (rho[p] * u[p, :, :, 0])[fl]
                net[p].append(q.sum()); fwd[p].append(q[q > 0].sum())
    m_hist = np.array(m_hist)
    if not np.all(np.isfinite(m_hist)):
        raise FloatingPointError("mass-box run diverged")
    flux = {}
    for p in planes:
        n_ = np.array(net[p]); f_ = np.array(fwd[p])
        flux[p] = {"mean_net_over_fwd": float(n_.mean() / f_.mean()),
                   "rms_net_over_fwd": float(np.sqrt(np.mean(n_**2)) / f_.mean()),
                   "mean_fwd_flux": float(f_.mean())}
    _, u = s.macros()
    jet_u = float(np.max(u[..., 0]))
    return {
        "lattice": lattice, "shape": shape, "F0": F0, "F_total": F_total, "steps": int(s.time_step),
        "momentum_injection_max_rel_err": mom_err,
        "mass_initial": m0, "mass_max_rel_drift": float(np.max(np.abs(m_hist / m0 - 1.0))),
        "flux": flux, "max_jet_velocity": jet_u,
        "pass": bool(np.max(np.abs(m_hist / m0 - 1)) < 1e-5 and
                     all(abs(v["mean_net_over_fwd"]) < 1e-3 for v in flux.values()) and mom_err < 1e-4),
        "_t": t_hist, "_m": m_hist / m0 - 1.0, "_u": u[:, :, nz // 2, 0],
    }


def run(lattices=("D3Q19", "D3Q27"), plot=True, **kw):
    out = {"cases": []}
    with Timer() as tm:
        for lat in lattices:
            out["cases"].append(run_case(lat, **kw))
    out["runtime_s"] = tm.dt
    if plot:
        d = outdir("05_mass_box")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
        for c in out["cases"]:
            axes[0].plot(c["_t"], c["_m"], label=c["lattice"])
        axes[0].set_xlabel("time step"); axes[0].set_ylabel("M(t)/M(0) - 1")
        axes[0].set_title("Total mass drift"); axes[0].legend(fontsize=7)
        c = out["cases"][0]
        im = axes[1].imshow(c["_u"].T, origin="lower", cmap="RdBu_r",
                            vmin=-abs(c["_u"]).max(), vmax=abs(c["_u"]).max())
        axes[1].set_title(f"u_x, mid-plane, step {c['steps']} ({c['lattice']})")
        fig.colorbar(im, ax=axes[1], shrink=0.8)
        fig.tight_layout(); fig.savefig(d / "mass_box.png"); plt.close(fig)
    for c in out["cases"]:
        for k in ("_t", "_m", "_u"):
            c.pop(k)
    save_json("05_mass_box", out)
    return out


if __name__ == "__main__":
    from .. import backend
    backend.init()
    for c in run()["cases"]:
        print(c["lattice"], "mass drift", c["mass_max_rel_drift"], "mom err", c["momentum_injection_max_rel_err"],
              {p: round(v["mean_net_over_fwd"], 6) for p, v in c["flux"].items()}, "pass", c["pass"])
