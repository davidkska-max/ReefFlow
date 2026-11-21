"""Verification 7 (Phase 2) - pump sources: delivered jet momentum flux and the
near-field turbulence they produce.

The tank model represents a powerhead as a body force F = J = rho Q^2 / A over a
short region at its outlet (DECISIONS D-202), and a return nozzle as a mass
source carrying the nozzle velocity (D-203). Both inject exactly the intended
force / mass / momentum (tests/test_sources.py). This case checks the
consequence the spec cares about: the jet each one produces carries momentum
flux J. It also measures the velocity fluctuations of the inflow-turbulence
setting (D-204) and the far-field decay of the jet.

Set-up: closed box 1.2 x 0.6 x 0.6 m, no-slip walls, free-slip lid, no sand or
rock; dx = 5 mm (Standard); production settings (D3Q19, RR, Cs = 0.10).
Spin-up 4 s, averaging 8 s (the jet front crosses the box in ~2 s). Source on
the left wall at mid-height, outlet 70 mm from the glass, aimed +x:
  powerhead       narrow-jet propeller archetype: 1500 GPH x 0.7, 32 mm outlet,
                  J = 1.55 N, modelled as 43.9 mm equivalent outlet (U_eq = 1.0 m/s);
                  archetype turbulence (default) and I = 0
  return          return-nozzle archetype (566 GPH from the pump curve, 19 mm,
                  2.1 m/s, J = 1.28 N; modelled as 40 mm equivalent), overflow
                  in the far back-right corner
  housing         the powerhead with a solid 70 mm housing behind the outlet
                  (informational: why housings are off by default, D-202)

Measured on planes x/D_eq downstream of the outlet:
    J(x) = integral of rho (U_x^2 + <u_x'^2> - <v_r'^2>) over the forward-flow part of the plane,
where the last term is the mean-pressure deficit of a thin turbulent shear layer,
p = -rho <v_r'^2> (Pope 2000, Sec. 5.2; <v_r'^2> taken as (<v'^2> + <w'^2>)/2),
and the centreline speed. The far-field decay is fitted over 8 <= x/D_eq <= 18 to the round-jet law
in momentum form, U_c = B sqrt(4 J / (pi rho)) / (x - x0), which needs no nozzle
diameter (for a top-hat jet sqrt(4J/(pi rho)) = U_0 D).

Pre-declared acceptance (powerhead default, powerhead I = 0, return):
    0.85 <= J(x)/J <= 1.20 for 3 <= x/D_eq <= 8.
Reported without acceptance: B, x0, centreline turbulence intensity.
"""
import json
import math

import numpy as np

from .. import project as PR
from .. import tank as TK
from ..run import run as run_tank
from .common import Timer, outdir, plt, save_json

BASE = {
    "reefflow_project": 1, "name": "Pump-source verification box", "display_units": "SI",
    "tank": {"external": {"length": "1.2 m", "width": "0.6 m", "height": "0.6 m"},
             "glass_thickness": "0 mm", "water_level_below_rim": "0 mm", "sand_depth": "0 mm"},
    "simulation": {"preset": "standard", "spinup": "4 s", "average": "8 s", "batches": 4},
}
POWERHEAD = {"name": "PH", "archetype": "narrow_jet_propeller", "rated_flow": "1500 GPH", "derating": 0.7,
             "position": ["0.07 m", "0.3 m", "0.3 m"], "aim": [1, 0, 0]}
RETURN = {"name": "R", "archetype": "return_nozzle", "rated_flow": "800 GPH", "static_head": "4 ft",
          "fittings_allowance": "1 ft", "position": ["0.07 m", "0.3 m", "0.3 m"], "aim": [1, 0, 0]}
OVERFLOW = {"corner": "back-right", "width": "150 mm", "depth": "150 mm", "crest_below_surface": "10 mm"}
CASES = [
    ("powerhead", POWERHEAD, {}, True),
    ("powerhead_I0", POWERHEAD, {"turbulence": 0.0}, True),
    ("return", RETURN, {}, True),
    ("powerhead_housing", POWERHEAD, {"housing": {"shape": "circle", "diameter": "70 mm", "length": "60 mm"}},
     False),
]
ACCEPT = (0.85, 1.20)
X_RANGE = (3.0, 8.0)
FIT = (8.0, 18.0)
RHO = 1023.0


def run_case(tag, pump, overrides, accept=True):
    raw = json.loads(json.dumps(BASE))
    raw["pumps"] = [dict(pump, **overrides)]
    if pump["archetype"] == "return_nozzle":
        raw["overflow"] = OVERFLOW
    P = PR.parse(raw)
    model = TK.build(P, P.dx())
    r = run_tank(model, outdir("07_powerhead") / tag, progress=print)
    f = r.pop("_fields")
    pr = model.pumps[0]
    D = pr.outlet_eq.diameter
    J0 = pr.pump.momentum_flux
    X, Y, Z = (model.node_coords(a) for a in range(3))
    fl = model.fluid_mask
    x0p = pr.pump.position[0]
    j0 = int(np.argmin(abs(Y - pr.pump.position[1])))
    k0 = int(np.argmin(abs(Z - pr.pump.position[2])))
    Ux = f["mean_u"][..., 0]
    Rxx = f["reynolds"][..., 0]
    Rrr = 0.5 * (f["reynolds"][..., 1] + f["reynolds"][..., 2])
    xs, Js, Uc, TI = [], [], [], []
    for i in range(model.shape[0]):
        xd = (X[i] - x0p) / D
        if xd < 0.5 or X[i] > X[-2] - 3 * D:
            continue
        m = fl[i] & (Ux[i] > 0)
        Js.append(float(RHO * np.sum((Ux[i] ** 2 + Rxx[i] - Rrr[i])[m]) * model.dx ** 2 / J0))
        xs.append(xd)
        uc = float(Ux[i, j0, k0])
        Uc.append(uc)
        TI.append(float(math.sqrt(max(Rxx[i, j0, k0], 0)) / uc) if uc > 0 else float("nan"))
    xs, Js, Uc, TI = map(np.array, (xs, Js, Uc, TI))
    win = (xs >= X_RANGE[0]) & (xs <= X_RANGE[1])
    fit = (xs >= FIT[0]) & (xs <= FIT[1])
    UmD = math.sqrt(4 * J0 / (math.pi * RHO))
    a, b = np.polyfit(xs[fit] * D, 1.0 / Uc[fit], 1)
    out = {"tag": tag, "kind": pr.pump.kind, "turbulence": pr.pump.turbulence,
           "housing": pr.pump.housing is not None, "D_eq_m": D, "U_eq_ms": pr.u_eq, "J_N": J0,
           "J_ratio_window": [float(Js[win].min()), float(Js[win].max())],
           "J_ratio_at": {f"x{q}D": float(np.interp(q, xs, Js)) for q in (1, 2, 4, 8)},
           "B": float(1.0 / (a * UmD)), "x0_over_Deq": float(-b / a / D),
           "centreline": {f"x{q}D": {"Uc_over_Ueq": float(np.interp(q, xs, Uc)) / pr.u_eq,
                                     "TI": float(np.interp(q, xs, TI))} for q in (1, 2, 4, 8)},
           "accept": accept, "mass_rel_drift": r["mass_rel_drift"], "wall_time_s": r["wall_time_s"],
           "_x": xs.tolist(), "_J": Js.tolist(), "_Uc": (Uc / pr.u_eq).tolist(), "_TI": TI.tolist(),
           "_UmD_over_D": UmD / D / pr.u_eq}
    out["pass"] = bool(ACCEPT[0] <= Js[win].min() and Js[win].max() <= ACCEPT[1]) if accept else None
    return out


def run(plot=True, cases=None, save=True):
    out = {"acceptance": {"J_ratio": ACCEPT, "x_over_D": X_RANGE}, "fit_window": FIT, "cases": []}
    with Timer() as tm:
        for tag, pump, ov, acc in CASES:
            if cases is None or tag in cases:
                out["cases"].append(run_case(tag, pump, ov, acc))
    out["runtime_s"] = tm.dt
    if plot:
        d = outdir("07_powerhead")
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        for c in out["cases"]:
            lab = f"{c['tag']} (I = {c['turbulence']:g})"
            ax[0].plot(c["_x"], c["_J"], label=lab)
            l, = ax[1].plot(c["_x"], c["_Uc"], label=lab)
            if c["tag"] != "powerhead_housing":
                xx = np.linspace(FIT[0], max(c["_x"]), 50)
                ax[1].plot(xx, c["B"] * c["_UmD_over_D"] / (xx - c["x0_over_Deq"]), ":", color=l.get_color())
            ax[2].plot(c["_x"], c["_TI"], label=lab)
        ax[0].axhspan(*ACCEPT, color="0.9", zorder=0)
        ax[0].axvspan(*X_RANGE, color="0.95", zorder=0)
        ax[0].axhline(1.0, color="k", lw=0.8)
        ax[0].set_ylabel("J(x) / (rho Q² / A)")
        ax[1].set_ylabel("U_c / U_eq  (dotted: fitted 1/x law)")
        ax[2].set_ylabel("sigma_x / U_c on the axis")
        ax[2].set_ylim(0, 0.6)
        for a in ax:
            a.set_xlabel("x / D_eq")
            a.legend(fontsize=7)
        ax[0].set_title("jet momentum flux (forward-flow region)", fontsize=9)
        ax[1].set_title("centreline velocity", fontsize=9)
        ax[2].set_title("centreline turbulence intensity", fontsize=9)
        fig.tight_layout()
        fig.savefig(d / "powerhead.png", dpi=110)
        plt.close(fig)
    for c in out["cases"]:
        for k in [k for k in c if k.startswith("_")]:
            c.pop(k)
    if save:
        save_json("07_powerhead", out)
    return out


if __name__ == "__main__":
    from .. import backend
    backend.init()
    for c in run()["cases"]:
        print(c["tag"], "J/J0", c["J_ratio_window"], "pass", c["pass"], "B", round(c["B"], 2),
              "x0/D", round(c["x0_over_Deq"], 2))
