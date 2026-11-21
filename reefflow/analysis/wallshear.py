"""Wall shear stress on the sand bed and upward-facing rock surfaces, by wall
function (DECISIONS D-006, user-approved), and the detritus-settling map.

Why a wall function: at tank resolutions the first fluid node sits at y+ ~ 10-25
and bounce-back gives a near-laminar gradient across one cell, which badly
underpredicts wall shear (PLAN C2). Instead the tangential speed is sampled at a
fixed height above the surface (15 mm, the same on every grid; y+ ~ 30-80 for
the demo, i.e. in the log layer) and Spalding's law of the wall, valid
continuously from the viscous sublayer to the log layer,

    y+ = u+ + e^{-kB} [ e^{k u+} - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6 ],
    k = 0.41, B = 5.2, u+ = u / u_tau, y+ = y u_tau / nu,

is solved for the friction velocity u_tau; tau_w = rho u_tau^2.

Speed used: u_eff = sqrt(U_t^2 + <u_t'^2>), the RMS tangential speed (mean
plus resolved fluctuations), because shear stress responds to the instantaneous
velocity; using only the mean velocity would understate tau in fluctuating flow.
Smooth-wall law: sand roughness (k_s ~ grain size ~ 1 mm, k_s+ ~ 3-10,
transitionally rough) is neglected; real sand shear is somewhat higher (A-304).

Detritus map: settling/accumulation likely where tau_w < tau_crit (default
0.02 Pa, a starting assumption for loose organic flocs; A-305). Sand movement
where tau_w > tau_sand (default 0.5 Pa: Shields theta_c ~ 0.04 for ~0.7 mm
aragonite, rho_s = 2850 kg/m^3).
"""
from __future__ import annotations

import numpy as np

KAPPA, B_LOG = 0.41, 5.2
NU = 0.94e-6
SAMPLE_HEIGHT = 0.015


def spalding_yplus(up):
    ku = np.minimum(KAPPA * up, 600.0)
    return up + np.exp(-KAPPA * B_LOG) * (np.exp(ku) - 1 - ku - ku ** 2 / 2 - ku ** 3 / 6)


def friction_velocity(u, y, nu=NU, iters=80):
    """Vectorised bisection (in log u_tau) of Spalding's law. u, y arrays (m/s, m)."""
    u = np.asarray(u, float)
    y = np.broadcast_to(np.asarray(y, float), u.shape)
    lo = np.full(u.shape, 1e-7)
    hi = np.maximum(u, 1e-6) * 1.0 + 1e-6       # u_tau < u always (u+ > 1 beyond y+ ~ 1)
    hi = np.maximum(hi, np.sqrt(nu * np.maximum(u, 1e-12) / y) * 1.01)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        f = y * mid / nu - spalding_yplus(u / mid)   # increasing in u_tau
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    out = np.sqrt(lo * hi)
    return np.where(u > 0, out, 0.0)


def _interp_up(field, fluid, base_nodes, h, dx):
    nf = h / dx - 0.5
    n0 = int(np.floor(nf))
    w = nf - n0
    i, j, k = base_nodes[:, 0], base_nodes[:, 1], base_nodes[:, 2] + n0
    nz = field.shape[2]
    ok = (k + 1 < nz)
    ok[ok] = fluid[i[ok], j[ok], k[ok]] & fluid[i[ok], j[ok], k[ok] + 1]
    val = np.full(len(base_nodes), np.nan)
    val[ok] = (1 - w) * field[i[ok], j[ok], k[ok]] + w * field[i[ok], j[ok], k[ok] + 1]
    return val


def horizontal_shear(run, rho=1023.0, h=SAMPLE_HEIGHT, tau_crit=0.02, tau_sand=0.5) -> dict:
    """Wall shear on upward-facing surfaces: the sand bed (2D map) and rock tops (points)."""
    from ..tank import TAGS
    m = run.model
    fl = m.fluid_mask
    U = run.mean_u
    R = run.reynolds if run.reynolds is not None else np.zeros(U.shape[:3] + (6,))
    ueff = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2 + np.clip(R[..., 0] + R[..., 1], 0, None))
    nx, ny, nz = m.shape
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    sand_base = np.stack([I.ravel(), J.ravel(), np.ones(I.size, int)], axis=1)
    u_s = _interp_up(ueff, fl, sand_base, h, m.dx).reshape(nx, ny)
    tau_sand_map = rho * friction_velocity(np.nan_to_num(u_s), h) ** 2
    tau_sand_map = np.where(np.isfinite(u_s), tau_sand_map, np.nan)
    below = np.zeros_like(fl)
    below[:, :, 1:] = m.tags[:, :, :-1] == TAGS["rock"]
    rb = np.argwhere(fl & below)
    u_r = _interp_up(ueff, fl, rb, h, m.dx)
    ok = np.isfinite(u_r)
    tau_rock = rho * friction_velocity(u_r[ok], h) ** 2
    sand_vals = tau_sand_map[np.isfinite(tau_sand_map)]
    return {
        "sand_tau_map": tau_sand_map, "rock_top_nodes": rb[ok], "rock_top_tau": tau_rock,
        "report": {
            "sample_height_m": h, "tau_crit_Pa": tau_crit, "tau_sand_move_Pa": tau_sand,
            "sand_tau_mean_Pa": float(sand_vals.mean()), "sand_tau_median_Pa": float(np.median(sand_vals)),
            "sand_tau_p10_Pa": float(np.percentile(sand_vals, 10)), "sand_tau_p90_Pa": float(np.percentile(sand_vals, 90)),
            "sand_area_settling_fraction": float(np.mean(sand_vals < tau_crit)),
            "sand_area_moving_fraction": float(np.mean(sand_vals > tau_sand)),
            "rock_top_tau_median_Pa": float(np.median(tau_rock)) if len(tau_rock) else float("nan"),
            "rock_top_settling_fraction": float(np.mean(tau_rock < tau_crit)) if len(tau_rock) else float("nan"),
            "sand_area_sampled_m2": float(len(sand_vals) * m.dx ** 2),
        },
    }
