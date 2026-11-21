"""Verification 9 (Phase 4 gate): photometric tracer against analytical solutions.

Results (JSON + PNG) go to verification/results/09_light/<case>/. Run all:
    python -m reefflow.light.verification          # full sizes (GPU: a few minutes)
Fast reduced versions run in tests/test_light_verification.py.

Statistics. Photons carry equal weight (apart from Beer-Lambert attenuation), so a
tally cell expecting lambda photons has Poisson noise sqrt(lambda). For a map with
analytical expectation, z = (n - lambda) / sqrt(lambda) per pixel and
chi^2/dof = mean(z^2) ~ 1 +- sqrt(2/dof) if the tracer is unbiased.

PRE-DECLARED ACCEPTANCE CRITERIA (written before the first run, DECISIONS D-110)

1  Inverse-square law. Isotropic point source (100 umol/s) in air; horizontal detector
   planes 0.1, 0.2, 0.4, 0.8 m below it, 10 mm pixels over +-0.4 m; probes of half-width
   0.05 h on the axis. Analytic pixel value E = Phi Omega_pixel / (4 pi A) (exact solid
   angle of a rectangle). Pass if, for every plane, 0.9 <= chi^2/dof <= 1.1 over pixels
   with lambda >= 50, max |z| < 5.5 over all planes, and the exponent n fitted to the probe
   values E(h) ~ h^n is -2.000 +- 0.015.
   AMENDMENT (after the first run, recorded in docs/drafts/phase4_decisions.md D-412): probes
   scaled with h intercept the same photon cone at every h, so their exponent is -2 by
   construction. Added: fixed 20 mm half-width probes, pass if their fitted exponent is within
   0.015 of the exponent of the exact probe averages. (A first version also required the exact
   exponent to be within 0.005 of -2; that was an error in the reference, not the tracer: a
   40 mm probe at 0.1 m is not a point, its exact exponent over 0.1-0.8 m is -1.982. Dropped.)
2  Lambertian emitter. 300 x 50 mm Lambertian rectangle (100 umol/s) facing down; planes
   0.05, 0.15, 0.40 m below, 5 mm pixels. Analytic E = M F (M = Phi/A_src, F the
   differential-element-to-parallel-rectangle configuration factor; pixel averages by
   4x4 Gauss-Legendre). Pass: same chi^2/dof and |z| limits as case 1, and plane flux
   within the window within 0.5 % of analytic.
3  Refraction and Fresnel.
   3a Collimated beam onto a flat water surface (n = 1.34) at incidence 0, 15, 30, 45, 60,
      75, 85 deg. Pass if the transmitted fraction is within 4 binomial sigma (+ 2e-6) of
      1 - R_unpolarised(theta) and the mean refracted angle within 0.01 deg of Snell.
   3b The refracted beam hits a 12.7 mm glass side pane (n = 1.5, clear, air outside) at
      incidence 75.2, 55.1, 47.4, 43.9, 42.0 deg (air incidence 20, 50, 65, 75, 85 deg).
      Pass if the fraction lost through the glass is within 4 sigma (+ 1e-4) of the
      incoherent slab value T = (1-R1)(1-R2)/(1-R1 R2) (R2 = 1 beyond the critical angle
      48.27 deg, i.e. exactly zero transmission under total internal reflection).
4  Beer-Lambert.
   4a Collimated normal beam, Kd = 0.5, 1, 2 1/m: downward flux through planes at depth z
      equals Phi T0 exp(-Kd z) within 4 binomial sigma of T0 (+ 1e-5 relative).
   4b Large Lambertian source 50 mm above a flat surface, Kd = 1 1/m, black floor/walls:
      E_d(z) at the centre (0.5 x 0.5 m window) equals
      M int_0^{pi/2} T(theta_a) exp(-Kd z / cos theta_w) 2 cos theta_a sin theta_a dtheta_a
      (Snell + Fresnel + path length z / cos theta_w) within 4 sigma + 0.1 % at
      z = 0.05, 0.1, 0.2, 0.4, 0.8 m.
   4c Spectral mode, two-channel (450 + 660 nm) collimated beam, Kd(lambda) = pure water
      + 0.03 1/m: flux ratio F(z)/F(0) = sum_b f_b exp(-Kd_b z) within 4 sigma (+ 1e-5).
5  Ray-count convergence (demo tank, rock, both fixtures). N = 2^19 ... 2^24 photons,
   4 seeds each. Empirical seed-to-seed relative noise of sand cells (5 mm) and of the
   150 mm plan map (over cells lit above half the mean; pooled RMS, see D-413). Pass if the fitted slope of
   log(noise) vs log(N) is -0.50 +- 0.03 for both, and the internal estimate sqrt(sum w^2)/sum w
   agrees with the empirical noise within 15 % at every N. (Added before case 5 was first run,
   with D-410:) the same two conditions for the node-stack meter tally on the node plane nearest
   150 mm depth, whose internal estimate is 1/sqrt(sum w).
6  Calibration round trip. Truth: led_puck with beam 70 deg and 150 umol/s; model starts
   from the archetype (90 deg, 180 umol/s); synthetic readings carry 2 % Gaussian noise.
   6a Manufacturer chart in air (8 points, 12 and 18 in below, 0-12 in off-axis), fit
      [scale, beam_angle]: pass if |beam - 70| <= 3 deg, |scale / 0.8333 - 1| <= 3 %,
      RMS residual <= 3.5 %.
   6b Own meter in the demo tank (6 points, in water, not immersion corrected, factor 1.08),
      fit [scale] with the true beam: pass if |scale / 0.8333 - 1| <= 2 % and RMS <= 3 %.
      Reported (no criterion): the scale obtained if the immersion metadata is ignored.
7  Rock geometry (SDF). Sphere R = 50 mm, 150 mm above black sand, collimated vertical beam,
   Kd = 0, black walls and rock. For the STL-derived SDF (h = 2.5 mm) and the voxel-derived
   SDF (CFD dx = 5 mm, smoothed): pass if the flux intercepted by the sphere is within
   1 % (STL) / 3 % (voxel) of E0 pi R^2, the sand shadow area (cells < E0/2) within 3 % /
   6 % of pi R^2, the area-weighted mean of E_patch / (E0 cos theta) over patches with
   cos theta > 0.2 within 2 % / 5 % of 1, and the RMS angle between SDF-gradient and exact
   normals on the surface below 2 deg / 8 deg.
8  Energy closure. Demo tank with clear, black and coralline walls, bounces on: the sinks
   sum to the emitted weight within 1e-5.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np

from .. import backend
from ..verification.common import outdir, plt, save_json
from . import config as CF
from .fixtures import load_library
from .tracer import Grid, build_scene, trace

ROOT = "09_light"
N_W = 1.34
N_G = 1.5


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _ensure_backend():
    if backend.current() is None:
        backend.init()


def _box_tank(L, W, zsand, zs, rim=None, glass=0.0127):
    return SimpleNamespace(length=L, width=W, sand_depth=zsand, water_surface=zs, height=rim or zs,
                           water_column=zs - zsand, glass=glass)


def _cfg(fixtures, tank, **kw):
    raw = {"fixtures": fixtures, "map_depths": kw.pop("map_depths", []), **kw}
    if not raw["map_depths"]:
        raw["map_depths"] = [0.5 * tank.water_column]
    return CF.parse_lighting(raw, tank)


def fresnel_unpolarised(theta_i, n1, n2):
    ci = np.cos(theta_i)
    s2 = (n1 / n2) ** 2 * (1 - ci ** 2)
    if np.any(s2 >= 1):
        out = np.ones_like(np.asarray(theta_i, float))
        ok = s2 < 1
        out[ok] = fresnel_unpolarised(np.asarray(theta_i)[ok], n1, n2)
        return out
    ct = np.sqrt(1 - s2)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return 0.5 * (rs ** 2 + rp ** 2)


def rect_solid_angle(x1, x2, y1, y2, h):
    """Solid angle of the rectangle [x1,x2] x [y1,y2] in a plane at distance h from the point
    above the origin."""
    def F(a, b):
        return np.arctan2(a * b, h * np.sqrt(a * a + b * b + h * h))
    return F(x2, y2) - F(x1, y2) - F(x2, y1) + F(x1, y1)


def parallel_rect_factor(x1, x2, y1, y2, h):
    """Configuration factor from a differential area (at the origin of its plane) to a parallel
    rectangle [x1,x2] x [y1,y2] at distance h (facing each other)."""
    def f(a, b):
        A, B = a / h, b / h
        sa, sb = np.sqrt(1 + A * A), np.sqrt(1 + B * B)
        return (A / sa * np.arctan(B / sa) + B / sb * np.arctan(A / sb)) / (2 * np.pi)
    return f(x2, y2) - f(x1, y2) - f(x2, y1) + f(x1, y1)


def _zstats(n, lam, mask):
    z = (n[mask] - lam[mask]) / np.sqrt(lam[mask])
    return float(np.mean(z ** 2)), float(np.max(np.abs(z))), int(mask.sum())


def _save(name, data, fig=None):
    d = outdir(f"{ROOT}/{name}")
    if fig is not None:
        fig.tight_layout()
        fig.savefig(d / f"{name}.png")
        plt.close(fig)
    save_json(f"{ROOT}/{name}", data)
    return data


def _air_scene(cfg, planes_z, half, pix, probes=None):
    sc = build_scene(cfg, None, box=(1, 1, -1, 0, 0), plan=False, probes=probes, has_water=False, air_floor=-5.0)
    n = int(round(2 * half / pix))
    sc.plan_z = np.asarray(planes_z, float)
    sc.plan_grid = (-half, -half, pix, n, n)
    return sc


# ----------------------------------------------------------------------
# 1 inverse square
# ----------------------------------------------------------------------
def case_inverse_square(rays=5e7, save=True):
    _ensure_backend()
    H, phi = 1.0, 100.0
    tank = _box_tank(1, 1, -1, 0)
    cfg = _cfg([{"name": "pt", "archetype": "isotropic_point", "ppf": phi, "position": [0, 0], "height": H}], tank)
    hs = np.array([0.1, 0.2, 0.4, 0.8])
    half, pix = 0.4, 0.01
    probes = [[0, 0, H - h, 0.05 * h] for h in hs] + [[0, 0, H - h, 0.02] for h in hs]
    sc = _air_scene(cfg, H - hs, half, pix, probes)
    N = int(rays)
    tal = trace(sc, N, seed=11)
    n = int(round(2 * half / pix))
    edges = -half + pix * np.arange(n + 1)
    planes = []
    chi_ok, zmax_all = True, 0.0
    for l, h in enumerate(hs):
        Om = rect_solid_angle(edges[:-1, None], edges[1:, None], edges[None, :-1], edges[None, 1:], h)
        lam = N * Om / (4 * np.pi)
        cnt = tal.plan_w[l]
        chi, zmax, dof = _zstats(cnt, lam, lam >= 50)
        zmax_all = max(zmax_all, zmax)
        chi_ok &= 0.9 <= chi <= 1.1
        planes.append({"h_m": h, "chi2_dof": chi, "max_abs_z": zmax, "pixels": dof})
    # probes (fixed size)
    Ef = tal.probe_w[4:] * tal.ppf_per_photon / 0.04 ** 2
    Om_f = np.array([rect_solid_angle(-0.02, 0.02, -0.02, 0.02, h) for h in hs])
    Efa = phi * Om_f / (4 * np.pi) / 0.04 ** 2
    nf_mc = float(np.polyfit(np.log(hs), np.log(Ef), 1)[0])
    nf_an = float(np.polyfit(np.log(hs), np.log(Efa), 1)[0])
    # probes (scaled with h)
    Ep = tal.probe_w[:4] * tal.ppf_per_photon / (0.1 * hs) ** 2
    Om_p = np.array([rect_solid_angle(-0.05 * h, 0.05 * h, -0.05 * h, 0.05 * h, h) for h in hs])
    Ea = phi * Om_p / (4 * np.pi) / (0.1 * hs) ** 2
    sig = Ea / np.sqrt(N * Om_p / (4 * np.pi))
    nfit = float(np.polyfit(np.log(hs), np.log(Ep), 1)[0])
    ok = chi_ok and zmax_all < 5.5 and abs(nfit + 2) <= 0.015 and abs(nf_mc - nf_an) <= 0.015
    out = {"case": "1 inverse square", "rays": N, "planes": planes, "max_abs_z": zmax_all,
           "fixed_probe_E": Ef.tolist(), "fixed_probe_E_analytic": Efa.tolist(),
           "fixed_probe_exponent_mc": nf_mc, "fixed_probe_exponent_analytic": nf_an,
           "probe_E": Ep.tolist(), "probe_E_analytic": Ea.tolist(), "probe_z": ((Ep - Ea) / sig).tolist(),
           "E_h2_over_point_law": (Ep * hs ** 2 / (phi / (4 * np.pi))).tolist(),
           "fitted_exponent": nfit, "rays_per_s": tal.rays_per_second, "pass": bool(ok)}
    fig = None
    if save:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        for l, h in enumerate(hs):
            c = tal.plan_w[l][n // 2 - 1:n // 2 + 1].mean(0) * tal.ppf_per_photon / pix ** 2
            yc = 0.5 * (edges[1:] + edges[:-1])
            Ean = phi / (4 * np.pi) * h / (h * h + yc ** 2) ** 1.5
            axs[0].semilogy(yc, c, ".", ms=3, label=f"MC h = {h} m")
            axs[0].semilogy(yc, Ean, "k-", lw=0.8)
        axs[0].set_xlabel("y (m), row through the axis"); axs[0].set_ylabel("E (µmol m$^{-2}$ s$^{-1}$)")
        axs[0].set_title("Plane irradiance: MC (dots) vs (Φ/4π) h/(h²+r²)^1.5 (lines)", fontsize=8)
        axs[0].legend(fontsize=7)
        axs[1].loglog(hs, Ef, "o", label="MC, fixed 40 mm probe")
        axs[1].loglog(hs, Efa, "k-", label="analytic (exact solid angle)")
        axs[1].set_xlabel("h (m)"); axs[1].set_ylabel("on-axis E (µmol m$^{-2}$ s$^{-1}$)")
        axs[1].set_title(f"fitted exponent {nf_mc:.4f} (analytic {nf_an:.4f})", fontsize=8)
        axs[1].legend(fontsize=7)
    return _save("01_inverse_square", out, fig) if save else out


# ----------------------------------------------------------------------
# 2 Lambertian rectangle
# ----------------------------------------------------------------------
def case_lambertian(rays=5e7, save=True):
    _ensure_backend()
    H, phi, a, b = 1.0, 100.0, 0.30, 0.05
    tank = _box_tank(1, 1, -1, 0)
    cfg = _cfg([{"name": "pan", "archetype": "lambertian_panel", "ppf": phi, "position": [0, 0], "height": H,
                 "footprint": {"length": f"{a} m", "width": f"{b} m"},
                 "emitters": {"nx": 1, "ny": 1, "size": [f"{a} m", f"{b} m"]}}], tank)
    hs = np.array([0.05, 0.15, 0.40])
    half, pix = 0.40, 0.005
    sc = _air_scene(cfg, H - hs, half, pix)
    N = int(rays)
    tal = trace(sc, N, seed=12)
    n = int(round(2 * half / pix))
    edges = -half + pix * np.arange(n + 1)
    gx, gw = np.polynomial.legendre.leggauss(4)
    M = phi / (a * b)
    planes, chi_ok, zmax_all, flux_ok = [], True, 0.0, True
    for l, h in enumerate(hs):
        Eav = np.zeros((n, n))
        for xi, wx in zip(gx, gw):
            for yi, wy in zip(gx, gw):
                px = 0.5 * (edges[:-1] + edges[1:]) + 0.5 * pix * xi
                py = 0.5 * (edges[:-1] + edges[1:]) + 0.5 * pix * yi
                X, Y = np.meshgrid(px, py, indexing="ij")
                F = parallel_rect_factor(-a / 2 - X, a / 2 - X, -b / 2 - Y, b / 2 - Y, h)
                Eav += 0.25 * wx * wy * M * F
        lam = Eav * pix ** 2 / tal.ppf_per_photon
        cnt = tal.plan_w[l]
        chi, zmax, dof = _zstats(cnt, lam, lam >= 50)
        zmax_all = max(zmax_all, zmax)
        chi_ok &= 0.9 <= chi <= 1.1
        fl_mc, fl_an = cnt.sum(), lam.sum()
        flux_ok &= abs(fl_mc / fl_an - 1) < 0.005
        planes.append({"h_m": h, "chi2_dof": chi, "max_abs_z": zmax, "pixels": dof,
                       "flux_ratio_mc_over_analytic": float(fl_mc / fl_an),
                       "on_axis_E_analytic": float(M * parallel_rect_factor(-a / 2, a / 2, -b / 2, b / 2, h))})
        if l == 0:
            keep = (cnt, lam)
    ok = chi_ok and zmax_all < 5.5 and flux_ok
    out = {"case": "2 Lambertian rectangle", "rays": N, "planes": planes, "max_abs_z": zmax_all, "pass": bool(ok)}
    fig = None
    if save:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        xc = 0.5 * (edges[1:] + edges[:-1])
        for l, h in enumerate(hs):
            pass
        cnt, lam = keep
        k = tal.ppf_per_photon / pix ** 2
        axs[0].plot(xc, cnt[:, n // 2] * k, ".", ms=3, label="MC, y = 0")
        axs[0].plot(xc, lam[:, n // 2] * k, "k-", lw=0.8, label="analytic (M·F)")
        axs[0].plot(xc, cnt[n // 2, :] * k, ".", ms=3, label="MC, x = 0")
        axs[0].plot(xc, lam[n // 2, :] * k, "k--", lw=0.8, label="analytic")
        axs[0].set_xlabel("position (m)"); axs[0].set_ylabel("E (µmol m$^{-2}$ s$^{-1}$)")
        axs[0].set_title(f"Lambertian 300x50 mm panel, plane {hs[0]} m below", fontsize=8)
        axs[0].legend(fontsize=7)
        z = (cnt - lam) / np.sqrt(np.maximum(lam, 1e-9))
        im = axs[1].imshow(np.where(lam >= 50, z, np.nan).T, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4,
                           extent=[-half, half, -half, half])
        axs[1].set_title("z-score per pixel (h = 0.05 m)", fontsize=8); axs[1].grid(False)
        fig.colorbar(im, ax=axs[1])
    return _save("02_lambertian", out, fig) if save else out


# ----------------------------------------------------------------------
# 3 Fresnel / Snell
# ----------------------------------------------------------------------
def case_fresnel(rays=4e6, save=True):
    _ensure_backend()
    N = int(rays)
    L, depth = 4.0, 1.0
    tank = _box_tank(L, L, 0.0, depth)
    rows_a = []
    ok_a = True
    for th in [0, 15, 30, 45, 60, 75, 85]:
        t = math.radians(th)
        hgt = 0.05
        x0 = L / 2 - hgt * math.tan(t)
        cfg = _cfg([{"name": "beam", "archetype": "collimated_beam", "position": [x0, L / 2], "height": hgt,
                     "tilt_y": th}], tank, bounces=False, albedo={"sand": 0, "rock": 0},
                   glass={"walls": {w: "black" for w in CF.WALL_NAMES}}, water={"kd": 0})
        sc = build_scene(cfg, tank, plan=False)
        tal = trace(sc, N, seed=31 + th)
        c = tal.counters
        T_mc = c["entered_water"] / c["emitted"]
        T_an = 1 - float(fresnel_unpolarised(np.array([t]), 1.0, N_W)[0])
        sig = math.sqrt(T_an * (1 - T_an) / N)
        dz = c["entry_dir_z"] / c["entered_water"]
        dx_ = c["entry_dir_x"] / c["entered_water"]
        tt_mc = math.degrees(math.atan2(dx_, -dz))
        tt_an = math.degrees(math.asin(math.sin(t) / N_W))
        good = abs(T_mc - T_an) < 4 * sig + 2e-6 and abs(tt_mc - tt_an) < 0.01
        ok_a &= good
        rows_a.append({"theta_i_deg": th, "T_mc": T_mc, "T_analytic": T_an, "sigma": sig,
                       "z": (T_mc - T_an) / sig if sig > 0 else 0.0, "theta_t_mc_deg": tt_mc,
                       "theta_t_snell_deg": tt_an, "pass": bool(good)})
    # 3b glass slab
    rows_b = []
    ok_b = True
    for th in [20, 50, 65, 75, 85]:
        t = math.radians(th)
        tw = math.asin(math.sin(t) / N_W)
        alpha = math.pi / 2 - tw
        hgt = 0.05
        xe = 0.05                                   # entry point 50 mm from the left glass
        x0 = xe + hgt * math.tan(t)
        cfg = _cfg([{"name": "beam", "archetype": "collimated_beam", "position": [x0, L / 2], "height": hgt,
                     "tilt_y": -th, "footprint": {"length": "1 mm", "width": "1 mm"},
                     "emitters": {"nx": 1, "ny": 1, "size": ["1 mm", "1 mm"]}}], tank, bounces=False,
                   albedo={"sand": 0, "rock": 0}, water={"kd": 0},
                   glass={"walls": {"left": "clear", "right": "black", "front": "black", "back": "black"}})
        sc = build_scene(cfg, tank, plan=False)
        tal = trace(sc, N, seed=51 + th)
        c = tal.counters
        T_mc = c["lost_through_glass"] / c["entered_water"]
        R1 = float(fresnel_unpolarised(np.array([alpha]), N_W, N_G)[0])
        ag = math.asin(N_W / N_G * math.sin(alpha))
        R2 = float(fresnel_unpolarised(np.array([ag]), N_G, 1.0)[0])
        T_an = (1 - R1) * (1 - R2) / (1 - R1 * R2)
        sig = math.sqrt(max(T_an * (1 - T_an), 0) / c["entered_water"])
        good = abs(T_mc - T_an) < 4 * sig + 1e-4 if T_an > 0 else T_mc == 0.0
        ok_b &= good
        rows_b.append({"theta_air_deg": th, "incidence_on_glass_deg": math.degrees(alpha), "R1_water_glass": R1,
                       "R2_glass_air": R2, "T_slab_mc": T_mc, "T_slab_analytic": T_an, "sigma": sig,
                       "tir": bool(R2 >= 1.0), "pass": bool(good)})
    out = {"case": "3 refraction + Fresnel", "rays_per_angle": N, "air_water": rows_a, "glass_slab": rows_b,
           "critical_angle_water_air_deg": math.degrees(math.asin(1 / N_W)), "pass": bool(ok_a and ok_b)}
    fig = None
    if save:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        th = np.linspace(0, 89.9, 300)
        axs[0].plot(th, 1 - fresnel_unpolarised(np.radians(th), 1, N_W), "k-", lw=0.8, label="1 - R (analytic)")
        axs[0].plot([r["theta_i_deg"] for r in rows_a], [r["T_mc"] for r in rows_a], "o", label="MC")
        axs[0].set_xlabel("incidence in air (deg)"); axs[0].set_ylabel("transmitted fraction")
        axs[0].set_title("Air -> water (n = 1.34)", fontsize=8); axs[0].legend(fontsize=7)
        al = np.linspace(40, 90, 400)
        R1 = fresnel_unpolarised(np.radians(al), N_W, N_G)
        ag = np.arcsin(np.clip(N_W / N_G * np.sin(np.radians(al)), -1, 1))
        R2 = fresnel_unpolarised(ag, N_G, 1.0)
        axs[1].plot(al, (1 - R1) * (1 - R2) / (1 - R1 * R2), "k-", lw=0.8, label="incoherent slab (analytic)")
        axs[1].plot([r["incidence_on_glass_deg"] for r in rows_b], [r["T_slab_mc"] for r in rows_b], "o", label="MC")
        axs[1].axvline(math.degrees(math.asin(1 / N_W)), color="0.5", ls=":", lw=0.8)
        axs[1].set_xlabel("incidence on glass from water (deg)"); axs[1].set_ylabel("fraction through glass")
        axs[1].set_title("Water -> 12.7 mm glass -> air (TIR beyond 48.3 deg)", fontsize=8)
        axs[1].legend(fontsize=7)
    return _save("03_fresnel", out, fig) if save else out


# ----------------------------------------------------------------------
# 4 Beer-Lambert
# ----------------------------------------------------------------------
def diffuse_Ed(z, kd, M=1.0):
    """E_d(z) below a flat surface lit by an infinite Lambertian plane of exitance M."""
    from scipy.integrate import quad

    def f(ta):
        tw = math.asin(math.sin(ta) / N_W)
        T = 1 - float(fresnel_unpolarised(np.array([ta]), 1.0, N_W)[0])
        return T * math.exp(-kd * z / math.cos(tw)) * 2 * math.cos(ta) * math.sin(ta)
    return M * quad(f, 0, math.pi / 2, limit=200, epsabs=1e-12, epsrel=1e-10)[0]


def case_beer_lambert(rays=4e6, rays_diffuse=5e7, save=True):
    _ensure_backend()
    rows_a, ok_a = [], True
    depths = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.9])
    L, D = 2.0, 1.0
    tank = _box_tank(L, L, 0.0, D)
    T0 = 1 - float(fresnel_unpolarised(np.array([0.0]), 1, N_W)[0])
    N = int(rays)
    for kd in [0.5, 1.0, 2.0]:
        cfg = _cfg([{"name": "beam", "archetype": "collimated_beam", "position": [L / 2, L / 2], "height": 0.05}], tank,
                   bounces=False, albedo={"sand": 0, "rock": 0}, water={"kd": f"{kd} 1/m"},
                   glass={"walls": {w: "black" for w in CF.WALL_NAMES}}, map_depths=depths.tolist(),
                   map_resolution="20 mm")
        sc = build_scene(cfg, tank)
        tal = trace(sc, N, seed=71)
        F = tal.plan_w.sum((1, 2)) / N
        Fan = T0 * np.exp(-kd * depths)
        sig = math.sqrt(T0 * (1 - T0) / N)
        err = np.abs(F / Fan - 1)
        good = bool(np.all(err < 4 * sig / T0 + 1e-5))
        ok_a &= good
        rows_a.append({"kd": kd, "depth_m": depths.tolist(), "F_over_phi_mc": F.tolist(),
                       "F_over_phi_analytic": Fan.tolist(), "max_rel_err": float(err.max()),
                       "tolerance": 4 * sig / T0 + 1e-5, "pass": good})
    # 4b diffuse
    S, Dd = 6.0, 1.0
    tank2 = _box_tank(S, S, 0.0, Dd)
    dd = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    phi = 1000.0
    cfg = _cfg([{"name": "sky", "archetype": "lambertian_panel", "ppf": phi, "position": [S / 2, S / 2],
                 "height": 0.05, "footprint": {"length": f"{S} m", "width": f"{S} m"},
                 "emitters": {"nx": 1, "ny": 1, "size": [f"{S} m", f"{S} m"]}}], tank2,
               bounces=False, albedo={"sand": 0, "rock": 0}, water={"kd": "1 1/m"},
               glass={"walls": {w: "black" for w in CF.WALL_NAMES}}, map_depths=dd.tolist(), map_resolution="0.5 m")
    sc = build_scene(cfg, tank2)
    # one 0.5 m pixel centred in the tank
    sc.plan_grid = (S / 2 - 0.25, S / 2 - 0.25, 0.5, 1, 1)
    Nd = int(rays_diffuse)
    tal = trace(sc, Nd, seed=72)
    M = phi / S ** 2
    E_mc = tal.plan_w[:, 0, 0] * tal.ppf_per_photon / 0.25
    rel = np.sqrt(tal.plan_w2[:, 0, 0]) / tal.plan_w[:, 0, 0]
    E_an = np.array([diffuse_Ed(z, 1.0, M) for z in dd])
    errb = E_mc / E_an - 1
    ok_b = bool(np.all(np.abs(errb) < 4 * rel + 1e-3))
    # 4c spectral
    tank3 = tank
    raw_fx = [{"name": "beam", "archetype": "collimated_beam", "position": [L / 2, L / 2], "height": 0.05,
               "channels": {"flat": 100}}]
    lib = load_library()
    lib["archetypes"]["two_channel_beam"] = dict(lib["archetypes"]["collimated_beam"])
    lib["archetypes"]["two_channel_beam"]["channels"] = {"rb": {"share": 0.6, "spectrum": "royal_blue"},
                                                         "red": {"share": 0.4, "spectrum": "red"}}
    raw_fx[0]["archetype"] = "two_channel_beam"
    raw_fx[0].pop("channels")
    cfg3 = CF.parse_lighting({"fixtures": raw_fx, "spectral": True, "bounces": False,
                              "albedo": {"sand": 0, "rock": 0}, "water": {"kd_extra": "0.03 1/m"},
                              "glass": {"walls": {w: "black" for w in CF.WALL_NAMES}},
                              "map_depths": depths.tolist(), "map_resolution": "20 mm"}, tank3, lib)
    sc3 = build_scene(cfg3, tank3)
    tal3 = trace(sc3, N, seed=73)
    F3 = tal3.plan_w.sum((1, 2)) / tal3.counters["entered_water"]
    fb = cfg3.fixtures[0].spectrum(cfg3.spectra)
    kdb = cfg3.kd_bins()
    F3an = np.array([np.sum(fb * np.exp(-kdb * z)) for z in depths])
    # binomial-like sigma of a bin-weighted mean: sqrt(var(exp(-Kd z)) / N)
    sig3 = np.array([math.sqrt(max(np.sum(fb * np.exp(-2 * kdb * z)) - np.sum(fb * np.exp(-kdb * z)) ** 2, 0)
                               / tal3.counters["entered_water"]) for z in depths])
    ok_c = bool(np.all(np.abs(F3 - F3an) < 4 * sig3 + 1e-5 * F3an))
    out = {"case": "4 Beer-Lambert", "collimated": rows_a,
           "diffuse": {"kd": 1.0, "depth_m": dd.tolist(), "E_mc": E_mc.tolist(), "E_analytic": E_an.tolist(),
                       "rel_err": errb.tolist(), "mc_rel_sigma": rel.tolist(), "exitance_M": M,
                       "E_d_0_over_M": diffuse_Ed(0.0, 0.0, 1.0), "rays": Nd, "pass": ok_b},
           "spectral": {"depth_m": depths.tolist(), "F_ratio_mc": F3.tolist(), "F_ratio_analytic": F3an.tolist(),
                        "sigma": sig3.tolist(), "kd_bins_1_m": kdb.tolist(), "pass": ok_c},
           "pass": bool(ok_a and ok_b and ok_c)}
    fig = None
    if save:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        for r in rows_a:
            axs[0].semilogy(r["depth_m"], r["F_over_phi_mc"], "o", ms=4, label=f"MC collimated Kd = {r['kd']}")
            axs[0].semilogy(r["depth_m"], r["F_over_phi_analytic"], "k-", lw=0.8)
        axs[0].semilogy(depths, F3 * T0, "s", ms=4, label="MC spectral (x T0)")
        axs[0].semilogy(depths, F3an * T0, "k--", lw=0.8)
        axs[0].set_xlabel("depth (m)"); axs[0].set_ylabel("F / Φ"); axs[0].legend(fontsize=7)
        axs[0].set_title("Collimated: T0 exp(-Kd z); spectral Σ f_b exp(-Kd_b z)", fontsize=8)
        axs[1].plot(dd, E_mc, "o", label="MC (centre window)")
        axs[1].plot(dd, E_an, "k-", lw=0.8, label="quadrature: Fresnel + Snell + z/cosθ_w")
        axs[1].plot(dd, M * np.exp(-dd) * diffuse_Ed(0, 0, 1), ":", color="0.5", lw=0.8,
                    label="naive E0 exp(-Kd z)")
        axs[1].set_xlabel("depth (m)"); axs[1].set_ylabel("E_d (µmol m$^{-2}$ s$^{-1}$)"); axs[1].legend(fontsize=7)
        axs[1].set_title("Diffuse (Lambertian) illumination, Kd = 1 1/m", fontsize=8)
    return _save("04_beer_lambert", out, fig) if save else out


# ----------------------------------------------------------------------
# 5 convergence
# ----------------------------------------------------------------------
def case_convergence(exponents=(19, 20, 21, 22, 23, 24), seeds=4, save=True):
    _ensure_backend()
    from .demo import demo_project
    from .run import prepare_geometry
    proj = demo_project()
    cfg = CF.lighting_from_project(proj, overrides={"map_depths": [0.15]})
    geo = prepare_geometry(proj, cfg, dx=0.005)
    sc = build_scene(cfg, proj.tank, grid=geo.grid, sdf=geo.sdf)
    kst = int(np.argmin(np.abs(geo.grid.coords(2) - (proj.tank.water_surface - 0.15))))
    rows = []
    for e in exponents:
        N = 2 ** e
        sand, sand2, plan, plan2, stack, stack2 = [], [], [], [], [], []
        for s in range(seeds):
            tal = trace(sc, N, seed=1000 + 17 * e + s)
            sand.append(tal.sand_w[1:-1, 1:-1].copy()); sand2.append(tal.sand_w2[1:-1, 1:-1].copy())
            plan.append(tal.plan_w[0].copy()); plan2.append(tal.plan_w2[0].copy())
            st = tal.stack_w[1:-1, 1:-1, kst].copy()
            stack.append(st); stack2.append(st)              # internal estimate 1/sqrt(sum w)
        row = {"N": N}
        for name, A, A2 in (("sand", np.array(sand), np.array(sand2)), ("plan_150mm", np.array(plan), np.array(plan2)),
                            ("stack_150mm", np.array(stack), np.array(stack2))):
            mean = A.mean(0)
            lit = mean > 0.5 * mean[mean > 0].mean()
            emp = A.std(0, ddof=1)[lit] / mean[lit]
            internal = (np.sqrt(A2) / np.where(A > 0, A, np.inf))[:, lit]
            # pooled RMS (unbiased variance, ddof = 1). NOT the median of per-cell std: with 4 seeds
            # the median of s is 0.888 sigma (chi^2_3), which made a first analysis read the internal
            # estimate as 13 % high (D-413).
            row[f"{name}_empirical_rel"] = float(np.sqrt(np.mean(emp ** 2)))
            row[f"{name}_internal_rel"] = float(np.sqrt(np.mean(internal ** 2)))
            row[f"{name}_empirical_rel_median"] = float(np.median(emp))
            row[f"{name}_internal_rel_median"] = float(np.median(internal))
            row[f"{name}_cells"] = int(lit.sum())
        rows.append(row)
    Ns = np.array([r["N"] for r in rows], float)
    res = {}
    ok = True
    for name in ("sand", "plan_150mm", "stack_150mm"):
        emp = np.array([r[f"{name}_empirical_rel"] for r in rows])
        intr = np.array([r[f"{name}_internal_rel"] for r in rows])
        slope = float(np.polyfit(np.log(Ns), np.log(emp), 1)[0])
        ratio = intr / emp
        good = abs(slope + 0.5) <= 0.03 and bool(np.all(np.abs(ratio - 1) <= 0.15))
        ok &= good
        res[name] = {"slope": slope, "internal_over_empirical": ratio.tolist(), "pass": bool(good)}
    # rays needed for 2 % / 1 % on a 5 mm average-lit sand cell (from the fit)
    emp = np.array([r["sand_empirical_rel"] for r in rows])
    c = np.exp(np.mean(np.log(emp) + 0.5 * np.log(Ns)))
    out = {"case": "5 convergence", "seeds": seeds, "rows": rows, "fits": res,
           "rays_for_2pct_5mm_sand_cell": float((c / 0.02) ** 2), "rays_for_1pct_5mm_sand_cell": float((c / 0.01) ** 2),
           "rays_for_2pct_10mm_cell_est": float((c / 0.02) ** 2 / 4), "pass": bool(ok)}
    fig = None
    if save:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for name, mk in (("sand", "o"), ("plan_150mm", "s"), ("stack_150mm", "^")):
            ax.loglog(Ns, [r[f"{name}_empirical_rel"] for r in rows], mk, label=f"{name}: seed-to-seed")
            est = "1/√Σw" if name.startswith("stack") else "√Σw²/Σw"
            ax.loglog(Ns, [r[f"{name}_internal_rel"] for r in rows], mk, mfc="none", label=f"{name}: internal {est}")
        ax.loglog(Ns, c / np.sqrt(Ns), "k-", lw=0.8, label="N$^{-1/2}$")
        ax.set_xlabel("photons N"); ax.set_ylabel("RMS relative noise, 5 mm cells")
        ax.set_title("Ray-count convergence (demo tank)", fontsize=8); ax.legend(fontsize=7)
    return _save("05_convergence", out, fig) if save else out


# ----------------------------------------------------------------------
# 6 calibration round trip
# ----------------------------------------------------------------------
def case_calibration(rays_truth=2e7, rays_fit=4e6, save=True):
    _ensure_backend()
    import copy
    from .calibration import _parse_readings, calibrate, predict
    from .demo import demo_project
    from .run import prepare_geometry, with_lighting
    proj0 = demo_project()
    base = copy.deepcopy(proj0.raw["lighting"])
    truth_l = copy.deepcopy(base)
    for f in truth_l["fixtures"]:
        f["distribution"] = {"type": "beam_angle", "beam_angle": "70 deg"}
        f["power"] = "75 W"                                  # 150 umol/s at 2.0 umol/J
    truth = with_lighting(proj0, truth_l)
    cfg_t = CF.lighting_from_project(truth)
    geo = prepare_geometry(truth, cfg_t, dx=0.005)
    rng = np.random.default_rng(2024)
    IN = 0.0254
    # 6a manufacturer chart in air under the first fixture
    pts_a = [(12, 0), (12, 3), (12, 6), (12, 9), (12, 12), (18, 0), (18, 6), (18, 12)]
    rd_a = [{"scene": "air", "below": f"{b} in", "offset": ["0 in", f"{o} in"], "par": 1.0} for b, o in pts_a]
    cal_a = {"readings": rd_a}
    p_true = predict(truth, cfg_t, geo, _parse_readings(cal_a, cfg_t), [cfg_t.fixtures[0]], int(rays_truth), seed=1001)
    for r, p in zip(rd_a, p_true):
        r["par"] = float(p * (1 + 0.02 * rng.standard_normal()))
    nominal = CF.lighting_from_project(proj0)
    fit_a = calibrate(proj0, nominal, geo, {"fit": ["scale", "beam_angle"], "fixtures": ["LED-left"],
                                            "readings": rd_a, "rays": rays_fit})
    # 6b own meter in the tank (in water, not immersion corrected)
    pts_b = [(11.75, 8.5, 4), (11.75, 8.5, 10), (23.5, 8.5, 4), (23.5, 8.5, 10), (35.25, 4, 6), (30, 13, 12)]
    rd_b = [{"scene": "tank", "x": f"{x} in", "y": f"{y} in", "depth": f"{d} in", "par": 1.0,
             "immersion_corrected": False} for x, y, d in pts_b]
    p_true_b = predict(truth, cfg_t, geo, _parse_readings({"readings": rd_b}, cfg_t), cfg_t.fixtures,
                       int(rays_truth), seed=1002)
    for r, p in zip(rd_b, p_true_b):
        r["par"] = float(p / 1.08 * (1 + 0.02 * rng.standard_normal()))
    # model: true beam angle, nominal power (180 umol/s)
    beam_l = copy.deepcopy(base)
    for f in beam_l["fixtures"]:
        f["distribution"] = {"type": "beam_angle", "beam_angle": "70 deg"}
    cfg_b = CF.lighting_from_project(with_lighting(proj0, beam_l))
    fit_b = calibrate(proj0, cfg_b, geo, {"fit": ["scale"], "readings": rd_b, "rays": rays_fit,
                                          "immersion_factor": 1.08})
    rd_b_wrong = [dict(r, immersion_corrected=True) for r in rd_b]
    fit_bw = calibrate(proj0, cfg_b, geo, {"fit": ["scale"], "readings": rd_b_wrong, "rays": rays_fit})
    s_true = 150.0 / 180.0
    ok_a = (abs(fit_a["beam_angle_deg"] - 70) <= 3 and abs(fit_a["scale"] / s_true - 1) <= 0.03
            and fit_a["rms_rel"] <= 0.035)
    ok_b = abs(fit_b["scale"] / s_true - 1) <= 0.02 and fit_b["rms_rel"] <= 0.03
    strip = lambda d: {k: v for k, v in d.items() if k != "cfg"}  # noqa: E731
    out = {"case": "6 calibration round trip", "truth": {"beam_angle_deg": 70, "ppf_umol_s": 150, "scale": s_true},
           "nominal": {"beam_angle_deg": 90, "ppf_umol_s": 180},
           "a_air_chart": strip(fit_a), "b_tank_meter": strip(fit_b), "b_ignoring_immersion": strip(fit_bw),
           "b_ignoring_immersion_scale_error": fit_bw["scale"] / s_true - 1,
           "pass_a": bool(ok_a), "pass_b": bool(ok_b), "pass": bool(ok_a and ok_b)}
    fig = None
    if save:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        tr = fit_a["trials"]
        axs[0].plot(tr["beam_angle_deg"], tr["cost"], "o-", ms=4, label="Σ residual² at best scale")
        axs[0].axvline(70, color="k", lw=0.8, ls="--", label="truth 70°")
        axs[0].axvline(fit_a["beam_angle_deg"], color="C1", lw=0.8, label=f"fit {fit_a['beam_angle_deg']:.1f}°")
        axs[0].set_yscale("log"); axs[0].set_xlabel("beam angle (deg)"); axs[0].legend(fontsize=7)
        axs[0].set_title("6a: in-air chart, fit [scale, beam angle]", fontsize=8)
        for fit, mk, lab in ((fit_a, "o", "6a air chart"), (fit_b, "s", "6b tank meter")):
            m = [r["measured_corrected"] for r in fit["readings"]]
            p = [r["model_calibrated"] for r in fit["readings"]]
            axs[1].loglog(m, p, mk, label=lab)
        lim = [min(axs[1].get_xlim()[0], axs[1].get_ylim()[0]), max(axs[1].get_xlim()[1], axs[1].get_ylim()[1])]
        axs[1].plot(lim, lim, "k-", lw=0.8)
        axs[1].set_xlabel("reading (immersion-corrected)"); axs[1].set_ylabel("calibrated model")
        axs[1].set_title("Calibrated model vs synthetic readings", fontsize=8); axs[1].legend(fontsize=7)
    return _save("06_calibration", out, fig) if save else out


# ----------------------------------------------------------------------
# 7 rock SDF: sphere
# ----------------------------------------------------------------------
def case_sphere(rays=2e7, save=True):
    _ensure_backend()
    out = {"case": "7 rock SDF (sphere)"}
    ok = True
    figs = []
    for src in ("stl", "voxels"):
        r = sphere_case(src, int(rays))
        lim = {"stl": (0.01, 0.03, 0.02, 2.0), "voxels": (0.03, 0.06, 0.05, 8.0)}[src]
        good = (abs(r["intercepted_ratio"] - 1) <= lim[0] and abs(r["shadow_area_ratio"] - 1) <= lim[1]
                and abs(r["cosine_law_ratio"] - 1) <= lim[2] and r["normal_rms_deg"] < lim[3])
        r["limits"] = {"intercepted": lim[0], "shadow": lim[1], "cosine": lim[2], "normal_deg": lim[3]}
        r["pass"] = bool(good)
        ok &= good
        figs.append(r.pop("_plot"))
        out[src] = r
    out["pass"] = bool(ok)
    fig = None
    if save:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, (src, (cth, ratio, w)) in zip(axs, zip(("stl", "voxels"), figs)):
            ax.scatter(cth, ratio, s=np.clip(w / w.max() * 20, 1, 20), alpha=0.5)
            ax.axhline(1, color="k", lw=0.8)
            ax.set_ylim(0.5, 1.5)
            ax.set_xlabel("cos θ (exact normal · z)"); ax.set_ylabel("E_patch / (E0 cos θ)")
            ax.set_title(f"Sphere patches, {src} SDF (dot size ∝ area)", fontsize=8)
    return _save("07_sphere", out, fig) if save else out


def sphere_case(src: str, rays: int) -> dict:
    """Collimated vertical beam on a sphere above black sand; STL- or voxel-derived SDF."""
    from .. import geometry as G
    from .results import rel_error
    from .sdf import SDF, sphere_sdf
    from .tracer import grid_for_tank
    L, zsand, zs, R = 0.6, 0.05, 0.45, 0.05
    c = np.array([0.3, 0.3, zsand + 0.15])
    tank = _box_tank(L, L, zsand, zs)
    phi, A_b = 1000.0, 0.3 * 0.3
    cfg = _cfg([{"name": "beam", "archetype": "collimated_beam", "ppf": phi, "position": [0.3, 0.3], "height": 0.05,
                 "footprint": {"length": "0.3 m", "width": "0.3 m"},
                 "emitters": {"nx": 1, "ny": 1, "size": ["0.3 m", "0.3 m"]}}], tank,
               bounces=False, albedo={"sand": 0, "rock": 0}, water={"kd": 0},
               glass={"walls": {w: "black" for w in CF.WALL_NAMES}})
    grid = grid_for_tank(tank, 0.005)
    if src == "stl":
        v, f = G.sdf_to_mesh(sphere_sdf(c, R), c - 1.3 * R, c + 1.3 * R, 0.0015)
        sdf = SDF.from_mesh(v, f, 0.0025, source="stl")
    else:
        X, Y, Z = (grid.coords(a) for a in range(3))
        P = np.stack(np.meshgrid(X, Y, Z, indexing="ij"), -1)
        sdf = SDF.from_mask(sphere_sdf(c, R)(P) < 0, grid.origin, grid.dx, smooth=1.0)
    pt = sdf.patches(grid.origin, grid.dx, grid.shape, zmin=zsand, zmax=zs)
    sc = build_scene(cfg, tank, grid=grid, sdf=sdf, plan=False)
    tal = trace(sc, rays, seed=91)
    T0 = 1 - float(fresnel_unpolarised(np.array([0.0]), 1, N_W)[0])
    E0 = phi * T0 / A_b
    k = tal.ppf_per_photon
    inter = tal.counters["incident_rock"] * k / (E0 * math.pi * R * R)
    Es = tal.sand_w * k / grid.dx ** 2
    X, Y = grid.coords(0), grid.coords(1)
    inbeam = (np.abs(X - 0.3) < 0.14)[:, None] & (np.abs(Y - 0.3) < 0.14)[None, :]
    shadow = float(((Es < 0.5 * E0) & inbeam).sum() * grid.dx ** 2) / (math.pi * R * R)
    A = pt["area"]
    ok = A >= 0.1 * grid.dx ** 2
    cen = pt["centroid"][ok]
    cth = ((cen - c) / np.linalg.norm(cen - c, axis=1, keepdims=True))[:, 2]
    E = tal.rock_w[ok] * k / A[ok]
    rel = rel_error(tal.rock_w[ok], tal.rock_w2[ok])
    sel = cth > 0.2
    ratio = E[sel] / (E0 * cth[sel])
    w = A[ok][sel]
    cos_ratio = float(np.sum(ratio * w) / np.sum(w))
    # scatter beyond MC noise (rms of ratio-1 after removing noise in quadrature)
    excess = float(np.sqrt(max(np.average((ratio - 1) ** 2, weights=w) - np.average(rel[sel] ** 2, weights=w), 0)))
    # normals at exact surface points
    n_pts = 4000
    i = np.arange(n_pts) + 0.5
    th = np.arccos(1 - 2 * i / n_pts)
    ph = np.pi * (1 + 5 ** 0.5) * i
    nex = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    ng = sdf.gradient(c + R * nex)
    ang = np.degrees(np.arccos(np.clip(np.sum(ng * nex, 1), -1, 1)))
    return {"sdf_source": src, "sdf_h_m": sdf.h, "rays": rays, "E0": E0, "intercepted_ratio": inter,
            "shadow_area_ratio": shadow, "cosine_law_ratio": cos_ratio, "patch_excess_scatter_rms": excess,
            "patches_used": int(sel.sum()), "median_patch_mc_rel": float(np.median(rel[sel])),
            "normal_rms_deg": float(np.sqrt(np.mean(ang ** 2))), "normal_max_deg": float(ang.max()),
            "surface_area_ratio": float(A.sum() / (4 * math.pi * R * R)),
            "_plot": (cth[sel], ratio, w)}


# ----------------------------------------------------------------------
# 8 energy closure
# ----------------------------------------------------------------------
def case_energy(rays=4e6, save=True):
    _ensure_backend()
    from .demo import demo_project
    from .run import prepare_geometry
    proj = demo_project()
    rows, ok = [], True
    geo = None
    for walls in ({w: "clear" for w in CF.WALL_NAMES}, {w: "black" for w in CF.WALL_NAMES},
                  {"front": "clear", "back": "coralline", "left": "coralline", "right": "black"}):
        lt = {**proj.raw["lighting"], "glass": {"walls": walls}}
        cfg = CF.lighting_from_project(proj, overrides=lt)
        geo = geo or prepare_geometry(proj, cfg, dx=0.005)
        sc = build_scene(cfg, proj.tank, grid=geo.grid, sdf=geo.sdf)
        tal = trace(sc, int(rays), seed=81)
        eb = tal.energy_balance()
        good = abs(eb["closure_residual"]) < 1e-5
        ok &= good
        rows.append({"walls": walls, "balance": eb, "pass": bool(good)})
    out = {"case": "8 energy closure", "rows": rows, "pass": bool(ok)}
    return _save("08_energy", out) if save else out


# ----------------------------------------------------------------------
# throughput
# ----------------------------------------------------------------------
def throughput(rays=2 ** 23, arches=("cuda", "cpu")):
    """Rays/s on the demo tank for several feature sets, per backend (each in a subprocess)."""
    import subprocess
    import sys
    rows = []
    for a in arches:
        cmd = [sys.executable, "-c", f"from reefflow.light.verification import _throughput_one; _throughput_one({a!r}, {int(rays)})"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        line = [l for l in p.stdout.splitlines() if l.startswith("THROUGHPUT ")]
        if line:
            rows.extend(json.loads(line[-1][len("THROUGHPUT "):]))
        else:
            rows.append({"arch": a, "error": p.stderr[-2000:]})
    return _save("throughput", {"rows": rows})


def _throughput_one(arch, rays):
    import reefflow  # noqa: F401
    backend.init(arch)
    from .demo import demo_project
    from .run import prepare_geometry
    proj = demo_project()
    cfg = CF.lighting_from_project(proj)
    geo = prepare_geometry(proj, cfg, dx=0.005)
    rows = []
    n = rays if arch != "cpu" else max(rays // 8, 1 << 19)
    for label, kw in (("empty tank, plan maps only", dict(grid=None, sdf=None)),
                      ("empty tank + node stack", dict(grid=geo.grid, sdf=None)),
                      ("rock, plan maps only", dict(grid=None, sdf=geo.sdf)),
                      ("full (rock + node stack + bounces)", dict(grid=geo.grid, sdf=geo.sdf))):
        sc = build_scene(cfg, proj.tank, **kw)
        trace(sc, 1 << 16, seed=1)                      # compile / warm-up
        tal = trace(sc, n, seed=2)
        rows.append({"arch": arch, "case": label, "rays": n, "seconds": tal.seconds,
                     "rays_per_s": tal.rays_per_second})
    print("THROUGHPUT " + json.dumps(rows))


CASES = {"1": case_inverse_square, "2": case_lambertian, "3": case_fresnel, "4": case_beer_lambert,
         "5": case_convergence, "6": case_calibration, "7": case_sphere, "8": case_energy}


def run_all(which=None):
    _ensure_backend()
    out = {}
    for k, fn in CASES.items():
        if which and k not in which:
            continue
        r = fn()
        out[k] = r["pass"]
        print(f"case {k}: {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    import reefflow  # noqa: F401
    run_all(sys.argv[1:] or None)
