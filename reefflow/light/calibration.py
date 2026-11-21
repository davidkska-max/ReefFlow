"""Calibration of the lighting model against PAR readings (spec 'Calibration option', PLAN C7).

Schema::

    lighting:
      calibration:
        fit: [scale]                 # or [scale, beam_angle]
        immersion_factor: 1.08       # multiplies in-water readings NOT immersion-corrected (A-409)
        fixtures: [LED-left]         # fixtures whose beam angle is fitted, and the fixture used
                                     # for 'air' / 'open_water' readings (default: all / the first)
        rays: 4.0e6                  # photons per scene evaluation
        readings:
          # own meter in the user's tank (depth below the water surface)
          - {scene: tank, x: 12 in, y: 8.5 in, depth: 6 in, par: 420, immersion_corrected: false}
          # manufacturer chart measured in AIR, a distance below the emitting face
          - {scene: air, below: 12 in, offset: [0 in, 6 in], par: 800}
          # manufacturer chart in water: fixture 'height' above the water, sensor 'depth' below it
          - {scene: open_water, height: 8 in, depth: 12 in, offset: [0 in, 0 in], par: 350,
             immersion_corrected: true}

Model: every reading is predicted by a probe tally (downward photons through a
horizontal square of side ``lighting.probe_size``, i.e. a cosine-corrected
upward-facing sensor) in its scene:
    tank        the full tank (glass, rock, sand bounces) with all fixtures;
    air         the calibration fixture alone in air, pointing straight down;
    open_water  the calibration fixture above laterally unbounded water (black floor,
                no walls), i.e. a manufacturer's open-water PAR chart.
In-water readings without immersion correction are multiplied by the immersion
factor before fitting (a submerged cosine diffuser under-reads; C7).

Fit: the scale s minimising sum(((s p_i - m_i) / m_i)^2) (relative residuals,
since meter errors are proportional) is s = sum(p/m) / sum((p/m)^2). With
'beam_angle' the cos^m beam angle (FWHM) of the calibration fixtures is found by
a 1-D search (11-point coarse scan, 9-point fine scan, parabolic refinement;
common random numbers across trial angles), s in closed form at each angle. Residuals are reported per reading.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from . import lunits as LU
from .fixtures import Distribution, beam_angle_to_m
from .tracer import build_scene, trace


def _parse_readings(cal: dict, cfg) -> list:
    out = []
    for i, r in enumerate(cal.get("readings") or []):
        sc = r.get("scene", "tank")
        if sc not in ("tank", "air", "open_water"):
            raise ValueError(f"calibration reading {i}: scene must be tank | air | open_water")
        par = LU.parse(r["par"], "par")
        if par <= 0:
            raise ValueError(f"calibration reading {i}: PAR must be > 0")
        in_water = sc != "air"
        corrected = bool(r.get("immersion_corrected", True)) if in_water else True
        d = {"scene": sc, "par": par, "in_water": in_water, "immersion_corrected": corrected}
        if sc == "tank":
            d.update(x=LU.parse(r["x"], "length"), y=LU.parse(r["y"], "length"), depth=LU.parse(r["depth"], "length"))
        elif sc == "air":
            off = r.get("offset", [0, 0])
            d.update(below=LU.parse(r["below"], "length"), ox=LU.parse(off[0], "length"), oy=LU.parse(off[1], "length"))
        else:
            off = r.get("offset", [0, 0])
            d.update(height=LU.parse(r["height"], "length"), depth=LU.parse(r["depth"], "length"),
                     ox=LU.parse(off[0], "length"), oy=LU.parse(off[1], "length"))
        out.append(d)
    if not out:
        raise ValueError("calibration: no readings")
    return out


def _alone(fx, centre):
    """Copy of a fixture alone, untilted, emitting face centred at ``centre``."""
    f = copy.deepcopy(fx)
    f.centre = np.asarray(centre, float)
    f.R = np.eye(3)
    return f


def predict(project, cfg, geo, readings, cal_fixtures, rays: int, seed: int = 7) -> np.ndarray:
    """Model PAR (uncalibrated, umol m^-2 s^-1) at each reading."""
    pred = np.full(len(readings), np.nan)
    hw = 0.5 * cfg.probe_size
    area = cfg.probe_size ** 2
    t = project.tank
    groups = {}
    for i, r in enumerate(readings):
        key = r["scene"] if r["scene"] != "open_water" else ("open_water", round(r["height"], 6))
        groups.setdefault(key, []).append(i)
    fx0 = cal_fixtures[0]
    for key, idx in groups.items():
        if key == "tank":
            probes = [[readings[i]["x"], readings[i]["y"], t.water_surface - readings[i]["depth"], hw] for i in idx]
            sc = build_scene(cfg, t, grid=None, sdf=geo.sdf if geo is not None else None, plan=False, probes=probes)
        elif key == "air":
            H = 2.0
            f = _alone(fx0, [0, 0, H])
            probes = [[readings[i]["ox"], readings[i]["oy"], H - readings[i]["below"], hw] for i in idx]
            sc = build_scene(cfg, None, box=(1, 1, -1, 0, 0), fixtures=[f], plan=False, probes=probes,
                             has_water=False, air_floor=-1.0)
        else:
            h = key[1]
            depth = max(readings[i]["depth"] for i in idx) + 0.5
            S = 20.0
            f = _alone(fx0, [0.5 * S, 0.5 * S, depth + h])
            probes = [[0.5 * S + readings[i]["ox"], 0.5 * S + readings[i]["oy"], depth - readings[i]["depth"], hw]
                      for i in idx]
            c2 = copy.copy(cfg)
            c2.walls = {w: "black" for w in cfg.walls}
            c2.albedo_sand = 0.0
            sc = build_scene(c2, None, box=(S, S, 0.0, depth, depth + h + 1.0), fixtures=[f], plan=False,
                             probes=probes)
        tal = trace(sc, rays, seed=seed, batch=cfg.batch)
        pred[idx] = tal.probe_w * tal.ppf_per_photon / area
    return pred


def _with_beam(cfg, names, beam):
    c = copy.copy(cfg)
    c.fixtures = copy.deepcopy(cfg.fixtures)
    for f in c.fixtures:
        if f.name in names:
            f.distribution = Distribution("cosine_power", m=beam_angle_to_m(beam), cutoff=f.distribution.cutoff)
    return c


def _scale_fit(p, m):
    q = p / m
    s = float(q.sum() / (q * q).sum())
    r = (s * p - m) / m
    return s, r


def calibrate(project, cfg, geo=None, cal: dict | None = None) -> dict:
    """Fit the calibration in ``cfg.calibration`` (or ``cal``). Returns a dict with ok, scale,
    beam_angle_deg (if fitted), per-reading residuals, rms_rel and the adjusted config ('cfg')."""
    cal = dict(cal or cfg.calibration or {})
    readings = _parse_readings(cal, cfg)
    fit = cal.get("fit", ["scale"])
    imm = float(cal.get("immersion_factor", 1.08))
    names = cal.get("fixtures") or [f.name for f in cfg.fixtures]
    unknown = set(names) - {f.name for f in cfg.fixtures}
    if unknown:
        raise ValueError(f"calibration: unknown fixture(s) {sorted(unknown)}")
    rays = int(float(cal.get("rays", 4e6)))
    m = np.array([r["par"] * (imm if (r["in_water"] and not r["immersion_corrected"]) else 1.0) for r in readings])

    def fixtures_of(c):
        return [f for f in c.fixtures if f.name in names]

    beam = None
    trials = None
    if "beam_angle" in fit:
        d0 = fixtures_of(cfg)[0].distribution
        if d0.kind != "cosine_power":
            raise ValueError("calibration: beam_angle can only be fitted for cos^m / beam_angle distributions")
        b0 = d0.beam_angle
        lo, hi = max(0.35 * b0, math.radians(10)), min(1.8 * b0, math.radians(170))
        def cost(b):
            c = _with_beam(cfg, names, b)
            p = predict(project, c, geo, readings, fixtures_of(c), rays)
            return float(np.sum(_scale_fit(p, m)[1] ** 2))

        # coarse scan, then a fine scan over +-1 coarse step, then a parabola through the fine
        # minimum and its neighbours (common random numbers keep the cost curve smooth)
        coarse = np.linspace(lo, hi, 11)
        cc = np.array([cost(b) for b in coarse])
        k = int(np.argmin(cc))
        step = coarse[1] - coarse[0]
        fine = np.linspace(max(lo, coarse[k] - step), min(hi, coarse[k] + step), 9)
        fc = np.array([cost(b) for b in fine])
        j = int(np.argmin(fc))
        beam = float(fine[j])
        if 0 < j < len(fine) - 1:
            a = np.polyfit(fine[j - 1:j + 2], fc[j - 1:j + 2], 2)
            if a[0] > 0:
                beam = float(np.clip(-a[1] / (2 * a[0]), fine[j - 1], fine[j + 1]))
        allb = np.concatenate([coarse, fine]); allc = np.concatenate([cc, fc])
        o = np.argsort(allb)
        trials = {"beam_angle_deg": np.degrees(allb[o]).tolist(), "cost": allc[o].tolist()}
        cfg = _with_beam(cfg, names, beam)
    p = predict(project, cfg, geo, readings, fixtures_of(cfg), rays)
    s, r = _scale_fit(p, m)
    n = len(m)
    nfit = 1 + (beam is not None)
    rms = float(np.sqrt(np.mean(r ** 2)))
    out = {"ok": bool(np.isfinite(s) and s > 0), "scale": s, "n_readings": n, "rms_rel": rms,
           "scale_se_rel": float(np.sqrt(np.sum(r ** 2) / max(n - nfit, 1)) / math.sqrt(n)),
           "immersion_factor": imm, "fit": list(fit), "fixtures": list(names),
           "beam_angle_deg": None if beam is None else math.degrees(beam), "trials": trials,
           "readings": [{**{k: v for k, v in rd.items()}, "measured_corrected": float(mm),
                         "model_uncalibrated": float(pp), "model_calibrated": float(s * pp),
                         "residual_rel": float(rr)} for rd, mm, pp, rr in zip(readings, m, p, r)],
           "cfg": cfg}
    if n <= nfit:
        out["warning"] = "as many fitted parameters as readings: residuals are zero by construction"
    return out
