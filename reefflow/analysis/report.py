"""Self-contained HTML summary report (figures embedded; prints to PDF from any browser)."""
from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path

from .. import units as U
from ..metrics import KEY_METRICS

ASSUMPTION_SUMMARY = [
    ("A-104/A-105", "Under-resolved LES (Smagorinsky C_s = 0.10); molecular viscosity is negligible at tank grids."),
    ("A-201", "Sand bed is a flat, smooth, no-slip floor; water surface is a rigid free-slip lid."),
    ("A-203", "Powerheads are momentum sources of force J = rho Q^2/A (no swirl, no pump body)."),
    ("A-205", "Delivered powerhead flow = rated x derating (default 0.7) - the largest input uncertainty."),
    ("A-206", "Narrow or fast outlets are modelled wider at the same momentum flux; the first ~5 outlet "
              "diameters are approximate."),
    ("A-207", "Inflow turbulence I = 0.05 at pump outlets; jet throw beyond ~5 diameters is uncertain by ~1.5-2x."),
    ("A-209", "Rock is solid and impermeable."),
    ("A-210", "Dead zone = time-averaged speed below the threshold (default 2.5 cm/s): a starting assumption."),
    ("A-302", "Water age: steady mean-field solve, turbulent Schmidt number 0.7."),
    ("A-304/A-305", "Wall shear by smooth-wall function 15 mm above the surface; detritus settles below 0.02 Pa "
                    "(starting assumption)."),
    ("A-306", "Coral profiles (flow and PAR bands) are starting points, not husbandry standards."),
    ("A-310", "Near-surface flow is under-predicted at tank resolutions (verifications 6 and 8): dead-zone volume "
              "is an upper bound, sand shear and flow over rock read low. Compare designs on the same grid; use "
              "Standard or Fine for near-surface numbers."),
]


def _img(path: Path) -> str:
    if not path.exists():
        return ""
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b}" alt="{html.escape(path.stem)}">'


def _fmt_len(v, us):
    return f"{U.from_si(v, 'in'):.1f} in" if us else f"{v * 1000:.0f} mm"


def _fmt_vel(v, us):
    return f"{U.from_si(v, 'in/s'):.2f} in/s" if us else f"{v * 100:.1f} cm/s"


def _pos(p, us):
    return "(" + ", ".join(_fmt_len(v, us).split()[0] for v in p) + (") in" if us else ") mm")


CSS = """
:root{--fg:#1d2433;--muted:#5b6475;--line:#d9dee7;--bg:#ffffff;--card:#f6f8fb;--ok:#1f7a3a;--warn:#a15c00;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root{--fg:#e7ebf2;--muted:#9aa4b5;--line:#344055;--bg:#141a24;--card:#1c2431;
--ok:#5fcf86;--warn:#f0b35a;--bad:#ff8a80}}
body{font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif;color:var(--fg);background:var(--bg);max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:24px;margin:8px 0}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-size:13px}th,td{border:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{background:var(--card)}.muted{color:var(--muted)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.cards{display:flex;flex-wrap:wrap;gap:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;min-width:150px}
.card b{display:block;font-size:20px}img{max-width:100%;border:1px solid var(--line);margin:6px 0}
@media print{h2{page-break-after:avoid}img{page-break-inside:avoid}body{max-width:none}}
"""


def write_report(run, out: dict, path) -> Path:
    m = run.model
    P = m.project
    us = P.display_units == "US"
    info = run.info
    met, unc = out["metrics"], out.get("metrics_stat_uncertainty", {})
    t = P.tank.summary()
    H = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>ReefFlow report</title><style>{CSS}</style></head><body>"]
    H.append(f"<h1>ReefFlow report: {html.escape(P.name)}</h1>")
    H.append(f"<p class='muted'>Run {html.escape(str(run.path))} - {info.get('lattice')} / {info.get('collision')} / "
             f"C_s {info.get('smagorinsky')} - grid {m.shape} at {m.dx * 1000:g} mm - {info.get('backend')} "
             f"{info.get('precision')} - {info.get('wall_time_s', 0) / 60:.1f} min - report generated "
             f"{dt.datetime.now():%Y-%m-%d %H:%M}</p>")

    # headline cards
    ke1, ke2 = info.get("ke_avg_first_half"), info.get("ke_avg_second_half")
    drift = abs(ke2 / ke1 - 1) if ke1 else float("nan")
    cards = [("Mean speed", _fmt_vel(met["mean_speed"], us)),
             ("Dead-zone volume", f"{100 * met['dead_fraction']:.1f} %"),
             ("Median speed", _fmt_vel(met["median_speed"], us))]
    if out.get("age"):
        cards.append(("Turnover V/Q (return)", f"{out['age']['tau_nominal_s'] / 60:.1f} min"))
    cards.append(("Sand area where detritus settles", f"{100 * out['wall_shear']['sand_area_settling_fraction']:.0f} %"))
    H.append("<div class='cards'>" + "".join(f"<div class='card'><span class='muted'>{a}</span><b>{b}</b></div>"
                                             for a, b in cards) + "</div>")

    # status
    H.append("<h2>Run status</h2><table>")
    st = [("Statistically steady (KE, 1st vs 2nd half of averaging)", f"{100 * drift:.1f} % difference",
           "ok" if drift < 0.05 else "warn"),
          ("Mass conservation (total drift)", f"{info.get('mass_rel_drift', 0):.1e}", "ok"),
          ("Peak lattice speed (target <= 0.1, warn 0.15)", f"{info.get('u_max_overall_lattice', 0):.3f}",
           "ok" if info.get("u_max_overall_lattice", 0) < 0.15 else "warn"),
          ("Spin-up / averaging", f"{info.get('spinup_s', 0):.0f} s / {info.get('average_s', 0):.0f} s", "ok"),
          ("Grid", f"{m.dx * 1000:g} mm cells. Near-surface metrics change with resolution (Preview overstates dead "
                   "water ~2.7x, Standard ~1.5x vs Fine; verification 6)" if m.dx > 0.004 else
                   f"{m.dx * 1000:g} mm cells (finest preset)", "warn" if m.dx > 0.004 else "ok")]
    for a, b, c in st:
        H.append(f"<tr><td>{a}</td><td class='{c}'>{b}</td></tr>")
    H.append("</table>")

    # inputs
    H.append("<h2>Inputs</h2><table>")
    H.append(f"<tr><th>Tank (internal)</th><td>{t['internal_in'][0]:.2f} x {t['internal_in'][1]:.2f} x "
             f"{t['internal_in'][2]:.2f} in ({' x '.join(f'{v * 1000:.0f}' for v in t['internal_m'])} mm); "
             f"water above sand {t['water_volume_above_sand_gal']:.1f} gal gross; model water volume "
             f"{U.from_si(met['water_volume_m3'], 'gal'):.1f} gal</td></tr>")
    H.append(f"<tr><th>Sand / water level</th><td>sand {_fmt_len(P.tank.sand_depth, us)}, water "
             f"{_fmt_len(P.tank.water_gap, us)} below rim</td></tr>")
    H.append(f"<tr><th>Rock</th><td>{m.info['rock_volume_m3'] * 1000:.1f} L above the sand</td></tr></table>")
    H.append("<table><tr><th>Pump</th><th>Type</th><th>Rated</th><th>Delivered (basis)</th><th>Outlet / modelled</th>"
             "<th>J = rho Q^2/A</th><th>Mode</th><th>Turb. I</th></tr>")
    for pi in m.info["pumps"]:
        H.append(f"<tr><td>{html.escape(pi['name'])}</td><td>{pi['archetype']}</td><td>{pi['rated_flow_GPH']:.0f} GPH</td>"
                 f"<td>{pi['delivered_flow_GPH']:.0f} GPH ({pi['delivered_flow_Lh']:.0f} L/h)<br><span class='muted'>"
                 f"{html.escape(pi['flow_basis'])}</span></td><td>{pi['outlet']}<br>{pi['outlet_model']}</td>"
                 f"<td>{pi['momentum_flux_N']:.2f} N</td><td>{html.escape(pi['mode'])}</td>"
                 f"<td>{pi['turbulence_intensity']:g}</td></tr>")
    H.append("</table><p class='muted'>Manufacturer flow ratings are free-flow values and overstate delivered flow; "
             "derating is an editable assumption (A-205).</p>")
    if m.notes:
        H.append("<p><b>Model notes:</b></p><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in m.notes) + "</ul>")

    # metrics
    H.append("<h2>Flow metrics</h2><table><tr><th>Metric</th><th>Value</th><th>± 2σ statistical</th></tr>")
    for k, (lab, unit, sc) in KEY_METRICS.items():
        u = unc.get(k)
        H.append(f"<tr><td>{lab}</td><td>{met[k] * sc:.2f} {unit}</td><td>{'' if u is None else f'{2 * u * sc:.2f}'}</td></tr>")
    H.append("</table>")

    # dead zones
    dz = out["dead_zones"]
    H.append(f"<h2>Dead zones</h2><p>{100 * dz['dead_fraction']:.1f} % of the water volume has time-averaged speed "
             f"below {_fmt_vel(dz['threshold_ms'], us)}, in {dz['n_zones']} separate regions. If delivered pump flow "
             f"were 20 % lower / higher (derating uncertainty): "
             + " / ".join(f"{100 * s['dead_fraction']:.1f} %" for s in out["derating_sensitivity"] if s["flow_factor"] != 1.0)
             + ".</p>")
    H.append("<table><tr><th>#</th><th>Volume</th><th>Core location (x, y, z)</th><th>Mean speed</th><th>Against</th></tr>")
    for n, z in enumerate(dz["zones"][:8]):
        vol = f"{U.from_si(z['volume_L'] / 1000, 'gal'):.2f} gal" if us else f"{z['volume_L']:.2f} L"
        H.append(f"<tr><td>{n + 1}</td><td>{vol}</td><td>{_pos(z['core_m'], us)}</td>"
                 f"<td>{_fmt_vel(z['mean_speed_ms'], us)}</td><td>{', '.join(z['touches'])}</td></tr>")
    H.append("</table>" + _img(run.path / "dead_zones.png"))

    # age
    if out.get("age"):
        a = out["age"]
        H.append(f"<h2>Water age</h2><p>Nominal turnover through the return pump V/Q = {a['tau_nominal_s'] / 60:.1f} min. "
                 f"Mean age {a['mean_age_over_tau']:.2f}, 95th percentile {a['p95_age_over_tau']:.2f}, maximum "
                 f"{a['max_age_over_tau']:.2f} times V/Q. Values close to 1 everywhere mean the powerheads mix the tank "
                 f"much faster than the return pump exchanges it, so no region is markedly stagnant in the age sense; "
                 f"local speed (dead zones above) is then the more useful indicator. Check: flow-weighted age at the "
                 f"overflow / (V/Q) = {a['outlet_age_over_tau']:.6f} (exactly 1 by construction).</p>"
                 + _img(run.path / "age.png"))

    # wall shear
    w = out["wall_shear"]
    H.append(f"<h2>Sand bed and detritus</h2><p>Wall shear on the sand: median {w['sand_tau_median_Pa'] * 1000:.1f} mPa "
             f"(10-90 %: {w['sand_tau_p10_Pa'] * 1000:.1f}-{w['sand_tau_p90_Pa'] * 1000:.1f} mPa). Detritus is likely to "
             f"settle on {100 * w['sand_area_settling_fraction']:.0f} % of the sand area (shear below "
             f"{w['tau_crit_Pa']} Pa) and on {100 * w['rock_top_settling_fraction']:.0f} % of rock tops; sand is moved on "
             f"{100 * w['sand_area_moving_fraction']:.1f} % (above {w['tau_sand_move_Pa']} Pa). Both thresholds are "
             f"starting assumptions; read the map as relative.</p>" + _img(run.path / "sand_shear.png"))

    # light
    L = out.get("light")
    if L:
        def _u(d):
            return (f"min {d.get('min', float('nan')):.0f} / mean {d.get('mean', float('nan')):.0f} / "
                    f"max {d.get('max', float('nan')):.0f} (CV {d.get('cv', float('nan')):.2f})") if d else "n/a"
        lab = L.get("label", "")
        cls = "ok" if L.get("calibrated") else "warn"
        H.append(f"<h2>Lighting (PAR)</h2><p class='{cls}'><b>{html.escape(lab)}</b></p>"
                 f"<p>Total photon output {L.get('total_ppf_umol_s', 0):.0f} µmol/s; {L.get('rays', 0):.3g} photons "
                 f"traced. Meter-equivalent PAR (upward-facing cosine sensor, the basis of the coral PAR bands) in "
                 f"µmol m⁻² s⁻¹: rock tops {_u(L.get('rock_tops_meter'))}; sand {_u(L.get('sand'))}. Daily light "
                 f"integral = PAR x {L.get('dli_factor_mol_per_umol_s', 0):.4f} mol m⁻² d⁻¹ per µmol m⁻² s⁻¹ "
                 f"(photoperiod).</p>")
        if not L.get("calibrated"):
            H.append("<p class='muted'>Uncalibrated: absolute PAR depends on generic fixture data. Enter a "
                     "manufacturer PAR value or your own meter readings under lighting: calibration to calibrate "
                     "(spec Feature 3).</p>")
        for f_ in ("par_plan.png", "par_surfaces.png", "dli_surfaces.png"):
            H.append(_img(run.path / f_))

    # coral
    c = out["coral"]
    H.append(f"<h2>Coral placement</h2><p>{c['surface_points']} exposed rock-surface points (tops and upper sides). "
             f"Median speed there {_fmt_vel(c['speed_median_ms'], us)}, median turbulence intensity {c['ti_median']:.2f}. "
             + ("Scores combine flow and light." if c["light_available"] else
                "<b>Flow-only scores</b>: no PAR field was supplied for this run.") + "</p>")
    H.append("<table><tr><th>Profile</th><th>Mean score</th><th>Surface with score >= 0.7</th>"
             "<th>Jet-impingement penalised</th><th>Best spots (x, y, z): score, speed</th></tr>")
    for name, s in c["profiles"].items():
        spots = "<br>".join(f"{n + 1}. {_pos(b['position_m'], us)}: {b['score']:.2f}, {_fmt_vel(b['speed_ms'], us)}"
                            for n, b in enumerate(c["best_spots"][name][:5]))
        H.append(f"<tr><td>{name}</td><td>{s['mean_score']:.2f}</td><td>{100 * s['fraction_score_ge_0.7']:.0f} %</td>"
                 f"<td>{100 * s['fraction_jet_penalised']:.0f} %</td><td>{spots}</td></tr>")
    H.append("</table>" + _img(run.path / "coral_scores.png"))

    H.append("<h2>Flow field</h2>" + _img(run.path / "slices.png") + _img(run.path / "history.png"))
    H.append("<h2>Key assumptions</h2><table>" + "".join(f"<tr><td>{a}</td><td>{html.escape(b)}</td></tr>"
                                                          for a, b in ASSUMPTION_SUMMARY)
             + "</table><p class='muted'>Full list with justifications: ASSUMPTIONS.md and DECISIONS.md.</p>")
    H.append("</body></html>")
    Path(path).write_text("\n".join(H), encoding="utf-8")
    return Path(path)
