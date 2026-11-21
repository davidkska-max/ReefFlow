"""Research YAML -> return-pump entries, merged with the powerheads into reefflow/data/pump_products.yaml.

Q0 / H0 read by hand from each entry (the research strings mix sources). Curves: flow units
normalised to L/h or GPH, '~' dropped, ranges averaged. When the curve has no 0-head point and
its first point is above the rating, Q0 is the curve extrapolated to 0 head; when the curve's own
0-head value or end head exceeds the rating, the curve wins (stated in the basis)."""
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from reefflow import units as U  # noqa: E402

OUT = REPO / "reefflow/data/pump_products.yaml"
R = {p["id"]: p for p in yaml.safe_load((HERE / "research" / "returns.yaml").read_text(encoding="utf-8"))}

# id: (Q0, H0, curve to use: 'curve' | 'curve_alt' | None, curve label, extra note)
Q = {
    "ecotech_vectra_s2": ("1400 GPH", "11.5 ft", "curve", "read from EcoTech's chart", ""),
    "ecotech_vectra_m2": ("2000 GPH", "21.5 ft", "curve", "read from EcoTech's chart",
                          "EcoTech's page still shows the first-generation M1 chart (ends near 19.5 ft); the M2 is rated 21.5 ft."),
    "ecotech_vectra_l2": ("3100 GPH", "21.5 ft", "curve", "read from EcoTech's chart",
                          "EcoTech's page still shows the first-generation L1 chart (ends near 19.5 ft); the L2 is rated 21.5 ft."),
    "sicce_syncra_sdc_6": ("5500 L/h", "3.5 m", "curve", "Sicce catalogue table", ""),
    "sicce_syncra_sdc_7": ("7000 L/h", "5 m", "curve_alt", "read from Sicce's SDC data-sheet chart",
                           "Sicce's 2018 catalogue table is much lower (e.g. 1300 vs 2080 L/h at 4 m); the data-sheet chart agrees with "
                           "Red Sea's ReefRun 7000 chart (same ratings), so it is used."),
    "sicce_syncra_sdc_9": ("9000 L/h", "7 m", "curve_alt", "read from Sicce's SDC data-sheet chart",
                           "Sicce's 2018 catalogue table is much lower (4000 vs 6200 L/h at 4 m); the data-sheet chart agrees with "
                           "Red Sea's ReefRun 9000 chart (same ratings), so it is used."),
    "sicce_syncra_silent_1_5": ("1350 L/h", "1.8 m", "curve", "Sicce catalogue table", ""),
    "sicce_syncra_silent_2_0": ("2150 L/h", "2 m", "curve", "Sicce catalogue table", ""),
    "sicce_syncra_silent_3_0": ("2700 L/h", "3 m", "curve", "Sicce catalogue table", ""),
    "sicce_syncra_silent_3_5": ("2500 L/h", "3.7 m", "curve", "Sicce catalogue table", ""),
    "sicce_syncra_silent_4_0": ("3500 L/h", "3.7 m", "curve", "Sicce catalogue table", ""),
    "redsea_reefrun_5500": ("5500 L/h", "3.5 m", "curve", "read from Red Sea's chart (about +/-5 %)", ""),
    "redsea_reefrun_7000": ("7000 L/h", "5 m", "curve", "read from Red Sea's chart (about +/-5 %)", ""),
    "redsea_reefrun_9000": ("9000 L/h", "7 m", "curve", "read from Red Sea's chart (about +/-5 %)", ""),
    "redsea_reefrun_g2_4000": ("4000 L/h", "3.4 m", None, None, ""),
    "redsea_reefrun_g2_6000": ("6500 L/h", "4.4 m", None, None, ""),
    "redsea_reefrun_g2_8000": ("8300 L/h", "5.3 m", None, None, ""),
    "redsea_reefrun_g2_12000": ("12000 L/h", "5.8 m", None, None, "Manual 5.8 m, web page 5.9 m."),
    "neptune_cor_15": ("1500 GPH", "14 ft", None, None, "Neptune does not publish the shut-off head; 14 ft is BRS's figure."),
    "neptune_cor_20": ("2000 GPH", "20 ft", "curve", "read from Neptune's flow graph (assumed to be the COR-20)",
                       "Neptune does not publish the shut-off head; 20 ft is BRS's figure."),
    "jebao_dcp_3500": ("925 GPH", "11.5 ft", None, None, "Not in Jebao's own model table: retailer figures only."),
    "jebao_dcp_5000": ("5000 L/h", "4 m", None, None, "From a distributor copy of Jebao's manual (low confidence)."),
    "jebao_dcp_6500": ("6500 L/h", "4.5 m", None, None, "From a distributor copy of Jebao's manual (low confidence)."),
    "jebao_dcp_8000": ("8000 L/h", "5.2 m", None, None, "From a distributor copy of Jebao's manual (low confidence)."),
    "jebao_dcp_10000": ("10000 L/h", "6 m", None, None, "From a distributor copy of Jebao's manual (low confidence)."),
    "reefoctopus_varios_2": ("792 GPH", "13 ft", None, None, "Retailer table only."),
    "reefoctopus_varios_4": ("1050 GPH", "14.7 ft", "curve", "read from Reef Octopus's chart (hosted by a retailer)", ""),
    "reefoctopus_varios_6": ("1720 GPH", "16 ft", "curve", "read from Reef Octopus's chart (hosted by a retailer)", ""),
    "reefoctopus_varios_8": ("2700 GPH", "18 ft", "curve", "read from Reef Octopus's chart (hosted by a retailer)",
                             "The chart (2800 GPH at 0 head, 0 at 18.7 ft) is above the 2700 GPH / 18 ft rating; the chart is used."),
    "abyzz_a200": ("14500 L/h", "8.8 m", "curve", "read from Abyzz's chart",
                   "Rated 14,500 L/h at 100 % (17,000 L/h in boost mode, not modelled). The original-driver A200 is listed as not "
                   "available; the A200 IPU is the same pump with a new driver."),
    "aquamedic_dc_runner_3": ("3000 L/h", "2.7 m", None, None, ""),
    "aquamedic_dc_runner_5": ("5000 L/h", "3.5 m", None, None, ""),
    "aquamedic_dc_runner_9": ("9000 L/h", "4.8 m", None, None, ""),
    "tunze_silence_1073_040": ("3000 L/h", "3 m", "curve", "Tunze's chart (a straight-line drawing)", ""),
    "tunze_silence_electronic_1073_050": ("3000 L/h", "3.9 m", "curve", "Tunze's chart (a straight-line drawing)", ""),
    "eheim_compacton_2100": ("2100 L/h", "2.4 m", "curve", "Eheim's chart (a straight-line drawing)",
                             "Eheim's chart line starts at about 1720 L/h at 0 head, below the 2100 L/h rating; the chart is used."),
    "eheim_compacton_3000": ("3000 L/h", "2.7 m", "curve", "Eheim's chart (a straight-line drawing)",
                             "Eheim's chart line starts at about 2390 L/h at 0 head, below the 3000 L/h rating; the chart is used."),
    "eheim_compacton_5000": ("5000 L/h", "3.6 m", "curve", "Eheim's chart (a straight-line drawing)",
                             "Eheim's chart line starts at about 4370 L/h at 0 head, below the 5000 L/h rating; the chart is used."),
    "danner_supreme_mag_drive_7": ("700 GPH", "11.67 ft", "curve", "Danner pumping table", ""),
    "danner_supreme_mag_drive_9_5": ("950 GPH", "13.58 ft", "curve", "Danner pumping table", ""),
    "danner_supreme_mag_drive_12": ("1200 GPH", "13.58 ft", "curve", "Danner pumping table", ""),
}
missing = set(R) - set(Q)
assert not missing, missing


def flow_str(s):
    s = str(s).replace("~", "").strip()
    m = re.match(r"^([\d.]+)(?:-([\d.]+))?\s*(GPH|gph|l/h|L/h|m3/h)", s)
    if not m:
        raise ValueError(s)
    v = float(m.group(1)) if not m.group(2) else (float(m.group(1)) + float(m.group(2))) / 2
    u = m.group(3).lower()
    if u == "m3/h":
        return f"{v * 1000:g} L/h"
    return f"{v:g} {'GPH' if u == 'gph' else 'L/h'}"


def head_str(s):
    s = str(s).strip()
    m = re.match(r"^([\d.]+)\s*(ft|m)$", s)
    if not m:
        raise ValueError(s)
    return f"{float(m.group(1)):g} {m.group(2)}"


LINEAR_BASIS = ("no published curve: straight line Q0 (1 - H/H0). The 14 published DC-pump curves in this catalogue are "
                "matched by a straight line within 7.7 % of Q0 on average (bias -0.4 %); the generic sqrt shape reads 17 % high (D-845)")
P = {}
cmp_rows = []
for k, r in R.items():
    q0s, h0s, which, lab, extra = Q[k]
    q0, h0 = U.parse(q0s, "flow"), U.parse(h0s, "head")
    q0_basis = f"{r['brand']}: {r.get('max_flow')}"
    pts = None
    notes = [extra] if extra else []
    if which:
        raw = r.get(which)
        pts = [[head_str(h), flow_str(f)] for h, f in raw]
        si = sorted((U.parse(h, "head"), U.parse(f, "flow")) for h, f in pts)
        # drop a 0-flow end point beyond which nothing is defined; take H0 from it if larger
        if si[-1][1] <= 1e-12:
            if si[-1][0] > h0 + 1e-9:
                notes.append(f"The curve reaches zero flow at {U.from_si(si[-1][0], 'ft'):.1f} ft, above the "
                             f"{U.from_si(h0, 'ft'):.1f} ft rating; the curve's end is used as the shut-off head.")
                h0 = si[-1][0]
        if si[-1][0] >= h0 - 1e-9 and si[-1][1] > 0:
            h0 = si[-1][0] * 1.02
            notes.append("Curve continues past the rated shut-off head: shut-off set just beyond its last point.")
        if si[0][0] < 1e-9:
            if si[0][1] > q0 * (1 + 1e-6):
                q0 = si[0][1]
                q0_basis += " (the curve's 0-head value, above the rating, is used)"
        else:
            (ha, qa), (hb, qb) = si[0], si[1]
            q_ext = qa + (qa - qb) * ha / (hb - ha)
            if q_ext > q0:
                q0 = q_ext
                q0_basis += f" (the table's first points extrapolate to {U.from_si(q0, 'GPH'):.0f} GPH at 0 head, above the rating; that is used)"
        pts = [[f"{U.from_si(h, 'ft'):.4g} ft", f"{U.from_si(f, 'GPH'):.5g} GPH"] for h, f in si if h < h0 - 1e-9]
        # how far is the generic sqrt curve from the published one?
        if "straight-line" not in lab:
            err = [abs(q0 * math.sqrt(max(1 - h / h0, 0)) - f) / q0 for h, f in si if 0 < h < h0]
            cmp_rows.append((k, max(err) * 100, float(np.mean(err)) * 100, lab))
    st = str(r.get("status") or "")
    P[k] = {
        "kind": "return", "brand": r["brand"], "model": r["model"],
        "description": f"{r['brand']} {r['model']} (return pump)",
        "rated_flow": f"{U.from_si(q0, 'GPH'):.5g} GPH", "rated_flow_basis": q0_basis,
        "pump_curve": {"max_head": f"{U.from_si(h0, 'ft'):.4g} ft", **({"points": pts, "basis": lab} if pts else
                       {"shape": "linear", "basis": LINEAR_BASIS} if r.get("dc_controllable") else {})},
        "power": str(r.get("power_max") or "").split("(")[0].strip() or None,
        "outlet_size": r.get("outlet_size"), "inlet_size": r.get("inlet_size"),
        "dc_controllable": r.get("dc_controllable"),
        "status": "current" if st.startswith("current") else "discontinued" if st.startswith("discontinued") else "unknown",
        "checked": str(r.get("checked") or "2026-11-12"),
        "sources": [s for s in r.get("sources") or [] if str(s).startswith("http")],
        "notes": " ".join(notes + [" ".join(str(r.get("notes") or "").split())]).strip(),
    }
    P[k] = {kk: vv for kk, vv in P[k].items() if vv is not None}

PH = yaml.safe_load((HERE / "research" / "_powerheads_built.yaml").read_text(encoding="utf-8"))
head = """# ReefFlow pump catalogue (D-844): real powerheads and return pumps, checked 2026-11-12.
#
# Same fields as the generic archetypes in pump_library.yaml (kind, outlet, rated_flow,
# pump_curve, ...) plus brand, model, sources and the basis of each number. Loaded by
# reefflow.pumps.load_products(); modelling defaults that are not product data (derating 0.7,
# turbulence 0.05, return nozzle 19 mm, static head 4 ft, fittings 1 ft) are filled in there.
#
# Powerheads: rated_flow is the maker's maximum (free-flow) rating; delivered = rated x derating.
#   outlet: published (Tunze), retailer-published (Aqua Medic), measured from the maker's drawing
#   (Sicce), or, when not published, sized for the median jet velocity of those pumps (A-844).
# Returns: rated_flow is the zero-head flow Q0, pump_curve.max_head the shut-off head H0;
#   pump_curve.points is the maker's head-flow curve where published, used as given (D-845).
# Regenerate with the research scripts; keep the *_basis fields in step with the numbers.
"""
OUT.write_text(head + yaml.safe_dump({**PH, **P}, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
print(len(PH), "powerheads +", len(P), "returns,", sum(1 for v in P.values() if v["pump_curve"].get("points")), "with curves")
print("\ngeneric sqrt curve vs published (fraction of Q0), measured/charted curves only:")
for k, mx, mean, lab in cmp_rows:
    print(f"  {k:34s} max {mx:5.1f} %  mean {mean:5.1f} %   ({lab})")
print(f"  ALL: mean of means {np.mean([c[2] for c in cmp_rows]):.1f} %, worst {max(c[1] for c in cmp_rows):.1f} %")
