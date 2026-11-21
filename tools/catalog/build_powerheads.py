"""Research YAML -> reefflow/data/pump_products.yaml (powerheads; returns are added by build_returns.py).

Outlet rule (A-844): published outlet (Tunze; Aqua Medic via retailer; Sicce measured from the
maker's drawing) is used as is. Otherwise the outlet is sized so the jet leaves at the median
outlet velocity of those published pumps at rated flow, capped by the wet-side diameter when
that is known. Cross-flow pumps: body length x smallest body dimension (upper bound)."""
import math
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
OUT = REPO / "reefflow/data/pump_products.yaml"
GPH, LPH = 3.785411784e-3 / 3600, 1e-3 / 3600
R = {p["id"]: p for p in yaml.safe_load((HERE / "research" / "powerheads.yaml").read_text(encoding="utf-8"))}

# rated (maximum) flow as published, read by hand from each entry; 'basis' is what the number means
FLOW = {
    "ecotech_mp10wqd": (1500, "GPH", "EcoTech: '1,500+ gph'"),
    "ecotech_mp40wqd": (4500, "GPH", "EcoTech: '4,500+ gph'"),
    "ecotech_mp60wqd": (7500, "GPH", "EcoTech: '7,500+ gph'"),
    "ecotech_mp10mqd": (2500, "GPH", "EcoTech's current page (retailers still print 1,500 gph; EcoTech gives no reason for the change)"),
    "ecotech_mp40mqd": (5000, "GPH", "EcoTech's current page"),
    "ecotech_mp60mqd": (9000, "GPH", "EcoTech's current page"),
    "neptune_wav": (4000, "GPH", "Neptune: '~4000+ gph'"),
    "ai_nero_3": (2000, "GPH", "AI"),
    "ai_nero_5": (3000, "GPH", "AI"),
    "maxspect_gyre_xf330": (9000, "L/h", "Maxspect"),
    "maxspect_gyre_xf350": (20000, "L/h", "Maxspect"),
    "maxspect_gyre_xf280": (6000, "GPH", "BRS listing (2017 launch coverage: 22,500 l/h, the same within 1 %)"),
    "maxspect_mj_gf308": (8000, "L/h", "Maxspect"),
    "maxspect_mj_gf316": (16000, "L/h", "Maxspect"),
    "redsea_reefwave_25": (7500, "L/h", "Red Sea: 'true maximum flow rate'"),
    "redsea_reefwave_45": (15000, "L/h", "Red Sea: 'true maximum flow rate'"),
    "tunze_nanostream_6040": (4500, "L/h", "Tunze: approx. 200-4,500 l/h with the 7020 controller (maximum used)"),
    "tunze_nanostream_6055": (5500, "L/h", "Tunze: 1,000-5,500 l/h (maximum used)"),
    "tunze_nanostream_6095": (9500, "L/h", "Tunze: 2,000-9,500 l/h (maximum used)"),
    "tunze_stream_6105_eco": (12000, "L/h", "Tunze: approx. 3,000-12,000 l/h at 12 V (maximum used)"),
    "tunze_nanostream_6075": (7500, "L/h", "Tunze: 2,700-7,500 l/h (maximum used)"),
    "tunze_stream_3_6150": (15000, "L/h", "Tunze: 2,500-15,000 l/h (maximum used)"),
    "sicce_xstream_sdc": (8500, "L/h", "Sicce: 1,000-8,500 l/h Q-max (maximum used)"),
    "sicce_xstream_8000": (8000, "L/h", "Sicce: Q-max"),
    "sicce_voyager_nano_1000": (1000, "L/h", "Sicce: Q-max"),
    "sicce_voyager_nano_2000": (2000, "L/h", "Sicce: Q-max"),
    "sicce_voyager_hp10": (15000, "L/h", "Sicce: Q-max"),
    "jebao_mlw_10": (4000, "L/h", "Jebao (retailer copies; low confidence)"),
    "jebao_mlw_20": (10000, "L/h", "Jebao (retailer copies; low confidence)"),
    "jebao_mlw_30": (13000, "L/h", "Jebao (retailer copies; low confidence)"),
    "jebao_slw_20": (10000, "L/h", "Jebao (retailer copies; low confidence)"),
    "jebao_slw_30": (13000, "L/h", "Jebao (retailer copies; low confidence)"),
    "aquamedic_ecodrift_4_3": (4000, "L/h", "Aqua Medic: up to 4,000 l/h"),
    "aquamedic_ecodrift_8_3": (8000, "L/h", "Aqua Medic: up to 8,000 l/h"),
    "aquamedic_ecodrift_15_3": (15000, "L/h", "Aqua Medic: up to 15,000 l/h"),
}
# outlet diameter that is a real (published or drawing-measured) opening
REAL = {k for k, v in R.items() if str(v.get("outlet_basis", "")).startswith(("manufacturer", "retailer-published", "derived (drawing)"))}
# upper bounds (wet side / body) and stated wet-side sizes for those with no outlet
BOUND = {k: float(str(v["outlet"]["diameter"]).split()[0]) / 1000 for k, v in R.items()
         if v.get("outlet") and v["outlet"]["shape"] == "circle" and k not in REAL}
BOUND.update({"ai_nero_3": 1.54 * 0.0254, "ai_nero_5": 2.1 * 0.0254})       # smallest wet-side dimension


def q_si(k):
    v, u, _ = FLOW[k]
    return v * (GPH if u == "GPH" else LPH)


vel = {k: q_si(k) / (math.pi * (float(str(R[k]["outlet"]["diameter"]).split()[0]) / 1000) ** 2 / 4) for k in sorted(REAL)}
U_MED = float(np.median(list(vel.values())))
print(f"outlet velocity at rated flow, {len(vel)} pumps with a real outlet: median {U_MED:.2f} m/s, range "
      f"{min(vel.values()):.2f}-{max(vel.values()):.2f}")
for k, u in vel.items():
    print(f"   {k:28s} {u:.2f} m/s")

P = {}
for k, r in R.items():
    if k not in FLOW:
        raise SystemExit(f"no flow for {k}")
    v, u, fb = FLOW[k]
    q = q_si(k)
    if r["pump_type"] == "crossflow":
        o = r["outlet"]
        outlet = {"shape": "rect", "width": o["width"], "height": o["height"]}
        ob = ("upper bound: body length x smallest body dimension (the slot size is not published). A slot smaller than this "
              "gives a stronger jet, so the momentum here leans low.")
    elif k in REAL:
        outlet = {"shape": "circle", "diameter": r["outlet"]["diameter"]}
        b = str(r["outlet_basis"])
        ob = ("manufacturer" if b.startswith("manufacturer") else
              "retailer-published (Aqua Medic's own manual could not be read)" if b.startswith("retailer") else
              "measured from Sicce's dimensioned drawing, about +/-3 mm")
    else:
        d = math.sqrt(4 * q / (math.pi * U_MED))
        cap = BOUND.get(k)
        if cap and cap < d:
            d, how = cap, f"capped at the wet-side size {cap * 1000:.0f} mm"
        else:
            how = f"below the wet-side size {cap * 1000:.0f} mm" if cap else "no wet-side size published"
        outlet = {"shape": "circle", "diameter": f"{d * 1000:.0f} mm"}
        ob = (f"not published: sized so the jet leaves at {U_MED:.2f} m/s at rated flow, the median of the {len(vel)} pumps with "
              f"a published or measured outlet ({how})")
    st = str(r.get("status") or "")
    P[k] = {
        "kind": "powerhead", "brand": r["brand"], "model": r["model"], "pump_type": r["pump_type"],
        "description": f"{r['brand']} {r['model']} ({'cross-flow' if r['pump_type'] == 'crossflow' else 'propeller'} powerhead)",
        "rated_flow": f"{v:g} {u}", "rated_flow_basis": fb + "; delivered flow = rated x derating 0.7 (A-205)",
        "outlet": outlet, "outlet_basis": ob,
        "power": str(r.get("power_max") or "").strip() or None,
        "dimensions": r.get("dimensions"), "tank_range": r.get("tank_range"), "modes": r.get("modes"),
        "status": "current" if st.startswith("current") else "discontinued" if st.startswith("discontinued") else "unknown",
        "checked": str(r.get("checked") or "2026-11-12"),
        "sources": [s for s in r.get("sources") or [] if str(s).startswith("http")],
        "notes": " ".join(str(r.get("notes") or "").split()),
    }
    P[k] = {kk: vv for kk, vv in P[k].items() if vv is not None}

(HERE / "research" / "_powerheads_built.yaml").write_text(yaml.safe_dump(P, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
print(len(P), "powerheads")
for k, v in P.items():
    o = v["outlet"]
    os_ = o.get("diameter") or f"{o['width']} x {o['height']}"
    print(f"  {k:28s} {v['rated_flow']:>10s}  {os_:>16s}  {v['status']}")
