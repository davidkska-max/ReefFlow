"""Research YAML + PAR fits -> reefflow/light/data/fixture_products.yaml (curated).

Photon output (no maker publishes PPF):
  * fitted to independent in-air PAR grids (Reef Builders) or the maker's own centre PAR, using
    the ReefFlow emitter model (fit_par.field): XR30 G6 Pro, Blade Coral Grow 21, CoralCare Gen2,
    Atlantik iCon, Red Sea ReefLED (surface centre value ~ in-air value at the mounting height);
  * same-family fixtures use the fitted sibling's umol/J;
  * the rest: power x the mean umol/J of the three independently measured fixtures.
"""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import fit_par as F  # noqa: E402
from reefflow.light.fixtures import Distribution, beam_angle_to_m  # noqa: E402

OUT = REPO / "reefflow/light/data/fixture_products.yaml"
R = {p["id"]: p for p in yaml.safe_load((HERE / "research" / "lights.yaml").read_text(encoding="utf-8"))}
IN, CM = 0.0254, 0.01


def m_of(beam):
    return beam_angle_to_m(math.radians(beam))


def fit_fixed(L, W, beam, data):
    """PPF for a fixed beam: geometric-mean ratio of measured to model PAR. data [(h m, pts, PAR)]."""
    m = m_of(beam)
    model = np.array([F.field(1.0, m, L, W, h, pts).mean() for h, pts, _ in data])
    meas = np.array([v for _, _, v in data])
    ppf = math.exp(np.mean(np.log(meas / model)))
    err = np.log(ppf * model / meas)
    return ppf, float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err)))


def rb(L, W, d):
    """Reef Builders data [(h in, kind, PAR)] -> [(h m, pts, PAR)]."""
    return [(h * IN, {"peak": F.C, "g3": F.G3, "g2": F.G2}[k], v) for h, k, v in d]


FITS = {}
# --- independent grids (Reef Builders, in air, full power, measured watts)
L, W, w, d = F.FIX["XR30 G6 Pro"]
ppf, rms, mx = fit_fixed(L, W, 125, rb(L, W, d))
FITS["xr30"] = dict(ppf=ppf, eff=ppf / w, rms=rms, mx=mx, beam=125, watts=w)
L, W, w, d = F.FIX["Blade Coral Grow 21"]
ppf, rms, mx = fit_fixed(L, W, 95, rb(L, W, d))
FITS["blade"] = dict(ppf=ppf, eff=ppf / w, rms=rms, mx=mx, beam=95, watts=w)
# CoralCare: no published beam -> fit both (as fit_par.py: beam 103 deg)
L, W, w, d = F.FIX["CoralCare Gen2"]
best = min(((b,) + fit_fixed(L, W, b, rb(L, W, d)) for b in range(80, 141)), key=lambda t: t[2])
FITS["coralcare"] = dict(ppf=best[1], eff=best[1] / w, rms=best[2], mx=best[3], beam=best[0], watts=w)
# --- Atlantik iCon: DaniReef in-air centre readings (light base to sensor base), 229 W measured, 120 deg lens
icon = [(0.17, F.C, 1774), (0.37, F.C, 966), (0.57, F.C, 620)]
best = min(((b,) + fit_fixed(0.615, 0.238, b, icon) for b in range(30, 131, 5)), key=lambda t: t[2])
FITS["icon"] = dict(ppf=best[1], eff=best[1] / 229, rms=best[2], mx=best[3], beam=best[0], watts=229)
INDEP = [FITS[k]["eff"] for k in ("xr30", "blade", "coralcare")]
EFF_DEFAULT = round(float(np.mean(INDEP)), 2)
for k, v in FITS.items():
    print(f"fit {k:10s}: PPF {v['ppf']:6.0f} umol/s  {v['eff']:.2f} umol/J  beam {v['beam']} deg  rms {100 * v['rms']:.1f} %  max {100 * v['mx']:.0f} %")
print("default efficacy (mean of independent fits):", EFF_DEFAULT, [round(x, 2) for x in INDEP])


def redsea_fit(L, W, h, par):
    ppf, _, _ = fit_fixed(L, W, 120, [(h, F.C, par)])
    return ppf


# channel helpers -------------------------------------------------------------
def chans(*items):
    """items (name, weight, spectrum) -> {name: {share, spectrum}} normalised."""
    tot = sum(w for _, w, _ in items)
    return {n: {"share": round(w / tot, 4), "spectrum": s} for n, w, s in items}


AI7 = lambda k: chans(("cool_white", 4 * k, "cool_white"), ("blue", 4 * k, "blue"), ("royal_blue", 4 * k, "royal_blue"),
                      ("red", k, "red"), ("green", k, "green"), ("violet", k, "indigo"), ("uv", k, "violet"))
XR = lambda cw, ww, rbl, bl, v430, v415, uv, gr, red: chans(
    ("cool_white", cw, "cool_white"), ("warm_white", ww, "warm_white"), ("royal_blue", rbl, "royal_blue"), ("blue", bl, "blue"),
    ("indigo", v430, "indigo"), ("violet", v415 + uv, "violet"), *([("green", gr, "green")] if gr else []), ("red", red, "red"))
BLADE_GROW = chans(("royal_blue", 12, "royal_blue"), ("blue", 6, "blue"), ("cool_white", 6, "cool_white"))
BLADE_GLOW = chans(("royal_blue", 12, "royal_blue"), ("indigo", 6, "indigo"), ("violet", 6, "violet"))
KESSIL = chans(("blue", 0.6, "royal_blue"), ("white", 0.25, "cool_white"), ("purple", 0.1, "indigo"), ("red", 0.03, "red"), ("green", 0.02, "green"))
REDSEA = lambda blue_w, white_w: chans(("reef_spec_blue", blue_w, "royal_blue"), ("white", white_w, "cool_white"))
OR3_DAY = lambda c: chans(("violet", c[0], "violet"), ("indigo", c[1] + c[2], "indigo"), ("royal_blue", c[3], "royal_blue"),
                          ("blue", c[4] + c[5], "blue"), ("amber", c[6], "amber"), ("white", c[7] + c[8], "cool_white"))
OR3_BLUE = chans(("violet", 4, "violet"), ("indigo", 12, "indigo"), ("royal_blue", 11, "royal_blue"), ("blue", 9, "blue"))
ICON = chans(("ch1_sun", 13, "warm_white"), ("ch2_noon", 13, "cool_white"), ("ch3_cyan_blue", 13, "cyan"),
             ("ch4_blue", 13, "royal_blue"), ("ch5_violet", 13, "indigo"), ("ch6_uv_violet", 13, "violet"))
GENERIC_REEF = chans(("blue", 0.55, "royal_blue"), ("white", 0.3, "cool_white"), ("violet", 0.1, "indigo"), ("red", 0.05, "red"))

ASSUMED_BEAM = 110        # deg, when not published: mid of the fitted beams (95-125 deg)
EST = lambda w: (w * EFF_DEFAULT, f"estimated: {w:g} W x {EFF_DEFAULT} umol/J, the mean of three fixtures fitted to independent PAR "
                                  f"grids (Reef Builders: {', '.join(f'{x:.2f}' for x in INDEP)} umol/J). Calibrate with a PAR meter if you have one.")
FAM = lambda w, key, what: (w * FITS[key]["eff"], f"estimated: {w:g} W x {FITS[key]['eff']:.2f} umol/J, the efficacy fitted for the {what} (same family)")

xr30, bl, cc, ic = FITS["xr30"], FITS["blade"], FITS["coralcare"], FITS["icon"]
P = {}


def add(pid, typ, power, L, W, beam, beam_basis, ppf_basis_pair, channels, notes, emitters=None, dist=None, status=None):
    r = R[pid]
    ppf, basis = ppf_basis_pair
    P[pid] = {
        "brand": r["brand"], "model": r["model"], "type": typ,
        "description": f"{r['brand']} {r['model']}",
        "footprint": {"length": f"{L * 1000:.0f} mm", "width": f"{W * 1000:.0f} mm"},
        "emitters": emitters or {"nx": 1, "ny": 1, "size": [f"{L * 1000:.0f} mm", f"{W * 1000:.0f} mm"]},
        "distribution": dist or {"type": "beam_angle", "beam_angle": f"{beam:g} deg"},
        "beam_basis": beam_basis,
        "power": f"{power:g} W", "ppf": f"{ppf:.0f} umol/s", "ppf_basis": basis,
        "channels": channels,
        "par_data": [{k: v for k, v in x.items() if v is not None} for x in (r.get("par_data") or [])],
        "status": status or ("current" if str(r.get("status", "")).startswith("current") else
                             "discontinued" if "discontinued" in str(r.get("status", "")) or "superseded" in str(r.get("status", "")) else "unknown"),
        "checked": str(r.get("checked") or "2026-11-12"),
        "sources": [s for s in r.get("sources") or [] if str(s).startswith("http")],
        "notes": notes,
        "research_notes": " ".join(str(r.get("notes") or "").split()),
    }


FP = "Footprint is the housing size (no maker publishes the emitting area). "
CH = "Channel shares follow the LED counts per colour (all LEDs assumed equal output). "

# ---- EcoTech Radion G6
add("ecotech_radion_xr30_g6_pro", "led_panel", 215, 0.30, 0.18, 125, "EcoTech: standard optic 124-126 deg",
    (xr30["ppf"] * 215 / xr30["watts"], f"fitted to Reef Builders' in-air PAR grid (24 and 30 in, 7 readings, 215.9 W measured) with the published 125 deg beam: "
     f"{xr30['eff']:.2f} umol/J, readings matched to {100 * xr30['rms']:.0f} % rms (max {100 * xr30['mx']:.0f} %)"),
    XR(16, 8, 32, 16, 8, 4, 6, 6, 4), FP + CH)
add("ecotech_radion_xr30_g6_blue", "led_panel", 215, 0.30, 0.18, 125, "EcoTech: standard optic 124-125 deg",
    FAM(215, "xr30", "XR30 G6 Pro"), XR(8, 4, 32, 32, 12, 4, 6, 0, 2), FP + CH)
add("ecotech_radion_xr15_g6_pro", "led_puck", 95, 0.18, 0.18, 125, "EcoTech: standard optic 124-126 deg",
    FAM(95, "xr30", "XR30 G6 Pro"), XR(8, 4, 16, 8, 4, 2, 3, 3, 2), FP + CH)
add("ecotech_radion_xr15_g6_blue", "led_puck", 95, 0.18, 0.18, 125, "EcoTech: standard optic 124-126 deg",
    FAM(95, "xr30", "XR30 G6 Pro"), XR(4, 2, 16, 16, 6, 2, 3, 0, 1), FP + CH)
# ---- AI
add("ai_blade_coral_grow_21", "led_bar", 40, 0.536, 0.07, 95, "AI: EdgeField optics, 95 deg spread",
    (bl["ppf"], f"fitted to Reef Builders' in-air PAR grid (24 and 30 in, 7 readings, 42.2 W measured) with the published 95 deg beam: "
     f"{bl['eff']:.2f} umol/J, readings matched to {100 * bl['rms']:.0f} % rms (max {100 * bl['mx']:.0f} %)"),
    BLADE_GROW, FP + CH)
add("ai_blade_coral_grow_48", "led_bar", 100, 1.222, 0.07, 95, "AI: EdgeField optics, 95 deg spread",
    FAM(100, "blade", "Blade Coral Grow 21 in"), BLADE_GROW, FP + CH)
add("ai_blade_coral_glow_48", "led_bar", 100, 1.222, 0.07, 95, "AI: 95 deg spread",
    FAM(100, "blade", "Blade Coral Grow 21 in"), BLADE_GLOW, FP + CH + "Glow is the violet-heavy version; its 405 nm LEDs sit at the PAR band edge.")
add("ai_prime_16hd", "led_puck", 59, 0.124, 0.124, ASSUMED_BEAM, f"not published (TIR optics): {ASSUMED_BEAM} deg assumed",
    EST(59), AI7(1), FP + CH + "AI's spec table gives 59 W; the same page also says 55 W from the wall.")
add("ai_hydra_32hd", "led_puck", 95, 0.185, 0.137, ASSUMED_BEAM, f"not published (TIR optics): {ASSUMED_BEAM} deg assumed",
    EST(95), AI7(2), FP + CH + "AI lists 95 W, BRS 90 W. LED counts per colour from retailer listings.")
add("ai_hydra_64hd", "led_puck", 135, 0.302, 0.137, ASSUMED_BEAM, f"not published (TIR optics): {ASSUMED_BEAM} deg assumed",
    EST(135), AI7(4), FP + CH + "LED counts per colour from retailer listings.")
# ---- Philips
add("philips_coralcare_gen2", "led_panel", 170, 0.46, 0.41, cc["beam"],
    f"not published: {cc['beam']} deg fitted to the PAR grid together with the output",
    (cc["ppf"] * 170 / cc["watts"], f"fitted to Reef Builders' in-air PAR grid (24 and 30 in, 7 readings, 167.9 W measured), beam and output together: "
     f"{cc['eff']:.2f} umol/J, readings matched to {100 * cc['rms']:.0f} % rms (max {100 * cc['mx']:.0f} %)"),
    chans(("daylight", 0.5, "cool_white"), ("blue", 0.5, "royal_blue")),
    FP + "Two user channels (daylight and blue); the colour mix of the 68 LEDs is not published, an even split is assumed.")
# ---- Orphek
add("orphek_atlantik_icon", "led_panel", 230, 0.615, 0.238, ic["beam"],
    f"Orphek's standard lens is 120 deg (options 90/60/45/15/5), but the published readings fall off like a {ic['beam']} deg beam; "
    "the lens on the tested unit is not stated. With 120 deg the fit needs 2.67 umol/J (19 % rms), which is not plausible for a reef LED",
    (ic["ppf"] * 230 / ic["watts"], f"fitted to DaniReef's in-air centre PAR at 17, 37 and 57 cm (229 W measured, republished by Orphek), beam and output together: "
     f"{ic['eff']:.2f} umol/J, readings matched to {100 * ic['rms']:.0f} % rms (max {100 * ic['mx']:.0f} %)"),
    ICON, FP + "Six channels of 13 LEDs each; each channel is modelled with one representative spectrum.")
add("orphek_atlantik_icon_compact", "led_panel", 130, 0.345, 0.238, ic["beam"], f"not stated for the Compact: {ic['beam']} deg as fitted for the Atlantik iCon",
    FAM(130, "icon", "Atlantik iCon"), ICON, FP + "Orphek gives 123-130 W. Per-channel counts not published; the iCon layout is assumed.")
or3_basis = "lens options 15/30/60/90 deg, default not stated: 90 deg assumed"
add("orphek_or3_120_reef_day_plus", "led_bar", 60, 1.235, 0.05, 90, or3_basis, EST(60),
    OR3_DAY([2, 2, 4, 7, 4, 2, 2, 5, 6]), FP + CH + "The 730-740 nm LEDs are outside the PAR band and left out. "
    "Orphek's catalogue shows 240 PAR at 37 cm in air, but does not say which bar or lens was tested.")
add("orphek_or3_120_blue_plus", "led_bar", 60, 1.235, 0.05, 90, or3_basis, EST(60), OR3_BLUE, FP + CH)
add("orphek_or3_60_reef_day_plus", "led_bar", 30, 0.635, 0.05, 90, or3_basis, EST(30),
    OR3_DAY([1, 1, 2, 4, 2, 1, 2, 1, 3]), FP + CH + "The 730-740 nm LEDs are outside the PAR band and left out.")
# ---- Kessil
kb = f"bare-unit beam not published: {ASSUMED_BEAM} deg assumed"
add("kessil_a360x", "led_puck", 90, 0.11, 0.11, ASSUMED_BEAM, kb, EST(90), KESSIL,
    FP + "Kessil's single Dense Matrix array is tuned with two knobs (colour, intensity); the channel mix here is a typical "
    "'Tuna Blue' setting, not a published one. Superseded by the A360XE. Optional 55/35 deg reflectors are not modelled.")
add("kessil_a500x", "led_puck", 185, 0.133, 0.133, 130, "130 deg standard (BRStv; not stated by Kessil)", EST(185), KESSIL,
    FP + "Channel mix is a typical 'Tuna Blue' setting, not a published one. The included 55 deg reflector is not modelled.")
add("kessil_ap9x", "led_panel", 185, 0.418, 0.145, ASSUMED_BEAM, f"elliptical optics, angle not published: {ASSUMED_BEAM} deg assumed",
    EST(185), KESSIL, FP + "Dimensions are the AP9XE's (successor), assumed equal. Channel mix is a typical setting, not a published one.")
add("kessil_a160we", "led_puck", 40, 0.063, 0.063, ASSUMED_BEAM, kb, EST(40), KESSIL,
    "Body diameter from retailers. No current Kessil product page (the manual is still on kessil.com). Channel mix is a typical "
    "setting, not a published one.", status="unknown")
# ---- Neptune
add("neptune_sky", "led_panel", 200, 0.325, 0.235, ASSUMED_BEAM, f"diffuser + lenses, angle not published: {ASSUMED_BEAM} deg assumed",
    EST(200), GENERIC_REEF, FP + "Neptune gives 200 W nominal (BRS: 219 W max). The colour mapping of its 4 channels is not published; "
    "a typical reef mix is assumed.")
# ---- Red Sea: fitted to the maker's surface-centre PAR at the stated mounting height (Lambertian assumed)
rs = [("redsea_reefled_50", 50, 0.12, 0.12, 0.17, 500, (40, 10), "discontinued"),
      ("redsea_reefled_90", 90, 0.155, 0.135, 0.21, 500, (80, 10), "discontinued"),
      ("redsea_reefled_160s", 160, 0.18, 0.16, 0.31, 550, (140, 20), "discontinued"),
      ("redsea_reefled_g2_60", 60, 0.12, 0.12, 0.17, 500, (50, 10), None),
      ("redsea_reefled_g2_115", 115, 0.155, 0.135, 0.21, 500, (95, 20), None),
      ("redsea_reefled_g2_170", 170, 0.18, 0.16, 0.31, 550, (140, 30), None)]
for pid, w, L, W, h, par, (bw, ww), st in rs:
    ppf = redsea_fit(L, W, h, par)
    est = EST(w)
    add(pid, "led_puck", w, L, W, ASSUMED_BEAM, f"not published: {ASSUMED_BEAM} deg assumed", (
        est[0], est[1] + f" Red Sea's own figure ({par} PAR at the surface, centre, light {h * 100:.0f} cm up, settings not stated) "
                         f"would imply {ppf / w:.2f} umol/J with a Lambertian beam; it is shown but not used, because the settings "
                         "are not given and models of different wattage share the same figure."),
        REDSEA(bw, ww), FP + "Channels: REEF-SPEC blue and white (moonlight left out)"
        + (" in proportion to their published watts." if st else "; the G2 channel watts are not published, a split like the G1's is assumed."),
        status=st)
# ---- Nicrew (no PAR published)
NICREW = chans(("cool_white", 15, "cool_white"), ("warm_white", 9, "warm_white"), ("royal_blue", 30, "royal_blue"), ("blue", 30, "blue"),
               ("violet", 18, "violet"), ("green", 3, "green"), ("red", 3, "red"))
add("nicrew_hyperreef_100_gen2", "led_panel", 100, 0.249, 0.160, 120, "retailers: 120 deg wide-angle lens", EST(100), NICREW,
    FP + CH + "LED counts from Nicrew's HyperReef 200 Gen 2 manual, the same mix assumed for the 100 (violet and UV merged). "
    "Nicrew recommends mounting it 8-10 in above the water; maximum coverage 24 x 24 in.")
add("nicrew_hyperreef_100_gen1", "led_panel", 100, 0.249, 0.160, 90, "90 deg (a retailer's Gen 1 / Gen 2 comparison)", EST(100), NICREW,
    "Housing size and LED mix of the original HyperReef 100 were not found: the Gen 2's are assumed. Measure your fixture and "
    "check the colour channels if you need more than an estimate.")

# ---- ATI T5
add("ati_sunpower_6x54", "t5", 324, 1.188, 0.34, 0, "", (324 * 1.0, "estimated: 6 x 54 W lamps x 1.0 umol/J (the generic T5 fixture assumption, "
    "A-404); depends on the lamps you fit"),
    chans(("actinic", 0.5, "t5_actinic"), ("daylight", 0.5, "t5_10k")),
    "Six 48 in T5 HO lamps under individual parabolic reflectors; lamp choice is yours (3 actinic + 3 daylight assumed). "
    "Power is the lamps' nominal 6 x 54 W.",
    emitters={"nx": 1, "ny": 6, "size": ["1150 mm", "16 mm"]}, dist={"type": "lambertian", "cutoff": "85 deg"})
P["ati_sunpower_6x54"]["beam_basis"] = "parabolic reflectors: Lambertian with an 85 deg cut-off (generic T5 model)"

head = f"""# ReefFlow light-fixture catalogue (D-844). Real products, checked 2026-11-12.
#
# Same fields as the generic archetypes in fixture_library.yaml, plus brand, model, sources
# and the basis of each modelled number. No maker publishes total photon output (PPF), so
# every `ppf` here is DERIVED (see ppf_basis):
#   * fitted to published PAR readings with the ReefFlow emitter model where readings exist
#     (Reef Builders in-air grids; DaniReef in-air readings; Red Sea's own surface figures);
#   * same-family fixtures use the fitted sibling's efficacy (umol/J);
#   * otherwise power x {EFF_DEFAULT} umol/J, the mean of the three independently fitted fixtures.
# Footprints are housing sizes. Uncalibrated light results stay labelled "estimated";
# a PAR-meter calibration (lighting.calibration) replaces these numbers.
# Regenerate with the research scripts; do not hand-edit numbers without updating ppf_basis.
"""
OUT.write_text(head + yaml.safe_dump(P, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
print(len(P), "fixtures")
for k, v in P.items():
    print(f"  {k:32s} {v['power']:>6s} {v['ppf']:>12s}  {v['distribution']}  {v['status']}")
