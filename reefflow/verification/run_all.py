"""Run the Phase-1 verification suite and write verification/VERIFICATION.md.

    python -m reefflow.verification.run_all            # all cases
    python -m reefflow.verification.run_all jet cavity # selected cases
    python -m reefflow.verification.run_all --report   # rebuild report from saved JSON only
"""
import json
import sys

from .common import RESULTS, ROOT

CASES = ["poiseuille", "cavity", "cylinder", "jet", "mass_box", "powerhead"]
DIRS = {"poiseuille": "01_poiseuille", "cavity": "02_cavity", "cylinder": "03_cylinder",
        "jet": "04_jet", "mass_box": "05_mass_box", "grid": "06_grid", "powerhead": "07_powerhead"}


def _load(name, filename="results.json"):
    p = RESULTS / DIRS[name] / filename
    return json.loads(p.read_text()) if p.exists() else None


def run_poiseuille_f64():
    """fp64 Poiseuille sweep on the CPU backend (CUDA fp64 vector arrays fault in Taichi 1.7.4)."""
    import os, subprocess
    env = dict(os.environ, REEFFLOW_ARCH="cpu", REEFFLOW_PRECISION="f64")
    code = ("from reefflow import backend; backend.init();"
            "from reefflow.verification import poiseuille as P;"
            "P.run(combos=[('D3Q19','bgk'),('D3Q19','rr'),('D3Q27','rr')])")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def report() -> str:
    L = ["# Verification report (Phase 1: cases 1-5; Phase 2: cases 6-7)", ""]
    first = True
    for prec in ("f32", "f64"):
        r = _load("poiseuille", f"results_{prec}.json")
        if not r:
            continue
        if first:
            L += ["## 1. Poiseuille channel (force-driven)", ""]
            first = False
        note = ("order fitted over H <= 32 (fp32 round-off floor beyond; see poiseuille.py)"
                if prec == "f32" else "all points fitted")
        L += [f"**{prec}**: tau = {r['tau']}, residual-converged, backend {r.get('backend')}; {note}.", "",
              "| Case | H (channel width) | L2 error | observed order |", "|---|---|---|---|"]
        for k, c in r["cases"].items():
            errs = ", ".join(f"{e:.2e}" for e in c["err_l2"])
            L.append(f"| {k} | {c['H']} | {errs} | **{c['order_l2']:.3f}** |")
        L += ["", "BGK at tau = 1/2 + sqrt(3/16) (analytically exact), L2 error for N = 8, 16, 24, 32: " + "; ".join(
            f"{k}: " + ", ".join(f"{e:.1e}" for e in v["err_l2"]) for k, v in r["magic_tau"].items()), "",
              f"![poiseuille {prec}](results/{DIRS['poiseuille']}/poiseuille_{prec}.png)", ""]
    r = _load("cavity")
    if r:
        L += ["## 2. Lid-driven cavity vs Ghia et al. (1982)", "",
              "| Re | N | lattice | collision | τ | steps | max dev /U | RMS dev /U | tolerance | pass |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in r["cases"]:
            L.append(f"| {c['Re']} | {c['N']} | {c['lattice']} | {c['collision']} | {c['tau']:.4f} | {c['steps']} | "
                     f"{c['max_dev']:.4f} | {c['rms_dev']:.4f} | {c['tolerance']} | {'✅' if c['pass'] else '❌'} |")
        capped = [c for c in r["cases"] if c["final_residual"] >= 1e-6]
        if capped:
            L += ["", "Note: " + ", ".join(f"Re={c['Re']} {c['lattice']} {c['collision']}" for c in capped) +
                  f" stopped at the step cap with residual max|du|/U per 1000 steps = "
                  + ", ".join(f"{c['final_residual']:.1e}" for c in capped) + " (target 1e-6)."]
        L += ["", f"![cavity](results/{DIRS['cavity']}/cavity.png)", ""]
    r = _load("cylinder")
    if r:
        L += ["## 3. Cylinder, Re = 100", "",
              "| lattice | collision | D (cells) | St (D) | St (D_eff) | cycles | period scatter | pass (0.16–0.17) |",
              "|---|---|---|---|---|---|---|---|"]
        for c in r["cases"]:
            L.append(f"| {c['lattice']} | {c['collision']} | {c['D']} | **{c['St']:.4f}** | {c['St_Deff']:.4f} | "
                     f"{c['n_cycles']} | {c['period_rel_std']:.1e} | {'✅' if c['pass'] else '❌'} |")
        L += ["", f"![cylinder](results/{DIRS['cylinder']}/cylinder.png)", ""]
    jet_files = sorted((RESULTS / DIRS["jet"]).glob("results_Cs*.json"))
    jet_files = [p for p in jet_files if "INVALID" not in p.name]
    if jet_files:
        rows, notes, pngs, r = [], [], [], None
        for p in jet_files:
            r = json.loads(p.read_text())
            if r.get("note"):
                notes.append(r["note"])
            for c in r["cases"]:
                j = c["momentum_flux_ratio_range_x5_25"]
                fr = c.get("fit_range", [15, 25])
                rows.append(f"| {c['lattice']} | {c['Cs']} | {c.get('inflow_eps', 0.05):.2f} | {c.get('Lx_over_D', 30)} | "
                            f"{fr[0]:g}-{fr[1]:g} | **{c['B']:.2f}** | {c['x0_over_D']:.2f} | **{c['S']:.4f}** | "
                            f"{c.get('potential_core_over_D', float('nan')):.1f} | {j[0]:.2f}–{j[1]:.2f} | "
                            f"{'✅' if c['pass'] else '❌'} |")
        pngs = sorted(p.name for p in (RESULTS / DIRS["jet"]).glob("jet_Cs*.png"))
        L += ["## 4. Turbulent round jet", "",
              f"Acceptance: B in {r['B_range']}, S in {r['S_range']} (fit over 15 <= x/D <= 25). "
              "Experimental potential core: ~4-6 D.", "",
              "| lattice | Cs | inflow disturbance | domain /D | fit window x/D | B | x0/D | S | potential core /D | J/J0 (5-25 D) | pass |",
              "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
        L += ["", "Literature: " + "; ".join(f"{n}: B={b}, S={s}" for n, b, s in r["literature"]), "",
              "Interpretation (DECISIONS.md D-120): pass/fail in the 15-25 D window is governed by when the "
              "nozzle shear layer transitions (core <= ~9 D passes). D3Q27 transitions later than D3Q19 but "
              "its far field on a 45 D domain matches the literature (B = 5.69, S = 0.090).", ""]
        for p in pngs:
            if "initial_run" in p:
                L += ["", "Initial run, Cs = 0.17, both lattices (before the half-width analysis fix: its "
                      "r_1/2 and self-similarity panels are invalid). The centreline panel is valid and shows "
                      "the laminar potential core persisting to ~12 D (D3Q19) and ~25 D (D3Q27): Cs = 0.17 "
                      "over-damps the nozzle shear layer at D = 10 cells.", ""]
            L.append(f"![{p}](results/{DIRS['jet']}/{p})")
        L.append("")
    r = _load("mass_box")
    if r:
        L += ["## 5. Closed box with momentum-source jet", "",
              "| lattice | max mass drift | momentum-injection error | net/forward flux at planes | pass |",
              "|---|---|---|---|---|"]
        for c in r["cases"]:
            fl = ", ".join(f"x={p}: {v['mean_net_over_fwd']:.1e}" for p, v in c["flux"].items())
            L.append(f"| {c['lattice']} | {c['mass_max_rel_drift']:.1e} | {c['momentum_injection_max_rel_err']:.1e} | "
                     f"{fl} | {'✅' if c['pass'] else '❌'} |")
        L += ["", f"![mass box](results/{DIRS['mass_box']}/mass_box.png)", ""]
    g = RESULTS / DIRS["grid"] / "grid_study.md"
    if g.exists():
        body = g.read_text(encoding="utf-8").splitlines()[1:]
        body = [ln.replace("](grid_", f"](results/{DIRS['grid']}/grid_") for ln in body]
        L += ["## 6. Grid-convergence study, demo 75-gallon tank", ""] + body + [""]
    r = _load("powerhead")
    if r:
        L += ["## 7. Pump sources: delivered jet momentum flux", "",
              f"Acceptance: {r['acceptance']['J_ratio'][0]} <= J(x)/J <= {r['acceptance']['J_ratio'][1]} for "
              f"{r['acceptance']['x_over_D'][0]:g} <= x/D_eq <= {r['acceptance']['x_over_D'][1]:g}. "
              f"B, x0: round-jet law in momentum form fitted over {r['fit_window'][0]:g}-{r['fit_window'][1]:g} "
              "D_eq (literature B = 5.0-6.1).", "",
              "| case | turbulence I | J/J (3-8 D) | J/J at 1 / 2 / 4 / 8 D | U_c/U_eq at 1 / 2 / 4 / 8 D | "
              "axis TI at 1 / 2 / 4 / 8 D | B | x0/D | pass |", "|---|---|---|---|---|---|---|---|---|"]
        for c in r["cases"]:
            ja = " / ".join(f"{v:.2f}" for v in c["J_ratio_at"].values())
            uc = " / ".join(f"{v['Uc_over_Ueq']:.2f}" for v in c["centreline"].values())
            ti = " / ".join(f"{v['TI']:.2f}" for v in c["centreline"].values())
            ps = "—" if c["pass"] is None else ("✅" if c["pass"] else "❌")
            L.append(f"| {c['tag']} | {c['turbulence']:g} | {c['J_ratio_window'][0]:.2f}–"
                     f"{c['J_ratio_window'][1]:.2f} | {ja} | {uc} | {ti} | {c['B']:.2f} | "
                     f"{c['x0_over_Deq']:.2f} | {ps} |")
        L += ["", f"![pump sources](results/{DIRS['powerhead']}/powerhead.png)", ""]
    ch = RESULTS / "08_channel" / "results.json"
    if ch.exists():
        r = json.loads(ch.read_text())
        L += ["## 8. Turbulent channel flow: near-wall behaviour and the wall model (DECISIONS D-702)", "",
              "Body-force-driven periodic channel: the wall shear is fixed at rho u_tau^2 by the momentum balance, "
              "and the flow it carries is compared with Dean's (1978) correlation. Pre-declared acceptance for the "
              "wall model: |U_b/u_tau - ref| <= 7 %. **Not met: the wall model diverged and is left off.** The "
              "bounce-back rows quantify the near-wall bias of production runs (ASSUMPTIONS A-310).", "",
              "| case | first node y+ | U_b/u_tau | reference | error |", "|---|---|---|---|---|"]
        for c in r["cases"]:
            ub = "diverged" if c["Ub_over_utau"] is None else f"{c['Ub_over_utau']:.2f}"
            er = "—" if c["Ub_rel_err"] is None else f"{100 * c['Ub_rel_err']:+.0f} %"
            L.append(f"| {c['label']} | {c['y1_plus']:.0f} | {ub} | {c['Ub_ref']:.1f} | {er} |")
        if (RESULTS / "08_channel" / "channel.png").exists():
            L += ["", "![channel](results/08_channel/channel.png)"]
        L.append("")
    light_cases = [("01_inverse_square", "Isotropic point source in air vs inverse square"),
                   ("02_lambertian", "Lambertian panel vs analytic irradiance"),
                   ("03_fresnel", "Air-water and water-glass-air Fresnel / Snell / TIR vs analytic"),
                   ("04_beer_lambert", "Attenuation vs Beer-Lambert (collimated, diffuse, spectral)"),
                   ("05_convergence", "Ray-count convergence (noise slope -1/2)"),
                   ("06_calibration", "Calibration round trip (recover a known fixture)"),
                   ("07_sphere", "Rock geometry: sphere interception, shadow, cosine law, normals"),
                   ("08_energy", "Energy closure")]
    rows = []
    for d_, label in light_cases:
        p = RESULTS / "09_light" / d_ / "results.json"
        if p.exists():
            rows.append(f"| {label} | {'✅' if json.loads(p.read_text()).get('pass') else '❌'} | "
                        f"results/09_light/{d_}/ |")
    if rows:
        L += ["## 9. Lighting / PAR verification (Phase 4)", "",
              "Acceptance criteria are in `reefflow/light/verification.py`; numbers in DECISIONS (Phase 4) and in "
              "each case's results.json and PNG.", "", "| case | pass | results |", "|---|---|---|"] + rows + [""]
    ph = RESULTS / "10_photo" / "results.json"
    if ph.exists():
        r = json.loads(ph.read_text())
        labels = {"C1": "Camera recovery, exact clicks", "C2": "Camera recovery, 2 px click noise",
                  "V1": "Carving, exact cameras + true masks (3 views)", "V2": "Carving, noisy calibration + true masks",
                  "V3": "Full pipeline (noisy clicks + automatic segmentation)", "V4": "Click-noise sweep",
                  "V6": "STL export round trip", "V7": "Cave restoration with the carve tool"}
        L += ["## 10. Photo reconstruction validation (Phase 6)", "",
              "Synthetic photos of the demo rockscape rendered through the refraction model; criteria in "
              "`reefflow/photo/validation.py`. V1 fails its pre-declared IoU >= 0.70 (0.679, volume +47 %): "
              "silhouettes cannot see dips between rock blobs and hidden corner wedges, so the hull overfills "
              "(a fourth, oblique photo gives 0.705). Reconstructed volume is an upper bound; use the edit tools.",
              "", "| case | pass |", "|---|---|"]
        for k, lab in labels.items():
            v = r.get(k, {})
            ok = v.get("pass") if isinstance(v, dict) else None
            L.append(f"| {lab} | {'✅' if ok else ('❌' if ok is False else '—')} |")
        L += ["", "![photo](results/10_photo/fig3_voxel_slices.png)", ""]
    pm = RESULTS / "08_precision" / "precision.md"
    if pm.exists():
        r = json.loads((RESULTS / "08_precision" / "results.json").read_text())
        L += ["## Precision check: fp32 vs fp64 on the demo tank (ASSUMPTIONS A-109)", "",
              "Preview grid, D3Q19. Turbulent runs decorrelate whatever the precision, so the fp64 - fp32 "
              "difference is judged against the seed-to-seed difference of two fp32 runs and the 2σ "
              "statistical noise (pre-declared in precision.py).", "",
              pm.read_text(encoding="utf-8"), "",
              f"Round-off immaterial for every metric: **{'yes' if r['roundoff_immaterial'] else 'NO'}**.", ""]
    text = "\n".join(L)
    (ROOT / "verification" / "VERIFICATION.md").write_text(text, encoding="utf-8")
    return text


def main(argv):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from .. import backend
    if "--report" in argv:
        print(report())
        return
    names = [a for a in argv if not a.startswith("-")] or CASES
    print("backend:", backend.init())
    import importlib
    for n in names:
        mod = importlib.import_module(f"reefflow.verification.{n}")
        print(f"--- {n}", flush=True)
        mod.run()
        if n == "poiseuille":
            print("--- poiseuille (fp64, CPU)", flush=True)
            run_poiseuille_f64()
    print(report())


if __name__ == "__main__":
    main(sys.argv[1:])
