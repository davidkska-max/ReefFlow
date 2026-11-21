"""fp32 vs fp64 on the demo tank at Preview (ASSUMPTIONS A-109 re-check).

Runs (verification/fp_check.py):
    fp32 GPU seed 1  = grid-study run D3Q19 Preview (identical settings)
    fp32 GPU seed 2  = runs/fpcheck_fp32_seed2
    fp64 CPU seed 1  = runs/fpcheck_fp64_seed1
The flow is turbulent, so runs decorrelate within seconds whatever the precision;
the comparison is statistical. Pre-declared reading: round-off is immaterial if
every |fp64 - fp32(seed 1)| is no larger than ~2x the seed-to-seed difference
|fp32(seed 2) - fp32(seed 1)| or 2 sigma statistical, whichever is larger.
"""
import json
from pathlib import Path

import numpy as np

from .. import metrics as MET
from .. import project as PR
from .. import tank as TK
from .common import RESULTS, ROOT, save_json

RUNS = {"fp32 seed 1 (GPU)": RESULTS / "06_grid" / "D3Q19_preview",
        "fp32 seed 2 (GPU)": ROOT / "runs" / "fpcheck_fp32_seed2",
        "fp64 seed 1 (CPU)": ROOT / "runs" / "fpcheck_fp64_seed1"}


def run():
    vals, unc, meta = {}, {}, {}
    model = None
    for name, d in RUNS.items():
        r = json.loads((d / "run.json").read_text())
        if model is None:
            P = PR.load(r["project_file"]).with_overrides(preset="preview", dx=None)
            model = TK.build(P, P.dx())
        f = np.load(d / "fields.npz")
        m = MET.compute(model, f["mean_u"].astype(float), f["mean_speed"].astype(float), f["tke"].astype(float),
                        r["metrics"]["dead_speed_threshold"])
        u = {}
        for k, v in m.items():
            if k in r["metrics_stat_uncertainty"] and r["metrics"].get(k):
                u[k] = r["metrics_stat_uncertainty"][k] / abs(r["metrics"][k]) * abs(v)
        vals[name], unc[name] = m, u
        meta[name] = {"backend": r["backend"], "precision": r["precision"], "wall_time_s": r["wall_time_s"],
                      "mass_rel_drift": r["mass_rel_drift"]}
    a, b, c = (vals[k] for k in RUNS)
    ua, ub, uc = (unc[k] for k in RUNS)
    rows, ok_all = [], True
    for k in MET.KEY_METRICS:
        seed = abs(b[k] - a[k])
        prec = abs(c[k] - a[k])
        noise = 2 * np.hypot(ua[k], uc[k])
        ok = prec <= max(2 * seed, noise)
        ok_all &= bool(ok)
        rows.append({"metric": k, "fp32_s1": a[k], "fp32_s2": b[k], "fp64_s1": c[k], "seed_diff": seed,
                     "precision_diff": prec, "noise_2sigma": noise, "immaterial": bool(ok)})
    out = {"runs": meta, "rows": rows, "roundoff_immaterial": ok_all}
    save_json("08_precision", out)
    L = ["| metric | fp32 seed 1 | fp32 seed 2 | fp64 seed 1 | |seed diff| | |fp64 − fp32| | 2σ noise | immaterial |",
         "|---|---|---|---|---|---|---|---|"]
    for rw in rows:
        lab, unit, sc = MET.KEY_METRICS[rw["metric"]]
        L.append(f"| {lab} [{unit}] | {rw['fp32_s1'] * sc:.2f} | {rw['fp32_s2'] * sc:.2f} | {rw['fp64_s1'] * sc:.2f} | "
                 f"{rw['seed_diff'] * sc:.2f} | {rw['precision_diff'] * sc:.2f} | {rw['noise_2sigma'] * sc:.2f} | "
                 f"{'yes' if rw['immaterial'] else 'NO'} |")
    (RESULTS / "08_precision" / "precision.md").write_text("\n".join(L), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    o = run()
    print((RESULTS / "08_precision" / "precision.md").read_text(encoding="utf-8"))
    print("round-off immaterial:", o["roundoff_immaterial"])
