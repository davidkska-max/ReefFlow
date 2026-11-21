"""Jet Smagorinsky-constant sensitivity study (DECISIONS D-118).

D3Q19 at Cs = 0.10 was completed earlier (results_Cs0.10.json); this script runs
D3Q27 at Cs = 0.10, then Cs = 0.12 and 0.17 on both lattices. Each run writes its
own results_*.json / jet_*.png / mean_*.npz into verification/results/04_jet.

    .venv\\Scripts\\python verification\\jet_cs_study.py
"""
import subprocess

from reefflow import backend
from reefflow.verification import jet

RUNS = [(("D3Q27",), 0.10, "_Cs0.10_D3Q27"),
        (("D3Q19", "D3Q27"), 0.12, "_Cs0.12"),
        (("D3Q19", "D3Q27"), 0.17, "_Cs0.17")]


def vram():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "n/a"


if __name__ == "__main__":
    print("backend:", backend.init(), flush=True)
    for lats, cs, tag in RUNS:
        print(f"=== {tag}: lattices {lats}, Cs={cs}; GPU before run: {vram()}", flush=True)
        out = jet.run(lattices=lats, cs=cs, tag=tag)
        for c in out["cases"]:
            print(f"RESULT {c['lattice']} Cs={cs}: B={c['B']:.3f} x0/D={c['x0_over_D']:.2f} S={c['S']:.4f} "
                  f"core/D={c['potential_core_over_D']:.1f} J/J0={c['momentum_flux_ratio_range_x5_25']} "
                  f"pass={c['pass']} t={c['runtime_s']:.0f}s", flush=True)
    print("STUDY DONE", flush=True)
