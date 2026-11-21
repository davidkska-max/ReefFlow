"""Jet diagnostics: why does D3Q27 fail the 15-25 D fit window? (DECISIONS D-120)

These runs do NOT change the verification case or its acceptance criteria. They
separate transition triggering from far-field accuracy:

E1  'turbulent nozzle exit': inflow disturbance 15 % instead of 5 % (both lattices).
    A propeller powerhead's outflow is swirling and highly turbulent, so this is
    closer to the tank application than a quiet nozzle.
E2  longer domain: 45 D (D3Q27, original 5 % inflow), fit window moved to 25-40 D,
    to see whether D3Q27 reaches the self-similar far field once it has room to.

    .venv\\Scripts\\python verification\\jet_diagnostics.py
"""
import subprocess

from reefflow import backend
from reefflow.verification import jet

RUNS = [
    dict(lattices=("D3Q19", "D3Q27"), cs=0.10, eps=0.15, tag="_Cs0.10_eps0.15"),
    dict(lattices=("D3Q27",), cs=0.10, eps=0.05, Lx=45, fit=(25.0, 40.0), average_steps=170_000,
         tag="_Cs0.10_Lx45_D3Q27"),
]


def vram():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "n/a"


if __name__ == "__main__":
    print("backend:", backend.init(), flush=True)
    for kw in RUNS:
        kw = dict(kw)
        tag = kw["tag"]
        print(f"=== {tag}: {kw}; GPU before run: {vram()}", flush=True)
        out = jet.run(**kw)
        for c in out["cases"]:
            print(f"RESULT {tag} {c['lattice']}: B={c['B']:.3f} x0/D={c['x0_over_D']:.2f} S={c['S']:.4f} "
                  f"core/D={c['potential_core_over_D']:.1f} J/J0={c['momentum_flux_ratio_range_x5_25']} "
                  f"fit={c['fit_range']} pass={c['pass']} t={c['runtime_s']:.0f}s", flush=True)
    print("DIAG DONE", flush=True)
