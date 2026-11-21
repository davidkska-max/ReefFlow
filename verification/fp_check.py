r"""fp32 vs fp64 check on the demo tank at Preview (ASSUMPTIONS A-109).

Runs: fp32 GPU seed 1 (runs/demo_preview_D3Q19, the standard Preview demo run),
fp32 GPU seed 2, fp64 CPU seed 1. Turbulent runs decorrelate within seconds, so
the comparison is statistical: the fp32-fp64 difference is judged against the
seed-to-seed difference of two fp32 runs.

    set REEFFLOW_ARCH=cpu & set REEFFLOW_PRECISION=f64 & python verification/fp_check.py 1 fp64_seed1
    python verification/fp_check.py 2 fp32_seed2
"""
import sys

from reefflow import backend

if __name__ == "__main__":
    seed, tag = int(sys.argv[1]), sys.argv[2]
    print("backend:", backend.init(), backend.precision(), flush=True)
    from reefflow import project, tank, run
    P = project.load("examples/demo_75gal/demo_75gal.yaml").with_overrides(seed=seed)
    m = tank.build(P, 0.01)
    r = run.run(m, f"runs/fpcheck_{tag}", progress=lambda s: print(s, flush=True))
    print("DONE", tag, r["metrics"]["mean_speed"], r["metrics"]["dead_fraction"], flush=True)
