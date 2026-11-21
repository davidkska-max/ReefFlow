r"""Phase-2 verification runs (GPU): powerhead momentum-flux check, then the grid study.

    .venv\Scripts\python verification\phase2_runs.py
"""
import subprocess
import sys

from reefflow import backend


def vram():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "n/a"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("backend:", backend.init(), "GPU:", vram(), flush=True)
    if "--skip-powerhead" not in sys.argv:
        from reefflow.verification import powerhead
        for c in powerhead.run()["cases"]:
            print("POWERHEAD", c["tag"], "J/J0", c["J_ratio_window"], "pass", c["pass"], c["centreline"], flush=True)
    print("GPU before grid study:", vram(), flush=True)
    from reefflow.gridstudy import grid_study
    grid_study("examples/demo_75gal/demo_75gal.yaml", progress=lambda m: print(m, flush=True))
    print("PHASE2 RUNS DONE", flush=True)
