"""Shared helpers for verification cases: output folders, JSON records, plotting style."""
import json
import os
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("REEFFLOW_RESULTS", ROOT / "verification" / "results"))

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


def outdir(name: str) -> Path:
    p = RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(name: str, data: dict, filename: str = "results.json") -> Path:
    from .. import backend
    data = dict(data)
    data.setdefault("backend", backend.current())
    data.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    p = outdir(name) / filename
    p.write_text(json.dumps(_clean(data), indent=2))
    return p


def fit_order(N, err) -> float:
    """Least-squares slope of log(err) vs log(1/N)."""
    N = np.asarray(N, float)
    e = np.asarray(err, float)
    return float(-np.polyfit(np.log(N), np.log(e), 1)[0])


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
