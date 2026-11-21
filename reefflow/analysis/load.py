"""Load a finished run: rebuild its tank model exactly and read its averaged fields."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import project as PR
from .. import tank as TK


@dataclass
class Run:
    path: Path
    info: dict          # run.json
    model: TK.TankModel
    mean_u: np.ndarray  # (nx, ny, nz, 3) m/s
    speed: np.ndarray   # <|u|> m/s
    tke: np.ndarray     # m^2/s^2
    reynolds: np.ndarray | None   # (nx, ny, nz, 6): xx, yy, zz, xy, xz, yz  m^2/s^2

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(2.0 / 3.0 * np.clip(self.tke, 0, None))


def load_run(run_dir, project_file=None, **sim_overrides) -> Run:
    d = Path(run_dir)
    info = json.loads((d / "run.json").read_text(encoding="utf-8"))
    pf = project_file or info["project_file"]
    P = PR.load(pf)
    sim = dict(info.get("sim") or {})
    sim.update(sim_overrides)
    keep = {k: v for k, v in sim.items() if k in PR.SIM_DEFAULTS}
    if keep:
        P = P.with_overrides(**keep)
    model = TK.build(P, info["dx_m"])
    f = np.load(d / "fields.npz")
    if not np.array_equal(f["tags"], model.tags):
        raise ValueError(f"rebuilt tank model does not match the run in {d} (project or settings changed?)")
    return Run(path=d, info=info, model=model, mean_u=f["mean_u"].astype(np.float64),
               speed=f["mean_speed"].astype(np.float64), tke=f["tke"].astype(np.float64),
               reynolds=f["reynolds"].astype(np.float64) if "reynolds" in f.files else None)
