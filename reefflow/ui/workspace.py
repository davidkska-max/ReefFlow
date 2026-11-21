"""User workspace: projects and their runs on disk.

Layout::

    <workspace>/
      projects/<pid>/project.yaml      the project file (core schema + UI 'designer' section)
      projects/<pid>/meshes/           imported STL/OBJ/PLY rock meshes
      projects/<pid>/rockscape.stl     union built by the rockscape designer
      projects/<pid>/runs/<rid>/       one directory per run (core outputs + project snapshot)
      comparisons/<name>/              comparison outputs

A run directory holds a *snapshot* of the project (``project.yaml`` + copies of the
rock meshes under ``inputs/``), so a run stays loadable (``analysis.load.load_run``
rebuilds the tank model from the project file) after the user edits the project.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import re
import shutil
from pathlib import Path

import yaml

from .. import project as PR

PROJECT_FILE = "project.yaml"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_.-")
    return (s or "project")[:60]


def check_id(s: str, what: str = "id", max_len: int = 100) -> str:
    """Reject anything that is not a plain file-name-safe identifier (no path traversal)."""
    if not isinstance(s, str) or len(s) > max_len or not _ID.match(s) or ".." in s:
        raise ValueError(f"invalid {what} {s!r}")
    return s


BLANK = {
    "reefflow_project": 1,
    "name": "New tank",
    "display_units": "US",
    "tank": {"external": {"length": "48 in", "width": "18 in", "height": "21 in"},
             "glass_thickness": "1/2 in", "water_level_below_rim": "1 in", "sand_depth": "2 in"},
    "rockscape": [],
    "overflow": {"corner": "back-right", "width": "6 in", "depth": "6 in", "crest_below_surface": "1/2 in"},
    "pumps": [],
    "simulation": {"lattice": "D3Q19", "collision": "rr", "smagorinsky": 0.10, "preset": "preview",
                   "spinup": "60 s", "average": "120 s", "seed": 1},
    "criteria": {"dead_zone_speed": "2.5 cm/s"},
}


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        (self.root / "projects").mkdir(parents=True, exist_ok=True)
        (self.root / "comparisons").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ projects
    def project_dir(self, pid: str) -> Path:
        d = self.root / "projects" / check_id(pid, "project id")
        if not d.is_dir():
            raise KeyError(f"no project {pid!r}")
        return d

    def project_path(self, pid: str) -> Path:
        return self.project_dir(pid) / PROJECT_FILE

    def list_projects(self) -> list:
        out = []
        for d in sorted((self.root / "projects").iterdir()):
            f = d / PROJECT_FILE
            if not f.is_file():
                continue
            try:
                raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                name = raw.get("name", d.name)
            except Exception:
                name = d.name + " (unreadable)"
            runs = self.list_runs(d.name)
            out.append({"id": d.name, "name": name, "modified": f.stat().st_mtime, "runs": len(runs)})
        return out

    def _new_dir(self, name: str) -> Path:
        base = slugify(name)
        d = self.root / "projects" / base
        i = 2
        while d.exists():
            d = self.root / "projects" / f"{base}_{i}"
            i += 1
        d.mkdir(parents=True)
        (d / "meshes").mkdir()
        (d / "runs").mkdir()
        return d

    def create_project(self, name: str, template: str = "blank", raw: dict | None = None,
                       files: dict | None = None) -> str:
        """New project from 'blank', 'demo' (the 75-gallon reference tank) or a given raw dict.
        ``files``: {relative name: bytes} written next to the project file (imported meshes)."""
        d = self._new_dir(name)
        if raw is not None:
            data = copy.deepcopy(raw)
        elif template == "demo":
            from ..demo import write_demo
            p = write_demo(d)                       # demo_75gal.yaml + rock.stl
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            p.unlink()
            try:                                    # Phase 4 demo lighting (two LED pucks, 8 in)
                from .lighting_adapter import demo_lighting
                lt = demo_lighting()
                if lt and "lighting" not in data:
                    data["lighting"] = lt
            except Exception:
                pass
        elif template == "blank":
            data = copy.deepcopy(BLANK)
        else:
            raise ValueError(f"unknown template {template!r}")
        data["name"] = name or data.get("name", "untitled")
        for rel, content in (files or {}).items():
            target = (d / rel).resolve()
            if d not in target.parents:
                raise ValueError(f"bad file name {rel!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        PR.dump(data, d / PROJECT_FILE)
        return d.name

    def read_raw(self, pid: str) -> dict:
        return yaml.safe_load(self.project_path(pid).read_text(encoding="utf-8")) or {}

    def write_raw(self, pid: str, raw: dict):
        p = self.project_path(pid)
        tmp = p.with_suffix(".yaml.tmp")
        PR.dump(raw, tmp)
        tmp.replace(p)

    def delete_project(self, pid: str):
        shutil.rmtree(self.project_dir(pid))

    def copy_project(self, pid: str, name: str) -> str:
        src = self.project_dir(pid)
        d = self._new_dir(name)
        for item in src.iterdir():
            if item.name == "runs":
                continue
            if item.is_dir():
                shutil.copytree(item, d / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, d / item.name)
        raw = self.read_raw(d.name)
        raw["name"] = name
        self.write_raw(d.name, raw)
        return d.name

    # ------------------------------------------------------------------ runs
    def runs_dir(self, pid: str) -> Path:
        d = self.project_dir(pid) / "runs"
        d.mkdir(exist_ok=True)
        return d

    def run_dir(self, pid: str, rid: str) -> Path:
        d = self.runs_dir(pid) / check_id(rid, "run id")
        if not d.is_dir():
            raise KeyError(f"no run {rid!r} in project {pid!r}")
        return d

    def new_run(self, pid: str, label: str) -> tuple[str, Path]:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        rid = f"{stamp}_{slugify(label)}"[:80]
        d = self.runs_dir(pid) / rid
        i = 2
        while d.exists():
            d = self.runs_dir(pid) / f"{rid}_{i}"
            i += 1
        d.mkdir(parents=True)
        return d.name, d

    def snapshot_project(self, pid: str, run_dir: Path, raw: dict | None = None) -> Path:
        """Copy the project (and the rock meshes it references) into the run directory."""
        src_dir = self.project_dir(pid)
        raw = copy.deepcopy(raw if raw is not None else self.read_raw(pid))
        raw.pop("designer", None)                    # UI state, not a run input
        inp = run_dir / "inputs"
        inp.mkdir(exist_ok=True)
        new_rocks = []
        for i, r in enumerate(raw.get("rockscape") or []):
            f = Path(r["file"])
            f = f if f.is_absolute() else src_dir / f
            if not f.is_file():
                raise FileNotFoundError(f"rock mesh {r['file']!r} not found")
            dst = inp / f"{i:02d}_{f.name}"
            shutil.copy2(f, dst)
            new_rocks.append({**r, "file": f"inputs/{dst.name}"})
        if raw.get("rockscape") is not None:
            raw["rockscape"] = new_rocks
        p = run_dir / PROJECT_FILE
        PR.dump(raw, p)
        return p

    def list_runs(self, pid: str) -> list:
        d = self.project_dir(pid) / "runs"
        if not d.is_dir():
            return []
        out = []
        for r in sorted(d.iterdir(), reverse=True):
            if r.is_dir():
                out.append(run_status(r, pid))
        return out

    def all_runs(self) -> list:
        out = []
        for p in (self.root / "projects").iterdir():
            if (p / PROJECT_FILE).is_file():
                out += self.list_runs(p.name)
        out.sort(key=lambda r: r["created"], reverse=True)
        return out


def run_status(d: Path, pid: str) -> dict:
    """Status of a run directory from its files (and the UI job record, when present)."""
    job = {}
    jf = d / "ui_job.json"
    if jf.is_file():
        try:
            job = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            job = {}
    has_fields = (d / "fields.npz").is_file()
    has_analysis = (d / "analysis.json").is_file()
    status = job.get("status") or ("analysed" if has_analysis else ("finished" if has_fields else "incomplete"))
    return {"project": pid, "id": d.name, "label": job.get("label", d.name), "status": status,
            "created": job.get("created", d.stat().st_mtime), "preset": job.get("preset"),
            "lattice": job.get("lattice"), "dx_m": job.get("dx_m"), "has_fields": has_fields,
            "has_analysis": has_analysis, "has_par": (d / "par.npz").is_file(),
            "has_report": (d / "report.html").is_file(), "error": job.get("error"),
            "job_id": job.get("job_id")}
