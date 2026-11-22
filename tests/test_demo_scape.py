"""UI demo aquascape and the broken-rock options of the rock element (D-850)."""
import numpy as np
import pytest
import yaml

from reefflow import project as PR
from reefflow.demo import DEMO_SCAPE_ELEMENTS, PROJECT_YAML, demo_scape
from reefflow.ui import rockgen as RG


def _grid(lo, hi, h=0.004):
    ax = [np.arange(a, b, h) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1)


def test_rock_defaults_unchanged():
    el = {"type": "rock", "params": {"size": [0.2, 0.15, 0.12], "seed": 3}}
    el0 = {"type": "rock", "params": {"size": [0.2, 0.15, 0.12], "seed": 3, "facets": 0, "chunk": 0.0}}
    lo, hi = RG.local_bounds(el)
    P = _grid(lo, hi)
    assert np.array_equal(RG.local_field(el, P), RG.local_field(el0, P))


def test_facets_cut_and_chunk_stay_in_bounds():
    base = {"size": [0.2, 0.15, 0.12], "seed": 3, "roughness": 0.18}
    plain = {"type": "rock", "params": base}
    cut = {"type": "rock", "params": {**base, "facets": 7}}
    lumpy = {"type": "rock", "params": {**base, "facets": 7, "chunk": 0.35}}
    lo, hi = RG.local_bounds(lumpy)
    P = _grid(lo - 0.02, hi + 0.02)
    v_plain = (RG.local_field(plain, P) < 0).sum()
    v_cut = (RG.local_field(cut, P) < 0).sum()
    assert 0.5 * v_plain < v_cut < 0.97 * v_plain              # facets only remove rock
    inside = RG.local_field(lumpy, P) < 0
    pts = P[inside]
    assert np.all(pts >= lo - 1e-9) and np.all(pts <= hi + 1e-9)  # the bounds still contain the rock


def test_demo_scape_builds_clean(tmp_path):
    raw = yaml.safe_load(PROJECT_YAML)
    tank = PR.parse({**raw, "rockscape": [], "lighting": None}).tank
    sc = demo_scape(tmp_path, tank)
    assert (tmp_path / "rockscape.stl").is_file()
    assert 0.008 < sc["volume_m3"] < 0.013                    # ~10.2 L of rock (~36 lb dry at 1.6 kg/L)
    assert len(sc["designer"]["elements"]) == len(DEMO_SCAPE_ELEMENTS)
    res = RG.union(sc["designer"]["elements"], tank, tmp_path, 0.004)
    assert res["issues"] == [] and res["watertight"]
    lo, hi = np.asarray(res["bounds_m"][0]), np.asarray(res["bounds_m"][1])
    assert hi[2] < tank.water_surface - 0.04                   # peak stays under the water
    assert lo[0] > 0.05 and hi[0] < tank.length - 0.05 and hi[1] < tank.width - 0.06   # room for flow at the glass


def test_ui_demo_template_is_designer_scape(tmp_path):
    from reefflow.ui.workspace import Workspace
    ws = Workspace(tmp_path)
    pid = ws.create_project("demo", "demo")
    raw = ws.read_raw(pid)
    assert raw["rockscape"][0]["file"] == "rockscape.stl"
    assert len(raw["designer"]["elements"]) == len(DEMO_SCAPE_ELEMENTS)
    assert not (ws.project_dir(pid) / "rock.stl").exists()
    PR.parse(raw, ws.project_dir(pid) / "project.yaml")
