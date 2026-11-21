"""Sculpt brushes (static/js/sculpt.js), run in Node on a box of rock (D-843).

Raise / Lower must work like a terrain brush: a steady surface speed at the centre,
a dome (not a spike), passes that add up, and never the wrong direction. Skipped
when Node.js is not installed (it is not needed to run ReefFlow, only for this check).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import reefflow  # noqa: F401
from reefflow import project as PR
from reefflow.ui import sculpt as SC

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")

TANK = PR.Tank(length=0.6, width=0.3, height=0.35, glass=0.006, bottom_glass=0.006, water_gap=0.025,
               sand_depth=0.03, external=(0.612, 0.312, 0.356))


def _speed():
    src = (REPO / "reefflow/ui/static/js/sculpt.js").read_text(encoding="utf-8")
    return float(re.search(r"const SHIFT_SPEED = ([0-9.]+);", src).group(1))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    import trimesh
    d = tmp_path_factory.mktemp("brush")
    box = trimesh.creation.box(extents=(0.24, 0.15, 0.14))      # front face at y = 0.075
    box.apply_translation([0.3, 0.15, 0.03 + 0.07 - 0.01])        # stands in the sand
    grid = SC.grid_for(TANK, 0.004)
    D = SC.field_from_meshes([(np.asarray(box.vertices), np.asarray(box.faces))], grid)
    (d / "meta.json").write_text(json.dumps({"grid": grid}))
    (d / "field.bin").write_bytes(SC.to_wire(D))
    r = subprocess.run([NODE, str(REPO / "tests/js/sculpt_brush_check.mjs"), str(d), str(REPO)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_raise_lower_speed_and_dome(result):
    v = _speed() * 0.5 * 0.5 * 1000          # strength 50 %, 0.5 s, in mm
    assert result["hold_half_s_centre_mm"] == pytest.approx(v, rel=0.12)
    assert result["lower_half_s_centre_mm"] == pytest.approx(v, rel=0.12)
    # dome: at half the radius the weight is (1 - 0.25)^2 = 0.56 of the centre; outside the brush 0
    ratio = result["hold_half_s_half_radius_mm"] / result["hold_half_s_centre_mm"]
    assert 0.4 < ratio < 0.75
    assert abs(result["hold_half_s_outside_mm"]) < 1.0


def test_passes_add_up_and_undo(result):
    one, two = result["scrub_one_pass_mm"], result["scrub_two_passes_mm"]
    assert one > 1.0 and two > 1.7 * one
    assert result["undo_differing_nodes"] == 0


def test_raise_lower_never_wrong_way_and_glass_gap(result):
    assert result["raise_removed_nodes"] == 0 and result["lower_added_nodes"] == 0
    assert result["raise_rock_nodes_within_10mm_of_front_glass"] == 0
    assert 6.0 < result["raise_front_after_5s_mm"] < 20.0      # stopped near the 10 mm gap (surface between nodes), not at the glass


def test_raise_on_sand_and_tunnel(result):
    assert result["sand_start_is_sand"] and result["sand_mound_mm"] > 20
    assert result["tunnel_through_one_click"] and result["tunnel_open_line"]
