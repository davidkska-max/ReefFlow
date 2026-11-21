"""Overflow placement along any glass wall (D-801..D-804) and multi-nozzle return pumps (D-805..D-808)."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from reefflow import project as PR
from reefflow import pumps as PM
from reefflow import sources as S
from reefflow import units as U

IN = 0.0254


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from reefflow.demo import write_demo
    return write_demo(tmp_path_factory.mktemp("demo_layout"))


@pytest.fixture()
def raw(demo):
    return yaml.safe_load(Path(demo).read_text())


# ---------------------------------------------------------------- overflow
def test_corner_form_footprint_unchanged(raw, demo):
    """The Phase-2 corner form gives exactly the Phase-2 footprint (formula re-stated here)."""
    for corner in PR.CORNERS:
        r = copy.deepcopy(raw)
        r["overflow"] = {"corner": corner, "width": "6 in", "depth": "5 in", "crest_below_surface": "1/2 in"}
        P = PR.parse(r, demo)
        t, w, d = P.tank, 6 * IN, 5 * IN
        x0, x1 = (t.length - w, t.length) if "right" in corner else (0.0, w)
        y0, y1 = (t.width - d, t.width) if "back" in corner else (0.0, d)
        assert P.overflow.box(t) == (x0, x1, y0, y1)


def _ov(raw, demo, **o):
    r = copy.deepcopy(raw)
    r["overflow"] = {"crest_below_surface": "1/2 in", **o}
    return PR.parse(r, demo)


def test_wall_form_footprints(raw, demo):
    P = _ov(raw, demo, wall="back", center="20 in", width="6 in", depth="4 in")
    t = P.tank
    np.testing.assert_allclose(P.overflow.box(t), (17 * IN, 23 * IN, t.width - 4 * IN, t.width), atol=1e-12)
    P = _ov(raw, demo, wall="front", center="10 in", width="6 in", depth="4 in")
    np.testing.assert_allclose(P.overflow.box(t), (7 * IN, 13 * IN, 0, 4 * IN), atol=1e-12)
    P = _ov(raw, demo, wall="left", center="8 in", width="6 in", depth="4 in")
    np.testing.assert_allclose(P.overflow.box(t), (0, 4 * IN, 5 * IN, 11 * IN), atol=1e-12)
    P = _ov(raw, demo, wall="right", center="8 in", width="6 in", depth="4 in")
    np.testing.assert_allclose(P.overflow.box(t), (t.length - 4 * IN, t.length, 5 * IN, 11 * IN), atol=1e-12)
    P = _ov(raw, demo, wall="back", width="6 in", depth="4 in")          # no centre: mid-wall
    b = P.overflow.box(t)
    assert (b[0] + b[1]) / 2 == pytest.approx(t.length / 2)


def test_wall_form_validation(raw, demo):
    with pytest.raises(ValueError, match="longer than the left glass"):
        _ov(raw, demo, wall="left", center="8 in", width="20 in", depth="4 in")      # 17 in inside width
    with pytest.raises(ValueError, match="outside the tank"):
        _ov(raw, demo, wall="back", center="2 in", width="6 in", depth="4 in")       # needs >= 3 in
    with pytest.raises(ValueError, match="outside the tank"):
        _ov(raw, demo, wall="front", center="45 in", width="6 in", depth="4 in")     # needs <= 44 in
    with pytest.raises(ValueError, match="opposite glass"):
        _ov(raw, demo, wall="back", center="20 in", width="6 in", depth="18 in")
    with pytest.raises(ValueError, match="wall"):
        _ov(raw, demo, wall="top", center="20 in", width="6 in", depth="4 in")
    with pytest.raises(ValueError, match="corner"):
        _ov(raw, demo, corner="middle", width="6 in", depth="4 in")
    # flush to a corner is allowed (centre exactly w/2)
    P = _ov(raw, demo, wall="right", center="3 in", width="6 in", depth="4 in")
    assert P.overflow.box(P.tank)[2] == pytest.approx(0.0, abs=1e-12)


def test_coast_to_coast_and_mid_wall_model(raw, demo):
    from reefflow import tank
    P = _ov(raw, demo, wall="back", width="47 in", depth="3 in")
    t = P.tank
    assert P.overflow.box(t) == pytest.approx((0.0, t.length, t.width - 3 * IN, t.width))
    assert "coast to coast" in P.overflow.describe(t)
    m = tank.build(P, 0.02)
    sink = np.argwhere(m.tags == tank.TAGS["sink"])
    nx = m.shape[0]
    assert sink[:, 0].min() == 1 and sink[:, 0].max() == nx - 2            # spans the whole length
    # mid-back-wall box: sink nodes centred on the requested centre
    P = _ov(raw, demo, wall="back", center="24 in", width="6 in", depth="6 in")
    m = tank.build(P, 0.01)
    sink = np.argwhere(m.tags == tank.TAGS["sink"])
    xs = m.node_coords(0)[sink[:, 0]]
    assert xs.mean() == pytest.approx(24 * IN, abs=0.006)
    assert m.spec.table[:, S.S_MASS].sum() == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------- nozzles
def _ret(raw, nozzles, **kw):
    r = copy.deepcopy(raw)
    ret = [p for p in r["pumps"] if p["archetype"] == "return_nozzle"][0]
    ret.pop("position"); ret.pop("aim")
    ret["nozzles"] = nozzles
    ret.update(kw)
    return r


N1 = {"position": ["40 in", "14 in", "17 in"], "aim": [-1, -0.5, -0.25]}
N2 = {"position": ["30 in", "14 in", "17 in"], "aim": [-1, -0.8, -0.2]}


def test_nozzle_split_by_area(raw, demo):
    P = PR.parse(_ret(raw, [N1, dict(N2, diameter="1/2 in")]), demo)
    p = [x for x in P.pumps if x.kind == "return"][0]
    A1, A2 = p.nozzles[0].outlet.area, math.pi * (0.5 * IN) ** 2 / 4
    assert p.nozzles[1].outlet.area == pytest.approx(A2)
    assert p.nozzles[0].share == pytest.approx(A1 / (A1 + A2))         # 3/4 in vs 1/2 in -> 0.6923
    assert p.nozzles[0].share == pytest.approx(19 ** 2 / (19 ** 2 + 12.7 ** 2))   # 19 mm vs 1/2 in: 0.691
    assert sum(n.share for n in p.nozzles) == pytest.approx(1.0, abs=1e-15)
    # Q_i ~ A_i -> equal exit velocities, both = Q / (A1 + A2)
    assert p.nozzle_velocity_of(0) == pytest.approx(p.nozzle_velocity_of(1))
    assert p.momentum_flux == pytest.approx(sum(1023 * p.nozzle_flow(k) ** 2 / p.nozzles[k].outlet.area for k in (0, 1)))
    np.testing.assert_allclose(p.position, P.pumps[0].nozzles[0].position)


def test_nozzle_shares(raw, demo):
    P = PR.parse(_ret(raw, [dict(N1, share=0.6), N2]), demo)
    p = [x for x in P.pumps if x.kind == "return"][0]
    assert [n.share for n in p.nozzles] == pytest.approx([0.6, 0.4])
    P = PR.parse(_ret(raw, [dict(N1, share=0.705), dict(N2, share=0.3)]), demo)    # 1.005: normalised
    p = [x for x in P.pumps if x.kind == "return"][0]
    assert [n.share for n in p.nozzles] == pytest.approx([0.705 / 1.005, 0.3 / 1.005])
    with pytest.raises(ValueError, match="sum to"):
        PR.parse(_ret(raw, [dict(N1, share=0.5), dict(N2, share=0.4)]), demo)
    with pytest.raises(ValueError, match="leave no flow"):
        PR.parse(_ret(raw, [dict(N1, share=1.0), N2]), demo)
    with pytest.raises(ValueError, match="not inside"):
        PR.parse(_ret(raw, [N1, dict(N2, position=["30 in", "14 in", "25 in"])]), demo)
    r = copy.deepcopy(raw)
    ph = [p for p in r["pumps"] if p["archetype"] != "return_nozzle"][0]
    ph["nozzles"] = [N1, N2]
    with pytest.raises(ValueError, match="only return pumps"):
        PR.parse(r, demo)


def test_exit_velocity_head():
    """19 mm (3/4 in) nozzle at 700 GPH: U = 2.60 m/s, U^2/2g = 1.13 ft; split over two: 0.28 ft each."""
    lib = PM.load_library()
    d = {"name": "r", "archetype": "return_nozzle", "rated_flow": "800 GPH", "delivered_flow": "700 GPH",
         "position": [0.5, 0.2, 0.3], "aim": [1, 0, 0]}
    p = PM.build_pump(d, lib)
    assert p.outlet.diameter == pytest.approx(0.019)
    s = p.summary()
    assert s["nozzles"][0]["exit_velocity_head_ft"] == pytest.approx(1.127, abs=0.005)
    d2 = dict(d)
    d2.pop("position"); d2.pop("aim")
    d2["nozzles"] = [{"position": [0.5, 0.2, 0.3], "aim": [1, 0, 0]}, {"position": [0.6, 0.2, 0.3], "aim": [1, 0, 0]}]
    s2 = PM.build_pump(d2, lib).summary()
    assert [n["exit_velocity_head_ft"] for n in s2["nozzles"]] == pytest.approx([0.28, 0.28], abs=0.005)
    assert s2["delivered_flow_GPH"] == pytest.approx(700)                     # total flow unchanged


def test_multi_nozzle_source_table(raw, demo):
    from reefflow import tank
    P = PR.parse(_ret(raw, [N1, dict(N2, diameter="1/2 in")]), demo)
    m = tank.build(P, 0.01)
    lu, tab = m.lu, m.spec.table
    ret = [p for p in P.pumps if p.kind == "return"][0]
    outs = [pr for pr in m.pumps if pr.pump is ret]
    assert [pr.nozzle for pr in outs] == [0, 1] and len({pr.index for pr in outs}) == 2
    Qsum = 0.0
    for pr in outs:
        rows = tab[tab[:, S.S_PUMP] == pr.index]
        Q = rows[:, S.S_MASS].sum() * lu.C_Q
        assert Q == pytest.approx(ret.nozzle_flow(pr.nozzle), rel=1e-12)
        mom = (rows[:, S.S_MASS:S.S_MASS + 1] * rows[:, S.S_U0:S.S_U0 + 3]).sum(axis=0) * lu.C_J
        J = 1023 * ret.nozzle_flow(pr.nozzle) ** 2 / ret.nozzles[pr.nozzle].outlet.area
        np.testing.assert_allclose(mom, J * ret.nozzles[pr.nozzle].aim, rtol=1e-12)
        Qsum += Q
    assert Qsum == pytest.approx(ret.delivered_flow, rel=1e-12)
    assert tab[:, S.S_MASS].sum() == pytest.approx(0.0, abs=1e-15)          # overflow sink balances
    st = m.pump_state(1.0, 0.01)
    assert st.shape == (len(m.pumps) + 1, S.P_NCOL) and len(m.pumps) == len(P.pumps) + 1
    assert st[m.overflow_index, S.P_SM] == pytest.approx(1.0)
    info = [d for d in m.info["pumps"] if d["name"] == ret.name][0]
    assert len(info["nozzles"]) == 2 and info["source_nodes"] == sum(pr.n_nodes for pr in outs)
    assert len(m.info["pumps"]) == len(P.pumps)


def test_single_nozzle_model_unchanged(demo):
    """The single-outlet form builds the same source table as a one-entry nozzle list."""
    from reefflow import tank
    raw = yaml.safe_load(Path(demo).read_text())
    m0 = tank.build(PR.parse(raw, demo), 0.02)
    ret = [p for p in raw["pumps"] if p["archetype"] == "return_nozzle"][0]
    r1 = _ret(raw, [{"position": ret["position"], "aim": ret["aim"]}])
    m1 = tank.build(PR.parse(r1, demo), 0.02)
    np.testing.assert_array_equal(m0.spec.cells, m1.spec.cells)
    np.testing.assert_array_equal(m0.spec.table, m1.spec.table)
    np.testing.assert_array_equal(m0.domain.flags, m1.domain.flags)


# ---------------------------------------------------------------- editor round trips
def test_editor_roundtrip_demo_unchanged(raw, demo):
    from reefflow.ui import editor_model as EM
    ed = EM.to_editor(raw)
    assert ed["overflow"]["wall"] == "back"
    out = EM.from_editor(ed, raw)
    assert all("nozzles" not in p and "position" in p for p in out["pumps"])    # single-outlet form kept
    P0, P1 = PR.parse(raw, demo), PR.parse(out, demo)
    np.testing.assert_allclose(P1.overflow.box(P1.tank), P0.overflow.box(P0.tank), atol=1e-6)
    for a, b in zip(P0.pumps, P1.pumps):
        np.testing.assert_allclose(a.position, b.position, atol=1e-6)
        assert a.delivered_flow == pytest.approx(b.delivered_flow, rel=1e-5)


def test_editor_roundtrip_new_forms(raw, demo):
    from reefflow.ui import editor_model as EM
    r = _ret(raw, [N1, dict(N2, diameter="1/2 in", share=0.3)])
    r["overflow"] = {"wall": "left", "center": "9 in", "width": "5 in", "depth": "4 in", "crest_below_surface": "1 in"}
    ed = EM.to_editor(r)
    rp = [p for p in ed["pumps"] if p["kind"] == "return"][0]
    assert len(rp["nozzles"]) == 2 and rp["nozzles"][1]["share"] == 0.3
    assert rp["nozzles"][1]["diameter"] == pytest.approx(0.5 * IN)
    ov = dict(ed["overflow"])
    assert ov.pop("product") is None and ov.pop("capacity") is None        # catalogue fields (D-844/846), unset here
    assert ov == pytest.approx({"wall": "left", "center": 9 * IN, "width": 5 * IN, "depth": 4 * IN,
                                "crest_below_surface": 1 * IN})
    out = EM.from_editor(ed, r)
    assert out["overflow"]["wall"] == "left" and "corner" not in out["overflow"]
    retd = [p for p in out["pumps"] if p["archetype"] == "return_nozzle"][0]
    assert "position" not in retd and len(retd["nozzles"]) == 2
    P0, P1 = PR.parse(r, demo), PR.parse(out, demo)
    np.testing.assert_allclose(P1.overflow.box(P1.tank), P0.overflow.box(P0.tank), atol=1e-6)
    q0 = [p for p in P0.pumps if p.kind == "return"][0]
    q1 = [p for p in P1.pumps if p.kind == "return"][0]
    assert [n.share for n in q1.nozzles] == pytest.approx([n.share for n in q0.nozzles])
    for a, b in zip(q0.nozzles, q1.nozzles):
        np.testing.assert_allclose(a.position, b.position, atol=1e-6)
    # dropping back to one nozzle writes the single-outlet form again
    rp2 = [p for p in ed["pumps"] if p["kind"] == "return"][0]
    rp2["nozzles"] = rp2["nozzles"][:1]
    out2 = EM.from_editor(ed, r)
    retd2 = [p for p in out2["pumps"] if p["archetype"] == "return_nozzle"][0]
    assert "nozzles" not in retd2 and "position" in retd2
