"""Equipment catalogue (D-844 to D-846): real pumps, lights and overflows.

Checks that every catalogue entry is complete and builds into the same objects the
simulation uses, that published return-pump curves are used as given, and that the
overflow capacity check warns only when the return flow is higher than the rating."""
import math
import re

import numpy as np
import pytest
import yaml

from reefflow import pumps as PM
from reefflow import project as PR
from reefflow import units as U
from reefflow.demo import PROJECT_YAML
from reefflow.light import fixtures as FX
from reefflow.light import lunits as LU

PUMP_PRODUCTS = PM.load_products()
LIGHT_PRODUCTS = FX.load_products()
OVERFLOWS = PR.load_overflow_catalog()
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _meta_ok(k, v):
    for f in ("brand", "model", "sources", "checked", "status"):
        assert v.get(f), f"{k}: missing {f}"
    assert DATE.match(str(v["checked"])), f"{k}: checked must be a date"
    assert v["status"] in ("current", "discontinued", "unknown"), k
    assert all(str(u).startswith("http") for u in v["sources"]), f"{k}: sources must be URLs"


def test_catalogue_is_not_empty():
    kinds = [v["kind"] for v in PUMP_PRODUCTS.values()]
    assert kinds.count("powerhead") >= 10 and kinds.count("return") >= 10
    assert len(LIGHT_PRODUCTS) >= 10
    assert len(OVERFLOWS["products"]) >= 5 and len(OVERFLOWS["drain_capacity"]) >= 3


@pytest.mark.parametrize("pid", sorted(PUMP_PRODUCTS))
def test_pump_product_builds(pid):
    v = PUMP_PRODUCTS[pid]
    _meta_ok(pid, v)
    assert v.get("rated_flow_basis"), f"{pid}: say where the rated flow comes from"
    assert v.get("outlet_basis"), f"{pid}: say where the outlet size comes from"
    p = PM.build_pump({"name": "x", "archetype": pid, "position": [0.2, 0.2, 0.2], "aim": [1, 0, 0]})
    assert p.kind == v["kind"] and p.rated_flow > 0 and p.outlet.area > 0
    assert 0 < p.delivered_flow <= p.rated_flow * (1 + 1e-9)
    if v["kind"] == "return":
        assert v.get("pump_curve", {}).get("max_head"), f"{pid}: return pumps need a shut-off head"
    # the outlet is a pump outlet, not a typo (5 mm to 0.5 m)
    assert 0.005 <= p.outlet.hydraulic_size <= 0.5, pid


@pytest.mark.parametrize("pid", sorted(k for k, v in PUMP_PRODUCTS.items() if (v.get("pump_curve") or {}).get("points")))
def test_published_curve_is_used_as_given(pid):
    v = PUMP_PRODUCTS[pid]
    pts = PM.parse_curve(v["pump_curve"])
    H0 = U.parse(v["pump_curve"]["max_head"], "head")
    Q0 = U.parse(v["rated_flow"], "flow")
    for h, q in pts:
        if h < H0:
            assert PM.return_pump_flow(Q0, H0, h, pts) == pytest.approx(q, rel=1e-9)
    hs = np.linspace(0, H0 * 0.999, 50)
    qs = [PM.return_pump_flow(Q0, H0, h, pts) for h in hs]
    assert all(b <= a + 1e-12 for a, b in zip(qs, qs[1:])), f"{pid}: flow rises with head"
    # a user-typed rated flow different from the product's Q0 falls back to the generic shape
    base = {"name": "r", "archetype": pid, "position": [0.2, 0.2, 0.2], "aim": [1, 0, 0],
            "static_head": "4 ft", "fittings_allowance": "1 ft"}
    p = PM.build_pump(base)
    assert p.flow_basis.startswith("manufacturer curve")
    p2 = PM.build_pump(dict(base, rated_flow=f"{U.from_si(Q0, 'GPH') * 0.8:.3f} GPH"))
    assert p2.flow_basis.startswith("pump curve Q0 sqrt")


def test_curve_interpolation_and_generic_unchanged():
    pts = [(0.0, 1.0), (1.0, 0.8), (2.0, 0.5)]
    assert PM.return_pump_flow(1.0, 3.0, 1.5, pts) == pytest.approx(0.65)
    assert PM.return_pump_flow(1.0, 3.0, 2.5, pts) == pytest.approx(0.25)     # closes at (H0, 0)
    assert PM.return_pump_flow(1.0, 4.0, 2.0) == pytest.approx(math.sqrt(0.5))  # generic: as before
    with pytest.raises(ValueError):
        PM.parse_curve({"points": [["0 ft", "100 GPH"], ["2 ft", "120 GPH"]]})
    assert PM.return_pump_flow(1.0, 4.0, 1.0, shape="linear") == pytest.approx(0.75)
    with pytest.raises(ValueError):
        PM.return_pump_flow(1.0, 4.0, 1.0, shape="cubic")


def test_dc_returns_without_curve_use_straight_line():
    lin = [k for k, v in PUMP_PRODUCTS.items() if (v.get("pump_curve") or {}).get("shape") == "linear"]
    assert lin
    for k in lin:
        v = PUMP_PRODUCTS[k]
        p = PM.build_pump({"name": "r", "archetype": k, "position": [0.2, 0.2, 0.2], "aim": [1, 0, 0],
                           "static_head": "4 ft", "fittings_allowance": "1 ft"})
        q0, h0 = U.parse(v["rated_flow"], "flow"), U.parse(v["pump_curve"]["max_head"], "head")
        assert p.delivered_flow == pytest.approx(q0 * (1 - U.parse("5 ft", "head") / h0))
        assert p.flow_basis.startswith("straight-line")


@pytest.mark.parametrize("fid", sorted(LIGHT_PRODUCTS))
def test_light_product_builds(fid):
    v = LIGHT_PRODUCTS[fid]
    _meta_ok(fid, v)
    assert v.get("ppf_basis"), f"{fid}: say where the photon output comes from"
    lib = FX.load_library()
    f = FX.build_fixture({"name": "L", "archetype": fid, "position": ["12 in", "9 in"], "height": "8 in"},
                         lib, water_surface=0.5)
    assert 1 < f.ppf < 3000, f"{fid}: photon output {f.ppf:.0f} umol/s"
    assert min(f.footprint) > 0
    spec = f.spectrum(lib["spectra"])
    assert spec.sum() == pytest.approx(1.0)
    if v.get("power") is not None:
        eff = f.ppf / LU.parse(v["power"], "power")
        assert 0.8 < eff < 3.5, f"{fid}: {eff:.2f} umol/J is not a plausible fixture efficacy"


@pytest.mark.parametrize("oid", sorted(OVERFLOWS["products"]))
def test_overflow_product(oid):
    v = OVERFLOWS["products"][oid]
    _meta_ok(oid, v)
    b = v.get("box") or {}
    assert b.get("width") or v.get("rated_flow"), f"{oid}: neither a size nor a capacity: nothing to use"
    if b.get("width"):
        assert 0.02 < U.parse(b["width"], "length") < 2.5
    if b.get("depth"):
        assert 0.01 < U.parse(b["depth"], "length") < 0.5
    if v.get("rated_flow"):
        assert U.parse(v["rated_flow"], "flow") > 0


def _demo_raw():
    return yaml.safe_load(PROJECT_YAML)


def test_overflow_capacity_warning():
    raw = _demo_raw()
    assert any(p.get("archetype") == "return_nozzle" for p in raw["pumps"])
    P = PR.parse(raw)
    q = sum(p.delivered_flow for p in P.pumps if p.kind == "return")
    assert PR.overflow_check(P) == []                                   # no capacity given: no check
    raw["overflow"]["capacity"] = f"{U.from_si(q * 1.2, 'GPH'):.1f} GPH"
    assert PR.overflow_check(PR.parse(raw)) == []
    raw["overflow"]["capacity"] = f"{U.from_si(q * 0.8, 'GPH'):.1f} GPH"
    w = PR.overflow_check(PR.parse(raw))
    assert len(w) == 1 and "rated capacity" in w[0]
    raw["overflow"]["capacity"] = "0 GPH"
    with pytest.raises(ValueError):
        PR.parse(raw)


def test_editor_round_trip_keeps_products():
    from reefflow.ui import editor_model as EM
    raw = _demo_raw()
    ph = next(k for k, v in PUMP_PRODUCTS.items() if v["kind"] == "powerhead")
    raw["pumps"][0]["archetype"] = ph
    oid = next(iter(OVERFLOWS["products"]))
    raw["overflow"]["product"] = oid
    raw["overflow"]["capacity"] = "1200 GPH"
    ed = EM.to_editor(raw)
    assert ed["pumps"][0]["archetype"] == ph
    assert ed["overflow"]["product"] == oid
    assert ed["overflow"]["capacity"] == pytest.approx(U.parse("1200 GPH", "flow"))
    back = EM.from_editor(ed, raw)
    assert back["pumps"][0]["archetype"] == ph and back["overflow"]["product"] == oid
    assert U.parse(back["overflow"]["capacity"], "flow") == pytest.approx(U.parse("1200 GPH", "flow"))
    PR.parse(back)


def test_library_api_lists_products():
    from reefflow.ui.server import library
    lib = library()
    prods = [v for v in lib["pumps"].values() if v.get("product")]
    assert len(prods) == len(PUMP_PRODUCTS)
    assert all(v["brand"] and v["model"] and v["sources"] for v in prods)
    assert "narrow_jet_propeller" in lib["pumps"] and not lib["pumps"]["narrow_jet_propeller"].get("product")
    lp = [v for v in lib["fixtures"].values() if v.get("product")]
    assert len(lp) == len(LIGHT_PRODUCTS) and all(v["ppf_umol_s"] for v in lp)
    assert len(lib["overflows"]) == len(OVERFLOWS["products"])


def test_return_exit_head_warning():
    raw = _demo_raw()
    assert PR.return_check(PR.parse(raw)) == []                  # demo: 2.1 m/s, 0.74 ft < 1 ft allowance
    big = next(k for k, v in PUMP_PRODUCTS.items() if v["kind"] == "return" and U.parse(v["rated_flow"], "flow") > U.parse("1500 GPH", "flow"))
    raw["pumps"][0]["archetype"] = big
    raw["pumps"][0].pop("rated_flow")
    w = PR.return_check(PR.parse(raw))
    assert len(w) == 1 and "exit head" in w[0]
    raw["pumps"][0]["delivered_flow"] = "500 GPH"                # a measured flow: no check
    assert PR.return_check(PR.parse(raw)) == []
