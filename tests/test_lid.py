"""Lid over the tank (D-848): presets, analytic transmission, tracer agreement, PAR effect."""
import math

import numpy as np
import pytest

from reefflow.light import config as CF


def _lid(**kw):
    from reefflow.light.verification import _box_tank
    tank = _box_tank(1.0, 1.0, 0.0, 0.5)
    from reefflow.light.fixtures import build_fixture, load_library
    fx = [build_fixture({"name": "b", "archetype": "collimated_beam", "position": [0.5, 0.5], "height": 0.2},
                        load_library(), tank.water_surface)]
    return CF.parse_lid(kw, tank, fx)


def test_sheet_presets_match_published_transmittance():
    # typical visible transmittance of 6 mm sheet: float glass ~88 %, low-iron ~91 %, cast acrylic ~92 %
    assert _lid(preset="glass_float_6mm").transmission() == pytest.approx(0.88, abs=0.01)
    assert _lid(preset="glass_lowiron_6mm").transmission() == pytest.approx(0.91, abs=0.01)
    assert _lid(preset="acrylic_6mm").transmission() == pytest.approx(0.92, abs=0.01)
    g = _lid(preset="glass_float_6mm")
    assert g.transmission(math.radians(60)) < g.transmission(math.radians(30)) < g.transmission()


def test_screen_open_fraction_and_angle():
    s = _lid(preset="screen_black_1_8")
    p = (0.125 * 25.4 + 0.6)
    assert s.open_fraction == pytest.approx((0.125 * 25.4 / p) ** 2)
    assert s.transmission() == pytest.approx(s.open_fraction)
    # a slanted ray: the round strands across its lean cast a shadow d / cos(theta)
    t = s.transmission(math.radians(30), 0.0)
    assert t == pytest.approx((s.pitch - s.strand / math.cos(math.radians(30))) / s.pitch * s.opening / s.pitch)
    e = _lid(preset="eggcrate_white")                     # deep cells shade slanted light much more
    assert e.transmission(math.radians(30)) / e.transmission() < t / s.transmission()
    assert _lid(preset="screen_aluminum_window").open_fraction == pytest.approx(0.64, abs=0.02)


def test_lid_validation():
    with pytest.raises(ValueError):
        _lid(preset="nope")
    with pytest.raises(ValueError):
        _lid(type="glass", height="1 m")                    # above the light
    with pytest.raises(ValueError):
        _lid(type="screen", opening="0 mm", strand="1 mm")
    assert _lid(type="open").is_open and _lid().is_open


def _beam_lost(lid, tilt=0.0, n=1_000_000):
    from reefflow.light.tracer import build_scene, trace
    from reefflow.light.verification import _box_tank, _cfg
    tank = _box_tank(2.0, 2.0, 0.0, 0.5)
    cfg = _cfg([{"name": "b", "archetype": "collimated_beam", "position": [1.0, 1.0], "height": 0.2, "tilt_x": tilt}],
               tank, bounces=False, albedo={"sand": 0, "rock": 0}, water={"kd": 0}, lid=lid, map_depths=[0.25])
    tal = trace(build_scene(cfg, tank), n, seed=4)
    c = tal.counters
    assert abs(tal.energy_balance()["closure_residual"]) < 1e-6
    return c["absorbed_lid"] / c["emitted"], cfg.lid


@pytest.mark.parametrize("preset", ["glass_float_6mm", "acrylic_3mm", "screen_black_1_8", "screen_aluminum_window",
                                    "eggcrate_white"])
@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_tracer_matches_analytic_transmission(backend_name, preset, tilt):
    lost, lid = _beam_lost({"preset": preset, "height": "50 mm"}, tilt)
    T = lid.transmission(math.radians(tilt), math.pi / 2)      # tilt_x leans the beam toward the back glass (y)
    passed = T + (1 - T) * lid.scatter if lid.type == "screen" else T
    assert 1 - lost == pytest.approx(passed, abs=0.002), (preset, tilt, 1 - lost, passed)


def test_open_lid_traces_identically(backend_name):
    from reefflow.light.tracer import build_scene, trace
    from reefflow.light.verification import _box_tank, _cfg
    tank = _box_tank(1.0, 0.5, 0.02, 0.5)
    fx = [{"name": "p", "archetype": "led_puck", "position": [0.5, 0.25], "height": 0.2}]
    a = trace(build_scene(_cfg(fx, tank, map_depths=[0.2]), tank), 1 << 20, seed=9)
    b = trace(build_scene(_cfg(fx, tank, map_depths=[0.2], lid={"type": "open"}), tank), 1 << 20, seed=9)
    assert np.array_equal(a.plan_w, b.plan_w) and a.counters == b.counters


def test_lid_effect_on_par(backend_name):
    """Puck over a tank: PAR at 20 cm depth falls by about the lid transmission (glass) and
    by less than the open fraction would suggest for a bright screen (scatter)."""
    from reefflow.light.tracer import build_scene, trace
    from reefflow.light.verification import _box_tank, _cfg
    tank = _box_tank(0.6, 0.6, 0.02, 0.5)
    fx = [{"name": "p", "archetype": "led_puck", "position": [0.3, 0.3], "height": 0.2}]
    mean = {}
    for name in ("open", "glass_float_6mm", "screen_black_1_8", "screen_clear_1_8"):
        cfg = _cfg(fx, tank, map_depths=[0.2], lid={"preset": name, "height": "25 mm"})
        tal = trace(build_scene(cfg, tank), 1 << 21, seed=2)
        mean[name] = tal.plan_w[0].sum()
    g = mean["glass_float_6mm"] / mean["open"]
    assert 0.80 < g < 0.89                       # 88 % straight down, a little less on the slant
    b = mean["screen_black_1_8"] / mean["open"]
    c = mean["screen_clear_1_8"] / mean["open"]
    assert 0.5 < b < _lid(preset="screen_black_1_8").open_fraction   # slanted rays: strands shade a bit more
    assert c > b
