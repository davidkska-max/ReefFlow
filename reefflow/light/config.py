"""``lighting:`` section of the project file -> LightingConfig.

Schema (every field optional except ``fixtures``; defaults in brackets)::

    lighting:
      fixtures:                          # list; archetype from the fixture library + overrides
        - name: left
          archetype: led_puck
          position: [12 in, 8.5 in]      # fixture centre over the tank (tank x, y)
          height: 8 in                   # emitting face above the water surface
          tilt_x: 0                      # deg, + aims the beam toward the back glass  [0]
          tilt_y: 0                      # deg, + aims the beam toward +x (right)       [0]
          rotate_z: 0                    # deg, footprint orientation                   [0]
          offset: [0 in, 0 in, 0 in]     # bracket / hanging offset                     [0]
          intensity: 100                 # master %                                     [100]
          channels: {white: 70}          # % per channel                                [100 each]
          # any archetype field can be overridden: ppf | power + efficacy, distribution,
          # footprint, emitters
      photoperiod:                       # either on/off/ramp/peak, or points
        on: "10:00"                      # [10:00]
        off: "20:00"                     # [20:00]
        ramp: 60 min                     # linear ramp up after 'on' and down before 'off' [60 min]
        peak: 100                        # % of the configured intensities               [100]
        # points: [["09:00", 0], ["10:00", 100], ["19:00", 100], ["20:00", 0]]  (piecewise linear)
      water:
        kd: 0.08 1/m                     # broadband diffuse attenuation (clear reef water) [0.08 1/m]
        kd_extra: 0.03 1/m               # spectral mode: Kd(l) = a_pure_water(l) + kd_extra  [0.03 1/m]
        n: 1.34                          # refractive index of seawater                    [1.34]
      surface:
        roughness: 0.0                   # RMS facet slope of the water surface (0 = flat)  [0]
      glass:
        n: 1.5                           #                                                  [1.5]
        walls: {front: clear, back: black, left: clear, right: clear}   # clear|black|coralline [all clear]
        coralline_albedo: 0.3            #                                                  [0.3]
      albedo: {sand: 0.6, rock: 0.15}    # Lambertian reflectance                           [0.6, 0.15]
      bounces: true                      # diffuse reflection off sand / rock / coralline    [true]
      rock_geometry: auto                # auto | stl | voxels | none                       [auto]
      sdf_resolution: 2.5 mm             # rock signed-distance grid spacing (STL)          [2.5 mm]
      spectral: false                    # per-channel spectra, wavelength-dependent Kd      [false]
      rays: 2.0e7                        # photons traced (minimum when target_noise is set) [2e7]
      target_noise: 0.02                 # trace until the average-lit surface patch has this
                                         # relative 1-sigma noise (null = exactly 'rays')    [0.02]
      max_rays: 2.0e8                    # cap when target_noise is used                    [2e8]
      seed: 1
      map_depths: [3 in, 6 in]           # plan PAR maps (depth below the water surface)
      map_spacing: 75 mm                 # ... or every this much down to the sand          [75 mm]
      map_resolution: 5 mm               # plan-map pixel size                               [5 mm]
      section_y: null                    # front-elevation section plane (tank y)           [mid-width]
      calibration: {...}                 # see calibration.py

All quantities are SI internally (lengths m, angles rad, PPF umol/s,
PAR umol m^-2 s^-1, Kd 1/m).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from . import lunits as LU
from .fixtures import LAMBDA_CENTRES, Fixture, build_fixture, load_library

WALL_CONDITIONS = {"clear": 0, "black": 1, "coralline": 2}
WALL_NAMES = ("front", "back", "left", "right")          # y = 0, y = W, x = 0, x = L

DEFAULTS = {
    "water": {"kd": "0.08 1/m", "kd_extra": "0.03 1/m", "n": 1.34},
    "surface": {"roughness": 0.0},
    "glass": {"n": 1.5, "walls": {}, "coralline_albedo": 0.3},
    "albedo": {"sand": 0.6, "rock": 0.15},
    "bounces": True,
    "rock_geometry": "auto",
    "sdf_resolution": "2.5 mm",
    "spectral": False,
    "rays": 2.0e7,
    "target_noise": 0.02,
    "max_rays": 2.0e8,
    "seed": 1,
    "map_spacing": "75 mm",
    "map_resolution": "5 mm",
    "section_y": None,
    "photoperiod": {"on": "10:00", "off": "20:00", "ramp": "60 min", "peak": 100},
    "probe_size": "25 mm",
    "max_events": 200,
    "batch": 1 << 22,
}

# Pure-water absorption coefficient a_w (1/m), Pope & Fry (1997), 400-700 nm (approximate,
# 20-nm samples). Used as the spectral part of Kd in spectral mode (assumption A-407).
_AW_NM = np.array([400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700], float)
_AW = np.array([0.00663, 0.00454, 0.00635, 0.00916, 0.0145, 0.0204, 0.0477, 0.0568, 0.0619, 0.0894,
                0.2224, 0.2755, 0.3170, 0.4100, 0.4650, 0.6240])


def pure_water_absorption(lam_nm) -> np.ndarray:
    return np.interp(np.asarray(lam_nm, float), _AW_NM, _AW)


# ----------------------------------------------------------------------
# photoperiod
# ----------------------------------------------------------------------
@dataclass
class Photoperiod:
    """Daily intensity schedule s(t) in [0, 1] (fraction of the configured intensities),
    piecewise linear and 24-h periodic."""
    t: np.ndarray                    # s after midnight (sorted)
    s: np.ndarray

    @classmethod
    def parse(cls, d: dict | None) -> "Photoperiod":
        d = dict(d or DEFAULTS["photoperiod"])
        # YAML 1.1 parses unquoted keys on:/off: as booleans True/False; accept them
        if True in d:
            d["on"] = d.pop(True)
        if False in d:
            d["off"] = d.pop(False)
        if "points" in d:
            pts = sorted((LU.parse_clock(a), LU.parse(b, "percent") / 100.0) for a, b in d["points"])
            t = np.array([p[0] for p in pts]); s = np.array([p[1] for p in pts])
        else:
            dd = {**DEFAULTS["photoperiod"], **d}
            on, off = LU.parse_clock(dd["on"]), LU.parse_clock(dd["off"])
            if off <= on:
                off += 86400.0
            ramp = LU.parse(dd["ramp"], "time")
            peak = LU.parse(dd["peak"], "percent") / 100.0
            if 2 * ramp > off - on:
                raise ValueError("photoperiod: ramps longer than the photoperiod")
            t = np.array([on, on + ramp, off - ramp, off]); s = np.array([0.0, peak, peak, 0.0])
            t = np.mod(t, 86400.0)
            o = np.argsort(t, kind="stable"); t, s = t[o], s[o]
        if np.any((s < 0) | (s > 1.0 + 1e-9)):
            raise ValueError("photoperiod intensities must be 0-100 %")
        return cls(t, s)

    def __call__(self, tod) -> np.ndarray:
        tt = np.concatenate([self.t - 86400.0, self.t, self.t + 86400.0])
        ss = np.concatenate([self.s, self.s, self.s])
        return np.interp(np.mod(np.asarray(tod, float), 86400.0), tt, ss)

    @property
    def full_intensity_seconds(self) -> float:
        """Integral of s(t) over 24 h (s): DLI = PAR x this x 1e-6 (mol m^-2 d^-1)."""
        t = np.linspace(0.0, 86400.0, 86401)
        return float(np.trapezoid(self(t), t))

    @property
    def dli_factor(self) -> float:
        """mol m^-2 d^-1 per umol m^-2 s^-1 at full intensity."""
        return self.full_intensity_seconds * 1e-6


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
@dataclass
class LightingConfig:
    fixtures: list
    spectra: dict
    photoperiod: Photoperiod
    kd: float                        # broadband Kd (1/m)
    kd_extra: float
    n_water: float
    roughness: float
    n_glass: float
    walls: dict                      # name -> 'clear' | 'black' | 'coralline'
    coralline_albedo: float
    albedo_sand: float
    albedo_rock: float
    bounces: bool
    rock_geometry: str
    sdf_resolution: float
    spectral: bool
    rays: int
    target_noise: float | None
    max_rays: int
    seed: int
    map_depths: list
    map_resolution: float
    section_y: float | None
    probe_size: float
    max_events: int
    batch: int
    calibration: dict | None
    raw: dict = field(default_factory=dict)

    @property
    def total_ppf(self) -> float:
        return float(sum(f.ppf for f in self.fixtures))

    def kd_bins(self) -> np.ndarray:
        """Kd per spectral bin (spectral mode) or the single broadband value."""
        if self.spectral:
            return pure_water_absorption(LAMBDA_CENTRES) + self.kd_extra
        return np.array([self.kd])

    def wall_codes(self) -> list:
        return [WALL_CONDITIONS[self.walls[w]] for w in WALL_NAMES]

    def summary(self) -> dict:
        return {"fixtures": [f.summary() for f in self.fixtures], "total_ppf_umol_s": self.total_ppf,
                "kd_1_m": self.kd, "spectral": self.spectral, "n_water": self.n_water, "n_glass": self.n_glass,
                "surface_roughness_rms_slope": self.roughness, "walls": dict(self.walls),
                "albedo": {"sand": self.albedo_sand, "rock": self.albedo_rock,
                           "coralline": self.coralline_albedo}, "bounces": self.bounces,
                "photoperiod_full_intensity_h": self.photoperiod.full_intensity_seconds / 3600.0,
                "rays": self.rays, "target_noise": self.target_noise, "seed": self.seed,
                "map_depths_m": list(self.map_depths), "rock_geometry": self.rock_geometry}


def parse_lighting(raw: dict, tank, library: dict | None = None) -> LightingConfig:
    """Parse the ``lighting`` mapping for a ``reefflow.project.Tank``."""
    if raw is None:
        raise ValueError("project has no 'lighting' section")
    raw = copy.deepcopy(raw)
    lib = library or load_library()
    d = {**DEFAULTS, **{k: v for k, v in raw.items() if v is not None or k in ("target_noise", "section_y")}}
    fx_raw = raw.get("fixtures") or []
    if not fx_raw:
        raise ValueError("lighting: at least one fixture is required")
    fixtures: list[Fixture] = [build_fixture(f, lib, tank.water_surface, i) for i, f in enumerate(fx_raw)]
    names = [f.name for f in fixtures]
    if len(set(names)) != len(names):
        raise ValueError("lighting: fixture names must be unique")
    water = {**DEFAULTS["water"], **(raw.get("water") or {})}
    glass = {**DEFAULTS["glass"], **(raw.get("glass") or {})}
    walls = {w: "clear" for w in WALL_NAMES}
    for k, v in (glass.get("walls") or {}).items():
        if k not in walls:
            raise ValueError(f"lighting.glass.walls: unknown wall {k!r}; use {WALL_NAMES}")
        if v not in WALL_CONDITIONS:
            raise ValueError(f"lighting.glass.walls.{k}: {v!r} must be one of {sorted(WALL_CONDITIONS)}")
        walls[k] = v
    alb = {**DEFAULTS["albedo"], **(raw.get("albedo") or {})}
    for k in ("sand", "rock"):
        if not 0 <= float(alb[k]) <= 1:
            raise ValueError(f"lighting.albedo.{k} must be in [0, 1]")
    if "map_depths" in raw and raw["map_depths"] is not None:
        depths = sorted(LU.parse(v, "length") for v in raw["map_depths"])
    else:
        sp = LU.parse(d["map_spacing"], "length")
        depths = list(np.arange(1, int(tank.water_column / sp) + 1) * sp)
        depths = [float(x) for x in depths if x < tank.water_column - 1e-6]
    for z in depths:
        if not 0 < z < tank.water_column:
            raise ValueError(f"lighting: map depth {z:.3f} m is not inside the water column")
    rg = d["rock_geometry"]
    if rg not in ("auto", "stl", "voxels", "none"):
        raise ValueError("lighting.rock_geometry must be auto | stl | voxels | none")
    surf = {**DEFAULTS["surface"], **(raw.get("surface") or {})}
    cfg = LightingConfig(
        fixtures=fixtures, spectra=lib.get("spectra", {}),
        photoperiod=Photoperiod.parse(raw.get("photoperiod")),
        kd=LU.parse(water["kd"], "attenuation"), kd_extra=LU.parse(water["kd_extra"], "attenuation"),
        n_water=float(water["n"]), roughness=float(surf["roughness"]),
        n_glass=float(glass["n"]), walls=walls, coralline_albedo=float(glass["coralline_albedo"]),
        albedo_sand=float(alb["sand"]), albedo_rock=float(alb["rock"]), bounces=bool(d["bounces"]),
        rock_geometry=rg, sdf_resolution=LU.parse(d["sdf_resolution"], "length"),
        spectral=bool(d["spectral"]), rays=int(float(d["rays"])),
        target_noise=None if d.get("target_noise") is None else float(d["target_noise"]),
        max_rays=int(float(d["max_rays"])), seed=int(d["seed"]), map_depths=depths,
        map_resolution=LU.parse(d["map_resolution"], "length"),
        section_y=None if d.get("section_y") is None else LU.parse(d["section_y"], "length"),
        probe_size=LU.parse(d["probe_size"], "length"), max_events=int(d["max_events"]),
        batch=int(d["batch"]), calibration=raw.get("calibration"), raw=raw)
    if cfg.kd < 0 or cfg.kd_extra < 0:
        raise ValueError("lighting.water: Kd must be >= 0")
    if not 1.0 < cfg.n_water < 2.0 or not 1.0 < cfg.n_glass < 2.5:
        raise ValueError("lighting: implausible refractive index")
    if cfg.roughness < 0:
        raise ValueError("lighting.surface.roughness must be >= 0")
    if cfg.rays <= 0:
        raise ValueError("lighting.rays must be positive")
    return cfg


def lighting_from_project(project, library: dict | None = None, overrides: dict | None = None) -> LightingConfig:
    """LightingConfig for a ``reefflow.project.Project`` (``overrides`` merged into the raw section)."""
    raw = copy.deepcopy(project.raw.get("lighting"))
    if raw is None:
        raise ValueError(f"project {project.name!r} has no 'lighting' section")
    for k, v in (overrides or {}).items():
        raw[k] = v
    return parse_lighting(raw, project.tank, library)


def dli(par, cfg: LightingConfig):
    """DLI (mol m^-2 d^-1) from PAR at full configured intensity (umol m^-2 s^-1)."""
    return np.asarray(par, float) * cfg.photoperiod.dli_factor
