"""Photometric units for the lighting model (kept here so ``reefflow.units`` stays CFD-only).

Internal units
    photon flux (PPF)          umol/s          (micromoles of 400-700 nm photons per second)
    photon irradiance (PAR)    umol m^-2 s^-1
    daily light integral (DLI) mol m^-2 d^-1
    power                      W
    efficacy                   umol/J
    attenuation coefficient    1/m
    wavelength                 nm
    angle                      rad internally; plain numbers in a project file are DEGREES

The photon unit is the micromole (not the SI mole) because every horticultural
PAR quantity (PPF, PPFD, fixture data sheets, PAR meters) is quoted in umol.
Lengths, times etc. are delegated to ``reefflow.units.parse``.
"""
from __future__ import annotations

import math

from .. import units as U

# unit -> (kind, factor to the internal unit above)
_LIGHT_UNITS = {
    # photon flux
    "umol/s": ("ppf", 1.0), "mmol/s": ("ppf", 1e3), "mol/s": ("ppf", 1e6),
    # photon irradiance
    "umol/m2/s": ("par", 1.0), "umol/m^2/s": ("par", 1.0), "PAR": ("par", 1.0), "PPFD": ("par", 1.0),
    # DLI
    "mol/m2/d": ("dli", 1.0),
    # power / efficacy
    "W": ("power", 1.0), "kW": ("power", 1e3),
    "umol/J": ("efficacy", 1.0),
    # attenuation
    "1/m": ("attenuation", 1.0), "1/cm": ("attenuation", 100.0), "m-1": ("attenuation", 1.0),
    # wavelength
    "nm": ("wavelength", 1.0),
    # angle (to radians)
    "deg": ("angle", math.pi / 180.0), "rad": ("angle", 1.0),
    # percent
    "%": ("percent", 1.0),
}

# what a bare number means for each light kind
_BARE = {"ppf": 1.0, "par": 1.0, "dli": 1.0, "power": 1.0, "efficacy": 1.0, "attenuation": 1.0,
         "wavelength": 1.0, "angle": math.pi / 180.0, "percent": 1.0}


def parse(value, kind: str) -> float:
    """Parse a quantity into the internal unit of ``kind``.

    Light kinds: ppf, par, dli, power, efficacy, attenuation, wavelength, angle, percent.
    Any other kind (length, time, ...) is passed to ``reefflow.units.parse``.
    Bare numbers: internal unit, except angles (degrees).
    """
    if kind not in _BARE:
        return U.parse(value, kind)
    if isinstance(value, bool):
        raise ValueError(f"expected a {kind}, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value) * _BARE[kind]
    if not isinstance(value, str):
        raise ValueError(f"expected a {kind} (number or '<number> <unit>'), got {value!r}")
    s = value.strip()
    if s.endswith("%") and kind == "percent":
        return float(s[:-1])
    parts = s.split()
    if len(parts) != 2:
        raise ValueError(f"cannot parse {value!r}: expected '<number> <unit>'")
    num, unit = parts
    try:
        x = float(num.split("/")[0]) / float(num.split("/")[1]) if "/" in num else float(num)
    except ValueError:
        raise ValueError(f"cannot parse number in {value!r}") from None
    if unit not in _LIGHT_UNITS:
        raise ValueError(f"unknown unit {unit!r} in {value!r}. Known light units: {sorted(_LIGHT_UNITS)}")
    k, f = _LIGHT_UNITS[unit]
    if k != kind:
        raise ValueError(f"{value!r} is a {k}, expected a {kind}")
    return x * f


def parse_clock(value) -> float:
    """'HH:MM' (or hours as a number) -> seconds after midnight."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * 3600.0
    s = str(value).strip()
    if ":" not in s:
        raise ValueError(f"time of day {value!r}: expected 'HH:MM'")
    h, m = s.split(":")[:2]
    t = int(h) * 3600 + int(m) * 60
    if not 0 <= t <= 24 * 3600:
        raise ValueError(f"time of day {value!r} outside 00:00-24:00")
    return float(t)
