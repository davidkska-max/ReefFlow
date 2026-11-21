"""Light-fixture archetypes: angular distributions, spectra and emitter layouts.

A fixture is expanded into *emitters*. Each emitter is a uniformly emitting
rectangle (or point) with an emission axis and an angular distribution of
radiant intensity I(theta) about that axis. Photons are emitted with equal
weight; an emitter is chosen with probability proportional to its photon
output, a point on it uniformly, and a direction from its distribution.

Direction sampling uses a tabulated inverse CDF of mu = cos(theta): the solid
angle element is dOmega = 2 pi dmu, so the pdf of mu is proportional to
I(theta(mu)). For I ~ cos^m on 0 <= theta <= theta_c the inverse CDF is
    mu(u) = [1 - u (1 - mu_c^(m+1))]^(1/(m+1)),
and for the isotropic (full-sphere) source mu(u) = 1 - 2u.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

from . import lunits as LU

N_TABLE = 4096                       # inverse-CDF table length
LAMBDA_EDGES = np.arange(400.0, 701.0, 10.0)      # spectral bins (nm), PAR band
LAMBDA_CENTRES = 0.5 * (LAMBDA_EDGES[1:] + LAMBDA_EDGES[:-1])


# ----------------------------------------------------------------------
# library
# ----------------------------------------------------------------------
def load_library(path=None) -> dict:
    """Fixture library (archetypes + spectra). Package data, with a file fallback
    so it also works from a source tree / editable install without package-data."""
    if path is not None:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    lib = yaml.safe_load(_data_text("fixture_library.yaml"))
    prods = load_products()
    clash = set(lib["archetypes"]) & set(prods)
    if clash:
        raise ValueError(f"fixture product ids clash with generic archetypes: {sorted(clash)}")
    lib["archetypes"].update(prods)
    return lib


def _data_text(name: str) -> str:
    try:
        return resources.files("reefflow.light").joinpath(f"data/{name}").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return (Path(__file__).parent / "data" / name).read_text(encoding="utf-8")


def load_products() -> dict:
    """Real light fixtures (D-844): {id: archetype fields + brand, model, sources}. Missing file -> {}."""
    try:
        prods = yaml.safe_load(_data_text("fixture_products.yaml")) or {}
    except (FileNotFoundError, OSError):
        return {}
    for v in prods.values():
        v["product"] = True
    return prods


# ----------------------------------------------------------------------
# angular distributions
# ----------------------------------------------------------------------
def beam_angle_to_m(beam: float) -> float:
    """cos^m exponent whose full width at half maximum is ``beam`` (rad)."""
    c = math.cos(0.5 * beam)
    if not 0.0 < c < 1.0:
        raise ValueError("beam angle must be between 0 and 180 deg")
    return math.log(0.5) / math.log(c)


@dataclass
class Distribution:
    """Angular distribution of radiant intensity about the emission axis."""
    kind: str                        # isotropic | cosine_power | collimated | table
    m: float = 1.0                   # cosine power
    cutoff: float = math.pi / 2      # rad
    table_theta: np.ndarray | None = None
    table_value: np.ndarray | None = None

    @classmethod
    def parse(cls, d: dict) -> "Distribution":
        d = dict(d or {"type": "lambertian"})
        t = d.get("type", "lambertian")
        cut = LU.parse(d["cutoff"], "angle") if d.get("cutoff") is not None else math.pi / 2
        if t == "isotropic":
            return cls("isotropic", cutoff=math.pi)
        if t == "lambertian":
            return cls("cosine_power", m=1.0, cutoff=cut)
        if t == "cosine_power":
            return cls("cosine_power", m=float(d["m"]), cutoff=cut)
        if t == "beam_angle":
            return cls("cosine_power", m=beam_angle_to_m(LU.parse(d["beam_angle"], "angle")), cutoff=cut)
        if t == "collimated":
            return cls("collimated", cutoff=0.0)
        if t == "table":
            th = np.radians(np.asarray(d["angles"], float))
            v = np.asarray(d["values"], float)
            if th.ndim != 1 or th.shape != v.shape or len(th) < 2 or np.any(np.diff(th) <= 0) or np.any(v < 0):
                raise ValueError("table distribution: 'angles' (deg, increasing) and 'values' (>= 0) of equal length")
            return cls("table", cutoff=min(cut, float(th[-1])), table_theta=th, table_value=v)
        raise ValueError(f"unknown distribution type {t!r}")

    @property
    def beam_angle(self) -> float:
        """Full width at half maximum (rad) of a cos^m distribution."""
        if self.kind != "cosine_power":
            return float("nan")
        return 2.0 * math.acos(0.5 ** (1.0 / self.m))

    def intensity(self, theta) -> np.ndarray:
        """Relative radiant intensity I(theta) (I(0) = 1), zero beyond the cutoff."""
        th = np.asarray(theta, float)
        if self.kind == "isotropic":
            return np.ones_like(th)
        if self.kind == "collimated":
            return np.where(th == 0, np.inf, 0.0)
        if self.kind == "cosine_power":
            v = np.clip(np.cos(th), 0.0, None) ** self.m
        else:
            v = np.interp(th, self.table_theta, self.table_value, right=0.0)
        return np.where(th <= self.cutoff + 1e-12, v, 0.0)

    def normalised_intensity(self, theta) -> np.ndarray:
        """I(theta) / Phi: radiant intensity per unit emitted flux (1/sr)."""
        if self.kind == "isotropic":
            return np.full_like(np.asarray(theta, float), 1.0 / (4 * math.pi))
        if self.kind == "cosine_power":
            mc = math.cos(self.cutoff)
            norm = 2 * math.pi * (1.0 - max(mc, 0.0) ** (self.m + 1)) / (self.m + 1)
            return self.intensity(theta) / norm
        mu = np.linspace(math.cos(self.cutoff), 1.0, 20001)
        norm = 2 * math.pi * np.trapezoid(self.intensity(np.arccos(mu)), mu)
        return self.intensity(theta) / norm

    def inverse_cdf(self, n: int = N_TABLE) -> np.ndarray:
        """mu = cos(theta) at u = k/(n-1), k = 0..n-1 (u = 0 -> on-axis)."""
        u = np.linspace(0.0, 1.0, n)
        if self.kind == "isotropic":
            return 1.0 - 2.0 * u
        if self.kind == "collimated":
            return np.ones(n)
        if self.kind == "cosine_power":
            a = self.m + 1.0
            mc = max(math.cos(self.cutoff), 0.0)
            return (1.0 - u * (1.0 - mc ** a)) ** (1.0 / a)
        mu = np.linspace(1.0, math.cos(self.cutoff), 20001)          # decreasing
        p = self.intensity(np.arccos(np.clip(mu, -1, 1)))
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * -np.diff(mu))])
        if cdf[-1] <= 0:
            raise ValueError("table distribution emits nothing")
        cdf /= cdf[-1]
        return np.interp(u, cdf, mu)

    def describe(self) -> str:
        if self.kind == "cosine_power":
            s = f"cos^{self.m:.2f} (beam {math.degrees(self.beam_angle):.0f} deg)"
            if self.cutoff < math.pi / 2 - 1e-9:
                s += f", cutoff {math.degrees(self.cutoff):.0f} deg"
            return s
        return self.kind


# ----------------------------------------------------------------------
# spectra
# ----------------------------------------------------------------------
def spectrum_bins(peaks) -> np.ndarray:
    """Relative photon spectrum (sum of Gaussians [peak, FWHM, height]) integrated
    over the 10-nm PAR bins and normalised to sum 1 (fraction of PAR photons per bin)."""
    lam = np.arange(400.0, 700.0, 0.25) + 0.125
    s = np.zeros_like(lam)
    for pk, fwhm, h in peaks:
        sig = float(fwhm) / 2.3548
        s += float(h) * np.exp(-0.5 * ((lam - float(pk)) / sig) ** 2)
    b = np.add.reduceat(s, np.searchsorted(lam, LAMBDA_EDGES[:-1]))
    if b.sum() <= 0:
        raise ValueError("spectrum has no photons in 400-700 nm")
    return b / b.sum()


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _rot(axis: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Channel:
    name: str
    share: float                     # normalised fraction of the 100 % PPF
    intensity: float                 # % setting
    spectrum: str


@dataclass
class Fixture:
    name: str
    archetype: str
    ppf_rated: float                 # umol/s at 100 % on all channels
    master: float                    # master intensity %
    channels: list
    distribution: Distribution
    footprint: tuple                 # (length, width) m
    grid: tuple                      # emitters (nx, ny)
    emitter_size: tuple              # (length, width) m of each emitter
    centre: np.ndarray               # emitting-face centre, tank coordinates (m)
    R: np.ndarray                    # local -> tank rotation; local -z is the emission axis
    height: float                    # emitting face above the water surface (m), before offset
    raw: dict = field(default_factory=dict)

    @property
    def ppf(self) -> float:
        """Photon output at the configured channel and master intensities (umol/s)."""
        return self.ppf_rated * self.master / 100.0 * sum(c.share * c.intensity / 100.0 for c in self.channels)

    @property
    def axis(self) -> np.ndarray:
        return self.R @ np.array([0.0, 0.0, -1.0])

    def emitter_rows(self, profile_index: int, fixture_index: int) -> tuple[np.ndarray, np.ndarray]:
        """(rows (n, 16), photon output per emitter (n,)). Row layout: centre(3), axis(3),
        u-axis(3), v-axis(3), half-length u, half-width v, profile index, fixture index."""
        nx, ny = self.grid
        L, W = self.footprint
        ex = (np.arange(nx) + 0.5) / nx * L - 0.5 * L
        ey = (np.arange(ny) + 0.5) / ny * W - 0.5 * W
        u = self.R @ np.array([1.0, 0.0, 0.0])
        v = self.R @ np.array([0.0, 1.0, 0.0])
        rows = []
        for x in ex:
            for y in ey:
                c = self.centre + x * u + y * v
                rows.append(np.concatenate([c, self.axis, u, v,
                                            [0.5 * self.emitter_size[0], 0.5 * self.emitter_size[1],
                                             profile_index, fixture_index]]))
        rows = np.array(rows)
        return rows, np.full(len(rows), self.ppf / len(rows))

    def spectrum(self, spectra: dict) -> np.ndarray:
        """Photon spectrum of the fixture at its channel settings (fractions per bin)."""
        tot = np.zeros(len(LAMBDA_CENTRES))
        for c in self.channels:
            if c.spectrum not in spectra:
                raise ValueError(f"fixture {self.name!r}: unknown spectrum {c.spectrum!r}")
            tot += c.share * c.intensity * spectrum_bins(spectra[c.spectrum])
        s = tot.sum()
        return tot / s if s > 0 else np.full(len(tot), 1.0 / len(tot))

    def summary(self) -> dict:
        return {"name": self.name, "archetype": self.archetype, "ppf_rated_umol_s": self.ppf_rated,
                "ppf_umol_s": self.ppf, "master_pct": self.master,
                "channels": {c.name: {"share": c.share, "intensity_pct": c.intensity} for c in self.channels},
                "distribution": self.distribution.describe(), "centre_m": self.centre.tolist(),
                "axis": self.axis.tolist(), "height_above_water_m": self.height,
                "footprint_m": list(self.footprint), "emitters": list(self.grid)}


def build_fixture(d: dict, library: dict, water_surface: float, index: int = 0) -> Fixture:
    """Fixture from a project-file entry (archetype + overrides).

    Mounting: ``position`` [x, y] of the fixture centre over the tank (tank
    coordinates), ``height`` of the emitting face above the water surface,
    ``tilt_x`` (deg; + aims the beam toward the back glass), ``tilt_y`` (deg;
    + aims toward +x / right), ``rotate_z`` (deg; footprint orientation),
    ``offset`` [dx, dy, dz] bracket/hanging offset added to the position."""
    arches = library["archetypes"]
    arch = d.get("archetype")
    if arch not in arches:
        raise ValueError(f"fixture {d.get('name')!r}: unknown archetype {arch!r}; choose from {sorted(arches)}")
    p = copy.deepcopy(arches[arch])
    for k, v in d.items():
        if k == "distribution" and isinstance(v, dict) and "type" not in v:
            p["distribution"] = {**p.get("distribution", {}), **v}
        elif k == "channels" and isinstance(v, dict) and all(not isinstance(x, dict) for x in v.values()):
            pass                                        # channel intensities, handled below
        elif v is not None:
            p[k] = v
    name = d.get("name") or f"{arch}-{index + 1}"
    if p.get("ppf") is not None:
        ppf = LU.parse(p["ppf"], "ppf")
    elif p.get("power") is not None and p.get("efficacy") is not None:
        ppf = LU.parse(p["power"], "power") * LU.parse(p["efficacy"], "efficacy")
    else:
        raise ValueError(f"fixture {name!r}: give 'ppf' or 'power' and 'efficacy'")
    if ppf < 0:
        raise ValueError(f"fixture {name!r}: negative photon output")
    chans_def = p.get("channels") or {"all": {"share": 1.0, "spectrum": "flat"}}
    settings = d.get("channels") if isinstance(d.get("channels"), dict) else {}
    settings = {k: v for k, v in settings.items() if not isinstance(v, dict)}
    unknown = set(settings) - set(chans_def)
    if unknown:
        raise ValueError(f"fixture {name!r}: unknown channel(s) {sorted(unknown)}; archetype has {sorted(chans_def)}")
    tot = sum(float(c.get("share", 1.0)) for c in chans_def.values())
    chans = []
    for cn, c in chans_def.items():
        pct = LU.parse(settings.get(cn, c.get("intensity", 100)), "percent")
        if not 0 <= pct <= 100:
            raise ValueError(f"fixture {name!r}: channel {cn} intensity {pct} % outside 0-100")
        chans.append(Channel(cn, float(c.get("share", 1.0)) / tot, pct, c.get("spectrum", "flat")))
    master = LU.parse(p.get("intensity", 100), "percent")
    if not 0 <= master <= 100:
        raise ValueError(f"fixture {name!r}: intensity {master} % outside 0-100")
    fp = p.get("footprint", {"length": 0, "width": 0})
    em = p.get("emitters", {"nx": 1, "ny": 1, "size": [0, 0]})
    size = tuple(LU.parse(v, "length") for v in em.get("size", [0, 0]))
    pos = p.get("position")
    if pos is None or len(pos) not in (2, 3):
        raise ValueError(f"fixture {name!r}: 'position' [x, y] over the tank is required")
    if len(pos) == 3:
        raise ValueError(f"fixture {name!r}: give 'position' as [x, y] and the mounting height as 'height' "
                         "(emitting face above the water surface)")
    if p.get("height") is None:
        raise ValueError(f"fixture {name!r}: 'height' above the water surface is required")
    h = LU.parse(p["height"], "length")
    off = np.array([LU.parse(v, "length") for v in p.get("offset", [0, 0, 0])], float)
    centre = np.array([LU.parse(pos[0], "length"), LU.parse(pos[1], "length"), water_surface + h]) + off
    if centre[2] <= water_surface:
        raise ValueError(f"fixture {name!r}: emitting face is not above the water surface")
    R = (_rot("y", -LU.parse(p.get("tilt_y", 0), "angle")) @ _rot("x", LU.parse(p.get("tilt_x", 0), "angle"))
         @ _rot("z", LU.parse(p.get("rotate_z", 0), "angle")))
    return Fixture(name=name, archetype=arch, ppf_rated=ppf, master=master, channels=chans,
                   distribution=Distribution.parse(p.get("distribution")),
                   footprint=(LU.parse(fp.get("length", 0), "length"), LU.parse(fp.get("width", 0), "length")),
                   grid=(int(em.get("nx", 1)), int(em.get("ny", 1))), emitter_size=size, centre=centre, R=R,
                   height=h, raw=dict(d))
