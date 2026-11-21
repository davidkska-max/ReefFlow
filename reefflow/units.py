"""Unit handling.

Two separate concerns live here:

1. ``to_si`` / ``from_si``: user-facing US <-> SI conversion (in, ft, GPH, L/h ...).
   Everything inside ReefFlow is SI.

2. ``LatticeUnits``: the physical <-> lattice mapping for the LBM.  This is the
   classic place LBM codes go wrong, so every conversion factor is spelled out
   and unit-tested.

Lattice units: dx_lat = dt_lat = 1, rho_lat = 1 (reference density),
c_s,lat^2 = 1/3.  With physical spacing dx [m], step dt [s] and reference
density rho0 [kg/m^3] the conversion factors are

    length        C_L   = dx
    time          C_T   = dt
    velocity      C_U   = dx / dt
    kinematic visc C_nu = dx^2 / dt
    density       C_rho = rho0
    acceleration  C_A   = dx / dt^2
    force density C_F   = rho0 dx / dt^2          (N/m^3)
    pressure      C_P   = rho0 dx^2 / dt^2         (Pa)
    volume flow   C_Q   = dx^3 / dt
    momentum flux C_J   = rho0 dx^4 / dt^2         (N, i.e. rho Q^2 / A)

BGK relaxation time: nu_lat = c_s^2 (tau - 1/2).
Pressure (gauge): p = c_s^2 (rho_lat - 1) C_P.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CS2 = 1.0 / 3.0

# ----------------------------------------------------------------------
# US / SI conversion
# ----------------------------------------------------------------------
INCH = 0.0254                 # m (exact)
FOOT = 0.3048                 # m (exact)
US_GALLON = 3.785411784e-3    # m^3 (exact)
LITRE = 1e-3                  # m^3
HOUR = 3600.0                 # s
POUND = 0.45359237            # kg (exact)

# unit name -> (quantity kind, factor to SI)
_UNITS = {
    # length
    "m": ("length", 1.0), "cm": ("length", 0.01), "mm": ("length", 1e-3),
    "in": ("length", INCH), "ft": ("length", FOOT),
    # volume
    "m3": ("volume", 1.0), "L": ("volume", LITRE), "gal": ("volume", US_GALLON),
    # volumetric flow
    "m3/s": ("flow", 1.0), "L/h": ("flow", LITRE / HOUR), "L/min": ("flow", LITRE / 60.0),
    "GPH": ("flow", US_GALLON / HOUR), "GPM": ("flow", US_GALLON / 60.0),
    # velocity
    "m/s": ("velocity", 1.0), "cm/s": ("velocity", 0.01), "in/s": ("velocity", INCH),
    "ft/s": ("velocity", FOOT),
    # mass
    "kg": ("mass", 1.0), "lb": ("mass", POUND),
    # density
    "kg/m3": ("density", 1.0), "lb/ft3": ("density", POUND / FOOT**3),
    # head (pump head as height of water column)
    "m_head": ("head", 1.0), "ft_head": ("head", FOOT),
    # time
    "s": ("time", 1.0), "ms": ("time", 1e-3), "min": ("time", 60.0), "h": ("time", HOUR),
}

# kinds that also accept units of another kind (pump head is a length of water)
_ALIASES = {"head": ("length",)}


def to_si(value: float, unit: str) -> float:
    """Convert ``value`` in ``unit`` to SI."""
    try:
        return value * _UNITS[unit][1]
    except KeyError:
        raise ValueError(f"Unknown unit {unit!r}. Known: {sorted(_UNITS)}") from None


def from_si(value: float, unit: str) -> float:
    """Convert an SI ``value`` to ``unit``."""
    try:
        return value / _UNITS[unit][1]
    except KeyError:
        raise ValueError(f"Unknown unit {unit!r}. Known: {sorted(_UNITS)}") from None


def kind(unit: str) -> str:
    return _UNITS[unit][0]


def parse(value, kind_: str, default_unit: str | None = None) -> float:
    """Parse a project-file quantity into SI.

    Accepts a number (taken in ``default_unit``, or SI if none) or a string
    "<number> <unit>", e.g. "48 in", "2500 GPH", "0.5 s", "1/2 in", "10 ft".
    The unit must be of the requested kind ('length', 'flow', 'time', ...).
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a {kind_}, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value) if default_unit is None else to_si(float(value), default_unit)
    if not isinstance(value, str):
        raise ValueError(f"expected a {kind_} (number or '<number> <unit>'), got {value!r}")
    parts = value.strip().split()
    if len(parts) != 2:
        raise ValueError(f"cannot parse {value!r}: expected '<number> <unit>'")
    num, unit = parts
    try:
        if "/" in num:
            a, b = num.split("/")
            x = float(a) / float(b)
        else:
            x = float(num)
    except ValueError:
        raise ValueError(f"cannot parse number in {value!r}") from None
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r} in {value!r}. Known: {sorted(_UNITS)}")
    k = _UNITS[unit][0]
    if k != kind_ and k not in _ALIASES.get(kind_, ()):
        raise ValueError(f"{value!r} is a {k}, expected a {kind_}")
    return to_si(x, unit)


DISPLAY = {
    "SI": {"length": "mm", "flow": "L/h", "velocity": "cm/s", "mass": "kg", "volume": "L"},
    "US": {"length": "in", "flow": "GPH", "velocity": "in/s", "mass": "lb", "volume": "gal"},
}


# ----------------------------------------------------------------------
# Physical <-> lattice
# ----------------------------------------------------------------------
class LatticeParameterError(ValueError):
    pass


@dataclass(frozen=True)
class LatticeUnits:
    dx: float              # m
    dt: float              # s
    rho0: float = 1023.0   # kg/m^3, reference density (seawater; ASSUMPTIONS A-3)

    def __post_init__(self):
        if not (self.dx > 0 and self.dt > 0 and self.rho0 > 0):
            raise LatticeParameterError("dx, dt and rho0 must all be positive")

    # --- constructors -------------------------------------------------
    @classmethod
    def from_velocity(cls, dx: float, u_phys: float, u_lat: float = 0.1, rho0: float = 1023.0):
        """Acoustic scaling: choose dt so that physical speed ``u_phys`` maps to ``u_lat``."""
        if u_phys <= 0:
            raise LatticeParameterError("u_phys must be positive")
        return cls(dx=dx, dt=u_lat * dx / u_phys, rho0=rho0)

    @classmethod
    def from_tau(cls, dx: float, nu_phys: float, tau: float, rho0: float = 1023.0):
        """Diffusive scaling: choose dt so that molecular viscosity gives relaxation time ``tau``."""
        if tau <= 0.5:
            raise LatticeParameterError("tau must exceed 0.5")
        return cls(dx=dx, dt=CS2 * (tau - 0.5) * dx * dx / nu_phys, rho0=rho0)

    # --- conversion factors -------------------------------------------
    @property
    def C_U(self): return self.dx / self.dt
    @property
    def C_nu(self): return self.dx * self.dx / self.dt
    @property
    def C_A(self): return self.dx / self.dt**2
    @property
    def C_F(self): return self.rho0 * self.dx / self.dt**2
    @property
    def C_P(self): return self.rho0 * self.dx**2 / self.dt**2
    @property
    def C_Q(self): return self.dx**3 / self.dt
    @property
    def C_J(self): return self.rho0 * self.dx**4 / self.dt**2

    # --- conversions ---------------------------------------------------
    def velocity_to_lattice(self, u): return u / self.C_U
    def velocity_to_physical(self, u): return u * self.C_U
    def time_to_steps(self, t): return t / self.dt
    def steps_to_time(self, n): return n * self.dt
    def length_to_cells(self, L): return L / self.dx
    def nu_to_lattice(self, nu): return nu / self.C_nu
    def nu_to_physical(self, nu_lat): return nu_lat * self.C_nu
    def tau_from_nu(self, nu_phys): return 0.5 + self.nu_to_lattice(nu_phys) / CS2
    def force_density_to_lattice(self, f): return f / self.C_F
    def force_density_to_physical(self, f): return f * self.C_F
    def acceleration_to_lattice(self, a): return a / self.C_A
    def flow_to_lattice(self, Q): return Q / self.C_Q
    def flow_to_physical(self, Q): return Q * self.C_Q
    def momentum_flux_to_lattice(self, J): return J / self.C_J
    def pressure_from_density(self, rho_lat): return CS2 * (rho_lat - 1.0) * self.C_P

    @property
    def sound_speed(self) -> float:
        """Artificial (lattice) speed of sound in m/s. Real seawater is ~1500 m/s."""
        return math.sqrt(CS2) * self.C_U

    # --- checks --------------------------------------------------------
    def check(self, u_max_phys: float, nu_phys: float, *, u_lat_limit: float = 0.1,
              tau_min: float = 0.5, strict: bool = True) -> dict:
        """Report the lattice parameters of a run and refuse unsafe ones.

        u_lat_limit applies to the lattice velocity (spec: u_lat <~ 0.1, i.e. Ma = u/c_s ~ 0.17).
        """
        u_lat = self.velocity_to_lattice(u_max_phys)
        nu_lat = self.nu_to_lattice(nu_phys)
        tau = 0.5 + nu_lat / CS2
        rep = {
            "dx_m": self.dx, "dt_s": self.dt, "u_max_lat": u_lat, "mach_lat": u_lat / math.sqrt(CS2),
            "nu_lat": nu_lat, "tau_molecular": tau,
            "re_cell": u_lat / nu_lat if nu_lat > 0 else math.inf,
            "density_variation_est": (u_lat / math.sqrt(CS2)) ** 2,
        }
        problems = []
        if u_lat > u_lat_limit:
            problems.append(f"lattice velocity {u_lat:.3f} exceeds limit {u_lat_limit}")
        if tau <= tau_min:
            problems.append(f"tau {tau:.6f} <= {tau_min}")
        rep["problems"] = problems
        if problems and strict:
            raise LatticeParameterError("; ".join(problems))
        return rep
