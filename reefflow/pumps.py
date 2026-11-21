"""Pump models: archetype library, delivered flow, momentum flux, equivalent
outlets, operating modes and inflow turbulence (host side, SI units).

Physics (see DECISIONS D-202 to D-205)
--------------------------------------
* Powerhead: momentum source without net mass (it recirculates tank water).
  Total force F = J = rho Q^2 / A, the momentum flux of its outlet jet, with
  Q the *delivered* flow (rated x derating) and A the true outlet area.
* Return nozzle: mass source rho Q carrying the nozzle velocity U = Q / A, i.e.
  momentum rate rho Q U = rho Q^2 / A. Q comes from a generic pump curve
  Q = Q0 sqrt(1 - H/H0) at total head H = static head + fittings allowance.
* Several return nozzles on one pump (D-805): the pump's total Q is unchanged;
  nozzle i gets Q_i = share_i Q with share_i proportional to its outlet area by
  default (orifices at a common manifold pressure), or the user's shares.
* Equivalent outlet (PLAN C3): an outlet narrower than ``min_cells`` cells, or
  whose equivalent velocity sqrt(J / (rho A)) exceeds ``velocity_cap``, is
  modelled over a larger area A_eq at the same J. The far field of a jet
  depends on J only; the near field (< ~5 D) is approximate. Logged per run.
* Modes: s(t) = Q(t)/Q scales the flow; the force scales as s^2, mass as s.
* Inflow turbulence: random (Ornstein-Uhlenbeck) amplitudes of 7 outlet modes
  (uniform, and r cos m theta, r sin m theta for m = 1..3) plus a random lateral
  component, relative RMS intensity I, correlation time D_eq / U_eq.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import yaml

from . import units as U

RHO0 = 1023.0
G0 = 9.80665


def _read_data(name: str) -> dict:
    with resources.files("reefflow.data").joinpath(name).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_products() -> dict:
    """Real pumps (D-844): {id: entry}. Each entry has the archetype fields (kind, outlet,
    rated_flow, derating / pump_curve, ...) plus brand, model and where each number came from."""
    try:
        prods = _read_data("pump_products.yaml")
    except FileNotFoundError:
        return {}
    for k, v in prods.items():
        v["product"] = True
        # modelling defaults that are not product data (same as the generic archetypes)
        v.setdefault("turbulence", 0.05)
        v.setdefault("housing", None)
        if v.get("kind") == "powerhead":
            v.setdefault("derating", 0.7)                         # A-205, unchanged
        elif v.get("kind") == "return":
            v.setdefault("outlet", {"shape": "circle", "diameter": "19 mm"})   # the nozzle in the tank, not the pump
            v.setdefault("outlet_basis", "generic 3/4 in return nozzle (the pump's own outlet size does not set the jet)")
            v.setdefault("static_head", "4 ft")
            v.setdefault("fittings_allowance", "1 ft")
    return prods


def load_library() -> dict:
    """Generic archetypes plus the real-product catalogue (product ids never clash with archetypes)."""
    lib = _read_data("pump_library.yaml")
    prods = load_products()
    clash = set(lib) & set(prods)
    if clash:
        raise ValueError(f"pump product ids clash with generic archetypes: {sorted(clash)}")
    return {**lib, **prods}


# ----------------------------------------------------------------------
# outlet geometry
# ----------------------------------------------------------------------
@dataclass
class Outlet:
    shape: str                   # 'circle' | 'rect'
    diameter: float = 0.0        # m (circle)
    width: float = 0.0           # m (rect, along the pump's lateral axis)
    height: float = 0.0          # m (rect, along the pump's 'up' axis)

    @classmethod
    def parse(cls, d: dict) -> "Outlet":
        shape = d.get("shape", "circle")
        if shape == "circle":
            return cls("circle", diameter=U.parse(d["diameter"], "length"))
        if shape == "rect":
            return cls("rect", width=U.parse(d["width"], "length"), height=U.parse(d["height"], "length"))
        raise ValueError(f"unknown outlet shape {shape!r}")

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4 if self.shape == "circle" else self.width * self.height

    @property
    def hydraulic_size(self) -> float:
        """Smallest outlet dimension (circle: diameter; rect: min(width, height))."""
        return self.diameter if self.shape == "circle" else min(self.width, self.height)

    def scaled(self, k: float) -> "Outlet":
        return Outlet(self.shape, self.diameter * k, self.width * k, self.height * k)

    def describe(self) -> str:
        if self.shape == "circle":
            return f"circle D = {self.diameter * 1e3:.1f} mm"
        return f"rect {self.width * 1e3:.1f} x {self.height * 1e3:.1f} mm"


def equivalent_outlet(outlet: Outlet, J: float, dx: float, min_cells: float = 4.0,
                      velocity_cap: float | None = None, min_size: float | None = None,
                      rho: float = RHO0) -> tuple[Outlet, list[str]]:
    """Outlet used in the simulation, same momentum flux J. Returns (outlet, notes).

    min_size (m) overrides min_cells * dx when given (used by the grid study, so
    every grid models the same physical source)."""
    notes = []
    smin = min_size if min_size is not None else min_cells * dx
    o = outlet
    if o.shape == "circle":
        if o.diameter < smin:
            o = Outlet("circle", diameter=smin)
    else:
        w, h = max(o.width, smin), max(o.height, smin)
        if (w, h) != (o.width, o.height):
            o = Outlet("rect", width=w, height=h)
    if o is not outlet:
        notes.append(f"outlet {outlet.describe()} is narrower than {smin * 1e3:.1f} mm "
                     f"(= {smin / dx:.1f} cells); modelled as {o.describe()} at the same momentum flux")
    if velocity_cap is not None:
        ueq = math.sqrt(J / (rho * o.area))
        if ueq > velocity_cap:
            k = ueq / velocity_cap          # area scales by k^2 -> U_eq by 1/k
            o2 = o.scaled(k)
            notes.append(f"equivalent velocity {ueq:.2f} m/s exceeds the cap {velocity_cap:.2f} m/s; "
                         f"outlet enlarged to {o2.describe()} at the same momentum flux")
            o = o2
    return o, notes


# ----------------------------------------------------------------------
# operating modes
# ----------------------------------------------------------------------
@dataclass
class Mode:
    """Flow fraction s(t) = Q(t) / Q_delivered.

    type 'constant' : s = 1
         'sine'     : s = min + (max - min) (1 + sin 2 pi (t/T + phase)) / 2
         'pulse'    : square wave between min and max with duty cycle 'duty' and
                      half-cosine ramps of 'ramp' x period (smooth, avoids pressure shocks)
    Alternating ('wave') operation of two pumps = same mode, phases 0 and 0.5.
    """
    type: str = "constant"
    period: float = 0.0
    min: float = 1.0
    max: float = 1.0
    duty: float = 0.5
    ramp: float = 0.1
    phase: float = 0.0

    @classmethod
    def parse(cls, d: dict | None) -> "Mode":
        if not d:
            return cls()
        t = d.get("type", "constant")
        if t == "constant":
            return cls()
        if t not in ("sine", "pulse"):
            raise ValueError(f"unknown pump mode {t!r} (constant, sine, pulse)")
        m = cls(t, period=U.parse(d["period"], "time"), min=float(d.get("min", 0.3)),
                max=float(d.get("max", 1.0)), duty=float(d.get("duty", 0.5)),
                ramp=float(d.get("ramp", 0.1)), phase=float(d.get("phase", 0.0)))
        if not (0 <= m.min <= m.max) or m.period <= 0:
            raise ValueError("pump mode needs 0 <= min <= max and period > 0")
        if not (0 < m.duty < 1) or not (0 <= m.ramp < min(m.duty, 1 - m.duty)):
            raise ValueError("pulse mode needs 0 < duty < 1 and 0 <= ramp < min(duty, 1 - duty)")
        return m

    def s(self, t):
        t = np.asarray(t, dtype=np.float64)
        if self.type == "constant":
            return np.ones_like(t)
        x = np.mod(t / self.period + self.phase, 1.0)
        if self.type == "sine":
            w = 0.5 * (1.0 + np.sin(2 * np.pi * x))
        else:
            r, d = self.ramp, self.duty
            w = np.where(x < d, 1.0, 0.0)
            if r > 0:
                up = x < r                                  # ramp 0 -> 1 at the start of the 'on' part
                dn = (x >= d) & (x < d + r)                 # ramp 1 -> 0 after it
                w = np.where(up, 0.5 * (1 - np.cos(np.pi * x / r)), w)
                w = np.where(dn, 0.5 * (1 + np.cos(np.pi * (x - d) / r)), w)
        return self.min + (self.max - self.min) * w

    def mean_s2(self) -> float:
        """Time-mean of s^2 (the mean momentum flux relative to constant operation)."""
        if self.type == "constant":
            return 1.0
        t = np.linspace(0, self.period, 20001)[:-1]
        return float(np.mean(self.s(t) ** 2))

    def describe(self) -> str:
        if self.type == "constant":
            return "constant"
        return (f"{self.type}, period {self.period:g} s, flow {self.min:.2f}-{self.max:.2f} x Q, "
                f"phase {self.phase:g}" + (f", duty {self.duty:g}" if self.type == "pulse" else ""))


# ----------------------------------------------------------------------
# pumps
# ----------------------------------------------------------------------
@dataclass
class Housing:
    shape: str
    length: float
    diameter: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @classmethod
    def parse(cls, d: dict | None) -> "Housing | None":
        if not d:
            return None
        shape = d.get("shape", "circle")
        L = U.parse(d["length"], "length")
        if shape == "circle":
            return cls("circle", L, diameter=U.parse(d["diameter"], "length"))
        return cls("rect", L, width=U.parse(d["width"], "length"), height=U.parse(d["height"], "length"))


@dataclass
class Nozzle:
    """One outlet of a pump: position, aim, outlet geometry and share of the pump's flow."""
    position: np.ndarray              # outlet centre, tank coordinates (m)
    aim: np.ndarray                   # unit vector
    up: np.ndarray                    # unit vector perpendicular to aim
    outlet: Outlet
    share: float = 1.0                # fraction of the pump's delivered flow
    share_basis: str = "single outlet"


@dataclass
class Pump:
    name: str
    kind: str                         # 'powerhead' | 'return'
    archetype: str
    outlet: Outlet
    position: np.ndarray              # outlet centre, tank coordinates (m)
    aim: np.ndarray                   # unit vector
    up: np.ndarray                    # unit vector perpendicular to aim (orients rect outlets)
    rated_flow: float                 # m^3/s
    delivered_flow: float             # m^3/s
    flow_basis: str                   # how delivered_flow was obtained
    mode: Mode = field(default_factory=Mode)
    housing: Housing | None = None
    turbulence: float = 0.0
    group: str | None = None          # pumps sharing a group share s(t) (return + its overflow)
    raw: dict = field(default_factory=dict)
    nozzles: list = field(default_factory=list)   # [Nozzle]; empty = single outlet at position/aim
    fittings_allowance: float | None = None       # returns: head allowance (m) that includes the nozzle exit head

    def __post_init__(self):
        if not self.nozzles:
            self.nozzles = [Nozzle(self.position, self.aim, self.up, self.outlet)]

    def nozzle_flow(self, k: int) -> float:
        return self.delivered_flow * self.nozzles[k].share

    def nozzle_velocity_of(self, k: int) -> float:
        return self.nozzle_flow(k) / self.nozzles[k].outlet.area

    def nozzle_momentum_flux(self, k: int) -> float:
        """J_i = rho Q_i^2 / A_i (N)."""
        return RHO0 * self.nozzle_flow(k) ** 2 / self.nozzles[k].outlet.area

    @property
    def momentum_flux(self) -> float:
        """Total J = sum_i rho Q_i^2 / A_i (N) at the delivered flow and true outlet areas."""
        return sum(self.nozzle_momentum_flux(k) for k in range(len(self.nozzles)))

    @property
    def nozzle_velocity(self) -> float:
        """Exit velocity of the first nozzle (the only one for a single-outlet pump)."""
        return self.nozzle_velocity_of(0)

    def nozzle_summaries(self) -> list:
        out = []
        for k, n in enumerate(self.nozzles):
            u = self.nozzle_velocity_of(k)
            out.append({"index": k, "position_m": [float(v) for v in n.position], "aim": [float(v) for v in n.aim],
                        "outlet": n.outlet.describe(), "outlet_area_m2": n.outlet.area,
                        "share": n.share, "share_basis": n.share_basis,
                        "flow_m3s": self.nozzle_flow(k), "flow_GPH": U.from_si(self.nozzle_flow(k), "GPH"),
                        "velocity_ms": u, "exit_velocity_head_m": u * u / (2 * G0),
                        "exit_velocity_head_ft": U.from_si(u * u / (2 * G0), "ft"),
                        "momentum_flux_N": self.nozzle_momentum_flux(k)})
        return out

    def summary(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "archetype": self.archetype,
            "outlet": self.outlet.describe(), "outlet_area_m2": self.outlet.area,
            "position_m": [float(v) for v in self.position], "aim": [float(v) for v in self.aim],
            "rated_flow_m3s": self.rated_flow, "rated_flow_GPH": U.from_si(self.rated_flow, "GPH"),
            "delivered_flow_m3s": self.delivered_flow,
            "delivered_flow_GPH": U.from_si(self.delivered_flow, "GPH"),
            "delivered_flow_Lh": U.from_si(self.delivered_flow, "L/h"),
            "flow_basis": self.flow_basis, "nozzle_velocity_ms": self.nozzle_velocity,
            "momentum_flux_N": self.momentum_flux, "mode": self.mode.describe(),
            "mean_momentum_flux_N": self.momentum_flux * self.mode.mean_s2(),
            "turbulence_intensity": self.turbulence,
            "exit_velocity_head_m": self.nozzle_velocity ** 2 / (2 * G0),
            "nozzles": self.nozzle_summaries(),
        }


def _unit(v, name) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a)
    if a.shape != (3,) or n == 0:
        raise ValueError(f"{name} must be a non-zero 3-vector")
    return a / n


def _default_up(aim: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    ref = z if abs(aim @ z) < 0.9 else np.array([0.0, 1.0, 0.0])
    up = ref - (ref @ aim) * aim
    return up / np.linalg.norm(up)


def return_pump_flow(rated: float, max_head: float, head: float, points=None, shape: str = "sqrt") -> float:
    """Delivered flow of a return pump at total head ``head`` (m).

    Without ``points``: generic centrifugal curve Q = Q0 sqrt(1 - H/H0) (PLAN C4), or the
    straight line Q = Q0 (1 - H/H0) for ``shape='linear'`` (D-845: DC catalogue pumps
    without a published curve; published DC curves are close to straight lines).
    With ``points`` [(H_i, Q_i)] (SI, a published head-flow curve, D-845): linear
    interpolation in H between the points, closed at (0, Q0) and (H0, 0) when those
    ends are not in the table."""
    if head >= max_head:
        raise ValueError(f"total head {head:.2f} m reaches the pump's shut-off head {max_head:.2f} m: no flow")
    if not points:
        if shape == "linear":
            return rated * (1.0 - head / max_head)
        if shape != "sqrt":
            raise ValueError(f"pump_curve shape {shape!r}: use sqrt or linear")
        return rated * math.sqrt(1.0 - head / max_head)
    pts = sorted((float(h), float(q)) for h, q in points)
    if pts[0][0] > 1e-9:
        pts.insert(0, (0.0, rated))
    if pts[-1][0] < max_head - 1e-9:
        pts.append((max_head, 0.0))
    H = np.array([p[0] for p in pts])
    Q = np.array([p[1] for p in pts])
    return float(np.interp(head, H, Q))


def parse_curve(pc: dict | None) -> list | None:
    """Published curve points of a ``pump_curve`` entry as [(H m, Q m^3/s)], or None."""
    if not pc or not pc.get("points"):
        return None
    pts = [(U.parse(h, "head"), U.parse(q, "flow")) for h, q in pc["points"]]
    hs = [h for h, _ in pts]
    if len(set(hs)) != len(hs) or any(q < 0 for _, q in pts):
        raise ValueError("pump_curve points need distinct heads and flows >= 0")
    sp = sorted(pts)
    if any(b[1] > a[1] + 1e-12 for a, b in zip(sp, sp[1:])):
        raise ValueError("pump_curve points: flow must not rise with head")
    return pts


def parse_vector(v, kind_="length") -> np.ndarray:
    if isinstance(v, dict):
        v = [v["x"], v["y"], v["z"]]
    if len(v) != 3:
        raise ValueError(f"expected a 3-vector, got {v!r}")
    return np.array([U.parse(c, kind_) for c in v])


def build_pump(d: dict, library: dict | None = None) -> Pump:
    """Pump from a project-file entry: archetype defaults overridden by the entry."""
    library = library or load_library()
    arch = d.get("archetype")
    if arch not in library:
        raise ValueError(f"pump {d.get('name')!r}: unknown archetype {arch!r}; choose from {sorted(library)}")
    p = {**library[arch], **{k: v for k, v in d.items() if v is not None or k == "housing"}}
    name = p.get("name") or arch
    kind_ = p["kind"]
    outlet = Outlet.parse(p["outlet"])
    nozzles = _parse_nozzles(p, name, kind_, outlet)
    first = nozzles[0]
    rated = U.parse(p["rated_flow"], "flow")
    fit = None
    if p.get("delivered_flow") is not None:
        q = U.parse(p["delivered_flow"], "flow")
        basis = "user override"
    elif kind_ == "powerhead":
        der = float(p.get("derating", 0.7))
        if not 0 < der <= 1:
            raise ValueError(f"pump {name}: derating must be in (0, 1]")
        q = der * rated
        basis = f"rated x derating {der:g}"
    elif kind_ == "return":
        H0 = U.parse(p["pump_curve"]["max_head"], "head")
        fit = U.parse(p.get("fittings_allowance", 0.0), "head")
        H = U.parse(p.get("static_head", 0.0), "head") + fit
        # a user-typed rated flow that differs from the product's Q0 means the published curve no
        # longer applies as given: fall back to the generic curve (stated in the basis)
        pts = parse_curve(p["pump_curve"])
        if pts and d.get("rated_flow") is not None and arch in library and library[arch].get("rated_flow") is not None \
                and abs(rated - U.parse(library[arch]["rated_flow"], "flow")) > 1e-3 * rated:
            pts = None
        shape = p["pump_curve"].get("shape", "sqrt")
        q = return_pump_flow(rated, H0, H, pts, shape)
        if pts:
            basis = (f"manufacturer curve ({len(pts)} points): H = {U.from_si(H, 'ft'):.2f} ft "
                     f"(H0 = {U.from_si(H0, 'ft'):.2f} ft)")
        elif shape == "linear":
            basis = (f"straight-line curve Q0 (1 - H/H0) (no published curve): Q0 = {U.from_si(rated, 'GPH'):.0f} GPH, "
                     f"H = {U.from_si(H, 'ft'):.2f} ft, H0 = {U.from_si(H0, 'ft'):.2f} ft")
        else:
            basis = (f"pump curve Q0 sqrt(1 - H/H0): Q0 = {U.from_si(rated, 'GPH'):.0f} GPH, "
                     f"H = {U.from_si(H, 'ft'):.2f} ft, H0 = {U.from_si(H0, 'ft'):.2f} ft")
    else:
        raise ValueError(f"pump {name}: kind must be 'powerhead' or 'return'")
    return Pump(name=name, kind=kind_, archetype=arch, outlet=outlet,
                position=first.position, aim=first.aim, up=first.up, rated_flow=rated,
                delivered_flow=q, flow_basis=basis, mode=Mode.parse(p.get("mode")),
                housing=Housing.parse(p.get("housing")), turbulence=float(p.get("turbulence", 0.0)),
                raw=dict(d), nozzles=nozzles, fittings_allowance=fit if kind_ == "return" else None)


def _aim_up(aim_raw, up_raw, label):
    aim = _unit(aim_raw, f"{label}: aim")
    up = _unit(up_raw, f"{label}: up") if up_raw is not None else _default_up(aim)
    up = up - (up @ aim) * aim
    if np.linalg.norm(up) < 1e-6:
        raise ValueError(f"{label}: 'up' is parallel to 'aim'")
    return aim, up / np.linalg.norm(up)


def _parse_nozzles(p: dict, name: str, kind_: str, outlet: Outlet) -> list:
    """Nozzles of a pump entry: a ``nozzles:`` list (return pumps) or the single position/aim.

    Flow split (D-805): explicit ``share`` values are kept; nozzles without one split the
    remaining flow in proportion to their outlet areas (orifices fed from one manifold at a
    common pressure with equal discharge coefficients: Q_i ~ A_i). If every nozzle has a
    share, the shares must sum to 1 within 1 % and are then normalised exactly."""
    lst = p.get("nozzles")
    if not lst:
        if p.get("position") is None or p.get("aim") is None:
            raise ValueError(f"pump {name}: needs position and aim (or a nozzles list)")
        aim, up = _aim_up(p["aim"], p.get("up"), f"pump {name}")
        return [Nozzle(parse_vector(p["position"]), aim, up, outlet)]
    if kind_ != "return":
        raise ValueError(f"pump {name}: only return pumps can have several nozzles")
    if not isinstance(lst, list):
        raise ValueError(f"pump {name}: nozzles must be a list")
    out = []
    for i, n in enumerate(lst):
        lab = f"pump {name} nozzle {i + 1}"
        if not isinstance(n, dict) or n.get("position") is None or n.get("aim") is None:
            raise ValueError(f"{lab}: needs position and aim")
        aim, up = _aim_up(n["aim"], n.get("up"), lab)
        if n.get("outlet") is not None:
            o = Outlet.parse(n["outlet"])
        elif n.get("diameter") is not None:
            o = Outlet("circle", diameter=U.parse(n["diameter"], "length"))
        else:
            o = outlet
        if o.area <= 0:
            raise ValueError(f"{lab}: outlet area must be > 0")
        out.append(Nozzle(parse_vector(n["position"]), aim, up, o))
    given = [None if n.get("share") is None else float(n["share"]) for n in lst]
    if any(g is not None and not (0 < g <= 1) for g in given):
        raise ValueError(f"pump {name}: nozzle shares must be in (0, 1]")
    fixed = sum(g for g in given if g is not None)
    free = [i for i, g in enumerate(given) if g is None]
    if not free:
        if abs(fixed - 1.0) > 0.01:
            raise ValueError(f"pump {name}: nozzle shares sum to {fixed:.3f}; they must sum to 1 "
                             "(or leave some empty to split the rest by outlet area)")
        for nz, g in zip(out, given):
            nz.share, nz.share_basis = g / fixed, "user share"
    else:
        rest = 1.0 - fixed
        if rest <= 1e-6:
            raise ValueError(f"pump {name}: the given nozzle shares ({fixed:.3f}) leave no flow for "
                             f"{len(free)} nozzle(s) without a share")
        a_free = sum(out[i].outlet.area for i in free)
        for i, nz in enumerate(out):
            if given[i] is not None:
                nz.share, nz.share_basis = given[i], "user share"
            else:
                nz.share = rest * nz.outlet.area / a_free
                nz.share_basis = "single outlet" if len(out) == 1 else "split by outlet area"
    return out


# ----------------------------------------------------------------------
# inflow turbulence
# ----------------------------------------------------------------------
class OUProcess:
    """Independent unit-variance Ornstein-Uhlenbeck processes (exact discrete update)."""

    def __init__(self, n: int, tcorr: float, rng: np.random.Generator):
        self.n, self.tcorr, self.rng = n, tcorr, rng
        self.x = rng.standard_normal(n)

    def advance(self, dt: float) -> np.ndarray:
        if self.tcorr <= 0:
            self.x = self.rng.standard_normal(self.n)
        else:
            a = math.exp(-dt / self.tcorr)
            self.x = a * self.x + math.sqrt(1.0 - a * a) * self.rng.standard_normal(self.n)
        return self.x


def mode_shapes(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """phi_k (N, 7) at outlet coordinates t1, t2 normalised by the outlet half-sizes:
    1, r cos(m theta), r sin(m theta) for m = 1..3; each mode scaled to mean square 1/7,
    so sum_k a_k phi_k has unit variance for unit-variance a_k."""
    r = np.sqrt(t1 ** 2 + t2 ** 2)
    th = np.arctan2(t2, t1)
    cols = [np.ones_like(r)]
    for m in (1, 2, 3):
        cols += [r * np.cos(m * th), r * np.sin(m * th)]
    phi = np.stack(cols, axis=1)
    ms = np.mean(phi ** 2, axis=0)
    ms[ms == 0] = 1.0
    return phi / np.sqrt(7.0 * ms)
