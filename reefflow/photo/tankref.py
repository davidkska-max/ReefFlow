"""Tank geometry and named reference points for photo calibration.

Tank coordinates (same as the rest of ReefFlow): origin at the inside
bottom-left-front glass corner, x along the length (left -> right seen from the
front), y front -> back, z up from the inside of the bottom glass, metres.

Reference points are the inside corners of the glass box at four levels:

    rim     z = internal height H  (top edge of the glass)
    water   z = water surface      (where the still surface meets the glass)
    sand    z = sand depth         (where the sand surface meets the glass)
    bottom  z = 0                  (inside face of the bottom glass; visible through
                                    a side pane as the bottom of the sand layer)

and four corners FL (x=0, y=0), FR (x=L, y=0), BR (x=L, y=W), BL (x=0, y=W).
A point is named "<level>_<corner>", e.g. "sand_BL".

Outside corners (DECISIONS D-836), visible on rimmed and rimless tanks alike:

    top     z = H,  outside corner of the box (top of the frame on a rimmed tank,
                    top edge of the glass on a rimless one)
    base    z = -gb, outside corner where the tank (or its base frame) sits on the stand
                    (gb = bottom glass thickness)

at x in {-g, L+g}, y in {-g, W+g}. The external size the user enters is exactly this
box, so these clicks match the entered dimensions by construction.

Seams: "seam_<corner>" (and "seam_<corner>_2") is a click anywhere on the vertical
inside corner line at that corner (x, y as above, z from 0 to H). It is used as an
edge constraint (one equation per click), not as a point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import units as U

N_AIR = 1.0          # 1.0003 at STP; neglected (ASSUMPTIONS A-603)
N_WATER = 1.34       # seawater, visible band (spec; A-603)
N_GLASS = 1.5        # soda-lime / low-iron float glass (A-603)

LEVELS = ("rim", "water", "sand", "bottom")
OUTER_LEVELS = ("top", "base")
CORNERS = ("FL", "FR", "BR", "BL")
VIEW_KINDS = ("top", "front", "back", "left", "right")


@dataclass
class TankGeometry:
    """Everything the photo module needs to know about the tank (SI units)."""
    length: float                    # internal, x
    width: float                     # internal, y
    height: float                    # internal, z (inside of bottom glass to rim)
    glass: float                     # side-pane thickness
    water_surface: float             # z of the still water surface
    sand_depth: float                # z of the sand surface
    n_water: float = N_WATER
    n_glass: float = N_GLASS
    n_air: float = N_AIR
    bottom_glass: float | None = None    # base corners sit at z = -bottom_glass (default: glass)

    def __post_init__(self):
        if min(self.length, self.width, self.height) <= 0:
            raise ValueError("tank dimensions must be positive")
        if not (0 <= self.sand_depth < self.water_surface <= self.height):
            raise ValueError("need 0 <= sand_depth < water_surface <= height")
        if self.glass < 0:
            raise ValueError("glass thickness must be >= 0")

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_tank(cls, tank, **kw) -> "TankGeometry":
        """From a :class:`reefflow.project.Tank`."""
        return cls(length=tank.length, width=tank.width, height=tank.height, glass=tank.glass,
                   water_surface=tank.water_surface, sand_depth=tank.sand_depth,
                   bottom_glass=getattr(tank, "bottom_glass", None), **kw)

    @classmethod
    def from_project(cls, path, **kw) -> "TankGeometry":
        from ..project import load
        return cls.from_tank(load(path).tank, **kw)

    @classmethod
    def from_dims(cls, length, width, height, glass, water_level_below_rim, sand_depth,
                  unit: str = "in", internal: bool = True, **kw) -> "TankGeometry":
        """From user-entered numbers in ``unit``. If ``internal`` is False the
        length/width/height are external and the glass is subtracted (bottom glass
        assumed equal to the side glass, as in the project file default)."""
        s = U.to_si(1.0, unit)
        L, W, H, g = length * s, width * s, height * s, glass * s
        if not internal:
            L, W, H = L - 2 * g, W - 2 * g, H - g
        return cls(length=L, width=W, height=H, glass=g, water_surface=H - water_level_below_rim * s,
                   sand_depth=sand_depth * s, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TankGeometry":
        return cls(**d)

    # -- reference points -------------------------------------------------
    def level_z(self, level: str) -> float:
        return {"rim": self.height, "water": self.water_surface, "sand": self.sand_depth,
                "bottom": 0.0}[level]

    def corner_xy(self, corner: str):
        return {"FL": (0.0, 0.0), "FR": (self.length, 0.0), "BR": (self.length, self.width),
                "BL": (0.0, self.width)}[corner]

    def outer_corner_xy(self, corner: str):
        g = self.glass
        return {"FL": (-g, -g), "FR": (self.length + g, -g), "BR": (self.length + g, self.width + g),
                "BL": (-g, self.width + g)}[corner]

    def reference_points(self) -> dict:
        """{name: (x, y, z)} for every level x corner, inside (rim/water/sand/bottom)
        and outside (top/base)."""
        out = {}
        for lv in LEVELS:
            for c in CORNERS:
                x, y = self.corner_xy(c)
                out[f"{lv}_{c}"] = (x, y, self.level_z(lv))
        gb = self.glass if self.bottom_glass is None else self.bottom_glass
        for lv, z in (("top", self.height), ("base", -gb)):
            for c in CORNERS:
                x, y = self.outer_corner_xy(c)
                out[f"{lv}_{c}"] = (x, y, z)
        return out

    def seam(self, name: str):
        """End points (A, B) of the vertical inside corner line named seam_<corner>[_2]."""
        parts = name.split("_")
        if (len(parts) not in (2, 3) or parts[0] != "seam" or parts[1] not in CORNERS
                or (len(parts) == 3 and parts[2] != "2")):
            raise KeyError(f"unknown seam {name!r}; names are seam_<corner> or seam_<corner>_2")
        x, y = self.corner_xy(parts[1])
        return np.array([x, y, 0.0]), np.array([x, y, self.height])

    def point(self, name: str) -> np.ndarray:
        try:
            return np.asarray(self.reference_points()[name], dtype=float)
        except KeyError:
            raise KeyError(f"unknown reference point {name!r}; names are <level>_<corner> with "
                           f"level in {LEVELS} and corner in {CORNERS}") from None

    def in_water(self, P) -> np.ndarray:
        """True where the point is inside the tank and at or below the water surface."""
        P = np.asarray(P, float)
        return ((P[..., 0] >= 0) & (P[..., 0] <= self.length) & (P[..., 1] >= 0)
                & (P[..., 1] <= self.width) & (P[..., 2] <= self.water_surface + 1e-9))


# Points a user can normally see and click in each view, with a plain-language hint.
_HINTS = {
    "rim": "top inside corner of the glass (rimless tanks; hidden under the frame on a rimmed tank)",
    "water": "where the still water surface meets the glass corner",
    "sand": "where the sand surface meets the glass corner",
    "bottom": "bottom inside corner of the glass (often hidden by a base frame or the sand)",
    "top": "top outside corner of the tank: the top of the frame (rimmed tank) or of the glass (rimless)",
    "base": "bottom outside corner: where the tank, or its black base frame, sits on the stand",
    "seam": "any point on this vertical corner joint of the glass, seen through the water (low down is best)",
    "seam2": "a second point on the same corner joint, higher up",
}
_SIDE = {"FL": "front left", "FR": "front right", "BR": "back right", "BL": "back left"}


def all_names() -> list:
    """Every name a click may use: points (inside and outside levels) and seams."""
    pts = [f"{lv}_{c}" for lv in LEVELS + OUTER_LEVELS for c in CORNERS]
    return pts + [f"seam_{c}" for c in CORNERS] + [f"seam_{c}_2" for c in CORNERS]


def is_seam(name: str) -> bool:
    return name.startswith("seam_")


def _entry(n):
    lv, c = n.split("_")[0], n.split("_")[1]
    key = "seam2" if is_seam(n) and n.endswith("_2") else lv
    return {"name": n, "type": "seam" if is_seam(n) else "point", "hint": f"{_SIDE[c]}: {_HINTS[key]}"}


def describe(name: str) -> dict:
    """{name, type, hint} for any valid click name."""
    if name not in all_names():
        raise KeyError(f"unknown reference point {name!r}")
    return _entry(name)


def recommended_points(kind: str, scheme: str = "outer") -> list[dict]:
    """Ordered list of {name, type, hint} that are usually visible in a view of this kind.

    scheme 'outer' (default, D-836): the outside corners of the near face (frame top
    and where the tank sits on the stand) -- visible on rimmed and rimless tanks --
    plus, through the water, the far corners at the water line and the far vertical
    corner seams. Top view: outside top corners and the sand corners.
    scheme 'inner' (before D-836): inside glass corners, which a frame or the sand hides.

    The set always mixes two or more levels / depths so it is not coplanar
    (PLAN C9: a single-plane set cannot fix focal length and distance)."""
    if kind not in VIEW_KINDS:
        raise ValueError(f"view kind must be one of {VIEW_KINDS}")
    if scheme not in ("outer", "inner"):
        raise ValueError("scheme must be 'outer' or 'inner'")
    if kind == "top":
        names = ([f"{'top' if scheme == 'outer' else 'rim'}_{c}" for c in CORNERS]
                 + [f"sand_{c}" for c in CORNERS])
    else:
        near, far = {"front": (("FL", "FR"), ("BL", "BR")), "back": (("BR", "BL"), ("FR", "FL")),
                     "left": (("BL", "FL"), ("BR", "FR")), "right": (("FR", "BR"), ("FL", "BL"))}[kind]
        if scheme == "outer":
            names = ([f"top_{c}" for c in near] + [f"base_{c}" for c in near]
                     + [f"water_{c}" for c in far] + [f"seam_{c}" for c in far])
        else:
            names = ([f"rim_{c}" for c in near] + [f"bottom_{c}" for c in near]
                     + [f"sand_{c}" for c in far] + [f"water_{c}" for c in far])
    return [_entry(n) for n in names]
