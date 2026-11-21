"""Node-type map (flags) and boundary-condition data for an LBM run.

Node types
----------
FLUID     ordinary fluid node (collide + stream).
SOLID     wall. Halfway bounce-back: the wall sits midway between the solid node
          and the adjacent fluid node. A non-zero ``u_bc`` on a solid node makes it
          a moving wall (Ladd correction), e.g. the cavity lid.
FREESLIP  impermeable, zero-shear plane (specular reflection, halfway). Only
          axis-aligned planes are supported; the plane normal is inferred.
INLET     prescribed velocity ``u_bc`` (regularized non-equilibrium extrapolation).
OUTLET    prescribed density ``rho_bc`` i.e. pressure (same scheme).
SOURCE    fluid node carrying a volumetric source term (pump force, mass
          source or sink; see sources.py). Streams and collides like FLUID.

INLET/OUTLET nodes are not part of the flow domain: they hold ghost populations
that are rebuilt every step from their inward neighbour, so the boundary
surface coincides with the INLET/OUTLET node plane.
"""
from __future__ import annotations

import numpy as np

FLUID, SOLID, FREESLIP, INLET, OUTLET, SOURCE = 0, 1, 2, 3, 4, 5
NAMES = {FLUID: "fluid", SOLID: "solid", FREESLIP: "freeslip", INLET: "inlet", OUTLET: "outlet",
         SOURCE: "source"}
FLUID_LIKE = (FLUID, SOURCE)
_FACES = {"x-": (0, 0), "x+": (0, 1), "y-": (1, 0), "y+": (1, 1), "z-": (2, 0), "z+": (2, 1)}


def encode_normal(n) -> int:
    return (int(n[0]) + 1) * 9 + (int(n[1]) + 1) * 3 + (int(n[2]) + 1)


def decode_normal(a: int) -> tuple[int, int, int]:
    return (a // 9 - 1, (a // 3) % 3 - 1, a % 3 - 1)


class Domain:
    def __init__(self, shape, periodic=(False, False, False)):
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != 3:
            raise ValueError("shape must be (nx, ny, nz)")
        self.periodic = tuple(bool(p) for p in periodic)
        self.flags = np.full(self.shape, FLUID, dtype=np.int8)
        self.u_bc = np.zeros(self.shape + (3,), dtype=np.float32)
        self.rho_bc = np.ones(self.shape, dtype=np.float32)
        self.aux = np.full(self.shape, encode_normal((0, 0, 0)), dtype=np.int8)
        self._final = False

    # ------------------------------------------------------------------
    def face_slice(self, face: str):
        axis, side = _FACES[face]
        sl = [slice(None)] * 3
        sl[axis] = 0 if side == 0 else self.shape[axis] - 1
        return tuple(sl)

    def set_face(self, face: str, kind: int, velocity=None, rho=None):
        sl = self.face_slice(face)
        self.flags[sl] = kind
        if velocity is not None:
            self.u_bc[sl] = np.asarray(velocity, dtype=np.float32)
        if rho is not None:
            self.rho_bc[sl] = rho
        self._final = False

    def set_mask(self, mask: np.ndarray, kind: int, velocity=None, rho=None):
        mask = np.asarray(mask, dtype=bool)
        self.flags[mask] = kind
        if velocity is not None:
            v = np.asarray(velocity, dtype=np.float32)
            self.u_bc[mask] = v if v.ndim == 1 else v[mask]
        if rho is not None:
            self.rho_bc[mask] = rho
        self._final = False

    # ------------------------------------------------------------------
    def finalize(self) -> "Domain":
        """Infer inward normals for FREESLIP/INLET/OUTLET nodes and validate."""
        bmask = np.isin(self.flags, (FREESLIP, INLET, OUTLET))
        idx = np.argwhere(bmask)
        if len(idx):
            n = np.zeros((len(idx), 3), dtype=np.int64)
            for a in range(3):
                if self.periodic[a]:
                    continue
                n[idx[:, a] == 0, a] += 1
                n[idx[:, a] == self.shape[a] - 1, a] -= 1
            interior = ~np.any(n != 0, axis=1)
            if np.any(interior):
                for row in np.where(interior)[0]:
                    p = idx[row]
                    acc = np.zeros(3, dtype=np.int64)
                    for a in range(3):
                        for s in (-1, 1):
                            q = p.copy(); q[a] += s
                            if self.periodic[a]:
                                q[a] %= self.shape[a]
                            elif not (0 <= q[a] < self.shape[a]):
                                continue
                            if self.flags[tuple(q)] in FLUID_LIKE:
                                acc[a] += s
                    n[row] = np.clip(acc, -1, 1)
            for row, p in enumerate(idx):
                self.aux[tuple(p)] = encode_normal(n[row])
            # validate open boundaries: neighbour along the normal must be fluid
            for row, p in enumerate(idx):
                kind = self.flags[tuple(p)]
                if kind in (INLET, OUTLET):
                    if not np.any(n[row]):
                        raise ValueError(f"{NAMES[kind]} node {tuple(p)} has no fluid neighbour")
                    q = p + n[row]
                    for a in range(3):
                        if self.periodic[a]:
                            q[a] %= self.shape[a]
                    if self.flags[tuple(q)] not in FLUID_LIKE:
                        raise ValueError(f"{NAMES[kind]} node {tuple(p)}: neighbour {tuple(q)} along "
                                         f"normal {tuple(n[row])} is {NAMES[int(self.flags[tuple(q)])]}, "
                                         "not fluid")
        self._final = True
        return self

    @property
    def has_moving_walls(self) -> bool:
        solid = self.flags == SOLID
        return bool(np.any(self.u_bc[solid] != 0.0))

    @property
    def has_open_boundaries(self) -> bool:
        return bool(np.any(np.isin(self.flags, (INLET, OUTLET))))

    def count(self, kind) -> int:
        return int(np.count_nonzero(np.isin(self.flags, kind)))
