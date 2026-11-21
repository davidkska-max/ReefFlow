"""Camera model with a flat refractive interface (air | glass | water, or air | water).

Pixel convention (OpenCV): u to the right, v down, the centre of the top-left
pixel is (0, 0). Images are H x W (x 3, RGB) arrays; ``mask[v, u]``.

Camera: pinhole with square pixels, focal length ``f`` [px], principal point
(cx, cy) (default: image centre) and optional one-term radial distortion ``k1``
on normalized coordinates (default 0). Pose: X_cam = R X_world + t, camera
axes x right, y down, z forward. Camera centre C = -R^T t.

Refraction (DECISIONS D-602, ASSUMPTIONS A-602/A-603)
------------------------------------------------------
Every view looks through a stack of *parallel planes* with a common unit normal
``n`` pointing from the camera into the tank:

    top view            air | water            plane: the still water surface z = z_w
    front/back/left/    air | glass | inner    planes: outer and inner face of the
    right view                                 pane facing the camera

``inner`` is water for points at or below the water surface and air above it.
For parallel planes Snell's law conserves the tangential quantity
``s = n_i sin(theta_i)`` along the whole ray, and the ray stays in the plane
spanned by ``n`` and the camera-to-point offset. Forward projection therefore
reduces to one scalar equation per point,

    sum_i d_i * s / sqrt(n_i^2 - s^2) = h,

where d_i are the layer thicknesses along ``n`` between camera and point and
``h`` the tangential (in-plane) distance from camera to point. The left side
is increasing and convex in s on [0, n_min); it is solved by safeguarded
Newton iteration to 1e-14 (vectorized). The air-side ray direction
``s e_t + sqrt(1 - s^2) n`` is then imaged with the pinhole model. This is
exact for flat interfaces -- no small-angle (apparent-depth) approximation.

Back-projection inverts this: pixel -> air ray -> refracted at every plane ->
(entry point on the last plane, direction in the inner medium).

Validity: a projection is flagged invalid when the straight-layer model does
not describe the real light path, e.g. a submerged point whose ray would have
to cross the water surface in a side view, a point seen through a side pane
outside the pane, or a top-view ray that enters the water outside the tank
opening. Invalid projections carry no information (carving ignores them).
"""
from __future__ import annotations

import numpy as np

from .tankref import TankGeometry, VIEW_KINDS


# ----------------------------------------------------------------------
# rotations
# ----------------------------------------------------------------------
def rotvec_to_matrix(r) -> np.ndarray:
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-15:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def matrix_to_rotvec(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


# ----------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------
class Camera:
    """Pinhole camera (square pixels, optional radial k1) with pose X_c = R X_w + t."""

    def __init__(self, f, R, t, width, height, cx=None, cy=None, k1=0.0):
        self.f = float(f)
        self.R = np.asarray(R, float).reshape(3, 3)
        self.t = np.asarray(t, float).reshape(3)
        self.width, self.height = int(width), int(height)
        self.cx = (self.width - 1) / 2.0 if cx is None else float(cx)
        self.cy = (self.height - 1) / 2.0 if cy is None else float(cy)
        self.k1 = float(k1)

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @classmethod
    def look_at(cls, eye, target, f, width, height, up=(0, 0, 1), roll_deg=0.0, k1=0.0):
        """Camera at ``eye`` looking at ``target``; image 'up' is as close to ``up`` as possible."""
        eye = np.asarray(eye, float)
        z = np.asarray(target, float) - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, np.asarray(up, float))
        if np.linalg.norm(x) < 1e-9:
            raise ValueError("view direction parallel to 'up'")
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        if roll_deg:
            a = np.radians(roll_deg)
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            R = Rz @ R
        return cls(f, R, -R @ eye, width, height, k1=k1)

    def project_dirs(self, dirs):
        """World-frame directions from the camera centre -> pixels (N, 2), in_front (N,)."""
        d = np.atleast_2d(np.asarray(dirs, float)) @ self.R.T
        front = d[:, 2] > 1e-12
        z = np.where(front, d[:, 2], 1.0)
        x, y = d[:, 0] / z, d[:, 1] / z
        if self.k1:
            r2 = x * x + y * y
            x, y = x * (1 + self.k1 * r2), y * (1 + self.k1 * r2)
        uv = np.stack([self.f * x + self.cx, self.f * y + self.cy], axis=-1)
        return uv, front

    def project_points_pinhole(self, P):
        """Ordinary pinhole projection (no refraction) of world points."""
        P = np.atleast_2d(np.asarray(P, float))
        return self.project_dirs(P - self.centre)

    def pixel_dirs(self, uv) -> np.ndarray:
        """Pixels (N, 2) -> unit world-frame ray directions (N, 3)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        xd = (uv[:, 0] - self.cx) / self.f
        yd = (uv[:, 1] - self.cy) / self.f
        x, y = xd.copy(), yd.copy()
        if self.k1:
            for _ in range(20):              # fixed-point undistortion
                r2 = x * x + y * y
                x, y = xd / (1 + self.k1 * r2), yd / (1 + self.k1 * r2)
        d = np.stack([x, y, np.ones_like(x)], axis=-1) @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def in_image(self, uv, margin: float = 0.0) -> np.ndarray:
        uv = np.atleast_2d(uv)
        return ((uv[:, 0] >= -0.5 + margin) & (uv[:, 0] < self.width - 0.5 - margin)
                & (uv[:, 1] >= -0.5 + margin) & (uv[:, 1] < self.height - 0.5 - margin))

    def to_dict(self) -> dict:
        return {"f_px": self.f, "cx": self.cx, "cy": self.cy, "k1": self.k1,
                "R": self.R.tolist(), "t": self.t.tolist(), "rvec": matrix_to_rotvec(self.R).tolist(),
                "width": self.width, "height": self.height, "centre_m": self.centre.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        R = np.asarray(d["R"]) if "R" in d else rotvec_to_matrix(d["rvec"])
        return cls(d["f_px"], R, d["t"], d["width"], d["height"], d.get("cx"), d.get("cy"), d.get("k1", 0.0))


# ----------------------------------------------------------------------
# flat-interface refraction
# ----------------------------------------------------------------------
def solve_tangential(h, d, nidx, tol=1e-14, maxit=60):
    """Solve sum_i d_i s / sqrt(n_i^2 - s^2) = h for s (vectorized over N points).

    h: (N,) tangential distances >= 0; d: (M, N) layer thicknesses >= 0 (layer 0 is
    the camera's medium); nidx: (M, N) refractive indices. Returns s (N,)."""
    h = np.asarray(h, float)
    d = np.asarray(d, float)
    n2 = np.asarray(nidx, float) ** 2
    D = d.sum(0)
    # smallest index among layers actually traversed bounds s
    nmin = np.min(np.where(d > 0, np.sqrt(n2), np.inf), axis=0)
    nmin = np.where(np.isfinite(nmin), nmin, 1.0)
    hi = nmin * (1 - 1e-15)
    # all-at-n_min straight-line solution is a lower bound (every n_i >= n_min)
    lo = np.where(D > 0, nmin * h / np.sqrt(h * h + D * D + 1e-300), 0.0)
    lo = np.minimum(lo, hi)
    s = lo.copy()
    for _ in range(maxit):
        q = np.maximum(n2 - s * s, 1e-300)
        fv = (d * s / np.sqrt(q)).sum(0) - h
        fp = (d * n2 / q ** 1.5).sum(0)
        lo = np.where(fv <= 0, s, lo)
        hi = np.where(fv > 0, s, hi)
        sn = s - fv / np.maximum(fp, 1e-300)
        bad = ~((sn > lo) & (sn < hi))
        sn = np.where(bad, 0.5 * (lo + hi), sn)
        done = np.max(np.abs(sn - s)) < tol if s.size else True
        s = sn
        if done:
            break
    return s


class View:
    """A calibrated photo: camera + which pane / surface it looks through.

    kind: 'top' | 'front' | 'back' | 'left' | 'right'. ``refraction=False`` gives
    the plain pinhole model (used only for the ablation study)."""

    def __init__(self, kind: str, camera: Camera, tank: TankGeometry, refraction: bool = True):
        if kind not in VIEW_KINDS:
            raise ValueError(f"view kind must be one of {VIEW_KINDS}")
        self.kind, self.camera, self.tank, self.refraction = kind, camera, tank, bool(refraction)
        T = tank
        g = T.glass
        if kind == "top":
            self.normal = np.array([0.0, 0.0, -1.0])
            self.planes = [-T.water_surface]
            self.mid_indices = []                    # no glass
        else:
            self.normal, near = {
                "front": (np.array([0.0, 1.0, 0.0]), 0.0),
                "back": (np.array([0.0, -1.0, 0.0]), -T.width),
                "left": (np.array([1.0, 0.0, 0.0]), 0.0),
                "right": (np.array([-1.0, 0.0, 0.0]), -T.length),
            }[kind]
            self.planes = [near - g, near]           # outer face, inner face (s = n . X)
            self.mid_indices = [T.n_glass]

    # -- helpers ------------------------------------------------------------
    def _indices(self, inner):
        """(M, N) index table: air, [glass], inner."""
        N = inner.shape[0]
        rows = [np.full(N, self.tank.n_air)] + [np.full(N, v) for v in self.mid_indices] + [inner]
        idx = np.vstack(rows)
        if not self.refraction:
            idx = np.ones_like(idx)
        return idx

    def _layers(self, sC, sP):
        b = [sC] + list(self.planes) + [np.inf]
        d = []
        for i in range(len(b) - 1):
            lo = np.maximum(sC, b[i])
            hi = np.minimum(sP, b[i + 1])
            d.append(np.clip(hi - lo, 0.0, None))
        return np.vstack(d)

    def _inner_index(self, z):
        if self.kind == "top":
            return np.full(z.shape, self.tank.n_water)
        return np.where(z <= self.tank.water_surface + 1e-9, self.tank.n_water, self.tank.n_air)

    def camera_ok(self) -> bool:
        """Camera on the outside of the interface (above the water / in front of the pane)."""
        return float(self.normal @ self.camera.centre) < self.planes[0]

    def _aperture_ok(self, E, submerged):
        """Is the entry point E on the inner plane inside the opening/pane, and on the
        correct side of the water surface for the point's medium?"""
        T, tol = self.tank, 1e-9
        if self.kind == "top":
            return ((E[:, 0] >= -tol) & (E[:, 0] <= T.length + tol)
                    & (E[:, 1] >= -tol) & (E[:, 1] <= T.width + tol))
        lat = E[:, 0] if self.kind in ("front", "back") else E[:, 1]
        span = T.length if self.kind in ("front", "back") else T.width
        ok = (lat >= -tol) & (lat <= span + tol) & (E[:, 2] >= -tol) & (E[:, 2] <= T.height + tol)
        side = np.where(submerged, E[:, 2] <= T.water_surface + 1e-6, E[:, 2] >= T.water_surface - 1e-6)
        return ok & side

    # -- forward projection ---------------------------------------------------
    def project(self, P, return_info: bool = False):
        """3D tank points (N, 3) -> pixels (N, 2) and valid (N,) through the interface.

        ``valid`` means: camera in front, the layered light path is physical (see
        module docstring). It does NOT test image bounds (use camera.in_image)."""
        P = np.atleast_2d(np.asarray(P, float))
        C = self.camera.centre
        n = self.normal
        sC = float(n @ C)
        sP = P @ n
        off = P - C
        along = off @ n
        tang = off - along[:, None] * n
        h = np.linalg.norm(tang, axis=1)
        et = tang / np.where(h > 1e-15, h, 1.0)[:, None]
        inner = self._inner_index(P[:, 2])
        idx = self._indices(inner)
        d = self._layers(sC, sP)
        s = solve_tangential(h, d, idx)
        n0 = idx[0]
        sin0 = s / n0
        dir0 = sin0[:, None] * et + np.sqrt(np.maximum(1 - sin0 ** 2, 0))[:, None] * n
        uv, front = self.camera.project_dirs(dir0)
        valid = front & (sC < self.planes[0]) & (sP >= sC)
        # entry point on the last plane, for points beyond it
        beyond = sP >= self.planes[-1] - 1e-12
        tan_i = s[None, :] / np.sqrt(np.maximum(idx ** 2 - s[None, :] ** 2, 1e-300))
        lat_to_last = (d[:-1] * tan_i[:-1]).sum(0)
        E = C + (self.planes[-1] - sC) * n + lat_to_last[:, None] * et
        submerged = P[:, 2] <= self.tank.water_surface + 1e-9
        T = self.tank
        inside = ((P[:, 0] >= -1e-9) & (P[:, 0] <= T.length + 1e-9) & (P[:, 1] >= -1e-9)
                  & (P[:, 1] <= T.width + 1e-9))
        # beyond the interface the point must be inside the tank, so that (box being
        # convex) the straight segment entry -> point never crosses another pane
        valid &= np.where(beyond, self._aperture_ok(E, submerged) & inside, True)
        if not return_info:
            return uv, valid
        sin_in = s / idx[-1]
        dir_in = sin_in[:, None] * et + np.sqrt(np.maximum(1 - sin_in ** 2, 0))[:, None] * n
        cos_i = np.sqrt(np.maximum(1 - (s[None, :] / idx) ** 2, 1e-300))
        seg_len = d / cos_i
        water_path = np.where(submerged & beyond, seg_len[-1], 0.0)
        return uv, valid, {"entry": E, "dir_inner": dir_in, "s": s, "water_path": water_path,
                           "beyond": beyond}

    # -- back-projection --------------------------------------------------------
    def back_project(self, uv) -> dict:
        """Pixels (N, 2) -> refracted rays in the tank.

        Returns dict: camera_centre (3,), dir_air (N, 3), entry (N, 3) point on the
        last plane (water surface for top views, inner glass face for side views),
        dir (N, 3) unit direction beyond it, n_inner (N,), valid (N,)."""
        C = self.camera.centre
        n = self.normal
        dirs = self.camera.pixel_dirs(uv)
        cos0 = dirs @ n
        ok = cos0 > 1e-9
        N = len(dirs)
        n_air = self.tank.n_air
        s = np.sqrt(np.maximum(1 - cos0 ** 2, 0)) * n_air
        tang = dirs - cos0[:, None] * n
        tn = np.linalg.norm(tang, axis=1)
        et = tang / np.where(tn > 1e-15, tn, 1.0)[:, None]
        X = np.repeat(C[None, :], N, axis=0)
        layer_n = [n_air] + list(self.mid_indices)
        if not self.refraction:
            layer_n = [1.0] * len(layer_n)
        for p, ni in zip(self.planes, layer_n):
            sin_i = s / ni
            cos_i = np.sqrt(np.maximum(1 - sin_i ** 2, 1e-300))
            dist = p - X @ n
            X = X + (dist / cos_i)[:, None] * (sin_i[:, None] * et + cos_i[:, None] * n)
        inner = self._inner_index(X[:, 2])
        if not self.refraction:
            inner = np.ones(N)
        sin_in = s / inner
        dir_in = sin_in[:, None] * et + np.sqrt(np.maximum(1 - sin_in ** 2, 0))[:, None] * n
        submerged = inner == self.tank.n_water if self.refraction else X[:, 2] <= self.tank.water_surface
        valid = ok & self._aperture_ok(X, submerged) & self.camera_ok()
        return {"camera_centre": C, "dir_air": dirs, "entry": X, "dir": dir_in,
                "n_inner": inner, "valid": valid}

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.kind, "refraction": self.refraction, "camera": self.camera.to_dict()}

    @classmethod
    def from_dict(cls, d: dict, tank: TankGeometry) -> "View":
        return cls(d["kind"], Camera.from_dict(d["camera"]), tank, d.get("refraction", True))


def ray_point_distance(origin, direction, P) -> np.ndarray:
    """Distance from points P (N, 3) to rays origin + t direction (N, 3), t >= 0."""
    o = np.atleast_2d(origin)
    dvec = np.atleast_2d(direction)
    w = np.atleast_2d(P) - o
    t = np.maximum((w * dvec).sum(1), 0)
    return np.linalg.norm(w - t[:, None] * dvec, axis=1)
