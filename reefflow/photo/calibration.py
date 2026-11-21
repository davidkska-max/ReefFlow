"""Camera calibration from clicked tank points: intrinsics + full pose by PnP,
with the flat-interface refraction model in the loop (PLAN C9, DECISIONS D-601).

Click input (JSON-able, what a UI sends)
----------------------------------------
    clicks = {"top_FL": [u, v], "water_BR": [u, v],           # named reference points
              "seam_BL": [u, v], ...}                         # and seams (any point on that
                                                              # vertical corner line)
    custom = [{"xyz": [x, y, z], "uv": [u, v], "name": "..."}]  # any other known point (m)
    edges  = [{"a": "sand_BL", "b": "rim_BL", "uv": [u, v]}]    # a click anywhere on the
                                                                # straight tank edge a-b

Named points: see :mod:`reefflow.photo.tankref` (``recommended_points(kind)``).

Unknowns: rotation (3), translation (3), focal length f (1) -- unless the user
supplies f -- and optionally radial k1 (1). Principal point fixed at the image
centre, square pixels (ASSUMPTIONS A-601).

Solve
-----
1. Initial pose: OpenCV SQPnP (pinhole) for a log-spaced sweep of f; keep the f
   with the smallest pinhole reprojection error.
2. Pinhole refinement of (R, t, f) by Levenberg-Marquardt.
3. Refinement with the refractive projection (camera.View.project), which is
   the model used for carving. Residuals are reported from this model.

Reported: per-point residuals (px), RMS and max, parameter standard deviations
from the Gauss-Newton covariance  s^2 (J^T J)^-1  with s^2 = SSR/(m - p), and
the propagated 1-sigma camera-centre uncertainty. Requirements: >= 4 points
and >= 1 more equation than unknowns; a coplanar point set with free f is
refused (focal length and distance are then not separable -- PLAN C9).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera, View, matrix_to_rotvec, rotvec_to_matrix
from .tankref import TankGeometry, VIEW_KINDS, is_seam


class CalibrationError(ValueError):
    pass


def focal_px_from_35mm(f35_mm: float, width: int, height: int) -> float:
    """Focal length in pixels from the EXIF 35-mm-equivalent focal length
    (diagonal convention: full-frame diagonal 43.27 mm)."""
    return float(f35_mm) * np.hypot(width, height) / 43.267


@dataclass
class CalibrationResult:
    view: View
    names: list
    xyz: np.ndarray
    uv_clicked: np.ndarray
    uv_model: np.ndarray
    residual_px: np.ndarray            # (N, 2) model - clicked
    edge_residual_px: np.ndarray       # (E,) signed-free distance to the projected edge
    rms_px: float
    max_px: float
    param_std: dict
    centre_std_mm: list
    warnings: list = field(default_factory=list)
    pinhole_rms_px: float | None = None   # same points, refraction ignored (diagnostic)
    edge_names: list = field(default_factory=list)
    _solver: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "view": self.view.to_dict(),
            "points": [{"name": n, "xyz_m": x.tolist(), "uv_clicked": c.tolist(), "uv_model": m.tolist(),
                        "residual_px": float(np.linalg.norm(r))}
                       for n, x, c, m, r in zip(self.names, self.xyz, self.uv_clicked, self.uv_model,
                                                self.residual_px)],
            "edge_residuals_px": self.edge_residual_px.tolist(),
            "seams": [{"name": n, "residual_px": float(abs(r))} for n, r in zip(self.edge_names, self.edge_residual_px)],
            "rms_px": self.rms_px, "max_px": self.max_px,
            "param_std": self.param_std, "camera_centre_std_mm": self.centre_std_mm,
            "pinhole_only_rms_px": self.pinhole_rms_px, "warnings": self.warnings,
        }


def _split_clicks(clicks):
    """Named clicks -> (point clicks, seam clicks)."""
    pts, seams = {}, {}
    for k, v in (clicks or {}).items():
        (seams if is_seam(k) else pts)[k] = v
    return pts, seams


def _model(tank: TankGeometry, point_names, custom, seam_names, edges):
    """3-D model of every observation for this tank: X (N, 3) and edges [(A, B)]."""
    X = [tank.point(k) for k in point_names] + [np.asarray(c["xyz"], float) for c in custom]
    E = [tank.seam(k) for k in seam_names]
    for e in edges:
        E.append(tuple(tank.point(e[k]) if isinstance(e[k], str) else np.asarray(e[k], float) for k in ("a", "b")))
    return np.asarray(X, float).reshape(-1, 3), E


def _point_to_polyline(p, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


_T41 = np.linspace(0, 1, 41)[:, None]


def _edge_residuals(view, E, uv_e):
    """Distance (px) from each click to its projected edge. The edge is curved slightly
    by refraction, so it is sampled at 41 points, then again at 41 points around the
    nearest sample (chord error ~ h^2: about 400x smaller)."""
    r = []
    for (A, B), c in zip(E, uv_e):
        u_e, ok = view.project(A + _T41 * (B - A))
        if ok.sum() < 2:
            r.append(1e3)
            continue
        t = _T41[:, 0]
        i = int(np.argmin(np.where(ok, np.linalg.norm(u_e - c, axis=1), np.inf)))
        t0, t1 = t[max(i - 1, 0)], t[min(i + 1, len(t) - 1)]
        tt = np.linspace(t0, t1, 41)[:, None]
        u2, ok2 = view.project(A + tt * (B - A))
        r.append(_point_to_polyline(c, u2[ok2]) if ok2.sum() >= 2 else _point_to_polyline(c, u_e[ok]))
    return r


def calibrate_view(kind: str, image_size, tank: TankGeometry, clicks: dict | None = None,
                   custom: list | None = None, edges: list | None = None, f_px: float | None = None,
                   f_guess: float | None = None, fit_k1: bool = False,
                   refraction: bool = True, f_prior: tuple | None = None) -> CalibrationResult:
    """Solve one camera from clicked points.

    image_size: (width, height) px. clicks: named points and seams (names in
    :func:`reefflow.photo.tankref.all_names`; a seam click is a point anywhere on
    that vertical corner line). f_px: known focal length in pixels (e.g. from EXIF
    via :func:`focal_px_from_35mm`) -> held fixed. f_guess: optional start value.
    f_prior: (f0_px, sigma_log) weak prior on a free focal length (D-838): one extra
    residual (ln f - ln f0) / sigma_log, in px units, so one sigma costs as much as a
    1 px click error. refraction=False calibrates a plain pinhole (ablation only)."""
    import cv2
    from scipy.optimize import least_squares

    if kind not in VIEW_KINDS:
        raise CalibrationError(f"view kind must be one of {VIEW_KINDS}")
    W, H = (int(v) for v in image_size)
    pclicks, sclicks = _split_clicks(clicks)
    custom, edges = list(custom or []), list(edges or [])
    point_names = list(pclicks) + [c.get("name", f"custom_{i}") for i, c in enumerate(custom)]
    names = point_names
    uv = np.asarray(list(pclicks.values()) + [c["uv"] for c in custom], float).reshape(-1, 2)
    seam_names = list(sclicks)
    edge_names = seam_names + [e.get("name", f"{e['a']}-{e['b']}" if isinstance(e["a"], str) else f"edge_{i}")
                               for i, e in enumerate(edges)]
    uv_e = [np.asarray(v, float) for v in sclicks.values()] + [np.asarray(e["uv"], float) for e in edges]
    X, E = _model(tank, list(pclicks), custom, seam_names, edges)
    N = len(X)
    warnings = []
    if N < 4:
        raise CalibrationError(f"need at least 4 clicked corner points (seams do not count), got {N}")
    use_prior = bool(f_prior) and not f_px
    n_unknown = 6 + (0 if f_px else 1) + (1 if fit_k1 else 0)
    m = 2 * N + len(E) + (1 if use_prior else 0)
    if m <= n_unknown:
        raise CalibrationError(f"{m} equations for {n_unknown} unknowns: click more points")
    # Edge clicks constrain points off the plane too, so they count for this test
    # (the clicked point lies somewhere on segment a-b; both ends are used).
    Xall = np.vstack([X] + [np.vstack([A, B]) for A, B in E]) if E else X
    sv = np.linalg.svd(Xall - Xall.mean(0), compute_uv=False)
    coplanar = sv[2] < 1e-3 * sv[0]
    if coplanar and not f_px:
        raise CalibrationError("all clicked points lie in one plane: focal length and camera distance "
                               "cannot be separated. Add points at another depth (the back corners at the "
                               "water line, or a back corner seam), or enter the focal length.")
    if N + len(E) < 6:
        warnings.append(f"only {N + len(E)} clicks: residuals have few degrees of freedom; 6-8 recommended")

    # --- 1. initial pose: SQPnP over an f sweep (pinhole) -----------------
    # Scored on points and edges: four coplanar corners fit any f exactly, the
    # seams (at another depth) then pick the right one.
    cxy = np.array([(W - 1) / 2.0, (H - 1) / 2.0])
    side = max(W, H)
    f_cands = [f_px] if f_px else ([f_guess] if f_guess else []) + list(np.geomspace(0.3, 6.0, 31) * side)
    if use_prior:
        f_cands = [f_prior[0]] + f_cands
    best = None
    for f in f_cands:
        K = np.array([[f, 0, cxy[0]], [0, f, cxy[1]], [0, 0, 1.0]])
        try:
            ok, rv, tv = cv2.solvePnP(X.astype(np.float64), uv.astype(np.float64), K, None,
                                      flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if not ok:
            continue
        Rm = rotvec_to_matrix(rv.ravel())
        if np.any((X @ Rm.T + tv.ravel())[:, 2] <= 0):
            continue
        proj, _ = cv2.projectPoints(X, rv, tv, K, None)
        sq = list(np.sum((proj.reshape(-1, 2) - uv) ** 2, 1))
        for (A, B), c in zip(E, uv_e):
            P = A + _T41 * (B - A)
            if np.any((P @ Rm.T + tv.ravel())[:, 2] <= 0):
                sq.append(1e6); continue
            pe, _ = cv2.projectPoints(P, rv, tv, K, None)
            sq.append(_point_to_polyline(c, pe.reshape(-1, 2)) ** 2)
        if use_prior:
            sq.append((np.log(f / f_prior[0]) / f_prior[1]) ** 2)
        err = float(np.sqrt(np.mean(sq)))
        if best is None or err < best[0]:
            best = (err, f, rv.ravel(), tv.ravel())
    if best is None:
        raise CalibrationError("PnP initialisation failed: check the point names / click order")
    _, f0, rv0, tv0 = best

    def unpack(p):
        rv, tv = p[:3], p[3:6]
        i = 6
        f = f_px if f_px else np.exp(p[i]); i += 0 if f_px else 1
        k1 = p[i] if fit_k1 else 0.0
        return Camera(f, rotvec_to_matrix(rv), tv, W, H, k1=k1)

    def residuals(p, refr):
        cam = unpack(p)
        v = View(kind, cam, tank, refraction=refr)
        u_m, _ = v.project(X)
        r = [(u_m - uv).ravel(), _edge_residuals(v, E, uv_e)]
        if use_prior:
            r.append([np.log(cam.f / f_prior[0]) / f_prior[1]])
        return np.concatenate([np.ravel(x) for x in r])

    p0 = np.concatenate([rv0, tv0, [] if f_px else [np.log(f0)], [0.0] if fit_k1 else []])
    # --- 2. pinhole refinement ----------------------------------------------
    sol = least_squares(residuals, p0, args=(False,), method="lm", x_scale="jac", max_nfev=2000)
    pin_rms = float(np.sqrt(np.mean(sol.fun[:2 * N].reshape(-1, 2) ** 2 @ np.ones(2))))
    p1 = sol.x
    # --- 3. refractive refinement -------------------------------------------
    if refraction:
        sol = least_squares(residuals, p1, args=(True,), method="lm", x_scale="jac", max_nfev=4000)
    cam = unpack(sol.x)
    view = View(kind, cam, tank, refraction=refraction)
    if not view.camera_ok():
        raise CalibrationError("solved camera lies inside the tank / below the interface: check the clicks")
    u_m, valid = view.project(X)
    res = u_m - uv
    rn = np.linalg.norm(res, axis=1)
    if not np.all(valid):
        warnings.append("points with a non-physical light path in the solved model: "
                        + ", ".join(n for n, ok in zip(names, valid) if not ok))
    e_res = sol.fun[2 * N:2 * N + len(E)]
    allr = np.concatenate([rn, np.abs(e_res)])
    rms = float(np.sqrt(np.mean(allr ** 2)))

    # --- covariance ---------------------------------------------------------
    J = sol.jac
    dof = m - len(sol.x)
    s2 = float(np.sum(sol.fun ** 2) / dof) if dof > 0 else float("nan")
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((len(sol.x), len(sol.x)), np.nan)
        warnings.append("singular normal matrix: pose poorly determined")
    sd = np.sqrt(np.abs(np.diag(cov)))
    pstd = {"rvec_rad": sd[:3].tolist(), "t_m": sd[3:6].tolist()}
    if not f_px:
        pstd["f_px"] = float(cam.f * sd[6])                      # d f = f d(log f)
    if fit_k1:
        pstd["k1"] = float(sd[-1])

    def centre(p):
        return unpack(p).centre
    Jc = np.zeros((3, len(sol.x)))
    for j in range(len(sol.x)):
        h = 1e-6 * max(1.0, abs(sol.x[j]))
        dp = np.zeros_like(sol.x); dp[j] = h
        Jc[:, j] = (centre(sol.x + dp) - centre(sol.x - dp)) / (2 * h)
    c_std = np.sqrt(np.abs(np.diag(Jc @ cov @ Jc.T))) * 1e3
    if rms > 5.0:
        warnings.append(f"RMS reprojection residual {rms:.1f} px is large: a click may be on the wrong "
                        "corner, the tank size may be wrong, or the photo is strongly distorted")
    out = CalibrationResult(view=view, names=names, xyz=X, uv_clicked=uv, uv_model=u_m, residual_px=res,
                            edge_residual_px=np.asarray(e_res), rms_px=rms, max_px=float(allr.max()),
                            param_std=pstd, centre_std_mm=c_std.tolist(), warnings=warnings,
                            pinhole_rms_px=pin_rms, edge_names=edge_names)
    out._solver = dict(kind=kind, image_size=(W, H), point_names=list(pclicks), custom=custom,
                       seam_names=seam_names, edges=edges, uv=uv, uv_e=uv_e, f_px=f_px,
                       f_prior=f_prior if use_prior else None, x=sol.x, refraction=refraction)
    return out


def check_tank_size(cal: CalibrationResult, tank: TankGeometry, min_gain: float = 0.35,
                    min_rel_change: float = 0.02) -> dict:
    """Does the photo fit a differently sized tank much better? (D-837)

    Refits the camera together with the tank's height and width relative to its
    length (a photo has no absolute scale, so the length is held). The water level
    keeps its distance below the rim and the sand keeps its depth. Returns
    {rms_px, height_m, width_m, height_std_m, width_std_m, external_m: [L, W, H],
    suggest: bool}. ``suggest`` is True when the RMS falls by more than ``min_gain``
    (fraction) and a dimension moves by more than ``min_rel_change`` and > 3 sigma."""
    from scipy.optimize import least_squares
    S = cal._solver
    kind, (W, H) = S["kind"], S["image_size"]
    f_px, prior = S["f_px"], S["f_prior"]
    uv, uv_e = S["uv"], S["uv_e"]
    below_rim = tank.height - tank.water_surface

    def tank_for(lh, lw):
        h, w = tank.height * np.exp(lh), tank.width * np.exp(lw)
        return TankGeometry(length=tank.length, width=w, height=h, glass=tank.glass,
                            water_surface=max(h - below_rim, tank.sand_depth + 1e-3), sand_depth=tank.sand_depth,
                            n_water=tank.n_water, n_glass=tank.n_glass, n_air=tank.n_air)

    nfix = len(S["x"])

    def residuals(p):
        T = tank_for(p[nfix], p[nfix + 1])
        X, E = _model(T, S["point_names"], S["custom"], S["seam_names"], S["edges"])
        rv, tv = p[:3], p[3:6]
        f = f_px if f_px else np.exp(p[6])
        v = View(kind, Camera(f, rotvec_to_matrix(rv), tv, W, H), T, refraction=S["refraction"])
        u_m, _ = v.project(X)
        r = [(u_m - uv).ravel(), _edge_residuals(v, E, uv_e)]
        if prior:
            r.append([np.log(f / prior[0]) / prior[1]])
        return np.concatenate([np.ravel(x) for x in r])

    p0 = np.concatenate([S["x"], [0.0, 0.0]])
    try:
        sol = least_squares(residuals, p0, method="lm", x_scale="jac", max_nfev=1500)
    except Exception as e:                      # noqa: BLE001 - diagnostic only
        return {"error": str(e), "suggest": False}
    n_obs = 2 * len(uv) + len(uv_e)
    r = sol.fun[:n_obs]
    pts = np.linalg.norm(r[:2 * len(uv)].reshape(-1, 2), axis=1)
    rms = float(np.sqrt(np.mean(np.concatenate([pts, np.abs(r[2 * len(uv):])]) ** 2)))
    dof = len(sol.fun) - len(sol.x)
    try:
        cov = (np.sum(sol.fun ** 2) / max(dof, 1)) * np.linalg.inv(sol.jac.T @ sol.jac)
        sd = np.sqrt(np.abs(np.diag(cov)))[-2:]
    except np.linalg.LinAlgError:
        sd = np.array([np.inf, np.inf])
    T = tank_for(sol.x[-2], sol.x[-1])
    g = tank.glass
    dh, dw = sol.x[-2], sol.x[-1]
    moved = [abs(d) > min_rel_change and abs(d) > 3 * s for d, s in ((dh, sd[0]), (dw, sd[1]))]
    gain = 1 - rms / max(cal.rms_px, 1e-9)
    return {"rms_px": rms, "rms_now_px": cal.rms_px, "height_m": T.height, "width_m": T.width,
            "height_std_m": float(T.height * sd[0]), "width_std_m": float(T.width * sd[1]),
            "external_m": [tank.length + 2 * g, T.width + 2 * g, T.height + g],
            "external_now_m": [tank.length + 2 * g, tank.width + 2 * g, tank.height + g],
            "height_changed": bool(moved[0]), "width_changed": bool(moved[1]),
            "suggest": bool(gain > min_gain and any(moved))}
