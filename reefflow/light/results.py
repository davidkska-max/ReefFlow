"""Lighting results: tallies -> PAR fields in physical units, noise, DLI, and the CFD-grid interface.

Quantities (PLAN C6)
    meter-equivalent PAR   downwelling planar photon irradiance E_d on a horizontal,
                           upward-facing, cosine-corrected sensor (what a quantum PAR
                           meter reads, and the basis of all hobby/manufacturer PAR
                           targets). Estimated at node (i, j, k) from the photons
                           crossing the horizontal dx x dx square centred on the node
                           downward, divided by the part of that square that is water
                           (rock-occupied part removed using the SDF).
    surface-normal PAR     photon irradiance on the actual rock / sand surface
                           (incident photons per unit true surface area); differs from
                           the meter value on sloped or shaded faces.
Both are in umol m^-2 s^-1 at the configured (100 % schedule) intensities; DLI
(mol m^-2 d^-1) multiplies by the photoperiod's full-intensity seconds x 1e-6.

Statistical noise: for a tally sum S = sum(w) with sum of squares Q = sum(w^2),
the relative standard error is sqrt(Q)/S (= 1/sqrt(hits) for equal weights).
95 % confidence interval: E (1 +- 1.96 rel).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import LightingConfig
from .sdf import SDF
from .tracer import Grid, Tallies

MIN_FLUID_FRACTION = 0.25            # meter cells with less water area than this are not reported
MIN_PATCH_AREA = 0.1                 # rock patches with area < this x dx^2 are not reported


def rel_error(w, w2):
    w = np.asarray(w, float); w2 = np.asarray(w2, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, np.sqrt(w2) / np.where(w > 0, w, 1), np.inf)


def water_area_fraction(sdf: SDF | None, grid: Grid, sub: int = 4) -> np.ndarray:
    """Water fraction of the horizontal dx x dx square centred on each node (at the node's
    height), from sub x sub samples of the SDF. 1 where no rock is near."""
    nx, ny, nz = grid.shape
    frac = np.ones(grid.shape)
    if sdf is None:
        return frac
    X, Y, Z = (grid.coords(a) for a in range(3))
    P = np.stack(np.meshgrid(X, Y, Z, indexing="ij"), -1)
    phi = sdf.sample(P)
    frac[phi < -0.75 * grid.dx] = 0.0
    near = np.abs(phi) <= 0.75 * grid.dx
    idx = np.nonzero(near)
    if len(idx[0]):
        off = (np.arange(sub) + 0.5) / sub - 0.5
        ox, oy = np.meshgrid(off, off, indexing="ij")
        offs = np.stack([ox.ravel(), oy.ravel(), np.zeros(sub * sub)], -1) * grid.dx
        Pc = P[idx]
        f = np.zeros(len(Pc))
        for s0 in range(0, len(Pc), 200_000):
            Q = Pc[s0:s0 + 200_000, None, :] + offs[None]
            f[s0:s0 + 200_000] = (sdf.sample(Q) > 0).mean(1)
        frac[idx] = f
    return frac


@dataclass
class LightResult:
    cfg: LightingConfig
    tank: object                     # reefflow.project.Tank
    grid: Grid
    tallies: Tallies
    sdf: SDF | None
    patches: dict | None
    scale: float = 1.0               # calibration scale factor
    calibration: dict | None = None
    notes: list = field(default_factory=list)

    def __post_init__(self):
        g, t, tal = self.grid, self.tank, self.tallies
        k = tal.ppf_per_photon * self.scale
        dx2 = g.dx ** 2
        Z = g.coords(2)
        zs, zsand = t.water_surface, t.sand_depth
        # ---- meter-equivalent PAR at every node --------------------------------
        self.fluid_fraction = water_area_fraction(self.sdf, g)
        if tal.stack_w is not None:
            valid = (self.fluid_fraction >= MIN_FLUID_FRACTION) & ((Z > zsand) & (Z < zs))[None, None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                self.meter = np.where(valid, tal.stack_w * k / (dx2 * self.fluid_fraction), np.nan)
            # stack noise: Poisson in the (unsplit, 0.9-1 weight) photon count, D-410
            with np.errstate(invalid="ignore", divide="ignore"):
                self.meter_rel = np.where(valid, 1.0 / np.sqrt(tal.stack_w), np.nan)
        else:
            self.meter = self.meter_rel = None
        # ---- rock patches -------------------------------------------------------
        if self.patches is not None and tal.rock_w is not None:
            A = self.patches["area"]
            ok = A >= MIN_PATCH_AREA * dx2
            with np.errstate(invalid="ignore", divide="ignore"):
                self.rock_normal = np.where(ok, tal.rock_w * k / np.where(ok, A, 1), np.nan)
            self.rock_rel = np.where(ok, rel_error(tal.rock_w, tal.rock_w2), np.nan)
            self.rock_valid = ok
            self.rock_meter = self._patch_meter(ok)
        else:
            self.rock_normal = self.rock_rel = self.rock_meter = None
            self.rock_valid = None
        # ---- sand ---------------------------------------------------------------
        if tal.sand_w is not None:
            sp = np.stack(np.meshgrid(g.coords(0), g.coords(1), [zsand + 1e-4], indexing="ij"), -1)[:, :, 0]
            if self.sdf is not None:
                off = (np.arange(4) + 0.5) / 4 - 0.5
                fr = np.zeros(sp.shape[:2])
                for ox in off:
                    for oy in off:
                        fr += (self.sdf.sample(sp + np.array([ox, oy, 0]) * g.dx) > 0)
                fr /= 16
            else:
                fr = np.ones(sp.shape[:2])
            X, Y = g.coords(0), g.coords(1)
            inside = ((X > 0) & (X < t.length))[:, None] & ((Y > 0) & (Y < t.width))[None, :]
            ok = (fr >= MIN_FLUID_FRACTION) & inside
            self.sand_fraction = fr
            with np.errstate(invalid="ignore", divide="ignore"):
                self.sand = np.where(ok, tal.sand_w * k / (dx2 * np.where(ok, fr, 1)), np.nan)
            self.sand_rel = np.where(ok, rel_error(tal.sand_w, tal.sand_w2), np.nan)
        else:
            self.sand = self.sand_rel = None
        # ---- plan maps ----------------------------------------------------------
        if len(self.cfg.map_depths) and tal.plan_w.size:
            px = self.cfg.map_resolution
            self.plan = tal.plan_w * k / px ** 2
            self.plan_rel = rel_error(tal.plan_w, tal.plan_w2)
            nx, ny = self.plan.shape[1:]
            self.plan_x = (np.arange(nx) + 0.5) * px
            self.plan_y = (np.arange(ny) + 0.5) * px
            # mask pixels whose centre is inside rock or outside the tank
            if self.sdf is not None:
                for l, d in enumerate(self.cfg.map_depths):
                    P = np.stack(np.meshgrid(self.plan_x, self.plan_y, [zs - d], indexing="ij"), -1)[:, :, 0]
                    inrock = self.sdf.sample(P) <= 0
                    self.plan[l][inrock] = np.nan
                    self.plan_rel[l][inrock] = np.nan
            out = (self.plan_x[:, None] > t.length) | (self.plan_y[None, :] > t.width)
            self.plan[:, out] = np.nan
        else:
            self.plan = self.plan_rel = None

    # ------------------------------------------------------------------
    def _patch_meter(self, ok) -> np.ndarray:
        """Meter-equivalent PAR at each rock patch: the node-plane value one cell off the
        surface along the patch's mean normal (two cells if that node is not reliable)."""
        out = np.full(self.grid.shape, np.nan)
        if self.meter is None:
            return out
        idx = np.nonzero(ok)
        c = self.patches["centroid"][idx]
        n = self.patches["normal"][idx]
        vals = np.full(len(c), np.nan)
        for step in (1.0, 2.0):
            todo = np.isnan(vals)
            q = c[todo] + n[todo] * step * self.grid.dx
            ijk = np.floor((q - self.grid.origin) / self.grid.dx + 0.5).astype(int)
            inb = np.all((ijk >= 0) & (ijk < np.array(self.grid.shape)), 1)
            v = np.full(len(q), np.nan)
            ii = ijk[inb]
            v[inb] = self.meter[ii[:, 0], ii[:, 1], ii[:, 2]]
            vals[todo] = v
        out[idx] = vals
        return out

    # ------------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.calibration is not None and self.calibration.get("ok", False)

    @property
    def label(self) -> str:
        if self.calibrated:
            return f"calibrated (scale {self.scale:.3f}, RMS residual {100 * self.calibration['rms_rel']:.1f} %)"
        return "relative/estimated (uncalibrated)"

    def dli(self, par):
        return np.asarray(par, float) * self.cfg.photoperiod.dli_factor

    def section(self, y: float | None = None):
        """Front-elevation section of meter-equivalent PAR: (E (nx, nz), x, z, y_used)."""
        y = self.cfg.section_y if y is None else y
        y = 0.5 * self.tank.width if y is None else y
        j = int(np.clip(np.round((y - self.grid.origin[1]) / self.grid.dx), 1, self.grid.shape[1] - 2))
        return self.meter[:, j, :], self.grid.coords(0), self.grid.coords(2), float(self.grid.coords(1)[j])

    def rock_top_mask(self, nz_min: float = 0.5) -> np.ndarray:
        """Valid rock patches facing up (mean normal z-component >= nz_min, i.e. within 60 deg)."""
        if self.rock_valid is None:
            return None
        return self.rock_valid & (self.patches["normal"][..., 2] >= nz_min)

    def noise_summary(self) -> dict:
        """Relative 1-sigma noise of an average-lit surface patch (rock + sand patches whose
        irradiance is within +-20 % of the mean), and its spread."""
        return surface_noise(self)

    def summary(self) -> dict:
        from .analysis import uniformity
        out = {"label": self.label, "rays": self.tallies.photons, "seconds": self.tallies.seconds,
               "rays_per_second": self.tallies.rays_per_second, "backend": self.tallies.arch,
               "total_ppf_umol_s": self.cfg.total_ppf * self.scale, "scale": self.scale,
               "dli_factor_mol_per_umol_s": self.cfg.photoperiod.dli_factor,
               "energy_balance": self.tallies.energy_balance(), "noise": self.noise_summary(),
               "grid_dx_m": self.grid.dx, "notes": list(self.notes)}
        if self.sand is not None:
            out["sand"] = uniformity(self.sand)
        if self.rock_normal is not None:
            top = self.rock_top_mask()
            out["rock_tops_meter"] = uniformity(np.where(top, self.rock_meter, np.nan))
            out["rock_tops_normal"] = uniformity(np.where(top, self.rock_normal, np.nan))
            out["rock_all_normal"] = uniformity(self.rock_normal)
        if self.plan is not None:
            out["plan"] = [dict(depth_m=d, **uniformity(self.plan[l])) for l, d in enumerate(self.cfg.map_depths)]
        if self.calibration is not None:
            out["calibration"] = self.calibration
        return out


def surface_noise(res) -> dict:
    vals, rels = [], []
    if res.sand is not None:
        m = np.isfinite(res.sand)
        vals.append(res.sand[m]); rels.append(res.sand_rel[m])
    if res.rock_normal is not None:
        m = np.isfinite(res.rock_normal) & (res.patches["area"] >= 0.5 * res.grid.dx ** 2)
        vals.append(res.rock_normal[m]); rels.append(res.rock_rel[m])
    if not vals:
        return {}
    v = np.concatenate(vals); r = np.concatenate(rels)
    ok = np.isfinite(r) & (v > 0)
    v, r = v[ok], r[ok]
    if not len(v):
        return {}
    mean = v.mean()
    band = (v > 0.8 * mean) & (v < 1.2 * mean)
    return {"average_lit_patch_rel": float(np.median(r[band])) if band.any() else float("nan"),
            "patches_in_band": int(band.sum()), "median_rel_all": float(np.median(r)),
            "p90_rel_all": float(np.percentile(r, 90)), "mean_irradiance": float(mean),
            "patch_size_m": float(res.grid.dx)}


def noise_function(grid: Grid, patches: dict | None, sand_fraction=None):
    """Callable(tallies) -> average-lit-patch relative noise, for target-noise tracing."""
    dx2 = grid.dx ** 2

    def fn(tal: Tallies) -> float:
        vals, rels = [], []
        if tal.sand_w is not None:
            w = tal.sand_w[1:-1, 1:-1]
            m = w > 0
            vals.append(w[m] / dx2); rels.append(rel_error(w, tal.sand_w2[1:-1, 1:-1])[m])
        if patches is not None and tal.rock_w is not None:
            A = patches["area"]
            m = (A >= 0.5 * dx2) & (tal.rock_w > 0)
            vals.append(tal.rock_w[m] / A[m]); rels.append(rel_error(tal.rock_w, tal.rock_w2)[m])
        v = np.concatenate(vals) if vals else np.zeros(0)
        r = np.concatenate(rels) if rels else np.zeros(0)
        if not len(v):
            return np.inf
        band = (v > 0.8 * v.mean()) & (v < 1.2 * v.mean())
        return float(np.median(r[band])) if band.any() else np.inf
    return fn


# ----------------------------------------------------------------------
# CFD-grid interface (coral scoring)
# ----------------------------------------------------------------------
@dataclass
class CfdPar:
    """PAR on the CFD node grid of a TankModel. Arrays are (nx, ny, nz), aligned with
    ``model.origin`` / ``model.dx`` / ``model.node_coords``; NaN where not evaluated.

    meter            meter-equivalent PAR (umol m^-2 s^-1) at every fluid node (PLAN C6; the
                     coral-scoring basis). At surface-adjacent nodes whose water fraction is
                     below 0.5 the value is taken one/two nodes outward along the rock normal.
    normal           surface-normal PAR at fluid nodes face-adjacent to rock or sand: local
                     area-weighted mean over the rock patches in the node's 3x3x3 neighbourhood
                     (rock), or the sand-cell value (sand).
    meter_rel, normal_rel   relative 1-sigma statistical noise of the two.
    surface_adjacent fluid nodes face-adjacent to a rock or sand node.
    adjacent_tag     TAGS value of the adjacent solid (rock wins over sand); 0 elsewhere.
    dli_meter, dli_normal   DLI (mol m^-2 d^-1) from the photoperiod.
    label            'relative/estimated (uncalibrated)' or 'calibrated (...)'.
    """
    meter: np.ndarray
    normal: np.ndarray
    meter_rel: np.ndarray
    normal_rel: np.ndarray
    surface_adjacent: np.ndarray
    adjacent_tag: np.ndarray
    dli_meter: np.ndarray
    dli_normal: np.ndarray
    label: str
    calibrated: bool


def par_on_cfd_grid(model, result: LightResult) -> CfdPar:
    """Map a LightResult computed on ``model``'s node grid onto its fluid nodes."""
    from ..tank import TAGS
    g = result.grid
    if tuple(g.shape) != tuple(model.shape) or abs(g.dx - model.dx) > 1e-12 or \
            not np.allclose(g.origin, model.origin, atol=1e-9):
        raise ValueError("lighting result was not computed on this TankModel's grid; "
                         "run run_lighting(project, model=model)")
    fluid = model.fluid_mask
    rock = model.tags == TAGS["rock"]
    sand = model.tags == TAGS["sand"]

    def adj(mask):
        a = np.zeros_like(mask)
        a[1:] |= mask[:-1]; a[:-1] |= mask[1:]
        a[:, 1:] |= mask[:, :-1]; a[:, :-1] |= mask[:, 1:]
        a[:, :, 1:] |= mask[:, :, :-1]; a[:, :, :-1] |= mask[:, :, 1:]
        return a

    ar, asd = adj(rock) & fluid, adj(sand) & fluid
    surf = ar | asd
    tag = np.zeros(model.shape, np.uint8)
    tag[asd] = TAGS["sand"]
    tag[ar] = TAGS["rock"]
    # ---- meter ---------------------------------------------------------------
    meter = np.where(fluid, result.meter, np.nan)
    mrel = np.where(fluid, result.meter_rel, np.nan)
    if result.sdf is not None:
        weak = surf & (result.fluid_fraction < 0.5)
        idx = np.nonzero(weak)
        if len(idx[0]):
            P = np.stack([model.node_coords(a)[idx[a]] for a in range(3)], -1)
            n = result.sdf.gradient(P)
            got = np.full(len(P), np.nan); grel = np.full(len(P), np.nan)
            for step in (1.0, 2.0):
                todo = np.isnan(got)
                ijk = np.floor((P[todo] + n[todo] * step * g.dx - g.origin) / g.dx + 0.5).astype(int)
                ijk = np.clip(ijk, 0, np.array(g.shape) - 1)
                ok = result.fluid_fraction[ijk[:, 0], ijk[:, 1], ijk[:, 2]] >= 0.5
                v = np.where(ok, result.meter[ijk[:, 0], ijk[:, 1], ijk[:, 2]], np.nan)
                r = np.where(ok, result.meter_rel[ijk[:, 0], ijk[:, 1], ijk[:, 2]], np.nan)
                got[todo] = v; grel[todo] = r
            meter[idx] = got; mrel[idx] = grel
    # ---- surface-normal --------------------------------------------------------
    normal = np.full(model.shape, np.nan)
    nrel = np.full(model.shape, np.nan)
    k = result.tallies.ppf_per_photon * result.scale
    if result.tallies.rock_w is not None and result.patches is not None:
        from scipy import ndimage
        box = np.ones((3, 3, 3))
        Wn = ndimage.convolve(result.tallies.rock_w, box, mode="constant")
        W2 = ndimage.convolve(result.tallies.rock_w2, box, mode="constant")
        An = ndimage.convolve(result.patches["area"], box, mode="constant")
        ok = ar & (An >= MIN_PATCH_AREA * g.dx ** 2)
        normal[ok] = Wn[ok] * k / An[ok]
        nrel[ok] = rel_error(Wn[ok], W2[ok])
    if result.sand is not None:
        sel = asd & ~ar
        i, j, _ = np.nonzero(sel)
        normal[sel] = result.sand[i, j]
        nrel[sel] = result.sand_rel[i, j]
    f = result.cfg.photoperiod.dli_factor
    return CfdPar(meter=meter, normal=normal, meter_rel=mrel, normal_rel=nrel, surface_adjacent=surf,
                  adjacent_tag=tag, dli_meter=meter * f, dli_normal=normal * f, label=result.label,
                  calibrated=result.calibrated)
