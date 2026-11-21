"""Tank model: project file + cell size -> LBM domain, surface tags and source table.

Grid layout (node = cell centre)
--------------------------------
* Fluid nodes span the internal length/width and the water column above the sand,
  rounded to whole cells and centred in x and y. Walls are one SOLID node layer;
  with halfway bounce-back each wall surface lies half a cell outside the
  outermost fluid nodes.
* k = 0 is the sand surface (SOLID, tag 'sand'): the sand bed is a solid floor at
  the user's depth (spec). The top layer is FREESLIP: the rigid free-slip lid
  standing in for the water surface (spec, PLAN C10).
* Rock: nodes whose centres lie inside a rock mesh become SOLID (tag 'rock').
* Overflow box: SOLID from the sand to the weir crest; the fluid nodes above it,
  up to the surface, form the mass sink (mass leaves at the return pumps' total
  flow, with its own momentum).
* Pumps: optional solid housing behind the outlet; SOURCE nodes in front of it.
  Every outlet (a powerhead, or each nozzle of a return pump) is its own source
  region with its own pump-state row and turbulence process; the nozzles of one
  pump share its mode s(t) (D-805).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geometry as G
from . import sources as S
from . import units as U
from .domain import Domain, SOLID, FREESLIP, SOURCE, FLUID
from .project import Project
from .pumps import OUProcess, Pump, equivalent_outlet, mode_shapes

TAGS = {"water": 0, "glass": 1, "sand": 2, "rock": 3, "overflow": 4, "housing": 5, "surface": 6,
        "source": 7, "sink": 8}
SQRT2 = math.sqrt(2.0)


@dataclass
class PumpRuntime:
    """One modelled outlet: a powerhead, or one nozzle of a return pump."""
    pump: Pump
    index: int                       # pump-state row
    outlet_eq: object
    n_nodes: int
    u_eq: float                      # equivalent outlet velocity sqrt(J / (rho A_eq)), m/s
    tcorr: float                     # turbulence correlation time, s
    lateral: tuple                   # (e1, e2) unit vectors perpendicular to aim
    ou: OUProcess | None = None
    notes: list = field(default_factory=list)
    nozzle: int = 0                  # index into pump.nozzles
    flow: float = 0.0                # delivered flow of this outlet, m^3/s


@dataclass
class TankModel:
    project: Project
    dx: float
    shape: tuple
    origin: np.ndarray               # tank coordinates of node (0, 0, 0)
    domain: Domain
    tags: np.ndarray
    lu: U.LatticeUnits
    spec: S.SourceSpec | None
    pumps: list
    overflow_index: int | None
    info: dict
    notes: list

    # -- coordinates ----------------------------------------------------
    def node_coords(self, axis: int) -> np.ndarray:
        return self.origin[axis] + self.dx * np.arange(self.shape[axis])

    @property
    def fluid_mask(self) -> np.ndarray:
        return np.isin(self.domain.flags, (FLUID, SOURCE))

    # -- pump state -------------------------------------------------------
    def pump_state(self, t: float, dt_update: float, advance: bool = True) -> np.ndarray:
        """Pump-state table at physical time t (advancing the turbulence processes by dt_update)."""
        st = self.spec.default_state()
        q_tot, q_now = 0.0, 0.0
        for pr in self.pumps:
            p = pr.pump
            s = float(p.mode.s(t))
            st[pr.index, S.P_SF] = s * s
            st[pr.index, S.P_SM] = s
            st[pr.index, S.P_I] = p.turbulence
            if p.turbulence > 0 and pr.ou is not None:
                x = pr.ou.advance(dt_update) if advance else pr.ou.x
                st[pr.index, S.P_A0:S.P_A0 + S.N_MODES] = x[:S.N_MODES]
                e1, e2 = pr.lateral
                st[pr.index, S.P_LAT:S.P_LAT + 3] = (x[S.N_MODES] * e1 + x[S.N_MODES + 1] * e2) / SQRT2
            if p.kind == "return":
                q_tot += pr.flow
                q_now += pr.flow * s
        if self.overflow_index is not None:
            st[self.overflow_index, S.P_SF] = 0.0
            st[self.overflow_index, S.P_SM] = q_now / q_tot if q_tot > 0 else 0.0
        return st


def _fill_pockets(d: Domain, tags, interior) -> dict:
    """Fluid regions not face-connected to the main water body (voids sealed inside
    rock) become rock: they exchange no water with the tank and would otherwise be
    counted as dead water."""
    from scipy import ndimage
    fluid = (d.flags == FLUID) & interior
    lab, n = ndimage.label(fluid)
    if n <= 1:
        return {"nodes": 0, "regions": 0, "mask": None}
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    mask = fluid & (lab != main)
    d.flags[mask] = SOLID
    tags[mask] = TAGS["rock"]
    return {"nodes": int(mask.sum()), "regions": n - 1, "mask": mask}


def build(project: Project, dx: float | None = None, *, verbose: bool = False) -> TankModel:
    p = project
    t = p.tank
    dx = float(dx or p.dx())
    notes: list[str] = []
    nxf = max(int(round(t.length / dx)), 4)
    nyf = max(int(round(t.width / dx)), 4)
    nzf = max(int(round(t.water_column / dx)), 4)
    shape = (nxf + 2, nyf + 2, nzf + 2)
    x_off = 0.5 * (t.length - nxf * dx)
    y_off = 0.5 * (t.width - nyf * dx)
    origin = np.array([x_off - 0.5 * dx, y_off - 0.5 * dx, t.sand_depth - 0.5 * dx])
    lid_z = t.sand_depth + nzf * dx
    eff = (nxf * dx, nyf * dx, nzf * dx)
    for name, true, got in (("length", t.length, eff[0]), ("width", t.width, eff[1]),
                            ("water column", t.water_column, eff[2])):
        if abs(got - true) > 1e-9:
            notes.append(f"grid {name} {got * 1e3:.1f} mm vs tank {true * 1e3:.1f} mm "
                         f"({100 * (got / true - 1):+.2f} %, whole cells)")

    d = Domain(shape)
    tags = np.zeros(shape, dtype=np.uint8)
    for f in ("x-", "x+", "y-", "y+"):
        d.set_face(f, SOLID)
        tags[d.face_slice(f)] = TAGS["glass"]
    d.flags[:, :, -1] = FREESLIP
    tags[:, :, -1] = TAGS["surface"]
    for f in ("x-", "x+", "y-", "y+"):                 # lid edges stay walls
        d.flags[d.face_slice(f)] = SOLID
    d.flags[:, :, 0] = SOLID
    tags[:, :, 0] = TAGS["sand"]
    tags[0, :, :] = tags[-1, :, :] = TAGS["glass"]
    tags[:, 0, :] = tags[:, -1, :] = TAGS["glass"]

    X = origin[0] + dx * np.arange(shape[0])
    Y = origin[1] + dx * np.arange(shape[1])
    Z = origin[2] + dx * np.arange(shape[2])
    interior = np.zeros(shape, bool)
    interior[1:-1, 1:-1, 1:-1] = True

    # ---- rock -----------------------------------------------------------
    rock_info = []
    rock = np.zeros(shape, bool)
    for r in p.rocks:
        v, fc = G.load_mesh(r.file, r.unit, r.translate, r.rotate_z, r.scale)
        rep = G.mesh_report(v, fc)
        if not rep["watertight"]:
            notes.append(f"rock mesh {r.file.name} is not watertight; voxelization may leak")
        m = G.voxelize_mesh(v, fc, origin, dx, shape)
        rock |= m
        rep.update(file=str(r.file.name))
        rock_info.append(rep)
    rock &= interior
    d.flags[rock] = SOLID
    tags[rock] = TAGS["rock"]
    pockets = _fill_pockets(d, tags, interior)
    if pockets["nodes"]:
        rock |= pockets["mask"]
        notes.append(f"filled {pockets['regions']} sealed water pocket(s) inside the rock "
                     f"({pockets['nodes']} nodes, {pockets['nodes'] * dx ** 3 * 1e6:.1f} cm3): no flow can reach them")

    # ---- overflow ----------------------------------------------------------
    sink_mask = np.zeros(shape, bool)
    ov_info = None
    if p.overflow is not None:
        x0, x1, y0, y1 = p.overflow.box(t)
        crest = t.water_surface - p.overflow.crest_below_surface
        fx = (X >= x0) & (X <= x1)
        fy = (Y >= y0) & (Y <= y1)
        foot = fx[:, None] & fy[None, :]
        kc = int(np.searchsorted(Z, crest))            # first node centre above the crest
        kc = min(kc, shape[2] - 2)                     # keep at least one fluid layer under the lid
        box = foot[:, :, None] & (np.arange(shape[2]) < kc)[None, None, :] & interior
        d.flags[box] = SOLID
        tags[box] = TAGS["overflow"]
        sink_mask = foot[:, :, None] & (np.arange(shape[2]) >= kc)[None, None, :] & interior & ~rock
        crest_model = origin[2] + (kc - 0.5) * dx
        if abs(crest_model - crest) > 0.5 * dx + 1e-9:
            notes.append(f"overflow crest modelled at {crest_model * 1e3:.1f} mm above the bottom "
                         f"(true {crest * 1e3:.1f} mm): at least one fluid cell must remain above the box")
        ov_info = {"footprint_m": [x0, x1, y0, y1], "placement": p.overflow.describe(t),
                   "crest_m": crest, "crest_model_m": crest_model,
                   "sink_nodes": int(sink_mask.sum()),
                   "sink_layers": int(shape[2] - 1 - kc)}

    # ---- lattice units: dt from the peak source velocity ---------------
    sim = p.sim
    outlets = [(pp, k) for pp in p.pumps for k in range(len(pp.nozzles))]   # one source region each
    J = {(pp.name, k): pp.nozzle_momentum_flux(k) for pp, k in outlets}
    eq = {}
    for pp, k in outlets:
        o_eq, n = equivalent_outlet(pp.nozzles[k].outlet, J[(pp.name, k)], dx, min_cells=sim["min_source_cells"],
                                    velocity_cap=sim["source_velocity_cap_ms"],
                                    min_size=sim["source_min_size_m"], rho=p.fluid["density"])
        if len(pp.nozzles) > 1:
            n = [f"nozzle {k + 1}: {s_}" for s_ in n]
        eq[(pp.name, k)] = (o_eq, n)
    u_eq = {key: math.sqrt(J[key] / (p.fluid["density"] * eq[key][0].area)) for key in J}
    # peak: ideal actuator-disk wake contraction gives sqrt(2) x U_eq for a force source;
    # used for both source types (conservative); the run monitors the actual maximum.
    u_peak = max([SQRT2 * v for v in u_eq.values()] + [0.05])
    lu = U.LatticeUnits.from_velocity(dx, u_peak, u_lat=sim["u_lattice_max"], rho0=p.fluid["density"])

    # ---- pumps -----------------------------------------------------------------
    pts_all = None
    rows_cells, rows_tab, pumps_rt = [], [], []
    rng = np.random.default_rng(int(sim["seed"]))
    taken = np.zeros(shape, bool)
    for idx, (pp, kz) in enumerate(outlets):
        key = (pp.name, kz)
        nz = pp.nozzles[kz]
        q_out = pp.nozzle_flow(kz)
        o_eq, pnotes = eq[key]
        a, l, up = G.frame(nz.aim, nz.up)
        Ls = max(sim["source_length_m"], 2 * dx)
        # bounding box of housing + source region
        ext = max(o_eq.hydraulic_size, o_eq.diameter, o_eq.width, o_eq.height,
                  (pp.housing.diameter if pp.housing else 0), (pp.housing.width if pp.housing else 0)) \
            + Ls + (pp.housing.length if pp.housing else 0) + 2 * dx
        lo = nz.position - ext
        hi = nz.position + ext
        ii = np.nonzero((X >= lo[0]) & (X <= hi[0]))[0]
        jj = np.nonzero((Y >= lo[1]) & (Y <= hi[1]))[0]
        kk = np.nonzero((Z >= lo[2]) & (Z <= hi[2]))[0]
        I, Jg, K = np.meshgrid(ii, jj, kk, indexing="ij")
        P = np.stack([X[I], Y[Jg], Z[K]], axis=-1)
        if pp.housing is not None:
            pnotes = pnotes + ["solid housing modelled behind the outlet: with no intake path the jet "
                               "delivers only ~55 % of J (verification 7, D-202); use for blockage studies only"]
            h = pp.housing
            hm = G.region_mask(P, nz.position, a, up, h.shape, -h.length, 0.0,
                               diameter=h.diameter, width=h.width, height=h.height)
            sel = (I[hm], Jg[hm], K[hm])
            ok = interior[sel]
            sel = tuple(s[ok] for s in sel)
            d.flags[sel] = SOLID
            tags[sel] = TAGS["housing"]
        sm = G.region_mask(P, nz.position, a, up, o_eq.shape, 0.0, Ls,
                           diameter=o_eq.diameter, width=o_eq.width, height=o_eq.height)
        sel = (I[sm], Jg[sm], K[sm])
        okm = (d.flags[sel] == FLUID) & ~sink_mask[sel] & ~taken[sel]
        lost = int((~okm).sum())
        sel = tuple(s[okm] for s in sel)
        n = len(sel[0])
        if n == 0:
            raise ValueError(f"pump {pp.name!r}{f' nozzle {kz + 1}' if len(pp.nozzles) > 1 else ''}: "
                             "its outlet region contains no free water nodes "
                             "(inside rock, glass or another source?)")
        if lost:
            pnotes = pnotes + [f"{lost} of {lost + n} outlet nodes fell in solid/occupied cells and were dropped"]
        taken[sel] = True
        s_, t1, t2 = G.local_coords(np.stack([X[sel[0]], Y[sel[1]], Z[sel[2]]], axis=-1), nz.position, a, up)
        half1 = (o_eq.diameter if o_eq.shape == "circle" else o_eq.width) / 2
        half2 = (o_eq.diameter if o_eq.shape == "circle" else o_eq.height) / 2
        phi = mode_shapes(t1 / half1, t2 / half2)
        tab = np.zeros((n, S.S_NCOL))
        tab[:, S.S_PUMP] = idx
        if pp.kind == "powerhead":
            F_lat = J[key] / lu.C_J                        # total force, lattice units
            tab[:, S.S_F0:S.S_F0 + 3] = (F_lat / n) * a
        else:
            Q_lat = q_out / lu.C_Q
            u_noz = pp.nozzle_velocity_of(kz)
            tab[:, S.S_MASS] = Q_lat / n
            tab[:, S.S_U0:S.S_U0 + 3] = lu.velocity_to_lattice(u_noz) * a
            if lu.velocity_to_lattice(u_noz) > 0.3:
                raise ValueError(f"return {pp.name!r} nozzle {kz + 1}: nozzle velocity {u_noz:.2f} m/s is "
                                 f"{lu.velocity_to_lattice(u_noz):.2f} in lattice units (> 0.3)")
        tab[:, S.S_PHI0:S.S_PHI0 + S.N_MODES] = phi
        rows_cells.append(np.stack(sel, axis=1))
        rows_tab.append(tab)
        d.flags[sel] = SOURCE
        tags[sel] = TAGS["source"]
        tc = (o_eq.hydraulic_size if o_eq.shape == "rect" else o_eq.diameter) / u_eq[key]
        pr = PumpRuntime(pump=pp, index=idx, outlet_eq=o_eq, n_nodes=n, u_eq=u_eq[key], tcorr=tc,
                         lateral=(l, up), notes=pnotes, nozzle=kz, flow=q_out,
                         ou=OUProcess(S.N_MODES + 2, tc, np.random.default_rng(rng.integers(2**63))))
        pumps_rt.append(pr)
        notes += [f"{pp.name}: {s}" for s in pnotes]

    ov_index = None
    if p.overflow is not None and any(pp.kind == "return" for pp in p.pumps):
        ov_index = len(outlets)
        sel = np.nonzero(sink_mask & (d.flags == FLUID))
        n = len(sel[0])
        if n == 0:
            raise ValueError("overflow has no free water nodes above its crest")
        Q_lat = sum(pp.delivered_flow for pp in p.pumps if pp.kind == "return") / lu.C_Q
        tab = np.zeros((n, S.S_NCOL))
        tab[:, S.S_PUMP] = ov_index
        tab[:, S.S_MASS] = -Q_lat / n
        rows_cells.append(np.stack(sel, axis=1))
        rows_tab.append(tab)
        d.flags[sel] = SOURCE
        tags[sel] = TAGS["sink"]

    spec = None
    if rows_cells:
        spec = S.SourceSpec(np.vstack(rows_cells), np.vstack(rows_tab),
                            n_pumps=len(outlets) + (1 if ov_index is not None else 0))
    d.finalize()

    n_fluid = int(np.isin(d.flags, (FLUID, SOURCE)).sum())
    info = {
        "dx_m": dx, "shape": list(shape), "cells": int(np.prod(shape)), "fluid_nodes": n_fluid,
        "grid_fluid_extent_m": list(eff), "origin_m": origin.tolist(), "lid_z_m": lid_z,
        "water_volume_model_m3": n_fluid * dx ** 3,
        "water_volume_model_gal": U.from_si(n_fluid * dx ** 3, "gal"),
        "rock_nodes": int(rock.sum()), "rock_volume_m3": float(rock.sum() * dx ** 3),
        "rock_meshes": rock_info, "overflow": ov_info,
        "lattice": {"dt_s": lu.dt, "C_U_ms": lu.C_U, "u_peak_est_ms": u_peak,
                    "nu_lat": lu.nu_to_lattice(p.fluid["viscosity"]),
                    "tau_molecular": lu.tau_from_nu(p.fluid["viscosity"]),
                    "sound_speed_ms": lu.sound_speed},
        "pumps": _pump_info(p.pumps, pumps_rt),
    }
    return TankModel(project=p, dx=dx, shape=shape, origin=origin, domain=d, tags=tags, lu=lu, spec=spec,
                     pumps=pumps_rt, overflow_index=ov_index, info=info, notes=notes)


def _pump_info(pumps, runtimes) -> list:
    """One entry per pump (keys as in Phase 2); per-nozzle model details under 'nozzles'."""
    out = []
    for pp in pumps:
        rts = [r for r in runtimes if r.pump is pp]
        d = pp.summary()
        first = rts[0]
        d.update(outlet_model=first.outlet_eq.describe() if len(rts) == 1 else
                 "; ".join(f"nozzle {r.nozzle + 1}: {r.outlet_eq.describe()}" for r in rts),
                 outlet_model_area_m2=sum(r.outlet_eq.area for r in rts),
                 source_nodes=sum(r.n_nodes for r in rts), equivalent_velocity_ms=first.u_eq,
                 turbulence_tcorr_s=first.tcorr, notes=[n for r in rts for n in r.notes])
        for nzd, r in zip(d["nozzles"], rts):
            nzd.update(outlet_model=r.outlet_eq.describe(), outlet_model_area_m2=r.outlet_eq.area,
                       source_nodes=r.n_nodes, equivalent_velocity_ms=r.u_eq, turbulence_tcorr_s=r.tcorr)
        out.append(d)
    return out
