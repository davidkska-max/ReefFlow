"""Run a tank simulation: spin-up, time averaging, monitoring, outputs.

Outputs in the run directory
    run.json       inputs, derived lattice parameters, pump summaries, metrics with
                   statistical uncertainty, run history summary, warnings, timing
    fields.npz     time-averaged fields (SI): mean velocity, mean speed, TKE, tags
    tank.vti       the same fields for ParaView
    slices.png     plan / front / side slices of mean speed with vectors
    history.png    kinetic energy, max speed and probe speeds vs time
    geometry.png   voxel model check
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import metrics as MET
from . import plots
from .io_vtk import write_vti
from .stats import Monitor, RunningStats
from .tank import TankModel

U_LAT_ABORT = 0.35     # lattice speed at which a run is stopped (unstable)
U_LAT_WARN = 0.15


def estimate(model: TankModel, mlups: float | None = None, arch: str | None = None) -> dict:
    """Memory and run-time estimate shown before a run."""
    from . import backend
    p = model.project
    lat = p.sim["lattice"].upper()
    Q = 19 if lat == "D3Q19" else 27
    cells = int(np.prod(model.shape))
    dt = model.lu.dt
    steps = int(math.ceil(p.sim["spinup_s"] / dt) + math.ceil(p.sim["average_s"] / dt))
    gpu_bytes = cells * (2 * Q * 4 + 1 + 1 + 4 + 12 + 4 + 40)
    host_bytes = cells * (2 * 80 + 40)
    arch = arch or backend.current() or "cuda"
    if mlups is None:
        # measured, D-119 (RTX 3070) and PLAN (CPU); derated 10 % for sources + statistics
        base = {"cuda": {19: 1900, 27: 1360}, "vulkan": {19: 1900, 27: 1360},
                "cpu": {19: 15, 27: 10}}.get(arch, {19: 15, 27: 10})[Q]
        mlups = 0.9 * base
    secs = cells * steps / (mlups * 1e6)
    return {"cells": cells, "steps": steps, "dt_s": dt, "gpu_memory_GB": gpu_bytes / 1e9,
            "host_memory_GB": host_bytes / 1e9, "backend": arch, "mlups_assumed": mlups,
            "runtime_s_est": secs, "free_vram_GB": _free_vram()}


def _free_vram():
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return None


def _probe_points(model: TankModel):
    """Six probe nodes: left/centre/right thirds at 30 % and 70 % of depth, mid-width,
    moved to the nearest fluid node."""
    fl = model.fluid_mask
    nx, ny, nz = model.shape
    pts, names = [], []
    for fx, lab in ((0.2, "left"), (0.5, "centre"), (0.8, "right")):
        for fz in (0.3, 0.7):
            i, j, k = int(nx * fx), int(ny * 0.5), 1 + int((nz - 3) * fz)
            # walk upward (then sideways) to the nearest fluid node
            for dk in range(nz):
                kk = min(k + dk, nz - 2)
                if fl[i, j, kk]:
                    break
            pts.append((i, j, kk)); names.append(f"{lab} {int(fz * 100)}%")
    return np.array(pts), names


class RunCancelled(RuntimeError):
    pass


def run(model: TankModel, out_dir, *, progress=print, write_outputs: bool = True,
        spinup: float | None = None, average: float | None = None, progress_interval_s: float = 30.0,
        on_status=None, should_stop=None) -> dict:
    """progress: callable(str) for human-readable lines every progress_interval_s.
    on_status: optional callable(dict) with fraction, simulated time, max speed... at every
    monitor point (~4 per simulated second). should_stop: optional callable() -> bool, checked
    at every monitor point; True stops the run cleanly with RunCancelled."""
    from .solver import LBMSolver
    from . import backend
    p = model.project
    sim = p.sim
    lu = model.lu
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_spin = spinup if spinup is not None else sim["spinup_s"]
    t_avg = average if average is not None else sim["average_s"]
    se = int(sim["sample_every"])
    nb = int(sim["batches"])
    n_spin = int(math.ceil(t_spin / lu.dt / se)) * se
    n_avg = int(math.ceil(t_avg / lu.dt / (se * nb))) * se * nb
    batch_len = n_avg // nb

    # pump-state update interval: >= 8 updates per turbulence correlation time and
    # >= 200 per mode period
    lim = [rt.tcorr / 8 for rt in model.pumps if rt.pump.turbulence > 0]
    lim += [rt.pump.mode.period / 200 for rt in model.pumps if rt.pump.mode.type != "constant"]
    n_lim = max(1, min([int(v / lu.dt) for v in lim] + [se]))
    n_up = max(d for d in range(1, se + 1) if se % d == 0 and d <= n_lim)   # divides se: aligned with samples

    nu_lat = lu.nu_to_lattice(p.fluid["viscosity"])
    est = estimate(model)
    progress(f"[run] {p.name}: grid {model.shape} ({est['cells'] / 1e6:.2f} M cells), dx = "
             f"{model.dx * 1e3:g} mm, dt = {lu.dt * 1e3:.4f} ms, steps {n_spin} + {n_avg}, "
             f"est. {est['runtime_s_est'] / 60:.1f} min, GPU memory {est['gpu_memory_GB']:.2f} GB "
             f"(free {est['free_vram_GB'] if est['free_vram_GB'] is not None else 'n/a'} GB)")
    if est["free_vram_GB"] is not None and est["gpu_memory_GB"] > 0.9 * est["free_vram_GB"] \
            and backend.current() == "cuda":
        progress("[run] WARNING: estimated GPU memory exceeds free VRAM; expect paging/slowdown (D-119)")

    use_wm = bool(sim.get("wall_model"))
    s = LBMSolver(model.domain, lattice=sim["lattice"].upper(), collision=sim["collision"], nu=nu_lat,
                  smagorinsky=float(sim["smagorinsky"]) if sim["smagorinsky"] else None,
                  sources=model.spec, sponge_nu=np.zeros(model.shape) if use_wm else None)
    wm = None
    if use_wm:
        from .wallmodel import WallModel, wall_nodes
        wm = WallModel(s, *wall_nodes(model.domain.flags))
        progress(f"[run] WARNING: experimental wall model on ({wm.n} wall nodes). It FAILED verification "
                 "(diverged in channel flow, DECISIONS D-702); results may be unstable.")
    stats = RunningStats(s)
    mon = Monitor(s)
    pidx, pnames = _probe_points(model)
    mon.set_probes(pidx)
    M = MET.masks(model)
    dead = p.criteria["dead_zone_speed"]

    hist = {"t": [], "ke": [], "u_max": [], "probes": [], "mass": []}
    mass0 = s.total_mass()
    n_mon = max(se, int(round(0.25 / lu.dt / se)) * se)
    warn_steps = 0
    batches = []
    prev = None
    t0 = time.perf_counter()
    last_print = t0
    total = n_spin + n_avg
    step = 0
    st = model.pump_state(0.0, 0.0, advance=False) if model.spec is not None else None
    if st is not None:
        s.set_pump_state(st)
    while step < total:
        if model.spec is not None and step % n_up == 0:
            s.set_pump_state(model.pump_state(step * lu.dt, n_up * lu.dt))
        s.step(n_up, store_last=True)
        if wm is not None:
            wm.update()
        step += n_up
        if step % se:
            continue
        averaging = step > n_spin
        if averaging:
            stats.sample()
            if (step - n_spin) % batch_len == 0:
                n_now, sums = stats.snapshot()
                if prev is None:
                    bs, bn = sums, n_now
                else:
                    bs, bn = sums - prev[1], n_now - prev[0]
                r = RunningStats.derive(bs, bn)
                bm = MET.compute(model, r["mean_u"] * lu.C_U, r["mean_speed"] * lu.C_U,
                                 r["tke"] * lu.C_U ** 2, dead, M)
                batches.append(bm)
                prev = (n_now, sums)
        if step % n_mon == 0 or step == total:
            m = mon()
            if m["nonfinite"] or m["u_max"] > U_LAT_ABORT:
                raise FloatingPointError(f"run unstable at step {step} (t = {step * lu.dt:.2f} s): "
                                         f"max lattice speed {m['u_max']:.3f}, non-finite nodes {m['nonfinite']}")
            if m["u_max"] > U_LAT_WARN:
                warn_steps += n_mon
            hist["t"].append(step * lu.dt)
            hist["ke"].append(m["ke_mean"] * lu.C_U ** 2)
            hist["u_max"].append(m["u_max"] * lu.C_U)
            hist["probes"].append((np.linalg.norm(mon.probes(), axis=1) * lu.C_U).tolist())
            now = time.perf_counter()
            if on_status is not None:
                el_ = now - t0
                on_status({"fraction": step / total, "t_sim_s": step * lu.dt, "t_total_s": total * lu.dt,
                           "phase": "averaging" if step > n_spin else "spin-up", "u_max_ms": m["u_max"] * lu.C_U,
                           "u_max_lattice": m["u_max"], "elapsed_s": el_,
                           "eta_s": el_ / step * (total - step) if step else None})
            if should_stop is not None and should_stop():
                raise RunCancelled(f"run cancelled at t = {step * lu.dt:.1f} s")
            if now - last_print > progress_interval_s or step == total:
                el = now - t0
                progress(f"[run] t = {step * lu.dt:6.1f} s / {total * lu.dt:.1f} s  "
                         f"({100 * step / total:5.1f} %), max|u| = {m['u_max'] * lu.C_U:.3f} m/s "
                         f"(lattice {m['u_max']:.3f}), elapsed {el / 60:.1f} min, "
                         f"ETA {el / step * (total - step) / 60:.1f} min")
                last_print = now
    wall = time.perf_counter() - t0
    mass1 = s.total_mass()

    res = stats.result()
    mean_u = res["mean_u"] * lu.C_U
    speed = res["mean_speed"] * lu.C_U
    tke = res["tke"] * lu.C_U ** 2
    met = MET.compute(model, mean_u, speed, tke, dead, M)
    unc = {}
    for k in met:
        vals = [b[k] for b in batches if isinstance(b.get(k), float)]
        if len(vals) == len(batches) and len(vals) > 1:
            unc[k] = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    ke = np.asarray(hist["ke"])
    t = np.asarray(hist["t"])
    a = ke[t > t_spin]
    half = len(a) // 2
    result = {
        "project": p.name, "project_file": str(p.path.resolve()) if p.path else None,
        "backend": backend.current(), "precision": backend.precision(),
        "lattice": sim["lattice"].upper(), "collision": sim["collision"], "smagorinsky": sim["smagorinsky"],
        "dx_m": model.dx, "spinup_s": n_spin * lu.dt, "average_s": n_avg * lu.dt,
        "steps": total, "sample_every": se, "samples": res["n"], "pump_update_every": n_up,
        "wall_time_s": wall, "mlups": float(np.prod(model.shape)) * total / wall / 1e6,
        "estimate": est, "model": model.info, "notes": model.notes,
        "metrics": met, "metrics_stat_uncertainty": unc, "batches": batches,
        "mass_rel_drift": mass1 / mass0 - 1.0,
        "ke_spinup_end": float(ke[t <= t_spin][-1]) if np.any(t <= t_spin) else None,
        "ke_avg_first_half": float(a[:half].mean()) if half else None,
        "ke_avg_second_half": float(a[half:].mean()) if half else None,
        "u_max_overall_ms": float(np.max(hist["u_max"])),
        "u_max_overall_lattice": float(np.max(hist["u_max"]) / lu.C_U),
        "fraction_time_above_u_lat_0.15": warn_steps / total,
        "probe_names": pnames, "probe_nodes": pidx.tolist(),
        "sim": {k: v for k, v in sim.items() if not k.endswith(("_s", "_ms", "_m"))},
        "wall_model_nodes": wm.n if wm is not None else 0,
    }
    if write_outputs:
        np.savez_compressed(out / "fields.npz", mean_u=mean_u.astype(np.float32),
                            mean_speed=speed.astype(np.float32), tke=tke.astype(np.float32),
                            reynolds=(res["reynolds"] * lu.C_U ** 2).astype(np.float32),
                            tags=model.tags, flags=model.domain.flags, origin=model.origin, dx=model.dx)
        fl = model.fluid_mask
        sig = np.sqrt(2.0 / 3.0 * np.clip(tke, 0, None))
        write_vti(out / "tank.vti", {
            "mean_velocity": np.where(fl[..., None], mean_u, 0.0),
            "mean_speed": np.where(fl, speed, 0.0),
            "rms_fluctuation": np.where(fl, sig, 0.0),
            "turbulence_intensity": np.where(fl, sig / np.maximum(speed, 1e-4), 0.0),
            "tke": np.where(fl, tke, 0.0),
            "tag": model.tags,
        }, model.origin, model.dx)
        plots.slice_figure(model, speed, mean_u, out / "slices.png")
        plots.history_figure(hist, out / "history.png", t_spin, pnames)
        plots.geometry_figure(model, out / "geometry.png")
        np.savez_compressed(out / "history.npz", **{k: np.asarray(v) for k, v in hist.items()})
        with open(out / "run.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, default=_json_default)
    result["_fields"] = {"mean_u": mean_u, "mean_speed": speed, "tke": tke,
                         "reynolds": res["reynolds"] * lu.C_U ** 2}
    return result


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
