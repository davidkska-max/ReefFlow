"""ReefFlow browser UI: FastAPI backend (local only, offline).

``create_app(workspace)`` returns the ASGI app. All engineering logic lives in the
core library (reefflow.project / tank / run / analysis, and reefflow.geometry for
rock meshes); this module only translates HTTP requests into core calls.

Endpoints (JSON unless noted; binary = little-endian float32/uint32 with a JSON
``X-Meta`` header):

    GET  /                                       the single-page app
    GET  /api/health
    GET  /api/library                            pump archetypes, coral profiles, fixtures, presets, units
    GET  /api/colormaps                          256-entry RGB tables (viridis, magma, ...)
    GET  /api/projects                           list
    POST /api/projects                           {name, template: blank|demo}
    POST /api/projects/import                    multipart: project YAML (+ mesh files)
    GET  /api/projects/{pid}                     editor model + YAML + validation
    PUT  /api/projects/{pid}                     {editor} -> save (always) + validation
    DELETE /api/projects/{pid}
    POST /api/projects/{pid}/copy                {name}
    POST /api/projects/{pid}/validate            {editor} -> validation (no save)
    GET  /api/projects/{pid}/yaml                text/yaml download
    PUT  /api/projects/{pid}/yaml                {text} -> save raw YAML + validation
    POST /api/projects/{pid}/estimate            {editor?, preset, lattice, dx} -> tank model summary + estimate
    POST /api/projects/{pid}/meshes              multipart: STL/OBJ/PLY + unit -> new 'mesh' element
    POST /api/projects/{pid}/scans ...           phone-scan clean-up (scan_routes.py)
    POST /api/phone/start | GET status | POST stop   phone photo link on the home network (phone.py)
    POST /api/projects/{pid}/element_mesh        {element} -> binary mesh (local coordinates)
    POST /api/projects/{pid}/rockscape/build     {editor, resolution} -> union STL, saved project, issues
    GET  /api/projects/{pid}/rockscape.stl       export the built rockscape (STL, mm)
    GET  /api/projects/{pid}/rock_mesh           binary mesh of the project's rockscape entries (world)
    GET  /api/projects/{pid}/runs                runs of a project
    POST /api/projects/{pid}/runs                {preset, lattice, dx?, spinup?, average?, label?, lighting?} -> job
    GET  /api/runs                               all runs
    GET  /api/runs/{pid}/{rid}                   run + analysis summary, available layers
    DELETE /api/runs/{pid}/{rid}
    POST /api/runs/{pid}/{rid}/analyze           -> job
    GET  /api/runs/{pid}/{rid}/project           editor model of the run's project snapshot
    GET  /api/runs/{pid}/{rid}/slice             ?field=&axis=x|y|z&pos=<m>  -> binary float32 2-D
    GET  /api/runs/{pid}/{rid}/vectors           ?axis=&pos=&stride=         -> binary
    GET  /api/runs/{pid}/{rid}/sand_shear        -> binary float32 2-D (Pa)
    GET  /api/runs/{pid}/{rid}/scores            ?profile=                   -> binary points + scores
    GET  /api/runs/{pid}/{rid}/rock_mesh         -> binary mesh of the snapshot rockscape (world)
    GET  /api/runs/{pid}/{rid}/file/{name}       report.html, PNG figures, PLY score meshes, logs
    POST /api/compare                            {a: [pid, rid], b: [pid, rid], label_a, label_b}
    GET  /api/comparisons/{name}/{file}
    GET  /api/jobs | /api/jobs/{id} | /api/jobs/{id}/events (SSE) ; POST /api/jobs/{id}/cancel
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import yaml
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import reefflow  # noqa: F401  (sets TI_SKIP_VERSION_CHECK before anything imports Taichi)

from .. import project as PR
from .. import units as U
from . import editor_model as EM
from . import lighting_adapter as LA
from . import results as RS
from . import rockgen as RG
from .jobs import TERMINAL, JobManager, mark_interrupted
from .workspace import Workspace, check_id, slugify

STATIC = Path(__file__).resolve().parent / "static"
MESH_EXT = (".stl", ".obj", ".ply")
RUN_FILE_EXT = (".html", ".png", ".ply", ".json", ".txt", ".log", ".vti", ".stl", ".yaml")


def _err(e: Exception) -> str:
    if isinstance(e, (ValueError, FileNotFoundError)):
        return str(e)
    if isinstance(e, KeyError):
        return f"missing field {e}"
    return f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------
# validation and summaries (core calls)
# ----------------------------------------------------------------------
def validate_raw(raw: dict, base_dir: Path) -> dict:
    errors, warnings = [], []
    P = None
    try:
        P = PR.parse(raw, base_dir / "project.yaml")
    except Exception as e:                                # noqa: BLE001 (any schema error goes to the user)
        errors.append(_err(e))
    out = {"ok": False, "errors": errors, "warnings": warnings}
    if P is not None:
        for r in P.rocks:
            if not Path(r.file).is_file():
                errors.append(f"rock mesh {Path(r.file).name!r} not found next to the project file")
        errors += LA.validate(raw, P.tank)
        t = P.tank
        out["tank"] = {"internal_m": [t.length, t.width, t.height], "water_surface_m": t.water_surface,
                       "water_column_m": t.water_column, "sand_depth_m": t.sand_depth,
                       "gross_volume_m3": t.length * t.width * t.height,
                       "water_volume_above_sand_m3": t.length * t.width * t.water_column}
        out["pumps"] = [p.summary() for p in P.pumps]
        if not P.pumps:
            warnings.append("no pumps: the water will not move")
        warnings += PR.return_check(P) + PR.overflow_check(P)
        if not P.rocks:
            warnings.append("no rockscape yet")
    if EM.designer_stale(raw):
        warnings.append("rockscape designer has unbuilt changes: press 'Build rockscape' before running")
    out["ok"] = not errors
    return out


def library() -> dict:
    from ..analysis.coral import load_profiles
    from ..pumps import Outlet, load_library
    pumps = {}
    for name, a in load_library().items():
        o = Outlet.parse(a["outlet"])
        pumps[name] = {"kind": a["kind"], "description": a.get("description", ""),
                       "outlet": {"shape": o.shape, "diameter": o.diameter, "width": o.width, "height": o.height,
                                  "area": o.area},
                       "rated_flow": U.parse(a["rated_flow"], "flow"), "derating": float(a.get("derating", 0.7)),
                       "turbulence": float(a.get("turbulence", 0.05)),
                       "static_head": U.parse(a.get("static_head", 0), "head"),
                       "fittings_allowance": U.parse(a.get("fittings_allowance", 0), "head"),
                       "max_head": U.parse(a["pump_curve"]["max_head"], "head") if a.get("pump_curve") else None}
        if a.get("product"):
            from ..pumps import parse_curve
            pumps[name].update(LA._product_meta(a))
            pumps[name].update({
                "power_w": LA._num(a.get("power")) if a.get("power") is not None else None,
                "rated_flow_basis": a.get("rated_flow_basis"), "outlet_basis": a.get("outlet_basis"),
                "curve": [[h, q] for h, q in parse_curve(a.get("pump_curve")) or []],
                "curve_basis": (a.get("pump_curve") or {}).get("basis"), "pump_type": a.get("pump_type")})
    oc = PR.load_overflow_catalog()
    overflows = {}
    for k, v in oc["products"].items():
        b = v.get("box") or {}
        overflows[k] = {**LA._product_meta(v), "mount": v.get("mount"), "design": v.get("design"),
                        "drains": v.get("drains"), "rated_flow_basis": v.get("rated_flow_basis"),
                        "width": U.parse(b["width"], "length") if b.get("width") else None,
                        "depth": U.parse(b["depth"], "length") if b.get("depth") else None,
                        "height": U.parse(b["height"], "length") if b.get("height") else None,
                        "capacity": U.parse(v["rated_flow"], "flow") if v.get("rated_flow") else None}
    return {
        "overflows": overflows, "drain_capacity": oc["drain_capacity"],
        "pumps": pumps, "coral_profiles": load_profiles(), "fixtures": LA.fixture_library(),
        "lighting_available": LA.available(),
        "presets": PR.PRESETS, "sim_defaults": {k: v for k, v in PR.SIM_DEFAULTS.items() if v is not None},
        "element_types": RG.TYPES, "element_defaults": RG.DEFAULTS,
        "units": {k: [v[0], v[1]] for k, v in U._UNITS.items()},
        "display": {"US": {"length": "in", "flow": "GPH", "head": "ft", "velocity": "in/s", "volume": "gal",
                           "mass": "lb", "small_velocity": "in/s"},
                    "SI": {"length": "mm", "flow": "L/h", "head": "m", "velocity": "cm/s", "volume": "L",
                           "mass": "kg", "small_velocity": "cm/s"}},
        "overflow_walls": [["back", "back glass"], ["front", "front glass"], ["left", "left glass"],
                           ["right", "right glass"]],
    }


def colormaps() -> dict:
    import matplotlib
    out = {}
    for name in ("viridis", "magma", "cividis", "RdBu_r", "YlOrRd", "plasma"):
        cm = matplotlib.colormaps[name]
        out[name] = (np.asarray(cm(np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(int).ravel().tolist()
    return out


def estimate_for(raw: dict, base_dir: Path, preset: str | None, lattice: str | None, dx: float | None) -> dict:
    from .. import tank as TK
    from ..run import estimate
    P = PR.parse(raw, base_dir / "project.yaml")
    over = {"preset": preset, "lattice": lattice, "dx": f"{dx * 1000:.6g} mm" if dx else None}
    P = P.with_overrides(**over)
    m = TK.build(P, P.dx())
    est = estimate(m)
    info = copy.deepcopy(m.info)
    return {"estimate": est, "info": info, "notes": m.notes, "dx_m": m.dx, "preset": P.sim["preset"],
            "lattice": P.sim["lattice"], "spinup_s": P.sim["spinup_s"], "average_s": P.sim["average_s"],
            "warning": ("estimated GPU memory exceeds free VRAM" if est.get("free_vram_GB") is not None
                        and est["gpu_memory_GB"] > 0.9 * est["free_vram_GB"] else None)}


def _binary(arrays: list, meta: dict) -> Response:
    body = b"".join(np.ascontiguousarray(a).astype("<f4" if a.dtype.kind == "f" else "<u4").tobytes() for a in arrays)
    return Response(content=body, media_type="application/octet-stream",
                    headers={"X-Meta": json.dumps(meta, separators=(",", ":")), "Cache-Control": "no-store"})


def rock_mesh_from_raw(raw: dict, base_dir: Path) -> Response:
    from .. import geometry as G
    vs, fs, off = [], [], 0
    for r in raw.get("rockscape") or []:
        f = Path(r["file"])
        f = f if f.is_absolute() else base_dir / f
        if not f.is_file():
            continue
        tr = [U.parse(c, "length") for c in r.get("translate", (0, 0, 0))]
        v, fc = G.load_mesh(f, r.get("unit", "mm"), tr, float(r.get("rotate_z", 0.0)), float(r.get("scale", 1.0)))
        vs.append(v)
        fs.append(fc + off)
        off += len(v)
    if not vs:
        return Response(content=np.array([0, 0], "<u4").tobytes(), media_type="application/octet-stream")
    return Response(content=RG.pack_mesh(np.vstack(vs), np.vstack(fs)), media_type="application/octet-stream")


# ----------------------------------------------------------------------
# app
# ----------------------------------------------------------------------
def create_app(workspace=None, start_worker: bool = True) -> FastAPI:
    ws = Workspace(workspace or os.environ.get("REEFFLOW_WORKSPACE") or Path.cwd() / "reefflow_workspace")
    mark_interrupted([Path(r) for p in (ws.root / "projects").glob("*/runs") for r in p.iterdir() if r.is_dir()])
    jobs = JobManager(start_worker=start_worker)                         # GPU queue (CFD runs, analysis)
    photo_jobs = JobManager(start_worker=start_worker, prefix="p")       # CPU queue (photo carving)

    def find_job(jid: str):
        return (photo_jobs if jid.startswith("p") else jobs).get(jid)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        jobs.shutdown()
        photo_jobs.shutdown()
        if getattr(_app.state, "phone", None):
            _app.state.phone.stop()

    app = FastAPI(title="ReefFlow", version="0.1", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
    app.state.ws = ws
    app.state.jobs = jobs
    app.state.photo_jobs = photo_jobs
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(KeyError)
    async def _keyerror(request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc).strip("'\"")})

    @app.exception_handler(ValueError)
    async def _valueerror(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def _fnf(request: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    # ---------------------------------------------------------------- basics
    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        return {"ok": True, "workspace": str(ws.root), "lighting_available": LA.available()}

    @app.get("/api/library")
    def get_library():
        return library()

    @app.get("/api/colormaps")
    def get_colormaps():
        return colormaps()

    # ---------------------------------------------------------------- projects
    @app.get("/api/projects")
    def list_projects():
        return {"projects": ws.list_projects(), "workspace": str(ws.root)}

    @app.post("/api/projects")
    def create_project(payload: dict = Body(...)):
        name = str(payload.get("name") or "New tank")
        pid = ws.create_project(name, payload.get("template", "blank"))
        return {"id": pid}

    @app.post("/api/projects/import")
    async def import_project(project_file: UploadFile = File(...), meshes: list[UploadFile] | None = File(None)):
        try:
            raw = yaml.safe_load((await project_file.read()).decode("utf-8"))
        except Exception as e:                            # noqa: BLE001
            raise HTTPException(400, f"not a readable YAML file: {e}")
        if not isinstance(raw, dict):
            raise HTTPException(400, "project file must be a YAML mapping")
        files = {}
        for m in meshes or []:
            nm = Path(m.filename or "mesh.stl").name
            if Path(nm).suffix.lower() not in MESH_EXT:
                raise HTTPException(400, f"{nm}: only STL/OBJ/PLY meshes")
            files[nm] = await m.read()
        pid = ws.create_project(str(raw.get("name") or Path(project_file.filename or "imported").stem), raw=raw,
                                files=files)
        return {"id": pid, "validation": validate_raw(ws.read_raw(pid), ws.project_dir(pid))}

    def project_payload(pid: str) -> dict:
        raw = ws.read_raw(pid)
        d = ws.project_dir(pid)
        return {"id": pid, "dir": str(d), "editor": EM.to_editor(raw), "yaml": ws.project_path(pid).read_text(encoding="utf-8"),
                "validation": validate_raw(raw, d), "has_rockscape_stl": (d / "rockscape.stl").is_file(),
                "designer_stale": EM.designer_stale(raw)}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        return project_payload(pid)

    @app.put("/api/projects/{pid}")
    def save_project(pid: str, payload: dict = Body(...)):
        ed = payload.get("editor")
        if not isinstance(ed, dict):
            raise HTTPException(400, "body must be {editor: {...}}")
        old = ws.read_raw(pid)
        try:
            raw = EM.from_editor(ed, old)
        except Exception as e:                            # noqa: BLE001
            raise HTTPException(400, f"cannot convert the editor model: {_err(e)}")
        ws.write_raw(pid, raw)
        return project_payload(pid)

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str):
        if any(j["project"] == pid and j["status"] not in TERMINAL for j in jobs.list()):
            raise HTTPException(409, "this project has a queued or running job")
        ws.delete_project(pid)
        return {"ok": True}

    @app.post("/api/projects/{pid}/copy")
    def copy_project(pid: str, payload: dict = Body(...)):
        return {"id": ws.copy_project(pid, str(payload.get("name") or f"{pid} copy"))}

    @app.post("/api/projects/{pid}/validate")
    def validate_project(pid: str, payload: dict = Body(...)):
        d = ws.project_dir(pid)
        try:
            raw = EM.from_editor(payload["editor"], ws.read_raw(pid))
        except Exception as e:                            # noqa: BLE001
            return {"ok": False, "errors": [f"cannot convert the editor model: {_err(e)}"], "warnings": []}
        return validate_raw(raw, d)

    @app.get("/api/projects/{pid}/yaml")
    def get_yaml(pid: str):
        return PlainTextResponse(ws.project_path(pid).read_text(encoding="utf-8"), media_type="text/yaml",
                                 headers={"Content-Disposition": f'attachment; filename="{pid}.yaml"'})

    @app.put("/api/projects/{pid}/yaml")
    def put_yaml(pid: str, payload: dict = Body(...)):
        try:
            raw = yaml.safe_load(payload.get("text", ""))
        except yaml.YAMLError as e:
            raise HTTPException(400, f"YAML syntax error: {e}")
        if not isinstance(raw, dict):
            raise HTTPException(400, "project file must be a YAML mapping")
        ws.write_raw(pid, raw)
        return project_payload(pid)

    @app.post("/api/projects/{pid}/estimate")
    async def estimate(pid: str, payload: dict = Body(...)):
        d = ws.project_dir(pid)
        raw = ws.read_raw(pid)
        if payload.get("editor"):
            raw = EM.from_editor(payload["editor"], raw)
        v = validate_raw(raw, d)
        if not v["ok"]:
            raise HTTPException(400, "; ".join(v["errors"]))
        try:
            return await run_in_threadpool(estimate_for, raw, d, payload.get("preset"), payload.get("lattice"),
                                           payload.get("dx"))
        except Exception as e:                            # noqa: BLE001 (tank.build errors are user errors)
            raise HTTPException(400, _err(e))

    # ---------------------------------------------------------------- rockscape designer
    @app.post("/api/projects/{pid}/meshes")
    async def upload_mesh(pid: str, file: UploadFile = File(...), unit: str = Form("mm")):
        d = ws.project_dir(pid)
        name = Path(file.filename or "mesh.stl").name
        ext = Path(name).suffix.lower()
        if ext not in MESH_EXT:
            raise HTTPException(400, "only STL, OBJ or PLY meshes can be imported")
        if U.kind(unit) != "length":
            raise HTTPException(400, f"{unit!r} is not a length unit")
        stem = slugify(Path(name).stem)
        target = d / "meshes" / f"{stem}{ext}"
        i = 2
        while target.exists():
            target = d / "meshes" / f"{stem}_{i}{ext}"
            i += 1
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(await file.read())
        from .. import geometry as G
        try:
            v, f = await run_in_threadpool(G.load_mesh, target, unit)
            rep = G.mesh_report(v, f)
        except Exception as e:                            # noqa: BLE001
            target.unlink(missing_ok=True)
            raise HTTPException(400, f"cannot read {name}: {_err(e)}")
        rel = target.relative_to(d).as_posix()
        lo, hi = np.asarray(rep["bounds_m"])
        return {"element": {"type": "mesh", "name": name, "params": {"file": rel, "unit": unit}},
                "report": rep, "bounds_m": [lo.tolist(), hi.tolist()],
                "warning": None if rep["watertight"] else "mesh is not watertight: it may voxelize incorrectly"}

    @app.post("/api/projects/{pid}/element_mesh")
    async def element_mesh(pid: str, payload: dict = Body(...)):
        d = ws.project_dir(pid)
        el = payload.get("element") or {}
        try:
            v, f = await run_in_threadpool(RG.element_mesh, el, d)
        except Exception as e:                            # noqa: BLE001
            raise HTTPException(400, _err(e))
        return Response(content=RG.pack_mesh(v, f), media_type="application/octet-stream")

    @app.post("/api/projects/{pid}/rockscape/build")
    async def build_rockscape(pid: str, payload: dict = Body(...)):
        d = ws.project_dir(pid)
        ed = payload.get("editor")
        if not isinstance(ed, dict):
            raise HTTPException(400, "body must be {editor: {...}, resolution}")
        old = ws.read_raw(pid)
        raw = EM.from_editor(ed, old)
        try:
            P = PR.parse({**raw, "rockscape": []}, d / "project.yaml")
        except Exception as e:                            # noqa: BLE001
            raise HTTPException(400, f"fix the project first: {_err(e)}")
        dz = raw.get("designer") or {}
        els = dz.get("elements") or []
        h = float(payload.get("resolution") or dz.get("resolution") or 0.004)
        try:
            res = await run_in_threadpool(RG.union, els, P.tank, d, h)
        except Exception as e:                            # noqa: BLE001
            raise HTTPException(400, _err(e))
        out = d / "rockscape.stl"
        tmp = d / "rockscape.tmp.stl"
        RG.save_union_stl(tmp, res)
        tmp.replace(out)
        dz.update(resolution=h, rockscape_file="rockscape.stl", built_hash=EM.elements_hash(els))
        raw["designer"] = dz
        raw["rockscape"] = [{"file": "rockscape.stl", "unit": "mm", "translate": ["0 mm", "0 mm", "0 mm"]}]
        ws.write_raw(pid, raw)
        rep = {k: v for k, v in res.items() if k not in ("vertices", "faces")}
        return {"report": rep, "project": project_payload(pid)}

    @app.post("/api/projects/{pid}/rockscape/clear")
    def clear_rockscape(pid: str, payload: dict = Body(...)):
        """Remove the rockscape entry (the designer elements stay)."""
        old = ws.read_raw(pid)
        raw = EM.from_editor(payload["editor"], old) if payload.get("editor") else old
        raw["rockscape"] = []
        if raw.get("designer"):
            raw["designer"]["built_hash"] = None
        ws.write_raw(pid, raw)
        return project_payload(pid)

    @app.get("/api/projects/{pid}/rockscape.stl")
    def export_rockscape(pid: str):
        f = ws.project_dir(pid) / "rockscape.stl"
        if not f.is_file():
            raise HTTPException(404, "no rockscape built yet: press 'Build rockscape' first")
        return FileResponse(f, media_type="model/stl", filename=f"{pid}_rockscape.stl")

    @app.get("/api/projects/{pid}/rock_mesh")
    async def project_rock_mesh(pid: str):
        return await run_in_threadpool(rock_mesh_from_raw, ws.read_raw(pid), ws.project_dir(pid))

    # ---------------------------------------------------------------- runs
    @app.get("/api/projects/{pid}/runs")
    def project_runs(pid: str):
        return {"runs": _live(ws.list_runs(pid))}

    def _live(runs):
        live = {(j["project"], j["run_id"]): j for j in jobs.list() if j["status"] not in TERMINAL}
        for r in runs:
            j = live.get((r["project"], r["id"]))
            if j:
                r["status"], r["job_id"], r["pct"] = j["status"], j["id"], j["pct"]
        return runs

    @app.post("/api/projects/{pid}/runs")
    def start_run(pid: str, payload: dict = Body(...)):
        d = ws.project_dir(pid)
        raw = ws.read_raw(pid)
        v = validate_raw(raw, d)
        if not v["ok"]:
            raise HTTPException(400, "the project has errors: " + "; ".join(v["errors"]))
        if EM.designer_stale(raw):
            raise HTTPException(409, "the rockscape designer has unbuilt changes: press 'Build rockscape' "
                                     "(or undo the changes) before running")
        preset = payload.get("preset") or raw.get("simulation", {}).get("preset", "standard")
        if preset not in PR.PRESETS:
            raise HTTPException(400, f"unknown preset {preset!r}")
        lattice = (payload.get("lattice") or "D3Q19").upper()
        if lattice not in ("D3Q19", "D3Q27"):
            raise HTTPException(400, "lattice must be D3Q19 or D3Q27")
        dx = payload.get("dx")
        if dx is not None and not 0.0015 <= float(dx) <= 0.05:
            raise HTTPException(400, "custom cell size must be between 1.5 and 50 mm")
        lighting = bool(payload.get("lighting")) and bool(raw.get("lighting"))
        label = payload.get("label") or f"{preset if not dx else f'dx{float(dx) * 1e3:g}mm'}_{lattice}"
        rid, rdir = ws.new_run(pid, label)
        try:
            pf = ws.snapshot_project(pid, rdir, raw)
        except Exception as e:                            # noqa: BLE001
            shutil.rmtree(rdir, ignore_errors=True)
            raise HTTPException(400, _err(e))
        spec = {"project_file": str(pf), "preset": preset, "lattice": lattice, "dx": float(dx) if dx else None,
                "spinup": payload.get("spinup"), "average": payload.get("average"), "lighting": lighting,
                "arch": payload.get("arch") or "auto", "analyze": payload.get("analyze", True),
                "rays": payload.get("rays")}
        job = jobs.submit("run", pid, rid, rdir, spec, label=label)
        return {"job": job.public(), "run_id": rid}

    @app.get("/api/runs")
    def all_runs():
        return {"runs": _live(ws.all_runs())}

    @app.get("/api/runs/{pid}/{rid}")
    def run_summary(pid: str, rid: str):
        d = ws.run_dir(pid, rid)
        out = RS.summary(d)
        from .workspace import run_status
        out["status"] = _live([run_status(d, pid)])[0]
        return out

    @app.delete("/api/runs/{pid}/{rid}")
    def delete_run(pid: str, rid: str):
        d = ws.run_dir(pid, rid)
        if jobs.active_for_run(pid, rid):
            raise HTTPException(409, "cancel the job first")
        shutil.rmtree(d)
        return {"ok": True}

    @app.post("/api/runs/{pid}/{rid}/analyze")
    def reanalyze(pid: str, rid: str):
        d = ws.run_dir(pid, rid)
        if not (d / "fields.npz").is_file():
            raise HTTPException(400, "this run has no flow results to analyze")
        if jobs.active_for_run(pid, rid):
            raise HTTPException(409, "a job for this run is already queued or running")
        job = jobs.submit("analyze", pid, rid, d, {}, label=f"analyze {rid}")
        return {"job": job.public()}

    @app.get("/api/runs/{pid}/{rid}/project")
    def run_project(pid: str, rid: str):
        d = ws.run_dir(pid, rid)
        raw = yaml.safe_load((d / "project.yaml").read_text(encoding="utf-8"))
        return {"editor": EM.to_editor(raw)}

    def _need_fields(d: Path, analysis: bool = False):
        if not (d / "fields.npz").is_file():
            raise HTTPException(404, "this run has no flow results yet")
        if analysis and not (d / "analysis.npz").is_file():
            raise HTTPException(404, "this run has not been analysed yet")

    @app.get("/api/runs/{pid}/{rid}/slice")
    async def get_slice(pid: str, rid: str, field: str = "speed", axis: str = "z", pos: float = 0.1):
        d = ws.run_dir(pid, rid)
        _need_fields(d, analysis=field in ("age", "dead"))
        if axis not in ("x", "y", "z"):
            raise HTTPException(400, "axis must be x, y or z")
        if field not in RS.FIELDS:
            raise HTTPException(400, f"unknown field {field!r}; choose from {sorted(RS.FIELDS)}")
        try:
            s = await run_in_threadpool(RS.slice_field, d, field, axis, pos)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))
        data = s["data"]
        fin = data[np.isfinite(data)]
        meta = {"field": field, "axis": axis, "shape": list(data.shape), "extent": s["extent"], "index": s["index"],
                "pos": s["pos"], "axes": s["axes"], "dx": s["dx"], "unit": RS.FIELDS[field][1],
                "min": float(fin.min()) if fin.size else None, "max": float(fin.max()) if fin.size else None,
                "p99": float(np.percentile(fin, 99)) if fin.size else None,
                "p1": float(np.percentile(fin, 1)) if fin.size else None}
        return _binary([data], meta)

    @app.get("/api/runs/{pid}/{rid}/vectors")
    async def get_vectors(pid: str, rid: str, axis: str = "z", pos: float = 0.1, stride: int | None = None):
        d = ws.run_dir(pid, rid)
        _need_fields(d)
        if axis not in ("x", "y", "z"):
            raise HTTPException(400, "axis must be x, y or z")
        s = await run_in_threadpool(RS.slice_vectors, d, axis, pos, stride)
        meta = {"n": int(len(s["points"])), "axes": s["axes"], "index": s["index"], "pos": s["pos"],
                "stride": s["stride"]}
        return _binary([s["points"], s["vectors"]], meta)

    @app.get("/api/runs/{pid}/{rid}/sand_shear")
    async def get_sand(pid: str, rid: str):
        d = ws.run_dir(pid, rid)
        _need_fields(d, analysis=True)
        s = await run_in_threadpool(RS.sand_shear, d)
        data = s["data"]
        fin = data[np.isfinite(data)]
        an = json.loads((d / "analysis.json").read_text(encoding="utf-8")) if (d / "analysis.json").is_file() else {}
        wsh = an.get("wall_shear") or {}
        meta = {"shape": list(data.shape), "extent": s["extent"], "z": s["z"], "unit": "Pa",
                "min": float(fin.min()) if fin.size else None, "max": float(fin.max()) if fin.size else None,
                "p99": float(np.percentile(fin, 99)) if fin.size else None,
                "tau_crit_Pa": wsh.get("tau_crit_Pa"), "tau_sand_move_Pa": wsh.get("tau_sand_move_Pa")}
        return _binary([data], meta)

    @app.get("/api/runs/{pid}/{rid}/scores")
    async def get_scores(pid: str, rid: str, profile: str = "SPS"):
        d = ws.run_dir(pid, rid)
        _need_fields(d, analysis=True)
        try:
            s = await run_in_threadpool(RS.surface_scores, d, profile)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))
        return _binary([s["points"], s["score"], s["speed"], s["ti"]], {"n": int(len(s["score"])), "profile": profile})

    @app.get("/api/runs/{pid}/{rid}/rock_mesh")
    async def run_rock_mesh(pid: str, rid: str):
        d = ws.run_dir(pid, rid)
        raw = yaml.safe_load((d / "project.yaml").read_text(encoding="utf-8"))
        return await run_in_threadpool(rock_mesh_from_raw, raw, d)

    @app.get("/api/runs/{pid}/{rid}/file/{name}")
    def run_file(pid: str, rid: str, name: str):
        d = ws.run_dir(pid, rid)
        check_id(name, "file name")
        f = d / name
        if f.suffix.lower() not in RUN_FILE_EXT or not f.is_file():
            raise HTTPException(404, f"no file {name!r} in this run")
        return FileResponse(f, headers={"Cache-Control": "no-store"})

    # ---------------------------------------------------------------- comparison
    @app.post("/api/compare")
    async def compare(payload: dict = Body(...)):
        from ..analysis.compare import compare as core_compare
        (pa, ra), (pb, rb) = payload["a"], payload["b"]
        da, db = ws.run_dir(pa, ra), ws.run_dir(pb, rb)
        for dd in (da, db):
            _need_fields(dd)
        name = f"{slugify(pa)}-{slugify(ra)}__vs__{slugify(pb)}-{slugify(rb)}"[:150]
        out = ws.root / "comparisons" / name
        la, lb = payload.get("label_a") or "A", payload.get("label_b") or "B"
        try:
            res = await run_in_threadpool(core_compare, da, db, out, la, lb)
        except Exception as e:                            # noqa: BLE001
            raise HTTPException(400, f"comparison failed: {_err(e)}")
        return {"name": name, "result": res, "html": f"/api/comparisons/{name}/comparison.html",
                "image": f"/api/comparisons/{name}/difference.png"}

    @app.get("/api/comparisons/{name}/{file}")
    def comparison_file(name: str, file: str):
        check_id(name, "comparison", max_len=200)
        check_id(file, "file name")
        f = ws.root / "comparisons" / name / file
        if not f.is_file() or f.suffix.lower() not in (".html", ".png", ".json"):
            raise HTTPException(404, "not found")
        return FileResponse(f, headers={"Cache-Control": "no-store"})

    # ---------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": jobs.list(), "photo_jobs": photo_jobs.list()}

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return find_job(jid).public(with_lines=True)

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        return (photo_jobs if jid.startswith("p") else jobs).cancel(jid).public()

    @app.get("/api/jobs/{jid}/events")
    async def job_events(jid: str, since: int = 0):
        job = find_job(jid)

        async def gen():
            i = max(0, int(since))
            last = None
            idle = 0.0
            while True:
                lines = job.lines[i:]
                i += len(lines)
                state = (job.status, round(job.pct, 1), job.phase, job.error)
                if lines or state != last or idle > 15:
                    payload = {"lines": lines, "next": i, "status": job.status, "pct": round(job.pct, 1),
                               "phase": job.phase, "error": job.error, "info": job.status_info}
                    yield f"data: {json.dumps(payload)}\n\n"
                    last, idle = state, 0.0
                if job.status in TERMINAL:
                    yield "event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.5)
                idle += 0.5

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    from .photo_routes import register as register_photo_routes
    register_photo_routes(app, ws, photo_jobs, _err)
    from .sculpt_routes import register as register_sculpt_routes
    register_sculpt_routes(app, ws, _err, project_payload)
    from .scan_routes import register as register_scan_routes
    register_scan_routes(app, ws, _err)
    from .phone import register as register_phone_routes
    phone_link = register_phone_routes(app, ws)

    @app.exception_handler(Exception)
    async def _any(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": _err(exc), "traceback": traceback.format_exc()})

    return app
