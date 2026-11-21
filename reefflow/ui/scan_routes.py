"""HTTP routes of the phone-scan clean-up (Rockscape tab -> "Import phone scan"). Logic: reefflow.scan.

    POST   /api/projects/{pid}/scans                 multipart file (.obj .glb .gltf .ply .stl, or a .zip holding one)
                                                     -> {scan, info}
    POST   /api/projects/{pid}/scans/{sid}/preview   {unit, scale, up} -> binary mesh (decimated, upright, m), X-Meta = info
    POST   /api/projects/{pid}/scans/{sid}/clean     {unit, scale, up, ground, dx, keep}
                                                     -> watertight STL in meshes/, {element, report, bounds_m}
    DELETE /api/projects/{pid}/scans/{sid}

Raw uploads live in projects/<pid>/scans/<sid>/ (kept, so the clean-up can be redone).
"""
from __future__ import annotations

import json
import secrets
import shutil
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
from fastapi import Body, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from .. import scan as SC
from .workspace import check_id, slugify

UPLOAD_EXT = SC.SCAN_EXT + (".zip",)
MAX_BYTES = 1_000_000_000            # 1 GB, raw or unzipped
_CACHE: dict = {}                    # (path, mtime) -> (v, f, preview v, preview f)
_LOCK = threading.Lock()


def _scan_dir(pdir: Path, sid: str) -> Path:
    check_id(sid, "scan id")
    return pdir / "scans" / sid


def _meta(d: Path) -> dict:
    f = d / "meta.json"
    if not f.is_file():
        raise HTTPException(404, "no such scan (upload it again)")
    return json.loads(f.read_text())


def _loaded(d: Path, meta: dict):
    p = d / meta["raw"]
    key = (str(p), p.stat().st_mtime)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    v, f = SC.load_raw(p)
    pv, pf = SC.decimate_preview(v, f)
    with _LOCK:
        if len(_CACHE) > 3:
            _CACHE.clear()
        _CACHE[key] = (v, f, pv, pf)
    return _CACHE[key]


def _from_zip(zpath: Path, d: Path) -> Path:
    """Extract the one mesh file of a scan-app ZIP export (glb > obj > ply > stl > gltf)."""
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in SC.SCAN_EXT
                   and not Path(m.filename).name.startswith(".")]
        if not members:
            raise ValueError("the ZIP holds no OBJ, GLB, PLY or STL mesh")
        rank = {".glb": 0, ".obj": 1, ".ply": 2, ".stl": 3, ".gltf": 4}
        m = min(members, key=lambda m: (rank[Path(m.filename).suffix.lower()], -m.file_size))
        if m.file_size > MAX_BYTES:
            raise ValueError("the mesh in the ZIP is larger than 1 GB")
        out = d / f"raw{Path(m.filename).suffix.lower()}"
        with z.open(m) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    return out


def _params(body: dict) -> dict:
    up = body.get("up") or "auto"
    unit = body.get("unit") or "auto"
    try:
        scale = float(body.get("scale") or 1.0)
    except (TypeError, ValueError):
        raise HTTPException(400, "scale must be a number")
    return {"unit": unit, "scale": scale, "up": up}


def register(app, ws, err):
    @app.post("/api/projects/{pid}/scans")
    async def upload_scan(pid: str, file: UploadFile = File(...)):
        pdir = ws.project_dir(pid)
        name = Path(file.filename or "scan.obj").name
        ext = Path(name).suffix.lower()
        if ext not in UPLOAD_EXT:
            raise HTTPException(400, "scans must be OBJ, GLB, glTF, PLY or STL (or a ZIP export holding one). "
                                     "USDZ is not supported: export OBJ or GLB from the scan app")
        sid = f"{slugify(Path(name).stem)[:40]}_{secrets.token_hex(3)}"
        d = _scan_dir(pdir, sid)
        d.mkdir(parents=True)
        tmp = d / f"upload{ext}"
        with open(tmp, "wb") as fh:
            await run_in_threadpool(shutil.copyfileobj, file.file, fh, 1 << 20)
        try:
            if tmp.stat().st_size > MAX_BYTES:
                raise ValueError("the file is larger than 1 GB")
            if ext == ".zip":
                raw = await run_in_threadpool(_from_zip, tmp, d)
                tmp.unlink()
            else:
                raw = tmp.rename(d / f"raw{ext}")
            meta = {"id": sid, "name": name, "raw": raw.name, "ext": raw.suffix.lower(), "time": time.time()}
            (d / "meta.json").write_text(json.dumps(meta))
            v, f, _, _ = await run_in_threadpool(_loaded, d, meta)
            info = await run_in_threadpool(SC.analyze, v, f, meta["ext"])
        except Exception as e:                                # noqa: BLE001 (bad files are user errors)
            shutil.rmtree(d, ignore_errors=True)
            msg = err(e)
            if ext in (".gltf", ".zip"):
                msg += " (a .gltf file needs its separate .bin file: export GLB instead)"
            raise HTTPException(400, f"cannot read {name}: {msg}")
        return {"scan": meta, "info": info}

    @app.post("/api/projects/{pid}/scans/{sid}/preview")
    async def scan_preview(pid: str, sid: str, body: dict = Body(default={})):
        d = _scan_dir(ws.project_dir(pid), sid)
        meta = _meta(d)
        p = _params(body)
        try:
            v, f, pv, pf = await run_in_threadpool(_loaded, d, meta)
            info = await run_in_threadpool(SC.analyze, v, f, meta["ext"], p["unit"], p["scale"], p["up"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        s = SC.U.to_si(1.0, info["unit"]) * p["scale"]
        vo = SC.orient(pv * s, info["up"]).astype("<f4")
        body_ = (np.array([len(vo), len(pf)], "<u4").tobytes() + np.ascontiguousarray(vo).tobytes()
                 + np.ascontiguousarray(pf, dtype="<u4").tobytes())
        info["preview_triangles"] = int(len(pf))
        return Response(content=body_, media_type="application/octet-stream",
                        headers={"X-Meta": json.dumps(info, separators=(",", ":")), "Cache-Control": "no-store"})

    @app.post("/api/projects/{pid}/scans/{sid}/clean")
    async def scan_clean(pid: str, sid: str, body: dict = Body(default={})):
        pdir = ws.project_dir(pid)
        d = _scan_dir(pdir, sid)
        meta = _meta(d)
        p = _params(body)
        ground = body.get("ground", "auto")
        if ground not in ("auto", "none"):
            try:
                ground = float(ground)
            except (TypeError, ValueError):
                raise HTTPException(400, "ground must be auto, none or a height in metres")
        dx = body.get("dx")
        keep = body.get("keep") or "pieces"
        if keep not in ("main", "pieces"):
            raise HTTPException(400, "keep must be main or pieces")
        try:
            v, f, _, _ = await run_in_threadpool(_loaded, d, meta)
            res = await run_in_threadpool(lambda: SC.clean(v, f, meta["ext"], ground=ground,
                                                           dx=float(dx) if dx else None, keep=keep, **p))
        except ValueError as e:
            raise HTTPException(400, str(e))
        rep = res["report"]
        if not rep["watertight"]:
            raise HTTPException(500, "the cleaned mesh is not watertight (please report this scan)")
        stem = slugify(Path(meta["name"]).stem)[:40] + "_scan"
        target = pdir / "meshes" / f"{stem}.stl"
        i = 2
        while target.exists():
            target = pdir / "meshes" / f"{stem}_{i}.stl"
            i += 1
        target.parent.mkdir(exist_ok=True)
        from .. import geometry as G
        await run_in_threadpool(G.save_stl, target, res["vertices"], res["faces"], "mm")
        lo, hi = res["vertices"].min(axis=0), res["vertices"].max(axis=0)
        rep["file"] = target.relative_to(pdir).as_posix()
        rep["settings"] = {**p, "ground": ground, "dx": dx, "keep": keep}
        (d / "last_clean.json").write_text(json.dumps(rep, default=float))
        label = Path(meta["name"]).stem + " (scan)"
        return {"element": {"type": "mesh", "name": label, "params": {"file": rep["file"], "unit": "mm"}},
                "report": rep, "bounds_m": [lo.tolist(), hi.tolist()]}

    @app.delete("/api/projects/{pid}/scans/{sid}")
    def delete_scan(pid: str, sid: str):
        d = _scan_dir(ws.project_dir(pid), sid)
        shutil.rmtree(d, ignore_errors=True)
        return {"deleted": sid}
