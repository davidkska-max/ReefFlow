"""HTTP routes of the Photos tab (photo reconstruction, spec Feature 1). Logic: reefflow.ui.photos
-> reefflow.photo.

    GET    /api/projects/{pid}/photos                          session state (views, calibrations, recon, ...)
    GET    /api/photo/capabilities                             segmentation models (SAM / GrabCut) and stubs
    POST   /api/projects/{pid}/photos/views                    multipart: file + kind -> new photo
    DELETE /api/projects/{pid}/photos/views/{vid}
    GET    /api/projects/{pid}/photos/views/{vid}/image        working image (JPEG; all pixel coordinates refer to it)
    PUT    /api/projects/{pid}/photos/views/{vid}/clicks       {clicks, custom?, focal?} -> calibration + overlay
    GET    /api/projects/{pid}/photos/views/{vid}/overlay      reprojected reference points + tank edges
    POST   /api/projects/{pid}/photos/segment                  {views?: [vid], method?, cue?, sensitivity?} -> automatic mask suggestion
    POST   /api/projects/{pid}/photos/views/{vid}/sensitivity  {sensitivity} -> move the outline (fast)
    GET    /api/projects/{pid}/photos/views/{vid}/mask.png
    PUT    /api/projects/{pid}/photos/views/{vid}/mask         multipart PNG upload (replace the mask)
    POST   /api/projects/{pid}/photos/views/{vid}/mask_ops     {ops: [{op: brush|polygon|fill_holes|remove_small|smooth|clear|invert, ...}]}
    POST   /api/projects/{pid}/photos/views/{vid}/mask_undo
    PUT    /api/projects/{pid}/photos/settings                 {dx?, tolerance_px?, bulk_density?, solid_fraction?}
    POST   /api/projects/{pid}/photos/carve                    -> background job (progress via /api/jobs/{id}/events)
    GET    /api/projects/{pid}/photos/recon/mesh               binary mesh of the edited reconstruction
    PUT    /api/projects/{pid}/photos/edits                    {edits: [...]} (reefflow.photo.edit schema) -> volume
    POST   /api/projects/{pid}/photos/compare                  side-by-side images + silhouette IoU per view
    GET    /api/projects/{pid}/photos/recon/compare/{vid}      comparison image (JPEG)
    POST   /api/projects/{pid}/photos/volume                   {bulk_density?, solid_fraction?} -> volume / weight report
    GET    /api/projects/{pid}/photos/recon.stl                download the reconstruction (STL, mm)
    POST   /api/projects/{pid}/photos/use                      write the STL into the project's rockscape
"""
from __future__ import annotations

import copy

from fastapi import Body, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from . import editor_model as EM
from . import photos as PH
from . import rockgen as RG
from .workspace import check_id

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
PHOTO_STL = "rockscape_photo.stl"


def register(app, ws, photo_jobs, err):
    from .. import project as PR

    def sdir(pid):
        d = PH.session_dir(ws.project_dir(pid))
        raw = ws.read_raw(pid)
        try:
            P = PR.parse({**raw, "rockscape": [], "pumps": []}, ws.project_path(pid))
        except Exception as e:                            # noqa: BLE001
            raise HTTPException(400, f"fix the tank dimensions first: {err(e)}")
        PH.sync_tank(d, P.tank)
        return d

    def vcheck(d, vid):
        check_id(vid, "photo id")
        if vid not in PH.load(d)["views"]:
            raise HTTPException(404, f"no photo {vid!r}")
        return vid

    def payload(d, pid):
        st = PH.state(d)
        live = [j for j in photo_jobs.list() if j["project"] == pid and j["status"] in ("queued", "running")]
        st["job"] = live[0] if live else None
        return st

    @app.get("/api/photo/capabilities")
    async def photo_capabilities():
        from reefflow.photo import capabilities
        return await run_in_threadpool(capabilities)

    @app.get("/api/projects/{pid}/photos")
    async def get_photos(pid: str):
        d = sdir(pid)
        return await run_in_threadpool(payload, d, pid)

    @app.post("/api/projects/{pid}/photos/views")
    async def add_photo(pid: str, file: UploadFile = File(...), kind: str = Form(...)):
        d = sdir(pid)
        name = file.filename or "photo.jpg"
        if not name.lower().endswith(IMG_EXT):
            raise HTTPException(400, "photos must be JPEG or PNG (HEIC is not supported: export as JPEG)")
        data = await file.read()
        try:
            v = await run_in_threadpool(PH.add_view, d, kind, name, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"view": v, "state": payload(d, pid)}

    @app.delete("/api/projects/{pid}/photos/views/{vid}")
    def delete_photo(pid: str, vid: str):
        d = sdir(pid)
        PH.delete_view(d, vcheck(d, vid))
        return payload(d, pid)

    @app.get("/api/projects/{pid}/photos/views/{vid}/image")
    def photo_image(pid: str, vid: str):
        d = PH.session_dir(ws.project_dir(pid))
        vcheck(d, vid)
        return FileResponse(d / vid / "image.jpg", media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.put("/api/projects/{pid}/photos/views/{vid}/clicks")
    async def put_clicks(pid: str, vid: str, body: dict = Body(...)):
        d = sdir(pid)
        vcheck(d, vid)
        try:
            return await run_in_threadpool(PH.set_clicks, d, vid, body.get("clicks"), body.get("custom"),
                                           body.get("focal"), bool(body.get("solve", True)), body.get("size_ok"))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/projects/{pid}/photos/views/{vid}/overlay")
    async def get_overlay(pid: str, vid: str):
        d = sdir(pid)
        vcheck(d, vid)
        return {"overlay": await run_in_threadpool(PH.get_overlay, d, vid)}

    @app.post("/api/projects/{pid}/photos/segment")
    async def segment(pid: str, body: dict = Body(...)):
        d = sdir(pid)
        vids = body.get("views")
        for v in vids or []:
            vcheck(d, v)
        try:
            res = await run_in_threadpool(PH.auto_segment, d, vids, body.get("method", "auto"),
                                          not body.get("overwrite_edited", False), body.get("cue", "auto"),
                                          float(body.get("sensitivity", 50)))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"result": res, "state": payload(d, pid)}

    @app.post("/api/projects/{pid}/photos/views/{vid}/sensitivity")
    async def mask_sensitivity(pid: str, vid: str, body: dict = Body(...)):
        """{sensitivity: 0-100} -> move the last suggestion's outline (fast; 409 if none cached)."""
        d = sdir(pid)
        vcheck(d, vid)
        try:
            view = await run_in_threadpool(PH.set_sensitivity, d, vid, float(body.get("sensitivity", 50)))
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"view": view}

    @app.get("/api/projects/{pid}/photos/views/{vid}/mask.png")
    def get_mask(pid: str, vid: str):
        d = PH.session_dir(ws.project_dir(pid))
        vcheck(d, vid)
        p = PH.mask_path(d, vid)
        if not p.is_file():
            raise HTTPException(404, "no mask yet")
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.put("/api/projects/{pid}/photos/views/{vid}/mask")
    async def put_mask(pid: str, vid: str, file: UploadFile = File(...)):
        from reefflow.photo import mask_from_png
        d = sdir(pid)
        vcheck(d, vid)
        try:
            m = mask_from_png(await file.read())
            return {"view": PH.write_mask(d, vid, m, method="user", edited=True)}
        except (ValueError, IOError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/projects/{pid}/photos/views/{vid}/mask_ops")
    async def post_mask_ops(pid: str, vid: str, body: dict = Body(...)):
        d = sdir(pid)
        vcheck(d, vid)
        try:
            return {"view": await run_in_threadpool(PH.mask_ops, d, vid, body.get("ops") or [])}
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/projects/{pid}/photos/views/{vid}/mask_undo")
    def post_mask_undo(pid: str, vid: str):
        d = sdir(pid)
        vcheck(d, vid)
        try:
            return {"view": PH.undo_mask(d, vid)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.put("/api/projects/{pid}/photos/settings")
    def put_settings(pid: str, body: dict = Body(...)):
        d = sdir(pid)
        keys = ("dx", "tolerance_px", "bulk_density", "solid_fraction", "min_component_voxels")
        try:
            PH.update_settings(d, **{k: body.get(k) for k in keys})
        except ValueError as e:
            raise HTTPException(400, str(e))
        return payload(d, pid)

    @app.post("/api/projects/{pid}/photos/carve")
    def carve(pid: str):
        d = sdir(pid)
        s = PH.load(d)
        if not PH.ready_views(d, s):
            raise HTTPException(400, "no photo is both calibrated and masked yet")
        if any(j["project"] == pid and j["status"] in ("queued", "running") for j in photo_jobs.list()):
            raise HTTPException(409, "a carving job for this project is already queued or running")
        (d / "recon").mkdir(exist_ok=True)
        job = photo_jobs.submit("carve", pid, "photos", d / "recon", {"session_dir": str(d)}, label="photo carving")
        return {"job": job.public()}

    @app.get("/api/projects/{pid}/photos/recon/mesh")
    async def recon_mesh(pid: str):
        d = PH.session_dir(ws.project_dir(pid))
        try:
            v, f = await run_in_threadpool(PH.mesh, d)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return Response(content=RG.pack_mesh(v, f), media_type="application/octet-stream")

    @app.put("/api/projects/{pid}/photos/edits")
    async def put_edits(pid: str, body: dict = Body(...)):
        d = sdir(pid)
        try:
            vol = await run_in_threadpool(PH.set_edits, d, list(body.get("edits") or []))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, f"bad edit: {e}")
        return {"volume": vol, "edits": PH.load(d)["edits"]}

    @app.post("/api/projects/{pid}/photos/compare")
    async def compare(pid: str):
        d = sdir(pid)
        try:
            return await run_in_threadpool(PH.comparison, d)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/projects/{pid}/photos/recon/compare/{vid}")
    def compare_image(pid: str, vid: str):
        d = PH.session_dir(ws.project_dir(pid))
        check_id(vid, "photo id")
        f = d / "recon" / f"compare_{vid}.jpg"
        if not f.is_file():
            raise HTTPException(404, "no comparison image yet")
        return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/projects/{pid}/photos/volume")
    async def post_volume(pid: str, body: dict = Body(...)):
        d = sdir(pid)
        bd, sf = body.get("bulk_density"), body.get("solid_fraction")
        if bd is not None and not 300 <= float(bd) <= 3000:
            raise HTTPException(400, "bulk density must be 300-3000 kg/m3")
        if sf is not None and not 0 < float(sf) <= 1:
            raise HTTPException(400, "solid fraction must be in (0, 1]")
        try:
            return await run_in_threadpool(PH.volume, d, bd, sf)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/projects/{pid}/photos/recon.stl")
    async def recon_stl(pid: str):
        d = PH.session_dir(ws.project_dir(pid))
        try:
            await run_in_threadpool(PH.export, d, d / "recon" / "recon.stl")
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(404, str(e))
        return FileResponse(d / "recon" / "recon.stl", media_type="model/stl", filename=f"{pid}_photo_rock.stl")

    @app.post("/api/projects/{pid}/photos/use")
    async def use_as_rockscape(pid: str):
        """Write the edited reconstruction as rockscape_photo.stl and make it the project's rockscape.
        The designer then holds it as one 'mesh' element (already built), so designer pieces can be
        added on top and the union rebuilt."""
        pdir = ws.project_dir(pid)
        d = PH.session_dir(pdir)
        try:
            ex = await run_in_threadpool(PH.export, d, pdir / PHOTO_STL)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, str(e))
        raw = ws.read_raw(pid)
        raw["rockscape"] = [ex["rockscape_entry"]]
        els = [{"id": "photo", "type": "mesh", "name": "photo reconstruction", "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "params": {"file": PHOTO_STL, "unit": "mm"}}]
        dz = copy.deepcopy(raw.get("designer") or {})
        dz.update(elements=els, rockscape_file=None, built_hash=EM.elements_hash(els))
        dz.setdefault("resolution", 0.004)
        dz.setdefault("rock_density", 1600.0)
        raw["designer"] = dz
        ws.write_raw(pid, raw)
        with PH._lock(d):
            s = PH.load(d)
            s["used_as_rockscape"] = {"time": __import__("time").time(), "file": PHOTO_STL}
            PH.save(d, s)
        return {"export": ex, "note": "The project's rockscape is now the photo reconstruction "
                                      f"({PHOTO_STL}); previous designer elements were replaced."}
