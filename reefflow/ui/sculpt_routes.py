"""HTTP routes of the rock sculpt tool (see sculpt.py).

    POST /api/projects/{pid}/sculpt/start   {dx?, fresh?} -> uint8 grid bytes, X-Meta = grid + source
    PUT  /api/projects/{pid}/sculpt/draft?dx=&base_sig=    body = grid bytes -> saved draft
    POST /api/projects/{pid}/sculpt/apply?dx=              body = grid bytes -> STL + project rockscape
"""
from __future__ import annotations

import json

from fastapi import Body, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from .. import project as PR
from . import sculpt as SC


def _tank(raw: dict, pdir):
    """The tank alone (pumps / rock / overflow problems must not block sculpting)."""
    r = {**raw, "rockscape": [], "pumps": [], "overflow": None}
    try:
        return PR.parse(r, pdir / "project.yaml").tank
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, f"fix the tank dimensions first: {e}")


def register(app, ws, err, project_payload):
    @app.post("/api/projects/{pid}/sculpt/start")
    async def sculpt_start(pid: str, payload: dict = Body(default={})):
        pdir = ws.project_dir(pid)
        raw = ws.read_raw(pid)
        tank = _tank(raw, pdir)
        dx = payload.get("dx")
        try:
            D, info = await run_in_threadpool(SC.start, raw, pdir, tank, float(dx) if dx else None,
                                              bool(payload.get("fresh")))
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, err(e))
        g = info["grid"]
        info["rock_volume_m3"] = float((D[:, :, g["k_sand"]:] >= SC.ISO).sum() * g["dx"] ** 3)
        return Response(content=SC.to_wire(D), media_type="application/octet-stream",
                        headers={"X-Meta": json.dumps(info, separators=(",", ":")), "Cache-Control": "no-store"})

    async def _field(request: Request, pid: str, dx: float):
        pdir = ws.project_dir(pid)
        raw = ws.read_raw(pid)
        tank = _tank(raw, pdir)
        try:
            g = SC.grid_for(tank, dx)
            D = SC.from_wire(await request.body(), g["shape"])
        except ValueError as e:
            raise HTTPException(400, err(e))
        return pdir, raw, tank, g, D

    @app.put("/api/projects/{pid}/sculpt/draft")
    async def sculpt_draft(pid: str, request: Request, dx: float, base_sig: str = ""):
        pdir, raw, _tank_, g, D = await _field(request, pid, dx)
        st = await run_in_threadpool(SC.save_field, pdir, D, g, base_sig, draft=True)
        return {"saved": st["time"]}

    @app.post("/api/projects/{pid}/sculpt/apply")
    async def sculpt_apply(pid: str, request: Request, dx: float):
        pdir, raw, tank, g, D = await _field(request, pid, dx)
        try:
            new_raw, rep = await run_in_threadpool(SC.apply, D, g, tank, pdir, raw)
        except ValueError as e:
            raise HTTPException(400, err(e))
        ws.write_raw(pid, new_raw)
        sig = SC.rockscape_signature(new_raw, pdir)
        await run_in_threadpool(SC.save_field, pdir, D, g, sig, draft=False)
        return {"report": rep, "base_sig": sig, "project": project_payload(pid)}
