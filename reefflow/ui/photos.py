"""Photo-reconstruction sessions for the UI (spec Feature 1, Phase 6 workflow).

All engineering work is done by ``reefflow.photo`` (calibration with refraction,
GrabCut/SAM segmentation, visual-hull carving, edits, side-by-side check, STL
export, volume report). This module only keeps the interactive state on disk so
it survives a restart, and chains the library calls.

Layout (per project)::

    projects/<pid>/photos/session.json        clicks, focal settings, calibrations, edits, settings
    projects/<pid>/photos/<vid>/original.*    the uploaded file, untouched (provenance)
    projects/<pid>/photos/<vid>/image.jpg     working copy: EXIF orientation applied, longest side
                                              <= MAX_SIDE px (all pixel coordinates refer to it)
    projects/<pid>/photos/<vid>/mask.png      rock mask (0/255), user-editable
    projects/<pid>/photos/recon/              carving result (occupancy.npz, carve.json),
                                              report.json, compare_<vid>.jpg, recon.stl

A view id is the view kind ('top', 'front', 'back', 'left', 'right'), or
'<kind>-2', '-3', ... for additional photos of the same side (e.g. an oblique
front photo, which reduces the silhouette overfill).
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

MAX_SIDE = 2000
SESSION = "session.json"
DEFAULTS = {"dx": 0.005, "tolerance_px": 1, "min_component_voxels": 8, "bulk_density": 1400.0,
            "solid_fraction": 1.0, "edits": []}
_locks: dict = {}
_mask_history: dict = {}
_glock = threading.Lock()


def _lock(d: Path) -> threading.RLock:
    with _glock:
        return _locks.setdefault(str(d), threading.RLock())


def session_dir(project_dir: Path) -> Path:
    d = Path(project_dir) / "photos"
    d.mkdir(exist_ok=True)
    return d


def load(d: Path) -> dict:
    f = d / SESSION
    s = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    for k, v in DEFAULTS.items():
        s.setdefault(k, copy.deepcopy(v))
    s.setdefault("views", {})
    s.setdefault("order", list(s["views"]))
    return s


def save(d: Path, s: dict):
    tmp = d / (SESSION + ".tmp")
    tmp.write_text(json.dumps(s, indent=1, default=_json), encoding="utf-8")
    tmp.replace(d / SESSION)


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def tank_geometry(s: dict):
    from reefflow.photo import TankGeometry
    return TankGeometry.from_dict(s["tank"])


def sync_tank(d: Path, project_tank) -> dict:
    """Store the project's (saved) tank geometry in the session. If it changed since the
    photos were calibrated, every photo with enough clicks is calibrated again with the
    new size (the clicks are kept)."""
    from reefflow.photo import TankGeometry

    def num(t, k):
        v = t.get(k)
        if k == "bottom_glass" and v is None:
            v = t.get("glass")
        return float(v) if v is not None else None

    with _lock(d):
        s = load(d)
        t = TankGeometry.from_tank(project_tank).to_dict()
        changed = bool(s.get("tank")) and any(
            num(s["tank"], k) is None or abs(num(s["tank"], k) - num(t, k)) > 1e-9 for k in t if num(t, k) is not None)
        redo = []
        if changed:
            for vid, v in s["views"].items():
                if v.get("calibration") or v.get("clicks"):
                    v["calibration"] = None
                    v["calib_error"] = "tank dimensions changed: calibrating again with the new size"
                    npts = sum(1 for k in v.get("clicks", {}) if not k.startswith("seam_")) + len(v.get("custom", []))
                    if npts >= 4:
                        redo.append(vid)
            s["notice"] = ("The tank size changed since these photos were calibrated: they were calibrated "
                           "again with the new size (your clicks are kept).")
        s["tank"] = t
        save(d, s)
    for vid in redo:
        try:
            set_clicks(d, vid)
        except Exception:     # noqa: BLE001 - the panel shows the error on the next load
            pass
    return load(d) if redo else s


# ----------------------------------------------------------------------
# photos
# ----------------------------------------------------------------------
def _exif_f35(data: bytes):
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        ex = im.getexif()
        sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
        f35 = sub.get(0xA405) or ex.get(0xA405)
        f = sub.get(0x920A) or ex.get(0x920A)
        return (float(f35) if f35 else None), (float(f) if f else None)
    except Exception:
        return None, None


def add_view(d: Path, kind: str, filename: str, data: bytes) -> dict:
    import cv2
    from reefflow.photo import focal_px_from_35mm
    from reefflow.photo.tankref import VIEW_KINDS
    if kind not in VIEW_KINDS:
        raise ValueError(f"view must be one of {VIEW_KINDS}")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)      # applies EXIF orientation
    if img is None:
        raise ValueError(f"{filename}: not a readable image (use JPEG or PNG)")
    h0, w0 = img.shape[:2]
    k = min(1.0, MAX_SIDE / max(h0, w0))
    if k < 1.0:
        img = cv2.resize(img, (int(round(w0 * k)), int(round(h0 * k))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    f35, f_mm = _exif_f35(data)
    with _lock(d):
        s = load(d)
        vid, i = kind, 2
        while vid in s["views"]:
            vid = f"{kind}-{i}"
            i += 1
        vd = d / vid
        vd.mkdir(exist_ok=True)
        ext = Path(filename or "photo.jpg").suffix.lower() or ".jpg"
        (vd / f"original{ext}").write_bytes(data)
        cv2.imwrite(str(vd / "image.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        s["views"][vid] = {
            "id": vid, "kind": kind, "file": f"{vid}/image.jpg", "original": f"{vid}/original{ext}",
            "source_name": filename, "width": w, "height": h, "scale": k, "original_size": [w0, h0],
            "exif_f35_mm": f35, "exif_f_mm": f_mm,
            "exif_f_px": focal_px_from_35mm(f35, w, h) if f35 else None,
            "focal": {"mode": "exif" if f35 else "free", "value": None},
            "clicks": {}, "custom": [], "calibration": None, "calib_error": None,
            "mask": False, "mask_method": None, "mask_notes": [], "mask_edited": False, "added": time.time()}
        s["order"] = [v for v in s.get("order", []) if v in s["views"]] + [vid]
        save(d, s)
        return s["views"][vid]


def delete_view(d: Path, vid: str):
    import shutil
    with _lock(d):
        s = load(d)
        if vid not in s["views"]:
            raise KeyError(f"no photo {vid!r}")
        del s["views"][vid]
        s["order"] = [v for v in s["order"] if v != vid]
        shutil.rmtree(d / vid, ignore_errors=True)
        save(d, s)


def image(d: Path, v: dict) -> np.ndarray:
    from reefflow.photo import load_image
    return load_image(d / v["file"])


PHONE_F35_MM = 26.0          # main camera of most phones, 35-mm equivalent (D-838)
SIZE_CHECK_RMS_PX = 4.0      # run the tank-size check above this RMS (D-837)
GUIDE_MAX_RMS_PX = 6.0       # a top photo guides the side masks only below this RMS (D-841)


def focal_px(v: dict):
    from reefflow.photo import focal_px_from_35mm
    f = v.get("focal") or {}
    mode = f.get("mode", "free")
    if mode == "exif":
        return v.get("exif_f_px")
    if mode == "px" and f.get("value"):
        return float(f["value"])
    if mode == "mm35" and f.get("value"):
        return focal_px_from_35mm(float(f["value"]), v["width"], v["height"])
    return None


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------
def view_object(s: dict, v: dict):
    from reefflow.photo import View
    if not v.get("calibration"):
        return None
    return View.from_dict(v["calibration"]["view"], tank_geometry(s))


def overlay(view, tank, clicked=()) -> dict:
    """Reprojection of every named reference point and of the tank edges (polylines), for drawing.

    Lines: the outside box ('top', 'base' loops and 'outer' verticals), the water and
    sand lines, and the inside vertical corner seams ('vertical'). The inside rim and
    bottom loops are drawn only when one of their corners was clicked."""
    names = list(tank.reference_points())
    P = np.array([tank.point(n) for n in names])
    uv, ok = view.project(P)
    inimg = view.camera.in_image(uv, margin=-200)
    pts = {n: (u.tolist() if o and i else None) for n, u, o, i in zip(names, uv, ok, inimg)}
    edges = []
    L, W, H, g = tank.length, tank.width, tank.height, tank.glass
    inner = [(0, 0), (L, 0), (L, W), (0, W)]
    outer = [(-g, -g), (L + g, -g), (L + g, W + g), (-g, W + g)]
    used = {n.split("_")[0] for n in clicked}
    segs = []
    loops = [("water", inner, tank.water_surface), ("sand", inner, tank.sand_depth),
             ("top", outer, H), ("base", outer, -(g if tank.bottom_glass is None else tank.bottom_glass))]
    loops += [(lv, inner, z) for lv, z in (("rim", H), ("bottom", 0.0)) if lv in used]
    for lv, cs, z in loops:
        for a, b in zip(cs, cs[1:] + cs[:1]):
            segs.append((lv, (a[0], a[1], z), (b[0], b[1], z)))
    for c in inner:
        segs.append(("vertical", (c[0], c[1], 0.0), (c[0], c[1], H)))
    gb = g if tank.bottom_glass is None else tank.bottom_glass
    for c in outer:
        segs.append(("outer", (c[0], c[1], -gb), (c[0], c[1], H)))
    t = np.linspace(0, 1, 33)[:, None]
    for lv, a, b in segs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        u, ok = view.project(a + t * (b - a))
        ok &= view.camera.in_image(u, margin=-400)
        run = []
        for q, o in zip(u, ok):
            if o:
                run.append([round(float(q[0]), 1), round(float(q[1]), 1)])
            elif len(run) > 1:
                edges.append({"level": lv, "uv": run})
                run = []
            else:
                run = []
        if len(run) > 1:
            edges.append({"level": lv, "uv": run})
    return {"points": pts, "edges": edges}


def set_clicks(d: Path, vid: str, clicks: dict | None = None, custom: list | None = None,
               focal: dict | None = None, solve: bool = True, size_ok: bool | None = None) -> dict:
    """Store clicks / focal settings and (re)calibrate when there are enough points.
    size_ok=True: the user confirmed the tank size; the size check is skipped for this
    photo until the size changes."""
    from reefflow.photo import CalibrationError, calibrate_view, check_tank_size, focal_px_from_35mm
    from reefflow.photo.tankref import all_names
    with _lock(d):
        s = load(d)
        v = s["views"].get(vid)
        if v is None:
            raise KeyError(f"no photo {vid!r}")
        if clicks is not None:
            ok_names = set(all_names())
            bad = set(clicks) - ok_names
            if bad:
                raise ValueError(f"unknown reference point(s) {sorted(bad)}")
            v["clicks"] = {k: [float(x[0]), float(x[1])] for k, x in clicks.items()}
        if custom is not None:
            v["custom"] = custom
        if focal is not None:
            if focal.get("mode") not in ("free", "exif", "px", "mm35"):
                raise ValueError("focal mode must be free, exif, px or mm35")
            v["focal"] = {"mode": focal["mode"], "value": focal.get("value")}
        if size_ok is not None:
            v["size_ok_tank"] = s.get("tank") if size_ok else None
        save(d, s)
        tank = tank_geometry(s)
    n = len(v["clicks"]) + len(v["custom"])
    result, err, ov = None, None, None
    n = sum(1 for k in v["clicks"] if not k.startswith("seam_")) + len(v["custom"])
    if solve and n >= 4:
        try:
            f_known = focal_px(v)
            # No focal length known (phone browsers strip it from uploads): weak prior at a
            # phone main camera, 26 mm equivalent, sigma 25 % (D-838).
            f0 = v.get("exif_f_px") or focal_px_from_35mm(PHONE_F35_MM, v["width"], v["height"])
            prior = None if f_known else (f0, 0.25)
            cal = calibrate_view(v["kind"], (v["width"], v["height"]), tank, v["clicks"], v["custom"],
                                 f_px=f_known, f_prior=prior)
            result = cal.to_dict()
            result["focal_px"] = float(cal.view.camera.f)
            result["focal_prior_35mm"] = (v.get("exif_f35_mm") or PHONE_F35_MM) if prior else None
            result["camera_centre_m"] = cal.view.camera.centre.tolist()
            if cal.rms_px > SIZE_CHECK_RMS_PX and v.get("size_ok_tank") != tank.to_dict():
                result["size_check"] = check_tank_size(cal, tank)
            ov = overlay(cal.view, tank, v["clicks"])
        except (CalibrationError, ValueError) as e:
            err = str(e)
    elif n < 4:
        err = f"click at least {4 - n} more point(s) at the corners (6-8 clicks recommended)"
    with _lock(d):
        s = load(d)
        v = s["views"][vid]
        v["calibration"], v["calib_error"] = result, err
        save(d, s)
    return {"view": v, "overlay": ov}


def get_overlay(d: Path, vid: str):
    s = load(d)
    v = s["views"][vid]
    view = view_object(s, v)
    return overlay(view, tank_geometry(s), v.get("clicks", {})) if view is not None else None


# ----------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------
def mask_path(d: Path, vid: str) -> Path:
    return d / vid / "mask.png"


def read_mask(d: Path, vid: str):
    from reefflow.photo import mask_from_png
    p = mask_path(d, vid)
    return mask_from_png(p.read_bytes()) if p.is_file() else None


def write_mask(d: Path, vid: str, mask, method=None, notes=None, edited=None, history: bool = True):
    from reefflow.photo import mask_to_png
    with _lock(d):
        s = load(d)
        v = s["views"][vid]
        if mask.shape != (v["height"], v["width"]):
            raise ValueError(f"mask is {mask.shape[1]} x {mask.shape[0]} px, photo is {v['width']} x {v['height']}")
        p = mask_path(d, vid)
        if history and p.is_file():
            h = _mask_history.setdefault((str(d), vid), [])
            h.append(p.read_bytes())
            del h[:-20]
        p.write_bytes(mask_to_png(mask))
        v["mask"] = True
        if method is not None:
            v["mask_method"] = method
        if notes is not None:
            v["mask_notes"] = notes
        if edited is not None:
            v["mask_edited"] = edited
        v["mask_fraction"] = float(np.count_nonzero(mask) / mask.size)
        save(d, s)
        return v


def undo_mask(d: Path, vid: str):
    from reefflow.photo import mask_from_png
    h = _mask_history.get((str(d), vid)) or []
    if not h:
        raise ValueError("nothing to undo for this mask")
    return write_mask(d, vid, mask_from_png(h.pop()), history=False)


# Base texture-cue masks kept in memory per photo, so the Sensitivity slider can move the
# outline without re-running GrabCut (D-840). Keyed by (session dir, view id).
_base_cache: dict = {}


def _base_key(v: dict) -> str:
    cal = v.get("calibration") or {}
    return json.dumps([cal.get("view"), v.get("added")], sort_keys=True, default=str)


def auto_segment(d: Path, vids: list | None = None, method: str = "auto", keep_edited: bool = True,
                 cue: str = "auto", sensitivity: float = 50.0) -> dict:
    """Automatic rock-mask suggestion (top view first; its prism guides the side views).
    Views whose mask the user edited are kept unless keep_edited is False. cue: 'auto'
    (roughness for side photos, colour for the top), 'texture' or 'colour'.
    sensitivity: 0-100, 50 = default (D-840)."""
    if cue not in ("auto", "texture", "colour"):
        raise ValueError("cue must be auto, texture or colour")
    sensitivity = float(np.clip(float(sensitivity), 0, 100))
    from reefflow.photo import auto_segment_views, guide_from_views, segment_view
    s = load(d)
    views, imgs = {}, {}
    for vid in s["order"]:
        v = s["views"][vid]
        view = view_object(s, v)
        if view is not None:
            views[vid], imgs[vid] = view, image(d, v)
    if not views:
        raise ValueError("calibrate at least one photo first (the suggestion uses the calibration)")
    targets = [x for x in (vids or list(views)) if x in views]
    if not targets:
        raise ValueError("the selected photo is not calibrated yet")
    if keep_edited and vids is None:
        targets = [x for x in targets if not s["views"][x].get("mask_edited")]
    out = {}
    # a badly calibrated top photo would cut the side masks short: only well-fitting
    # top views guide the others (D-841)
    bad_tops = {k for k in views if views[k].kind == "top"
                and (s["views"][k].get("calibration") or {}).get("rms_px", 0) > GUIDE_MAX_RMS_PX}
    if vids is None or len(targets) > 1:
        use = {k: views[k] for k in views if k not in bad_tops}
        res = auto_segment_views({k: imgs[k] for k in use}, use, method=method, cue=cue,
                                 sensitivity=sensitivity) if use else {}
        for k in bad_tops:
            res[k] = segment_view(imgs[k], views[k], method=method, cue=cue, sensitivity=sensitivity)
        for k in targets:
            out[k] = res[k]
            if bad_tops and views[k].kind != "top":
                out[k]["notes"].append("not guided by the top photo: its calibration error is over "
                                       f"{GUIDE_MAX_RMS_PX:g} px")
    else:
        k = targets[0]
        guide = None
        tops = [t for t in views if views[t].kind == "top" and t != k and t not in bad_tops
                and read_mask(d, t) is not None]
        if tops and views[k].kind != "top":
            guide = guide_from_views(views[k], [views[t] for t in tops], [read_mask(d, t) for t in tops])
        out[k] = segment_view(imgs[k], views[k], method=method, guide=guide, cue=cue, sensitivity=sensitivity)
        if guide is not None:
            out[k]["notes"].append("guided by the top-view mask")
    for k, r in out.items():
        if r.get("base") is not None:
            _base_cache[(str(d), k)] = {"key": _base_key(s["views"][k]), "base": r["base"], "edge": r["edge"],
                                        "method": r["method"],
                                        "notes": [n for n in r["notes"] if not n.startswith("sensitivity ")]}
        else:
            _base_cache.pop((str(d), k), None)
        write_mask(d, k, r["mask"], method=r["method"], notes=r["notes"], edited=False)
        _set_view_field(d, k, "mask_sensitivity", sensitivity if r.get("base") is not None else None)
    return {k: {"method": r["method"], "notes": r["notes"]} for k, r in out.items()}


def _set_view_field(d: Path, vid: str, key: str, value):
    with _lock(d):
        s = load(d)
        s["views"][vid][key] = value
        save(d, s)


def set_sensitivity(d: Path, vid: str, sensitivity: float) -> dict:
    """Move the outline of the last texture suggestion for this photo (no GrabCut re-run).
    Raises ValueError when there is no cached suggestion (the caller then suggests again)."""
    from reefflow.photo.segment import finish_texture_mask
    sensitivity = float(np.clip(float(sensitivity), 0, 100))
    s = load(d)
    v = s["views"].get(vid)
    if v is None:
        raise KeyError(f"no photo {vid!r}")
    c = _base_cache.get((str(d), vid))
    if c is None or c["key"] != _base_key(v):
        raise ValueError("no suggestion to adjust for this photo: press Suggest mask")
    m = finish_texture_mask(c["base"], c["edge"], sensitivity)
    notes = c["notes"] + ([f"sensitivity {sensitivity:g} (50 = default)"] if sensitivity != 50 else [])
    view = write_mask(d, vid, m, method=c["method"], notes=notes, edited=False)
    _set_view_field(d, vid, "mask_sensitivity", sensitivity)
    view["mask_sensitivity"] = sensitivity
    return view


def mask_ops(d: Path, vid: str, ops: list) -> dict:
    """Apply brush / polygon / cleanup operations (reefflow.photo mask edit functions)."""
    from reefflow.photo import brush, fill_holes, polygon, remove_small, smooth
    s = load(d)
    v = s["views"][vid]
    m = read_mask(d, vid)
    if m is None:
        m = np.zeros((v["height"], v["width"]), bool)
    for op in ops:
        kind = op.get("op")
        mode = op.get("mode", "add")
        if mode not in ("add", "subtract"):
            raise ValueError("mode must be add or subtract")
        if kind == "brush":
            m = brush(m, op["stroke"], float(op.get("radius", 10)), mode)
        elif kind == "polygon":
            if len(op["vertices"]) < 3:
                raise ValueError("a polygon needs at least 3 points")
            m = polygon(m, op["vertices"], mode)
        elif kind == "fill_holes":
            m = fill_holes(m, op.get("max_area"))
        elif kind == "remove_small":
            m = remove_small(m, int(op.get("min_area", 200)))
        elif kind == "smooth":
            m = smooth(m, int(op.get("radius", 2)))
        elif kind == "clear":
            m = np.zeros_like(m)
        elif kind == "invert":
            m = ~m
        else:
            raise ValueError(f"unknown mask operation {kind!r}")
    return write_mask(d, vid, m, edited=True)


# ----------------------------------------------------------------------
# carving (runs in a job process)
# ----------------------------------------------------------------------
def inputs_signature(d: Path, s: dict) -> str:
    """Hash of everything the carving depends on (calibrations, masks, voxel size, tolerance)."""
    h = hashlib.sha1()
    for vid in s["order"]:
        v = s["views"][vid]
        if not v.get("calibration") or not mask_path(d, vid).is_file():
            continue
        h.update(vid.encode())
        h.update(json.dumps(v["calibration"]["view"], sort_keys=True).encode())
        h.update(hashlib.sha1(mask_path(d, vid).read_bytes()).digest())
    h.update(json.dumps([s["dx"], s["tolerance_px"], s["min_component_voxels"], s["tank"]], sort_keys=True).encode())
    return h.hexdigest()[:16]


def ready_views(d: Path, s: dict) -> list:
    return [vid for vid in s["order"] if s["views"][vid].get("calibration") and mask_path(d, vid).is_file()]


def grid_for(s: dict):
    from reefflow.photo import VoxelGrid
    return VoxelGrid.for_tank(tank_geometry(s), float(s["dx"]))


def carve_job(d: Path, progress=print, phase=None, status=None, should_stop=None) -> bool:
    """Visual-hull carving of the session, in z-slabs so progress can be reported (each voxel
    is independent, so slabs give exactly the result of one call). Writes recon/occupancy.npz,
    recon/carve.json, then the edited result's report and comparison images. Returns False if
    cancelled."""
    from reefflow.photo import VoxelGrid, carve_visual_hull, remove_small_components
    phase = phase or (lambda p: None)
    status = status or (lambda st: None)
    phase("load")
    s = load(d)
    vids = ready_views(d, s)
    if not vids:
        raise ValueError("no photo is both calibrated and masked")
    if not any(s["views"][v]["kind"] == "top" for v in vids):
        progress("[photo] warning: no top view; the plan shape is constrained only by side views")
    views = [view_object(s, s["views"][v]) for v in vids]
    masks = [read_mask(d, v) for v in vids]
    grid = grid_for(s)
    sig = inputs_signature(d, s)
    nx, ny, nz = grid.shape
    progress(f"[photo] carving {len(vids)} view(s) ({', '.join(vids)}) on a {nx} x {ny} x {nz} grid "
             f"({nx * ny * nz / 1e6:.2f} M voxels at {grid.dx * 1e3:g} mm), mask tolerance {s['tolerance_px']} px")
    phase("carve")
    occ = np.zeros(grid.shape, bool)
    seen = np.zeros(grid.shape, np.uint8)
    per_view = [{"view": v, "kind": s["views"][v]["kind"], "seen": 0, "carved": 0} for v in vids]
    step = max(1, int(400_000 // max(nx * ny, 1)))
    t0 = time.perf_counter()
    for k0 in range(0, nz, step):
        if should_stop and should_stop():
            progress("[photo] carving cancelled")
            return False
        k1 = min(nz, k0 + step)
        sub = VoxelGrid(grid.origin + np.array([0.0, 0.0, k0 * grid.dx]), grid.dx, (nx, ny, k1 - k0))
        r = carve_visual_hull(views, masks, sub, tolerance_px=int(s["tolerance_px"]))
        occ[:, :, k0:k1] = r.occupancy
        seen[:, :, k0:k1] = r.seen_count
        for a, b in zip(per_view, r.stats["per_view"]):
            a["seen"] += b["seen"]
            a["carved"] += b["carved"]
        el = time.perf_counter() - t0
        status({"fraction": k1 / nz, "elapsed_s": el, "eta_s": el / k1 * (nz - k1)})
    n_raw = int(occ.sum())
    occ = remove_small_components(occ, int(s["min_component_voxels"])) if s["min_component_voxels"] else occ
    stats = {"voxels": int(occ.size), "occupied_raw": n_raw, "occupied": int(occ.sum()),
             "removed_small": n_raw - int(occ.sum()), "unconstrained": int((seen < 1).sum()),
             "seen_by_n_views": {int(n): int((seen == n).sum()) for n in range(len(vids) + 1)},
             "per_view": per_view, "tolerance_px": int(s["tolerance_px"]), "views": vids,
             "time_s": time.perf_counter() - t0}
    out = d / "recon"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "occupancy.npz", occ=np.packbits(occ), seen=seen, shape=np.array(grid.shape),
                        origin=grid.origin, dx=grid.dx)
    (out / "carve.json").write_text(json.dumps({"signature": sig, "stats": stats, "grid": grid.to_dict(),
                                                "time": time.time()}, indent=1, default=_json), encoding="utf-8")
    progress(f"[photo] carved in {stats['time_s']:.0f} s: {stats['occupied']} voxels occupied "
             f"({stats['occupied'] * grid.dx ** 3 * 1e3:.2f} L before edits)")
    phase("finish")
    progress("[photo] comparing the reconstruction with each photo")
    comparison(d)
    progress("[photo] done")
    return True


# ----------------------------------------------------------------------
# result: edits, mesh, comparison, volume, export
# ----------------------------------------------------------------------
def carve_info(d: Path):
    f = d / "recon" / "carve.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def raw_occupancy(d: Path):
    from reefflow.photo import VoxelGrid
    f = d / "recon" / "occupancy.npz"
    if not f.is_file():
        raise FileNotFoundError("no reconstruction yet: run 'Carve' first")
    z = np.load(f)
    shape = tuple(int(x) for x in z["shape"])
    occ = np.unpackbits(z["occ"])[:int(np.prod(shape))].reshape(shape).astype(bool)
    return occ, VoxelGrid(z["origin"], float(z["dx"]), shape)


def final_occupancy(d: Path, edits: list | None = None):
    from reefflow.photo import apply_edits
    occ, grid = raw_occupancy(d)
    edits = load(d)["edits"] if edits is None else edits
    return (apply_edits(occ, grid, edits) if edits else occ), grid


def validate_edit(e: dict):
    if e.get("op") not in ("carve", "add"):
        raise ValueError("edit op must be 'carve' or 'add'")
    sh = e.get("shape") or {}
    t = sh.get("type")
    need = {"sphere": ("center", "radius"), "box": ("lo", "hi"), "obox": ("center", "size"),
            "capsule": ("points", "radius"), "arch": ("base_center", "axis", "span", "rise", "length")}
    if t not in need:
        raise ValueError(f"unknown edit shape {t!r}")
    miss = [k for k in need[t] if k not in sh]
    if miss:
        raise ValueError(f"{t} edit needs {miss}")
    if t == "arch" and float(sh["rise"]) < float(sh["span"]) / 2:
        raise ValueError("arch rise must be at least half the span (semicircular top)")


def set_edits(d: Path, edits: list) -> dict:
    for e in edits:
        validate_edit(e)
    with _lock(d):
        s = load(d)
        s["edits"] = edits
        save(d, s)
    return volume(d)


def mesh(d: Path):
    """Display mesh (vertices m, faces) of the edited reconstruction, lightly smoothed."""
    from reefflow.photo import occupancy_to_mesh
    occ, grid = final_occupancy(d)
    if not occ.any():
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return occupancy_to_mesh(occ, grid, extend_below=1, smooth_sigma=0.6)


def volume(d: Path, bulk_density: float | None = None, solid_fraction: float | None = None) -> dict:
    from reefflow.photo import volume_report
    with _lock(d):
        s = load(d)
        if bulk_density is not None:
            s["bulk_density"] = float(bulk_density)
        if solid_fraction is not None:
            s["solid_fraction"] = float(solid_fraction)
        save(d, s)
    occ, grid = final_occupancy(d)
    raw, _ = raw_occupancy(d)
    rep = volume_report(occ, grid, bulk_density=float(s["bulk_density"]), solid_fraction=float(s["solid_fraction"]))
    rep["volume_before_edits_m3"] = float(raw.sum()) * grid.cell_volume
    rep["edits"] = len(s["edits"])
    return rep


def comparison(d: Path) -> dict:
    """Side-by-side reconstruction-vs-photo images and silhouette IoU per view."""
    import cv2
    from reefflow.photo import side_by_side
    s = load(d)
    occ, grid = final_occupancy(d)
    out = {}
    for vid in ready_views(d, s):
        v = s["views"][vid]
        r = side_by_side(image(d, v), view_object(s, v), occ, grid, read_mask(d, vid), label=vid)
        cv2.imwrite(str(d / "recon" / f"compare_{vid}.jpg"), cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        out[vid] = {k: r[k] for k in ("silhouette_iou", "overfill_px", "underfill_px")}
    rep = {"views": out, "edits": len(s["edits"]), "time": time.time()}
    (d / "recon" / "compare.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def export(d: Path, path: Path) -> dict:
    from reefflow.photo import export_stl
    occ, grid = final_occupancy(d)
    if not occ.any():
        raise ValueError("the reconstruction is empty")
    return export_stl(occ, grid, path)


def estimate_grid(s: dict, dx: float | None = None) -> dict:
    t = s["tank"]
    dx = float(dx or s["dx"])
    n = [max(1, math.floor(t["length"] / dx)), max(1, math.floor(t["width"] / dx)),
         max(1, math.floor((t["water_surface"] - t["sand_depth"]) / dx))]
    vox = n[0] * n[1] * n[2]
    return {"shape": n, "voxels": vox, "est_seconds_per_view": vox / 1.8e6 * 25 / 3}


def state(d: Path) -> dict:
    """Everything the Photos panel shows (small JSON)."""
    from reefflow.photo import recommended_points
    from reefflow.photo.tankref import VIEW_KINDS, all_names, describe
    s = load(d)
    ci = carve_info(d)
    sig = inputs_signature(d, s) if s.get("tank") else None
    rec = None
    if ci is not None:
        rec = {"stats": ci["stats"], "grid": ci["grid"], "time": ci["time"], "stale": ci["signature"] != sig}
        cf = d / "recon" / "compare.json"
        if cf.is_file():
            rec["comparison"] = json.loads(cf.read_text(encoding="utf-8"))
        try:
            rec["volume"] = volume(d)
        except Exception as e:                            # noqa: BLE001
            rec["volume_error"] = str(e)
    return {"views": [s["views"][v] for v in s["order"]], "tank": s.get("tank"), "settings":
            {k: s[k] for k in ("dx", "tolerance_px", "min_component_voxels", "bulk_density", "solid_fraction")},
            "edits": s["edits"], "notice": s.get("notice"), "ready_views": ready_views(d, s), "recon": rec,
            "grid_estimate": estimate_grid(s) if s.get("tank") else None,
            "recommended": {k: recommended_points(k) for k in VIEW_KINDS},
            "all_points": [describe(n) for n in all_names()], "used_as_rockscape": s.get("used_as_rockscape")}


def update_settings(d: Path, **kw) -> dict:
    with _lock(d):
        s = load(d)
        for k, v in kw.items():
            if v is None:
                continue
            if k == "dx" and not 0.002 <= float(v) <= 0.02:
                raise ValueError("voxel size must be between 2 and 20 mm")
            if k == "tolerance_px" and not 0 <= int(v) <= 10:
                raise ValueError("mask tolerance must be 0-10 px")
            s[k] = v
        save(d, s)
    return state(d)
