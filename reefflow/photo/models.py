"""Optional local ML models: registry, license gate, cache, explicit download, stubs.

Rules (spec 'Deployment and cost constraints'; DECISIONS D-604):
* Local open-weight models only; nothing is downloaded unless
  :func:`download_model` is called explicitly with ``accept_license=True``.
  Nothing in ReefFlow calls it automatically, and the tests never do.
* Only models whose weights AND code permit commercial use are downloadable.
  Non-commercial entries are listed (so the UI can say why they are absent)
  but :func:`download_model` refuses them.
* The classical pipeline (GrabCut + manual editing) never depends on this module.
* Torch and the model packages are optional and import-guarded.

Cache directory: ``$REEFFLOW_MODEL_DIR`` if set, else
``%LOCALAPPDATA%/ReefFlow/models`` (Windows) or ``~/.cache/reefflow/models``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path


class ModelNotInstalled(RuntimeError):
    """Raised when an optional model (weights or runtime package) is missing."""


# sha256: None = not yet pinned; the hash of the downloaded file is recorded in a
# sidecar JSON on first download (open issue: pin hashes before release).
MODELS = {
    "mobile_sam": {
        "task": "segmentation", "license": "Apache-2.0", "commercial_use": True,
        "url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "filename": "mobile_sam.pt", "size_mb": 39, "sha256": None,
        "runtime": ["torch", "mobile_sam"],
        "install_hint": "pip install torch (CPU build) and MobileSAM "
                        "(git+https://github.com/ChaoningZhang/MobileSAM, Apache-2.0)",
        "notes": "TinyViT encoder distilled from SAM; CPU inference ~0.5-2 s per image.",
    },
    "sam_vit_b": {
        "task": "segmentation", "license": "Apache-2.0", "commercial_use": True,
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth", "size_mb": 375, "sha256": None,
        "runtime": ["torch", "segment_anything"],
        "install_hint": "pip install torch (CPU build) and segment-anything (Apache-2.0)",
        "notes": "Original SAM ViT-B; CPU inference ~5-15 s per image.",
    },
    "depth_anything_v2_small": {
        "task": "depth", "license": "Apache-2.0", "commercial_use": True,
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
               "depth_anything_v2_vits.pth",
        "filename": "depth_anything_v2_vits.pth", "size_mb": 99, "sha256": None,
        "runtime": ["torch"],
        "install_hint": "interface stub only (spec item 6); not used by the pipeline yet",
        "notes": "Relative (affine-invariant) depth; would need metric scaling from the calibration.",
    },
    "depth_anything_v2_base": {
        "task": "depth", "license": "CC-BY-NC-4.0", "commercial_use": False,
        "url": None, "filename": None, "size_mb": 390, "sha256": None, "runtime": ["torch"],
        "install_hint": "EXCLUDED: non-commercial license", "notes": "Listed for the audit only.",
    },
    "depth_anything_v2_large": {
        "task": "depth", "license": "CC-BY-NC-4.0", "commercial_use": False,
        "url": None, "filename": None, "size_mb": 1300, "sha256": None, "runtime": ["torch"],
        "install_hint": "EXCLUDED: non-commercial license", "notes": "Listed for the audit only.",
    },
}

LICENSE_ALLOWLIST = ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause")


def cache_dir() -> Path:
    env = os.environ.get("REEFFLOW_MODEL_DIR")
    if env:
        return Path(env)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "ReefFlow" / "models"
    return Path.home() / ".cache" / "reefflow" / "models"


def _spec(name: str) -> dict:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name]


def weights_path(name: str, directory=None) -> Path | None:
    s = _spec(name)
    if not s["filename"]:
        return None
    return Path(directory or cache_dir()) / s["filename"]


def runtime_available(name: str) -> tuple[bool, str]:
    missing = [m for m in _spec(name)["runtime"] if importlib.util.find_spec(m) is None]
    return (not missing, "ok" if not missing else "missing python packages: " + ", ".join(missing))


def model_status(directory=None) -> list[dict]:
    """One JSON-able row per model, for the UI 'models' panel."""
    rows = []
    for name, s in MODELS.items():
        p = weights_path(name, directory)
        rt, why = runtime_available(name)
        installed = bool(p and p.exists())
        rows.append({"name": name, "task": s["task"], "license": s["license"],
                     "commercial_use": s["commercial_use"], "size_mb": s["size_mb"],
                     "weights_installed": installed, "runtime_available": rt,
                     "usable": installed and rt and s["commercial_use"],
                     "status": ("excluded (license)" if not s["commercial_use"] else
                                "ready" if installed and rt else
                                "not installed" + ("" if installed else " (weights)")
                                + ("" if rt else f" ({why})")),
                     "path": str(p) if p else None, "install_hint": s["install_hint"]})
    return rows


def check_license(name: str) -> None:
    s = _spec(name)
    if not s["commercial_use"] or s["license"] not in LICENSE_ALLOWLIST:
        raise PermissionError(f"model {name!r} is licensed {s['license']}, which does not permit "
                              "commercial use; it is excluded from ReefFlow (spec license audit)")


def download_model(name: str, directory=None, accept_license: bool = False, progress=None) -> Path:
    """Download a model's weights into the cache. Explicit, user-initiated only.

    accept_license must be True (the UI shows the license first). Refuses
    non-commercial models. Writes ``<file>.json`` with url, license, sha256, date."""
    check_license(name)
    s = _spec(name)
    if not accept_license:
        raise PermissionError(f"downloading {name!r} requires accept_license=True "
                              f"(license: {s['license']})")
    if os.environ.get("REEFFLOW_OFFLINE") == "1":
        raise ModelNotInstalled("REEFFLOW_OFFLINE=1: downloads disabled")
    import urllib.request
    dest = weights_path(name, directory)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(s["url"], timeout=60) as r, open(tmp, "wb") as fh:  # noqa: S310
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk); h.update(chunk); done += len(chunk)
            if progress:
                progress(done, total)
    digest = h.hexdigest()
    if s["sha256"] and digest != s["sha256"]:
        tmp.unlink(missing_ok=True)
        raise IOError(f"sha256 mismatch for {name}: got {digest}")
    tmp.replace(dest)
    dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(
        {"model": name, "url": s["url"], "license": s["license"], "sha256": digest,
         "downloaded_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    return dest


# ----------------------------------------------------------------------
# SAM-family segmentation (optional)
# ----------------------------------------------------------------------
class SamSegmenter:
    """Prompted segmentation with MobileSAM / SAM ViT-B, if installed.

    NOTE: this path is not exercised by the test suite (no weights/torch in the
    default install). It follows the published SamPredictor API."""

    def __init__(self, name: str = "mobile_sam", directory=None, device: str = "cpu"):
        check_license(name)
        if _spec(name)["task"] != "segmentation":
            raise ValueError(f"{name} is not a segmentation model")
        self.name, self.directory, self.device = name, directory, device
        self._predictor = None

    def available(self) -> tuple[bool, str]:
        rt, why = runtime_available(self.name)
        p = weights_path(self.name, self.directory)
        if not (p and p.exists()):
            return False, f"not installed: weights {p} missing (download_model('{self.name}', accept_license=True))"
        if not rt:
            return False, "not installed: " + why
        return True, "ok"

    def _load(self):
        ok, why = self.available()
        if not ok:
            raise ModelNotInstalled(why)
        if self._predictor is None:
            if self.name == "mobile_sam":
                from mobile_sam import SamPredictor, sam_model_registry   # type: ignore
                model = sam_model_registry["vit_t"](checkpoint=str(weights_path(self.name, self.directory)))
            else:
                from segment_anything import SamPredictor, sam_model_registry   # type: ignore
                model = sam_model_registry["vit_b"](checkpoint=str(weights_path(self.name, self.directory)))
            model.to(self.device).eval()
            self._predictor = SamPredictor(model)
        return self._predictor

    def predict(self, image, box=None, fg_points=None, bg_points=None):
        """image RGB uint8 HxWx3; box [u0, v0, u1, v1]; points [[u, v], ...]. -> bool mask."""
        import numpy as np
        pred = self._load()
        pred.set_image(np.ascontiguousarray(image))
        pts, lab = [], []
        for p in fg_points or []:
            pts.append(p); lab.append(1)
        for p in bg_points or []:
            pts.append(p); lab.append(0)
        masks, scores, _ = pred.predict(point_coords=np.asarray(pts, float) if pts else None,
                                        point_labels=np.asarray(lab) if lab else None,
                                        box=None if box is None else np.asarray(box, float),
                                        multimask_output=False)
        return masks[0].astype(bool)


# ----------------------------------------------------------------------
# stubs for spec item 6 (better concavity capture) -- interface only
# ----------------------------------------------------------------------
def estimate_depth(image, model: str = "depth_anything_v2_small", directory=None):
    """Monocular relative depth map (H x W float). STUB (spec Feature 1 item 6).

    Planned use: per-view depth to carve concavities the silhouettes cannot see,
    scaled to metric with the calibrated camera and the visual hull surface."""
    check_license(model)
    p = weights_path(model, directory)
    if not (p and p.exists()):
        raise ModelNotInstalled(f"{model}: not installed")
    raise NotImplementedError("monocular depth estimation is a planned enhancement (spec item 6)")


def photogrammetry_reconstruct(image_paths, tank=None, **kw):
    """Multi-photo / video structure-from-motion + dense reconstruction. STUB.

    Planned interface: list of image paths (or a video) -> (vertices, faces) in tank
    coordinates, scaled with the clicked tank corners of at least one frame."""
    raise NotImplementedError("multi-photo photogrammetry is a planned enhancement (spec item 6); "
                              "no engine is bundled")


def capabilities(directory=None) -> dict:
    """What the photo pipeline can do on this machine (for the UI)."""
    st = {r["name"]: r for r in model_status(directory)}
    sam = [n for n in ("mobile_sam", "sam_vit_b") if st[n]["usable"]]
    return {"segmentation": ["grabcut"] + sam, "grabcut": True, "sam": sam,
            "depth": "interface stub (not implemented)", "photogrammetry": "interface stub (not implemented)",
            "models": list(st.values())}
