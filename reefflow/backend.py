"""Compute-backend selection (Taichi).

All GPU/CPU specifics are confined to this module and to the kernel modules
(``solver``, ``stats``), so a port to another kernel framework (e.g. NVIDIA
Warp) touches only those files.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="taichi")


import taichi as ti  # noqa: E402

_ARCHES = {"cuda": "cuda", "vulkan": "vulkan", "metal": "metal", "cpu": "cpu"}
_state: dict = {"arch": None, "kwargs": None}


_FP = {"f32": ti.f32, "f64": ti.f64}


def _try(name: str, precision: str = "f32", **kw) -> bool:
    arch = getattr(ti, name)
    ti.init(arch=arch, default_fp=_FP[precision], offline_cache=True, log_level=ti.WARN, **kw)
    got = ti.lang.impl.current_cfg().arch
    return got == arch


def init(arch: str = "auto", precision: str = "f32", **kw) -> str:
    """Initialise Taichi. ``arch`` is 'auto', 'cuda', 'vulkan', 'metal' or 'cpu'.

    'auto' tries CUDA, then Vulkan, then Metal, then falls back to CPU.
    ``precision`` 'f32' (production) or 'f64' (verification of round-off floors;
    needs a backend with fp64 support, e.g. CUDA or CPU). Returns the backend in use.
    """
    arch = os.environ.get("REEFFLOW_ARCH", arch)
    precision = os.environ.get("REEFFLOW_PRECISION", precision)
    kw["precision"] = precision
    order = ["cuda", "vulkan", "metal", "cpu"] if arch == "auto" else [arch]
    for name in order:
        if name not in _ARCHES:
            raise ValueError(f"unknown arch {name!r}")
        if _try(name, **kw):
            _state.update(arch=name, kwargs=kw)
            return name
    if arch != "auto":
        raise RuntimeError(f"Requested backend {arch!r} is not available on this machine")
    raise RuntimeError("No Taichi backend available")


def fresh() -> str:
    """Reset Taichi (frees all fields) and re-initialise on the same backend."""
    arch = _state["arch"] or "auto"
    kw = _state["kwargs"] or {}
    ti.reset()
    return init(arch, **kw)


def current() -> str | None:
    return _state["arch"]


def precision() -> str:
    return (_state["kwargs"] or {}).get("precision", "f32")
