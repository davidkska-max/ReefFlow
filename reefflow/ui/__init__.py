"""ReefFlow browser UI (Phase 5): local FastAPI backend + three.js frontend.

    python -m reefflow.ui [--workspace DIR] [--port N] [--no-browser]

The UI is a layer over the core library: every engineering calculation is a call
into reefflow.project / tank / run / analysis / geometry (and reefflow.light).
"""
from .launcher import main, serve  # noqa: F401


def create_app(*args, **kwargs):
    from .server import create_app as _c
    return _c(*args, **kwargs)
