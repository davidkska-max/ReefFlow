"""Phase 4: photometric / PAR analysis with a Monte Carlo forward photon tracer (Taichi).

Public API::

    from reefflow.light import run_lighting, par_on_cfd_grid
    res = run_lighting(project)                  # LightResult on the project's node grid
    res = run_lighting(project, model=tank)      # ... on a CFD TankModel's grid
    par = par_on_cfd_grid(tank, res)             # CfdPar: meter / normal PAR at fluid nodes

Importing this package imports Taichi (via reefflow.backend) but does not initialise it;
the kernels compile on the first trace.
"""
from .analysis import compare, hotspots_and_shadows, sweep_height, uniformity
from .config import LightingConfig, Photoperiod, lighting_from_project, parse_lighting
from .fixtures import Distribution, Fixture, build_fixture, load_library
from .results import CfdPar, LightResult, par_on_cfd_grid
from .run import Geometry, prepare_geometry, rock_sdf, roughness_from_cfd, run_lighting, with_lighting
from .sdf import SDF

__all__ = ["run_lighting", "par_on_cfd_grid", "CfdPar", "LightResult", "LightingConfig", "Photoperiod",
           "parse_lighting", "lighting_from_project", "load_library", "build_fixture", "Fixture", "Distribution",
           "prepare_geometry", "rock_sdf", "roughness_from_cfd", "Geometry", "with_lighting", "SDF", "uniformity",
           "hotspots_and_shadows", "compare", "sweep_height"]
