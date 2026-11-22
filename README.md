# ReefFlow

Water circulation (lattice Boltzmann CFD) and lighting (Monte Carlo PAR ray tracing) for reef aquariums. ReefFlow maps dead spots, water age, detritus settling and coral placement scores. It runs entirely on your own machine; after installation no internet connection is needed.

**ReefFlow is free to use.** If it helps your tank, you can [buy me a coffee ☕](https://buymeacoffee.com/davidkska).

- **Walkthrough video:** attached to each [release](../../releases)
- **Illustrated guide:** the **Help** button in the app, or `reefflow/ui/static/guide/index.html`
- **Reference manual:** `docs/USER_GUIDE.md`
- **Design choices and physical assumptions (with numbers):** `DECISIONS.md`, `ASSUMPTIONS.md`
- **Verification evidence:** `verification/VERIFICATION.md` (in the GitHub repository; not in the download zip)
- **Licence:** `LICENSE` (free for personal use; all rights reserved). Third-party libraries: `LICENSES.md`.

## Download and install (Windows)

1. Download `ReefFlow-<version>.zip` from the latest [release](../../releases) and unzip it somewhere permanent, e.g. `Documents\ReefFlow`.
2. Right-click `install.ps1` and choose **Run with PowerShell**. It needs the internet once, to download Python and the libraries (about 1.3 GB).
3. Double-click `ReefFlow.bat`. ReefFlow opens in your browser; choose **From the 75-gallon demo** to try it.

**What you need:** Windows 10 or 11, about 2 GB of disk space plus room for runs (a few hundred MB each), and ideally an NVIDIA graphics card. Other graphics cards and CPU-only PCs work, but runs take much longer.

Everything runs on your own PC. Nothing is uploaded, and no internet connection is needed after installation.

## Install details

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1              # add -SystemCerts on TLS-inspecting networks
```

The installer sets up `uv` and a Python 3.12 environment in `.venv`, installs ReefFlow, writes `ReefFlow.bat` and runs a self-check.

Manual equivalent:

```powershell
uv venv .venv --python 3.12
uv pip install --system-certs --python .venv\Scripts\python.exe -e ".[dev]"
```

Python 3.13+ is not supported: Taichi 1.7 ships no wheels for it. An NVIDIA GPU (CUDA) is used automatically. Other GPUs use Vulkan; without a GPU, the CPU backend is used and is about 50–100× slower.

## Use

**Browser interface.** Double-click `ReefFlow.bat`, or run `.venv\Scripts\reefflow ui`. From there you can:
- design or import a rockscape;
- place pumps and lights;
- run a simulation and browse the results.

**Command line**, with the 75-gallon demo:

```powershell
.venv\Scripts\reefflow demo examples\demo_75gal                         # write the demo project + rock STL
.venv\Scripts\reefflow info examples\demo_75gal\demo_75gal.yaml --preset preview
.venv\Scripts\reefflow run  examples\demo_75gal\demo_75gal.yaml --preset preview     # CFD, then the analysis + report
.venv\Scripts\reefflow light examples\demo_75gal\demo_75gal.yaml                     # PAR / DLI maps only
.venv\Scripts\reefflow compare runs\A runs\B --label-a before --label-b after        # difference maps, metric deltas
```

A project is one human-readable YAML file containing:
- tank dimensions;
- rockscape (STL/OBJ/PLY references);
- pumps (from `reefflow/data/pump_library.yaml`);
- overflow;
- lighting (fixtures from `reefflow/light/data/fixture_library.yaml`);
- simulation settings;
- metric thresholds (coral profiles in `reefflow/data/coral_profiles.yaml`).

Quantities are written with units, e.g. `"48 in"`, `"2500 GPH"`, `"8 in"`, `"60 s"`. Set `display_units: US` or `SI`. Internally everything is SI.

**Resolution presets** (75-gal tank, RTX 3070):

| Preset | Cell size | Cells | Run time | Use |
|---|---|---|---|---|
| preview | 10 mm | 0.25 M | ~5 min | bulk flow pattern only |
| standard | 5 mm | 2 M | ~13 min | design comparisons |
| fine | 3 mm | 8.9 M | ~75 min | best near-wall accuracy |

Near-surface quantities (dead-zone volume, sand shear, flow over rock) depend strongly on resolution; see `verification/VERIFICATION.md` section 6.

**Outputs of a run** (`runs/<name>/`):
- `report.html`: self-contained summary, prints to PDF.
- `run.json`, `analysis.json`: all numbers.
- `tank.vti`, `analysis.vti`: open in ParaView.
- `rock_score_<profile>.ply`, `dead_zones.ply`: 3D meshes.
- PNG figures.

## Tests and verification

```powershell
.venv\Scripts\python -m pytest                    # fast suite
.venv\Scripts\python -m pytest -m slow            # full verification cases (GPU, tens of minutes)
.venv\Scripts\python -m reefflow.verification.run_all --report   # rebuild verification/VERIFICATION.md
```

Environment switches:
- `REEFFLOW_ARCH = auto | cuda | vulkan | cpu`
- `REEFFLOW_PRECISION = f32 | f64` (fp64 on CPU only)
- `REEFFLOW_OFFLINE = 1`: blocks even the explicit optional-model download

## Layout

| Path | Contents |
|---|---|
| `reefflow/units.py`, `lattice.py`, `kernels.py`, `solver.py`, `domain.py`, `stats.py` | LBM core (D3Q19/D3Q27, recursive-regularized collision, Smagorinsky LES), units, statistics |
| `reefflow/sources.py`, `pumps.py`, `geometry.py`, `project.py`, `tank.py`, `run.py` | Tank model: pumps, overflow, rock voxelization, project file, run driver |
| `reefflow/metrics.py`, `reefflow/analysis/` | Dead zones, water age, wall shear/detritus, coral scores, report, comparison |
| `reefflow/light/` | Photon tracer, fixtures, calibration, PAR/DLI maps |
| `reefflow/photo/` | Photo calibration (PnP with refraction), segmentation, visual hull, edit tools |
| `reefflow/ui/` | FastAPI backend and three.js frontend |
| `reefflow/wallmodel.py` | Experimental log-law wall model (off by default; see DECISIONS) |
| `reefflow/verification/` | Verification cases with plots |
| `install.ps1`, `reefflow/selfcheck.py` | Installer and self-check |
