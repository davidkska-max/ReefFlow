# ReefFlow release notes

## 1.0.0 (first public release)

ReefFlow simulates water flow and light in a reef tank on your own Windows PC. Build a model of your tank, rock, pumps and lights, run it, and see dead spots, detritus build-up, water age, PAR and the best spots on the rock for SPS, LPS and softies. Then change one thing and compare.

### What's in it

- **Tank and rock.** Enter your tank size, glass and sand depth.
  - Build a rockscape from pieces, from photos of your tank, or from a 3D scan or STL/OBJ/PLY file.
  - Sculpt the result: push, pull, smooth, and drill caves and holes.
- **Real equipment.** Pick from 35 powerheads, 41 return pumps (27 with published head–flow curves), 30 light fixtures and 21 overflows. Each model shows its published specs and where they came from.
- **Lids.** Model an open top or a lid, and see the effect on PAR. The choices are glass, low-iron glass, acrylic, aluminium or black window screen, 1/8 in or 1/4 in netting, and an egg-crate grid. You can set the height, hole size and extra loss from salt or condensation.
- **Flow simulation** (lattice Boltzmann, on the GPU). A quick preview of a 75-gallon tank takes about 5 minutes on an RTX 3070; a standard run with lighting takes about 13.
- **Light simulation.** Light is traced through the lid, the water surface and the water, including reflections off the glass and shadows from the rock. You get PAR and DLI maps, and you can calibrate them to your own PAR meter.
- **Results.**
  - Speed slices, dead zones, sand and rock-top flow, and water age.
  - Coral placement scores on the rock, with editable coral profiles.
  - Animated particles that follow the flow.
  - A printable report.
- **Compare.** Put two runs side by side, with difference maps and the change in every number.
- **Demo.** A ready-to-run 75-gallon reef with a natural rockscape (two islands, an arch and open sand).
- **Help.** An illustrated guide is built into the app, and a walkthrough video is attached to this release.

### Limitations worth knowing

- It is most useful for comparing setups. Trust "layout B beats layout A" more than any single number.
- Close to the sand and rock surfaces it under-predicts flow, so dead spots look a little worse than they are. Higher resolution helps.
- Rock built from photos comes out chunkier than the real thing, because cameras can't see into every gap. Clean it up with the sculpt tools.
- Without a PAR-meter calibration, light values are estimates. The light scattered by bright screens and egg crate is an estimate too.
- Windows only. An NVIDIA graphics card is strongly recommended; other GPUs and CPU-only PCs work, but runs are much slower.

### Install

Download `ReefFlow-1.0.0.zip` below, unzip it, right-click `install.ps1` and choose **Run with PowerShell**, then double-click `ReefFlow.bat`. The installer needs the internet once; after that ReefFlow runs offline.

ReefFlow is free for personal use (see `LICENSE`). If it helps your tank, you can [buy me a coffee ☕](https://buymeacoffee.com/davidkska).
