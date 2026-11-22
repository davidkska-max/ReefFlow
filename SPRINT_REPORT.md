# ReefFlow sprint report: Phases 2–7 (2026-10-31)

On 2026-10-31 you asked for Phases 2–7 to be completed in one sprint, without per-phase stops. This file is the review package.

Detail lives in four places:
- `verification/VERIFICATION.md`: every verification case, with plots;
- `DECISIONS.md`: what was decided and why, with numbers;
- `ASSUMPTIONS.md`: physical and modelling assumptions;
- `docs/USER_GUIDE.md`: how to use the browser interface.

## How to try it

1. `ReefFlow.bat`, or `.venv\Scripts\reefflow ui`, opens the browser interface. It runs locally only.
2. Or use the command line with the 75-gallon demo:
   - `.venv\Scripts\reefflow run examples\demo_75gal\demo_75gal.yaml --preset preview` takes about 5 min and writes `runs\...\report.html`.
   - `.venv\Scripts\reefflow light examples\demo_75gal\demo_75gal.yaml` produces PAR/DLI maps in about 20 s.

**Showcase result:** `runs/demo_standard/report.html`. It is the demo tank at Standard resolution with lighting: 10 min simulation + 2.5 min analysis.
- Mean speed 11.9 cm/s; dead-zone volume 3.7 % (an upper bound, see below).
- Water age nearly uniform: mean 0.98, max 1.11 × V/Q, with V/Q = 5.7 min.
- Sand shear median 28 mPa; detritus likely to settle on 39 % of the sand.
- PAR (uncalibrated): rock tops ≈ 380, sand ≈ 360 µmol m⁻² s⁻¹.
- Best SPS spots: upper right island, around (32, 7, 9) in.

## What was built

| Phase | Content | Key evidence |
|---|---|---|
| 2 Tank model + pumps | YAML project file (US/SI); rock STL import and voxelization; pumps (momentum sources), return (mass source), overflow (sink); pump modes; inflow turbulence; time-averaging and RMS; VTK export; CLI | Pump jets carry the specified momentum flux J = ρQ²/A: 0.95–1.01 (verification 7). Mass drift ≤ 3 × 10⁻⁶ |
| 3 Metrics | Dead zones (ranked, with core locations), steady water age, sand/rock shear by wall function, detritus map, coral scores (flow × light) with best spots, HTML report, comparison mode, derating sensitivity | Age solver exact to 10⁻⁸ on plug flow and to 2 × 10⁻¹² on the outlet identity. Derating shortcut confirmed by rerun |
| 4 Lighting / PAR | Monte Carlo photon tracer (Fresnel, Snell, TIR in glass, Kd, rock shadowing, bounces); fixture library; calibration with immersion metadata; PAR/DLI maps; uniformity; height sweep; comparison | All 8 photometric verification cases pass |
| 5 Browser UI | FastAPI + three.js (vendored, offline). 7 tabs: Tank, Photos, Rockscape designer (rock/slab/pillar/arch/bridge, union → STL), Pumps & lights, Run (live progress, clean cancel), Results (3D, slices, dead zones, shear, coral scores, PAR), Compare | 25 API tests; headless-Chrome check of every tab, including the full photo workflow |
| 6 Photo reconstruction | PnP calibration with refraction through glass and water; GrabCut segmentation (SAM optional); mask editing; visual hull; carve/add tools; STL export; volume and weight | 7 of 8 validation criteria pass; one fails (below) |
| 7 Polish | Compute-graph stepping (bit-identical; Preview runs 3.2× faster); installer + self-check; README; precision check | fp32 vs fp64 on the demo tank: differences within run-to-run noise |

## Results you should look at first

**1. Near-surface flow is resolution-limited; this is the main open issue.**

Verification 6, grid study on the demo tank (D3Q19):

| Metric | Preview | Standard | Fine | Standard → Fine |
|---|---|---|---|---|
| Mean speed (cm/s) | 10.85 | 11.98 | 12.57 | +5 % ✅ |
| Dead-zone volume | 6.6 % | 3.6 % | 2.5 % | −32 % ❌ |
| Speed 15 mm above sand (cm/s) | 3.7 | 8.0 | 9.8 | +23 % ❌ |

- A channel-flow test at tank-like near-wall resolution carries 28 % too little flow for its wall friction (verification 8).
- **Meaning:** near-wall flow is under-predicted. Dead-zone volume is an upper bound, and sand shear and flow over rock read low.
- Bulk flow, and comparisons between designs on the same grid, are much less affected.
- I tried a wall model; it failed verification (diverged) and is left off (D-702).
- **Recommendation:** use Standard or Fine for near-surface numbers, and treat absolute dead-zone volumes as conservative.

**2. Pump model corrected during the sprint.**
- The first version put a solid pump body behind the force region, and the jet carried only 50 % of its momentum.
- Housings are now off by default, and jets carry 96–100 % of J.
- The inflow-turbulence default was recalibrated from 0.15 to 0.05. It changes tank-wide mean speed by only ±5 % over 0.02–0.10, but a single jet's throw by a factor of about 1.5–2.

**3. D3Q19 vs D3Q27 on the tank.**
- They agree on bulk metrics within 0.4–2 % at Standard; D3Q27 reads 19 % more dead volume (near-wall sensitive) and runs 2× slower.
- This supports the D3Q19 choice you approved. On the Fine grid, mean speed is 12.55 vs 12.57 cm/s and dead volume 2.88 vs 2.45 %.

**4. Photo reconstruction overfills.**
- With 3 perfect views, IoU is 0.679 against the pre-declared 0.70 (volume +47 %). Kept as a failure.
- Cause: silhouettes cannot see the dips between rock blobs. A fourth, oblique photo helps (0.705), and the edit tools are there for this. The reconstructed volume is an upper bound.

**5. The demo lighting is strong:** about 400 µmol m⁻² s⁻¹ on rock tops, uncalibrated. So coral scores favour SPS; LPS and soft corals are penalised for excess light.

## Follow-up after your review (2026-10-31)

| Request | What was built | Evidence |
|---|---|---|
| Easier rock carving | Rockscape tab → **Sculpt the rockscape**: Pull / Push / Smooth / Flatten brushes. Hold to dig deeper; the wheel sets the brush size. Works on built pieces, imported scans and photo reconstructions, with undo/redo, drafts and Apply to project (D-820 to D-825) | Picks land on the surface to 10⁻⁵ m; sculpt volume within 0.02 % of the exported mesh; an unedited round trip changes CFD rock volume by under 0.2 %; a sculpted 115 cm³ cave changes the CFD rock by the same 115 cm³ |
| Overflow anywhere along the glass | `wall` + `center` form; slide it along the glass in the view; coast to coast (D-801 to D-804) | The old corner form gives the identical footprint |
| Sculpt round 2 (2026-11-03): openings were hard to make, brush tilted at odd angles | Brush locked to a straight direction: the tank axis nearest your view, or a fixed one turned/tilted in 15° steps. Push drills a straight round hole with a preview of where it breaks through, and stops when through (or bores through in one click); Pull extrudes along the same line; Flatten can cut a vertical face; left-drag off the rock orbits; caves render darker (D-826 to D-830) | Predicted exit 176 mm vs 174.5 mm of rock on the line; hole radius 20.0–20.2 mm for R = 20 mm; holding after breakthrough removes 0 cm³; Pull 1 s at 50 % moves the face 49.9 mm (expected 50) |
| Calibration and rock mask for real (rimmed) tanks (2026-11-03) | Click the **outside** corners (top of the frame, where the tank sits on the stand), the back water-line corners and the back corner joints, with a where-to-click sketch, a magnifier and an expected-position ring. A **tank-size check** says when the entered size cannot fit the photo. The mask suggestion uses **texture** on side photos, and only rough areas that reach the sand (D-836 to D-839) | Your front photo: fit 34.9 px with the old clicks; 24 px with the new clicks at 21 in; 6.1 px at 23 in. The check reports 22.8 ± 0.2 in tall. Synthetic mask IoU 0.993–0.998 (colour unguided: 0.51 on the side view); photo validation outcome unchanged |
| Sculpt round 3 (2026-11-11): the brush barely changed the rock; wanted a Planet Coaster-style push / pull | **Raise / Lower** terrain brush: everything under the circle that faces you moves toward / away from you at a steady speed, most in the middle; passes add up, holding still grows a dome or hollow; never the wrong way; 10 mm gap at the glass kept. The straight drill is now **Tunnel** (Ctrl: Extrude). Default brush 50 mm (D-843) | Old: 3 s scrub moved the surface 0.2 mm (airbrush) or not at all on repeat passes (drill). New: 40.7 mm in a 1 s hold in the browser (design 40), half-radius 0.56 of centre (design 0.56), two passes = 2.0x one pass, 0 nodes changed the wrong way, undo exact |
| Release prep (2026-11-21): lid, particles, demo, guide, video | **Lid over the tank** (glass, acrylic, window screens, netting, egg crate; D-848). **Particle animation** in Results (D-849). **Demo aquascape** built from designer pieces, with a new broken-rock look (D-850). **Illustrated guide** in the app (Help button). **2-minute captioned walkthrough video** (docs/media, recorded from the app; tools/video) | Lid: tracer matches the analytic transmission within 0.2 % (glass 88 %, low-iron 91 %, acrylic 92 % straight down); open lid bit-identical to no lid. PAR at 20 cm vs open top: glass 87 %, black 1/8 in netting 68 %. Particles: 2 ms/frame for 3,000. Demo standard run: 12 min, mean speed 11.7 cm/s, 4.2 % dead water. 413 tests pass |
| Look of the first release (2026-11-21) | Light working panel with a deep-water 3D view; styled controls; step tabs with ticks for finished steps; foldable panel sections; welcome dialog with the new logo (D-847). Display only | Before / after screenshots of every tab; 48 UI tests pass |
| Real equipment (2026-11-12): pumps, powerheads, lights and overflows | **Model** lists with 35 powerheads, 41 return pumps, 30 lights and 21 overflows, each with sources, the date checked and the basis of every number. Published return-pump curves are used as given; DC pumps without one use a straight line. Light output fitted to published PAR readings where they exist. Warnings for return flow above the overflow capacity and for a nozzle exit head above the fittings allowance (D-844 to D-846) | Light fits match the published readings to 3-11 % rms (1.14-1.76 umol/J). The old generic curve read DC pumps 17 % high on average (up to 38 %); a straight line is within 7.7 %. 159 catalogue tests; every product builds into the simulation objects; browser check of all four pickers |
| Other ways to make rock modelling easier (2026-11-03) | **Phone 3-D scans** (Rockscape → 📱 Import phone scan): OBJ/GLB/PLY/STL/ZIP from Polycam, Scaniverse and similar apps. Finds the up direction and the table, cuts the table away, closes holes and the open bottom, drops loose bits, scales to a tape measurement, outputs a watertight rock element. **Phone photos** (Photos → 📱 Use my phone): QR code → guided shot list on the phone (filled or dry tank) → photos land in the Photos tab (D-831 to D-835) | Synthetic scans: volume 0.98–1.00 of truth, IoU 0.97–0.99, 50 mm tunnel kept open, up axis right in all 10 cases including inside-out and wrong-convention files; about 1 s per rock. Phone link: upload, retake, permissions and stop checked; page fits a 390 px phone screen |
| Several return nozzles on one pump | `nozzles:` list. Default split proportional to area; each nozzle is its own source and turbulence process (D-805 to D-808, A-801, A-802) | Source = sink to machine precision; mass drift 1.4 × 10⁻⁸ |

## Open items

- A robust wall model for near-surface accuracy (D-702).
- Transient water-age cross-check (D-302). Deferred; the demo's age is nearly uniform.
- Darcy–Brinkman porous rock (D-212). Deferred.
- Photo auto-segmentation can fail on some views (one full-size synthetic side view: mask IoU 0.51; D-705). The mask tools fix it, but treat the suggestion as a starting point.
- SAM segmentation path: written, never run (no weights installed); model sha256 hashes not pinned.
- Lighting: in-water scattering and per-channel schedules not modelled; calibration fits one scale for all fixtures.
- UI: single user; the job list resets on restart (runs stay on disk).
- First commit made 2026-11-21 (2880d01). Your own projects in reefflow_workspace/ are kept out of git.
