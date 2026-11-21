# ReefFlow: program overview for planning a rock scanner

**Purpose of this file.** It describes what ReefFlow does and how rock geometry gets into it today, so a 3-D scanner (lidar or similar) can be planned to fit. It covers the data the scanner has to produce, the accuracy the simulation needs, and the physics of scanning a filled reef tank.

**Labels used here.** "Today" means built and tested. "Proposed" means a plan, not built.

Status as of 2026-11-07. Nothing is committed to git yet.

---

## 1. What ReefFlow is

ReefFlow simulates a reef aquarium on your own PC. It runs entirely offline after installation.

**Water flow:** a GPU lattice Boltzmann CFD solver. It runs on Taichi, and uses CUDA on NVIDIA cards, Vulkan on other GPUs, or the CPU as a fallback.

**Light:** a Monte Carlo PAR ray tracer.

**Outputs:**
- dead spots;
- water age;
- detritus settling (sand shear);
- flow over the rock;
- PAR/DLI maps;
- coral placement scores.

**Results:**
- an HTML report;
- ParaView files;
- 3-D meshes;
- comparisons between designs.

The **rockscape** is the biggest input, and the one users find hardest to get right. Rock blocks and steers the flow, so its shape, the openings through it, and the gaps between it and the glass decide where the dead zones are. A scanner is meant to make this step fast and accurate.

## 2. How it is built

| Part | Technology | Where |
|---|---|---|
| CFD core | Lattice Boltzmann (D3Q19/D3Q27, recursive-regularised collision, Smagorinsky LES) in Taichi | `reefflow/solver.py`, `kernels.py`, `lattice.py`, `domain.py` |
| Tank model | Pumps, overflow, rock voxelisation, project file | `reefflow/project.py`, `tank.py`, `geometry.py`, `pumps.py`, `sources.py` |
| Analysis | Dead zones, water age, wall shear, coral scores, report | `reefflow/metrics.py`, `reefflow/analysis/` |
| Light | Photon tracer, fixture library, PAR/DLI | `reefflow/light/` |
| Photo reconstruction | Camera calibration through glass and water, rock masks, visual hull | `reefflow/photo/` |
| Phone scan clean-up | Raw scan → watertight rock mesh | `reefflow/scan.py`, `reefflow/ui/scan_routes.py` |
| Sculpt tool | Density grid edited with brushes in the browser | `reefflow/ui/sculpt.py`, `static/js/sculpt.js` |
| User interface | FastAPI server + three.js in the browser (`ReefFlow.bat`) | `reefflow/ui/` |

**Environment:**
- Python 3.12, installed in `.venv` by `install.ps1`.
- Every dependency has a licence that allows commercial use (see `LICENSES.md`).
- The suite has 228 automated tests. Design choices with their numbers are in `DECISIONS.md`, physical assumptions in `ASSUMPTIONS.md`.

## 3. The pipeline, end to end

```
project.yaml ──► tank + rockscape + pumps + overflow + lights
                      │
   rock sources ──────┤  (designer shapes, imported meshes, phone scan, photos, sculpt)
                      ▼
            rockscape STL (tank coordinates)
                      ▼
      voxelised onto the CFD grid (a cell is rock if its centre is inside the rock)
                      ▼
      CFD spin-up (60 s) + averaging (120 s)  ──►  analysis + light  ──►  report.html, .vti, .ply, .json
```

**Project file.** Each tank is one readable YAML file. Quantities carry units, for example `"48 in"`, `"2500 GPH"` or `"60 s"`. Internally everything is SI (metres, seconds, kilograms). Your tank's file:

```yaml
tank:
  external: {length: 48 in, width: 18 in, height: 23 in}
  glass_thickness: 0.5 in
  water_level_below_rim: 1 in
  sand_depth: 1 in
rockscape: []          # list of {file, unit, translate, rotate_z, scale}
designer: {elements: [], resolution: 0.004, rock_density: 1600.0, ...}
```

The inside of your tank is 47 × 17 × 22.5 in (1194 × 432 × 572 mm). The water column above the sand is about 20.5 in (521 mm).

## 4. Tank coordinate frame (the scanner must output in this frame, or be convertible to it)

**Origin:** the inside bottom-left-front corner of the glass box. "Left" is as seen from the front.

**Axes:**
- **x** runs along the length, left → right, from 0 to L (inside length).
- **y** runs front → back, from 0 to W (inside width).
- **z** points up from the inside face of the bottom glass. The sand surface is at z = sand depth, the water surface at z = water level, and the top of the glass at z = H.

**Units:** metres in code. Meshes are stored in mm by default; the unit is stated per file.

**Outside corners** used by photo calibration:
- x ∈ {−g, L+g} and y ∈ {−g, W+g}, where g is the glass thickness;
- the top corners are at z = H;
- the base corners are at z = −(bottom glass thickness).

These corners match the external size the user enters, so the tank itself is a known calibration object. A scanner can use the same corners to place itself; see §8.

**Pixel convention** (for camera-based scanners): OpenCV. u points right and v points down; the camera's own axes are x right, y down, z forward.

## 5. How rock gets into ReefFlow today

| Source | What the user does | What it produces | Limits |
|---|---|---|---|
| **Designer** | Places parametric shapes: rock, slab, pillar, arch, bridge, mesh | A signed-distance field per element → marching cubes at 4 mm → STL | Hand-built, so only an approximation of real rock |
| **Mesh import** | Adds STL/OBJ/PLY files with unit, translate, rotate_z, scale | Rockscape entries in `project.yaml` | The mesh must be closed (watertight) |
| **Phone scan import** (Rockscape → 📱 Import phone scan) | Scans **dry** rock with Polycam, Scaniverse or a similar app; uploads OBJ/GLB/glTF/PLY/STL or a ZIP | A watertight rock element, scaled to a tape measurement | Dry rock only. **Meshes only: a point cloud is rejected.** Each rock is placed in the tank by hand afterwards. |
| **Photo reconstruction** (Photos tab) | Top + front + side photos; clicks tank corners; checks the rock mask | A visual hull (space carving on a 5 mm grid) → `rock_photo.stl` | Silhouettes can't see caves, dips between blobs or hidden gaps, so the hull **overfills**: validation IoU is 0.679 against a 0.70 target, and volume comes out +47 %. It needs manual carving. |
| **Sculpt** | Pushes and pulls the rock surface with brushes | Edits a density grid (0 = water, 255 = rock), then marching cubes → STL | Manual. The brush is being reworked: the user wants a Planet Coaster-style push/pull. |

**Details that matter for a scanner:**

**Phone scan clean-up** (`scan.py`, `clean()`) already handles messy real-world scans:
1. Guesses the unit and applies the scale.
2. Finds the up axis from the table the rock sat on.
3. Cuts the table away.
4. Builds a solid on a voxel grid. It casts 26 rays per voxel, so holes and an open bottom are handled.
5. Drops loose pieces, fills sealed voids, flattens the base, and runs marching cubes.

The output is centred on x = y = 0 with its base at z = 0, in metres.

**The photo pipeline already models refraction exactly.** `photo/camera.py` traces rays through air | glass | water (side views) or air | water (top view) with Snell's law on flat parallel planes, with no small-angle shortcut. **Any optical scanner looking through the glass can reuse this to correct its rays.**

**The occupancy grid** (`photo/carve.py`, `VoxelGrid`):
- node (i, j, k) has centre `origin + (i, j, k)·dx`;
- the edit tools (`photo/edit.py`) carve or add spheres, capsules, boxes and arches, and replay edits as a JSON list.

A scanner's output can feed straight into this grid.

## 6. What accuracy the simulation needs

| CFD preset | Cell size | Cells (75 gal) | Run time (RTX 3070) |
|---|---|---|---|
| preview | 10 mm | 0.25 M | ~5 min |
| standard | 5 mm | 2 M | ~13 min |
| fine | 3 mm | 8.9 M | ~75 min |

Rock is voxelised at cell centres, so:
- **Surface error well below one cell is invisible to the CFD.** A target of **≤ 2 mm RMS** covers every preset; 3–5 mm is fine for preview and standard.
- **Openings matter more than surface detail.** A gap or tunnel narrower than about 2 cells (6–10 mm) is effectively closed in the simulation. Missing a 30 mm cave changes the flow; missing 1 mm of surface texture does not.
- **Getting the free space right is the most valuable thing a scanner can add.** The photo method can only remove what every view sees past. A range sensor knows the water is empty along each ray up to the hit, which removes the photo method's overfill.
- **Coverage:** the back faces of rock against the back glass, and the undersides of overhangs, are hard to see from any side. Unseen space has to be marked "unknown", not guessed as rock or water (§9).

## 7. Physics of scanning a filled reef tank (read before choosing hardware)

### 7.1 Water absorbs infrared: most lidar wavelengths will not work through it

Most lidar uses near-infrared:
- automotive lidar and hobby time-of-flight sensors at 850–940 nm;
- the iPhone/iPad LiDAR at about 940 nm;
- long-range lidar at 1550 nm.

Water absorbs these strongly. The table gives approximate pure-water absorption coefficients from the published literature (please re-check the exact values before buying hardware). The last column is the transmission over a **0.9 m round trip**: through the front glass to the back of your 17 in deep tank and back.

| Wavelength | Typical source | Absorption ≈ | Round-trip transmission (0.9 m) |
|---|---|---|---|
| 450 nm (blue) | Blue laser diode | 0.01 /m | ~99 % |
| 520–532 nm (green) | Green laser line, bathymetric lidar | 0.04–0.05 /m | ~96 % |
| 650 nm (red) | Red laser line | 0.3–0.35 /m | ~73 % |
| 850 nm | Many ToF cameras | ≈ 4 /m | ~2 % |
| 905 nm | Most hobby/automotive lidar | ≈ 6–7 /m | ~0.3 % |
| 940 nm | iPhone LiDAR, many ToF sensors | ≈ 25–30 /m | ≈ 0 |
| 1550 nm | Long-range lidar | ≈ 1000 /m | 0 |

**Conclusions:**
- **In the water, use blue or green light.** Red works over short paths.
- **Phone LiDAR and off-the-shelf IR lidar** are only useful for **dry rock**, which ReefFlow already supports, or for the outside of the tank: finding the glass box and the stand.
- Ocean "bathymetric" lidar uses green (532 nm) for exactly this reason.

### 7.2 Refraction and the speed of light in water

**Refraction.** A ray bends at the air → glass → water faces (n = 1.00 / 1.50 / 1.34). Without correction, rock seen through the front glass appears about 25 % closer at normal viewing angles, and more distorted towards the edges. `camera.py` already corrects this exactly for flat glass.

**Time of flight** is slower in water by n ≈ 1.34. A ToF sensor calibrated for air reads in-water distances about 34 % too long. It also measures time along a bent path. Both have to be corrected per ray. That is possible, but it ties the sensor's internals to the tank model, which is one more reason to prefer **triangulation** (laser + camera, or projector + camera). Triangulation only needs ray directions, and those the refraction model already handles.

**Timing precision.** 1 mm of range is about 6.7 ps of round-trip time in air (about 9 ps in water). Millimetre-level direct ToF needs specialised timing hardware. Triangulation gets 0.5–2 mm over a 0.5 m range with ordinary cameras.

### 7.3 Other practical problems

**Glass reflections.** The front glass reflects part of the beam straight back (about 4 % per air/glass face), which shows up as a ghost hit or a bright flare. Ways to handle it:
- tilt the sensor 5–15° off square to the glass;
- use a polariser;
- reject hits closer than the inside face of the glass, a distance the tank model knows.

**Water surface (top view).** Ripples from pumps bend rays randomly. Turn the pumps off and wait, or float a flat clear acrylic plate on the surface as a window. A surface skimmer and overflow flow also ripple the water.

**Bubbles and particles.** Micro-bubbles (seen in your latest photo) scatter light and give false near hits. Mitigations:
- pumps off for a few minutes before scanning;
- filter isolated points (a hit with no neighbours);
- require several sweeps to agree.

**Lighting.** Turn the reef lights off, or at least down. Blue actinics swamp a blue laser line, and a green line is easier to see against blue light.

**Fluorescence.** Corals glow under blue and violet light, and a blue laser can make them glow. A green line, with a band-pass filter matched to the laser on the camera, avoids that.

**Motion.** Fish, corals and soft tissue move. Scan fast, and take a few sweeps so moving things can be voted out. Soft corals will be ragged in the result. Brush them out, or keep them on purpose.

**Occlusion.** One viewpoint sees only the faces turned towards it. Plan for several viewpoints: front, both ends, and top, if the lights can be moved.

**Laser safety.** Use a Class 2 or 3R line laser (≤ 5 mW). Never look into the beam or its reflection in the glass; the glass can send it back into the room. Higher-power lasers need eye protection and are not worth the risk here.

## 8. Scanner concepts compared

| Concept | Works in a filled tank? | Expected accuracy | Effort / cost | Fit with ReefFlow |
|---|---|---|---|---|
| **A. Phone LiDAR / photogrammetry of dry rock** | No (dry rock only) | 2–5 mm, scale from a tape | None: works today | Built: phone scan import. Placing each rock in the tank is manual. |
| **B. Turntable + structured light or photogrammetry, dry rock** | No | 0.1–1 mm | Low to medium | Same import path as A, better detail |
| **C. Green line laser + camera on a rail outside the glass (laser triangulation)** | **Yes**, through the glass | ~0.5–2 mm | Medium, DIY-friendly | Best fit: reuses the refraction model and the tank calibration, and gives free-space information |
| **D. Projector + camera (structured light) through the glass** | Yes, but the projector must be calibrated through the glass too | ~0.5–1 mm, dense, fast | Medium to high | Good. More calibration work than C. |
| **E. Green-wavelength ToF / underwater lidar** | Yes | mm-level needs expensive timing hardware | High (commercial) | Needs per-ray speed and refraction correction (§7.2) |
| **F. Stereo cameras (passive) through the glass** | Yes | 1–5 mm on textured rock | Low to medium | Easy to add. Fails on smooth surfaces, and bubbles cause false matches. |

**Recommendation:** use A now for any rock you can take out (no new hardware). Build **C** as the in-tank scanner.

C uses a green line laser and a global-shutter camera fixed together on a motorised linear rail running along the front of the tank, about 20–40 cm from the glass, with the laser sheet roughly vertical. Each camera frame gives one line of 3-D points. The rail sweeps the whole front. Repeat from each end, and from the top if the lights can be moved.

Why C:
- It uses visible green light, which water barely absorbs.
- Each point comes from the camera ray meeting the laser plane. Both refract at the glass, and `camera.py` already traces that exactly.
- The tank corners and glass give a free, accurate reference for calibration.
- Each detected point also says "the water is empty from the glass to here", the free-space information the photo method lacks.

## 9. Where a scanner plugs into ReefFlow (proposed)

There are three possible levels. **Level 2 is recommended.**

**Level 1: mesh file (no code changes).** The scanner software builds its own mesh in tank coordinates and saves it as STL/PLY in mm. It is added as a rockscape mesh, or through the phone scan import with up axis "+z", unit "mm" and ground cut off.
- Pros: works today.
- Cons: no free-space or unknown information, holes must be closed by the scanner software, and no way to merge with photos.

**Level 2: scan session file → new fusion module.** The scanner saves its points **and where each was seen from**, in tank coordinates. A new module `reefflow/scanfuse.py` fuses them into the existing `VoxelGrid`:
1. **Free space:** every voxel a ray passes through before its hit becomes water.
2. **Surface:** voxels at hits become rock, with the surface position kept as a signed distance so the result isn't stair-stepped. This reuses the method in `scan.py`.
3. **Unknown:** voxels no ray reached are resolved by a rule:
   - behind a seen surface, as seen from every sensor: rock (the solid interior);
   - otherwise use the photo hull if one exists, or leave it as water and flag it in the report.
4. Result: occupancy → marching cubes → rockscape STL. The same "Use as rockscape" step as the photo and sculpt tools, with the edit and sculpt tools available afterwards.

It can combine with the photo hull: final rock = (photo hull) minus (scanner free space), plus (scanner surface). That removes the overfill the photo method can't fix.

**Level 3: live device in the UI.** A "Scan" tab drives the rail and camera, shows points arriving, and flags uncovered areas ("scan from the right end next"). It needs device drivers, so it is best left until the hardware is proven.

### 9.1 Proposed scan session format (Level 2)

One folder per scan session:

```
scan_session/
  session.json
  sweep_01.npz        # one file per sweep (one rail pass from one side)
  sweep_02.npz
  calib/              # images used for calibration (optional, for re-processing)
```

`session.json`:

```json
{
  "format": "reefflow-scan/1",
  "units": "m",
  "frame": "tank",                       
  "tank": {"external_m": [1.2192, 0.4572, 0.5842], "glass_m": 0.0127},
  "sensor": {"type": "laser_line", "wavelength_nm": 520,
              "camera": {"width": 1920, "height": 1200, "f_px": 1400.0, "cx": 960, "cy": 600, "k1": 0.0}},
  "refraction_corrected": true,
  "sweeps": [
    {"file": "sweep_01.npz", "view": "front", "note": "pumps off, lights off"}
  ]
}
```

Each `sweep_XX.npz`:

| Array | Shape | Meaning |
|---|---|---|
| `points` | N × 3 float32 | Hit positions, tank frame, m (already refraction-corrected) |
| `origins` | N × 3 float32 | Where each ray **enters the water**, at the inside face of the glass or the water surface, tank frame, m |
| `quality` | N float32 | 0–1: line peak strength or sharpness, used to weight or reject points |
| `frame_id` | N int32 | Which camera frame or rail position the point came from (lets moving fish be voted out) |
| `misses` *(optional)* | M × 6 float32 | Rays that reached the back glass without hitting rock (origin + end): free space with no rock |

The ray's path inside the water is the straight line `origins → points`. The fusion module only needs that, because refraction has already been handled. If the scanner software can't correct refraction, save the raw camera rays instead (`ray_px`: N × 2 pixel positions plus the rail position). ReefFlow can then correct them with `camera.py`.

## 10. Software work in ReefFlow (proposed, by phase)

| Phase | Work | Reuses |
|---|---|---|
| 0 | Nothing: scan dry rock with a phone. | `scan.py` |
| 1 | **Accept point clouds** in the scan import, for scanners that can't make meshes. Today `scan.load_raw` rejects them. Add a Poisson or voxel surface step. | `scan.py` voxel solid |
| 2 | **Synthetic laser scanner** for testing before any hardware exists: render laser-line images of the demo rock through glass and water, with bubbles and noise. | `photo/synthetic.py`, `camera.py` |
| 3 | **Laser-line processing**: find the line in each frame (sub-pixel peak); triangulate through the glass (camera ray meets the refracted laser plane); calibrate the camera and laser plane from tank corners plus a target. | `camera.py`, `calibration.py` |
| 4 | **Fusion module** `scanfuse.py` (§9): free space, surface, unknown; merge with the photo hull. | `carve.VoxelGrid`, `scan.py`, `edit.py` |
| 5 | **UI**: import a scan session, show coverage, "Use as rockscape". | `scan_routes.py` pattern |
| 6 | **Device control (optional)**: drive the rail and camera from the Scan tab. | — |

**Validation plan** (matches how the photo tools were checked):
- **Synthetic:** scan the demo rock in software. Report surface RMS error (mm), IoU against the true rock, volume error, and how much of the photo method's overfill is removed. A useful target: IoU ≥ 0.90 against the photo method's 0.679.
- **Known objects in the real tank:**
  - a PVC pipe or ball of measured size, to check scale and shape;
  - a 25–50 mm tunnel, to check that openings survive;
  - a rock of measured displacement volume, to check overall volume.
- **Repeatability:** scan twice, compare surface differences (mm).

## 11. Hardware sketch for concept C (to refine)

| Item | Notes |
|---|---|
| Line laser, green 515–532 nm, ≤ 5 mW (Class 2/3R), fine line (< 1 mm at 0.5 m) | A diode at 515–520 nm is cheaper and more stable than a 532 nm DPSS laser |
| Global-shutter camera, ≥ 1.3 MP, USB3, fixed lens about 6–12 mm | A global shutter avoids line distortion while moving. Fit a band-pass filter matched to the laser on the lens. |
| Linear rail ≥ 1.25 m (your tank is 48 in), stepper + belt or lead screw, end-stop | The position is logged per frame. It doesn't have to be precise if the frames are also tied to tank features. |
| Rigid bracket holding the camera and laser 15–25 cm apart, angled 20–35° to each other | Wider spacing gives better depth accuracy but more occlusion |
| Controller (Raspberry Pi, Arduino + PC, or the PC directly) | Drives the stepper, triggers the camera |
| Calibration target (printed ChArUco board on a flat plate) | Used dry for the camera's own calibration; the tank corners do the rest |
| Optional: rotation stage or second mount for the end views, and a flat acrylic surface plate for the top view | Covers what the front can't see |

Rough capture numbers (to check): at 1 mm rail steps, a 48 in sweep is about 1200 frames. At 30–60 frames/s that's under a minute per sweep, plus processing.

## 12. Open questions to decide

1. **In-tank or dry?** Is scanning the rock in the filled tank the goal, or is dry rock during a new setup (where a phone already works) enough for most users?
2. **Accuracy target:** is 2 mm (all presets) needed, or is 5 mm enough?
3. **Viewpoints:** front only, or front + ends + top? This decides how much of the back and the undersides is covered.
4. **Who processes the frames?** Scanner software outputs points (Level 2, recommended), or ReefFlow takes raw frames and does the triangulation itself (more reuse of `camera.py`, tighter coupling)?
5. **Budget and build skills:** rail, electronics, 3-D printed brackets.
6. **Product direction:** a DIY add-on for your own tank, or a kit other reefers could build? A kit needs a simpler calibration and wider camera support.

## 13. Key files

| File | Why it matters for a scanner |
|---|---|
| `reefflow/photo/camera.py` | Exact refraction through glass and water; pixel ↔ ray in tank coordinates |
| `reefflow/photo/tankref.py` | Tank coordinate frame; named corner points for calibration |
| `reefflow/photo/calibration.py` | Solves camera pose and focal length from clicked tank corners |
| `reefflow/photo/carve.py` | `VoxelGrid`, space carving, mesh export, volume and weight |
| `reefflow/photo/edit.py` | Carve/add tools, edits replayable as JSON |
| `reefflow/photo/synthetic.py` | Renders synthetic tank photos: the base for a synthetic scanner |
| `reefflow/scan.py` | Raw scan → watertight mesh (units, up axis, ground cut, voxel solid) |
| `reefflow/ui/scan_routes.py` | HTTP pattern for uploading and cleaning scans |
| `reefflow/geometry.py` | Mesh loading and voxelisation onto the CFD grid |
| `reefflow/project.py` | Project file format, presets (cell sizes), rockscape entries |
| `DECISIONS.md` D-601…D-842 | Photo and scan design choices with their measured numbers |
| `verification/VERIFICATION.md` §10 | Photo reconstruction validation (IoU 0.679, overfill) |
