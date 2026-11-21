# ASSUMPTIONS.md — physical and modeling assumptions

Each entry states the assumption, its justification, and its expected effect on results. Threshold values for coral husbandry metrics (Phases 3–4) are **starting assumptions, not authoritative values**.

## Fluid properties

**A-101 Seawater density ρ₀ = 1023 kg/m³** (reference density for lattice ↔ physical conversion).
- Reef tanks are typically kept at SG 1.025–1.026 (hydrometer convention, 25 °C), which is ≈ 1022–1023 kg/m³.
- Only affects force, pressure and momentum-flux conversions, linearly. A 0.2 % change has no practical effect.
- Phase 0 draft said 1025; corrected here.

**A-102 Kinematic viscosity ν = 0.94 × 10⁻⁶ m²/s** (seawater, 25 °C, S = 35 g/kg; Sharqawy et al. 2010).
- Irrelevant in practice at tank resolutions; see A-104.

**A-103 Incompressible flow approximated by weakly compressible LBM, with lattice velocity u_lat ≤ 0.1 (Ma = u/c_s ≈ 0.17).**
- Density fluctuations scale as Ma² ≈ 3 % at the peak-velocity location and are far smaller elsewhere.
- The artificial sound speed is c_s = dx/(√3 dt) (≈ 5.8 m/s at 5 mm cells, 1 m/s peak), not 1500 m/s. Acoustic transients therefore travel slowly but are irrelevant to time-averaged circulation.

## Turbulence

**A-104 Tank flow is computed as an under-resolved LES; molecular viscosity is effectively negligible.**
- At 5 mm cells and dt = 0.5 ms, ν_lattice ≈ 2 × 10⁻⁵ (τ − ½ ≈ 6 × 10⁻⁵).
- Dissipation comes almost entirely from the Smagorinsky model and the collision operator. Results are model-dependent by construction.
- Mitigated by: the jet verification (test 4), the grid-convergence study (test 6, Phase 2), and reporting sensitivity rather than single numbers.

**A-105 Smagorinsky constant C_s = 0.10 (user-approved default; see D-118), user-editable, with no near-wall damping.**
- C_s = 0.17 (Lilly 1967, isotropic turbulence) over-damped the jet shear layer in verification.
- C_s = 0.10 reproduces far-field jet decay and spreading (B = 5.21, S = 0.104).
- The near field remains too laminar at D = 10 cells (potential core 8.4 D vs ~4–6 D measured).
- Without van Driest damping, eddy viscosity near walls is overpredicted. That is acceptable because near-wall flow is unresolved anyway (y⁺ ≈ 12–25 at the first node) and wall shear is reported through a wall function (D-006).

## Boundaries and geometry

**A-106 Geometry is voxelized (staircase).** With halfway bounce-back, wall position uncertainty is up to ±dx/2 locally.
- The BGK wall-location error is τ-dependent: uniform slip u_s = g(16(τ−½)² − 3)/(24ν) in Poiseuille flow, verified against He et al. (1997).
- Curved-boundary schemes (Bouzidi interpolated bounce-back) are a possible later refinement.

**A-107 Moving-wall correction uses ρ_w = 1.** Error O(Ma²); only used in the cavity verification.

**A-108 Open-boundary nodes (inlet/outlet) carry no body force.** Respected by construction in all cases.

## Numerics

**A-109 fp32 arithmetic and storage, with shifted populations.**
- **Measured round-off floor:** in steady, weakly forced laminar flow, rounding of the per-step force increment repeats identically every step. It biases the steady state by ≈ ulp × H²/ν.
- Poiseuille at τ = 0.8:
  - The floor is negligible for H ≤ 32 (fp32 and fp64 agree to < 1 %).
  - It is ~10⁻⁴ relative at H = 64.
  - At the "exact" magic τ the fp32 error is 10⁻⁶–2 × 10⁻⁵ against 3 × 10⁻⁹ in fp64.
- **Relevance to tank runs:** low. Those are unsteady (rounding decorrelates), use acoustic scaling with u_lat ≈ 0.1 (large per-step increments relative to ulp), and the quantities of interest are time averages at the percent level.
- **Re-checked in Phase 2 on the demo tank** (Preview, 180 s; verification/VERIFICATION.md, precision section).
  - Runs: fp32 GPU at two random seeds, and fp64 CPU at seed 1.
  - For all 10 metrics, |fp64 − fp32| is no larger than the seed-to-seed difference or the 2σ statistical noise. Example, mean speed: 10.84 vs 10.85 cm/s, against 10.64 cm/s for the second fp32 seed.
  - Conclusion: fp32 is adequate for tank runs.
- Closed-box mass drift is reported in verification test 5.

## Phase 2 — tank model and pumps

**A-201 Sand bed = flat, smooth, no-slip solid floor at the user's sand depth** (spec). Sand roughness and ripples are not modelled. Wall shear on the sand comes from a wall function in Phase 3 (D-006).

**A-202 Water surface = rigid free-slip lid** (spec, PLAN C10), at whole-cell height. The difference from the true water level is reported: −1.0 % of the water column at Preview, +0.1 % at Standard, for the demo. There are no surface waves, so no momentum is lost to them.

**A-203 Powerhead = momentum source of total force J = ρQ²/A at the outlet** (spec).
- No net mass: the pump recirculates.
- The source region doubles as the propeller disc and the intake. Water is drawn from behind and around it, as through a real propeller's intake cage.
- Not modelled:
  - swirl, apart from the random lateral fluctuations of A-207;
  - the solid pump body and mounting magnet. Housings are off by default (D-202).
- Expected effect: correct jet momentum (verified, case 7), but no blockage from the body (a few cm) and no swirl-driven spreading close to the pump.

**A-204 Return-pump flow from a generic centrifugal curve Q = Q₀√(1 − H/H₀).**
- H = static head + fittings allowance (default 1 ft; editable). The user can override the flow directly.
- The demo pump is rated 800 GPH with 10 ft shut-off, working at 4 + 1 ft, which gives 566 GPH.

**A-205 Powerhead derating (delivered/rated flow) default 0.7, editable.**
- There is no reliable published basis for one number: free-flow ratings are measured differently by each manufacturer. This is a starting assumption.
- Mean speeds scale roughly linearly with delivered flow, so ±20 % derating means roughly ±20 % on speeds (PLAN C4).
- The Phase-3 report will show dead-zone volume at ±20 % derating.

**A-206 Equivalent outlets** (D-203). Outlets narrower than 4 cells, or with equivalent velocity above 1.0 m/s, are modelled over a larger area at the same momentum flux J.
- The far field is unaffected (it depends on J only).
- Within ~5 equivalent diameters of the outlet, the speed is lower and the jet wider than in reality.
- Demo, all grids:
  - return nozzle 19 mm at 2.1 m/s → 40 mm at 1.0 m/s;
  - left powerhead 32 mm at 1.37 m/s → 44 mm at 1.0 m/s;
  - right powerhead 60 mm at 0.68 m/s, unchanged.

**A-207 Inflow turbulence intensity I = 0.05 at every pump outlet** (D-204), editable per pump.
- It fixes the jet's transition 3–4 equivalent diameters from the outlet, independent of the grid.
- The modelled pump jet then decays about twice as fast as a plain round jet of the same momentum. That is plausible for propeller pumps but not calibrated.
- Effect: jet throw beyond ~5 D is uncertain by a factor of ~1.5–2 through this parameter.
  - Larger I gives a shorter throw and more mixing near the pump.
  - Smaller I gives a longer, more laminar jet.
- The momentum delivered to the tank is unaffected (verified, case 7).

**A-208 Overflow = mass sink over the box footprint, from the weir crest to the water surface.**
- Removes exactly the return flow.
- Surface skimming and the weir's free overfall are not represented.
- At least one cell of water is kept above the box, so the modelled crest can sit up to one cell below the true crest. Demo: 1.8 mm.

**A-209 Rock is solid and impermeable** (spec default). Flow through porous live rock is neglected; Darcy–Brinkman is deferred (D-212). Real rock passes some flow, so results slightly overpredict sheltering behind rock.

**A-210 Dead-zone threshold 2.5 cm/s, applied to the time-averaged speed ⟨|u|⟩**, which is what a coral or a settling particle experiences.
- The spec gives ~2–3 cm/s. It is a starting assumption, not a husbandry standard, and user-configurable.
- The fraction below threshold based on the mean-velocity magnitude |U| is reported alongside. It is always larger, because fluctuating flow can have a small mean velocity.

**A-211 Run length: 60 s spin-up + 120 s averaging.**
- Demo turnover through the pumps is ~2 s per powerhead volume; the large-scale circulation time is ~10–15 s.
- Stationarity is checked by kinetic energy in the two halves of the averaging window. Values for each grid-study run are in verification/VERIFICATION.md, section 6.
- Statistical uncertainty is the standard error over 4 batches of 30 s. Batches are assumed independent; the large-scale circulation decorrelates in ~10 s.

## Phase 3 — metrics

**A-301 Water age is a steady mean-field quantity, normalised by V/Q** (tank water volume / return flow).
- 1 means "as refreshed as a perfectly mixed tank", > 1 stagnant, < 1 short-circuit.
- Turbulent transport is modelled by gradient diffusion, which is approximate for large-scale recirculation.

**A-302 Turbulent Schmidt number 0.7; eddy viscosity from the resolved Reynolds stresses (Boussinesq fit) plus SGS.** The demo's age statistics move by < 1 % in the mean and < 5 % in the maximum over Sc_t = 0.5–1.0.

**A-303 First-order upwind advection in the age solve.** Numerical diffusion is about U dx/2, comparable to the turbulent diffusivity at Preview (cell Péclet ~1–2). It acts in the same direction as turbulent mixing; at Standard and Fine it is halved or less.

**A-304 Wall shear from a smooth-wall Spalding law at 15 mm above the surface, using the RMS tangential speed.**
- Neglected:
  - sand roughness (k_s⁺ ~ 3–10, transitionally rough), which would raise shear by up to ~30 %;
  - non-equilibrium effects of jets impinging on the sand.
- Read shear maps as relative patterns with factor-of-1.5 absolute uncertainty.

**A-305 Detritus settles where sand or rock-top shear is below 0.02 Pa; sand moves above 0.5 Pa.** Starting assumptions, user-configurable (`criteria: detritus_shear_Pa`, `sand_move_shear_Pa`).
- 0.02 Pa: the order of reported critical erosion stresses for loose organic flocs (~0.01–0.05 Pa).
- 0.5 Pa: Shields θ_c ≈ 0.04 for ~0.7 mm aragonite sand.

**A-306 Coral profiles (`reefflow/data/coral_profiles.yaml`) are starting points, not husbandry standards** (spec).
- SPS: 10–30 cm/s, PAR 250–450 (limit 600).
- LPS: 5–15 cm/s, PAR 75–200 (limit 350).
- Soft corals and zoanthids: 3–10 cm/s, PAR 50–150 (limit 300).
- Turbulence preference and jet-penalty thresholds as listed in the file. Users can edit them or add profiles (`criteria: coral_profiles`).

**A-307 Coral flow is sampled 15 mm off the rock surface** (D-305), on tops and upper sides (outward normal n_z > −0.25). Corals on overhang ceilings or vertical-to-undercut faces are not scored.

## Phase 4 — lighting / PAR

Each entry: assumption, justification, expected effect. Values marked "starting value" are
editable defaults, not authoritative data.

**A-401 Geometric optics, incoherent light; unpolarised Fresnel at every interface.**
- Wavelengths (0.4–0.7 µm) ≪ every geometric feature; no interference in 12 mm glass.
- Polarisation after the first reflection is not tracked (each interface uses (Rs+Rp)/2). Effect on
  multiply reflected light at the glass is a few % of that light, which itself is a minor part of
  the PAR at any point.

**A-402 Water surface: rigid plane at the tank's water level; flat by default (spec).**
- Optional roughness = isotropic Gaussian facet slopes with RMS slope σ (verified against
  small-slope theory, D-406). Typical visual agitation corresponds to σ ≈ 0.05–0.2 (no reference
  data for aquaria; starting values only).
- The CFD hook (σ from near-surface speed) is a heuristic with a user-supplied coefficient; there is
  no validated relation between a rigid-lid LES velocity and surface agitation. Off by default.
- Effect of a flat surface: sharper fixture footprint and hot spot under each fixture than a real
  agitated surface; time-averaged PAR patterns are slightly smoothed by real ripples (caustics
  average out over seconds).

**A-403 n_seawater = 1.34 at all PAR wavelengths; n_glass = 1.5.**
Real dispersion 1.343 (400 nm) – 1.331 (700 nm) changes normal-incidence Fresnel transmission by
< 0.1 % and the critical angle by 0.4°. Low-iron and float glass are both ≈ 1.5.

**A-404 Fixture library values are generic starting values (product class, 2024–26).**
- LED puck 90 W × 2.0 µmol/J = 180 µmol/s, 90° optic (cos² beam); LED bar 40 W × 2.2, Lambertian;
  T5 4 × 54 W × 1.0 µmol/J (fixture level), Lambertian with 85° cut-off; MH 250 W × 1.0, 110° beam.
- Every emitter is a uniformly emitting rectangle with one angular distribution; the individual LEDs
  and colour separation of real modules are not resolved. Within ~2 footprint sizes of a fixture the
  pattern is approximate; all channels share the same distribution (colour shadowing not modelled).
- Hence all uncalibrated results are labelled "relative/estimated". Calibration (one global scale
  and optionally the beam angle) is the intended route to absolute values.

**A-405 In-water attenuation: Kd = 0.08 1/m broadband (clear reef water; starting value), applied as
a beam attenuation coefficient along every path (PLAN C8).**
- Kd is strictly an apparent optical property of diffuse ocean light. In a 0.45 m tank the
  attenuation is 2–4 % (demo: 3.5 % of emitted photons absorbed by water, including bounced light),
  so the approximation matters little; it is verified as implemented (case 4).

**A-406 No in-water scattering.** Clear, filtered tank water scatters little over < 1 m; turbid water
would mainly broaden shadows. Not modelled.

**A-407 Spectral mode: Kd(λ) = a_w(λ) + 0.03 1/m**, a_w = pure-water absorption (Pope & Fry 1997,
approximate 20-nm table), 0.03 1/m spectrally flat for dissolved/particulate matter (starting value).
Red (660 nm) is attenuated ~10× more than royal blue (450 nm): 0.45 vs 0.04 1/m.

**A-408 Glass and walls.**
- Panes are smooth, non-absorbing slabs (absorption of 12 mm low-iron glass ≈ 1–2 %, float glass
  ≈ 5–10 % — neglected); pane edges, silicone and bracing are ignored.
- "clear": air outside, light leaving is lost. "black": absorbing paint in optical contact with the
  outer face (no TIR there). "coralline": Lambertian reflector at the inner face, albedo 0.3
  (starting value), no glass interaction.
- Default all clear; the demo uses a black back wall. The clear front pane produces a bright TIR
  "mirror" band along it in the demo maps — a real effect (spec: "significant in real tanks").

**A-409 PAR meters: cosine-corrected point sensors; immersion factor 1.08 by default.**
- Modelled as a 25 mm horizontal square counting downward photons.
- 1.08 is typical of modern full-spectrum quantum sensors; older designs need up to ~1.3 — the user
  must enter their sensor's value. Ignoring the immersion metadata biases the calibration by
  1/factor (case 6: −6.8 %) (PLAN C7).
- Manufacturer charts: `air` = fixture alone in air; `open_water` = laterally unbounded water with a
  black floor. Charts measured in a specific tank (walls, sand) are better entered as `tank` readings
  of a matching project.

**A-410 Sand and rock are Lambertian with albedo 0.6 (aragonite sand) and 0.15 (coralline-covered
live rock); starting values.** Light hitting sand/rock is either absorbed or re-emitted diffusely.
Rock above the water surface is ignored by rays in air. Effect: sand albedo raises sand-level PAR
noticeably (sand-reflected light partly returns by surface TIR: ~44 % of a Lambertian upwelling flux
is beyond the 48.3° critical angle).

**A-411 No light from outside the fixtures; light reflected out of the tank is lost** (no canopy or
hood reflector, no room light, no second pass through the glass from outside).

**A-412 "PAR on rock" for coral scoring = meter-equivalent planar PAR at the surface location
(PLAN C6).** A horizontal upward-facing cosine sensor placed at the patch: the basis of the
250–450 / 75–200 / 50–150 µmol m⁻² s⁻¹ coral bands. Surface-normal irradiance (what the tissue on
a sloped face actually receives) is reported as a separate layer. On demo rock tops the two differ
by ~30 % on average (379 vs 261 mean).

**A-413 Photoperiod default 10:00–20:00 with 60-min linear ramps at 100 %** (9 h full-intensity
equivalent, DLI factor 0.0324 mol m⁻² d⁻¹ per µmol m⁻² s⁻¹). One schedule scales all fixtures.

**A-414 Patch size = CFD cell (5 mm at Standard).** Noise reported per patch; 2 % on an average-lit
patch needs ~9e7 photons; shaded patches at 20 % of the mean have ~2.2× the noise. Coarser
reporting (e.g. 3×3×3 neighbourhood for surface-normal PAR at CFD nodes) is ~3× less noisy.

## Phase 6 — photo reconstruction

Each entry states the assumption, its justification, and its expected effect on results. Sensitivities were computed for the demo tank and the validation cameras (1600 × 1200 px, f = 1200 px): the camera was held fixed and one parameter changed, and the table gives the image displacement of rock-surface points.

### Camera and optics

**A-601 Camera = pinhole with square pixels, principal point at the image centre, no lens distortion by default.**
- Unknowns per photo: focal length f (px) plus pose (6). Optional: one radial term k1 (`fit_k1=True`), recommended only with ≥ 9 points.
- Justification: phone main cameras apply distortion correction in-camera, and the principal point is within ~1 % of the image centre. With 8 clicked points (16 equations), more intrinsic unknowns would leave too few degrees of freedom to check the fit.
- Violated by: ultra-wide lenses, cropped or rotated-then-cropped images, digital zoom crops whose centre is not the frame centre.
- Symptom: a systematic residual pattern and RMS > 3–5 px. The calibration then reports a warning.

**A-602 Flat refractive interfaces.**
- Side photos look through a flat, parallel-faced glass pane. The top photo looks through a flat, still water surface (pumps off, as the spec requires).
- The camera is in air outside the interface, and ray paths are exact for flat parallel planes (D-602).
- The meniscus at the glass and pane bowing (a few mm on large tanks) are ignored. Surface ripple is not modelled; a rippled top photo blurs and misplaces the top silhouette.
- Points the layered model cannot describe (e.g. a submerged point seen from a high side camera through the water surface) are flagged invalid and ignored. They are never silently mis-projected.

**A-603 Refractive indices: air 1.000, seawater 1.34, glass 1.50 (spec values).**
- Seawater at 35 g/kg and 25 °C is 1.339 at 589 nm. Dispersion over 400–700 nm is ±0.004.
- Soda-lime float glass is 1.52; low-iron glass is 1.49–1.52. Air (1.0003) is neglected.
- Sensitivities (fixed camera, rock-surface points, mean / max displacement):

| Change | Top view | Front view | Left view |
|---|---|---|---|
| n_water + 0.005 | 0.17 / 0.36 px | 0.09 / 0.19 px | 0.19 / 0.40 px |
| n_glass + 0.03 | 0 | 0.02 / 0.04 px | 0.02 / 0.06 px |
| glass + 1 mm | 0 | 0.05 / 0.09 px | 0.05 / 0.13 px |
| water level − 5 mm | 0.23 / 0.44 px | 0 | 0 |

- Every entry is far below click precision. Calibration re-fits the camera anyway, which absorbs part of any error.
- For scale: ignoring refraction entirely moves rock points by tens of pixels (validation `refraction_displacement_rock_px`).

**A-604 The tank is a rectangular box, and the reference points are its inside corners.**
- Levels: rim (internal height), water line (project water level), sand (project sand depth), bottom (inside face of the bottom glass).
- Sand is rarely level. If the sand line at a corner differs from the project's sand depth, click a different point there (rim, water line or bottom corner), or enter it as a custom point with measured coordinates.
- A 5 mm error in a reference point's height is roughly a 5 mm error in that point's 3D position. The fit spreads it, and it shows up as residual.
- *Updated 2026-11-03 (D-836):* the default points are now the **outside** corners of the box (top of the frame or glass at z = H; where the tank sits on the stand at z = −g; x, y on the outer glass faces) plus the far water-line corners and the far vertical corner seams. This assumes the external size the user enters is the outside of the frame, and that the bottom glass is as thick as the side glass. On a rimmed tank the frame usually adds 5–15 mm under the bottom glass, so the inside bottom (z = 0) sits that much too low. Rock height above the sand is unaffected, because the sand depth is entered separately.

### Segmentation

**A-605 Rock is distinguishable by colour from sand, glass walls and the back wall, at least once background colours are learned from regions that cannot contain rock.**
- The automatic threshold (ΔE 14 in CIE-Lab) was chosen on synthetic images (render seeds 1–3); the validation uses a held-out seed.
- Real photos will often need user strokes or brush edits: coralline-covered glass, rock-coloured backgrounds, shadows on the sand, corals, equipment.
- Equipment in front of rock should be marked with the per-view **ignore** mask, not as rock or background.
- *Updated 2026-11-03 (D-839):* this assumption fails under blue reef lighting, where rock and water have nearly the same colour (user photo: the colour suggestion took the empty water at both ends of the tank for rock). Side views now use **texture** instead: porous rock is rough, and open water, glass and the back wall are smooth. Only rough blobs that reach the sand are kept. The top view still uses colour. Assumptions behind the texture cue: rock is rougher than the background at the photo's resolution; the rock rests on the sand; equipment does not reach the sand (otherwise it stays in the mask, to be brushed out).

### Reconstruction

**A-606 Dry-rock bulk density 1.4 kg/L by default (user-editable; range shown 1.1–1.8 kg/L).**
- Basis: aragonite mineral density 2.93 kg/L, and dry porous reef rock porosity of roughly 40–60 % (bulk 1.2–1.8 kg/L). Dense "base rock" sits at the top of that range, light porous rock at the bottom.
- This is an **assumption**, not a measured value. Wet weight of live rock is higher, because water-filled pores add 0.4–0.6 kg/L.
- `solid_fraction` (default 1.0) scales for stacked rubble with gaps the silhouettes cannot see.
- The reported weight is a sanity check against what the user bought, not a structural load.

**A-607 The visual hull overfills: the reconstructed volume is an upper bound.**
- Caves, arches, overhangs, and gaps between rocks that no silhouette sees are filled. With axis-aligned views (top, front, side), the diagonal "corner wedges" of each rock are also unconstrained.
- In the validation, with perfect masks and cameras, the overfill was +47 % of the true rock volume (see the decisions draft, validation section).
- CFD consequence: less open water and blocked passages behind and under the rock, so dead zones may be over-predicted there. The edit tools (tunnels, arches, open regions) are the mitigation. An extra oblique photo through a front corner reduces the wedges.

**A-608 Only rock between the sand surface and the water surface is reconstructed.**
- The buried base is irrelevant, because the sand bed is solid in the CFD model. The exported mesh extends one cell into the sand so the sand-surface layer under the rock is also solid.
- Rock above the water surface is not reconstructed.

**A-609 Silhouette test at voxel centres, with masks dilated by 1 px.**
- This is the same centre-inside rule as `geometry.voxelize_mesh` (A-106), so reconstruction and ground truth are compared like-for-like.
- The 1 px dilation absorbs calibration error and can only add volume. At 1600 px over a 1.2 m tank, 1 px ≈ 1 mm, about 0.2 of a 5 mm cell.
- Voxels that no view constrains (outside every field of view, or only behind ignore masks) are left empty and counted in `carving.unconstrained`.

## Results-based assumptions (grid study and channel verification)

**A-310 Near-surface flow is under-predicted at tank resolutions, so near-surface metrics are biased.**
- Evidence:
  - Verification 6: near-surface speeds rise with every refinement; at Standard → Fine the speed 15 mm above the sand rises 23 % and dead volume falls 32 %.
  - Verification 8: a channel at tank-like first-node y⁺ ≈ 8 carries 28 % too little flow for its wall shear.
- The bias, in plain terms:
  - **Dead-zone volume is an upper bound** (conservative).
  - Sand shear, rock-top speed and the flow part of coral scores read low.
  - Bulk speeds and the large-scale circulation are much less affected (+5 % Standard → Fine).
- Design *comparisons* on the same grid are more reliable than absolute values. Report near-surface numbers from Standard or Fine.
- Revisit when a verified wall model is available (D-702).

## Return manifolds (multi-nozzle returns)

**A-801 The nozzle exit velocity head is assumed to be inside the user's fittings allowance.**
- Splitting a return over more nozzles therefore does not change the total pump flow (D-805).
- In reality, more nozzles lower the exit velocity, so the exit losses fall and the flow rises a little. The UI and the pump summary show the exit head U²/2g per nozzle, so the user can lower the allowance:
  - 19 mm (3/4 in) nozzle at 700 GPH: U = 2.60 m/s, exit head 1.13 ft;
  - two such nozzles: 1.30 m/s and 0.28 ft each.
- On the demo return (static 4 ft + allowance 1 ft, shut-off head 10 ft), a 0.85 ft lower head would raise Q by ≈ 8 % (√(1 − 4.15/10) / √(1 − 5/10) = 1.082).

**A-802 Nozzles on one manifold see the same pressure and have equal discharge coefficients** (the default split Q_i ∝ A_i). Unequal pipe runs to the nozzles are not modelled; enter shares if they matter.

## Equipment catalogue (D-844 to D-846)

**A-844 Catalogue numbers that are not published.**
- **Powerhead outlets.** Only Tunze publishes the outlet size. Aqua Medic's comes from a retailer, and Sicce's was measured from Sicce's dimensioned drawings (±3 mm).
  - **Jet speed of those 12 pumps** at rated flow: median 1.34 m/s, range 0.35–2.62 m/s.
  - **The 16 propeller pumps without a usable outlet size** get an outlet that makes their jet leave at 1.34 m/s. It is capped at the wet-side diameter where that is published. Example: EcoTech MP40mQD at 5,000 GPH → 71 mm, against a 76 mm wet side.
  - **Why it matters:** the jet's momentum flux is ρQ²/A, so a 10 % larger diameter means about 17 % less momentum at the same flow.
  - **Cross-flow pumps** (Maxspect Gyre, Red Sea ReefWave) use body length × smallest body dimension as the slot. That is an upper bound, so their momentum leans low.
  - Rated flows are the makers' free-flow maxima; delivered = rated × 0.7 as before (A-205).
- **Light output.** Fixtures without their own PAR data use power × 1.41 µmol/J (D-844). The fitted fixtures span 1.14–1.76 µmol/J, so expect roughly ±25 % on uncalibrated PAR from these.
  - **Beams that are not published:** 110° (the middle of the fitted 95–125°).
  - **Footprints** are housing sizes.
  - **Channel shares** follow the LED counts per colour, all LEDs assumed equal. For Kessil, Neptune SKY and Philips the mix is a typical one, not a published one.
  - A PAR-meter calibration replaces all of this.
- **Return pumps** without a published curve use a straight line (D-845). The nozzle in the tank stays the generic 19 mm (3/4 in) unless you change it: the pump's own outlet does not set the jet.
- **Overflow ratings** are the makers' claims (often optimistic for one 1 in drain, as the drain table shows). Red Sea Reefer "capacities" are its recommended return-pump flows.
