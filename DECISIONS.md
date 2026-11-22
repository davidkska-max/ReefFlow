# DECISIONS.md — design choices log

Each entry: what was decided, why, and what would make us revisit it. Newest at the bottom of each phase.

## Phase 0 — plan (approved 2026-10-30)

**D-001 Python 3.12, Taichi 1.7.4 (pinned).**
Taichi ships no wheels for Python ≥ 3.13. The machine's default Python is 3.14, so the project venv uses uv's CPython 3.12.
*Revisit if:* Taichi publishes 3.13+ wheels, or a port to Warp happens (D-002).

**D-002 GPU framework: Taichi.**
- JAX has no native-Windows GPU support.
- NVIDIA Warp is CUDA-only.
- Taichi runs CUDA, Vulkan (integrated GPUs), Metal and CPU from one kernel source. On the dev machine Vulkan matched CUDA (529 vs 506 MLUPS, simple benchmark kernel).
- Risk: Taichi development has slowed. All kernel code is confined to `kernels.py`, `stats.py` and `backend.py`, so a port touches only those files.

**D-003 Lattices: implement both D3Q19 and D3Q27; choose the production lattice from verification test 4 (turbulent jet).**
D3Q19 lacks rotational invariance in higher moments, with documented artifacts in under-resolved turbulent jets and pipes. *(User-approved.)*

**D-004 Collision: BGK for verification, recursive-regularized (RR) for production.** See D-103 for the implementation.

**D-005 Grid-convergence study (verification test 6) moved to the start of Phase 2.** It needs the tank model. *(User-approved.)*

**D-006 Wall function on by default for reported wall shear (sand-bed detritus metric).** Momentum BC stays bounce-back. Implemented in Phase 3. *(User-approved.)*

**D-007 Demo tank default glass thickness ½ in.** *(User-approved.)*

## Phase 1 — solver core

**D-101 Kernels take runtime-shaped Taichi ndarrays, not fields.**
With fields, every new grid size forced a full recompile (10–60 s per configuration and size). With ndarrays a kernel set compiles once per configuration — lattice, collision, LES on/off, feature switches, periodicity — and serves every grid size. Compiled kernels are cached on disk by Taichi.
No measurable throughput penalty: 2,429 MLUPS (D3Q19, RR + Smagorinsky) and 1,729 MLUPS (D3Q27) on an RTX 3070 at 2 M cells.

**D-102 Populations stored shifted, g = f − w (fp32).**
Skordos 1993; standard in waLBerla and FluidX3D. The quantity stored is the deviation from rest equilibrium, so fp32 round-off scales with the flow perturbation rather than with ρ ≈ 1. Total-mass reductions are finished in fp64 on the host.

**D-103 Recursive-regularized collision, lattice-generic implementation.**
- Equilibrium and non-equilibrium reconstruction use the third-order Hermite basis each lattice can represent, with lattice norms computed numerically:
  - D3Q27: the 7 individual H3 polynomials.
  - D3Q19: (H_xxy ± H_yzz) and cyclic permutations. Norms 2/27 and 6/27 give the 3:1 weighting of Jacob et al. (2018). H_xyz is not representable on D3Q19.
- The third-order non-equilibrium moments are rebuilt recursively, a3 = sym(u ⊗ a2) (Malaspinas 2015).
- Orthogonality and moment reproduction are unit-tested (`tests/test_lattice.py`).

**D-104 Forcing: Guo, Zheng & Shi (2002).**
- Velocity u = (Σ f c + F/2)/ρ.
- The non-equilibrium stress includes the (uF + Fu)/2 correction.
- Forcing is implemented identically for BGK and RR: f* = f_eq + (1 − ω) f_neq′ + F_q/2.
- Tested: momentum grows by exactly ΣF per step (`test_body_force_momentum_rate`).

**D-105 Smagorinsky LES evaluated locally from the non-equilibrium stress** (Hou et al. 1996). No finite differences, so it costs nothing extra in bandwidth.
- Default C_s = 0.17 (Lilly's isotropic-turbulence value), user-editable.
- No near-wall (van Driest) damping in Phase 1; see A-105.
- The production default is reviewed against the jet test.

**D-106 Wall boundaries: halfway bounce-back.**
- Moving walls use the Ladd correction with ρ_w = 1, an O(Ma²) approximation.
- Free-slip is halfway specular reflection on axis-aligned planes only. That is sufficient for the rigid water-surface lid; at edges and corners it falls back to bounce-back.

**D-107 Open boundaries (velocity inlet, pressure outlet): regularized non-equilibrium extrapolation** (Guo et al. 2002, regularized as in Latt et al. 2008).
- Used in the verification cases only; the tank model has no open boundaries except the overflow sink (Phase 2).
- Assumes no body force on the boundary's neighbour node.

**D-108 Output velocity from post-collision populations: u = (Σ f c − F/2)/ρ.**
*Bug found and fixed during verification.* The first `macros()` used +F/2, which is correct only for pre-collision populations. That biased every reported velocity in forced flows by exactly g. It was found because BGK Poiseuille at the "magic" τ = ½ + √(3/16) showed an error of exactly g instead of zero. After the fix, the error at the magic τ is 8 × 10⁻⁷ (fp32 floor). The kernel's own state was never affected. A regression test is included (`test_body_force_momentum_rate` checks both quantities).

**D-109 Steady verification cases stop on a residual, not a fixed step count.**
A fixed multiple of the diffusion time (3 N²/ν) left one Poiseuille case visibly unconverged (6.5 × 10⁻⁴ vs converged 7.0 × 10⁻⁴).

**D-110 Acceptance criteria for every verification case are written in the case's module docstring before it is run.**

**D-111 Known overhead: ~150–250 µs Python-side cost per Taichi kernel launch on this machine (any backend, including CPU).**
- Irrelevant for tank-size grids (a 2 M-cell D3Q19 step takes ~0.8 ms).
- It dominates small 2D verification cases, which need 10⁵–10⁶ steps.
- Taichi compute graphs (pre-recorded launch sequences) measured 101 µs vs 238 µs per launch. That gain matters only for small grids, so adoption is deferred to Phase 7.

**D-112 Initial state at open boundaries.**
- `initialize()` sets only the prescribed quantity on boundary nodes: u on INLET, ρ on OUTLET. The other quantity comes from the initial field.
- *Bug found by test:* the first version also imposed u_bc = 0 on pressure outlets. That launched a pressure pulse (Δρ ≈ 0.02) which a weakly damped acoustic mode carried for thousands of steps.
- After the fix, a uniform stream through inlet + outlet + free-slip walls is preserved to fp32 precision (`test_open_boundaries_plug_flow`).

**D-113 Float precision is a build option (fp32 production; fp64 for verification).**
- Set via `backend.init(precision=...)` or `REEFFLOW_PRECISION`, and part of the kernel cache key.
- fp64 currently runs on the CPU backend only: fp64 vector ndarrays raise a CUDA misaligned-address fault in Taichi 1.7.4.
- Purpose: separate discretization error from round-off in verification (see A-109).

**D-115 Sponge layers (optional per-cell added viscosity) next to open boundaries.**
- *Found in verification:* the first turbulent-jet run diverged at step 80,000.
- An instrumented rerun showed inflow through the constant-pressure lateral boundaries growing steadily to ~12 % of U_j. Entrainment requires only ~0.4 %. There was also outlet backflow of 25–35 % of U_j from eddies leaving the domain.
- Fix: `LBMSolver(sponge_nu=...)` adds viscosity cell by cell. In the jet it ramps quadratically to ν = 0.05 over 26 < x/D < 30 and over the outer 1.5 D of the side faces. Both zones are outside the fitting window.
- This is standard practice in jet LES.
- Test: a uniform sponge must equal extra viscosity (`test_sponge_is_extra_viscosity`).
- Tank runs have no open boundaries (the overflow is a sink), so tank runs do not use sponges.

**D-116 Poiseuille acceptance split by precision.**
- The scheme's order is verified strictly in fp64 (order 2.00 ± 0.02; magic-τ error < 10⁻⁸). Achieved: 2.000 everywhere and 2.7 × 10⁻⁹.
- fp32 (production) must agree with fp64 within 25 % for H ≤ 32. Achieved: ≤ 20 %, with the worst case D3Q27-RR free-slip at H = 32.
- Reason: the fp32 round-off floor (A-109) is larger on D3Q27 (27 populations rounded per node) and already ~20 % of the discretization error at H = 32. That pulled one fp32 order fit to 1.87.

**D-117 Jet-analysis bug fixed; post-processing now verified independently.**
- Lattice radii are discrete. The 0.5-cell radial bins between r = 0.5 and 1 contain no node, and were read as zero velocity, so the half-width was found in the first empty bin (S = 0 in the first report).
- Empty bins are now dropped. A synthetic self-similar jet (B = 5.8, S = 0.094, x₀ = 4 D) is recovered exactly (`test_jet_analysis_recovers_synthetic_jet`).
- J₀ is now the momentum flux of the actual (smoothed) inflow profile, which is 0.83 of the ideal top-hat.

**D-118 Smagorinsky C_s = 0.17 over-damps resolved shear layers. Production default C_s = 0.10.** *(User-approved 2026-10-30.)*
- At D = 10 cells, C_s = 0.17 gives the nozzle shear layer a local eddy-viscosity Reynolds number of ~50. Instabilities do not grow and the laminar potential core persists to ~12 D (D3Q19) or ~25 D (D3Q27); experiments give ~4–6 D.
- With C_s = 0.10 (D3Q19): B = 5.21, S = 0.104, momentum flux conserved within 11 %, profiles self-similar. Passes. The potential core is still long (8.4 D).
- C_s ≈ 0.1 is the customary value for shear and wall-bounded LES (Deardorff 1970 onward). 0.17 is Lilly's isotropic-turbulence value.
- Still to run (interrupted when the GPU was taken over by another application): D3Q27 at C_s = 0.10, and C_s = 0.12 / 0.17 on both lattices with the corrected analysis.
- **Tank implication:** jets resolved by only a few cells will still tend to stay coherent too long (overpredicted throw, underpredicted near-field mixing). This adds weight to the momentum-equivalent nozzle treatment (PLAN C3) and to the Phase-2 grid study.

**D-119 Wall-clock timings depend on other GPU load on the workstation.**
- Clean measurements in the jet configuration (7.7 M cells, RR + LES + sponge + open boundaries): D3Q19 1,901 MLUPS, D3Q27 1,358 MLUPS (ratio 1.40).
- One D3Q27 jet run took 5.6× longer than expected while another application held ~7 GB of the 8 GB of VRAM, causing GPU memory paging.
- Future run-time estimates should check free VRAM first.

**D-120 Jet study: the pass/fail outcome is governed by when the shear layer transitions. Production lattice: D3Q19.** *(User-approved 2026-10-31; D3Q27 stays selectable and is re-checked in the Phase-2 grid study.)*

Runs, all at D = 10 cells. Acceptance: B 4.5–6.7, S 0.077–0.112, fitted over 15–25 D unless noted.

| Lattice | C_s | Inflow disturbance | Core /D | B | S | Pass |
|---|---|---|---|---|---|---|
| D3Q19 | 0.10 | 5 % | 8.4 | 5.21 | 0.104 | yes |
| D3Q19 | 0.12 | 5 % | 8.7 | 5.45 | 0.095 | yes |
| D3Q19 | 0.17 | 5 % | 12.8 | 4.11 | 0.137 | no |
| D3Q27 | 0.10 | 5 % | 14.2 | 4.35 | 0.126 | no |
| D3Q27 | 0.12 | 5 % | 15.9 | 4.10 | 0.135 | no |
| D3Q27 | 0.17 | 5 % | 23.4 | still laminar | — | no |
| D3Q19 | 0.10 | 15 % | 6.4 | 5.40 | 0.093 | yes |
| D3Q27 | 0.10 | 15 % | 8.7 | 4.73 | 0.1125 | no (S marginal) |
| D3Q27, 45 D domain, fit 25–40 D | 0.10 | 5 % | 14.4 | 5.69 | 0.090 | yes |

Findings:
1. Every run whose laminar core ends by ~9 D reproduces the literature decay and spreading in the 15–25 D window. Every later-transitioning run fails because the window then catches the rapid post-transition development, not the self-similar far field.
2. The core lengthens with C_s (more eddy viscosity in the nozzle shear layer) and shortens with inflow disturbance.
3. At equal settings D3Q27 transitions 2–6 D later than D3Q19. D3Q19's earlier transition is plausibly helped by its known anisotropy seeding disturbances. Neither lattice reaches the experimental ~4–6 D core at D = 10 cells.
4. **D3Q27's far field is correct.** On a 45 D domain, fitted over 25–40 D, it gives B = 5.69 and S = 0.090, within 2–4 % of Hussein et al. (5.8, 0.094), with clean profile collapse. Its jet-test failures are purely late transition.
5. In tests 1, 2, 3 and 5 the two lattices agree to within the reported precision.
6. **Tank scale:** a powerhead nozzle is ~40–50 mm across and the tank 1.2 m long, so the whole tank is ~25–30 D, i.e. the jet near field. Transition timing therefore matters more for our application than far-field slope. D3Q19's near field is closer to experiment: core 6.4–8.4 D against 8.7–14.4 D for D3Q27, with ~4–6 D measured. D3Q27 also costs 1.40× run time and 1.42× memory.
7. Caveat on J/J₀: the momentum-flux estimate omits the pressure term. Inside a turbulent jet p′ ≈ −ρ⟨v′²⟩ < 0, so the 10–20 % post-transition overshoot is expected rather than a conservation error. The closed-box test (test 5) verifies momentum injection exactly.

**Recommendation:** D3Q19 + recursive-regularized + C_s = 0.10 for production, with D3Q27 selectable.
- This reverses the Phase-0 lean toward D3Q27, which was based on literature reports of D3Q19 artifacts.
- Our data show both lattices resolve the self-similar far field correctly. D3Q19 gets the near field (which is what a tank is) closer to experiment, at lower cost.
- Caveat: part of D3Q19's earlier transition may come from its own anisotropy rather than better physics. The Phase-2 grid study should be run on both lattices for the demo tank, so the choice is confirmed on the actual application.

**Phase 2** *(user-approved 2026-10-31, implemented as D-204)*: powerhead and return momentum sources should inject inflow turbulence (configurable intensity). Propeller outflow is swirling and highly turbulent, and the jet study shows near-field mixing depends strongly on it. A quiet source would overpredict jet throw.

**D-114 Smagorinsky implementation checked quantitatively.**
- Test: decay of a shear wave against the single-mode analytical eddy viscosity ν_t,eff = (8/3π) C_s² k A.
- Measured/expected ratio: 1.08 for RR (D3Q19 and D3Q27) and 1.05 for BGK. The single-mode estimate ignores transfer to the 3k harmonic.
- Test tolerance is 10 %.

## Phase 2 — tank model and pumps (CLI)

**D-201 Taichi's start-up update check is disabled (`TI_SKIP_VERSION_CHECK=ON`, set in `reefflow/__init__.py`).**
- On `ti.init()` Taichi starts a background HTTPS request to check for new versions. ReefFlow must run fully offline, so it should make no network calls.
- *Found in Phase 2:* on this workstation's TLS-inspecting network that thread began aborting the whole process about 0.5 s after start-up (`OPENSSL_Uplink ... no OPENSSL_Applink`). Every GPU test died silently.
- With the check disabled, all tests pass again.

**D-202 Powerhead = body force over a short source region; total force F = J = ρQ²/A.**
- Q is the delivered flow (rated × derating) and A the true outlet area. The force acts along the aim vector, spread uniformly over the SOURCE nodes of a disc (or rectangle) of the equivalent outlet size (D-203).
- The region is 20 mm long, at least 2 cells, and starts at the outlet plane.
- The source region acts as propeller disc and intake together: water is drawn from behind and around it.
- The force is injected exactly: `tests/test_sources.py` shows momentum rising by exactly ΣF per step.
- **The resulting jet carries momentum flux J** (verification case 7): J(x)/J = 0.96–1.00 for 3–8 equivalent diameters downstream, and 1.00 at 1–2 D.
- *Changed during Phase 2: solid pump housings are now off by default.*
  - The first version put a solid housing (archetype body size) directly behind the force region.
  - Verification 7 then showed the jet carrying only **49–55 % of J**. With no intake path, the pump drew its water sideways across the housing face, and the suction on that face cancelled about half the thrust.
  - Removing the housing restored J/J = 1.00. A real powerhead draws through an intake behind or around its propeller, so the no-housing model is the closer physical match.
  - A housing can still be switched on, for blockage studies; the run then prints a warning.
- A force-only source behaves like an actuator disc, so the jet contracts and its peak velocity exceeds the equivalent outlet velocity U_eq.
  - Measured: 1.43–1.53 × U_eq on the axis at 1–4 D (ideal disc: √2 = 1.41).
  - The time step uses √2 × U_eq (D-209), so peak lattice speed in the demo is 0.11, slightly above the 0.10 target (Ma 0.19). Runs warn at 0.15.

**D-203 Return pump = mass source; overflow = matching mass sink. Equivalent outlets keep J.**
- Return: SOURCE nodes add mass at rate ρQ in equilibrium at the true nozzle velocity U = Q/A. That gives mass flux ρQ and momentum flux ρQU = ρQ²/A = J for any source-region size.
- Overflow: the fluid nodes above the overflow box, from the weir crest up to the lid, remove the same total mass. Each removes a fraction of all its populations, so it takes out mass together with its own momentum and exerts no spurious force.
- Source and sink share one time-scale factor, so mass balances exactly. Measured drift: 5 × 10⁻⁷ over a 180 s Preview run, which is fp32 round-off.
- Why not a velocity-inlet boundary on a nozzle face? A mass source works at any aim angle, needs no pipe geometry (sub-cell at Preview), and balances the overflow exactly.
- Equivalent outlet (PLAN C3, extended): the modelled outlet is enlarged, at the same J, when
  - it is narrower than 4 cells, or
  - its equivalent velocity √(J/(ρA)) exceeds a cap of 1.0 m/s (project setting `source_velocity_cap`).
- Why the cap: the time step scales with 1/(peak velocity). The demo's ¾" return nozzle runs at 2.1 m/s, and modelling it at true size would halve the time step and double every run time.
- The jet far field depends on J only. The near field within ~5 equivalent diameters is approximate. Every enlargement is listed in the run's notes (A-206).

**D-204 Inflow turbulence at pump outlets** *(user-approved 2026-10-31)*.
- Each pump's force (powerheads) or injected velocity (returns) carries random fluctuations with relative RMS intensity I.
- The axial part is the sum of 7 outlet modes (uniform; r cos mθ and r sin mθ for m = 1–3) with independent unit-variance Ornstein–Uhlenbeck amplitudes.
- A random lateral component perpendicular to the aim adds jet flapping.
- Correlation time: D_eq/U_eq, the transit time of one outlet diameter. The host updates amplitudes at least 8 times per correlation time.
- **Default I = 0.05 for all archetypes**, chosen from a sweep in the verification-7 box: narrow-jet powerhead, Standard grid, no housing.

  | I | U_c/U_eq at 1 / 2 / 4 / 8 / 12 D | axis turbulence intensity at 2 / 4 / 8 D |
  |---|---|---|
  | 0 | 1.43 / 1.51 / 1.53 / 1.47 / 1.29 | 0.00 / 0.00 / 0.00 (stays laminar) |
  | 0.03 | 1.43 / 1.50 / 1.28 / 0.52 / 0.35 | 0.03 / 0.20 / 0.38 |
  | 0.05 (verification run) | 1.43 / 1.47 / 1.07 / 0.43 / — | 0.05 / 0.30 / 0.42 |
  | 0.06 | 1.44 / 1.45 / 0.95 / 0.41 / 0.30 | 0.06 / 0.37 / 0.44 |
  | 0.10 | 1.44 / 1.33 / 0.72 / 0.36 / 0.27 | 0.16 / 0.49 / 0.34 |
  | 0.15 (first default) | 1.43 / 1.16 / 0.57 / 0.31 / 0.23 | 0.30 / 0.57 / 0.43 |

- Reading of the sweep:
  - Without inflow turbulence the jet stays laminar for more than 12 D. That is unrealistic for a propeller pump, and the transition point would depend on the grid.
  - From I ≈ 0.03 the jet transitions at a fixed place, 3–4 D from the outlet. A grid-independent transition point is what the grid study needs.
  - Above I ≈ 0.06 the near field breaks up almost at once (I = 0.15 already 0.57 U_eq at 4 D).
  - 0.05 sits in the middle of the band where the transition point is fixed but the near field is not destroyed.
- **Comparison with a plain jet, and the open uncertainty.**
  - A plain turbulent round jet with the same J would have U_c ≈ 0.88 U_eq at 8 D (B = 5.8, Hussein et al. 1994; momentum form).
  - The modelled pump jets have about half that (0.41–0.52 for I = 0.03–0.06), so they spread and slow down faster.
  - Propeller jets are reported to decay faster than plain jets because of swirl and blade-wake turbulence (review: Lam et al. 2011, *Ocean Eng.* 38:1–10), so the direction is plausible. It is not calibrated against powerhead measurements.
  - Jet "throw" at 8 D is therefore uncertain by roughly a factor of 1.5–2 through this parameter. That is comparable to the derating uncertainty (A-205). Phase 3 will report sensitivity to it.
  - The far-field constant B could not be measured reliably in this box: fits over 8–18 D scatter between 2.6 and 5.0 with no trend in I, because the end wall is ~26 D away. It is reported but not used.

**D-205 Pump modes.**
- Flow fraction s(t): constant, sine, or pulse. Pulse is a square wave with half-cosine ramps (default 10 % of the period) to avoid pressure shocks.
- Alternating ("wave") operation is two pumps with the same mode at phases 0 and 0.5.
- Force scales as s² (J ∝ Q²); mass and injected velocity scale as s.
- Each pump reports its time-mean momentum flux J⟨s²⟩.

**D-206 SOURCE node type handled inside the main stream–collide kernel.**
- Each SOURCE node indexes a row of a small source table.
- Each pump has a pump-state row (scale, turbulence amplitudes) that the host updates during the run.
- Ordinary nodes pay nothing: no extra kernel launch and no extra memory traffic.
- Mass is added or removed after collision (operator splitting). The force enters the Guo forcing like a body force.

**D-207 Voxelization: a node is solid if its centre lies inside the mesh (ray parity along +z).**
- Own NumPy implementation (no rtree or Embree dependency). It is exact for an axis-aligned box and within 2 % of the volume of a 10-cell-radius sphere (tests).
- Fluid regions not face-connected to the main water body (voids sealed inside rock) are filled as rock and reported. The demo has one such node at Standard and Fine.

**D-208 Grid layout.**
- Whole cells of the preset size, centred in x and y.
- The grid's length, width and water column differ from the tank's by less than ½ cell; each difference is reported (e.g. −1.0 % water column at Preview).
- The sand surface is a solid floor (spec). The water surface is the free-slip lid (spec, PLAN C10).

**D-209 Time step from the estimated peak speed; runtime guard.**
- dt = 0.1 dx / (√2 max U_eq).
- Each run monitors the maximum lattice speed. It warns above 0.15 and stops above 0.35 or on any non-finite value.
- Demo: estimated peak 1.41 m/s; measured maximum about 1.55 m/s, lattice speed 0.11 (D-202). Exact values per run are in the grid-study table.

**D-210 Statistics and their uncertainty.**
- Velocity is sampled every 10 steps.
- The averaging window is split into 4 batches; the statistical uncertainty of each metric is the standard error of the batch values.
- Stationarity is checked by comparing kinetic energy in the two halves of the window.

**D-211 The grid study holds the physical model fixed.**
- Equivalent pump outlets use one minimum size on every grid: 4 Preview cells = 40 mm, instead of 4 cells of each grid.
- Without this, the source itself would change between grids and the study would mix model change with discretization error.

**D-212 Darcy–Brinkman porous rock: deferred.**
- The spec lists it as optional; rock is solid by default, as specified (A-209).
- It needs a velocity-dependent (implicit) drag term in the kernel. It will be added when there is a use case to verify it against.

**D-213 Result files: VTK `.vti` written directly (own writer, no PyVista/VTK dependency).** It opens in ParaView; tested by round trip.

## Phase 3 — metrics, report, comparison

*Process note (2026-10-31):* the user asked for all remaining phases to be completed in one sprint, without stopping for review between phases. Each phase is still documented here with its numbers.

**D-301 Analysis is a post-processing step on the saved, time-averaged fields** (`reefflow/analysis/`).
- `reefflow run` runs it automatically at the end; `reefflow analyze RUN` reruns it; `reefflow compare A B` compares two runs.
- Each run stores its mean velocity, mean speed, TKE, full Reynolds-stress tensor and settings. The analysis rebuilds the identical tank model and checks it voxel for voxel.
- Run time on the Preview demo: 21 s, of which 15 s is the water-age solve.

**D-302 Water age: steady mean-field solve (PLAN C5).**
- Finite volumes on the LBM node grid, upwind advection, central diffusion.
- Age-0 water enters at the return-nozzle source nodes; water leaves at its local age through the overflow sink nodes.
- Face fluxes are first projected onto discrete continuity with the pump sources. The LBM mean field's divergence error is 0.9 % of the total face flux; the projection changes fluxes by 0.9 % and leaves 4 × 10⁻¹⁵.
- Diffusivity: D = ν + (ν_t,resolved + ν_t,SGS)/Sc_t.
  - ν_t,resolved: least-squares Boussinesq fit of the resolved Reynolds stress to the mean strain. Demo median 3.1 × 10⁻⁴ m²/s.
  - ν_t,SGS: Smagorinsky on the mean strain. Median 1.7 × 10⁻⁶ m²/s, which is negligible.
- Solver: algebraic multigrid (pyamg) with GMRES, iterated on the true residual. Preview demo: 15 s, residual 6 × 10⁻¹⁰. An incomplete-LU GMRES fallback took 440 s.
- Exact checks (tests):
  - plug flow gives the travel time per cell to 10⁻⁸;
  - outlet flow-weighted age = V/Q to 10⁻⁶ for random non-solenoidal fields with obstacles; 2 × 10⁻¹² on the demo;
  - strong diffusion gives uniform age = V/Q.
- **Demo result:** mean age 0.97 V/Q, 95th percentile 1.02, maximum 1.15 (V/Q = 5.7 min).
  - The powerheads mix the tank far faster than the return pump exchanges it, so age is nearly uniform. Speed-based dead zones are the more informative indicator for this tank; the report says so.
  - Sensitivity over Sc_t = 0.5 / 0.7 / 1.0: mean 0.974 / 0.969 / 0.964; maximum 1.12 / 1.15 / 1.20.
  - With SGS diffusivity only, the maximum would be 1.81: the overstatement PLAN C5 warned about.
- **Not done:** PLAN C5's cross-check against a direct transient age run (age carried by the instantaneous LBM velocity). It needs a coupled scalar-transport kernel.
  - Deferred, because for this tank the result is insensitive to the mixing model (above) and the exact checks cover the solver.
  - *Revisit if* a tank shows strongly non-uniform age.

**D-303 Dead zones.**
- Connected regions (face connectivity) of water with ⟨|u|⟩ below the threshold, ranked by volume.
- Each zone reports volume, centroid, bounding box, the surfaces it touches, and its *core*: the point farthest from faster water.
- The core was added because the largest zone in the demo is an 11.5 L network threading along glass and rock, so its centroid lies in open water.
- Derating sensitivity (PLAN C4): the speed field is rescaled by 0.8 and 1.2 rather than rerun. Checked by an actual −20 % rerun (D-310).

**D-304 Wall shear on sand and rock tops by wall function (D-006).**
- Spalding's law of the wall, solved for u_τ from the RMS tangential speed √(U_t² + ⟨u_t′²⟩) sampled 15 mm above the surface (same height on every grid; log layer).
- Test: within 1.2 % of the pure log law at y⁺ = 80, and exact in the viscous sublayer.
- Demo Preview: sand shear median 7.8 mPa (10–90 %: 1.8–21 mPa). The surface fraction below the 0.02 Pa detritus threshold is 88 % for the sand and 88 % for rock tops.

**D-305 Coral placement score.**
- Surface points: fluid nodes face-adjacent to rock with outward normal n_z > −0.25 (tops and upper sides, excluding overhang ceilings). Demo Preview: 1,440 points.
- **Deviation from the spec's wording:** flow is sampled 15 mm out from the rock surface along the normal, not "one cell off". One cell is 10, 5 or 3 mm depending on the preset, and the near-wall speed changes strongly with it (verification 6: speed 15 mm above the sand is 3.7 cm/s at Preview, 7.9 at Standard). A fixed distance makes the score a property of the tank, not of the grid.
- Sub-scores, each 0–1: speed band (trapezoid), turbulence modifier (0.6–1.0), laminar-jet penalty, light band with upper limit. Combined = speed × turbulence × jet × light. Formulas are in `analysis/coral.py`; profiles in `data/coral_profiles.yaml` (A-306).
- Best spots: greedy non-maximum suppression, at least 50 mm apart, top 10 per profile.
- The rock surface (marching cubes) is exported per profile as a vertex-coloured PLY.
- Demo Preview, flow only: mean score SPS 0.19, LPS 0.54, soft 0.75. Surface scoring ≥ 0.7: 13 % / 46 % / 70 %.

**D-306 Comparison mode.**
- Both runs' metrics are recomputed with identical (current) definitions. Each change is flagged when it exceeds 2σ of the combined statistical noise.
- Difference maps: speed change in plan and elevation, plus dead water gained and removed.
- Runs on different grids are compared by trilinear interpolation onto run A's nodes.
- Demo, constant vs alternating powerheads (10 s pulse, 30–100 %): mean speed 10.85 → 9.37 cm/s, dead-zone volume 6.6 → 8.1 %. All 10 metrics change by more than the noise.

**D-307 Report: one self-contained HTML file per run** (figures embedded; prints to PDF from any browser; light and dark display).
- Contents: run status, inputs and pump table, model notes, metrics with ±2σ, dead zones with derating sensitivity, water age, sand shear and detritus, coral placement with best spots, flow slices, key assumptions.

**D-308 Metric definitions tightened during Phase 3, before the grid study was analysed.**
- "Speed one cell above sand / rock tops" is now speed at a fixed 15 mm above the surface (interpolated).
- Plan-plane speeds are now interpolated at exact fractions of the true water column.
- The grid-study analysis recomputes every metric from the saved fields with the current definitions. Statistical uncertainties of redefined metrics are scaled from the as-run values; this is flagged per run.

**D-309 New dependency: pyamg 5.4 (MIT)** for the multigrid solves.

## Phase 4 — lighting / PAR

Same format as DECISIONS.md: what, why, numbers, revisit-if. Verification numbers refer to
`verification/results/09_light/<case>/results.json` (module `reefflow/light/verification.py`;
acceptance criteria pre-declared in its docstring, D-110).

**D-401 Custom Monte Carlo forward photon tracer in Taichi (as planned, PLAN §3).**
- One thread traces one photon from emission to absorption/escape; tallies are photon flux on
  planes and surfaces, so irradiance comes out directly in µmol m⁻² s⁻¹.
- Equal-weight photons. Emitter chosen ∝ its photon output, uniform point on it, direction from a
  tabulated inverse CDF of μ = cos θ (4096 entries; exact closed forms for cos^m and isotropic).
- Per-photon PCG32 streams (own odd increment per photon, hashed seed per batch): deterministic and
  identical on CUDA and CPU; a shared-increment LCG would give overlapping (shifted) streams.
- Same backend as the CFD (`reefflow.backend.init()`); kernels in `reefflow/light/kernels.py` only.
- *Revisit if:* scenes need many more surface types (then a BVH/mesh tracer, e.g. Warp/OptiX).

**D-402 Scene = analytic planes + glass slabs traced explicitly + rock SDF.**
- Water box [0, L] × [0, W] × [z_sand, z_s] from the `Tank` dataclass; open top at the rim height.
- Light that misses the open top (or the water footprint) is lost; light reflected back up at the
  surface is lost (no canopy/hood).
- Side panes are slabs of thickness g, traced explicitly inside the glass (lateral displacement and
  multiple internal reflections, TIR at glass→air); each pane is treated as an infinite slab, and a
  ray leaving the glass outside the water column is discarded ("lost_other", 1–5 % in the demo).
- *Revisit if:* rimless tanks with euro-bracing or a canopy reflector need modelling.

**D-403 Meter-equivalent PAR = downward plane crossings at the node heights ("stack" tally) (PLAN C6).**
- For every node (i, j, k) of the tally grid, photons crossing the horizontal dx × dx square at the
  node's height *downward* are summed: that is the downwelling planar irradiance on an upward-facing
  cosine sensor, the basis of all PAR targets. Upwelling light is excluded (sensor back side).
- The grid is the CFD node grid when a `TankModel` is given (same layout rule otherwise), so the
  field maps 1:1 onto CFD nodes. Cells partly inside rock are divided by their water area fraction
  (4×4 SDF sub-samples); cells with < 25 % water are not reported.
- Plan maps at user depths are separate plane tallies with their own pixel size (default 5 mm).
  The front-elevation section is the stack slice at the nearest node row.
- Surface-normal PAR on rock: hits binned to the nearest node, divided by the true rock area in that
  cell (marching-cubes triangulation of the SDF, each triangle subdivided 16×, binned by centroid).
- Case 7: sphere patches E / (E0 cos θ) area-weighted mean 1.0011 (STL SDF).

**D-404 Rock SDF from STL: exact narrow band.**
- Grid h = 2.5 mm (default) over rock bbox ∩ water box. Sign from ray-parity voxelization (existing
  `geometry.voxelize_mesh`), magnitude = exact point–triangle distance (Ericson, vectorised) for
  |φ| < 2.5 h, candidate triangles by KD-tree after subdividing the mesh to edges ≤ h; outside the
  band a conservative (under-estimated) EDT distance so sphere-tracing steps never overshoot.
- Hits: sphere tracing on the trilinear field, then 24-step bisection; normals from central
  differences of the trilinear field (δ = h/2).
- Case 7 (R = 50 mm sphere): intercepted flux 1.0002 × E0πR², shadow area 1.0059 × πR²,
  normals 0.06° RMS (0.19° max), surface area 0.9989 × 4πR².
- No new dependency (scipy cKDTree / EDT, trimesh subdivision, scikit-image marching cubes).

**D-405 Rock SDF from the CFD voxels (fallback), and why STL is the default.**
- EDT of voxel centres (surface half-way between solid and fluid centres), Gaussian-smoothed σ = 1
  cell; rock is extended into the sand and glass layers so no spurious faces appear there.
- Case 7 at dx = 5 mm (R = 10 cells): intercepted 0.982, shadow 0.980, normals 2.8° RMS (6.1° max),
  patch-level scatter 9.9 % beyond MC noise (STL: 0.8 %).
- Demo tank, Standard grid, node-wise STL/voxel ratio of meter PAR at rock-adjacent CFD nodes:
  median 0.953, p10 0.44, p90 1.02. The low tail is side/underside faces (normal z ≈ −0.09): smoothing
  erodes overhang lips, so the voxel SDF lets light reach nodes the true geometry shades
  (47 vs 164 µmol m⁻² s⁻¹ median at those nodes). Rock tops agree within 8 % (341 vs 370 mean).
- Decision: `rock_geometry: auto` uses the STL when the project has one, voxels only otherwise
  (results then carry a note). *Revisit:* smaller smoothing (σ = 0.5) for the voxel path.

**D-406 Interface optics.**
- Unpolarised Fresnel (Rs+Rp)/2 at every interface, Snell refraction; n_water 1.34, n_glass 1.5.
- Case 3a: transmission 0–85° within |z| ≤ 1.7 of 1 − R; refraction angle within 5e-6° of Snell.
  Case 3b (water→12.7 mm glass→air): slab transmission within |z| ≤ 1.1 of the incoherent slab
  formula; exactly zero beyond the 48.27° critical angle.
- Optional rough surface: isotropic Gaussian facet slopes (RMS slope σ), facet sampled by rejection
  ∝ projected area |d·m|/(m_z|d_z|); a reflected ray that would leave through the surface again is
  mirrored back down. Verified (`test_rough_surface_beam_spread`): beam-spot RMS at 0.5 m depth vs
  small-slope theory (1 − 1/n)(σ/√2)D: ratio 1.006, 1.000, 0.993, 0.984 for σ = 0.05, 0.1, 0.2, 0.3.
- CFD hook `roughness_from_cfd(model, mean_speed, slope_per_ms)`: σ = σ0 + k|U_top|, k with no
  default (no validated relation; see A-402).

**D-407 Beer–Lambert as continuous weight attenuation w·exp(−Kd s)** (not analog absorption): lower
variance, exact per segment. Detector crossings use the weight at the crossing point.
Case 4a: flux vs T0 exp(−Kd z) error 1.9e-5 (Kd 0.5–2 1/m); case 4b (diffuse light, Fresnel + Snell +
z/cos θ_w path lengths): −0.34 … +0.11 % vs quadrature, MC σ 0.18 %; case 4c spectral: |z| ≤ 0.92.

**D-408 Diffuse bounces (sand, rock, coralline walls): Lambertian, Russian roulette with p = albedo,**
so photon weights stay equal and the noise estimate stays 1/√hits. On by default (sand albedo 0.6
makes sand-reflected light + surface TIR a significant contributor).

**D-409 Tally precision (found in verification).**
- First version: all tallies fp32 atomics. Case 4a failed: a collimated beam puts every photon of a
  batch into one cell; fp32 accumulation of millions of near-identical weights rounds systematically
  (same mechanism as A-109). Flux errors were 1.26 % (identical at every Kd) and spectral ratios off
  by up to 0.6 %.
- Now: grid tallies (stack, rock, sand) fp32 on the device — a cell sees ~10²–10³ photons per batch,
  bound n·2⁻²⁴ ≈ 1e-5, checked every batch (warning above 2¹⁵) — flushed to fp64 on the host every
  batch (2²² photons). Plan maps, probes and counters (can be hot) are fp64 on CUDA/CPU; on
  Vulkan/Metal they fall back to fp32 (fp64 atomics unavailable) — a known limitation for
  concentrated-beam verification scenes, not for tank scenes.
- Counters are replicated in 4096 rows (was 256): the refraction-angle read-back error dropped from
  0.0065° to 0.0006° (fp32), then to < 1e-5° with fp64 counters.
- fp64 everywhere was measured at 2.05 vs 2.85 Mrays/s (shared GPU) — rejected for grid tallies.

**D-410 The stack tally stores Σw only (no Σw²).**
- It is the costliest tally (~90 plane crossings × atomics per photon descent). Dropping Σw² raised
  the empty-tank-with-stack throughput 5.7 → 11.2 Mrays/s and the full demo 5.7 → 10.2 Mrays/s.
- Stack weights are never split, only attenuated (0.9–1.0), so the relative noise is 1/√Σw.
  Case 5: this estimate is 0.6–4 % above the empirical seed-to-seed noise (conservative).

**D-411 Calibration.**
- Scenes: `tank` (own meter in the user's tank, full scene), `air` (manufacturer chart: fixture alone
  in air), `open_water` (manufacturer in-water chart: fixture above unbounded water, black floor).
  Readings are predicted by probe tallies (25 mm cosine squares).
- In-water readings without immersion correction × immersion factor (default 1.08, A-409) (C7).
- Scale: closed-form relative least squares s = Σ(p/m)/Σ(p/m)². Beam angle (optional): 11-point coarse
  scan, 9-point fine scan over ±1 coarse step, parabola through the fine minimum; common random
  numbers make the cost curve smooth.
- *Found in verification:* coarse-scan-plus-parabola alone gave beam 71.3° (truth 70°) and RMS
  residual 3.8 % (criterion 3.5 %): the cost is V-shaped at 13° spacing. Two-stage: 68.6°, RMS 1.7 %.
- One global scale for all fixtures; results labelled "calibrated (scale, RMS residual)", otherwise
  "relative/estimated (uncalibrated)".
- Case 6a: beam 68.6° (truth 70), scale 0.8324 (truth 0.8333), RMS 1.7 %. 6b: scale 0.8392 (+0.7 %),
  RMS 2.1 %. Ignoring the immersion metadata: scale −6.8 % (≈ 1/1.08).

**D-412 Verification-case amendments (recorded as required by D-110).**
- Case 1: probes scaled with h intercept the same photon cone at every h, so their exponent is −2
  identically (bookkeeping only). Added fixed-size (40 mm) probes, criterion |n_MC − n_exact| ≤ 0.015:
  −1.9781 vs −1.9823. A first version also required n_exact within 0.005 of −2; that was my error in
  the reference (a 40 mm probe at 0.1 m is not a point), dropped. The plane χ²/dof tests (0.96–0.99)
  are the substantive inverse-square check.
- Case 5: the stack noise check was added (before the case was first run) with D-410.

**D-413 Convergence analysis: pooled RMS, not median of per-cell standard deviations.**
The first analysis read the internal estimate as 8–16 % high at every N. With 4 seeds the median of
the sample standard deviation is 0.888 σ (χ²₃ median), i.e. ratio 1.126 expected. With the unbiased
pooled RMS: internal/empirical 1.00–1.04, slopes −0.503 / −0.499 / −0.498 (sand, plan, stack).

**D-414 Default ray count = noise target.** `lighting.target_noise: 0.02` (relative 1σ of an
average-lit 5 mm surface patch, i.e. rock + sand patches within ±20 % of the mean irradiance),
`rays: 2e7` minimum, `max_rays: 2e8` cap. Demo: 1.17e8 photons, 2.0 %, 16.5 s on the shared GPU.
An explicit `rays=` argument traces exactly that many. PLAN's 4e7 for 2 % assumed 1 cm² patches;
5 mm cells (0.25 cm²) need 9.0e7 for 2 % and 3.6e8 for 1 % (case 5 fit).

**D-415 Throughput (measured; GPU shared with the running CFD grid study at 97–100 % utilisation).**

| Scene (demo tank) | CUDA (RTX 3070, shared) | CPU (5600X, 12 thr) |
|---|---|---|
| empty tank, plan maps only | 44–59 Mrays/s | 17.4 |
| empty tank + node stack | 9.5–11.2 | 5.0 |
| rock, plan maps only | 23–29 | 8.8 |
| full (rock + stack + bounces) | 7–21 (contention-dependent) | 3.5 |

PLAN estimated ≥ 50 Mrays/s GPU and ~2 Mrays/s CPU. CPU beats the estimate; GPU reaches it only
without the node-stack tally, which is atomic-bound. A 2 %-noise demo run: ~6–17 s GPU, ~35 s CPU.
*Revisit:* warp-aggregated atomics or a coarser stack (every 2nd node) if sweeps become slow.

**D-416 CFD-grid interface (for coral scoring).** `par_on_cfd_grid(model, result) -> CfdPar`:
meter-equivalent PAR at every fluid node; surface-normal PAR at fluid nodes face-adjacent to rock or
sand (rock: area-weighted mean of the rock patches in the node's 3×3×3 neighbourhood; sand: the sand
cell). Where a surface-adjacent node's water fraction is < 0.5 (STL rock reaches into the CFD cell)
the meter value is taken one, then two, nodes outward along the SDF normal. Demo Preview: 100 % of
the 7,247 surface-adjacent nodes evaluated.

**D-417 DLI from one photoperiod schedule that scales all fixtures.** PAR is linear in output, so
DLI = PAR(100 % schedule) × ∫s(t)dt × 1e-6. Per-channel / per-fixture schedules are not supported
(would need per-channel tallies). Default 10:00–20:00, 60-min ramps → 9 h full-intensity equivalent.

**D-418 Photon unit µmol internally** (PPF µmol/s, PAR µmol m⁻² s⁻¹), DLI mol m⁻² d⁻¹. Light units
live in `reefflow/light/lunits.py`; `reefflow/units.py` is unchanged. Bare angles are degrees.

**D-419 Spectral mode**: 30 bins of 10 nm over 400–700 nm; each photon samples a bin from its
fixture's channel-weighted spectrum (no extra cost); Kd per bin. Refractive-index dispersion is
neglected (A-403). Channel spectra are generic Gaussian sums in the fixture library.

## Phase 5 — browser interface

**D-501 — UI architecture: local FastAPI + plain ES-module three.js, no build step.**
The backend (`reefflow/ui/server.py`, `create_app(workspace)`) only translates HTTP requests into core calls: `project.parse` (validation), `tank.build` + `run.estimate` (pre-run summary), `run.run` and `analysis.analyze` (jobs), `analysis.compare.compare`, `geometry` (meshes). The frontend is plain ES modules plus an import map. There is no bundler and no Node toolchain, so the package ships exactly the files the browser loads. The server listens on 127.0.0.1 only (`--host` can override this). There is no authentication: it is a single-user desktop tool.

**D-502 — Offline: three.js r160 (0.160.0) vendored in `reefflow/ui/static/vendor/three/`.**
The vendored files are `three.module.js`, OrbitControls, TransformControls, STLLoader, PLYLoader and the MIT LICENSE, each unmodified. Their SHA-256 hashes are in `VERSION.txt`. Nothing loads from a CDN at runtime; a test checks that `index.html` references no external asset. r160 is the last release with a single-file `three.module.js` and the old `TransformControls`-is-an-`Object3D` API, so no `three.core.js` split is needed.

**D-503 — Project file: UI-owned `designer:` section.** *(Accepted by the lead on 2026-10-31: one file per tank is preserved and the core ignores the section.)*
The parametric rock elements (type, parameters, position, rotation and scale, in metres and degrees) are saved under a top-level `designer:` key in the same YAML. That keeps the spec's rule of one project file per tank. `project.parse` ignores unknown keys, so the core does not change. The simulation still reads only `rockscape:`. After "Build rockscape" that section holds a single entry, `rockscape.stl` (mm), and `designer.built_hash` records the element geometry it was built from. When the elements change without a rebuild, the server refuses to start a run (HTTP 409) and the UI shows "rock not built". When an existing project has `rockscape:` but no `designer:`, its meshes become movable `mesh` elements and are marked as already built, so nothing has to be rebuilt.

**D-504 — Rock geometry is generated server-side only.**
Each element is a signed-distance-like field: rock = `geometry.rock_sdf`, the same noise-deformed blobs as the demo. Slab, pillar, arch and bridge are analytic fields with mild noise. An imported mesh is voxelized with `geometry.voxelize_mesh` and turned into a signed distance with a Euclidean distance transform. The browser preview of each element is the server's marching-cubes mesh of that same field, so the preview cannot diverge from the build.
- The union is the min of the element fields on one grid. The default spacing is 4 mm; 3 to 8 mm is selectable, and it should be at most the CFD cell size of the finest planned run.
- The union is clipped to the tank interior (glass, bottom glass, rim), closed at the grid boundary, meshed with marching cubes, and cleaned with trimesh. The result is a single watertight STL. Watertightness is tested.

**D-505 — Element placement convention.**
world = position + R · (scale ∘ local), with R = Rx·Ry·Rz (three.js Euler order 'XYZ'). The same matrices are used in Python (`rockgen.rotation_matrix`) and in the browser. The local origin is the element's base. Rocks extend `bury` (default 15 % of their height) below the base, so they sit in the sand. "Snap down" casts rays straight down from 5 footprint points and puts the base on the highest sand or rock hit.

**D-506 — Geometry checks are informational only.**
- Glass contact: the gap is ≤ 1 grid cell. A gap under 25 mm is reported as "close", with cleaning access in mind.
- Above water: some rock is above the water surface.
- Floating: a 26-connected rock group never reaches the sand layer.
- Unsupported overhang: a group's centre of mass, assuming uniform density, lies outside the convex hull of its cells within one cell of the sand top.

None of these block a run. The spec asks only for a simple footprint/CoM check, and the UI says that this is not a structural analysis. The dry-weight estimate is solid volume × an editable bulk density. The 1.6 kg/L default is a placeholder assumption.

**D-507 — Units in the UI.**
The editor model exchanged with the browser holds plain SI numbers. The browser converts for display and input using the unit table from `reefflow.units`, which the server sends. On save, the server writes quantities as "<number> <unit>" strings in the project's display system (US: in / GPH / ft head; SI: mm / L/h / m head). The time unit is s and the dead-zone threshold is written in cm/s. Velocities are displayed in in/s (US) or cm/s (SI), following `units.DISPLAY`. Keys the editor does not know are preserved ("extra").

**D-508 — Jobs: one GPU job at a time, each in its own spawned process.**
A single manager thread runs a FIFO queue. Each job runs in a `multiprocessing` 'spawn' process, which isolates the Taichi/CUDA state and survives worker crashes. Cancel terminates that process, because Taichi kernels cannot be interrupted from Python. The worker sends progress lines; the manager derives a percentage from them and streams lines, percentage, phase and status to the browser over SSE.
- Phase bands: build 0–3 %, flow run 3–92 % (3–86 % plus light 86–92 % when PAR is included), analysis 92–100 %.
- The flow-run percentage is parsed from `run.run`'s "( 12.3 %)" lines, which the core prints at most every 30 s (see the requested core change in the report).
- Logs are kept in `ui_log.txt` and the job record in `ui_job.json`. Runs left queued or running by a server stop are marked "interrupted" at the next start.

**D-509 — Run snapshot.**
A run directory gets `project.yaml` with the `designer` section removed and the rock meshes copied to `inputs/`. `run.json` therefore points at a file that never changes, and `analysis.load.load_run` can rebuild the identical tank model after the user edits the project. Comparison and re-analysis keep working too.

**D-510 — Results go to the browser as 2-D slices only.**
- Slices are cut server-side from `fields.npz`, `analysis.npz` or `par.npz` without rebuilding the tank model. They are sent as raw little-endian float32 with a JSON `X-Meta` header carrying shape, extent, index and percentiles.
- A Fine run's plan slice is about 400×145×4 B ≈ 230 kB. The 3-D arrays (8.9 M cells) never leave the server.
- Colouring happens in the browser, using 256-entry matplotlib colormaps served by `/api/colormaps`. This allows hover readouts of real values and user-set colour ranges.
- NaN marks non-water cells, which are transparent.
- Vectors are sub-sampled to about 48 arrows along the long axis.

**D-511 — Coral scores in 3-D use the analysis outputs as written.**
The viewer shows the core's `rock_score_<profile>.ply` meshes; their sRGB vertex colours are converted to linear for three.js. Best spots come from `analysis.json`. The UI does not recompute any score.

**D-512 — Lighting goes through one adapter (`reefflow/ui/lighting_adapter.py`).**
- The UI edits only the Phase-4 mounting fields: name, archetype, position [x, y], height, tilt_x, tilt_y, rotate_z, intensity. All other fixture keys and the rest of `lighting:` are preserved.
- The beam cone is drawn with Phase 4's rotation R = Ry(−tilt_y)·Rx(tilt_x)·Rz(rotate_z). The fixture library is read from the Phase-4 YAML directly, so the server does not import Taichi for it.
- Optional PAR: after the flow run, the worker calls `reefflow.light.run_lighting(project, model=model)` and `par_on_cfd_grid`, writes `par.npz` (meter, normal, label), and passes `meter` to `analyze(par=...)`. A lighting failure is logged and the flow results are kept.
- The PAR slice layer appears whenever `par.npz` exists.
- YAML 1.1 reads unquoted `on:`/`off:` keys as booleans. The adapter restores them to the strings "on"/"off", and PyYAML then writes them quoted.

**D-513 — Undo/redo.**
Undo works on whole-editor snapshots: the JSON of the editor model is taken at each committed change, and the last 100 are kept. It covers rock elements, pumps, lights, tank and units.

**D-514 — The Photos tab follows Phase 6's workflow; the UI holds only state.**
`reefflow/ui/photos.py` keeps the interactive session; all computation is a call into `reefflow.photo`. The steps map to: `calibrate_view`, `auto_segment_views` / `segment_view` plus the mask edit functions, `carve_visual_hull`, `apply_edits`, `side_by_side`, `export_stl` and `volume_report`.

The state is stored per project in `photos/session.json` and is all JSON: clicks, focal settings, calibration results (`View.to_dict`), the edit list, density settings. Alongside it: masks as PNG (`<vid>/mask.png`), the uploaded original, and a working JPEG. It survives restarts. The reconstruction is reproducible from (photos, clicks, masks, edits).

A photo id is its view kind, with `-2`, `-3` for extra photos of the same side, so an oblique fourth photo can be added.

**D-515 — Working image.**
Each photo is decoded once with OpenCV, which applies the EXIF orientation, and reduced to at most 2000 px on the long side (INTER_AREA). It is stored as a JPEG at quality 93. Every pixel coordinate (clicks, masks, overlays) refers to this working image, so the browser, calibration, segmentation and carving all agree.

The EXIF 35-mm-equivalent focal length is read with Pillow and offered as a known focal length via `focal_px_from_35mm`, on the working size. The user can also give it in px or 35-mm mm, or let the solve find it. 2000 px keeps GrabCut and the reprojections interactive; the validation used 1600 x 1200.

**D-516 — Carving runs as a background job on a separate CPU queue.**
It runs in a second `JobManager` (ids start with "p"), so it never waits behind a GPU run. It runs in its own process, like the CFD runs.

The job carves in z-slabs of about 0.4 M voxels to report real progress. Every voxel is independent, so the result is identical to a single `carve_visual_hull` call; the per-view statistics are summed. It then removes small islands, writes `recon/occupancy.npz` (bit-packed) and `carve.json`, and runs the comparison.

Cancel is cooperative (a stop event checked between slabs). An input signature covers the calibrations, mask bytes, voxel size, tolerance and tank; it flags the result as stale when any of them changes.

**D-517 — Mask editing is server-side.**
Brush strokes and polygons are drawn on the client for immediate feedback, then sent as operations to `reefflow.photo.brush` / `polygon` (and `fill_holes` / `remove_small` / `smooth`). The server returns the mask PNG, so it is the single authoritative copy. Server memory keeps a 20-step mask undo history per photo, lost on restart.

"Suggest all" keeps any mask the user has edited.

**D-518 — Using the reconstruction as the rockscape.**
The edited occupancy is exported with `export_stl` (exact marching cubes on the voxels, base extended one cell into the sand) to `rockscape_photo.stl` in the project folder, and becomes the project's only `rockscape:` entry.

The designer section is set to one `mesh` element pointing at that file, marked as built. The project can be run immediately, and designer pieces can be added on top and rebuilt; their union includes the photo mesh. The 3-D display uses a lightly smoothed mesh (sigma 0.6 voxel) for appearance only; the volume, the STL and the simulation use the exact voxels.

**D-519 — Job worker uses the core's structured progress and cooperative cancel** (`run.run(on_status=, should_stop=)`).
Status messages are throttled to 2 per second and drive the progress bar and the simulated time / max speed / ETA line. Cancel sets a stop event: `run.run` raises `RunCancelled` at the next monitor point (about 0.5 s in a test). The worker is still terminated after a 20 s grace period if it does not respond, for example while Taichi compiles or during the analysis.

## Phase 6 — photo reconstruction

Code: `reefflow/photo/` · tests: `tests/test_photo.py` (fast), `tests/test_photo_pipeline.py` (slow) · validation: `python -m reefflow.photo.validation` → `verification/results/10_photo/`.

**D-601 Calibration by full PnP with refraction in the loop, not a rim homography (PLAN C9).**
- Per photo the user clicks named tank points. These are the inside corners of the glass box at four levels (rim, water line, sand line, bottom), named `<level>_<FL|FR|BR|BL>`, plus optional custom points with known coordinates and optional "edge" clicks (anywhere on a straight tank edge; 1 residual each).
- `recommended_points(kind)` always mixes ≥ 2 levels, so the set is never coplanar.
- Unknowns: focal length f (px) and pose (6). Principal point fixed at the image centre, square pixels, k1 = 0 by default (A-601).
- Solve in three stages:
  1. OpenCV SQPnP for 31 log-spaced f candidates (0.3–6 × image size), keeping the best pinhole fit.
  2. Levenberg–Marquardt pinhole refinement.
  3. LM refinement with the exact refractive projection (D-602). Reported residuals come from this model.
- The calibration also reports:
  - per-point residuals, RMS and max;
  - parameter 1σ from s²(JᵀJ)⁻¹;
  - propagated camera-centre 1σ;
  - the pinhole-only RMS as a diagnostic.
- Refused: fewer than 4 points; no redundancy; a coplanar set with free f. In the last case f and camera distance are inseparable; the user can enter f, e.g. `focal_px_from_35mm` from EXIF.
- Verified:
  - Exact recovery with exact clicks, all five view kinds (tests; validation C1, residual ~1e-11 px).
  - With 1 px noise, the normalized camera-centre errors have RMS of order 1 against the reported 1σ (test).
- *Revisit if:* real photos show systematic residuals; then free k1 and the principal point, and require more points.

**D-602 Exact flat-interface refraction model (air | glass | water, air | water).**
- For parallel planes the tangential invariant s = nᵢ sin θᵢ is conserved, so forward projection is one scalar equation per point: Σ dᵢ s/√(nᵢ² − s²) = h. It is solved by safeguarded Newton to 1e-14, vectorized over millions of voxels.
- Back-projection follows Snell plane by plane.
- No apparent-depth (paraxial) approximation. The paraxial limit is tested: d/n.
- Top view: one interface (the still water surface). Side views: two planes (outer and inner pane faces). The medium beyond the pane is water below the water line and air above it.
- Validity is tracked, not assumed. A projection is invalid when:
  - the camera is not outside the interface;
  - the entry point falls outside the pane or opening;
  - the entry is on the wrong side of the water line for the point's medium (the ray would cross the surface);
  - a submerged point lies outside the tank footprint (bug found by test and fixed: such a ray crosses a side pane).
- Invalid projections carry no information.
- Round trip: project → back-project → distance to ray < 1e-9 m; Snell invariant to 1e-12 (all five view kinds).
- Magnitude of the correction: uncorrected refraction displaces rock points by tens of pixels (validation `refraction_displacement_rock_px`).

**D-603 Segmentation: GrabCut with an automatic, calibration-driven initialisation; SAM optional.**
- First attempt, rejected: the purely geometric prior, with every rock-able pixel set to "probable rock". GrabCut then labelled the whole tank as rock on every synthetic view (mask IoU 0.06–0.20), because the background model had no back-wall or open-sand samples.
- Adopted (`segment.auto_segment_views`):
  - The top view goes first. Background colours (k-means, CIE-Lab) are learned from rays that cannot meet rock: those that miss the rock-able box inset 3 cm from the glass and 6 cm below the surface, plus top-view rays ending on sand next to the glass.
  - Pixels with ΔE > 14 from every background colour are "probable rock". GrabCut refines this.
  - The top mask's prism is carved at 1 cm and reprojected into each side view as a guide. Background colours are then learned from every tank pixel outside the guide.
  - If GrabCut swallows > 85 % of the candidate region, the threshold is raised ×1.4 and it retries.
- The threshold of 14 was chosen on render seeds 1–3; the validation uses held-out seed 7.
- A SAM-family model is used instead when installed and requested (`method='auto'|'sam'`). Otherwise the result notes "SAM … not installed; used GrabCut".
- Mask editing is pure functions on bool arrays: `brush`, `polygon` (add/subtract), `fill_holes`, `remove_small`, `smooth`, PNG encode/decode. A per-view `ignore` mask covers equipment, trim and glare.

**D-604 Optional models: registry, license gate, explicit download only.**
- `photo.models.MODELS` lists MobileSAM and SAM ViT-B (Apache-2.0), Depth Anything V2 Small (Apache-2.0), and V2 Base/Large (CC-BY-NC-4.0, excluded, listed so the UI can explain).
- `download_model(name, accept_license=True)` is the only network path. It refuses non-commercial models, honours `REEFFLOW_OFFLINE=1`, caches to `%LOCALAPPDATA%/ReefFlow/models` or `$REEFFLOW_MODEL_DIR`, and writes a sha256 sidecar.
- torch and the model packages are not dependencies; imports are guarded.
- `estimate_depth` and `photogrammetry_reconstruct` are interface stubs (spec item 6). They raise `ModelNotInstalled` or `NotImplementedError`.
- The SAM inference path follows the published `SamPredictor` API but is untested here (no weights installed).

**D-605 Carving rule: any-view veto at voxel centres, 1 px mask dilation, explicit "unknown".**
- A voxel is removed if any view that validly sees its centre (D-602; inside the image; not on an ignore pixel) projects it onto non-rock.
- Voxels seen by no view are left empty and counted ("unconstrained").
- The grid convention is identical to `geometry.voxelize_mesh` (node centre = origin + i·dx; centre-inside rule), so ground-truth comparison is like-for-like.
- Default grid: the water column above the sand, at any dx; the validation uses 5 mm (the standard CFD preset).
- Cost: ~20–25 s for 1.8 M voxels × 3 views on one CPU core (numpy). This is acceptable for a one-time step. A coarse-to-fine octree is a Phase-7 option.

**D-606 Mesh export: marching cubes at level 0.5 on the zero-padded occupancy.**
- Without smoothing, re-voxelizing the exported STL on the same grid reproduces the occupancy exactly (test; validation V6: 0 mismatched voxels).
- The base is extended one cell into the sand. Winding is forced outward (positive signed volume).
- The STL is in tank coordinates in mm, referenced from the project file like any rock: `{file: rock_photo.stl, unit: mm, translate: [0 mm, 0 mm, 0 mm]}`.

**D-607 Validation against a known synthetic rockscape, with criteria declared before running (as D-110).**
- Renderer (`photo.synthetic`): numpy/OpenCV only, no heavy renderer.
  - Background: per-pixel back-projection through the interface.
  - Rock: exact vertex projection plus a painter's algorithm on ~4 mm triangles.
  - Also: water tint, sensor noise σ = 3, blur 0.8 px, and an exact ground-truth mask.
- Cameras: hand-held positions (top 1.05 m above the rim; front 1.45 m from the glass; left 0.8 m), 1600 × 1200 px, f = 1200 px.
- Clicks: exact projections plus Gaussian noise.
- V1 (perfect masks and cameras) failed its pre-declared bound. The bound was **not** changed. An informational case (V1c, a 4th oblique photo, plus a per-island breakdown) was added afterwards and is labelled as such.

**D-608 Occupancy edits are a JSON-replayable list.**
- `apply_edits(occ, grid, [{"op": "carve"|"add", "shape": {...}}])`. Shapes: sphere, box, oriented box, capsule/polyline tunnel, and arch (flat floor, semicircular top, extruded along a horizontal axis).
- The UI stores the list in the project. Photos + calibration + masks + edits then reproduce the rockscape.
- Validation V7: a blind cave invisible to every silhouette (silhouette change 0.000 in all views) is overfilled by the hull. Carving the same capsule restores IoU to within 0.005 of the cave-free value.

**D-609 Found in shared code (not changed, outside this phase's files): `geometry.sdf_to_mesh` returns inward-facing triangles.**
- The signed volume is negative (sphere test: −0.52 for +0.52), contrary to its "outward normals" comment.
- The ray-parity voxelizer and `mesh_report` (which uses `abs`) are unaffected. The demo `rock.stl` therefore has inverted normals, which matters for renderers, lighting and back-face culling (Phase 4 ray tracer!).
- The photo module orients meshes by signed volume.
- Suggested fix: drop the `[:, ::-1]` in `sdf_to_mesh`.

**D-610 Conventions.**
- Images are RGB uint8 H×W×3; masks are bool H×W indexed `mask[v, u]`.
- Pixel centres are at integer coordinates, with (0, 0) the centre of the top-left pixel (OpenCV).
- All 3D quantities are tank coordinates in metres.
- Click lists are `{name: [u, v]}`.

### Phase 6 validation results (verification/results/10_photo/results.json)

Demo 75-gal tank and demo rock (12.10 L above the sand, 96,799 voxels at 5 mm). Run time 953 s on CPU.

| Case | What | Result | Criterion (pre-declared) | Verdict |
|---|---|---|---|---|
| C1 | exact clicks → camera | RMS ≤ 6e-12 px; centre error ≤ 8e-11 mm | < 1e-6 px, < 1e-3 mm, f < 1e-6 | PASS |
| C2 | σ = 2 px clicks, 8 trials × 3 views | RMS 2.13 px (expected 2.1); centre error 14.8 mm (top 10.8 / front 27.8 / left 5.7, against reported 1σ of 12.6 / 29.1 / 8.4); \|Δf\| 0.88 % | ≤ 3 px, ≤ 25 mm, ≤ 2 % | PASS |
| V1 | exact cameras + true masks, 3 views | IoU **0.679**, recall 1.000, volume **+47.3 %** | IoU ≥ 0.70, overfill ≤ 45 % | **FAIL** |
| V1c (info, added after the first run) | + oblique front photo (4 views) | IoU 0.705, +41.9 % | — | — |
| V1 (info) | top + front only | IoU 0.618, +61.9 % | — | — |
| V2 | σ = 2 px calibration + true masks, 8 trials | IoU 0.682, recall 0.99995, +46.6 % | IoU ≥ 0.629, recall ≥ 0.95 | PASS |
| V3 | **full pipeline**: σ = 2 px + automatic GrabCut, held-out seed | mask IoU 0.988 / 0.997 / 0.998; voxel IoU 0.678, recall 0.9999, +47.4 % | masks ≥ 0.90, IoU ≥ 0.65, 0 … +50 % | PASS |
| V4 | click-noise sweep, 6 trials (true masks) | IoU σ=0: 0.679, 1: 0.679, 2: 0.682, 4: 0.683 ± 0.005, 8: 0.674 ± 0.015; recall at 8 px 0.993; centre error 6 / 13 / 39 / 77 mm at 1 / 2 / 4 / 8 px | IoU(4 px) ≥ 0.579 | PASS |
| V5 (info) | refraction ablation | (a) true cameras, projection without refraction: IoU 0.602, recall 0.994, +64.7 %. (b) pinhole calibration + pinhole carving: IoU 0.687, recall 0.9989, +45.2 %, **but the fitted cameras are physically wrong**: centre off by 276–543 mm, f +35–38 % | — | — |
| V6 | STL export → re-voxelize | 0 mismatched voxels of 142,657; watertight | ≤ 0.1 % | PASS |
| V7 | blind cave (0.108 L) + `carve_tunnel` | silhouette change 0.000 (all views); IoU against the cave truth 0.673 → 0.678 | ≥ 0.679 − 0.005 | PASS |

Volume and weight (V3): reconstructed 17.8 L, i.e. 25.0 kg at the default 1.4 kg/L (range 19.6–32.1 kg). The truth is 12.1 L, i.e. 16.9 kg. The overestimate is the hull overfill.

Interpretation:
1. **V1 failure.** The overfill is the visual-hull limit, not calibration or segmentation error: V1, V2, V3 and V4 all give 0.68 IoU. It is located in the diagonal "corner wedges" of each island in plan view, plus the concavities between blobs (fig3).
   - The pre-declared bound assumed the left island would suffer most from superposition in the side view. Measured per-island IoU is 0.676 (left) and 0.683 (right), so that hypothesis was wrong. The surface roughness and multi-blob shape cost more than the convex-sphere analogue predicted.
   - Practical consequence: the reconstructed volume is ~+45 % too high for a 3-photo set. A 4th oblique photo gives +42 %.
   - Users must expect an upper bound and use the edit tools. Further mitigation is the stubbed depth/photogrammetry route.
2. **Click noise is not the limiting error.** Even 8 px noise costs only 0.005 IoU, because the 1 px mask tolerance and the 5 mm cells dominate.
3. **The refraction model matters for physical consistency, not for this particular IoU.**
   - With 8 spread points per photo, a pinhole fit absorbs flat-interface refraction into a wrong "effective camera" (≈ 35 % longer focal length, 0.3–0.5 m farther away). That camera carves almost as well.
   - The pinhole shortcut fails as soon as the camera is constrained by physics: a known focal length (EXIF), fewer points, or reuse of a pose. With true cameras and no refraction correction, IoU drops 0.68 → 0.60 and 0.6 % of the true rock is wrongly carved away.
   - Only the refractive model recovers the true camera (C1, C2).

## Phase 2–3 results recorded after the runs finished

**D-214 Grid-convergence study (verification 6), D3Q19: bulk flow converges, near-surface quantities do not.** Demo tank; Preview / Standard / Fine (10 / 5 / 3 mm); identical physical model on every grid (D-211). Values ± 2σ statistical.

| Metric | Preview | Standard | Fine | Standard → Fine |
|---|---|---|---|---|
| Volume-mean speed (cm/s) | 10.85 ± 0.15 | 11.98 ± 0.14 | 12.57 ± 0.09 | +5.0 % |
| Median speed (cm/s) | 9.02 | 10.06 | 10.91 | +8.4 % |
| Speed at 75 % depth (cm/s) | 17.14 | 17.88 | 18.24 | +2.0 % |
| RMS fluctuation (cm/s) | 5.08 | 5.35 | 5.71 | +6.6 % |
| Volume above 30 cm/s (%) | 2.73 | 3.22 | 3.20 | −0.6 % (below noise) |
| **Dead-zone volume (%)** | 6.58 ± 0.44 | 3.59 ± 0.18 | 2.45 ± 0.04 | **−32 %** |
| **Speed 15 mm above the sand (cm/s)** | 3.71 | 7.95 | 9.79 | **+23 %** |
| **Speed 15 mm above rock tops (cm/s)** | 4.29 | 6.36 | 8.12 | **+28 %** |

- Changes are monotonic. The observed orders are low (0.3–1.4), so GCI_fine is large for the near-surface metrics (42–147 %).
- Run cost and quality:
  - Fine: 76 min at 1,622 MLUPS; Standard: 12.6 min at 1,297 MLUPS.
  - Mass drift ≤ 3 × 10⁻⁶; peak lattice speed 0.125–0.138.
  - Kinetic energy differs by ≤ 1.7 % between the two halves of the averaging window.
- Speed profiles: Standard and Fine agree in the interior; the differences sit in near-wall layers (`verification/results/06_grid/grid_profiles.png`).
- **Interpretation:** bounce-back walls at these resolutions produce a near-wall layer about one cell thick, so everything measured close to a surface changes with the grid. Verification 8 (D-702) shows this layer over-predicts wall friction. Coarser grids therefore overstate dead water: about 2.7× at Preview and 1.5× at Standard, relative to Fine.
- **Guidance:**
  - Preview is for the bulk flow pattern only.
  - Standard is the minimum for dead-zone, sand and coral metrics, and is suitable for comparing designs.
  - Absolute near-surface values are conservative even at Fine (A-310).
- **D3Q27 shows the same pattern.** Fine: mean speed 12.55 vs 12.57 cm/s (D3Q19), dead volume 2.88 vs 2.45 %.
  - Standard → Fine: dead volume −32.5 %, sand speed +19 %, mean speed +5.2 %.
  - Run times: Standard 25.6 vs 12.6 min, Fine 92 vs 76 min.
  - The lattice choice (D-120) is confirmed on the application: bulk metrics agree within 0.2–2 %. D3Q27 reads 15–19 % more dead volume at every resolution, consistent with its slower jet transition (D-120).

**D-310 Sensitivity runs (Preview demo, D3Q19).**

| Case | Mean speed | Median | Dead-zone volume | RMS fluctuation |
|---|---|---|---|---|
| Baseline (derating 0.7, I = 0.05) | 10.85 | 9.02 | 6.58 % | 5.08 |
| All pump flows −20 % (rerun) | 8.58 | 7.06 | 9.73 % | 4.00 |
| Same, predicted by rescaling the baseline (D-303) | 8.68 | — | 9.40 % | — |
| Inflow turbulence I = 0.02 | 10.27 | 8.39 | 7.11 % | 4.88 |
| Inflow turbulence I = 0.10 | 11.36 | 9.68 | 5.19 % | 5.41 |

- The flow-rescaling shortcut used for the derating sensitivity is confirmed: speeds within 1 %, dead volume within 0.3 points (noise ≈ 0.4).
- The inflow-turbulence setting (A-207) moves tank-wide mean speed by about ±5 % and dead volume by about ±1 point over I = 0.02–0.10. That is much less than its effect on a single jet's centreline (D-204), because the tank-scale circulation depends on J.

## Phase 7 — performance, robustness, packaging

**D-701 Batched stepping with Taichi compute graphs.**
- `LBMSolver.step(n)` runs pre-recorded graphs of up to 16 steps instead of n Python-side kernel calls. Falls back to the plain loop if the backend cannot build graphs.
- Bit-identical to the plain loop on both lattices, including odd step counts (`tests/test_sources.py::test_compute_graph_stepping_is_bit_identical`).
- Measured on the demo tank with the GPU otherwise idle:
  - Preview (0.25 M cells): 0.567 → 0.179 ms per step, 442 → 1,402 MLUPS (**3.2× faster**).
  - Standard (1.9 M cells): 1.022 → 1.021 ms per step (1,890 MLUPS; already GPU-bound).
- A first measurement under GPU contention gave only +26 %.

**D-702 Wall model: attempted, failed verification, left OFF (experimental). Near-wall bias quantified instead.**
- Verification 8: turbulent plane channel, periodic, body-force driven, so the wall shear is fixed at ρu_τ² by the momentum balance; the carried flow is compared with Dean's (1978) correlation. Production LES settings.

  | Case | First node y⁺ | U_b/u_τ | Reference | Error |
  |---|---|---|---|---|
  | Re_τ 2000, h = 16, bounce-back | 62 | 10.89 | 21.7 | −50 % |
  | same, wall model v1 (viscosity added only) | 62 | 10.85 | 21.7 | −50 % |
  | same, wall model v2 (relaxation time overridden) | 62 | diverged | — | — |
  | Re_τ 2000, h = 32, bounce-back | 31 | 12.84 | 21.7 | −41 % |
  | **Re_τ 500, h = 32, bounce-back (tank-like y⁺)** | 8 | 13.02 | 18.0 | **−28 %** |

- **Finding:** at under-resolved walls the channel carries too *little* flow for its wall shear. The near-wall layer is too frictional: Smagorinsky without near-wall damping puts too much eddy viscosity next to the wall.
- Wall model v1 (Spalding-based, only able to add viscosity) did nothing, because the needed wall viscosity is *below* the Smagorinsky value.
- Wall model v2 replaced the wall nodes' relaxation time with τ = ½ + 3ν_wall (≈ 0.5004). It diverged even from a gentle start: the wall nodes lose the stabilising eddy viscosity.
- A robust wall model (imposed-stress / slip-velocity boundary, or near-wall SGS damping with a stability safeguard) is the main open item for accuracy near surfaces.
  - The code (`reefflow/wallmodel.py`, `simulation: wall_model`) remains but is off by default, and a run with it on prints a warning.
  - The kernel change for v2 only adds a branch for negative entries of the per-node relaxation array. Sponge layers are unchanged, and all tests pass.
- Consequences are recorded in A-310.

**D-703 Bug fixed: procedurally generated rock meshes had inward-facing triangles** (found by the Phase 6 work).
- `geometry.sdf_to_mesh` reversed the face order of marching cubes, which already returns outward normals for an SDF that is negative inside.
- Voxelization (ray parity) and the light tracer (sign from voxel parity) were unaffected. Vertices are identical, so existing runs still match their rebuilt models (checked).
- The demo `rock.stl` was regenerated: signed volume +0.01364 m³.

**D-704 Installer and self-check for non-programmers.**
- `install.ps1`: installs uv, creates the Python 3.12 environment, installs the package, writes `ReefFlow.bat`, runs `python -m reefflow.selfcheck` (backend detection plus a 64³ solver run with a mass check).
- `-SystemCerts` handles TLS-inspecting networks.

**D-705 Known weakness found in UI testing: automatic rock segmentation fails on some full-size views.**
- On the full-size (1600 × 1200) synthetic left view of the demo, the default GrabCut suggestion marked the sand floor as rock: mask IoU 0.51, against 0.91 for the top view and 0.94 for the front. The same view at 640 × 480 gave 0.985.
- The UI makes the failure obvious (wireframe overlay, red/blue comparison), and the brush and polygon tools correct it.
- The automatic suggestion should be treated as a starting point. *Revisit:* make the segmentation threshold resolution-aware, and test on real photos.

## Layout features after the sprint review (D-801 to D-808)

**D-801 Overflow box on any glass wall.** The user asked to place the overflow anywhere along the glass, not only in a corner.
- New project-file form: `wall` (back, front, left, right), `center` (box centre along that glass: from the left glass for back/front, from the front glass for left/right; default mid-wall), `width` along the glass, `depth` out from it.
- The Phase-2 `corner` form is still read, and gives exactly the Phase-2 footprint (same formula; test `test_corner_form_footprint_unchanged`).
- The editor converts a corner box to the wall form when it saves the project.

**D-802 Overflow validation instead of silent clamping.** A box wider than its glass, a centre that puts the box outside the tank, or a depth reaching the opposite glass is a project-file error with the allowed range in the message. A tolerance of 1 µm absorbs unit rounding. Width equal to the glass length is allowed (coast to coast). The UI clamps while dragging or typing, so it never writes an invalid box.

**D-803 Overflow modelling unchanged.** Only the footprint moves; the box is still solid from the sand to the weir crest with a mass sink above it (D-203). Check on the demo at 10 mm with the box at mid back wall (centre 24 in): the sink nodes are centred within 6 mm of 24 in, and the sink balances the return to machine precision.

**D-804 Overflow dragging in the 3-D view.** The overflow is selectable; the move gizmo shows only the axis along its glass, and the position is clamped so the box stays inside the tank.

**D-805 Several return nozzles on one pump: flow split.**
- The pump's total delivered flow Q is unchanged: it still comes from the pump curve at static head + fittings allowance (see A-801).
- Default split Q_i ∝ A_i: orifices fed from one manifold at a common pressure with equal discharge coefficients. Q = C_d A √(2Δp/ρ) gives the same exit velocity for every nozzle.
- Nozzles with an explicit `share` keep it. Nozzles without one split the remaining flow by area. If every nozzle has a share, the shares must sum to 1 within 1 % and are normalised exactly; otherwise it is an error. Shares that leave no flow for the remaining nozzles are an error too.

**D-806 Each nozzle is its own source.** Each nozzle is a mass source with U_i = Q_i/A_i and J_i = ρQ_i²/A_i, with its own equivalent outlet (minimum cells and velocity cap per nozzle, D-203), its own pump-state row and its own turbulence (OU) process. All nozzles of one pump share its mode s(t). The overflow sink removes the sum of all return flows, as before.
- A single-outlet pump builds a bit-identical model and draws the same random numbers as before (test `test_single_nozzle_model_unchanged`).

**D-807 Project-file form.** A return pump may give `nozzles:` (each with `position`, `aim`, optional `diameter` or `outlet`, optional `share`) instead of `position`/`aim`. Pump.position/aim mean "first nozzle". The editor writes the old single form when there is one nozzle without overrides, so old files round-trip unchanged. Only return pumps may have several nozzles.

**D-808 Reporting.** `model.info["pumps"]` keeps one entry per pump with the Phase-2 keys (J is the sum over nozzles), plus a `nozzles` list giving per-nozzle flow, share, velocity, exit velocity head U²/2g, momentum flux and modelled outlet. `model.pumps` (the runtime list) now has one entry per outlet.

## Rock sculpt tool after the sprint review (D-820 to D-825)

**D-820 Sculpting on a voxel density grid in the browser.** The user found the shape-based carve tools hard to use and asked for push / pull brushes with long-press-to-dig and wheel-sized brushes.
- The rockscape becomes a density grid (0 = water, 255 = rock, surface at 127.5) that the browser edits in real time.
- Mesh edits (vertex pushing) were ruled out because they cannot open caves or tunnels without changing topology; a grid can.
- The server builds the starting grid, stores drafts and turns the result into the project rockscape (`rockscape_sculpt.stl`), the same way "Use as rockscape" does for photos. The designer then holds it as one built `mesh` element.

**D-821 Brushes.**
- Pull and Push change density by up to 3 × strength full-scale per second at the brush centre, with falloff (1 − r²/R²)². They are time-based ("airbrush"): holding still keeps working. The brush re-picks the surface every frame, so a held Push drills along the view ray.
- Smooth relaxes toward the 6-neighbour mean.
- Flatten blends toward a horizontal plane at the height where the stroke started (2-cell ramp), so it both cuts and fills.
- The two node layers below the sand always copy the first layer above it, so rock stays buried in the sand as the CFD model expects.

**D-822 Display and picking.**
- Display uses naive surface nets with gradient normals, re-meshed per 16-cell chunk, and only where a brush touched.
- Picking ray-marches the density grid (0.4-cell steps, then bisection), so it always matches the current edits.
- Node check on the demo rock at 4 mm:
  - the surface-nets volume is within 0.02 % of the server's marching-cubes volume;
  - the mesh is consistently oriented, with no edge used twice in the same direction;
  - picks land on the 127.5 level to 10⁻⁵ m;
  - a brush dab plus pick costs 0.24 ms, and re-meshing after a stroke takes about 5 ms;
  - undo restores the grid exactly.

**D-823 Starting grid: 3 × 3 × 3 supersampled volume fraction.**
- Each node's density is the rock fraction of its cell, so the starting surface is not stair-stepped.
- With 2 × 2 × 2 subsamples, half-filled cells landed exactly on the 127.5 threshold and counted as rock: node volume +0.8 to +1.2 % on the demo rock, +1.9 % on a sphere. An odd 3 × 3 × 3 count has no ties.
- Demo rock, true volume above the sand 12.098 L:
  - the sculpt grid gives 12.092–12.095 L;
  - an unedited round trip (mesh → sculpt grid → STL → CFD voxels) changes the CFD rock volume by −0.02 to −0.19 % (sculpt 3/4/5 mm × CFD 3/5 mm).
- Start-up time on the demo: 1.3 s at 5 mm, 2.5 s at 4 mm, 6.4 s at 3 mm.

**D-824 Drafts and resume.** The grid is saved with the signature of the rockscape it belongs to (entries plus mesh file sizes and times). Opening the sculpt tool again resumes the saved grid losslessly only if the rockscape is unchanged and the resolution matches. Otherwise it re-voxelizes the current rockscape. Applying updates the signature, so sculpt → apply → sculpt never re-samples.

**D-825 The CFD sees the sculpt at its own cell size.** The sculpt resolution (3–6 mm) only sets how fine the user can sculpt. Applying a sculpted cave of 115 cm³ on the demo changed the CFD rock volume (5 mm cells) by the same amount as the grid: 11.978 L vs 11.978 L. The panel warns that caves narrower than about 3 CFD cells carry little or no flow.

## Sculpt tool, second round (D-826 to D-830), 2026-11-03

The user's feedback on D-820 to D-825: it was hard to push and pull the rock to make openings, and the brush tilted to odd angles on the rough surface. They asked for a brush locked to an orthogonal axis that can be turned in steps.

**D-826 The brush works along a locked direction by default.**
- The surface normal under the cursor jumps around on rough rock, and so did the old cursor ring and the drill direction (the view ray). Default mode is now **Straight in**: the tank axis (±x, ±y, ±z) nearest to the view direction. Looking through the front glass, it is front → back; from above, it is straight down.
- **Fixed** sets a direction from presets (front → back, back → front, left → right, right → left, down, up) and turns or tilts it in 15° steps (Q / E, W / S). **Follow surface** keeps the D-821 ball airbrush, with the cursor normal now averaged over the brush radius and eased between frames.

**D-827 Locked Push drills a straight hole; locked Pull extrudes.**
- The tool is a round-ended cylinder of radius R along the direction. Push sets density = min(current, profile); Pull sets max. The soft edge is max(1.5 cells, 0.2 R) each side of R. Repeating a stroke over the same place changes nothing: depth comes from hold time, not from repetition.
- Depth grows at 0.1 m/s × strength while held. The anchor is where the cursor ray meets the rock **as it was before the stroke**, read from the stroke's undo copies. Holding still therefore drills one straight line, and dragging routes a groove (or ridge) at the stroke depth along the original surface.
- Push stops by itself once it is through. "Through" means the first water stretch after rock at least max(R/2, 2 cells) long. The tip runs on to R past it, so the opening has full width. Pull stops 10 mm short of the glass and the water surface, about 2 Standard CFD cells, so the flow model still sees a gap. Push stops at the glass, sand or top.
- An optional **Click bores through** option drills the whole way through in one click.
- Before clicking, the cursor shows the direction arrow, a dashed line to the predicted exit, a ring at the exit, and a label: "through at 6.8 in", or "no way through before the sand".

Node check on the demo rock at 4 mm, R = 20 mm, strength 50 %, front → back:

| Check | Result |
|---|---|
| Predicted exit vs rock on that line (measured independently) | 176 mm vs 174.5 mm |
| Time to break through | 3.9 s |
| Hole radius at mid-depth (up / down / left / right) | 20.2 / 20.0 / 20.0 / 20.0 mm |
| Samples still rock on the hole axis | 0 |
| Removed vs ideal cylinder | 214 cm³ vs 219 cm³ |
| Extra 2 s hold after breaking through | 0 cm³ removed |
| Undo | restores the grid exactly |

Other checks:
- **Tilted 15° down:** through at 188 mm; axis clear.
- **Pull out toward the front, 1 s:** extruded 50 mm, and the front face moved 49.9 mm. A 10 s hold stopped 10.1 mm from the glass.
- **Drag groove at a 15 mm set depth:** 14.7–16.6 mm deep along the path.
- **Cost:** 0.13 ms per drill frame; the hover path preview takes 0.015 ms.
- **Browser (headless Chrome, 5 mm grid):**
  - From the front view the direction snapped to front → back.
  - Holding 4.5 s removed 211 cm³, and one click with bore-through removed 278 cm³ (R = 22 mm).
  - Straight down on the tall rock, the preview showed "through at 1.89 in", where it opens into the space under the overhang. One click removed 49 cm³.
  - Apply gave a watertight mesh with no geometry issues.

**D-828 Flatten.** Flatten levels a horizontal plane in Straight-in and Follow-surface modes, as in D-821, because flat frag shelves are its main use. With a Fixed direction it cuts a flat face square to that direction. Node check, fixed front → back: five picks across the face all hit y = 0.1585 with normal y = −1.000.

**D-829 Left-drag off the rock orbits.** Left-drag on rock or sand sculpts. Left-drag where the cursor ray misses both orbits the view, in the manner of mobile sculpting apps; right-drag still orbits. Browser check: a left-drag over open water moved the camera 1.48 m and changed the rock by 0 cm³.

**D-830 Cave and hole shading (display only).**
- Each sculpt-mesh vertex is darkened by ambient occlusion. Short rays (up to 12 cells) go out in the outward-facing grid directions, and the weighted share that hits rock or sand gives occ. Shade = (1 − 0.75 occ)^2.2; the exponent is there because vertex colours are linear.
- Through-tunnel on the demo: wall shading 0.55 (median, before the exponent) vs 0.94 on the front face.
- Chunks within 12 cells of an edit re-mesh too, so the shading has no seams. Cost on the demo at 4 mm: full mesh 224 → 306 ms; re-mesh after a stroke 5 → 32 ms, spread over frames by the existing 12 ms per-frame budget.
- The exported STL and the CFD are unaffected.

## Phone scans and phone photos (D-831 to D-835), 2026-11-03

You asked for other ways to make rock modelling easier. Two were built: clean-up of phone 3-D scans of dry rock (new setups), and a phone photo page for tanks that are already filled. Neither changes the physics: both only produce the rock geometry that the existing pipeline voxelizes.

**D-831 Scan import: units, up axis, ground (reefflow/scan.py).**
- Formats: OBJ, GLB, glTF (embedded only), PLY, STL, or a ZIP holding one (glb > obj > ply > stl is preferred). USDZ is refused with a message to export OBJ/GLB. Point clouds are refused.
- Unit guess from the largest dimension: ≤ 3 → m, ≤ 30 → cm, else mm (phone apps write metres). The user can override it and can scale to one tape measurement.
- Up axis: for each of the six axes, look for the largest flat, horizontal area (normals within 15° of the axis, either winding) at the bottom of the scan: ≥ 4 % of the scanned area with ≤ 3 % of the area below it. This is the table. If none is found, use the format convention: +Y for GLB/glTF/OBJ, +Z for STL/PLY. The user can always pick an axis.
- Ground cut: table level + max(3 mm, 3σ of the table heights, bin width). Faces wholly below are dropped; the base is cut flat there.

**D-832 Inside / outside without a closed mesh.** Ray parity (geometry.voxelize_mesh) needs a closed mesh, and scans are open at the bottom and have holes. Instead:
- The surface is sampled at < dx/2.5 spacing into surface voxels. Each holds the mean point and the summed face normal.
- Every voxel casts 26 rays along the grid directions (row sweeps, no per-ray loops). A ray that first meets a surface from behind (normal·ray > 0.15) votes inside. A ray that meets it from the front, or leaves the grid, votes outside. Grazing hits and exits through the cut base do not vote. Inside wins on a strict majority.
- The scan's normals must point outward. If the area-weighted Σ (c − centre)·n < 0 above the cut, they are flipped first (reported as "inside-out").
- Near the surface (2-voxel band) the signed distance is the point-to-plane distance to the nearest surface voxel, where the voxel lies along that normal (|d·n| ≥ 0.5 |d|). Elsewhere it is the vote sign times the Euclidean distance. This gives a sub-voxel surface, and patches holes between the rim points.

**D-833 Clean-up.**
- Solid pieces under 5 % of the largest are dropped (option: keep the largest only). Pieces of 1 cm³ or more are counted for the report.
- Sealed voids (water regions not connected to the sides or top) are filled; water cannot reach them.
- The result is intersected with z ≥ cut, and the layer under the base is set to +dx/2, so the base is exactly flat at the cut. Marching cubes then closes the mesh.
- Output: base on z = 0, centred on x = y = 0, saved as an STL in mm under meshes/. It is added as an ordinary mesh element. The raw upload is kept in projects/<pid>/scans/.
- Auto detail: dx = clip(extent / 150, 1.5 mm, 4 mm). The grid is capped at 7 M voxels (dx grows 15 % per step until it fits).

**D-834 Scan clean-up checks (synthetic scans, tests/test_scan.py).** The synthetic scan is a ~26 cm noisy rock with a 50 mm tunnel, standing on a 60 cm table. Like a phone scan, its bottom is open, it has 2 holes of 36 mm, a 24 mm loose piece, 0.3 mm vertex noise and a glTF Y-up frame.

| Case | Up found | Volume / truth | IoU | Tunnel open | Loose piece removed |
|---|---|---|---|---|---|
| GLB, Y up | +y | 0.995 | 0.989 | yes | yes |
| Inside-out normals | +y (flipped) | 0.995 | 0.989 | yes | yes |
| STL, Z up | +z | 0.995 | 0.989 | yes | yes |
| OBJ but Z up (convention wrong) | +z | 0.995 | 0.989 | yes | yes |
| No table (open bottom only) | +y | 1.001 | 0.988 | yes | yes |
| 6 holes | +y | 0.994 | 0.988 | yes | yes |
| Two 70 mm holes | +y | 0.989 | 0.983 | yes | yes |
| One 100 mm hole | +y | 0.982 | 0.976 | yes | yes |
| 1 mm vertex noise | +y | 0.998 | 0.966 | yes | yes |
| 2 mm detail | +y | 0.996 | 0.990 | yes | yes |

- Truth is the rock above the same cut, on a 1 mm grid. The cut was found at 4.1 mm above the lowest point; the table is at 0 with a 3 mm clearance.
- Every result is watertight. Time is about 1.0 s at the auto 2.6 mm detail and 1.6 s at 2 mm. A 0.8 mm-sampled scan (about 1 M triangles) takes 5.4 s.
- A cm file at scale × 2 gives 8.0× the volume (within 3 %).
- In the browser, a 49 k-triangle GLB cleaned in 1.07 s to 4.68 L. Building the rockscape from it gave 4.65 L, watertight, with 0 geometry issues.
- Limits:
  - This is checked on synthetic scans only; no real phone scan was available offline.
  - Holes are closed by the votes. A hole about the size of the rock's own cross-section would leave a dent.
  - Two flat levels (a table on a floor) give the floor; the user then raises the cut.

**D-835 Phone photo link (reefflow/ui/phone.py, static/phone.html).**
- The main UI stays on 127.0.0.1. "Use my phone" starts a second uvicorn server in a thread on 0.0.0.0, on a free port from 8770.
- Every path is under /p/<token>, a 128-bit random token compared in constant time. The QR code is made by segno, BSD-3-Clause.
- Requests from non-private, non-loopback addresses get 403. The link is bound to one project. It can add photos and delete only the photos it sent itself; a retake replaces that card's photo.
- It stops after 60 min without a request, on "Stop the link", or on app shutdown. Photos are capped at 40 MB.
- Plain HTTP: `<input type=file capture>` reaches the camera without HTTPS, and a self-signed certificate would only show a warning page on the phone.
- The phone page sends camera JPEG/PNG unchanged, so EXIF focal length is kept. HEIC and files over 25 MB are re-encoded to JPEG at up to 4000 px, which loses the EXIF.
- Shot list: front and top (required); left, right and a second, higher front shot (recommended); back (optional). The higher front shot is stored as front-2, which earlier improved IoU from 0.679 to 0.705 (Phase 6). Tips differ for a filled and a dry tank.
- Checks (tests/test_phone.py and headless Chrome):
  - upload → appears in the Photos session (front, front-2);
  - retake replaces;
  - deleting a desktop photo from the phone gives 403;
  - wrong token gives 404;
  - unknown shot or bad image gives 400;
  - after Stop the port refuses connections;
  - the phone page fits 390 px with no sideways scroll;
  - the desktop dialog listed the phone photo within 2 s.
- Not tested: a real phone on a real Wi-Fi, and the Windows firewall prompt. Tests bind 127.0.0.1 (REEFFLOW_PHONE_HOST) so that no firewall prompt is raised.

## Photo calibration and rock mask, round 2 (D-836 to D-839), 2026-11-03

You reported that calibration was not intuitive and did not fit a rimmed tank whose black base frame hides most of the sand bed, and that the suggested mask was not close. Diagnosis on your own front photo (iPhone, 4032 × 3024 reduced to 2000 × 1500, project "Demo 75-gallon reef"):

- **The inside corners the tool asked for are hidden on a rimmed tank.** The rim (inside top) corners are under the top frame. The bottom corners and the front sand line are behind the base frame. The clicks therefore landed on the frame, which is up to 18 mm outside the model points. The fit error was 34.9 px RMS, the worst click 49 px.
- **The tank height in the project does not match the photo.** The project still has the demo size, 48 × 18 × 21 in outside. Your photo fits a tank 22.8 ± 0.2 in tall (D-837). With 21 in, no set of clicks fits better than about 22 px.
- **The mask suggestion used colour.** Under blue light, the empty water at both ends of the tank differs from the thin strip of water the colour model learns from, so it was labelled rock. The 35 px calibration error also shifted the regions.
- **No focal length.** iOS Safari strips most EXIF from photos uploaded through the browser, so the focal length is unknown. It was the same for all five photos.

None of this changes the physics. Calibration and the mask only shape the rock geometry that the existing pipeline voxelizes.

**D-836 Outside corners and corner seams (reefflow/photo/tankref.py, calibration.py).**
- New named points:
  - `top_<corner>` at z = H and `base_<corner>` at z = −g, both at the outside corners (x ∈ {−g, L+g}, y ∈ {−g, W+g}). These form exactly the box of the external size the user enters, so they match the entered dimensions by construction. On a rimmed tank they are the outside corners of the frame; on a rimless tank, the outside corners of the glass.
  - `seam_<corner>` and `seam_<corner>_2`: a click anywhere on the inside vertical corner line. Each is one line constraint (distance to the projected line), not a point.
- Default list for a side view:
  - the two near top corners and the two near base corners;
  - the two far corners at the water line;
  - the two far seams.
- Default list for the top view: the four outside top corners and the four sand corners.
- The old inside-corner list is still available (`recommended_points(kind, scheme="inner")`), and saved clicks keep working.
- The edge distance is computed exactly enough for a zero-noise check:
  - the line is sampled at 41 points, then again at 41 points around the nearest sample;
  - exact synthetic clicks fit to < 1e-5 px.
- The focal-length start search now scores the seams as well. Four coplanar corners fit any focal length exactly; the seams, at another depth, pick the right one.
- The coplanarity refusal counts both ends of each seam.

**D-837 Tank-size check (calibration.check_tank_size).**
- When a fit's RMS exceeds 4 px, the camera is refitted together with the tank's height and width relative to its length. A photo has no absolute scale, so the length is held fixed.
  - The water keeps its distance below the rim, and the sand keeps its depth.
  - It costs 0.2 s.
- The UI shows a red box with the external size that fits better when both of these hold:
  - the RMS falls by more than 35 %;
  - a dimension moves by more than 2 % and by more than 3σ.
- The UI asks the user to measure and correct the size on the Tank tab. Nothing is changed automatically.
- On your photo:

| Tank height in the fit | Fit RMS (8 clicks) | Size check |
|---|---|---|
| 21 in (project) | 24.1 px | height 22.81 ± 0.21 in fits better (5.5 px); width 20.0 ± 2.8 in: not flagged (within 3σ) |
| 23 in | 6.1 px | no change suggested |

- Synthetic check (tests/test_calib_round2.py): a 23-in tank calibrated as 21 in, with 1 px click noise.
  - The fit RMS is above 4 px.
  - The check reports 23 ± 0.3 in and an RMS under 2 px.
  - With the right size it suggests nothing.

**D-838 Focal-length prior when EXIF is missing.**
- In "solve from the clicks" mode without EXIF, a weak prior is added at a phone main camera: 26 mm equivalent, σ = 25 % in log f. It is one extra residual, (ln f − ln f0)/σ, in px units, so 1σ costs as much as a 1 px click error.
- It keeps the four-corners-plus-seams case from running away: one prototype without water points went to f = 49,000 px. It barely moves a well-determined fit:
  - a prior 30 % off moved an 8-click synthetic fit by < 1 %;
  - on your photo, f went from 1564 to 1564 px.
- The free fit on your photo gives 1564–1606 px, against 1502 px for 26 mm, which supports the default.

**D-839 Texture cue for the rock-mask suggestion (reefflow/photo/segment.py).**
- Side views now start GrabCut from roughness instead of colour:
  1. Texture energy: the Gaussian-averaged gradient magnitude (σ = 6 px per 2000 px) divided by the local brightness + 8.
  2. The projected vertical tank corners are zeroed, 24 px wide (glass edges glow).
  3. The threshold is Otsu over the tank pixels.
  4. On pixels whose ray ends on the sand (sand is rough too), a seed must also differ in colour from the water and sand samples by ΔE > 14.
  5. Seeds are opened with a 15 px disk, and only blobs that reach the sand (within 3 cm, along the ray) are kept.
  6. GrabCut runs on a (fine roughness at 2.5 px, grey level, colour distance) image. Pixels whose ray misses the rock-able box are definite background.
  7. Only result blobs that touch the sand are kept.
  8. The outline is snapped with a colour GrabCut restricted to ±8 px around it (the blurred roughness puts the edge a few px outside).
  9. A 4 px opening, specks under 0.067 % of the photo removed, holes under 0.1 % filled.
- The top view keeps the colour cue. The UI has a "Find rock by" choice (auto / texture / colour).
- Synthetic renders (held-out seed 7 and seed 11), mask IoU against the true silhouette:

| View | Colour, guided by top | Texture, guided | Colour, unguided | Texture, unguided |
|---|---|---|---|---|
| front, seed 7 | 0.993 | 0.997 | 0.993 | 0.993 |
| left, seed 7 | 0.997 | 0.998 | **0.507** | 0.997 |
| front, seed 11 | 0.997 | 0.997 | 0.994 | 0.994 |
| left, seed 11 | 0.997 | 0.998 | **0.506** | 0.997 |

- Without the edge snap, texture alone scored 0.92–0.99 (a halo of a few px, and a strip along the back sand line). With the snap it matches or beats colour in every case.
- A rough 46 × 36 px "powerhead" painted high on the back glass is excluded (< 2 % of it in the mask), and the rock IoU stays ≥ 0.97.
- On your photo, with the 23 in height and the new clicks (figures in the scratch run, not committed):
  - the mask follows the rock outline with the cave open;
  - the clip on the left glass, the powerhead, overflow and heater at the right, and the empty water at both ends are excluded;
  - 12 s per photo.
  - With the 21 in height the base of the rock is cut off: the wrong size shifts the sand line.
- Limits:
  - Equipment that rests on the sand, and corals, stay in the mask.
  - Rock smoother than the background (bare dry rock in front of a textured wall) would need the colour cue.
  - Checked on two synthetic seeds and one real photo only.

**Photo validation re-run with the new defaults** (`python -m reefflow.photo.validation`, 1099 s, verification/results/10_photo). The pass/fail outcome is unchanged: C1, V2, C2, V3, V4, V6 and V7 pass; V1 still fails, as before.

| Criterion | Before (inside corners, colour) | After (outside corners + seams, texture) |
|---|---|---|
| C2 mean click RMS, 2 px noise | 2.13 px | 1.95 px |
| C2 camera centre error, top / front / left | 10.8 / 27.8 / 5.7 mm | 10.6 / 33.0 / 4.7 mm |
| C2 mean focal error | 0.88 % | 0.97 % |
| V3 mask IoU, top / front / left | 0.988 / 0.997 / 0.998 | 0.988 / 0.997 / 0.998 |
| V3 voxel IoU, volume error | 0.678, +47.4 % | 0.680, +47.1 % |
| V3 segmentation time, 3 views | 10.8 s | 14.5 s |
| V1 (exact masks; not affected) | 0.679 | 0.679 |

- The front camera is a little less well placed along its view axis (+5 mm), because a seam click is one equation and a corner click two.
- In exchange, every default point can be seen on a rimmed tank.

## Calibration and mask, round 3 (D-840, D-841), 2026-11-07

Your feedback on round 2:

1. "Tank size looks wrong" even after you corrected it on the Tank tab.
2. It was unclear whether to click the frame or the glass, and there was no way to say a tank is rimless, or rimless only at the top.
3. You asked for a Photoshop-style sensitivity, because the mask stopped early at the sand.

**Cause of 1.** The Photos tab calibrates with the *saved* project. Your Tank tab showed 48 × 18 × 23 in, unsaved. On disk the project still said 48 × 18 × 21 in, and the warning said "not the 21.0 in entered". The same stale size caused most of the early cut-off in 3: with 21 in, the lowest mask row was 987 px; with 23 in it was 1044 px, reaching the sand. The photo was New_tank/front, iPhone, EXIF 26 mm.

**D-840 Sensitivity (segment.sensitivity_params, adjust_outline, finish_texture_mask).** The slider runs 0–100, with 50 as the default.
- Things tried first, on your photo:
  - lowering the global roughness threshold: ×0.7 flooded the tank (mask 22.7 % of the photo, against 14.0 %);
  - growing the GrabCut seeds: sensitivity 80 flooded too (26.4 %), because the window blinds behind the back glass are textured as well;
  - changing seeds at all also changed the base GrabCut result unpredictably: at 25 the arches filled in.
- Shipped: the base suggestion is the same for every setting. The slider only moves the outline, like "expand / contract" with an edge test.
  - Above 50: pixels within up to 40 px of the outline (per 2000 px of photo, at 100) are added if they look like rock. That means roughness above Otsu × 1.25^k, or a colour difference from the water and sand samples above ΔE 14 × 1.5^k, with k = (50 − s)/50. Only parts connected to the outline count.
  - Below 50: pixels within up to 15 px of the edge are dropped unless they are rough.
- The base mask and its texture maps are cached per photo in the server, so a slider move takes about 0.2 s, against about 14 s for a new suggestion. After a restart, the slider simply suggests again.
- On your photo (23 in): sensitivity 0 / 25 / 50 / 75 / 100 gives 12.7 / 13.2 / 14.0 / 15.4 / 16.3 % of the photo. Each setting contains the one below it: 0 px lost between steps.
- Moving the slider replaces brush edits, so it asks first.
- Test: tests/test_ui_photos.py::test_sensitivity_style_and_size_ok checks the API returns nested masks and that IoU at 50 is ≥ 0.97 on the synthetic front view.

**D-841 Tank style, unsaved size, re-solve, size confirmation.**
- **Tank style:**
  - `tank.style` in the project file: rimmed, rimless_top (rimless top with a base frame) or rimless. It can be set on the Tank tab or in the Photos panel.
  - It changes only what the calibration asks for: "top of the frame" or "top of the glass", and "bottom of the base frame where it sits on the stand" or "bottom of the glass". The sketch draws the frame bands to match.
  - The geometry is unchanged: both are the outside corners of the entered external size.
- **Frame overhang.** Modelling the frame sticking out past the glass by ¼–½ in made your fit *worse*:

| Glass, overhang, depth (outside) | Fit error, EXIF focal length |
|---|---|
| ½ in glass, overhang 0 / ¼ / ½ in, 18 in deep | 8.5 / 13.2 / 18.0 px |
| ⅜ in glass, overhang 0, 16 / 17 / 18 in deep | 4.5 / 3.2 / 7.2 px |

  So it is not modelled. The simulated water volume is unchanged.
- **Base corners.** They now use the bottom glass thickness (z = −bottom glass), not the side glass.
- **Unsaved size.** The Photos panel shows a red "Unsaved tank size" box when the Tank tab's external size differs from the saved one by more than 0.5 mm, with a "Save project now" button.
- **Re-solve after a size change.** When the saved size changes, every photo with four or more corner clicks is calibrated again at once with the stored clicks. Before, the calibrations were only dropped.
- **Size confirmation.**
  - "My measurements are right: stop asking" skips the size check for that photo until the tank size changes again.
  - Why it is needed: one front photo pins the height well (23.1 ± 0.1 in on your new photo, matching your 23 in) but the depth only roughly. Your new photo prefers about 16–17 in outside over 18 in (fit error 3.7–4.2 against 8.5 px with EXIF focal length). With ⅜ in instead of ½ in glass, 17 in fits at 3.2 px.
  - The warning now says the depth estimate is rough and the tape wins.
- **Focal prior.** In "solve from the clicks" mode, a photo with EXIF uses its own focal length as the centre of the weak prior, instead of 26 mm.
- Top photo guide: your New_tank top photo (EXIF 14 mm, lights in place) had 4 of its 8 points hidden behind the light fixtures. Its fit error was 25.7 px, and it was guiding the front mask. Top photos with RMS > 6 px no longer guide the side masks; the side mask notes say so.

## Rock mask, round 4 (D-842), 2026-11-07

**D-842 Bubbles, a bounded outline, rubble on the sand, open arches (segment.texture_energy, texture_init, adjust_outline).** On your new blue-lit photo (IMG_9531) the suggestion took in the dark water behind the left rocks and most of the sand, at every sensitivity (15.6 % of the photo).
- **Cause 1: micro-bubbles.** The water is full of bright 2–3 px specks. Roughness is divided by brightness, so in the dark water behind the rock they scored as rough as rock, and the rough seed areas spread into the water.
  - Fix: the roughness is measured on a median-filtered image (5 px per 2000 px). Rock texture is coarser and survives it.
  - The filter is only used when its kernel is ≥ 5 px, i.e. photos of about 1800 px and up. The app always stores 2000 px. On the 640 px synthetic test render a 3 px median removed the fine rock texture (front IoU 0.99 → 0.74).
- **Cause 2: the outline step could reach the whole tank.** GrabCut treated every pixel in the rock-able region as "probable background" that could switch, and its colour model took the dark water and the bright sand as rock.
  - Fix: GrabCut may now only move the outline within 20 px (per 2000 px) of the seeds, or inside areas they enclose once gaps up to 60 px are closed. Everything else is definite background.
  - The enclosed-area part matters: smooth rock faces ringed by rough edges were otherwise lost (synthetic front 0.66 before it was added).
- **Rubble on the sand.** Where the ray would end on the sand, a pixel needed a colour difference above ΔE 14. Under blue light, rubble and sand differ by about ΔE 12–14, so the rubble was lost.
  - Measured on your photo: rubble L* 108–118 against sand 138–152; roughness 0.96–1.01 against 0.75–0.90 × the threshold.
  - New rule: on the sand, a pixel is rock if it is rough (> 0.8 × the threshold) **and** either clearly darker than the sand around it (L* below 0.85 × the 80th percentile within ±150 px of columns) or different in colour (ΔE > 14).
- **Sensitivity on the sand and at arches** (D-840 update):
  - On the sand, the slider now loosens the darkness test (0.80 at 0 to 0.90 at 100) and the roughness. The colour test there stays at ΔE 14, because lower lets the sand in.
  - Openings in the suggestion bigger than the hole-fill size (0.1 % of the photo) are never grown over. At 75 and above, growth from all sides used to close the arches.
- **Results on real photos** (mask % of the photo at sensitivity 0 / 25 / 50 / 75 / 100; before at 50):

| Photo | Now | Before (at 50) |
|---|---|---|
| Your new photo (blue light, bubbles) | 10.9 / 11.4 / 11.9 / 12.8 / 13.2 | 15.6 (water and sand included) |
| New_tank front | 9.3 / 9.9 / 10.2 / 11.1 / 11.6 | 11.1 |
| Demo front (old 21 in calibration) | 10.9 / 11.2 / 11.4 / 12.2 / 12.5 | 18.5 |

  Visual check on your photo: the water behind the left rocks is excluded, both arches stay open up to 100, and the rubble line is included down to the sand. Not taken: the two soft corals at the far left (brush them in if you want them in the model).
- **Synthetic tests** (640 px, below the speck-filter size): front 0.992 / left 0.933, before 0.992 / 0.921. Photo validation at 1600 px, where the filter is used: same pass/fail as before (V1 still fails at 0.679); V3 mask IoU 0.988 / 0.997 / 0.997 against 0.988 / 0.997 / 0.998.
- **Tests** in tests/test_calib_round2.py:
  - the speck filter makes bubbly water much smoother than rock and keeps the rock texture;
  - rubble darker than the sand is found and plain sand is not;
  - the candidate region outside the band is definite background;
  - an arch stays open at sensitivity 100.

## Sculpt brushes, third round (D-843), 2026-11-11

**D-843 Raise / Lower as a terrain brush (static/js/sculpt.js `shift()`).** You said the brush barely changed the rock and asked for a Planet Coaster-style push and pull.
- **Measured on the photo reconstruction of your tank** (Node, the browser code itself):
  - **Default "Straight in" Push / Pull** was a drill: it extruded or dug a 25 mm tube along the view axis. It took the min / max with what was there, so going back over the same spot changed nothing.
  - **"Follow surface"** (the ball airbrush): a 3 s scrub across a 30 cm patch moved the surface **0.2 mm** on average (max 3 mm). Held still, it grew a narrow spike at about 32 mm/s, because it re-centred on its own growing tip every frame.
- **New Raise / Lower:**
  - **Footprint:** a round column of radius R around the line of sight through the cursor. Weight w = (1 − r²/R²)², with r the distance from the line.
  - **Motion:** every rock surface in the column that faces you moves along the line of sight by speed × strength × w per second. Raise moves it toward you, Lower away.
  - **Method:** semi-Lagrangian. The new density at x is the old one at x ± step·(view direction), read from a copy. Raise keeps the max with the old value and Lower the min, so Raise never removes rock and Lower never adds any, and surfaces facing away never move.
  - **Column depth:** while a stroke lasts, the column reaches R plus the distance already moved, so a held brush grows a dome as a whole.
  - **Sand:** for Raise the sand counts as solid, so raising on the sand grows rock out of it.
  - **Glass and water surface:** Raise adds nothing within 10 mm of the glass or the water surface, the same gap as the old Pull.
  - **Settings:** speed 80 mm/s at 100 % (40 mm/s at the default 50 %). The default radius is now 50 mm (was 25 mm); the maximum is 200 mm (was 150 mm).
- **Tunnel:** the old straight-line drill is now its own tool (5). Ctrl turns it into Extrude (the old Pull arm), unchanged. The "Follow surface" mode is removed; the brush direction setting now applies only to Tunnel and Flatten.
- **Results:**

| Check | Old | New |
|---|---|---|
| Hold still 1 s, 50 %, R 50 mm, centre | spike, 32 mm/s (R 25 mm) | 29.3 mm at 60 mm/s; final 80 mm/s: 40.7 mm in the browser |
| Shape after holding (points at half radius) | narrow spike | 0.56 of the centre (design 0.5625) |
| One / two scrub passes over the same line | no build-up (drill) / ~0 (airbrush) | 17.6 / 35.3 mm (exactly double) |
| Raise: rock removed; rock within 10 mm of the glass | – | 0 nodes; 0 nodes |
| Lower: rock added | – | 0 nodes |
| Undo after two strokes | – | 0 nodes differ |
| Time per frame, R 50 mm / 100 mm | – | 3 ms / 13 ms (Node, real rock) |

- Repeated small shifts with linear interpolation soften fine texture where you raise or lower a lot: roughly a few millimetres after moving the surface 30 mm. This is fine for the flow model's 3–10 mm cells.
- **Tests:** tests/test_sculpt_brush.py runs tests/js/sculpt_brush_check.mjs in Node on a box of rock: speed, dome shape, passes add up, no wrong-way change, glass gap, raising on sand, Tunnel bores through, undo. It is skipped when Node.js is not installed. Node is not needed to run ReefFlow.

## Real equipment catalogue (D-844 to D-846), 2026-11-12

**D-844 Products are library entries with real specs and their sources.** You asked for real lights, pumps, powerheads and overflows.
- **Files:**
  - `reefflow/data/pump_products.yaml`: 35 powerheads and 41 return pumps.
  - `reefflow/light/data/fixture_products.yaml`: 30 light fixtures (Nicrew HyperReef 100 Gen 1 and Gen 2 added on request).
  - `reefflow/data/overflow_products.yaml`: 21 overflows and 11 drain-capacity rows.
- **How they load:** they merge into the existing archetype libraries, so a project says `archetype: ecotech_mp40mqd` and the solver, the light tracer and old project files are unchanged. The generic archetypes stay, listed first as "Generic (typical values, not a product)".
- **What each entry carries:** brand, model, status (current / discontinued / unknown), the date checked, source URLs, and a `*_basis` for every modelled number saying where it came from: manufacturer, retailer, read from a chart, measured from a drawing, fitted, or assumed.
- **Research:** done 2026-11-12. The raw research and the build scripts are in `tools/catalog/`; the scripts regenerate the three files byte for byte.
- **Overflow entries:** an overflow product sets the box width and depth and a rated capacity (`overflow.product`, `overflow.capacity`). Hang-on-back boxes set the width only: the published depth includes the part behind the glass. Red Sea Reefer entries set the capacity only, because no box size is published.
- **Left out:**
  - Glass-Holes (company gone, no sizes);
  - Reef Brite XHO (no wattage);
  - Nicrew lights other than the HyperReef 100 (not researched; added on request only);
  - Jebao cross-flow pumps (contradictory specs);
  - Waterbox and IM NUVO overflows (no box size or rating).
- **Light photon output.** No maker publishes total photon output (PPF), so every catalogue PPF is derived:

| Fixture | How | µmol/J | Readings matched |
|---|---|---|---|
| EcoTech Radion XR30 G6 Pro | Reef Builders in-air grid, 7 readings, published 125° beam | 1.14 | 11 % rms (max 16 %) |
| AI Blade Coral Grow 21" | Reef Builders in-air grid, 7 readings, published 95° beam | 1.76 | 5 % rms (max 10 %) |
| Philips CoralCare Gen2 | Reef Builders in-air grid, beam and output fitted together → 103° | 1.34 | 3 % rms (max 4 %) |
| Orphek Atlantik iCon | DaniReef in-air centre PAR at 17 / 37 / 57 cm, beam fitted → 60° (with the standard 120° lens the fit needs 2.67 µmol/J at 19 % rms, which is not plausible) | 1.31 | 4 % rms (max 5 %) |

  - **Families:** the XR15 / XR30 Blue use the XR30 Pro's efficacy, Blade 48" Grow and Glow use the Blade 21"'s, and the iCon Compact uses the iCon's.
  - **The rest** (AI Prime / Hydra, Kessil, Neptune SKY, Nicrew HyperReef, Orphek OR3, Red Sea ReefLED) use power × 1.41 µmol/J, the mean of the three independent fits (A-844). Red Sea's own figures (500 PAR at the surface) are shown but not used: they give no settings, and the G1 90 and the G2 115 share the same figure at different wattages.
- **The fit model** (`tools/catalog/fit_par.py`) is the tracer's own emitter model: a uniform rectangle of the housing size with a cos^m intensity, so the fitted numbers carry straight into the simulation.

**D-845 Return pumps use the maker's head-flow curve; DC pumps without one use a straight line.**
- **Published curves:** 27 of the 41 returns have one: a manufacturer table, or read from the maker's chart. Q(H) is interpolated linearly between the points, closed at (0, Q0) and (H0, 0).
- **How wrong the generic shape was:** compared with the published curves, Q0·√(1 − H/H0) is off by:

| Pumps | √ shape: mean error, bias | Straight line Q0 (1 − H/H0): mean error, bias |
|---|---|---|
| 14 DC pumps (Vectra, COR-20, VarioS, Abyzz, Syncra SDC, ReefRun) | 17.8 % of Q0, reads 17.4 % high | 7.7 %, −0.4 % |
| 8 AC pumps (Syncra Silent, Danner) | 7.0 %, +7.0 % | 10.9 %, −10.9 % |

  Worst single pump on the √ shape: VarioS-8, 38 % of Q0 too high at mid head.
- **Products without a curve:** the 14 catalogue returns without a published curve are all DC pumps, so they use the straight line (`pump_curve.shape: linear`).
- **The generic `return_nozzle` archetype keeps √**, so existing projects (the demo and yours) give exactly the same flow as before. Changing it is your call; on the demo (5 ft head, 10 ft shut-off) it would lower the return from 566 to 400 GPH.
- **Typing over the rated flow:** if you type a rated flow different from a product's Q0, the published curve no longer applies and the generic √ shape is used. The flow basis says so.

**D-846 Two plumbing checks (warnings only; the flow is not changed).**
- **Overflow capacity:** a warning when the total return flow is above the overflow's rated capacity. Typed or from the product; the default is no capacity and no check. The Pumps panel lists drain capacities by pipe size from three sources that disagree by up to 2×: a measured test (1 in gravity drain 350 GPH, siphon 957 GPH), Durso's FAQ and Simplicity Aquatics.
- **Nozzle exit head:** a warning when a return nozzle's exit velocity head U²/2g is above the pump's whole fittings allowance. The curve flow assumes the exit head is inside that allowance (A-801), so the real flow would be lower. Example: the Sicce SDC 7 through one 3/4 in nozzle gives 1,502 GPH at 5.6 m/s, an exit head of 5.2 ft against a 1 ft allowance. The demo return (2.1 m/s, 0.74 ft) passes.
- **Panel refresh:** the Pumps panel now redraws after each validation (not while you type in it), so derived flows are current.
- **Tests:** tests/test_catalog.py, 159 tests:
  - every product builds into the same pump / fixture / overflow objects the simulation uses;
  - published curves are reproduced at their points and never rise with head;
  - the straight-line shape is used for DC pumps without a curve;
  - both warnings fire only when they should;
  - editor round trip;
  - library API.

## Look of the first release (D-847), 2026-11-21

**D-847 Light working panel, deep-water 3D view.** You chose this look and the small layout touches. Display only: no physics, project-file or API change.
- **Theme:** all colours are tokens at the top of `static/css/app.css`. Controls are styled; checkboxes and sliders use the accent colour, and file pickers match the other buttons. Number inputs use tabular figures.
- **3D view:**
  - the WebGL canvas is now transparent over a CSS deep-water gradient;
  - tank edges, glass, waterline, overflow box, pump bodies and axes are lightened to read on it;
  - 3D labels are light text with a dark halo; dark label colours passed by older code are lightened automatically, keeping their hue;
  - the legend and readout are dark glass cards.
- **Step tabs:**
  - each tab has a number badge;
  - a green tick marks the steps that are done: tank valid, rock built and up to date, at least one pump, a run, an analysed run;
  - Photos and Compare are marked optional;
  - run progress shows next to Run.
- **Foldable sections:** click any section heading in the left panel to fold it. The choice is remembered per tab, in the browser only.
- **Welcome / projects dialog:** logo and one-line description; projects newest first, with run count and "saved … ago"; two cards to start from the demo or a blank tank; import folded away.
- **Logo:** a new mark (water lines over coral) in the header, the welcome dialog and the browser tab.
- **Checks:**
  - headless-Chrome screenshots of every tab, the welcome dialog, sculpt, Photos 3D result and the project check, before and after;
  - folding hides 9 rows on the Tank tab and survives a re-render;
  - the Run badge shows "42 %" for a running job;
  - the 48 UI, phone, sculpt and scan tests pass.

## Lid, particle animation, demo aquascape (D-848 to D-850), 2026-11-21

**D-848 A lid over the tank (light tracer).** You asked to see how a lid affects PAR. The new `lighting.lid` section is open by default.
- **Default unchanged:** with no lid (or `type: open`) the tracer draws no extra random numbers and gives bit-identical results to before (test).
- **Geometry:** a horizontal sheet over the tank opening at `height` above the water surface (default: on the rim). Photons from the lights cross it on the way down.
- **Glass and acrylic:**
  - an incoherent slab: Fresnel reflection R(θ) at both faces, all internal reflections, and absorption along the refracted path;
  - T = (1 − R)² τ / (1 − R² τ²), with τ = exp(−α t / cos θ_t);
  - the lateral offset of the ray inside the sheet (a few mm) is ignored;
  - α comes from typical visible transmittance of 6 mm sheet (A-848).
- **Screens:**
  - a square mesh of round strands (diameter d, hole size a, pitch p = a + d);
  - per axis, the open fraction is (p − d·√(1 + tan²) − e·|tan|) / p, where tan is the slope of the ray along that axis;
  - d·√(1 + tan²) is the shadow of a round strand; e is extra wall depth, used only for egg-crate grids;
  - light that hits a strand goes on down with probability `scatter` (A-848), re-emitted Lambertian downward; otherwise it is lost.
- **Extra loss:** `dirt` (salt creep, condensation) multiplies the transmission by (1 − dirt).
- **Energy balance:** light lost at the lid is tallied as `absorbed_lid` and closes to 10⁻⁶.
- **Presets:**

| Preset | Straight down | 45° | PAR at 20 cm depth vs open top (puck 20 cm up, lid 25 mm above water) |
|---|---|---|---|
| Glass 6 mm standard float | 88.1 % | 85.9 % | 86.6 % |
| Glass 6 mm low-iron | 90.9 % | – | 89.5 % |
| Acrylic 6 mm clear cast | 91.6 % | – | 90.3 % |
| Aluminium window screen 18 × 16 | 64 % open | – | 67.9 % (bright strands scatter some light on down) |
| Black fibreglass window screen 18 × 16 | 59 % open | – | 55.3 % |
| Black netting 1/8 in | 71 % open | – | 67.7 % |
| Black netting 1/4 in | 79 % open | – | 76.6 % |
| Clear / white netting 1/8 in | 71 % open | – | 75.3 % |
| White egg-crate grid (1/2 in cells, 3/8 in deep) | 83 % open | – | 55.6 % (deep cells shade slanted light) |

- **A model I dropped:** a first version treated strands as walls two strand-widths deep. It cut a black window screen to 33 % of open, below its 59 % open area, which is not credible for round strands. It was replaced by the cylinder shadow above.
- **Tests:** tests/test_lid.py, 15 tests:
  - presets match published transmittance (88 / 91 / 92 %);
  - the tracer matches the analytic transmission within 0.2 % for glass, acrylic, two screens and egg crate, straight down and at 30°;
  - an open lid traces bit-identically to no lid;
  - the PAR effect ordering;
  - validation (lid below the lights, positive hole size).
- **UI:** Pumps & lights → **Lid over the tank**:
  - preset, height, hole size and strand width for screens, sheet thickness for glass and acrylic, extra loss;
  - the transmission straight down and at 45°, and the open area;
  - the lid drawn in 3D: glass tinted, screens as a mesh.

**D-849 Particle animation (Results).** Tracer particles carried by the run's time-averaged velocity field.
- **Data:** `GET /api/runs/{pid}/{rid}/flowfield` sends `mean_u`, every `stride`-th node so at most 600,000 nodes go out (about 7 MB), NaN in rock.
- **Integration (in the browser):** trilinear velocity; midpoint rule, sub-stepped so no step moves more than half a cell; corner weights renormalised over water nodes; a particle with less than a quarter of its weight in water counts as in rock and respawns.
- **Display:** particles live 4–12 s (random) and fade in and out. Trails are coloured by speed (blue slow, cyan, pale yellow at the 95th-percentile speed and up).
- **Options:** release everywhere or at the pump outlets; 1,000–12,000 particles; ¼× to 4× real time; trail length.
- **Speed:** 2.0 ms per frame for 3,000 particles and 6.0 ms for 12,000 (headless Chrome, standard demo run, 121 × 44 × 48 nodes at 10 mm).
- **What it shows:** particles follow the average flow only; the turbulent swirl on top of it is not added, and the panel says so.
- **Test:** tests/test_particles.py (field, NaN in rock, stride and origin).

**D-850 Demo aquascape in the UI template; broken-rock options.**
- **The new scape:** "From the 75-gallon demo" builds its rock from 13 designer pieces, so every piece can be selected, moved and edited:
  - left island: a broad mound with a ledge and a swim-through arch;
  - right island: a taller stack with an overhang, peak about 10 cm under the water;
  - open sand between them.
- **Build:** about 10.2 L of rock (about 36 lb dry at 1.6 kg/L), built at 4 mm when the project is created (about 3 s), watertight, no designer warnings.
- **CLI demo unchanged:** `reefflow demo` and the verification reference case keep the original two procedural islands, so the numbers in VERIFICATION.md stay valid.
- **Rock element: two new options, both off by default** (existing rocks rebuild exactly as before, tested):
  - `facets`: cuts the rock with random planes (smooth maximum), giving flat broken faces;
  - `chunk`: adds large lumps (low-frequency noise of amplitude chunk × smallest lobe radius).
- **The demo rocks** use facets 7, chunk 0.35 and roughness 0.18. Higher roughness alone gave spiky "hedgehog" rocks; higher chunk (0.5) left small floating bits.
- **Tests:** tests/test_demo_scape.py (defaults unchanged, facets only remove rock, bounds hold, the demo builds clean and stays under water, the template uses the designer scape).
