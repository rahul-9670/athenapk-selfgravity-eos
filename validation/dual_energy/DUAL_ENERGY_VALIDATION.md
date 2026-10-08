# Dual-energy formalism in AthenaPK: validation report (paper I)

Version 2026-10-07 (public edition). Folder `validation/dual_energy/`. `README.md` is the index of decks, scripts and
results; every number below is read from the table file named with it (`results/tables/`).

*Public edition.* The validation was carried out in a larger private research code from which this repository was
extracted. That code also couples the dual energy to further physics modules; those couplings, their tests and findings,
the private development history and the production adoption are not part of this repository and are left out. Section numbers follow the full report, so Sections 6 and 9
are absent. Every hydro result is unchanged.

---

## 0. Summary

`hydro/dual_energy` carries the internal energy rho e as an extra advected variable and uses it instead of
E - KE - ME where that difference is untrustworthy. It was tested with the keys of the production collapse runs
(Sec. 2.1) against known answers in 1D, 2D and 3D, on CPU and H100 / GH200 GPUs, and across restarts.

| group | tests | verdict |
|---|---|---|
| Cold hypersonic flow | D1a contact at Mach 0..1e4; D1b cold velocity sinusoid (non-expanding Zel'dovich pancake), M 1..1e4, N 64..16384 | PASS for smooth flow: switched cells stay on the table adiabat to 4e-5 (L1) at any Mach number, 2nd order in resolved flow; dual off fails by up to dP ~ 1e4. Contacts: **DE-F3**, fixed by opt-in keys |
| Magnetically dominated gas | D2a low-beta sinusoid (par/perp B, beta 1..1e-4), D2c low-beta contact, D2b div B blob (2D, dedner_extended), D4 homologous compression x101 | PASS: flux freezing to 1e-14; compression on the adiabat to 3e-6 vs 0.76 off (beta 1e-4); the div B cleaning heat suppressed ~100-2000x. Strong blobs at beta 1e-4 drive shocks and leave the known-answer regime (C-1) |
| Shocks | D3 planar Noh (colliding cold streams, M 3/10/30) | PASS: exact star state to <= 3.4e-4, cold inflow exactly at T0; the shock exemption is essential (-16 % without); `dual_comp_ghost` makes the result decomposition-independent (bitwise) |
| Thresholds | D6 eta1 1e-3..0.2, eta2 off | eta1 trades cold-flow accuracy against shock heating; 5e-2 is a sound compromise |
| Portability | D8 GPU vs CPU; restart exactness | PASS: GPU = CPU to <= 1e-12, metrics identical; dual-energy restarts BITWISE |
| Gap closure (Sec. 12.2) | refined meshes (SMR/AMR), 3D, opt-in fixes of the measured residuals, GPU long blocks | PASS on refined meshes and in 3D as on uniform meshes; opt-in keys make switched contacts exact (also across level jumps), shocks eta1-independent, perpendicular-field compression 2nd order; GPU long 1D blocks were SILENTLY wrong (E-F8b), fixed |

Findings: **DE-F1** `cons_diag` 1D segfault (FIXED); **DE-F2** `boundary_flux_clamp` ignored the BC type (FIXED);
**DE-F3** dual-energy contacts not in pressure equilibrium (opt-in fixes `hydro/dual_energy_flux = density` and
`hydro/dual_energy_double_flux`); **E-F8b** GPU full-row tiles silently wrong for 1024-cell rows (FIXED).
Characterisation **C-1**: strong div B blobs at beta 1e-4 drive shocks, outside the known-answer regime.

## 1. Scope

In scope: the dual-energy variable, its advection, the -P div v source, the switch (eta1, two-sided eta2 / kappa /
shock_comp), exact resync, unsigned prim, step-end sync, `dual_comp_ghost`, `boundary_clamp_dual` (not exercised by a
known-answer test, Sec. 12.2 a), and the opt-in keys `dual_energy_flux`, `dual_energy_double_flux`, `dual_energy_pdv`,
`dual_energy_shock_eta1`. Not in scope: couplings to physics that is not part of this repository.

## 2. Code under test

### 2.1 Configuration (decks: `scripts/de_common.py`)
`hydro/dual_energy=true`, `dual_energy_eta1=5e-2`, `dual_energy_eta2=0.1`, `dual_energy_kappa=3.0`,
`dual_energy_shock_comp=0.02`, `dual_energy_sync_step_end`, `dual_energy_exact_resync`, `dual_energy_unsigned_prim`,
`eos_floor_consistent`, `dual_comp_ghost`, `boundary_clamp_dual`, `ghost_sync_rounds=1`, with the tabulated EOS
(`Tfloor=0.2726` = 2.73 K, `double_flux=true`, table `eos_table_hires_v5_cold_rho26.bin`, `vl2_predictor_recon=dc`,
`eos_interp=p`) and the hydro numerics of the collapse runs (hlld / plm / vl2 / cfl 0.3 / dfloor 1e-6 / pfloor 1e-8).
Stated deviations: `boundary_flux_clamp=false` on periodic/inflow boxes (DE-F2), `dedner_plain` in 1D (E-F3).
Problem generator: `eos_test` (modes slab, riemann, compress, sinus, divb).

### 2.2 Source and binaries
The tests ran with binaries of the private research tree (the hashes quoted below identify them; run directories with
`cmd.txt` are not shipped). The dual-energy code shipped here is that tree's final version minus its couplings to
physics modules that are not part of this repository.

**Port check (2026-10-08).** The code shipped here was compared with the research tree it was taken from (source of
the final dual-energy CPU binary `273f3124`, which is bitwise identical to the binary that ran the tests in this report):

1. *Dual energy off.* The 47 regression runs of this repository's self-gravity, EOS and `collapse_be` validation are
   bitwise identical to the previous public version (commit `9fd9c7b`) in every field of every output dump; the history
   files are identical apart from the added `cons-ndual` column.
2. *Dual energy on, default compiler flags.* 147 runs sampled from the run lists of this report (D1-D4, D6, D8b, T1-T3b,
   F1-F4, X2; 1 to 16 MPI ranks). Of the 92 that completed on both sides, none is bitwise; the largest relative
   difference (max |d| / max |ref| over all dumps and fields) is <= 1e-13 in 39, <= 1e-10 in 78, <= 1e-6 in 86, at most
   7.8e-4 (`T3/D3_M10_clamp_prod`, boundary clamp). Runs with dual energy off differ in the same way, so the cause is
   not the dual-energy code but GCC contracting a*b+c into fused multiply-adds differently in the two builds.
3. *Dual energy on, both trees compiled with `-ffp-contract=off`.* All 147 runs: the 140 that completed are bitwise
   identical in every field of every dump (1055 dumps). The other 7 were stopped before their end time and are bitwise
   in every dump both sides wrote (16 dumps): the two `T3b` boundary-clamp runs (dt collapse, see Sec. 12.2) and five
   long 16-rank `D2b` runs (the same setup completed bitwise in the `D2x` runs).

Three source files had comments edited after these builds; the shipped source was rebuilt the same way and
31 of the runs (1 rank, 21 of the 30 test groups) were repeated with it: bitwise identical again.

So the port changes no arithmetic: every difference between the two builds is FMA contraction.

## 3. References and tools
Known answers: the table adiabat de/dln rho = P(rho, e)/rho of the runtime interpolant (RK4, `eos_tabulated/scripts/
analyze_E8.adiabat`); the exact general-EOS Riemann solver (`eos_tabulated/scripts/riemann_exact.py`). Run harness
`scripts/run_list.sh` (lists `inputs/<test>_runs.txt`, 4th field = MPI ranks).

## 4. Cold hypersonic flow (D1)

### 4.1 Design note
Uniform advection carries no kinetic-energy truncation error (div v = 0, the -P div v source vanishes), so the
catalogue's "uniform cold state" is exact to round-off for any scheme. The discriminating test is a hypersonic flow
WITH a velocity gradient: D1b, v = u0 + M cs0 sin(2 pi x), uniform rho = 1 code, T = 10 K, run to 0.8 t_caustic. Before
the caustic the flow is smooth, so every cell must stay on the ONE table adiabat through (rho0, T0): metric dP =
P/P_ad(rho) - 1. The flow is resolved only while the velocity jump per cell is below cs, s = |dv|/cs < 1 (N >~ 10 M).

### 4.2 D1b results (`D1b_adiabat.txt`; cells split by the `dual_switched` field and by s)
* Resolved, switched cells converge at ~2nd order: M 100, max error 6.6e-3 (N 1024) -> 4.9e-5 (N 16384). L1 errors
  are 17x (M 100) and 180x (M 1000) below dual off at N 16384.
* All switched (boosted u0 = 10 vamp): L1 4e-5 at 0.4 t_c for EVERY M from 1 to 1e4 (Galilean); dual off grows to
  L1 2.8e3 at M 1e4 with thousands of Tfloor hits.
* Under-resolved (s up to 350): switched cells still on the adiabat (M 1e4: max 5e-4).
* The largest errors sit in the UNSWITCHED cells around the velocity nodes (v -> 0, e/E > eta1): they keep E - KE,
  whose truncation error from the steep gradient is amplified by up to 1/eta1 = 20. For M 100, N 1024 these are
  exactly the 38 cells with |sin kx| < 5.8/M. Larger eta1 shrinks this region (Sec. 8).

### 4.3 D1a contact and finding DE-F3 (`D1a_contact.txt`, `D1x_contact.txt`, `D1y_density_flux.txt`)
A pure contact rho 1|2 at equal P (T 10|5 K, above Tfloor) translated one box length at u = M cs0, N 256:

| | dual on | dual off |
|---|---|---|
| M 0, 1 (not switched) | 7e-15, 4e-14 | same |
| M 10 | 6.6e-3 | 4.9e-12 |
| M >= 100 | **5.6e-2** | 1.6e-5 |

Isolation (D1x, one change each, M 100): donor-cell reconstruction 1.4e-3; 5 crossings 5.6 -> 6.6e-2 (bounded);
N 64 -> 1024: 4.0 -> 6.8e-2 (NOT converging, confined to the contact cells); density ratio 1.1 / 2 / 10: 1.7e-3 /
5.6e-2 / **0.42**; eta1 1e-3 (nothing switches): identical to dual off. Double flux is inert in switched cells (on =
on_nodf bitwise).

Mechanism: (a) the dual component is advected like a passive scalar, flux = mass flux x the upwind RECONSTRUCTED
SPECIFIC energy (`hydro.cpp`, "Passive scalar fluxes"); PLM limits rho and e separately, so rho_face e_face !=
rho e = P/(Gamma - 1) at a contact; (b) the residual 1.4e-3 is the EOS-nonlinearity contact error that double flux
corrects in E - but a switched cell rewrites E from the dual energy.

Candidate fix for (a), opt-in key `hydro/dual_energy_flux = density` (own translation unit `hydro/dual_flux_density.cpp`,
task after CalculateFluxes on PLM stages): F_dual = F_mass q_face/rho_face with q = rho e and rho PLM-reconstructed on
the upwind side; exact at a contact (q_face = rho e, F_mass = rho_face v). Results: rho 1|2 5.6e-2 -> **1.4e-3** at
every M >= 10 and flat over 5 crossings; rho 1|10 0.42 -> 1.4e-2; low-beta contacts (D2c) 3.5-6.2e-2 -> 1.1-1.4e-3;
smooth flows (D1b, D2a) unchanged or better (boosted D1b at 0.8 t_c: max 1.5e-2 -> 6.9e-3). GPU = CPU to 6e-15.
Key off: bitwise legacy. Part (b): Sec. 12.2 (b).

## 5. Magnetically dominated gas (D2, D4)

### 5.1 D2a low-beta sinusoid (`D2a_lowbeta_sinus.txt`, `D2y_resolution.txt`)
D1b at M 3 with a uniform field. Parallel field (no force): the switch fires at beta <= 0.01 (estimate 0.035), dual on
6-9x better than off at beta 1e-4 (7.8e-4 vs 7.3e-3, N 256). Perpendicular field: flux freezing (B_y/rho)/(B0/rho0) - 1
<= 1e-14; with every cell switched the MAX error converges at ~1st order (order 0.7-0.95, 7.2e-5 at N 4096, beta 0.01)
while L1 converges at 2nd order (1.7e-7); dual off reaches 1e-5 in max. Small and localised; characterised.

### 5.2 D4 homologous compression (`D4_compress.txt`)
E8 harness (33 cells, exact homologous ghosts, now B- and dual-aware), x101 from (1e-13 g/cc, 10 K), transverse B
frozen into the flow (beta falls by ~5.5x):

| beta0 | dual on: e/e_ad - 1 | dual off |
|---|---|---|
| 1 (not switched) | 7.6e-3 | 7.6e-3 |
| 1e-2 | **-3.3e-6** | 7.8e-3 |
| 1e-4 | **-3.3e-6** | **+0.76** |

The -P div v source on the linear homologous velocity is exact; dual off at beta 1e-4 is wrong by 76 % (the E - KE - ME
cancellation). Compression rate alpha scaled per beta for cost (the homologous solution is alpha-independent).

### 5.3 D2b div B blob (`D2b_divb_blob.txt`)
2D 128^2, `dedner_extended`, uniform rho, T, v, B0z, plus a Dedner-shaped div B != 0 blob in Bx (rb = 0.125). Nothing
physical heats the gas; metrics: max|dP| vs the adiabat and the spurious heat sum rho (e - e_ad) dV over the blob's
excess magnetic energy. Static, at the end:

| beta, bamp/B0z | dual on: max dP / heat | dual off: max dP / heat |
|---|---|---|
| 1, 1 (not switched) | 2.1e-2 / 1.5e-2 | same |
| 1e-2, 0.1 | 1.0e-2 / 5.6e-5 | 0.12 / 1.3e-2 |
| 1e-2, 1 | 0.16 / 6.1e-4 | **70** / 0.13 |
| 1e-4, 0.1 | 0.12 / 8.9e-5 | **45** / 0.20 (64 148 Tfloor hits) |
| 1e-4, 1 (to t/tlim 0.5) | **7.3** / 2.3e-3 (11 924 Tfloor hits) | 1.7e4 |

Dual off reproduces the defect that motivated the module: cleaning turns up to 20 % of the unphysical
magnetic energy into heat. Dual on suppresses it 100-2000x; the strongest blob leaves the known-answer regime (Sec. 5.4).

### 5.4 C-1: the strong blob at beta 1e-4 (characterisation; `D2x_divb_mechanism.txt`)
The blob with bamp = B0z at beta 1e-4 still leaves max|dP| 1.4 (N 64, moving) to 7.3 (N 128, static) with the dual
energy on. Mechanism test at N 64, |v| = 10 cs0 (4 ranks): the extended-GLM momentum source -B div B drives motions of
up to 11 cs0 relative to the bulk flow (t/tlim 0.25), i.e. SHOCKS, so "every cell on the adiabat" stops being the known
answer once they form. Two hypotheses were tested and FALSIFIED:
* the shock exemption hands shocked low-beta cells E - KE - ME: production, shock exemption off (shock_comp 1e30) and
  eta2 off are IDENTICAL, and all 2047 deviating cells are switched (none in the E branch);
* the DE-F3 contact error at the compression-made density contrasts (rho 0.46-2.24): `dual_energy_flux = density`
  gives the same (max 1.25 vs 1.37, L1 4.65e-2 both).
82 % of the deviating cells lie ABOVE the adiabat, as shocked gas should; the switched cells' -P div v update in strong
compressions produces entropy, but no exact shock-heating reference exists for this flow. Dual off is wrong everywhere
(max|dP| 150, 4452 Tfloor hits). Conclusion: no defect demonstrated; the case is outside the known-answer regime.

*(Section 6 concerns a physics module that is not part of this repository.)*

## 7. Shocks (D3, `D3_noh.txt`)
Two cold streams (T0 10 K, u = +-M cs0) collide; exact star state from the general-EOS solver (M 30: rho2 12.7,
T2 1172 K). Final dump, plateau excluding the wall-heating cells:

| M, N 1024 | production | dual off | shock exemption off |
|---|---|---|---|
| 10 | P -1.1e-4 | +1.2e-3 | **-0.16**, shock 83 cells behind |
| 30 | P +3.4e-4 | +5.3e-4 | -4.1e-3, 16 cells behind |

Pre-shock |P/P0 - 1| <= 1e-16 (the cold inflow stays at T0). eta2 off: identical. With `dual_comp_ghost`
(production) 32 blocks on 1 or 8 ranks are BITWISE identical to one block; without it (legacy) the result depends on
the decomposition (max rel 1.6e3 at M 10, 4.5e3 at M 30; plateau max error 2.8e-3 -> 2.7e-2).

## 8. Thresholds and energy budget (D6, `D6_threshold.txt`)
eta1 in {1e-3, 1e-2, 5e-2, 0.1, 0.2}, eta2 off at 5e-2:
* Cold flow improves with eta1: D1b M 1000 N 1024 max|dP| 20 / 12.5 / 0.40 and Tfloor hits 702 / 130 / 0 at eta1 1e-3 /
  5e-2 / 0.2; resolved M 100: L1 1.1e-3 -> 6.8e-5.
* Shocks degrade with eta1: Noh M 10 plateau -1.1e-4 / -5.2e-3 / -1.7e-2 at 5e-2 / 0.1 / 0.2. The shock exemption
  only exempts from the eta2 test, not from the eta1 criterion: shock-front cells with e/E < eta1 take the cold
  advected energy.
* The div B blob needs eta1 above the gas e/E (beta 0.01: eta1 <= 1e-2 leaves it unswitched, 13 % heat).
* eta2 off: identical in every case (it targets low-beta upward swings these tests do not produce).
Production 5e-2 keeps the Noh error <= 3.4e-4.

*(Section 9 concerns a physics module that is not part of this repository.)*

## 10. Portability and restarts (D8, `D8_gpu_restart.txt`)
GPU (H100) vs CPU: max relative difference in rho and P <= 1.2e-12, every known-answer metric identical to the printed
digits. Restart at tlim/2 vs uninterrupted (CPU): BITWISE for D1b M 1000, D3 M 30 on 4 ranks and D2b. GPU observation
(= EOS E-F8, refined): the double-flux coefficient kernel's level-0 scratch (3 x nvar x (nx1 + 4) doubles) does not
launch for 1D meshblocks >~ 200 cells; 32^3 blocks need 8.6 KB; `hydro/scratch_level = 1` works (D1a at nx1 256).

## 11. Findings

**DE-F1** `cons_diag = true` segfaults in 1D (= EOS E-F4): the solver-flux columns read the X2/X3 flux arrays a 1D/2D
mesh does not allocate. FIXED (`cons_diag.cpp`, read only along existing dimensions).

**DE-F2** `boundary_flux_clamp` zeroes the inward flux on every mesh-edge face without checking the boundary type: a
periodic box loses its through-flow. FIXED in the gap closure (the clamp skips periodic faces).

**DE-F3** contacts in switched gas leave pressure equilibrium (Sec. 4.3): 5.6e-2 at rho 1|2, 0.42 at 1|10, bounded,
not converging. Fix `hydro/dual_energy_flux = density` (opt-in; adopted in the production runs).

**DE-F3 part (b)** (gap closure) switched contacts with double flux: the dual energy missed the own-EOS correction;
`hydro/dual_energy_double_flux` makes them exact (8e-11) on uniform AND refined meshes (the coarse side of a level
jump first mixed the restricted fine mass flux with its own face states: 3e-5..4e-4; fixed by the flux-corrected
`dual_vflux`). Opt-in.

**E-F8b** (gap closure; GPU builds, not dual-energy specific) kernels with an explicit full-row MDRange tile
{1, 1, 1, nx1} (16 AthenaPK reductions incl. the CFL time step; Parthenon's `MeshBlock::par_reduce`, which selects the md
pattern, e.g. `CalculateGlobalMinDx`) run one thread block per row: on the GH200 a row of 1024 cells made the time-step reduction return DBL_MIN
SILENTLY (rc 0; Noh M 30 blown to vacuum), rows > 2048 abort; 512 and 256 are correct. Rows of 32 cells are
unaffected. The original E-F8 abort (per-team scratch, rows >~ 200-400) had masked it. FIXED in device builds: row tile
capped at 256 (`utils/row_tile.hpp`, Parthenon md dispatch); host builds unchanged. GPU verification (GH200, `G_gpu_final.txt`): FINAL GPU 4f64c1e5 runs one 16384-cell block
(= CPU to 1.3e-13; before: abort) and one 1024-cell block (time step = CPU from cycle 0, same 12481 cycles, Noh plateau
+5.2838e-4 on both; max |dP/P| 2.9e-6 confined to the two shock-front cells; before: rho -> 0); the same 1024-cell deck
on the previous GPU binary still aborts with the original E-F8 TeamPolicy error (falsification of the scratch fallback). Small-block
runs (H2/H3) are identical to the pre-cap binary (G2/G3).

**C-1** (characterisation, not a finding) strong div B blobs at beta 1e-4: the GLM force drives Mach-11 motions;
the remaining O(1) deviations are in switched, mostly shocked cells; shock-exemption and contact-flux hypotheses
falsified (Sec. 5.4).

## 12. Defaults and recommendations
* The configuration of Sec. 2.1 is validated for its purpose: cold hypersonic and magnetically dominated gas stays on the
  adiabat where the single-energy scheme fails by orders of magnitude; shock heating is correct; `dual_comp_ghost` makes
  results decomposition-independent; eta1 5e-2 is a sound compromise; GPU and restarts are consistent.
* `hydro/dual_energy_flux = density` (DE-F3) is recommended (it is used in the production collapse runs).
* Every dual-energy key defaults to off/legacy; with `hydro/dual_energy = false` the code is the single-energy scheme.
* With `collapse_be` the dual energy is refused: its barotropic step overwrites the thermal energy and is not wired to
  the advected internal energy.

### 12.2 Gap closure
Every number below is in the named table. Binary hashes identify the binary of each run (recorded in its `cmd.txt`).

**(a) Untested -> tested.**
* *Refined meshes* (`T1_refinement.txt`, SMR one level / AMR maxdensity three levels, 1D). The
  production configuration behaves as on uniform meshes: D1b cold sinusoid M 100 L1 (all cells) 4.1e-3 (SMR) / 1.45e-2
  (AMR) vs 5.4e-2 / 0.17 with the dual energy off; boosted M 100 AMR 5.5e-4 vs 10.5 off; Noh M 10 / M 30 plateau
  -1.1e-4 / +3.4e-4 on AMR (uniform -1.1e-4 / +3.4e-4), cold inflow exact (1.1e-16); 4 ranks = 1 rank BITWISE (position
  matched). The DE-F3 contact (1.45e-3 with the density flux on SMR) is as on the uniform mesh.
* *3D* (`T2_3d.txt`, 64^3 on 8 ranks along k = 2 pi (1,1,1); Noh 1024x8x8): same verdicts as 1D. D1b oblique M 100 L1
  0.85 vs 2.95 off; contact M 100 4.2e-2 production / 1.1e-3 density flux / 2.0e-5 off (as 1D); 3D Noh plateau -1.1e-4,
  transverse spread exactly 0.
* *P-10 `boundary_clamp_dual`* (`T3_clamp_dual.txt`): no feasible known-answer test. With the inflow clamped at a box
  edge, the edge cells stagnate and heat to dP ~ +23 in every variant (production, legacy, dual off) -- the clamp's own
  physics, not the dual energy's; a beta 1e-4 variant (to keep the edge cells switched) collapsed dt to 1.2e-9 by t = 1.1e-4 with the clamp on
  (1.9e-6 without; run logs, not shipped) and was cancelled. P-10 is not covered by a known-answer test.

**(b) Measured -> fixed (new opt-in keys; key off = BIT-IDENTICAL on the test legs and on production restarts).**
* *DE-F3 part (b), contacts with double flux* -> `hydro/dual_energy_double_flux` (`D1z_dual_dflux.txt`,
  `F1_level_jump.txt`). The dual energy receives the own-EOS (frozen stiffened-gas) flux correction in double-flux
  contact cells and is reset to the table energy at the frozen-EOS pressure, as double flux does for E. Uniform meshes:
  rho 1|10 at M 100 max|P/P0 - 1| 8.0e-11 (density flux alone 1.4e-2, specific flux 0.42, dual off 1.2e-5), up to M 1e4 1.2e-7. Refined meshes
  first kept 3.0e-5 (SMR, 2:1) to 3.9e-4 (AMR): a coarse cell next to a finer block combined the restricted fine mass flux
  with its OWN face density/pressure. An EOS-consistent dual-aware ghost fill did not change it (F1d; kept for
  consistency). Fix: the face volume fluxes F_m/rho_f and F_m P_f/rho_f are fluxes of `dual_vflux` and are
  flux-corrected with the mass flux, the applied dual flux is the corrected one -> SMR 1.4e-10 / 6.4e-11 (rho 1|10), AMR
  2.6e-10 / 1.4e-10, i.e. the uniform-mesh value, 1e5 below dual off.
* *Shock heating at larger eta1* -> `hydro/dual_energy_shock_eta1` (`F2_shock_eta1.txt`): the shock exemption also
  overrides eta1 (not where e_int < eta1 (e_int + e_B), i.e. magnetically dominated cells). Noh M 10 plateau error
  becomes eta1-independent (+1.2e-3 at eta1 5e-2 / 0.1 / 0.2; without: -1.1e-4 / -5.2e-3 / -1.7e-2), M 30 +5.3e-4
  (without up to -2.1e-3); smooth flows unchanged at every eta1 (identical to the printed 3 digits). The cost at production eta1 5e-2: Noh M 10
  plateau +1.2e-3 vs -1.1e-4 -- the key buys eta1-robustness, not accuracy at 5e-2.
* *First-order max error with a perpendicular field* -> `hydro/dual_energy_pdv = face` (`F3b_face_pdv.txt`,
  `F3c_face_pdv_refined.txt`): -P div v from the face velocities the mass fluxes imply (stage-consistent; flux-corrected
  on refined meshes) instead of centred differences. D2a perpendicular, beta 0.01: max 2.1e-4 -> 9.3e-8 (N 1024), order
  ~1 -> ~2 (6.2e-9 at N 4096); beta 0.001 7.5e-5 -> 1.0e-8. SMR: 3.4e-6 at the level jumps, 2nd order (1.8-1.9). AMR: 48-794x
  below centred at every resolution but irregular order (1.0-1.7), maxima next to freshly refined blocks (inferred:
  regrid prolongation). Costs: Noh M 10 plateau -4.0e-4 vs -1.1e-4; boosted D1b max 1.67e-2 vs 1.52e-2.
* *C-1 second law* (`F4_fixes_secondlaw.txt`): with all opt-in keys the strong beta 1e-4 blob has 10 cells below the
  adiabat by > 1e-2 instead of 360 (min dP -1.5e-2 vs -2.1e-2); still no exact answer for the shocked flow (inherent).

**(c) Small items.** E-F3 (`dedner_extended` in 1D) and DE-F2 (clamp on a periodic box) FIXED: each pair BITWISE
(`F4_fixes_secondlaw.txt`). E-F8 / E-F8b, long 1D meshblocks on the GPU (`G_gpu_final.txt`, finding
E-F8b in Sec. 11): FIXED and verified on the GH200. New keys on the GPU (vs CPU): dual double flux SMR 1.2e-11, face pdV AMR 1.5e-14. Noh Tfloor hits
(`X2_noh_tfloor.txt`): with the dual energy OFF the cell just ahead of each shock (r/r_s 1.03-1.22) undershoots to
P/P0 = 0.273 = Tfloor/T0 in 9 of 25 snapshots -> a pre-shock undershoot of the hydro scheme (not a dual-energy effect).

**(d) Inherent, documented.** *C-1*: no exact answer for strong-blob shocks (Sec. 5.4).

**(e) Recommendation on the opt-in keys.**
* `hydro/dual_energy_pdv = face`: large gain for perpendicular-field compression (2nd order, 48-7500x smaller error),
  small costs (Noh M 10 plateau -4.0e-4 vs -1.1e-4).
* `hydro/dual_energy_double_flux`: makes switched contacts exact with double flux; correct, low cost.
* `hydro/dual_energy_shock_eta1`: makes shock heating eta1-independent, but worsens the Noh plateau at eta1 5e-2
  (+1.2e-3 vs -1.1e-4); only useful if eta1 is raised.

## 13. Test-infrastructure traps met in this module
* `xargs -L 1` joins a line ending in a blank with the next line (empty override field): garbled arguments.
* `mpirun` inside `while read` consumes the loop's stdin (only the first line ran): redirect `< /dev/null`.
* A run-list default binary that predates a fix silently reintroduces the bug: pass BIN explicitly.
* 1D periodic decks with the production `boundary_flux_clamp` drain to vacuum (DE-F2).
* A contact's dense side below Tfloor makes the floor the answer (first D1a).
* The E8 compression rate costs ~1e7 cycles at beta 1e-4: scale alpha (the solution is alpha-independent).
* `refinement/type = maxdensity` with a threshold above the flow's rho_max silently never refines: the "AMR" run is a
  uniform run (first F3c beta 1e-3 rows, moved to `runs/F3c_invalid/`). The analyses now flag NO REFINEMENT.
* `analyze_D2.cells()` returns ONE uniform-mesh cell volume: on refined meshes weight by the per-cell widths.
* `run_list.sh` FILTER is anchored at the start of the run name (`^FILTER`): use `.*_old`, not `_old`.
* A "static region" covering every root block refines the whole box (no level jump).

## 14. Reproduction map
Decks: `scripts/make_decks_D{1,1x,1y,2,3,4,6}.py` -> `inputs/`; runs: `scripts/run_list.sh` with `inputs/*_runs.txt`
(4th field = ranks), `scripts/D8_gpu.sh`, `scripts/D8b_gpu.sh`; analysis: `scripts/analyze_D{1,1x,1y,1z,2,2x,2y,3,4,6}.py`,
`scripts/analyze_F3.py` -> `results/tables/`. Gap-closure run lists `inputs/{T1,T2,T34,T3b,F1b,F1c,F1d,F1e,F2,F3,F3b,F3c,
F3d,F3e,F4,X2}_runs.txt`; their tables (T1, T2, T3, F1-F4, X2, G) were produced by analysis scripts that also cover tests
of modules not in this repository and are shipped as tables only.

## 15. References
Bryan, Norman, Stone, Cen & Ostriker 1995, Comput. Phys. Commun. 89, 149 (dual energy); Ryu, Ostriker, Kang & Cen
1993, ApJ 414, 1; Noh 1987, J. Comput. Phys. 72, 78; Dedner et al. 2002, J. Comput. Phys. 175, 645 (GLM);
Abgrall 1996, J. Comput. Phys. 125, 150; Abgrall & Karni 2001, J. Comput. Phys. 169, 594 (double flux).
