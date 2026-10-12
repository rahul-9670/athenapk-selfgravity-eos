# Non-ideal MHD (Ohmic + ambipolar diffusion, RKL2 super-time-stepping, ionization coefficients) in AthenaPK: validation report (paper I)

Version 2026-10-12 (public edition of the report of 2026-10-11). Folder `validation/nonideal_mhd/`. `README.md` is the
index of decks, scripts and results; every number below is read from the table file named with it (`results/tables/`),
each written by a saved script with a `--check` mode.

*Public edition.* The validation was carried out in a larger private research code from which this repository was
extracted. That code also has the Hall term, thermal conduction, viscosity and a dust-evolution package coupled to the
ionization model, and its non-ideal module was additionally gated on 3-D production collapse restarts (a cell-by-cell
check of the production diffusivities against an independent evaluation, a diagnosis of an envelope hot spot that
turned out to be the donor-cell VL2 predictor, the production ceilings, restart and GPU equivalence). Those parts, the
private development history and the production adoption are not part of this repository and are left out. Section
numbers follow the full report. Every result of the tests shipped here is unchanged.

---

## 0. Summary

| # | test | exact answer / criterion | result | verdict |
|---|---|---|---|---|
| A7 | RKL2 amplification of single Fourier modes, ratio 1-1000 | Legendre stability polynomial R_s(z)^2 | 36/36 Ohmic runs to <= 3.4e-7 | **PASS** |
| A2 | per-cell energy budget of the super-step | total E conserved; Joule heat = magnetic energy lost, per cell | total E 1.6e-16; per cell 7e-13 where hydro-inert | **PASS** |
| A1 | Ohmic Gaussian | heat-equation solution (periodic image sum) | order 2.00, L1 1.8e-5 at N 512 | **PASS** |
| A3 | damped Alfven eigenmode with AD, aligned and 45 deg | complex dispersion relation | order 1.9-2.1, 3e-5 at N 512 | **PASS** |
| A4 | 3-D ABC (Beltrami) field | Ohmic exp(-eta k^2 t); AD exactly zero | Ohmic order 2.00; AD null order 1.95 (0.42 % at 64 cells/wavelength) | **PASS** |
| A5 | Ohmic diffusion across an eta jump (contrast 10, 100) | two-medium (composite) solution | arithmetic face mean FIRST order (4.4 % at the interface); harmonic: order 2-3, 295x smaller | **NI-F2, FIXED (Ohmic): `eta_face_harmonic`** |
| A6 | nonlinear AD spreading, eta_A = Q B^2 | Barenblatt-Pattle (m = 3) | RKL2 ratio 1000 + frozen eta = unsplit (L1 1.5e-4); harmonic face mean breaks it | **PASS**; harmonic restricted to Ohmic |
| A9 | A1 across a static coarse-fine boundary at the peak | same solution | order 2.00, coarse-fine cells 2nd order | **PASS** |
| A10 | ambipolar C-shock | steady single-fluid profile (`scripts/cshock_ref.py`) | profile L1 1e-4, downstream 1e-3 (long box) | **PASS** (caveat NI-O1) |
| A11 | conductivity gates 1-8 | identities, limits, C++ vs independent Python | all pass; eta_A tensor C++ vs Python 2.9e-3 | **PASS** |

## 1. Scope

In scope: the Ohmic and ambipolar flux operators (`src/hydro/diffusion/resistivity.cpp`, `ambipolar.cpp`), the
cell-centred eta cache and its face average, the RKL2 integrator, its stage count and ratio cap
(`src/hydro/hydro_driver.cpp` AddSTSTasks, `src/hydro/hydro.cpp` EstimateTimestep), the hand-over of Joule / ambipolar
heat to the dual energy (`diffusion/dual_energy_heating.cpp`, `hydro/dual_heat_exact.cpp`), and the ionization
coefficient model (`diffusion/ionization.hpp`, `ionization_build.hpp`, the fused evaluator in `diffusion.hpp`). The
ideal-MHD step (`../`, other modules), the EOS (`../eos_tabulated/`) and the dual-energy scheme (`../dual_energy/`) have
their own reports.

## 2. Code under test

### 2.1 Keys
`<diffusion>`: `integrator = unsplit | rkl2`, `rkl2_max_dt_ratio` (cap on dt / dt_diff), `rkl2_freeze_eta` (eta frozen
at the start of each Strang half for all RKL2 stages), `cfl` (= cfl_diff: dt_diff = cfl x 0.5 dx^2/eta in 1-D,
x 1/6 in 3-D), `resistivity = ohmic` with `resistivity_coeff = fixed | ionization | density_power` (the last one is
validation-only: eta = c rho^p, `ohm_rho_power`), `ambipolar = ambipolar` with `ambipolar_coeff = fixed | ionization`
(fixed: eta_A = Q B^2, `ambipolar_coeff_code` = Q), `ion_*` (ionization model; `ion_ad_closure = single_fluid` (default)
or `tensor`; `ion_cr_model`, `ion_cr_column_model = local | calibrated`), `eta_ohm_cap_code`, `eta_ad_cap_code`,
`eta_cache` (default on for ionization coefficients), `eta_face_harmonic` (NI-F2, Ohmic faces, default false),
`dt_sum_ohm_ad`, `cap_diag`, `rkl2_skip_empty`. `<hydro>` with dual energy: `dual_energy_heat_rkl2` (heat as a source
inside the RKL2 recurrence) and `dual_energy_heat_exact` (exact end-of-half closure; needs `heat_rkl2` and, under rkl2,
`rkl2_freeze_eta`). The research code's production setting is rkl2, ratio cap 1000, frozen eta, cfl 0.3, ionization
coefficients with the tensor closure and the calibrated column, `heat_exact`, `eta_face_harmonic`; the test decks carry
those integration keys with constant coefficients where an exact answer needs them.

RKL2 in brief: each Strang half advances tau = dt/2 with s stages, s = ceil(0.5 (sqrt(9 + 16 tau/dt_diff) - 1)) made
odd; stability needs tau/dt_FE <= (s^2+s-2)/4. At the cap (ratio dt/dt_diff = 1000) s = 45 per half.

### 2.2 Source and binaries
The tests ran with binaries of the private research tree (validation branch; run directories with `cmd.txt` are not
shipped). Every code change there is a key, default off. The non-ideal code shipped here is that tree's version of
2026-10-11 restricted to Ohmic + ambipolar diffusion: the Hall term, thermal conduction, viscosity and the per-cell
dust-package coupling of the ionization model are removed and refused at startup (`diffusion/hall`, `conduction`,
`viscosity`, `dust_coupling`); the eta_H slot of the cache and the tensor solve are kept unchanged (inert), so the
evaluator is the research code's line for line.

**Port check (2026-10-12, `results/tables/PORT_check_2026-10-12.txt`).** This repository's build: `cd69c066` (default
flags), `2d66e869` (`-ffp-contract=off`).

1. *Diffusion off.* The 47 regression runs of this repository's self-gravity, EOS, dual-energy and `collapse_be`
   validation against the previous public version (commit `40a3fb0`, build `46b778e5`), default flags: all 47 bitwise
   identical in every field of every dump and in the history files (including the magnetized turbulent `collapse_be`
   run L10, 33 dumps).
2. *Diffusion on.* 164 runs of this report's decks (A7: all 72; A1/A3 N <= 256; A4 N <= 32; A5 N <= 512, both face
   averages; A6 N <= 256, all five arms; A9 N <= 256; A10 N <= 256 with and without dual energy, and the long box at
   N 512) with this repository and with the research tree's validation branch (`987a21a5`), both `-ffp-contract=off`:
   all 164 bitwise identical in every field of every dump and in the history files.
3. *Radiation on, diffusion off.* The 605 radiation runs of the radiation port check (`../radiation_sn/`, Sec. 2.2: the
   593 sampled legs plus its two follow-up lists) against commit `40a3fb0` (`f8233dda`), both `-ffp-contract=off`: all
   605 bitwise identical (5940 dumps); the runs that stop with rc 134 do so identically on both sides (the R5bis
   BiCGSTAB trust audit and the WF lines written with the old `sn_sweep_wavefront = true`, as in that check).
4. Host tests (`src/hydro/diffusion/tests/test_conductivity.py` gates 1-7 and, with `xcheck_conductivity.cpp` built
   from this tree, gate 8): all pass; the C++ dump is byte-identical to the research tree's.

Source comments were edited after these builds (line numbers unchanged); the shipped source rebuilds to byte-identical
binaries (`cd69c066`, `2d66e869`). The GPU build of this tree was not compiled or run for this check.

## 4. A7 — RKL2 amplification of single Fourier modes (PASS)

**Set-up** (`make_decks_A7.py`, `A7_template.in`, `run_A7.sh`, `analyze_A7.py`, table `A7.{json,txt}`). 1-D periodic,
N = 64, iprob 41 with Bx = 0: B = amp (0, sin kx, cos kx), |B| uniform, so the hydro step is inert and one cycle is two
RKL2 halves of pure diffusion of mode m. eta = 100 (Ohmic, fixed), so dt_hyp/dt_diff >> 1000 and dt = r x dt_diff is set
by the cap r in {1, 10, 100, 300, 500, 1000}; m in {1, 4, 8, 16, 24, 31}.

**Exact answer.** The discrete operator has eigenvalue lambda = eta (4/dx^2) sin^2(k dx/2) (compact face gradient); one
cycle multiplies the mode by R_s(z)^2, z = -(dt/2) lambda, with the RKL2 stability polynomial
R_s(z) = a_s + b_s P_s(1 + w1 z), P_s Legendre, b_s = (s^2+s-2)/(2s(s+1)), a_s = 1 - b_s, w1 = 4/(s^2+s-2) (closed
form; independent of the code's three-term recurrence). s and dt are read from the run log.

**Result.** All 36 Ohmic runs match R_s(z)^2 to <= 3.4e-7 relative. Selected rows (G = measured one-cycle amplification):

| ratio | s | m | z / bound | G (= R_s^2) | exact exp(-lambda dt) |
|---|---|---|---|---|---|
| 1 | 3 | 1 | 0.000 | 0.99856 | 0.99856 |
| 1 | 3 | 31 | 0.060 | 0.55312 | 0.54957 |
| 100 | 15 | 1 | 0.001 | 0.86553 | 0.86553 |
| 100 | 15 | 8 | 0.037 | 0.33533 | 1.5e-4 |
| 1000 | 45 | 1 | 0.001 | 0.25032 | 0.23582 |
| 1000 | 45 | 4 | 0.011 | 0.15768 | 1.2e-10 |
| 1000 | 45 | 8 | 0.042 | 0.32338 | 6.9e-39 |
| 1000 | 45 | 16 | 0.145 | 0.18633 | 5.1e-131 |
| 1000 | 45 | 31 | 0.289 | 0.30273 | 1.1e-260 |

**What it quantifies.** The stiffest mode sits at 29 % of the RKL2 stability bound (the 0.3 diffusion CFL). At the
production ratio the integrator is stable but does not damp grid-scale modes: they keep 16-32 % of their amplitude per
cycle where the exact decay is < 1e-38, and the resolved m = 1 mode is damped 6 % too weakly per cycle (second-order
error growing as ratio^2). **AD** on the same field: the circularly polarized field is force-free (J = k B), so AD must do
nothing: |G - 1| <= 1.4e-6 in all 36 AD runs (the 1-D A4 null).

## 5. A2 — energy budget of the super-step (PASS)

**On the A7 Ohmic runs** (`analyze_A2_budget.py`, `A2_budget.{json,txt}`). Exact: total energy conserved (periodic), and
since |B| stays uniform the Joule heat equals the magnetic energy lost in every cell: d(e_int) = -d(ME).
* Total E: conserved to <= 1.6e-16 relative in all 36 runs.
* Per cell, max |d(e_int) + d(ME)| / mean |d(ME)| (E-KE-ME path = dual-energy path):

| m \ ratio | 1 | 10 | 100 | 300 | 500 | 1000 |
|---|---|---|---|---|---|---|
| 1 | 4.4e-5 | 3.7e-5 | 4.1e-5 | 6.1e-5 | 8.3e-5 | 1.1e-4 |
| 4 | 3.7e-5 | 3.2e-5 | 3.0e-5 | 3.2e-4 | 3.3e-3 | 1.7e-3 |
| 8 | 1.7e-10 | 7.0e-10 | 4.1e-8 | 5.9e-8 | 1.3e-6 | 3.6e-6 |
| 16 | 1.5e-15 | 7.7e-15 | 3.6e-14 | 1.0e-13 | 1.0e-13 | 2.2e-13 |
| 24 | 1.0e-15 | 1.2e-14 | 1.4e-13 | 2.5e-13 | 3.1e-13 | 4.5e-13 |
| 31 | 4.8e-16 | 1.4e-15 | 3.8e-14 | 1.6e-13 | 3.5e-13 | 6.9e-13 |

For m >= 16 the super-step deposits exactly the heat the field loses, cell by cell, even at ratio 1000 where those modes
keep 20-30 % of their amplitude. The long modes' residuals correlate with the density change of the hydro step
(r = 1.000; at ratio 1 the slope 1.58 ~ (gamma e + B^2)/(gamma e) = 1.67 = compression work on gas and field): the
sampled force-free field is not an exact equilibrium of PLM + HLLD (face-reconstructed |B|^2 is not uniform), so those
runs are not hydro-inert. Attributed to the set-up (inferred, mechanism measured).
* **Alfvenic coupling** (Bx != 0, exact dispersion, exact per-cell heat) in the production RKL2 regime was measured in
  the dual-energy module (D5, `../dual_energy/DUAL_ENERGY_VALIDATION.md` Sec. 6): with the exact heat closure every case
  within 2e-9 of the energy actually lost; underdamped AD -1.1e-4. Not repeated.

## 6. A1 — Ohmic diffusion of a Gaussian, production keys (PASS)

**Set-up** (`make_decks_A1A3.py`, `analyze_A1A3.py`, `A1A3.{json,txt}`): iprob 40, B_y Gaussian on the periodic box
[-4, 4] (x 8 cells in y), eta = 0.25, t0 = 0.5, tlim 2; unsplit vs rkl2 (cap 1000), N = 64..512. Exact: the periodic image
sum amp/sqrt(4 pi eta T) sum_n exp(-(x - 8n)^2 / (4 eta T)), T = t0 + t (a single Gaussian plateaus at 4.2e-4, audit
F-A14).

| N | unsplit L1 | rkl2 L1 | order |
|---|---|---|---|
| 64 | 1.168e-3 | 1.169e-3 | -- |
| 128 | 2.918e-4 | 2.927e-4 | 2.00 |
| 256 | 7.287e-5 | 7.310e-5 | 2.00 |
| 512 | 1.820e-5 | 1.825e-5 | 2.00 |

Identical to the 2026-09-14 audit: the production dual-energy keys do not affect the field diffusion.

## 7. A3 — damped Alfven eigenmode with AD (PASS)

**Set-up**: iprob 50 (the 2026-09-14 audit deck + production keys), guide field Bx = 1, By = amp sin kx, Q_A = 0.5,
tlim 0.3; aligned and oblique (`Bguide_y = 1`, 45 deg: J has a component along B, so the (J.b)b projection of the AD EMF
is exercised; damping eta_A k^2 cos^2 theta). Exact: s^2 + eta_A k^2 cos^2(theta) s + k^2 v_A^2 cos^2(theta) = 0; the
pgen prints the relative error of the final amplitude.

| N | aligned unsplit | aligned rkl2 | oblique unsplit | oblique rkl2 |
|---|---|---|---|---|
| 64 | 1.92e-3 | 1.86e-3 | 1.92e-3 | 1.88e-3 |
| 128 | 5.47e-4 | 4.90e-4 | 5.47e-4 | 5.03e-4 |
| 256 | 1.45e-4 | 1.16e-4 | 1.45e-4 | 1.34e-4 |
| 512 | 3.71e-5 | 3.01e-5 | 3.71e-5 | 3.54e-5 |

Orders 1.8-2.1; RKL2 slightly more accurate than unsplit (fewer, larger steps of a second-order scheme).

## 8. A4 — 3-D ABC (Beltrami) field (PASS)

**Set-up** (pgen iprob 44, `make_decks_A4.py`, `analyze_A4.py`, `A4.{json,txt}`): B = amp (A sin kz + C cos ky,
B sin kx + A cos kz, C sin ky + B cos kx), A = 1, B = sqrt(2/3), C = sqrt(1/3), k = 2 pi, amp 1e-3, periodic unit cube,
t = 0.025; Ohmic eta = 1 (eta k^2 t = 0.99); AD Q = 1e6 (eta_A 0.3-2.4); N = 16, 32, 64. curl B = k B, so J x B = 0:
Ohmic decays every component as exp(-eta k^2 t); AD must leave the field unchanged.

| N | Ohmic unsplit G/exact - 1 | Ohmic rkl2 | shape error | AD unsplit max dB/B | AD rkl2 |
|---|---|---|---|---|---|
| 16 | 1.27e-2 | 1.69e-2 | 7.8e-7 | 5.70e-2 | 5.76e-2 |
| 32 | 3.17e-3 | 4.28e-3 | 1.5e-7 | 1.62e-2 | 1.63e-2 |
| 64 | 7.93e-4 | 1.08e-3 | 4.4e-8 | 4.20e-3 | 4.20e-3 |

Ohmic order 2.00 (unsplit) / 1.99 (rkl2). The AD null converges at order 1.95, unsplit = rkl2 to 4 digits: spatial
truncation of the cell-centred J x B projection, ~ (k dx)^2 (5.8 % at 16 cells per wavelength over eta_A k^2 t ~ 1).
In 1-D (A7) the null holds to 1.4e-6 because J x B = 0 holds discretely there.

## 9. A5 — Ohmic diffusion across an eta jump: finding NI-F2

**Set-up** (pgen iprob 42, validation-only `resistivity_coeff = density_power` with eta = 100 rho^1;
`make_decks_A5.py [--harmonic]`, `analyze_A5.py`, `A5.{json,txt}`): 1-D periodic [0, 1], rho = rho_in on 0.25 < x < 0.75
and 1 outside at uniform p = 1 (static contact), By = 1e-3 inside, 0 outside (beta ~ 2e6, hydro inert), so eta jumps
from 100 to 100 rho_in at the same faces where B jumps; contrasts 10 and 100; t = 2.5e-5 (diffusion length 0.05 on the
high-eta side); schemes unsplit, rkl2 cap 100 and 1000; N = 64..1024.

**Exact answer.** Two semi-infinite media with continuous B and continuous flux eta dB/dx (Carslaw & Jaeger): interface
value B_i = (B_in sqrt(eta_in) + B_out sqrt(eta_out)) / (sqrt(eta_in) + sqrt(eta_out)), erf profiles on each side
(effusivity sqrt(eta)). Errors: L1 over the box and the two cells beside the interface, relative to B_in.

| contrast | N | unsplit L1 (arithm.) | interface | unsplit L1 (harmonic) | interface | rkl2-1000 L1 (arithm.) | rkl2-1000 L1 (harm.) |
|---|---|---|---|---|---|---|---|
| 10 | 128 | 6.67e-3 | 5.84e-2 | 6.14e-4 | 1.89e-3 | 9.02e-3 | 8.90e-3 |
| 10 | 256 | 3.33e-3 | 2.95e-2 | 1.52e-4 | 2.28e-4 | 3.27e-3 | 3.14e-3 |
| 10 | 512 | 1.67e-3 | 1.48e-2 | 4.02e-5 | 2.83e-5 | 1.64e-3 | 7.85e-5 |
| 10 | 1024 | 8.34e-4 | 7.40e-3 | 1.25e-5 | 3.53e-6 | 8.33e-4 | 1.37e-5 |
| 100 | 128 | 1.54e-2 | 3.32e-1 | 3.21e-3 | 8.11e-2 | 9.62e-3 | 3.37e-3 |
| 100 | 256 | 6.93e-3 | 1.74e-1 | 7.04e-4 | 1.13e-2 | 5.96e-3 | 1.25e-3 |
| 100 | 512 | 3.30e-3 | 8.81e-2 | 1.68e-4 | 1.23e-3 | 3.28e-3 | 1.75e-4 |
| 100 | 1024 | 1.61e-3 | 4.42e-2 | 4.21e-5 | 1.50e-4 | 1.61e-3 | 4.25e-5 |

* Arithmetic face mean (production until 2026-10-11): **first order** in L1 and at the interface, identical for unsplit,
  rkl2-100 and rkl2-1000 at N >= 512 -> a spatial error, not the integrator. At N = 1024 the interface cells are wrong by
  4.4 % (contrast 100) / 0.74 % (10) of the jump.
* Harmonic face mean: unsplit order 2.0 in L1 and 3.0 at the interface; at N = 1024 / contrast 100 the L1 error is 38x and
  the interface error 295x smaller. RKL2 at ratio 1000 with the harmonic mean converges to the same answer for N >= 512
  (coarse grids take 2-3 cycles and are irregular).
* Mechanism: flux continuity across a jump needs the series (harmonic) conductance of the two half-cells; the arithmetic
  mean overestimates the conductance of the low-eta side by up to (1 + c)^2 / (4c) for contrast c.
* Relevance: the production eta varies by orders of magnitude across density steps (ionization model), so the Ohmic
  fluxes at those steps were first-order accurate. **Fix:** key `diffusion/eta_face_harmonic` (FaceEta in
  `diffusion.hpp`), Ohmic faces only (A6). Key off = bitwise unchanged.

## 10. A6 — nonlinear ambipolar spreading (Barenblatt-Pattle): PASS; harmonic mean rejected for AD

**Set-up** (pgen iprob 43, `make_decks_A6.py`, `analyze_A6.py`, `A6.{json,txt}`): Bz(x) only (B perpendicular to x,
J perpendicular to B), AD with eta_A = Q B^2, Q = 1e6, so dBz/dt = d/dx(Q Bz^2 dBz/dx) (porous-medium equation, m = 3).
Exact: Bz = B0 T^(-1/4) sqrt(1 - (x - xc)^2 / (xf0^2 T^(1/2)))_+, T = (t0 + t)/t0, t0 = xf0^2/(4 Q B0^2); B0 = 1e-3,
xf0 = 0.1 at t0 = 2.5e-3, run 80 t0 (front 0.1 -> 0.3). Arms: unsplit; rkl2 cap 100 and 1000 with freeze_eta
(production); cap 1000 refilling eta every stage; cap 1000 + harmonic face mean.

| arm | N | L1 / peak | peak error | front vs 0.300 | flux drift |
|---|---|---|---|---|---|
| unsplit | 512 | 1.78e-4 | -2.0e-4 | +0.77 % | 1e-15 |
| rkl2 1000 (production) | 512 | 2.07e-4 | +2.1e-4 | +0.58 % | 1e-14 |
| rkl2 1000 (production) | 1024 | 1.50e-4 | +3.2e-5 | +0.38 % | 6e-15 |
| rkl2 1000, eta refilled per stage | 1024 | 1.60e-4 | -1.7e-4 | +0.40 % | 9e-15 |
| rkl2 100 | 1024 | 1.51e-4 | -1.4e-4 | +0.39 % | 2e-14 |
| rkl2 1000 + harmonic | 1024 | 5.48e-1 | **+1.34** | **0.101 (frozen)** | 1e-14 |

* The production arm agrees with unsplit and with per-stage eta refill to within the spatial error: freezing eta over a
  ratio-1000 super-step adds no measurable error to the B^2 nonlinearity. The front (threshold 1e-3 of the peak on a
  degenerate square-root profile) converges at about first order; L1 plateaus near 1.5e-4 (front-corner limited).
* Harmonic face mean: eta_A = Q B^2 is zero outside the support and the harmonic mean of a zero is zero, so no flux
  crosses the front; the field piles up (peak +134 %) at every N. -> `eta_face_harmonic` acts on Ohmic faces only; AD is
  hard-wired arithmetic (AD with the key on is bitwise identical to arithmetic).

## 11. A9 — Ohmic Gaussian across a static coarse-fine boundary (PASS)

**Set-up** (`make_decks_A9.py`, `analyze_A9.py`, `A9.{json,txt}`): the A1 deck with a static refinement level on
0 < x < 4 (blocks of width 1): coarse-fine faces at x = 0 (the Gaussian's peak) and +-4. Volume-weighted L1 vs the image
sum, by half, and the two cells beside x = 0.

| N | L1 (unsplit) | order | coarse half | uniform N (A1) | fine half | uniform 2N (A1) | C/F cells |
|---|---|---|---|---|---|---|---|
| 64 | 6.97e-4 | -- | 1.15e-3 | 1.17e-3 | 4.72e-4 | 2.92e-4 | 1.39e-3 |
| 128 | 1.74e-4 | 2.01 | 2.86e-4 | 2.92e-4 | 1.17e-4 | 7.29e-5 | 3.45e-4 |
| 256 | 4.33e-5 | 2.00 | 7.16e-5 | 7.29e-5 | 2.92e-5 | 1.82e-5 | 8.58e-5 |
| 512 | 1.08e-5 | 2.00 | 1.79e-5 | 1.82e-5 | 7.28e-6 | -- | 2.14e-5 |

rkl2 (cap 1000) identical to 3 digits. Prolongation and the diffusive flux correction keep the scheme second order.

## 12. A10 — ambipolar C-shock (PASS; caveat NI-O1)

**Reference** (`scripts/cshock_ref.py`, written for this module): perpendicular field B = By(x) y, adiabatic gamma 5/3,
single-fluid AD with eta_A = Q By^2, shock frame with gas flowing in +x. Conserved: m = rho v;
Pi = m v + P + By^2/2; F = m v^2/2 + gamma/(gamma-1) P v + v0 B0 By (the last term is the Poynting flux -E_z By with
E_z = -v0 B0 everywhere). Given By these give v (quadratic, root continuous with v0); the induction equation
Q By^2 dBy/dx = v By - v0 B0 gives the profile, integrated from the upstream state. The lab-frame shock speed V follows
from the ideal perpendicular jump conditions with the downstream gas at rest.
**Set-up** (pgen iprob 45, `make_decks_A10.py`, `analyze_A10.py`, `A10.{json,txt}`): uniform rho0 = 1, u = 2, P0 = 0.1,
By = 2 (c_s 0.41, v_A 2) driven into a `reflecting_val` wall at x = L, outflow inlet; Q = 0.01. Reference: V = 1.9106,
fast Mach 1.92, compression 2.0468, C-type (no sub-shock), 10-90 width W = 0.117; downstream rho1 2.0468, P1 1.5425.
Comparison: By aligned at the midpoint (B0+B1)/2, L1 over +-3W relative to B1 - B0; downstream mean 2-3W behind the shock.

| box | arm | N per unit length | By L1 | By Linf | downstream rho / B / P |
|---|---|---|---|---|---|
| L 1, t 0.3 | rkl2 (production keys) | 128 .. 1024 | 2.1-2.3e-3 | 1.7e-2 | +1.1 % / -0.48 % / +5.2 % |
| L 1, t 0.3 | unsplit, dual energy off | 128 .. 1024 | 2.1-2.3e-3 | 1.7e-2 | same to 3 digits |
| L 1, t 0.3 | rkl2, dual energy off | 128 .. 1024 | 2.1-2.3e-3 | 1.7e-2 | same to 3 digits |
| L 2, t 0.7 | rkl2 (production keys) | 512 / 1024 | 1.07e-4 / 9.27e-5 | 2.8e-4 | -7.0e-4 / -3.2e-4 / -9.8e-4 |
| L 2, t 0.7 | rkl2, dual energy off | 512 / 1024 | 1.08e-4 / 9.29e-5 | 2.8e-4 | -7.1e-4 / -3.2e-4 / -9.6e-4 |
| L 1, t 0.3 | **unsplit + dual energy** | 128 .. 1024 | 3.3e-2 | 9.3e-2 | +4.4 % / +4.7 % / **-76 %** |

* In the short box every consistent arm gives the same answer at every N: the shock (born at the wall at t = 0) has not
  yet relaxed; the shock speed already matches V. Gas next to the wall carries the start-up (wall-heating) anomaly
  (rho +7 %, P +40 % at x 0.85-0.95), so the downstream window must sit right behind the shock.
* In the long box the profile matches to 1e-4 and the downstream state to 1e-3; the production keys (RKL2 + dual energy
  + exact heat closure) equal dual energy off to 3 digits.
* **NI-O1:** unsplit diffusion WITH dual energy (the heat closures require rkl2) does not deliver the AD heat to the dual
  energy: downstream P -76 %, the shock 0.05 behind. Not a production combination; a startup guard is recommended.

## 14. A11 — conductivity gates (PASS)

`src/hydro/diffusion/tests/test_conductivity.py --xcheck-csv runs/A11/xcheck_grains.csv` (`xcheck_conductivity.cpp`
built with host shims; it evaluates the tensor closure) -> `A11_gates.txt`:

| gate | check | result |
|---|---|---|
| 1-2 | sigma_O tensor = scalar conductivity; eta_O = c^2/(4 pi sigma_O) | 4.4e-16, 4.6e-16 |
| 3 | parity: eta_O, eta_A even in B, eta_H odd | 0 |
| 4 | sigma_perp^2 = sigma_H^2 + sigma_P^2 | 0 |
| 5 | eta_A/eta_O crossover along a collapse track | 2.7e6 (low rho) -> 2.7e-3 (high rho) |
| 6 | grain neutrality, ionization balance | 7.0e-14, 4.6e-11 |
| 7 | grains negative, Z ~ a, electron depletion, sublimation | all true |
| 8 | C++ vs independent Python (40 points, 5 bins): n_e, n_i, Z, eta_O / eta_H / eta_A | <= 6.5e-11 / 4.2e-4 / 2.9e-3 |

NICIL (Wurster 2016) not compared (no offline copy).

## 17. Findings

| ID | finding | evidence | status |
|---|---|---|---|
| NI-F2 | the arithmetic face mean of the cached eta is first order at eta jumps (4.4 % at the interface, N 1024, contrast 100) | A5 | FIXED for Ohmic: `diffusion/eta_face_harmonic` (the research code's production setting) |
| NI-F2b | the harmonic mean blocks degenerate AD fronts (eta_A = Q B^2 -> 0) | A6 | AD stays arithmetic (hard-wired) |
| NI-O1 | unsplit diffusion + dual energy loses the AD heating | A10 | open; use rkl2 with the heat closures |
| NI-O2 | AD on a force-free field: spurious change ~(k dx)^2 | A4 | expected truncation, quantified |
| NI-O3 | RKL2 at ratio 1000 does not damp grid-scale modes (16-32 % per cycle) | A7 | expected property; per-cell energy budget exact (A2) |

A further research-code finding concerns the ideal-MHD step rather than this module: the donor-cell (first-order)
VL2 predictor (`hydro/vl2_predictor_recon = dc`) can drive a cell on a sharp low-beta density step to a negative
internal energy in one stage; the PLM predictor (`same`, the default) does not.

## 18. Reproduction

`README.md`: `scripts/make_decks_<T>.py` -> `inputs/<T>/`, `scripts/run_set.sh` (BIN, SET, LIST) -> `runs/<T>/`,
`scripts/analyze_<T>.py --check` -> `results/tables/<T>.{json,txt}`; A2 reads the A7 runs; A9 reads `A1A3.json`;
A11 from the source tree.

## 19. References

Meyer, Balsara & Aslam 2012, MNRAS 422, 2102; 2014, JCP 257, 594 (RKL2). Masson et al. 2012, ApJS 201, 24 (Ohmic test).
Balsara 1996, ApJ 465, 775; Choi, Kim & Wiita 2009, ApJS 181, 413 (AD damped Alfven wave). Mac Low et al. 1995, ApJ 442,
726; Wardle 1991, MNRAS 250, 523 (C-shocks). Barenblatt 1952; Pattle 1959 (porous-medium similarity solution). Carslaw &
Jaeger 1959, Conduction of Heat in Solids (composite media). Umebayashi & Nakano 1990, MNRAS 243, 103; Pandey & Wardle
2008, MNRAS 385, 2269; Wurster 2016, PASA 33, e041 (NICIL). Citations not yet checked against the publisher record are
quoted from memory.
