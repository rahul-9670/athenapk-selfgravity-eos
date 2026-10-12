# Non-ideal MHD (Ohmic + ambipolar diffusion) — validation suite

`<diffusion>` block: Ohmic resistivity (`resistivity = ohmic`) and ambipolar diffusion (`ambipolar = ambipolar`),
integrated unsplit (`integrator = unsplit`) or with second-order Runge-Kutta-Legendre super-time-stepping
(`integrator = rkl2`, ratio cap `rkl2_max_dt_ratio`, optional `rkl2_freeze_eta`), with constant coefficients
(`*_coeff = fixed`; ambipolar eta_A = Q B^2) or from the equilibrium ionization model (`*_coeff = ionization`: cosmic-ray
+ thermal (K, H) ionization, MRN grains, Pandey & Wardle conductivity tensor, single-fluid or tensor ambipolar closure,
optional ceilings `eta_ohm_cap_code`, `eta_ad_cap_code`), a cell-centred eta cache (`eta_cache`) with the face average
`eta_face_harmonic` (Ohmic faces), and the hand-over of the Joule / ambipolar heat to the dual energy
(`hydro/dual_energy_heat_rkl2`, `hydro/dual_energy_heat_exact`). Problem generator: `diffusion` (iprob 40-45, 50, 51).
Hall, thermal conduction and viscosity are not part of this repository.

**Report: [`NONIDEAL_MHD_VALIDATION.md`](NONIDEAL_MHD_VALIDATION.md)** (every test with numbers, findings, keys, and the
check of the code shipped here against the research code it was taken from, Sec. 2.2).

| ID | test | result |
|---|---|---|
| A7 | RKL2 amplification of single Fourier modes, super-step ratio 1-1000 | PASS: = the Legendre stability polynomial R_s(z)^2 to <= 3.4e-7; at ratio 1000 grid-scale modes keep 16-32 % per cycle |
| A2 | energy budget of the super-step | PASS: total E to 1.6e-16; per-cell Joule heat = magnetic energy lost to 7e-13 where the hydro is inert |
| A1 | Ohmic diffusion of a Gaussian | PASS: 2nd order, unsplit = RKL2 to 0.3 % |
| A3 | damped Alfven eigenmode with AD, aligned and 45 deg | PASS: 2nd order, 3e-5 at N 512 |
| A4 | 3-D ABC (Beltrami) field: Ohmic decay, AD null | PASS: 2nd order (AD null 0.42 % at 64 cells per wavelength) |
| A5 | Ohmic diffusion across an eta jump | **NI-F2**: arithmetic face mean first order (4.4 % at the interface); `eta_face_harmonic` restores 2nd order |
| A6 | nonlinear AD spreading (Barenblatt-Pattle) | PASS; `rkl2_freeze_eta` at ratio 1000 adds no error; harmonic face mean must not be used for AD |
| A9 | Ohmic Gaussian across a coarse-fine boundary | PASS: 2nd order |
| A10 | ambipolar C-shock vs the steady single-fluid profile | PASS: profile 1e-4, downstream state 1e-3 |
| A11 | ionization model and conductivity tensor gates 1-8 | PASS (C++ vs an independent Python solve: eta_O 6.5e-11, eta_A 2.9e-3) |

## Layout
* `inputs/<test>/` — decks (written by `scripts/make_decks_<test>.py`) and `runs.txt` (one deck name per line);
  `inputs/base/` — the two base decks of A1/A3.
* `scripts/` — `run_set.sh` (runs one test set), deck generators and their templates (`A5/A6/A7_template.in`),
  analyses (`analyze_<test>.py`, each with `--check`), the C-shock reference (`cshock_ref.py`).
* `results/tables/` — the tables quoted in the report.
* A11 runs from the source tree: `src/hydro/diffusion/tests/test_conductivity.py` (gates 1-7) and, after building
  `xcheck_conductivity.cpp` with the host shims, `--xcheck-csv` (gate 8).

## Reproduce
```bash
BIN=/path/to/build/bin/athenaPK SET=A7 bash scripts/run_set.sh        # -> runs/A7/...
python3 scripts/analyze_A7.py --check && python3 scripts/analyze_A2_budget.py --check
```
A5's harmonic arm: `SET=A5 LIST=runs_H.txt`; A10's no-dual-energy and long-box arms: `LIST=runs_nd.txt`,
`LIST=runs_long.txt`.

## Not shipped
Run directories (`runs/`) and the pinned research binaries.
