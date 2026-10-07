# Self-gravity — validation suite

Configuration under test: BiCGSTAB preconditioned by FAS multigrid (SRJ2 smoothers), `relative_residual = true`,
`residual_tolerance = 1e-6`, warm start from the previous potential, multipole (monopole + quadrupole, shell-split)
boundary conditions on all six faces, `phi_operator_split`, linear phi prolongation, gravity source = rho g momentum +
flux-weighted energy work, coupled to glmmhd + HLLD + PLM + VL2. The gravity keys are in
`inputs/_grav_production.inc`, included by every deck; a test departs from it only where the problem requires
(periodic box: Jeans swindle + periodic gravity BCs).

**Report: [`SELF_GRAVITY_VALIDATION.md`](SELF_GRAVITY_VALIDATION.md)** (algorithm, every test with numbers, findings
SG-F1..SG-F7, reproduction map). Tables: `results/tables/`, figures: `results/figures/`, 1D reference solutions:
`results/ref1d/`.

| group | tests | result |
|---|---|---|
| A. Poisson solve | S1 periodic sinusoid, S2 Gaussian, S3/S3a uniform sphere, S4 binary on 4 AMR levels, S5 Maclaurin spheroid, S6 Plummer, S7 collapse IC, S8 MPI invariance, S9 solver controls | PASS (2nd order) |
| B. Gravity + hydro | D1/D2 Jeans waves, D3 dispersion relation, D4 advected slab, D5/D6 polytropes, D7 pressure-free collapse | PASS |
| C. Deep AMR | DA1 6 static levels, DA2 free fall through 4 levels, DA3 = L7 collapse to AMR level 13 vs exact 1D | PASS |
| D. GPU | Poisson, deep AMR, Jeans, free fall, adaptive collapse | PASS (GPU = CPU to 1e-13) |
| E. Literature | L1 Mullen et al. polytrope + atmosphere, L2 Spitzer sheet, L3 n = 5 polytrope, L4 Lin-Mestel-Shu, L5 Evrard, L6 Bonnor-Ebert stability, L7 Larson-Penston, L8 SIS, L9 Boss-Bodenheimer, L10 Hennebelle et al. 2011 | PASS L1-L8 static; L9/L10 qualitative (no sink particles) |
| F. Final configuration | S1, D1, D2, D4, D7 with the final defaults | PASS |

## Layout
* `inputs/` — decks (`scripts/make_decks.py`; literature initial states in `inputs/tables/` from `scripts/make_tables.py`).
* `runs/*.sh` — the SLURM scripts that ran each test (`suite_*`, `lit_*`, `chain_*` for restart chains), as run.
* `scripts/` — `run_one.sh`, `suite_common.sh` (MPI settings incl. the SG-F7 fix), analyses, exact potentials,
  the 1D Lagrangian reference code (`ref1d.py`, `lagr1d.py`).
* `results/` — tables, figures and 1D references quoted in the report.

Paths: as in `../eos_tabulated/README.md` (`$REPO` = repository root; `BIN` = an `athenaPK` built from this
repository).
