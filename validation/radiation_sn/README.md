# Radiation transport (discrete ordinates, Sn) and opacities — validation suite

`<physics> radiation = true` with `radiation/transport = sn`: gray-in-group discrete-ordinates transport (32 ordinates
by default, step / SOU / limited-SOU differencing, BiCGSTAB preconditioned by ordered transport sweeps, optional gray
diffusion-synthetic acceleration), implicit Fleck matter coupling (optionally the coupled Fleck & Cummings solve over
all groups), multigroup, and tabulated Planck / Rosseland mean opacities (`src/radiation/opacity_table_*.bin`, written
by `src/radiation/gen_opacity_table.py`). Problem generators: `rad_pulse` (all R tests except R8b), `rad_shock` (R8b),
`eos_test` (R4 hand-over decks).

**Report: [`RADIATION_VALIDATION.md`](RADIATION_VALIDATION.md)** (every test with numbers, findings, keys, and the
check of the code shipped here against the research code it was taken from, Sec. 2.2).

| ID | test | result |
|---|---|---|
| R0 | the earlier Sn gates: G8 thick equilibrium, G9 coupling, G10 scattering, G11 Fleck, K1/K3/K5 Krylov gates, K4 diffusion limit, FLECK order, RAY ray effects, SEAM refinement seam, two-leg restart; host tests | all reproduce; restart re-seeded the ordinates: **R-F1, FIXED** |
| R1 | 0-D multigroup relaxation vs the exact multigroup ODE (91 runs, stiffness 0.07-1000) | code = its own discrete scheme to <= 2e-6; one-Fleck-factor scattering lag: **R-F3, FIXED** (`sn_fc_scatter`) |
| R2 | Su-Olson non-equilibrium Marshak wave (linearised, +A/-A pairs) | PASS: L1 5.4e-4 vs the code-ordinate reference; 2.1 % vs S64 with 4x8 ordinates |
| R3 | searchlight, crossing beams, shadow (2-D) | PASS after **R-F10 FIXED** (`sn_sou_limiter = positive`) |
| R4 | conservation: the dual-energy D7 hand-over | the 0.29 % loss = the convergence norm: **R-F6, FIXED** (`sn_norm = b`, `sn_local_tol`) |
| R5, R13 | group count / edges (1-D hot core in a cold envelope), ordinate count (3-D) | **R-F13**: 8 groups (X8 edges) within 1.4-3.6 % of a 34-group reference; 32 ordinates within 0.4 % of 288 |
| R5, R13 | solver in a hot, optically thick core | **R-F12, FIXED** (`sn_precond_stencil = sou` + gray DSA `sn_gray_dsa`) |
| R6 | GPU (H100) vs CPU, same source / deck / ranks | identical to round-off or the solver tolerance; iteration counts equal |
| R7 | opacities: provenance (O1), table vs definition (O2), lookup (O3), Semenov et al. 2003 (O5) | **R-F4, R-F5, R-F9, R-F11 FIXED** (tables v10b/v11/v12, `opacity_interp = log`) |
| R8 | Ganapol plane source, Lowrie & Edwards radiative shocks (M 2, 3), 3-D point source | PASS: shock T1 to 0.04 %, spike to 0.3 %; plane-source variance to 0.2 % |
| RF* / WF* | regression lists of the fixes (R-F12 DSA variants, R-F15 boundaries, R-F16 operator check, sweep variants) | see report Sec. 14 |

## Layout
* `inputs/` — decks (`scripts/make_decks_R*.py`) and run lists `<test>_runs.txt` ("rundir | deck | overrides | ranks").
* `scripts/` — runners (`run_list_srun.sh` CPU, `run_list_gpu.sh` one GPU, `run_list.sh`), deck generators, analyses
  (`analyze_R*.py`, `r7_*.py`, each with `run | check`), exact / reference solutions (`r1_model.py`, `su_olson_ref.py`,
  `ganapol_ref.py`, `radshock_ref.py`), the R0 checkers (`r0_checkers/`), restart scripts (`r0_restart*.sh`,
  `cmp_bitwise.py`), opacity-table builds for R5 / R13 (`gen_tables_R*.sh`; they write `runs/R5/tables`,
  `runs/R13/tables`, which the R5 / R13 decks read), `remap_restart_groups.py` (change `n_group` across a restart,
  conservatively), `host/opacity_v4_dump.cpp` (C++ loader of the format-4 tables, for R7).
* `results/tables/` — the tables quoted in the report; `results/figures/` — its figures (`scripts/make_figs.py`).

## Not shipped
* Run directories (`runs/`), the pinned research binaries, and the authors' Semenov et al. (2003) opacity code and data
  used by R7-O5 and by the table v12 build (download them from the authors' web page; point
  `gen_opacity_table.py --semenov_dir` at the directory for `src/radiation/build_opacity_v12*.sh`, and
  `SEMENOV_DIR` for `scripts/r7_o5_semenov.py` / `r7_v12.py`).
* Tests of physics that is not part of this repository (the dust-evolution factor of the opacities, R7-O6 / R-F8, and
  the 3-D collapse-restart gates of the research code). In this repository the dust multipliers of the opacities are
  exactly 1: the opacity table already contains the dust and gas opacity, which is what the research code computes
  with its dust package off. Decks keep the `dust/...` keys of the research runs; they are ignored here.
* Literature (copyrighted text).

Paths: as in `../eos_tabulated/README.md` (`$REPO` = repository root, `export REPO=/path/to/clone`; in decks
substitute it, e.g. `sed -i "s#\$REPO#$REPO#g" inputs/*.in inputs/*/*.in`). `BIN` is an `athenaPK` built from this
repository; the run scripts name the pinned binaries of the original runs (`$REPO/validation/_build/bin/athenaPK_cpu_<md5>`),
which are not shipped. Tests with more than 8 groups need a build with `-DATHENAPK_MAX_GROUP=24` (or 40) in
`CMAKE_CXX_FLAGS`. Table headers that named a generator path now name it relative to the repository root.
`inputs/WF_cpu_runs.txt` / `WF_gpu_runs.txt` were written when `radiation/sn_sweep_wavefront` was a switch; it is now a
mode (0-5) and the lists use mode 1, the schedule the switch selected (`true` no longer parses).
