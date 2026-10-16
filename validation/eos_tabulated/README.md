# Tabulated EOS — validation suite

Configuration under test: `eos = hydrogen` with `src/eos/eos_table_hires_v5_cold.bin` (md5 `f712d6a8`; multi-Saha
H/He, H2 dissociation, H2 rotation/vibration), HLLD + PLM + VL2, `eos_floor_consistent = true`, 20 P -> e bisections
(default). Problem generator: `eos_test` (`src/pgen/eos_test.cpp`).

**Report: [`EOS_VALIDATION.md`](EOS_VALIDATION.md)** (authoritative). Part A is the implementation guide (physics,
derivations, execution path, every lookup, numerics, units, configuration keys, worked examples); Part B the
validation (every test with numbers, findings E-F1..E-F11, reproduction map); Part C what each test establishes,
limitations, observations O-1..O-7 (fixed in 5649acd, checks in `results/tables/O_fixes_check_2026-10-16.txt`) and
proposed tests. Tables: `results/tables/`, figures: `results/figures/`.

| ID | test | reference | result |
|---|---|---|---|
| E1 | rebuild the table with the documented generator command | byte identity | PASS |
| E2 | table thermodynamics: charge neutrality, element conservation, Maxwell identity, P and cs2 vs direct Saha | round-off / interpolation error | PASS |
| E3 | asymptotic limits: Gamma1 -> 5/3 (cold), 7/5 (warm H2), dips at dissociation/ionization | known limits | PASS |
| E4 | interpolation error structure | — | finding E-F1 |
| E5 | inversions on the real header (host harness `host/`): monotonicity, P -> e bisection, T -> e -> T, edges, floors | analytic bound / identity | PASS (bisection error = bound 8.08e-6); E-F5, E-F6, E-F7 |
| E6 | acoustic waves at the table sound speed, 5 states, N = 16..256 | IC after one period | PASS (phase speed = interpolant's, E-F1) |
| E7 | 4 general-EOS Riemann problems x N = 100..6400 x 2 interpolation modes; contacts; double flux; 2D/3D slabs; refined meshes | exact general-EOS Riemann solver (`scripts/riemann_exact.py`) | PASS; slow R2/R4 convergence explained (E-F10) and fixed opt-in by `hydro/double_flux` |
| E8 | homologous compression 10 K -> 12 500 K | exact adiabat of the interpolant | PASS (<= 4e-7) |
| E9 | GPU (H100) vs CPU | bitwise / round-off | PASS |
| E10 | table footprint of the production runs; floor-visibility probe | clamp and floor counters | E-F5, E-F6 (fixed) |

## Layout
* `inputs/` — decks (`make_decks_eos.py`, `make_decks_eos2.py` regenerate them).
* `runs/*.sh` — the SLURM scripts that ran each test, as run (one node, Kokkos OpenMP; GPU scripts on one H100).
* `scripts/` — analysis, exact solvers, the table reader `eostab.py`, `run_one_nobind.sh`.
* `host/` — host-only harnesses that compile the real `src/eos/eos_table.hpp` with plain g++ (`run_e5.sh` for E5,
  `run_worked.sh` for the worked examples of Part A, output `results/tables/W_worked_examples.txt`; the physics side is
  `scripts/worked_saha.py` -> `results/tables/W_worked_saha.txt`).
* `results/` — tables and figures quoted in the report.

Paths: decks and scripts refer to the repository root as `$REPO` (`export REPO=/path/to/clone`; in decks substitute
it, e.g. `sed -i "s#\$REPO#$REPO#g" inputs/*.in`). `BIN` is an `athenaPK` built from this repository; the run scripts
name the pinned binaries of the original runs (`$REPO/validation/_build/bin/athenaPK_cpu_<md5>`), which are not
shipped. Run scripts `source ~/athenapk_env.sh` (site module setup: GCC 13, Open MPI 5.0.7, parallel HDF5); adapt to
your system.
