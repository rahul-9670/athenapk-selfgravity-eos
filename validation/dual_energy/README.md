# Dual energy — validation suite

`hydro/dual_energy` carries the internal energy rho e as an extra advected variable and uses it instead of
E - KE - ME where that difference is untrustworthy (cold hypersonic or magnetically dominated gas).

Configuration under test (`scripts/de_common.py`): `dual_energy_eta1 = 5e-2`, two-sided switch (`eta2 = 0.1`,
`kappa = 3`, `shock_comp = 0.02`), `exact_resync`, `unsigned_prim`, `sync_step_end`, `dual_comp_ghost`,
`boundary_clamp_dual`, on top of the tabulated EOS with double flux. Problem generator: `eos_test`.

**Report: [`DUAL_ENERGY_VALIDATION.md`](DUAL_ENERGY_VALIDATION.md)** (every test with numbers, findings, keys).

| ID | test | result |
|---|---|---|
| D1 | cold hypersonic contact (D1a) and velocity sinusoid (D1b), Mach 1..1e4 | PASS (smooth flow); contacts: DE-F3, fixed by opt-in keys |
| D2, D4 | low-beta sinusoid and contact, 2D div B blob, homologous compression | PASS (compression on the adiabat to 3e-6 vs 0.76 off) |
| D3 | planar Noh, M 3/10/30 | PASS (star state to <= 3.4e-4) |
| D6 | eta1 / eta2 thresholds | eta1 = 5e-2 recommended |
| D8 | GPU vs CPU, restarts | PASS (<= 1.2e-12; restarts bitwise) |
| T1, T2, T3, F1-F4, X2, G | refined meshes, 3D, boundary clamp, opt-in keys, GPU long rows | see report Sec. 12.2 |

## Layout
* `inputs/` — decks (`scripts/make_decks_D*.py`) and run lists `<test>_runs.txt` ("rundir | deck | overrides | ranks").
* `scripts/` — `run_list.sh` (runs a list), deck generators, analyses, GPU scripts.
* `results/tables/` — the tables quoted in the report.

Rows and run-list lines of tests that need physics not in this repository were removed from the shipped tables and
lists; the analysis scripts of the gap-closure tables also covered such tests and are not shipped (the tables are).
Paths: as in `../eos_tabulated/README.md` (`$REPO` = repository root; `BIN` = an `athenaPK` built from this repository).
