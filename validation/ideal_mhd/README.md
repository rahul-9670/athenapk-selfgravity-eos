# Ideal MHD (HLLD + PLM + VL2, GLM/Dedner divergence cleaning, boundary treatment) — validation suite

The MHD scheme of the collapse problem: HLLD fluxes (`riemann = hlld`), PLM reconstruction, the VL2 predictor-corrector
with the PLM predictor, GLM divergence cleaning (`glmmhd_source = dedner_extended`, `glmmhd_alpha = 0.1`), diode
boundaries with the boundary flux clamp, and the interfaces of the MHD step to the dual-energy selector (shocks) and the
tabulated EOS (waves). Problem generators: `linear_wave_mhd`, `cpaw`, `field_loop`, `ideal_mhd` (shock tubes, Noh,
div-B pulse) and `eos_test` (`wave_fast`). Three defects were found and fixed behind default-off keys:
`hydro/dual_energy_shock_eta1` + `hydro/dual_energy_shock_detector = kv` (IM-F1), `hydro/boundary_flux_clamp_wall` +
`hydro/diode_energy_consistent` (IM-F2), `hydro/glmmhd_damping = midpoint` (IM-F3).

**Report: [`IDEAL_MHD_VALIDATION.md`](IDEAL_MHD_VALIDATION.md)** (every test with numbers, findings IM-F1..IM-F6, keys,
and the check of the code shipped here against the research code it was taken from, Sec. 2.4).

| ID | test | result |
|---|---|---|
| I1 | linear MHD waves (fast, Alfven, slow, entropy), 1D/2D/3D | PASS: 2nd order (1D 2.01-2.04, 3D N 128 1.93-1.98) |
| I1c | linear waves through the tabulated EOS | = E-F1: 5.6 % phase floor with `eos_interp = p`, 0.26 % with `p_over_rho` |
| I2 | circularly polarised Alfven wave | PASS: amplitude loss 3rd order, phase 2nd order |
| I3 | MHD Riemann problems vs a new exact solver (`scripts/exact_mhd_riemann.py`) | PASS: L1 order 0.83-1.05 |
| I4 | field-loop advection 2D/3D | PASS: ME 0.946 at N 256; mean div B order ~1.7 |
| I5 | GLM div-B pulse vs the telegraph equation (+ SMR) | PASS; **IM-F3** damping 1st order in dt -> fixed (2nd order); IM-F4 coarse-level over-damping documented |
| I6 | Noh 1D/2D/3D (Mach 8-775), tabulated Noh | **IM-F1**: dual energy under-heated strong shocks (+0.8 %, non-converging) -> fixed (<= 5e-4, converging) |
| I6x2 | dual-energy hand-over X2 | CLOSED (dc predictor; 0 Tfloor hits with the PLM predictor) |
| I7 | outgoing waves and gas at rest at the diode + clamp boundary | **IM-F2**: the clamp removed the outside pressure (rho -4.8 %) -> fixed (wall state to 1e-5 .. 2e-3) |
| I6d*, I8* | the fixes and div B on a 3-D production collapse state | measured (research code) |

\* results of the research code on a 3-D production collapse state (report Sec. 9.3, 11, 12.4): the state, its decks
and the run scripts are not shipped, so these cannot be re-run from this repository.

## Layout
* `inputs/<test>/` — decks (written by `scripts/make_decks_<test>.py`); `inputs/*.txt` — run lists, one line per run:
  `rundir | deck | overrides | nranks`.
* `scripts/` — `run_list.sh` (runs a run list), deck generators (`im_common.py` holds the shared deck pieces),
  analyses (`analyze_<test>.py`, each with `--check`), the exact MHD Riemann solver, `make_figures.py`.
* `results/tables/` — the tables quoted in the report; `results/fig_*.png` — its figures.

## Running
Paths: as in `../eos_tabulated/README.md` (`$REPO` = repository root, `export REPO=/path/to/clone`; in decks substitute
it, e.g. `sed -i "s#\$REPO#$REPO#g" inputs/*/*.in`). Then e.g.

    BIN=$REPO/build/bin/athenaPK LIST=I1_runs.txt NPAR=8 bash scripts/run_list.sh
    python3 scripts/analyze_I1.py --check

`analyze_I6c.py` / `analyze_I6x2.py` use helpers from `../dual_energy/scripts/` (the tabulated Noh decks are the
dual-energy module's D3 decks, copied to `inputs/I6/` with their `D3_meta.json`).
