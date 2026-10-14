# Dust evolution (grain growth, irreversible sublimation, dust scalars, dust -> conductivity coupling) — validation suite

`<physics> dust = true`, `<dust> evolve = true`: two passive scalars per cell (`hydro/nscalars >= 2`, `dust/scalar_index`)
carry the dust-to-gas ratio f_dg and the grain radius a_c. `ReactDust` (once per cycle, final stage) applies irreversible
sublimation toward f_ref s(T) at fixed grain number and monodisperse sweep-up growth (Brownian + turbulent relative
velocity, RK2 sub-cycled). The scalars multiply the radiation opacities (`src/radiation/radiation_opacity.hpp`
`DustOpacityFactors`) and, with `diffusion/dust_coupling = true`, set the grain population of the ionization /
conductivity model (`src/hydro/diffusion/diffusion.hpp` `CellIon`). Keys from this validation (all default off):
`dust/subl_tau_s` (finite sublimation time, DU-F1), `dust/subl_model = semenov` + `dust/subl_width_K` (sublimate at the
opacity table's T5(rho), DU-F2), `dust/growth_prefactor` (DU-F3), `diffusion/dust_destroyed_zero` (DU-F4),
`dust/subl_shock_hold`, `dust/subl_T_filter` (measured, not recommended).

**Report: [`DUST_VALIDATION.md`](DUST_VALIDATION.md)** (every test with numbers, findings DU-F1..DU-F8, keys, and the
check of the code shipped here against the research code it was taken from, Sec. 2.3).

| ID | test | result |
|---|---|---|
| D0 | host gate `src/dust/tests/test_dust.cpp`, 17 analytic sub-gates (CTest option `ATHENAPK_DUST_HOST_TESTS`) | PASS |
| D1 | grain growth in the code vs the ODE of the same law, 5 states, 5 dt regimes | PASS: -3.2e-5 resolved; -1.55e-3 sub-cycled at growth_cap 0.1 (2nd order in the cap) |
| D2 | the growth law vs the literature (Birnstiel et al. 2012; Ormel & Cuzzi 2007) | **DU-F3**: prefactor 1/4 vs 1/2 .. 1 -> key `dust/growth_prefactor` |
| D3 | one-zone sublimation (heat through H2 dissociation; cool) | PASS: f = f_ref s(T) to 2.9e-13; irreversible |
| D4 | dust fronts through moving slabs; restart | PASS: budget <= 8e-12, restart bit-identical; **DU-F5** contact mixing across T_subl destroys dust (convergent, order 0.7) |
| D5 / F2 | Noh with T_subl = T2 | **DU-F1**: the post-shock overshoot is recorded (-2.9 % / -1.8 % at every N) -> `dust/subl_tau_s` converges it |
| D6 | dust field through refine/derefine events (2D AMR) | PASS: budget 3e-14; AMR adds no loss |
| D7 | diode + clamp boundary, oblique through-flow | PASS: 4.4e-16 |
| F5 | `subl_model = semenov` one-zone | PASS: 2.1e-12 per cell |
| D8* | dust -> conductivity coupling cell by cell; destroyed grains | PASS 9.4e-13; **DU-F4** one-cycle runaway without `dust_destroyed_zero` |
| D9* | dust census of a production collapse run | T_max 129 K; DU-F6 fossil core deficit <= 1.7e-6 |
| D10* | GPU (GH200) vs CPU on the decks of this suite | PASS: <= 2.2e-12 (smooth), Noh deficits to 2.1e-8 |

\* results of the research code on its 3-D production collapse state or GPU builds (report Sec. 10-12): not re-runnable
from this repository.

## Layout
* `inputs/<test>/` — decks (written by `scripts/make_decks_<test>.py`, shared pieces in `scripts/du_common.py`) and
  `*_meta.json` (the exact answers the analyses use); `inputs/*.txt` — run lists, one line per run:
  `rundir | deck | overrides | nranks`.
* `scripts/` — `run_list.sh` (runs a run list), `run_D4d.sh` (two-leg restart test), deck generators, analyses
  (`analyze_<test>.py`, each with `--check`; `D2_sensitivity.py`), `make_figures.py`, `bitcmp_phdf.py`.
* `results/tables/` — the tables quoted in the report; `results/figures/` — its figures.

## Running
Paths: as in `../eos_tabulated/README.md` (`$REPO` = repository root, `export REPO=/path/to/clone`; in decks substitute
it, e.g. `sed -i "s#\$REPO#$REPO#g" inputs/*/*.in`). Then e.g.

    BIN=$REPO/build/bin/athenaPK LIST=D1_runs.txt NPAR=8 bash scripts/run_list.sh
    python3 scripts/analyze_D1.py --check

The host gate needs no build of the code: `g++ -O2 -std=c++17 -o test_dust $REPO/src/dust/tests/test_dust.cpp && ./test_dust`.
