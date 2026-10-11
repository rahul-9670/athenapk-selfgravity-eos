# AthenaPK with self-gravity, a tabulated protostellar EOS, dual energy and radiation transport

A modified copy of [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk) (Parthenon + Kokkos, CPU and GPU) that
contains four validated additions:

* **Self-gravity** (`src/self_gravity/`): multigrid-preconditioned BiCGSTAB Poisson solver over the AMR hierarchy,
  multipole (monopole + quadrupole) isolated boundary conditions, Jeans swindle for periodic boxes, flux-weighted
  energy coupling. Ported from [Artemis](https://github.com/lanl/artemis) (LANL, BSD-3; see `LICENSE-ARTEMIS.md`) and
  modified. Report: [`validation/self_gravity/SELF_GRAVITY_VALIDATION.md`](validation/self_gravity/SELF_GRAVITY_VALIDATION.md).
* **Tabulated protostellar EOS** (`src/eos/`, `<hydro> eos = hydrogen`): multi-Saha H/He with H2 dissociation and
  H2 rotational/vibrational levels, two tables shipped (`TABLE_PROVENANCE.md`), optional double-flux energy update
  for pressure-exact contacts (`hydro/double_flux`). Report:
  [`validation/eos_tabulated/EOS_VALIDATION.md`](validation/eos_tabulated/EOS_VALIDATION.md).
* **Dual energy** (`hydro/dual_energy`): the internal energy is carried as an extra advected variable (with the
  -P div v source and a two-sided switch) and used where E - KE - ME is ill-conditioned, i.e. in cold hypersonic and
  magnetically dominated gas. Report:
  [`validation/dual_energy/DUAL_ENERGY_VALIDATION.md`](validation/dual_energy/DUAL_ENERGY_VALIDATION.md).
* **Radiation transport** (`src/radiation/`, `<physics> radiation = true`, `radiation/transport = sn`): multigroup
  discrete ordinates (Sn) with step / second-order upwind / positivity-limited differencing, BiCGSTAB preconditioned by
  ordered transport sweeps and an optional gray diffusion-synthetic acceleration, implicit Fleck (or coupled Fleck &
  Cummings) matter coupling at the full speed of light, Planck / Rosseland mean opacity tables (`opacity_model =
  tabulated`; tables v9, v11, v12 (Semenov et al. 2003 dust + Helling 2001 gas, 3 and 8 groups) shipped, generator
  `src/radiation/gen_opacity_table.py`). Report:
  [`validation/radiation_sn/RADIATION_VALIDATION.md`](validation/radiation_sn/RADIATION_VALIDATION.md).

Problem generators: `collapse_be` (magnetized Bonnor-Ebert sphere collapse, barotropic or with radiation) and the
validation problems `poisson_test`, `grav_analytic`, `jeans_wave`, `grav_slab`, `polytrope`, `profile1d`, `eos_test`,
`fofc_scalar`, `rad_pulse`, `rad_shock`.
Fluid: `glmmhd` only. With `eos = hydrogen`, `collapse_be`'s barotropic step prescribes the temperature,
T = T0 sqrt(1 + (rho/rhocrit)^(2(gamma-1))), and takes the internal energy from the table at that temperature
(with `eos = adiabatic` it is the usual ideal-gas barotropic energy). The dual energy is not wired into that
barotropic step and is refused for `collapse_be` without radiation. With `<physics> radiation = true` the radiation owns
the gas thermal energy: the barotropic step is skipped (only the outside-sphere momentum condition is kept), the
radiation field is seeded in LTE with the initial gas, and the dual energy is allowed.

## Build
```bash
git clone --recursive <this repo>          # Kokkos and Catch2 are submodules; Parthenon is vendored (external/parthenon_local)
cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON \
      -DPARTHENON_DISABLE_EXAMPLES=ON -DPARTHENON_DISABLE_OPENMP=ON -DPARTHENON_DISABLE_SPARSE=ON
make -C build_cpu athenaPK -j
# GPU (H100): -DCMAKE_CXX_COMPILER=$PWD/external/Kokkos/bin/nvcc_wrapper -DKokkos_ENABLE_CUDA=ON
#             -DKokkos_ARCH_HOPPER90=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON (plus the PARTHENON_* flags above)
```
Needs MPI and parallel HDF5. `-DATHENAPK_EOS_LEGACY_LOOKUPS=ON` builds the pre-fix EOS lookups (see the EOS report, E-F9).
At most 8 radiation groups by default; add `-DATHENAPK_MAX_GROUP=<n>` to `CMAKE_CXX_FLAGS` for more. The radiation
package has no dust-evolution coupling here: the dust multipliers of the opacities are 1 (the tables already contain
the dust and gas opacity).

## Relation to the validated code
The validation reports were produced with a larger private research code from which this tree was extracted. The
extraction was checked against the validated binary on 47 paired runs: identical up to compiler FMA contraction (every
rerun leg bitwise identical when both are compiled with `-ffp-contract=off`); details in either report, Sec. 2.2. The
dual-energy and radiation ports were checked the same way (their reports, Sec. 2.2).

## Licence
BSD 3-Clause (`LICENSE`, AthenaPK/Parthenon); the self-gravity package derives from Artemis (`LICENSE-ARTEMIS.md`).
