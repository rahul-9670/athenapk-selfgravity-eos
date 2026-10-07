# AthenaPK with self-gravity and a tabulated protostellar EOS

A modified copy of [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk) (Parthenon + Kokkos, CPU and GPU) that
contains two validated additions:

* **Self-gravity** (`src/self_gravity/`): multigrid-preconditioned BiCGSTAB Poisson solver over the AMR hierarchy,
  multipole (monopole + quadrupole) isolated boundary conditions, Jeans swindle for periodic boxes, flux-weighted
  energy coupling. Ported from [Artemis](https://github.com/lanl/artemis) (LANL, BSD-3; see `LICENSE-ARTEMIS.md`) and
  modified. Report: [`validation/self_gravity/SELF_GRAVITY_VALIDATION.md`](validation/self_gravity/SELF_GRAVITY_VALIDATION.md).
* **Tabulated protostellar EOS** (`src/eos/`, `<hydro> eos = hydrogen`): multi-Saha H/He with H2 dissociation and
  H2 rotational/vibrational levels, two tables shipped (`TABLE_PROVENANCE.md`), optional double-flux energy update
  for pressure-exact contacts (`hydro/double_flux`). Report:
  [`validation/eos_tabulated/EOS_VALIDATION.md`](validation/eos_tabulated/EOS_VALIDATION.md).

Problem generators: `collapse_be` (magnetized Bonnor-Ebert sphere collapse, barotropic) and the validation problems
`poisson_test`, `grav_analytic`, `jeans_wave`, `grav_slab`, `polytrope`, `profile1d`, `eos_test`, `fofc_scalar`.
Fluid: `glmmhd` only. With `eos = hydrogen`, `collapse_be`'s barotropic step prescribes the temperature,
T = T0 sqrt(1 + (rho/rhocrit)^(2(gamma-1))), and takes the internal energy from the table at that temperature
(with `eos = adiabatic` it is the usual ideal-gas barotropic energy).

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

## Relation to the validated code
The validation reports were produced with a larger private research code from which this tree was extracted. The
extraction was checked against the validated binary on 47 paired runs: identical up to compiler FMA contraction (every
rerun leg bitwise identical when both are compiled with `-ffp-contract=off`); details in either report, Sec. 2.2.

## Licence
BSD 3-Clause (`LICENSE`, AthenaPK/Parthenon); the self-gravity package derives from Artemis (`LICENSE-ARTEMIS.md`).
