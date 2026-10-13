# AthenaPK with self-gravity, a tabulated protostellar EOS, dual energy, radiation transport and non-ideal MHD

A modified copy of [AthenaPK](https://github.com/parthenon-hpc-lab/athenapk) (Parthenon + Kokkos, CPU and GPU) that
contains six validated additions:

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
* **Non-ideal MHD** (`src/hydro/diffusion/`, `<diffusion>` block): Ohmic resistivity and ambipolar diffusion,
  integrated unsplit or with RKL2 super-time-stepping (ratio cap, frozen-eta option), with constant coefficients or from
  an equilibrium ionization model (cosmic rays with column attenuation, thermal K/H ionization, MRN grains, Pandey &
  Wardle conductivity tensor, single-fluid or tensor ambipolar closure), a cell-centred eta cache with a harmonic Ohmic
  face average (`eta_face_harmonic`), and the Joule / ambipolar heat handed to the dual energy (`hydro/dual_energy_heat_rkl2`,
  `hydro/dual_energy_heat_exact`). Hall, thermal conduction and viscosity are not included. Report:
  [`validation/nonideal_mhd/NONIDEAL_MHD_VALIDATION.md`](validation/nonideal_mhd/NONIDEAL_MHD_VALIDATION.md).
* **Ideal-MHD validation and fixes** (HLLD + PLM + VL2, GLM/Dedner cleaning, diode + clamp boundaries): linear waves,
  circularly polarised Alfven wave, MHD Riemann problems against a new exact solver, field loop, GLM telegraph test, Noh,
  boundary tests; three fixes as default-off keys: `hydro/dual_energy_shock_eta1` + `hydro/dual_energy_shock_detector =
  kv` (dual energy under-heated strong shocks), `hydro/boundary_flux_clamp_wall` + `hydro/diode_energy_consistent` (the
  boundary clamp removed the outside pressure), `hydro/glmmhd_damping = midpoint` (second-order psi damping). Report:
  [`validation/ideal_mhd/IDEAL_MHD_VALIDATION.md`](validation/ideal_mhd/IDEAL_MHD_VALIDATION.md).

Problem generators: `collapse_be` (magnetized Bonnor-Ebert sphere collapse, barotropic or with radiation) and the
validation problems `poisson_test`, `grav_analytic`, `jeans_wave`, `grav_slab`, `polytrope`, `profile1d`, `eos_test`,
`fofc_scalar`, `rad_pulse`, `rad_shock`, `diffusion` (non-ideal MHD tests), `linear_wave_mhd`, `cpaw`, `field_loop`,
`ideal_mhd` (ideal-MHD tests).
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
dual-energy, radiation and non-ideal MHD ports were checked the same way (their reports, Sec. 2.2), the ideal-MHD port
in its report, Sec. 2.4.

## Licence
BSD 3-Clause (`LICENSE`, AthenaPK/Parthenon); the self-gravity package derives from Artemis (`LICENSE-ARTEMIS.md`).
