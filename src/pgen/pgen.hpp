#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//
// Problem generators of the self-gravity + tabulated-EOS tree: `collapse_be` (magnetized
// Bonnor-Ebert sphere collapse) plus the validation problems used by validation/self_gravity
// and validation/eos_tabulated. Adding one means adding its .cpp, declaring its namespace here,
// adding it to pgen/CMakeLists.txt and adding its branch to main.cpp.

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace collapse_be {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
// WP-13: registers the collapse_be_* Params. MUST be wired in main.cpp -- it runs on both a
// fresh start and a restart, whereas ProblemGenerator does NOT run on restart, which silently
// dropped the ApplyBarotropicCooling task (and its outside-sphere momentum BC) on every resume.
void ProblemInitPackageData(parthenon::ParameterInput *pin,
                            parthenon::StateDescriptor *hydro_pkg);
parthenon::TaskStatus ApplyBarotropicCooling(parthenon::MeshData<parthenon::Real> *md,
                                              const parthenon::SimTime &tm,
                                              const parthenon::Real dt);
} // namespace collapse_be

// The self-gravity multipole boundary condition's analytic gate (WS-5a gate (i)).
namespace poisson_test {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace poisson_test

// Radiation (Sn) validation problems (validation/radiation_sn): a static or uniformly moving medium with an
// optional Gaussian / top-hat radiation or temperature profile (rad_pulse), and the Lowrie & Edwards radiative
// shock initial state (rad_shock).
namespace rad_pulse {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace rad_pulse
namespace rad_shock {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace rad_shock

// Non-ideal MHD validation suite (validation/nonideal_mhd): Ohmic Gaussian (iprob 40), circularly
// polarized decay (41), eta jump (42), Barenblatt AD spreading (43), 3-D ABC field (44), ambipolar
// C-shock (45), damped Alfven wave with AD (50) and its eta_A-bump variant (51).
namespace diffusion {
using namespace parthenon::driver::prelude;
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin, parthenon::SimTime &tm);
} // namespace diffusion

// Self-gravity validation suite (validation/self_gravity, 2026-10-03).
namespace grav_analytic {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace grav_analytic
namespace jeans_wave {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace jeans_wave
namespace grav_slab {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace grav_slab
namespace polytrope {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace polytrope
namespace eos_test {
using namespace parthenon::driver::prelude;
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void HomologousInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void HomologousOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
} // namespace eos_test

namespace profile1d {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);
} // namespace profile1d

namespace fofc_scalar {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace fofc_scalar

// Ideal-MHD validation suite (validation/ideal_mhd): linear MHD waves, circularly polarized Alfven wave, field
// loop advection, and shock tubes / Noh / div B tests (ideal_mhd).
namespace linear_wave_mhd {
using namespace parthenon::driver::prelude;
void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin, parthenon::SimTime &tm);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
} // namespace linear_wave_mhd
namespace cpaw {
using namespace parthenon::driver::prelude;
void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin, parthenon::SimTime &tm);
} // namespace cpaw
namespace field_loop {
using namespace parthenon::driver::prelude;
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
} // namespace field_loop
namespace ideal_mhd {
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace ideal_mhd

#endif // PGEN_PGEN_HPP_
