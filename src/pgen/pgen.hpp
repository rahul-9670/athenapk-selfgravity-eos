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

// Self-gravity validation suite (validation_tests/self_gravity, 2026-10-03).
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

#endif // PGEN_PGEN_HPP_
