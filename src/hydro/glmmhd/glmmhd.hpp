#ifndef HYDRO_GLMMHD_GLMMHD_HPP_
#define HYDRO_GLMMHD_GLMMHD_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::GLMMHD {

template <bool extended>
void DednerSource(MeshData<Real> *md, const Real beta_dt);

using SourceFun_t = std::function<void(MeshData<Real> *md, const Real beta_dt)>;

// hydro/glmmhd_damping = midpoint (validation_tests/ideal_mhd IM-F3, 2026-10-12): second-order (exponential
// integrator) psi damping for VL2. md0 = the live register after the stage update, md1 = "u1" = the step-start state.
TaskStatus GLMDampingExponential(MeshData<Real> *md0, MeshData<Real> *md1, const int stage, const Real dt);

} // namespace Hydro::GLMMHD

#endif // HYDRO_GLMMHD_GLMMHD_HPP_
