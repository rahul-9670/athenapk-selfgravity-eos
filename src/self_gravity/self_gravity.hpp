//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Self-gravity package, ported from Artemis (LANL).
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
#ifndef SELF_GRAVITY_SELF_GRAVITY_HPP_
#define SELF_GRAVITY_SELF_GRAVITY_HPP_

#include <memory>
#include <string>

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

#include <parthenon/driver.hpp>
using namespace parthenon::driver::prelude;

namespace SelfGravity {

// Field types — use the VARIABLE macro pattern from Parthenon's poisson_package.hpp
// "grav.phi" and "grav.rhs" are the HDF5 field names you'll see in outputs.
#define SG_VARIABLE(ns, varname)                                                         \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace grav {
SG_VARIABLE(grav, phi);
SG_VARIABLE(grav, rhs);
} // namespace grav

// Package registration — called from Hydro::ProcessPackages
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// FillDerived function: computes rhs = 4piG * (rho - rho_mean) on every cell
// including ghosts (ghosts so the solver's boundary logic sees the right values).
void FillPoissonRHS(MeshData<Real> *md);

// P2 (2026-08-13): driver-set gate telling FillPoissonRHS whether the grav::rhs it is about to
// write will be read by a Poisson solve before the next FillDerived overwrites it. When false
// the global reductions (Jeans-swindle mean density, multipole moments) and their MPI_Allreduce
// are skipped; the RHS itself is still written IN FULL over IndexDomain::entire, so no stale
// rhs can survive into a consumed call. MUST be set from the driver task graph so that it is
// rank-uniform -- a rank-dependent skip would deadlock the collective.
// Default is TRUE, i.e. a driver that never calls this is bit-identical to pre-P2.
parthenon::TaskStatus SetGravRhsConsumed(parthenon::Mesh *pm, bool consumed);

// Apply gravitational acceleration to momentum and energy using phi.
// Flux-weighted energy update (Artemis style) for better AMR energy conservation.
TaskStatus ApplyGravitySource(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real beta_dt);

// Build and submit the Poisson solve into the task collection.
// Called from HydroDriver::MakeTaskCollection on the final stage.
void SolvePoisson(TaskCollection &tc, Mesh *pmesh);
//! Same, but added to an EXISTING TaskRegion so an independent chain can share the task lists
//! and run during this one's Allreduce waits. See the long note at the definition.
void SolvePoissonInRegion(parthenon::TaskRegion &region, Mesh *pmesh);

} // namespace SelfGravity

#endif // SELF_GRAVITY_SELF_GRAVITY_HPP_
