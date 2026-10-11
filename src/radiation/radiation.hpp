//========================================================================================
// AthenaPK - radiation package interface. Discrete-ordinates (Sn) multigroup transport with
// Fleck matter coupling (radiation_sn.*), registration (radiation.cpp), task assembly and
// diagnostics (radiation_tasks.cpp). The moment fields rad.Er / rad.Fr1..3 (per group) are the
// Sn writeback of the converged intensities (outputs + restart).
//========================================================================================
#ifndef RADIATION_RADIATION_HPP_
#define RADIATION_RADIATION_HPP_

#include <memory>
#include <string>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;
using namespace parthenon::driver::prelude;

namespace Radiation {

// Field-name macro (mirrors SelfGravity's SG_VARIABLE). HDF5 field names are
// "rad.Er", "rad.Fr1", ...
#define RAD_VARIABLE(varname)                                                            \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return "rad." #varname; }                                \
  }

namespace rad {
RAD_VARIABLE(Er);   // radiation energy density (lab frame), code units
RAD_VARIABLE(Fr1);  // radiation flux, x1-component
RAD_VARIABLE(Fr2);  // radiation flux, x2-component
RAD_VARIABLE(Fr3);  // radiation flux, x3-component
} // namespace rad

// Package factory -- called from Hydro::ProcessPackages when <physics> radiation = true.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

//! Per-group extinction chi_g = rho*(kappa_R,g + kappa_s)*kdust for the rslaMax monitor
//! (radiation_tasks.cpp). No-op with rsla_diag = false or sn_skip_extinction = true.
TaskStatus ComputeRadExtinction(MeshData<Real> *md);
//! RSLA validity monitor, max |v| tau_loc / chat (`rslaMax` history column).
Real RslaMaxReport(MeshData<Real> *md);
//! Radiation history reducers: total and dense-gas Er, dense volume, peak |Fr|.
Real RadErTotal(MeshData<Real> *md);
Real RadErDense(MeshData<Real> *md);
Real RadVolDense(MeshData<Real> *md);
Real RadFrMax(MeshData<Real> *md);

//! Operator-split discrete-ordinates radiation for the final stage (radiation_tasks.cpp).
void AddRadiationTasks(TaskCollection &tc, Mesh *pmesh, const Real dt);

} // namespace Radiation

#endif // RADIATION_RADIATION_HPP_
