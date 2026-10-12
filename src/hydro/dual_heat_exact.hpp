//========================================================================================
// AthenaPK - exact energy closure of the Ohmic/ambipolar heating for the dual energy (DE-F6).
//========================================================================================
#ifndef HYDRO_DUAL_HEAT_EXACT_HPP_
#define HYDRO_DUAL_HEAT_EXACT_HPP_

#include <parthenon/package.hpp>

namespace Hydro {
//! hydro/dual_energy_heat_exact, see dual_heat_exact.cpp. Init after the half-start CalcDiffFluxes + flux correction;
//! Stage after each RKL2StepOther (flux arrays = fluxes of Y_{j-1}); Finish after the last stage's update, before its
//! boundary exchange / FillDerived.
parthenon::TaskStatus DualExactInit(parthenon::MeshData<parthenon::Real> *md, const int s_rkl,
                                    const parthenon::Real tau);
parthenon::TaskStatus DualExactStage(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real mu_j,
                                     const parthenon::Real nu_j, const parthenon::Real mu_tilde_j,
                                     const parthenon::Real gamma_tilde_j, const parthenon::Real tau);
parthenon::TaskStatus DualExactFinish(parthenon::MeshData<parthenon::Real> *md,
                                      parthenon::MeshData<parthenon::Real> *md_y0);
} // namespace Hydro

#endif // HYDRO_DUAL_HEAT_EXACT_HPP_
