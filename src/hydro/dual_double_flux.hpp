//========================================================================================
// AthenaPK - double flux for the dual energy (DE-F3 part (b)).
//========================================================================================
#ifndef HYDRO_DUAL_DOUBLE_FLUX_HPP_
#define HYDRO_DUAL_DOUBLE_FLUX_HPP_

#include <parthenon/package.hpp>

namespace Hydro {
//! hydro/dual_energy_double_flux, see dual_double_flux.cpp. Call after DoubleFluxCorrect (same stage, same beta_dt).
//! Face volume fluxes F_mass/rho_f and F_mass P_f/rho_f (upwind face states; dc = donor-cell stage) into the flux
//! slots of dual_vflux. Call after the hydro fluxes, before the flux correction is sent.
parthenon::TaskStatus DualFaceVolumeFlux(parthenon::MeshData<parthenon::Real> *md, const bool dc);
parthenon::TaskStatus DualDoubleFluxCorrect(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real beta_dt,
                                            const int stage);
} // namespace Hydro

#endif // HYDRO_DUAL_DOUBLE_FLUX_HPP_
