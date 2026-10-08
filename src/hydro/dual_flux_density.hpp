//========================================================================================
// AthenaPK - dual-energy flux from the reconstructed internal-energy density (DE-F3).
//========================================================================================
#ifndef HYDRO_DUAL_FLUX_DENSITY_HPP_
#define HYDRO_DUAL_FLUX_DENSITY_HPP_

#include <parthenon/package.hpp>

namespace Hydro {
//! hydro/dual_energy_flux = density: overwrite the dual-energy face fluxes with F_mass * q_face / rho_face
//! (q = rho e, both PLM on the upwind side). Call right after a PLM CalculateFluxes, before the flux
//! correction is sent. See dual_flux_density.cpp.
parthenon::TaskStatus DualEnergyDensityFlux(parthenon::MeshData<parthenon::Real> *md);
//! hydro/dual_energy_pdv = face: -P div v of the dual energy from the face velocities F_mass/rho_f (experiment F3).
void DualEnergyPdVFace(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real beta_dt);
} // namespace Hydro

#endif // HYDRO_DUAL_FLUX_DENSITY_HPP_
