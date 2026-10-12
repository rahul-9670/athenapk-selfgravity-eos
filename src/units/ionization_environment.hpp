//========================================================================================
// AthenaPK - shared ionization environment (cosmic-ray rate + composition).
//
// A single owner of the ionization parameters (cosmic-ray rate, neutral composition) used by
// the equilibrium ionization model behind Ohmic/Hall/ambipolar_coeff = ionization. It was
// introduced so the (now removed, 2026-09-24) chemistry network and the equilibrium model
// could not disagree on the charge population.
//
// Ownership split vs PhysicalUnits: PhysicalUnits owns *units* (incl. mu_thermal, the
// temperature calibration). IonizationEnvironment owns the *charge microphysics inputs*:
// the cosmic-ray ionization rate and the neutral mean molecular weight mu_n used for
// number densities n = rho/(mu_n m_H). (Keeping mu_thermal and mu_n in separate objects
// is the structural fix for audit finding #4's conflation of the two.)
//
// Room to grow (audit Workstream A.2 / Phase 3): CR attenuation with column, radionuclide
// floor, grain population, thermal-ionization params can move here so every non-ideal
// coefficient draws one composition.
//========================================================================================
#ifndef UNITS_IONIZATION_ENVIRONMENT_HPP_
#define UNITS_IONIZATION_ENVIRONMENT_HPP_

#include <string>

#include <parameter_input.hpp>
#include <parthenon/package.hpp>

#include "basic_types.hpp"

namespace PhysUnits {

using parthenon::Real;

//----------------------------------------------------------------------------------------
//! Shared charge-microphysics environment. POD, trivially copyable.
struct IonizationEnvironment {
  Real zeta_cr_cgs = 1.0e-16; // cosmic-ray ionization rate [s^-1] (single source of truth)
  Real mu_n = 2.33;           // neutral mean molecular weight (n = rho/(mu_n m_H))
};

//----------------------------------------------------------------------------------------
//! Build the environment from the input deck (<diffusion> ion_zeta / ion_mu_n). Deterministic
//! in `pin`, so every consumer obtains identical values from independent Initialize() calls.
//! (Before 2026-09-24 the chemistry package could own the rate; chemistry was removed.)
inline IonizationEnvironment BuildIonizationEnvironment(parthenon::ParameterInput *pin) {
  IonizationEnvironment env;
  env.zeta_cr_cgs = pin->GetOrAddReal("diffusion", "ion_zeta", 1.0e-17);
  env.mu_n = pin->GetOrAddReal("diffusion", "ion_mu_n", 2.33);
  return env;
}

} // namespace PhysUnits

#endif // UNITS_IONIZATION_ENVIRONMENT_HPP_
