//========================================================================================
// AthenaPK - host-side construction of the shared ionization model.
//
// Split out of hydro.cpp on 2026-08-11 so a second consumer (the chemistry package, removed
// 2026-09-24) could build the same object from the same <diffusion> ion_* keys.
//
// Kept OUT of ionization.hpp deliberately: that header is device-pure (Kokkos + basic_types
// only) and pulling ParameterInput into it would drag the parthenon input machinery into
// every kernel translation unit.
//========================================================================================
#ifndef HYDRO_DIFFUSION_IONIZATION_BUILD_HPP_
#define HYDRO_DIFFUSION_IONIZATION_BUILD_HPP_

#include <iostream>
#include <string>

#include <parthenon/package.hpp>

#include "../../units/ionization_environment.hpp"
#include "../../units/physical_units.hpp"
#include "ionization.hpp"

namespace Ionization {

//----------------------------------------------------------------------------------------
//! Build the reduced CR+thermal+grain ionization model (NICIL-class, MRN charged grains)
//! shared by the ambipolar / Ohmic / Hall non-ideal terms when their coefficient is
//! "ionization". Reads <diffusion> ion_* keys; all default
//! to the shared FHC code-unit calibration. The MRN grain bins are filled host-side here.
inline IonizationModel BuildIonizationModel(parthenon::ParameterInput *pin) {
  IonizationModel ion;
  // Phase 1: unit conversions come from the single authoritative unit system
  // (was: ion_rho_unit_cgs/ion_T_unit_K/ion_B_unit_G/ion_eta_unit hardcoded defaults here,
  // duplicating the radiation package). B_unit now the exact sqrt(4*pi*rho)*v (audit #3); eta_unit = time/length^2.
  const auto U = PhysUnits::BuildPhysicalUnits(pin);
  ion.rho_unit = U.rho_unit;
  ion.T_unit = U.temperature_unit;
  ion.B_unit = U.magnetic_unit;
  ion.eta_unit = U.diffusivity_unit;
  // Cosmic-ray rate + neutral composition from the shared IonizationEnvironment.
  const auto env = PhysUnits::BuildIonizationEnvironment(pin);
  ion.zeta = env.zeta_cr_cgs;
  ion.mu_n = env.mu_n;
  ion.m_ion = pin->GetOrAddReal("diffusion", "ion_m_ion", ion.m_ion);
  ion.alpha0 = pin->GetOrAddReal("diffusion", "ion_alpha0", ion.alpha0);
  ion.sigma_en = pin->GetOrAddReal("diffusion", "ion_sigma_en", ion.sigma_en);
  ion.sigv_in = pin->GetOrAddReal("diffusion", "ion_sigv_in", ion.sigv_in);
  // Audit P-05: 1.7333 = (4/3) x delta_gn(1.3) reproduces the NICIL / Wardle & Ng rate.
  ion.grain_drag = pin->GetOrAddReal("diffusion", "ion_grain_drag_factor", 1.0);
  ion.gamma_AD = pin->GetOrAddReal("diffusion", "ion_gamma_AD", ion.gamma_AD);
  // --- CR attenuation with column. OFF by default => bit-identical. -----------------
  // `ion.zeta` above is the UNSHIELDED (diffuse-ISM) rate; ZetaLocal supplies the shielded
  // rate per cell. `spectrum` is the realistic model and attenuates the envelope as well as
  // the core; `exponential` is the legacy single-exponential, inert below Sigma_0, kept only
  // so the first-pass result stays reproducible. See ionization.hpp for the law and for an
  // explicit statement of how well the default coefficients are known.
  {
    // AUDIT F-34 (2026-08-28) -- PROVENANCE OF THE ATTENUATION COEFFICIENTS, AT THE POINT OF USE.
  // `ion_cr_model = spectrum` is live in all four production decks and materially sets x_e,
  // hence all three diffusivities. ionization.hpp states plainly that the FORM is standard but
  // the DEFAULTS (a = 0.4, N_0 = 1e20, N_c = 2.05e25, zeta_rn = 1e-19) are "NOT a transcription
  // of any specific paper's fit". Separately the local column proxy Sigma = 0.34 rho lambda_J
  // is calibrated against ONE collapse over a range where rho ~ r^-2, with a residual spread of
  // 1.89-5.61 about its median, and a 2x column error becomes 2^0.4 = 1.32x in zeta and hence
  // ~1.32x in eta_A. Neither is a defect and both are honestly stated in the source -- but they
  // are LOAD-BEARING for any published flux-retention or eta_A number and appear nowhere in the
  // deck or the run banner. If the paper quotes eta_A, this is the dominant systematic to
  // declare. Recorded here so it is seen by anyone changing ion_cr_model.
  const auto cm = pin->GetOrAddString("diffusion", "ion_cr_model", "none");
    if (cm == "none") {
      ion.cr_model = CRAtten::none;
    } else if (cm == "exponential") {
      ion.cr_model = CRAtten::exponential;
    } else if (cm == "spectrum") {
      ion.cr_model = CRAtten::spectrum;
    } else {
      PARTHENON_FAIL("diffusion/ion_cr_model must be 'none', 'exponential' or 'spectrum'.");
    }
  }
  ion.cr_zeta_rn = pin->GetOrAddReal("diffusion", "ion_cr_zeta_rn", ion.cr_zeta_rn);
  ion.cr_N0 = pin->GetOrAddReal("diffusion", "ion_cr_N0_cm2", ion.cr_N0);
  ion.cr_index = pin->GetOrAddReal("diffusion", "ion_cr_index", ion.cr_index);
  ion.cr_Nc = pin->GetOrAddReal("diffusion", "ion_cr_Nc_cm2", ion.cr_Nc);
  ion.cr_mu_col = pin->GetOrAddReal("diffusion", "ion_cr_mu_col", ion.cr_mu_col);
  ion.cr_sigma0 = pin->GetOrAddReal("diffusion", "ion_cr_sigma0_cgs", ion.cr_sigma0);
  ion.cr_column_fac = pin->GetOrAddReal("diffusion", "ion_cr_column_fac", ion.cr_column_fac);
  ion.cr_column_ext = pin->GetOrAddReal("diffusion", "ion_cr_column_ext_cgs",
                                        ion.cr_column_ext);
  {
    const auto cc = pin->GetOrAddString("diffusion", "ion_cr_column_model", "local");
    if (cc == "calibrated") ion.cr_column_calibrated = true;
    else if (cc != "local")
      PARTHENON_FAIL("diffusion/ion_cr_column_model must be 'local' or 'calibrated'.");
  }
  ion.cr_gamma = pin->GetOrAddReal("hydro", "gamma", ion.cr_gamma);
  if (ion.cr_model != CRAtten::none) {
    if (parthenon::Globals::my_rank == 0) {
      if (ion.cr_model == CRAtten::spectrum) {
        std::cout << "## CR attenuation (spectrum): zeta(N) = " << ion.zeta
                  << " min(1,(N/" << ion.cr_N0 << ")^-" << ion.cr_index << ") exp(-N/"
                  << ion.cr_Nc << ") + " << ion.cr_zeta_rn << " s^-1;  Sigma = "
                  << ion.cr_column_fac << (ion.cr_column_calibrated ? " g(rho)" : "")
                  << " rho lambda_J + " << ion.cr_column_ext
                  << " g/cm^2,  N = Sigma/(" << ion.cr_mu_col << " m_H)" << std::endl;
      } else {
        std::cout << "## CR attenuation (exponential, LEGACY -- inert below Sigma_0): zeta = "
                  << ion.zeta << " exp(-Sigma/" << ion.cr_sigma0 << ") + " << ion.cr_zeta_rn
                  << " s^-1" << std::endl;
      }
    }
  }
  ion.rho_grain = pin->GetOrAddReal("diffusion", "ion_rho_grain", ion.rho_grain);
  ion.T_subl = pin->GetOrAddReal("diffusion", "ion_T_subl", ion.T_subl);
  const Real a_min = pin->GetOrAddReal("diffusion", "ion_a_min_cm", 5.0e-7);
  const Real a_max = pin->GetOrAddReal("diffusion", "ion_a_max_cm", 2.5e-5);
  const Real mrn_p = pin->GetOrAddReal("diffusion", "ion_mrn_p", 3.5);
  SetupGrainBins(ion, a_min, a_max, mrn_p);
  // thermal Saha
  ion.x_K = pin->GetOrAddReal("diffusion", "ion_x_K", ion.x_K);
  ion.chi_K = pin->GetOrAddReal("diffusion", "ion_chi_K", ion.chi_K);
  ion.xK_per_H = pin->GetOrAddBoolean("diffusion", "ion_xK_per_H", false); // audit P-14
  // ambipolar closure: single_fluid (Athena++-matched, default) or tensor (grain-modified)
  const auto cl = pin->GetOrAddString("diffusion", "ion_ad_closure", "single_fluid");
  if (cl == "tensor") {
    ion.ad_closure = ADClosure::tensor;
  } else if (cl == "single_fluid") {
    ion.ad_closure = ADClosure::single_fluid;
  } else {
    PARTHENON_FAIL("diffusion/ion_ad_closure must be 'single_fluid' or 'tensor'.");
  }
  return ion;
}

} // namespace Ionization

#endif // HYDRO_DIFFUSION_IONIZATION_BUILD_HPP_
