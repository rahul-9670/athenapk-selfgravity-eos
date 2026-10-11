#ifndef RADIATION_RADIATION_SN_UNITS_HPP_
#define RADIATION_RADIATION_SN_UNITS_HPP_
//========================================================================================
// AthenaPK - discrete-ordinates (Sn) transport: the MOMENT <-> INTENSITY UNIT CALIBRATION.
//
// WHY THIS IS A SEPARATE, TEST-COVERED HEADER. radiation_moments.cpp's task list carried this
// note for the whole prototype phase:
//
//     "the PROTOTYPE Sn path deliberately does not write its moments back into rad.Er/rad.Fr
//      (the unit calibration against M1 is a step-3 gate, and a WRONG FACTOR THERE WOULD
//      SILENTLY CORRUPT THE GAS)."
//
// That is the correct fear and it names the remedy: the factor must be in ONE place, and that
// place must be compiled by a host test that checks the round trip and the equilibrium limit
// numerically rather than by inspection. A 4*pi or a chat in the wrong position does not crash
// -- it produces a plausible-looking radiation field that is wrong by a constant, which is the
// single hardest class of error to catch downstream. Every conversion between the ordinate
// representation and the moment representation goes through this header, and nothing else in
// the package is allowed to write a bare 4*pi.
//
// THE CONVENTION, stated once (radiation_sn_angles.hpp fixes sum_n w_n = 4*pi):
//
//   J   = sum_n w_n I_n / (4*pi)          mean intensity      [I]
//   H_d = sum_n w_n I_n Omega_d / (4*pi)  Eddington flux      [I]
//   Er  = (4*pi/chat) * J                 radiation energy density, M1's rad.Er
//   Fr_d= 4*pi * H_d                      radiation flux,           M1's rad.Fr
//
// chat, NOT c: under RSLA the transport runs on the reduced speed, and rad.Er/rad.Fr are the
// reduced-speed moments that M1's own closure and drag are written against (|Fr| <= chat*Er,
// which the free-streaming limit |H| <= J maps onto exactly). Using c here would break the
// closure's reduced-flux bound f = |Fr|/(chat*Er) <= 1 by the factor creduc -- 1000 on the
// collapse run.
//
// EQUILIBRIUM. In a static, optically thick, non-scattering medium the ordinate equation
// relaxes to I_n = S/chi_a = the source function, isotropically. Writing that back must give
// Er = B_g, the group-integrated Planck energy density a_R T^4 f_g(T) -- i.e. the SAME number
// M1's matter coupling drives Er toward (radiation_thermal_m1.cpp `Band`). That identity is
// the physical content of the calibration and it is gate G8.
//========================================================================================

#ifndef KOKKOS_INLINE_FUNCTION
#define SN_UNITS_DEFINED_KIF
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace Radiation {
namespace Sn {

//! 4*pi, spelled once. (M_PI is not guaranteed device-visible on every toolchain.)
constexpr double kFourPi = 12.566370614359172953850573533118;

//----------------------------------------------------------------------------------------
//! \brief Isotropic specific intensity carrying radiation energy density `E`.
//!
//! The inverse of SnEnergyFromJ for an isotropic field, and the value the sweep must relax to
//! in the equilibrium limit. Used to seed the ordinates from an existing moment field.
KOKKOS_INLINE_FUNCTION double SnIntensityFromEnergy(const double E, const double chat) {
  return chat * E / kFourPi;
}

//----------------------------------------------------------------------------------------
//! \brief Radiation energy density from the mean intensity: Er = 4*pi*J/chat.
KOKKOS_INLINE_FUNCTION double SnEnergyFromJ(const double J, const double chat) {
  return kFourPi * J / chat;
}

//----------------------------------------------------------------------------------------
//! \brief Radiation flux component from the Eddington flux: Fr_d = 4*pi*H_d.
//!
//! No chat. H and J share units, so |H| <= J (every |Omega_d| <= 1) becomes |Fr| <= chat*Er
//! automatically -- the M1 closure's reduced-flux bound falls out of the quadrature rather
//! than being imposed. Dividing by chat here as well is the tempting symmetric mistake; it
//! would make the field look sub-luminal by creduc and quietly kill every radiation force.
KOKKOS_INLINE_FUNCTION double SnFluxFromH(const double H) { return kFourPi * H; }

//----------------------------------------------------------------------------------------
//! \brief Thermal EMISSIVITY of group g: eta = chi_a,g * chat * B_g / (4*pi).
//!
//! `chi_a` is the ABSORPTION coefficient rho*kappa_P,g [1/length], NOT the total extinction
//! chi = chi_a + chi_s. Emission is an absorption process by Kirchhoff's law; feeding total
//! chi here (which the prototype did, because its placeholder source folded the multiply into
//! the sweep) over-emits by (1 + chi_s/chi_a) wherever scattering is non-zero.
//!
//! `B_g` is the group-integrated Planck ENERGY DENSITY a_R T^4 f_g(T) in code units, exactly
//! what radiation_thermal_m1.cpp's Material::Band returns -- so the two coupling paths cannot
//! drift apart in their definition of equilibrium.
//!
//! Returned as an emissivity (added to the RHS directly) rather than a source function
//! (multiplied by chi inside the sweep): with scattering live the source is
//! chi_a*B + chi_s*J, which is NOT any single coefficient times a single function.
KOKKOS_INLINE_FUNCTION double SnEmissivity(const double chi_a, const double B_g,
                                           const double chat) {
  return chi_a * chat * B_g / kFourPi;
}

//----------------------------------------------------------------------------------------
//! \brief The Fleck split: how one group's opacities become the coefficients the sweep uses.
//!
//! IN THIS HEADER, AND NOT INLINE IN THE SEED, because getting it wrong is silent and it
//! already happened once. The first implementation stored the BARE chi_a as the absorption and
//! folded the Fleck remainder (1-f)*chi_a only into the scattering. The sweep forms its total
//! extinction as chi_a + chi_s, so that made the diagonal too large by exactly (1-f)*chi_a and
//! the steady state came out
//!     (chi_a + chi_s_eff) J - chi_s_eff J = eta   =>   Er = f*B  instead of  Er = B,
//! which gate G8 measured as Er = 5.96e-3 against a target of 1. The host test that was
//! supposed to cover this had been written with the fixed point solved in CLOSED FORM -- so it
//! encoded the intended algebra rather than the code's decomposition, and passed throughout.
//! Both now go through this function.
//!
//! THE SPLIT. Emission is linearised about T^n by the Fleck factor f, and the absorption that
//! removes, (1-f)*chi_a, is returned to the field as isotropic effective scattering:
//!
//!     chi_a_eff = f * chi_a                                  (exchanges energy with the gas)
//!     chi_s_eff = chi_s + max(0, chi_R - chi_a) + (1-f)*chi_a (returns it to the field)
//!     eta       = chi_a_eff * chat * B / (4*pi)
//!
//! TWO INVARIANTS make it correct, and the test checks both rather than trusting this comment:
//!   1. chi_a_eff + chi_s_eff = chi_a + chi_s + max(0, chi_R - chi_a), INDEPENDENT of f. A
//!      total extinction is a property of the medium; an implicitness knob must not change how
//!      far a photon travels.
//!   2. chi_a_eff * J = eta at equilibrium, i.e. Er = B EXACTLY for any f. The knob changes the
//!      path, never the answer.
//!
//! The Rosseland excess is carried as scattering so that attenuation happens on the Rosseland
//! length while energy exchange stays on chi_a: charging the difference to the gas would break
//! conservation by (chi_R - chi_a)*Er, and dropping it would make the medium thinner than its
//! own opacity table says.
struct SnCoeffs {
  double chi_a_eff;  //!< effective absorption, f*chi_a
  double chi_s_eff;  //!< effective scattering, incl. the Rosseland excess and (1-f)*chi_a
  double eta;        //!< emissivity, chi_a_eff*chat*B/(4*pi)
};

//! MULTIGROUP LIMITATION (audit P-07, 2026-09-23; documented, not changed). The (1-f)*chi_a,g
//! remainder is returned to the SAME group g. Multigroup IMC re-emits absorbed energy across
//! groups with weights chi_a,g' dB_g'/dT / sum; doing that would couple the groups inside the
//! Krylov operator (it is block-diagonal in g today). Equilibrium (Er_g = B_g) and energy
//! conservation are exact either way; what lags is the inter-group energy flow inside one step
//! in stiff (f << 1) cells, which then relaxes through the f*chi_a channel over ~a step.
KOKKOS_INLINE_FUNCTION SnCoeffs SnFleckSplit(const double chi_a, const double chi_s,
                                             const double chi_ross, const double B_g,
                                             const double chat, const double fleck) {
  SnCoeffs c;
  c.chi_a_eff = fleck * chi_a;
  c.chi_s_eff = chi_s + (chi_ross > chi_a ? chi_ross - chi_a : 0.0) + (1.0 - fleck) * chi_a;
  c.eta = c.chi_a_eff * chat * B_g / kFourPi;
  return c;
}

//----------------------------------------------------------------------------------------
//! \brief The equilibrium intensity implied by an emissivity and a total extinction.
//!
//! I_eq = eta/chi. TEST-ONLY helper: the sweep reaches this by iterating, and G8 checks that
//! it does. Keeping the closed form here lets the host test state the expected answer without
//! re-deriving the factors it is supposed to be checking.
KOKKOS_INLINE_FUNCTION double SnEquilibriumIntensity(const double eta, const double chi) {
  return (chi > 0.0) ? eta / chi : 0.0;
}

} // namespace Sn
} // namespace Radiation

#ifdef SN_UNITS_DEFINED_KIF
#undef KOKKOS_INLINE_FUNCTION
#undef SN_UNITS_DEFINED_KIF
#endif

#endif // RADIATION_RADIATION_SN_UNITS_HPP_
