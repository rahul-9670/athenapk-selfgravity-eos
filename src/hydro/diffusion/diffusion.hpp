//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusion.hpp
//! \brief

#ifndef HYDRO_DIFFUSION_DIFFUSION_HPP_
#define HYDRO_DIFFUSION_DIFFUSION_HPP_

// Parthenon headers
#include <limits>

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "ionization.hpp"

using namespace parthenon::package::prelude;

// (the slope limiters of the thermal-conduction kernel of the full code are not needed here)

struct OhmicDiffusivity {
 private:
  Real mbar_, me_, kb_;
  Resistivity resistivity_;
  ResistivityCoeff resistivity_coeff_type_;
  // "free" coefficient/prefactor. Value depends on resistivity set in the constructor.
  Real coeff_;
  // Reduced ionization model used when resistivity_coeff_type_ == ionization.
  Ionization::IonizationModel ion_;
  // Ceiling on the ionization-model eta_O (code units), applied identically in the flux
  // kernels (via the eta cache), the dt estimators, and the fused evaluator. In the
  // first-core interior eta_O grows without bound as x_e collapses while the field is
  // already fully decoupled there; the cap bounds the parabolic dt cost of resolving a
  // diffusion the dynamics no longer depends on. Disabled by default (huge value).
  Real eta_cap_;
  // VALIDATION-ONLY coefficient (nonideal_mhd A5, 2026-10-11): eta = coeff_ * rho^rho_pow_
  // (ResistivityCoeff::density_power) -- an eta jump at a density step with an exact answer.
  Real rho_pow_;

 public:
  KOKKOS_INLINE_FUNCTION
  OhmicDiffusivity(Resistivity resistivity, ResistivityCoeff resistivity_coeff_type,
                   Real coeff, Real mbar, Real me, Real kb,
                   Ionization::IonizationModel ion = Ionization::IonizationModel(),
                   Real eta_cap = std::numeric_limits<Real>::max(), Real rho_pow = 0.0)
      : resistivity_(resistivity), resistivity_coeff_type_(resistivity_coeff_type),
        coeff_(coeff), mbar_(mbar), me_(me), kb_(kb), ion_(ion), eta_cap_(eta_cap),
        rho_pow_(rho_pow) {}

  // bmag, temp (= code-unit T = p/rho) are used only for the ionization model;
  // ignored for ResistivityCoeff::fixed.
  KOKKOS_INLINE_FUNCTION
  Real Get(const Real bmag, const Real rho, const Real temp) const;

  KOKKOS_INLINE_FUNCTION
  Resistivity GetType() const { return resistivity_; }

  KOKKOS_INLINE_FUNCTION
  ResistivityCoeff GetCoeffType() const { return resistivity_coeff_type_; }

  // The reduced ionization model (shared by all non-ideal terms). Used by the fused
  // non-ideal timestep estimator to evaluate all three eta's from ONE Diffusivities call.
  KOKKOS_INLINE_FUNCTION
  const Ionization::IonizationModel &GetIonModel() const { return ion_; }

  // Ceiling on the ionization-model eta_O (code units; huge = disabled).
  KOKKOS_INLINE_FUNCTION
  Real EtaCap() const { return eta_cap_; }
};

// Face diffusivity from the two cached cell values. Arithmetic mean = legacy/default. Harmonic mean
// (diffusion/eta_face_harmonic, VALIDATION nonideal_mhd A5, default false) keeps the flux eta dB/dx continuous across
// an eta jump (the flux-continuous two-point average; arithmetic is first order at a jump). Ohmic only: for a field-
// dependent eta that vanishes (AD, eta_A = Q B^2) the harmonic mean blocks degenerate diffusion fronts.
KOKKOS_INLINE_FUNCTION
Real FaceEta(const Real a, const Real b, const bool harmonic) {
  if (!harmonic) return 0.5 * (a + b);
  const Real s = a + b;
  return (s > 0.0) ? 2.0 * a * b / s : 0.0;
}

// NOTE: the Get() bodies live here in the header (not in resistivity.cpp) because they
// are KOKKOS_INLINE_FUNCTION (implicitly inline): every translation unit that calls them
// (flux kernels, dt estimators, PrecomputeNonidealEta) must see the definition.
KOKKOS_INLINE_FUNCTION
Real OhmicDiffusivity::Get(const Real bmag, const Real rho, const Real temp) const {
  if (resistivity_coeff_type_ == ResistivityCoeff::fixed) {
    return coeff_;
  } else if (resistivity_coeff_type_ == ResistivityCoeff::ionization) {
    // Self-consistent eta_O from the reduced CR+thermal+grain ionization model
    // (Ohmic = c^2/(4 pi sigma_O)).
    Real eta_O, eta_H, eta_A;
    Ionization::Diffusivities(ion_, rho, temp, bmag, eta_O, eta_H, eta_A);
    return (eta_O < eta_cap_) ? eta_O : eta_cap_;
  } else if (resistivity_coeff_type_ == ResistivityCoeff::density_power) {
    return coeff_ * std::pow(rho, rho_pow_);
  } else if (resistivity_coeff_type_ == ResistivityCoeff::spitzer) {
    PARTHENON_FAIL("needs impl");
  } else {
    PARTHENON_FAIL("Unknown Resistivity coeff");
  }
}

Real EstimateResistivityTimestep(MeshData<Real> *md);

//! Calculate isotropic resistivity with fixed coefficient
void OhmicDiffFluxIsoFixed(MeshData<Real> *md);

//! Calculate resistivity (general case incl. Spitzer)
void OhmicDiffFluxGeneral(MeshData<Real> *md);

//----------------------------------------------------------------------------------------
//! Ambipolar diffusion. Following Athena++ (ConstDiffusivity), the ambipolar diffusivity
//! is eta_A = coeff * B^2 and the EMF is the perpendicular current,
//! E_A = eta_A (J - (J.bhat) bhat). Parabolic, so it is compatible with both the unsplit
//! and the RKL2 super-time-stepping integrators.
struct AmbipolarDiffusivity {
 private:
  Real mbar_, me_, kb_;
  Ambipolar ambipolar_;
  AmbipolarCoeff ambipolar_coeff_type_;
  // "free" coefficient/prefactor Q_A such that eta_A = Q_A * B^2 (AmbipolarCoeff::fixed).
  Real coeff_;
  // Reduced ionization model used when ambipolar_coeff_type_ == ionization.
  Ionization::IonizationModel ion_;
  // Ceiling on eta_A for the IONIZATION-family coefficients (code units), the ambipolar
  // analogue of OhmicDiffusivity::eta_cap_ / HallDiffusivity::eta_cap_ and applied
  // identically everywhere eta_A is consumed (flux kernels via the eta cache, dt
  // estimators, fused evaluator). The single-fluid/tensor eta_A = B^2/(4 pi gamma rho_i rho)
  // runs away (~1/rho^2) in low-density magnetized cells and drives the explicit/RKL2
  // parabolic dt to ~0 there; this bounds that runaway in the diffusion-decoupled regime
  // exactly as eta_ohm_cap does for Ohm. Default (huge) = disabled (bit-identical).
  Real eta_cap_;

 public:
  KOKKOS_INLINE_FUNCTION
  AmbipolarDiffusivity(Ambipolar ambipolar, AmbipolarCoeff ambipolar_coeff_type, Real coeff,
                       Real mbar, Real me, Real kb,
                       Ionization::IonizationModel ion = Ionization::IonizationModel(),
                       Real eta_cap = std::numeric_limits<Real>::max())
      : ambipolar_(ambipolar), ambipolar_coeff_type_(ambipolar_coeff_type), coeff_(coeff),
        mbar_(mbar), me_(me), kb_(kb), ion_(ion), eta_cap_(eta_cap) {}

  // temp = code-unit temperature (p = rho*T); ignored for AmbipolarCoeff::fixed.
  KOKKOS_INLINE_FUNCTION
  Real Get(const Real bmag, const Real rho, const Real temp) const;

  KOKKOS_INLINE_FUNCTION
  Ambipolar GetType() const { return ambipolar_; }

  KOKKOS_INLINE_FUNCTION
  AmbipolarCoeff GetCoeffType() const { return ambipolar_coeff_type_; }

  KOKKOS_INLINE_FUNCTION
  Real EtaCap() const { return eta_cap_; }

  KOKKOS_INLINE_FUNCTION
  const Ionization::IonizationModel &GetIonModel() const { return ion_; }
};

// Get() bodies in the header for cross-TU inlining (see note above OhmicDiffusivity::Get).
KOKKOS_INLINE_FUNCTION
Real AmbipolarDiffusivity::Get(const Real bmag, const Real rho, const Real temp) const {
  if (ambipolar_coeff_type_ == AmbipolarCoeff::fixed) {
    // Athena++ convention: eta_A = coeff * B^2 (uncapped: fixed-coeff test/idealised path)
    return coeff_ * SQR(bmag);
  } else if (ambipolar_coeff_type_ == AmbipolarCoeff::ionization) {
    // Self-consistent eta_A from the reduced CR+thermal+grain ionization model.
    Real eta_O, eta_H, eta_A;
    Ionization::Diffusivities(ion_, rho, temp, bmag, eta_O, eta_H, eta_A);
    return (eta_A < eta_cap_) ? eta_A : eta_cap_;
  } else {
    PARTHENON_FAIL("Unknown Ambipolar coeff");
  }
}

Real EstimateAmbipolarTimestep(MeshData<Real> *md);

//! Calculate ambipolar diffusion with fixed coefficient
void AmbipolarDiffFluxIsoFixed(MeshData<Real> *md);

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//! Per-cell fused evaluation of all three non-ideal diffusivities for the ionization
//! coefficient family. Each term's per-term Get() recomputes the FULL Pandey-Wardle
//! conductivity tensor (the dominant GPU cost) but keeps only one eta; this helper
//! reproduces the per-term Get() results from ONE Ionization::Diffusivities call.
struct FusedNonidealEval {
  Ionization::IonizationModel ion;
  bool ohm_on = false, ad_on = false, hall_on = false;
  // eta_O ceiling mirrored from OhmicDiffusivity::EtaCap() (huge = disabled), so the
  // fused path clamps exactly where the per-term Get() does.
  Real ohm_cap = std::numeric_limits<Real>::max();
  // |eta_H| ceiling (inert: Hall is not part of this tree, hall_on is always false).
  Real hall_cap = std::numeric_limits<Real>::max();
  // eta_A ceiling mirrored from AmbipolarDiffusivity::EtaCap() (huge = disabled). Bounds the
  // low-density single-fluid/tensor eta_A runaway that otherwise sets the parabolic dt.
  Real ad_cap = std::numeric_limits<Real>::max();

  // --- WS-4 DUST COUPLING (audit item 5) ---------------------------------------
  // prim-pack indices of the evolved dust scalars (f_dg, a_c) and the reference grain
  // radius the STATIC MRN bins correspond to. dust_on=false (default) => the ionization
  // model keeps its frozen ISM MRN population and every result is bit-identical.
  int i_fdg = -1;      // prim index of the dust-to-gas mass ratio
  int i_ac = -1;       // prim index of the characteristic grain radius a_c [cm]
  Real dust_a_ref = 1.0e-5; // [cm] a_c the static bins represent (DustModel::a_ref)
  Real dust_fdg_ref = 0.01; // dust-to-gas ratio the static bins represent
  bool dust_on = false;
  //! diffusion/dust_destroyed_zero (validation/dust DU-F4; default false = bit-identical): a cell whose
  //! grains were DESTROYED (f_dg <= 0 with a_c > 0 -- the dust package's own 'destroyed, not unseeded' test,
  //! dust.hpp apply_sublimation F7a) gets NO grains (f_dg_scale = 0) instead of the reference population. Without
  //! it, f_dg == 0 exactly (tanh underflow above ~3400 K, or a sublimated cell whose advected rho f undershot to
  //! <= 0) falls into the 'uninitialised' guard below and the charge solve sees the full ISM dust mass -- in grains of
  //! a_floor = a_ref/100 on the growth path, i.e. ~100x the reference grain area. Grains drop out of the charge
  //! balance anyway above diffusion/ion_T_subl (step, 1500 K), so this matters for destroyed gas BELOW 1500 K.
  bool dust_destroyed_zero = false;

  //! Per-cell ionization model: the static one, or a copy rescaled to the evolved grain
  //! population. The copy is only made when dust coupling is ON, so uncoupled runs pay
  //! nothing (the model is a ~40-Real POD; copying it per cell is not free on GPU).
  KOKKOS_INLINE_FUNCTION
  Ionization::IonizationModel CellIon(const Real f_dg, const Real a_c) const {
    Ionization::IonizationModel mc = ion;
    // Guard degenerate/uninitialized scalars: fall back to the static population rather
    // than divide by zero or hand the charge solve a zero-radius grain.
    //
    // HISTORY (2026-07-30): collapse_be used to leave these scalars at 0.0 even with
    // <dust> evolve=true (its species zero-fill ran past the since-removed chemistry block), so both
    // guards fired and this coupling was a silent no-op. FIXED in collapse_be.cpp, which
    // now seeds f_dg = <dust> f_dg_ref and a_c = <dust> a_ref_cm. The guards stay as a
    // backstop for other pgens: falling back to the static population is always safer than
    // handing the charge solve a zero-radius grain. NOTE that a_scale == 1 at t=0 BY
    // DESIGN (the static MRN bins represent exactly the reference population), so this
    // coupling only bites once the dust package moves (f_dg, a_c) away from the reference
    // -- changing <dust> a_ref_cm alone moves the IC and the reference together and does
    // nothing.
    mc.f_dg_scale = (f_dg > 0.0 && dust_fdg_ref > 0.0) ? f_dg / dust_fdg_ref : 1.0;
    if (dust_destroyed_zero && !(f_dg > 0.0) && a_c > 0.0) mc.f_dg_scale = 0.0;   // destroyed, not unseeded
    mc.a_scale = (a_c > 0.0 && dust_a_ref > 0.0) ? a_c / dust_a_ref : 1.0;
    return mc;
  }

  //! UNCAPPED diffusivities: everything Eta() does except the three ceiling clamps.
  //! Eta() == EtaRaw() + clamps, so the physics path is unchanged by construction; this
  //! exists so the cap diagnostic (FillNonidealCapDiag) can report how much diffusivity
  //! the ceilings actually remove, without a second Wardle tensor solve.
  KOKKOS_INLINE_FUNCTION
  void EtaRaw(const Real rho, const Real temp, const Real bmag, Real &eta_O, Real &eta_H,
              Real &eta_A, const Real f_dg = -1.0, const Real a_c = -1.0) const {
    // Dust coupling: rescale the grain population to the evolved (f_dg, a_c) for THIS cell.
    // Pointer (not a ternary) on purpose: `dust_on ? CellIon(...) : ion` is a PRVALUE, so it
    // would copy the model even in the OFF branch -- the exact cost this is avoiding.
    Ionization::IonizationModel mcell;
    const Ionization::IonizationModel *pim = &ion;
    if (dust_on) {
      mcell = CellIon(f_dg, a_c);
      pim = &mcell;
    }
    const Ionization::IonizationModel &im = *pim;
    Real eO_eq = 0.0, eH_eq = 0.0, eA_eq = 0.0;
    if (ohm_on || hall_on || ad_on) {
      Ionization::Diffusivities(im, rho, temp, bmag, eO_eq, eH_eq, eA_eq);
    }
    eta_O = ohm_on ? eO_eq : 0.0;
    eta_H = hall_on ? eH_eq : 0.0;
    eta_A = ad_on ? eA_eq : 0.0;
  }

  //! Apply the three explicit numerical ceilings in place. Split out of Eta() so the cap
  //! diagnostic applies EXACTLY the same clamps the physics path does (no drift between a
  //! diagnostic copy and the real thing). When a term is off its eta is 0 and its cap is
  //! positive, so clamping unconditionally is a no-op there.
  KOKKOS_INLINE_FUNCTION
  void Clamp(Real &eta_O, Real &eta_H, Real &eta_A) const {
    if (eta_O > ohm_cap) eta_O = ohm_cap;
    if (eta_H > hall_cap) eta_H = hall_cap;
    if (eta_H < -hall_cap) eta_H = -hall_cap;
    // ad_on guard so an off-AD eta_A stays exactly 0 even for a pathological ad_cap <= 0
    // (pre-refactor the eta_A clamp lived inside the ad_on branch).
    if (ad_on && eta_A > ad_cap) eta_A = ad_cap;
  }

  KOKKOS_INLINE_FUNCTION
  void Eta(const Real rho, const Real temp, const Real bmag, Real &eta_O, Real &eta_H,
           Real &eta_A, const Real f_dg = -1.0, const Real a_c = -1.0) const {
    EtaRaw(rho, temp, bmag, eta_O, eta_H, eta_A, f_dg, a_c);
    Clamp(eta_O, eta_H, eta_A);
  }
};

//----------------------------------------------------------------------------------------
//! Build the fused evaluator from the package's active diffusivity objects; returns false when
//! the coefficient mix is not fusable (defined in diffusion.cpp).
bool MakeFusedNonidealEval(parthenon::StateDescriptor *pkg, FusedNonidealEval &fe);

//! Fused non-ideal (Ohmic + ambipolar + Hall) diffusive timestep for the IONIZATION
//! coefficient family. The three separate EstimateX Timestep reductions
//! each recompute the full conductivity tensor but keep only one eta; this routine uses
//! FusedNonidealEval (minimum tensor solves per cell) and reduces the min over whichever
//! of the three terms are active, using each term's own CFL factor. The eta's match the
//! per-term Get() exactly. The parabolic constraint uses the SUM
//! eta_O + hall_ohmic_floor (the flux kernels apply them additively), so with a Hall floor
//! this is slightly stricter (safer) than the separate per-term estimators.
//! Valid when every ACTIVE non-ideal term uses an "ionization" coeff
//! (NOT "fixed"); the caller gates on the "nonideal_dt_fused" package flag.
Real EstimateNonidealTimestepIonizationFused(MeshData<Real> *md);


//----------------------------------------------------------------------------------------
//! Component indices of the cell-centered non-ideal diffusivity cache field
//! "nonideal_eta" (registered by the Hydro package when diffusion/eta_cache is on).
namespace NonidealEtaIdx {
constexpr int O = 0; // Ohmic eta_O
constexpr int H = 1; // Hall eta_H (signed)
constexpr int A = 2; // ambipolar eta_A
} // namespace NonidealEtaIdx

//----------------------------------------------------------------------------------------
//! Component indices of the cap-activation diagnostic field "diff.capdiag" (registered when
//! diffusion/cap_diag=true). AUDIT ITEM 1: eta_ohm_cap_code / eta_ad_cap_code /
//! eta_hall_cap_code and the Hall Ohmic floor are numerical stabilizers that MODIFY the
//! induction equation. Without this diagnostic there is no way to tell whether a flux-
//! retention result is physics or a stabilizer. Components 0-2 are a 0/1 activation flag
//! (a capped cell returns the ceiling exactly, so the test is exact); components 3-5 are
//! log10(eta_raw / eta_applied) -- how many DECADES of diffusivity the ceiling removed
//! (0 where inactive). Both are per-cell, so the phdf gives the spatial picture (where the
//! caps bite relative to the core/accretion shock) for free.
namespace CapDiagIdx {
constexpr int flagO = 0;
constexpr int flagH = 1;
constexpr int flagA = 2;
constexpr int decO = 3;
constexpr int decH = 4;
constexpr int decA = 5;
constexpr int NCOMP = 6;
} // namespace CapDiagIdx

//! Fill "diff.capdiag" from the same per-cell state PrecomputeNonidealEta uses. Called
//! from PrecomputeNonidealEta itself (gated on the "nonideal_cap_diag" package flag) so
//! there is no second Wardle tensor solve and no extra task in the driver.
//! Volume-weighted history reductions over "diff.capdiag" (see CapDiagIdx). All are
//! plain sums so they compose across meshblocks/ranks; form fractions in analysis by
//! dividing by "cap-Vtot" (the summed volume of the same cells).
//!
//! AUDIT F-32 (2026-08-28) -- QUOTE THE MASS FRACTION, NOT THE VOLUME FRACTION.
//! Both forms are registered (cap-V* = sum flag*dV, cap-M* = sum flag*rho*dV), but every
//! analysis so far quoted the VOLUME one, and the caps bite in the DENSEST cells, so the two
//! differ by seven orders of magnitude. Measured on the production run's final hst row:
//!     term    volume fraction   MASS fraction   mass/volume
//!     Ohmic     6.361e-11        1.457e-03       2.29e+07
//!     Hall      6.062e-11        1.436e-03       2.37e+07
//! "6.3e-11 of the volume" reads as utterly negligible; "0.15 % of the MASS -- concentrated
//! in exactly the first-core gas whose field structure is the science result" does not.
//! Neither number is wrong; the volume one is simply the least informative choice for this
//! question. Divide cap-M* by the hst `mass` column, not cap-M* by cap-Vtot.
//! (The mean magnitude removed is cap-D*/cap-V* in DECADES: 0.497 dex = 3.14x for Ohmic,
//! 0.491 dex = 3.09x for Hall -- volume-weighted means whose PEAK is the cap ratio itself.)
Real NonidealCapHstVol(MeshData<Real> *md);                   // cap-Vtot  = sum dV
Real NonidealCapHstFlag(MeshData<Real> *md, int comp);        // sum flag*dV
Real NonidealCapHstDecade(MeshData<Real> *md, int comp);      // sum flag*decades*dV
Real NonidealCapHstMassFlag(MeshData<Real> *md, int comp);    // sum flag*rho*dV

//! Fill the cell-centered (eta_O, eta_H, eta_A) cache from prim over the interior
//! expanded by one ghost layer (all cells the face-averaging in the flux kernels
//! touches). Uses the fused FusedNonidealEval (minimum tensor solves per cell) when
//! every active non-ideal term uses an "ionization" coefficient;
//! per-term Get() otherwise (still once per cell instead of once per face per term).
//! Called from CalcDiffFluxes when the "nonideal_eta_cache" package flag is set.
TaskStatus PrecomputeNonidealEta(StateDescriptor *hydro_pkg, MeshData<Real> *md);

//! Which subset of the active diffusion terms CalcDiffFluxes applies.
//!   all        -- every active term (the unsplit integrator).
//!   parabolic  -- Ohmic and ambipolar (the RKL2 stage operator).
//!   dispersive -- the Hall term of the full code; selects nothing in this tree.
enum class DiffTermSet { all, parabolic, dispersive };

// Calculate diffusion fluxes for the selected term set, i.e., update the .flux views in md.
// refill_eta=false skips the PrecomputeNonidealEta cache refresh so the flux kernels reuse
// the previously cached diffusivities ("diffusion/rkl2_freeze_eta": the RKL2 stage loop
// refreshes eta only at the first stage of each Strang half; RKL2's stability polynomial
// (Meyer+2012) assumes a fixed operator across the super-step, and the resulting
// coefficient lag is O(dt/2) on eta(rho,T,B), which evolves on the collapse timescale).
//! Dual energy (hydro/dual_energy): add the Ohmic + ambipolar dissipation of this Strang half to
//! the advected internal energy. See dual_energy_heating.cpp. No-op when dual energy is off.
TaskStatus DualEnergyDiffusionHeating(MeshData<Real> *md, const Real dt);
//! hydro/dual_energy_heat_rkl2 (audit P-03): the same heating as a source inside the RKL2
//! recurrence. MY0.dual += q(base); and base.dual += coef * q(base prim) after a later stage.
TaskStatus DualHeatIntoMY0(MeshData<Real> *md_base, MeshData<Real> *md_my0);
TaskStatus DualHeatStage(MeshData<Real> *md_base, const Real coef);

TaskStatus CalcDiffFluxes(StateDescriptor *hydro_pkg, MeshData<Real> *md,
                          DiffTermSet term_set = DiffTermSet::all, bool refill_eta = true);

#endif //  HYDRO_DIFFUSION_DIFFUSION_HPP_
