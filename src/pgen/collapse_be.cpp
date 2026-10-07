//========================================================================================
// AthenaPK - Bonnor-Ebert sphere collapse problem generator
// Ported from Athena++'s collapse.cpp (Kengo Tomida).
// Adapted for AthenaPK's hydrodynamics interface and self-gravity package.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file collapse_be.cpp
//  \brief Bonnor-Ebert sphere collapse. Ambient density = the BE profile value at r = rc,
//         so the sphere/ambient interface is pressure-continuous.
//
//  GRAVITY BOUNDARY CONDITIONS (this header used to say multipole BCs were unsupported --
//  they landed in WS-5a and are what production uses; corrected 2026-08-05):
//    * `self_gravity/*_bc = multipole` -- isolated exterior potential via an RHS
//      Green's-function lift (src/self_gravity/multipole.hpp). PRODUCTION. Lets a
//      physically MATCHED box (L=16) collapse freely; validated to 0.54% against the
//      analytic exterior potential.
//    * `... = zero` -- zero-Dirichlet. Still supported and still correct, but it needs a
//      PADDED box (domain ~4x the sphere radius) because clamping phi=0 on the face of a
//      matched box freezes the collapse. This is the setup the old header described.
//    * `... = neumann` -- known broken; do not use.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "../eos/adiabatic_glmmhd.hpp" // AUDIT F-45: EOS-table IC thermal state
#include "../units.hpp" // canonical Heaviside-Lorentz magnetic-unit definition (audit #3)
#include "../units/be_normalization.hpp" // Phase 1: shared BE base-scale derivation
#include "../units/physical_units.hpp"   // Phase 1: authoritative units (consistency guard)
#include "pgen.hpp"

namespace collapse_be {
using namespace parthenon::driver::prelude;

// ---------------------------------------------------------------------------
// Dimensionless constants from Tomida's normalization
// In code units: 4piG = 1, c_s(T=10K) = 1, central density of BE sphere = 1
// ---------------------------------------------------------------------------
constexpr Real rc_code        = 6.45;        // BE radius in code units
constexpr Real rcsq_code      = 26.0 / 3.0;  // BE profile scale^2
constexpr Real bemass_code    = 197.561;     // total mass of critical BE sphere

// VALIDATION (2026-10-03, validation_tests/self_gravity L6/L7): optional EXACT isothermal
// Bonnor-Ebert profile, rho = exp(-psi(r)) from the isothermal Lane-Emden equation (in code
// units xi = r), truncated at r = xi_max. Default `be_profile = tomida` with xi_max = 6.45 is the
// historical (production) profile and radius -- identical arithmetic. Host-side only (the IC is
// built on a host mirror).
namespace {
Real g_rc = rc_code;
bool g_exact = false;
bool g_henn = false;
Real g_r0 = 1.0;
bool g_be_init = false;
std::vector<Real> g_le_rho;  // exp(-psi) on a uniform grid of spacing g_le_dr
Real g_le_dr = 1.0e-3;

void InitBEProfile(parthenon::ParameterInput *pin) {
  if (g_be_init) return;
  g_be_init = true;
  const std::string prof = pin->GetOrAddString("problem/collapse_be", "be_profile", "tomida");
  PARTHENON_REQUIRE(prof == "tomida" || prof == "exact" || prof == "hennebelle",
                    "problem/collapse_be/be_profile must be 'tomida', 'exact' or 'hennebelle'");
  g_exact = (prof == "exact");
  g_henn = (prof == "hennebelle");
  g_rc = pin->GetOrAddReal("problem/collapse_be", "xi_max", rc_code);
  // Hennebelle et al. 2011 (A&A 528, A72) core: rho = 1/(1 + (r/r0)^2) inside xi_max.
  g_r0 = pin->GetOrAddReal("problem/collapse_be", "profile_r0", 1.0);
  if (!g_exact) return;
  // RK4 for y = (psi, dpsi/dxi): psi'' = exp(-psi) - 2 psi'/xi; series start at xi0.
  const int n = static_cast<int>(std::ceil((g_rc + 1.0) / g_le_dr)) + 2;
  g_le_rho.assign(n, 1.0);
  const Real x0 = 1.0e-6;
  Real psi = x0 * x0 / 6.0, dpsi = x0 / 3.0, x = x0;
  auto f = [](Real xx, Real p, Real dp, Real &d1, Real &d2) {
    d1 = dp;
    d2 = std::exp(-p) - 2.0 * dp / xx;
  };
  const int sub = 20;
  const Real h = g_le_dr / sub;
  g_le_rho[0] = 1.0;
  for (int i = 1; i < n; ++i) {
    const Real target = i * g_le_dr;
    while (x < target - 0.5 * h) {
      const Real hh = std::min(h, target - x);
      Real a1, a2, b1, b2, c1, c2, d1, d2;
      f(x, psi, dpsi, a1, a2);
      f(x + 0.5 * hh, psi + 0.5 * hh * a1, dpsi + 0.5 * hh * a2, b1, b2);
      f(x + 0.5 * hh, psi + 0.5 * hh * b1, dpsi + 0.5 * hh * b2, c1, c2);
      f(x + hh, psi + hh * c1, dpsi + hh * c2, d1, d2);
      psi += hh / 6.0 * (a1 + 2 * b1 + 2 * c1 + d1);
      dpsi += hh / 6.0 * (a2 + 2 * b2 + 2 * c2 + d2);
      x += hh;
    }
    g_le_rho[i] = std::exp(-psi);
  }
}

Real BEProfileHost(Real r) {
  if (g_henn) return 1.0 / (1.0 + r * r / (g_r0 * g_r0));
  if (!g_exact) return std::pow(1.0 + r * r / rcsq_code, -1.5);
  const Real u = r / g_le_dr;
  const int i = std::min(static_cast<int>(u), static_cast<int>(g_le_rho.size()) - 2);
  const Real w = u - i;
  return (1.0 - w) * g_le_rho[i] + w * g_le_rho[i + 1];
}
} // namespace

// Dimensional constants (cgs)
constexpr Real msun = 1.9891e33;    // g
constexpr Real au   = 1.4959787e13; // cm
constexpr Real yr   = 3.15569e7;    // s

// These hold the derived physical parameters for logging and the cooling fn.
// Parthenon doesn't give us a natural "namespace-scoped globals" pattern like
// Athena++, so we stash them in the Hydro package params after problem init.
struct CollapseParams {
  Real rho0;     // density normalization (cgs)
  Real rhocrit;  // barotropic transition density in CODE units
  Real f;        // BE density enhancement factor
  Real rc;       // sphere radius in code units
  Real omega;    // angular velocity in code units
  Real amp;      // m=2 perturbation amplitude
  Real gamma_eos;
};

// Approximated BE profile (Tomida 2011 PhD thesis)
KOKKOS_INLINE_FUNCTION
Real BEProfile(Real r) {
  return std::pow(1.0 + r * r / rcsq_code, -1.5);
}

// ---------------------------------------------------------------------------
// Problem generator: initialize cons variables on each block.
// No rotation-vs-ambient distinction — inside sphere has solid-body rotation,
// outside has zero velocity and ambient density = rho(rc).
// ---------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//! Register the collapse_be package Params, plus the cgs unit sidecar.
//!
//! WP-13: registered here, not in ProblemGenerator, because Parthenon does not call the pgen on a
//! restart -- the driver gates the exterior-momentum/barotropic task on collapse_be_rhocrit, and
//! losing it on a resume floored the whole domain. Idempotent (guarded on the key).
void RegisterCollapseBeParams(parthenon::ParameterInput *pin,
                              parthenon::StateDescriptor *hydro_pkg) {
  if (hydro_pkg->AllParams().hasKey("collapse_be_rhocrit")) return;

  const Real gam = pin->GetReal("hydro", "gamma");
  const Real mass_msun = pin->GetReal("problem/collapse_be", "mass");
  const Real temp_K = pin->GetReal("problem/collapse_be", "temperature");
  const Real f = pin->GetReal("problem/collapse_be", "f");
  const Real rhocrit_cgs = pin->GetReal("problem/collapse_be", "rhocrit");

  // The ONE shared BE derivation -- identical to the values
  // ProblemGenerator computes, so moving the registration here changes no number.
  const auto be = PhysUnits::DeriveBENormalization(mass_msun, temp_K, f);
  const Real m0 = be.m0, v0 = be.v0, rho0 = be.rho0, t0 = be.t0, l0 = be.l0;
  const Real rhocrit_code = rhocrit_cgs / rho0;
  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);

  hydro_pkg->AddParam("collapse_be_rhocrit", rhocrit_code);
  hydro_pkg->AddParam("collapse_be_mhd", mhd);
  InitBEProfile(pin);
  hydro_pkg->AddParam("collapse_be_rc", g_rc);
  hydro_pkg->AddParam("collapse_be_gamma", gam);

  // Consistency guard: the authoritative unit system (PhysUnits) must reproduce this BE normalization
  // exactly. Both now route through DeriveBENormalization, so this is a regression sentinel
  // -- it fires only if that sharing is broken or a <units> override slips in.
  const auto U = PhysUnits::BuildPhysicalUnits(pin);
  const Real utol = 1.0e-10;
  PARTHENON_REQUIRE(std::abs(U.rho_unit - rho0) <= utol * rho0 &&
                        std::abs(U.v_unit - v0) <= utol * v0 &&
                        std::abs(U.length_unit - l0) <= utol * l0,
                    "collapse_be: PhysicalUnits base scales disagree with the BE IC "
                    "normalization -- microphysics units would desync from the dynamics.");

  // === Unit conversion factors to cgs ===
  // Multiply a code-unit value by these to obtain cgs.
  const Real code_length_cgs = l0;
  const Real code_time_cgs = t0;
  const Real code_mass_cgs = m0;
  const Real code_density_cgs = rho0;
  const Real code_velocity_cgs = l0 / t0;
  const Real code_energy_cgs = m0 * (l0 / t0) * (l0 / t0);
  // Audit fix #3: the Heaviside-Lorentz magnetic unit
  // B_cgs = sqrt(4*pi*rho0)*v0 (P_mag = B^2/2) is taken from the ONE definition in units.hpp
  // (Units::CodeMagneticCgs), the same formula PhysicalUnits uses -- so this sidecar cannot
  // drift from the code's internal B convention. The BE derivation gives mass_unit =
  // rho0*l0^3 exactly, so CodeMagneticCgs(m0,l0,t0) == sqrt(4*pi*rho0)*v0.
  // (Previously the sqrt(4*pi) was dropped, under-reporting physical B by ~3.545.)
  const Real code_bfield_cgs =
      Units::CodeMagneticCgs(code_mass_cgs, code_length_cgs, code_time_cgs);

  hydro_pkg->AddParam("units/code_length_cgs", code_length_cgs);
  hydro_pkg->AddParam("units/code_time_cgs", code_time_cgs);
  hydro_pkg->AddParam("units/code_mass_cgs", code_mass_cgs);
  hydro_pkg->AddParam("units/code_density_cgs", code_density_cgs);
  hydro_pkg->AddParam("units/code_velocity_cgs", code_velocity_cgs);
  hydro_pkg->AddParam("units/code_energy_cgs", code_energy_cgs);
  hydro_pkg->AddParam("units/code_bfield_cgs", code_bfield_cgs);

  // Sidecar JSON dump (rank 0 only; once per run).
  if (parthenon::Globals::my_rank == 0) {
    std::ofstream js("units.json");
    js.precision(12);
    js << "{\n"
       << "  \"code_length_cgs\":   " << code_length_cgs << ",\n"
       << "  \"code_time_cgs\":     " << code_time_cgs << ",\n"
       << "  \"code_mass_cgs\":     " << code_mass_cgs << ",\n"
       << "  \"code_density_cgs\":  " << code_density_cgs << ",\n"
       << "  \"code_velocity_cgs\": " << code_velocity_cgs << ",\n"
       << "  \"code_energy_cgs\":   " << code_energy_cgs << ",\n"
       << "  \"code_bfield_cgs\":   " << code_bfield_cgs << "\n"
       << "}\n";
  }
}

//! Package-init hook. Runs on BOTH a fresh start and a restart (unlike ProblemGenerator).
void ProblemInitPackageData(parthenon::ParameterInput *pin,
                            parthenon::StateDescriptor *hydro_pkg) {
  RegisterCollapseBeParams(pin, hydro_pkg);
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  // For the tabulated protostellar EOS, cold molecular H2 (10 K, rotation frozen) is
  // monatomic-like: e_int = 1.5 p (gamma=5/3), not p/(gamma-1). Seeding the ideal value
  // would make the table read the IC as hot. (At 10 K the table's pressure already equals
  // the isothermal p=rho since it uses the same mu=2.33; only the energy differs.)
  const bool use_h2diss = (pin->GetOrAddString("hydro", "eos", "adiabatic") == "hydrogen");
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = gam - 1.0;
  const Real igm1 = 1.0 / gm1;
  // e_int_over_p is computed further down, once hydro_pkg and `f` exist (audit F-45).

  // Read problem params
  const Real mass_msun = pin->GetReal("problem/collapse_be", "mass");
  const Real temp_K    = pin->GetReal("problem/collapse_be", "temperature");
  const Real f         = pin->GetReal("problem/collapse_be", "f");
  const Real amp       = pin->GetOrAddReal("problem/collapse_be", "amp", 0.0);
  const Real omegatff  = pin->GetOrAddReal("problem/collapse_be", "omegatff", 0.0);
  const Real rhocrit_cgs = pin->GetReal("problem/collapse_be", "rhocrit");

  // Derive normalization units via the ONE shared BE derivation.
  // Bit-identical to the historical inline block; PhysicalUnits now uses the same function so
  // the microphysics units match these exactly (no rounded-<units>-defaults drift).
  const auto be   = PhysUnits::DeriveBENormalization(mass_msun, temp_K, f);
  const Real m0   = be.m0;   // total mass = bemass*f in code units
  const Real v0   = be.v0;   // code c_s = 1
  const Real rho0 = be.rho0;
  const Real t0   = be.t0;   // code t = 1
  const Real l0   = be.l0;

  // Angular velocity: omega * t_ff = omegatff
  const Real tff_code = std::sqrt(3.0 / (8.0 * f)) * M_PI;
  const Real omega_code = omegatff / tff_code;

  // rhocrit in code units
  const Real rhocrit_code = rhocrit_cgs / rho0;

  // Stash params in the hydro package for the cooling function to read.
  // (Only rank 0 prints; all ranks need the params.)
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
  const Real B0z = mhd ? pin->GetOrAddReal("problem/collapse_be", "B0z", 0.0) : 0.0;

  // Audit F-45: under the tabulated EOS the IC internal energy comes from inverting the table at
  // T_code = 1 (the old constant 1.5 put the IC at 10.194 K, p/rho = 1.0013, and under-seeded Er by
  // 7.3 %). Ideal gas unchanged; collapse_be/eint_over_p_override restores a constant.
  Real e_int_over_p = igm1;
  if (use_h2diss) {
    e_int_over_p = pin->GetOrAddReal("problem/collapse_be", "eint_over_p_override", -1.0);
    if (!(e_int_over_p > 0.0)) {
      // esp at T_code = 1 from the RT grid, then e_int/p = esp/T_code = esp (since p = rho*T
      // and T_code = 1 => p = rho => e_int/p = (esp*rho)/rho = esp).
      const auto &eos_ic = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
      const auto &tab_ic = eos_ic.GetEosTable();
      const auto U_ic = PhysUnits::BuildPhysicalUnits(pin);
      // Evaluate at the BE central density (rho_code = f); esp is essentially rho-independent
      // at 10 K (verified: 1.473638 across three decades in rho), so any interior state does.
      // HOST call: EintFromRhoTk reads a device View and is illegal here. GH200 traps it
      // at startup; x86 tolerated it silently. Host twin, same arithmetic.
      e_int_over_p = tab_ic.EintFromRhoTkHost(f, U_ic.temperature_unit) / f;
    }
  }
  // NOTE(WP-13, 2026-08-02): the registration below now lives in RegisterCollapseBeParams(),
  // which ProblemInitPackageData calls during package Initialize -- a hook that runs on BOTH
  // a fresh start AND a restart. This call is therefore a no-op in normal operation (the key
  // already exists) and is kept only as a belt-and-braces fallback. It must NOT be the sole
  // registration site: ProblemGenerator is not called on restart, so anything registered only
  // here silently vanishes on resume -- which is exactly the bug WP-13 found.
  RegisterCollapseBeParams(pin, hydro_pkg.get());

  // Log (rank 0 only)
  if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
    std::cout << "\n---  Dimensional parameters of the simulation  ---\n"
              << "Total mass          : " << mass_msun << " [Msun]\n"
              << "Initial temperature : " << temp_K    << " [K]\n"
              << "Sound speed         : " << v0        << " [cm/s]\n"
              << "Central density     : " << rho0 * f  << " [g/cm^3]\n"
              << "Cloud radius        : " << g_rc * l0 / au << " [au]\n"
              << "Free-fall time      : " << tff_code * t0 / yr << " [yr]\n"
              << "rhocrit (cgs)       : " << rhocrit_cgs << " [g/cm^3]\n"
              << "rhocrit (code)      : " << rhocrit_code << "\n"
              << "Density enhancement : " << f << "\n\n"
              << "---   Normalization Units    ---\n"
              << "Mass                : " << m0 / msun << " [Msun]\n"
              << "Length              : " << l0 / au   << " [au]\n"
              << "Time                : " << t0 / yr   << " [yr]\n"
              << "Density             : " << rho0      << " [g/cm^3]\n\n"
              << "--- Dimensionless parameters ---\n"
              << "Total mass          : " << bemass_code * f << "\n"
              << "Sound speed         : 1.0\n"
              << "Central density     : 1.0\n"
              << "Cloud radius        : " << g_rc << "\n"
              << "Free-fall time      : " << tff_code << "\n"
              << "m=2 perturbation    : " << amp << "\n"
              << "Omega * tff         : " << omegatff << std::endl;
  }

  auto &data = pmb->meshblock_data.Get();
  auto &u_dev = data->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();

  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);

  auto &coords = pmb->coords;
  // Ambient density: BE profile value at r = rc (pressure-continuous)
  InitBEProfile(pin);
  const Real rho_amb = f * BEProfileHost(g_rc);

  // Passive scalars: zero-filled. The first scalar_density component sits just past the hydro
  // vars in the cons pack.
  const int nscalars_ic = pin->GetOrAddInteger("hydro", "nscalars", 0);
  const int iscal0 = mhd ? (IPS + 1) : NHYDRO;


  for (int k = kb.s; k <= kb.e; ++k) {
    Real z = coords.Xc<3>(k);
    for (int j = jb.s; j <= jb.e; ++j) {
      Real y = coords.Xc<2>(j);
      for (int i = ib.s; i <= ib.e; ++i) {
        Real x = coords.Xc<1>(i);
        Real r = std::sqrt(x * x + y * y + z * z);

        Real rho, v1, v2, v3;
        if (r < g_rc) {
          // Inside sphere: BE profile + m=2 azimuthal perturbation + solid-body rotation
          const Real phi = std::atan2(y, x);
          rho = f * BEProfileHost(r) * (1.0 + amp * (r*r/(g_rc*g_rc)) * std::cos(2.0 * phi));
          v1 =  omega_code * y;
          v2 = -omega_code * x;
          v3 = 0.0;
        } else {
          // Outside: ambient, at rest
          rho = rho_amb;
          v1 = 0.0;
          v2 = 0.0;
          v3 = 0.0;
        }

        // Pressure: isothermal c_s = 1 (T=10K), so the BE-equilibrium pressure is
        // p = rho * c_s^2 = rho.
        //
        // ApplyBarotropicCooling overwrites e_th every step, so this sets only the t = 0 state.
        const Real p = rho;

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * v1;
        u(IM2, k, j, i) = rho * v2;
        u(IM3, k, j, i) = rho * v3;
        u(IEN, k, j, i) = p * e_int_over_p + 0.5 * rho * (v1*v1 + v2*v2 + v3*v3);
        if (mhd) {
          u(IB1, k, j, i) = 0.0;
          u(IB2, k, j, i) = 0.0;
          u(IB3, k, j, i) = B0z;
          u(IPS, k, j, i) = 0.0;
          u(IEN, k, j, i) += 0.5 * B0z * B0z;
        }
        for (int s = 0; s < nscalars_ic; ++s) u(iscal0 + s, k, j, i) = 0.0;
      }
    }
  }
  // ============================================================================
  // Initial-only turbulent velocity field (gated by turb_mach > 0).
  // Adds a random divergence-free + compressive velocity perturbation on top
  // of the deterministic IC above. The field is scaled to `turb_mach` via the
  // ANALYTIC ensemble RMS of the mode spectrum, and the ANALYTIC region-mean
  // velocity is subtracted so the cloud doesn't translate as a unit (both are
  // closed-form so every rank/block applies identical factors; see notes below).
  // ============================================================================
  const Real turb_mach = pin->GetOrAddReal("problem/collapse_be", "turb_mach", 0.0);
  if (turb_mach > 0.0) {
    const int   k_min   = pin->GetOrAddInteger("problem/collapse_be", "turb_kmin", 2);
    const int   k_max   = pin->GetOrAddInteger("problem/collapse_be", "turb_kmax", 6);
    const int   nmodes  = pin->GetOrAddInteger("problem/collapse_be", "turb_nmodes", 64);
    const Real  zeta    = pin->GetOrAddReal("problem/collapse_be", "turb_zeta", 0.5);
    const Real  alpha_s = pin->GetOrAddReal("problem/collapse_be", "turb_alpha", 2.0);
    // WP-20: the meaning of turb_alpha depends on how |k| is sampled:
    //   flat (legacy default): dN/dk = const -> E(k) ~ k^-alpha (1D shell slope; Kolmogorov 5/3);
    //   k2 (production): uniform over 3D k-space, dN/dk ~ k^2 -> E(k) ~ k^(2-alpha) (3D PSD slope;
    //   Kolmogorov 11/3 = 3.667). The legacy decks paired 3.667 with `flat`, i.e. far steeper than
    //   Kolmogorov (85 % of the energy at k = 1-2).
    const std::string ksample = pin->GetOrAddString(
        "problem/collapse_be", "turb_ksample", std::string("flat"));
    PARTHENON_REQUIRE(ksample == "flat" || ksample == "k2",
                      "problem/collapse_be/turb_ksample must be 'flat' or 'k2'.");
    const bool ksample_k2 = (ksample == "k2");
    // AUDIT F-43: solenoidal polarization. "random" (DEFAULT) draws the polarization angle in
    // the plane perpendicular to k, giving a statistically isotropic solenoidal field.
    // "axis" is the legacy a x k_hat rule, which is anisotropic by 2.9x with its strong axis
    // along z -- retained ONLY to reproduce pre-2026-08-28 initial conditions bit-for-bit.
    const std::string tpol = pin->GetOrAddString(
        "problem/collapse_be", "turb_polarization", std::string("random"));
    PARTHENON_REQUIRE(tpol == "random" || tpol == "axis",
                      "problem/collapse_be/turb_polarization must be 'random' or 'axis'.");
    const bool pol_random = (tpol == "random");
    const std::string region = pin->GetOrAddString(
        "problem/collapse_be", "turb_region", std::string("sphere"));
    const uint32_t seed = static_cast<uint32_t>(
        pin->GetOrAddInteger("problem/collapse_be", "turb_seed", 12345));

    const Real x1min_ = pin->GetReal("parthenon/mesh", "x1min");
    const Real x1max_ = pin->GetReal("parthenon/mesh", "x1max");
    const Real Lbox  = x1max_ - x1min_;
    const Real two_pi_over_L = 2.0 * M_PI / Lbox;

    // Build the SAME mode list on every rank (deterministic with seed).
    std::mt19937 mrng(seed);
    std::uniform_real_distribution<Real> uphase(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<Real> ucos(-1.0, 1.0);
    std::uniform_real_distribution<Real> uang(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<Real> ukmag(static_cast<Real>(k_min),
                                               static_cast<Real>(k_max));
    // k^2-weighted path (turb_ksample=k2). Inverse transform of p(k) ∝ k^2 on
    // [k_min,k_max]: F(k) = (k^3-k_min^3)/(k_max^3-k_min^3), so k = cbrt(k_min^3 +
    // u*(k_max^3-k_min^3)) with u~U(0,1). Consumes EXACTLY ONE variate per mode, the
    // same as the flat path, so the rest of the RNG stream (directions, phases) is
    // drawn in the identical order and the "flat" branch is bit-identical to legacy.
    std::uniform_real_distribution<Real> uunit(0.0, 1.0);
    const Real k3lo = std::pow(static_cast<Real>(k_min), 3);
    const Real k3hi = std::pow(static_cast<Real>(k_max), 3);

    std::vector<std::array<Real,3>> k_vec(nmodes), phase(nmodes),
                                     eps_sol(nmodes), eps_comp(nmodes);
    std::vector<Real> t_amp(nmodes);

    for (int m = 0; m < nmodes; ++m) {
      const Real kmag = ksample_k2 ? std::cbrt(k3lo + uunit(mrng) * (k3hi - k3lo))
                                   : ukmag(mrng);
      const Real cos_t = ucos(mrng);
      const Real sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
      const Real phi_k = uang(mrng);
      const Real kxh = sin_t * std::cos(phi_k);
      const Real kyh = sin_t * std::sin(phi_k);
      const Real kzh = cos_t;

      k_vec[m] = {kmag * two_pi_over_L * kxh,
                  kmag * two_pi_over_L * kyh,
                  kmag * two_pi_over_L * kzh};
      t_amp[m] = std::pow(kmag, -0.5 * alpha_s);
      phase[m] = {uphase(mrng), uphase(mrng), uphase(mrng)};

      // Solenoidal polarisation, audit F-43: the legacy `axis` rule (eps = a x k_hat with a = x or y)
      // left the field with no x-component in ~2/3 of modes (measured <vx^2>:<vy^2>:<vz^2> =
      // 0.18:0.31:0.52, anisotropic along B0z for every seed). `random` (production) draws the angle
      // uniformly in the plane perpendicular to k_hat; `axis` reproduces the legacy field bit-identically.
      Real sx, sy, sz;
      if (pol_random) {
        // Orthonormal transverse basis via the numerically stable Hughes-Moller construction:
        // reflect the axis with the SMALLEST |component|, which is guaranteed non-degenerate.
        Real ax = 0.0, ay = 0.0, az = 0.0;
        if (std::fabs(kxh) <= std::fabs(kyh) && std::fabs(kxh) <= std::fabs(kzh)) ax = 1.0;
        else if (std::fabs(kyh) <= std::fabs(kzh)) ay = 1.0;
        else az = 1.0;
        Real e1x = ay * kzh - az * kyh;
        Real e1y = az * kxh - ax * kzh;
        Real e1z = ax * kyh - ay * kxh;
        Real n1 = std::sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
        if (n1 < 1e-12) n1 = 1.0;
        e1x /= n1; e1y /= n1; e1z /= n1;
        // e2 = k_hat x e1 (already unit, since both are unit and orthogonal)
        const Real e2x = kyh * e1z - kzh * e1y;
        const Real e2y = kzh * e1x - kxh * e1z;
        const Real e2z = kxh * e1y - kyh * e1x;
        const Real psi = uang(mrng);   // polarization angle, U(0, 2pi)
        const Real cp = std::cos(psi), sp = std::sin(psi);
        sx = cp * e1x + sp * e2x;
        sy = cp * e1y + sp * e2y;
        sz = cp * e1z + sp * e2z;
      } else {
        // LEGACY (anisotropic; kept only to reproduce pre-2026-08-28 ICs bit-for-bit).
        Real ax = 1.0, ay = 0.0, az = 0.0;
        if (std::fabs(kxh) > std::fabs(kyh) && std::fabs(kxh) > std::fabs(kzh)) {
          ax = 0.0; ay = 1.0; az = 0.0;
        }
        sx = ay * kzh - az * kyh;
        sy = az * kxh - ax * kzh;
        sz = ax * kyh - ay * kxh;
      }
      Real snorm = std::sqrt(sx*sx + sy*sy + sz*sz);
      if (snorm < 1e-12) snorm = 1.0;
      eps_sol[m]  = {sx/snorm, sy/snorm, sz/snorm};
      eps_comp[m] = {kxh, kyh, kzh};
    }

    // Evaluate field on this block's local cells (host scratch).
    const int nx_loc = ib.e - ib.s + 1;
    const int ny_loc = jb.e - jb.s + 1;
    const int nz_loc = kb.e - kb.s + 1;
    std::vector<Real> v_turb(3 * nx_loc * ny_loc * nz_loc, 0.0);
    auto vt = [&](int c, int kk, int jj, int ii) -> Real & {
      return v_turb[((c * nz_loc + (kk - kb.s)) * ny_loc + (jj - jb.s)) * nx_loc
                    + (ii - ib.s)];
    };

    for (int k = kb.s; k <= kb.e; ++k) {
      const Real z = coords.Xc<3>(k);
      for (int j = jb.s; j <= jb.e; ++j) {
        const Real y = coords.Xc<2>(j);
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real x = coords.Xc<1>(i);
          Real vx_t = 0.0, vy_t = 0.0, vz_t = 0.0;
          for (int m = 0; m < nmodes; ++m) {
            const Real kdotr = k_vec[m][0] * x + k_vec[m][1] * y + k_vec[m][2] * z;
            const Real w_sol  = std::sqrt(zeta) * t_amp[m]
                              * std::cos(kdotr + phase[m][0]);
            const Real w_comp = std::sqrt(1.0 - zeta) * t_amp[m]
                              * std::cos(kdotr + phase[m][1]);
            vx_t += w_sol * eps_sol[m][0]  + w_comp * eps_comp[m][0];
            vy_t += w_sol * eps_sol[m][1]  + w_comp * eps_comp[m][1];
            vz_t += w_sol * eps_sol[m][2]  + w_comp * eps_comp[m][2];
          }
          vt(0, k, j, i) = vx_t;
          vt(1, k, j, i) = vy_t;
          vt(2, k, j, i) = vz_t;
        }
      }
    }

    // -----------------------------------------------------------------------
    // Apply analytical normalization to the unscaled mode-sum field.
    // We do NOT use a mass-weighted RMS reduction across the mesh — that
    // approach fails when ProblemGenerator is called per-block and block 0
    // (which prints diagnostics) sits entirely outside the sphere. With
    // 1 MPI rank the MPI_Allreduce returns this block's own data, not the
    // global mesh state, so per-block rescaling becomes inconsistent and
    // outside-sphere blocks silently skip turbulence entirely.
    //
    // Instead we compute the field's expected RMS analytically from the
    // mode spectrum and apply a fixed scale factor identical on every block.
    //
    // For an isotropic random field with t_amp[m] = kmag^(-alpha/2),
    // unit-variance cos terms, and orthogonal sol/comp projections each
    // unit-norm, the per-component variance is:
    //   <v_x^2> = sum_m [zeta * <cos^2> * eps_sol_x^2 + (1-zeta) * <cos^2> * eps_comp_x^2]
    //          * t_amp[m]^2
    // Summed over (x,y,z), the eps^2 components sum to 1 for each mode (since
    // both eps_sol and eps_comp are unit vectors), so:
    //   <|v|^2> = sum_m t_amp[m]^2 * <cos^2>(zeta + (1-zeta)) = 0.5 * sum_m t_amp[m]^2
    // -----------------------------------------------------------------------
    Real sum_amp_sq = 0.0;
    for (int m = 0; m < nmodes; ++m) {
      sum_amp_sq += t_amp[m] * t_amp[m];
    }
    const Real vrms_analytic = std::sqrt(0.5 * sum_amp_sq);
    const Real scale = (vrms_analytic > 0.0) ? (turb_mach / vrms_analytic) : 0.0;

    // -----------------------------------------------------------------------
    // Analytic mean of the mode-sum field over the application region, subtracted
    // before scaling so the cloud receives no net momentum kick (for ~64 modes the
    // sphere-restricted mean is O(a few %) of v_rms). Computed analytically -- so it
    // is deterministic and identical on every rank -- for the same reason as the RMS
    // normalization above: a mesh-wide reduction is not per-block consistent here.
    //   sphere (centered on origin): <cos(k.r + phi)> = cos(phi) * W(|k| rc),
    //       W(x) = 3 (sin x - x cos x) / x^3   (-> 1 as x -> 0)
    //   box:  <cos(k.r + phi)> = cos(k.r_c + phi) * prod_d sinc(k_d L/2)
    // -----------------------------------------------------------------------
    const bool in_sphere = (region == "sphere");
    Real vmean[3] = {0.0, 0.0, 0.0};
    for (int m = 0; m < nmodes; ++m) {
      Real wcos_sol, wcos_comp; // region-mean of the sol/comp cosine terms
      if (in_sphere) {
        const Real kmagp =
            std::sqrt(k_vec[m][0] * k_vec[m][0] + k_vec[m][1] * k_vec[m][1] +
                      k_vec[m][2] * k_vec[m][2]);
        const Real x = kmagp * g_rc;
        const Real W =
            (x < 1.0e-4) ? 1.0 : 3.0 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
        wcos_sol = W * std::cos(phase[m][0]);
        wcos_comp = W * std::cos(phase[m][1]);
      } else {
        // Cubic box; same center in each dimension (matches the k-space setup above).
        const Real xc = 0.5 * (x1min_ + x1max_);
        Real sincs = 1.0;
        Real kdotc = 0.0;
        for (int d = 0; d < 3; ++d) {
          const Real arg = 0.5 * k_vec[m][d] * Lbox;
          sincs *= (std::fabs(arg) < 1.0e-8) ? 1.0 : std::sin(arg) / arg;
          kdotc += k_vec[m][d] * xc;
        }
        wcos_sol = sincs * std::cos(kdotc + phase[m][0]);
        wcos_comp = sincs * std::cos(kdotc + phase[m][1]);
      }
      const Real a_sol = std::sqrt(zeta) * t_amp[m] * wcos_sol;
      const Real a_comp = std::sqrt(1.0 - zeta) * t_amp[m] * wcos_comp;
      for (int c = 0; c < 3; ++c) {
        vmean[c] += a_sol * eps_sol[m][c] + a_comp * eps_comp[m][c];
      }
    }

    if (parthenon::Globals::my_rank == 0 && pmb->gid == 0) {
      std::cout << "\n--- Initial turbulence ---\n"
                << "Region              : " << region << "\n"
                << "Target Mach (RMS)   : " << turb_mach << "\n"
                << "Modes               : " << nmodes
                << "  k in [" << k_min << ", " << k_max << "]"
                << "  zeta=" << zeta << " (solenoidal energy fraction)"
                << "  alpha=" << alpha_s << "\n"
                // WP-20: state the resulting spectrum explicitly. turb_alpha means
                // different things under the two samplers and the difference is large.
                << "Solenoidal polariz. : " << tpol
                << (pol_random ? "  (isotropic; audit F-43)"
                               : "  (LEGACY axis rule -- ANISOTROPIC 2.9x, strong axis z)")
                << "\n"
                << "k sampling          : " << ksample
                << (ksample_k2 ? "  (dN/dk ~ k^2; alpha is the 3D PSD slope)"
                               : "  (dN/dk = const; alpha is the 1D shell slope)")
                << "\n"
                << "=> E(k) ~ k^-"
                << (ksample_k2 ? alpha_s - 2.0 : alpha_s)
                << "   [Kolmogorov = k^-1.667]\n"
                << "Analytic vrms       : " << vrms_analytic << " (code v)\n"
                << "Scale factor        : " << scale << "\n"
                << "Mean subtracted     : (" << vmean[0] * scale << ", "
                << vmean[1] * scale << ", " << vmean[2] * scale << ") (code v)"
                << std::endl;
    }

    // Apply the scaled, mean-subtracted turbulent velocity to the region's cells.
    for (int k = kb.s; k <= kb.e; ++k) {
      const Real z = coords.Xc<3>(k);
      for (int j = jb.s; j <= jb.e; ++j) {
        const Real y = coords.Xc<2>(j);
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real x = coords.Xc<1>(i);
          if (in_sphere && (x*x + y*y + z*z) >= g_rc * g_rc) continue;
          const Real dvx = (vt(0, k, j, i) - vmean[0]) * scale;
          const Real dvy = (vt(1, k, j, i) - vmean[1]) * scale;
          const Real dvz = (vt(2, k, j, i) - vmean[2]) * scale;
          const Real rho     = u(IDN, k, j, i);
          const Real vx_old  = u(IM1, k, j, i) / rho;
          const Real vy_old  = u(IM2, k, j, i) / rho;
          const Real vz_old  = u(IM3, k, j, i) / rho;
          const Real vx_new  = vx_old + dvx;
          const Real vy_new  = vy_old + dvy;
          const Real vz_new  = vz_old + dvz;
          u(IM1, k, j, i) = rho * vx_new;
          u(IM2, k, j, i) = rho * vy_new;
          u(IM3, k, j, i) = rho * vz_new;
          const Real ke_old = 0.5 * rho * (vx_old*vx_old + vy_old*vy_old + vz_old*vz_old);
          const Real ke_new = 0.5 * rho * (vx_new*vx_new + vy_new*vy_new + vz_new*vz_new);
          u(IEN, k, j, i) += (ke_new - ke_old);
        }
      }
    }

    // Audit F-44 (diagnostic only, reads u): report the REALISED angular momentum -- with turb_kmin = 1
    // the box-scale mode adds net rotation (omega_z 0.19-2.3x the omegatff value over 8 seeds, axis
    // tilted > 45 deg from z in 5 of 8), which controls braking relative to B0z.
    {
      Real Lx = 0.0, Ly = 0.0, Lz = 0.0, Ixx = 0.0, Iyy = 0.0, Izz = 0.0, Msph = 0.0;
      for (int k = kb.s; k <= kb.e; ++k) {
        const Real z = coords.Xc<3>(k);
        for (int j = jb.s; j <= jb.e; ++j) {
          const Real y = coords.Xc<2>(j);
          for (int i = ib.s; i <= ib.e; ++i) {
            const Real x = coords.Xc<1>(i);
            if (x*x + y*y + z*z >= g_rc * g_rc) continue;
            const Real m = u(IDN, k, j, i) * coords.CellVolume(k, j, i);
            const Real vx = u(IM1, k, j, i) / u(IDN, k, j, i);
            const Real vy = u(IM2, k, j, i) / u(IDN, k, j, i);
            const Real vz = u(IM3, k, j, i) / u(IDN, k, j, i);
            Lx += m * (y*vz - z*vy);
            Ly += m * (z*vx - x*vz);
            Lz += m * (x*vy - y*vx);
            Ixx += m * (y*y + z*z);
            Iyy += m * (z*z + x*x);
            Izz += m * (x*x + y*y);
            Msph += m;
          }
        }
      }
      if (Msph > 0.0) {
        // SIGN: the pgen imposes v1 = +omega*y, v2 = -omega*x, so the physical angular
        // momentum is Lz = -omega*I_zz -- the deck's `omegatff` has the OPPOSITE sign to the
        // conventional omega = L/I. Negate here so the printed value is directly comparable
        // to `omega_code` below, which is what a reader will check it against.
        const Real ox = (Ixx > 0.0) ? -Lx / Ixx : 0.0;
        const Real oy = (Iyy > 0.0) ? -Ly / Iyy : 0.0;
        const Real oz = (Izz > 0.0) ? -Lz / Izz : 0.0;
        const Real omag = std::sqrt(ox*ox + oy*oy + oz*oz);
        const Real tilt = (omag > 0.0)
            ? std::acos(std::min(1.0, std::max(-1.0, std::abs(oz) / omag))) * 180.0 / M_PI
            : 0.0;
        std::cout << "## [IC angular momentum, THIS BLOCK] M=" << Msph
                  << " omega=(" << ox << "," << oy << "," << oz << ")"
                  << " |omega|=" << omag << " tilt_from_z=" << tilt << " deg"
                  << "  (imposed omega_z = " << omega_code
                  << "; audit F-44: omegatff does NOT describe the realized rotation --"
                     " sum over blocks for the whole-sphere value)" << std::endl;
      }
    }
  }
  u_dev.DeepCopy(u);
}

// ---------------------------------------------------------------------------
// Barotropic cooling source term (Masunaga-Inutsuka / Tomida form):
//   e_th = rho/(gamma-1) * sqrt(1 + (rho/rhocrit)^(2(gamma-1)))
// For rho << rhocrit: e_th ~ rho/(gamma-1) (isothermal, c_s = 1)
// For rho >> rhocrit: e_th ~ (rho/(gamma-1)) * (rho/rhocrit)^(gamma-1) (adiabatic)
// This OVERWRITES the thermal energy every step — it's a barotropic EOS
// enforcement, not a differential cooling equation.
//
// Also zeroes momentum outside the sphere (fixed boundary condition on velocity).
// ---------------------------------------------------------------------------
TaskStatus ApplyBarotropicCooling(MeshData<Real> *md, const parthenon::SimTime &tm,
                                   const Real dt) {
  auto pm = md->GetParentPointer();
  auto hydro_pkg = pm->packages.Get("Hydro");
  const Real rhocrit = hydro_pkg->Param<Real>("collapse_be_rhocrit");
  const Real rc = hydro_pkg->Param<Real>("collapse_be_rc");
  const Real gam = hydro_pkg->Param<Real>("collapse_be_gamma");
  const bool mhd = hydro_pkg->Param<bool>("collapse_be_mhd");
  const Real gm1 = gam - 1.0;
  const Real igm1 = 1.0 / gm1;
  const Real rcsq = rc * rc;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  parthenon::IndexRange ib = md->GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb = md->GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb = md->GetBoundsK(parthenon::IndexDomain::interior);
  const int nblocks = md->NumBlocks();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "BEcollapse::BaroCool", parthenon::DevExecSpace(),
      0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.template Xc<1>(i);
        const Real y = coords.template Xc<2>(j);
        const Real z = coords.template Xc<3>(k);
        const Real r2 = x*x + y*y + z*z;

        // Outside sphere: zero velocity (fixed BC on momentum)
        if (r2 > rcsq) {
          cons(IM1, k, j, i) = 0.0;
          cons(IM2, k, j, i) = 0.0;
          cons(IM3, k, j, i) = 0.0;
        }

        // Barotropic EOS enforcement (everywhere); IEN is fully overwritten.
        {
          const Real rho = cons(IDN, k, j, i);
          const Real ke = 0.5 / rho * (cons(IM1, k, j, i) * cons(IM1, k, j, i)
                                     + cons(IM2, k, j, i) * cons(IM2, k, j, i)
                                     + cons(IM3, k, j, i) * cons(IM3, k, j, i));
          const Real me = mhd ? 0.5 * (cons(IB1, k, j, i) * cons(IB1, k, j, i)
                                     + cons(IB2, k, j, i) * cons(IB2, k, j, i)
                                     + cons(IB3, k, j, i) * cons(IB3, k, j, i))
                               : 0.0;
          const Real te = igm1 * rho * std::sqrt(1.0 + std::pow(rho / rhocrit, 2.0 * gm1));
          cons(IEN, k, j, i) = te + ke + me;
        }
      });
  return TaskStatus::complete;
}

} // namespace collapse_be
