//========================================================================================
// AthenaPK - WS-4 single-moment dust evolution (device/host).
//
// Evolves two passive scalars: the dust-to-gas mass ratio f_dg and the mass-weighted
// characteristic grain radius a_c [cm]. Grains with a_c < um have Stokes number << 1 at
// core densities => they co-move with the gas (no drift); what matters is (i) GROWTH
// (raises a_c -> lowers total cross-section -> lowers kappa, raises the non-ideal etas) and
// (ii) SUBLIMATION above T_subl. This is the monodisperse sweep-up model (NOT the Artemis
// two-fluid dust + coagulation, which is the drift-limited fallback).
//
//   da_c/dt = C (f_dg * rho_gas / rho_grain) * v_rel, C = growth_pref = 1/4 (monodisperse sweep-up)
//   v_rel   = sqrt(v_Brownian^2 + v_turb^2)
//     v_Brownian = sqrt(8 k_B T / (pi m_red)),  m_red = m_grain/2, m_grain=(4/3)pi a^3 rho_gr
//     v_turb     = sqrt(alpha_turb) * c_s * St,  St = (rho_grain a_c)/(rho_gas v_th) * Omega
//                  Omega = sqrt(4 pi G rho_gas), the LOCAL dynamical frequency (2026-08-15;
//                  was a single hard-coded omega_ref -- see omega_dyn for why that mattered
//                  in principle and why it turns out not to matter in practice)
//   f_dg -> 0 smoothly above T_subl (sublimation). IRREVERSIBLE by default since 2026-08-15:
//     grains are destroyed but not re-created, and a_c shrinks with them at fixed grain
//     number (a ~ f_dg^(1/3)). <dust> subl_irreversible=false restores the old reversible,
//     lag-free, a_c-preserving behaviour bit-identically. See integrate_cell.
//
// UNITS: this header works in cgs (rho_gas g/cm^3, T K, a_c cm, c_s cm/s). The caller
// converts code<->cgs. All growth is capped at <=10%/step (the a_c->ionization->dt coupling
// risk) inside integrate_cell.
//========================================================================================
#ifndef DUST_DUST_HPP_
#define DUST_DUST_HPP_

#ifdef KOKKOS_INLINE_FUNCTION
#define DUST_FN KOKKOS_INLINE_FUNCTION
#else
#define DUST_FN inline
#endif

#include <cmath>

namespace Dust {

namespace dcgs {
constexpr double kB = 1.380649e-16;    // erg/K
constexpr double pi = 3.14159265358979323846;
} // namespace dcgs

//----------------------------------------------------------------------------------------
//! Single-moment dust parameters (cgs unless noted). Defaults are FHC-representative.
struct DustModel {
  double rho_grain = 3.0;    // internal grain density [g/cm^3] (silicate ~3)
  double a_ref = 1.0e-5;     // reference grain radius [cm] (0.1 um, MRN a_max-ish)
  double f_dg_ref = 0.01;    // canonical dust-to-gas ratio (ISM)
  double alpha_turb = 1.0e-2; // turbulent alpha for v_turb
  double omega_ref = 1.0e-13; // reference frequency [s^-1] setting St (~ 1/t_ff at n~1e6)
  double mu_gas = 2.33;      // mean molecular weight (for v_th, c_s)
  double m_H = 1.6726219e-24;
  double T_subl = 1.5e3;     // sublimation temperature [K]
  double subl_width = 100.0; // K, smoothness of the sublimation switch
  double a_floor = 1.0e-7;   // minimum grain radius [cm] (1 nm)
  double growth_cap = 0.10;  // max fractional a_c change per sub-step (dt-thrash guard)
  int    nsub_max = 200;
  //! Sweep-up prefactor C in da/dt = C (f_dg rho / rho_grain) v_rel. <dust> growth_prefactor
  //! (validation_tests/dust D2, 2026-10-12). Default 0.25 = the historical "/ 4.0" BIT-FOR-BIT
  //! (x * 0.25 and x / 4.0 are the same correctly rounded value in IEEE-754). The monodisperse
  //! Smoluchowski limit (dn/dt = -n^2 sigma v_rel / 2, sigma = pi (2a)^2, mass conserved) gives
  //! C = 0.5; the "one grain sweeps up the others through pi (2a)^2" picture gives C = 1;
  //! C = 0.25 corresponds to sigma = pi a^2. See validation_tests/dust D2 for the A/B.
  double growth_pref = 0.25;
  //! Sublimation centre (validation_tests/dust DD5, 2026-10-12). 0 = `fixed` (default): T_subl, the historical
  //! constant. 1 = `semenov`: T5(rho), the refractory (silicate + iron) evaporation temperature of Semenov et al.
  //! (2003) opacity.f -- the SAME function gen_opacity_table.py --model semenov used to build the v4 (v12) opacity
  //! table, so the dust package and the table agree on where the grains are gone. <dust> subl_model.
  int subl_model = 0;
  //! Finite sublimation time tau [s] (validation_tests/dust DD2, 2026-10-13). 0 (default) = the instantaneous
  //! equilibrium switch, bit-identical. > 0: grains erode toward f_eq = f_ref s(T) as f -> f_eq + (f - f_eq) exp(-dt/tau)
  //! (exact for a constant-T step; still irreversible: f never rises), a_c shrinking at fixed grain number. WHY: the
  //! instantaneous rule records the MAXIMUM temperature a parcel is ever seen at, and every captured shock shows each
  //! parcel a post-shock overshoot of ~0.5 % T2 lasting ~1.2 dx/S (D5 trace) -- self-similar in dx, so no grid-scaled
  //! filter converges (D5/F2). A fixed PHYSICAL tau integrates over that transient once 1.2 dx/S << tau, so the
  //! deficit falls with resolution. tau is a model parameter (grain evaporation kinetics), not a numerical one.
  double subl_tau = 0.0;
  // --- 2026-08-15 fixes; both have an OFF value that reproduces the old code exactly ---
  //! Use the LOCAL dynamical frequency Omega = sqrt(4 pi G rho) for the Stokes number
  //! instead of the hard-coded omega_ref. <dust> omega_local, default true.
  bool omega_local = true;
  //! Sublimation destroys grains irreversibly (f_dg can only DECREASE toward the local
  //! equilibrium) instead of being re-created the instant the gas cools. <dust>
  //! subl_irreversible, default true. See integrate_cell for the argument.
  bool subl_irreversible = true;
  //! --- 2026-09-16 opacity-coupling corrections (AUDIT F-06 / F-07). BOTH default ON (the
  //! fixes) since the 2026-09-16 A/B; the OFF values execute the IDENTICAL historical
  //! expression, so geometric/false is bit-identical to the pre-fix binary. See DustFactor*.
  //! <dust> opacity_size_law: "bridged" (default, F-06 fix) or "geometric" (historical).
  bool opac_size_bridged = true;
  //! <dust> opacity_gas_split (F-07 fix): restrict the dust multiplier to the DUST part of
  //! the tabulated opacity, leaving the molecular / H- / Kramers / electron-scattering GAS
  //! opacity alone. Applied at the consumer as kd_eff = 1 + w*(kd-1). Default true.
  bool opac_gas_split = true;
  // code-unit conversions (for the in-code path; the standalone test works in cgs directly)
  double rho_unit = 5.467e-19;
  double t_unit = 1.476998822551e12;

  //! Local dynamical frequency [s^-1] used for the Stokes number.
  //!
  //! WHY THIS IS NOT omega_ref (2026-08-15). `omega_ref = 1e-13 s^-1` was a single hard-coded
  //! constant "~ 1/t_ff at n~1e6" used at EVERY density. Over the collapse this problem spans
  //! rho = 1e-19 -> 1e-6 g/cm^3, where sqrt(4 pi G rho) runs 2.9e-13 -> 9.2e-7 s^-1: the
  //! constant is wrong by up to 9.2e6, and no deck key could correct it per-cell.
  //!
  //! HONESTY ABOUT THE SIZE OF THIS FIX, measured rather than asserted (offline integration of
  //! this very law over one local t_ff at eight (rho,T) states spanning the collapse):
  //! it changes essentially NOTHING, because the sweep-up velocity is BROWNIAN-dominated
  //! wherever it matters. At rho = 1e-15 g/cm^3, v_B = 0.75 cm/s vs v_turb = 1.9e-4 (fixed)
  //! or 5.4e-2 (local); at rho = 1e-10, v_B = 4.1 vs 1.9e-9 or 1.7e-4. Integrated over the
  //! whole collapse a_c goes 1.000e-05 -> 1.0369e-04 cm under BOTH -- identical to four
  //! significant figures. Only the outermost envelope (rho <~ 1e-17, where Omega_loc/omega_ref
  //! = 2.9-29 and v_turb finally exceeds v_B) sees any difference at all, and there the total
  //! growth is ~1e-4 in a_c/a. So this is a correctness fix, NOT a result-changing one, and it
  //! must not be quoted as one.
  DUST_FN double omega_dyn(double rho_gas) const {
    if (!omega_local) return omega_ref;
    constexpr double G_cgs = 6.674e-8;
    return std::sqrt(4.0 * dcgs::pi * G_cgs * rho_gas);
  }

  //! Relative grain-grain velocity [cm/s] at (rho_gas, T, a_c).
  DUST_FN double v_rel(double rho_gas, double T, double a_c) const {
    const double cs = std::sqrt(dcgs::kB * T / (mu_gas * m_H)); // isothermal sound speed
    const double v_th = std::sqrt(8.0 / dcgs::pi) * cs;         // mean thermal speed
    const double m_grain = (4.0 / 3.0) * dcgs::pi * a_c * a_c * a_c * rho_grain;
    const double m_red = 0.5 * m_grain;
    const double v_B = std::sqrt(8.0 * dcgs::kB * T / (dcgs::pi * m_red));
    // Epstein Stokes number: St = t_stop * Omega, t_stop = rho_grain a_c/(rho_gas v_th).
    const double St = (rho_grain * a_c / (rho_gas * v_th)) * omega_dyn(rho_gas);
    const double v_t = std::sqrt(alpha_turb) * cs * St;
    return std::sqrt(v_B * v_B + v_t * v_t);
  }

  //! da_c/dt [cm/s] (monodisperse sweep-up) at fixed gas state and f_dg.
  DUST_FN double dadt(double rho_gas, double T, double a_c, double f_dg) const {
    return (f_dg * rho_gas / rho_grain) * v_rel(rho_gas, T, a_c) * growth_pref;
  }

  //! Sublimation factor in [0,1]: 1 below T_subl, ->0 above (smooth tanh switch).
  //!
  //! AUDIT F-26 (2026-08-28) -- THREE UNCOUPLED SUBLIMATION PRESCRIPTIONS RUN AT ONCE, AND
  //! THE RUN HAS ALREADY CROSSED THE ONE THEY DISAGREE ABOUT.
  //!   (1) THIS tanh, width 100 K about <dust> T_subl_K = 1500 K. Sets f_dg, hence kdust,
  //!       hence ALL THREE opacities AND the ionization grain population.
  //!   (2) The opacity TABLE's own two-species cascade, baked into opacity_table_v*.bin
  //!       (gen_opacity_table.py DUST_SPECIES): volatile ice mantles 140-170 K carrying 35 %
  //!       of the dust opacity, refractories 1400-1600 K carrying the other 65 %.
  //!   (3) <radiation> dust_Tmax_K = 1500 -- inert under opacity_model = tabulated (audit
  //!       F-25), commented out of the live decks.
  //! Nothing couples (1) and (2). The production run's maximum gas temperature is 192.9 K,
  //! i.e. ALREADY PAST the ice front: in those cells the opacity table has removed 35 % of
  //! the dust opacity while this factor still returns ~1.0 and the dust package reports
  //! f_dg = 0.01 exactly. That is not double counting -- grain growth and ice sublimation are
  //! different physics -- but the two modules hold inconsistent views of how much dust exists,
  //! and the inconsistency grows through the first-core to second-core transition where both
  //! fronts are crossed. Coupling them (or deriving one from the other) is a physics decision,
  //! not a bug fix; it is recorded here so the split is explicit rather than accidental.
  DUST_FN double subl_factor(double T) const {
    return 0.5 * (1.0 - std::tanh((T - T_subl) / subl_width));
  }

  //! Sublimation factor at (T [K], rho [g/cm^3]): subl_factor(T) for subl_model = fixed (the SAME expression, so
  //! the default is bit-identical), the tanh about SemenovT5(rho) for subl_model = semenov.
  DUST_FN double subl_factor_at(double T, double rho_cgs) const;

  //! Apply the sublimation rule to (a_c, f_dg) at temperature T. Shared by integrate_cell and
  //! by the freeze_growth path in dust.cpp so the two cannot drift apart.
  //!
  // ---- THE DEFECT THIS REPLACES (2026-08-15) -------------------------------------------
  // The old line was, unconditionally,
    //     f_dg = f_dg_ref * subl_factor(T);
    // i.e. f_dg was RECOMPUTED from the local temperature every step and the advected value
    // read out of the scalar was discarded. Two consequences, both wrong:
    //   (1) Sublimation was instantaneous AND fully reversible. A parcel heated through
    //       T_subl lost its grains; the moment it cooled again -- by expansion, by radiative
    //       cooling, by mixing -- the FULL ISM dust abundance reappeared. Dust mass was
    //       created from nothing. This is not a corner case here: the first core interior
    //       reaches T_subl on its way to the second collapse, and shocked material that
    //       subsequently cools is exactly the material whose opacity and grain-surface
    //       ionisation the run depends on.
    //   (2) a_c was untouched by sublimation, so the resurrected dust came back at whatever
    //       grain size it had grown to. Measured offline over this problem's own (rho,T)
    //       track, a_c grows 10.4x across the collapse, and both the dust opacity and the
    //       grain-surface term in the ionisation model scale as 1/a_c -- so the size that
    //       came back for free was a factor ~10 wrong in the consumer.
    //
    // THE FIX. Grains can be destroyed but not re-created: f_dg only ever moves DOWN toward
    // the local equilibrium f_dg_ref*subl_factor(T). Re-condensation is a real process but it
    // is slow compared with the collapse and is not modelled, so the honest statement is
    // "no re-formation", not "instant re-formation".
    // Sublimation erodes grains at FIXED GRAIN NUMBER (they shrink, they do not vanish one by
    // one), so with n_gr ~ f_dg/a_c^3 held fixed, a_c scales as f_dg^(1/3). Applying that
    // keeps f_dg and a_c mutually consistent instead of letting them drift apart.
    //
  // subl_irreversible = false restores the old expression EXACTLY (bit-identical), which is
  // the anchor the A/B is measured against.
  DUST_FN void apply_sublimation(double &a_c, double &f_dg, double T, double rho_cgs = -1.0,
                                 double dt_s = -1.0) const {
    // rho_cgs < 0 (host gates that predate DD5): the fixed centre, exactly as before.
    const double f_eq = f_dg_ref * (rho_cgs > 0.0 ? subl_factor_at(T, rho_cgs) : subl_factor(T));
    if (!subl_irreversible) {
      f_dg = f_eq; // LEGACY: instantaneous and reversible. The bit-identity anchor.
      return;
    }
    // f_dg <= 0 means the scalar was never seeded (the pgen writes rho*f_dg_ref, and the tanh
    // switch bottoms out at ~1e-9, never at exactly 0). Treat that as "unseeded" and start
    // from the reference rather than locking the cell at zero dust forever -- the failure mode
    // a one-sided rule would otherwise introduce on any restart taken from a run with dust off.
    // F7a (code review): discriminate "never seeded" from "destroyed" using a_c. The old test
    // treated BOTH as unseeded, so a cell whose grains had genuinely sublimated (the tanh bottoms
    // at ~1e-9 f_dg_ref) and whose advected rho*f_dg then undershot to <= 0 in a shock or across
    // an AMR restriction was restored to the FULL ISM abundance -- dust mass created from
    // nothing, the exact defect the irreversible path exists to remove. A truly unseeded cell
    // (dust off, or a restart from a run without dust) has a_c == 0 as well; a destroyed cell
    // retains its grain size.
    // AUDIT 2026-09-15 (docs/AUDIT_DUST_2026-09-15.md F-14): "start from the reference" must
    // restore BOTH scalars. The unseeded branch used to reset f_dg only and leave a_c = 0, so
    // the very next consumer saw a_c = a_floor (1e-7 cm, growth path => kdust = a_ref/a_floor
    // = 100) or a_c = 0 (freeze_growth path, never floored => kdust = 1e25): a hundred-fold to
    // astronomically wrong opacity in every cell that entered dust-on from a dust-off state,
    // and (growth path) ~2e3 yr of spurious 1-nm-grain sweep-up before a_c recovered. Never
    // fired in the production chain (seeded by the pgen at t=0; F-05 finds a_c >= a_ref in all
    // 4.9e7 cells), so this changes nothing there; it makes the documented recovery correct.
    double f_in;
    if (f_dg > 0.0)      f_in = f_dg;        // normal
    else if (a_c <= 0.0) { f_in = f_dg_ref; a_c = a_ref; } // never seeded -> the reference pair
    else                 f_in = 0.0;         // destroyed and undershot -> stays destroyed
    // F7b: only shrink when sublimation is ACTUALLY removing dust. `f_eq < f_in` alone fires on
    // truncation-level overshoot above f_dg_ref at temperatures where subl_factor == 1,
    // ratcheting a_c down once per step in cells nowhere near T_subl -- and a_c feeds both the
    // dust opacity and the grain-surface ionisation as 1/a_c.
    constexpr double kSublRelTol = 1.0e-12;
    if (f_in > 0.0 && f_eq < f_in * (1.0 - kSublRelTol)) {
      // DD2: finite sublimation time (subl_tau > 0 and a step length given); else the instantaneous switch
      const double f_new = (subl_tau > 0.0 && dt_s > 0.0) ? f_eq + (f_in - f_eq) * std::exp(-dt_s / subl_tau) : f_eq;
      const double ratio = f_new / f_in;
      a_c *= std::cbrt(ratio); // fixed grain number: a ~ f_dg^(1/3)
      f_in = f_new;
    }
    f_dg = f_in;
  }

  //! Advance (a_c, f_dg) over dt_s [s] at fixed gas state. Sublimation first (see
  //! apply_sublimation), then a_c grows by monodisperse sweep-up (RK2 midpoint; sub-step
  //! capped at growth_cap/step to bound the a_c->ionization->dt coupling). Writes a_c, f_dg
  //! in place; returns nsub used.
  DUST_FN int integrate_cell(double &a_c, double &f_dg, double rho_gas, double T,
                             double dt_s) const {
    apply_sublimation(a_c, f_dg, T, rho_gas, dt_s);
    if (a_c < a_floor) a_c = a_floor; // guard v_Brownian ~ 1/sqrt(a^3) against a_c -> 0
    return grow_cell(a_c, f_dg, rho_gas, T, dt_s);
  }

  //! The growth half of integrate_cell (split out 2026-10-12 for dust/subl_shock_hold; integrate_cell's arithmetic is
  //! unchanged: the same statements in the same order). Expects a_c >= a_floor.
  DUST_FN int grow_cell(double &a_c, const double f_dg, double rho_gas, double T, double dt_s) const {
    double t = 0.0;
    int nsub = 0;
    const double dt_floor = dt_s / static_cast<double>(nsub_max);
    while (t < dt_s && nsub < nsub_max) {
      const double rate = dadt(rho_gas, T, a_c, f_dg);
      double dt_sub = dt_s - t;
      if (rate > 0.0) {
        const double dt_cap = growth_cap * a_c / rate; // |da|/a <= growth_cap
        if (dt_cap < dt_sub) dt_sub = dt_cap;
      }
      if (dt_sub < dt_floor) dt_sub = dt_floor;
      if (t + dt_sub > dt_s) dt_sub = dt_s - t;
      // RK2 midpoint (2nd order -> tracks exponential/power-law growth accurately).
      const double a_mid = a_c + 0.5 * dt_sub * rate;
      const double rate_mid = dadt(rho_gas, T, a_mid, f_dg);
      a_c += dt_sub * rate_mid;
      if (a_c < a_floor) a_c = a_floor;
      t += dt_sub;
      ++nsub;
    }
    return nsub;
  }
};

//----------------------------------------------------------------------------------------
//! Refractory evaporation temperature T5(rho) [K] of Semenov et al. (2003, A&A 410, 611) opacity.f (`cop`): the
//! three refractory rows of its evaporation table (8 densities 1e-18 .. 1e-4 g/cm^3), each interpolated LINEARLY
//! IN RHO between the bracketing grid points and clamped at the grid ends (opacity.f `bint`), then
//! T5 = min(max(T_2, T_3), max(T_3, T_4)). A line-by-line port of src/radiation/opacity_semenov.py
//! evap_temperatures()[4], which gen_opacity_table.py --model semenov used for the v4/v12 tables (gated against it
//! in test_dust.cpp gate (n)). 902 K at 1e-18, 1129 K at 1e-12, 1245 K at 1e-9, 1395 K at 1e-8 g/cm^3.
DUST_FN double SemenovT5(double rho_cgs) {
  constexpr double ro[8] = {1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4};
  constexpr double t2[8] = {835.0, 908.0, 994.0, 1100.0, 1230.0, 1395.0, 1612.0, 1908.0};
  constexpr double t3[8] = {902.0, 980.0, 1049.0, 1129.0, 1222.0, 1331.0, 1462.0, 1621.0};
  constexpr double t4[8] = {929.0, 997.0, 1076.0, 1168.0, 1277.0, 1408.0, 1570.0, 1774.0};
  const double x = (rho_cgs < ro[0]) ? ro[0] : ((rho_cgs > ro[7]) ? ro[7] : rho_cgs);
  int i = 0;
  while (i < 6 && x > ro[i + 1]) ++i;
  const double w = (x - ro[i]) / (ro[i + 1] - ro[i]);
  const double a = t2[i] + w * (t2[i + 1] - t2[i]);
  const double b = t3[i] + w * (t3[i + 1] - t3[i]);
  const double c = t4[i] + w * (t4[i + 1] - t4[i]);
  const double ab = (a > b) ? a : b, bc = (b > c) ? b : c;
  return (ab < bc) ? ab : bc;
}

//! Survival of the refractory grains across T5(rho) with Semenov's own compact ramp (opacity.f `cop`, the region-5 ->
//! gas blend the v4 table uses): 1 for T <= T5 - w, 0.5 (1 - sin(pi/2 (T - T5)/w)) inside, and the floor 1e-12 (not
//! 0) above T5 + w. COMPACT on purpose: a tanh about T5 ~ 900-1100 K leaves 1 - s ~ 1e-8 at 10-100 K, which the
//! irreversible rule would turn into a tiny but permanent loss (and a_c ratchet) over the whole cold cloud. The
//! floor keeps a_c > 0 (a_c = 0 with f = 0 would read as 'never seeded' on the freeze_growth path).
DUST_FN double SublSemenov(double T, double rho_cgs, double w) {
  const double x = (T - SemenovT5(rho_cgs)) / w;
  if (x <= -1.0) return 1.0;
  if (x >= 1.0) return 1.0e-12;
  const double s = 0.5 * (1.0 - std::sin(0.5 * dcgs::pi * x));
  return (s > 1.0e-12) ? s : 1.0e-12;
}

DUST_FN double DustModel::subl_factor_at(double T, double rho_cgs) const {
  if (subl_model == 0) return subl_factor(T);
  return SublSemenov(T, rho_cgs, subl_width);
}

//----------------------------------------------------------------------------------------
//! Dust consumer multiplier for opacity / grain cross-section (WS-4 inc2). In the geometric
//! limit kappa ∝ total grain area ∝ (dust mass) / a  => factor = (f_dg/f_ref)*(a_ref/a_c).
//! At the reference state (f_dg=f_ref, a_c=a_ref) this is EXACTLY 1.0 => wiring it into
//! AbsorptionOpacity/RosselandOpacity and the ionization grain bins is bit-identical to the
//! static-grain code when grains are frozen at reference. Guarded against a_c -> 0.
DUST_FN double DustFactor(double f_dg, double a_c, double f_ref, double a_ref) {
  const double a = (a_c > 1.0e-30) ? a_c : 1.0e-30;
  return (f_dg / f_ref) * (a_ref / a);
}

//----------------------------------------------------------------------------------------
// AUDIT F-06 FIX (2026-09-16): the GEOMETRIC law above is only the a >> lambda limit.
//
// kappa per unit DUST MASS for a sphere of radius a is  kappa = 3 Q_abs(x) / (4 rho_gr a),
// with the size parameter x = 2 pi a / lambda. The standard bridging form
//     Q_abs(x) ~= Q_geo * x / (1 + x)
// (Rayleigh Q ∝ x for x << 1, saturating to the geometric value for x >> 1) gives
//     kappa_mass ∝ (1/a) * x/(1+x) ∝ 1 / (1 + x).
//   * x << 1 (Rayleigh): kappa_mass is INDEPENDENT of a. The nu^beta far-IR slope the opacity
//     table carries is a MATERIAL property (optical constants), not a size effect, and it is
//     already baked into the tabulated Planck/Rosseland means -- growing the grains must NOT
//     move it. This is the regime the whole first core lives in: at T <= 500 K less than 4 %
//     of the Rosseland and 1.3 % of the Planck weight sits above the table's own grain break
//     (AUDIT_DUST_2026-09-15 F-06 addendum).
//   * x >> 1 (geometric): kappa_mass -> ∝ 1/a, i.e. EXACTLY the historical a_ref/a_c law.
//
// The break is anchored to the table's own NU_BREAK = 1e14 Hz AT THE REFERENCE RADIUS, so the
// two modules agree by construction: nu_break(a) = NU_BREAK * (a_ref/a)  =>
//     x = (nu/NU_BREAK) * (a/a_ref) = (a/a_ref) * u * T / T_break,   u = h nu /(k T),
//     T_break = h * NU_BREAK / k_B = 4799.24 K.
// Replacing u by its weight mean <u> (exact for the Rosseland channel, see below) leaves a
// one-line factor with all three required limits built in:
//     g(a,T) = (1 + s) / (1 + (a/a_ref) * s),   s = <u> * T / T_break.
//
// ACCURACY, stated rather than implied: <1/(1+x)> is replaced by 1/(1+<x>).
//   * ROSSELAND is EXACT: the Rosseland mean is HARMONIC, so 1/kappa_R ∝ <1 + x>_R
//     = 1 + <u>_R s -- linear, no approximation (verified numerically to 5 digits).
//   * PLANCK is a lower bound (Jensen): the true <1/(1+x)>_P exceeds the Pade by <= 8 %
//     at s ~ 0.3-1 and by < 2 % for s < 0.05, i.e. the Pade slightly OVER-corrects the
//     Planck opacity downward -- conservative with respect to the defect it fixes.
//   * <u> uses the pure Planck / Rosseland weights, not the weight x the table's own psi
//     shape. Including psi shifts <u> by O(1) in opposite directions for the two channels;
//     the LIMITS above are shape-independent, the mid-transition value is a model choice.
//----------------------------------------------------------------------------------------
namespace dopac {
//! h * NU_BREAK / k_B [K] with NU_BREAK = 1e14 Hz (gen_opacity_table.py:82).
constexpr double kTBreakK = 4799.243073366221;
//! <u>_Planck   = Gamma(5) zeta(5) / (Gamma(4) zeta(4))
constexpr double kUPlanck = 3.832229496128944;
//! <u>_Rosseland = Gamma(6) zeta(5) / (Gamma(5) zeta(4))
constexpr double kURoss = 4.790286870161177;
} // namespace dopac

//! Grain-size dependence of the dust mass absorption coefficient, bridged Rayleigh<->geometric.
//! Returns EXACTLY 1.0 (bitwise) when a_c == a_ref, for any T: the numerator and denominator are
//! then the same double, because q*s with q == 1.0 is s exactly in IEEE-754.
DUST_FN double DustSizeFactor(double a_c, double a_ref, double T_K, double u_mean) {
  const double a = (a_c > 1.0e-30) ? a_c : 1.0e-30;
  const double ar = (a_ref > 1.0e-30) ? a_ref : 1.0e-30;
  const double q = a / ar;
  const double s = u_mean * ((T_K > 0.0 ? T_K : 0.0) / dopac::kTBreakK);
  return (1.0 + s) / (1.0 + q * s);
}

//! Dust multiplier for a RADIATION opacity. `bridged == false` reproduces DustFactor()
//! expression-for-expression (bit-identical); `true` applies the F-06 bridged size law with the
//! channel's weight mean u_mean (dopac::kUPlanck or dopac::kURoss).
DUST_FN double DustFactorOpacity(double f_dg, double a_c, double f_ref, double a_ref,
                                 double T_K, double u_mean, bool bridged) {
  const double a = (a_c > 1.0e-30) ? a_c : 1.0e-30;
  if (!bridged) return (f_dg / f_ref) * (a_ref / a);
  return (f_dg / f_ref) * DustSizeFactor(a, a_ref, T_K, u_mean);
}

//----------------------------------------------------------------------------------------
//! AUDIT F-07 FIX, part 1: how much of the gray opacity do GRAINS actually carry?
//!
//! Bell & Lin (1994)'s eight regimes split cleanly into DUST (0 ice grains 2e-4 T^2,
//! 1 ice evaporation 2e16 T^-7, 2 metal grains 0.1 T^0.5, 3 dust sublimation 2e81 rho T^-24)
//! and GAS (4 molecular 1e-8 rho^2/3 T^3, 5 H- 1e-36 rho^1/3 T^10, 6 Kramers 1.5e20 rho T^-2.5,
//! 7 electron scattering 0.348). `BellLinKappaFrom(...,0)` is the full regime-skip-robust walk;
//! `(...,4)` is the opacity the SAME gas would have with no grains at all.
//!
//! NOTE ON DUPLICATION (deliberate, gated): `Radiation::BellLinKappaFixed`
//! (radiation_opacity.hpp) carries its own copy of these coefficients because it works in
//! parthenon::Real and is the opacity MAGNITUDE; this copy is `double` and host-compilable so
//! the standalone dust gate can exercise it. Both are gated against the SAME canonical anchors
//! (kappa = 0.02 at 10 K, 2.0 at 100 K, 0.348 electron scattering) -- test_dust.cpp gate (k)
//! here and test_opacity.cpp gates 5/6 there.
//!
//! WHY NOT `dust_survival(T)` from gen_opacity_table.py: that is a SHAPE blend, not a magnitude
//! split. It returns 0.65 for 170 K < T < 1400 K, but in that window Bell & Lin is regime 2,
//! which is 100 % grains -- the ice removal is already in the MAGNITUDE via regime 1. Using S
//! there would shield 35 % of a purely-dust opacity from the dust multiplier.
DUST_FN double BellLinKappaFrom(double rho, double T, int istart) {
  constexpr double k0[8] = {2.0e-4, 2.0e16, 0.1, 2.0e81, 1.0e-8, 1.0e-36, 1.5e20, 0.348};
  constexpr double aa[8] = {0.0, 0.0, 0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0};
  constexpr double bb[8] = {2.0, -7.0, 0.5, -24.0, 3.0, 10.0, -2.5, 0.0};
  auto Tcross = [&](int p, int q) {
    return std::pow((k0[p] / k0[q]) * std::pow(rho, aa[p] - aa[q]), 1.0 / (bb[q] - bb[p]));
  };
  int i = istart;
  while (i < 7) {
    int j = i + 1;
    while (j < 7 && Tcross(j, j + 1) <= Tcross(i, j)) ++j; // skip closed-window regimes
    if (T < Tcross(i, j)) return k0[i] * std::pow(rho, aa[i]) * std::pow(T, bb[i]);
    i = j;
  }
  return k0[7] * std::pow(rho, aa[7]) * std::pow(T, bb[7]);
}

//! R-F8 (validation_tests/radiation_sn, 2026-10-08): the same walk with the regime-3 (grain evaporation) coefficients
//! of the opacity TABLE's evaporation model. pollack = gen_opacity_table.py --evap pollack (the production table v9):
//! kappa_3 = 0.1 * 2000^24.5 * rho^(0.0195*24.5) * T^-24, i.e. evaporation at T_ev = 2000 rho^0.0195 K instead of
//! Bell & Lin's 420-810 K. With BL94 coefficients the split called the table's still-dusty opacity "gas" in a band
//! above T_ev,pollack (1120-1590 K at 1e-18 g/cm^3), so grain growth / depletion was ignored there.
DUST_FN double BellLinKappaFromEvap(double rho, double T, int istart, bool pollack) {
  double k0[8] = {2.0e-4, 2.0e16, 0.1, 2.0e81, 1.0e-8, 1.0e-36, 1.5e20, 0.348};
  double aa[8] = {0.0, 0.0, 0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0};
  const double bb[8] = {2.0, -7.0, 0.5, -24.0, 3.0, 10.0, -2.5, 0.0};
  if (pollack) {
    k0[3] = 0.1 * std::pow(2000.0, 24.5);
    aa[3] = 0.0195 * 24.5;
  }
  auto Tcross = [&](int p, int q) {
    return std::pow((k0[p] / k0[q]) * std::pow(rho, aa[p] - aa[q]), 1.0 / (bb[q] - bb[p]));
  };
  int i = istart;
  while (i < 7) {
    int j = i + 1;
    while (j < 7 && Tcross(j, j + 1) <= Tcross(i, j)) ++j;
    if (T < Tcross(i, j)) return k0[i] * std::pow(rho, aa[i]) * std::pow(T, bb[i]);
    i = j;
  }
  return k0[7] * std::pow(rho, aa[7]) * std::pow(T, bb[7]);
}
DUST_FN double DustOpacityWeightEvap(double rho_cgs, double T_K, bool pollack) {
  if (!(rho_cgs > 0.0) || !(T_K > 0.0)) return 1.0;
  const double ktot = BellLinKappaFromEvap(rho_cgs, T_K, 0, pollack);
  if (!(ktot > 0.0)) return 0.0;
  const double w = 1.0 - BellLinKappaFromEvap(rho_cgs, T_K, 4, pollack) / ktot;
  return (w < 0.0) ? 0.0 : ((w > 1.0) ? 1.0 : w);
}

//! Fraction of the gray opacity at (rho_cgs, T_K) carried by GRAINS, clamped to [0,1].
//! -> 1 in the cold dusty envelope and first core; -> 0 above the sublimation gap, where the
//! opacity is molecular / H- / Kramers / electron scattering and has nothing to do with grains.
DUST_FN double DustOpacityWeight(double rho_cgs, double T_K) {
  if (!(rho_cgs > 0.0) || !(T_K > 0.0)) return 1.0;
  const double ktot = BellLinKappaFrom(rho_cgs, T_K, 0);
  if (!(ktot > 0.0)) return 0.0;
  const double w = 1.0 - BellLinKappaFrom(rho_cgs, T_K, 4) / ktot;
  return (w < 0.0) ? 0.0 : ((w > 1.0) ? 1.0 : w);
}

//! AUDIT F-07 FIX: restrict a dust multiplier to the DUST part of a mixed dust+gas opacity.
//! kappa_nu is additive, kappa' = kd*kappa_dust + kappa_gas = kappa*(1 + w*(kd-1)) with
//! w = kappa_dust/kappa the dust fraction -- EXACT for the Planck (arithmetic) mean, an
//! approximation for the Rosseland (harmonic) one. w == 1 leaves kd unchanged mathematically
//! but NOT bitwise, which is why the caller must branch on the switch rather than pass w = 1.
DUST_FN double DustGasSplit(double kd, double w) {
  const double ww = (w < 0.0) ? 0.0 : ((w > 1.0) ? 1.0 : w);
  const double r = 1.0 + ww * (kd - 1.0);
  return (r > 0.0) ? r : 0.0;
}

} // namespace Dust

#endif // DUST_DUST_HPP_
