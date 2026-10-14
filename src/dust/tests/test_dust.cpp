// WS-4 dust: host gates for DustModel against analytic solutions.
//
// HISTORY. Written 2026-07-14 for the original model (fixed omega_ref Stokes number, instantaneous
// REVERSIBLE sublimation). On 2026-08-15 both defaults changed (omega_local=true,
// subl_irreversible=true) and this file was not touched, so gates (b) and (c) FAILED from that day
// until the 2026-09-15 dust audit re-ran it (docs/AUDIT_DUST_2026-09-15.md F-03): (b) 99.88 %
// because the analytic C still used omega_ref, (c) because it asserted the legacy round-trip.
// Nothing ever re-ran it because it is not wired into CMake/ctest. Build and run by hand:
//     g++ -O2 -std=c++17 -o /tmp/test_dust src/dust/tests/test_dust.cpp && /tmp/test_dust
//
// GATES (all analytic, each with a deliberate-break falsifier documented in the audit):
//   (a) Brownian-dominated (alpha_turb=0): v_B ~ a^{-3/2} => a^{5/2} = a0^{5/2} + (5/2)K t.
//   (b) Turbulent-dominated (St ~ a): da/dt = C a => a(t) = a0 exp(C t) [3 decades], with the
//       Stokes frequency pinned (omega_local=false) so C is analytic; (b2) the same with
//       omega_local=true, C built from sqrt(4 pi G rho) -- grades the LOCAL-Omega path.
//   (c) Sublimation, DEFAULT (irreversible): hot -> f_dg ~ 0 and a_c shrinks as f^{1/3}; cooling
//       does NOT restore. (c2) LEGACY switch (subl_irreversible=false) restores f_dg_ref exactly.
//   (e) Sublimation FRONT: sweep T across T_subl; f_dg(T) must equal f_ref*subl_factor(T) and
//       the opacity multiplier kdust must be monotone non-increasing and drop below 1e-2 by
//       T_subl + 4*width (subl^{2/3} = 4.8e-3 there); the fixed-grain-number rule
//       kdust = subl^{2/3} must hold to 1e-12.
//   (f) Ratchet: T oscillating around T_subl leaves f_dg at the minimum equilibrium ever seen.
//   (g) Unseeded-vs-destroyed discrimination (F7a): (f<=0, a<=0) -> f_ref; (f<=0, a>0) -> 0.
//   (d) DustFactor: bit-exact 1 at reference; 0.5 for 2x grain; 2 for 2x dust.
// Gate: model integrate_cell matches the analytic solution to < 5 % (a,b,b2); exact elsewhere.
#include "../dust.hpp"
#include <cstdio>
#include <cmath>
#include <initializer_list>
using namespace Dust;

int main() {
  int fail = 0;
  const double rho_gas = 1.0e-13; // g/cm^3 (~ n_H 1e11? just a fixed test state)
  const double T = 20.0;          // K
  const double f_dg = 0.01;

  // ---- (a) Brownian regime: alpha_turb = 0 -> pure a^{-3/2} growth ----
  {
    DustModel d;
    d.alpha_turb = 0.0; // kill turbulent term
    d.growth_cap = 0.05;
    d.nsub_max = 100000;
    // analytic K: da/dt = (f_dg rho/(4 rho_gr)) * V0 * a^{-3/2}, V0 = sqrt(12 kB T/(pi^2 rho_gr))
    const double V0 = std::sqrt(12.0 * dcgs::kB * T / (dcgs::pi * dcgs::pi * d.rho_grain));
    const double K = (f_dg * rho_gas / (4.0 * d.rho_grain)) * V0;
    double a = 1.0e-7, fdg = f_dg;
    const double a0 = a;
    const double tmax = 3.0e12; // s
    d.integrate_cell(a, fdg, rho_gas, T, tmax);
    const double a_ana = std::pow(std::pow(a0, 2.5) + 2.5 * K * tmax, 0.4);
    const double rel = std::fabs(a - a_ana) / a_ana;
    printf("(a) Brownian: a_num=%.4e a_ana=%.4e cm  rel=%.2f%%  %s\n", a, a_ana, 100 * rel,
           rel < 0.05 ? "PASS" : "FAIL");
    if (rel >= 0.05) ++fail;
  }

  // ---- (b) Turbulent regime: large omega + large a so St-driven exp growth dominates ----
  // Two variants: omega pinned to omega_ref (legacy switch), and the default LOCAL Omega.
  for (int variant = 0; variant < 2; ++variant) {
    DustModel d;
    d.alpha_turb = 0.1;
    d.growth_cap = 0.02;
    d.nsub_max = 100000;
    double omega;
    if (variant == 0) {
      d.omega_local = false;
      d.omega_ref = 1.0e-8; // boost v_turb so St term dominates over Brownian across the range
      omega = d.omega_ref;
    } else {
      d.omega_local = true; // Omega = sqrt(4 pi G rho): at 1e-13 g/cm3 that is 2.9e-10 s^-1.
      // Computed HERE, independently of DustModel::omega_dyn, so a wrong Omega in the model is
      // caught (a falsifier that used omega_dyn passed with 4 pi -> 2 pi; audit F-13).
      omega = std::sqrt(4.0 * dcgs::pi * 6.674e-8 * rho_gas);
    }
    const double cs = std::sqrt(dcgs::kB * T / (d.mu_gas * d.m_H));
    const double v_th = std::sqrt(8.0 / dcgs::pi) * cs;
    const double C = (f_dg / 4.0) * std::sqrt(d.alpha_turb) * (cs / v_th) * omega;
    // start large enough that v_turb (~a) beats v_B (~a^-3/2) from the first step
    double a = (variant == 0) ? 1.0e-4 : 1.0e-3, fdg = f_dg;
    const double a0 = a;
    const double tmax = std::log(1000.0) / C; // grow ~3 decades
    d.integrate_cell(a, fdg, rho_gas, T, tmax);
    const double a_ana = a0 * std::exp(C * tmax);
    const double rel = std::fabs(a - a_ana) / a_ana;
    // check v_turb truly dominates at the start (else the exp form is contaminated)
    const double m_grain = (4.0 / 3.0) * dcgs::pi * a0 * a0 * a0 * d.rho_grain;
    const double vB = std::sqrt(8.0 * dcgs::kB * T / (dcgs::pi * 0.5 * m_grain));
    const double St = (d.rho_grain * a0 / (rho_gas * v_th)) * omega;
    const double vt = std::sqrt(d.alpha_turb) * cs * St;
    printf("(b%s) Turbulent %s: a_num=%.4e a_ana=%.4e cm (%.1f decades)  rel=%.2f%%  "
           "vt/vB@start=%.1f  %s\n",
           variant ? "2" : "", variant ? "omega_local" : "omega_ref  ", a, a_ana,
           std::log10(a / a0), 100 * rel, vt / vB, rel < 0.05 ? "PASS" : "FAIL");
    if (rel >= 0.05) ++fail;
  }

  // ---- (c) Sublimation, DEFAULT irreversible: hot destroys, cooling does NOT restore ----
  {
    DustModel d; // defaults: subl_irreversible = true
    double a = 1.0e-5, fdg = 0.01;
    const double s_hot = d.subl_factor(2.0e3);
    d.integrate_cell(a, fdg, rho_gas, 2.0e3, 1.0e3); // T > T_subl; short dt so growth ~0
    const double fdg_hot = fdg, a_hot = a;
    d.integrate_cell(a, fdg, rho_gas, 10.0, 1.0e3); // T < T_subl -> must stay destroyed
    const double fdg_cool = fdg;
    const double a_expect = 1.0e-5 * std::cbrt(s_hot); // fixed grain number
    const bool ok = (fdg_hot < 1.0e-3) && (fdg_cool == fdg_hot) &&
                    (std::fabs(a_hot - a_expect) / a_expect < 1.0e-6); // 1e3 s of growth at 2000 K ~ 2e-8
    printf("(c) Sublimation irreversible: f_dg hot=%.3e cool=%.3e (must stay)  a_c hot=%.4e "
           "(expect %.4e = a0*subl^(1/3))  %s\n",
           fdg_hot, fdg_cool, a_hot, a_expect, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }
  // ---- (c2) LEGACY reversible switch: round-trip restores f_dg_ref exactly, a_c untouched ----
  {
    DustModel d;
    d.subl_irreversible = false;
    double a = 1.0e-5, fdg = 0.01;
    d.integrate_cell(a, fdg, rho_gas, 2.0e3, 1.0e3);
    const double fdg_hot = fdg;
    d.integrate_cell(a, fdg, rho_gas, 10.0, 1.0e3);
    const bool ok = (fdg_hot < 1.0e-3) && (std::fabs(fdg - 0.01) / 0.01 < 1.0e-9) &&
                    (std::fabs(a - 1.0e-5) / 1.0e-5 < 1.0e-6);
    printf("(c2) Sublimation legacy reversible: hot=%.3e cool=%.6f a_c=%.4e  %s\n", fdg_hot, fdg,
           a, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (e) Sublimation FRONT: kdust(T) across T_subl, fixed-grain-number rule ----
  {
    DustModel d;
    bool ok = true;
    double kd_prev = 2.0, kd_at_4w = 1.0;
    for (int n = 0; n <= 40; ++n) {
      const double Tn = d.T_subl - 5.0 * d.subl_width + n * 0.25 * d.subl_width; // -5w .. +5w
      double a = 1.0e-5, fdg = 0.01; // fresh parcel heated monotonically to Tn
      d.apply_sublimation(a, fdg, Tn);
      const double s = d.subl_factor(Tn);
      const double kd = DustFactor(fdg, a, 0.01, 1.0e-5);
      const double kd_expect = std::pow(s, 2.0 / 3.0); // (f/f_ref)*(a_ref/a) = s / s^{1/3}
      if (std::fabs(fdg - 0.01 * s) > 1.0e-15) ok = false;      // f follows the switch
      if (std::fabs(kd - kd_expect) > 1.0e-12 * kd_expect) ok = false;
      if (kd > kd_prev * (1.0 + 1.0e-14)) ok = false;            // monotone non-increasing
      kd_prev = kd;
      if (std::fabs(Tn - (d.T_subl + 4.0 * d.subl_width)) < 1.0e-9) kd_at_4w = kd;
    }
    if (!(kd_at_4w < 1.0e-2)) ok = false; // by T_subl + 4 width the multiplier is < 1 % (4.8e-3)
    const double kd_mid = std::pow(d.subl_factor(d.T_subl), 2.0 / 3.0);
    printf("(e) Sublimation front: kdust = subl^(2/3) along T_subl-5w..+5w; kdust(T_subl)=%.4f "
           "kdust(T_subl+4w)=%.3e monotone=%s  %s\n",
           kd_mid, kd_at_4w, ok ? "yes" : "no", ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (f) Ratchet: oscillating T leaves f_dg at the minimum equilibrium ever visited ----
  {
    DustModel d;
    double a = 1.0e-5, fdg = 0.01;
    const double Tpk = d.T_subl + 1.5 * d.subl_width;
    double fmin = 0.01;
    for (int n = 0; n < 200; ++n) {
      const double Tn = d.T_subl + 1.5 * d.subl_width * std::sin(0.1 * n); // swings +-1.5w
      d.apply_sublimation(a, fdg, Tn);
      fmin = std::fmin(fmin, 0.01 * d.subl_factor(Tn));
    }
    const double f_expect = 0.01 * d.subl_factor(Tpk - 0.1 * d.subl_width); // sin peaks ~1
    const bool ok = (std::fabs(fdg - fmin) <= 1.0e-15) && (fmin < f_expect);
    printf("(f) Ratchet: f_dg after +-1.5w oscillation = %.4e, min equilibrium visited = %.4e  %s\n",
           fdg, fmin, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (g) F7a: unseeded (f<=0, a<=0) -> reference; destroyed (f<=0, a>0) -> stays 0 ----
  {
    DustModel d;
    double a1 = 0.0, f1 = 0.0;
    d.apply_sublimation(a1, f1, 10.0);
    double a2 = 3.0e-6, f2 = -1.0e-30; // destroyed grains whose rho*f undershot in a shock
    d.apply_sublimation(a2, f2, 10.0);
    // F-14: the unseeded recovery must restore the reference PAIR (f_ref, a_ref), so the
    // consumer sees kdust = 1, not a_ref/a_floor = 100 (growth path) or 1e25 (freeze path).
    const double kd1 = DustFactor(f1, a1, d.f_dg_ref, d.a_ref);
    const bool ok = (f1 == d.f_dg_ref) && (a1 == d.a_ref) && (kd1 == 1.0) && (f2 == 0.0) &&
                    (a2 == 3.0e-6);
    printf("(g) Unseeded->(f=%.4f,a=%.1e) kdust=%.3f (=1) ; destroyed-undershoot->%.1e (=0), "
           "a_c kept=%.1e  %s\n",
           f1, a1, kd1, f2, a2, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (d) Consumer factor: bit-identical at reference, correct geometric scaling ----
  {
    const double f_ref = 0.01, a_ref = 1.0e-5;
    const double k1 = DustFactor(f_ref, a_ref, f_ref, a_ref);      // reference -> 1
    const double k2 = DustFactor(f_ref, 2.0e-5, f_ref, a_ref);     // 2x grain -> 0.5
    const double k3 = DustFactor(0.02, a_ref, f_ref, a_ref);       // 2x dust -> 2
    const bool ok = std::fabs(k1 - 1.0) < 1e-12 && std::fabs(k2 - 0.5) < 1e-12 &&
                    std::fabs(k3 - 2.0) < 1e-12;
    printf("(d) DustFactor: ref=%.6f (=1) bigger-grain=%.3f (=0.5) more-dust=%.3f (=2)  %s\n",
           k1, k2, k3, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ======================================================================================
  // AUDIT F-06 / F-07 opacity-coupling gates (2026-09-16). Each has a named falsifier; each
  // falsifier was RUN (see docs/FIX_DUST_OPACITY_2026-09-16.md) and made exactly this gate fail.
  // ======================================================================================

  // ---- (h) REFERENCE STATE IS EXACTLY 1.0, for BOTH laws, at EVERY temperature ----
  // This is what makes the fix bit-identical for frozen-reference grains: the bridged size law
  // must not perturb a run whose grains never grew. Requires EXACT equality, not a tolerance.
  {
    const double f_ref = 0.01, a_ref = 1.0e-5;
    const double Ts[] = {3.0, 30.0, 193.0, 500.0, 1500.0, 2500.0, 5000.0, 1.0e5};
    bool ok = true;
    for (double T : Ts) {
      for (double u : {dopac::kUPlanck, dopac::kURoss}) {
        if (DustSizeFactor(a_ref, a_ref, T, u) != 1.0) ok = false;
        if (DustFactorOpacity(f_ref, a_ref, f_ref, a_ref, T, u, true) != 1.0) ok = false;
        // and the geometric branch must be the byte-for-byte historical expression
        if (DustFactorOpacity(0.004, 3.1e-5, f_ref, a_ref, T, u, false) !=
            DustFactor(0.004, 3.1e-5, f_ref, a_ref))
          ok = false;
      }
    }
    printf("(h) reference kdust == 1.0 EXACTLY for both laws over 3..1e5 K, and the geometric "
           "branch is byte-identical to DustFactor  %s\n", ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (i) RAYLEIGH LIMIT: the size dependence must VANISH as T -> 0 ----
  // Analytic: g = (1+s)/(1+q s) => 1 - g = (q-1)s/(1+q s), so (1-g)/((q-1)s) -> 1 as s -> 0.
  // Gate both the asymptotic slope AND the physically relevant statement: at the first-core
  // temperatures the production run actually occupies, growing the grains 4.13x must change
  // kappa by < 10 % at 30 K -- where the historical geometric law changes it by a factor 4.13.
  {
    const double a_ref = 1.0e-5, q = 4.129654; // the census maximum a_c/a_ref (AUDIT F-05)
    bool ok = true;
    double worst = 0.0;
    for (double T : {0.03, 0.3, 3.0}) { // s_R = 3e-5 .. 3e-3: deep Rayleigh
      const double s = dopac::kURoss * T / dopac::kTBreakK;
      const double g = DustSizeFactor(q * a_ref, a_ref, T, dopac::kURoss);
      const double slope = (1.0 - g) / ((q - 1.0) * s); // -> 1
      worst = std::fmax(worst, std::fabs(slope - 1.0));
      if (std::fabs(slope - 1.0) > 1.05 * q * s) ok = false; // bound: |slope-1| <= q s
    }
    const double g30 = DustSizeFactor(q * a_ref, a_ref, 30.0, dopac::kURoss);
    const double ggeo = a_ref / (q * a_ref);
    // 30 K, grains grown 4.13x: the bridged law moves kappa by 8.3 %, the geometric law by
    // 313 %. Gate both the absolute size and the ratio, so neither can drift unnoticed.
    if (!(g30 > 0.90 && (1.0 - g30) < 0.15 * (1.0 - ggeo))) ok = false;
    printf("(i) Rayleigh: (1-g)/((q-1)s) -> 1 (worst dev %.2e, bound q*s); g(30 K, a=4.13 a_ref)"
           " = %.4f (>0.90, and 1-g = %.1f%% of the geometric law's) vs geometric %.4f  %s\n",
           worst, g30, 100.0 * (1.0 - g30) / (1.0 - ggeo), ggeo, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (j) GEOMETRIC LIMIT: 1/a is recovered where the weight is above the size break ----
  // s >> 1 means the spectral weight sits far above the grain-size break; there the bridged law
  // must reproduce the historical a_ref/a_c to better than 1 %.
  {
    const double a_ref = 1.0e-5;
    bool ok = true;
    double worst = 0.0;
    for (double q : {2.0, 4.129654, 10.0}) {
      for (double T : {1.0e5, 1.0e6}) { // s_R = 100 .. 1000
        const double g = DustSizeFactor(q * a_ref, a_ref, T, dopac::kURoss);
        const double rel = std::fabs(g * q - 1.0); // g -> 1/q
        worst = std::fmax(worst, rel);
        if (rel > 1.0e-2) ok = false;
      }
    }
    printf("(j) geometric: |g*(a_c/a_ref) - 1| <= 1e-2 for s >= 100 (worst %.2e)  %s\n", worst,
           ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (k) HOT-GAS LIMIT (F-07): the dust multiplier must not touch the GAS opacity ----
  // Two halves, and BOTH must hold or the fix is either inert or destructive:
  //   cold: w == 1 below the sublimation gap, so today's dusty first core is UNCHANGED;
  //   hot : w -> 0 above it, so kdust_eff -> 1 and the molecular / H- / e-scattering opacity
  //         survives. The historical code multiplied it by 1.3e-3 at 2000 K and by 0 above
  //         ~4000 K (AUDIT F-07 addendum).
  {
    const double rho = 1.0e-8;               // g/cm^3, the audit addendum's reference density
    const double kd_hot = 1.3e-3 * 0.2422;   // kdust at 2000 K incl. the grown-grain factor
    bool ok = true;
    for (double T : {10.0, 100.0, 193.0, 1000.0}) {
      if (!(DustOpacityWeight(rho, T) > 0.999)) ok = false; // cold: grains carry it all
    }
    const double w2000 = DustOpacityWeight(rho, 2000.0);
    const double w4000 = DustOpacityWeight(rho, 4000.0);
    const double w1e4 = DustOpacityWeight(1.0e-6, 2.0e4); // electron scattering
    if (!(w2000 < 1.0e-2 && w4000 < 1.0e-2 && w1e4 < 1.0e-2)) ok = false;
    const double eff2000 = DustGasSplit(kd_hot, w2000);
    const double eff4000 = DustGasSplit(0.0, w4000); // total tanh underflow: kdust == 0
    if (!(eff2000 > 0.99 && eff4000 > 0.99)) ok = false;
    // and the canonical Bell & Lin anchors, shared with test_opacity.cpp gates 5/6, so the
    // second copy of the coefficients here cannot drift from the radiation one.
    const double anchors_ok = std::fabs(BellLinKappaFrom(1.0e-14, 10.0, 0) - 0.02) < 1.0e-8 &&
                              std::fabs(BellLinKappaFrom(1.0e-14, 100.0, 0) - 2.0) < 1.0e-6 &&
                              std::fabs(BellLinKappaFrom(1.0e-6, 1.0e7, 0) - 0.348) < 1.0e-9;
    if (!anchors_ok) ok = false;
    printf("(k) gas split: w(1e-8,10..1000 K) > 0.999 ; w(2000 K)=%.2e w(4000 K)=%.2e "
           "w(e-scat)=%.2e ; kd_eff(2000 K)=%.5f kd_eff(4000 K, kdust=0)=%.5f ; B&L anchors %s "
           " %s\n", w2000, w4000, w1e4, eff2000, eff4000, anchors_ok ? "ok" : "BAD",
           ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (l) MONOTONICITY + the census delta the science measurement is built on ----
  // g must fall monotonically with both a_c and T (more weight above the break => more
  // geometric), and at the production census state the correction must be the factor the
  // report quotes -- so a silent coefficient change shows up here as a number, not a mood.
  {
    const double a_ref = 1.0e-5;
    bool ok = true;
    double prev = 2.0;
    for (int n = 1; n <= 40; ++n) { // a_c = 1..40 a_ref at fixed T
      const double g = DustSizeFactor(n * a_ref, a_ref, 193.0, dopac::kURoss);
      if (!(g < prev)) ok = false;
      prev = g;
    }
    prev = 2.0;
    for (int n = 1; n <= 40; ++n) { // T = 25..1000 K at fixed a_c
      const double g = DustSizeFactor(4.13 * a_ref, a_ref, 25.0 * n, dopac::kURoss);
      if (!(g < prev)) ok = false;
      prev = g;
    }
    const double gR = DustSizeFactor(4.129654 * a_ref, a_ref, 193.0, dopac::kURoss);
    const double gP = DustSizeFactor(4.129654 * a_ref, a_ref, 193.0, dopac::kUPlanck);
    const double kd_old = DustFactor(0.01, 4.129654e-5, 0.01, 1.0e-5);
    if (!(std::fabs(gR - 0.66424) < 1.0e-4 && std::fabs(gP - 0.70519) < 1.0e-4)) ok = false;
    printf("(l) monotone in a_c and T ; census cell (193 K, a_c=4.1297 a_ref): kdust_R %.4f "
           "kdust_P %.4f vs historical %.4f (kappa_R up %.2fx)  %s\n", gR, gP, kd_old,
           gR / kd_old, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (m) growth_prefactor (validation_tests/dust D2, 2026-10-12) ----
  // Default 0.25 must equal the historical expression (f rho/rho_gr) v_rel / 4.0 BIT FOR BIT over a
  // grid of states; C = 0.5 must reproduce the Brownian closed form with K -> 2K (falsifier: the
  // closed form with the old K misses by 2^0.4 - 1 = 32 %).
  {
    DustModel d;
    bool bit = true;
    for (double r : {1.0e-19, 1.0e-15, 1.0e-13, 1.0e-10, 1.0e-7})
      for (double Tk : {3.0, 10.0, 150.0, 1499.0})
        for (double a : {1.0e-7, 1.0e-5, 3.3e-5, 1.0e-3})
          for (double fd : {1.0e-9, 0.004, 0.01}) {
            const double hist = (fd * r / d.rho_grain) * d.v_rel(r, Tk, a) / 4.0;
            if (d.dadt(r, Tk, a, fd) != hist) bit = false;
          }
    DustModel d2;
    d2.alpha_turb = 0.0;
    d2.growth_cap = 0.05;
    d2.nsub_max = 100000;
    d2.growth_pref = 0.5;
    const double V0 = std::sqrt(12.0 * dcgs::kB * T / (dcgs::pi * dcgs::pi * d2.rho_grain));
    const double K2 = (f_dg * rho_gas * 0.5 / d2.rho_grain) * V0;
    double a = 1.0e-7, fdg = f_dg;
    const double tmax = 3.0e12;
    d2.integrate_cell(a, fdg, rho_gas, T, tmax);
    const double a_ana = std::pow(std::pow(1.0e-7, 2.5) + 2.5 * K2 * tmax, 0.4);
    const double rel = std::fabs(a - a_ana) / a_ana;
    const bool ok = bit && rel < 0.05;
    printf("(m) growth_prefactor: default 0.25 bitwise == historical /4.0 over 300 states: %s ; "
           "C = 0.5 Brownian a_num=%.4e a_ana=%.4e rel=%.2f%%  %s\n",
           bit ? "yes" : "NO", a, a_ana, 100 * rel, ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (n) DD5 subl_model (validation_tests/dust, 2026-10-12) ----
  // SemenovT5 must reproduce opacity_semenov.py evap_temperatures()[4] (the function the v4/v12 opacity table was
  // built with) at 14 densities incl. both clamped ends; subl_model = fixed must give subl_factor(T) BIT FOR BIT
  // through the new rho-aware path; semenov must centre the tanh on T5(rho).
  {
    const double ref[14][2] = {{1e-19, 902.0}, {1e-18, 902.0}, {3e-17, 924.8484848484849}, {1e-15, 986.2727272727273},
                               {1e-13, 1056.2727272727273}, {1e-12, 1129.0}, {5e-12, 1132.7575757575758},
                               {1e-11, 1137.4545454545455}, {1e-10, 1230.0}, {1e-09, 1245.0}, {1e-08, 1395.0},
                               {1e-07, 1414.7272727272727}, {1e-05, 1588.5454545454545}, {0.001, 1774.0}};
    double worst = 0.0;
    for (const auto &r : ref) worst = std::fmax(worst, std::fabs(SemenovT5(r[0]) - r[1]));
    DustModel d;
    bool bit = true;
    for (double Tk : {10.0, 900.0, 1400.0, 1500.0, 1650.0, 3000.0})
      for (double r : {1e-15, 1e-9})
        if (d.subl_factor_at(Tk, r) != d.subl_factor(Tk)) bit = false;
    double a1 = 1e-5, f1 = 0.01, a2 = 1e-5, f2 = 0.01;
    d.apply_sublimation(a1, f1, 1550.0);
    d.apply_sublimation(a2, f2, 1550.0, 1e-9);
    if (!(a1 == a2 && f1 == f2)) bit = false;
    DustModel ds;
    ds.subl_model = 1;
    const double s_mid = ds.subl_factor_at(SemenovT5(1e-9), 1e-9);   // exactly 0.5 at the centre
    const double s_old = ds.subl_factor_at(1300.0, 1e-9);            // 1300 K at 1e-9: T5 = 1245 -> x = 0.55
    const double s_ana = 0.5 * (1.0 - std::sin(0.5 * dcgs::pi * 0.55));  // = 0.1198
    // compact ramp: EXACTLY 1 below T5 - w (no tail at cloud temperatures), the 1e-12 floor above T5 + w
    const bool compact = ds.subl_factor_at(1145.0, 1e-9) == 1.0 && ds.subl_factor_at(10.0, 1e-18) == 1.0 &&
                         ds.subl_factor_at(1345.0, 1e-9) == 1.0e-12;
    const bool ok = worst < 1e-9 && bit && s_mid == 0.5 && std::fabs(s_old - s_ana) < 1e-12 && compact;
    printf("(n) subl_model: SemenovT5 vs opacity_semenov.py max |dT| = %.1e K (14 densities) ; fixed path bitwise: %s"
           " ; semenov s(T5)=%.3f s(1300 K, 1e-9)=%.4f (ana %.4f) compact %s  %s\n", worst, bit ? "yes" : "NO", s_mid, s_old, s_ana, compact ? "yes" : "NO",
           ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  // ---- (o) DD2 subl_tau (validation_tests/dust, 2026-10-13) ----
  // tau = 0: the dt-aware call is bitwise the instantaneous one. tau > 0 at constant T: f(t) = f_eq + (f0 - f_eq)
  // exp(-t/tau) (one step = analytic to 1e-14; 50 steps = one step to 1e-12: the exponential composes);
  // a_c = a0 (f/f0)^(1/3); irreversible: cooling afterwards leaves f unchanged.
  {
    DustModel d0;
    double a1 = 1e-5, f1 = 0.01, a2 = 1e-5, f2 = 0.01;
    d0.apply_sublimation(a1, f1, 1520.0, 1e-9);
    d0.apply_sublimation(a2, f2, 1520.0, 1e-9, 3.0e5);
    bool ok = (a1 == a2 && f1 == f2);
    DustModel d;
    d.subl_tau = 1.0e4;
    const double T = 1560.0, feq = 0.01 * d.subl_factor(T), t = 2.0e4;
    double a = 1e-5, f = 0.01;
    d.apply_sublimation(a, f, T, 1e-9, t);
    const double f_ana = feq + (0.01 - feq) * std::exp(-t / d.subl_tau);
    const double e1 = std::fabs(f / f_ana - 1), ea = std::fabs(a / (1e-5 * std::cbrt(f / 0.01)) - 1);
    double an = 1e-5, fn = 0.01;
    for (int n = 0; n < 50; ++n) d.apply_sublimation(an, fn, T, 1e-9, t / 50);
    const double e2 = std::fabs(fn / f - 1);
    const double f_hot = fn;
    d.apply_sublimation(an, fn, 30.0, 1e-9, 1.0e9);   // long cold step: must not restore
    ok = ok && e1 < 1e-14 && ea < 1e-14 && e2 < 1e-12 && fn == f_hot;
    printf("(o) subl_tau: tau=0 bitwise %s ; one step vs analytic %.1e, a_c rule %.1e ; 50 steps vs 1 step %.1e ; "
           "irreversible %s  %s\n", (a1 == a2 && f1 == f2) ? "yes" : "NO", e1, ea, e2, fn == f_hot ? "yes" : "NO",
           ok ? "PASS" : "FAIL");
    if (!ok) ++fail;
  }

  printf("\n%s (%d failures)\n", fail == 0 ? "ALL GATES PASS" : "GATE FAILURES", fail);
  return fail == 0 ? 0 : 1;
}
