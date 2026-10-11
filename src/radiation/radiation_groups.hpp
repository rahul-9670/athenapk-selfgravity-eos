//========================================================================================
// AthenaPK - M1 radiation: multigroup frequency structure (Phase 4 scaffold, increment 1).
//
// Foundation for multigroup RHD. This increment adds ONLY the frequency-group structure and
// the group-integrated Planck function (no transport changes yet). It is designed so that
// N_GROUP = 1 reduces EXACTLY to the current gray behavior (a single group spanning [0, inf)
// has Planck fraction 1, so B_group = a_R T^4), keeping production bit-identical until the
// multigroup path is wired into the moments update + opacity in later increments.
//
// Design (later increments): per-group moments (Er_g, Fr_g), per-group M1 closure (reuses
// radiation_closure.hpp unchanged -- the closure is frequency-independent), per-group Planck/
// Rosseland opacity means from a monochromatic dataset, and IMEX coupling summing the group
// emission/absorption. The M1 closure and signal speeds are per-group; only the source term
// couples groups (through the matter temperature).
//========================================================================================
#ifndef RADIATION_RADIATION_GROUPS_HPP_
#define RADIATION_RADIATION_GROUPS_HPP_

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <iostream> // std::cout -- this header prints table diagnostics itself; it was
                    // relying on a transitive include from the CMake build, which the host
                    // test suites (tests/run_host_tests.sh) do not have.
#include <vector>

#include <Kokkos_Core.hpp>

#include <basic_types.hpp>      // parthenon::Real
#include <parthenon_arrays.hpp> // parthenon::ParArray2D (device tables)

#include "opacity_table_format.hpp" // shared table header parser (audit N13)

namespace Radiation {

using parthenon::ParArray2D;
using parthenon::ParArray3D;   // R-F11: format-v4 rho-dependent multipliers
using parthenon::Real;

// Max compile-time group count (POD-friendly; N_GROUP=1 default keeps the gray path).
// R5 (validation/radiation_sn, 2026-10-08): a test build may raise the cap with -DATHENAPK_MAX_GROUP=<n> to get a
// converged group-count REFERENCE (18 groups); production builds do not define it and keep 8.
#ifndef ATHENAPK_MAX_GROUP
#define ATHENAPK_MAX_GROUP 8
#endif
constexpr int MAX_GROUP = ATHENAPK_MAX_GROUP;

//----------------------------------------------------------------------------------------
// NOTE (audit N9, 2026-08-05): two device-callable band-moment helpers, PlanckMoment() and
// RossMoment(), used to live here. They were DEAD -- nothing in src/ or the tests ever called
// them. They were superseded by the `phimean` quadrature inside GroupMultsAtT() below, which
// computes the same band means but in the form actually needed (a ratio of a phi-weighted to
// an unweighted moment) and on the host, where the table is built. Removed rather than kept
// "for later": a device-callable helper with a subtle e^{-a}-factored-out return convention
// is exactly the kind of thing a future caller uses without reading the convention. The
// convention itself survives, documented at its one real use site in `phimean`.
//----------------------------------------------------------------------------------------
//! Cumulative Planck energy fraction F(x) = (15/pi^4) * Int_0^x u^3/(e^u - 1) du, x = h*nu/kT.
//! F(0)=0, F(inf)=1; the fraction in a group [x1,x2] is F(x2)-F(x1). The tail integral
//! Int_x^inf u^3/(e^u-1) du = sum_{n>=1} e^{-n x}( x^3/n + 3 x^2/n^2 + 6 x/n^3 + 6/n^4 )
//! (Clark 1965), and Int_0^inf = pi^4/15, so F(x) = 1 - (15/pi^4)*tail. The series converges
//! fast; at x=0 the tail sums to pi^4/15 (F=0), at large x it -> 0 (F=1).
//! TWO BRANCHES (2026-08-18). The single exponential-tail branch that used to be here was both
//! the hottest function in the code and the least accurate part of it.
//!
//! SPEED. `nsys` on three production cycles (job 2536309) put
//! `Radiation::MatterCouplingMultigroup` at 8.025 s of 23.893 s of total GPU kernel time --
//! 33.6 %, in THREE launches of 2.675 s each. That kernel is a safeguarded Newton solve whose
//! residual calls PlanckFraction for every group, and PlanckFraction calls this function twice.
//! The tail series only terminates early via `n*x > 60`, so for x < 2 it ran the FULL 30 terms
//! and 30 `exp()` calls -- roughly 2e10 `exp()` per hydro step. x = h*nu/(k*T), so x < 2 is the
//! COLD gas, i.e. most of a collapsing cloud.
//!
//! ACCURACY. Truncating Clark's tail series at n=30 leaves sum_{n>30} 6/n^4 ~ 7.4e-5 unaccounted,
//! so F(x) was too large by a FIXED ~1.1e-5 as x -> 0. Measured against a quadrature of the
//! integral: at x = 1e-4 it returned 1.085e-05 where the answer is 5.1e-14 -- eight orders out.
//! It largely cancels in PlanckFraction = F(xhi) - F(xlo) when both edges are small, which is
//! why it never showed up as an obvious wrong answer, but it put a ~1e-5 floor under every
//! group fraction.
//!
//! Below x = 2 use the Debye/Bernoulli expansion of the integral itself,
//!   Int_0^x u^3/(e^u-1) du = sum_n B_n x^(n+3) / (n! (n+3)),  B_1 = -1/2,
//! truncated at n = 18 (powers up to x^21). It converges for |x| < 2*pi and needs NO exp() at
//! all. Above x = 2 the tail series is already accurate to ~1e-16 and now terminates in 2-28
//! terms rather than always 30.
//!
//! VERIFIED against scipy quadrature over x in [1e-7, 60]:
//!   max abs error   NEW 1.117e-11   OLD 1.085e-05     (9.7e5x better)
//!   seam at x = 2   the two branches agree to 1.1e-11
//! The Horner rearrangement below was separately checked against the direct power sum
//! (max relative difference 8.0e-16), so the restructuring itself introduces nothing.
//!
//! THIS IS RESULT-CHANGING at the level of the OLD error (~1e-5 in a group fraction, less after
//! the cancellation in the difference). It is a bug fix, not a refactor.
KOKKOS_INLINE_FUNCTION Real PlanckCumFraction(const Real x) {
  if (x <= 0.0) return 0.0;
  constexpr Real k15_pi4 = 15.0 / (M_PI * M_PI * M_PI * M_PI);
  if (x <= 2.0) {
    // Debye/Bernoulli series, exp()-free. Coefficients are B_n/(n!(n+3)) for the nonzero B_n
    // up to n = 18, i.e. powers x^3, x^4, x^5, then odd powers x^7..x^21.
    constexpr Real a0 = 3.33333333333333315e-01;   // x^3
    constexpr Real a1 = -1.25000000000000000e-01;  // x^4
    constexpr Real a2 = 1.66666666666666664e-02;   // x^5
    constexpr Real a3 = -1.98412698412698413e-04;  // x^7
    constexpr Real a4 = 3.67430922986478553e-06;   // x^9
    constexpr Real a5 = -7.51563251563251607e-08;  // x^11
    constexpr Real a6 = 1.60590438368216149e-09;   // x^13
    constexpr Real a7 = -3.52279342579166215e-11;  // x^15
    constexpr Real a8 = 7.87208031216745774e-13;   // x^17
    constexpr Real a9 = -1.78404226122241216e-14;  // x^19
    constexpr Real a10 = 4.08860097917992578e-16;  // x^21
    const Real y = x * x;
    const Real p =
        a2 + y * (a3 + y * (a4 + y * (a5 + y * (a6 + y * (a7 + y * (a8 + y * (a9 + y * a10)))))));
    const Real f = k15_pi4 * (x * y) * (a0 + a1 * x + y * p);
    return (f < 0.0) ? 0.0 : ((f > 1.0) ? 1.0 : f);
  }
  // x > 2: Clark's tail series. Int_x^inf u^3/(e^u-1) du =
  //   sum_{n>=1} e^{-n x}( x^3/n + 3 x^2/n^2 + 6 x/n^3 + 6/n^4 ), and Int_0^inf = pi^4/15.
  //
  // ONE exp() INSTEAD OF ONE PER TERM (2026-08-19). e^{-n x} = (e^{-x})^n, so the whole series
  // needs a single transcendental and a running multiply. This is the branch PRODUCTION spends
  // its time in: x = h*nu/(k*T), the group edge is 1e12 Hz and the envelope sits near 10 K, so
  // x ~ 4.8 and the `n*x > 60` break lands at n = 13. Thirteen exp() per call, two calls per
  // PlanckFraction, three groups, two residuals per Newton iteration = 156 exp() per iteration
  // per cell.
  //
  // WHY THIS WAS MISSED UNTIL NOW. The Debye branch added earlier tonight covers x <= 2 -- i.e.
  // T > 24 K -- and its A/B ran on a 10 K state, so the new branch was never executed and the
  // kernel time did not move. I read that as "the Planck series is not the bottleneck". It does
  // not follow: the test was vacuous for performance exactly as it was for correctness.
  //
  // VERIFIED BIT-IDENTICAL to the per-term form over x in [2,60] (max |difference| 0.000e+00 at
  // 400 sample points; both agree with quadrature to 3.331e-15). The terms decay fast enough
  // that the ~n*eps accumulated by the running product never reaches the printed digits.
  const Real x2 = x * x, x3 = x2 * x;
  const Real E = std::exp(-x); // the only transcendental in this branch
  Real en = E;                 // e^{-n x}, advanced by one multiply per term
  Real tail = 0.0;
  for (int n = 1; n <= 64; ++n) {
    const Real nn = static_cast<Real>(n);
    tail += en * (x3 / nn + 3.0 * x2 / (nn * nn) + 6.0 * x / (nn * nn * nn) +
                  6.0 / (nn * nn * nn * nn));
    if (nn * x > 60.0) break; // remaining terms are below double precision
    en *= E;
  }
  Real f = 1.0 - k15_pi4 * tail;
  return (f < 0.0) ? 0.0 : ((f > 1.0) ? 1.0 : f);
}

//----------------------------------------------------------------------------------------
//! Frequency-group structure. POD captured by value into device kernels. Group boundaries are
//! stored as x = h*nu/(k*T_ref) reference values are NOT used -- instead the group boundaries
//! are physical frequencies [Hz]; the Planck fraction below uses the LOCAL matter T. Edges are
//! nu[0]=0 .. nu[n_group]=inf(large). N_GROUP=1 with nu=[0,inf) => gray.
struct RadGroups {
  int n_group = 1;
  // group frequency edges [Hz], size n_group+1; nu_edge[0]=0, nu_edge[n_group]=+inf (HUGE).
  Real nu_edge[MAX_GROUP + 1] = {0.0};
  Real h_over_k = 4.799243e-11; // h/k_B [s*K]  (Planck const / Boltzmann)

  //! Equilibrium Planck energy fraction in group g at matter temperature T [K].
  //! sum_g PlanckFraction(g,T) == 1 exactly (edges span [0,inf)).
  KOKKOS_INLINE_FUNCTION Real PlanckFraction(const int g, const Real T) const {
    if (n_group == 1) return 1.0; // gray: the single group holds all the energy
    const Real xlo = h_over_k * nu_edge[g] / T;
    const Real xhi = (g == n_group - 1) ? 1.0e30 : h_over_k * nu_edge[g + 1] / T;
    return PlanckCumFraction(xhi) - PlanckCumFraction(xlo);
  }
};

//----------------------------------------------------------------------------------------
//! Semenov-class monochromatic dust+gas opacity SHAPE phi(nu,T): the frequency dependence of
//! kappa_nu(rho,T) = kappa_gray(rho,T) * phi(nu,T), where kappa_gray is the existing gray
//! Bell&Lin/dust law (which already carries the rho,T magnitude AND the sublimation drop). phi
//! only redistributes that magnitude across frequency, normalized (by the ratio anchoring in the
//! table build) so the full-spectrum Planck/Rosseland means of kappa_nu equal kappa_gray exactly.
//!
//! Physics captured (protostellar dust, Semenov et al. 2003 / Draine grain emissivity):
//!  - Dust emissivity rises as nu^beta in the far-IR (Q_abs ~ nu^beta, beta~1.5-2 for
//!    silicate/carbon+ice aggregates) and FLATTENS to the geometric (gray) limit above a
//!    grain-size break nu_break (lambda ~ 2*pi*a ~ few micron => nu_break ~ 1e14 Hz).
//!  - Above dust sublimation (~1500 K) grains are destroyed: the opacity becomes gas-dominated
//!    (molecular / H- / Kramers continuum), taken here as ~gray (phi -> 1). survival(T) ramps
//!    the dust frequency structure off across [T_sub_lo, T_sub_hi].
//! beta=0 => phi==1 everywhere => gray-per-group (equivalence gate exact).
struct DustOpacityModel {
  Real beta = 0.0;             // far-IR dust emissivity index (0 => gray)
  Real nu_break_hz = 1.0e14;   // grain-size turnover: Q_abs flattens to gray above this
  Real T_sub_lo = 1400.0;      // dust sublimation ramp start [K]
  Real T_sub_hi = 1600.0;      // dust sublimation ramp end   [K] (phi->1, gray gas)
  Real h_over_k = 4.799243e-11;

  //! Dust survival fraction (1 cold, 0 above sublimation).
  Real survival(const Real Tk) const {
    if (Tk <= T_sub_lo) return 1.0;
    if (Tk >= T_sub_hi) return 0.0;
    return (T_sub_hi - Tk) / (T_sub_hi - T_sub_lo);
  }
  //! Opacity frequency shape at x = h*nu/(k*T), matter temperature Tk. Host-side (table build).
  Real phi(const Real x, const Real Tk) const {
    if (beta == 0.0) return 1.0;
    const Real S = survival(Tk);
    const Real xb = h_over_k * nu_break_hz / Tk;         // break in x-units at this T
    Real psi = std::pow(x / (xb + 1.0e-300), beta);      // nu^beta far-IR rise
    if (psi > 1.0) psi = 1.0;                            // geometric (gray) limit above break
    return S * psi + (1.0 - S);                          // dust shape blended toward gray gas
  }
};

//----------------------------------------------------------------------------------------
//! Ratio-anchored per-group Planck & Rosseland opacity multipliers at matter temperature Tk:
//!   mult_P,g = <phi>_{Planck,band g} / <phi>_{Planck,full}       (=> sum_g mult_P,g*frac_g = 1)
//!   mult_R,g = <1/phi>_{Ross,full}  / <1/phi>_{Ross,band g}      (harmonic; mult_R,full = 1)
//! <.>_Planck weights by B_nu (x^3/(e^x-1)); <.>_Ross by dB/dT (x^4 e^x/(e^x-1)^2). The e^{-a}
//! factor is pulled out of each band ratio (integrand carries e^{a-x}) so deep-Wien bands stay
//! finite. Host-side, called once per T-grid point in the table build.
inline void GroupMultsAtT(const DustOpacityModel &m, const RadGroups &g, const Real Tk,
                          Real *mP, Real *mR) {
  const int NQ = 240;
  const Real X = 80.0;
  // <phi^{inv?}>_weight over [a,b]; weight: 0=Planck (x^3), 1=Rosseland (x^4 * e^x/(e^x-1))
  auto phimean = [&](Real a, Real b, bool ross, bool inv) -> Real {
    if (a < 1.0e-8) a = 1.0e-8;
    if (b <= a) return 1.0;
    const int n = NQ;
    const Real h = (b - a) / n;
    Real num = 0.0, den = 0.0;
    for (int k = 0; k <= n; ++k) {
      const Real x = a + k * h;
      const Real w = (k == 0 || k == n) ? 1.0 : ((k % 2) ? 4.0 : 2.0);
      const Real ex = std::exp(-x);
      Real base = std::exp(a - x); // e^{a-x}, common factor pulled out (cancels in the ratio)
      base *= ross ? (x * x * x * x / ((1.0 - ex) * (1.0 - ex))) : (x * x * x / (1.0 - ex));
      Real ph = m.phi(x, Tk);
      if (inv) ph = 1.0 / ph;
      num += w * ph * base;
      den += w * base;
    }
    return num / (den + 1.0e-300);
  };
  const Real phiP_full = phimean(1.0e-8, X, false, false);
  const Real invphiR_full = phimean(1.0e-8, X, true, true);
  for (int gi = 0; gi < g.n_group; ++gi) {
    Real a = g.h_over_k * g.nu_edge[gi] / Tk;
    Real b = (gi == g.n_group - 1) ? (a + 60.0) : g.h_over_k * g.nu_edge[gi + 1] / Tk;
    if (b > a + 60.0) b = a + 60.0;
    mP[gi] = phimean(a, b, false, false) / (phiP_full + 1.0e-300);
    mR[gi] = invphiR_full / (phimean(a, b, true, true) + 1.0e-300);
  }
}

//----------------------------------------------------------------------------------------
//! Tabulated per-group opacity multipliers vs matter temperature. The band means are a
//! per-T Simpson quadrature of the Semenov-class shape phi -- precomputed here on a log10(T[K])
//! grid once at init and O(1) linear-interpolated on device (the by-value-Views-into-kernel
//! pattern of EosTable). active_=false (beta=0 or gray) => mult=1, exact gray-per-group.
struct GroupOpacityTable {
  ParArray2D<Real> wP_, wR_; // [n_group][nT] Planck- and Rosseland-mean multipliers
  Real lT0_ = 0.0, dlT_ = 1.0; // log10(T[K]) grid origin + spacing
  int ng_ = 1, nT_ = 0;
  bool active_ = false;
  //! R-F5 (radiation/opacity_interp = log): interpolate log(multiplier) in log T (the multipliers vary by
  //! orders of magnitude across a cell near the 140 K ice ramp and in the Wien tail). Default false = legacy.
  bool log_ = false;
  ParArray2D<Real> lwP_, lwR_;

  // OUT-OF-RANGE CLAMP TALLY -- see the long note in radiation_opacity.hpp. Layout: [0] = T
  // below the table, [1] = T above. Allocated by the builder; a default-constructed table
  // keeps a null View and the kernels skip the atomic entirely.
  Kokkos::View<unsigned long long *> clamp_ctr_;
  // 2026-08-21: DOUBLED into a state bank and a probe bank. [off+0] = T below, [off+1] = T
  // above, off = 0 for a real cell state and 2 for a root-finder TRIAL value. MGResidual probes
  // T across a bracket every Newton step, so its clamps fired benignly every cycle and masked
  // any genuine one -- the same reason the note above gives for keying the count off f.
  static constexpr int kNClampState = 2;
  static constexpr int kNClampCtr = 4;

  KOKKOS_INLINE_FUNCTION
  Real Lookup(const ParArray2D<Real> &W, const int g, const Real Tk,
              const bool probe = false) const {
    if (!active_) return 1.0;
    const int ctr_off = probe ? kNClampState : 0;
    const bool count = (clamp_ctr_.data() != nullptr);
    Real f = (std::log10(Tk) - lT0_) / dlT_;
    int i = static_cast<int>(f);
    // AUDIT 2026-08-05 (N10): clamp HIGH first, then LOW. The old order (low then high) left
    // i = -1 whenever nT_ < 2, because the second clamp `i > nT_-2` then pushed i BELOW the
    // zero it had just been raised to -- an out-of-bounds device read of W(g,-1). nT_ < 2 is
    // rejected at construction now (see BuildGroupOpacityTable*), so this is belt-and-braces,
    // but the clamp order was wrong on its own terms and cost nothing to fix.
    // Counting is keyed off the ORIGINAL f, not the clamped index: with the N10 clamp order
    // (high first, then low) a single lookup can trip both branches when nT_ < 2, and keying
    // off the index would then tally a low clamp for a high-side temperature.
    //
    // THE THRESHOLD IS nT_-1, NOT nT_-2, and the difference is not cosmetic. The interpolation
    // cells are [0,1] .. [nT_-2, nT_-1], so f = nT_-1 is the table's exact TOP NODE -- in
    // range, fully interpolated, not clamped. Keying the count off `f > nT_-2` flagged the
    // whole last cell as out-of-range.
    // Caught by the fix it was meant to verify: capping the rtsafe bracket at the table's true
    // upper edge made the solver probe exactly AT that edge, and the high-T count went UP
    // (41232 -> 59343) where it should have gone to zero. A counter that reports more clamping
    // after the clamping is removed is measuring itself, not the code.
    if (count && f > static_cast<Real>(nT_ - 1))
      Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 1), 1ull);
    if (count && f < 0.0) Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 0), 1ull);
    if (i > nT_ - 2) i = nT_ - 2;
    if (i < 0) i = 0;
    Real t = f - i;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return (1.0 - t) * W(g, i) + t * W(g, i + 1);
  }

  //! Host-side: drain the tallies and zero them for the next interval. False if never built.
  bool ReadAndResetClampCounts(unsigned long long out[kNClampCtr]) const {
    if (clamp_ctr_.data() == nullptr) return false;
    auto h = Kokkos::create_mirror_view(clamp_ctr_);
    Kokkos::deep_copy(h, clamp_ctr_);
    for (int q = 0; q < kNClampCtr; ++q) out[q] = h(q);
    Kokkos::deep_copy(clamp_ctr_, 0ull);
    return true;
  }
  //! R-F11 (format v4, validation/radiation_sn 2026-10-09): rho-DEPENDENT multipliers W3(g, i_rho, j_T)
  //! on the gray table's own (log10 rho_cgs, log10 T_K) grid. Semenov's evaporation temperatures and the
  //! dust -> gas blend depend on rho, so a T-only multiplier is wrong in every transition band. Only a v4 file
  //! sets rho_dep_; PlanckMultRT / RossMultRT then use this path, otherwise they ARE PlanckMult / RossMult.
  bool rho_dep_ = false;
  int nr_ = 0;
  Real lr0_ = 0.0, dlr_ = 1.0, rho_unit_ = 1.0;
  ParArray3D<Real> wP3_, wR3_, lwP3_, lwR3_;
  KOKKOS_INLINE_FUNCTION
  Real Lookup3(const ParArray3D<Real> &W, const int g, const Real rho_code, const Real Tk,
               const bool probe = false) const {
    const int ctr_off = probe ? kNClampState : 0;
    const bool count = (clamp_ctr_.data() != nullptr);
    Real f = (std::log10(Tk) - lT0_) / dlT_;
    int j = static_cast<int>(f);
    if (count && f > static_cast<Real>(nT_ - 1))
      Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 1), 1ull);
    if (count && f < 0.0) Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 0), 1ull);
    if (j > nT_ - 2) j = nT_ - 2;
    if (j < 0) j = 0;
    Real t = f - j;
    t = (t < 0.0) ? 0.0 : ((t > 1.0) ? 1.0 : t);
    Real fr = (std::log10(rho_code * rho_unit_) - lr0_) / dlr_;   // rho clamps: tallied by the gray table
    int i = static_cast<int>(fr);
    if (i > nr_ - 2) i = nr_ - 2;
    if (i < 0) i = 0;
    Real u = fr - i;
    u = (u < 0.0) ? 0.0 : ((u > 1.0) ? 1.0 : u);
    return (1.0 - u) * ((1.0 - t) * W(g, i, j) + t * W(g, i, j + 1)) +
           u * ((1.0 - t) * W(g, i + 1, j) + t * W(g, i + 1, j + 1));
  }
  KOKKOS_INLINE_FUNCTION Real PlanckMultRT(const int g, const Real rho_code, const Real Tk,
                                           const bool probe = false) const {
    if (!rho_dep_) return PlanckMult(g, Tk, probe);
    if (log_) return std::exp(Lookup3(lwP3_, g, rho_code, Tk, probe));
    return Lookup3(wP3_, g, rho_code, Tk, probe);
  }
  KOKKOS_INLINE_FUNCTION Real RossMultRT(const int g, const Real rho_code, const Real Tk) const {
    if (!rho_dep_) return RossMult(g, Tk);
    if (log_) return std::exp(Lookup3(lwR3_, g, rho_code, Tk));
    return Lookup3(wR3_, g, rho_code, Tk);
  }
  KOKKOS_INLINE_FUNCTION Real PlanckMult(const int g, const Real Tk, const bool probe = false) const {
    if (log_ && active_) return std::exp(Lookup(lwP_, g, Tk, probe));
    return Lookup(wP_, g, Tk, probe);
  }
  KOKKOS_INLINE_FUNCTION Real RossMult(const int g, const Real Tk) const {
    if (log_ && active_) return std::exp(Lookup(lwR_, g, Tk));
    return Lookup(wR_, g, Tk);
  }
  //! R-F5: log copies of the multiplier tables (host, once at init).
  void EnableLogInterp() {
    if (!active_) return;
    if (rho_dep_) {   // R-F11 (v4): only the 3-D multipliers exist
      auto mk3 = [&](const ParArray3D<Real> &src, ParArray3D<Real> &dst, const char *name) {
        dst = ParArray3D<Real>(name, ng_, nr_, nT_);
        auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src);
        auto hd = Kokkos::create_mirror_view(dst);
        for (int g = 0; g < ng_; ++g)
          for (int i = 0; i < nr_; ++i)
            for (int j = 0; j < nT_; ++j) hd(g, i, j) = std::log(std::max(hs(g, i, j), Real(1.0e-300)));
        Kokkos::deep_copy(dst, hd);
      };
      mk3(wP3_, lwP3_, "rad_lwP3");
      mk3(wR3_, lwR3_, "rad_lwR3");
      log_ = true;
      return;
    }
    auto mk = [&](const ParArray2D<Real> &src, ParArray2D<Real> &dst, const char *name) {
      dst = ParArray2D<Real>(name, ng_, nT_);
      auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src);
      auto hd = Kokkos::create_mirror_view(dst);
      for (int g = 0; g < ng_; ++g)
        for (int j = 0; j < nT_; ++j) hd(g, j) = std::log(std::max(hs(g, j), Real(1.0e-300)));
      Kokkos::deep_copy(dst, hd);
    };
    mk(wP_, lwP_, "rad_lwP");
    mk(wR_, lwR_, "rad_lwR");
    log_ = true;
  }
};

//! Build the tabulated multipliers from the Semenov-class band means over [Tmin,Tmax] K,
//! log-spaced with nT points. beta=0 / gray => inactive (mult==1, exact gray equivalence).
inline GroupOpacityTable BuildGroupOpacityTable(const DustOpacityModel &m, const RadGroups &g,
                                                const Real Tmin_K, const Real Tmax_K,
                                                const int nT) {
  GroupOpacityTable tab;
  tab.ng_ = g.n_group;
  tab.nT_ = nT;
  if (g.n_group == 1 || m.beta == 0.0) return tab; // inactive => Lookup returns 1
  // AUDIT 2026-08-05 (N10): validate the T grid before it can reach a device kernel.
  // radiation/opacity_table_nT, _Tmin_K, _Tmax_K are free deck knobs with no checks: nT < 2
  // made dlT_ divide by zero AND drove Lookup's index negative; Tmin <= 0 makes lT0_ = -inf
  // (every lookup then NaN); Tmax <= Tmin makes dlT_ <= 0 so the interpolation runs backwards.
  // All three are silent on the host and only show up as garbage opacity multipliers.
  if (nT < 2)
    throw std::runtime_error("GroupOpacityTable: radiation/opacity_table_nT must be >= 2 (got " +
                             std::to_string(nT) + ")");
  if (!(Tmin_K > 0.0) || !(Tmax_K > Tmin_K))
    throw std::runtime_error(
        "GroupOpacityTable: need 0 < radiation/opacity_table_Tmin_K < opacity_table_Tmax_K (got " +
        std::to_string(Tmin_K) + ", " + std::to_string(Tmax_K) + ")");
  tab.active_ = true;
  tab.lT0_ = std::log10(Tmin_K);
  tab.dlT_ = (std::log10(Tmax_K) - tab.lT0_) / (nT - 1);
  // Clamp tally (see the note on clamp_ctr_): allocated only on a real build, so a
  // default-constructed table keeps a null View and the device kernels skip the atomic.
  tab.clamp_ctr_ = Kokkos::View<unsigned long long *>("rad_grp_clamp_ctr",
                                                     GroupOpacityTable::kNClampCtr);
  tab.wP_ = ParArray2D<Real>("rad_wP", g.n_group, nT);
  tab.wR_ = ParArray2D<Real>("rad_wR", g.n_group, nT);
  auto hP = Kokkos::create_mirror_view(tab.wP_);
  auto hR = Kokkos::create_mirror_view(tab.wR_);
  Real mP[MAX_GROUP], mR[MAX_GROUP];
  for (int it = 0; it < nT; ++it) {
    const Real Tk = std::pow(10.0, tab.lT0_ + it * tab.dlT_);
    GroupMultsAtT(m, g, Tk, mP, mR);
    for (int ig = 0; ig < g.n_group; ++ig) {
      hP(ig, it) = mP[ig];
      hR(ig, it) = mR[ig];
    }
  }
  Kokkos::deep_copy(tab.wP_, hP);
  Kokkos::deep_copy(tab.wR_, hR);
  return tab;
}

//! Build the per-group multiplier table by reading the TABULATED opacity file (produced by
//! gen_opacity_table.py). Binary layout: int64 [ng,nr,nT]; double [lr0,dlr,lT0,dlT]; float64
//! gray kP[nr,nT], kR, ks (skipped here); then multipliers mP[ng,nT], mR[ng,nT] (T-only,
//! rho-independent). This REPLACES the analytic DustOpacityModel band means with the tabulated
//! frequency-resolved values. n_group must match the file's ng.
//!
//! The multipliers are RATIOS of opacities, so they are dimensionless and carry no unit
//! error in either format version -- this reader only needs the header to know where the
//! payload starts. It parses it through the SAME helper as OpacityTable::Load (audit N13);
//! the two used to carry independent copies of the layout.
inline GroupOpacityTable BuildGroupOpacityTableFromFile(const std::string &path,
                                                        const int n_group,
                                                        const Real *runtime_edges,
                                                        const Real rho_unit = 1.0) {
  GroupOpacityTable tab;
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("GroupOpacityTable: cannot open " + path);
  const auto hdr = ReadOpacityHeader(f, path);
  const int ng = hdr.ng;
  const int nr = hdr.nr;
  const int nT = hdr.nT;
  if (ng != n_group)
    throw std::runtime_error("GroupOpacityTable: file n_group mismatch (" +
                             std::to_string(ng) + " != " + std::to_string(n_group) + ")");
  // AUDIT F-24 (2026-08-28) -- VALIDATE THE GROUP EDGES, NOT JUST THE COUNT.
  // Until v3 the file carried only ng/nr/nT: the runtime partition was built independently by
  // BuildRadGroups(n_group, nu_min_hz, nu_max_hz) from deck defaults, and it matched the
  // table's --edges only because two independent defaults happened to coincide (both
  // {0,1e12,1e15,1e30} for n_group=3). Setting radiation/nu_min_hz or nu_max_hz, or
  // regenerating with different --edges, would have silently multiplied
  // groups.PlanckFraction(g,T) (RUNTIME edges) against optab.PlanckMult(g,T) (FILE edges) from
  // DIFFERENT band partitions in the same sum (radiation_moments.cpp:726). No error, no warning.
  // v1/v2 files carry no edges; they are accepted with a warning rather than rejected, so
  // existing decks keep working.
  if (!hdr.nu_edge.empty()) {
    for (int g = 0; g <= ng; ++g) {
      const Real fe = static_cast<Real>(hdr.nu_edge[g]);
      const Real re = runtime_edges[g];
      // Both ends carry sentinels (0 and ~1e30); compare the finite interior edges relatively.
      const bool ok = (g == 0) ? (std::abs(fe - re) <= 1.0)
                    : (g == ng) ? true
                    : (std::abs(fe - re) <= 1.0e-6 * std::max(std::abs(fe), std::abs(re)));
      if (!ok)
        throw std::runtime_error(
            "GroupOpacityTable: group edge " + std::to_string(g) + " of " + path +
            " is " + std::to_string(fe) + " Hz but the runtime partition has " +
            std::to_string(re) + " Hz. The per-group multipliers in the file and the Planck "
            "fractions computed at runtime would then describe DIFFERENT bands. Fix "
            "radiation/nu_min_hz / nu_max_hz to match the table, or regenerate the table "
            "with matching --edges. (Audit F-24.)");
    }
  } else if (parthenon::Globals::my_rank == 0) {
    std::cout << "### WARNING GroupOpacityTable: " << path
              << " is format v" << hdr.version
              << ", which stores NO group frequency edges, so the runtime band partition "
                 "cannot be checked against the one the table was built with. Regenerate with "
                 "gen_opacity_table.py to get a v3 file. (Audit F-24.)" << std::endl;
  }
  tab.ng_ = ng;
  tab.nT_ = nT;
  tab.lT0_ = hdr.lT0;
  tab.dlT_ = hdr.dlT;
  if (ng == 1) return tab; // gray => inactive (Lookup returns 1)
  // audit N10: the T grid comes straight out of a binary header here, so it is even less
  // checked than the analytic path. Same three conditions, same reasons.
  if (nT < 2 || !(tab.dlT_ > 0.0))
    throw std::runtime_error("GroupOpacityTable: bad T grid in " + path + " (nT=" +
                             std::to_string(nT) + ", dlogT=" + std::to_string(tab.dlT_) +
                             "); need nT >= 2 and dlogT > 0");
  // skip the 3 gray arrays kP,kR,ks (each nr*nT doubles)
  f.seekg(static_cast<std::streamoff>(3) * nr * nT * sizeof(double), std::ios::cur);
  tab.active_ = true;
  // Clamp tally (see the note on clamp_ctr_): allocated only on a real build, so a
  // default-constructed table keeps a null View and the device kernels skip the atomic.
  tab.clamp_ctr_ = Kokkos::View<unsigned long long *>("rad_grp_clamp_ctr",
                                                     GroupOpacityTable::kNClampCtr);
  if (hdr.version >= 4) {
    // R-F11: multipliers per (rho, T) on the gray grid, mP then mR, each [ng, nr, nT].
    tab.rho_dep_ = true;
    tab.nr_ = nr;
    tab.lr0_ = hdr.lr0;
    tab.dlr_ = hdr.dlr;
    tab.rho_unit_ = rho_unit;
    if (nr < 2 || !(tab.dlr_ > 0.0))
      throw std::runtime_error("GroupOpacityTable: bad rho grid in v4 file " + path);
    tab.wP3_ = ParArray3D<Real>("rad_wP3", ng, nr, nT);
    tab.wR3_ = ParArray3D<Real>("rad_wR3", ng, nr, nT);
    std::vector<double> b3(static_cast<size_t>(ng) * nr * nT);
    auto read3 = [&](ParArray3D<Real> &v) {
      auto h = Kokkos::create_mirror_view(v);
      f.read(reinterpret_cast<char *>(b3.data()), static_cast<std::streamsize>(b3.size() * sizeof(double)));
      for (int ig = 0; ig < ng; ++ig)
        for (int i = 0; i < nr; ++i)
          for (int it = 0; it < nT; ++it)
            h(ig, i, it) = static_cast<Real>(b3[(static_cast<size_t>(ig) * nr + i) * nT + it]);
      Kokkos::deep_copy(v, h);
    };
    read3(tab.wP3_);
    read3(tab.wR3_);
    if (!f) throw std::runtime_error("GroupOpacityTable: truncated v4 file " + path);
    return tab;
  }
  tab.wP_ = ParArray2D<Real>("rad_wP", ng, nT);
  tab.wR_ = ParArray2D<Real>("rad_wR", ng, nT);
  auto hP = Kokkos::create_mirror_view(tab.wP_);
  auto hR = Kokkos::create_mirror_view(tab.wR_);
  std::vector<double> buf(static_cast<size_t>(ng) * nT);
  auto read_mult = [&](decltype(hP) &h) {
    f.read(reinterpret_cast<char *>(buf.data()),
           static_cast<std::streamsize>(buf.size() * sizeof(double)));
    for (int ig = 0; ig < ng; ++ig)
      for (int it = 0; it < nT; ++it) h(ig, it) = static_cast<Real>(buf[ig * nT + it]);
  };
  read_mult(hP); // mP
  read_mult(hR); // mR
  if (!f) throw std::runtime_error("GroupOpacityTable: truncated file " + path);
  Kokkos::deep_copy(tab.wP_, hP);
  Kokkos::deep_copy(tab.wR_, hR);
  return tab;
}

//----------------------------------------------------------------------------------------
//! HDF5/pack field names for group g's four M1 moments (Er, Fr1, Fr2, Fr3). Group 0 keeps
//! the ORIGINAL gray names ("rad.Er", ...) so a gray (n_group=1) run's output/restart is
//! byte-for-byte unchanged; groups g>0 get a "_gG" suffix. Order is always {Er,Fr1,Fr2,Fr3}.
inline std::vector<std::string> GroupFieldNames(const int g) {
  if (g == 0) return {"rad.Er", "rad.Fr1", "rad.Fr2", "rad.Fr3"};
  const std::string s = "_g" + std::to_string(g);
  return {"rad.Er" + s, "rad.Fr1" + s, "rad.Fr2" + s, "rad.Fr3" + s};
}

//! All radiation moment field names across n_group groups (group 0 first). For n_group=1
//! this is exactly the four gray names, in the original order.
inline std::vector<std::string> AllRadFieldNames(const int n_group) {
  std::vector<std::string> v;
  v.reserve(4 * n_group);
  for (int g = 0; g < n_group; ++g)
    for (const auto &nm : GroupFieldNames(g)) v.push_back(nm);
  return v;
}

//! Per-group EXTINCTION-COEFFICIENT field name, chi_g = rho*(kappa_R,g + kappa_s) [1/length,
//! code units]. Always suffixed -- unlike the moments there is no legacy gray name to preserve.
//! Used by the asymptotic-preserving HLL correction (see CalculateRadFluxes): the cell optical
//! depth in direction d is tau = chi_g * dx_d, and it is computed ONCE per hydro step because
//! rho and T are frozen for the whole operator-split radiation sub-cycle.
inline std::string GroupChiName(const int g) {
  return "rad.chi_g" + std::to_string(g);
}

//----------------------------------------------------------------------------------------
//! Build a default (gray) or log-spaced multigroup structure. n_group=1 -> gray (bit-identical).
inline RadGroups BuildRadGroups(int n_group, Real nu_min_hz, Real nu_max_hz) {
  RadGroups g;
  if (n_group < 1) n_group = 1;
  if (n_group > MAX_GROUP) n_group = MAX_GROUP;
  g.n_group = n_group;
  g.nu_edge[0] = 0.0;
  if (n_group == 1) {
    g.nu_edge[1] = 1.0e30; // [0, inf) -> gray
    return g;
  }
  // interior edges log-spaced in [nu_min, nu_max]; first edge 0, last +inf so nothing is lost.
  // There are n_group-1 interior edges, so the log-spacing denominator is n_group-2.
  // AUDIT 2026-08-05 (N11): n_group == 2 has exactly ONE interior edge, so it cannot span
  // [nu_min, nu_max] -- the loop places that edge at nu_min and nu_max is silently DROPPED.
  // (The `+1e-30` below existed only to keep that degenerate case from dividing 0 by 0; it
  // gave the right answer for the wrong-looking reason.) Say so out loud instead: a deck that
  // sets nu_max and gets two groups deserves to know its upper edge was ignored.
  const Real ll = std::log10(nu_min_hz), lh = std::log10(nu_max_hz);
  if (n_group == 2) {
    g.nu_edge[1] = nu_min_hz; // the single split point
    g.nu_edge[2] = 1.0e30;
    return g;
  }
  for (int i = 1; i < n_group; ++i)
    g.nu_edge[i] = std::pow(10.0, ll + (lh - ll) * (i - 1) / (n_group - 2));
  g.nu_edge[n_group] = 1.0e30;
  return g;
}

//! R-F13 (validation/radiation_sn, 2026-10-09): explicit interior edges [Hz] (n_group - 1 of them, strictly
//! increasing, positive) from radiation/nu_edges_hz, for partitions that are not log-uniform. Edge 0 = 0, last = 1e30
//! as in BuildRadGroups.
inline RadGroups BuildRadGroupsFromEdges(const int n_group, const std::vector<Real> &interior) {
  if (n_group < 2 || n_group > MAX_GROUP || static_cast<int>(interior.size()) != n_group - 1)
    throw std::runtime_error("radiation/nu_edges_hz: need n_group - 1 = " + std::to_string(n_group - 1) +
                             " interior edges (got " + std::to_string(interior.size()) + "), 2 <= n_group <= MAX_GROUP");
  RadGroups g;
  g.n_group = n_group;
  g.nu_edge[0] = 0.0;
  for (int i = 1; i < n_group; ++i) {
    const Real e = interior[i - 1];
    if (!(e > 0.0) || !(e > g.nu_edge[i - 1]))
      throw std::runtime_error("radiation/nu_edges_hz: edges must be positive and strictly increasing");
    g.nu_edge[i] = e;
  }
  g.nu_edge[n_group] = 1.0e30;
  return g;
}

} // namespace Radiation

#endif // RADIATION_RADIATION_GROUPS_HPP_
