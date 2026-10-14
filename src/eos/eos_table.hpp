#ifndef EOS_EOS_TABLE_HPP_
#define EOS_EOS_TABLE_HPP_
//========================================================================================
// AthenaPK -- device-callable tabulated protostellar EOS. Holds 2D Kokkos tables generated
// offline by src/eos/gen_eos_table.py (full multi-Saha: H2 dissociation + H ionization +
// He/He+ ionization + H2 rot/vib + inert He), all in the shared FHC code units. On device
// it does O(1) bilinear interpolation on evenly-spaced log10 grids; the pressure inverse
// (needed by the Riemann solver and sound speed) is a short bisection over the monotone
// esp axis. The struct is small (holds Kokkos View handles + grid scalars) and is copied
// BY VALUE into kernels, so the handles ride along to the device correctly.
//
// Forward grid  (log10 rho_code, log10 esp_code) -> P_code, cs2_code, log10 T[K]
// RT grid       (log10 rho_code, log10 T[K])     -> esp_code
// esp = specific internal energy (code v0^2 units); e_int DENSITY = esp * rho.
//========================================================================================
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

#include "Kokkos_Core.hpp"
#include "basic_types.hpp"       // parthenon::Real
#include "parthenon_arrays.hpp"  // parthenon::ParArray2D

#include "eos_table_format.hpp"  // shared table header parser (audit N14)

namespace EOSTable {
using parthenon::ParArray2D;
using parthenon::Real;

struct EosTable {
  ParArray2D<Real> P_, cs2_, logT_, espT_;

  // HOST MIRROR of the (rho, T) -> esp grid. See EintFromRhoTkHost below for why this
  // exists. Deliberately a plain std::vector and not a Kokkos View: it is touched only by
  // host setup code, and keeping Kokkos out of it means nvcc never has to reason about a
  // HostSpace array inside a __host__ __device__ function.
  //
  // SHARED, not owned: every KOKKOS_LAMBDA that captures an EosTable (or the AdiabaticGLMMHDEOS holding one) copies it
  // by value on the host at EVERY launch. As std::vector members the two mirrors (5.8 MB each for the rho26 table) were
  // deep-copied each time -- the EosTable copy constructor was the hottest host symbol (CPU: 10-23x slower steps on
  // tabulated-EOS problems). A shared_ptr makes the copy a reference-count increment; the device never touches these
  // members. Results are bit-identical.
  std::shared_ptr<std::vector<Real>> espT_h_;
  std::shared_ptr<std::vector<Real>> logT_h_;   // host mirror of the (rho, esp) -> log10 T[K] grid
  Real lr0_ = 0.0, dlr_ = 1.0, le0_ = 0.0, dle_ = 1.0, lT0_ = 0.0, dlT_ = 1.0;
  int nr_ = 0, ne_ = 0, nT_ = 0;
  // Fixed bisection count of EspFromP (hydro/eos_bisections, audit P-13). 20 over the v5_cold
  // table's 7.36-dex esp axis resolves e(rho,P) to 8.1e-6 relative -- and that error is piecewise
  // CONSTANT in P, i.e. non-smooth, unlike the table's own interpolation error; it enters every
  // HLLD face energy. 30 gives 7.9e-9. Default 20 = legacy.
  int n_bisect_ = 20;
  // hydro/eos_interp (validation_tests/eos_tabulated E-F1, 2026-10-04). DEFAULT "p_over_rho" since
  // 2026-10-04 (hydro.cpp); set "p" to bit-reproduce runs made before. "p" (legacy):
  // P is interpolated bilinearly in (log rho, log esp) -- linear in P on a 0.05-dex log-rho axis,
  // which biases P by 0..+0.17 % and the effective sound speed sqrt(dP/drho|_s) by up to +-2.5 %
  // as a sawtooth in rho (measured: acoustic phase speeds match that prediction to 1e-4).
  // "p_over_rho": the table stores P/rho_node and Pof multiplies by rho, which is exact for
  // P proportional to rho at fixed esp; P error <= 2e-4, c_eff error <= 1.4e-3. MUST be set before Load().
  bool p_over_rho_ = false;
  // hydro/eos_extrapolate (validation_tests/eos_tabulated E-F5/E-F6, 2026-10-05). false = legacy edge clamp.
  // true: OUT-OF-TABLE states are continued as the ideal gas they are, instead of nearest-neighbour clamped:
  //   * esp below the esp axis (T < 0.063 K: cold molecular gas, rotation frozen, gamma = 5/3 exactly): P, cs2 and
  //     T scale with esp/esp_min at fixed rho, and EspFromP inverts that exactly. Before, P was pinned at
  //     P(rho, esp_min) = 6.2e-3 rho (code), 4-19 dex above the production pfloor = 1e-8: the pressure floor could
  //     never bind, a cell with 0 < e < e_min carried more pressure than its energy, and a floored cell
  //     (e <= 0 -> e_min, energy injected) was invisible to cons-npfloor and to the clamp counters (E-F5).
  //   * rho below the rho axis with eos_interp = p: P = rho (P/rho)(rho_min, esp), the ideal-gas continuation that
  //     eos_interp = p_over_rho already does; before, P was frozen at P(rho_min, esp) (x10 at 0.1 rho_min, E-F6).
  //   * T below the T axis (RT grid): esp ~ T.  cs2 and T at low rho are intensive and stay clamped.
  // The clamp counters still count these lookups (they ARE outside the table). MUST be set before Load().
  //
  // BIT-IDENTITY (gates on a 3D production restart, 2026-10-05): ANY change to the
  // inline lookup bodies -- even code never executed with the key off -- changes GCC's inlining decisions and the FMA
  // contraction in the callers, and a stiff coupled solver can amplify the last bit to 1e-4 in cons in 3 cycles. A runtime
  // key therefore cannot reproduce the legacy binary bit for bit. To CONTINUE a pre-2026-10-05 run bit-continuously,
  // build with -DATHENAPK_EOS_LEGACY_LOOKUPS=ON: the legacy bodies are compiled verbatim and hydro/eos_extrapolate is
  // refused (default false there). The default build has the extrapolating lookups (key default true).
  bool extrapolate_ = false;
  Real rho_min_ = 1.0, esp_min_ = 1.0, T_min_ = 1.0; // 10^(axis origins), set by Load()
  bool loaded_ = false;

  //----------------------------------------------------------------------------------------
  // SILENT OUT-OF-RANGE CLAMP -- INSTRUMENTED 2026-08-15.
  //
  // `bilin` below clamps BOTH the cell index AND the interpolation weight, so a lookup outside
  // the table returns the nearest EDGE value: nearest-neighbour extrapolation, with no counter,
  // no flag and no warning. That is the right NUMERICAL behaviour (extrapolating a multi-Saha
  // EOS off the end of its grid is worse than clamping), so the numerics are
  // deliberately left alone. What was wrong is that it was UNOBSERVABLE.
  //
  // This project has just been burned by exactly this failure class twice over: the self-gravity
  // non-convergence alarm that could never fire, and an opacity table that was silently two
  // decades short in rho and 0.48 dex short in T while every run reported success. A diagnostic
  // that has never fired is an untested code path; a clamp that cannot be counted is a wrong
  // answer that reports success.
  //
  // COST. The counter is incremented ONLY on the branch that was already taken when a clamp
  // occurs, so a healthy run (nothing out of range) pays zero atomics -- just the comparisons
  // it was doing anyway. If the atomic traffic is ever high, that is itself the finding.
  //
  // NUMERICS-NEUTRAL BY CONSTRUCTION: nothing here is read back into the interpolation. The
  // counter View is a separate allocation and is write-only from the kernel's point of view,
  // so this needs no gate to prove it bit-identical -- there is no path from it to a flux.
  //
  // Layout: [0]=rho below, [1]=rho above, [2]=esp below, [3]=esp above, [4]=T below,
  //         [5]=T above.  jbase selects the second axis pair (2 = esp grid, 4 = T grid).
  // 2026-08-21: the bank is DOUBLED. Slots 0-5 count lookups made on a real CELL STATE;
  // slots 6-11 count lookups made on a root-finder TRIAL value (MGResidual/GrayResidual probe
  // T over a bracket, so they legitimately step outside the table and get pulled back).
  // Before this split the counter could not tell the two apart, and since the probes fire
  // benignly every cycle they MASKED any genuine cold-gas clamp -- an alarm that is always on
  // is not an alarm. Layout repeats: [off+0]=rho below, [off+1]=rho above, [off+2]=esp below,
  // [off+3]=esp above, [off+4]=T below, [off+5]=T above, with off = 0 (state) or 6 (probe).
  static constexpr int kNClampState = 6;
  static constexpr int kNClampCtr = 12;
  Kokkos::View<unsigned long long *> clamp_ctr_;

  // bilinear interpolation of A[n1 x n2] over axes (x0,dx) and (y0,dy); edge-clamped, and the
  // clamps are counted (see above).
  KOKKOS_INLINE_FUNCTION
  static Real bilin(const ParArray2D<Real> &A, int n1, int n2, Real x0, Real dx, Real y0,
                    Real dy, Real x, Real y,
                    const Kokkos::View<unsigned long long *> &ctr, const int jbase,
                    const bool do_count = true, const int ctr_off = 0) {
    // A default-constructed EosTable (built when the tabulated path is off) has an
    // unallocated counter; never dereference it.
    const bool count = do_count && (ctr.data() != nullptr);
    Real fi = (x - x0) / dx;
    int i = static_cast<int>(fi);
    // 2026-08-21 (code review F5): key the LOW-side count off `fi`, not off the truncated
    // index. C++ truncates toward zero, so fi in (-1,0) gives i = 0 and the old `i < 0` test
    // never fired -- yet `ti = fi - i` is negative and IS clamped to 0, so the returned value
    // is edge-clamped nearest-neighbour all the same. Any excursion of less than one grid cell
    // below the floor was therefore silently uncounted: a clamp that cannot be counted is a
    // wrong answer that reports success, which is the exact failure this counter exists to
    // prevent. GroupOpacityTable::Lookup already keys off `f` for this reason.
    if (count && fi < 0.0) Kokkos::atomic_fetch_add(&ctr(ctr_off + 0), 1ull);
    if (i < 0) i = 0;
    if (i > n1 - 2) {
      i = n1 - 2;
      if (count) Kokkos::atomic_fetch_add(&ctr(ctr_off + 1), 1ull);
    }
    Real ti = fi - i;
    if (ti < 0.0) ti = 0.0;
    if (ti > 1.0) ti = 1.0;
    Real fj = (y - y0) / dy;
    int j = static_cast<int>(fj);
    if (count && fj < 0.0) Kokkos::atomic_fetch_add(&ctr(ctr_off + jbase), 1ull);
    if (j < 0) j = 0;
    if (j > n2 - 2) {
      j = n2 - 2;
      if (count) Kokkos::atomic_fetch_add(&ctr(ctr_off + jbase + 1), 1ull);
    }
    Real tj = fj - j;
    if (tj < 0.0) tj = 0.0;
    if (tj > 1.0) tj = 1.0;
    return (1.0 - ti) * (1.0 - tj) * A(i, j) + ti * (1.0 - tj) * A(i + 1, j) +
           (1.0 - ti) * tj * A(i, j + 1) + ti * tj * A(i + 1, j + 1);
  }

#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
  // LEGACY lookups, verbatim (compile-time switch, CMake -DATHENAPK_EOS_LEGACY_LOOKUPS=ON): for continuing a run made
  // before 2026-10-05 BIT FOR BIT. Any change to these inline bodies -- even code that is never executed with the
  // runtime key off -- changes GCC's inlining and FMA contraction in the callers and hence the last bit of a
  // production restart (gates efix1-5, bisection B1, validation_tests/eos_tabulated). hydro/eos_extrapolate is
  // refused in this build.
  KOKKOS_INLINE_FUNCTION
  Real Pof(Real rho, Real esp) const {
    return (p_over_rho_ ? rho : 1.0) *
           bilin(P_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), std::log10(esp),
                 clamp_ctr_, 2);
  }
  KOKKOS_INLINE_FUNCTION
  Real Cs2of(Real rho, Real esp) const {
    return bilin(cs2_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), std::log10(esp),
                 clamp_ctr_, 2);
  }
  KOKKOS_INLINE_FUNCTION
  Real LogTof(Real rho, Real esp) const {
    return bilin(logT_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), std::log10(esp),
                 clamp_ctr_, 2);
  }
  // Invert P over log10(esp) at fixed rho (P is monotone increasing in esp).
  KOKKOS_INLINE_FUNCTION
  Real EspFromP(Real rho, Real P) const {
    const Real lr = std::log10(rho);
    Real lelo = le0_, lehi = le0_ + (ne_ - 1) * dle_;
    // 20 bisections => interval 2^-20 ~ 1e-6 of the log-esp range, still ~1e4x below the ~1%
    // table interpolation accuracy. FIXED count (not early-exit) is deliberate: on GPU a warp
    // runs until its slowest lane converges, so a fixed low count is faster + divergence-free than
    // an early-exit. 20 (was 30) trims this hot per-cell inversion (sound speed etc.) ~33%.
    // CLAMP COUNTING (2026-08-15): the bisection below probes lem across the table's OWN esp
    // range, so it can never leave the esp axis -- but it calls bilin 20 times, and if rho is
    // out of range every one of those would tally the rho clamp, inflating the count 20x and
    // making the diagnostic unreadable. So the loop runs with counting OFF and the rho axis is
    // probed exactly once, here.
    if (clamp_ctr_.data() != nullptr) {
      const Real fi = (lr - lr0_) / dlr_;
      if (fi < 0.0) Kokkos::atomic_fetch_add(&clamp_ctr_(0), 1ull);
      if (static_cast<int>(fi) > nr_ - 2) Kokkos::atomic_fetch_add(&clamp_ctr_(1), 1ull);
    }
    for (int it = 0; it < n_bisect_; ++it) {
      const Real lem = 0.5 * (lelo + lehi);
      const Real Pm =
          (p_over_rho_ ? rho : 1.0) *
          bilin(P_, nr_, ne_, lr0_, dlr_, le0_, dle_, lr, lem, clamp_ctr_, 2, false);
      if (Pm < P) lelo = lem; else lehi = lem;
    }
    return std::pow(10.0, 0.5 * (lelo + lehi));
  }

#else
  // Private copy of bilin for the helpers below: their calls must not change GCC's inlining decisions for the
  // legacy bilin (call counts enter the heuristics). Same arithmetic.
  KOKKOS_INLINE_FUNCTION
  static Real bilin_x(const ParArray2D<Real> &A, int n1, int n2, Real x0, Real dx, Real y0,
                    Real dy, Real x, Real y,
                    const Kokkos::View<unsigned long long *> &ctr, const int jbase,
                    const bool do_count = true, const int ctr_off = 0) {
    // A default-constructed EosTable (built when the tabulated path is off) has an
    // unallocated counter; never dereference it.
    const bool count = do_count && (ctr.data() != nullptr);
    Real fi = (x - x0) / dx;
    int i = static_cast<int>(fi);
    // 2026-08-21 (code review F5): key the LOW-side count off `fi`, not off the truncated
    // index. C++ truncates toward zero, so fi in (-1,0) gives i = 0 and the old `i < 0` test
    // never fired -- yet `ti = fi - i` is negative and IS clamped to 0, so the returned value
    // is edge-clamped nearest-neighbour all the same. Any excursion of less than one grid cell
    // below the floor was therefore silently uncounted: a clamp that cannot be counted is a
    // wrong answer that reports success, which is the exact failure this counter exists to
    // prevent. GroupOpacityTable::Lookup already keys off `f` for this reason.
    if (count && fi < 0.0) Kokkos::atomic_fetch_add(&ctr(ctr_off + 0), 1ull);
    if (i < 0) i = 0;
    if (i > n1 - 2) {
      i = n1 - 2;
      if (count) Kokkos::atomic_fetch_add(&ctr(ctr_off + 1), 1ull);
    }
    Real ti = fi - i;
    if (ti < 0.0) ti = 0.0;
    if (ti > 1.0) ti = 1.0;
    Real fj = (y - y0) / dy;
    int j = static_cast<int>(fj);
    if (count && fj < 0.0) Kokkos::atomic_fetch_add(&ctr(ctr_off + jbase), 1ull);
    if (j < 0) j = 0;
    if (j > n2 - 2) {
      j = n2 - 2;
      if (count) Kokkos::atomic_fetch_add(&ctr(ctr_off + jbase + 1), 1ull);
    }
    Real tj = fj - j;
    if (tj < 0.0) tj = 0.0;
    if (tj > 1.0) tj = 1.0;
    return (1.0 - ti) * (1.0 - tj) * A(i, j) + ti * (1.0 - tj) * A(i + 1, j) +
           (1.0 - ti) * tj * A(i, j + 1) + ti * tj * A(i + 1, j + 1);
  }
  // ---- out-of-line helpers of hydro/eos_extrapolate (see the note on extrapolate_) ----
#if defined(__CUDACC__)
#define EOS_NOINLINE __noinline__
#else
#define EOS_NOINLINE __attribute__((noinline))
#endif
  KOKKOS_INLINE_FUNCTION EOS_NOINLINE void CountEspLow() const {
    if (clamp_ctr_.data() != nullptr) Kokkos::atomic_fetch_add(&clamp_ctr_(2), 1ull);
  }
  // Pof operands: fac (P per stored value) and the log-esp coordinate
  KOKKOS_INLINE_FUNCTION EOS_NOINLINE void XPofOperands(const Real rho, const Real esp, Real &fac,
                                                         Real &lesp) const {
    if (!p_over_rho_ && rho < rho_min_) fac = rho / rho_min_; // E-F6
    if (esp < esp_min_) {                                     // E-F5: P ~ esp at fixed rho
      CountEspLow();
      fac *= esp / esp_min_;
      lesp = le0_;
    }
  }
  KOKKOS_INLINE_FUNCTION EOS_NOINLINE Real XCs2Low(const Real rho, const Real esp) const {
    CountEspLow();
    return bilin_x(cs2_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), le0_, clamp_ctr_, 2) *
           (esp / esp_min_);
  }
  KOKKOS_INLINE_FUNCTION EOS_NOINLINE Real XLogTLow(const Real rho, const Real esp,
                                                    const bool count) const {
    if (count) CountEspLow();
    return bilin_x(logT_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), le0_, clamp_ctr_, 2,
                 count) +
           std::log10(esp / esp_min_);
  }
  // EspFromP below the lowest tabulated pressure: exact inverse of the continuation (true = handled)
  KOKKOS_INLINE_FUNCTION EOS_NOINLINE bool XEspLowP(const Real rho, const Real P, Real &esp) const {
    const Real lr = std::log10(rho);
    const Real fac = p_over_rho_ ? rho : (rho < rho_min_ ? rho / rho_min_ : 1.0);
    const Real Pmin = fac * bilin_x(P_, nr_, ne_, lr0_, dlr_, le0_, dle_, lr, le0_, clamp_ctr_, 2, false);
    if (!(P < Pmin)) return false;
    CountEspLow();
    esp = esp_min_ * (P / Pmin);
    return true;
  }
  KOKKOS_INLINE_FUNCTION EOS_NOINLINE void XEintOperands(const Real rho, const Real Tk, const int off,
                                                          Real &lT, Real &r) const {
    if (Tk < T_min_) {
      if (clamp_ctr_.data() != nullptr) Kokkos::atomic_fetch_add(&clamp_ctr_(off + 4), 1ull);
      lT = lT0_;
      r = rho * (Tk / T_min_);
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real Pof(Real rho, Real esp) const {
    Real fac = (p_over_rho_ ? rho : 1.0), lesp = std::log10(esp);
    if (extrapolate_) XPofOperands(rho, esp, fac, lesp);
    return fac * bilin(P_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), lesp, clamp_ctr_, 2);
  }
  KOKKOS_INLINE_FUNCTION
  Real Cs2of(Real rho, Real esp) const {
    if (extrapolate_ && esp < esp_min_) return XCs2Low(rho, esp);
    return bilin(cs2_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), std::log10(esp),
                 clamp_ctr_, 2);
  }
  KOKKOS_INLINE_FUNCTION
  Real LogTof(Real rho, Real esp) const {
    if (extrapolate_ && esp < esp_min_) return XLogTLow(rho, esp, true);
    return bilin(logT_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho), std::log10(esp),
                 clamp_ctr_, 2);
  }
  // Invert P over log10(esp) at fixed rho (P is monotone increasing in esp).
  KOKKOS_INLINE_FUNCTION
  Real EspFromP(Real rho, Real P) const {
    const Real lr = std::log10(rho);
    Real lelo = le0_, lehi = le0_ + (ne_ - 1) * dle_;
    // 20 bisections => interval 2^-20 ~ 1e-6 of the log-esp range, still ~1e4x below the ~1%
    // table interpolation accuracy. FIXED count (not early-exit) is deliberate: on GPU a warp
    // runs until its slowest lane converges, so a fixed low count is faster + divergence-free than
    // an early-exit. 20 (was 30) trims this hot per-cell inversion (sound speed etc.) ~33%.
    // CLAMP COUNTING (2026-08-15): the bisection below probes lem across the table's OWN esp
    // range, so it can never leave the esp axis -- but it calls bilin 20 times, and if rho is
    // out of range every one of those would tally the rho clamp, inflating the count 20x and
    // making the diagnostic unreadable. So the loop runs with counting OFF and the rho axis is
    // probed exactly once, here.
    if (clamp_ctr_.data() != nullptr) {
      const Real fi = (lr - lr0_) / dlr_;
      if (fi < 0.0) Kokkos::atomic_fetch_add(&clamp_ctr_(0), 1ull);
      if (static_cast<int>(fi) > nr_ - 2) Kokkos::atomic_fetch_add(&clamp_ctr_(1), 1ull);
    }
    Real fac = (p_over_rho_ ? rho : 1.0);
    if (extrapolate_) {
      Real e_low;
      if (XEspLowP(rho, P, e_low)) return e_low; // E-F5
      if (!p_over_rho_ && rho < rho_min_) fac = rho / rho_min_; // E-F6
    }
    for (int it = 0; it < n_bisect_; ++it) {
      const Real lem = 0.5 * (lelo + lehi);
      const Real Pm =
          fac *
          bilin(P_, nr_, ne_, lr0_, dlr_, le0_, dle_, lr, lem, clamp_ctr_, 2, false);
      if (Pm < P) lelo = lem; else lehi = lem;
    }
    return std::pow(10.0, 0.5 * (lelo + lehi));
  }

#endif
  // ---- code-unit interface (mirrors the analytic HydrogenEOS names) ----
  // gas pressure from (rho, internal-energy DENSITY)
  KOKKOS_INLINE_FUNCTION
  Real PresFromRhoEint(Real rho, Real eint) const { return Pof(rho, eint / rho); }
  // internal-energy DENSITY from (rho, pressure)   [inverse of the above]
  KOKKOS_INLINE_FUNCTION
  Real EintFromRhoPres(Real rho, Real pres) const { return EspFromP(rho, pres) * rho; }
  // adiabatic sound speed^2 from (rho, pressure)
  KOKKOS_INLINE_FUNCTION
  Real AsqFromRhoPres(Real rho, Real pres) const { return Cs2of(rho, EspFromP(rho, pres)); }
  // gas temperature [K] from (rho, internal-energy DENSITY)
  KOKKOS_INLINE_FUNCTION
  Real TemperatureK(Real rho, Real eint) const {
    return std::pow(10.0, LogTof(rho, eint / rho));
  }
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
  // gas temperature [K] from (rho, pressure). The T_code = p/rho identity only holds for the
  // ideal gas; with dissociation/ionization
  // mu varies, so T must come from the EOS. (esp inverse then the log10T table.)
  KOKKOS_INLINE_FUNCTION
  Real TemperatureKFromPres(Real rho, Real pres) const {
    const Real esp = EspFromP(rho, pres);
    // EspFromP already tallied the rho axis for this call; count only the esp axis here.
    return std::pow(10.0, bilin(logT_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho),
                                std::log10(esp), clamp_ctr_, 2, false));
  }
  // internal-energy DENSITY from (rho, T[K])   [for RT]
  KOKKOS_INLINE_FUNCTION
  //! `probe = true` marks a root-finder TRIAL evaluation (MGResidual / GrayResidual), which is
  //! tallied in the separate probe bank. Default false keeps every existing caller counting as
  //! a real cell state, so no call site changes behaviour by accident.
  Real EintFromRhoTk(Real rho, Real Tk, const bool probe = false) const {
    // jbase = 4: this is the (rho, T) grid, whose second axis is TEMPERATURE, not esp.
    return bilin(espT_, nr_, nT_, lr0_, dlr_, lT0_, dlT_, std::log10(rho), std::log10(Tk),
                 clamp_ctr_, 4, true, probe ? kNClampState : 0) *
           rho;
  }

#else
  // gas temperature [K] from (rho, pressure). The T_code = p/rho identity only holds for the
  // ideal gas; with dissociation/ionization
  // mu varies, so T must come from the EOS. (esp inverse then the log10T table.)
  KOKKOS_INLINE_FUNCTION
  Real TemperatureKFromPres(Real rho, Real pres) const {
    const Real esp = EspFromP(rho, pres);
    // EspFromP already tallied the rho axis for this call; count only the esp axis here.
    if (extrapolate_ && esp < esp_min_) return std::pow(10.0, XLogTLow(rho, esp, false));
    return std::pow(10.0, bilin(logT_, nr_, ne_, lr0_, dlr_, le0_, dle_, std::log10(rho),
                                std::log10(esp), clamp_ctr_, 2, false));
  }
  // internal-energy DENSITY from (rho, T[K])   [for RT]
  KOKKOS_FORCEINLINE_FUNCTION // B1 (legacy) inlined it at every call site; keep that with the operand hook
  //! `probe = true` marks a root-finder TRIAL evaluation (MGResidual / GrayResidual), which is
  //! tallied in the separate probe bank. Default false keeps every existing caller counting as
  //! a real cell state, so no call site changes behaviour by accident.
  Real EintFromRhoTk(Real rho, Real Tk, const bool probe = false) const {
    // jbase = 4: this is the (rho, T) grid, whose second axis is TEMPERATURE, not esp.
    Real lT = std::log10(Tk), r = rho;
    if (extrapolate_) XEintOperands(rho, Tk, probe ? kNClampState : 0, lT, r);
    return bilin(espT_, nr_, nT_, lr0_, dlr_, lT0_, dlT_, std::log10(rho), lT,
                 clamp_ctr_, 4, true, probe ? kNClampState : 0) *
           r;
  }

#endif
  //! True when the file was a legacy (v1) table, i.e. its axes and arrays were baked in the
  //! GENERATOR's code units rather than this run's. Lets the caller warn without re-parsing.
  bool used_legacy_code_units = false;

  //----------------------------------------------------------------------------------------
  // HOST-ONLY EOS LOOKUP.
  //
  // WHY THIS EXISTS. `EintFromRhoTk` is KOKKOS_INLINE_FUNCTION and reads the DEVICE View
  // `espT_`. Two SETUP-TIME callers run on the host, not in a kernel:
  //     pgen/collapse_be.cpp  -- the WP-13 inversion that makes e_int/p exact at T_code = 1
  //     hydro/hydro.cpp       -- the audit F-22 `hydro/Tfloor` inversion (inert: 0 of 4 decks)
  // Dereferencing a device pointer from host code is illegal on any machine. x86 + H100
  // tolerated it silently, but GH200 traps it at startup:
  //     Kokkos::View ERROR: attempt to access inaccessible memory space (label="eos_espT")
  // (every rank, before cycle 0.)
  //
  // KEEP `bilin_host` AND `bilin` IN LOCKSTEP. The arithmetic below is a line-for-line copy
  // of `bilin`'s, in the same association order, so the host and device paths return
  // bit-identical values for the same inputs. The ONLY differences are that this one takes
  // a flat std::vector instead of a ParArray2D, and drops the clamp counters -- those live
  // in a device View, and these are one-off setup calls, not cell-state lookups. If you edit
  // one of the two, edit the other.
  static Real bilin_host(const std::vector<Real> &A, int n1, int n2, Real x0, Real dx,
                         Real y0, Real dy, Real x, Real y) {
    Real fi = (x - x0) / dx;
    int i = static_cast<int>(fi);
    if (i < 0) i = 0;
    if (i > n1 - 2) i = n1 - 2;
    Real ti = fi - i;
    if (ti < 0.0) ti = 0.0;
    if (ti > 1.0) ti = 1.0;
    Real fj = (y - y0) / dy;
    int j = static_cast<int>(fj);
    if (j < 0) j = 0;
    if (j > n2 - 2) j = n2 - 2;
    Real tj = fj - j;
    if (tj < 0.0) tj = 0.0;
    if (tj > 1.0) tj = 1.0;
    const auto at = [&](int a, int b) { return A[static_cast<size_t>(a) * n2 + b]; };
    return (1.0 - ti) * (1.0 - tj) * at(i, j) + ti * (1.0 - tj) * at(i + 1, j) +
           (1.0 - ti) * tj * at(i, j + 1) + ti * tj * at(i + 1, j + 1);
  }

#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
  //! Host twin of EintFromRhoTk. Returns the internal-energy DENSITY at (rho, T[K]).
  //! Throws rather than returning garbage if the table was never loaded, because both
  //! callers are setup code where a silent zero would poison the initial condition.
  Real EintFromRhoTkHost(Real rho, Real Tk) const {
    if (!espT_h_ || espT_h_->empty())
      throw std::runtime_error("EosTable::EintFromRhoTkHost called before Load()");
    return bilin_host(*espT_h_, nr_, nT_, lr0_, dlr_, lT0_, dlT_, std::log10(rho),
                      std::log10(Tk)) *
           rho;
  }

  //! Host twin of TemperatureK: gas temperature [K] from (rho, internal-energy DENSITY).
  //! For problem generators, which run on HOST mirrors of the conserved variables and so
  //! cannot call the device lookup.
  Real TemperatureKHost(Real rho, Real eint) const {
    if (!logT_h_ || logT_h_->empty())
      throw std::runtime_error("EosTable::TemperatureKHost called before Load()");
    const Real esp = eint / rho;
    return std::pow(10.0, bilin_host(*logT_h_, nr_, ne_, lr0_, dlr_, le0_, dle_,
                                     std::log10(rho), std::log10(esp)));
  }

#else
  //! Host twin of EintFromRhoTk. Returns the internal-energy DENSITY at (rho, T[K]).
  //! Throws rather than returning garbage if the table was never loaded, because both
  //! callers are setup code where a silent zero would poison the initial condition.
  Real EintFromRhoTkHost(Real rho, Real Tk) const {
    if (!espT_h_ || espT_h_->empty())
      throw std::runtime_error("EosTable::EintFromRhoTkHost called before Load()");
    const bool lowT = extrapolate_ && Tk < T_min_; // same continuation as EintFromRhoTk
    return bilin_host(*espT_h_, nr_, nT_, lr0_, dlr_, lT0_, dlT_, std::log10(rho),
                      lowT ? lT0_ : std::log10(Tk)) *
           (lowT ? rho * (Tk / T_min_) : rho);
  }

  //! Host twin of TemperatureK: gas temperature [K] from (rho, internal-energy DENSITY).
  //! For problem generators, which run on HOST mirrors of the conserved variables and so
  //! cannot call the device lookup.
  Real TemperatureKHost(Real rho, Real eint) const {
    if (!logT_h_ || logT_h_->empty())
      throw std::runtime_error("EosTable::TemperatureKHost called before Load()");
    const Real esp = eint / rho;
    const bool lowe = extrapolate_ && esp < esp_min_; // same continuation as LogTof
    return std::pow(10.0, bilin_host(*logT_h_, nr_, ne_, lr0_, dlr_, le0_, dle_,
                                     std::log10(rho), lowe ? le0_ : std::log10(esp)) +
                              (lowe ? std::log10(esp / esp_min_) : 0.0));
  }

#endif
  // ---- host-side loader (reads the flat binary from gen_eos_table.py) ----
  //! `rho_unit` [g/cm^3] and `v_unit` [cm/s] are this run's authoritative scales. A v2 (cgs)
  //! table is converted into code units with them here, so the file is independent of any
  //! particular IC (audit N14). A legacy table is consumed verbatim -- bit-identical to the
  //! historical behaviour, and carrying the historical bias, which the caller reports.
  void Load(const std::string &path, const Real rho_unit, const Real v_unit) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("EosTable: cannot open table file " + path);
    const auto hdr = ReadEosHeader(f, path);
    nr_ = hdr.nr;
    ne_ = hdr.ne;
    nT_ = hdr.nT;
    lr0_ = hdr.lr0; dlr_ = hdr.dlr;
    le0_ = hdr.le0; dle_ = hdr.dle;
    lT0_ = hdr.lT0; dlT_ = hdr.dlT;
    if (nr_ < 2 || ne_ < 2 || nT_ < 2 || !(dlr_ > 0.0) || !(dle_ > 0.0) || !(dlT_ > 0.0))
      throw std::runtime_error("EosTable: bad grid in " + path);

    used_legacy_code_units = !hdr.in_cgs;
    // cgs -> code. The axes are log10-spaced, so this is a pure shift of the ORIGIN and the
    // SPACING is invariant; the payload arrays scale by fixed factors. log10 T[K] is
    // unit-free and is left alone. All factors are 1 / no-op for a legacy table.
    const Real esp_unit = v_unit * v_unit;         // erg/g
    const Real e_unit = rho_unit * esp_unit;       // erg/cm^3
    Real scale_P = 1.0, scale_cs2 = 1.0, scale_esp = 1.0;
    if (hdr.in_cgs) {
      lr0_ -= static_cast<Real>(std::log10(static_cast<double>(rho_unit)));
      le0_ -= static_cast<Real>(std::log10(static_cast<double>(esp_unit)));
      scale_P = 1.0 / e_unit;
      scale_cs2 = 1.0 / esp_unit;
      scale_esp = 1.0 / esp_unit;
    }

    rho_min_ = std::pow(10.0, lr0_);
    esp_min_ = std::pow(10.0, le0_);
    T_min_ = std::pow(10.0, lT0_);

    auto load2d = [&](ParArray2D<Real> &view, const char *name, int n1, int n2,
                      const Real scale) {
      view = ParArray2D<Real>(name, n1, n2);
      auto h = Kokkos::create_mirror_view(view);
      std::vector<double> buf(static_cast<size_t>(n1) * n2);
      f.read(reinterpret_cast<char *>(buf.data()),
             static_cast<std::streamsize>(buf.size() * sizeof(double)));
      for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
          h(i, j) = static_cast<Real>(buf[i * n2 + j]) * scale;
      Kokkos::deep_copy(view, h);
    };
    load2d(P_, "eos_P", nr_, ne_, scale_P);
    if (p_over_rho_) { // store P / rho_node (code units; lr0_ is already the code-unit origin)
      auto Pd = P_;
      const Real lr0 = lr0_, dlr = dlr_;
      Kokkos::parallel_for(
          "eos_P_over_rho", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nr_, ne_}),
          KOKKOS_LAMBDA(const int i, const int j) {
            Pd(i, j) /= Kokkos::pow(10.0, lr0 + i * dlr);
          });
      Kokkos::fence();
    }
    load2d(cs2_, "eos_cs2", nr_, ne_, scale_cs2);
    load2d(logT_, "eos_logT", nr_, ne_, 1.0); // log10 T[K]: unit-free
    load2d(espT_, "eos_espT", nr_, nT_, scale_esp);
    if (!f) throw std::runtime_error("EosTable: truncated table file " + path);
    // Host mirror of the (rho, T) grid for the two setup-time callers (see EintFromRhoTkHost).
    // Copied back from the device View rather than from `buf` so it is guaranteed to hold
    // exactly what the kernels will read, scaling included.
    {
      auto hm = Kokkos::create_mirror_view(espT_);
      Kokkos::deep_copy(hm, espT_);
      auto v = std::make_shared<std::vector<Real>>(static_cast<size_t>(nr_) * nT_);
      for (int i = 0; i < nr_; ++i)
        for (int j = 0; j < nT_; ++j)
          (*v)[static_cast<size_t>(i) * nT_ + j] = hm(i, j);
      espT_h_ = v;
    }
    {
      auto hm = Kokkos::create_mirror_view(logT_);
      Kokkos::deep_copy(hm, logT_);
      auto v = std::make_shared<std::vector<Real>>(static_cast<size_t>(nr_) * ne_);
      for (int i = 0; i < nr_; ++i)
        for (int j = 0; j < ne_; ++j)
          (*v)[static_cast<size_t>(i) * ne_ + j] = hm(i, j);
      logT_h_ = v;
    }
    // Out-of-range clamp tally (see the note on bilin). Allocated ONLY here, so every
    // default-constructed EosTable keeps a null View and the kernels skip the atomic.
    // Kokkos zero-initialises a View by default, which is what we want.
    clamp_ctr_ = Kokkos::View<unsigned long long *>("eos_clamp_ctr", kNClampCtr);
    loaded_ = true;
  }

  //! Host-side: copy the clamp tallies out and zero them for the next interval.
  //! Returns false if the table was never loaded (nothing to report).
  //! NOTE this is a HOST function -- it fences and does a device->host copy, so call it once
  //! per cycle at most, never inside a kernel or a per-block loop.
  bool ReadAndResetClampCounts(unsigned long long out[kNClampCtr]) const {
    if (clamp_ctr_.data() == nullptr) return false;
    auto h = Kokkos::create_mirror_view(clamp_ctr_);
    Kokkos::deep_copy(h, clamp_ctr_);
    for (int q = 0; q < kNClampCtr; ++q) out[q] = h(q);
    Kokkos::deep_copy(clamp_ctr_, 0ull);
    return true;
  }

  //! Axis extents in the table's own log10 coordinates -- so a report can say WHERE the wall
  //! is, not merely that one was hit.
  void AxisRanges(Real &lr_lo, Real &lr_hi, Real &le_lo, Real &le_hi, Real &lT_lo,
                  Real &lT_hi) const {
    lr_lo = lr0_; lr_hi = lr0_ + (nr_ - 1) * dlr_;
    le_lo = le0_; le_hi = le0_ + (ne_ - 1) * dle_;
    lT_lo = lT0_; lT_hi = lT0_ + (nT_ - 1) * dlT_;
  }
};

} // namespace EOSTable

#endif // EOS_EOS_TABLE_HPP_
