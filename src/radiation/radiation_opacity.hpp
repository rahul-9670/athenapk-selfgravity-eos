//========================================================================================
// AthenaPK - M1 moment radiation transport package.
// Gray opacity model + code-unit calibration for the FHC collapse problem.
//
// Two responsibilities:
//   (1) Convert the physical radiation constants / opacities into the run's SHARED
//       code units (the same rho0/v0/l0 normalization the two codes use), so that
//       arad and the absorption coefficient are dimensionally consistent with the
//       gas energy density (= pressure unit rho0*v0^2) and the M1 transport.
//   (2) Provide a device-callable gray opacity kappa(rho,T): either a constant
//       placeholder or a Bell & Lin (1994) low-temperature dust law, which is what
//       produces the radiative trapping that sets the first-hydrostatic-core entropy.
//========================================================================================
#ifndef RADIATION_RADIATION_OPACITY_HPP_
#define RADIATION_RADIATION_OPACITY_HPP_

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <basic_types.hpp>      // parthenon::Real
#include <parthenon_arrays.hpp> // parthenon::ParArray2D (tabulated opacity, device)

#include "opacity_table_format.hpp" // shared table header parser (audit N13)

namespace Radiation {

using parthenon::Real;
using parthenon::ParArray2D;

//! Tiny guard added to denominators (formerly in the removed M1 closure header).
KOKKOS_FORCEINLINE_FUNCTION Real RadFuzz() { return 1.0e-100; }

// CGS physical constants used only for the one-time code-unit calibration (host side).
namespace cgs {
constexpr Real a_rad = 7.5657e-15;     // radiation constant a_R [erg cm^-3 K^-4]
constexpr Real k_B = 1.380649e-16;     // Boltzmann [erg/K]
constexpr Real m_H = 1.6726219e-24;    // hydrogen mass [g]
constexpr Real c_light = 2.99792458e10; // speed of light [cm/s]
} // namespace cgs

enum class OpacityModel { constant, dust, belllin, tabulated };

//----------------------------------------------------------------------------------------
//! Tabulated GRAY opacity magnitude (code units) on a (log10 rho_cgs, log10 T_K) grid, produced
//! offline by gen_opacity_table.py: true Planck mean kappa_P, Rosseland mean kappa_R, and
//! scattering kappa_s. State-of-the-art: the frequency-resolved dust+gas physics lives in the
//! generator (swappable for a real Semenov/DSHARP monochromatic dataset); the device only
//! bilinear-interpolates. Small (Kokkos View handles + scalars) => copied BY VALUE into kernels.
//! The per-group MULTIPLIERS (kappa_{P,R},g/kappa_gray, rho-independent => T-only) from the SAME
//! file feed GroupOpacityTable (radiation_groups.hpp).
struct OpacityTable {
  ParArray2D<Real> kP_, kR_, ks_; // [nr][nT], code units
  Real lr0_ = 0.0, dlr_ = 1.0, lT0_ = 0.0, dlT_ = 1.0; // log10(rho_cgs), log10(T_K) axes
  Real rho_unit = 1.0, T_unit = 1.0; // code->cgs for the lookup axes
  int ng_ = 1, nr_ = 0, nT_ = 0;
  bool loaded_ = false;
  //! R-F5 (radiation/opacity_interp = log, validation/radiation_sn 2026-10-08): interpolate log(kappa)
  //! bilinearly instead of kappa. Exact on every Bell & Lin power-law piece kappa ~ rho^a T^b; linear
  //! interpolation overestimates by up to +25.6 % at cell centres in the T^-24 evaporation regime (0.025 dex
  //! grid). Default false = the historical linear lookup.
  bool log_interp_ = false;
  ParArray2D<Real> lkP_, lkR_;   // log(kappa) copies, filled by EnableLogInterp()

  //----------------------------------------------------------------------------------------
  // SILENT OUT-OF-RANGE CLAMP -- INSTRUMENTED 2026-08-17, mirroring src/eos/eos_table.hpp.
  //
  // `bilin` clamps BOTH the cell index AND the interpolation weight, so a lookup outside the
  // table returns the nearest EDGE value: nearest-neighbour extrapolation, with no counter, no
  // flag and no warning. The numerics are deliberately left alone -- extrapolating a dust
  // opacity off the end of its grid is worse than clamping -- but it was UNOBSERVABLE.
  //
  // THIS HALF WAS MISSED. The EOS table got this treatment on 2026-08-15 and the opacity tables
  // did not; the item was reported closed on the strength of the EOS half alone. Which is
  // itself the failure mode: this project has been burned by a clamped opacity table that was
  // two decades short in rho and 0.48 dex short in T while every run reported success (+477%
  // error), and by a convergence alarm that could never fire. A clamp that cannot be counted is
  // a wrong answer that reports success.
  //
  // Layout: [0]=rho below, [1]=rho above, [2]=T below, [3]=T above.
  // Cost: the atomic fires ONLY on the branch that was already taken when a clamp occurs, so a
  // healthy run pays zero atomics. High traffic here is itself the finding.
  // Numerics-neutral by construction: the counter is a separate allocation, write-only from the
  // kernel's point of view, with no path back into the interpolation -- so it needs no gate to
  // be provably bit-identical.
  // 2026-08-21: DOUBLED into a state bank and a probe bank -- see eos_table.hpp for the
  // rationale. [off+0]=rho below, [off+1]=rho above, [off+2]=T below, [off+3]=T above,
  // off = 0 for a real cell state, 4 for a root-finder trial value.
  static constexpr int kNClampState = 4;
  static constexpr int kNClampCtr = 8;
  Kokkos::View<unsigned long long *> clamp_ctr_;

  KOKKOS_INLINE_FUNCTION
  Real bilin(const ParArray2D<Real> &A, const Real lr, const Real lT,
             const int ctr_off = 0) const {
    const bool count = (clamp_ctr_.data() != nullptr);
    Real fi = (lr - lr0_) / dlr_;
    int i = static_cast<int>(fi);
    // see eos_table.hpp: count off the coordinate, not the truncated index
    if (count && fi < 0.0) Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 0), 1ull);
    if (i < 0) i = 0;
    if (i > nr_ - 2) {
      i = nr_ - 2;
      if (count) Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 1), 1ull);
    }
    Real ti = fi - i;
    ti = (ti < 0.0) ? 0.0 : ((ti > 1.0) ? 1.0 : ti);
    Real fj = (lT - lT0_) / dlT_;
    int j = static_cast<int>(fj);
    // see eos_table.hpp: count off the coordinate, not the truncated index
    if (count && fj < 0.0) Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 2), 1ull);
    if (j < 0) j = 0;
    if (j > nT_ - 2) {
      j = nT_ - 2;
      if (count) Kokkos::atomic_fetch_add(&clamp_ctr_(ctr_off + 3), 1ull);
    }
    Real tj = fj - j;
    tj = (tj < 0.0) ? 0.0 : ((tj > 1.0) ? 1.0 : tj);
    return (1.0 - ti) * (1.0 - tj) * A(i, j) + ti * (1.0 - tj) * A(i + 1, j) +
           (1.0 - ti) * tj * A(i, j + 1) + ti * tj * A(i + 1, j + 1);
  }
  KOKKOS_INLINE_FUNCTION Real KappaP(const Real rho_code, const Real T_code,
                                     const bool probe = false) const {
    if (log_interp_)
      return std::exp(bilin(lkP_, std::log10(rho_code * rho_unit), std::log10(T_code * T_unit),
                            probe ? kNClampState : 0));
    return bilin(kP_, std::log10(rho_code * rho_unit), std::log10(T_code * T_unit),
                 probe ? kNClampState : 0);
  }
  KOKKOS_INLINE_FUNCTION Real KappaR(const Real rho_code, const Real T_code) const {
    if (log_interp_)
      return std::exp(bilin(lkR_, std::log10(rho_code * rho_unit), std::log10(T_code * T_unit)));
    return bilin(kR_, std::log10(rho_code * rho_unit), std::log10(T_code * T_unit));
  }
  //! R-F5: build the log(kappa) copies (host, once at init) and switch the P/R lookups to them.
  void EnableLogInterp() {
    auto mk = [&](const ParArray2D<Real> &src, ParArray2D<Real> &dst, const char *name) {
      dst = ParArray2D<Real>(name, nr_, nT_);
      auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src);
      auto hd = Kokkos::create_mirror_view(dst);
      for (int i = 0; i < nr_; ++i)
        for (int j = 0; j < nT_; ++j) hd(i, j) = std::log(std::max(hs(i, j), Real(1.0e-300)));
      Kokkos::deep_copy(dst, hd);
    };
    mk(kP_, lkP_, "opac_lkP");
    mk(kR_, lkR_, "opac_lkR");
    log_interp_ = true;
  }
  KOKKOS_INLINE_FUNCTION Real KappaS(const Real rho_code, const Real T_code) const {
    return bilin(ks_, std::log10(rho_code * rho_unit), std::log10(T_code * T_unit));
  }

  //! R-F11 (format v4, flags bit 1): the grains' share of the gray kappa_P / kappa_R, for
  //! dust/opacity_evap = table. Plain bilinear on the table grid, edge-clamped and NOT counted (the gray
  //! lookups at the same (rho, T) already tally any clamp; counting here would double it).
  ParArray2D<Real> wdP_, wdR_;
  bool has_dust_weight_ = false;
  KOKKOS_INLINE_FUNCTION
  Real bilin_nc(const ParArray2D<Real> &A, const Real lr, const Real lT) const {
    Real fi = (lr - lr0_) / dlr_;
    int i = static_cast<int>(fi);
    if (i < 0) i = 0;
    if (i > nr_ - 2) i = nr_ - 2;
    Real ti = fi - i;
    ti = (ti < 0.0) ? 0.0 : ((ti > 1.0) ? 1.0 : ti);
    Real fj = (lT - lT0_) / dlT_;
    int j = static_cast<int>(fj);
    if (j < 0) j = 0;
    if (j > nT_ - 2) j = nT_ - 2;
    Real tj = fj - j;
    tj = (tj < 0.0) ? 0.0 : ((tj > 1.0) ? 1.0 : tj);
    return (1.0 - ti) * (1.0 - tj) * A(i, j) + ti * (1.0 - tj) * A(i + 1, j) +
           (1.0 - ti) * tj * A(i, j + 1) + ti * tj * A(i + 1, j + 1);
  }
  KOKKOS_INLINE_FUNCTION Real DustWeightP(const Real rho_code, const Real T_code) const {
    return bilin_nc(wdP_, std::log10(rho_code * rho_unit), std::log10(T_code * T_unit));
  }
  KOKKOS_INLINE_FUNCTION Real DustWeightR(const Real rho_code, const Real T_code) const {
    return bilin_nc(wdR_, std::log10(rho_code * rho_unit), std::log10(T_code * T_unit));
  }

  //! Load the gray magnitude tables (host). Header is parsed by ReadOpacityHeader (see
  //! opacity_table_format.hpp); the payload is float64 kP[nr,nT], kR, ks, then mP[ng,nT], mR
  //! (the multipliers are read separately by GroupOpacityTable).
  //!   ru, tu  = rho_unit, T_unit -- used ONLY to map code (rho,T) onto the file's cgs axes.
  //!   kappa_u = the run's opacity_unit (rho_unit*length_unit).
  //!
  //! AUDIT N13: a v2 file stores kappa in cgs and is scaled by `kappa_u` here, so it is
  //! correct for ANY normalization. A legacy v1 file stores kappa already in the generator's
  //! code units and is used verbatim -- bit-identical to the historical behaviour, and
  //! carrying the historical +0.135 % bias, which the caller reports.
  //! `used_legacy_code_units` lets the caller warn without re-parsing the file.
  bool used_legacy_code_units = false;
  void Load(const std::string &path, const Real ru, const Real tu, const Real kappa_u) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("OpacityTable: cannot open " + path);
    const auto hdr = ReadOpacityHeader(f, path);
    ng_ = hdr.ng;
    nr_ = hdr.nr;
    nT_ = hdr.nT;
    lr0_ = hdr.lr0; dlr_ = hdr.dlr; lT0_ = hdr.lT0; dlT_ = hdr.dlT;
    rho_unit = ru; T_unit = tu;
    if (nr_ < 2 || nT_ < 2 || !(dlr_ > 0.0) || !(dlT_ > 0.0))
      throw std::runtime_error("OpacityTable: bad (rho,T) grid in " + path);
    // cgs table => convert to code units with THIS run's opacity unit (the N13 fix).
    // Legacy table => already in the generator's code units; use verbatim.
    used_legacy_code_units = !hdr.kappa_in_cgs;
    const Real scale = hdr.kappa_in_cgs ? kappa_u : static_cast<Real>(1.0);
    auto load2d = [&](ParArray2D<Real> &view, const char *name) {
      view = ParArray2D<Real>(name, nr_, nT_);
      auto h = Kokkos::create_mirror_view(view);
      std::vector<double> buf(static_cast<size_t>(nr_) * nT_);
      f.read(reinterpret_cast<char *>(buf.data()),
             static_cast<std::streamsize>(buf.size() * sizeof(double)));
      for (int i = 0; i < nr_; ++i)
        for (int j = 0; j < nT_; ++j)
          h(i, j) = static_cast<Real>(buf[i * nT_ + j]) * scale;
      Kokkos::deep_copy(view, h);
    };
    load2d(kP_, "opac_kP");
    load2d(kR_, "opac_kR");
    load2d(ks_, "opac_ks");
    if (hdr.has_dust_weight) {
      // R-F11 (v4): skip the [ng, nr, nT] multipliers mP, mR (GroupOpacityTable reads them), then the weights.
      // The weights are fractions: never scaled by the opacity unit.
      f.seekg(static_cast<std::streamoff>(2) * ng_ * nr_ * nT_ * sizeof(double), std::ios::cur);
      auto loadw = [&](ParArray2D<Real> &view, const char *name) {
        view = ParArray2D<Real>(name, nr_, nT_);
        auto h = Kokkos::create_mirror_view(view);
        std::vector<double> buf(static_cast<size_t>(nr_) * nT_);
        f.read(reinterpret_cast<char *>(buf.data()),
               static_cast<std::streamsize>(buf.size() * sizeof(double)));
        for (int i = 0; i < nr_; ++i)
          for (int j = 0; j < nT_; ++j) h(i, j) = static_cast<Real>(buf[i * nT_ + j]);
        Kokkos::deep_copy(view, h);
      };
      loadw(wdP_, "opac_wdP");
      loadw(wdR_, "opac_wdR");
      has_dust_weight_ = true;
    }
    // Clamp tally, allocated ONLY on a real load so every default-constructed OpacityTable
    // keeps a null View and its kernels skip the atomic. Kokkos zero-initialises by default.
    clamp_ctr_ = Kokkos::View<unsigned long long *>("opac_clamp_ctr", kNClampCtr);
    if (!f) throw std::runtime_error("OpacityTable: truncated file " + path);
    loaded_ = true;
  }

  //! Host-side: drain the tallies and zero them for the next interval. False if never loaded.
  //! HOST only -- it fences and does a device->host copy, so call it at most once per cycle.
  bool ReadAndResetClampCounts(unsigned long long out[kNClampCtr]) const {
    if (clamp_ctr_.data() == nullptr) return false;
    auto h = Kokkos::create_mirror_view(clamp_ctr_);
    Kokkos::deep_copy(h, clamp_ctr_);
    for (int q = 0; q < kNClampCtr; ++q) out[q] = h(q);
    Kokkos::deep_copy(clamp_ctr_, 0ull);
    return true;
  }

  //! Axis extents in the table's own log10 coordinates, so a report can say WHERE the wall is.
  void AxisRanges(Real &lr_lo, Real &lr_hi, Real &lT_lo, Real &lT_hi) const {
    lr_lo = lr0_; lr_hi = lr0_ + (nr_ - 1) * dlr_;
    lT_lo = lT0_; lT_hi = lT0_ + (nT_ - 1) * dlT_;
  }
};

//----------------------------------------------------------------------------------------
//! Device-side opacity parameters (all in CODE units / conversion factors). Captured by
//! value into the matter-coupling kernel; no host pointers, safe on GPU.
struct OpacityParams {
  OpacityModel model = OpacityModel::constant;
  // constant model: code-unit opacities (per unit mass), coefficient = rho_code*kappa.
  Real kappa_a0 = 0.0;
  Real kappa_s0 = 0.0;
  // dust model (Bell & Lin 1994 ice-grain regime kappa = k0 * T_phys^2 [cm^2/g]):
  Real dust_k0_cgs = 2.0e-4; // [cm^2 g^-1 K^-2]
  Real dust_Tmax = 1.5e3;    // sublimation cap on T_phys [K] (kappa frozen above)
  Real kappa_s_dust = 0.0;   // optional gray scattering opacity (code units)
  // unit conversions:
  Real T_unit = 1.0;     // K per unit code temperature (T_phys = T_unit * T_code)
  Real rho_unit = 1.0;   // g/cm^3 per unit code density (rho_phys = rho_unit * rho_code)
  Real kappa_conv = 1.0; // kappa_code = kappa_phys[cm^2/g] * kappa_conv (= rho0*l0)
  // Planck/Rosseland split: Bell&Lin/dust laws are ROSSELAND fits; the emission (matter-
  // coupling) term should use the PLANCK mean, which for dust exceeds the Rosseland mean
  // (kappa_P/kappa_R ~ 2-5). Modeled as kappa_P = planck_ross_ratio * kappa_R. Default 1.0
  // => bit-identical to the pre-split single-opacity behavior. A full (log rho, log T) ->
  // (kappa_P, kappa_R) table (Semenov et al. 2003) is the follow-on increment.
  Real planck_ross_ratio = 1.0;
  // Regime-skip-robust Bell&Lin walk (fixes the low-rho/high-T kappa discontinuity).
  //
  // DEFAULT FLIPPED false -> true, 2026-08-08. The old comment justified `false` by saying the
  // two walks are identical "on the collapse track", quoting rho >= 1e-10 g/cm^3. That bound is
  // real but it describes the WRONG DOMAIN: this problem has rho0 = 5.467e-19 g/cm^3 and
  // rhocrit = 1e-13, so the collapse run lives at rho ~ 1e-21 .. 1e-12 -- entirely BELOW the range
  // over which the walks were checked to agree. The agreement was never evidence about this run.
  //
  // Re-measured over the domain the collapse run actually occupies (both walks evaluated directly,
  // 121x121 log grids):
  //   rho 1e-21..1e-13, T   5..100 K  (pre-first-core; the ENTIRE ensemble epoch)  0/14641 differ
  //   rho 1e-21..1e-12, T   5..2000 K (through first core)      3/14641 = 0.02 %, worst 0.09 dex
  //   rho 1e-21..1e-10, T   5..1e4 K  (into second core)     1442/14641 = 9.85 %, worst 8.59 dex
  //                                                          (plain 0.348 vs fixed 8.87e-10)
  // So `true` is BIT-IDENTICAL for everything measured to date, and prevents an up-to-8.6-decade
  // kappa error the moment a run gets hot at low density -- which is precisely where the collapse run
  // is headed. Making the correct branch the default is therefore free today and load-bearing
  // later; leaving it opt-in meant 148 belllin decks and 0 of them setting it.
  //
  // Canonical values are preserved by the fixed walk: 0.02 @ 10 K, 2.0 @ 100 K, 0.348 e-scattering.
  // The tabulated model is unaffected either way -- gen_opacity_table.py already bakes in the
  // fixed walk, so `opacity_model = tabulated` (the collapse run) never used the plain one.
  // Set <radiation> bell_lin_fix_regime_skip = false to reproduce the historical behaviour.
  bool bell_lin_fix_regime_skip = true;
  //! The two flags below select the dust/gas split of the research code's dust package (not part of
  //! this repository; they stay false here). R-F8: <dust> opacity_evap = pollack -> the split uses the table's Pollack evaporation
  //! coefficients (gen_opacity_table.py --evap pollack). Default false = Bell & Lin 1994 (historical).
  bool dust_evap_pollack = false;
  //! R-F11: <dust> opacity_evap = table -> the split uses the grain share stored in a format-v4 table
  //! (gen_opacity_table.py --model semenov), separately for kappa_P and kappa_R. Default false.
  bool dust_evap_table = false;
  // model == tabulated: gray kappa_P/kappa_R/kappa_s come from this device table (frequency-
  // resolved dust+gas physics precomputed offline). Inactive for the other models (View handles
  // default-constructed, never dereferenced). This gives a SELF-CONSISTENT Planck mean (not the
  // planck_ross_ratio fudge) and a real Rosseland mean.
  OpacityTable table;
};

//----------------------------------------------------------------------------------------
//! Bell & Lin (1994) Rosseland-mean gray opacity [cm^2/g] (rho,T in cgs). Eight piecewise
//! power-law regimes kappa = k0 * rho^a * T^b: ice grains, ice evaporation, metal/dust
//! grains, DUST SUBLIMATION (the opacity gap ~1500-2000 K where kappa plunges T^-24),
//! molecular, H- (negative hydrogen ion), bound-free+free-free (Kramers), and electron
//! scattering. This is the physical opacity through the SECOND-core regime (T>1500 K),
//! replacing the frozen-dust cap. Regime is chosen by walking up the density-dependent
//! transition temperatures (where adjacent regimes' opacities cross).
KOKKOS_INLINE_FUNCTION
Real BellLinKappa(const Real rho, const Real T) {
  constexpr Real k0[8] = {2.0e-4, 2.0e16, 0.1, 2.0e81, 1.0e-8, 1.0e-36, 1.5e20, 0.348};
  constexpr Real aa[8] = {0.0, 0.0, 0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0};
  constexpr Real bb[8] = {2.0, -7.0, 0.5, -24.0, 3.0, 10.0, -2.5, 0.0};
  int i = 0;
  for (; i < 7; ++i) {
    // transition T where regime i meets i+1: k0[i] rho^aa[i] T^bb[i] = k0[i+1] rho^.. T^..
    const Real Tt = std::pow((k0[i] / k0[i + 1]) * std::pow(rho, aa[i] - aa[i + 1]),
                             1.0 / (bb[i + 1] - bb[i]));
    if (T < Tt) break;
  }
  return k0[i] * std::pow(rho, aa[i]) * std::pow(T, bb[i]);
}

//----------------------------------------------------------------------------------------
//! Regime-skip-robust Bell & Lin walk. The plain BellLinKappa walk assumes the adjacent-regime
//! transition temperatures are encountered in order; at low rho (<~1e-11) they go OUT OF ORDER
//! (the Kramers regime's window closes) and the walk jumps across a skipped regime, giving a
//! kappa DISCONTINUITY of up to ~4 decades at T~4-9 kK. This version skips any regime whose
//! forward window has closed [cross(j,j+1) <= cross(i,j)] and bridges regime i directly to the
//! next ACTIVE regime, so kappa is CONTINUOUS everywhere. VERIFIED: identical to BellLinKappa
//! on the collapse track (rho>=1e-10, rel diff 0) and gives the correct canonical values
//! (0.02 @ 10 K, 2.0 @ 100 K, 0.348 e-scatter); only the off-track corner changes.
KOKKOS_INLINE_FUNCTION
Real BellLinKappaFixed(const Real rho, const Real T) {
  constexpr Real k0[8] = {2.0e-4, 2.0e16, 0.1, 2.0e81, 1.0e-8, 1.0e-36, 1.5e20, 0.348};
  constexpr Real aa[8] = {0.0, 0.0, 0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 0.0};
  constexpr Real bb[8] = {2.0, -7.0, 0.5, -24.0, 3.0, 10.0, -2.5, 0.0};
  auto Tcross = [&](int p, int q) { // T where regime p == regime q at this rho
    return std::pow((k0[p] / k0[q]) * std::pow(rho, aa[p] - aa[q]), 1.0 / (bb[q] - bb[p]));
  };
  int i = 0;
  while (i < 7) {
    int j = i + 1;
    while (j < 7 && Tcross(j, j + 1) <= Tcross(i, j)) ++j; // skip closed-window regimes
    if (T < Tcross(i, j)) return k0[i] * std::pow(rho, aa[i]) * std::pow(T, bb[i]);
    i = j;
  }
  return k0[7] * std::pow(rho, aa[7]) * std::pow(T, bb[7]);
}


//----------------------------------------------------------------------------------------
//! ROSSELAND-mean absorption opacity in CODE units (per unit mass), kappa_R(rho,T). Used
//! for flux attenuation / the diffusion limit. Bell & Lin is itself a Rosseland fit, so
//! this is the base opacity; the coefficient in the flux term is rho_code * (this + scat).
KOKKOS_INLINE_FUNCTION
Real RosselandOpacity(const OpacityParams &op, const Real rho_code, const Real T_code) {
  if (op.model == OpacityModel::constant) return op.kappa_a0;
  if (op.model == OpacityModel::tabulated) return op.table.KappaR(rho_code, T_code);
  const Real T_phys = op.T_unit * T_code;
  if (op.model == OpacityModel::belllin) {
    const Real rho_phys = op.rho_unit * rho_code;
    const Real kap = op.bell_lin_fix_regime_skip ? BellLinKappaFixed(rho_phys, T_phys)
                                                 : BellLinKappa(rho_phys, T_phys);
    return kap * op.kappa_conv;
  }
  // dust: Bell & Lin low-T ice-grain law only, frozen above sublimation.
  const Real Td = std::min(T_phys, op.dust_Tmax);
  const Real kappa_phys = op.dust_k0_cgs * Td * Td; // [cm^2/g]
  return kappa_phys * op.kappa_conv;
}

//----------------------------------------------------------------------------------------
//! PLANCK-mean absorption opacity in CODE units (per unit mass), kappa_P(rho,T). Used for
//! the emission/absorption matter coupling S_E = chat*rho*kappa_P*(E-B). Modeled as
//! kappa_P = planck_ross_ratio * kappa_R (ratio=1 -> identical to kappa_R). See
//! OpacityParams::planck_ross_ratio.
KOKKOS_INLINE_FUNCTION
Real PlanckOpacity(const OpacityParams &op, const Real rho_code, const Real T_code,
                   const bool probe = false) {
  // Tabulated: TRUE Planck mean (self-consistent, not planck_ross_ratio * kappa_R).
  // `probe` marks a root-finder TRIAL temperature; it only selects which counter bank the
  // edge-clamp tally lands in and cannot change the returned opacity.
  if (op.model == OpacityModel::tabulated) return op.table.KappaP(rho_code, T_code, probe);
  return op.planck_ross_ratio * RosselandOpacity(op, rho_code, T_code);
}

//----------------------------------------------------------------------------------------
//! Gray SCATTERING opacity in CODE units (per unit mass). Constant for both models.
KOKKOS_INLINE_FUNCTION
Real ScatteringOpacity(const OpacityParams &op, const Real rho_code, const Real T_code) {
  if (op.model == OpacityModel::tabulated) return op.table.KappaS(rho_code, T_code);
  return (op.model == OpacityModel::constant) ? op.kappa_s0 : op.kappa_s_dust;
}

} // namespace Radiation

#endif // RADIATION_RADIATION_OPACITY_HPP_
