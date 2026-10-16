#ifndef EOS_ADIABATIC_GLMMHD_HPP_
#define EOS_ADIABATIC_GLMMHD_HPP_
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers
#include <limits> // std::numeric_limits<float>

// Parthenon headers
#include "mesh/mesh.hpp"

// Athena headers
#include "../main.hpp"
#include "eos.hpp"
#include "eos_table.hpp"

using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshBlockVarPack;
using parthenon::Real;

class AdiabaticGLMMHDEOS : public EquationOfState {
 public:
  // Optional tabulated protostellar EOS (second-core physics) is selected by
  // use_h2diss=true; then pressure/sound-speed/internal-energy come from the interpolated
  // table (eos_table.hpp). When use_h2diss=false the class is bit-identical to the ideal
  // gamma-law EOS (all table branches compile out at runtime; defaults keep old constructor
  // call sites valid).
  AdiabaticGLMMHDEOS(Real pressure_floor, Real density_floor, Real internal_e_floor,
                     Real velocity_ceiling, Real internal_e_ceiling, Real gamma,
                     bool use_h2diss = false,
                     EOSTable::EosTable eos_tab = EOSTable::EosTable(),
                     int dual_energy_idx = -1, Real dual_energy_eta1 = 1.0e-2,
                     bool floor_consistent = false, bool dual_unsigned = false,
                     bool dual_exact = false, Real dual_eta2 = 0.0, Real dual_kappa = 3.0,
                     Real dual_shock = 0.02, bool dual_shock_eta1 = false)
      : EquationOfState(pressure_floor, density_floor, internal_e_floor, velocity_ceiling,
                        internal_e_ceiling),
        gamma_{gamma}, use_h2diss_{use_h2diss}, eos_tab_{eos_tab},
        dual_idx_{dual_energy_idx}, dual_eta1_{dual_energy_eta1},
        floor_consistent_{floor_consistent}, dual_unsigned_{dual_unsigned},
        dual_exact_{dual_exact}, dual_eta2_{dual_eta2}, dual_kappa_{dual_kappa},
        dual_shock_{dual_shock}, dual_shock_eta1_{dual_shock_eta1} {}

  void ConservedToPrimitive(MeshData<Real> *md) const override;

  KOKKOS_INLINE_FUNCTION
  Real GetGamma() const { return gamma_; }

  KOKKOS_INLINE_FUNCTION
  bool UseH2Diss() const { return use_h2diss_; }

  // Internal-energy DENSITY from (rho, pressure). Ideal: P/(gamma-1); H2-diss: Saha
  // inversion. Used by the Riemann solvers to reconstruct the L/R total energy so the
  // energy flux stays consistent with the (general) EOS.
  KOKKOS_INLINE_FUNCTION
  Real InternalEnergyFromPres(const Real rho, const Real pres) const {
    if (use_h2diss_) return eos_tab_.EintFromRhoPres(rho, pres);
    return pres / (gamma_ - 1.0);
  }

  // Accessor so other code (diagnostics, double flux, problem generators) can share the same table.
  KOKKOS_INLINE_FUNCTION
  const EOSTable::EosTable &GetEosTable() const { return eos_tab_; }

#ifndef ATHENAPK_EOS_LEGACY_LOOKUPS
  //! Tabulated EOS: the temperature floor / ceiling in KELVIN (hydro/Tfloor, hydro/Tceil in code temperature units
  //! times temperature_unit), set by hydro.cpp after construction; <= 0 = off. With them the floor and ceiling
  //! energies are taken from the table AT EACH CELL'S DENSITY (EOS_VALIDATION.md O-3, O-5), not from one specific
  //! energy evaluated at rho_code = 1, which is exact only where e(T) does not depend on density (cold gas).
  void SetTableTemperatureLimits(const Real Tfloor_K, const Real Tceil_K) {
    Tfloor_K_ = Tfloor_K;
    Tceil_K_ = Tceil_K;
  }
  //! Internal-energy DENSITY of the temperature floor at density rho (ideal gas: rho * e_floor, as before).
  KOKKOS_INLINE_FUNCTION
  Real EintFloor(const Real rho) const {
    return (use_h2diss_ && Tfloor_K_ > 0.0) ? eos_tab_.EintFromRhoTk(rho, Tfloor_K_)
                                            : rho * GetInternalEFloor();
  }
  //! Internal-energy DENSITY of the temperature ceiling at density rho (ideal gas: rho * e_ceiling, as before).
  KOKKOS_INLINE_FUNCTION
  Real EintCeil(const Real rho) const {
    return (use_h2diss_ && Tceil_K_ > 0.0) ? eos_tab_.EintFromRhoTk(rho, Tceil_K_)
                                           : rho * GetInternalECeiling();
  }
#endif

  //----------------------------------------------------------------------------------------
  // \!fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
  // \brief returns adiabatic sound speed given vector of primitive variables
  // TODO(pgrete): need to fix idx defs
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(const Real prim[NHYDRO]) const {
    if (use_h2diss_)
      return std::sqrt(eos_tab_.AsqFromRhoPres(prim[IDN], prim[IPR]));
    return std::sqrt(gamma_ * prim[IPR] / prim[IDN]);
  }
  // fast magnetosonic speed function for adiabatic EOS
  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(const Real d, const Real p, const Real bx, const Real by,
                             const Real bz) const {
    // asq here is rho*c_s^2 (= gamma*p for an ideal gas); use the general Saha c_s^2 when
    // the H2-dissociation EOS is active.
    Real asq = use_h2diss_ ? d * eos_tab_.AsqFromRhoPres(d, p) : gamma_ * p;
    const Real bx2 = bx * bx;
    Real ct2 = by * by + bz * bz;
    Real qsq = bx2 + ct2 + asq;
    Real tmp = bx2 + ct2 - asq;
    return std::sqrt(0.5 * (qsq + std::sqrt(tmp * tmp + 4.0 * asq * ct2)) / d);
  }
  //

  //----------------------------------------------------------------------------------------
  // \!fn Real EquationOfState::ConsToPrim(View4D cons, View4D prim, const int& k, const
  // int& j, const int& i) \brief Fills an array of primitives given an array of
  // conserveds, potentially updating the conserved with floors
  template <typename View4D>
  // Returns whether the cell took the advected internal energy (dual-energy switch).
  KOKKOS_INLINE_FUNCTION bool ConsToPrim(View4D cons, View4D prim, const int &nhydro,
                                         const int &nscalars, const int &k, const int &j,
                                         const int &i, const Real comp = 0.0) const {
    auto gam = GetGamma();
    auto gm1 = gam - 1.0;
    auto density_floor_ = GetDensityFloor();
    auto pressure_floor_ = GetPressureFloor();
    auto e_floor_ = GetInternalEFloor();

    auto velocity_ceiling_ = GetVelocityCeiling();
    auto e_ceiling_ = GetInternalECeiling();

    Real &u_d = cons(IDN, k, j, i);
    Real &u_m1 = cons(IM1, k, j, i);
    Real &u_m2 = cons(IM2, k, j, i);
    Real &u_m3 = cons(IM3, k, j, i);
    Real &u_e = cons(IEN, k, j, i);
    Real &u_b1 = cons(IB1, k, j, i);
    Real &u_b2 = cons(IB2, k, j, i);
    Real &u_b3 = cons(IB3, k, j, i);
    Real &u_psi = cons(IPS, k, j, i);

    Real &w_d = prim(IDN, k, j, i);
    Real &w_vx = prim(IV1, k, j, i);
    Real &w_vy = prim(IV2, k, j, i);
    Real &w_vz = prim(IV3, k, j, i);
    Real &w_p = prim(IPR, k, j, i);
    Real &w_Bx = prim(IB1, k, j, i);
    Real &w_By = prim(IB2, k, j, i);
    Real &w_Bz = prim(IB3, k, j, i);
    Real &w_psi = prim(IPS, k, j, i);

    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative density is encountered.
    PARTHENON_REQUIRE(u_d > 0.0 || density_floor_ > 0.0,
                      "Got negative density. Consider enabling first-order flux "
                      "correction or setting a reasonble density floor.");
    // apply density floor, without changing momentum or energy
    u_d = (u_d > density_floor_) ? u_d : density_floor_;
    w_d = u_d;

    Real di = 1.0 / u_d;
    w_vx = u_m1 * di;
    w_vy = u_m2 * di;
    w_vz = u_m3 * di;

    w_Bx = u_b1;
    w_By = u_b2;
    w_Bz = u_b3;
    w_psi = u_psi;

    Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    Real e_B = 0.5 * (SQR(u_b1) + SQR(u_b2) + SQR(u_b3));
    Real e_int = u_e - e_k - e_B;

    // DUAL ENERGY (hydro/dual_energy). e_int = E - KE - ME is a small difference wherever the
    // magnetic or kinetic energy dominates: in low-beta gas it goes to zero or negative and the
    // pressure floor then injects energy (measured: up to ~2e4 floored cells per step in a
    // magnetized collapse). When that happens, take the separately advected internal energy
    // instead and make the total energy consistent with it; otherwise keep the conservative
    // value and re-sync the advected one. dual_idx_ < 0 (default) compiles this out at runtime
    // and the run is bit-identical to the single-energy scheme.
    bool dual_switched = false;
    Real dual_keep = 0.0;         // the advected e_int a switched cell used (dual_exact_)
    bool energy_rewritten = false; // a floor/ceiling below rewrote u_e after the switch
    if (dual_idx_ >= 0) {
      Real &u_ie = cons(dual_idx_, k, j, i);
      // hydro/dual_energy_eta2 > 0 (default 0 = off): the switch above is ONE-SIDED -- it catches an
      // E - KE - ME that is too SMALL, not one that is wrong by being too LARGE. In low-beta cells the
      // VL2 predictor's E - KE - ME swings by +-1e5 around a true ~1e3-1e4 (measured in a cell
      // trace, 2026-09-23); a positive swing passed eta1, overwrote e_dual and inflated P 11x. So ALSO take
      // the advected value when the cell is thermally negligible by its own advected estimate
      // (u_ie < eta2*E) and E - KE - ME exceeds it by more than kappa -- UNLESS the cell compressed
      // by more than dual_shock this stage (comp = -div v * dt from DualEnergyPdV): shock heating
      // is real, never reaches the advected energy, and must keep coming from E.
      const bool distrust_high = (dual_eta2_ > 0.0) && (u_ie > 0.0) && (u_ie < dual_eta2_ * u_e) &&
                                 (e_int > dual_kappa_ * u_ie) && (comp < dual_shock_);
      // hydro/dual_energy_shock_eta1 (validation_tests/dual_energy F2, 2026-10-07, default false): a cell compressing
      // by more than shock_comp this stage keeps E also when e_int <= eta1 E, unless it is magnetically dominated
      // (e_int <= eta1 (e_int + e_B): there E - KE - ME is noise even in a shock). Without it, shock-front cells of a
      // kinetically dominated flow take the cold advected energy and the shock under-heats as eta1 grows (D6: Noh
      // M 10 plateau -5.2e-3 / -1.7e-2 at eta1 0.1 / 0.2).
      const bool shock_keep =
          dual_shock_eta1_ && (comp >= dual_shock_) && (e_int > dual_eta1_ * (e_int + e_B));
      if ((e_int > dual_eta1_ * u_e || shock_keep) && !distrust_high) {
        u_ie = e_int; // total energy is trustworthy here: it stays authoritative
      } else {
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
        e_int = (u_ie > 0.0) ? u_ie : e_floor_ * u_d;
#else
        e_int = (u_ie > 0.0) ? u_ie : EintFloor(u_d);
#endif
        dual_keep = e_int;
        u_e = e_int + e_k + e_B; // local, bounded non-conservation, counted by cons_diag
        dual_switched = true;
        // hydro/dual_energy_exact_resync: settle on the state a SECOND pass would reach. The
        // rewritten cell re-evaluates with e2 = u_e - e_k - e_B (= u_ie to ~1e-14); if that is now
        // above the threshold, a second pass takes the conservative branch with e_int = e2, so
        // take it here already. Otherwise the second pass switches again, reads back dual_keep
        // (stored exactly, see the re-sync below) and rebuilds the identical u_e.
        if (dual_exact_) {
          const Real e2 = u_e - e_k - e_B;
          if (e2 > dual_eta1_ * u_e) {
            e_int = e2;
            dual_switched = false;
          }
        }
      }
    }
    // Table lookup interpolates in log10(e_int/rho): for e_int <= 0 it would return NaN,
    // and NaN compares false against every floor below (silently poisoning the cell).
    // Hand non-positive e_int to the ideal-gamma expression instead so w_p goes negative
    // and the existing PARTHENON_REQUIRE / pressure-floor logic catches it.
    // Audit F-21: this lookup runs before the floors and over ghosts too, so a transiently sub-floor
    // e_int (low-density tail, prolongated coarse-fine ghosts) is counted as an `esp<` clamp and then
    // floored away -- such clamps are not evidence the table is too short; compare with cons-npfloor.
    w_p = (use_h2diss_ && e_int > 0.0) ? eos_tab_.PresFromRhoEint(u_d, e_int)
                                       : gm1 * e_int;

    // apply velocity ceiling. By default ceiling is std::numeric_limits<Real>::infinity()
    const Real w_v2 = SQR(w_vx) + SQR(w_vy) + SQR(w_vz);
    if (w_v2 > SQR(velocity_ceiling_)) {
      const Real w_v = sqrt(w_v2);
      w_vx *= velocity_ceiling_ / w_v;
      w_vy *= velocity_ceiling_ / w_v;
      w_vz *= velocity_ceiling_ / w_v;

      u_m1 *= velocity_ceiling_ / w_v;
      u_m2 *= velocity_ceiling_ / w_v;
      u_m3 *= velocity_ceiling_ / w_v;

      Real e_k_new = 0.5 * u_d * SQR(velocity_ceiling_);
      u_e -= e_k - e_k_new;
      energy_rewritten = true;
      e_k = e_k_new;
    }

    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative pressure is encountered.
    PARTHENON_REQUIRE(w_p > 0.0 || pressure_floor_ > 0.0 || e_floor_ > 0.0,
                      "Got negative pressure. Consider enabling first-order flux "
                      "correction or setting a reasonble pressure or temperature floor.");

    // Pressure floor (if present) takes precedence over temperature floor
    if ((pressure_floor_ > 0.0) && (w_p < pressure_floor_)) {
      // apply pressure floor, correct total energy. InternalEnergyFromPres keeps the
      // e<->P mapping consistent with the active EOS (ideal: pressure_floor/gm1,
      // bit-identical to the old expression; table: Saha inversion).
      const Real e_fl = InternalEnergyFromPres(u_d, pressure_floor_);
      u_e = e_fl + e_k + e_B;
      energy_rewritten = true;
      // hydro/eos_floor_consistent (2026-09-22, default false). Under the tabulated EOS the
      // inversion CLAMPS at the table's bottom esp edge, where the pressure OF the stored energy is
      // far above pressure_floor_ (measured: e = 38.4 -> P = 25.6 against a 1e-8 floor). Writing
      // w_p = pressure_floor_ then leaves prim inconsistent with cons for the rest of the stage, and
      // makes ConsToPrim non-idempotent (a second pass returns 25.6) -- a restart asymmetry. With
      // the key on, the pressure is that of the energy actually stored. Ideal gas: the two are
      // exact inverses, so the branch is unchanged.
      w_p = (floor_consistent_ && use_h2diss_) ? eos_tab_.PresFromRhoEint(u_d, e_fl)
                                               : pressure_floor_;
    }

    // temperature (internal energy) based pressure floor. Under the tabulated EOS the
    // pressure corresponding to the specific-internal-energy floor comes from the table
    // (the gm1 conversion would set w_p inconsistent with the u_e written below); the
    // ideal-gas branch is unchanged.
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
    const Real eff_pressure_floor = (use_h2diss_ && e_floor_ > 0.0)
                                        ? eos_tab_.PresFromRhoEint(u_d, u_d * e_floor_)
                                        : gm1 * u_d * e_floor_;
    if (w_p < eff_pressure_floor) {
      // apply temperature floor, correct total energy
      u_e = (u_d * e_floor_) + e_k + e_B;
      energy_rewritten = true;
      w_p = eff_pressure_floor;
    }
#else
    // Tabulated EOS: the floor energy is e(rho, Tfloor) at THIS cell's density (O-5); ideal gas: rho * e_floor.
    const Real eint_floor = EintFloor(u_d);
    const Real eff_pressure_floor = (use_h2diss_ && e_floor_ > 0.0)
                                        ? eos_tab_.PresFromRhoEint(u_d, eint_floor)
                                        : gm1 * u_d * e_floor_;
    if (w_p < eff_pressure_floor) {
      // apply temperature floor, correct total energy
      u_e = eint_floor + e_k + e_B;
      energy_rewritten = true;
      w_p = eff_pressure_floor;
    }
#endif

    // temperature (internal energy) based pressure ceiling (guard the table against the
    // default e_ceiling = infinity: log10(inf) would clamp to the table edge and
    // spuriously trigger the ceiling).
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
    const Real eff_pressure_ceiling =
        (use_h2diss_ && e_ceiling_ < std::numeric_limits<Real>::infinity())
            ? eos_tab_.PresFromRhoEint(u_d, u_d * e_ceiling_)
            : gm1 * u_d * e_ceiling_;
    if (w_p > eff_pressure_ceiling) {
      // apply temperature ceiling, correct total energy
      u_e = (u_d * e_ceiling_) + e_k + e_B;
      energy_rewritten = true;
      w_p = eff_pressure_ceiling;
    }
#else
    // Tabulated EOS: the ceiling energy is e(rho, Tceil) at THIS cell's density (O-3/O-5); ideal gas as before.
    const bool ceil_on = e_ceiling_ < std::numeric_limits<Real>::infinity();
    const Real eint_ceil = (use_h2diss_ && ceil_on) ? EintCeil(u_d) : u_d * e_ceiling_;
    const Real eff_pressure_ceiling = (use_h2diss_ && ceil_on) ? eos_tab_.PresFromRhoEint(u_d, eint_ceil)
                                                               : gm1 * u_d * e_ceiling_;
    if (w_p > eff_pressure_ceiling) {
      // apply temperature ceiling, correct total energy
      u_e = eint_ceil + e_k + e_B;
      energy_rewritten = true;
      w_p = eff_pressure_ceiling;
    }
#endif

    // Dual energy: the floors/ceilings above may have rewritten u_e; keep the advected internal
    // energy consistent with what the cell ended up with.
    if (dual_idx_ >= 0) {
      // hydro/dual_energy_exact_resync (default false = legacy). A SWITCHED cell computed its
      // pressure from dual_keep = u_ie and set u_e = u_ie + e_k + e_B. Re-deriving the advected
      // energy as u_e - e_k - e_B does NOT return u_ie: with KE+ME ~99.5 % of E the subtraction
      // loses ~1e-14 relative, so the next ConsToPrim evaluates P at a different e_int. That made
      // ConsToPrim non-idempotent (period-2 oscillation) and was the last source of restart
      // non-reproducibility: a few switched cells grew into a whole-mesh difference within one step
      // (found by per-stage state hashing, 2026-09-22). With the key on, a switched cell
      // that no floor/ceiling touched keeps u_ie itself, so P and the stored energy come from the
      // same number and a second pass reproduces the first exactly.
      cons(dual_idx_, k, j, i) = (dual_exact_ && dual_switched && !energy_rewritten)
                                     ? dual_keep
                                     : u_e - e_k - e_B;
      // The primitive carries the specific internal energy. LEGACY (dual_unsigned_ = false): with
      // a NEGATIVE SIGN when this cell took the advected value, as a flag for cons-ndual. That
      // comment used to say "nothing reads this component except the diagnostics" -- WRONG: the
      // hydro flux advects the dual component like a passive scalar, flux = mass flux times the
      // RECONSTRUCTED prim (hydro.cpp "Passive scalar fluxes"), so every switched cell exported
      // NEGATIVE internal energy and PLM took slopes between +e and -e. It was also a restart
      // seed: a cell switched by the last ConsToPrim of a step is usually NOT switched when a
      // resumed run re-derives prim from the same cons (E was rewritten), so the sign -- and
      // with it the next step's dual flux -- differed (2026-09-22).
      // hydro/dual_energy_unsigned_prim = true stores the magnitude only; the flag goes to the
      // separate `dual_switched` field (written by ConservedToPrimitive from the return value).
      prim(dual_idx_, k, j, i) =
          ((dual_switched && !dual_unsigned_) ? -1.0 : 1.0) * cons(dual_idx_, k, j, i) * di;
    }

    // Convert passive scalars
    for (auto n = nhydro; n < nhydro + nscalars; ++n) {
      prim(n, k, j, i) = cons(n, k, j, i) * di;
    }
    return dual_switched;
  }

  bool DualUnsigned() const { return dual_unsigned_; }

  //! The internal energy ConsToPrim's dual-energy switch would select, WITHOUT touching any state.
  //! For operator-split source terms that act on `cons` between a hydro update and the next
  //! ConsToPrim and need the gas internal energy: they must use the same two-sided rule as
  //! ConsToPrim, or writing their result back into the dual energy would disable the eta2 test.
  //! (No such source term is part of this repository.) Same branches as ConsToPrim above (the
  //! exact-resync refinement is left to the ConsToPrim that follows). `comp` is the stage
  //! compression (`dual_comp`), 0 when eta2 is off. `e_B` = magnetic energy density, used only by the
  //! hydro/dual_energy_shock_eta1 exemption (as in ConsToPrim; 2026-10-07).
  KOKKOS_INLINE_FUNCTION
  Real DualSelectEint(const Real e_E, const Real E, const Real u_ie, const Real rho,
                      const Real comp, const Real e_B) const {
    const bool distrust_high = (dual_eta2_ > 0.0) && (u_ie > 0.0) && (u_ie < dual_eta2_ * E) &&
                               (e_E > dual_kappa_ * u_ie) && (comp < dual_shock_);
    const bool shock_keep =
        dual_shock_eta1_ && (comp >= dual_shock_) && (e_E > dual_eta1_ * (e_E + e_B));
    if ((e_E > dual_eta1_ * E || shock_keep) && !distrust_high) return e_E;
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
    return (u_ie > 0.0) ? u_ie : GetInternalEFloor() * rho;
#else
    return (u_ie > 0.0) ? u_ie : EintFloor(rho);
#endif
  }

 private:
  int dual_idx_ = -1;         // cons/prim component holding rho*e_int (dual energy), -1 = off
  Real dual_eta1_ = 1.0e-2;   // switch threshold on e_int/E. Measured 2026-09-23 with all fixes on
                              // (production restart, 10 cycles): 5e-2 leaves 1 cell < 8 K, 1e-2 leaves 41; the
                              // 09-19 "5e-2 is worse than off" was the signed-prim advection bug.
  bool floor_consistent_ = false; // hydro/eos_floor_consistent, see the pressure-floor branch
  bool dual_unsigned_ = false;    // hydro/dual_energy_unsigned_prim, see the dual re-sync in ConsToPrim
  bool dual_exact_ = false;       // hydro/dual_energy_exact_resync, idem
  Real dual_eta2_ = 0.0;          // hydro/dual_energy_eta2: two-sided switch (0 = off)
  Real dual_kappa_ = 3.0;         // hydro/dual_energy_kappa
  Real dual_shock_ = 0.02;        // hydro/dual_energy_shock_comp: -div v dt above this = shock
  bool dual_shock_eta1_ = false;  // hydro/dual_energy_shock_eta1: the shock exemption also from eta1 (F2)
#ifndef ATHENAPK_EOS_LEGACY_LOOKUPS
  Real Tfloor_K_ = -1.0; // tabulated EOS: temperature floor [K], <= 0 = off (SetTableTemperatureLimits)
  Real Tceil_K_ = -1.0;  // tabulated EOS: temperature ceiling [K], <= 0 = off
#endif
public:
  Real DualEta2() const { return dual_eta2_; }
private:
  Real gamma_; // ratio of specific heats
  bool use_h2diss_;            // enable tabulated protostellar EOS (second-core physics)
  EOSTable::EosTable eos_tab_; // interpolated table (View handles; device-copyable)
};

#endif // EOS_ADIABATIC_GLMMHD_HPP_
