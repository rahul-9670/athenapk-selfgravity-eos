//========================================================================================
// AthenaPK - radiation package registration: units, opacities (gray + per-group tables), the
// EOS table copy, coupling temperature brackets, the rad.Er/rad.Fr moment fields (Sn
// writeback), the rslaMax diagnostic, and Sn::Initialize. Discrete ordinates is the only
// transport (the M1 / FLD / thermal-M1 transports were removed 2026-09-24).
// Task assembly: radiation_tasks.cpp; transport: radiation_sn.cpp.
//========================================================================================

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../eos/eos_table.hpp"
#include "../units/physical_units.hpp" // Phase 1: one unit system (audit #4)
#include "radiation.hpp"
#include "radiation_sn.hpp"
#include "radiation_groups.hpp"
#include "radiation_opacity.hpp"

namespace Radiation {

using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("radiation");
  const std::string bn = "radiation";

  // --- Code-unit calibration (no free knobs; from the single PhysicalUnits system) ---
  //   e_unit = rho_unit v_unit^2,  T_unit = mu_thermal m_H v_unit^2 / k_B  (mu_thermal 2.29 -> 10.015 K)
  //   arad_code = a_R T_unit^4 / e_unit,  c_code = c / v_unit (~1.58e6),  kappa_conv = rho_unit l_unit
  // mu_thermal (the T unit) is deliberately not the neutral mu_n = 2.33 used for number densities;
  // the old radiation-local mu = 2.33 override (T0 = 10.19 K) is structurally impossible now.
  const auto U = PhysUnits::BuildPhysicalUnits(pin);
  const Real rho_unit = U.rho_unit;   // g/cm^3
  const Real v_unit = U.v_unit;       // cm/s
  const Real l_unit = U.length_unit;  // cm
  const Real T_unit = U.temperature_unit; // K
  const Real arad_default = U.arad_code();
  const Real c_default = U.c_code();
  const Real kappa_conv = U.opacity_unit;
  // A legacy <radiation> mu is now IGNORED (the T-unit is single-sourced). Warn loudly so a
  // deck carrying the old mu=2.33 override is not silently misread as still active.
  if (pin->DoesParameterExist(bn, "mu")) {
    const Real mu_legacy = pin->GetReal(bn, "mu");
    if (parthenon::Globals::my_rank == 0 &&
        std::abs(mu_legacy - U.mu_thermal) > 1.0e-6 * U.mu_thermal) {
      std::cout << "### WARNING Radiation: <radiation> mu=" << mu_legacy
                << " is IGNORED. The radiation temperature unit is now single-sourced from "
                   "PhysicalUnits (mu_thermal=" << U.mu_thermal << ", T0=" << T_unit
                << " K). Set <units> mu_thermal to change it." << std::endl;
    }
  }

  // AUDIT 2026-08-05 (N4): <radiation> rho_unit_cgs / v_unit_cgs / length_unit_cgs are DEAD
  // KEYS and have been since the Phase-1 unit consolidation -- the base scales come from
  // PhysUnits::BuildPhysicalUnits, which reads block <units> (or derives them from the BE IC).
  // 165 of the 295 decks in the tree still carry them, where they read as authoritative
  // (counted 2026-08-05; all 165 occurrences are in <radiation>). Worse, the
  // hard-fail that protects the BE normalisation only inspects <units>, so writing the
  // override in <radiation> gets SILENCE rather than the intended error. Warn instead of
  // failing: the values in every existing deck happen to agree with the derived ones to
  // rounding, and hard-failing would break decks belonging to queued jobs.
  if (parthenon::Globals::my_rank == 0) {
    const char *keys[3] = {"rho_unit_cgs", "v_unit_cgs", "length_unit_cgs"};
    const Real derived[3] = {rho_unit, v_unit, l_unit};
    for (int q = 0; q < 3; ++q) {
      if (!pin->DoesParameterExist(bn, keys[q])) continue;
      const Real given = pin->GetReal(bn, keys[q]);
      std::cout << "### WARNING Radiation: <radiation> " << keys[q] << " = " << given
                << " is IGNORED (dead key). The base scales are single-sourced from "
                   "PhysicalUnits; this run uses "
                << derived[q] << " (relative difference "
                << (derived[q] != 0.0 ? (given - derived[q]) / derived[q] : 0.0)
                << "). Remove it from the deck, or set it in <units> -- where, for a "
                   "collapse_be problem, it is correctly REJECTED rather than ignored."
                << std::endl;
    }
  }

  //   c      : speed of light  (code units; default = calibrated c_light/v_unit)
  //   arad   : radiation constant a_R (so E_eq = arad * T^4; default = calibrated)
  //   creduc : reduced-speed-of-light factor (RSLA); chat = c / creduc (>=1)
  const Real c_code = pin->GetOrAddReal(bn, "c_code", c_default);
  const Real arad_code = pin->GetOrAddReal(bn, "arad_code", arad_default);
  const Real creduc = pin->GetOrAddReal(bn, "creduc", 1.0);
  PARTHENON_REQUIRE(creduc >= 1.0, "radiation/creduc must be >= 1 (chat = c/creduc <= c).");
  pkg->AddParam("c", c_code);
  pkg->AddParam("arad", arad_code);
  pkg->AddParam("chat", c_code / creduc);
  pkg->AddParam("creduc", creduc);
  // Density threshold (code units) for the split Er history columns. 0 = off (default),
  // which registers no extra column and is therefore bit-identical for existing decks.
  pkg->AddParam("diag_rho_split",
                pin->GetOrAddReal(bn, "diag_rho_split", 0.0));
  pkg->AddParam("T_unit", T_unit);

  // --- Gray opacity model (code units) ---------------------------------------
  // "constant": kappa_a_code / kappa_s_code placeholders (per unit mass).
  // "dust":     Bell & Lin (1994) low-T ice-grain law kappa = k0 * T_phys^2 [cm^2/g],
  //             converted to code units internally (this is what traps the FHC).
  OpacityParams op;
  const std::string opmodel = pin->GetOrAddString(bn, "opacity_model", "constant");
  PARTHENON_REQUIRE(opmodel == "constant" || opmodel == "dust" || opmodel == "belllin" ||
                        opmodel == "tabulated",
                    "radiation/opacity_model must be 'constant', 'dust', 'belllin', or "
                    "'tabulated'.");
  op.model = (opmodel == "tabulated")
                 ? OpacityModel::tabulated
                 : (opmodel == "belllin")
                       ? OpacityModel::belllin
                       : (opmodel == "dust") ? OpacityModel::dust : OpacityModel::constant;
  op.kappa_a0 = pin->GetOrAddReal(bn, "kappa_a_code", 0.0);
  op.kappa_s0 = pin->GetOrAddReal(bn, "kappa_s_code", 0.0);
  op.dust_k0_cgs = pin->GetOrAddReal(bn, "dust_kappa0_cgs", 2.0e-4);
  op.dust_Tmax = pin->GetOrAddReal(bn, "dust_Tmax_K", 1.5e3);
  op.kappa_s_dust = pin->GetOrAddReal(bn, "kappa_s_code", 0.0);
  op.T_unit = T_unit;
  op.rho_unit = rho_unit;
  op.kappa_conv = kappa_conv;
  // Planck/Rosseland split: kappa_P = planck_ross_ratio * kappa_R for the emission term.
  // Default 1.0 => bit-identical to the single-opacity behavior.
  op.planck_ross_ratio = pin->GetOrAddReal(bn, "planck_ross_ratio", 1.0);
  {
    // R-F8 (validation/radiation_sn, 2026-10-08): the evaporation model of the dust/gas opacity split must be
    // the one the opacity table was generated with (v9 = pollack). Read from <dust> so it sits with the other
    // dust->opacity keys. Default "bl94" = historical.
    const std::string ev = pin->GetOrAddString("dust", "opacity_evap", "bl94");
    // R-F11 (2026-10-09): "table" = the grain share stored in a format-v4 table (--model semenov); checked
    // after the table is loaded, below.
    PARTHENON_REQUIRE(ev == "bl94" || ev == "pollack" || ev == "table",
                      "dust/opacity_evap must be 'bl94', 'pollack' or 'table'");
    op.dust_evap_pollack = (ev == "pollack");
    op.dust_evap_table = (ev == "table");
    // validation/dust DD5 / DU-F2: equilibrium-relative dust multiplier when the dust package sublimates at the
    // table's own T5(rho). Only meaningful with the table's grain share; any other combination is refused rather
    // than silently half-applied.
    const std::string sm = pin->GetOrAddString("dust", "subl_model", "fixed");
    op.dust_subl_eq = (sm == "semenov");
    op.dust_subl_width = pin->GetOrAddReal("dust", "subl_width_K", 100.0);
    PARTHENON_REQUIRE(!op.dust_subl_eq || op.dust_evap_table,
                      "dust/subl_model = semenov needs dust/opacity_evap = table (the v4 table's grain share is "
                      "built with the same Semenov T5(rho); with bl94/pollack the split uses another evaporation "
                      "model and the two would disagree again)");
  }
  PARTHENON_REQUIRE(op.planck_ross_ratio > 0.0,
                    "radiation/planck_ross_ratio must be > 0.");
  // Regime-skip-robust Bell&Lin walk (default false = bit-identical; fixes the off-track
  // low-rho/high-T kappa discontinuity). Only meaningful for opacity_model=belllin.
  // Default flipped false -> true 2026-08-08; see OpacityParams in radiation_opacity.hpp for the
  // measurement. Bit-identical below ~100 K (the whole ensemble epoch); up to 8.59 decades of
  // kappa error avoided once the run is hot at low density.
  op.bell_lin_fix_regime_skip =
      pin->GetOrAddBoolean(bn, "bell_lin_fix_regime_skip", true);
  // Tabulated model: load the offline frequency-resolved gray magnitude table (true Planck +
  // Rosseland means; self-consistent, replacing the planck_ross_ratio fudge). The per-group
  // multipliers from the SAME file feed GroupOpacityTable below.
  // Audit N13: default is the cgs (v2) table converted with this run's units; the legacy v1 table
  // was uniformly +0.135 % high (generator's rounded rho0*l0). Decks naming v1 explicitly still work.
  // R-F5 (validation/radiation_sn, 2026-10-08): "linear" (default, historical) or "log": interpolate
  // log(kappa) and log(group multiplier) -- exact on each Bell & Lin power-law piece.
  const std::string opacity_interp = pin->GetOrAddString(bn, "opacity_interp", "linear");
  PARTHENON_REQUIRE(opacity_interp == "linear" || opacity_interp == "log",
                    "radiation/opacity_interp must be 'linear' or 'log'");
  const bool opacity_interp_log = (opacity_interp == "log");
  const std::string opac_table_file = pin->GetOrAddString(
      bn, "opacity_table_file",
      "opacity_table.bin");
  if (op.model == OpacityModel::tabulated) {
    // AUDIT N13 (2026-08-05), FIXED. A v2 table stores kappa in cgs and Load() converts it
    // with THIS run's opacity_unit, so the file is independent of any particular IC. A legacy
    // v1 table stores kappa already in the generator's own rounded code units and is used
    // verbatim -- bit-identical to the historical behaviour, and carrying the historical bias.
    // Warn in that case, quantifying it, so an old table can never be mistaken for a fixed one.
    op.table.Load(opac_table_file, rho_unit, T_unit, kappa_conv);
    if (opacity_interp_log) op.table.EnableLogInterp();   // R-F5
    PARTHENON_REQUIRE(!op.dust_evap_table || op.table.has_dust_weight_,
                      "dust/opacity_evap = table needs a format-v4 opacity table with dust weights "
                      "(gen_opacity_table.py --model semenov); " + opac_table_file + " has none");
    if (parthenon::Globals::my_rank == 0 && op.table.used_legacy_code_units) {
      constexpr Real kGenRhoUnit = 5.467e-19; // gen_opacity_table.py RHO_UNIT, format v1
      constexpr Real kGenLenUnit = 2.81e16;   // gen_opacity_table.py LEN_UNIT, format v1
      const Real unit_ratio = kGenRhoUnit * kGenLenUnit / kappa_conv;
      std::cout << "### WARNING Radiation (audit N13): " << opac_table_file
                << " is a LEGACY (v1) opacity table: kappa is stored in the generator's own "
                   "code units and is used verbatim. This run's opacity_unit = "
                << kappa_conv << " vs the generator's " << kGenRhoUnit * kGenLenUnit
                << " => kappa is " << 100.0 * (unit_ratio - 1.0)
                << "% off. Regenerate with gen_opacity_table.py (which now emits cgs) to "
                   "remove the bias; the v2 file is correct for any normalization."
                << std::endl;
    }
  }
  PARTHENON_REQUIRE(!op.dust_evap_table || op.model == OpacityModel::tabulated,
                    "dust/opacity_evap = table needs radiation/opacity_model = tabulated");
  pkg->AddParam("opacity", op);
  // Back-compat scalar params (constant model still reads these directly).
  pkg->AddParam("kappa_a", op.kappa_a0);
  pkg->AddParam("kappa_s", op.kappa_s0);

  // --- Transport: discrete ordinates only ---------------------------------------------------
  // (radiation/closure is no longer read; production decks still carry `closure = M1`, inert.)
  const std::string transport = pin->GetOrAddString(bn, "transport", std::string("sn"));
  PARTHENON_REQUIRE(transport == "sn",
                    "radiation/transport must be 'sn' (got: " + transport + "). The m1, fld, "
                    "m1_implicit and m1_thermal transports were removed on 2026-09-24.");
  pkg->AddParam("transport", transport);

  // --- Multigroup frequency structure (Phase 4) ------------------------------
  // n_group=1 (default) => a single group spanning [0,inf) == the gray path, bit-identical.
  // n_group>1 partitions the spectrum into log-spaced groups [nu_min,nu_max] Hz (edges 0 and
  // +inf on the ends so no energy is lost); each group has its own ordinates and moments, and
  // the groups couple through the matter temperature (Sn matter coupling).
  const int n_group = pin->GetOrAddInteger(bn, "n_group", 1);
  PARTHENON_REQUIRE(n_group >= 1 && n_group <= MAX_GROUP,
                    "radiation/n_group must be in [1, MAX_GROUP=" + std::to_string(MAX_GROUP) + "].");
  const Real nu_min_hz = pin->GetOrAddReal(bn, "nu_min_hz", 1.0e12); // ~40 K blackbody peak
  const Real nu_max_hz = pin->GetOrAddReal(bn, "nu_max_hz", 1.0e15); // ~far-UV
  PARTHENON_REQUIRE(!(n_group > 1) || (nu_min_hz > 0.0 && nu_max_hz > nu_min_hz),
                    "radiation: multigroup needs 0 < nu_min_hz < nu_max_hz.");
  // audit N11 (2026-08-05): n_group=2 has one interior edge, so nu_max_hz cannot be honoured.
  if (n_group == 2 && parthenon::Globals::my_rank == 0)
    std::cout << "### WARNING Radiation: n_group = 2 has a single interior group edge, which is "
                 "placed at nu_min_hz = "
              << nu_min_hz << " Hz. radiation/nu_max_hz = " << nu_max_hz
              << " is IGNORED. Use n_group >= 3 to span [nu_min_hz, nu_max_hz]." << std::endl;
  // R-F13: radiation/nu_edges_hz = explicit interior edges (space-separated, n_group - 1 values); empty (default) =
  // the log-uniform partition over [nu_min_hz, nu_max_hz] as before.
  const std::string nu_edges = pin->GetOrAddString(bn, "nu_edges_hz", "");
  RadGroups groups;
  if (nu_edges.empty()) {
    groups = BuildRadGroups(n_group, nu_min_hz, nu_max_hz);
  } else {
    std::vector<Real> ie;
    std::istringstream ss(nu_edges);
    for (Real v; ss >> v;) ie.push_back(v);
    groups = BuildRadGroupsFromEdges(n_group, ie);
  }
  // Monochromatic dust opacity spectral index beta in kappa_nu = kappa_gray*(nu/nu_ref)^beta.
  // The per-group Planck/Rosseland means are the band-limited averages of kappa_nu with the
  // correct weights (B_nu for emission/absorption, dB_nu/dT for flux), normalized so the
  // full-spectrum means reproduce kappa_gray. beta=0 => every group mean == kappa_gray =>
  // multigroup reduces EXACTLY to n_group copies of gray (equivalence gate). `opacity_beta`
  // is the key; `opacity_nu_index` (the earlier representative-frequency exponent) is accepted
  // as a fallback so existing decks keep working.
  const Real beta_fallback = pin->GetOrAddReal(bn, "opacity_nu_index", 0.0);
  DustOpacityModel dmodel;
  dmodel.beta = pin->GetOrAddReal(bn, "opacity_beta", beta_fallback);
  dmodel.nu_break_hz = pin->GetOrAddReal(bn, "opacity_nu_break_hz", 1.0e14);
  dmodel.T_sub_lo = pin->GetOrAddReal(bn, "opacity_T_sub_lo_K", 1400.0);
  dmodel.T_sub_hi = pin->GetOrAddReal(bn, "opacity_T_sub_hi_K", 1600.0);
  pkg->AddParam("groups", groups);
  pkg->AddParam("n_group", n_group);
  pkg->AddParam("opacity_beta", dmodel.beta);
  pkg->AddParam("dust_opacity_model", dmodel);
  // Precompute the per-group Semenov-class band-mean multipliers on a log(T) grid (device
  // table) so the matter coupling does an O(1) interpolation instead of a per-cell quadrature.
  // Inactive (multiplier == 1) for gray/beta=0, so the equivalence gate is untouched.
  const int op_nT = pin->GetOrAddInteger(bn, "opacity_table_nT", 256);
  const Real op_Tmin = pin->GetOrAddReal(bn, "opacity_table_Tmin_K", 3.0);
  const Real op_Tmax = pin->GetOrAddReal(bn, "opacity_table_Tmax_K", 1.0e6);
  // Tabulated model: per-group multipliers come from the SAME frequency-resolved file (real
  // <psi>_band/<psi>_full ratios) instead of the analytic DustOpacityModel band means. Gray
  // (n_group=1) needs only the gray magnitude table (loaded above) -> inactive multiplier table,
  // so DON'T read the file's per-group block (and don't require the file's ng to match 1).
  {
    // AUDIT F-24: pass the RUNTIME band edges so the loader can check them against the file's.
    GroupOpacityTable gt = (op.model == OpacityModel::tabulated && n_group > 1)
                               ? BuildGroupOpacityTableFromFile(opac_table_file, n_group, groups.nu_edge, rho_unit)
                               : BuildGroupOpacityTable(dmodel, groups, op_Tmin, op_Tmax, op_nT);
    if (opacity_interp_log) gt.EnableLogInterp();   // R-F5
    pkg->AddParam("optable", gt);
  }

  pkg->AddParam("efloor", pin->GetOrAddReal(bn, "efloor", 1.0e-15)); // Er floor (writeback)

  // RSLA validity monitor -> history column `rslaMax` (see RslaMaxReport). Default ON: it is
  // one max-reduction per cycle over a field that is being computed anyway, and it answers a
  // question -- "is chat big enough here?" -- that has so far only ever been argued.
  pkg->AddParam("rsla_diag", pin->GetOrAddBoolean(bn, "rsla_diag", true));

  pkg->AddParam("gamma", pin->GetReal("hydro", "gamma"));
  // H2-dissociation general EOS: when active, RT owns the gas energy and must use the
  // SAME Saha EOS as the hydro to map internal-energy <-> temperature (else RT and hydro
  // disagree on T and the dissociation buffering is inconsistent). Mirror the hydro-block
  // keys (defaults = FHC-calibrated). T(e) uses T_code = T_phys[K]/T_unit.
  const bool use_h2diss = (pin->GetOrAddString("hydro", "eos", "adiabatic") == "hydrogen");
  pkg->AddParam("use_h2diss", use_h2diss);
  EOSTable::EosTable eos_tab;
  if (use_h2diss) {
    // Same units as the hydro package's load (audit N14): a v2 cgs table is converted with
    // this run's scales, a legacy table is consumed verbatim. hydro.cpp does the warning.
    // The default MUST match hydro.cpp's hydro/eos_table_file default exactly: Parthenon aborts when
    // one (block, key) is registered with two different defaults, even if the deck sets it.
    {
      const auto interp = pin->GetOrAddString("hydro", "eos_interp", "p_over_rho"); // E-F1; default MUST match hydro.cpp
      PARTHENON_REQUIRE(interp == "p" || interp == "p_over_rho", "hydro/eos_interp must be p or p_over_rho");
      eos_tab.p_over_rho_ = (interp == "p_over_rho");
      // E-F5/E-F6 (2026-10-05): ideal-gas continuation outside the table; false = legacy edge clamp. Default MUST
      // match the other load site (hydro.cpp / radiation.cpp): Parthenon aborts on two defaults for one key.
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
      eos_tab.extrapolate_ = pin->GetOrAddBoolean("hydro", "eos_extrapolate", false);
      PARTHENON_REQUIRE(!eos_tab.extrapolate_, "hydro/eos_extrapolate = true needs a build WITHOUT "
                        "ATHENAPK_EOS_LEGACY_LOOKUPS (the legacy build compiles the original lookups verbatim)");
#else
      eos_tab.extrapolate_ = pin->GetOrAddBoolean("hydro", "eos_extrapolate", true);
#endif
    }
    eos_tab.Load(pin->GetOrAddString("hydro", "eos_table_file",
                                     "eos_table_hires_v5_cold.bin"),
                 U.rho_unit, U.v_unit);
    eos_tab.n_bisect_ = pin->GetOrAddInteger("hydro", "eos_bisections", 20); // audit P-13
  }
  pkg->AddParam("eos_tab", eos_tab);

  // --- A3: announce the opacity-table domain ---  bilin edge-clamps silently (measured -95 % /
  // +477 % errors at corners v2 could not represent), so the domain is printed at startup and
  // clamps are counted per cycle. Warn, never abort (restart-hostile for older decks).
  if (op.model == OpacityModel::tabulated && parthenon::Globals::my_rank == 0) {
    const auto &ot = op.table;
    const Real op_rlo = std::pow(10.0, ot.lr0_);
    const Real op_rhi = std::pow(10.0, ot.lr0_ + (ot.nr_ - 1) * ot.dlr_);
    const Real op_tlo = std::pow(10.0, ot.lT0_);
    const Real op_thi = std::pow(10.0, ot.lT0_ + (ot.nT_ - 1) * ot.dlT_);
    std::cout << "### Radiation opacity table: " << opac_table_file << "\n"
              << "###   domain rho[g/cm^3] " << op_rlo << " .. " << op_rhi << "  (nr = "
              << ot.nr_ << ", dlogrho = " << ot.dlr_ << ")\n"
              << "###   domain T[K]        " << op_tlo << " .. " << op_thi << "  (nT = "
              << ot.nT_ << ", dlogT = " << ot.dlT_ << ")\n"
              << "###   lookups outside this box are CLAMPED to the edge value."
              << std::endl;
    if (eos_tab.loaded_) {
      // EosTable stores its rho axis in CODE units (eos_table.hpp:151 shifts a cgs file by
      // log10(rho_unit)); a legacy file carries the generator's own code units, which differ
      // by 0.14 % -- irrelevant against a domain check spanning decades. Its T axis is log10
      // K in both cases. Put both tables in cgs and compare.
      const Real eos_rlo = std::pow(10.0, eos_tab.lr0_) * U.rho_unit;
      const Real eos_rhi =
          std::pow(10.0, eos_tab.lr0_ + (eos_tab.nr_ - 1) * eos_tab.dlr_) * U.rho_unit;
      const Real eos_tlo = std::pow(10.0, eos_tab.lT0_);
      const Real eos_thi =
          std::pow(10.0, eos_tab.lT0_ + (eos_tab.nT_ - 1) * eos_tab.dlT_);
      const bool short_rho = op_rhi < eos_rhi * 0.999;
      const bool short_T = op_thi < eos_thi * 0.999;
      if (short_rho || short_T) {
        std::cout << "### WARNING Radiation (A3): the opacity table does NOT cover the "
                     "thermodynamic range the EOS table claims. The EOS stays valid where "
                     "the opacity is silently clamped to its edge value.\n";
        if (short_rho)
          std::cout << "###   rho: opacity ends at " << op_rhi << " g/cm^3, EOS reaches "
                    << eos_rhi << " g/cm^3 (short by "
                    << std::log10(eos_rhi / op_rhi) << " decades)\n";
        if (short_T)
          std::cout << "###   T:   opacity ends at " << op_thi << " K, EOS reaches "
                    << eos_thi << " K (short by " << std::log10(eos_thi / op_thi)
                    << " decades)\n";
        std::cout << "###   If this run can reach those conditions, regenerate the table:\n"
                     "###     gen_opacity_table.py --out <file> --nr 101 --nT 201 "
                     "--rho_min 1e-20 --rho_max 1e0 --T_min 3.0 --T_max 3.0e5\n"
                     "###   and record it in src/eos/TABLE_PROVENANCE.md."
                  << std::endl;
      } else {
        std::cout << "###   covers the EOS table domain (rho <= " << eos_rhi
                  << " g/cm^3, T <= " << eos_thi << " K): OK." << std::endl;
      }
    }
  }

  pkg->AddParam("tfloor", pin->GetOrAddReal(bn, "tfloor_code", 1.0e-3)); // T floor (code)
  // Upper temperature clamp for the Sn emission/coupling (code T).
  //
  // Default = the opacity table's actual T ceiling (the analytic-model key opacity_table_Tmax_K,
  // 1e6 K, sent the rtsafe bracket off the file's 3e5 K edge on every probe). A bracket, so capping
  // it does not restrict physics: a root above the table was extrapolated opacity anyway.
  Real ctmax_K = pin->GetOrAddReal(bn, "coupling_tmax_K", op_Tmax);
  if (op.model == OpacityModel::tabulated && op.table.loaded_) {
    Real lr_lo, lr_hi, lT_lo, lT_hi;
    op.table.AxisRanges(lr_lo, lr_hi, lT_lo, lT_hi);
    Real tab_Tmax = std::pow(10.0, lT_hi);
    // Pull the cap 1e-9 (relative) inside the table: exactly on the top node the bilinear index
    // tips on the last bit and floods the clamp counter (58M counts with v4). No physical effect.
    tab_Tmax *= (1.0 - 1.0e-9);
    if (ctmax_K > tab_Tmax) {
      if (parthenon::Globals::my_rank == 0) {
        std::cout << "### NOTE Radiation: coupling_tmax capped " << ctmax_K << " -> "
                  << tab_Tmax << " K, the actual upper edge of " << opac_table_file
                  << ". Bracketing above the table only ever produced edge-clamped "
                     "(extrapolated) opacity; see the opacity clamp report." << std::endl;
      }
      ctmax_K = tab_Tmax;
    }
    // The LOW end is the other half of what the counter found: 19,023 gray + 57,069 group
    // low-T clamps per cycle, because the radiation temperature floor sits BELOW the table.
    // Not silently adjusted -- tfloor is a physics knob and moving it would change results --
    // but it must not be silent either.
    const Real tab_Tmin = std::pow(10.0, lT_lo);
    const Real tfloor_K = pin->GetOrAddReal(bn, "tfloor_code", 1.0e-3) * T_unit;
    if (tfloor_K < tab_Tmin && parthenon::Globals::my_rank == 0) {
      std::cout << "### WARNING Radiation: <radiation> tfloor_code gives T_floor = " << tfloor_K
                << " K, BELOW the opacity table's lower edge " << tab_Tmin << " K. Every "
                   "floored cell will edge-clamp its opacity lookup (nearest-neighbour, not "
                   "interpolated). Either raise tfloor_code to >= "
                << tab_Tmin / T_unit << " (code) or regenerate the table down to T_floor."
                << std::endl;
    }
  }
  pkg->AddParam("coupling_tmax", ctmax_K / T_unit);

  // LOWER rtsafe bracket, by the same argument as the upper one. It used to be `tfloor`
  // (0.01 K for the collapse run), while the opacity table stops at 3 K -- so every bracket probe
  // at the low end fell off the bottom edge and was silently clamped, exactly mirroring the
  // high-side bug. Measured: 15,715 gray + 47,145 group low-T clamps per cycle, all of them
  // solver probes rather than physical cell states (the gas is at ~10 K at this epoch).
  // Raising the bracket floor to the table edge does not restrict the physics: a root below
  // the table's floor was extrapolated opacity, not data. tfloor itself is UNCHANGED -- it is
  // a physics knob and moving it would be result-changing.
  Real ctmin_K = pkg->Param<Real>("tfloor") * T_unit;
  if (op.model == OpacityModel::tabulated && op.table.loaded_) {
    Real lr_lo, lr_hi, lT_lo, lT_hi;
    op.table.AxisRanges(lr_lo, lr_hi, lT_lo, lT_hi);
    const Real tab_Tmin = std::pow(10.0, lT_lo);
    if (ctmin_K < tab_Tmin) ctmin_K = tab_Tmin;
  }
  pkg->AddParam("coupling_tmin", ctmin_K / T_unit);
  {
    const Real tfl_K = pkg->Param<Real>("tfloor") * T_unit;
    if (ctmin_K > tfl_K && parthenon::Globals::my_rank == 0) {
      std::cout << "### NOTE Radiation: matter-coupling bracket floor raised " << tfl_K
                << " -> " << ctmin_K
                << " K (opacity-table lower edge). Roots below this are returned AT the edge,"
                   " so this is an effective coupling temperature floor; non-binding while the"
                   " gas stays above it. Widen the opacity table to lower it." << std::endl;
    }
  }

  // --- Fields: radiation moments (Er, Fr1, Fr2, Fr3) per group ------------------------------
  // Written by the Sn writeback each step; outputs and restart datasets. The metadata is kept
  // exactly as the M1 era registered it (Independent, FillGhost, WithFluxes) so existing
  // restart files load unchanged. The OperatorSplit user flag keeps them out of the hydro
  // integrator's {WithFluxes, Cell} flag packs.
  const parthenon::MetadataFlag OperatorSplit =
      parthenon::Metadata::FlagNameExists("OperatorSplit")
          ? parthenon::Metadata::GetUserFlag("OperatorSplit")
          : parthenon::Metadata::AddUserFlag("OperatorSplit");
  std::vector<parthenon::MetadataFlag> flags{Metadata::Cell, Metadata::Independent,
                                             Metadata::FillGhost, Metadata::WithFluxes,
                                             OperatorSplit};
  Metadata m(flags);
  m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                          parthenon::refinement_ops::RestrictAverage>();
  pkg->AddField<rad::Er>(m);
  pkg->AddField<rad::Fr1>(m);
  pkg->AddField<rad::Fr2>(m);
  pkg->AddField<rad::Fr3>(m);
  // Multigroup: register the extra groups' moments by string name (same Metadata as the
  // gray fields). Group 0 is the four fields above (original names) so n_group=1 adds
  // nothing here -> registration is byte-for-byte the gray layout.
  for (int g = 1; g < n_group; ++g)
    for (const auto &nm : GroupFieldNames(g)) pkg->AddField(nm, m);

  // --- Per-group extinction chi = rho (kR,g + ks) for rslaMax ---  Derived, OneCopy, computed over
  // the whole block (no exchange). Registered only with rsla_diag.
  if (pkg->Param<bool>("rsla_diag")) {
    Metadata mchi({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    for (int g = 0; g < n_group; ++g) pkg->AddField(GroupChiName(g), mchi);
  }

  // History: RSLA validity. APPENDED at the end of the radiation package's own list, never
  // inserted -- hydro.cpp:~2193 records that inserting a history entry shifts every later
  // column index and silently breaks downstream analysis scripts.
  if (pkg->Param<bool>("rsla_diag")) {
    parthenon::HstVar_list hst_rad = {};
    hst_rad.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::max, RslaMaxReport, "rslaMax"));
    // APPENDED after rslaMax, never inserted: inserting a history entry shifts every later
    // column index and breaks downstream analysis.
    hst_rad.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, RadErTotal, "rad-Er"));
    hst_rad.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::max, RadFrMax, "rad-Frmax"));
    // Density-split Er. `rad-Er` above is a whole-box SUM and was measured USELESS on the
    // collapse run (exactly 3.170450e+02 for 3389 consecutive cycles of live leg C): the core is
    // ~5.9e-11 of the box, so its signal falls below the history file's 6 significant figures.
    // Registered ONLY when a split is set, and appended last, so every existing history file
    // and every notebook column index stays valid.
    if (pkg->Param<Real>("diag_rho_split") > 0.0) {
      hst_rad.emplace_back(parthenon::HistoryOutputVar(
          parthenon::UserHistoryOperation::sum, RadErDense, "rad-Er-dense"));
      hst_rad.emplace_back(parthenon::HistoryOutputVar(
          parthenon::UserHistoryOperation::sum, RadVolDense, "rad-Vdense"));
    }
    pkg->AddParam<>(parthenon::hist_param_key, hst_rad);
  }

  // --- Discrete ordinates (Sn) --------------------------------------------------------------
  Sn::Initialize(pin, pkg.get());
  return pkg;
}

} // namespace Radiation
