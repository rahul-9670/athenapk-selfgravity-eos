//========================================================================================
// AthenaPK WS-4 dust package — implementation. Two passive scalars carry the dust-to-gas
// ratio f_dg and the characteristic grain radius a_c [cm]; this package adds the operator-
// split growth + sublimation source (DustModel::integrate_cell). Gated behind <physics>
// dust=true (default false) => bit-identical to pre-dust runs.
//========================================================================================
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "dust_pkg.hpp"
#include "dust.hpp" // DustModel (after parthenon => DUST_FN = device)
// NSPEC_H2 / NSPEC_GOW for the F3 scalar-index collision guard (after parthenon headers
// so the networks' CHEMG_FN/CHEM_FN resolve to the device-callable form).
#include "../eos/adiabatic_glmmhd.hpp" // EOS-table T under eos=hydrogen (audit N2)
#include "../main.hpp"
#include "../units/physical_units.hpp" // the one authoritative unit system (audit N3)

using namespace parthenon::package::prelude;

namespace Dust {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("dust");
  const std::string bn = "dust";

  const bool evolve = pin->GetOrAddBoolean(bn, "evolve", false);
  pkg->AddParam("evolve", evolve);

  // Audit N3: dust units come from PhysUnits::BuildPhysicalUnits like every other package; the old
  // <dust> unit keys are dead. WARN, do not fail: restarts written by pre-N3 binaries carry the
  // keys as injected defaults, and a hard failure would make them unloadable.
  const auto U = PhysUnits::BuildPhysicalUnits(pin);
  if (parthenon::Globals::my_rank == 0) {
    const char *ukeys[3] = {"rho_unit_cgs", "t_unit_cgs", "T_unit_K"};
    const Real uderived[3] = {U.rho_unit, U.time_unit, U.temperature_unit};
    for (int q = 0; q < 3; ++q) {
      if (!pin->DoesParameterExist(bn, ukeys[q])) continue;
      const Real given = pin->GetReal(bn, ukeys[q]);
      std::cout << "### WARNING Dust: <dust> " << ukeys[q] << " = " << given
                << " is IGNORED (dead key). The dust cgs scales are single-sourced from "
                   "PhysUnits::BuildPhysicalUnits; this run uses "
                << uderived[q] << " (relative difference "
                << (uderived[q] != 0.0 ? (given - uderived[q]) / uderived[q] : 0.0)
                << "). Remove it from the deck. If it appears in a RESTART rather than a "
                   "deck it was injected as a GetOrAdd default by a pre-2026-08-05 binary "
                   "and is harmless -- the value above is what is actually used."
                << std::endl;
    }
  }
  DustModel d;
  d.rho_unit = U.rho_unit;
  d.t_unit = U.time_unit;
  d.rho_grain = pin->GetOrAddReal(bn, "rho_grain_cgs", 3.0);
  d.a_ref = pin->GetOrAddReal(bn, "a_ref_cm", 1.0e-5);
  d.f_dg_ref = pin->GetOrAddReal(bn, "f_dg_ref", 0.01);
  d.alpha_turb = pin->GetOrAddReal(bn, "alpha_turb", 1.0e-2);
  d.omega_ref = pin->GetOrAddReal(bn, "omega_ref_cgs", 1.0e-13);
  d.mu_gas = pin->GetOrAddReal(bn, "mu_n", 2.33);
  d.T_subl = pin->GetOrAddReal(bn, "T_subl_K", 1.5e3);
  d.growth_cap = pin->GetOrAddReal(bn, "growth_cap", 0.10);
  // validation_tests/dust D2 (2026-10-12): sweep-up prefactor, default 0.25 = historical "/4" bit-for-bit.
  d.growth_pref = pin->GetOrAddReal(bn, "growth_prefactor", 0.25);
  PARTHENON_REQUIRE_THROWS(d.growth_pref > 0.0, "<dust> growth_prefactor must be > 0");
  // validation_tests/dust DD5 (2026-10-12): sublimation centre and width. Defaults = the historical constant
  // T_subl_K with the hard-coded 100 K width, bit-for-bit. subl_model = semenov: centre T5(rho) of Semenov et al.
  // (2003) -- the evaporation temperature the v4/v12 opacity table was built with (Dust::SemenovT5).
  {
    const std::string sm = pin->GetOrAddString(bn, "subl_model", "fixed");
    PARTHENON_REQUIRE_THROWS(sm == "fixed" || sm == "semenov", "<dust> subl_model must be 'fixed' or 'semenov'");
    d.subl_model = (sm == "semenov") ? 1 : 0;
    d.subl_width = pin->GetOrAddReal(bn, "subl_width_K", 100.0);
    PARTHENON_REQUIRE_THROWS(d.subl_width > 0.0, "<dust> subl_width_K must be > 0");
    // DD2 (2026-10-13): finite sublimation time [s]; 0 = instantaneous (default, bit-identical). See DustModel::subl_tau.
    d.subl_tau = pin->GetOrAddReal(bn, "subl_tau_s", 0.0);
    PARTHENON_REQUIRE_THROWS(d.subl_tau >= 0.0, "<dust> subl_tau_s must be >= 0");
  }
  // 2026-08-15 fixes. Both default ON because this project has repeatedly been burned by
  // corrections that were "implemented and gated" and then reached zero decks; both have an
  // OFF value that reproduces the previous arithmetic bit-identically, which is what the A/B
  // is measured against. See src/dust/dust.hpp (omega_dyn, integrate_cell) for the arguments
  // and for the measured size of each.
  d.omega_local = pin->GetOrAddBoolean(bn, "omega_local", true);
  d.subl_irreversible = pin->GetOrAddBoolean(bn, "subl_irreversible", true);
  // --- AUDIT F-06 / F-07 opacity-coupling corrections (2026-09-16). Both DEFAULT ON since
  // 2026-09-16 (A/B jobs 2752382/2752383, cycles 8589->8789: dT/T <= 2.6% at matched density,
  // rho_max 0.905x, no instability; docs/FIX_DUST_OPACITY_2026-09-16.md). The OFF values
  // (geometric / false) execute the historical expression unchanged => bit-identical to the
  // pre-fix binary, which is how the A/B baseline is reproduced.
  {
    const std::string law = pin->GetOrAddString(bn, "opacity_size_law", "bridged");
    PARTHENON_REQUIRE_THROWS(law == "geometric" || law == "bridged",
                             "<dust> opacity_size_law must be 'geometric' (historical, the "
                             "a >> lambda limit) or 'bridged' (AUDIT F-06: Rayleigh<->geometric "
                             "bridge; grain growth stops changing kappa where the spectral "
                             "weight is below the grain-size break).");
    d.opac_size_bridged = (law == "bridged");
  }
  d.opac_gas_split = pin->GetOrAddBoolean(bn, "opacity_gas_split", true);
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "## Dust->opacity coupling: size_law="
              << (d.opac_size_bridged ? "bridged" : "geometric")
              << "  gas_split=" << (d.opac_gas_split ? "true" : "false")
              << "  (AUDIT F-06/F-07; defaults bridged/true; geometric/false = historical)" << std::endl;
  }
  // zero-growth switch for the inc-2 bit-identical gate (evolve on, but grains frozen).
  const bool freeze_growth = pin->GetOrAddBoolean(bn, "freeze_growth", false);
  pkg->AddParam("freeze_growth", freeze_growth);
  // dust/dual_energy (2026-09-23, default false = bit-identical): take the gas temperature from the
  // dual-energy internal energy where hydro/dual_energy switched the cell (same selection as
  // ConsToPrim), instead of E - KE - ME, which is ill-conditioned in low-beta cells and can be
  // negative there -- that would mis-drive growth and (irreversible) sublimation.
  pkg->AddParam("dual_energy", pin->GetOrAddBoolean(bn, "dual_energy", false));
  // dust/subl_shock_hold (validation_tests/dust DD2, 2026-10-12; default 0 = off, bit-identical). Irreversible
  // sublimation records the HOTTEST temperature a parcel is ever seen at, and a parcel crossing a captured shock is
  // seen inside the numerical shock profile at T above the Rankine-Hugoniot T2 (D5: +0.66 % of T2 at Mach 10, the
  // same at every resolution). That transient is not physics, and it is recorded as permanent dust loss that does
  // not converge away. 1 = no sublimation update in a cell the hydro shock detector flags this step
  // (dual_comp >= hydro/dual_energy_shock_comp, the flag shock_eta1/eta2 already use; kv in production); the
  // parcel sublimates on its next step out of the shock, at its post-shock temperature. 2 = also when a face
  // neighbour is flagged (one cell of dilation, for an overshoot that trails the flag). Growth is untouched.
  {
    const int hold = pin->GetOrAddInteger(bn, "subl_shock_hold", 0);
    PARTHENON_REQUIRE_THROWS(hold >= 0 && hold <= 2, "<dust> subl_shock_hold must be 0, 1 or 2");
    pkg->AddParam("subl_shock_hold", hold);
    // the shock threshold the hydro readers use lives inside the EOS object (not a package Param): read the same key
    // with the same default (hydro.cpp, dual_energy_shock_comp = 0.02).
    pkg->AddParam("subl_shock_thr", pin->GetOrAddReal("hydro", "dual_energy_shock_comp", 0.02));
  }
  // dust/subl_T_filter (validation_tests/dust DD2, 2026-10-13; default false = bit-identical). D5 showed WHY the
  // shock-recorded deficit does not converge: behind a captured shock the post-shock state carries odd-even (2 dx)
  // oscillations (+-2..4 K about T2 at Mach 10, independent of N), and irreversible sublimation records each parcel's
  // MAXIMUM. The kv shock flag covers only the 1-2 shock cells, not this trailing noise. With the filter, sublimation
  // reads T_eff = min(T, T <P/rho>/(P/rho)), <.> = separable 1-2-1 average over the face/edge/corner neighbours
  // (prim, ghosts included): the 2 dx mode is removed exactly, a smooth field is unchanged to O(dx^2), and T is never
  // RAISED (a cold cell beside a hot contact does not sublimate more than without the filter). Growth keeps the
  // cell's own T.
  pkg->AddParam("subl_T_filter", pin->GetOrAddBoolean(bn, "subl_T_filter", false));
  pkg->AddParam("model", d);
  pkg->AddParam("T_unit_K", U.temperature_unit);
  pkg->AddParam("gamma", pin->GetReal("hydro", "gamma"));
  // Scalar block index of f_dg (a_c = f_dg index + 1).
  pkg->AddParam("scalar_index", pin->GetOrAddInteger(bn, "scalar_index", 0));

  if (evolve) {
    const int nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
    const int si = pin->GetOrAddInteger(bn, "scalar_index", 0);
    PARTHENON_REQUIRE(nscalars >= si + 2,
                      "dust: <hydro> nscalars must be >= dust/scalar_index + 2 (f_dg, a_c).");
  }
  return pkg;
}

TaskStatus ReactDust(MeshData<Real> *md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->pmy_mesh->packages.Get("dust");
  if (!pkg->Param<bool>("evolve")) return TaskStatus::complete;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = pmb->pmy_mesh->packages.Get("Hydro");
  const int nhydro = hydro_pkg->Param<int>("nhydro");
  // audit N7: ask the package what fluid it is, rather than inferring it from the component
  // count. Same value today (glmmhd carries IB1..IB3,IPS past the hydro vars), one less
  // invariant to break silently when a variable is added.
  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
  // AUDIT 2026-08-05 (N2) -- THE SUBLIMATION TEMPERATURE MUST COME FROM THE EOS.
  // This derived T as (gamma-1) e/rho with the IDEAL gamma while every production deck sets
  // <hydro> eos = hydrogen, then compared it with T_subl = 1500 K. That is the same defect as
  // audit A1/A6, and it lands in the WORST possible place: 1500 K is inside the H2
  // dissociation zone, exactly where Gamma_1 -> 1.14 and the ideal mapping over-estimates T.
  // Dust therefore sublimated EARLY, removing opacity and grain-surface ionisation with it.
  // The (since removed) chemistry and M1 radiation modules had already got this fix; dust
  // was the one module left on the ideal mapping. Ideal-gas path is bit-identical.
  bool use_h2 = false;
  EOSTable::EosTable eos_tab;
  if (mhd) {
    const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
    use_h2 = eos.UseH2Diss();
    if (use_h2) eos_tab = eos.GetEosTable();
  }
  const DustModel d = pkg->Param<DustModel>("model");
  const bool freeze = pkg->Param<bool>("freeze_growth");
  // dust/dual_energy, see the package initialisation.
  const bool dual_on = pkg->Param<bool>("dual_energy") && hydro_pkg->Param<int>("dual_idx") >= 0;
  const int idual_c = dual_on ? hydro_pkg->Param<int>("dual_idx") : 0;
  const Real dual_eta1 = dual_on ? hydro_pkg->Param<Real>("dual_energy_eta1") : 0.0;
  // hydro/dual_energy_shared_select (audit P-01): ConsToPrim's own two-sided selection
  // (AdiabaticGLMMHDEOS::DualSelectEint with this stage's dual_comp) instead of eta1 only.
  const bool dual_shared = dual_on && mhd &&
                           hydro_pkg->AllParams().hasKey("dual_energy_shared_select") &&
                           hydro_pkg->Param<bool>("dual_energy_shared_select");
  const AdiabaticGLMMHDEOS eos_dsel =
      dual_shared ? hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos")
                  : AdiabaticGLMMHDEOS(0.0, 0.0, 0.0, 0.0, 0.0, 1.4);
  const bool dsel_comp = dual_shared && eos_dsel.DualEta2() > 0.0;
  const Real gm1 = pkg->Param<Real>("gamma") - 1.0;
  const Real T_unit = pkg->Param<Real>("T_unit_K");
  const int sidx = pkg->Param<int>("scalar_index");

  static const std::vector<std::string> names{"cons"};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ic = imap["cons"].first;
  // dust/subl_shock_hold: the shock flag lives in dual_comp, written by DualEnergyPdV when hydro/dual_energy_eta2 > 0.
  const int shold = pkg->Param<int>("subl_shock_hold");
  PARTHENON_REQUIRE_THROWS(shold == 0 || (hydro_pkg->Param<int>("dual_idx") >= 0 &&
                                          hydro_pkg->Param<Real>("dual_energy_eta2") > 0.0),
                           "dust/subl_shock_hold needs hydro/dual_energy = true with dual_energy_eta2 > 0 (the "
                           "shock flag dual_comp is written only then)");
  const Real shock_thr = (shold > 0) ? pkg->Param<Real>("subl_shock_thr") : 0.0;
  const int ndim_d = pmb->pmy_mesh->ndim;
  const auto comp_pack = (dsel_comp || shold > 0) ? md->PackVariables(std::vector<std::string>{"dual_comp"})
                                   : md->PackVariables(std::vector<std::string>{"cons"});
  const int is = ic + nhydro + sidx; // f_dg scalar_density; a_c at is+1
  // dust/subl_T_filter: P/rho of the neighbours from prim (filled incl. ghosts by the last FillDerived)
  const bool tfilt = pkg->Param<bool>("subl_T_filter");
  const auto prim_pack = tfilt ? md->PackVariables(std::vector<std::string>{"prim"})
                               : md->PackVariables(std::vector<std::string>{"cons"});
  const int ndim_f = pmb->pmy_mesh->ndim;

  const Real rho_to_cgs = d.rho_unit;
  const Real dt_s = static_cast<Real>(dt) * d.t_unit;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Dust::React", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real rho = pack(b, ic + IDN, k, j, i);
        const Real inv_rho = 1.0 / rho;
        const Real m1 = pack(b, ic + IM1, k, j, i);
        const Real m2 = pack(b, ic + IM2, k, j, i);
        const Real m3 = pack(b, ic + IM3, k, j, i);
        Real eint = pack(b, ic + IEN, k, j, i) - 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) * inv_rho;
        Real e_B = 0.0;
        if (mhd) {
          const Real b1 = pack(b, ic + IB1, k, j, i);
          const Real b2 = pack(b, ic + IB2, k, j, i);
          const Real b3 = pack(b, ic + IB3, k, j, i);
          e_B = 0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
          eint -= e_B;
        }
        if (dual_on) {
          // Same selection as AdiabaticGLMMHDEOS::ConsToPrim.
          const Real E = pack(b, ic + IEN, k, j, i);
          const Real e_dual = pack(b, ic + idual_c, k, j, i);
          if (dual_shared) {
            eint = eos_dsel.DualSelectEint(eint, E, e_dual, rho,
                                           dsel_comp ? comp_pack(b, 0, k, j, i) : 0.0, e_B);
          } else if (!(eint > dual_eta1 * E) && e_dual > 0.0) {
            eint = e_dual;
          }
        }
        // audit N2: tabulated EOS temperature when eos=hydrogen (guarded against eint <= 0,
        // where the table interpolates in log10(eint/rho) and would return NaN); otherwise
        // the ideal mapping, bit-identical to before.
        double T_K;
        if (use_h2 && eint > 0.0) {
          T_K = static_cast<double>(eos_tab.TemperatureK(rho, eint));
          if (T_K < 1.0e-4 * T_unit) T_K = 1.0e-4 * T_unit; // floor (matches ideal path)
        } else {
          Real T_code = gm1 * eint * inv_rho;
          if (T_code < 1.0e-4) T_code = 1.0e-4;
          T_K = static_cast<double>(T_code * T_unit);
        }
        const double rho_cgs = static_cast<double>(rho * rho_to_cgs);

        double f_dg = static_cast<double>(pack(b, is, k, j, i) * inv_rho);
        double a_c = static_cast<double>(pack(b, is + 1, k, j, i) * inv_rho);
        // dust/subl_shock_hold: inside a captured shock, hold the dust state for this step (no sublimation).
        bool hold_subl = false;
        if (shold > 0) {
          hold_subl = comp_pack(b, 0, k, j, i) >= shock_thr;
          if (shold == 2 && !hold_subl) {
            hold_subl = comp_pack(b, 0, k, j, i - 1) >= shock_thr || comp_pack(b, 0, k, j, i + 1) >= shock_thr;
            if (ndim_d > 1)
              hold_subl = hold_subl || comp_pack(b, 0, k, j - 1, i) >= shock_thr ||
                          comp_pack(b, 0, k, j + 1, i) >= shock_thr;
            if (ndim_d > 2)
              hold_subl = hold_subl || comp_pack(b, 0, k - 1, j, i) >= shock_thr ||
                          comp_pack(b, 0, k + 1, j, i) >= shock_thr;
          }
        }
        if (hold_subl) {
          // growth only (sublimation skipped): integrate_cell with T capped at the cell's current equilibrium
          // would still sublimate, so run the sweep-up directly at the current (a_c, f_dg).
          if (!freeze) {
            if (a_c < d.a_floor) a_c = d.a_floor;
            d.grow_cell(a_c, f_dg, rho_cgs, T_K, static_cast<double>(dt_s));
          }
        } else if (tfilt) {
          // separable 1-2-1 filter of q = P/rho over the 3^ndim stencil (inactive directions: weight 1 on the cell)
          const int dk = (ndim_f > 2) ? 1 : 0, dj = (ndim_f > 1) ? 1 : 0;
          Real qs = 0.0, ws = 0.0;
          for (int kk = -dk; kk <= dk; ++kk)
            for (int jj = -dj; jj <= dj; ++jj)
              for (int ii = -1; ii <= 1; ++ii) {
                const Real w = (2 - (kk < 0 ? -kk : kk)) * (2 - (jj < 0 ? -jj : jj)) * (2 - (ii < 0 ? -ii : ii));
                qs += w * prim_pack(b, IPR, k + kk, j + jj, i + ii) / prim_pack(b, IDN, k + kk, j + jj, i + ii);
                ws += w;
              }
          qs /= ws;
          const Real qi = prim_pack(b, IPR, k, j, i) / prim_pack(b, IDN, k, j, i);
          const double T_sub = (qi > 0.0 && qs < qi) ? T_K * static_cast<double>(qs / qi) : T_K;
          d.apply_sublimation(a_c, f_dg, T_sub, rho_cgs, static_cast<double>(dt_s));
          if (!freeze) {
            if (a_c < d.a_floor) a_c = d.a_floor;
            d.grow_cell(a_c, f_dg, rho_cgs, T_K, static_cast<double>(dt_s));
          }
        } else if (freeze) {
          // bit-identical guard: only apply the sublimation switch, no growth. Routed through
          // the SAME apply_sublimation the growth path uses (2026-08-15) -- it used to carry
          // its own inline copy of `f_dg = f_dg_ref*subl_factor(T)`, so the irreversibility
          // fix would have reached the growth path only and the two would have silently
          // disagreed about what "sublimation" means. freeze_growth means freeze GROWTH.
          d.apply_sublimation(a_c, f_dg, T_K, rho_cgs, static_cast<double>(dt_s));
        } else {
          d.integrate_cell(a_c, f_dg, rho_cgs, T_K, static_cast<double>(dt_s));
        }
        pack(b, is, k, j, i) = rho * static_cast<Real>(f_dg);
        pack(b, is + 1, k, j, i) = rho * static_cast<Real>(a_c);
      });
  return TaskStatus::complete;
}

void AddDustTasks(TaskCollection &tc, Mesh *pmesh, const Real dt) {
  using namespace parthenon;
  if (!pmesh->packages.Get("dust")->Param<bool>("evolve")) return;
  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    tl.AddTask(TaskID(0), ReactDust, md.get(), dt);
  }
}

} // namespace Dust
