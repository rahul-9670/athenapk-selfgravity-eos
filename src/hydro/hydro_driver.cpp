//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon headers
#include "amr_criteria/refinement_package.hpp"
#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include <cmath>
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/parthenon.hpp>
// AthenaPK headers
#include "../tracers/tracers.hpp"
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>
#include "glmmhd/glmmhd.hpp"
#include "hydro.hpp"
#include "double_flux.hpp"
#include "dual_flux_density.hpp"
#include "dual_double_flux.hpp"
#include "dual_heat_exact.hpp"
#include "diffusion/diffusion.hpp"
#include "hydro_driver.hpp"
#include "../diagnostics/cons_diag.hpp"
#include "../self_gravity/self_gravity.hpp"
#include "../pgen/pgen.hpp"
#include "../radiation/radiation.hpp"

using namespace parthenon::driver::prelude;

namespace Hydro {

HydroDriver::HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("hydro", "eos");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/time", "cfl");

  // WP-13: the barotropic/exterior-momentum task is added only if collapse_be registered its Params
  // (ProblemInitPackageData, which also runs on restart). Die at startup if that wiring breaks rather
  // than silently dropping the task on a resume. Read job/problem_id WITHOUT registering a default.
  const std::string problem_id = pin->DoesParameterExist("job", "problem_id")
                                     ? pin->GetString("job", "problem_id")
                                     : std::string("");
  if (pm != nullptr && problem_id == "collapse_be") {
    auto hydro_pkg = pm->packages.Get("Hydro");
    PARTHENON_REQUIRE(
        hydro_pkg->AllParams().hasKey("collapse_be_rhocrit"),
        "collapse_be: Param 'collapse_be_rhocrit' is missing, so ApplyBarotropicCooling -- "
        "and with it the outside-sphere momentum BC -- would be SILENTLY skipped. Is "
        "collapse_be::ProblemInitPackageData still wired up in main.cpp?");
  }
}


TaskStatus ResetFluxes(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Pack "cons" by NAME rather than by the {Independent} flag: the STS resets the diffusive
  // fluxes, which are deposited ONLY onto cons (the v6-STS-bug-hardened choice).
  const std::vector<std::string> cons_name{"cons"};
  auto cons_pack = md->PackVariablesAndFluxes(cons_name, cons_name);

  const int ndim = pmb->pmy_mesh->ndim;
  // Using separate loops for each dim as the launch overhead should be hidden
  // by enough work over the entire pack and it allows to not use any conditionals.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ResetFluxes X1", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        cons.flux(X1DIR, v, k, j, i) = 0.0;
      });

  if (ndim < 2) {
    return TaskStatus::complete;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ResetFluxes X2", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e + 1,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        cons.flux(X2DIR, v, k, j, i) = 0.0;
      });

  if (ndim < 3) {
    return TaskStatus::complete;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ResetFluxes X3", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e + 1, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        cons.flux(X3DIR, v, k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

TaskStatus RKL2StepFirst(MeshData<Real> *md_Y0, MeshData<Real> *md_Yjm1,
                         MeshData<Real> *md_Yjm2, MeshData<Real> *md_MY0, const int s_rkl,
                         const Real tau) {
  auto pmb = md_Y0->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Compute coefficients. Meyer+2014 eq. (18)
  Real mu_tilde_1 = 4. / 3. /
                    (static_cast<Real>(s_rkl) * static_cast<Real>(s_rkl) +
                     static_cast<Real>(s_rkl) - 2.);

  // Pack by NAME ("cons"), not by {Metadata::Independent}: the diffusion terms update
  // only the gas conserved variables, but the Independent flag also matches rad.Er/Fr
  // (radiation) and grav.phi. Since Yjm1 aliases "base" and the Y0 snapshot ("u1") only
  // deep-copies cons/prim, a flag-based pack overwrites the radiation fields and phi in
  // base with the u1 container's stale/zero content on every STS call (this silently
  // zeroed rad.Er for the whole run past cycle 71000, see DEV_LOG 2026-07-15).
  const std::vector<std::string> sts_vars{"cons"};
  auto Y0 = md_Y0->PackVariablesAndFluxes(sts_vars, sts_vars);
  auto Yjm1 = md_Yjm1->PackVariablesAndFluxes(sts_vars, sts_vars);
  auto Yjm2 = md_Yjm2->PackVariablesAndFluxes(sts_vars, sts_vars);
  auto MY0 = md_MY0->PackVariablesAndFluxes(sts_vars, sts_vars);

  // Using separate loops for each dim as the launch overhead should be hidden
  // by enough work over the entire pack and it allows to not use any conditionals.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RKL first step", parthenon::DevExecSpace(), 0,
      Y0.GetDim(5) - 1, 0, Y0.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        Yjm1(b, v, k, j, i) =
            Y0(b, v, k, j, i) + mu_tilde_1 * tau * MY0(b, v, k, j, i); // Y_1
        Yjm2(b, v, k, j, i) = Y0(b, v, k, j, i);                       // Y_0
      });

  return TaskStatus::complete;
}

TaskStatus RKL2StepOther(MeshData<Real> *md_Y0, MeshData<Real> *md_Yjm1,
                         MeshData<Real> *md_Yjm2, MeshData<Real> *md_MY0, const Real mu_j,
                         const Real nu_j, const Real mu_tilde_j, const Real gamma_tilde_j,
                         const Real tau) {
  auto pmb = md_Y0->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Pack by NAME ("cons") — same constraint as RKL2StepFirst above: a flag-based
  // {Independent} pack pulls in rad.Er/Fr and grav.phi and corrupts them via the
  // cons/prim-only u1 snapshot.
  const std::vector<std::string> sts_vars{"cons"};
  auto Y0 = md_Y0->PackVariablesAndFluxes(sts_vars, sts_vars);
  auto Yjm1 = md_Yjm1->PackVariablesAndFluxes(sts_vars, sts_vars);
  auto Yjm2 = md_Yjm2->PackVariablesAndFluxes(sts_vars, sts_vars);
  auto MY0 = md_MY0->PackVariablesAndFluxes(sts_vars, sts_vars);

  const int ndim = pmb->pmy_mesh->ndim;
  // Using separate loops for each dim as the launch overhead should be hidden
  // by enough work over the entire pack and it allows to not use any conditionals.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RKL other step", parthenon::DevExecSpace(), 0,
      Y0.GetDim(5) - 1, 0, Y0.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        // First calc this step
        const auto &coords = Yjm1.GetCoords(b);
        const Real MYjm1 =
            parthenon::Update::FluxDivHelper(v, k, j, i, ndim, coords, Yjm1(b));
        const Real Yj = mu_j * Yjm1(b, v, k, j, i) + nu_j * Yjm2(b, v, k, j, i) +
                        (1.0 - mu_j - nu_j) * Y0(b, v, k, j, i) +
                        mu_tilde_j * tau * MYjm1 +
                        gamma_tilde_j * tau * MY0(b, v, k, j, i);
        // Then shuffle vars for next step
        Yjm2(b, v, k, j, i) = Yjm1(b, v, k, j, i);
        Yjm1(b, v, k, j, i) = Yj;
      });

  return TaskStatus::complete;
}


// hydro/lean_registers (audit P-04, 2026-09-23). The integrator register "u1" (VL2 snapshot) used to be
// created with no variable list, and MeshBlockData::Initialize then
// allocates a private copy of EVERY non-OneCopy field, most of which nothing ever reads from a register.
// Every reader of these registers packs either "cons"/"prim" by name or {WithFluxes, Cell} minus the
// OperatorSplit user flag (parthenon Update::FluxDivergence / UpdateWithFluxDivergence), so a
// register built from the non-OperatorSplit fields of "base" serves every pack with the same shape.
// Empty list = legacy (copy everything).
std::vector<std::string> RegisterVarNames(MeshBlockData<Real> *base, const bool lean) {
  std::vector<std::string> names;
  if (!lean) return names;
  const bool has_os = parthenon::Metadata::FlagNameExists("OperatorSplit");
  for (const auto &v : base->GetVariableVector()) {
    if (has_os && v->IsSet(parthenon::Metadata::GetUserFlag("OperatorSplit"))) continue;
    names.push_back(v->label());
  }
  return names;
}

//! Speed option: hydro/lean_exchange = true -> the hydro-stage halo exchanges use a SHALLOW subset of
//! "base" without the Sn transport's own fields (sn.*: the nfa-wide iterate sn.u and the coupled-solve Krylov vector
//! sn.kc_y, most of the exchanged components with 8 groups), which the Sn solver exchanges itself before every use.
//! Default false = "base" (every FillGhost field), unchanged.
static std::shared_ptr<MeshData<Real>> &HydroExchangeData(Mesh *pmesh, const int i) {
  auto &base = pmesh->mesh_data.GetOrAdd("base", i);
  if (!pmesh->packages.Get("Hydro")->Param<bool>("lean_exchange")) return base;
  // hydro/lean_exchange_rad (default false): ALSO drop the radiation moments rad.* (4 per
  // group: 32 of ~48 exchanged components at 8 groups). Under transport = sn nothing reads their ghost cells (the Sn
  // writeback, matter coupling, P1 seed and history reducers all work on the interior; the ghost fill was for the retired
  // M1 transport), restarts and outputs store interiors only, and their AMR prolongation is piecewise constant.
  static std::vector<std::string> names;   // the field set is fixed for the run
  if (names.empty()) {
    const bool drop_rad = pmesh->packages.Get("Hydro")->Param<bool>("lean_exchange_rad");
    for (const auto &v : base->GetBlockData(0)->GetVariableVector()) {
      if (!v->IsSet(parthenon::Metadata::FillGhost) || v->label().rfind("sn.", 0) == 0) continue;
      if (drop_rad && v->label().rfind("rad.", 0) == 0) continue;
      names.push_back(v->label());
    }
  }
  return pmesh->mesh_data.AddShallow("base_hx", base, names);
}

// Assumes that prim and cons are in sync initially.
// Guarantees that prim and cons are in sync at the end.
void AddSTSTasks(TaskCollection *ptask_coll, Mesh *pmesh, BlockList_t &blocks,
                 const Real tau) {

  auto hydro_pkg = blocks[0]->packages.Get("Hydro");
  auto mindt_diff = hydro_pkg->Param<Real>("dt_diff");
  // EMPTY SUPER-STEPS (diffusion/rkl2_skip_empty, default false = unchanged).
  //
  // When every parabolic term has been taken off the explicit path -- which is exactly what
  // the (removed 2026-09-24) diffusion/ohm_ad_imex did when nothing else remained -- dt_diff is
  // still DBL_MAX here, so the stage count below evaluates to s_rkl = 3 and the cycle runs
  // 2 x 3 RKL2 stages whose diffusive fluxes are all zero, each paying a full boundary
  // exchange and ConsToPrim. Measured: the `STS ratio: 2.27e-314 Taking 3 steps.` line in
  // every implicit log of 2026-09-19/20, and that overhead is inside every implicit cost
  // number those entries quote.
  //
  // NOT bit-identical, which is why it is gated: the stages apply M = 0, but the RKL2
  // recurrence still forms a*u + b*u1 + ... with u == u1, and that need not round back to u.
  if (mindt_diff == std::numeric_limits<Real>::max() &&
      hydro_pkg->Param<bool>("rkl2_skip_empty")) {
    if (parthenon::Globals::my_rank == 0) {
      static bool said = false;
      if (!said) {
        said = true;
        std::cout << "## RKL2: no parabolic term left on the explicit path (dt_diff = inf); "
                     "skipping the empty super-steps (diffusion/rkl2_skip_empty)."
                  << std::endl;
      }
    }
    return;
  }
  // With diffusion/rkl2_freeze_eta the stage-init below refreshes the non-ideal eta
  // cache from the half-start state and the stages jj=2..s reuse it (refill=false).
  const bool refill_eta_stages = !hydro_pkg->Param<bool>("rkl2_freeze_eta");
  // hydro/dual_energy_heat_rkl2 (audit P-03): Joule/AD heating of the dual energy as a source
  // inside the recurrence (DualHeatIntoMY0 / DualHeatStage).
  const bool heat_exact = hydro_pkg->Param<bool>("dual_energy_heat_exact");
  // hydro/dual_energy_heat_exact (DE-F6) replaces the in-recurrence dual source by the exact end-of-half closure.
  const bool heat_rkl2 = hydro_pkg->Param<bool>("dual_energy_heat_rkl2") && !heat_exact;
  // get number of RKL steps
  // eq (21) using half hyperbolic timestep due to Strang split
  int s_rkl =
      static_cast<int>(0.5 * (std::sqrt(9.0 + 16.0 * tau / mindt_diff) - 1.0)) + 1;
  // ensure odd number of stages
  if (s_rkl % 2 == 0) s_rkl += 1;

  if (parthenon::Globals::my_rank == 0) {
    const auto ratio = 2.0 * tau / mindt_diff;
    std::cout << "STS ratio: " << ratio << " Taking " << s_rkl << " steps." << std::endl;
    if (ratio > 400.1) {
      std::cout << "WARNING: ratio is > 400. Proceed at own risk." << std::endl;
    }
  }

  TaskID none(0);

  // Store initial u0 in u1 as "base" will continuously be updated but initial state Y0 is
  // required for each stage.
  TaskRegion &region_copy_out = ptask_coll->AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = region_copy_out[i];
    auto &Y0 = blocks[i]->meshblock_data.Get("u1");
    auto &base = blocks[i]->meshblock_data.Get();
    tl.AddTask(
        none,
        [](MeshBlockData<Real> *dst, MeshBlockData<Real> *src) {
          dst->Get("cons").data.DeepCopy(src->Get("cons").data);
          dst->Get("prim").data.DeepCopy(src->Get("prim").data);
          return TaskStatus::complete;
        },
        Y0.get(), base.get());
  }

  TaskRegion &region_init = ptask_coll->AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = region_init[i];
    auto &base = pmb->meshblock_data.Get();

    // Add extra registers. No-op for existing variables so it's safe to call every
    // time.
    // TODO(pgrete) this allocates all Variables, i.e., prim and cons vector, but only a
    // subset is actually needed. Streamline to allocate only required vars.
    const auto reg_vars = RegisterVarNames(base.get(), hydro_pkg->Param<bool>("lean_registers"));
    pmb->meshblock_data.Add("MY0", base, reg_vars);
    pmb->meshblock_data.Add("Yjm2", base, reg_vars);
  }

  const int num_partitions = pmesh->DefaultNumPartitions();

  TaskRegion &region_calc_fluxes_step_init = ptask_coll->AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region_calc_fluxes_step_init[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    const auto any = parthenon::BoundaryType::any;
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, HydroExchangeData(pmesh, i));
    auto start_flxcor_recv =
        tl.AddTask(none, parthenon::StartReceiveFluxCorrections, base);

    // Reset flux arrays (not guaranteed to be zero)
    auto reset_fluxes = tl.AddTask(none, ResetFluxes, base.get());

    // Calculate the diffusive fluxes for Y0 (here still "base" as nothing has been
    // updated yet) so that we can store the result as MY0 and reuse later
    // (in every subsetp). Parabolic terms only: in the mixed rkl2+Hall mode the
    // dispersive Hall EMF is applied unsplit with the hyperbolic fluxes, not here.
    // Always refreshes the eta cache (refill=true): this is the half-start state the
    // frozen stages reuse.
    auto hydro_diff_fluxes = tl.AddTask(reset_fluxes, CalcDiffFluxes, hydro_pkg.get(),
                                        base.get(), DiffTermSet::parabolic, true);

    TaskID diff_ready = hydro_diff_fluxes;
    auto send_flx = tl.AddTask(diff_ready, parthenon::LoadAndSendFluxCorrections, base);
    auto recv_flx =
        tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, base);
    auto set_flx =
        tl.AddTask(recv_flx | diff_ready, parthenon::SetFluxCorrections, base);

    auto &Y0 = pmesh->mesh_data.GetOrAdd("u1", i);
    auto &MY0 = pmesh->mesh_data.GetOrAdd("MY0", i);
    auto &Yjm2 = pmesh->mesh_data.GetOrAdd("Yjm2", i);

    auto init_MY0 = tl.AddTask(set_flx, parthenon::Update::FluxDivergence<MeshData<Real>>,
                               base.get(), MY0.get());
    if (heat_rkl2) init_MY0 = tl.AddTask(init_MY0, DualHeatIntoMY0, base.get(), MY0.get());
    if (heat_exact) init_MY0 = tl.AddTask(init_MY0, Hydro::DualExactInit, base.get(), s_rkl, tau);

    // Initialize Y0 and Y1 and the recursion relation starting with j = 2 needs data from
    // the two preceeding stages.
    auto rkl2_step_first = tl.AddTask(init_MY0, RKL2StepFirst, Y0.get(), base.get(),
                                      Yjm2.get(), MY0.get(), s_rkl, tau);

    TaskID after_step = rkl2_step_first;
    if (heat_exact && s_rkl == 1)
      after_step = tl.AddTask(after_step, Hydro::DualExactFinish, base.get(), Y0.get());

    // Update ghost cells of Y1 (as MY1 is calculated for each Y_j).
    // Y1 stored in "base", see rkl2_step_first task.
    // Update ghost cells (local and non local), prolongate and apply bound cond.
    // TODO(someone) experiment with split (local/nonlocal) comms with respect to
    // performance for various tests (static, amr, block sizes) and then decide on the
    // best impl. Go with default call (split local/nonlocal) for now.
    // TODO(pgrete) optimize (in parthenon) to only send subset of updated vars
    auto bounds_exchange = parthenon::AddBoundaryExchangeTasks(
        after_step | start_bnd, tl, HydroExchangeData(pmesh, i), pmesh->multilevel);

    // P2: only the LAST RKL2 sub-stage's RHS is read by a Poisson solve; the other s_rkl-1
    // are overwritten before any read. s_rkl == 1 would make THIS stage-init call the last
    // one, hence the (s_rkl > 1) test rather than an unconditional false.
    auto rhs_gate_1 = tl.AddTask(bounds_exchange, SelfGravity::SetGravRhsConsumed, pmesh,
                                 !(s_rkl > 1));
    auto fill_derived_1 = tl.AddTask(
        rhs_gate_1, parthenon::Update::FillDerived<MeshData<Real>>, base.get());

  }

  // Compute coefficients. Meyer+2012 eq. (16)
  Real b_j = 1. / 3.;
  Real b_jm1 = 1. / 3.;
  Real b_jm2 = 1. / 3.;
  Real w1 = 4. / (static_cast<Real>(s_rkl) * static_cast<Real>(s_rkl) +
                  static_cast<Real>(s_rkl) - 2.);
  Real mu_j, nu_j, j, mu_tilde_j, gamma_tilde_j;

  // RKL loop
  for (int jj = 2; jj <= s_rkl; jj++) {
    j = static_cast<Real>(jj);
    b_j = (j * j + j - 2.0) / (2 * j * (j + 1.0));
    mu_j = (2.0 * j - 1.0) / j * b_j / b_jm1;
    nu_j = -(j - 1.0) / j * b_j / b_jm2;
    mu_tilde_j = mu_j * w1;
    gamma_tilde_j = -(1.0 - b_jm1) * mu_tilde_j; // -a_jm1*mu_tilde_j

    TaskRegion &region_calc_fluxes_step_other = ptask_coll->AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = region_calc_fluxes_step_other[i];
      auto &base = pmesh->mesh_data.GetOrAdd("base", i);

      // Only need boundaries for base as it's the only "active" container exchanging
      // data/fluxes with neighbors. All other containers are passive (i.e., data is only
      // used but not exchanged).
      const auto any = parthenon::BoundaryType::any;
      auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, HydroExchangeData(pmesh, i));
      auto start_flxcor_recv =
          tl.AddTask(none, parthenon::StartReceiveFluxCorrections, base);

      // Reset flux arrays (not guaranteed to be zero)
      auto reset_fluxes = tl.AddTask(none, ResetFluxes, base.get());

      // Calculate the diffusive fluxes for Yjm1 (here u1); parabolic terms only (the
      // dispersive Hall EMF is never super-time-stepped, see AddSTSTasks stage-init).
      // refill_eta_stages=false (diffusion/rkl2_freeze_eta) reuses the half-start eta.
      auto hydro_diff_fluxes = tl.AddTask(reset_fluxes, CalcDiffFluxes, hydro_pkg.get(),
                                          base.get(), DiffTermSet::parabolic,
                                          refill_eta_stages);

      TaskID diff_ready = hydro_diff_fluxes;

      auto send_flx = tl.AddTask(diff_ready, parthenon::LoadAndSendFluxCorrections, base);
      auto recv_flx =
          tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, base);
      auto set_flx =
          tl.AddTask(recv_flx | diff_ready, parthenon::SetFluxCorrections, base);

      auto &Y0 = pmesh->mesh_data.GetOrAdd("u1", i);
      auto &MY0 = pmesh->mesh_data.GetOrAdd("MY0", i);
      auto &Yjm2 = pmesh->mesh_data.GetOrAdd("Yjm2", i);

      auto rkl2_step_other =
          tl.AddTask(set_flx, RKL2StepOther, Y0.get(), base.get(), Yjm2.get(), MY0.get(),
                     mu_j, nu_j, mu_tilde_j, gamma_tilde_j, tau);

      TaskID after_step = rkl2_step_other;
      // Before the exchange/FillDerived, so `prim` is still Y_{j-1}'s.
      if (heat_rkl2)
        after_step = tl.AddTask(after_step, DualHeatStage, base.get(), mu_tilde_j * tau);
      if (heat_exact) {
        after_step = tl.AddTask(after_step, Hydro::DualExactStage, base.get(), mu_j, nu_j, mu_tilde_j,
                                gamma_tilde_j, tau);
        if (jj == s_rkl) after_step = tl.AddTask(after_step, Hydro::DualExactFinish, base.get(), Y0.get());
      }
      // update ghost cells of base (currently storing Yj)
      // Update ghost cells (local and non local), prolongate and apply bound cond.
      // TODO(someone) experiment with split (local/nonlocal) comms with respect to
      // performance for various tests (static, amr, block sizes) and then decide on the
      // best impl. Go with default call (split local/nonlocal) for now.
      // TODO(pgrete) optimize (in parthenon) to only send subset of updated vars
      auto bounds_exchange = parthenon::AddBoundaryExchangeTasks(
          after_step | start_bnd, tl, HydroExchangeData(pmesh, i), pmesh->multilevel);

      // P2: only jj == s_rkl (the final sub-stage) produces an RHS a solve reads.
      auto rhs_gate_j = tl.AddTask(bounds_exchange, SelfGravity::SetGravRhsConsumed, pmesh,
                                   jj == s_rkl);
      auto fill_derived_j = tl.AddTask(
          rhs_gate_j, parthenon::Update::FillDerived<MeshData<Real>>, base.get());
    }

    b_jm2 = b_jm1;
    b_jm1 = b_j;
  }
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection HydroDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  auto hydro_pkg = blocks[0]->packages.Get("Hydro");

  TaskID none(0);
  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();

  const int num_partitions = pmesh->DefaultNumPartitions();

  // hydro/ghost_sync_rounds (default 0 = off): N rounds of (halo exchange + FillDerived) at the start
  // of every step. Coarse->fine messages carry the sender's own ghost cells, whose history differs
  // between a continued and a resumed run; one round makes the halo a function of the interior only
  // (restart reproducibility, 2026-09-22/23).
  const int ghost_rounds = hydro_pkg->Param<int>("ghost_sync_rounds");
  // The ghost-sync FillDerived passes may build the Poisson RHS the next solve reads, so they
  // keep the multipole reduction on.
  const bool sync_rhs_consumed =
      !(hydro_pkg->Param<DiffInt>("diffint") == DiffInt::rkl2 &&
        !(hydro_pkg->AllParams().hasKey("rkl2_skip_empty") &&
          hydro_pkg->Param<bool>("rkl2_skip_empty")));
  if (stage == 1 && ghost_rounds > 0) {
    for (int r = 0; r < ghost_rounds; r++) {
      TaskRegion &gr = tc.AddRegion(num_partitions);
      for (int i = 0; i < num_partitions; i++) {
        auto &tl = gr[i];
        auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
        auto bnd = parthenon::AddBoundaryExchangeTasks(TaskID(0), tl, HydroExchangeData(pmesh, i), pmesh->multilevel);
        bnd = tl.AddTask(bnd, SelfGravity::SetGravRhsConsumed, pmesh, sync_rhs_consumed);
        tl.AddTask(bnd, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
      }
    }
  }

  // Dual energy: the RKL2 half just changed the total energy through the diffusion flux
  // divergence; tell the advected internal energy how much of that was dissipation.
  auto add_dual_diff_heating = [&](const Real dt_half) {
    if (!hydro_pkg->Param<bool>("dual_energy")) return;
    // hydro/dual_energy_heat_rkl2: already integrated inside the RKL2 stages.
    if (hydro_pkg->Param<bool>("dual_energy_heat_rkl2")) return;
    TaskRegion &r = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      r[i].AddTask(TaskID(0), DualEnergyDiffusionHeating, md.get(), dt_half);
    }
  };

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    // Using "base" as u0, which already exists (and returned by using plain Get())
    auto &u0 = pmb->meshblock_data.Get();

    // Create meshblock data for register u1.
    // This is a noop if u1 already exists.
    // TODO(pgrete) update to derive from other quanity as u1 does not require fluxes
    if (stage == 1) {
      pmb->meshblock_data.Add("u1", u0,
                              RegisterVarNames(u0.get(), hydro_pkg->Param<bool>("lean_registers")));
    }
  }

  // First add split sources before the main time integration
  if (stage == 1) {
    // If any tasks modify the conserved variables before this place, then
    // the STS tasks should be updated to not assume prim and cons are in sync.
    const auto &diffint = hydro_pkg->Param<DiffInt>("diffint");
    if (diffint == DiffInt::rkl2) {
      AddSTSTasks(&tc, pmesh, blocks, 0.5 * tm.dt);
      add_dual_diff_heating(0.5 * tm.dt);
    }
    TaskRegion &strang_init_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = strang_init_region[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);

      // Add initial Strang split source terms, i.e., a dt/2 update
      // IMPORTANT 1: This task must also update `prim` and `cons` variables so that
      // the source term is applied to all active registers in the flux calculation.
      // IMPORTANT 2: The tasks should work using `cons` variables as input as in the
      // final step, `prim` are not updated yet from the flux calculation.
      tl.AddTask(none, AddSplitSourcesStrang, mu0.get(), tm);
    }
  }

  // Now start the main time integration by resetting the registers
  TaskRegion &async_region_init_int = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_init_int[i];
    auto &u0 = pmb->meshblock_data.Get();
    // init u1, see (11) in Athena++ method paper
    if (stage == 1) {
      auto &u1 = pmb->meshblock_data.Get("u1");
      auto init_u1 = tl.AddTask(
          none,
          [](MeshBlockData<Real> *u0, MeshBlockData<Real> *u1, bool copy_prim) {
            u1->Get("cons").data.DeepCopy(u0->Get("cons").data);
            if (copy_prim) {
              u1->Get("prim").data.DeepCopy(u0->Get("prim").data);
            }
            return TaskStatus::complete;
          },
          // First order flux correction needs the original prim variables in the
          // during the correction.
          u0.get(), u1.get(), hydro_pkg->Param<bool>("first_order_flux_correct"));
    }
  }

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);

    const auto any = parthenon::BoundaryType::any;
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, HydroExchangeData(pmesh, i));
    auto start_flxcor_recv =
        tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mu0);

    const auto flux_str = (stage == 1) ? "flux_first_stage" : "flux_other_stage";
    FluxFun_t *calc_flux_fun = hydro_pkg->Param<FluxFun_t *>(flux_str);
    auto calc_flux = tl.AddTask(none, calc_flux_fun, mu0);

    // TODO(pgrete) figure out what to do about the sources from the first stage
    // that are potentially disregarded when the (m)hd fluxes are corrected in the second
    // stage.
    TaskID first_order_flux_correct = calc_flux;
    if (hydro_pkg->Param<bool>("first_order_flux_correct")) {
      auto *first_order_flux_correct_fun =
          hydro_pkg->Param<FirstOrderFluxCorrectFun_t *>("first_order_flux_correct_fun");
      first_order_flux_correct =
          tl.AddTask(calc_flux, first_order_flux_correct_fun, mu0.get(), mu1.get(),
                     integrator->gam0[stage - 1], integrator->gam1[stage - 1],
                     integrator->beta[stage - 1] * integrator->dt);
    }

    // hydro/double_flux (E-F10): own-EOS energy-flux face sums, computed from the stage's prim BEFORE the flux
    // correction is sent (they are fluxes of dflux_xyz and get flux-corrected with the hydro fluxes); stage 1 also
    // freezes the per-cell EOS from the step-start state. Off -> graph unchanged.
    const bool dflux = hydro_pkg->Param<bool>("double_flux");
    TaskID flx_done = first_order_flux_correct;
    // hydro/dual_energy_flux = density (DE-F3): rebuild the dual flux from the reconstructed rho*e on PLM stages
    // (the dc VL2 predictor already gives the legacy flux exactly), before the flux correction is sent.
    if (hydro_pkg->Param<bool>("dual_energy_flux_density") &&
        !(stage == 1 && flux_str == std::string("flux_first_stage") &&
          hydro_pkg->Param<FluxFun_t *>("flux_first_stage") != hydro_pkg->Param<FluxFun_t *>("flux_other_stage")))
      flx_done = tl.AddTask(flx_done, Hydro::DualEnergyDensityFlux, mu0.get());
    if (dflux)
      flx_done = flx_done | tl.AddTask(none, Hydro::DoubleFluxCoefficients, mu0.get(), stage);
    // face volume fluxes of the dual double flux / face pdV (flux-corrected with the mass flux; dual_double_flux.cpp)
    if (hydro_pkg->Param<bool>("dual_vflux")) {
      const bool vdc = (stage == 1 && hydro_pkg->Param<FluxFun_t *>("flux_first_stage") !=
                                          hydro_pkg->Param<FluxFun_t *>("flux_other_stage")) ||
                       hydro_pkg->Param<Reconstruction>("reconstruction") == Reconstruction::dc;
      flx_done = tl.AddTask(flx_done, Hydro::DualFaceVolumeFlux, mu0.get(), vdc);
    }
    auto send_flx = tl.AddTask(flx_done, parthenon::LoadAndSendFluxCorrections, mu0);
    auto recv_flx = tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, mu0);
    auto set_flx = tl.AddTask(recv_flx | flx_done, parthenon::SetFluxCorrections, mu0);

    // B1-remainder: clamp inward domain-boundary fluxes. MUST sit after SetFluxCorrections (so
    // the AMR flux-correction cannot reintroduce an inward flux the clamp already removed) and
    // before the divergence (so the clamped values are what is actually applied). Applied only
    // on the MAIN hydro update.
    auto flx_ready = set_flx;
    if (hydro_pkg->Param<bool>("boundary_flux_clamp")) {
      flx_ready = tl.AddTask(set_flx, Hydro::ClampBoundaryFluxes, mu0.get());
    }

    // compute the divergence of fluxes of conserved variables
    auto update = tl.AddTask(
        flx_ready, parthenon::Update::UpdateWithFluxDivergence<MeshData<Real>>, mu0.get(),
        mu1.get(), integrator->gam0[stage - 1], integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);
    if (dflux)
      update = tl.AddTask(update, Hydro::DoubleFluxCorrect, mu0.get(),
                          integrator->beta[stage - 1] * integrator->dt);
    // hydro/dual_energy_double_flux (DE-F3 part (b)): the same own-EOS treatment for the dual energy
    if (dflux && hydro_pkg->Param<bool>("dual_energy_double_flux"))
      update = tl.AddTask(update, Hydro::DualDoubleFluxCorrect, mu0.get(),
                          integrator->beta[stage - 1] * integrator->dt, stage);

    // WP-6: accumulate the APPLIED domain-face mass flux here, the only point where the flux arrays
    // hold exactly what `update` consumed (later history callbacks see cleared fluxes). Weighted by
    // the same beta*dt, so mass(t) - mass(0) = -cons-Mout-applied(t). See cons_diag.hpp. Gated on
    // hydro/cons_diag (off: after_update == update, graph unchanged).
    auto after_update = update;
    {
      auto hydro_pkg = pmesh->packages.Get("Hydro");
      if (hydro_pkg->Param<bool>("cons_diag")) {
        after_update = tl.AddTask(
            update,
            [](MeshData<Real> *md, StateDescriptor *pkg, const Real bdt,
               const bool is_last) {
              // Rank-local running total; the history callback returns it and Parthenon's
              // sum reduction supplies the cross-rank sum. Safe to mutate a Param here only
              // because cons_diag asserts DefaultNumPartitions() == 1 at registration, so
              // this task is unique per rank per stage -- see hydro.cpp.
              const Real f = Diagnostics::AppliedMassFluxSurface(md);
              pkg->UpdateParam("cons_Mout_applied",
                               pkg->Param<Real>("cons_Mout_applied") + f * bdt);
              // LAST-STAGE-ONLY companion. Parthenon's low-storage update is
              //   u0 <- gam0*u0 + gam1*u1 + beta*dt*L(u0),
              // and vl2 has gam0 = 0 on its final stage, so the final stage OVERWRITES u0
              // and every earlier stage's flux increment is discarded. For vl2 the applied
              // flux is therefore the last stage's alone, and summing all stages (the column
              // above) necessarily over-counts. Recording both makes the decomposition a
              // measurement instead of an argument about coefficients: whichever column
              // closes mass(t)-mass(0) identifies the correct weighting empirically.
              if (is_last) {
                pkg->UpdateParam("cons_Mout_applied_last",
                                 pkg->Param<Real>("cons_Mout_applied_last") + f * bdt);
              }
              return TaskStatus::complete;
            },
            mu0.get(), hydro_pkg.get(),
            integrator->beta[stage - 1] * integrator->dt,
            stage == integrator->nstages);
      }
    }

    // Add non-operator split source terms (cooling, MHD, problem-defined).
    auto source_unsplit = tl.AddTask(after_update, AddUnsplitSources, mu0.get(), tm,
                                     integrator->beta[stage - 1] * integrator->dt);
    // BE collapse barotropic cooling (only active if problem_id = collapse_be)
    auto after_cooling = source_unsplit;
    {
      auto hydro_pkg = pmesh->packages.Get("Hydro");
      // WP-13: this gate used to fail SILENTLY on every restart. It is now backed by a
      // startup assert in the HydroDriver constructor (which has `pin`), so a missing key is
      // a hard error rather than a quietly-dropped task. See the note there.
      if (hydro_pkg->AllParams().hasKey("collapse_be_rhocrit")) {
        after_cooling = tl.AddTask(source_unsplit, collapse_be::ApplyBarotropicCooling,
                                   mu0.get(), tm,
                                   integrator->beta[stage - 1] * integrator->dt);
      }
    }
    // Then change the next task's dependency from source_unsplit to after_cooling

    auto source_split_first_order = after_cooling;

    if (stage == integrator->nstages) {
      // Audit fix #7: the Strang split source must depend on after_cooling, NOT
      // source_unsplit. Both cooling and the split sources mutate cons; branching the
      // split chain from source_unsplit left them as unordered siblings (a GPU-async
      // race on the ambient momentum that barotropic cooling zeroes). Single chain:
      // update -> source_unsplit -> (cooling) -> strang -> first_order -> bnd_exchange.
      auto source_split_strang_final =
          tl.AddTask(after_cooling, AddSplitSourcesStrang, mu0.get(), tm);
      source_split_first_order =
          tl.AddTask(source_split_strang_final, AddSplitSourcesFirstOrder, mu0.get(), tm);
    }

    parthenon::AddBoundaryExchangeTasks(source_split_first_order | start_bnd, tl, HydroExchangeData(pmesh, i),
                                        pmesh->multilevel);
  }
  // --- END of single_tasklist_per_pack_region loop ---
  // ---- Self-gravity: a STAGE-CONSISTENT source (audit fix #2) --------------------------------
  // Each stage solves Poisson for the RHS the preceding FillDerived built -- the STAGE-START
  // density (rho^n on the predictor, rho* on the corrector) -- and ApplyGravitySource kicks with
  // that stage's prim density, beta*dt weight and mass flux (flux-weighted work). That is the VL2
  // midpoint rule for the corrector, i.e. 2nd order in time. Cost: one Poisson solve per stage.
  auto self_gravity_pkg = pmesh->packages.AllPackages().count("self_gravity") > 0
                              ? pmesh->packages.Get("self_gravity")
                              : nullptr;
  if (self_gravity_pkg != nullptr) {
    // Solve Poisson: this adds its own TaskRegion with num_partitions task lists.
    SelfGravity::SolvePoisson(tc, pmesh);

    // Apply gravitational source term per partition, weighted by this stage's beta*dt.
    TaskRegion &sg_source_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = sg_source_region[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      tl.AddTask(none, SelfGravity::ApplyGravitySource, mu0.get(), tm,
                 integrator->beta[stage - 1] * integrator->dt);
    }
  }

  // Radiation (Sn): operator-split transport + matter coupling on the final stage only.
  auto radiation_pkg = pmesh->packages.AllPackages().count("radiation") > 0
                           ? pmesh->packages.Get("radiation")
                           : nullptr;
  if (stage == integrator->nstages && radiation_pkg != nullptr) {
    // Adds its own TaskRegion.
    Radiation::AddRadiationTasks(tc, pmesh, integrator->dt);
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  // Operator-split sources modify INTERIOR conserved variables AFTER the stage boundary exchange
  // above (self-gravity on every stage, radiation matter coupling on the final stage). Refresh ghosts before FillDerived so the primitive ghosts
  // (the gravity-updated momentum used by the next stage's reconstruction) reflect the sourced
  // interior, not stale ghosts. The final stage refreshes too, as in the validated code.
  const bool sourced_interior =
      (stage == integrator->nstages) || (self_gravity_pkg != nullptr);
  // P2: this FillDerived's RHS is read by the NEXT stage's SolvePoisson -- except on the final
  // stage, where the RKL2 STS block that follows issues the last FillDerived of the cycle. With
  // rkl2 off there is no such block, so the final-stage call is the one that feeds the next
  // cycle and must reduce.
  const bool rhs_consumed_r3 =
      (stage != integrator->nstages) ||
      (hydro_pkg->Param<DiffInt>("diffint") != DiffInt::rkl2);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    TaskID pre = none;
    if (sourced_interior)
      pre = parthenon::AddBoundaryExchangeTasks(none, tl, HydroExchangeData(pmesh, i), pmesh->multilevel);
    pre = tl.AddTask(pre, SelfGravity::SetGravRhsConsumed, pmesh, rhs_consumed_r3);
    auto fill_derived =
        tl.AddTask(pre, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
  }

  const auto &diffint = hydro_pkg->Param<DiffInt>("diffint");
  // If any tasks modify the conserved variables before this place and after FillDerived,
  // then the STS tasks should be updated to not assume prim and cons are in sync.
  if (diffint == DiffInt::rkl2 && stage == integrator->nstages) {
    AddSTSTasks(&tc, pmesh, blocks, 0.5 * tm.dt);
    add_dual_diff_heating(0.5 * tm.dt);
  }

  // hydro/dual_energy_sync_step_end: end the final stage with halo exchange + FillDerived so a
  // continued run holds exactly the state a resumed run rebuilds (the dual heating writes cons after
  // the STS FillDerived, and tabulated-EOS pressure floors made ConsToPrim non-idempotent). Default false.
  if (stage == integrator->nstages && hydro_pkg->Param<bool>("dual_energy") &&
      hydro_pkg->Param<bool>("dual_energy_sync_step_end")) {
    TaskRegion &sync_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = sync_region[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto bnd = parthenon::AddBoundaryExchangeTasks(none, tl, HydroExchangeData(pmesh, i), pmesh->multilevel);
      bnd = tl.AddTask(bnd, SelfGravity::SetGravRhsConsumed, pmesh, sync_rhs_consumed);
      tl.AddTask(bnd, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
    }
  }
  // Single task in single (serial) region to reset global vars used in reductions in the
  // first stage.
  // TODO(pgrete) check if we logically need this reset or if we can reset within the
  // timestep task
  if (stage == integrator->nstages &&
      (hydro_pkg->Param<bool>("calc_c_h") ||
       hydro_pkg->Param<DiffInt>("diffint") != DiffInt::none)) {
    TaskRegion &reset_reduction_vars_region = tc.AddRegion(1);
    auto &tl = reset_reduction_vars_region[0];
    tl.AddTask(
        none,
        [](StateDescriptor *hydro_pkg) {
          hydro_pkg->UpdateParam("mindx", std::numeric_limits<Real>::max());
          hydro_pkg->UpdateParam("dt_hyp", std::numeric_limits<Real>::max());
          hydro_pkg->UpdateParam("dt_diff", std::numeric_limits<Real>::max());
          return TaskStatus::complete;
        },
        hydro_pkg.get());
  }

  if (stage == integrator->nstages) {
    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_dt = tl.AddTask(none, parthenon::Update::EstimateTimestep<MeshData<Real>>,
                               mu0.get());
    }
  }

  auto tracers_pkg = pmesh->packages.Get("tracers");
  // First order operator split tracer advection
  if (stage == integrator->nstages && tracers_pkg->Param<bool>("enabled")) {
    const std::string swarm_name = "tracers";
    TaskRegion &sync_region_tr = tc.AddRegion(1);
    {
      for (auto &pmb : blocks) {
        auto &tl = sync_region_tr[0];
        auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
        auto reset_comms =
            tl.AddTask(none, &SwarmContainer::ResetCommunication, sd.get());
      }
    }

    TaskRegion &async_region_tr = tc.AddRegion(blocks.size());
    for (int n = 0; n < blocks.size(); n++) {
      auto &tl = async_region_tr[n];
      auto &pmb = blocks[n];
      auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
      auto &mbd0 = pmb->meshblock_data.Get("base");
      auto tracer_advect =
          tl.AddTask(none, Tracers::AdvectTracers, mbd0.get(), integrator->dt);

      auto send = tl.AddTask(tracer_advect, &SwarmContainer::Send, sd.get(),
                             BoundaryCommSubset::all);

      auto receive =
          tl.AddTask(send, &SwarmContainer::Receive, sd.get(), BoundaryCommSubset::all);
    }
    // TODO(pgrete) Fix/cleanup once we got swarm packs.
    // We need just a single region with a single task in order to be able to use plain
    // MPI reductions (rather than Parthenon provided reduction tasks that work with
    // arbitrary packs).
    PARTHENON_REQUIRE_THROWS(num_partitions == 1,
                             "Only packs_per_rank=1 currently supported for tracers.")
    TaskRegion &single_tasklist_per_pack_region_4 = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = single_tasklist_per_pack_region_4[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto fill = tl.AddTask(none, Tracers::FillTracers, mu0.get(), tm);
      if (Tracers::ProblemFillTracers != nullptr) {
        fill =
            tl.AddTask(fill, Tracers::ProblemFillTracers, mu0.get(), tm, integrator->dt);
      }
    }
  }

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &u0 = blocks[i]->meshblock_data.Get("base");
      auto tag_refine =
          tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<Real>>, u0.get());
    }
  }

  return tc;
}
} // namespace Hydro
