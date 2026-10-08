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
#include "hydro_driver.hpp"
#include "../diagnostics/cons_diag.hpp"
#include "../self_gravity/self_gravity.hpp"
#include "../pgen/pgen.hpp"

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
  const bool sync_rhs_consumed = true;
  if (stage == 1 && ghost_rounds > 0) {
    for (int r = 0; r < ghost_rounds; r++) {
      TaskRegion &gr = tc.AddRegion(num_partitions);
      for (int i = 0; i < num_partitions; i++) {
        auto &tl = gr[i];
        auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
        auto bnd = parthenon::AddBoundaryExchangeTasks(TaskID(0), tl, mu0, pmesh->multilevel);
        bnd = tl.AddTask(bnd, SelfGravity::SetGravRhsConsumed, pmesh, sync_rhs_consumed);
        tl.AddTask(bnd, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
      }
    }
  }

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
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mu0);
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

    parthenon::AddBoundaryExchangeTasks(source_split_first_order | start_bnd, tl, mu0,
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

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  // Operator-split sources modify INTERIOR conserved variables AFTER the stage boundary exchange
  // above (self-gravity on every stage). Refresh ghosts before FillDerived so the primitive ghosts
  // (the gravity-updated momentum used by the next stage's reconstruction) reflect the sourced
  // interior, not stale ghosts. The final stage refreshes too, as in the validated code.
  const bool sourced_interior =
      (stage == integrator->nstages) || (self_gravity_pkg != nullptr);
  // This FillDerived's RHS is read by the next SolvePoisson (next stage or next cycle).
  const bool rhs_consumed_r3 = true;
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    TaskID pre = none;
    if (sourced_interior)
      pre = parthenon::AddBoundaryExchangeTasks(none, tl, mu0, pmesh->multilevel);
    pre = tl.AddTask(pre, SelfGravity::SetGravRhsConsumed, pmesh, rhs_consumed_r3);
    auto fill_derived =
        tl.AddTask(pre, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
  }

  // hydro/dual_energy_sync_step_end: end the final stage with halo exchange + FillDerived so a
  // continued run holds exactly the state a resumed run rebuilds (tabulated-EOS pressure floors made
  // ConsToPrim non-idempotent). Default false.
  if (stage == integrator->nstages && hydro_pkg->Param<bool>("dual_energy") &&
      hydro_pkg->Param<bool>("dual_energy_sync_step_end")) {
    TaskRegion &sync_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = sync_region[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto bnd = parthenon::AddBoundaryExchangeTasks(none, tl, mu0, pmesh->multilevel);
      bnd = tl.AddTask(bnd, SelfGravity::SetGravRhsConsumed, pmesh, sync_rhs_consumed);
      tl.AddTask(bnd, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
    }
  }
  // Single task in single (serial) region to reset global vars used in reductions in the
  // first stage.
  // TODO(pgrete) check if we logically need this reset or if we can reset within the
  // timestep task
  if (stage == integrator->nstages && hydro_pkg->Param<bool>("calc_c_h")) {
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
