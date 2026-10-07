//========================================================================================
// AthenaPK - Self-gravity package
// Ported from Artemis (LANL, BSD-licensed).
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_utils.hpp>

#include "../diagnostics/grav_diag.hpp"
#include "../main.hpp"          // for IDN
#include "self_gravity.hpp"
#include "poisson_equation.hpp"
#include "multipole.hpp"
#include <solvers/internal_prolongation.hpp>

namespace SelfGravity {

using namespace parthenon::BoundaryFunction;
using namespace parthenon::package::prelude;

// Selector for BC enrollment — matches any variable in the "grav" namespace.
struct any_grav : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_grav(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "grav[.].*"; }
};

// Zero Dirichlet (phi = 0 on face): FixedFace BC enforces phi(face) = 0 via
// linear extrapolation into ghosts. Constant = 0 here.
template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto DirZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    GenericBC<DIR, SIDE, BCType::FixedFace, any_grav>(rc, coarse, 0.0);
  };
}

// Neumann (dphi/dn = 0): just copy from interior, which is Outflow semantics.
template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto NeuZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    GenericBC<DIR, SIDE, BCType::Outflow, any_grav>(rc, coarse, 0.0);
  };
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("self_gravity");
  const std::string block_name = "self_gravity";

  // --- Coordinate system check -----------------------------------------------
  // This port supports Cartesian only. AthenaPK currently only supports
  // UniformCartesian anyway, so we just assert and move on.
  // (Parthenon's Coordinates_t is a typedef; check it here at runtime.)
  PARTHENON_REQUIRE(
      typeid(parthenon::Coordinates_t) == typeid(parthenon::UniformCartesian),
      "Self-gravity currently only supports UniformCartesian coordinates.");

  // --- Multigrid required ----------------------------------------------------
  const bool mg_enabled =
      pin->GetOrAddBoolean("parthenon/mesh", "multigrid", false);
  PARTHENON_REQUIRE(
      mg_enabled,
      "Self-gravity requires parthenon/mesh/multigrid = true. Set it in your input file.");

  // --- 4 pi G ----------------------------------------------------------------
  // v1: require units_override. Integrating with AthenaPK's Units class is a
  // follow-on. For G in code units, user sets four_pi_G explicitly.
  const bool units_override =
      pin->GetOrAddBoolean(block_name, "units_override", true);
  PARTHENON_REQUIRE(units_override,
                    "self_gravity/units_override must be true in v1. "
                    "Compute 4*pi*G in your code units and set four_pi_G directly.");
  const Real four_pi_G = pin->GetOrAddReal(block_name, "four_pi_G", 1.0);
  pkg->AddParam("four_pi_G", four_pi_G);

  // --- Jeans swindle ---------------------------------------------------------
  auto is_periodic = [&](const std::string &face) {
    return pin->GetOrAddString("parthenon/mesh", face, "outflow") == "periodic";
  };
  const bool fully_periodic =
      is_periodic("ix1_bc") && is_periodic("ox1_bc") &&
      is_periodic("ix2_bc") && is_periodic("ox2_bc") &&
      is_periodic("ix3_bc") && is_periodic("ox3_bc");
  const bool use_swindle =
      pin->GetOrAddBoolean(block_name, "use_swindle", fully_periodic);
  PARTHENON_REQUIRE(!fully_periodic || use_swindle,
                    "Fully periodic mesh BCs require Jeans swindle (use_swindle = true).");
  if (use_swindle && !fully_periodic) {
    PARTHENON_WARN("Jeans swindle enabled on non-fully-periodic mesh. Proceed carefully.");
  }
  pkg->AddParam("use_swindle", use_swindle);

  // --- phi boundary condition enrollment -------------------------------------
  auto valid_bc = [](const std::string &s) {
    return s == "default" || s == "zero" || s == "neumann" || s == "multipole";
  };
  const std::string b_ix1 = pin->GetOrAddString(block_name, "ix1_bc", "default");
  const std::string b_ox1 = pin->GetOrAddString(block_name, "ox1_bc", "default");
  const std::string b_ix2 = pin->GetOrAddString(block_name, "ix2_bc", "default");
  const std::string b_ox2 = pin->GetOrAddString(block_name, "ox2_bc", "default");
  const std::string b_ix3 = pin->GetOrAddString(block_name, "ix3_bc", "default");
  const std::string b_ox3 = pin->GetOrAddString(block_name, "ox3_bc", "default");
  PARTHENON_REQUIRE(valid_bc(b_ix1), "Invalid self_gravity/ix1_bc: " + b_ix1);
  PARTHENON_REQUIRE(valid_bc(b_ox1), "Invalid self_gravity/ox1_bc: " + b_ox1);
  PARTHENON_REQUIRE(valid_bc(b_ix2), "Invalid self_gravity/ix2_bc: " + b_ix2);
  PARTHENON_REQUIRE(valid_bc(b_ox2), "Invalid self_gravity/ox2_bc: " + b_ox2);
  PARTHENON_REQUIRE(valid_bc(b_ix3), "Invalid self_gravity/ix3_bc: " + b_ix3);
  PARTHENON_REQUIRE(valid_bc(b_ox3), "Invalid self_gravity/ox3_bc: " + b_ox3);

  using BF = parthenon::BoundaryFace;
  constexpr auto LL = BCSide::Inner;
  constexpr auto RR = BCSide::Outer;
  if (b_ix1 == "zero") pkg->UserBoundaryFunctions[BF::inner_x1].push_back(DirZ<parthenon::X1DIR, LL>());
  if (b_ox1 == "zero") pkg->UserBoundaryFunctions[BF::outer_x1].push_back(DirZ<parthenon::X1DIR, RR>());
  if (b_ix2 == "zero") pkg->UserBoundaryFunctions[BF::inner_x2].push_back(DirZ<parthenon::X2DIR, LL>());
  if (b_ox2 == "zero") pkg->UserBoundaryFunctions[BF::outer_x2].push_back(DirZ<parthenon::X2DIR, RR>());
  if (b_ix3 == "zero") pkg->UserBoundaryFunctions[BF::inner_x3].push_back(DirZ<parthenon::X3DIR, LL>());
  if (b_ox3 == "zero") pkg->UserBoundaryFunctions[BF::outer_x3].push_back(DirZ<parthenon::X3DIR, RR>());
  if (b_ix1 == "neumann") pkg->UserBoundaryFunctions[BF::inner_x1].push_back(NeuZ<parthenon::X1DIR, LL>());
  if (b_ox1 == "neumann") pkg->UserBoundaryFunctions[BF::outer_x1].push_back(NeuZ<parthenon::X1DIR, RR>());
  if (b_ix2 == "neumann") pkg->UserBoundaryFunctions[BF::inner_x2].push_back(NeuZ<parthenon::X2DIR, LL>());
  if (b_ox2 == "neumann") pkg->UserBoundaryFunctions[BF::outer_x2].push_back(NeuZ<parthenon::X2DIR, RR>());
  if (b_ix3 == "neumann") pkg->UserBoundaryFunctions[BF::inner_x3].push_back(NeuZ<parthenon::X3DIR, LL>());
  if (b_ox3 == "neumann") pkg->UserBoundaryFunctions[BF::outer_x3].push_back(NeuZ<parthenon::X3DIR, RR>());
  // Multipole (WS-5a): exterior Cartesian expansion of interior mass through the quadrupole.
  // Enrolled as the per-block post-solve BC; the packed solver-internal path is in
  // PoissonEquation::SetBoundary (both keyed on face type 3).
  if (b_ix1 == "multipole") pkg->UserBoundaryFunctions[BF::inner_x1].push_back(MultipoleBC<parthenon::X1DIR, LL>);
  if (b_ox1 == "multipole") pkg->UserBoundaryFunctions[BF::outer_x1].push_back(MultipoleBC<parthenon::X1DIR, RR>);
  if (b_ix2 == "multipole") pkg->UserBoundaryFunctions[BF::inner_x2].push_back(MultipoleBC<parthenon::X2DIR, LL>);
  if (b_ox2 == "multipole") pkg->UserBoundaryFunctions[BF::outer_x2].push_back(MultipoleBC<parthenon::X2DIR, RR>);
  if (b_ix3 == "multipole") pkg->UserBoundaryFunctions[BF::inner_x3].push_back(MultipoleBC<parthenon::X3DIR, LL>);
  if (b_ox3 == "multipole") pkg->UserBoundaryFunctions[BF::outer_x3].push_back(MultipoleBC<parthenon::X3DIR, RR>);

  // Cache the per-face BC type (0=default/other, 1=zero/Dirichlet, 2=neumann/outflow,
  // 3=multipole) so the packed PoissonEquation::SetBoundary can apply them without the
  // per-block BC dispatch. Order matches parthenon::BoundaryFace:
  // inner_x1,outer_x1,inner_x2,outer_x2,inner_x3,outer_x3.
  auto bc_code = [](const std::string &s) {
    return s == "zero" ? 1 : (s == "neumann" ? 2 : (s == "multipole" ? 3 : 0));
  };
  std::array<int, 6> grav_bc_face_type = {bc_code(b_ix1), bc_code(b_ox1), bc_code(b_ix2),
                                          bc_code(b_ox2), bc_code(b_ix3), bc_code(b_ox3)};
  pkg->AddParam("grav_bc_face_type", grav_bc_face_type);
  // Runtime escape hatch / A-B switch: self_gravity/packed_bc=false forces the original
  // per-block ApplyBoundaryConditionsOnCoarseOrFineMD path inside SetBoundary.
  const bool packed_bc = pin->GetOrAddBoolean(block_name, "packed_bc", true);
  pkg->AddParam("grav_packed_bc", packed_bc);

  // Multipole moments (WS-5a): mutable POD recomputed each step in FillPoissonRHS and
  // captured by value into the (device) boundary kernels. Registered unconditionally so
  // both BC paths can Param<>() it; four_pi_G is fixed, the rest start at zero.
  const bool has_multipole = (b_ix1 == "multipole") || (b_ox1 == "multipole") ||
                             (b_ix2 == "multipole") || (b_ox2 == "multipole") ||
                             (b_ix3 == "multipole") || (b_ox3 == "multipole");
  // Multipole BCs rely on the packed SetBoundary applying HOMOGENEOUS Dirichlet inside the
  // solve (with the inhomogeneous data lifted into the RHS). packed_bc=false would route
  // the operator through the enrolled inhomogeneous per-block MultipoleBC, making the
  // operator affine and diverging the preconditioned solve -- so forbid that combination.
  PARTHENON_REQUIRE(!has_multipole || packed_bc,
                    "self_gravity: multipole gravity BCs require packed_bc=true.");
  // NEW-1 (audit 2026-07-19): the multipole moments are assembled from the FULL density
  // while the swindle subtracts rho_mean from the RHS -- combining them would solve an
  // equation whose boundary data and source are mutually inconsistent. The multipole BC
  // is for isolated (non-periodic) boxes where the swindle is off anyway.
  PARTHENON_REQUIRE(!has_multipole || !use_swindle,
                    "self_gravity: multipole gravity BCs are incompatible with "
                    "use_swindle=true (moments use full rho, swindle RHS uses rho - "
                    "rho_mean). Disable the swindle for isolated-box multipole BCs.");
  // --- Shell-split multipole BC (2026-08-16) ---------------------------------------------
  // The pure exterior expansion assumes ALL mass is interior to the field point; on the domain
  // faces that is false by 34-45% of the mass for the production L=16 box, and it costs 7.8%
  // median error in |grad phi| at r = 1-2 (box ladder vs L = 32). Splitting the source at the
  // field radius fixes it without touching the box size -- which matters, because a small box
  // is exactly what this BC was written to make possible.
  // ON by default: the pure-exterior form is wrong wherever the split is needed, and correct
  // wherever it is not (the split reduces to it identically when all mass is interior), so
  // there is no regime in which the old form is preferable. `false` restores it bit-identically
  // for A/B.
  const bool mp_shell_split =
      pin->GetOrAddBoolean("self_gravity", "mp_shell_split", true);
  // nshell 12 -> 8 (2026-08-18): the deterministic accumulator costs one mesh sweep per bin, and
  // the boundary accuracy is converged well before 12. Measured against brute-force summation,
  // max|err| as a fraction of phi's face variation: nshell 4 = 7.95e-03, 8 = 5.43e-03,
  // 12 = 5.25e-03, 32 = 5.26e-03. Eight gives up 3% of the accuracy for a third fewer passes.
  const int mp_nshell = pin->GetOrAddInteger("self_gravity", "mp_nshell", 8);
  // Deterministic accumulation (par_reduce per bin) vs the original FP atomics. Default ON:
  // the atomics cost exact run-to-run reproducibility, which is what every A/B in this project
  // uses to separate signal from noise (measured: 0.0 exactly without, 1.21e-04 with).
  // `false` restores the atomic path for cost comparison.
  const bool mp_det = pin->GetOrAddBoolean("self_gravity", "mp_deterministic", true);
  pkg->AddParam("mp_deterministic", mp_det);
  PARTHENON_REQUIRE(mp_nshell >= 1 && mp_nshell <= 256,
                    "self_gravity/mp_nshell must be in [1, 256].");
  pkg->AddParam("mp_shell_split", mp_shell_split && has_multipole);
  // AUDIT F-38: the uniform-background split (see multipole.hpp UniformBoxPhi). Measured
  // 10.6 % -> 0.58 % of phi's face variation, i.e. 18x beyond the shell split alone. Default
  // true = the behaviour since 2026-08-16; false restores the pure shell-split expansion.
  pkg->AddParam("mp_uniform_split",
                pin->GetOrAddBoolean("self_gravity", "mp_uniform_split", true));
  pkg->AddParam("mp_nshell", mp_nshell);
  if (mp_shell_split && has_multipole) {
    MultipoleShells ms;
    ms.nshell = mp_nshell;
    ms.cum = Kokkos::View<Real *>("SG::mp_cum",
                                  static_cast<std::size_t>(mp_nshell + 1) * kMPNSlot);
    pkg->AddParam("grav_mp_shells", ms, /*is_mutable=*/true);
    // Scratch accumulator, zeroed and refilled every step. Separate from `cum` so the
    // cumulative arrays the kernels read are never half-written.
    pkg->AddParam("grav_mp_acc",
                  Kokkos::View<Real *>("SG::mp_acc",
                                       static_cast<std::size_t>(mp_nshell) * 2 * kMPNSlot),
                  /*is_mutable=*/true);
  }

  MultipoleMoments mm0{};
  mm0.four_pi_G = four_pi_G;
  pkg->AddParam("grav_multipole_moments", mm0, /*is_mutable=*/true);
  pkg->AddParam("grav_has_multipole", has_multipole);

  // AUDIT A2 (2026-08-05), FIXED. The global reductions in FillPoissonRHS (Jeans-swindle mean
  // density, multipole moments) now run ONCE PER FillDerived SWEEP over the WHOLE local block
  // list, instead of once per MeshData partition over that partition's blocks only. The two
  // Params below carry the published result from the sweep's leading call to the trailing ones.
  //   grav_reduction_tick counts calls within a sweep, modulo the rank's partition count; the
  //   call that sees tick == 0 is the one that reduces and Allreduces. Every FillDerivedMesh
  //   invocation in the code base is "for each default block partition" (Mesh::FillDerived,
  //   and TaskRegion(num_partitions) at hydro_driver.cpp:357/445/863), so the tick returns to 0
  //   exactly at sweep boundaries and every rank enters the collective exactly once per sweep --
  //   which is what removes the deadlock, since DefaultNumPartitions() itself is NOT rank-uniform.
  pkg->AddParam<int>("grav_reduction_tick", 0, /*is_mutable=*/true);
  pkg->AddParam<Real>("grav_mean_rho", 0.0, /*is_mutable=*/true);
  // P2: see self_gravity.hpp. Default TRUE == pre-P2 behaviour (reduce on every call).
  pkg->AddParam<bool>("grav_rhs_consumed", true, /*is_mutable=*/true);

  // --- Fields ----------------------------------------------------------------
  // phi: solution variable. Needs ghost fill, fluxes, GMG prolong/restrict for AMR MG.
  {
    std::vector<parthenon::MetadataFlag> flags{
        Metadata::Cell,       Metadata::Independent,   Metadata::FillGhost,
        Metadata::WithFluxes, Metadata::GMGRestrict,   Metadata::GMGProlongate};
    // Audit A-3: without the OperatorSplit flag grav.phi sits in the flag pack of
    // UpdateWithFluxDivergence and is overwritten from the (unused, zero) u1 copy every stage -- benign
    // only because the solve rebuilds it before use. self_gravity/phi_operator_split = true adds the
    // flag (production). Default false = previous pack contents, bit-identical.
    if (pin->GetOrAddBoolean(block_name, "phi_operator_split", false)) {
      const parthenon::MetadataFlag OperatorSplit =
          parthenon::Metadata::FlagNameExists("OperatorSplit")
              ? parthenon::Metadata::GetUserFlag("OperatorSplit")
              : parthenon::Metadata::AddUserFlag("OperatorSplit");
      flags.push_back(OperatorSplit);
      if (parthenon::Globals::my_rank == 0)
        std::cout << "## Self-gravity: grav.phi carries OperatorSplit (excluded from the "
                     "hydro low-storage update; audit A-3)" << std::endl;
    }
    Metadata m(flags);
    // Audit F-15: prolongation of phi into AMR ghosts (not the MG's internal prolongation). With
    // piecewise-constant the first fine cell at a coarse-fine face gets 1.25x the force of a linear
    // potential; ProlongateSharedLinear reproduces linear phi exactly (production, phi_prolongation =
    // linear). Audit F-17: float32 dumps cannot validate the solve (round-off dominated); use
    // grav-res / grav-res-rel, or write one snapshot at double precision.
  const auto phi_prol =
        pin->GetOrAddString(block_name, "phi_prolongation", "linear");
    PARTHENON_REQUIRE(phi_prol == "linear" || phi_prol == "piecewise_constant",
                      "self_gravity/phi_prolongation must be 'linear' or "
                      "'piecewise_constant'.");
    if (phi_prol == "linear") {
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                              parthenon::refinement_ops::RestrictAverage>();
    } else {
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                              parthenon::refinement_ops::RestrictAverage>();
    }
    if (parthenon::Globals::my_rank == 0)
      std::cout << "## Self-gravity phi prolongation: " << phi_prol
                << (phi_prol == "linear" ? "  (audit F-15; PWC cost 25 % of the force in the"
                                           " first fine cell at coarse-fine faces)"
                                         : "  (LEGACY piecewise-constant)")
                << std::endl;
    pkg->AddField<grav::phi>(m);
  }
  // rhs: source. Derived, OneCopy. Matches upstream Parthenon poisson_gmg example.
  // rhs carries FillGhost + refinement ops to match Artemis: the GMG V-cycle
  // restricts/exchanges rhs across ranks, so it needs ghost data and AMR ops.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy,
                Metadata::FillGhost});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::rhs>(m);
  }

  // --- FillDerived: compute rhs from density on every timestep ---------------
  pkg->FillDerivedMesh = FillPoissonRHS;

  // --- Solver construction ---------------------------------------------------
  using PoissEq = PoissonEquation<grav::phi>;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
  const std::string solver_params_block = block_name + "/solver_params";
  // Runtime-selectable solver. Default "BiCGSTAB" reproduces production behaviour
  // bit-for-bit. "MG" / "Multigrid" uses the *pure* geometric multigrid solver,
  // which has no global inner-products (BiCGSTAB needs ~2 all-reduces/iteration),
  // attacking the latency-bound bottleneck of the GPU self-gravity solve.
  // NOTE: solver=MG (pure multigrid) needs an adequate smoother on AMR. SRJ1 (one
  // weighted-Jacobi sweep) is enough on a UNIFORM grid but NOT across AMR fine-coarse
  // boundaries, where the coarse-grid correction injects high-frequency error each
  // V-cycle (the standalone V-cycle then has spectral radius > 1 and diverges). Use
  // SRJ2/SRJ3 for MG on AMR; SRJ2 is the parthenon MGParams default and converges in
  // ~6 V-cycles. BiCGSTAB tolerates SRJ1 because its Krylov outer loop stabilises a
  // non-contractive preconditioner.
  const std::string solver_type =
      pin->GetOrAddString(solver_params_block, "solver", "BiCGSTAB");
  std::shared_ptr<parthenon::solvers::SolverBase> psolver;
  if (solver_type == "MG" || solver_type == "Multigrid") {
    psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
        /*container_base=*/"base",
        /*container_u=*/"phi",
        /*container_rhs=*/"rhs",
        pin, solver_params_block, PoissEq(pin, block_name));
  } else {
    psolver =
        std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            /*container_base=*/"base",
            /*container_u=*/"phi",
            /*container_rhs=*/"rhs",
            pin, solver_params_block, PoissEq(pin, block_name));
  }
  // Audit G-9: BiCGSTAB warm-starts from the "phi" solver container, which is cleared on regrid
  // and absent after a resume -> continued and resumed runs converge to different phi inside the
  // tolerance. zero_initial_guess = true removes the asymmetry by cold-starting every solve;
  // seed_from_phi (production) removes it while keeping the warm start. Default false.
  const bool zero_guess =
      pin->GetOrAddBoolean(solver_params_block, "zero_initial_guess", false);
  psolver->initial_guess_is_zero = zero_guess;
  pkg->AddParam("solver_pointer", psolver);
  // SG-F1 (2026-10-03): round-off floor coefficient c (floor = c eps sqrt(N_cells) 4piG rms(rho)).
  // Default 64 under the Jeans swindle (the only configuration with a null space, where the stall
  // was measured), 0 otherwise -- so isolated boxes (production: multipole, no swindle) are
  // bit-identical by construction.
  const Real roundoff_floor =
      pin->GetOrAddReal(solver_params_block, "roundoff_floor", use_swindle ? 64.0 : 0.0);
  pkg->AddParam<Real>("roundoff_floor", roundoff_floor);
  if (solver_type != "MG" && solver_type != "Multigrid") {
    auto pbicg = std::static_pointer_cast<
        parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(psolver);
    pkg->AddParam<std::shared_ptr<Real>>("grav_roundoff_abs_tol",
                                         pbicg->GetParams().roundoff_abs_tolerance);
  }
  if (parthenon::Globals::my_rank == 0 && roundoff_floor > 0.0) {
    std::cout << "## Self-gravity solver round-off floor ON (SG-F1): |r| < " << roundoff_floor
              << " * eps * sqrt(N_cells) * 4piG rms(rho) also counts as converged" << std::endl;
  }
  // RESTART REPRODUCIBILITY (2026-09-22). zero_initial_guess removes the continue/resume
  // asymmetry above by discarding the warm start. seed_from_phi removes it by making the warm
  // start come from the RESTART-PERSISTED field: every solve first copies base grav.phi (the
  // Independent field written to every restart, holding the previous solution) into the solver's
  // "phi" container. A continued and a resumed run then start the first post-restart solve from
  // the same potential. It also keeps the warm start across regrids, where the container is
  // cleared and the default path silently cold-starts. Default false == previous behaviour.
  const bool seed_from_phi =
      pin->GetOrAddBoolean(solver_params_block, "seed_from_phi", false);
  PARTHENON_REQUIRE(!(seed_from_phi && zero_guess),
                    "self_gravity/solver_params: seed_from_phi and zero_initial_guess are "
                    "mutually exclusive (one warm-starts from grav.phi, the other from zero)");
  pkg->AddParam("seed_from_phi", seed_from_phi);

  // --- VALIDATION WP-5: Poisson-solver convergence reporting -----------------
  // The BiCGSTAB completion task returns TaskStatus::complete both when a tolerance is
  // met and when max_iterations is exhausted, and reports neither -- so a solve that
  // silently bails at the ceiling is indistinguishable from a converged one. Record the
  // iteration count / final residual and flag the ceiling case. See diagnostics/grav_diag.hpp.
  // Default OFF; when off nothing is registered and no task is added => bit-identical.
  const bool solver_diag = pin->GetOrAddBoolean(block_name, "solver_diag", false);
  pkg->AddParam<>("solver_diag", solver_diag);

  // B2 (2026-08-02): the NON-CONVERGENCE WARNING is unconditional, unlike the hst columns.
  // Rationale: a Poisson solve that quietly gave up corrupts the potential for that step, and
  // production runs do NOT set solver_diag (it adds hst columns, which would shift every column
  // index and break existing analysis scripts). Gating the warning on the diagnostic meant the
  // one failure mode that most needs to be loud was silent in exactly the runs that matter.
  // These two Params are always registered; the warn task reads solver state only and touches
  // no field data, so the OFF-state remains bit-identical.
  const int max_iters_always =
      pin->GetOrAddInteger(solver_params_block, "max_iterations", 1000);
  pkg->AddParam<int>("grav_max_iters", max_iters_always);
  pkg->AddParam<int>("grav_nonconv_count", 0, true);

  // B3: unless relative_residual = true (production), the criterion is an ABSOLUTE rms residual,
  // which tightens relative accuracy monotonically through a collapse. Not auto-corrected (result-
  // changing); announced loudly instead.
  const bool rel_res = pin->GetOrAddBoolean(solver_params_block, "relative_residual", false);
  if (!rel_res && parthenon::Globals::my_rank == 0) {
    const Real res_tol = pin->GetOrAddReal(solver_params_block, "residual_tolerance", 1.0e-12);
    std::cout << "## NOTE [self-gravity] convergence uses an ABSOLUTE residual tolerance ("
              << res_tol << "); <" << solver_params_block
              << "> relative_residual is false. As the collapse deepens the RHS grows while "
                 "this ceiling does not, so the effective relative tolerance tightens "
                 "monotonically. Watch grav-res (self_gravity/solver_diag=true) and the "
                 "non-convergence warnings. See B3 in VALIDATION_PLAN.md."
              << std::endl;
  }

  // Audits G-1/F-16: always print which gravity BC is in use (the banner used to sit inside
  // solver_diag, which no production deck set). Reads Params only.
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "## Self-gravity BCs: x1(" << b_ix1 << "," << b_ox1 << ") x2(" << b_ix2
              << "," << b_ox2 << ") x3(" << b_ix3 << "," << b_ox3 << ")";
    if (has_multipole) {
      std::cout << " | multipole: shell_split="
                << (pkg->Param<bool>("mp_shell_split") ? "ON" : "off");
      if (pkg->Param<bool>("mp_shell_split"))
        std::cout << " (nshell=" << pkg->Param<int>("mp_nshell") << ", uniform-box split "
                  << (pkg->Param<bool>("mp_uniform_split") ? "ON" : "off") << ")";
      std::cout << " deterministic=" << (pkg->Param<bool>("mp_deterministic") ? "ON" : "off");
    }
    std::cout << " | packed_bc=" << (pkg->Param<bool>("grav_packed_bc") ? "ON" : "off")
              << " swindle=" << (use_swindle ? "ON" : "off") << std::endl;
  }

  if (solver_diag) {
    // Mirror the ceiling the solver itself parsed, so the "did it bail" test uses the
    // same number rather than a hard-coded default. The key was already consumed above
    // by BiCGSTABParams/MGParams, so GetOrAdd here just reads it back.
    pkg->AddParam<Real>("grav_last_iters", -1.0, true);
    pkg->AddParam<Real>("grav_last_res", -1.0, true);
    pkg->AddParam<Real>("grav_last_nonconv", 0.0, true);
    // AUDIT F-14: residual / tolerance-actually-applied. -1 = not published by the solver.
    pkg->AddParam<Real>("grav_last_res_rel", -1.0, true);

    using Diagnostics::GravDiag;
    using Diagnostics::GravDiagReport;
    parthenon::HstVar_list hst_grav = {};
    auto add_grav = [&hst_grav](GravDiag w, const std::string &label) {
      // max, not sum: the underlying scalars come from a global all-reduce inside the
      // solver and are identical on every partition, so sum would multiply by the
      // partition count.
      hst_grav.emplace_back(parthenon::HistoryOutputVar(
          parthenon::UserHistoryOperation::max,
          [w](MeshData<Real> *md) { return GravDiagReport(md, w); }, label));
    };
    add_grav(GravDiag::iters, "grav-iters");
    add_grav(GravDiag::res, "grav-res");
    add_grav(GravDiag::nonconv, "grav-nonconv");
    // Multipole exterior-validity pair (2026-08-15). Registered only when the multipole BC is
    // actually in use -- for zero-Dirichlet walls there is no expansion whose validity could
    // be in question, and a column of zeros would be noise.
    // APPENDED, never inserted: hydro.cpp:~2193 records that inserting a history entry shifts
    // every later column index and breaks downstream analysis. These two land at the END of
    // the self_gravity block, after grav-nonconv.
    if (has_multipole) {
      pkg->AddParam<Real>("grav_mp_extmass", -1.0, true);
      pkg->AddParam<Real>("grav_mp_comoff", -1.0, true);
      add_grav(GravDiag::mp_extmass, "grav-mp-extmass");
      add_grav(GravDiag::mp_comoff, "grav-mp-comoff");
    }
    // AUDIT F-14 (2026-08-28) -- APPENDED LAST, so NO existing column index moves.
    // `grav-res` is the ABSOLUTE rms residual, while <self_gravity/solver_params>
    // residual_tolerance is RELATIVE (relative_residual = true in all four live decks), so
    // comparing the two is meaningless: 14 of 539 production rows show grav-res > 1e-6 while
    // being fully converged. This column is that residual divided by the tolerance the solve
    // was ACTUALLY judged against, so it is directly comparable to 1 -- > 1 means the
    // tolerance was not met. -1 when the solver did not publish one.
    {
      add_grav(GravDiag::res_rel, "grav-res-rel");
    }
    pkg->AddParam<>(parthenon::hist_param_key, hst_grav);

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "## Self-gravity solver diagnostics ON (self_gravity/solver_diag): "
                   "hst grav-iters, grav-res, grav-nonconv; max_iterations="
                << max_iters_always << "." << std::endl;

    }
  }

  return pkg;
}

// AUDIT A2 (2026-08-05). Local (this-partition) partial sums feeding the two global gravity
// reductions. Split out of FillPoissonRHS so the same kernels can be run over several MeshData
// partitions and summed before the single MPI_Allreduce -- see FillPoissonRHS for the full
// argument. `acc` is ACCUMULATED into, never zeroed here.
//   acc[0]      total mass          (swindle)
//   acc[1]      total volume        (swindle)
//   acc[2..11]  the ten raw multipole moments about the coordinate origin
//   acc[12]     mass outside the largest sphere inscribed in the domain (multipole validity)
//   acc[13..22] the same ten moments for a UNIT-DENSITY field (i.e. volume-weighted), used
//               to form the residual delta-rho moments for the uniform-background split
namespace {
constexpr int kNGravSums = 24;  // [23] = sum rho^2 vol (SG-F1 round-off floor)

void AccumulateGravSums(MeshData<Real> *md, const bool need_swindle,
                        const bool need_moments, const Real rho_floor,
                        const Real r_inscribed, Real (&acc)[kNGravSums]) {
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  if (nblocks == 0) return;

  if (need_swindle) {
    Real total_mass = 0.0, total_volume = 0.0, total_rho2 = 0.0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "SG::TotalMass", parthenon::DevExecSpace(), 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmass,
                      Real &lvol, Real &lrho2) {
          const auto &coords = cons_pack.GetCoords(b);
          const Real vv = coords.CellVolume(k, j, i);
          const Real rho = (cons_pack(b, IDN, k, j, i) > rho_floor ? cons_pack(b, IDN, k, j, i)
                                                                   : rho_floor);
          lvol += vv;
          lmass += rho * vv;
          lrho2 += rho * rho * vv;
        },
        Kokkos::Sum<Real>(total_mass), Kokkos::Sum<Real>(total_volume),
        Kokkos::Sum<Real>(total_rho2));
    Kokkos::fence();
    acc[0] += total_mass;
    acc[1] += total_volume;
    acc[23] += total_rho2;
  }

  if (need_moments) {
    // P1: tree reduction (par_reduce) instead of 10 contended atomics per cell (was 51.6 % of GPU
    // time). Also deterministic, which the atomic version was not (~1e-16 run to run).
    Real q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0, q4 = 0.0;
    Real q5 = 0.0, q6 = 0.0, q7 = 0.0, q8 = 0.0, q9 = 0.0;
    // 11th accumulator: mass outside the inscribed sphere (multipole exterior validity).
    // Rides the SAME reduction as the moments, so it adds one register and no launch, no
    // extra pass over the mesh and no extra MPI_Allreduce -- the P1 lesson (the moment
    // reduction was once 51.6% of all GPU kernel time) says never to add a second pass here.
    // Radius is measured from the COORDINATE ORIGIN, not the COM: the COM is not known until
    // this reduction completes, and for this problem the two coincide to within the offset
    // that grav-mp-comoff reports separately -- so the pair is self-checking.
    Real q10 = 0.0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "SG::Moments", parthenon::DevExecSpace(), 0,
        nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &l0,
                      Real &l1, Real &l2, Real &l3, Real &l4, Real &l5, Real &l6, Real &l7,
                      Real &l8, Real &l9, Real &l10) {
          const auto &coords = cons_pack.GetCoords(b);
          const Real vv = coords.CellVolume(k, j, i);
          const Real dm = (cons_pack(b, IDN, k, j, i) > rho_floor
                               ? cons_pack(b, IDN, k, j, i)
                               : rho_floor) *
                          vv;
          const Real x = coords.Xc<parthenon::X1DIR>(k, j, i);
          const Real y = coords.Xc<parthenon::X2DIR>(k, j, i);
          const Real z = coords.Xc<parthenon::X3DIR>(k, j, i);
          l0 += dm;
          l1 += dm * x;
          l2 += dm * y;
          l3 += dm * z;
          l4 += dm * x * x;
          l5 += dm * y * y;
          l6 += dm * z * z;
          l7 += dm * x * y;
          l8 += dm * x * z;
          l9 += dm * y * z;
          if (x * x + y * y + z * z > r_inscribed * r_inscribed) l10 += dm;
        },
        Kokkos::Sum<Real>(q0), Kokkos::Sum<Real>(q1), Kokkos::Sum<Real>(q2),
        Kokkos::Sum<Real>(q3), Kokkos::Sum<Real>(q4), Kokkos::Sum<Real>(q5),
        Kokkos::Sum<Real>(q6), Kokkos::Sum<Real>(q7), Kokkos::Sum<Real>(q8),
        Kokkos::Sum<Real>(q9), Kokkos::Sum<Real>(q10));
    const Real qs[10] = {q0, q1, q2, q3, q4, q5, q6, q7, q8, q9};
    for (int t = 0; t < 10; ++t) acc[2 + t] += qs[t];
    acc[12] += q10;
  }
}

//! Whole-mesh VOLUME moments (the same ten as the mass moments, but for unit density).
//!
//! Needed because the residual delta-rho = rho - rho_bar moments are formed by subtraction,
//! I(delta) = I(rho) - rho_bar * I(1), and the CORE part of I(delta) is itself obtained by
//! subtracting the shell bins from the whole-mesh total. Both totals must therefore be DISCRETE
//! sums over the same cells: the analytic continuum box moments differ from the discrete cell
//! sums at O(dx^2), and mixing the two would leave that mismatch in the core moments, where it
//! would look like a physical dipole/quadrupole that is not there.
//! A separate tree reduction rather than ten more reducers bolted onto SG::Moments: this keeps
//! the existing kernel's reducer list (already eleven) from growing to twenty-one, and it is
//! the same pattern P1 established as the cheap way to do this.
void AccumulateVolumeMoments(MeshData<Real> *md, Real (&vm)[10]) {
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  if (nblocks == 0) return;
  Real v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0, v7 = 0, v8 = 0, v9 = 0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "SG::VolMoments", parthenon::DevExecSpace(), 0,
      nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &l0, Real &l1,
                    Real &l2, Real &l3, Real &l4, Real &l5, Real &l6, Real &l7, Real &l8,
                    Real &l9) {
        const auto &coords = cons_pack.GetCoords(b);
        const Real vv = coords.CellVolume(k, j, i);
        const Real x = coords.Xc<parthenon::X1DIR>(k, j, i);
        const Real y = coords.Xc<parthenon::X2DIR>(k, j, i);
        const Real z = coords.Xc<parthenon::X3DIR>(k, j, i);
        l0 += vv; l1 += vv * x; l2 += vv * y; l3 += vv * z;
        l4 += vv * x * x; l5 += vv * y * y; l6 += vv * z * z;
        l7 += vv * x * y; l8 += vv * x * z; l9 += vv * y * z;
      },
      Kokkos::Sum<Real>(v0), Kokkos::Sum<Real>(v1), Kokkos::Sum<Real>(v2),
      Kokkos::Sum<Real>(v3), Kokkos::Sum<Real>(v4), Kokkos::Sum<Real>(v5),
      Kokkos::Sum<Real>(v6), Kokkos::Sum<Real>(v7), Kokkos::Sum<Real>(v8),
      Kokkos::Sum<Real>(v9));
  const Real vs[10] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9};
  for (int t = 0; t < 10; ++t) vm[t] += vs[t];
}

//! Accumulate the shell-binned inner/outer moments needed by the shell-split multipole BC.
//!
//! NOT bit-reproducible: double atomics resolve in hardware order (measured 1.2e-4 on cons between
//! identical runs, job 2531563, and it amplifies). Only cells with r > r_lo participate; the core's
//! moments come from the 10-scalar tree reduction by subtraction. Use the deterministic variant
//! (self_gravity/mp_deterministic = true, production).
//! ----
//! Deterministic variant: one par_reduce per bin (fixed tree -> reproducible), at one mesh sweep
//! per (bin, weight). nshell = 8 is converged (5.4e-3 of phi's face variation vs 5.3e-3 at 12).
//! Quantising the atomic sums was rejected: it fails intermittently near rounding boundaries.
void AccumulateShellBinsDet(MeshData<Real> *md, const Real rho_floor, const Real r_lo,
                            const Real inv_dr, const int nshell, const bool volume_weight,
                            Real *out) {
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  if (nblocks == 0) return;

  for (int kbin = 0; kbin < nshell; ++kbin) {
    Real a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0,a8=0,a9=0,a10=0;
    Real b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0,b7=0,b8=0,b9=0;
    const int kb_target = kbin;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "SG::ShellBinsDet", parthenon::DevExecSpace(),
        0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int bk, const int k, const int j, const int i,
                      Real &l0, Real &l1, Real &l2, Real &l3, Real &l4, Real &l5, Real &l6,
                      Real &l7, Real &l8, Real &l9, Real &l10, Real &m0, Real &m1, Real &m2,
                      Real &m3, Real &m4, Real &m5, Real &m6, Real &m7, Real &m8, Real &m9) {
          const auto &coords = cons_pack.GetCoords(bk);
          const Real x = coords.Xc<parthenon::X1DIR>(k, j, i);
          const Real y = coords.Xc<parthenon::X2DIR>(k, j, i);
          const Real z = coords.Xc<parthenon::X3DIR>(k, j, i);
          const Real r = std::sqrt(x * x + y * y + z * z);
          if (!(r > r_lo)) return;
          int kk = static_cast<int>((r - r_lo) * inv_dr);
          if (kk < 0) kk = 0;
          if (kk > nshell - 1) kk = nshell - 1;
          if (kk != kb_target) return;
          const Real vv = coords.CellVolume(k, j, i);
          const Real w = volume_weight
                             ? vv
                             : (cons_pack(bk, IDN, k, j, i) > rho_floor
                                    ? cons_pack(bk, IDN, k, j, i)
                                    : rho_floor) * vv;
          const Real invr = 1.0 / r, invr3 = invr * invr * invr, invr5 = invr3 * invr * invr;
          l0 += w;            l1 += w * x;        l2 += w * y;        l3 += w * z;
          l4 += w * x * x;    l5 += w * y * y;    l6 += w * z * z;
          l7 += w * x * y;    l8 += w * x * z;    l9 += w * y * z;
          l10 += w * invr;
          m0 += w * x * invr3;      m1 += w * y * invr3;      m2 += w * z * invr3;
          m3 += w * x * x * invr5;  m4 += w * y * y * invr5;  m5 += w * z * z * invr5;
          m6 += w * x * y * invr5;  m7 += w * x * z * invr5;  m8 += w * y * z * invr5;
          m9 += w * invr3;
        },
        Kokkos::Sum<Real>(a0), Kokkos::Sum<Real>(a1), Kokkos::Sum<Real>(a2),
        Kokkos::Sum<Real>(a3), Kokkos::Sum<Real>(a4), Kokkos::Sum<Real>(a5),
        Kokkos::Sum<Real>(a6), Kokkos::Sum<Real>(a7), Kokkos::Sum<Real>(a8),
        Kokkos::Sum<Real>(a9), Kokkos::Sum<Real>(a10),
        Kokkos::Sum<Real>(b0), Kokkos::Sum<Real>(b1), Kokkos::Sum<Real>(b2),
        Kokkos::Sum<Real>(b3), Kokkos::Sum<Real>(b4), Kokkos::Sum<Real>(b5),
        Kokkos::Sum<Real>(b6), Kokkos::Sum<Real>(b7), Kokkos::Sum<Real>(b8),
        Kokkos::Sum<Real>(b9));
    // Slot order must match the MPSlot enum exactly; kS0 sits between the inner block and
    // the S1 vector, which is why a10 is written there rather than appended.
    Real *o = out + static_cast<std::size_t>(kbin) * kMPNSlot;
    o[kM]=a0; o[kPx]=a1; o[kPy]=a2; o[kPz]=a3;
    o[kMxx]=a4; o[kMyy]=a5; o[kMzz]=a6; o[kMxy]=a7; o[kMxz]=a8; o[kMyz]=a9;
    o[kS0]=a10;
    o[kS1x]=b0; o[kS1y]=b1; o[kS1z]=b2;
    o[kS2xx]=b3; o[kS2yy]=b4; o[kS2zz]=b5; o[kS2xy]=b6; o[kS2xz]=b7; o[kS2yz]=b8;
    o[kT0]=b9;
  }
}

void AccumulateShellBins(MeshData<Real> *md, const Real rho_floor, const Real r_lo,
                         const Real inv_dr, const int nshell,
                         const Kokkos::View<Real *> &acc) {
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  if (nblocks == 0) return;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::ShellBins", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<parthenon::X1DIR>(k, j, i);
        const Real y = coords.Xc<parthenon::X2DIR>(k, j, i);
        const Real z = coords.Xc<parthenon::X3DIR>(k, j, i);
        const Real r2 = x * x + y * y + z * z;
        const Real r = std::sqrt(r2);
        if (!(r > r_lo)) return; // interior to every boundary point; handled by subtraction
        int kbin = static_cast<int>((r - r_lo) * inv_dr);
        if (kbin < 0) kbin = 0;
        if (kbin > nshell - 1) kbin = nshell - 1;
        const Real vv = coords.CellVolume(k, j, i);
        const Real dm = (cons_pack(b, IDN, k, j, i) > rho_floor ? cons_pack(b, IDN, k, j, i)
                                                                : rho_floor) *
                        vv;
        const Real invr = 1.0 / r;
        const Real invr3 = invr * invr * invr;
        const Real invr5 = invr3 * invr * invr;
        // TWO sets of slots per bin: density-weighted and volume-weighted. delta-rho moments
        // are linear in the weight, so  I(delta) = I(rho) - rho_bar * I(1), and rho_bar is not
        // known until after the reduction. Accumulating both here is what lets the residual be
        // formed later without a second pass over the mesh.
        Real *a = acc.data() + static_cast<std::size_t>(kbin) * 2 * kMPNSlot;
        Real *w = a + kMPNSlot;
        // inner-type raw moments (used when this shell is INSIDE the field point)
        Kokkos::atomic_add(a + kM, dm);
        Kokkos::atomic_add(a + kPx, dm * x);
        Kokkos::atomic_add(a + kPy, dm * y);
        Kokkos::atomic_add(a + kPz, dm * z);
        Kokkos::atomic_add(a + kMxx, dm * x * x);
        Kokkos::atomic_add(a + kMyy, dm * y * y);
        Kokkos::atomic_add(a + kMzz, dm * z * z);
        Kokkos::atomic_add(a + kMxy, dm * x * y);
        Kokkos::atomic_add(a + kMxz, dm * x * z);
        Kokkos::atomic_add(a + kMyz, dm * y * z);
        // outer-type moments (used when this shell is OUTSIDE the field point)
        Kokkos::atomic_add(a + kS0, dm * invr);
        Kokkos::atomic_add(a + kS1x, dm * x * invr3);
        Kokkos::atomic_add(a + kS1y, dm * y * invr3);
        Kokkos::atomic_add(a + kS1z, dm * z * invr3);
        Kokkos::atomic_add(a + kS2xx, dm * x * x * invr5);
        Kokkos::atomic_add(a + kS2yy, dm * y * y * invr5);
        Kokkos::atomic_add(a + kS2zz, dm * z * z * invr5);
        Kokkos::atomic_add(a + kS2xy, dm * x * y * invr5);
        Kokkos::atomic_add(a + kS2xz, dm * x * z * invr5);
        Kokkos::atomic_add(a + kS2yz, dm * y * z * invr5);
        Kokkos::atomic_add(a + kT0, dm * invr3);
        // volume-weighted twin (dm -> vv): the same 21 moments for a unit-density field.
        Kokkos::atomic_add(w + kM, vv);
        Kokkos::atomic_add(w + kPx, vv * x);
        Kokkos::atomic_add(w + kPy, vv * y);
        Kokkos::atomic_add(w + kPz, vv * z);
        Kokkos::atomic_add(w + kMxx, vv * x * x);
        Kokkos::atomic_add(w + kMyy, vv * y * y);
        Kokkos::atomic_add(w + kMzz, vv * z * z);
        Kokkos::atomic_add(w + kMxy, vv * x * y);
        Kokkos::atomic_add(w + kMxz, vv * x * z);
        Kokkos::atomic_add(w + kMyz, vv * y * z);
        Kokkos::atomic_add(w + kS0, vv * invr);
        Kokkos::atomic_add(w + kS1x, vv * x * invr3);
        Kokkos::atomic_add(w + kS1y, vv * y * invr3);
        Kokkos::atomic_add(w + kS1z, vv * z * invr3);
        Kokkos::atomic_add(w + kS2xx, vv * x * x * invr5);
        Kokkos::atomic_add(w + kS2yy, vv * y * y * invr5);
        Kokkos::atomic_add(w + kS2zz, vv * z * z * invr5);
        Kokkos::atomic_add(w + kS2xy, vv * x * y * invr5);
        Kokkos::atomic_add(w + kS2xz, vv * x * z * invr5);
        Kokkos::atomic_add(w + kS2yz, vv * y * z * invr5);
        Kokkos::atomic_add(w + kT0, vv * invr3);
      });
}

// Serializes the tick read-modify-write. Parthenon's default task execution is single-threaded,
// but nothing in the API forbids a threaded TaskRegion, and getting this wrong reintroduces
// exactly the mismatched-collective-count deadlock the fix exists to remove.
std::mutex grav_reduction_mutex;
} // namespace

// FillPoissonRHS: assemble rhs = 4 pi G (rho - rho_mean) from gas density.
// Runs over ENTIRE (including ghosts) so solver sees consistent ghost values.
// P2 (2026-08-13): driver-set, rank-uniform gate. See self_gravity.hpp and the P2 patch
// header. Called from the driver task graph immediately before each Update::FillDerived so
// that FillPoissonRHS knows whether the rhs it is about to write will actually be read.
parthenon::TaskStatus SetGravRhsConsumed(parthenon::Mesh *pm, bool consumed) {
  if (pm->packages.AllPackages().count("self_gravity") == 0)
    return parthenon::TaskStatus::complete;
  pm->packages.Get("self_gravity")->UpdateParam("grav_rhs_consumed", consumed);
  return parthenon::TaskStatus::complete;
}

void FillPoissonRHS(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &grav_pkg = pm->packages.Get("self_gravity");
  const bool use_swindle = grav_pkg->Param<bool>("use_swindle");
  const Real four_pi_G = grav_pkg->Param<Real>("four_pi_G");
  const bool has_multipole = grav_pkg->Param<bool>("grav_has_multipole");

  // WP-13: read the density from cons, not prim: package FillDerived order is not guaranteed, and on
  // a restart self-gravity's ran first with prim still zero (zero RHS -> NaN phi). Apply ConsToPrim's
  // density floor so the RHS equals prim(IDN) bit-for-bit.
  const Real rho_floor = pm->packages.Get("Hydro")->Param<Real>("grav_rho_floor");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto &resolved = pm->resolved_packages;
  auto desc_rhs = parthenon::MakePackDescriptor<grav::rhs>(resolved.get());
  auto rhs_pack = desc_rhs.GetPack(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);
  const int nblocks = md->NumBlocks();

  // Audit A2: the swindle mean and the multipole moments are GLOBAL, but FillDerivedMesh runs once
  // per partition. The leading call of each sweep (grav_reduction_tick == 0) reduces over ALL this
  // rank's partitions, so the collective is entered a rank-uniform number of times; trailing calls
  // read the published Params. With one partition (every deck) the arithmetic is unchanged.
  const int npart_local = static_cast<int>(pm->GetDefaultBlockPartitions().size());
  Real grav_mean_rho = 0.0;
  // P2: skip the reduction entirely when this call's grav::rhs cannot be read by any Poisson
  // solve before the next FillDerived overwrites it. 29 of 32 FillDerived calls per cycle are
  // in that category with RKL2 super-time-stepping at s_rkl = 15, and the 320 B multipole Allreduce they issue
  // was 98.96% of ALL blocking-collective time in a measured trace (48.72 s of 49.23 s).
  // The tick is NOT advanced in the skipped case, so the A2 sweep bookkeeping stays consistent:
  // a skipped sweep enters no collective on ANY rank, because the flag is rank-uniform.
  const bool rhs_consumed = grav_pkg->Param<bool>("grav_rhs_consumed");
  if (rhs_consumed && (use_swindle || has_multipole)) {
    bool lead_call = true;
    {
      std::lock_guard<std::mutex> lk(grav_reduction_mutex);
      int tick = grav_pkg->Param<int>("grav_reduction_tick");
      // A regrid between sweeps changes npart_local; resync rather than trust a stale tick.
      if (tick >= npart_local) tick = 0;
      lead_call = (tick == 0);
      grav_pkg->UpdateParam("grav_reduction_tick", (tick + 1) % npart_local);
    }

    if (lead_call) {
      // Largest sphere inscribed in the domain, about the coordinate ORIGIN. This is the
      // distance to the nearest point at which the multipole BC is evaluated, i.e. the radius
      // beyond which mass invalidates the exterior expansion. Half-width per dimension, min
      // over the three.
      const auto &msz = pm->mesh_size;
      const Real r_ins =
          std::min({0.5 * (msz.xmax(parthenon::X1DIR) - msz.xmin(parthenon::X1DIR)),
                    0.5 * (msz.xmax(parthenon::X2DIR) - msz.xmin(parthenon::X2DIR)),
                    0.5 * (msz.xmax(parthenon::X3DIR) - msz.xmin(parthenon::X3DIR))});

      // Shell-split accumulator: zeroed here, filled by AccumulateShellBins below, and
      // reduced together with the moments in the SAME MPI_Allreduce further down -- adding a
      // second collective would cost more than the arithmetic it carries (the per-level
      // Allreduce wait is the measured GPU bottleneck).
      const bool mp_split = grav_pkg->Param<bool>("mp_shell_split");
      const int mp_nshell = grav_pkg->Param<int>("mp_nshell");
      // r_hi: the furthest boundary point from the origin is a box corner. Cells beyond it
      // cannot exist, so the last bin is the outermost one that any face point can sit in.
      const Real r_hi = std::sqrt(
          0.25 * (SQR(msz.xmax(parthenon::X1DIR) - msz.xmin(parthenon::X1DIR)) +
                  SQR(msz.xmax(parthenon::X2DIR) - msz.xmin(parthenon::X2DIR)) +
                  SQR(msz.xmax(parthenon::X3DIR) - msz.xmin(parthenon::X3DIR))));
      const Real mp_inv_dr =
          (r_hi > r_ins) ? static_cast<Real>(mp_nshell) / (r_hi - r_ins) : 0.0;
      Kokkos::View<Real *> mp_acc;
      if (mp_split) {
        mp_acc = grav_pkg->Param<Kokkos::View<Real *>>("grav_mp_acc");
        Kokkos::deep_copy(mp_acc, 0.0);
      }

      Real acc[kNGravSums] = {};
      Real volmom[10] = {};
      const bool mp_det = grav_pkg->Param<bool>("mp_deterministic");
      const int nbin_slots = mp_nshell * kMPNSlot;
      std::vector<Real> det_rho(mp_det ? nbin_slots : 0, 0.0);
      std::vector<Real> det_vol(mp_det ? nbin_slots : 0, 0.0);
      if (npart_local == 1) {
        AccumulateGravSums(md, use_swindle, has_multipole, rho_floor, r_ins, acc);
        if (mp_split) {
          if (mp_det) {
            AccumulateShellBinsDet(md, rho_floor, r_ins, mp_inv_dr, mp_nshell, false, det_rho.data());
            AccumulateShellBinsDet(md, rho_floor, r_ins, mp_inv_dr, mp_nshell, true, det_vol.data());
          } else {
            AccumulateShellBins(md, rho_floor, r_ins, mp_inv_dr, mp_nshell, mp_acc);
          }
          AccumulateVolumeMoments(md, volmom);
        }
      } else {
        const std::string stage = md->StageName();
        for (int p = 0; p < npart_local; ++p) {
          auto *mdp = pm->mesh_data.GetOrAdd(stage, p).get();
          AccumulateGravSums(mdp, use_swindle, has_multipole, rho_floor, r_ins, acc);
          if (mp_split) {
            if (mp_det) {
              // NOTE: partial sums ACCUMULATE across partitions, so the per-bin results are
              // summed into scratch and added, not overwritten.
              std::vector<Real> t1(nbin_slots, 0.0), t2(nbin_slots, 0.0);
              AccumulateShellBinsDet(mdp, rho_floor, r_ins, mp_inv_dr, mp_nshell, false, t1.data());
              AccumulateShellBinsDet(mdp, rho_floor, r_ins, mp_inv_dr, mp_nshell, true, t2.data());
              for (int t = 0; t < nbin_slots; ++t) { det_rho[t] += t1[t]; det_vol[t] += t2[t]; }
            } else {
              AccumulateShellBins(mdp, rho_floor, r_ins, mp_inv_dr, mp_nshell, mp_acc);
            }
            AccumulateVolumeMoments(mdp, volmom);
          }
        }
      }

      // --- Mean density (for Jeans swindle) ------------------------------------
      if (use_swindle) {
        Real buf[3] = {acc[0], acc[1], acc[23]};
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, 3, MPI_PARTHENON_REAL,
                                          MPI_SUM, MPI_COMM_WORLD));
#endif
        grav_pkg->UpdateParam("grav_mean_rho", Real(buf[0] / buf[1]));
        // SG-F1 (2026-10-03): round-off floor for the Krylov solve. Under the swindle the
        // operator has a constant null space and rhs = 4piG (rho - rho_bar) carries a
        // round-off component of size ~eps * 4piG rho_bar that no phi can remove; with a
        // warm start the relative target (relative to the INITIAL residual) can fall below
        // it, the solve stalls to max_iterations and BiCGSTAB returns garbage. The floor
        // is scaled by the FULL density (rms rho), not by rhs, because that is what sets
        // the round-off. Only the swindle path reaches here; isolated boxes are untouched.
        // The attainable residual grows with the number of cells summed (measured 2026-10-03:
        // 8.9e-11 -> 1.8e-10 from 64^3 to 128^3 at 4piG rho_bar = 88.8, i.e. ~ sqrt(N)), so the
        // floor is c * eps * sqrt(N_cells) * 4piG rms(rho); measured c ~ 6, default c = 64.
        const Real floor_fac = grav_pkg->Param<Real>("roundoff_floor");
        if (floor_fac > 0.0 && grav_pkg->AllParams().hasKey("grav_roundoff_abs_tol")) {
          auto tol = grav_pkg->Param<std::shared_ptr<Real>>("grav_roundoff_abs_tol");
          *tol = floor_fac * std::numeric_limits<Real>::epsilon() *
                 std::sqrt(static_cast<Real>(pm->GetTotalCells())) * four_pi_G *
                 std::sqrt(buf[2] / buf[1]);
        }
      }

      // --- Multipole moments (for multipole exterior BC) -----------------------
      // Ten raw moments about the coordinate origin (mass, mass-weighted x_i, and
      // mass-weighted x_i x_j) summed over interior cells, then a global MPI_Allreduce.
      // From these derive the center of mass and the traceless quadrupole about it.
      // Uses the FULL density (not rho - rho_mean): the multipole BC is only used on
      // isolated (non-periodic) boxes where the swindle is off, so RHS = 4piG*rho and
      // the exterior expansion is of that same rho -- consistent by construction.
      if (has_multipole) {
        // 11 values, not 10: acc[12] (mass outside the inscribed sphere) rides the SAME
        // Allreduce as the moments, so the exterior-validity diagnostic costs no collective
        // of its own. Order matters -- h[10] is the exterior mass.
        // ONE collective for everything: the 10 raw moments, the exterior-mass diagnostic,
        // and (when active) the nshell*21 shell bins. Splitting these into separate
        // MPI_Allreduce calls would add barrier waits to the step's critical path for no
        // arithmetic benefit -- per-level Allreduce wait is the measured GPU bottleneck.
        const int nbin_vals = mp_split ? mp_nshell * 2 * kMPNSlot : 0;
        const int nvol = mp_split ? 10 : 0;
        std::vector<Real> h(11 + nvol + nbin_vals);
        for (int t = 0; t < 10; ++t) h[t] = acc[2 + t];
        h[10] = acc[12];
        typename Kokkos::View<Real *>::HostMirror mp_h;
        if (mp_split) {
          for (int t = 0; t < 10; ++t) h[11 + t] = volmom[t];
          if (mp_det) {
            // Deterministic path: interleave [rho-weighted | volume-weighted] per bin so the
            // downstream layout is identical to the atomic path's 2*kMPNSlot stride.
            for (int kbin = 0; kbin < mp_nshell; ++kbin)
              for (int sl = 0; sl < kMPNSlot; ++sl) {
                const std::size_t src = static_cast<std::size_t>(kbin) * kMPNSlot + sl;
                const std::size_t dst = static_cast<std::size_t>(kbin) * 2 * kMPNSlot + sl;
                h[11 + nvol + dst] = det_rho[src];
                h[11 + nvol + dst + kMPNSlot] = det_vol[src];
              }
          } else {
            mp_h = Kokkos::create_mirror_view(mp_acc);
            Kokkos::deep_copy(mp_h, mp_acc);
            for (int t = 0; t < nbin_vals; ++t) h[11 + nvol + t] = mp_h(t);
          }
        }
#ifdef MPI_PARALLEL
        PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, h.data(),
                                          static_cast<int>(h.size()), MPI_PARTHENON_REAL,
                                          MPI_SUM, MPI_COMM_WORLD));
#endif
        MultipoleMoments mm;
        mm.four_pi_G = four_pi_G;
        mm.M = h[0];
        const Real invM = (h[0] != 0.0) ? 1.0 / h[0] : 0.0;
        mm.cx = h[1] * invM;
        mm.cy = h[2] * invM;
        mm.cz = h[3] * invM;
        // Second moments of mass about the COM: mu_ij = <x_i x_j> - M c_i c_j.
        const Real muxx = h[4] - mm.M * mm.cx * mm.cx;
        const Real muyy = h[5] - mm.M * mm.cy * mm.cy;
        const Real muzz = h[6] - mm.M * mm.cz * mm.cz;
        const Real muxy = h[7] - mm.M * mm.cx * mm.cy;
        const Real muxz = h[8] - mm.M * mm.cx * mm.cz;
        const Real muyz = h[9] - mm.M * mm.cy * mm.cz;
        // Traceless Cartesian quadrupole Q_ij = 3 mu_ij - delta_ij tr(mu).
        const Real tr = muxx + muyy + muzz;
        mm.Qxx = 3.0 * muxx - tr;
        mm.Qyy = 3.0 * muyy - tr;
        mm.Qzz = 3.0 * muzz - tr;
        mm.Qxy = 3.0 * muxy;
        mm.Qxz = 3.0 * muxz;
        mm.Qyz = 3.0 * muyz;
        grav_pkg->UpdateParam("grav_multipole_moments", mm);

        // --- Build the cumulative shell-split arrays ---------------------------------
        // Bin index (kb+1) in `cum`, where kb is the field point's shell and kb = -1 means
        // "inside r_lo". Inner slots accumulate INWARD, outer slots OUTWARD.
        // The core (r <= r_lo) inner moments are recovered by SUBTRACTION -- total minus the
        // sum over bins -- which is why AccumulateShellBins can skip 52% of the cells.
        if (mp_split) {
          auto ms = grav_pkg->Param<MultipoleShells>("grav_mp_shells");
          ms.r_lo = r_ins;
          ms.inv_dr = mp_inv_dr;
          ms.four_pi_G = four_pi_G;
          ms.nshell = mp_nshell;
          ms.bx0 = msz.xmin(parthenon::X1DIR); ms.bx1 = msz.xmax(parthenon::X1DIR);
          ms.by0 = msz.xmin(parthenon::X2DIR); ms.by1 = msz.xmax(parthenon::X2DIR);
          ms.bz0 = msz.xmin(parthenon::X3DIR); ms.bz1 = msz.xmax(parthenon::X3DIR);
          ms.active = true;
          // AUDIT F-38 (2026-08-28): was hardwired `true` with no deck key and no OFF state,
          // unlike every other switch in this module (mp_shell_split, mp_deterministic,
          // packed_bc, use_swindle all reproduce the prior arithmetic when off). A
          // result-changing improvement with no OFF state cannot be bisected against if a
          // gravity regression appears later, and cannot reproduce any result predating
          // 2026-08-16. Default stays true, so this is bit-identical.
          ms.uniform_split = grav_pkg->Param<bool>("mp_uniform_split");

          const Real *vol = h.data() + 11;              // whole-mesh volume moments (10)
          const Real *bins = h.data() + 11 + nvol;      // [bin][rho(21) | volume(21)]
          // Box-mean density. h[0] is the total mass (rho-weighted monopole), vol[0] the total
          // volume -- both discrete sums over the same cells, so their ratio is exactly the
          // density that, spread uniformly, carries the same mass.
          // Audit G-5: rho_bar must be 0 when the uniform-box split is off, or the BC is that of a
          // zero-mass source (98.5 % error on poisson_test).
          ms.rho_bar =
              (ms.uniform_split && vol[0] > 0.0) ? h[0] / vol[0] : 0.0;

          // Residual moments: delta = rho-weighted - rho_bar * volume-weighted, per bin. This
          // is why both twins were accumulated: rho_bar is only knowable after the reduction.
          std::vector<Real> dbin(static_cast<std::size_t>(mp_nshell) * kMPNSlot);
          for (int kbin = 0; kbin < mp_nshell; ++kbin)
            for (int sl = 0; sl < kMPNSlot; ++sl) {
              const std::size_t src = static_cast<std::size_t>(kbin) * 2 * kMPNSlot + sl;
              dbin[static_cast<std::size_t>(kbin) * kMPNSlot + sl] =
                  bins[src] - ms.rho_bar * bins[src + kMPNSlot];
            }
          // Core (r <= r_lo) inner moments of delta-rho, by subtraction from the whole-mesh
          // totals -- which is what lets AccumulateShellBins skip the interior 52% of cells.
          Real core[10];
          for (int sl = 0; sl < 10; ++sl) {
            Real sum = 0.0;
            for (int kbin = 0; kbin < mp_nshell; ++kbin)
              sum += dbin[static_cast<std::size_t>(kbin) * kMPNSlot + sl];
            core[sl] = (h[sl] - ms.rho_bar * vol[sl]) - sum;
          }
          auto cum_h = Kokkos::create_mirror_view(ms.cum);
          for (int sl = 0; sl < 10; ++sl) cum_h(sl) = core[sl];
          for (int sl = 10; sl < kMPNSlot; ++sl) {
            Real sum = 0.0;
            for (int kbin = 0; kbin < mp_nshell; ++kbin)
              sum += dbin[static_cast<std::size_t>(kbin) * kMPNSlot + sl];
            cum_h(sl) = sum;
          }
          for (int kbin = 0; kbin < mp_nshell; ++kbin) {
            const std::size_t dst = static_cast<std::size_t>(kbin + 1) * kMPNSlot;
            const std::size_t prv = static_cast<std::size_t>(kbin) * kMPNSlot;
            const Real *bk = dbin.data() + static_cast<std::size_t>(kbin) * kMPNSlot;
            for (int sl = 0; sl < 10; ++sl) cum_h(dst + sl) = cum_h(prv + sl) + bk[sl];
            for (int sl = 10; sl < kMPNSlot; ++sl) cum_h(dst + sl) = cum_h(prv + sl) - bk[sl];
          }
          Kokkos::deep_copy(ms.cum, cum_h);
          grav_pkg->UpdateParam("grav_mp_shells", ms);
        }

        // --- Multipole exterior-validity diagnostic (2026-08-15) ---------------------
        // grav-mp-extmass = M(r > R_ins)/M_total. The expansion evaluated on a domain FACE
        // is only entitled to expand mass interior to that face; this is the fraction it is
        // not entitled to. grav-mp-comoff = |COM|/R_ins guards the assumption that measuring
        // r from the origin (which the reduction must, since the COM is its own output) is
        // the same as measuring it from the expansion centre.
        if (grav_pkg->AllParams().hasKey("grav_mp_extmass")) {
          grav_pkg->UpdateParam("grav_mp_extmass",
                                Real((h[0] != 0.0) ? h[10] / h[0] : 0.0));
          const Real com = std::sqrt(mm.cx * mm.cx + mm.cy * mm.cy + mm.cz * mm.cz);
          grav_pkg->UpdateParam("grav_mp_comoff", Real((r_ins > 0.0) ? com / r_ins : 0.0));
        }
      }
    }

  }

  // Read outside the rhs_consumed gate so a skipped (unconsumed) call writes the RHS with the
  // previous mean rather than zero. Inert without the Jeans swindle (all decks).
  if (use_swindle) grav_mean_rho = grav_pkg->Param<Real>("grav_mean_rho");

  // --- Fill rhs over entire domain (incl. ghosts) ----------------------------
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::SetRHS", parthenon::DevExecSpace(), 0, nblocks - 1,
      kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real rho = (cons_pack(b, IDN, k, j, i) > rho_floor ? cons_pack(b, IDN, k, j, i) : rho_floor);
        rhs_pack(b, te, grav::rhs(), k, j, i) = four_pi_G * (rho - grav_mean_rho);
      });


  // --- Multipole boundary lift ----------------------------------------------
  // The Poisson solver runs a LINEAR operator with HOMOGENEOUS Dirichlet ghosts on the
  // multipole faces (see PoissonEquation::SetBoundary). The inhomogeneous exterior-
  // expansion face value Phi_mp is folded into the RHS here: for an interior cell adjacent
  // to a boundary face, the true ghost 2*Phi_mp - phi_int vs the homogeneous ghost -phi_int
  // differ by 2*Phi_mp, which the discrete Laplacian adds as +2*Phi_mp/dx_n^2. Moving it to
  // the RHS:  b -= 2*Phi_mp(face) / dx_n^2  at each boundary-adjacent interior cell (edges/
  // corners get one term per touching face). Post-solve, the enrolled per-block MultipoleBC
  // restores the physical ghost values for ApplyGravitySource.
  if (has_multipole) {
    const MultipoleMoments mm =
        grav_pkg->Param<MultipoleMoments>("grav_multipole_moments");
    // Shell-split form when active (see multipole.hpp): handles boundary points that are not
    // exterior to the mass, which on this box is every one of them.
    const bool mp_split_bc = grav_pkg->Param<bool>("mp_shell_split");
    MultipoleShells ms_bc;
    if (mp_split_bc) ms_bc = grav_pkg->Param<MultipoleShells>("grav_mp_shells");
    const auto &bctype = grav_pkg->Param<std::array<int, 6>>("grav_bc_face_type");
    for (int b = 0; b < nblocks; ++b) {
      auto *pmb = md->GetBlockData(b)->GetBlockPointer();
      for (int f = 0; f < 6; ++f) {
        if (bctype[f] != 3) continue;
        if (pmb->boundary_flag[f] != parthenon::BoundaryFlag::user) continue;
        const int dd = f / 2;         // 0->x1, 1->x2, 2->x3
        const bool inner = (f % 2 == 0);
        const int foff = inner ? 0 : 1; // Xf face-index offset for this side
        // Interior cell slab adjacent to this face (one cell thick along dd).
        const int i0 = (dd == 0) ? (inner ? ib.s : ib.e) : ib.s;
        const int i1 = (dd == 0) ? (inner ? ib.s : ib.e) : ib.e;
        const int j0 = (dd == 1) ? (inner ? jb.s : jb.e) : jb.s;
        const int j1 = (dd == 1) ? (inner ? jb.s : jb.e) : jb.e;
        const int k0 = (dd == 2) ? (inner ? kb.s : kb.e) : kb.s;
        const int k1 = (dd == 2) ? (inner ? kb.s : kb.e) : kb.e;
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "SG::MultipoleRHSLift", parthenon::DevExecSpace(),
            b, b, k0, k1, j0, j1, i0, i1,
            KOKKOS_LAMBDA(const int bb, const int k, const int j, const int i) {
              const auto &coords = cons_pack.GetCoords(bb);
              Real px = coords.Xc<parthenon::X1DIR>(k, j, i);
              Real py = coords.Xc<parthenon::X2DIR>(k, j, i);
              Real pz = coords.Xc<parthenon::X3DIR>(k, j, i);
              Real dxn;
              if (dd == 0) {
                px = coords.Xf<parthenon::X1DIR>(k, j, i + foff);
                dxn = coords.Dxc<parthenon::X1DIR>(k, j, i);
              } else if (dd == 1) {
                py = coords.Xf<parthenon::X2DIR>(k, j + foff, i);
                dxn = coords.Dxc<parthenon::X2DIR>(k, j, i);
              } else {
                pz = coords.Xf<parthenon::X3DIR>(k + foff, j, i);
                dxn = coords.Dxc<parthenon::X3DIR>(k, j, i);
              }
              const Real val = (mp_split_bc && ms_bc.active)
                                   ? MultipoleShellPhi(ms_bc, px, py, pz)
                                   : MultipolePhi(mm, px, py, pz);
              rhs_pack(bb, te, grav::rhs(), k, j, i) -= 2.0 * val / (dxn * dxn);
            });
      }
    }
  }
}

// ApplyGravitySource: momentum += rho * g * dt, energy += flux-weighted work.
// Flux-weighted energy (Artemis style) uses the hydro mass flux across each face
// to compute gravitational work. Requires hydro fluxes still be in memory.
TaskStatus ApplyGravitySource(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real beta_dt) {
  auto pm = md->GetParentPointer();

  // Pack hydro cons + prim + phi, and cons-with-fluxes for mass flux access.
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto &resolved = pm->resolved_packages;
  auto desc_phi = parthenon::MakePackDescriptor<grav::phi>(resolved.get());
  auto phi_pack = desc_phi.GetPack(md);

  // Mass flux: pack the "cons" variable AND its fluxes BY NAME (density flux =
  // component IDN). Packing by name rather than the {Independent} metadata flag
  // is required here: grav::phi is also Independent + WithFluxes, so selecting by
  // flag would pull phi into the pack and make the flat IDN index point at phi's
  // flux instead of the gas mass flux whenever phi sorts ahead of cons.
  auto cons_flx_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"},
                                                  std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  const int nblocks = md->NumBlocks();
  const int ndim = pm->ndim;
  const bool multi_d = (ndim > 1);
  const bool three_d = (ndim > 2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::ApplyGravity", parthenon::DevExecSpace(),
      0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &cons_flx = cons_flx_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real dx1 = coords.Dxc<1>(k, j, i);
        const Real dx2 = multi_d ? coords.Dxc<2>(k, j, i) : 1.0;
        const Real dx3 = three_d ? coords.Dxc<3>(k, j, i) : 1.0;

        const Real hdtodx1 = 0.5 * beta_dt / dx1;
        const Real hdtodx2 = multi_d ? 0.5 * beta_dt / dx2 : 0.0;
        const Real hdtodx3 = three_d ? 0.5 * beta_dt / dx3 : 0.0;

        // phi differences (Artemis convention: dpl = -(phi_c - phi_{c-1}))
        const Real phic = phi_pack(b, te, grav::phi(), k, j, i);
        const Real dpl1 = -(phic - phi_pack(b, te, grav::phi(), k, j, i - 1));
        const Real dpr1 = -(phi_pack(b, te, grav::phi(), k, j, i + 1) - phic);
        const Real dpl2 = multi_d
            ? -(phic - phi_pack(b, te, grav::phi(), k, j - 1, i))
            : 0.0;
        const Real dpr2 = multi_d
            ? -(phi_pack(b, te, grav::phi(), k, j + 1, i) - phic)
            : 0.0;
        const Real dpl3 = three_d
            ? -(phic - phi_pack(b, te, grav::phi(), k - 1, j, i))
            : 0.0;
        const Real dpr3 = three_d
            ? -(phi_pack(b, te, grav::phi(), k + 1, j, i) - phic)
            : 0.0;

        // Momentum update: rho * g * dt, centered difference of phi.
        const Real rho = prim(IDN, k, j, i);
        cons(IM1, k, j, i) += rho * hdtodx1 * (dpl1 + dpr1);
        if (multi_d) cons(IM2, k, j, i) += rho * hdtodx2 * (dpl2 + dpr2);
        if (three_d) cons(IM3, k, j, i) += rho * hdtodx3 * (dpl3 + dpr3);

        // Energy update: flux-weighted work.
        // mass_flux(face) = cons.flux(IVn, IDN, k, j, i).
        // Work done by gravity = sum over faces of (mass_flux * dphi/face) * (0.5 dt / dx).
        Real de = hdtodx1 * (cons_flx.flux(X1DIR, IDN, k, j, i)     * dpl1 +
                             cons_flx.flux(X1DIR, IDN, k, j, i + 1) * dpr1);
        if (multi_d) {
          de += hdtodx2 * (cons_flx.flux(X2DIR, IDN, k, j,     i) * dpl2 +
                           cons_flx.flux(X2DIR, IDN, k, j + 1, i) * dpr2);
        }
        if (three_d) {
          de += hdtodx3 * (cons_flx.flux(X3DIR, IDN, k,     j, i) * dpl3 +
                           cons_flx.flux(X3DIR, IDN, k + 1, j, i) * dpr3);
        }
        cons(IEN, k, j, i) += de;
      });

  return TaskStatus::complete;
}

} // namespace SelfGravity
