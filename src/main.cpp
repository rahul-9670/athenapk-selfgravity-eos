// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE");

#include <sstream>

// Parthenon headers
#include "bvals/boundary_conditions_generic.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "parthenon_manager.hpp"

// AthenaPK headers
#include "bvals/boundary_conditions_apk.hpp"
#include "hydro/double_flux.hpp"
#include "hydro/hydro.hpp"
#include "hydro/hydro_driver.hpp"
#include "main.hpp"

#include "outputs/restart_hdf5.hpp"
#include "pgen/pgen.hpp"
#include "tracers/tracers.hpp"
// Initialize defaults for package specific callback functions
namespace Hydro {
InitPackageDataFun_t ProblemInitPackageData = nullptr;
SourceFun_t ProblemSourceFirstOrder = nullptr;
SourceFun_t ProblemSourceStrangSplit = nullptr;
SourceFun_t ProblemSourceUnsplit = nullptr;
EstimateTimestepFun_t ProblemEstimateTimestep = nullptr;
std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock = nullptr;
} // namespace Hydro

namespace Tracers {
InitPackageDataFun_t ProblemInitTracerData = nullptr;
SeedInitialFun_t ProblemSeedInitialTracers = nullptr;
FillTracersFun_t ProblemFillTracers = nullptr;
} // namespace Tracers

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos

  // Redefine defaults
  pman.app_input->ProcessPackages = Hydro::ProcessPackages;
  pman.app_input->PreStepMeshUserWorkInLoop = Hydro::PreStepMeshUserWorkInLoop;
  const auto problem = pman.pinput->GetOrAddString("job", "problem_id", "unset");

  // Problem generators of this tree (see pgen/CMakeLists.txt). Adding one means its .cpp, its
  // namespace declaration in pgen/pgen.hpp, its entry in pgen/CMakeLists.txt and a branch here.
  if (problem == "collapse_be") {
    pman.app_input->ProblemGenerator = collapse_be::ProblemGenerator;
    // WP-13: restart-safe Param registration. ProblemGenerator is NOT called on restart, so
    // this hook (run from Hydro::Initialize on both fresh start and resume) is what keeps
    // collapse_be_rhocrit alive -- and with it the ApplyBarotropicCooling task, which carries
    // the outside-sphere momentum BC. Without it every restart silently dropped that task.
    Hydro::ProblemInitPackageData = collapse_be::ProblemInitPackageData;
  } else if (problem == "poisson_test") {
    // Self-gravity analytic gate (uniform sphere); see pgen/pgen.hpp.
    pman.app_input->ProblemGenerator = poisson_test::ProblemGenerator;
  } else if (problem == "grav_analytic") {
    // Self-gravity validation suite (validation_tests/self_gravity): exact-potential fields.
    pman.app_input->ProblemGenerator = grav_analytic::ProblemGenerator;
  } else if (problem == "jeans_wave") {
    pman.app_input->ProblemGenerator = jeans_wave::ProblemGenerator;
  } else if (problem == "grav_slab") {
    pman.app_input->ProblemGenerator = grav_slab::ProblemGenerator;
  } else if (problem == "polytrope") {
    pman.app_input->ProblemGenerator = polytrope::ProblemGenerator;
  } else if (problem == "eos_test") {
    pman.app_input->ProblemGenerator = eos_test::ProblemGenerator;
  } else if (problem == "profile1d") {
    pman.app_input->ProblemGenerator = profile1d::ProblemGenerator;
    Hydro::ProblemInitPackageData = profile1d::ProblemInitPackageData;
  } else if (problem == "fofc_scalar") {
    // First-order flux correction x passive-scalar consistency gate (2026-09-16).
    pman.app_input->ProblemGenerator = fofc_scalar::ProblemGenerator;
  } else {
    std::stringstream msg;
    msg << "Problem ID '" << problem
        << "' is not available in this build. Available: collapse_be, poisson_test, "
           "grav_analytic, jeans_wave, grav_slab, polytrope, eos_test, profile1d, fofc_scalar.";
    PARTHENON_THROW(msg);
  }

  // The "reflecting" BC was removed on 2026-08-10 with the pure-hydro path. Its
  // implementation opened with PARTHENON_REQUIRE_THROWS(fluid == Fluid::euler,
  // "Reflecting boundary conditions for MHD need special treatment.") -- so in an MHD-only
  // build it could only ever abort. Production uses `diode` (registered below) or
  // Parthenon's built-in `outflow`/`periodic`.
  using BF = parthenon::BoundaryFace;
  using parthenon::BoundaryFunction::BCSide;

  // VALIDATION B1: inflow-suppressing outflow ("diode"). Parthenon's `outflow` is a plain
  // zero-gradient copy and admits inflow -- measured as ~-178 per face in WP-6's cons-Mout,
  // the source of the production mass rise. Registering it as a SEPARATE named BC leaves
  // `outflow` untouched, so every existing deck stays bit-identical; switching production to
  // it is result-changing and therefore an explicit, deliberate deck edit.
  const std::string DIODE = "diode";
  using Hydro::BoundaryFunction::DiodeBC;
  pman.app_input->RegisterBoundaryCondition(BF::inner_x1, DIODE,
                                            DiodeBC<X1DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x1, DIODE,
                                            DiodeBC<X1DIR, BCSide::Outer>);
  pman.app_input->RegisterBoundaryCondition(BF::inner_x2, DIODE,
                                            DiodeBC<X2DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x2, DIODE,
                                            DiodeBC<X2DIR, BCSide::Outer>);
  pman.app_input->RegisterBoundaryCondition(BF::inner_x3, DIODE,
                                            DiodeBC<X3DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x3, DIODE,
                                            DiodeBC<X3DIR, BCSide::Outer>);
  // VALIDATION-ONLY closed wall (self-gravity L6c): mirror + negate normal momentum and normal B.
  const std::string REFLV = "reflecting_val";
  using Hydro::BoundaryFunction::ReflectValBC;
  pman.app_input->RegisterBoundaryCondition(BF::inner_x1, REFLV, ReflectValBC<X1DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x1, REFLV, ReflectValBC<X1DIR, BCSide::Outer>);
  pman.app_input->RegisterBoundaryCondition(BF::inner_x2, REFLV, ReflectValBC<X2DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x2, REFLV, ReflectValBC<X2DIR, BCSide::Outer>);
  pman.app_input->RegisterBoundaryCondition(BF::inner_x3, REFLV, ReflectValBC<X3DIR, BCSide::Inner>);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x3, REFLV, ReflectValBC<X3DIR, BCSide::Outer>);
  // VALIDATION-ONLY exact homologous ghosts for eos_test mode = compress (validation_tests E8).
  pman.app_input->RegisterBoundaryCondition(BF::inner_x1, "eos_homologous",
                                            eos_test::HomologousInnerX1);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x1, "eos_homologous",
                                            eos_test::HomologousOuterX1);
  // Mesh-level fused diode BC hook; only acts when hydro/fused_bc_mesh = true.
  Hydro::BoundaryFunction::RegisterFusedDiodeBCMesh(pman.pinput.get());

  pman.ParthenonInitPackagesAndMesh();


  // hydro/double_flux on a refined mesh: install the EOS-consistent fill hooks (double_flux.hpp). Off -> nothing set,
  // Parthenon's hooks stay null. (The initial exchange inside ParthenonInitPackagesAndMesh runs before this, with
  // dflux_p still zero: FixE skips non-positive pressures, so the first stage's level-jump ghosts are unfixed.)
  {
    auto hydro_pkg = pman.pmesh->packages.Get("Hydro");
    if (hydro_pkg->AllParams().hasKey("dflux_ghost_fill") && hydro_pkg->Param<bool>("dflux_ghost_fill")) {
      pman.pmesh->PreSendBoundBufsHook = [](parthenon::MeshData<parthenon::Real> *md) {
        Hydro::DoubleFluxRefreshP(md);
      };
      pman.pmesh->PostRegridHook = Hydro::DoubleFluxPostRegrid;
    }
  }

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Starting up hydro driver" << std::endl;
  }

  {
    Hydro::HydroDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

    driver.Execute();
  }

  pman.ParthenonFinalize();


  return (0);
}
