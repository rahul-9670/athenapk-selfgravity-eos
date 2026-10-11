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
#include "radiation/radiation_sn.hpp"
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
  } else if (problem == "rad_pulse") {
    // Radiation (Sn) validation problem; registers no Params and has no restart-sensitive task.
    pman.app_input->ProblemGenerator = rad_pulse::ProblemGenerator;
  } else if (problem == "rad_shock") {
    // Lowrie & Edwards radiative shock (radiation in a moving flow).
    pman.app_input->ProblemGenerator = rad_shock::ProblemGenerator;
  } else if (problem == "poisson_test") {
    // Self-gravity analytic gate (uniform sphere); see pgen/pgen.hpp.
    pman.app_input->ProblemGenerator = poisson_test::ProblemGenerator;
  } else if (problem == "grav_analytic") {
    // Self-gravity validation suite (validation/self_gravity): exact-potential fields.
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
           "grav_analytic, jeans_wave, grav_slab, polytrope, eos_test, profile1d, fofc_scalar, rad_pulse, rad_shock.";
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
  // VALIDATION-ONLY exact homologous ghosts for eos_test mode = compress (validation E8).
  pman.app_input->RegisterBoundaryCondition(BF::inner_x1, "eos_homologous",
                                            eos_test::HomologousInnerX1);
  pman.app_input->RegisterBoundaryCondition(BF::outer_x1, "eos_homologous",
                                            eos_test::HomologousOuterX1);
  // Mesh-level fused diode/Sn BC hook; only acts when hydro/fused_bc_mesh = true.
  Hydro::BoundaryFunction::RegisterFusedDiodeBCMesh(pman.pinput.get());


  // --------------------------------------------------------------------------------------
  // Sn RESTART PRE-CHECK -- ask the restart FILE whether it carries the ordinate field.
  //
  // WHY HERE AND NOWHERE ELSE. parthenon's RestartReader is alive only between
  // ParthenonInitEnv() (which opens the .rhdf, parthenon_manager.cpp:106-119) and the end of
  // ParthenonInitPackagesAndMesh() (which nulls it at :223). Radiation::Sn::Initialize runs
  // inside that window, from ProcessPackages, and is handed nothing but ParameterInput -- so
  // this is the only point at which the question can be asked and the answer still be in
  // time to choose sn.u's metadata. This replaces a re-seeding of the ordinates at every
  // restart from their own first two angular moments, which did not conserve energy
  // (it created ~4200 code units of energy at one restart in a collapse run).
  //
  // EVERY RANK opens its own reader (parthenon_manager.cpp:110), so this needs no
  // communication and cannot disagree between ranks.
  //
  // NOT WRITTEN INTO pman.pinput deliberately: a fresh (non-restart) run must keep an
  // untouched ParameterInput so its outputs' Input string is exactly its deck, and a deck
  // without a <radiation> block must not acquire one.
  if (pman.restartReader != nullptr) {
    Radiation::Sn::g_restart_is_restart = true;
    bool has_u = false;
    try {
      has_u = pman.restartReader->VariableExists(Radiation::Sn::USolName());
    } catch (std::exception &) {
      has_u = false; // unreadable file: treat as "no ordinates", i.e. re-seed
    }
    Radiation::Sn::g_restart_u_present = has_u;
    Radiation::Sn::g_restart_u_ncomp = -1;
    // R-F1: the file's own cycle number. Sn::Initialize re-seeds the ordinates only at cycle 0; the pin's
    // parthenon/time/ncycle is stale at that point (0 in every restart of a run that began at t = 0).
    Radiation::Sn::g_restart_ncycle = -1;
    {
      auto *h5c = dynamic_cast<parthenon::RestartReaderHDF5 *>(pman.restartReader.get());
      if (h5c != nullptr) {
        try {
          Radiation::Sn::g_restart_ncycle = h5c->GetAttr<int>("Info", "NCycle");
        } catch (std::exception &) {
          Radiation::Sn::g_restart_ncycle = -1;
        }
      }
    }
    if (has_u) {
      // Component count, so a changed ordinate set is caught BEFORE parthenon's ReadBlocks
      // builds a hyperslab of the new nfa on the file's old nfa and H5Dread throws.
      // Info/OutputDatasetNames and Info/NumComponents are written together, entry for
      // entry, by parthenon_hdf5.cpp:478-480. GetAttrVec lives on the HDF5 reader rather
      // than the RestartReader interface, hence the cast; -1 (unknown) if it is not one.
      auto *h5 = dynamic_cast<parthenon::RestartReaderHDF5 *>(pman.restartReader.get());
      if (h5 != nullptr) {
        try {
          const auto names = h5->GetAttrVec<std::string>("Info", "OutputDatasetNames");
          // NumComponents is written from a std::vector<std::size_t> (parthenon_hdf5.cpp:459), and
          // parthenon's attribute reader TYPE-CHECKS with H5Tequal and throws on a mismatch --
          // so GetAttrVec<int> silently lost this to the catch below (observed: ncomp = -1 on a
          // file that plainly had NumComponents = [9 1 1 1 1 32]). Read the type it was written as.
          const auto ncmp = h5->GetAttrVec<std::size_t>("Info", "NumComponents");
          for (std::size_t i = 0; i < names.size() && i < ncmp.size(); ++i) {
            if (names[i] == Radiation::Sn::USolName())
              Radiation::Sn::g_restart_u_ncomp = static_cast<int>(ncmp[i]);
          }
        } catch (std::exception &) {
          Radiation::Sn::g_restart_u_ncomp = -1;
        }
      }
    }
  }

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
