//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
#include "../pgen/pgen.hpp"
#include "../recon/dc_simple.hpp"
#include "../recon/limo3_simple.hpp"
#include "../recon/plm_simple.hpp"
#include "../recon/ppm_simple.hpp"
#include "../recon/weno3_simple.hpp"
#include "../recon/wenoz_simple.hpp"
#include "../refinement/refinement.hpp"
#include "../tracers/tracers.hpp"
#include "../units.hpp"
#include "../units/physical_units.hpp"
#include "defs.hpp"
#include "../diagnostics/cons_diag.hpp"
#include <solvers/bicgstab_solver.hpp>
#include <solvers/mg_solver.hpp>
#include "glmmhd/glmmhd.hpp"
#include "double_flux.hpp"
#include "hydro.hpp"
#include "interface/params.hpp"
#include "outputs/outputs.hpp"
#include "prolongation/custom_ops.hpp"
#include "rsolvers/rsolvers.hpp"
#include "utils/error_checking.hpp"
#include "../self_gravity/self_gravity.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// define the "physics" package Hydro, which  *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace Hydro {

using parthenon::HistoryOutputVar;

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  parthenon::Packages_t packages;
  packages.Add(Hydro::Initialize(pin.get()));
  packages.Add(Tracers::Initialize(pin.get()));
  if (pin->GetOrAddBoolean("physics", "self_gravity", false)) {
    packages.Add(SelfGravity::Initialize(pin.get()));
  }
  // This tree carries hydro/MHD + self-gravity + the tabulated EOS only. Refuse switches for
  // packages that are not part of it rather than silently running without them.
  for (const char *phys : {"radiation", "chemistry", "dust"}) {
    PARTHENON_REQUIRE_THROWS(!pin->GetOrAddBoolean("physics", phys, false),
                             "<physics> " + std::string(phys) + " is not part of this build.");
  }
  PARTHENON_REQUIRE_THROWS(!pin->GetOrAddBoolean("sinks", "enabled", false),
                           "<sinks> enabled: sink particles are not part of this build.");
  return packages;
}

// Calculate mininum dx, which is used in calculating the divergence cleaning speed c_h
// TODO(PG) eventually move to calculating the timestep once the timestep calc
// has been moved to be done before Step()
Real CalculateGlobalMinDx(MeshData<Real> *md) {
  auto *pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real mindx = std::numeric_limits<Real>::max();

  bool nx2 = prim_pack.GetDim(2) > 1;
  bool nx3 = prim_pack.GetDim(3) > 1;
  pmb->par_reduce(
      "CalculateGlobalMinDx", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmindx) {
        const auto &coords = prim_pack.GetCoords(b);
        lmindx = fmin(lmindx, coords.Dxc<1>(k, j, i));
        if (nx2) {
          lmindx = fmin(lmindx, coords.Dxc<2>(k, j, i));
        }
        if (nx3) {
          lmindx = fmin(lmindx, coords.Dxc<3>(k, j, i));
        }
      },
      Kokkos::Min<Real>(mindx));

  return mindx;
}

// Using this per cycle function to populate various variables in
// Params that require global reduction *and* need to be set/known when
// the task list is constructed (versus when the task list is being executed).
// TODO(next person touching this function): If more/separate feature are required
// please separate concerns.
void PreStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, SimTime &tm) {
  auto hydro_pkg = pmesh->packages.Get("Hydro");

  // ---- TABULATED-EOS OUT-OF-RANGE CLAMP REPORT (2026-08-15) --------------------------------
  // EosTable::bilin edge-clamps index AND weight, i.e. it nearest-neighbour-extrapolates off
  // the end of the grid and returns a plausible number. That is the right numerics and the
  // wrong silence.
  // The kernels now tally the clamps (numerics-neutral -- the counter is write-only from the
  // kernel and feeds no flux); this drains the tally once per cycle and says so out loud.
  // Reduced over ranks so the message is a global statement, not rank 0's private view.
  // Guarded on the FLUID, not just on the key: Param<T> is type-checked, so asking for an
  // AdiabaticGLMMHDEOS in a fluid=hydro run would throw rather than no-op.
  // Fluid FIRST: the key is only registered on the glmmhd branch, and && short-circuits.
  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd &&
      hydro_pkg->Param<bool>("eos_clamp_report")) {
    const auto &etab = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos").GetEosTable();
    unsigned long long c[EOSTable::EosTable::kNClampCtr] = {0};
    bool have_eos_ctr = etab.ReadAndResetClampCounts(c);
    if (have_eos_ctr) {
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, c, EOSTable::EosTable::kNClampCtr,
                                        MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
#endif
      // 2026-08-21: slots 0-5 are real CELL-STATE lookups, 6-11 are root-finder TRIAL
      // (probe) lookups. The WARNING fires on the state bank only -- probes step outside the
      // table by design and fired every cycle, which masked any genuine clamp. Probes are
      // still reported, because a probe count that suddenly explodes is a solver pathology.
      unsigned long long tot = 0, tot_probe = 0;
      for (int q = 0; q < EOSTable::EosTable::kNClampState; ++q) tot += c[q];
      for (int q = EOSTable::EosTable::kNClampState;
           q < EOSTable::EosTable::kNClampCtr; ++q) tot_probe += c[q];
      if ((tot > 0 || tot_probe > 0) && parthenon::Globals::my_rank == 0) {
        Real lrl, lrh, lel, leh, lTl, lTh;
        etab.AxisRanges(lrl, lrh, lel, leh, lTl, lTh);
        std::cout << (tot > 0 ? "### WARNING EOS table: " : "### NOTE EOS table: ") << tot
                  << " CELL-STATE (+ " << tot_probe << " solver-probe)"
                  << " out-of-range lookups were EDGE-CLAMPED in the previous cycle"
                     " (nearest-neighbour extrapolation; the returned value is NOT"
                     " interpolated). Breakdown  rho<" << lrl << ": " << c[0]
                  << "  rho>" << lrh << ": " << c[1] << "  esp<" << lel << ": " << c[2]
                  << "  esp>" << leh << ": " << c[3] << "  T<" << lTl << ": " << c[4]
                  << "  T>" << lTh << ": " << c[5]
                  << " | PROBE  rho<: " << c[6] << "  rho>: " << c[7]
                  << "  esp<: " << c[8] << "  esp>: " << c[9]
                  << "  T<: " << c[10] << "  T>: " << c[11]
                  << "   (axes are log10; rho,esp in CODE units, T in K). With hydro/eos_extrapolate"
                     " (default) rho< (eos_interp=p), esp< and T< lookups are CONTINUED as an ideal gas,"
                     " not clamped; rho>/esp>/T> are clamped. CELL-STATE clamps"
                     " mean the table does not cover the run -- regenerate it"
                     " (src/eos/gen_eos_table.py). PROBE clamps are root-finder trial values"
                     " pulled back by bisection and are benign unless they explode."
                  << std::endl;
      }
    }
  }

  // D1 instrument (hydro/d1_meminfo, default false): per-rank block count, coarse-fine / same-level
  // neighbour counts, refinement-level range and free/total device memory, printed on every regrid
  // and every d1_meminfo_every cycles. Measurement only (docs/validation D1_gpu_memory_imbalance).
  if (hydro_pkg->Param<bool>("d1_meminfo")) {
    int nblocks = static_cast<int>(pmesh->block_list.size());
    int ncf = 0, nsame = 0;
    int maxlev = -1, minlev = 1 << 30;
    for (auto const &pmb : pmesh->block_list) {
      const int mylev = pmb->loc.level();
      maxlev = std::max(maxlev, mylev);
      minlev = std::min(minlev, mylev);
      for (auto const &nb : pmb->GetNeighbors()) {
        if (nb.loc.level() != mylev) {
          ncf++;
        } else {
          nsame++;
        }
      }
    }
    if (nblocks == 0) {
      minlev = -1;
    }
    const int every = hydro_pkg->Param<int>("d1_meminfo_every");
    const bool on_change = (nblocks != hydro_pkg->Param<int>("d1_last_nblocks"));
    const bool on_interval = (every > 0) && (tm.ncycle % every == 0);
    if (on_change || on_interval) {
      double freeGB = -1.0, totGB = -1.0;
#ifdef KOKKOS_ENABLE_CUDA
      size_t fb = 0, tb = 0;
      if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
        freeGB = static_cast<double>(fb) / (1024.0 * 1024.0 * 1024.0);
        totGB = static_cast<double>(tb) / (1024.0 * 1024.0 * 1024.0);
      }
#endif
      printf("[D1] cycle=%d rank=%d nblocks=%d coarse_fine_nbrs=%d same_level_nbrs=%d "
             "dev_free_GiB=%.2f dev_total_GiB=%.2f maxlev=%d minlev=%d trigger=%s\n",
             tm.ncycle, parthenon::Globals::my_rank, nblocks, ncf, nsame, freeGB, totGB,
             maxlev, minlev, on_change ? "regrid" : "interval");
      fflush(stdout);
      hydro_pkg->UpdateParam("d1_last_nblocks", nblocks);
    }
  }

  // Calculate hyperbolic divergence cleaning speed
  // TODO(pgrete) Calculating mindx is only required after remeshing. Need to
  // find a clean solution for this one-off global reduction.
  if (hydro_pkg->Param<bool>("calc_c_h") ||
      hydro_pkg->Param<DiffInt>("diffint") != DiffInt::none) {

    Real mindx = std::numeric_limits<Real>::max();
    // Going over default partitions. Not using a (new) single partition containing
    // all blocks here as this (default) split is also used main Step() function and
    // thus does not create an overhead (such as creating a new MeshBlockPack that is just
    // used here). All partitions are executed sequentially. Given that a par_reduce to a
    // host var is blocking it's save to dirctly use the return value.
    const int num_partitions = pmesh->DefaultNumPartitions();
    for (int i = 0; i < num_partitions; i++) {
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      mindx = std::min(mindx, CalculateGlobalMinDx(mu0.get()));
    }
#ifdef MPI_PARALLEL
    Real mins[3];
    mins[0] = mindx;
    mins[1] = hydro_pkg->Param<Real>("dt_hyp");
    mins[2] = hydro_pkg->Param<Real>("dt_diff");
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, mins, 3, MPI_PARTHENON_REAL, MPI_MIN,
                                      MPI_COMM_WORLD));

    hydro_pkg->UpdateParam("mindx", mins[0]);
    hydro_pkg->UpdateParam("dt_hyp", mins[1]);
    hydro_pkg->UpdateParam("dt_diff", mins[2]);
#else
    hydro_pkg->UpdateParam("mindx", mindx);
    // dt_hyp and dt_diff are already set directly in Params when they're calculated
#endif
    // Finally update c_h
    const auto &cfl_hyp = hydro_pkg->Param<Real>("cfl");
    const auto &dt_hyp = hydro_pkg->Param<Real>("dt_hyp");
    mindx = hydro_pkg->Param<Real>("mindx");
    hydro_pkg->UpdateParam("c_h", cfl_hyp * mindx / dt_hyp);
  }

  // DT REPORT (hydro/dt_report, default off => no output, no collective). Names the
  // constraint that set THIS step: the global hyperbolic and parabolic limits (reduced
  // above). If tm.dt sits below all of them, another package (gravity, the growth limit)
  // is the one binding. Added 2026-09-20 because the 20x dt collapse at cycle 688 of the
  // uncapped implicit leg was attributed to a mechanism without the binding constraint
  // ever being logged.
  if (hydro_pkg->AllParams().hasKey("dt_report") && hydro_pkg->Param<bool>("dt_report")) {
    if (parthenon::Globals::my_rank == 0) {
      printf("[DTREPORT] cycle=%d dt=%.6e dt_hyp=%.6e dt_diff=%.6e\n", tm.ncycle, tm.dt,
             hydro_pkg->Param<Real>("dt_hyp"), hydro_pkg->Param<Real>("dt_diff"));
      fflush(stdout);
    }
  }
}

template <Hst hst, int idx = -1>
Real HydroHst(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;
  const bool two_d = cons_pack.GetNdim() >= 2; // see the 1-D FIX note in the divb branch

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;

  // Sanity checks
  if ((hst == Hst::idx) && (idx < 0)) {
    PARTHENON_FAIL("Idx based hst output needs index >= 0");
  }
  Kokkos::parallel_reduce(
      "HydroHst",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        if (hst == Hst::idx) {
          lsum += cons(idx, k, j, i) * coords.CellVolume(k, j, i);
        } else if (hst == Hst::ekin) {
          lsum += 0.5 / cons(IDN, k, j, i) *
                  (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                   SQR(cons(IM3, k, j, i))) *
                  coords.CellVolume(k, j, i);
        } else if (hst == Hst::emag) {
          lsum += 0.5 *
                  (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                   SQR(cons(IB3, k, j, i))) *
                  coords.CellVolume(k, j, i);
          // relative divergence of B error, i.e., L * |div(B)| / |B|
        } else if (hst == Hst::divb) {
          // 1-D FIX (2026-08-16). The k+-1 term was guarded by `three_d` but the j+-1 term
          // was NOT guarded at all. In a 1-D mesh (nx2 = nx3 = 1) Parthenon allocates NO
          // ghost cells in x2, so `j+1` / `j-1` index outside the array and the run
          // SEGFAULTS -- in the history reduction, i.e. on the first output, immediately
          // after "Setup complete, executing driver...". Found 2026-08-16 while trying to
          // run the M1 diffusion benchmark (runs/ap_diffusion) in 1-D; it cost three
          // debugging cycles because the crash looks like it is in the physics.
          // Latent for production (3-D), real for every 1-D test problem in this build.
          Real divb =
              (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
          if (two_d) {
            divb +=
                (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
          }
          if (three_d) {
            divb += (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) /
                    coords.Dxc<3>(k, j, i);
          }

          Real abs_b = std::sqrt(SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                                 SQR(cons(IB3, k, j, i)));

          // same hardening as maxRelDivB above; abs_b + b_floor rather than an exact-zero test
          lsum += (abs_b != 0) ? 0.5 *
                                     (std::sqrt(SQR(coords.Dxc<1>(k, j, i)) +
                                                SQR(coords.Dxc<2>(k, j, i)) +
                                                SQR(coords.Dxc<3>(k, j, i)))) *
                                     std::abs(divb) / abs_b * coords.CellVolume(k, j, i)
                               : 0; // Add zero when abs_b ==0
        }
      },
      sum);

  return sum;
}

// WS-5c: peak (max over all cells) of the relative divergence error L*|div(B)|/|B|, for the
// div B fidelity audit. Registered with UserHistoryOperation::max so Parthenon takes the
// global max across blocks/ranks. Reports the WORST-cell cleaning error; near first-core
// formation the peak is what matters.
// NOTE (2026-08-20): the "relDivB" sum above is an UNNORMALIZED volume integral -- it accumulates
// 0.5*L*|divB|/|B| * CellVolume and is never divided by the total volume, so it is NOT a mean, and
// its value scales with the box. Do not compare relDivB across different box sizes. To read it as
// a volume-weighted mean, divide by the domain volume: on an L=16 box (volume 4096) the
// measured 0.775 is a mean relative cleaning error of 1.9e-04, against maxRelDivB = 2.3e-02
// worst-cell. (A previous version of this comment called the sum "a mean", which made the two
// numbers look mutually inconsistent -- a max cannot be below a mean.)
Real HydroHstMaxDivB(MeshData<Real> *md) {
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;
  const bool two_d = cons_pack.GetNdim() >= 2; // see the 1-D FIX note in HydroHst's divb
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real vmax = 0.0;
  Kokkos::parallel_reduce(
      "HydroHstMaxDivB",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        Real divb =
            (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) / coords.Dxc<1>(k, j, i);
        if (two_d) {
          divb += (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        }
        if (three_d) {
          divb += (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        }
        const Real abs_b = std::sqrt(SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                                     SQR(cons(IB3, k, j, i)));
        const Real dx = 0.5 * std::sqrt(SQR(coords.Dxc<1>(k, j, i)) +
                                        SQR(coords.Dxc<2>(k, j, i)) +
                                        SQR(coords.Dxc<3>(k, j, i)));
        // 2026-08-26: `abs_b != 0.0` guards only EXACT zero, so a cell with an anomalously
        // small |B| gave an unbounded relative error -- the same defect class as the rslaMax
        // scale-height division. MEASURED SAFE in the current campaign (min |B| over 48.9M
        // cells is 4.49e-02, and ZERO cells fall below 1e-6, because the uniform B0z threads
        // the whole domain and compression only strengthens it), so this changes nothing today;
        // it is hardened so a future IC with a weak or reversing field cannot silently produce
        // a meaningless maxRelDivB. Floor at a tiny fraction of the cell's own field scale.
        const Real b_floor = 1.0e-30;
        const Real rel = dx * std::abs(divb) / (abs_b + b_floor);
        if (rel > lmax) lmax = rel;
      },
      Kokkos::Max<Real>(vmax));
  return vmax;
}

// TOOD(pgrete) check is we can enlist this with FillDerived directly
// this is the package registered function to fill derived, here, convert the
// conserved variables to primitives
template <class T>
void ConsToPrim(MeshData<Real> *md) {
  const auto &eos =
      md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro")->Param<T>("eos");
  eos.ConservedToPrimitive(md);
}

// Add unsplit sources, i.e., source that are integrated in all stages of the
// explicit integration scheme.
// Note 1: Given that the sources are integrated in an unsplit manner, ensure
// that potential timestep constrains are also properly enforced when the
// respective source in active.
// Note 2: Directly update the "cons" variables based on the "prim" variables
// as the "cons" variables have already been updated when this function is called.
//----------------------------------------------------------------------------------------
//! B1-remainder: zero the inward part of the domain-boundary face fluxes. See hydro.hpp for
//! the rationale and the scope limits (hydro + scalars only, magnetic fluxes untouched).
TaskStatus ClampBoundaryFluxes(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto *pm = pmb->pmy_mesh;

  // Physical extent of the WHOLE mesh. A meshblock face coincides with it only on the domain
  // boundary; interior block faces must not be touched (they are genuine internal fluxes).
  // Same float-tolerance test cons_diag uses to identify surface cells.
  const Real xlo = pm->mesh_size.xmin(X1DIR), xhi = pm->mesh_size.xmax(X1DIR);
  const Real ylo = pm->mesh_size.xmin(X2DIR), yhi = pm->mesh_size.xmax(X2DIR);
  const Real zlo = pm->mesh_size.xmin(X3DIR), zhi = pm->mesh_size.xmax(X3DIR);
  const int ndim = pm->ndim;

  const int nhydro = hydro_pkg->Param<int>("nhydro");
  const int nscalars = hydro_pkg->Param<int>("nscalars");

  // Packed BY NAME: grav::phi is also Independent+WithFluxes, so a metadata-flag pack could put
  // phi's flux at the flat IDN index. Same trap cons_diag and SelfGravity::ApplyGravitySource
  // both document.
  auto cf = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"},
                                       std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ClampBoundaryFluxes", DevExecSpace(), 0, cf.GetDim(5) - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cfb = cf(b);
        const auto &coords = cf.GetCoords(b);
        const Real dx = coords.Dxc<1>(k, j, i);
        const Real dy = (ndim > 1) ? coords.Dxc<2>(k, j, i) : 1.0;
        const Real dz = (ndim > 2) ? coords.Dxc<3>(k, j, i) : 1.0;

        // Zero every hydro+scalar component of one face's flux. `sign` is +1 when a POSITIVE
        // flux points OUT of the domain (upper faces) and -1 on lower faces; the flux is
        // cleared only when it points IN, so genuine outflow is never touched.
        auto clamp = [&](const int dir, const int kk, const int jj, const int ii,
                         const Real sign) {
          const Real fm = cfb.flux(dir, IDN, kk, jj, ii);
          if (sign * fm >= 0.0) return; // outflow (or exactly zero) -- leave it alone
          cfb.flux(dir, IDN, kk, jj, ii) = 0.0;
          cfb.flux(dir, IM1, kk, jj, ii) = 0.0;
          cfb.flux(dir, IM2, kk, jj, ii) = 0.0;
          cfb.flux(dir, IM3, kk, jj, ii) = 0.0;
          cfb.flux(dir, IEN, kk, jj, ii) = 0.0;
          for (int n = 0; n < nscalars; ++n)
            cfb.flux(dir, nhydro + n, kk, jj, ii) = 0.0;
        };

        if (fabs(coords.Xf<1>(i) - xlo) < 1.0e-9 * dx) clamp(X1DIR, k, j, i, -1.0);
        if (fabs(coords.Xf<1>(i + 1) - xhi) < 1.0e-9 * dx) clamp(X1DIR, k, j, i + 1, 1.0);
        if (ndim > 1) {
          if (fabs(coords.Xf<2>(j) - ylo) < 1.0e-9 * dy) clamp(X2DIR, k, j, i, -1.0);
          if (fabs(coords.Xf<2>(j + 1) - yhi) < 1.0e-9 * dy) clamp(X2DIR, k, j + 1, i, 1.0);
        }
        if (ndim > 2) {
          if (fabs(coords.Xf<3>(k) - zlo) < 1.0e-9 * dz) clamp(X3DIR, k, j, i, -1.0);
          if (fabs(coords.Xf<3>(k + 1) - zhi) < 1.0e-9 * dz) clamp(X3DIR, k + 1, j, i, 1.0);
        }
      });
  return TaskStatus::complete;
}

TaskStatus AddUnsplitSources(MeshData<Real> *md, const SimTime &tm, const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  // GLM/Dedner sources.
  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    hydro_pkg->Param<GLMMHD::SourceFun_t>("glmmhd_source")(md, beta_dt);
  }
  if (ProblemSourceUnsplit != nullptr) {
    ProblemSourceUnsplit(md, tm, beta_dt);
  }

  return TaskStatus::complete;
}

TaskStatus AddSplitSourcesFirstOrder(MeshData<Real> *md, const SimTime &tm) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if (ProblemSourceFirstOrder != nullptr) {
    ProblemSourceFirstOrder(md, tm, tm.dt);
  }
  return TaskStatus::complete;
}

TaskStatus AddSplitSourcesStrang(MeshData<Real> *md, const SimTime &tm) {
  if (ProblemSourceStrangSplit != nullptr) {
    ProblemSourceStrangSplit(md, tm, tm.dt);
  }
  return TaskStatus::complete;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Hydro");

  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  bool pack_in_one = pin->GetOrAddBoolean("parthenon/mesh", "pack_in_one", true);
  pkg->AddParam<>("pack_in_one", pack_in_one);

  // Provenance: stamp the unit system into the Hydro package as Restart-flagged params so every
  // restart/output records the exact code->cgs mapping the run used. Numerically inert (metadata
  // only; no kernel reads these). Reading an older restart that predates these attrs is safe:
  // Parthenon's ReadFromHDF5AllParamsOfType keeps the value recomputed here.
  {
    const auto Uprov = PhysUnits::BuildPhysicalUnits(pin);
    using RM = Params::Mutability;
    pkg->AddParam<Real>("phys_units/rho_unit_cgs", Uprov.rho_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/v_unit_cgs", Uprov.v_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/length_unit_cgs", Uprov.length_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/time_unit_cgs", Uprov.time_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/mass_unit_cgs", Uprov.mass_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/energy_density_unit_cgs", Uprov.energy_density_unit,
                        RM::Restart);
    pkg->AddParam<Real>("phys_units/magnetic_unit_G", Uprov.magnetic_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/temperature_unit_K", Uprov.temperature_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/diffusivity_unit", Uprov.diffusivity_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/opacity_unit", Uprov.opacity_unit, RM::Restart);
    pkg->AddParam<Real>("phys_units/mu_thermal", Uprov.mu_thermal, RM::Restart);
  }

  // MHD only (glmmhd): the pure-hydro `euler` path is not part of this tree, so the default is
  // glmmhd and a deck asking for euler fails loudly below rather than silently running a
  // different fluid.

  const auto fluid_str = pin->GetOrAddString("hydro", "fluid", "glmmhd");
  auto fluid = Fluid::undefined;
  bool calc_c_h = false; // calculate hyperbolic divergence cleaning speed
  int nhydro = -1;

  if (fluid_str == "glmmhd") {
    fluid = Fluid::glmmhd;
    nhydro = GetNVars<Fluid::glmmhd>();
    // TODO(pgrete) reeval default value based on testing
    auto glmmhd_source_str =
        pin->GetOrAddString("hydro", "glmmhd_source", "dedner_plain");
    if (glmmhd_source_str == "dedner_plain") {
      pkg->AddParam<GLMMHD::SourceFun_t>("glmmhd_source", GLMMHD::DednerSource<false>);
    } else if (glmmhd_source_str == "dedner_extended") {
      pkg->AddParam<GLMMHD::SourceFun_t>("glmmhd_source", GLMMHD::DednerSource<true>);
    } else {
      PARTHENON_FAIL("AthenaPK hydro: Unknown glmmhd_source");
    }
    // ratio of diffusive to advective timescale of the divergence cleaning
    auto glmmhd_alpha = pin->GetOrAddReal("hydro", "glmmhd_alpha", 0.1);
    pkg->AddParam<Real>("glmmhd_alpha", glmmhd_alpha);
    calc_c_h = true;
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown fluid method. This build supports "
                   "hydro/fluid = glmmhd only.");
  }
  pkg->AddParam<>("fluid", fluid);
  pkg->AddParam<>("nhydro", nhydro);

  // B1-remainder: opt-in clamp on the domain-boundary face fluxes (see ClampBoundaryFluxes).
  // Default false => the task is never added to the driver and the OFF state is bit-identical.
  const bool bflux_clamp = pin->GetOrAddBoolean("hydro", "boundary_flux_clamp", false);
  pkg->AddParam<>("boundary_flux_clamp", bflux_clamp);
  if (bflux_clamp && parthenon::Globals::my_rank == 0) {
    std::cout << "## Boundary flux clamp ON (hydro/boundary_flux_clamp): inward mass/momentum/"
                 "energy/scalar fluxes on the physical domain faces are zeroed after the "
                 "Riemann solve. Magnetic fluxes are NOT clamped (that would inject div(B); "
                 "see hydro.hpp). This is the B1 remainder the `diode` BC cannot reach."
              << std::endl;
  }
  pkg->AddParam<>("calc_c_h", calc_c_h);

  // Divergence control: GLM/Dedner only. Constrained Transport (divergence_control = ct) was
  // removed 2026-09-24 (never used in production; 0 decks set the key).
  PARTHENON_REQUIRE_THROWS(pin->GetOrAddString("hydro", "divergence_control", "glm") == "glm",
                           "hydro/divergence_control must be 'glm' (CT was removed 2026-09-24).");
  // D1 instrument (see PreStepMeshUserWorkInLoop). Default off => no output, no cost.
  pkg->AddParam<>("d1_meminfo", pin->GetOrAddBoolean("hydro", "d1_meminfo", false));
  pkg->AddParam<>("d1_last_nblocks", -1, Params::Mutability::Mutable);
  // Sampling interval in cycles for the D1 instrument. 0 (default) = original behaviour:
  // print only when the block count changes. >0 also prints every N cycles, which is what
  // resolving the memory-growth curve between sparse regrids requires.
  pkg->AddParam<>("d1_meminfo_every", pin->GetOrAddInteger("hydro", "d1_meminfo_every", 0));
  // Following params should (currently) be present independent of solver because
  // they're all used in the main loop.
  // TODO(pgrete) think about which approach (selective versus always is preferable)
  pkg->AddParam<Real>(
      "c_h", 0.0, Params::Mutability::Mutable); // hyperbolic divergence cleaning speed
  // global minimum dx (used to calc c_h)
  pkg->AddParam<Real>("mindx", std::numeric_limits<Real>::max(),
                      Params::Mutability::Mutable);
  // hyperbolic timestep constraint
  pkg->AddParam<Real>("dt_hyp", std::numeric_limits<Real>::max(),
                      Params::Mutability::Mutable);

  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_nghost = 3; // largest number for the choices below
  auto recon = Reconstruction::undefined;
  if (recon_str == "dc") {
    recon = Reconstruction::dc;
    recon_need_nghost = 1;
  } else if (recon_str == "plm") {
    recon = Reconstruction::plm;
    recon_need_nghost = 2;
  } else if (recon_str == "ppm") {
    recon = Reconstruction::ppm;
    recon_need_nghost = 3;
  } else if (recon_str == "limo3") {
    recon = Reconstruction::limo3;
    recon_need_nghost = 2;
  } else if (recon_str == "weno3") {
    recon = Reconstruction::weno3;
    recon_need_nghost = 2;
  } else if (recon_str == "wenoz") {
    recon = Reconstruction::wenoz;
    recon_need_nghost = 3;
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown reconstruction method.");
  }
  // Adding recon independently of flux function pointer as it's used in 3D flux func.
  pkg->AddParam<>("reconstruction", recon);

  // Use hyperbolic timestep constraint by default
  bool calc_dt_hyp = true;
  const auto riemann_str = pin->GetString("hydro", "riemann");
  auto riemann = RiemannSolver::undefined;
  if (riemann_str == "llf") {
    riemann = RiemannSolver::llf;
    PARTHENON_REQUIRE(recon == Reconstruction::dc,
                      "LLF Riemann solver only implemented with DC reconstruction.")
  } else if (riemann_str == "hlle") {
    riemann = RiemannSolver::hlle;
  } else if (riemann_str == "hlld") {
    riemann = RiemannSolver::hlld;
  } else if (riemann_str == "none") {
    riemann = RiemannSolver::none;
    // If hyperbolic fluxes are disabled, there's no restriction from those
    // on the timestep
    calc_dt_hyp = false;
    PARTHENON_REQUIRE(recon == Reconstruction::dc,
                      "Disabling hyperbolic fluxes via 'none' Riemann solver only "
                      "supported in comination with DC reconstruction.")
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown riemann solver.");
  }
  pkg->AddParam<>("riemann", riemann);

  // Set calculation of hyperbolic timestep. Input file option takes precedence.
  if (pin->DoesParameterExist("hydro", "calc_dt_hyp")) {
    calc_dt_hyp = pin->GetBoolean("hydro", "calc_dt_hyp");
  }
  pkg->AddParam<>("calc_dt_hyp", calc_dt_hyp);
  // SG-F6 (2026-10-04): NaN guard. A non-finite primitive used to vanish from the timestep
  // reduction (fmin ignores NaN) and spread through gravity while the run went on and exited 0.
  // The hyperbolic-dt kernel now flags non-finite cells with a negative sentinel; finite cells
  // are untouched (bit-identical). On a flag the run aborts with the cell's location and values.
  pkg->AddParam<>("nan_check", pin->GetOrAddBoolean("hydro", "nan_check", true));

  // Maximum dt. Useful for debugging.
  const auto max_dt = pin->GetOrAddReal("hydro", "max_dt", -1.0);
  pkg->AddParam<>("max_dt", max_dt);

  // Map contaning all compiled in flux functions
  std::map<std::tuple<Fluid, Reconstruction, RiemannSolver>, FluxFun_t *>
      flux_functions{};
  // TODO(?) The following line could potentially be set by configure-time options
  // so that the resulting binary can only contain a subset of included flux functions
  // to reduce size.
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::none>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::plm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::ppm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::weno3, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::limo3, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::wenoz, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::plm, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::ppm, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::weno3, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::limo3, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::wenoz, RiemannSolver::hlld>(flux_functions);
  // Add first order recon with LLF fluxes (implemented for testing as tight loop)
  flux_functions[std::make_tuple(Fluid::glmmhd, Reconstruction::dc, RiemannSolver::llf)] =
      Hydro::CalculateFluxesTight<Fluid::glmmhd>;

  // flux used in all stages expect the first. First stage is set below based on integr.
  FluxFun_t *flux_other_stage = nullptr;
  flux_other_stage = flux_functions.at(std::make_tuple(fluid, recon, riemann));

  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IDN>, "mass"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM1>, "1-mom"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM2>, "2-mom"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM3>, "3-mom"));
  hst_vars.emplace_back(
      HistoryOutputVar(parthenon::UserHistoryOperation::sum, HydroHst<Hst::ekin>, "KE"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IEN>, "tot-E"));
  if (fluid == Fluid::glmmhd) {
    hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                           HydroHst<Hst::emag>, "ME"));
    hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                           HydroHst<Hst::divb>, "relDivB"));
    // WS-5c: peak relative div B error (max over cells), for the div B fidelity audit.
    hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::max,
                                           HydroHstMaxDivB, "maxRelDivB"));
  }
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars, true);

  // not using GetOrAdd here until there's a reasonable default
  const auto nghost = pin->GetInteger("parthenon/mesh", "nghost");
  if (nghost < recon_need_nghost) {
    PARTHENON_FAIL("AthenaPK hydro: Need more ghost zones for chosen reconstruction.");
  }

  const auto integrator_str = pin->GetString("parthenon/time", "integrator");
  auto integrator = Integrator::undefined;
  FluxFun_t *flux_first_stage = flux_other_stage;

  if (integrator_str == "rk1") {
    integrator = Integrator::rk1;
  } else if (integrator_str == "rk2") {
    integrator = Integrator::rk2;
  } else if (integrator_str == "rk3") {
    integrator = Integrator::rk3;
  } else if (integrator_str == "vl2") {
    integrator = Integrator::vl2;
    // override first stage (predictor) to first order
    // SG-F4 (2026-10-03): `hydro/vl2_predictor_recon = same` (DEFAULT since 2026-10-04) keeps the
    // main reconstruction in the predictor (VL2 then is the plain midpoint rule). The donor-cell
    // predictor (`dc`, the historical behaviour, kept for bit-reproducing older runs) leaves an
    // O(dt) error at converging stagnation points -- measured as a FIRST-order-in-dt error of the
    // collapse's central density, 35 % of the n_J = 8 collapse-time error
    // (validation_tests/self_gravity, L7).
    const auto vl2_pred = pin->GetOrAddString("hydro", "vl2_predictor_recon", "same");
    PARTHENON_REQUIRE(vl2_pred == "dc" || vl2_pred == "same",
                      "hydro/vl2_predictor_recon must be 'same' (default) or 'dc'");
    if (vl2_pred == "dc") {
      flux_first_stage =
          flux_functions.at(std::make_tuple(fluid, Reconstruction::dc, riemann));
    }
  }
  pkg->AddParam<>("integrator", integrator);
  pkg->AddParam<FluxFun_t *>("flux_first_stage", flux_first_stage);
  pkg->AddParam<FluxFun_t *>("flux_other_stage", flux_other_stage);

  // Fused domain-boundary kernel (bvals/boundary_conditions_apk.hpp): the GenericBC zero-gradient
  // copy and the diode clamp in ONE pack and ONE launch per boundary face (bit-identical to the
  // per-variable path).
  pkg->AddParam<>("fused_bc", pin->GetOrAddBoolean("hydro", "fused_bc", false));
  // Mesh-level version (bvals/fused_bc_mesh.cpp): six launches per exchange for ALL blocks instead
  // of one per block face (bit-identical; -14 % s/cycle measured on one GH200 node). Requires every
  // domain face to be `diode`; declines otherwise.
  pkg->AddParam<>("fused_bc_mesh", pin->GetOrAddBoolean("hydro", "fused_bc_mesh", false));
  auto first_order_flux_correct =
      pin->GetOrAddBoolean("hydro", "first_order_flux_correct", false);
  pkg->AddParam<>("first_order_flux_correct", first_order_flux_correct);
  // hydro/double_flux (validation_tests/eos_tabulated E7 / E-F10, 2026-10-05; default false): pressure-exact
  // contacts for the tabulated EOS via the double-flux energy update (src/hydro/double_flux.hpp). Opt-in;
  // refused outside its validated scope.
  {
    const bool dflux = pin->GetOrAddBoolean("hydro", "double_flux", false);
    pkg->AddParam<>("double_flux", dflux);
    pkg->AddParam<>("double_flux_kappa", pin->GetOrAddReal("hydro", "double_flux_kappa", 0.2));
    // EOS-consistent fill on refined meshes (double_flux.hpp, E7w); only with double flux on a refined mesh.
    // hydro/double_flux_fill_sensor: restrict the fill to contact-like cells (same test as the double-flux sensor, on
    // the transported pressure); elsewhere Parthenon's conservative interpolation is kept (E7z production test).
    pkg->AddParam<>("dflux_fill_sensor", pin->GetOrAddBoolean("hydro", "double_flux_fill_sensor", true));
    pkg->AddParam<>("dflux_ghost_fill",
                    dflux && pin->GetOrAddString("parthenon/mesh", "refinement", "none") != "none" &&
                        pin->GetOrAddBoolean("hydro", "double_flux_ghost_fill", true));

    pkg->AddParam<>("dflux_stage1_dc",
                    dflux && integrator == Integrator::vl2 &&
                        pin->GetOrAddString("hydro", "vl2_predictor_recon", "same") == "dc");
    if (dflux) {
      PARTHENON_REQUIRE(pin->GetOrAddString("hydro", "eos", "adiabatic") == "hydrogen",
                        "hydro/double_flux needs eos = hydrogen");
      PARTHENON_REQUIRE(riemann == RiemannSolver::hlld, "hydro/double_flux needs riemann = hlld");
      PARTHENON_REQUIRE(recon == Reconstruction::plm || recon == Reconstruction::dc,
                        "hydro/double_flux needs reconstruction = plm or dc");
      PARTHENON_REQUIRE(integrator == Integrator::vl2, "hydro/double_flux needs integrator = vl2");
      // mesh refinement allowed since 2026-10-06: coarse cells along fine/coarse faces are standard cells
      // (double_flux.cpp); validated by validation_tests/eos_tabulated E7r.
      PARTHENON_REQUIRE(!first_order_flux_correct,
                        "hydro/double_flux: first_order_flux_correct not supported (it replaces the fluxes the "
                        "double-flux increments were built for)");
      pkg->AddField("dflux_frozen",
                    Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>({2})));
      // Face sums X, Y, Z of the own-EOS energy-flux increments live in the FLUX slots of dflux_xyz (cell values
      // unused), so Parthenon's flux correction restricts them at fine/coarse faces together with the hydro fluxes.
      // OperatorSplit keeps it out of the low-storage update, Update::FluxDivergence and the lean registers (the
      // flag-pack trap: never let a new WithFluxes field reach a {WithFluxes, Cell} pack). No FillGhost.
      const parthenon::MetadataFlag OperatorSplit =
          parthenon::Metadata::FlagNameExists("OperatorSplit")
              ? parthenon::Metadata::GetUserFlag("OperatorSplit")
              : parthenon::Metadata::AddUserFlag("OperatorSplit");
      pkg->AddField("dflux_xyz", Metadata({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                                           OperatorSplit},
                                          std::vector<int>({3})));
      if (pkg->Param<bool>("dflux_ghost_fill")) {
        // transported pressure for the EOS-consistent fill on refined meshes (double_flux.hpp): exchanged and
        // regridded with the SAME refinement operators as cons; OperatorSplit keeps it out of the hydro update packs
        // and the lean registers
        Metadata mp({Metadata::Cell, Metadata::Independent, Metadata::FillGhost, OperatorSplit});
        if (pin->GetOrAddBoolean("hydro", "cons_prolong_pc", false)) {
          mp.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                                   parthenon::refinement_ops::RestrictAverage>();
        } else {
          mp.RegisterRefinementOps<refinement_ops::ProlongateCellMinModMultiD,
                                   parthenon::refinement_ops::RestrictAverage>();
        }
        pkg->AddField("dflux_p", mp);
      }
    }
  }
  // Per-call FOFC tallies on stdout (default off; diagnostic only, no effect on the update).
  pkg->AddParam<>("fofc_verbose", pin->GetOrAddBoolean("hydro", "fofc_verbose", false));
  if (first_order_flux_correct) {
    if (fluid == Fluid::glmmhd) {
      pkg->AddParam<FirstOrderFluxCorrectFun_t *>("first_order_flux_correct_fun",
                                                  FirstOrderFluxCorrect<Fluid::glmmhd>);
    }
  }

  if (pin->DoesBlockExist("units")) {
    Units units(pin, pkg);
  }

  auto eos_str = pin->GetString("hydro", "eos");
  if (eos_str == "adiabatic" || eos_str == "hydrogen") {
    Real gamma = pin->GetReal("hydro", "gamma");
    pkg->AddParam<>("AdiabaticIndex", gamma);

    // Optional tabulated protostellar EOS (second-core physics). eos=hydrogen loads the
    // offline full-Saha table (src/eos/gen_eos_table.py: H2 dissociation + H ionization +
    // He/He+ ionization + H2 rot/vib + inert He, in FHC code units) and interpolates it for
    // pressure/sound-speed/internal-energy. Requires riemann=hlld.
    const bool use_h2diss = (eos_str == "hydrogen");
    EOSTable::EosTable eos_tab;
    if (use_h2diss) {
      // The table file must be named in the deck (hydro/eos_table_file, e.g.
      // src/eos/eos_table_hires_v5_cold.bin); a missing key fails below unless
      // hydro/eos_table_allow_default = true, which falls back to the file name below in the
      // run directory. Tables are in cgs (v2 format) and converted by Load() with the run's units.
      const bool tabfile_given = pin->DoesParameterExist("hydro", "eos_table_file");
      const auto tabfile =
          pin->GetOrAddString("hydro", "eos_table_file", "eos_table_hires_v5_cold.bin");
      if (!tabfile_given) {
        const bool allow_default =
            pin->GetOrAddBoolean("hydro", "eos_table_allow_default", false);
        PARTHENON_REQUIRE(allow_default,
                          "<hydro> eos_table_file is not set. Name the table explicitly, e.g. "
                          "hydro/eos_table_file = <repo>/src/eos/eos_table_hires_v5_cold.bin, or set "
                          "hydro/eos_table_allow_default = true to read eos_table_hires_v5_cold.bin "
                          "from the run directory.");
      }
      // AUDIT N14 (2026-08-05), FIXED. A v2 table stores axes and arrays in cgs and Load()
      // converts them with THIS run's units, so the file is independent of any particular IC.
      // A legacy v1 table was built in the generator's own rounded code units and is consumed
      // verbatim -- bit-identical to the historical behaviour, and carrying the historical
      // bias. Warn in that case, quantifying it, so an old table cannot pass for a fixed one.
      {
        const auto Ueos = PhysUnits::BuildPhysicalUnits(pin);
        {
          const auto interp = pin->GetOrAddString("hydro", "eos_interp", "p_over_rho"); // E-F1 (eos_tabulated validation)
          PARTHENON_REQUIRE(interp == "p" || interp == "p_over_rho", "hydro/eos_interp must be p or p_over_rho");
          eos_tab.p_over_rho_ = (interp == "p_over_rho");
          // E-F5/E-F6 (2026-10-05): ideal-gas continuation outside the table; false = legacy edge clamp.
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
          eos_tab.extrapolate_ = pin->GetOrAddBoolean("hydro", "eos_extrapolate", false);
          PARTHENON_REQUIRE(!eos_tab.extrapolate_, "hydro/eos_extrapolate = true needs a build WITHOUT "
                            "ATHENAPK_EOS_LEGACY_LOOKUPS (the legacy build compiles the original lookups verbatim)");
#else
          eos_tab.extrapolate_ = pin->GetOrAddBoolean("hydro", "eos_extrapolate", true);
#endif
        }
        eos_tab.Load(tabfile, Ueos.rho_unit, Ueos.v_unit);
        eos_tab.n_bisect_ = pin->GetOrAddInteger("hydro", "eos_bisections", 20); // audit P-13
        PARTHENON_REQUIRE(eos_tab.n_bisect_ >= 10 && eos_tab.n_bisect_ <= 60,
                          "hydro/eos_bisections must lie in [10, 60].");
        if (parthenon::Globals::my_rank == 0 && eos_tab.used_legacy_code_units) {
          constexpr Real kGenEosRho0 = 5.467e-19; // gen_eos_table.py rho0, format v1
          constexpr Real kGenEosV0 = 1.9e4;       // gen_eos_table.py v0,   format v1
          const Real drho = kGenEosRho0 / Ueos.rho_unit - 1.0;
          const Real dv = kGenEosV0 / Ueos.v_unit - 1.0;
          if (std::abs(drho) > 1.0e-6 || std::abs(dv) > 1.0e-6) {
            std::cout << "### WARNING Hydro (audit N14): " << tabfile
                      << " is a LEGACY (v1) EOS table: its axes and arrays are in the "
                         "generator's own code units (rho0 = "
                      << kGenEosRho0 << ", v0 = " << kGenEosV0
                      << ") and are used verbatim. This run's PhysUnits give rho_unit = "
                      << Ueos.rho_unit << ", v_unit = " << Ueos.v_unit << " (rel diff " << drho
                      << ", " << dv
                      << "), so the table is consulted at slightly wrong (rho, esp). "
                         "Regenerate with gen_eos_table.py (which now emits cgs) to remove "
                         "the bias; the v2 file is correct for any normalization."
                      << std::endl;
          }
        }
      }
      const auto riem = pin->GetOrAddString("hydro", "riemann", "hlld");
      PARTHENON_REQUIRE(riem == "hlld",
                        "Tabulated protostellar EOS (eos=hydrogen) requires riemann=hlld "
                        "(HLLE's Roe average bakes in a constant gamma).");
      // The table EOS owns the gas thermodynamics. collapse_be's barotropic cooling overwrites the
      // thermal energy with an ideal-gamma law that is inconsistent with the table (a radiative
      // energy equation, not part of this tree, would be needed instead), so the table is refused for collapse_be and available for every other problem generator.
      PARTHENON_REQUIRE(!(pin->DoesParameterExist("job", "problem_id") &&
                          pin->GetString("job", "problem_id") == "collapse_be"),
                        "eos=hydrogen is not available for collapse_be in this build (its barotropic "
                        "cooling would overwrite the table's thermal energy); use eos=adiabatic.");
      if (parthenon::Globals::my_rank == 0) {
        std::cout << "## EOS: tabulated protostellar EOS ACTIVE (second-core). table="
                  << tabfile << " grid " << eos_tab.nr_ << "x" << eos_tab.ne_
                  << " (rho,esp), " << eos_tab.nr_ << "x" << eos_tab.nT_ << " (rho,T)"
                  << std::endl;
      }
    }

    if (pin->DoesParameterExist("hydro", "He_mass_fraction") &&
        pkg->AllParams().hasKey("units")) {
      auto units = pkg->Param<Units>("units");
      const auto He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
      const auto mu = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
      const auto mu_e = 1 / (He_mass_fraction * 2. / 4. + (1 - He_mass_fraction));
      pkg->AddParam<>("mu", mu);
      pkg->AddParam<>("mu_e", mu_e);
      pkg->AddParam<>("He_mass_fraction", He_mass_fraction);
      pkg->AddParam<>("mbar", mu * units.atomic_mass_unit());
      // Following convention in the astro community, we're using mh as unit for the mean
      // molecular weight
      pkg->AddParam<>("mbar_over_kb", mu * units.mh() / units.k_boltzmann());
    }

    // By default disable floors by setting a negative value
    Real dfloor = pin->GetOrAddReal("hydro", "dfloor", -1.0);
    // WP-13: expose the density floor as a package Param. SelfGravity::FillPoissonRHS needs
    // it to reproduce prim's floored density exactly while reading `cons` (it must read cons,
    // not prim, because prim is another package's FillDerived output and the ordering between
    // packages is not guaranteed -- on restart prim was still zero when the Poisson RHS was
    // assembled). Registered unconditionally so it does not depend on any diagnostic gate.
    pkg->AddParam<Real>("grav_rho_floor", dfloor);
    Real pfloor = pin->GetOrAddReal("hydro", "pfloor", -1.0);
    Real Tfloor = pin->GetOrAddReal("hydro", "Tfloor", -1.0);
    Real efloor = Tfloor;
    if (efloor > 0.0) {
      // validation_tests/eos_tabulated E-F5 (2026-10-05): the tabulated EOS takes the floor energy from the table
      // (below) and needs neither a units block nor hydro/He_mass_fraction -- the requirement applies to the
      // ideal-gas branch only. Tfloor is in CODE temperature units (x temperature_unit = K; 10.015 K for FHC).
      if (!use_h2diss && !pkg->AllParams().hasKey("mbar_over_kb")) {
        PARTHENON_FAIL("Temperature floor requires units and gas composition. "
                       "Either set a 'units' block and the 'hydro/He_mass_fraction' in "
                       "input file or use a pressure floor "
                       "(defined code units) instead.");
      }
      // AUDIT F-22 (2026-08-28): this is the IDEAL-GAS conversion. Under eos = hydrogen the
      // table's Gamma_1 is ~5/3 below theta_rot = 85 K, not the deck's 1.4, so the delivered
      // temperature floor would differ from the requested one by roughly
      // (1.4-1)/(5/3-1) = 0.6. Invert the table instead when it is active, so hydro/Tfloor
      // means the temperature it says. Inert today (0 of 4 live decks set hydro/Tfloor, so
      // efloor stays -1 and the e_floor_ > 0 guard never runs) -- fixed because the key looks
      // usable and was not. Ideal-gas path is unchanged and bit-identical.
      if (use_h2diss) {
        // esp [code] at the requested T; e_floor_ is a SPECIFIC internal energy.
        const auto U_fl = PhysUnits::BuildPhysicalUnits(pin);
        // HOST call -- see the note on EosTable::EintFromRhoTkHost.
        efloor = eos_tab.EintFromRhoTkHost(1.0, Tfloor * U_fl.temperature_unit) / 1.0;
      } else {
        efloor = Tfloor / pkg->Param<Real>("mbar_over_kb") / (gamma - 1.0);
      }
    }

    // By default disable ceilings by setting to infinity
    Real vceil =
        pin->GetOrAddReal("hydro", "vceil", std::numeric_limits<Real>::infinity());
    Real Tceil =
        pin->GetOrAddReal("hydro", "Tceil", std::numeric_limits<Real>::infinity());
    Real eceil = Tceil;
    if (eceil < std::numeric_limits<Real>::infinity()) {
      if (!pkg->AllParams().hasKey("mbar_over_kb")) {
        PARTHENON_FAIL("Temperature ceiling requires units and gas composition. "
                       "Either set a 'units' block and the 'hydro/He_mass_fraction' in "
                       "input file or use a pressure floor "
                       "(defined code units) instead.");
      }
      auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");
      eceil = Tceil / mbar_over_kb / (gamma - 1.0);
    }

    // No parabolic (diffusion) terms in this tree: the parabolic timestep is infinite and the
    // diffusion integrator is off. Both are still registered because the timestep report and the
    // driver read them.
    PARTHENON_REQUIRE_THROWS(
        pin->GetOrAddString("diffusion", "integrator", "none") == "none",
        "<diffusion> is not part of this build (no conduction, viscosity or resistivity).");
    pkg->AddParam<Real>("dt_diff", std::numeric_limits<Real>::max(),
                        Params::Mutability::Mutable);
    pkg->AddParam<>("diffint", DiffInt::none);

    if (fluid == Fluid::glmmhd) {
      const bool floor_consistent =
          pin->GetOrAddBoolean("hydro", "eos_floor_consistent", false);
      if (floor_consistent && use_h2diss && parthenon::Globals::my_rank == 0)
        std::cout << "## EOS pressure floor CONSISTENT with the table (hydro/eos_floor_consistent): "
                     "a floored cell stores the pressure of the energy it actually holds." << std::endl;
      AdiabaticGLMMHDEOS eos(pfloor, dfloor, efloor, vceil, eceil, gamma, use_h2diss, eos_tab,
                             floor_consistent);
      pkg->AddParam<>("eos", eos);
      // Per-cycle [DTREPORT] line naming the binding dt constraint (see PreStepMeshUserWorkInLoop).
      pkg->AddParam<>("dt_report", pin->GetOrAddBoolean("hydro", "dt_report", false));
      // Per-cycle report of out-of-range EOS-table lookups (see PreStepMeshUserWorkInLoop).
      // Default ON: it only prints, the tally itself is numerics-neutral, and a silent clamp
      // is a failure worth reporting.
      pkg->AddParam<>("eos_clamp_report",
                      pin->GetOrAddBoolean("hydro", "eos_clamp_report", true));
      pkg->FillDerivedMesh = pkg->Param<bool>("dflux_ghost_fill") ? ConsToPrimDoubleFlux
                                                                   : ConsToPrim<AdiabaticGLMMHDEOS>;
      pkg->EstimateTimestepMesh = EstimateTimestep<Fluid::glmmhd>;
    }
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown EOS");
  }

  // Tabular optically-thin cooling is not part of this tree; collapse_be cools barotropically
  // (ApplyBarotropicCooling). A deck that still sets <cooling> enable_cooling is caught here
  // rather than silently ignored.
  {
    const auto enable_cooling_str =
        pin->GetOrAddString("cooling", "enable_cooling", "none");
    PARTHENON_REQUIRE(
        enable_cooling_str == "none",
        "cooling/enable_cooling is set, but tabular cooling is not part of this build.");
  }

  auto scratch_level = pin->GetOrAddInteger("hydro", "scratch_level", 0);
  pkg->AddParam("scratch_level", scratch_level);

  auto nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
  pkg->AddParam("nscalars", nscalars);
  // Advected components after `nhydro`: the passive scalars.
  const int nadvected = nscalars;
  // See hydro_driver.cpp (start of stage 1): extra halo-exchange + FillDerived rounds so the ghost
  // zones a step starts from depend on the interior only (restart reproducibility). 0 = off.
  pkg->AddParam("ghost_sync_rounds", pin->GetOrAddInteger("hydro", "ghost_sync_rounds", 0));
  // See RegisterVarNames in hydro_driver.cpp. Default false = full register copies.
  pkg->AddParam("lean_registers", pin->GetOrAddBoolean("hydro", "lean_registers", false));
  pkg->AddParam("nadvected", nadvected);

  std::vector<std::string> cons_labels(nhydro);
  cons_labels[IDN] = "density";
  cons_labels[IM1] = "momentum_density_1";
  cons_labels[IM2] = "momentum_density_2";
  cons_labels[IM3] = "momentum_density_3";
  cons_labels[IEN] = "total_energy_density";
  if (fluid == Fluid::glmmhd) {
    cons_labels[IB1] = "magnetic_field_1";
    cons_labels[IB2] = "magnetic_field_2";
    cons_labels[IB3] = "magnetic_field_3";
    cons_labels[IPS] = "magnetic_psi";
  }

  // TODO(pgrete) check if this could be "one-copy" for two stage SSP integrators
  std::vector<std::string> prim_labels(nhydro);
  prim_labels[IDN] = "density";
  prim_labels[IV1] = "velocity_1";
  prim_labels[IV2] = "velocity_2";
  prim_labels[IV3] = "velocity_3";
  prim_labels[IPR] = "pressure";
  if (fluid == Fluid::glmmhd) {
    prim_labels[IB1] = "magnetic_field_1";
    prim_labels[IB2] = "magnetic_field_2";
    prim_labels[IB3] = "magnetic_field_3";
    prim_labels[IPS] = "magnetic_psi";
  }
  for (auto i = 0; i < nscalars; i++) {
    cons_labels.emplace_back("scalar_density_" + std::to_string(i));
    prim_labels.emplace_back("scalar_" + std::to_string(i));
  }

  Metadata m(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes},
      std::vector<int>({nhydro + nadvected}), cons_labels);
  // hydro/cons_prolong_pc (VALIDATION ONLY, default false = the production minmod operator): piecewise-constant
  // prolongation of cons, to isolate the fine/coarse source of the residual double-flux contact error
  // (validation_tests/eos_tabulated E7v, 2026-10-06).
  if (pin->GetOrAddBoolean("hydro", "cons_prolong_pc", false)) {
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                            parthenon::refinement_ops::RestrictAverage>();
  } else {
    m.RegisterRefinementOps<refinement_ops::ProlongateCellMinModMultiD,
                            parthenon::refinement_ops::RestrictAverage>();
  }
  pkg->AddField("cons", m);

  m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({nhydro + nadvected}),
               prim_labels);
  pkg->AddField("prim", m);

  // WP-6 -- conservation-budget history diagnostics. See src/diagnostics/cons_diag.hpp.
  // These exist because tot-E excludes the gravitational potential energy (so energy
  // conservation is currently UNVERIFIABLE, not merely unverified), mass RISES under
  // outflow BCs, and 1-mom drifts ~10%. All read-only => bit-identical when off.
  const bool cons_diag = pin->GetOrAddBoolean("hydro", "cons_diag", false);
  pkg->AddParam<>("cons_diag", cons_diag);
  if (cons_diag) {
    using Diagnostics::ConsDiag;
    using Diagnostics::ConsDiagReduce;
    // Cached for the reduction kernel: the floor values and whether B exists. Read from
    // pin with the SAME defaults hydro.cpp uses above, so a deck that omits them gets the
    // same -1.0 "disabled" sentinel here and the floor columns stay identically zero.
    pkg->AddParam<Real>("cons_diag_dfloor", pin->GetOrAddReal("hydro", "dfloor", -1.0));
    pkg->AddParam<Real>("cons_diag_pfloor", pin->GetOrAddReal("hydro", "pfloor", -1.0));
    pkg->AddParam<bool>("cons_diag_mhd", fluid == Fluid::glmmhd);

    // Cumulative APPLIED domain-face mass flux (a MASS, not a rate; see cons_diag.hpp).
    // Rank-local: the driver task adds beta*dt * (face flux) each stage, and the history
    // callback below returns it for Parthenon's sum reduction to combine across ranks.
    //
    // Correct only at ONE MeshData partition per rank -- otherwise the driver task would run
    // several times per stage racing on the same Param, and the history callback would return
    // the same running total once per partition and sum it. pack_size < 1 is Parthenon's
    // "all blocks in one partition" setting and is what every deck in this tree uses; assert
    // it rather than let either failure happen silently.
    const int cd_pack_size = pin->GetOrAddInteger("parthenon/mesh", "pack_size", -1);
    PARTHENON_REQUIRE(
        cd_pack_size < 1,
        "hydro/cons_diag: cons-Mout-applied accumulates into one rank-local Param and is "
        "returned once per MeshData partition, so parthenon/mesh/pack_size >= 1 would both "
        "race the accumulator and multiply-count it. Unset pack_size, or set "
        "hydro/cons_diag=false.");
    // NOTE ON RESTARTS: this is not checkpointed, so it restarts from zero and the identity
    // it closes becomes mass(t) - mass(t_resume) = -cons-Mout-applied(t). Compare it against
    // the mass column measured from the SAME resume point, never against t=0 of the chain.
    pkg->AddParam<Real>("cons_Mout_applied", 0.0, true);
    pkg->AddParam<Real>("cons_Mout_applied_last", 0.0, true);

    auto hst_c = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
    auto add_c = [&hst_c](ConsDiag w, const std::string &label) {
      hst_c.emplace_back(parthenon::HistoryOutputVar(
          parthenon::UserHistoryOperation::sum,
          [w](MeshData<Real> *md) { return ConsDiagReduce(md, w); }, label));
    };
    // W = 1/2 int rho Phi dV needs the solver's potential field to exist.
    const bool have_phi = pin->GetOrAddBoolean("physics", "self_gravity", false);
    if (have_phi) add_c(ConsDiag::Wgrav, "cons-W");
    add_c(ConsDiag::Mout, "cons-Mout");
    // The factor-2 experiment: the same faces read from the SOLVER's own flux array.
    // Agreement with cons-Mout localises the discrepancy to the time integration;
    // a factor of 2 between them localises it to the surface integral. See cons_diag.hpp.
    add_c(ConsDiag::MoutSolver, "cons-Mout-solver");
    add_c(ConsDiag::MoutSolverInner, "cons-Mout-solver-in");
    add_c(ConsDiag::MoutSolverOuter, "cons-Mout-solver-out");
    add_c(ConsDiag::MoutInner, "cons-Mout-in");
    add_c(ConsDiag::MoutOuter, "cons-Mout-out");
    // THE column to close the mass budget with. Already time-integrated and stage-weighted
    // by the driver, so the check is the exact identity
    //     mass(t) - mass(0) = -cons-Mout-applied(t)
    // with no quadrature. cons-Mout stays as the physical surface-integral cross-check and
    // cons-Mout-solver as the timing control: solver reads the flux array at HISTORY time,
    // applied reads it at UPDATE time, so their difference isolates flux-array clobbering by
    // the operator-split sources (in production cons-Mout-solver is identically zero, which
    // is that clobbering, not a physics result).
    hst_c.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum,
        [](MeshData<Real> *md) {
          return md->GetParentPointer()->packages.Get("Hydro")->Param<Real>(
              "cons_Mout_applied");
        },
        "cons-Mout-applied"));
    hst_c.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum,
        [](MeshData<Real> *md) {
          return md->GetParentPointer()->packages.Get("Hydro")->Param<Real>(
              "cons_Mout_applied_last");
        },
        "cons-Mout-applied-last"));
    add_c(ConsDiag::PoutX, "cons-Poutx");
    add_c(ConsDiag::PoutY, "cons-Pouty");
    add_c(ConsDiag::PoutZ, "cons-Poutz");
    add_c(ConsDiag::nfloor, "cons-nfloor");
    add_c(ConsDiag::Mfloor, "cons-Mfloor");
    add_c(ConsDiag::npfloor, "cons-npfloor");
    // E-F5 (2026-10-05): cells held at the TEMPERATURE floor (hydro/Tfloor), by the same floor pressure ConsToPrim
    // applies; cons-npfloor only sees P <= pfloor. Registered only when the floor is on (history layout unchanged).
    if (fluid == Fluid::glmmhd && pkg->Param<AdiabaticGLMMHDEOS>("eos").GetInternalEFloor() > 0.0)
      add_c(ConsDiag::nTfloor, "cons-nTfloor");
    pkg->UpdateParam(parthenon::hist_param_key, hst_c);
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "## Conservation-budget diagnostics ON (hydro/cons_diag): hst "
                << (have_phi ? "cons-W (grav. PE), " : "")
                << "cons-Mout, cons-Poutx/y/z (TOTAL momentum flux incl. pressure and"
                   " Maxwell stress), cons-nfloor/Mfloor/npfloor. All *out columns are"
                   " OUTWARD-POSITIVE, so a NEGATIVE cons-Mout is inflow through an"
                   " outflow face. Budgets: d(mass)/dt = -cons-Mout, d(i-mom)/dt ="
                   " -cons-Pouti. Floor columns are a PROXY for floor activity, NOT the"
                   " mass injected -- see cons_diag.hpp."
                << std::endl;
      if (!have_phi) {
        std::cout << "## cons_diag: self_gravity is off, so cons-W is not registered."
                  << std::endl;
      }
    }
  }

  const auto refine_str = pin->GetOrAddString("refinement", "type", "unset");
  if (refine_str == "pressure_gradient") {
    pkg->CheckRefinementBlock = refinement::gradient::PressureGradient;
    const auto thr = pin->GetOrAddReal("refinement", "threshold_pressure_gradient", 0.0);
    PARTHENON_REQUIRE(thr > 0.,
                      "Make sure to set refinement/threshold_pressure_gradient >0.");
    pkg->AddParam<Real>("refinement/threshold_pressure_gradient", thr);
  } else if (refine_str == "xyvelocity_gradient") {
    pkg->CheckRefinementBlock = refinement::gradient::VelocityGradient;
    const auto thr =
        pin->GetOrAddReal("refinement", "threshold_xyvelocity_gradient", 0.0);
    PARTHENON_REQUIRE(thr > 0.,
                      "Make sure to set refinement/threshold_xyvelocity_gradient >0.");
    pkg->AddParam<Real>("refinement/threshold_xyvelocity_gradient", thr);
  } else if (refine_str == "maxdensity") {
    pkg->CheckRefinementBlock = refinement::other::MaxDensity;
    const auto deref_below =
        pin->GetOrAddReal("refinement", "maxdensity_deref_below", 0.0);
    const auto refine_above =
        pin->GetOrAddReal("refinement", "maxdensity_refine_above", 0.0);
    PARTHENON_REQUIRE(deref_below > 0.,
                      "Make sure to set refinement/maxdensity_deref_below > 0.");
    PARTHENON_REQUIRE(refine_above > 0.,
                      "Make sure to set refinement/maxdensity_refine_above > 0.");
    PARTHENON_REQUIRE(deref_below < refine_above,
                      "Make sure to set refinement/maxdensity_deref_below < "
                      "refinement/maxdensity_refine_above");
    pkg->AddParam<Real>("refinement/maxdensity_deref_below", deref_below);
    pkg->AddParam<Real>("refinement/maxdensity_refine_above", refine_above);
  } else if (refine_str == "jeans") {
    pkg->CheckRefinementBlock = refinement::jeans::Jeans;
    const auto njeans = pin->GetOrAddReal("refinement", "njeans", 0.0);
    PARTHENON_REQUIRE(njeans > 0.,
                      "Make sure to set refinement/njeans > 0 (typically 8-16).");
    pkg->AddParam<Real>("refinement/njeans", njeans);
  } else if (refine_str == "user") {
    pkg->CheckRefinementBlock = Hydro::ProblemCheckRefinementBlock;
  } else {
    // Audit F-1: an unrecognised refinement/type leaves CheckRefinementBlock null, which Parthenon
    // treats as derefine-everything every cycle under adaptive refinement. Fail loudly.
    const bool adaptive =
        pin->DoesParameterExist("parthenon/mesh", "refinement") &&
        pin->GetString("parthenon/mesh", "refinement") == "adaptive";
    PARTHENON_REQUIRE(
        !adaptive,
        "parthenon/mesh/refinement = adaptive but <refinement> type = '" + refine_str +
            "' is not a known criterion. Valid: pressure_gradient, xyvelocity_gradient, "
            "maxdensity, jeans, user. With an unknown type the refinement "
            "hook stays null, which Parthenon reads as 'derefine' on every block every cycle "
            "-- the mesh collapses to the root grid silently. Set a valid type, or set "
            "parthenon/mesh/refinement = none for a uniform grid.");
    if (refine_str != "unset" && parthenon::Globals::my_rank == 0) {
      std::cout << "### WARNING <refinement> type = '" << refine_str
                << "' is not a known criterion and parthenon/mesh/refinement is not adaptive,"
                   " so it is IGNORED. Valid: pressure_gradient, xyvelocity_gradient,"
                   " maxdensity, jeans, user." << std::endl;
    }
  }

  if (ProblemInitPackageData != nullptr) {
    ProblemInitPackageData(pin, pkg.get());
  }

  return pkg;
}

template <Fluid fluid>
Real EstimateHyperbolicTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &cfl_hyp = hydro_pkg->Param<Real>("cfl");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eos_ =
      hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();
  const bool nan_check_ = hydro_pkg->Param<bool>("nan_check");

  const auto ndim_ = prim_pack.GetNdim();
  Kokkos::parallel_reduce(
      "EstimateHyperbolicTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        // Need to reference variables here so that they are properly caught by
        // nvcc, which cannot determine captured variables only used within constexpr if.
        const auto &ndim = ndim_;
        const auto &eos = eos_;
        const bool &nan_check = nan_check_;
        if (nan_check) {
          bool finite = true;
          for (int v = 0; v <= IPS; ++v) {
            finite = finite && Kokkos::isfinite(prim(v, k, j, i));
          }
          if (!finite) {
            min_dt = -1.0; // sentinel, located and reported after the reduction
            return;
          }
        }

        Real w[(NHYDRO)];
        w[IDN] = prim(IDN, k, j, i);
        w[IV1] = prim(IV1, k, j, i);
        w[IV2] = prim(IV2, k, j, i);
        w[IV3] = prim(IV3, k, j, i);
        w[IPR] = prim(IPR, k, j, i);
        Real lambda_max_x, lambda_max_y, lambda_max_z;
        if constexpr (fluid == Fluid::glmmhd) {
          lambda_max_x = eos.FastMagnetosonicSpeed(
              w[IDN], w[IPR], prim(IB1, k, j, i), prim(IB2, k, j, i), prim(IB3, k, j, i));
          if (ndim > 1) {
            lambda_max_y =
                eos.FastMagnetosonicSpeed(w[IDN], w[IPR], prim(IB2, k, j, i),
                                          prim(IB3, k, j, i), prim(IB1, k, j, i));
          }
          if (ndim > 2) {
            lambda_max_z =
                eos.FastMagnetosonicSpeed(w[IDN], w[IPR], prim(IB3, k, j, i),
                                          prim(IB1, k, j, i), prim(IB2, k, j, i));
          }
        } else {
          PARTHENON_FAIL("Unknown fluid in EstimateTimestep");
        }
        min_dt = fmin(min_dt, coords.Dxc<1>(k, j, i) / (fabs(w[IV1]) + lambda_max_x));
        if (ndim > 1) {
          min_dt = fmin(min_dt, coords.Dxc<2>(k, j, i) / (fabs(w[IV2]) + lambda_max_y));
        }
        if (ndim > 2) {
          min_dt = fmin(min_dt, coords.Dxc<3>(k, j, i) / (fabs(w[IV3]) + lambda_max_z));
        }
      },
      Kokkos::Min<Real>(min_dt_hyperbolic));

  if (nan_check_ && min_dt_hyperbolic < 0.0) {
    // Rare path: find the first non-finite cell (lowest linear index) and report it.
    const int nb = prim_pack.GetDim(5);
    const int nk = kb.e + 1 - kb.s, nj = jb.e + 1 - jb.s, ni = ib.e + 1 - ib.s;
    long first = std::numeric_limits<long>::max();
    Kokkos::parallel_reduce(
        "NaNGuardLocate",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(DevExecSpace(), {0, kb.s, jb.s, ib.s},
                                               {nb, kb.e + 1, jb.e + 1, ib.e + 1}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, long &lmin) {
          const auto &prim = prim_pack(b);
          bool finite = true;
          for (int v = 0; v <= IPS; ++v) {
            finite = finite && Kokkos::isfinite(prim(v, k, j, i));
          }
          if (!finite) {
            const long idx = ((static_cast<long>(b) * nk + (k - kb.s)) * nj + (j - jb.s)) * ni +
                             (i - ib.s);
            lmin = idx < lmin ? idx : lmin;
          }
        },
        Kokkos::Min<long>(first));
    const int b = static_cast<int>(first / (static_cast<long>(nk) * nj * ni));
    long rem = first % (static_cast<long>(nk) * nj * ni);
    const int k = kb.s + static_cast<int>(rem / (static_cast<long>(nj) * ni));
    rem %= static_cast<long>(nj) * ni;
    const int j = jb.s + static_cast<int>(rem / ni);
    const int i = ib.s + static_cast<int>(rem % ni);
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    auto prim_host = md->GetBlockData(b)->Get("prim").data.GetHostMirrorAndCopy();
    std::stringstream msg;
    msg << "NaN guard (hydro/nan_check): non-finite primitive on rank " << parthenon::Globals::my_rank
        << ", block gid " << pmb->gid << " level " << pmb->loc.level() << ", cell (k,j,i) = ("
        << k << "," << j << "," << i << "), x = (" << pmb->coords.Xc<1>(k, j, i) << ", "
        << pmb->coords.Xc<2>(k, j, i) << ", " << pmb->coords.Xc<3>(k, j, i)
        << "); prim[rho,v1,v2,v3,p,B1,B2,B3,psi] =";
    for (int v = 0; v <= IPS; ++v) {
      msg << " " << prim_host(v, k, j, i);
    }
    PARTHENON_FAIL(msg.str());
  }

  // TODO(pgrete) THIS WORKAROUND IS NOT THREAD SAFE (though this will only become
  // relevant once parthenon uses host-multithreading in the driver).
  // We need to save the the hyperbolic part to recover it later as
  // the divergence cleaning speed is only limited in relation to the other
  // hyperbolic signal speeds and not by (potentially more restrictive) diffusive
  // processes.
  if constexpr (fluid == Fluid::glmmhd) {
    auto dt_hyp_pkg = hydro_pkg->Param<Real>("dt_hyp");
    if (cfl_hyp * min_dt_hyperbolic < dt_hyp_pkg) {
      hydro_pkg->UpdateParam("dt_hyp", cfl_hyp * min_dt_hyperbolic);
    }
  }
  return cfl_hyp * min_dt_hyperbolic;
}

// provide the routine that estimates a stable timestep for this package
template <Fluid fluid>
Real EstimateTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto min_dt = std::numeric_limits<Real>::max();
  auto dt_hyp = std::numeric_limits<Real>::max();

  const auto calc_dt_hyp = hydro_pkg->Param<bool>("calc_dt_hyp");
  if (calc_dt_hyp) {
    dt_hyp = EstimateHyperbolicTimestep<fluid>(md);
    min_dt = std::min(min_dt, dt_hyp);
  }

  // No parabolic terms in this tree (dt_diff stays infinite, see Initialize).

  if (ProblemEstimateTimestep != nullptr) {
    min_dt = std::min(min_dt, ProblemEstimateTimestep(md));
  }

  // maximum user dt
  const auto max_dt = hydro_pkg->Param<Real>("max_dt");
  if (max_dt > 0.0) {
    min_dt = std::min(min_dt, max_dt);
  }

  return min_dt;
}

// Calculate fluxes using a tightly nested 3D loop over the entire block.
// Currently only used for testing the LLF Riemann solver used in first-order flux corr.
template <Fluid fluid>
TaskStatus CalculateFluxesTight(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Same named-pack hardening as CalculateFluxes below; see the note there.
  const std::vector<std::string> cons_name{"cons"};
  auto cons_in = md->PackVariablesAndFluxes(cons_name, cons_name);
  auto pkg = pmb->packages.Get("Hydro");

  const auto &eos =
      pkg->Param<AdiabaticGLMMHDEOS>("eos");

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }
  // TODO(pgrete) fix scalar fluxes, too
  auto const &prim_in = md->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;
  auto riemann = Riemann<fluid, RiemannSolver::llf>();
  // loop bounds are chosen so that all active fluxes are calculated
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DC LLF fluxes", parthenon::DevExecSpace(), 0,
      cons_in.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_in(b);
        const auto &prim = prim_in(b);
        riemann.Solve(eos, k, j, i, IV1, prim, cons, c_h);
        if (ndim >= 2) {
          riemann.Solve(eos, k, j, i, IV2, prim, cons, c_h);
        }
        if (ndim >= 3) {
          riemann.Solve(eos, k, j, i, IV3, prim, cons, c_h);
        }
      });

  return TaskStatus::complete;
}

// Calculate fluxes using scratch pad memory, i.e., over cached pencils in i-dir.
template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int il, iu, jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  // TODO(pgrete): are these looop limits are likely too large for 2nd order
  if (pmb->block_size.nx(X2DIR) > 1) {
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s, ku = kb.e;
    else // 3D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s - 1, ku = kb.e + 1;
  }

  // Pack "cons" BY NAME: a flag pack is ordered by registration, so any Independent+WithFluxes
  // field registered before cons would take the leading slots
  // and receive the hydro fluxes (measured mass loss, DEV_LOG 2026-09-19).
  const std::vector<std::string> cons_name{"cons"};
  auto cons_in = md->PackVariablesAndFluxes(cons_name, cons_name);
  auto pkg = pmb->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nadvected");

  const auto &eos =
      pkg->Param<AdiabaticGLMMHDEOS>("eos");

  auto num_scratch_vars = nhydro + nscalars;

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }

  auto const &prim_in = md->PackVariables(std::vector<std::string>{"prim"});

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 2;

  auto riemann = Riemann<fluid, rsolver>();

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "x1 flux", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &prim = prim_in(b);
        auto &cons = cons_in(b);
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        // get reconstructed state on faces
        Reconstruct<recon, X1DIR>(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        riemann.Solve(member, k, j, ib.s, ib.e + 1, IV1, wl, wr, cons, eos, c_h);
        member.team_barrier();

        // Passive scalar fluxes
        for (auto n = nhydro; n < nhydro + nscalars; ++n) {
          parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            if (cons.flux(IV1, IDN, k, j, i) >= 0.0) {
              cons.flux(IV1, n, k, j, i) = cons.flux(IV1, IDN, k, j, i) * wl(n, i);
            } else {
              cons.flux(IV1, n, k, j, i) = cons.flux(IV1, IDN, k, j, i) * wr(n, i);
            }
          });
        }
      });

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 3;
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, kl = kb.s, ku = kb.e;
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
      kl = kb.s, ku = kb.e;
    else // 3D
      kl = kb.s - 1, ku = kb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x2 flux", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
          const auto &prim = prim_in(b);
          auto &cons = cons_in(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
            // reconstruct L/R states at j
            Reconstruct<recon, X2DIR>(member, k, j, il, iu, prim, wlb, wr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (j > jb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV2, wl, wr, cons, eos, c_h);
              member.team_barrier();

              // Passive scalar fluxes
              for (auto n = nhydro; n < nhydro + nscalars; ++n) {
                parthenon::par_for_inner(member, il, iu, [&](const int i) {
                  if (cons.flux(IV2, IDN, k, j, i) >= 0.0) {
                    cons.flux(IV2, n, k, j, i) = cons.flux(IV2, IDN, k, j, i) * wl(n, i);
                  } else {
                    cons.flux(IV2, n, k, j, i) = cons.flux(IV2, IDN, k, j, i) * wr(n, i);
                  }
                });
              }
              member.team_barrier();
            }

            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
          }
        });
  }
  //--------------------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->ndim >= 3) {
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, jl = jb.s - 1, ju = jb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x3 flux", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int j) {
          const auto &prim = prim_in(b);
          auto &cons = cons_in(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          for (int k = kb.s - 1; k <= kb.e + 1; ++k) {
            // reconstruct L/R states at j
            Reconstruct<recon, X3DIR>(member, k, j, il, iu, prim, wlb, wr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (k > kb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV3, wl, wr, cons, eos, c_h);
              member.team_barrier();

              // Passive scalar fluxes
              for (auto n = nhydro; n < nhydro + nscalars; ++n) {
                parthenon::par_for_inner(member, il, iu, [&](const int i) {
                  if (cons.flux(IV3, IDN, k, j, i) >= 0.0) {
                    cons.flux(IV3, n, k, j, i) = cons.flux(IV3, IDN, k, j, i) * wl(n, i);
                  } else {
                    cons.flux(IV3, n, k, j, i) = cons.flux(IV3, IDN, k, j, i) * wr(n, i);
                  }
                });
              }
              member.team_barrier();
            }
            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
          }
        });
  }

  return TaskStatus::complete;
}

// Apply first order flux correction, i.e., use first order reconstruction and a
// diffusive LLF Riemann solver if a negative density or energy density is expected.
// The current implementation is computationally not the most efficient one, but works
// for all standard integrators (rk1, rk2, rk3, and vl) and with and without AMR.
// In principle, without AMR one could directly use the results from the actual
// flux divergence call.
// However, with AMR (and coarse/fine flux correction) we need to correct the local
// fluxes first before calling coarse/fine flux correction.
// In addition, it may be enough to call first order flux correction once at the
// final stage (rather than at every stage as right row).
// However, this'd require an additional register to store the initial state and
// we should first evaluate where the tradeoff between extra computational costs
// (multiple calls) versus extra memory usage is.
template <Fluid fluid>
TaskStatus FirstOrderFluxCorrect(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                 const Real gam0_, const Real gam1_,
                                 const Real beta_dt_) {
  // Work around for CUDA <=11.6
  const Real gam0 = gam0_;
  const Real gam1 = gam1_;
  const Real beta_dt = beta_dt_;

  auto pmb = u0_data->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Pack "cons" by NAME (only the cell-centered hydro cons is corrected).
  const std::vector<std::string> cons_name{"cons"};
  auto u0_cons_pack = u0_data->PackVariablesAndFluxes(cons_name, cons_name);
  auto const &u0_prim_pack = u0_data->PackVariables(std::vector<std::string>{"prim"});
  auto u1_cons_pack = u1_data->PackVariablesAndFluxes(cons_name, cons_name);
  auto pkg = pmb->packages.Get("Hydro");

  const auto &eos =
      pkg->Param<AdiabaticGLMMHDEOS>("eos");

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }

  const int ndim = pmb->pmy_mesh->ndim;
  const bool fofc_verbose = pkg->Param<bool>("fofc_verbose");

  constexpr auto NVAR = GetNVars<fluid>();

  auto riemann = Riemann<fluid, RiemannSolver::llf>();

  std::int64_t num_corrected, num_need_floor;
  // Potentially need multiple attempts as flux correction corrects 6 (in 3D) fluxes
  // of a single cell at the same time. So the neighboring cells need to be rechecked with
  // the corrected fluxes as the corrected fluxes in one cell may result in the need to
  // correct all the fluxes of an originally "good" neighboring cell.
  size_t num_attempts = 0;
  do {
    num_corrected = 0;

    Kokkos::parallel_reduce(
        "FirstOrderFluxCorrect",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {u0_cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      std::int64_t &lnum_corrected, std::int64_t &lnum_need_floor) {
          const auto &coords = u0_cons_pack.GetCoords(b);
          const auto &u0_prim = u0_prim_pack(b);
          auto &u0_cons = u0_cons_pack(b);

          // In principle, the u_cons.fluxes could be updated in parallel by a
          // different thread resulting in a race conditon here. However, if the
          // fluxes of a cell have been updated (anywhere) then the entire kernel will
          // be called again anyway, and, at that point the already fixed
          // u0_cons.fluxes will automaticlly be used here.
          Real new_cons[NVAR];
          for (auto v = 0; v < NVAR; v++) {
            new_cons[v] =
                gam0 * u0_cons(v, k, j, i) + gam1 * u1_cons_pack(b, v, k, j, i) +
                beta_dt *
                    parthenon::Update::FluxDivHelper(v, k, j, i, ndim, coords, u0_cons);
          }

          // no need to include gamma - 1 as we only care for negative values
          auto new_p =
              new_cons[IEN] -
              0.5 * (SQR(new_cons[IM1]) + SQR(new_cons[IM2]) + SQR(new_cons[IM3])) /
                  new_cons[IDN];
          if constexpr (fluid == Fluid::glmmhd) {
            new_p -= 0.5 * (SQR(new_cons[IB1]) + SQR(new_cons[IB2]) + SQR(new_cons[IB3]));
          }
          // no correction required
          if (new_cons[IDN] > 0.0 && new_p > 0.0) {
            return;
          }
          // if already tried 3 times and only pressure is negative, then we'll rely
          // on the pressure floor during ConsToPrim conversion
          if (num_attempts > 2 && new_cons[IDN] > 0.0 && new_p < 0.0) {
            lnum_need_floor += 1;
            return;
          }
          // In principle, there could be a racecondion as this loop goes over all
          // k,j,i and we updating the i+1 flux here. However, the results are
          // idential because u0_prim is never updated in this kernel so we don't
          // worry about it.
          // TODO(pgrete) as we need to keep the function signature idential for now
          // (due to Cuda compiler bug) we could potentially template these function
          // and get rid of the `if constexpr`
          riemann.Solve(eos, k, j, i, IV1, u0_prim, u0_cons, c_h);
          riemann.Solve(eos, k, j, i + 1, IV1, u0_prim, u0_cons, c_h);

          if (ndim >= 2) {
            riemann.Solve(eos, k, j, i, IV2, u0_prim, u0_cons, c_h);
            riemann.Solve(eos, k, j + 1, i, IV2, u0_prim, u0_cons, c_h);
          }
          if (ndim >= 3) {
            riemann.Solve(eos, k, j, i, IV3, u0_prim, u0_cons, c_h);
            riemann.Solve(eos, k + 1, j, i, IV3, u0_prim, u0_cons, c_h);
          }
          lnum_corrected += 1;
        },
        Kokkos::Sum<std::int64_t>(num_corrected),
        Kokkos::Sum<std::int64_t>(num_need_floor));
    if (fofc_verbose && (num_corrected > 0 || num_need_floor > 0)) {
      std::cout << "FOFC [" << parthenon::Globals::my_rank << "] attempt " << num_attempts
                << " corrected " << num_corrected << " need_floor " << num_need_floor
                << std::endl;
    }
    num_attempts += 1;
  } while (num_corrected > 0 && num_attempts < 4);

  return TaskStatus::complete;
}

} // namespace Hydro

