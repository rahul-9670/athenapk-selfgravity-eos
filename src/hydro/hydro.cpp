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
#include "../dust/dust_pkg.hpp"
#include <solvers/bicgstab_solver.hpp>
#include <solvers/mg_solver.hpp>
#include "glmmhd/glmmhd.hpp"
#include "double_flux.hpp"
#include "dual_flux_density.hpp"
#include "diffusion/diffusion.hpp"
#include "diffusion/ionization_build.hpp"
#include "hydro.hpp"
#include "interface/params.hpp"
#include "outputs/outputs.hpp"
#include "prolongation/custom_ops.hpp"
#include "rsolvers/rsolvers.hpp"
#include "utils/error_checking.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/radiation_groups.hpp"
#include "../radiation/radiation_opacity.hpp"
#include "../self_gravity/self_gravity.hpp"
#include "../utils/row_tile.hpp"

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
  if (pin->GetOrAddBoolean("physics", "radiation", false)) {
    packages.Add(Radiation::Initialize(pin.get()));
  }
  // This tree carries hydro/MHD + self-gravity + the tabulated EOS + radiation (Sn) + non-ideal MHD + dust
  // evolution. Refuse switches for packages that are not part of it rather than silently running without them.
  PARTHENON_REQUIRE_THROWS(!pin->GetOrAddBoolean("physics", "chemistry", false),
                           "<physics> chemistry is not part of this build.");
  if (pin->GetOrAddBoolean("physics", "dust", false)) {
    packages.Add(Dust::Initialize(pin.get()));
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
    // radiation.cpp loads its OWN EosTable (own clamp counter); drain and sum both banks
    // so the report covers the whole run (MatterCoupling/probe lookups live on radiation's copy).
    if (pmesh->packages.AllPackages().count("radiation") > 0) {
      const auto &retab =
          pmesh->packages.Get("radiation")->Param<EOSTable::EosTable>("eos_tab");
      unsigned long long cr[EOSTable::EosTable::kNClampCtr] = {0};
      if (retab.ReadAndResetClampCounts(cr)) {
        for (int q = 0; q < EOSTable::EosTable::kNClampCtr; ++q) c[q] += cr[q];
        have_eos_ctr = true;
      }
    }
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

  // ---- OPACITY-TABLE OUT-OF-RANGE CLAMP REPORT (2026-08-17) --------------------------------
  // The other half of the same job: both opacity tables clamp exactly like the EOS one, and a
  // silently clamped opacity gives a wrong answer while every run reports success.
  // Same reduction pattern as above, and drained in the same cycle so a run that walks off both
  // tables says so once, not twice.
  if (pmesh->packages.AllPackages().count("radiation") > 0) {
    auto rad_pkg = pmesh->packages.Get("radiation");
    unsigned long long cg[Radiation::OpacityTable::kNClampCtr] = {0};
    unsigned long long cm[Radiation::GroupOpacityTable::kNClampCtr] = {0};
    const auto &op = rad_pkg->Param<Radiation::OpacityParams>("opacity");
    const auto &gt = rad_pkg->Param<Radiation::GroupOpacityTable>("optable");
    const bool have_g = op.table.ReadAndResetClampCounts(cg);
    const bool have_m = gt.ReadAndResetClampCounts(cm);
    if (have_g || have_m) {
      // 2026-08-21: [0..3] gray STATE, [4..7] gray PROBE, [8..9] group-mult STATE,
      // [10..11] group-mult PROBE. See the EOS block above for why the split exists.
      unsigned long long buf[12] = {cg[0], cg[1], cg[2], cg[3], cg[4], cg[5],
                                    cg[6], cg[7], cm[0], cm[1], cm[2], cm[3]};
#ifdef MPI_PARALLEL
      PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, 12, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                                        MPI_COMM_WORLD));
#endif
      unsigned long long tot = 0, tot_probe = 0;
      for (int q = 0; q < 4; ++q) tot += buf[q];
      for (int q = 8; q < 10; ++q) tot += buf[q];
      for (int q = 4; q < 8; ++q) tot_probe += buf[q];
      for (int q = 10; q < 12; ++q) tot_probe += buf[q];
      if ((tot > 0 || tot_probe > 0) && parthenon::Globals::my_rank == 0) {
        Real lrl = 0, lrh = 0, lTl = 0, lTh = 0;
        if (have_g) op.table.AxisRanges(lrl, lrh, lTl, lTh);
        std::cout << (tot > 0 ? "### WARNING opacity table: " : "### NOTE opacity table: ") << tot
                  << " CELL-STATE (+ " << tot_probe << " solver-probe)"
                  << " out-of-range lookups were EDGE-CLAMPED in the previous cycle"
                     " (nearest-neighbour extrapolation, NOT interpolation). gray:"
                     "  rho<" << lrl << ": " << buf[0] << "  rho>" << lrh << ": " << buf[1]
                  << "  T<" << lTl << ": " << buf[2] << "  T>" << lTh << ": " << buf[3]
                  << " | group multipliers:  T below: " << buf[8]
                  << "  T above: " << buf[9]
                  << " | PROBE gray  rho<: " << buf[4] << "  rho>: " << buf[5]
                  << "  T<: " << buf[6] << "  T>: " << buf[7]
                  << " | PROBE mult  T below: " << buf[10] << "  T above: " << buf[11]
                  << "   (axes are log10; rho in CGS, T in K). Regenerate the table over the"
                     " range the run actually visits (src/radiation/gen_opacity_table.py)."
                  << std::endl;
      }
    }
  }

  // D1 instrument (hydro/d1_meminfo, default false): per-rank block count, coarse-fine / same-level
  // neighbour counts, refinement-level range and free/total device memory, printed on every regrid
  // and every d1_meminfo_every cycles. Measurement only.
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
          {1, 1, 1, RowTile(ib.e + 1 - ib.s)}),
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
          // after "Setup complete, executing driver...". Found 2026-08-16 running a 1-D test
          // problem; it cost three debugging cycles because the crash looks like it is in the
          // physics.
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
          {1, 1, 1, RowTile(ib.e + 1 - ib.s)}),
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
  // validation/dual_energy DE-F2 (2026-10-07): clamp only PHYSICAL boundaries. A periodic mesh face is an
  // interior face of the domain; clamping its inflow drained periodic test boxes to vacuum. Production: outflow
  // on all faces -> unchanged.
  using parthenon::BoundaryFace;
  using parthenon::BoundaryFlag;
  const auto &bcs = pm->mesh_bcs;
  const bool c1l = bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic;
  const bool c1u = bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic;
  const bool c2l = bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic;
  const bool c2u = bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic;
  const bool c3l = bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic;
  const bool c3u = bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic;

  const int nhydro = hydro_pkg->Param<int>("nhydro");
  // hydro/boundary_clamp_dual (audit P-10): with dual energy the advected internal energy sits in
  // the slot after the passive scalars and its flux is mass flux x reconstructed e -- zeroing the
  // mass flux but not that one left an energy inflow without mass. Legacy: scalars only.
  const int nscalars = (hydro_pkg->AllParams().hasKey("boundary_clamp_dual") &&
                        hydro_pkg->Param<bool>("boundary_clamp_dual"))
                           ? hydro_pkg->Param<int>("nadvected")
                           : hydro_pkg->Param<int>("nscalars");

  // Packed BY NAME: grav::phi is also Independent+WithFluxes, so a metadata-flag pack could put
  // phi's flux at the flat IDN index. Same trap cons_diag and SelfGravity::ApplyGravitySource
  // both document.
  auto cf = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"},
                                       std::vector<std::string>{"cons"});
  // hydro/boundary_flux_clamp_wall (validation_tests/ideal_mhd I7, 2026-10-12; default false = the zeroing below,
  // bit-identical): on a clamped (inflow) face, keep the clamp's zero mass/energy/scalar flux but give the face the
  // momentum flux of a WALL AT REST with the adjacent interior cell's state, P + B^2/2 - Bn^2 (normal) and -Bn Bt
  // (tangential). Zeroing the whole momentum flux removes the outside pressure: a boundary cell at rest with a tiny
  // inward mass flux is then pushed outward and a rarefaction runs into the box (I7: static atmosphere, rho -4.8 %).
  const bool clamp_wall = hydro_pkg->Param<bool>("boundary_flux_clamp_wall");
  const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

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
          if (clamp_wall) {
            // (k, j, i) is the interior cell adjacent to this face (both callers below). Flux of a reflecting
            // wall: the Riemann problem between the cell and its mirror image has u = 0 at the face and total
            // pressure P* = P_tot - rho c_f u_away (acoustic/linearised two-wave solution; u_away = the cell's
            // normal velocity pointing AWAY from the wall, > 0 on every clamped face because its mass flux is
            // inward): gas leaving the wall lowers the wall pressure (rarefaction), gas at rest keeps P_tot.
            const auto &w = prim_pack(b);
            const int dn = dir - X1DIR;
            const Real bn = w(IB1 + dn, k, j, i);
            const Real b2 = SQR(w(IB1, k, j, i)) + SQR(w(IB2, k, j, i)) + SQR(w(IB3, k, j, i));
            const Real u_away = -sign * w(IV1 + dn, k, j, i);
            const Real bt1 = w(IB1 + (dn + 1) % 3, k, j, i), bt2 = w(IB1 + (dn + 2) % 3, k, j, i);
            const Real cf = eos.FastMagnetosonicSpeed(w(IDN, k, j, i), w(IPR, k, j, i), bn, bt1, bt2);
            const Real ptot = w(IPR, k, j, i) + 0.5 * b2;
            const Real pstar = Kokkos::fmax(ptot - w(IDN, k, j, i) * cf * Kokkos::fmax(u_away, 0.0), 0.0);
            for (int m = 0; m < 3; ++m)
              cfb.flux(dir, IM1 + m, kk, jj, ii) = (m == dn) ? pstar - bn * bn : -bn * w(IB1 + m, k, j, i);
          } else {
            cfb.flux(dir, IM1, kk, jj, ii) = 0.0;
            cfb.flux(dir, IM2, kk, jj, ii) = 0.0;
            cfb.flux(dir, IM3, kk, jj, ii) = 0.0;
          }
          cfb.flux(dir, IEN, kk, jj, ii) = 0.0;
          for (int n = 0; n < nscalars; ++n)
            cfb.flux(dir, nhydro + n, kk, jj, ii) = 0.0;
        };

        if (c1l && fabs(coords.Xf<1>(i) - xlo) < 1.0e-9 * dx) clamp(X1DIR, k, j, i, -1.0);
        if (c1u && fabs(coords.Xf<1>(i + 1) - xhi) < 1.0e-9 * dx) clamp(X1DIR, k, j, i + 1, 1.0);
        if (ndim > 1) {
          if (c2l && fabs(coords.Xf<2>(j) - ylo) < 1.0e-9 * dy) clamp(X2DIR, k, j, i, -1.0);
          if (c2u && fabs(coords.Xf<2>(j + 1) - yhi) < 1.0e-9 * dy) clamp(X2DIR, k, j + 1, i, 1.0);
        }
        if (ndim > 2) {
          if (c3l && fabs(coords.Xf<3>(k) - zlo) < 1.0e-9 * dz) clamp(X3DIR, k, j, i, -1.0);
          if (c3u && fabs(coords.Xf<3>(k + 1) - zhi) < 1.0e-9 * dz) clamp(X3DIR, k + 1, j, i, 1.0);
        }
      });
  return TaskStatus::complete;
}

//! \brief Dual energy: the pdV work, d(rho e)/dt = ... - P div v.
//!
//! The advected part (d(rho e)/dt + div(rho e v)) comes for free: the internal-energy density
//! rides in the last conserved component and the flux loop upwinds it with the mass flux, like a
//! passive scalar. This adds the remaining non-conservative term with the same centred velocity
//! differences the rest of the unsplit sources use. Pressure is taken from `prim`, i.e. from the
//! energy ConsToPrim selected for this cell, so the two energies stay consistent.
void DualEnergyPdV(MeshData<Real> *md, const Real beta_dt) {
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const int idual = pkg->Param<int>("dual_idx");
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const int ndim = md->GetBlockData(0)->GetBlockPointer()->pmy_mesh->ndim;
  // hydro/dual_energy_eta2: record the stage compression for ConsToPrim's shock detector.
  const bool write_comp = pkg->Param<Real>("dual_energy_eta2") > 0.0;
  // hydro/dual_energy_pdv = face: the work from the face velocities instead (dual_flux_density.cpp); the shock
  // detector keeps the centred compression.
  const bool pdv_face = pkg->Param<bool>("dual_energy_pdv_face");
  auto comp_pack = write_comp ? md->PackVariables(std::vector<std::string>{"dual_comp"})
                              : cons_pack;
  // hydro/dual_energy_shock_detector (validation_tests/ideal_mhd IM-F1, 2026-10-12; default "comp" = the legacy
  // per-stage compression -div v * beta_dt, written as is). The alternatives write a dt-INDEPENDENT flag 1 / 0 into
  // dual_comp, so both readers (eta2's distrust_high and shock_eta1's shock_keep, each testing comp >= shock_comp)
  // see the physical shock and not the step size (the legacy value fired in smooth 3D infall at large dt/r and sat AT
  // the threshold in the production accretion shock):
  //   pjump: div v < 0 and the pressure across the two neighbours in some direction differs by > (1 + jump).
  //          Misfires in smooth, grid-scale-steep convergence (3D Noh near the origin: ratio ~2 per two cells).
  //   kjump: div v < 0 and the ENTROPY proxy K = P / rho^gk differs by > (1 + jump): smooth adiabatic compression
  //          keeps K whatever its steepness; a strong shock raises it by orders of magnitude.
  //          Fixes 1D, but in 3D the cell AHEAD of the front is flagged, takes E - KE - ME (truncation noise of
  //          oblique hypersonic flow), heats, and flags ITS upstream neighbour: a chain walks outward (I6 3D N 64).
  //   kv:    kjump AND a velocity jump: max_d -(v_d(+1) - v_d(-1)) > dv * max(|v|, c_s) over the neighbours. Across a
  //          strong shock the normal velocity drops by ~ the inflow speed within two cells; smooth convergence gives
  //          ~2 dx / r of it, and the cell ahead of a noise-heated neighbour sees almost no jump -> the chain breaks.
  const int shock_det = pkg->Param<int>("dual_energy_shock_detector");   // 0 comp, 1 pjump, 2 kjump, 3 kv
  const Real sdv = pkg->Param<Real>("dual_energy_shock_dv");
  const Real sjump = pkg->Param<Real>("dual_energy_shock_jump");
  const Real sgk = pkg->Param<Real>("dual_energy_shock_gamma");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DualEnergyPdV", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        Real divv = (prim(IV1, k, j, i + 1) - prim(IV1, k, j, i - 1)) /
                    (2.0 * coords.template Dxc<1>(k, j, i));
        if (ndim > 1)
          divv += (prim(IV2, k, j + 1, i) - prim(IV2, k, j - 1, i)) /
                  (2.0 * coords.template Dxc<2>(k, j, i));
        if (ndim > 2)
          divv += (prim(IV3, k + 1, j, i) - prim(IV3, k - 1, j, i)) /
                  (2.0 * coords.template Dxc<3>(k, j, i));
        if (!pdv_face) cons(idual, k, j, i) -= beta_dt * prim(IPR, k, j, i) * divv;
        if (write_comp) {
          if (shock_det > 0) {
            // q = P (pjump) or P / rho^gk (kjump) of a neighbour
            auto q = [&](const int kk, const int jj, const int ii) {
              const Real pr = prim(IPR, kk, jj, ii);
              return (shock_det == 1) ? pr : pr / Kokkos::pow(prim(IDN, kk, jj, ii), sgk);
            };
            auto ratio = [](const Real a, const Real c) { return Kokkos::fmax(a, c) / Kokkos::fmin(a, c); };
            Real r = ratio(q(k, j, i + 1), q(k, j, i - 1));
            if (ndim > 1) r = Kokkos::fmax(r, ratio(q(k, j + 1, i), q(k, j - 1, i)));
            if (ndim > 2) r = Kokkos::fmax(r, ratio(q(k + 1, j, i), q(k - 1, j, i)));
            bool shock = (divv < 0.0 && r > 1.0 + sjump);
            if (shock && shock_det == 3) {
              // velocity jump vs the largest speed scale among the neighbours (|v| and the sound speed)
              Real vscale = 0.0, dvmax = 0.0;
              auto spd = [&](const int kk, const int jj, const int ii) {
                const Real v2 = SQR(prim(IV1, kk, jj, ii)) + SQR(prim(IV2, kk, jj, ii)) + SQR(prim(IV3, kk, jj, ii));
                return Kokkos::sqrt(v2 + prim(IPR, kk, jj, ii) / prim(IDN, kk, jj, ii));
              };
              vscale = Kokkos::fmax(spd(k, j, i + 1), spd(k, j, i - 1));
              dvmax = -(prim(IV1, k, j, i + 1) - prim(IV1, k, j, i - 1));
              if (ndim > 1) {
                vscale = Kokkos::fmax(vscale, Kokkos::fmax(spd(k, j + 1, i), spd(k, j - 1, i)));
                dvmax = Kokkos::fmax(dvmax, -(prim(IV2, k, j + 1, i) - prim(IV2, k, j - 1, i)));
              }
              if (ndim > 2) {
                vscale = Kokkos::fmax(vscale, Kokkos::fmax(spd(k + 1, j, i), spd(k - 1, j, i)));
                dvmax = Kokkos::fmax(dvmax, -(prim(IV3, k + 1, j, i) - prim(IV3, k - 1, j, i)));
              }
              shock = dvmax > sdv * vscale;
            }
            comp_pack(b, 0, k, j, i) = shock ? 1.0 : 0.0;
          } else {
            comp_pack(b, 0, k, j, i) = -divv * beta_dt;
          }
        }
      });
  if (pdv_face) Hydro::DualEnergyPdVFace(md, beta_dt);
}

TaskStatus AddUnsplitSources(MeshData<Real> *md, const SimTime &tm, const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (hydro_pkg->Param<bool>("dual_energy")) DualEnergyPdV(md, beta_dt);

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

// BuildIonizationModel lives in diffusion/ionization_build.hpp.
using Ionization::BuildIonizationModel;

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
  const bool dual_energy = pin->GetOrAddBoolean("hydro", "dual_energy", false);
  const Real dual_eta1 = pin->GetOrAddReal("hydro", "dual_energy_eta1", 1.0e-2);

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
    // hydro/glmmhd_damping (IM-F3): "lie" (default, per-stage multiply in DednerSource) or "midpoint" (second-order
    // exponential integrator, GLMDampingExponential; VL2 only).
    const auto glm_damp = pin->GetOrAddString("hydro", "glmmhd_damping", "lie");
    PARTHENON_REQUIRE_THROWS(glm_damp == "lie" || glm_damp == "midpoint", "hydro/glmmhd_damping must be lie or midpoint");
    PARTHENON_REQUIRE_THROWS(glm_damp == "lie" || pin->GetString("parthenon/time", "integrator") == "vl2",
                             "hydro/glmmhd_damping = midpoint is written for the vl2 integrator");
    pkg->AddParam<bool>("glmmhd_damping_midpoint", glm_damp == "midpoint");
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
  pkg->AddParam<>("boundary_flux_clamp_wall",
                  bflux_clamp && pin->GetOrAddBoolean("hydro", "boundary_flux_clamp_wall", false));
  {
    // hydro/diode_energy_consistent (IM-F2): see DiodeBC. Only the standard diode kernel implements it.
    const bool dec = pin->GetOrAddBoolean("hydro", "diode_energy_consistent", false);
    PARTHENON_REQUIRE_THROWS(!dec || (!pin->GetOrAddBoolean("hydro", "fused_bc", false) &&
                                      !pin->GetOrAddBoolean("hydro", "fused_bc_mesh", false)),
                             "hydro/diode_energy_consistent is implemented in the standard DiodeBC only "
                             "(fused_bc / fused_bc_mesh must be false)");
    pkg->AddParam<>("diode_energy_consistent", dec);
  }
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
    // (validation/self_gravity, L7).
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
  // hydro/double_flux (validation/eos_tabulated E7 / E-F10, 2026-10-05; default false): pressure-exact
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
      // (double_flux.cpp); validated by validation/eos_tabulated E7r.
      PARTHENON_REQUIRE(!first_order_flux_correct,
                        "hydro/double_flux: first_order_flux_correct not supported (it replaces the fluxes the "
                        "double-flux increments were built for)");
      // Dual energy (2026-10-06): no extra coupling needed. ConsToPrim re-syncs the advected internal energy to
      // E - KE - ME in every non-switched cell at every stage, so the double-flux reset of E carries over; a
      // switched cell (e_int < dual_energy_eta1 E) rewrites E from the advected energy and discards the reset --
      // there the thermal pressure is dynamically minor by construction. Validated: eos_tabulated E7s.
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
      // collapse_be's barotropic step uses the table too: it prescribes T(rho) and takes the
      // internal energy from the table at that T (collapse_be.cpp, ApplyBarotropicCooling).
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
      // validation/eos_tabulated E-F5 (2026-10-05): the tabulated EOS takes the floor energy from the table
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

    // Only Ohmic and ambipolar diffusion are part of this tree. The Hall term, thermal conduction and
    // viscosity of the full code are refused rather than silently ignored.
    PARTHENON_REQUIRE_THROWS(pin->GetOrAddString("diffusion", "hall", "none") == "none",
                             "diffusion/hall is not part of this tree (Ohmic + ambipolar only).");
    PARTHENON_REQUIRE_THROWS(pin->GetOrAddString("diffusion", "conduction", "none") == "none",
                             "diffusion/conduction is not part of this tree.");
    PARTHENON_REQUIRE_THROWS(pin->GetOrAddString("diffusion", "viscosity", "none") == "none",
                             "diffusion/viscosity is not part of this tree.");
    auto resistivity = Resistivity::none;
    auto resistivity_str = pin->GetOrAddString("diffusion", "resistivity", "none");
    if (resistivity_str == "ohmic") {
      resistivity = Resistivity::ohmic;
    } else if (resistivity_str != "none") {
      PARTHENON_FAIL("Unknown resistivity method. Options are: none, ohmic");
    }
    // If resistivity is enabled, process supported coefficients
    if (resistivity != Resistivity::none) {
      auto resistivity_coeff_str =
          pin->GetOrAddString("diffusion", "resistivity_coeff", "none");
      auto resistivity_coeff = ResistivityCoeff::none;
      // Ceiling on the ionization-model eta_O (code units; default disabled). Bounds the
      // parabolic dt in the diffusion-decoupled first-core interior; ignored for
      // resistivity_coeff=fixed. See OhmicDiffusivity::eta_cap_.
      const Real eta_ohm_cap_code = pin->GetOrAddReal(
          "diffusion", "eta_ohm_cap_code", std::numeric_limits<Real>::max());

      if (resistivity_coeff_str == "spitzer") {
        // If this is implemented, check how the Spitzer coeff for thermal conduction is
        // handled.
        PARTHENON_FAIL("needs impl");

      } else if (resistivity_coeff_str == "fixed") {
        resistivity_coeff = ResistivityCoeff::fixed;
        Real ohm_diff_coeff_code = pin->GetReal("diffusion", "ohm_diff_coeff_code");
        auto ohm_diff = OhmicDiffusivity(resistivity, resistivity_coeff,
                                         ohm_diff_coeff_code, 0.0, 0.0, 0.0);
        pkg->AddParam<>("ohm_diff", ohm_diff);

      } else if (resistivity_coeff_str == "density_power") {
        // VALIDATION ONLY (nonideal_mhd A5): eta = ohm_diff_coeff_code * rho^ohm_rho_power.
        resistivity_coeff = ResistivityCoeff::density_power;
        const Real c = pin->GetReal("diffusion", "ohm_diff_coeff_code");
        const Real pw = pin->GetOrAddReal("diffusion", "ohm_rho_power", 0.0);
        pkg->AddParam<>("ohm_diff", OhmicDiffusivity(resistivity, resistivity_coeff, c, 0.0, 0.0, 0.0,
                                                     Ionization::IonizationModel(),
                                                     std::numeric_limits<Real>::max(), pw));

      } else if (resistivity_coeff_str == "ionization") {
        // Self-consistent Ohmic resistivity eta_O = c^2/(4 pi sigma_O) from the reduced
        // CR+thermal+grain ionization model (no ohm_diff_coeff_code needed).
        resistivity_coeff = ResistivityCoeff::ionization;
        Ionization::IonizationModel ion = BuildIonizationModel(pin);
        auto ohm_diff = OhmicDiffusivity(resistivity, resistivity_coeff, 0.0, 0.0, 0.0,
                                         0.0, ion, eta_ohm_cap_code);
        pkg->AddParam<>("ohm_diff", ohm_diff);
        if (parthenon::Globals::my_rank == 0) {
          std::cout << "## Ohmic resistivity: self-consistent ionization model "
                       "(conductivity-tensor sigma_O)"
                    << std::endl;
          if (eta_ohm_cap_code < std::numeric_limits<Real>::max()) {
            std::cout << "## Ohmic eta_O cap: " << eta_ohm_cap_code << " (code units)"
                      << std::endl;
          }
        }

      } else {
        PARTHENON_FAIL("Resistivity is enabled but no valid coefficient is set. Set "
                       "diffusion/resistivity_coeff to 'fixed' (+ohm_diff_coeff_code) or "
                       "'ionization'.");
      }
    }
    pkg->AddParam<>("resistivity", resistivity);

    auto ambipolar = Ambipolar::none;
    auto ambipolar_str = pin->GetOrAddString("diffusion", "ambipolar", "none");
    if (ambipolar_str == "ambipolar") {
      ambipolar = Ambipolar::ambipolar;
    } else if (ambipolar_str != "none") {
      PARTHENON_FAIL("Unknown ambipolar method. Options are: none, ambipolar");
    }
    // If ambipolar diffusion is enabled, process supported coefficients
    if (ambipolar != Ambipolar::none) {
      PARTHENON_REQUIRE_THROWS(fluid == Fluid::glmmhd,
                               "Ambipolar diffusion requires MHD (hydro/fluid=glmmhd).");
      auto ambipolar_coeff_str =
          pin->GetOrAddString("diffusion", "ambipolar_coeff", "none");
      auto ambipolar_coeff = AmbipolarCoeff::none;
      // Ceiling on eta_A for the ionization families (code units; ignored for `fixed`): bounds the
      // eta_A ~ B^2/(rho_i rho) low-density runaway. Default 10 (audit F-31) keeps the cap diagnostic
      // live while being a no-op on the measured physics; production passes its own value on the CLI
      // (100 through leg 2, 1e30 = uncapped from the next leg).
      const Real eta_ad_cap_code =
          pin->GetOrAddReal("diffusion", "eta_ad_cap_code", 10.0);
      if (parthenon::Globals::my_rank == 0) {
        std::cout << "## Ambipolar eta_A cap: " << eta_ad_cap_code
                  << " (code units). Default 10.0 is ~8x above the maximum eta_A measured on"
                     " the production snapshot (1.23), so it is a no-op that makes cap-VA/MA/DA"
                     " a live alarm rather than a constant zero. See audit F-31." << std::endl;
      }

      if (ambipolar_coeff_str == "fixed") {
        ambipolar_coeff = AmbipolarCoeff::fixed;
        Real ambipolar_coeff_code = pin->GetReal("diffusion", "ambipolar_coeff_code");
        auto ad_diff = AmbipolarDiffusivity(ambipolar, ambipolar_coeff,
                                            ambipolar_coeff_code, 0.0, 0.0, 0.0);
        pkg->AddParam<>("ad_diff", ad_diff);
        // AUDIT 2026-08-05 (B5). AmbipolarDiffusivity::Get's `fixed` branch returns
        // coeff*B^2 UNCAPPED by design (diffusion.hpp) -- eta_ad_cap_code is threaded only
        // into the ionization-family constructors above/below. That is correct (the fixed
        // path is the Athena++-matched idealised coefficient), but it makes the key a
        // silent no-op for anyone A/B-ing the cap against the fixed path. Say so.
        if (parthenon::Globals::my_rank == 0 &&
            eta_ad_cap_code < std::numeric_limits<Real>::max()) {
          std::cout << "## WARNING: diffusion/eta_ad_cap_code = " << eta_ad_cap_code
                    << " is IGNORED with ambipolar_coeff=fixed (eta_A = coeff*B^2 is "
                       "uncapped by design). The cap applies only to the "
                       "ionization coefficient family."
                    << std::endl;
        }

      } else if (ambipolar_coeff_str == "ionization") {
        // Self-consistent eta_A = B^2/(4 pi gamma_AD rho_i rho) with rho_i from the
        // reduced CR+thermal+grain ionization model (replaces the constant Q_A). Defaults
        // are the shared FHC code-unit calibration; override per <diffusion> ion_* keys.
        ambipolar_coeff = AmbipolarCoeff::ionization;
        Ionization::IonizationModel ion = BuildIonizationModel(pin);
        auto ad_diff = AmbipolarDiffusivity(ambipolar, ambipolar_coeff, 0.0, 0.0, 0.0, 0.0,
                                            ion, eta_ad_cap_code);
        pkg->AddParam<>("ad_diff", ad_diff);
        if (parthenon::Globals::my_rank == 0) {
          const bool tensor = ion.ad_closure == Ionization::ADClosure::tensor;
          std::cout << "## Ambipolar diffusion: self-consistent ionization model"
                    << " (CR zeta=" << ion.zeta << " s^-1, MRN dust f_dg=" << ion.f_dg
                    << ", thermal K x_K=" << ion.x_K
                    << ", AD closure=" << (tensor ? "tensor" : "single_fluid") << ")"
                    << std::endl;
          if (eta_ad_cap_code < std::numeric_limits<Real>::max()) {
            std::cout << "## Ambipolar eta_A cap: " << eta_ad_cap_code << " (code units)"
                      << std::endl;
          }
        }

      } else {
        PARTHENON_FAIL("Ambipolar diffusion is enabled but no valid coefficient is set. "
                       "Set diffusion/ambipolar_coeff to 'fixed' (+ambipolar_coeff_code) or "
                       "'ionization'.");
      }
    }
    pkg->AddParam<>("ambipolar", ambipolar);

    auto diffint_str = pin->GetOrAddString("diffusion", "integrator", "none");
    auto diffint = DiffInt::none;
    if (diffint_str == "unsplit") {
      diffint = DiffInt::unsplit;
    } else if (diffint_str == "rkl2") {
      diffint = DiffInt::rkl2;
      auto rkl2_dt_ratio = pin->GetOrAddReal("diffusion", "rkl2_max_dt_ratio", -1.0);
      pkg->AddParam<>("rkl2_max_dt_ratio", rkl2_dt_ratio);
    } else if (diffint_str != "none") {
      PARTHENON_FAIL("AthenaPK unknown integration method for diffusion processes. "
                     "Options are: none, unsplit, rkl2");
    }
    if (diffint != DiffInt::none) {
      // As in Athena++ a cfl safety factor is also applied to the theoretical limit.
      // By default it is equal to the hyperbolic cfl.
      auto cfl_diff = pin->GetOrAddReal("diffusion", "cfl", pkg->Param<Real>("cfl"));
      pkg->AddParam<>("cfl_diff", cfl_diff);
    }
    pkg->AddParam<Real>("dt_diff", std::numeric_limits<Real>::max(),
                        Params::Mutability::Mutable); // diffusive timestep constraint
    pkg->AddParam<>("diffint", diffint);

    // Fused non-ideal diffusive-timestep path. The per-cell Wardle conductivity tensor
    // (Ionization::Diffusivities) is the dominant GPU cost, and the three per-term dt
    // reductions each recompute the full tensor but keep only one eta. When EVERY active
    // non-ideal term uses the "ionization" coefficient, the reductions share ONE tensor
    // evaluation per cell (FusedNonidealEval) where the per-term path ran three.
    // Fixed-coeff (cheap) terms keep their exact per-term estimators.
    {
      bool any_nonideal = false;
      bool all_ionization = true;
      if (resistivity != Resistivity::none) {
        any_nonideal = true;
        const auto ct = pkg->Param<OhmicDiffusivity>("ohm_diff").GetCoeffType();
        if (ct != ResistivityCoeff::ionization)
          all_ionization = false;
      }
      if (ambipolar != Ambipolar::none) {
        any_nonideal = true;
        const auto ct = pkg->Param<AmbipolarDiffusivity>("ad_diff").GetCoeffType();
        if (ct != AmbipolarCoeff::ionization)
          all_ionization = false;
      }
      // NOTE (profiled 2026-07-11, job 2321477): the per-term estimators were ~15% of
      // all GPU kernel time in the mixed production run (2x843ms + 2x421ms per cycle).
      const bool fusable = any_nonideal && all_ionization;
      // Optional override (default = auto-on when fusable). Setting
      // diffusion/fused_nonideal_dt=false forces the exact per-term estimators (used for
      // A/B validation and as an escape hatch); it can never enable fusion where the
      // coefficients are not all ionization-family.
      const bool fused = pin->GetOrAddBoolean("diffusion", "fused_nonideal_dt", fusable);
      pkg->AddParam<>("nonideal_dt_fused", fused && fusable);
      // diffusion/dt_sum_ohm_ad (audit P-12, default false = legacy separate minima). For a current
      // perpendicular to B the EMF is (eta_O + eta_A) J, so the explicit limit is set by the SUM;
      // the legacy estimate can be up to 2x optimistic (covered so far only by cfl_diff = 0.3).
      pkg->AddParam<>("dt_sum_ohm_ad", pin->GetOrAddBoolean("diffusion", "dt_sum_ohm_ad", false));

    }

    // Cell-centered non-ideal diffusivity cache. When enabled, CalcDiffFluxes first
    // fills a 3-component derived field (eta_O, eta_H, eta_A) once per stage
    // (PrecomputeNonidealEta) and the Ohmic/ambipolar flux kernels face-average
    // the cached per-cell values instead of re-running the (expensive) ionization
    // solve per face -- cutting up to 9 tensor evaluations per cell per stage to 1.
    // This is the Athena++ pattern (cell-centered CalcDiffusivity, then face/edge
    // averaging). Face eta becomes the arithmetic mean of the two adjacent cell
    // values rather than eta(face-averaged state): same order of accuracy, but not
    // bit-identical -- diffusion/eta_cache=false restores the per-face evaluation
    // for A/B validation. Default: on when any active term uses an ionization-type
    // coefficient (where it pays), off for purely fixed coefficients (where the
    // per-face evaluation is trivially cheap and bit-identical history matters).
    {
      const bool any_nonideal = (resistivity != Resistivity::none) ||
                                (ambipolar != Ambipolar::none);
      bool any_ionization = false;
      if (resistivity != Resistivity::none) {
        const auto ct = pkg->Param<OhmicDiffusivity>("ohm_diff").GetCoeffType();
        any_ionization |= (ct == ResistivityCoeff::ionization);
      }
      if (ambipolar != Ambipolar::none) {
        const auto ct = pkg->Param<AmbipolarDiffusivity>("ad_diff").GetCoeffType();
        any_ionization |= (ct == AmbipolarCoeff::ionization);
      }
      const bool eta_cache =
          pin->GetOrAddBoolean("diffusion", "eta_cache", any_ionization) && any_nonideal;
      // Under eos=hydrogen the correct code temperature for the ionization model comes from
      // the EOS table; that plumbing exists on the cached path (PrecomputeNonidealEta + the
      // fused dt estimator) but not the per-face cache-off flux kernels, so require the
      // cache there (it is the default when any term uses an ionization coefficient).
      PARTHENON_REQUIRE(!(use_h2diss && any_ionization) || eta_cache,
                        "eos=hydrogen with ionization-coefficient non-ideal MHD requires "
                        "diffusion/eta_cache=true (the default).");
      // Audit A5: under eos=hydrogen the per-term dt estimators (T = P/rho, ideal-gas identity) are
      // wrong; only the fused estimator uses the table T. Refuse the non-fused path (mixed coefficient
      // families or fused_nonideal_dt=false). Production is fused.
      PARTHENON_REQUIRE(
          !(use_h2diss && any_ionization) || pkg->Param<bool>("nonideal_dt_fused"),
          "eos=hydrogen with ionization-coefficient non-ideal MHD requires the FUSED "
          "non-ideal dt estimator (diffusion/fused_nonideal_dt): the per-term estimators "
          "derive the ionization temperature as p/rho, which holds only for an ideal gas. "
          "Keep every active non-ideal coefficient in the ionization family so fusion is "
          "available, leave diffusion/fused_nonideal_dt at its default, or use eos=adiabatic.");
      pkg->AddParam<>("nonideal_eta_cache", eta_cache);
      // diffusion/eta_face_harmonic (nonideal_mhd A5, default false = arithmetic, unchanged): harmonic mean of the two
      // cached cell etas at OHMIC faces only (FaceEta, diffusion.hpp; AD stays arithmetic, see ambipolar.cpp).
      pkg->AddParam<>("eta_face_harmonic", pin->GetOrAddBoolean("diffusion", "eta_face_harmonic", false));
      if (eta_cache) {
        Metadata m_eta({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                       std::vector<int>({3}),
                       std::vector<std::string>{"eta_ohmic", "eta_hall", "eta_ambipolar"});
        pkg->AddField("nonideal_eta", m_eta);
        if (parthenon::Globals::my_rank == 0) {
          std::cout << "## Non-ideal eta cache: cell-centered (eta_O, eta_H, eta_A) "
                       "precomputed once per stage; flux kernels use face-averaged "
                       "cached values (diffusion/eta_cache=false to disable)."
                    << std::endl;
        }
      }

      // AUDIT ITEM 5 -- WS-4 DUST -> CONDUCTIVITY COUPLING.
      // Grains dominate recombination over most of the collapse, so they set x_e and hence
      // all three non-ideal diffusivities. Without the coupling the ionization model uses a FROZEN ISM
      // MRN population while the dust package evolves (f_dg, a_c) for the OPACITY only --
      // i.e. the run's own grain evolution never reaches its own magnetic microphysics.
      // With this on, the conductivity grain population is rescaled per cell to the evolved
      // scalars (see IonizationModel::FDG/Ak and FusedNonidealEval::CellIon).
      // OFF by default; when off nothing is copied and results are bit-identical.
      const bool dust_couple =
          pin->GetOrAddBoolean("diffusion", "dust_coupling", false);
      pkg->AddParam<>("nonideal_dust_on", dust_couple);
      if (dust_couple) {
        PARTHENON_REQUIRE(pin->GetOrAddBoolean("dust", "evolve", false),
                          "diffusion/dust_coupling=true requires the dust package to be "
                          "evolving (<dust> evolve=true) -- otherwise the grain population "
                          "is static and the coupling is a no-op with extra cost.");
        PARTHENON_REQUIRE(any_ionization,
                          "diffusion/dust_coupling=true only affects ionization-family "
                          "coefficients (resistivity/ambipolar _coeff = ionization).");
        // Dust stores f_dg at scalar_index and a_c at scalar_index+1 (see dust.cpp);
        // prim component = nhydro + scalar index, the same convention as x_e.
        const int dsi = pin->GetOrAddInteger("dust", "scalar_index", 0);
        pkg->AddParam<>("nonideal_dust_i_fdg", nhydro + dsi);
        pkg->AddParam<>("nonideal_dust_i_ac", nhydro + dsi + 1);
        // Reference population the STATIC MRN bins represent. a_ref/f_dg_ref must be the
        // values the bins were built from, or the scaling is biased at t=0.
        // NOTE the key is "a_ref_cm", not "a_ref" (see dust.cpp). Reading the wrong key
        // would silently fall back to 1e-5 and bias a_scale for any deck that sets it.
        pkg->AddParam<>("nonideal_dust_a_ref",
                        pin->GetOrAddReal("dust", "a_ref_cm", 1.0e-5));
        pkg->AddParam<>("nonideal_dust_fdg_ref",
                        pin->GetOrAddReal("dust", "f_dg_ref", 0.01));
        // validation/dust (DU-F4): destroyed grains (f_dg <= 0, a_c > 0) -> no grains in the charge solve
        // (FusedNonidealEval::dust_destroyed_zero). Default false = the historical 'reference population' fallback.
        pkg->AddParam<>("nonideal_dust_destroyed_zero",
                        pin->GetOrAddBoolean("diffusion", "dust_destroyed_zero", false));
        if (parthenon::Globals::my_rank == 0) {
          std::cout << "## Dust->conductivity coupling ON (diffusion/dust_coupling): the "
                       "ionization grain population is rescaled per cell from the evolved "
                       "(f_dg, a_c) at prim indices " << nhydro + dsi << ", "
                    << nhydro + dsi + 1 << "." << std::endl;
        }
      }
      // AUDIT ITEM 1 -- non-ideal cap-activation diagnostic.
      // eta_ohm_cap_code / eta_ad_cap_code are numerical stabilizers
      // that MODIFY the induction equation, so a flux-retention number is uninterpretable
      // without knowing where and how hard they bit. Filled inside PrecomputeNonidealEta
      // from the same tensor solve (no extra solve, no extra task); OFF by default, and
      // bit-identical when off (it only writes a Derived field, never feeds back).
      const bool cap_diag = pin->GetOrAddBoolean("diffusion", "cap_diag", false);
      PARTHENON_REQUIRE(!cap_diag || eta_cache,
                        "diffusion/cap_diag=true requires diffusion/eta_cache=true.");
      pkg->AddParam<>("nonideal_cap_diag", cap_diag);
      if (cap_diag) {
        Metadata m_cap({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                       std::vector<int>({CapDiagIdx::NCOMP}),
                       std::vector<std::string>{"cap_flag_O", "cap_flag_H", "cap_flag_A",
                                                "cap_dec_O", "cap_dec_H", "cap_dec_A"});
        pkg->AddField("diff.capdiag", m_cap);
        auto hst_cap = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
        using parthenon::HistoryOutputVar;
        using parthenon::UserHistoryOperation;
        // Normalizer: the summed volume of exactly the cells the flags are summed over.
        // Volume fraction capped = cap-V<term> / cap-Vtot; mass fraction = cap-M<term>/mass;
        // volume-weighted mean clipped decades over the capped set = cap-D<term>/cap-V<term>.
        hst_cap.emplace_back(HistoryOutputVar(UserHistoryOperation::sum,
                                              NonidealCapHstVol, "cap-Vtot"));
        const std::pair<int, const char *> terms[3] = {{CapDiagIdx::flagO, "O"},
                                                       {CapDiagIdx::flagH, "H"},
                                                       {CapDiagIdx::flagA, "A"}};
        for (const auto &tm : terms) {
          const int c = tm.first;
          hst_cap.emplace_back(HistoryOutputVar(
              UserHistoryOperation::sum,
              [c](MeshData<Real> *md) { return NonidealCapHstFlag(md, c); },
              std::string("cap-V") + tm.second));
          hst_cap.emplace_back(HistoryOutputVar(
              UserHistoryOperation::sum,
              [c](MeshData<Real> *md) { return NonidealCapHstMassFlag(md, c); },
              std::string("cap-M") + tm.second));
          hst_cap.emplace_back(HistoryOutputVar(
              UserHistoryOperation::sum,
              [c](MeshData<Real> *md) { return NonidealCapHstDecade(md, c); },
              std::string("cap-D") + tm.second));
        }
        pkg->UpdateParam(parthenon::hist_param_key, hst_cap);
        if (parthenon::Globals::my_rank == 0) {
          std::cout << "## Non-ideal cap diagnostic ON (diffusion/cap_diag): hst cap-V*/"
                       "cap-M*/cap-D* + field diff.capdiag (add to an output block for "
                       "the spatial picture)."
                    << std::endl;
        }
      }
      // Freeze the cached eta across the RKL2 stages of each Strang half: the stage
      // tasks skip the PrecomputeNonidealEta refresh, so eta stays at its value from
      // the half's first stage (refreshed 2x per cycle instead of once per stage).
      // RKL2's stability polynomial (Meyer+2012) is derived for an operator held fixed
      // across the super-step, and the coefficient lag is O(dt/2) on eta(rho,T,B),
      // which evolves on the collapse timescale >> dt. Requires the eta cache; only
      // meaningful with integrator=rkl2. Default off (bit-identical history matters
      // for A/B validation).
      const bool freeze_eta =
          pin->GetOrAddBoolean("diffusion", "rkl2_freeze_eta", false);
      PARTHENON_REQUIRE(!freeze_eta || (eta_cache && diffint == DiffInt::rkl2),
                        "diffusion/rkl2_freeze_eta=true requires diffusion/eta_cache=true "
                        "and diffusion/integrator=rkl2.");
      pkg->AddParam<>("rkl2_freeze_eta", freeze_eta && eta_cache);
      // See AddSTSTasks: skip the RKL2 super-steps when no parabolic term is left on the
      // explicit path at all (dt_diff = inf). Default false = unchanged / bit-identical.
      pkg->AddParam<>("rkl2_skip_empty",
                      pin->GetOrAddBoolean("diffusion", "rkl2_skip_empty", false));
      if (freeze_eta && parthenon::Globals::my_rank == 0) {
        std::cout << "## RKL2 eta freeze: non-ideal eta cache refreshed at the first "
                     "stage of each Strang half only (diffusion/rkl2_freeze_eta)."
                  << std::endl;
      }
    }

    if (fluid == Fluid::glmmhd) {
      // DUAL ENERGY. One extra conserved component holding rho*e_int, advected with the mass flux like a
      // passive scalar and sourced by -P div v. ConsToPrim uses it instead of E - KE - ME wherever that
      // difference is ill-conditioned (e_int <= eta1 * E). Off by default: the component is not registered
      // and the EOS branch is inert.
      const int nscalars_cfg = pin->GetOrAddInteger("hydro", "nscalars", 0);
      const int dual_idx = dual_energy ? nhydro + nscalars_cfg : -1;
      const bool floor_consistent =
          pin->GetOrAddBoolean("hydro", "eos_floor_consistent", false);
      if (floor_consistent && use_h2diss && parthenon::Globals::my_rank == 0)
        std::cout << "## EOS pressure floor CONSISTENT with the table (hydro/eos_floor_consistent): "
                     "a floored cell stores the pressure of the energy it actually holds." << std::endl;
      // hydro/dual_energy_unsigned_prim (default false = legacy signed prim, bit-identical):
      // prim's dual component carries the MAGNITUDE only and the switch flag moves to the
      // `dual_switched` field. The signed legacy value is advected by the scalar flux; see
      // AdiabaticGLMMHDEOS::ConsToPrim.
      const bool dual_unsigned =
          dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_unsigned_prim", false);
      if (dual_unsigned && parthenon::Globals::my_rank == 0)
        std::cout << "## Dual energy: UNSIGNED prim (hydro/dual_energy_unsigned_prim); the switch "
                     "flag is the `dual_switched` field." << std::endl;
      // hydro/dual_energy_exact_resync (default false = legacy, bit-identical): a switched cell
      // stores back the advected energy it used instead of re-deriving it from E (restart seed,
      // see AdiabaticGLMMHDEOS::ConsToPrim).
      const bool dual_exact =
          dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_exact_resync", false);
      if (dual_exact && parthenon::Globals::my_rank == 0)
        std::cout << "## Dual energy: EXACT re-sync in switched cells "
                     "(hydro/dual_energy_exact_resync)." << std::endl;
      const Real dual_eta2 =
          dual_energy ? pin->GetOrAddReal("hydro", "dual_energy_eta2", 0.0) : 0.0;
      const Real dual_kappa = pin->GetOrAddReal("hydro", "dual_energy_kappa", 3.0);
      const Real dual_shock = pin->GetOrAddReal("hydro", "dual_energy_shock_comp", 0.02);
      const auto shock_det_s = pin->GetOrAddString("hydro", "dual_energy_shock_detector", "comp");
      PARTHENON_REQUIRE_THROWS(shock_det_s == "comp" || shock_det_s == "pjump" || shock_det_s == "kjump" ||
                                   shock_det_s == "kv",
                               "hydro/dual_energy_shock_detector must be comp, pjump, kjump or kv");
      PARTHENON_REQUIRE_THROWS(shock_det_s == "comp" || dual_eta2 > 0.0,
                               "hydro/dual_energy_shock_detector != comp needs hydro/dual_energy_eta2 > 0 (it writes dual_comp)");
      const int shock_det_i = (shock_det_s == "comp") ? 0 : (shock_det_s == "pjump") ? 1 : (shock_det_s == "kjump") ? 2 : 3;
      // F2 (2026-10-07): shock exemption also from the eta1 criterion (needs eta2 > 0: it reads dual_comp)
      const bool dual_shock_eta1 =
          dual_eta2 > 0.0 && pin->GetOrAddBoolean("hydro", "dual_energy_shock_eta1", false);
      if (dual_eta2 > 0.0 && parthenon::Globals::my_rank == 0)
        std::cout << "## Dual energy: TWO-SIDED switch eta2=" << dual_eta2 << " kappa=" << dual_kappa
                  << " shock_comp=" << dual_shock << " (hydro/dual_energy_eta2)." << std::endl;
      AdiabaticGLMMHDEOS eos(pfloor, dfloor, efloor, vceil, eceil, gamma, use_h2diss,
                             eos_tab, dual_idx, dual_eta1,
                             floor_consistent, dual_unsigned, dual_exact, dual_eta2, dual_kappa,
                             dual_shock, dual_shock_eta1);
      pkg->AddParam<>("eos", eos);
      // IM-F1 (read by DualEnergyPdV)
      pkg->AddParam<>("dual_energy_shock_detector", shock_det_i);
      pkg->AddParam<>("dual_energy_shock_jump", pin->GetOrAddReal("hydro", "dual_energy_shock_jump", 1.0));
      pkg->AddParam<>("dual_energy_shock_gamma", pin->GetOrAddReal("hydro", "dual_energy_shock_gamma", 5.0 / 3.0));
      pkg->AddParam<>("dual_energy_shock_dv", pin->GetOrAddReal("hydro", "dual_energy_shock_dv", 0.3));
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
  // Advected components after `nhydro`: the passive scalars plus, with dual energy on, the
  // internal-energy density in the LAST slot (so no scalar index shifts).
  // GUARD. collapse_be's barotropic step overwrites the thermal energy every stage and does not update the
  // advected internal energy, so the two energies would drift apart; dual energy is refused there. With
  // radiation on the barotropic overwrite is skipped (radiation owns the energy, collapse_be.cpp) and the
  // radiation thermal exchange updates the internal energy, so that combination is allowed.
  if (dual_energy) {
    // Ohmic + ambipolar dissipation IS wired into the internal energy (diffusion/dual_energy_heating.cpp,
    // hydro/dual_heat_exact.cpp); it reads the cell-centred eta cache, so that has to be on.
    const auto res = pin->GetOrAddString("diffusion", "resistivity", "none");
    const auto amb = pin->GetOrAddString("diffusion", "ambipolar", "none");
    PARTHENON_REQUIRE_THROWS(
        (res == "none" && amb == "none") ||
            (pkg->AllParams().hasKey("nonideal_eta_cache") && pkg->Param<bool>("nonideal_eta_cache")),
        "hydro/dual_energy with resistivity/ambipolar needs diffusion/eta_cache=true: the "
        "dissipation is evaluated from the cached cell-centred eta.");
    PARTHENON_REQUIRE_THROWS(!(pin->DoesParameterExist("job", "problem_id") &&
                               pin->GetString("job", "problem_id") == "collapse_be") ||
                                 pin->GetOrAddBoolean("physics", "radiation", false),
                             "hydro/dual_energy is not wired into collapse_be's barotropic step "
                             "(it is allowed with <physics> radiation = true).");
  }
  const int nadvected = nscalars + (dual_energy ? 1 : 0);
  pkg->AddParam("dual_energy", dual_energy);
  // See hydro_driver.cpp (end of the final stage): re-sync cons_dual/prim at the end of the step so a
  // continued run matches a resumed one. Default false == previous behaviour bit-for-bit.
  pkg->AddParam("dual_energy_sync_step_end",
                pin->GetOrAddBoolean("hydro", "dual_energy_sync_step_end", false));
  pkg->AddParam("dual_idx", dual_energy ? nhydro + nscalars : -1);
  // hydro/dual_energy_flux (validation/dual_energy DE-F3, default "specific" = legacy, task graph unchanged):
  // "density" rebuilds the dual flux from the reconstructed rho*e (hydro/dual_flux_density.cpp) so switched cells
  // keep pressure equilibrium at contacts. PLM only; needs the unsigned prim (the legacy signed flag would flip q).
  {
    const auto dflx = pin->GetOrAddString("hydro", "dual_energy_flux", "specific");
    PARTHENON_REQUIRE_THROWS(dflx == "specific" || dflx == "density",
                             "hydro/dual_energy_flux must be 'specific' (default) or 'density'");
    const bool dens = dual_energy && dflx == "density";
    PARTHENON_REQUIRE_THROWS(!dens || (pin->GetString("hydro", "reconstruction") == "plm" &&
                                       pin->GetOrAddBoolean("hydro", "dual_energy_unsigned_prim", false)),
                             "hydro/dual_energy_flux = density needs reconstruction = plm and "
                             "hydro/dual_energy_unsigned_prim = true");
    pkg->AddParam("dual_energy_flux_density", dens);
    // hydro/dual_energy_double_flux (DE-F3 part (b), default false): own-EOS dual-energy update in double-flux contact
    // cells (hydro/dual_double_flux.cpp). Needs the density flux and hydro/double_flux.
    const bool ddf = dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_double_flux", false);
    PARTHENON_REQUIRE_THROWS(!ddf || (dens && pin->GetOrAddBoolean("hydro", "double_flux", false)),
                             "hydro/dual_energy_double_flux needs hydro/dual_energy_flux = density and "
                             "hydro/double_flux = true");
    pkg->AddParam("dual_energy_double_flux", ddf);
    const auto pdv = pin->GetOrAddString("hydro", "dual_energy_pdv", "centred");
    PARTHENON_REQUIRE_THROWS(pdv == "centred" || pdv == "face", "hydro/dual_energy_pdv must be 'centred' or 'face'");
    pkg->AddParam("dual_energy_pdv_face", dual_energy && pdv == "face");
    // Face volume fluxes of the dual-energy terms that need u_f = F_mass/rho_f (dual_double_flux.cpp
    // DualFaceVolumeFlux): component 0 = F_mass/rho_f, 1 = F_mass P_f/rho_f, in the FLUX slots so that Parthenon's
    // flux correction replaces them at fine/coarse faces together with the mass flux. Built from the coarse cell's
    // own reconstruction instead, a coarse cell next to a finer block mixed the fine mass flux with its own face
    // density: contacts crossing a level jump kept max|P/P0-1| 2e-4 (SMR rho 1|10) vs 8e-11 on the uniform mesh.
    // Metadata as dflux_xyz (OperatorSplit: never in a {WithFluxes, Cell} pack). Only registered when used.
    const bool vflux = ddf || (dual_energy && pdv == "face");
    pkg->AddParam("dual_vflux", vflux);
    if (vflux) {
      const parthenon::MetadataFlag OperatorSplit =
          parthenon::Metadata::FlagNameExists("OperatorSplit")
              ? parthenon::Metadata::GetUserFlag("OperatorSplit")
              : parthenon::Metadata::AddUserFlag("OperatorSplit");
      pkg->AddField("dual_vflux", Metadata({Metadata::Cell, Metadata::Independent, Metadata::WithFluxes,
                                            OperatorSplit},
                                           std::vector<int>({2})));
    }
  }
  pkg->AddParam("dual_energy_eta1", pin->GetOrAddReal("hydro", "dual_energy_eta1", 1.0e-2));
  const Real dual_eta2_p = dual_energy ? pin->GetOrAddReal("hydro", "dual_energy_eta2", 0.0) : 0.0;
  pkg->AddParam("dual_energy_eta2", dual_eta2_p);
  // hydro/dual_comp_ghost (audit P-02, default false = legacy). ConsToPrim runs over the ENTIRE
  // block and reads `dual_comp` in ghost cells, where DualEnergyPdV never writes it (allocation
  // zero). A ghost copy of a compressing cell could then take the dual branch while its owner keeps
  // the E-derived energy, and the two blocks computed DIFFERENT fluxes on their shared same-level
  // face (no flux correction there). With the key on the field is exchanged with the stage halo.
  const bool dual_comp_ghost =
      dual_eta2_p > 0.0 && pin->GetOrAddBoolean("hydro", "dual_comp_ghost", false);
  pkg->AddParam("dual_comp_ghost", dual_comp_ghost);
  if (dual_eta2_p > 0.0) {
    // -div v * (stage dt) per cell, written by DualEnergyPdV, read by ConsToPrim (shock detector).
    Metadata m_comp(dual_comp_ghost
                        ? std::vector<parthenon::MetadataFlag>{Metadata::Cell, Metadata::Derived,
                                                    Metadata::OneCopy, Metadata::FillGhost}
                        : std::vector<parthenon::MetadataFlag>{Metadata::Cell, Metadata::Derived,
                                                    Metadata::OneCopy});
    pkg->AddField("dual_comp", m_comp);
  }
  // hydro/dual_energy_shared_select (audit P-01, default false = legacy): the Sn seed and matter coupling pick
  // the gas internal energy with ConsToPrim's own two-sided rule (AdiabaticGLMMHDEOS::DualSelectEint) instead
  // of the one-sided eta1 rule. See SnDualSel in radiation/radiation_sn.cpp; no effect without radiation.
  const bool dual_shared_select =
      dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_shared_select", false);
  pkg->AddParam("dual_energy_shared_select", dual_shared_select);
  // hydro/dual_energy_heat_rkl2 (audit P-03, default false = legacy). The legacy path adds
  // tau * (eta_O J^2 + eta_A J_perp^2) once per RKL2 half, AFTER the half, i.e. with the DIFFUSED
  // end-of-half field -- in a stiff cell (t_diff << tau) J has decayed and the advected energy is
  // under-heated. An E-budget ((E-KE-ME)_end - (E-KE-ME)_start) does NOT work instead: every STS
  // stage's ConsToPrim rewrites E from the dual energy in switched cells, erasing exactly the heat
  // that matters. With the key on, the heating is a SOURCE of the dual component inside the RKL2
  // recurrence itself -- the same stages and weights that advance B (see AddSTSTasks) -- and the
  // post-half DualEnergyDiffusionHeating is skipped.
  pkg->AddParam("dual_energy_heat_rkl2",
                dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_heat_rkl2", false));
  // hydro/dual_energy_heat_face (validation_tests/dual_energy DE-F6, default false = centred-J heating): the heat of
  // the RKL2 source is the dissipation of the applied B diffusion fluxes, per face (dual_energy_heating.cpp).
  {
    const bool hf = dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_heat_face", false);
    PARTHENON_REQUIRE_THROWS(!hf || pkg->Param<bool>("dual_energy_heat_rkl2"),
                             "hydro/dual_energy_heat_face needs hydro/dual_energy_heat_rkl2 = true");
    pkg->AddParam("dual_energy_heat_face", hf);
    // hydro/dual_energy_heat_exact (DE-F6, default false): exact end-of-half energy closure (hydro/dual_heat_exact.cpp)
    const bool hx = dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_heat_exact", false);
    // The closure acts inside the RKL2 super-steps only; the frozen-eta requirement applies when they exist (a run
    // without parabolic terms has nothing to close and is unaffected by the key).
    const bool rkl2_on = pin->GetOrAddString("diffusion", "integrator", "none") == "rkl2";
    PARTHENON_REQUIRE_THROWS(!hx || (pkg->Param<bool>("dual_energy_heat_rkl2") &&
                                     (!rkl2_on || pin->GetOrAddBoolean("diffusion", "rkl2_freeze_eta", false))),
                             "hydro/dual_energy_heat_exact needs hydro/dual_energy_heat_rkl2 = true and, with "
                             "diffusion/integrator = rkl2, diffusion/rkl2_freeze_eta = true");
    pkg->AddParam("dual_energy_heat_exact", hx);
    if (hx) {
      Metadata m_ex({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
      pkg->AddField("dual_ex_a1", m_ex);
      pkg->AddField("dual_ex_a2", m_ex);
      pkg->AddField("dual_ex_m0", m_ex);
    }
  }
  // hydro/boundary_clamp_dual (audit P-10, default false = legacy): ClampBoundaryFluxes also zeroes
  // the dual-energy flux of a clamped inflow face (it zeroed only the passive scalars before).
  pkg->AddParam("boundary_clamp_dual",
                dual_energy && pin->GetOrAddBoolean("hydro", "boundary_clamp_dual", false));
  if (parthenon::Globals::my_rank == 0 && dual_energy &&
      (dual_comp_ghost || dual_shared_select || pkg->Param<bool>("boundary_clamp_dual"))) {
    std::cout << "## Dual energy keys: comp_ghost=" << dual_comp_ghost
              << " shared_select=" << dual_shared_select
              << " clamp_dual=" << pkg->Param<bool>("boundary_clamp_dual") << std::endl;
  }
  const bool dual_unsigned_prim =
      dual_energy && pin->GetOrAddBoolean("hydro", "dual_energy_unsigned_prim", false);
  pkg->AddParam("dual_energy_unsigned_prim", dual_unsigned_prim);
  // See hydro_driver.cpp (start of stage 1): extra halo-exchange + FillDerived rounds so the ghost
  // zones a step starts from depend on the interior only (restart reproducibility). 0 = off.
  pkg->AddParam("ghost_sync_rounds", pin->GetOrAddInteger("hydro", "ghost_sync_rounds", 0));
  // See RegisterVarNames in hydro_driver.cpp. Default false = full register copies.
  pkg->AddParam("lean_registers", pin->GetOrAddBoolean("hydro", "lean_registers", false));
  // Speed work (2026-10-08): hydro halo exchanges without the Sn fields (hydro_driver.cpp HydroExchangeData).
  pkg->AddParam("lean_exchange", pin->GetOrAddBoolean("hydro", "lean_exchange", false));
  {
    const bool lr = pin->GetOrAddBoolean("hydro", "lean_exchange_rad", false);
    PARTHENON_REQUIRE(!lr || pin->GetOrAddBoolean("hydro", "lean_exchange", false),
                      "hydro/lean_exchange_rad needs hydro/lean_exchange = true");
    // read-only (radiation.cpp owns the key and its default "sn"; a GetOrAdd here could register a second default)
    PARTHENON_REQUIRE(!lr || !pin->DoesParameterExist("radiation", "transport") ||
                          pin->GetString("radiation", "transport") == "sn",
                      "hydro/lean_exchange_rad is valid only with radiation/transport = sn (M1/FLD read rad.* ghosts)");
    pkg->AddParam("lean_exchange_rad", lr);
  }
  if (dual_unsigned_prim) {
    // Written over IndexDomain::entire by every ConsToPrim; never exchanged or restarted
    // (it is a pure function of cons at the last ConsToPrim, and only cons-ndual reads it).
    Metadata m_sw({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("dual_switched", m_sw);
  }
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
  if (dual_energy) {
    cons_labels.emplace_back("dual_energy_density");
    prim_labels.emplace_back("dual_specific_energy");
  }

  Metadata m(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes},
      std::vector<int>({nhydro + nadvected}), cons_labels);
  // hydro/cons_prolong_pc (VALIDATION ONLY, default false = the production minmod operator): piecewise-constant
  // prolongation of cons, to isolate the fine/coarse source of the residual double-flux contact error
  // (validation/eos_tabulated E7v, 2026-10-06).
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
    add_c(ConsDiag::ndual, "cons-ndual");
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
          {1, 1, 1, RowTile(ib.e + 1 - ib.s)}),
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

  auto dt_diff = std::numeric_limits<Real>::max();
  if (hydro_pkg->Param<DiffInt>("diffint") != DiffInt::none) {
    const auto diffint = hydro_pkg->Param<DiffInt>("diffint");
    // When every active non-ideal term uses the "ionization" coefficient, evaluate the
    // (expensive) Wardle conductivity tensor ONCE per cell and reduce the Ohmic/ambipolar
    // dt together; otherwise (fixed / density_power coefficients) the exact per-term estimators.
    if (hydro_pkg->Param<bool>("nonideal_dt_fused")) {
      dt_diff = std::min(dt_diff, EstimateNonidealTimestepIonizationFused(md));
    } else {
      if (hydro_pkg->Param<Resistivity>("resistivity") != Resistivity::none) {
        dt_diff = std::min(dt_diff, EstimateResistivityTimestep(md));
      }
      if (hydro_pkg->Param<Ambipolar>("ambipolar") != Ambipolar::none) {
        dt_diff = std::min(dt_diff, EstimateAmbipolarTimestep(md));
      }
    }
    if (diffint == DiffInt::unsplit) {
      // unsplit integration: the strict explicit limit
      min_dt = std::min(min_dt, dt_diff);
    } else if (diffint == DiffInt::rkl2) {
      // RKL2: no strict limit (that is what super-time-stepping is for), only the optional cap
      // on the super-step ratio dt / dt_diff
      const auto max_dt_ratio = hydro_pkg->Param<Real>("rkl2_max_dt_ratio");
      if (max_dt_ratio > 0.0 && dt_hyp / dt_diff > max_dt_ratio) {
        min_dt = std::min(min_dt, max_dt_ratio * dt_diff);
      }
    } else {
      PARTHENON_THROW("unknown diffusion integrator");
    }
    auto dt_diff_param = hydro_pkg->Param<Real>("dt_diff");
    hydro_pkg->UpdateParam("dt_diff", std::min(dt_diff, dt_diff_param));
  }

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
  // and receive the hydro fluxes (measured mass loss, 2026-09-19).
  const std::vector<std::string> cons_name{"cons"};
  auto cons_in = md->PackVariablesAndFluxes(cons_name, cons_name);
  auto pkg = pmb->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  // With dual energy on this also covers the internal-energy density (last component): it is
  // advected exactly like a passive scalar, i.e. upwind specific value times the mass flux.
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

  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  // 0 is actual scratch (tiny); 1 is HBM. EOS E-F8 / dual_energy D8 (2026-10-07): on a GPU a level-0 request above the
  // device limit makes Kokkos abort ("could not find a valid execution configuration", 1D meshblocks >~ 200-400
  // cells); fall back to level 1 for that call. The values are identical, only their memory location changes;
  // production 32^3 blocks are far below the limit, so the selected level is unchanged there.
  int scratch_level = pkg->Param<int>("scratch_level");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if (scratch_level == 0 &&
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 3 >
          static_cast<size_t>(Kokkos::TeamPolicy<parthenon::DevExecSpace>::scratch_size_max(0)))
    scratch_level = 1;
#endif

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

  // Unsplit diffusion: the Ohmic/ambipolar fluxes ride on the hyperbolic fluxes of every stage
  // under the strict explicit dt. With RKL2 they are applied by the super-step tasks instead.
  if (pkg->Param<DiffInt>("diffint") == DiffInt::unsplit) {
    CalcDiffFluxes(pkg.get(), md.get(), DiffTermSet::all);
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
            {1, 1, 1, RowTile(ib.e + 1 - ib.s)}),
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

