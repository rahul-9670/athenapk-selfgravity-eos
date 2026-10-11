//========================================================================================
// AthenaPK - mesh-level fused physical boundary conditions (hydro/fused_bc_mesh).
//========================================================================================
//! \file fused_bc_mesh.cpp
//  \brief Diode + Sn incoming-ordinate boundary conditions for EVERY block of a MeshData in six
//  launches (one per face direction), registered as parthenon::MeshDataBoundaryHook().
//
// WHY. Parthenon applies physical BCs block by block and face by face at every halo exchange;
// on GPUs the per-face kernel launches dominate when exchanges are frequent (the Sn Krylov vector
// is exchanged every BiCGSTAB iteration).
//
// WHAT IS IDENTICAL. The values written are those of Hydro::BoundaryFunction::DiodeBC with
// hydro/fused_bc=true: every ghost takes the reference interior cell, the inward normal
// momentum of `cons` is zeroed, and incoming Sn ordinates (sn.u, sn.k_y) are overwritten. The
// faces are applied in Parthenon's order (inner_x1, outer_x1, inner_x2, outer_x2, inner_x3,
// outer_x3) because corner ghosts are rewritten by later faces from cells earlier faces filled;
// blocks are independent (a block's BC reads and writes only that block), so face-major order
// across blocks equals Parthenon's block-major order. Same index ranges as par_for_bndry.
//
// WHEN IT DECLINES (returns false -> stock per-block path): the switch is off; the MeshData is
// not the leaf grid (multigrid levels); not all six domain faces are `diode`; or any FillGhost
// field is not cell-centred or is Fine (e.g. CT's Bf).
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/radiation_sn.hpp"
#include "bvals/boundary_conditions.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

namespace Hydro {
namespace BoundaryFunction {

using namespace parthenon::package::prelude;

namespace {
bool &AllDiode() {
  static bool v = false;
  return v;
}
} // namespace

bool FusedDiodeBCMesh(std::shared_ptr<MeshData<Real>> &md, bool coarse) {
  if (!AllDiode()) return false;
  // Task lists may run on several threads ; the
  // mask buffer below is shared, so one caller at a time.
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (md->NumBlocks() == 0) return true;
  if (md->grid.type() != parthenon::GridType::leaf || md->grid.IsMultigrid()) return false;
  MeshBlock *pmb0 = md->GetBlockData(0)->GetBlockPointer();
  if (!pmb0->packages.Get("Hydro")->Param<bool>("fused_bc_mesh")) return false;
  const int ndim = pmb0->pmy_mesh->ndim;

  // Same fusability rule as the per-block fused path.
  std::vector<std::string> names;
  for (const auto &v : md->GetBlockData(0)->GetVariableVector()) {
    if (!v->IsSet(Metadata::FillGhost)) continue;
    if (!v->IsSet(Metadata::Cell) || v->IsSet(Metadata::Fine)) return false;
    names.push_back(v->label());
  }
  if (names.empty()) return true;

  // Per-block physical-face mask. A block takes part in the coarse pass only if it has a coarser
  // neighbour, exactly as ApplyBoundaryConditionsOnCoarseOrFine skips the others.
  const int nb = md->NumBlocks();
  // Reused across calls (no allocation per exchange); released before Kokkos::finalize.
  static Kokkos::View<int *> mask_d;
  static bool hook_set = false;
  if (!hook_set) {
    Kokkos::push_finalize_hook([]() { mask_d = Kokkos::View<int *>(); });
    hook_set = true;
  }
  if (static_cast<int>(mask_d.extent(0)) < nb) mask_d = Kokkos::View<int *>("FusedBCMask", 2 * nb);
  auto mask_h = Kokkos::create_mirror_view(mask_d);
  bool any = false;
  for (int b = 0; b < nb; ++b) {
    MeshBlock *pmb = md->GetBlockData(b)->GetBlockPointer();
    int m = 0;
    if (!(coarse && !pmb->HasCoarserNeighbors())) {
      for (int f = 0; f < 6; ++f) {
        const auto flag = pmb->boundary_flag[f];
        const bool phys = flag != parthenon::BoundaryFlag::block &&
                          flag != parthenon::BoundaryFlag::undef &&
                          flag != parthenon::BoundaryFlag::periodic &&
                          !(ndim < 3 && f >= 4) && !(ndim < 2 && (f == 2 || f == 3));
        if (phys) m |= (1 << f);
      }
    }
    mask_h(b) = m;
    any = any || (m != 0);
  }
  if (!any) return true;
  Kokkos::deep_copy(mask_d, mask_h);

  parthenon::PackIndexMap imap;
  auto q = md->PackVariables(names, imap, coarse);
  const int nv = q.GetDim(4);
  auto rng = [&](const std::string &n, int &lo, int &cnt) {
    const auto &ip = imap[n];
    lo = ip.first < 0 ? -1 : ip.first;
    cnt = ip.first < 0 ? 0 : ip.second - ip.first + 1;
  };
  int ic, nc, iu, nu, iy, ny, icy, ncy;
  rng("cons", ic, nc);
  rng(Radiation::Sn::USolName(), iu, nu);
  rng(Radiation::Sn::KyName(), iy, ny);
  rng(Radiation::Sn::KcyName(), icy, ncy);   // R-F14: the coupled-solve (sn_fc_scatter) Krylov vector, nfa-wide
  int mode = 0, nang = 1;
  Kokkos::View<Real *> ubg;
  Radiation::Sn::DeviceAngles ang;
  auto &all = pmb0->packages.AllPackages();
  auto it = all.find("radiation");
  if (it != all.end() && it->second->AllParams().hasKey("sn_bc_mode")) {
    mode = it->second->Param<int>("sn_bc_mode");
    if (mode != 0) {
      nang = it->second->Param<Radiation::Sn::SnParams>("sn_params").nang;
      ubg = it->second->Param<Kokkos::View<Real *>>("sn_bc_ubg");
      ang = it->second->Param<Radiation::Sn::DeviceAngles>("sn_angles");
    }
  }
  if (mode == 0) nu = ny = ncy = 0;
  auto nxv = ang.nx, nyv = ang.ny, nzv = ang.nz;
  auto mask = mask_d;

  const auto &bounds = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;
  const IndexDomain doms[6] = {IndexDomain::inner_x1, IndexDomain::outer_x1,
                               IndexDomain::inner_x2, IndexDomain::outer_x2,
                               IndexDomain::inner_x3, IndexDomain::outer_x3};
  for (int f = 0; f < 6; ++f) {
    const int dir = f / 2 + 1;
    const bool inner = (f % 2 == 0);
    const auto rI = bounds.GetBoundsI(IndexDomain::interior);
    const auto rJ = bounds.GetBoundsJ(IndexDomain::interior);
    const auto rK = bounds.GetBoundsK(IndexDomain::interior);
    const auto &r = (dir == 1) ? rI : (dir == 2) ? rJ : rK;
    const int ref = inner ? r.s : r.e;
    const auto ib = bounds.GetBoundsI(doms[f]);
    const auto jb = bounds.GetBoundsJ(doms[f]);
    const auto kb = bounds.GetBoundsK(doms[f]);
    const int cmom = (ic >= 0) ? ic + dir : -1; // IM1/IM2/IM3 == dir within cons
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FusedBCMesh", parthenon::DevExecSpace(), 0, nb - 1, 0, nv - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
          if (!(mask(b) & (1 << f))) return;
          Real val = q(b, l, dir == 3 ? ref : k, dir == 2 ? ref : j, dir == 1 ? ref : i);
          if (l == cmom) {
            const bool inward = inner ? (val > 0.0) : (val < 0.0);
            if (inward) val = 0.0;
          }
          if (nu > 0 && l >= iu && l < iu + nu) {
            const int n = l - iu;
            const int a = n % nang;
            const Real nc_ = dir == 1 ? nxv(a) : (dir == 2 ? nyv(a) : nzv(a));
            if (inner ? (nc_ > 0.0) : (nc_ < 0.0)) val = (mode == 2) ? ubg(n / nang) : 0.0;
          } else if (ny > 0 && l >= iy && l < iy + ny && (l - iy) < nang) {
            const int a = l - iy;
            const Real nc_ = dir == 1 ? nxv(a) : (dir == 2 ? nyv(a) : nzv(a));
            if (inner ? (nc_ > 0.0) : (nc_ < 0.0)) val = 0.0;
          } else if (ncy > 0 && l >= icy && l < icy + ncy) {
            const int a = (l - icy) % nang;
            const Real nc_ = dir == 1 ? nxv(a) : (dir == 2 ? nyv(a) : nzv(a));
            if (inner ? (nc_ > 0.0) : (nc_ < 0.0)) val = 0.0;
          }
          q(b, l, k, j, i) = val;
        });
    // Parthenon's per-block path calls the tree's UserBoundaryFunctions right after the face BC
    // (self-gravity's zero/neumann/multipole phi BCs live there). Skipping them was the bug in the
    // first version (not bit-identical: grav.phi differed at 1e-11). Called here
    // per block for THIS face, i.e. in the same (face BC, then user BC) order per block/face as
    // ApplyBoundaryConditionsOnCoarseOrFine -- blocks are independent, so this is equivalent.
    for (int b = 0; b < nb; ++b) {
      if (!(mask_h(b) & (1 << f))) continue;
      auto rc = md->GetBlockData(b);
      MeshBlock *pmb = rc->GetBlockPointer();
      auto &user = pmb->pmy_mesh->forest.GetTreePtr(pmb->loc.tree())->UserBoundaryFunctions;
      for (auto &bnd_func : user[f]) bnd_func(rc, coarse);
    }
  }
  return true;
}

void RegisterFusedDiodeBCMesh(parthenon::ParameterInput *pin) {
  const char *keys[6] = {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"};
  bool all = true;
  // Read, never GetOrAdd: Parthenon errors on two different defaults for one key, and its own
  // default for *_bc is "outflow".
  for (auto k : keys)
    all = all && pin->DoesParameterExist("parthenon/mesh", k) &&
          pin->GetString("parthenon/mesh", k) == "diode";
  AllDiode() = all;
  if (all) parthenon::MeshDataBoundaryHook() = FusedDiodeBCMesh;
}

} // namespace BoundaryFunction
} // namespace Hydro
