//========================================================================================
// AthenaPK - Sn transport: lean zeroing for the coupled Fleck & Cummings Krylov solve (speed work, 2026-10-09).
// Key radiation/sn_fc_lean_zero = true (default false: SnZeroAll of the whole preconditioned vector sn.kc_y and of the
// operator output before every preconditioner apply, as before). Measured on the GPU production configuration (rank 0):
// those two SnZeroAll calls are 6 % of the GPU time (8.7 s of 140 s over 4 cycles).
// What actually needs to be zero: (a) the GHOST cells of sn.kc_y, which the block-local sweep reads on its upwind faces
// (the interior of every live group is overwritten by the sweep, in sweep order, before it is read), and (b) every cell
// of a FROZEN group's slice (sn_skip_group_rel: its sweep and its operator apply are skipped). The operator output is
// written in the interior by SnApplyA for every live group and only read in the interior (dots, vector updates), so
// only the frozen slices need zeroing there. Same values everywhere they are read -> BIT-IDENTICAL.
//========================================================================================
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "radiation.hpp"
#include "radiation_sn.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::MeshData;
using parthenon::Real;

namespace Radiation {
namespace Sn {

//! Zero the ghost cells (entire minus interior) of every component of field nm.
TaskStatus SnZeroGhosts(MeshData<Real> *md, const std::string &nm) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ie = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange je = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange ke = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{nm}, imap);
  const int i0 = imap[nm].first, nc = imap[nm].second - imap[nm].first + 1;
  const int is = ib.s, ie_ = ib.e, js = jb.s, je_ = jb.e, ks = kb.s, ke_ = kb.e;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::ZeroGhosts", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nc - 1, ke.s, ke.e,
      je.s, je.e, ie.s, ie.e, KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        if (i >= is && i <= ie_ && j >= js && j <= je_ && k >= ks && k <= ke_) return;
        pack(b, i0 + n, k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

//! Zero every cell (ghosts included) of the slices [g*nang, (g+1)*nang) of field nm for the groups with skip[g] != 0.
TaskStatus SnZeroSkippedGroups(MeshData<Real> *md, const std::string &nm, const std::vector<int> &skip, const int nang) {
  bool any = false;
  for (const int s : skip) any = any || (s != 0);
  if (!any) return TaskStatus::complete;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ie = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange je = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange ke = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{nm}, imap);
  const int i0 = imap[nm].first;
  for (std::size_t g = 0; g < skip.size(); ++g) {
    if (!skip[g]) continue;
    const int base = i0 + static_cast<int>(g) * nang;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Sn::ZeroSkippedGroup", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nang - 1,
        ke.s, ke.e, je.s, je.e, ie.s, ie.e,
        KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) { pack(b, base + a, k, j, i) = 0.0; });
  }
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
