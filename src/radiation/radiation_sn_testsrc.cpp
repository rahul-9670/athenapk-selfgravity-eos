//========================================================================================
// AthenaPK - Sn transport: TEST-ONLY volume radiation source (validation/radiation_sn R2, Su-Olson, 2026-10-08).
//
// radiation/sn_test_source = true (default false: nothing registered, task graph unchanged) adds, once per step and
// BEFORE SnSeedSource copies sn.u into the RHS anchor sn.un, the isotropic increment
//     dI = chat * Q * dt_on / (4 pi)      to every ordinate of group 0,
// in the slab |x1| < sn_test_source_x0, weighted by each cell's overlap fraction with the slab (so the injected energy
// is the exact cell average of a top-hat source), while the step start time is below sn_test_source_tend
// (dt_on = min(dt, tend - t)). Energy injected per step and unit volume: Q dt_on * overlap. Operator split, first
// order in time (like the coupling). Time is accumulated per partition from the dt handed to the task list.
//========================================================================================
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "radiation.hpp"
#include "radiation_sn.hpp"
#include "radiation_sn_angles.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::MeshData;
using parthenon::Real;

namespace Radiation {
namespace Sn {

TaskStatus SnTestSource(MeshData<Real> *md, const Real dt, const int partition) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  auto tvec = pkg->Param<std::shared_ptr<std::vector<Real>>>("sn_test_source_time");
  if (static_cast<int>(tvec->size()) <= partition) tvec->resize(partition + 1, 0.0);
  const Real t = (*tvec)[partition];
  (*tvec)[partition] = t + dt;
  const auto defs = pkg->Param<std::vector<SnTestSourceDef>>("sn_test_source_defs");
  const Real chat = pkg->Param<Real>("chat");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const int ndim = pmb->pmy_mesh->ndim;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{USolName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int nang = p.nang;   // group 0 = components [0, nang)
  auto wv = ang.w;
  for (const auto &d : defs) {
    const Real dt_on = std::max(Real(0.0), std::min(dt, d.tend - t));
    if (!(dt_on > 0.0)) continue;
    const int ord = d.ordinate;
    const bool slab = d.slab;
    const Real x0 = d.x0, q = d.q;
    const Real b0 = d.box[0], b1 = d.box[1], b2 = d.box[2], b3 = d.box[3], b4 = d.box[4], b5 = d.box[5];
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Sn::TestSource", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nang - 1, kb.s,
        kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) {
          if (ord >= 0 && a != ord) return;
          const auto &co = pack.GetCoords(b);
          auto ov = [](Real c, Real h, Real lo, Real hi) {
            const Real l = std::max(c - 0.5 * h, lo), u = std::min(c + 0.5 * h, hi);
            return (u > l) ? (u - l) / h : Real(0.0);
          };
          Real frac;
          if (slab) {
            frac = ov(co.Xc<1>(k, j, i), co.Dxc<1>(k, j, i), -x0, x0);
          } else {
            frac = ov(co.Xc<1>(k, j, i), co.Dxc<1>(k, j, i), b0, b1);
            if (ndim >= 2) frac *= ov(co.Xc<2>(k, j, i), co.Dxc<2>(k, j, i), b2, b3);
            if (ndim >= 3) frac *= ov(co.Xc<3>(k, j, i), co.Dxc<3>(k, j, i), b4, b5);
          }
          if (!(frac > 0.0)) return;
          // energy injected per unit volume: q dt_on frac (E = sum_a w_a I_a / chat, sum_a w_a = 4 pi)
          const Real dI = (ord >= 0) ? chat * q * dt_on / wv(a) : chat * q * dt_on / (4.0 * M_PI);
          pack(b, iu + a, k, j, i) += frac * dI;
        });
  }
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
