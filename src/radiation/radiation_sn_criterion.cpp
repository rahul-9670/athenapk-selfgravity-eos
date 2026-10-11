//========================================================================================
// AthenaPK - Sn transport: convergence norm and LOCAL convergence criterion (R-F6, validation/radiation_sn R4,
// 2026-10-08). Kernels only; the Krylov-state wrappers live in radiation_sn.cpp (KrylovState is private there).
//
// R-F6(a). The legacy denominator is SnResidual's sum |rhs(x)| with rhs = I^n/(chat dt) + eta + chi_s J(x)
// + sum_d (|Omega_d|/dx_d) x_upwind: it moves x-DEPENDENT operator terms (effective scattering, streaming inflow) into
// the "right-hand side". Where those dominate (Fleck-stiff cells, chat dt >> dx) the norm is ~(chi_s chat dt) or
// ~(chat dt/dx) times the per-step radiation change, so sn_rel_tol stops admitting a per-step energy error of that
// size; and it is evaluated at x0 for the recurrence but at the converged x for the audit, which made the trust audit
// compare two different denominators (R-F2: 100x false alarm when J changes 100x in a step).
// radiation/sn_norm = b uses the TRUE right-hand side of A x = b, b = I^n/(chat dt) + eta: fixed during the solve,
// independent of x, and the scale of the radiation the step actually carries.
//
// R-F6(b). One global L1 sum is dominated by the brightest cells; dim cells can be far from converged relative to their
// own radiation (a 3-D collapse restart at 1e-8: per-cell energy defect p99 8 %, max 3x E_rad per step). radiation/sn_local_tol > 0
// additionally requires, in EVERY cell and for the group being solved,
//     | sum_a w_a r_a |  <=  sn_local_tol * sum_g' sum_a w_a b_{g',a}      (r = b - A x, the recurrence residual)
// i.e. the step's energy defect in the cell below sn_local_tol of the radiation energy (incl. this step's emission)
// the cell carries, summed over all groups (so an empty group is measured against the cell's whole radiation).
//========================================================================================
#include <algorithm>
#include <cmath>
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

//! sum over interior cells and ordinates of |I^n/(chat dt) + eta_g| for group g (g < 0: all groups) into *bsum; for
//! g >= 0 also sum |k_r| (the initial residual r0 = b - A x0 already formed by SnApplyA mode 1) into *rsum.
TaskStatus SnBnormB(MeshData<Real> *md, const Real dt, const int g, Real *bsum, Real *rsum) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const Real rdt = 1.0 / (pkg->Param<Real>("chat") * dt);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{UnName(), SrcName()};
  const bool want_r = (g >= 0 && rsum != nullptr);   // k_r only exists on the per-group path
  if (want_r) names.push_back(KrName());
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iun = imap[UnName()].first, isrc = imap[SrcName()].first;
  const int ikr = want_r ? imap[KrName()].first : 0;
  const int nang = p.nang;
  const int glo = (g < 0) ? 0 : g, ghi = (g < 0) ? p.n_group - 1 : g;
  Real lb = 0.0, lr = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::BnormB", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &sb, Real &sr) {
        for (int gg = glo; gg <= ghi; ++gg) {
          const Real eta = pack(b, isrc + gg, k, j, i);
          for (int a = 0; a < nang; ++a) sb += std::abs(rdt * pack(b, iun + gg * nang + a, k, j, i) + eta);
        }
        if (want_r)
          for (int a = 0; a < nang; ++a) sr += std::abs(pack(b, ikr + a, k, j, i));
      },
      Kokkos::Sum<Real>(lb), Kokkos::Sum<Real>(lr));
  *bsum = lb;
  if (rsum != nullptr) *rsum = lr;
  return TaskStatus::complete;
}

//! Number of interior cells violating the local criterion for group g (residual vector in k_r).
TaskStatus SnLocalViolations(MeshData<Real> *md, const Real dt, const int g, const Real tol, Real *count) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real rdt = 1.0 / (pkg->Param<Real>("chat") * dt);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{UnName(), SrcName(), KrName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iun = imap[UnName()].first, isrc = imap[SrcName()].first, ikr = imap[KrName()].first;
  const int nang = p.nang, n_group = p.n_group;
  auto wv = ang.w;
  int lc = 0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::LocalViolations", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &c) {
        Real scale = 0.0;
        for (int gg = 0; gg < n_group; ++gg) {
          const Real eta = pack(b, isrc + gg, k, j, i);
          for (int a = 0; a < nang; ++a) scale += wv(a) * (rdt * pack(b, iun + gg * nang + a, k, j, i) + eta);
        }
        Real m = 0.0;
        for (int a = 0; a < nang; ++a) m += wv(a) * pack(b, ikr + a, k, j, i);
        if (std::abs(m) > tol * std::abs(scale)) c += 1;
      },
      Kokkos::Sum<int>(lc));
  *count = static_cast<Real>(lc);
  (void)g;
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
