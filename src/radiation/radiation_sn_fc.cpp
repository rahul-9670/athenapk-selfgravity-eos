//========================================================================================
// AthenaPK - Sn transport: Fleck & Cummings multigroup effective scattering (R-F3 / audit P-07,
// validation/radiation_sn, 2026-10-08). Kernels; the coupled BiCGSTAB task list is AddSnKrylovFC in
// radiation_sn.cpp.
//
// THE DEFECT (R-F3). The legacy split returns the Fleck remainder (1-f) chi_g J_g to the SAME group g as effective
// scattering. Fleck & Cummings (1971) re-emit the absorbed (1-f) sum_h chi_h J_h with the emission spectrum
// w_g = chi_g B_g / sum_h chi_h B_h. With one Planck-weighted f and S >> 1 the legacy split suppresses the emission of
// a low-opacity group by ~f (0-D, production dt, v9 table: far-IR group 25 % off after one step, 0.8 % after four;
// Fleck & Cummings 0.4 % -> 5e-13).
//
// THE OPERATOR. With A_g the legacy per-group operator (extinction chi_a,eff + chi_s, and -chi_s J_g with chi_s
// containing (1-f) chi_g), the Fleck & Cummings operator is A_g + C_g with the isotropic cross-group term
//     C_g(x) = (1-f) [ chi_g J_g(x) - w_g X(x) ],   X(x) = sum_h chi_h J_h(x),
// chi_g = chi_a,eff,g / f (sn.chia stores f chi_a), w_g = eta_g / sum_h eta_h (eta_g = f chi_g chat B_g / 4 pi).
// sum_g C_g = 0 exactly: the redistribution conserves energy, and the matter coupling (the f part) is unchanged.
// The groups are coupled, so all of them are solved in ONE BiCGSTAB on nfa-wide work vectors (sn.kc_*).
//========================================================================================
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "radiation.hpp"
#include "radiation_groups.hpp"   // MAX_GROUP
#include "radiation_sn.hpp"
#include "radiation_sn_angles.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::MeshData;
using parthenon::Real;

namespace Radiation {
namespace Sn {

TaskStatus SnResidual(MeshData<Real> *md, const Real dt, Real *res, Real *nrm, const std::string &store,
                      Real *tmax, const int gsel);

//! yn(g,a) += sign * C_g(xn) over the interior, for all groups; xn, yn nfa-wide (sn.u or sn.kc_*).
TaskStatus SnFCCross(MeshData<Real> *md, const std::string &xn, const std::string &yn, const Real sign) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{SrcName(), ChiaName(), FleckName()};
  for (const auto &nm : {xn, yn})
    if (std::find(names.begin(), names.end(), nm) == names.end()) names.push_back(nm);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ix = imap[xn].first, iy = imap[yn].first;
  const int isrc = imap[SrcName()].first, ichia = imap[ChiaName()].first, ifl = imap[FleckName()].first;
  const int nang = p.nang, n_group = p.n_group;
  auto wv = ang.w;
  const Real fourpi_inv = 1.0 / (4.0 * M_PI);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::FCCross", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real f = pack(b, ifl, k, j, i);
        if (!(f > 0.0) || !(f < 1.0)) return;   // f = 1: no Fleck remainder, nothing to redistribute
        Real etas = 0.0, X = 0.0;
        Real Jg[MAX_GROUP], chig[MAX_GROUP];
        for (int g = 0; g < n_group; ++g) {
          Real J = 0.0;
          for (int a = 0; a < nang; ++a) J += wv(a) * pack(b, ix + g * nang + a, k, j, i);
          Jg[g] = J * fourpi_inv;
          chig[g] = std::max(0.0, pack(b, ichia + g, k, j, i)) / f;
          X += chig[g] * Jg[g];
          etas += std::max(0.0, pack(b, isrc + g, k, j, i));
        }
        for (int g = 0; g < n_group; ++g) {
          const Real w = (etas > 0.0) ? std::max(0.0, pack(b, isrc + g, k, j, i)) / etas : 0.0;
          const Real c = sign * (1.0 - f) * (chig[g] * Jg[g] - w * X);
          for (int a = 0; a < nang; ++a) pack(b, iy + g * nang + a, k, j, i) += c;
        }
      });
  return TaskStatus::complete;
}

//! TRUE Fleck & Cummings residual sum |A_FC x - b| of sn.u over all groups (SnResidual's per-angle legacy residual
//! stored in `scratch` (nfa-wide), plus C(x)); *res accumulates. Needs x's ghosts and current sn.mom, like SnResidual.
TaskStatus SnFCTrueResidual(MeshData<Real> *md, const Real dt, const std::string &scratch, Real *res) {
  Real r_leg = 0.0, nrm = 0.0;
  SnResidual(md, dt, &r_leg, &nrm, scratch, nullptr, -1);
  SnFCCross(md, USolName(), scratch, 1.0);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{scratch}, imap);
  const int is = imap[scratch].first;
  const int nc = imap[scratch].second - is + 1;
  Real la = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::FCTrue", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nc - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &a) {
        a += std::abs(pack(b, is + n, k, j, i));
      },
      Kokkos::Sum<Real>(la));
  *res += la;
  return TaskStatus::complete;
}

//! R-F16: SECOND transcription of the cross-group term, for SnFCVerifyOp only. Written from the formula in the header
//! of this file and reading J_g from sn.mom (SnComputeMoments, a separate kernel) instead of re-integrating sn.u as
//! SnFCCross does: yn(g,a) += sign * (1-f) [chi_g J_g - w_g sum_h chi_h J_h]. Also returns max |sum_g C_g| (must be 0:
//! the redistribution conserves energy) and the TERM scale max (1-f) X over the interior (C_g is a difference of terms
//! <= (1-f) X, so near equilibrium |C_g| << X and round-off must be judged against X, not against |C_g|).
//! Valid only while sn.mom is current with sn.u.
TaskStatus SnFCCrossMom(MeshData<Real> *md, const std::string &yn, const Real sign, Real *csum, Real *cscale) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{MomName(), SrcName(), ChiaName(), FleckName(), yn};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int imom = imap[MomName()].first, isrc = imap[SrcName()].first, ichia = imap[ChiaName()].first;
  const int ifl = imap[FleckName()].first, iy = imap[yn].first;
  const int nang = p.nang, n_group = p.n_group;
  Real lsum = 0.0, lmax = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::FCCrossMom", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &s, Real &m) {
        const Real f = pack(b, ifl, k, j, i);
        if (!(f > 0.0) || !(f < 1.0)) return;
        // emission spectrum w_g = eta_g / sum_h eta_h and X = sum_h chi_h J_h, chi_h = (f chi_h) / f
        Real eta_sum = 0.0, X = 0.0;
        for (int h = 0; h < n_group; ++h) {
          eta_sum += std::max(0.0, pack(b, isrc + h, k, j, i));
          X += (std::max(0.0, pack(b, ichia + h, k, j, i)) / f) * pack(b, imom + 4 * h, k, j, i);
        }
        Real total = 0.0;
        m = std::max(m, (1.0 - f) * std::abs(X));
        for (int g = 0; g < n_group; ++g) {
          const Real chi = std::max(0.0, pack(b, ichia + g, k, j, i)) / f;
          const Real wg = (eta_sum > 0.0) ? std::max(0.0, pack(b, isrc + g, k, j, i)) / eta_sum : 0.0;
          const Real c = (1.0 - f) * (chi * pack(b, imom + 4 * g, k, j, i) - wg * X);
          total += c;
          for (int a = 0; a < nang; ++a) pack(b, iy + g * nang + a, k, j, i) += sign * c;
        }
        s = std::max(s, std::abs(total));
      },
      Kokkos::Max<Real>(lsum), Kokkos::Max<Real>(lmax));
  *csum = lsum;
  *cscale = lmax;
  return TaskStatus::complete;
}

//! Local criterion on an nfa-wide residual: cells where |sum_g sum_a w r| > tol * |sum_g sum_a w b|.
TaskStatus SnLocalViolationsWide(MeshData<Real> *md, const Real dt, const std::string &rn, const Real tol,
                                 Real *count) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real rdt = 1.0 / (pkg->Param<Real>("chat") * dt);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{UnName(), SrcName(), rn};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iun = imap[UnName()].first, isrc = imap[SrcName()].first, ir = imap[rn].first;
  const int nang = p.nang, n_group = p.n_group;
  auto wv = ang.w;
  int lc = 0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::LocalViolationsWide", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &c) {
        Real scale = 0.0, m = 0.0;
        for (int g = 0; g < n_group; ++g) {
          const Real eta = pack(b, isrc + g, k, j, i);
          for (int a = 0; a < nang; ++a) {
            scale += wv(a) * (rdt * pack(b, iun + g * nang + a, k, j, i) + eta);
            m += wv(a) * pack(b, ir + g * nang + a, k, j, i);
          }
        }
        if (std::abs(m) > tol * std::abs(scale)) c += 1;
      },
      Kokkos::Sum<int>(lc));
  *count = static_cast<Real>(lc);
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
