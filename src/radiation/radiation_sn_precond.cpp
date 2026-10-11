//========================================================================================
// AthenaPK - Sn transport: SOU-CONSISTENT preconditioner sweep (R-F12, validation/radiation_sn R5, 2026-10-08).
// Key radiation/sn_precond_stencil = sou (default step = SnPrecondSweep in radiation_sn.cpp, untouched: the dispatch
// is a host-side early return, so no existing lambda changes codegen). Needs sn_spatial = sou.
//
// PROBLEM. The BiCGSTAB preconditioner M is ONE ordered transport sweep. It is built from the STEP stencil
// (SnUpdateCell: one upwind value per direction) while the operator A is SOU (two upwind values, coefficients
// (1 + k/2, -(1 + k), k/2)). For pure streaming M_step^-1 A_sou is clustered (1.5 - 0.5 e^{-ik}), but in optically thick,
// Fleck-scattering-dominated cells -- a hot dense core at the production dt -- the mismatch is large: R5 (1-D hot core,
// 1e-11 g/cm^3, 1000 K, tau_cell ~ 30) needs ~1700-1900 iterations per step with SOU and the adopted keys vs ~270 with
// step (the same keys, i.e. a CONSISTENT step preconditioner), and after ~2000-3000 iterations BiCGSTAB loses its
// recurrence (true vs recurrence residual 8e-6 vs 5e-9 -> trust-audit abort) or diverges (rel 4e9 -> NaN).
//
// FIX. Sweep with the operator's OWN stencil: in sweep order both upwind values of a cell are already updated, so the
// SOU (or the R-F10 limited-SOU) transport + collision operator is lower-triangular and the sweep inverts it exactly
// (block-locally; zero upwind ghosts as in the step sweep, scattering lagged to the Krylov as before):
//     y_i = [ r_i + sum_d a_d ((1 + q0/2 + q1/2) y_up1 - (q1/2) y_up2) ] / (rdt + chi_a + chi_s + sum_d a_d (1 + q0/2)),
// q0 = phi of the cell's outgoing face, q1 = phi of the upwind cell's outgoing face, from the lag field sn.un exactly as
// in SnApplyASouLim (limiter on) or q0 = q1 = kappa (limiter off: then this is the SOU stencil of SnSouRhs/Diag).
// M stays a FIXED LINEAR operator (no fixup; it may produce negative y, which is fine for a preconditioner).
//========================================================================================
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

namespace {
//! Same face coefficient as radiation_sn_soulim.cpp SnSouPhi (copied, not shared, so neither TU's codegen changes).
KOKKOS_INLINE_FUNCTION Real PcPhi(const Real kap, const Real lc_, const Real lu_) {
  const Real lc = lc_ > 0.0 ? lc_ : 0.0;
  const Real lu = lu_ > 0.0 ? lu_ : 0.0;
  if (!(lu > lc)) return kap;
  const Real lim = 2.0 * lc / (lu - lc);
  return lim < kap ? lim : kap;
}
inline int PcWidth(parthenon::PackIndexMap &imap, const std::string &nm) {
  return imap[nm].second - imap[nm].first + 1;
}
inline int PcBase(parthenon::PackIndexMap &imap, const std::string &nm, const int g, const int nang) {
  return imap[nm].first + ((PcWidth(imap, nm) > nang) ? g * nang : 0);
}
} // namespace

TaskStatus SnPrecondSweepSou(MeshData<Real> *md, const Real dt, const int g, const std::string &rn,
                             const std::string &yn) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = (pmb->pmy_mesh->ndim >= 2);
  const bool three_d = (pmb->pmy_mesh->ndim >= 3);
  std::vector<std::string> names{rn, yn, ChiaName(), ChisName(), UnName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  const int ir = PcBase(imap, rn, g, nang);
  const int iy = PcBase(imap, yn, g, nang);
  const int iun = imap[UnName()].first + g * nang;
  const int ichia = imap[ChiaName()].first + g;
  const int ichis = imap[ChisName()].first + g;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);
  const Real kap = p.sou_kappa;
  const bool lim = p.sou_limit;
  const int is = ib.s, ie = ib.e, js = jb.s, je = jb.e, ks = kb.s, ke = kb.e;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::PrecondSweepSou", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nang - 1,
      KOKKOS_LAMBDA(const int b, const int a) {
        const int di = (nxv(a) > 0.0) ? 1 : -1;
        const int dj = (nyv(a) > 0.0) ? 1 : -1;
        const int dk = (nzv(a) > 0.0) ? 1 : -1;
        const int i0 = (di > 0) ? is : ie, i1 = (di > 0) ? ie : is;
        const int j0 = (dj > 0) ? js : je, j1 = (dj > 0) ? je : js;
        const int k0 = (dk > 0) ? ks : ke, k1 = (dk > 0) ? ke : ks;
        const auto &co = pack.GetCoords(b);
        for (int k = k0;; k += dk) {
          for (int j = j0;; j += dj) {
            for (int i = i0;; i += di) {
              const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
              const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
              const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
              const Real l0 = lim ? pack(b, iun + a, k, j, i) : 0.0;
              Real den = rdt + std::max(0.0, pack(b, ichia, k, j, i)) + std::max(0.0, pack(b, ichis, k, j, i));
              Real num = pack(b, ir + a, k, j, i);
              {
                const Real q0 = lim ? PcPhi(kap, l0, pack(b, iun + a, k, j, i - di)) : kap;
                const Real q1 = lim ? PcPhi(kap, pack(b, iun + a, k, j, i - di), pack(b, iun + a, k, j, i - 2 * di))
                                    : kap;
                den += ax * (1.0 + 0.5 * q0);
                num += ax * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k, j, i - di) -
                             0.5 * q1 * pack(b, iy + a, k, j, i - 2 * di));
              }
              if (two_d) {
                const Real q0 = lim ? PcPhi(kap, l0, pack(b, iun + a, k, j - dj, i)) : kap;
                const Real q1 = lim ? PcPhi(kap, pack(b, iun + a, k, j - dj, i), pack(b, iun + a, k, j - 2 * dj, i))
                                    : kap;
                den += ay * (1.0 + 0.5 * q0);
                num += ay * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k, j - dj, i) -
                             0.5 * q1 * pack(b, iy + a, k, j - 2 * dj, i));
              }
              if (three_d) {
                const Real q0 = lim ? PcPhi(kap, l0, pack(b, iun + a, k - dk, j, i)) : kap;
                const Real q1 = lim ? PcPhi(kap, pack(b, iun + a, k - dk, j, i), pack(b, iun + a, k - 2 * dk, j, i))
                                    : kap;
                den += az * (1.0 + 0.5 * q0);
                num += az * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k - dk, j, i) -
                             0.5 * q1 * pack(b, iy + a, k - 2 * dk, j, i));
              }
              pack(b, iy + a, k, j, i) = num / den;
              if (i == i1) break;
            }
            if (j == j1) break;
          }
          if (k == k1) break;
        }
      });
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
