//========================================================================================
// AthenaPK - Sn transport: LAGGED POSITIVITY LIMITER for second-order upwind (R-F10, validation/radiation_sn
// R3, 2026-10-08). Key radiation/sn_sou_limiter = positive (default none = the unlimited SOU kernels in
// radiation_sn.cpp, untouched: the dispatch is a host-side early return, so no existing lambda changes codegen).
//
// PROBLEM. SOU reconstructs the outgoing face of cell i along each direction as
//     f_i = I_i + (kappa/2) (I_i - I_{i-1})
// (Godunov: no linear second-order scheme is positive). Behind a sharp absorber or at a beam edge
// I_i << I_{i-1} and f_i < 0; the next cell receives negative inflow. R3: min I/I0 -13 % -> -19 % (dx 1/32 -> 1/64),
// transmitted flux behind an opaque column -7.5 % -> -5.2 % of the incident beam -- not converging.
//
// FIX. Replace the uniform kappa by a PER-FACE coefficient phi in [0, kappa]:
//     f_i = I_i + (phi_i/2) (I_i - I_{i-1}),
//     phi_i = kappa                                   if L_i >= L_{i-1}   (f_i >= I_i >= 0 already),
//           = min(kappa, 2 L_i / (L_{i-1} - L_i))     otherwise           (the largest phi with f_i >= 0),
// with L the intensity at the START OF THE STEP (sn.un = I^n, ghosts filled from the exchanged warm start, see
// SnSouLimLag). phi is FIXED during the solve, so A stays a fixed linear operator (BiCGSTAB valid, the step-sweep
// preconditioner unchanged) and the scheme stays CONSERVATIVE (one face value per face, the same on both sides):
//     adx (f_i - f_{i-1}) = adx [ (1 + phi_i/2) I_i - (1 + phi_i/2 + phi_{i-1}/2) I_{i-1} + (phi_{i-1}/2) I_{i-2} ],
// which at phi == kappa is exactly the SOU stencil of SnSouRhs/SnSouDiagonal. Positivity is exact wherever the new
// solution has the shape of I^n (steady beams, shadows, the quasi-static radiation of a collapse at full c); where
// it does not, phi is the I^n value -- the residual negativity is MEASURED (R3), not assumed away. phi = kappa
// wherever the field does not drop by more than a factor 3 (kappa = 1) per cell, so smooth fields -- the diffusion
// limit (gate K4) -- keep full SOU.
// Requires sn_solver = bicgstab (the only solver production uses; the Richardson sweeps are not limited).
//========================================================================================
#include <algorithm>
#include <cmath>
#include <iostream>
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

TaskStatus SnMomentOf(MeshData<Real> *md, const int g, const std::string &xn, const std::string &jn);

namespace {
//! phi of the face out of cell c (value lc) whose upwind cell has value lu (both from the lag field).
KOKKOS_INLINE_FUNCTION Real SnSouPhi(const Real kap, const Real lc_, const Real lu_) {
  const Real lc = lc_ > 0.0 ? lc_ : 0.0;
  const Real lu = lu_ > 0.0 ? lu_ : 0.0;
  if (!(lu > lc)) return kap;
  const Real lim = 2.0 * lc / (lu - lc);
  return lim < kap ? lim : kap;
}
//! Limited SOU diagonal and right-hand side (helper transcription, used by the residual kernels; SnApplyASouLim
//! writes the same operator out by hand so SnKrVerifyOp compares two independent transcriptions).
KOKKOS_INLINE_FUNCTION Real SnSouLimDiag(const Real rdt, const Real chi, const Real ax, const Real ay, const Real az,
                                         const Real px0, const Real py0, const Real pz0) {
  return rdt + chi + ax * (1.0 + 0.5 * px0) + ay * (1.0 + 0.5 * py0) + az * (1.0 + 0.5 * pz0);
}
KOKKOS_INLINE_FUNCTION Real SnSouLimRhs(const Real in_old, const Real rdt, const Real src, const Real ax,
                                        const Real ay, const Real az, const Real ux, const Real uy, const Real uz,
                                        const Real ux2, const Real uy2, const Real uz2, const Real px0,
                                        const Real py0, const Real pz0, const Real px1, const Real py1,
                                        const Real pz1) {
  const Real sx = ax * ((1.0 + 0.5 * (px0 + px1)) * ux - 0.5 * px1 * ux2);
  const Real sy = ay * ((1.0 + 0.5 * (py0 + py1)) * uy - 0.5 * py1 * uy2);
  const Real sz = az * ((1.0 + 0.5 * (pz0 + pz1)) * uz - 0.5 * pz1 * uz2);
  return in_old * rdt + src + sx + sy + sz;
}
inline int Width(parthenon::PackIndexMap &imap, const std::string &nm) {
  return imap[nm].second - imap[nm].first + 1;
}
inline int GBase(parthenon::PackIndexMap &imap, const std::string &nm, const int g, const int nang) {
  return imap[nm].first + ((Width(imap, nm) > nang) ? g * nang : 0);
}
} // namespace

//! Fill the GHOST cells of the lag field sn.un from sn.u right after the solve's boundary exchange, when sn.u is still
//! the warm start x0 = I^n (its interior equals sn.un's; checked here, warn-once if not). sn.un's ghosts are otherwise
//! unused ("read only at the owning cell"), so this costs no memory and no extra exchange. Both sides of every block
//! face then evaluate phi from the same values -> one face value -> conservative.
TaskStatus SnSouLimLag(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ie = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange je = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange ke = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{USolName(), UnName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first, iun = imap[UnName()].first;
  const int is = ib.s, ie_ = ib.e, js = jb.s, je_ = jb.e, ks = kb.s, ke_ = kb.e;
  Real dmax = 0.0, umax = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::SouLimLag", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, p.nfa - 1,
      ke.s, ke.e, je.s, je.e, ie.s, ie.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &sd, Real &su) {
        const bool inside = (i >= is && i <= ie_ && j >= js && j <= je_ && k >= ks && k <= ke_);
        if (!inside) {
          pack(b, iun + n, k, j, i) = pack(b, iu + n, k, j, i);
        } else {
          sd = std::max(sd, std::abs(pack(b, iu + n, k, j, i) - pack(b, iun + n, k, j, i)));
          su = std::max(su, std::abs(pack(b, iun + n, k, j, i)));
        }
      },
      Kokkos::Max<Real>(dmax), Kokkos::Max<Real>(umax));
  if (dmax > 0.0) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::cout << "### WARNING Sn sou_limiter: warm start differs from I^n (max |x0 - I^n| = " << dmax
                << ", max |I^n| = " << umax << ") on rank " << parthenon::Globals::my_rank
                << "; ghost lag values are then x0's, interior ones I^n's (warn-once)" << std::endl;
    }
  }
  return TaskStatus::complete;
}

//! Limited SOU operator, written out by hand (independent of SnSouLimRhs/Diag). Same contract as SnApplyA:
//! mode 0: y = A x; mode 1: y = b - A x; mode 2: y = b.
TaskStatus SnApplyASouLim(MeshData<Real> *md, const Real dt, const int g, const std::string &xn,
                          const std::string &yn, const int mode) {
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
  SnMomentOf(md, g, xn, KjName());
  std::vector<std::string> names{xn, yn, UnName(), SrcName(), ChiaName(), ChisName(), KjName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  const int ix = GBase(imap, xn, g, nang);
  const int iy = GBase(imap, yn, g, nang);
  const int iun = imap[UnName()].first + g * nang;
  const int isrc = imap[SrcName()].first + g;
  const int ichia = imap[ChiaName()].first + g;
  const int ichis = imap[ChisName()].first + g;
  const int ikj = imap[KjName()].first;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);
  const Real kap = p.sou_kappa;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::ApplyA_soulim", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nang - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) {
        const auto &co = pack.GetCoords(b);
        const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
        const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
        const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
        const int di = (nxv(a) > 0.0) ? -1 : 1;
        const int dj = (nyv(a) > 0.0) ? -1 : 1;
        const int dk = (nzv(a) > 0.0) ? -1 : 1;
        const Real l0 = pack(b, iun + a, k, j, i);
        Real Ax = (rdt + std::max(0.0, pack(b, ichia, k, j, i)) + std::max(0.0, pack(b, ichis, k, j, i))) *
                  pack(b, ix + a, k, j, i);
        {
          const Real l1 = pack(b, iun + a, k, j, i + di), l2 = pack(b, iun + a, k, j, i + 2 * di);
          const Real q0 = SnSouPhi(kap, l0, l1), q1 = SnSouPhi(kap, l1, l2);
          Ax += ax * ((1.0 + 0.5 * q0) * pack(b, ix + a, k, j, i) - (1.0 + 0.5 * q0 + 0.5 * q1) *
                      pack(b, ix + a, k, j, i + di) + 0.5 * q1 * pack(b, ix + a, k, j, i + 2 * di));
        }
        if (two_d) {
          const Real l1 = pack(b, iun + a, k, j + dj, i), l2 = pack(b, iun + a, k, j + 2 * dj, i);
          const Real q0 = SnSouPhi(kap, l0, l1), q1 = SnSouPhi(kap, l1, l2);
          Ax += ay * ((1.0 + 0.5 * q0) * pack(b, ix + a, k, j, i) - (1.0 + 0.5 * q0 + 0.5 * q1) *
                      pack(b, ix + a, k, j + dj, i) + 0.5 * q1 * pack(b, ix + a, k, j + 2 * dj, i));
        }
        if (three_d) {
          const Real l1 = pack(b, iun + a, k + dk, j, i), l2 = pack(b, iun + a, k + 2 * dk, j, i);
          const Real q0 = SnSouPhi(kap, l0, l1), q1 = SnSouPhi(kap, l1, l2);
          Ax += az * ((1.0 + 0.5 * q0) * pack(b, ix + a, k, j, i) - (1.0 + 0.5 * q0 + 0.5 * q1) *
                      pack(b, ix + a, k + dk, j, i) + 0.5 * q1 * pack(b, ix + a, k + 2 * dk, j, i));
        }
        Ax -= std::max(0.0, pack(b, ichis, k, j, i)) * pack(b, ikj, k, j, i);
        if (mode == 0) {
          pack(b, iy + a, k, j, i) = Ax;
        } else {
          const Real rhs = pack(b, iun + a, k, j, i) * rdt + pack(b, isrc, k, j, i);
          pack(b, iy + a, k, j, i) = (mode == 1) ? rhs - Ax : rhs;
        }
      });
  return TaskStatus::complete;
}

//! Limited SOU residual with SnResidual's contract (scattering lagged in the source through sn.mom J, optional
//! per-cell store, tmax, group selection).
TaskStatus SnResidualSouLim(MeshData<Real> *md, const Real dt, Real *res, Real *nrm, const std::string &store,
                            Real *tmax, const int gsel) {
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
  std::vector<std::string> names{USolName(), UnName(), SrcName(), ChiaName(), ChisName(), MomName()};
  if (!store.empty()) names.push_back(store);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int iun = imap[UnName()].first;
  const int isrc = imap[SrcName()].first;
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int imom = imap[MomName()].first;
  const int ist = store.empty() ? -1 : imap[store].first;
  const int nang = p.nang;
  const int nfa = p.nfa;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);
  const Real kap = p.sou_kappa;
  const int nlo = (gsel < 0) ? 0 : gsel * nang;
  const int nhi = (gsel < 0) ? nfa - 1 : gsel * nang + nang - 1;
  Real lres = 0.0, lnrm = 0.0, ltmax = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::Residual_soulim", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, nlo, nhi,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &sres, Real &snrm,
                    Real &stmax) {
        const int g = n / nang;
        const int a = n - g * nang;
        const auto &co = pack.GetCoords(b);
        const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
        const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
        const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
        const int di = (nxv(a) > 0.0) ? -1 : 1;
        const int dj = (nyv(a) > 0.0) ? -1 : 1;
        const int dk = (nzv(a) > 0.0) ? -1 : 1;
        const Real l0 = pack(b, iun + n, k, j, i);
        const Real lx1 = pack(b, iun + n, k, j, i + di), lx2 = pack(b, iun + n, k, j, i + 2 * di);
        const Real ly1 = two_d ? pack(b, iun + n, k, j + dj, i) : 0.0;
        const Real ly2 = two_d ? pack(b, iun + n, k, j + 2 * dj, i) : 0.0;
        const Real lz1 = three_d ? pack(b, iun + n, k + dk, j, i) : 0.0;
        const Real lz2 = three_d ? pack(b, iun + n, k + 2 * dk, j, i) : 0.0;
        const Real px0 = SnSouPhi(kap, l0, lx1), px1 = SnSouPhi(kap, lx1, lx2);
        const Real py0 = SnSouPhi(kap, l0, ly1), py1 = SnSouPhi(kap, ly1, ly2);
        const Real pz0 = SnSouPhi(kap, l0, lz1), pz1 = SnSouPhi(kap, lz1, lz2);
        const Real ux = pack(b, iu + n, k, j, i + di), ux2 = pack(b, iu + n, k, j, i + 2 * di);
        const Real uy = two_d ? pack(b, iu + n, k, j + dj, i) : 0.0;
        const Real uy2 = two_d ? pack(b, iu + n, k, j + 2 * dj, i) : 0.0;
        const Real uz = three_d ? pack(b, iu + n, k + dk, j, i) : 0.0;
        const Real uz2 = three_d ? pack(b, iu + n, k + 2 * dk, j, i) : 0.0;
        const Real chi_a = std::max(0.0, pack(b, ichia + g, k, j, i));
        const Real chi_s = std::max(0.0, pack(b, ichis + g, k, j, i));
        const Real src = pack(b, isrc + g, k, j, i) + chi_s * pack(b, imom + 4 * g + 0, k, j, i);
        const Real rhs = SnSouLimRhs(l0, rdt, src, ax, ay, az, ux, uy, uz, ux2, uy2, uz2, px0, py0, pz0, px1,
                                     py1, pz1);
        const Real dia = SnSouLimDiag(rdt, chi_a + chi_s, ax, ay, az, px0, py0, pz0);
        const Real dx_ = dia * pack(b, iu + n, k, j, i);
        const Real rr = dx_ - rhs;
        if (ist >= 0) pack(b, ist + n - nlo, k, j, i) = rr;
        sres += std::abs(rr);
        snrm += std::abs(rhs);
        stmax = std::max(stmax, std::max(std::abs(dx_), std::abs(rhs)));
      },
      Kokkos::Sum<Real>(lres), Kokkos::Sum<Real>(lnrm), Kokkos::Max<Real>(ltmax));
  *res = lres;
  *nrm = lnrm;
  if (tmax != nullptr) *tmax = ltmax;
  return TaskStatus::complete;
}

//! Limited twin of SnResidualMoment (radiation_sn_conserve.cpp): sn.resm_g = sum_a w_a (dia x - rhs).
TaskStatus SnResidualMomentSouLim(MeshData<Real> *md, const Real dt) {
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
  std::vector<std::string> names{USolName(), UnName(), SrcName(), ChiaName(), ChisName(), MomName(),
                                 ResMomName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int iun = imap[UnName()].first;
  const int isrc = imap[SrcName()].first;
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int imom = imap[MomName()].first;
  const int irm = imap[ResMomName()].first;
  const int nang = p.nang;
  const int n_group = p.n_group;
  const Real kap = p.sou_kappa;
  auto wv = ang.w;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::ResidualMoment_soulim", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &co = pack.GetCoords(b);
        for (int g = 0; g < n_group; ++g) {
          const Real chi_a = std::max(0.0, pack(b, ichia + g, k, j, i));
          const Real chi_s = std::max(0.0, pack(b, ichis + g, k, j, i));
          const Real src = pack(b, isrc + g, k, j, i) + chi_s * pack(b, imom + 4 * g + 0, k, j, i);
          Real acc = 0.0;
          for (int a = 0; a < nang; ++a) {
            const int n = g * nang + a;
            const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
            const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
            const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
            const int di = (nxv(a) > 0.0) ? -1 : 1;
            const int dj = (nyv(a) > 0.0) ? -1 : 1;
            const int dk = (nzv(a) > 0.0) ? -1 : 1;
            const Real l0 = pack(b, iun + n, k, j, i);
            const Real lx1 = pack(b, iun + n, k, j, i + di), lx2 = pack(b, iun + n, k, j, i + 2 * di);
            const Real ly1 = two_d ? pack(b, iun + n, k, j + dj, i) : 0.0;
            const Real ly2 = two_d ? pack(b, iun + n, k, j + 2 * dj, i) : 0.0;
            const Real lz1 = three_d ? pack(b, iun + n, k + dk, j, i) : 0.0;
            const Real lz2 = three_d ? pack(b, iun + n, k + 2 * dk, j, i) : 0.0;
            const Real px0 = SnSouPhi(kap, l0, lx1), px1 = SnSouPhi(kap, lx1, lx2);
            const Real py0 = SnSouPhi(kap, l0, ly1), py1 = SnSouPhi(kap, ly1, ly2);
            const Real pz0 = SnSouPhi(kap, l0, lz1), pz1 = SnSouPhi(kap, lz1, lz2);
            const Real ux = pack(b, iu + n, k, j, i + di), ux2 = pack(b, iu + n, k, j, i + 2 * di);
            const Real uy = two_d ? pack(b, iu + n, k, j + dj, i) : 0.0;
            const Real uy2 = two_d ? pack(b, iu + n, k, j + 2 * dj, i) : 0.0;
            const Real uz = three_d ? pack(b, iu + n, k + dk, j, i) : 0.0;
            const Real uz2 = three_d ? pack(b, iu + n, k + 2 * dk, j, i) : 0.0;
            const Real rhs = SnSouLimRhs(l0, rdt, src, ax, ay, az, ux, uy, uz, ux2, uy2, uz2, px0, py0, pz0,
                                         px1, py1, pz1);
            const Real dia = SnSouLimDiag(rdt, chi_a + chi_s, ax, ay, az, px0, py0, pz0);
            acc += wv(a) * (dia * pack(b, iu + n, k, j, i) - rhs);
          }
          pack(b, irm + g, k, j, i) = acc;
        }
      });
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
