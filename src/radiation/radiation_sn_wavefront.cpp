//========================================================================================
// AthenaPK - Sn transport: WAVEFRONT forms of the preconditioner sweeps (speed work, 2026-10-08).
// Key radiation/sn_sweep_wavefront = 0 (default: the serial-per-thread sweeps SnPrecondSweep / SnPrecondSweepSou,
// untouched; the dispatch is a host-side early return) | 1 | 2 | 3 | 4 (= 3 with the x-upwind values in registers).
//
// The ordered sweep gives every (block, ordinate) pair ONE thread that walks the whole block in sweep order, so a 32^3
// block is 32768 dependent cell updates in a row and a GPU sees only nblocks x nang threads. Within one ordinate a cell
// depends only on its upwind neighbours (i - di, j - dj, k - dk, and for SOU i - 2 di, ...). Three schedules that
// respect those dependencies:
//   1 PLANE:  one team per (block, ordinate); hyperplanes ii + jj + kk = s in order, a team barrier between planes,
//             the threads over the cells of a plane.
//   2 OCTANT PLANE: one team per (block, octant); the ordinates of an octant share the sweep order, so the team's
//             threads cover (ordinate, cell of the plane): 4x fewer teams and barriers at 4x8.
//   3 OCTANT PENCIL (KBA-like): one team per (block, octant); 2-D diagonals jj + kk = s of i-PENCILS in order, each
//             thread walks one pencil (ordinate, jj, kk) along i -- contiguous memory per thread.
// Every cell sees exactly the upwind values and executes exactly the arithmetic of the serial sweep (the cell update
// below is the serial kernels' body, verbatim), so the result is BIT-IDENTICAL; only the order of independent cells
// changes. 1-D meshes keep the serial sweep (a plane is one cell there).
//
// GROUP BATCH (radiation/sn_sweep_group_batch, 2026-10-10, 8-group speed work): SnPrecondSweepWave4Groups runs the
// mode-4 schedule for ALL active groups of the coupled solve in ONE launch (league = blocks x octants x groups). One
// group's mode-4 launch has nblocks x 8 teams of <= (ordinates per octant) x 32 threads -- about a quarter of a GPU's
// resident threads at production load -- and its 63 dependent diagonals are latency-bound, so the groups' launches in a
// row cost ~n_group x one. The groups' sweeps are independent (they couple only through the cross term, outside the
// sweep), each team runs WfCellCarry verbatim on its own group's slice: BIT-IDENTICAL to the per-group loop.
//========================================================================================
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "radiation.hpp"
#include "radiation_groups.hpp"   // MAX_GROUP
#include "radiation_sn.hpp"
#include "radiation_sn_angles.hpp"
#include "radiation_sn_sweep.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::MeshData;
using parthenon::Real;

namespace Radiation {
namespace Sn {

namespace {
using team_t = Kokkos::TeamPolicy<parthenon::DevExecSpace>::member_type;
constexpr int kMaxAng = 512;   // ordinates per call for the octant schedules (12x24 = 288, 16x32 = 512)

//! Same face coefficient as radiation_sn_soulim.cpp SnSouPhi (copied, as in radiation_sn_precond.cpp).
KOKKOS_INLINE_FUNCTION Real WfPhi(const Real kap, const Real lc_, const Real lu_) {
  const Real lc = lc_ > 0.0 ? lc_ : 0.0;
  const Real lu = lu_ > 0.0 ? lu_ : 0.0;
  if (!(lu > lc)) return kap;
  const Real lim = 2.0 * lc / (lu - lc);
  return lim < kap ? lim : kap;
}
inline int WfWidth(parthenon::PackIndexMap &imap, const std::string &nm) {
  return imap[nm].second - imap[nm].first + 1;
}
inline int WfBase(parthenon::PackIndexMap &imap, const std::string &nm, const int g, const int nang) {
  return imap[nm].first + ((WfWidth(imap, nm) > nang) ? g * nang : 0);
}

//! Everything a cell update needs, captured once by value.
template <typename Pack, typename AV>
struct WfArgs {
  Pack pack;
  AV nxv, nyv, nzv;
  int ir, iy, iun, ichia, ichis;
  Real rdt, kap;
  bool lim, two_d, three_d, sou;
  int is, ie, js, je, ks, ke;
};

//! One cell, one ordinate: the body of SnPrecondSweep (step) or SnPrecondSweepSou, verbatim.
template <typename A>
KOKKOS_INLINE_FUNCTION void WfCell(const A &w, const int b, const int a, const int di, const int dj, const int dk,
                                   const int k, const int j, const int i) {
  const auto &pack = w.pack;
  const auto &co = pack.GetCoords(b);
  const Real ax = std::abs(w.nxv(a)) / co.template Dxc<1>(k, j, i);
  const Real ay = w.two_d ? std::abs(w.nyv(a)) / co.template Dxc<2>(k, j, i) : 0.0;
  const Real az = w.three_d ? std::abs(w.nzv(a)) / co.template Dxc<3>(k, j, i) : 0.0;
  if (!w.sou) {
    const Real ux = pack(b, w.iy + a, k, j, i - di);
    const Real uy = w.two_d ? pack(b, w.iy + a, k, j - dj, i) : 0.0;
    const Real uz = w.three_d ? pack(b, w.iy + a, k - dk, j, i) : 0.0;
    const Real chi_a = std::max(0.0, pack(b, w.ichia, k, j, i));
    const Real chi_s = std::max(0.0, pack(b, w.ichis, k, j, i));
    pack(b, w.iy + a, k, j, i) =
        SnUpdateCell(0.0, w.rdt, chi_a + chi_s, pack(b, w.ir + a, k, j, i), ax, ay, az, ux, uy, uz);
    return;
  }
  const Real kap = w.kap;
  const bool lim = w.lim;
  const int iun = w.iun, iy = w.iy;
  const Real l0 = lim ? pack(b, iun + a, k, j, i) : 0.0;
  Real den = w.rdt + std::max(0.0, pack(b, w.ichia, k, j, i)) + std::max(0.0, pack(b, w.ichis, k, j, i));
  Real num = pack(b, w.ir + a, k, j, i);
  {
    const Real q0 = lim ? WfPhi(kap, l0, pack(b, iun + a, k, j, i - di)) : kap;
    const Real q1 = lim ? WfPhi(kap, pack(b, iun + a, k, j, i - di), pack(b, iun + a, k, j, i - 2 * di)) : kap;
    den += ax * (1.0 + 0.5 * q0);
    num += ax * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k, j, i - di) - 0.5 * q1 * pack(b, iy + a, k, j, i - 2 * di));
  }
  if (w.two_d) {
    const Real q0 = lim ? WfPhi(kap, l0, pack(b, iun + a, k, j - dj, i)) : kap;
    const Real q1 = lim ? WfPhi(kap, pack(b, iun + a, k, j - dj, i), pack(b, iun + a, k, j - 2 * dj, i)) : kap;
    den += ay * (1.0 + 0.5 * q0);
    num += ay * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k, j - dj, i) - 0.5 * q1 * pack(b, iy + a, k, j - 2 * dj, i));
  }
  if (w.three_d) {
    const Real q0 = lim ? WfPhi(kap, l0, pack(b, iun + a, k - dk, j, i)) : kap;
    const Real q1 = lim ? WfPhi(kap, pack(b, iun + a, k - dk, j, i), pack(b, iun + a, k - 2 * dk, j, i)) : kap;
    den += az * (1.0 + 0.5 * q0);
    num += az * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k - dk, j, i) - 0.5 * q1 * pack(b, iy + a, k - 2 * dk, j, i));
  }
  pack(b, iy + a, k, j, i) = num / den;
}
//! WfCell with the x-upwind values carried in registers along an i-pencil (mode 4): yx1 = y(i - di), yx2 = y(i - 2 di),
//! ux1 = un(i - di), ux2 = un(i - 2 di) on entry; shifted on exit. They hold exactly the values WfCell would load, so the
//! arithmetic (and the result) is identical.
template <typename A>
KOKKOS_INLINE_FUNCTION void WfCellCarry(const A &w, const int b, const int a, const int dj, const int dk, const int k,
                                        const int j, const int i, Real &yx1, Real &yx2, Real &ux1, Real &ux2) {
  const auto &pack = w.pack;
  const auto &co = pack.GetCoords(b);
  const Real ax = std::abs(w.nxv(a)) / co.template Dxc<1>(k, j, i);
  const Real ay = w.two_d ? std::abs(w.nyv(a)) / co.template Dxc<2>(k, j, i) : 0.0;
  const Real az = w.three_d ? std::abs(w.nzv(a)) / co.template Dxc<3>(k, j, i) : 0.0;
  if (!w.sou) {
    const Real ux = yx1;
    const Real uy = w.two_d ? pack(b, w.iy + a, k, j - dj, i) : 0.0;
    const Real uz = w.three_d ? pack(b, w.iy + a, k - dk, j, i) : 0.0;
    const Real chi_a = std::max(0.0, pack(b, w.ichia, k, j, i));
    const Real chi_s = std::max(0.0, pack(b, w.ichis, k, j, i));
    const Real y = SnUpdateCell(0.0, w.rdt, chi_a + chi_s, pack(b, w.ir + a, k, j, i), ax, ay, az, ux, uy, uz);
    pack(b, w.iy + a, k, j, i) = y;
    yx2 = yx1;
    yx1 = y;
    return;
  }
  const Real kap = w.kap;
  const bool lim = w.lim;
  const int iun = w.iun, iy = w.iy;
  const Real l0 = lim ? pack(b, iun + a, k, j, i) : 0.0;
  Real den = w.rdt + std::max(0.0, pack(b, w.ichia, k, j, i)) + std::max(0.0, pack(b, w.ichis, k, j, i));
  Real num = pack(b, w.ir + a, k, j, i);
  {
    const Real q0 = lim ? WfPhi(kap, l0, ux1) : kap;
    const Real q1 = lim ? WfPhi(kap, ux1, ux2) : kap;
    den += ax * (1.0 + 0.5 * q0);
    num += ax * ((1.0 + 0.5 * (q0 + q1)) * yx1 - 0.5 * q1 * yx2);
  }
  if (w.two_d) {
    const Real q0 = lim ? WfPhi(kap, l0, pack(b, iun + a, k, j - dj, i)) : kap;
    const Real q1 = lim ? WfPhi(kap, pack(b, iun + a, k, j - dj, i), pack(b, iun + a, k, j - 2 * dj, i)) : kap;
    den += ay * (1.0 + 0.5 * q0);
    num += ay * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k, j - dj, i) - 0.5 * q1 * pack(b, iy + a, k, j - 2 * dj, i));
  }
  if (w.three_d) {
    const Real q0 = lim ? WfPhi(kap, l0, pack(b, iun + a, k - dk, j, i)) : kap;
    const Real q1 = lim ? WfPhi(kap, pack(b, iun + a, k - dk, j, i), pack(b, iun + a, k - 2 * dk, j, i)) : kap;
    den += az * (1.0 + 0.5 * q0);
    num += az * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k - dk, j, i) - 0.5 * q1 * pack(b, iy + a, k - 2 * dk, j, i));
  }
  const Real y = num / den;
  pack(b, iy + a, k, j, i) = y;
  yx2 = yx1;
  yx1 = y;
  ux2 = ux1;
  ux1 = l0;
}
} // namespace

TaskStatus SnPrecondSweepWave(MeshData<Real> *md, const Real dt, const int g, const std::string &rn,
                              const std::string &yn, const bool sou, const int mode) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{rn, yn, ChiaName(), ChisName()};
  if (sou) names.push_back(UnName());
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  using Pack = decltype(pack);
  using AV = decltype(ang.nx);
  WfArgs<Pack, AV> w{pack, ang.nx, ang.ny, ang.nz, WfBase(imap, rn, g, nang), WfBase(imap, yn, g, nang),
                     sou ? imap[UnName()].first + g * nang : 0, imap[ChiaName()].first + g, imap[ChisName()].first + g,
                     1.0 / (chat * dt), p.sou_kappa, sou && p.sou_limit, pmb->pmy_mesh->ndim >= 2,
                     pmb->pmy_mesh->ndim >= 3, sou, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e};
  const int Ni = ib.e - ib.s + 1, Nj = jb.e - jb.s + 1, Nk = kb.e - kb.s + 1;
  const int nb = pack.GetDim(5);
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  if (mode == 1) {
    const int nplane = Ni + Nj + Nk - 2;
    Kokkos::parallel_for(
        "Sn::PrecondSweepWave1", Kokkos::TeamPolicy<parthenon::DevExecSpace>(nb * nang, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_t &m) {
          const int b = m.league_rank() / nang;
          const int a = m.league_rank() % nang;
          const int di = (nxv(a) > 0.0) ? 1 : -1, dj = (nyv(a) > 0.0) ? 1 : -1, dk = (nzv(a) > 0.0) ? 1 : -1;
          for (int s = 0; s < nplane; ++s) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(m, Nj * Nk), [&](const int q) {
              const int kk = q / Nj, jj = q - kk * Nj, ii = s - kk - jj;
              if (ii < 0 || ii >= Ni) return;
              WfCell(w, b, a, di, dj, dk, (dk > 0) ? w.ks + kk : w.ke - kk, (dj > 0) ? w.js + jj : w.je - jj,
                     (di > 0) ? w.is + ii : w.ie - ii);
            });
            m.team_barrier();
          }
        });
    return TaskStatus::complete;
  }
  // octant schedules: ordinates sorted by octant on the host (nang is small), passed by value
  PARTHENON_REQUIRE(nang <= kMaxAng, "radiation/sn_sweep_wavefront 2/3: more than 512 ordinates");
  // the host copy of the angular grid (the device views are built from the same call at Initialize; no device sync)
  const AngularGrid hg = BuildAngularGrid(p.nzeta, p.npsi);
  const auto &hx = hg.nx, &hy = hg.ny, &hz = hg.nz;
  Kokkos::Array<short, kMaxAng> olist;
  Kokkos::Array<short, 9> ooff;
  {
    int n = 0;
    for (int o = 0; o < 8; ++o) {
      ooff[o] = n;
      for (int a = 0; a < nang; ++a) {
        const int oa = (hx[a] > 0.0 ? 1 : 0) + (hy[a] > 0.0 ? 2 : 0) + (hz[a] > 0.0 ? 4 : 0);
        if (oa == o) olist[n++] = a;
      }
    }
    ooff[8] = n;
  }
  if (mode == 2) {
    const int nplane = Ni + Nj + Nk - 2;
    Kokkos::parallel_for(
        "Sn::PrecondSweepWave2", Kokkos::TeamPolicy<parthenon::DevExecSpace>(nb * 8, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_t &m) {
          const int b = m.league_rank() / 8, o = m.league_rank() % 8;
          const int a0 = ooff[o], na = ooff[o + 1] - a0;
          if (na == 0) return;
          const int di = (o & 1) ? 1 : -1, dj = (o & 2) ? 1 : -1, dk = (o & 4) ? 1 : -1;
          for (int s = 0; s < nplane; ++s) {
            const int klo = Kokkos::max(0, s - (Ni - 1) - (Nj - 1)), khi = Kokkos::min(Nk - 1, s);
            const int nk = khi - klo + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(m, na * nk * Nj), [&](const int q) {
              const int am = q / (nk * Nj), r = q - am * (nk * Nj);
              const int kk = klo + r / Nj, jj = r - (r / Nj) * Nj, ii = s - kk - jj;
              if (ii < 0 || ii >= Ni) return;
              WfCell(w, b, olist[a0 + am], di, dj, dk, (dk > 0) ? w.ks + kk : w.ke - kk,
                     (dj > 0) ? w.js + jj : w.je - jj, (di > 0) ? w.is + ii : w.ie - ii);
            });
            m.team_barrier();
          }
        });
    return TaskStatus::complete;
  }
  // mode 3: pencils along i, 2-D diagonals jj + kk = s (mode 4: the same with the x-upwind values carried in registers)
  const int ndiag = Nj + Nk - 1;
  if (mode == 4) {
    Kokkos::parallel_for(
        "Sn::PrecondSweepWave4", Kokkos::TeamPolicy<parthenon::DevExecSpace>(nb * 8, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_t &m) {
          const int b = m.league_rank() / 8, o = m.league_rank() % 8;
          const int a0 = ooff[o], na = ooff[o + 1] - a0;
          if (na == 0) return;
          const int di = (o & 1) ? 1 : -1, dj = (o & 2) ? 1 : -1, dk = (o & 4) ? 1 : -1;
          for (int s = 0; s < ndiag; ++s) {
            const int klo = Kokkos::max(0, s - (Nj - 1)), khi = Kokkos::min(Nk - 1, s);
            const int nk = khi - klo + 1;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(m, na * nk), [&](const int q) {
              const int am = q / nk, kk = klo + (q - am * nk), jj = s - kk;
              const int a = olist[a0 + am];
              const int k = (dk > 0) ? w.ks + kk : w.ke - kk, j = (dj > 0) ? w.js + jj : w.je - jj;
              const int i0 = (di > 0) ? w.is : w.ie;
              Real yx1 = w.pack(b, w.iy + a, k, j, i0 - di), yx2 = w.pack(b, w.iy + a, k, j, i0 - 2 * di);
              Real ux1 = w.lim ? w.pack(b, w.iun + a, k, j, i0 - di) : 0.0;
              Real ux2 = w.lim ? w.pack(b, w.iun + a, k, j, i0 - 2 * di) : 0.0;
              for (int ii = 0; ii < Ni; ++ii)
                WfCellCarry(w, b, a, dj, dk, k, j, (di > 0) ? w.is + ii : w.ie - ii, yx1, yx2, ux1, ux2);
            });
            m.team_barrier();
          }
        });
    return TaskStatus::complete;
  }
  Kokkos::parallel_for(
      "Sn::PrecondSweepWave3", Kokkos::TeamPolicy<parthenon::DevExecSpace>(nb * 8, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_t &m) {
        const int b = m.league_rank() / 8, o = m.league_rank() % 8;
        const int a0 = ooff[o], na = ooff[o + 1] - a0;
        if (na == 0) return;
        const int di = (o & 1) ? 1 : -1, dj = (o & 2) ? 1 : -1, dk = (o & 4) ? 1 : -1;
        for (int s = 0; s < ndiag; ++s) {
          const int klo = Kokkos::max(0, s - (Nj - 1)), khi = Kokkos::min(Nk - 1, s);
          const int nk = khi - klo + 1;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(m, na * nk), [&](const int q) {
            const int am = q / nk, kk = klo + (q - am * nk), jj = s - kk;
            const int a = olist[a0 + am];
            const int k = (dk > 0) ? w.ks + kk : w.ke - kk, j = (dj > 0) ? w.js + jj : w.je - jj;
            for (int ii = 0; ii < Ni; ++ii) WfCell(w, b, a, di, dj, dk, k, j, (di > 0) ? w.is + ii : w.ie - ii);
          });
          m.team_barrier();
        }
      });
  return TaskStatus::complete;
}

TaskStatus SnPrecondSweepWave4Groups(MeshData<Real> *md, const Real dt, const std::string &rn, const std::string &yn,
                                     const bool sou, const std::vector<int> &groups) {
  if (groups.empty()) return TaskStatus::complete;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{rn, yn, ChiaName(), ChisName()};
  if (sou) names.push_back(UnName());
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  PARTHENON_REQUIRE(nang <= kMaxAng, "radiation/sn_sweep_group_batch: more than 512 ordinates");
  PARTHENON_REQUIRE(static_cast<int>(groups.size()) <= MAX_GROUP, "radiation/sn_sweep_group_batch: too many groups");
  using Pack = decltype(pack);
  using AV = decltype(ang.nx);
  // group-0 bases; a team adds its group's offset exactly as WfBase / SnPrecondSweepWave would for that group
  WfArgs<Pack, AV> w{pack, ang.nx, ang.ny, ang.nz, WfBase(imap, rn, 0, nang), WfBase(imap, yn, 0, nang),
                     sou ? imap[UnName()].first : 0, imap[ChiaName()].first, imap[ChisName()].first,
                     1.0 / (chat * dt), p.sou_kappa, sou && p.sou_limit, pmb->pmy_mesh->ndim >= 2,
                     pmb->pmy_mesh->ndim >= 3, sou, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e};
  const int rstep = (WfWidth(imap, rn) > nang) ? nang : 0, ystep = (WfWidth(imap, yn) > nang) ? nang : 0;
  const int ustep = sou ? nang : 0;
  const int Ni = ib.e - ib.s + 1, Nj = jb.e - jb.s + 1, Nk = kb.e - kb.s + 1;
  const int nb = pack.GetDim(5);
  const AngularGrid hg = BuildAngularGrid(p.nzeta, p.npsi);
  const auto &hx = hg.nx, &hy = hg.ny, &hz = hg.nz;
  Kokkos::Array<short, kMaxAng> olist;
  Kokkos::Array<short, 9> ooff;
  {
    int n = 0;
    for (int o = 0; o < 8; ++o) {
      ooff[o] = n;
      for (int a = 0; a < nang; ++a) {
        const int oa = (hx[a] > 0.0 ? 1 : 0) + (hy[a] > 0.0 ? 2 : 0) + (hz[a] > 0.0 ? 4 : 0);
        if (oa == o) olist[n++] = a;
      }
    }
    ooff[8] = n;
  }
  Kokkos::Array<short, MAX_GROUP> gl;
  const int ngr = static_cast<int>(groups.size());
  for (int q = 0; q < ngr; ++q) gl[q] = static_cast<short>(groups[q]);
  const int ndiag = Nj + Nk - 1;
  Kokkos::parallel_for(
      "Sn::PrecondSweepWave4Groups", Kokkos::TeamPolicy<parthenon::DevExecSpace>(nb * 8 * ngr, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_t &m) {
        const int b = m.league_rank() / (8 * ngr), rem = m.league_rank() - b * (8 * ngr);
        const int g = gl[rem / 8], o = rem % 8;
        const int a0 = ooff[o], na = ooff[o + 1] - a0;
        if (na == 0) return;
        auto wg = w;
        wg.ir += g * rstep;
        wg.iy += g * ystep;
        wg.iun += g * ustep;
        wg.ichia += g;
        wg.ichis += g;
        const int di = (o & 1) ? 1 : -1, dj = (o & 2) ? 1 : -1, dk = (o & 4) ? 1 : -1;
        for (int s = 0; s < ndiag; ++s) {
          const int klo = Kokkos::max(0, s - (Nj - 1)), khi = Kokkos::min(Nk - 1, s);
          const int nk = khi - klo + 1;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(m, na * nk), [&](const int q) {
            const int am = q / nk, kk = klo + (q - am * nk), jj = s - kk;
            const int a = olist[a0 + am];
            const int k = (dk > 0) ? wg.ks + kk : wg.ke - kk, j = (dj > 0) ? wg.js + jj : wg.je - jj;
            const int i0 = (di > 0) ? wg.is : wg.ie;
            Real yx1 = wg.pack(b, wg.iy + a, k, j, i0 - di), yx2 = wg.pack(b, wg.iy + a, k, j, i0 - 2 * di);
            Real ux1 = wg.lim ? wg.pack(b, wg.iun + a, k, j, i0 - di) : 0.0;
            Real ux2 = wg.lim ? wg.pack(b, wg.iun + a, k, j, i0 - 2 * di) : 0.0;
            for (int ii = 0; ii < Ni; ++ii)
              WfCellCarry(wg, b, a, dj, dk, k, j, (di > 0) ? wg.is + ii : wg.ie - ii, yx1, yx2, ux1, ux2);
          });
          m.team_barrier();
        }
      });
  return TaskStatus::complete;
}

//! radiation/sn_sweep_wavefront = 5 (8-group speed work, 2026-10-10): the SOU preconditioner sweep with COALESCED memory
//! access. Mode 4 gives each thread one i-pencil, so a warp's 32 loads hit 32 different rows (measured: ~73 ms per 8-group
//! launch, ~3 TB/s of mostly wasted sector traffic). Here one WARP owns a row (vector lanes = consecutive i): the lanes
//! load everything that does not depend on the row's own unknowns (rhs, opacities, the lagged limiter inputs and the
//! j/k-upwind values of the previous diagonal) coalesced, fold it into  y_i = (F_i + A_i y_{i-1} + B_i y_{i-2}) / D_i  in
//! team scratch, one lane runs the 32-step recurrence, and the lanes write the row back. Same linear map as mode 4, other
//! association of the sums (round-off-level differences; the preconditioner stays a fixed linear operator). The
//! ThreadVectorRange loops end in a warp sync with memory visibility (Kokkos Cuda backend), single(PerThread) syncs after
//! itself; on CPU backends the vector range is serial.
TaskStatus SnPrecondSweepWave5Groups(MeshData<Real> *md, const Real dt, const std::string &rn, const std::string &yn,
                                     const bool sou, const std::vector<int> &groups) {
  if (!sou) return SnPrecondSweepWave4Groups(md, dt, rn, yn, sou, groups);   // mode 5 is the SOU kernel only
  if (groups.empty()) return TaskStatus::complete;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{rn, yn, ChiaName(), ChisName(), UnName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  PARTHENON_REQUIRE(nang <= kMaxAng, "radiation/sn_sweep_wavefront = 5: more than 512 ordinates");
  PARTHENON_REQUIRE(static_cast<int>(groups.size()) <= MAX_GROUP, "radiation/sn_sweep_wavefront = 5: too many groups");
  const int ir0 = WfBase(imap, rn, 0, nang), iy0 = WfBase(imap, yn, 0, nang), iun0 = imap[UnName()].first;
  const int ichia0 = imap[ChiaName()].first, ichis0 = imap[ChisName()].first;
  const int rstep = (WfWidth(imap, rn) > nang) ? nang : 0, ystep = (WfWidth(imap, yn) > nang) ? nang : 0;
  const Real rdt = 1.0 / (chat * dt), kap = p.sou_kappa;
  const bool lim = p.sou_limit, two_d = pmb->pmy_mesh->ndim >= 2, three_d = pmb->pmy_mesh->ndim >= 3;
  const int is = ib.s, ie = ib.e, js = jb.s, je = jb.e, ks = kb.s, ke = kb.e;
  const int Ni = ie - is + 1, Nj = je - js + 1, Nk = ke - ks + 1;
  const int nb = pack.GetDim(5);
  auto nxv = ang.nx, nyv = ang.ny, nzv = ang.nz;
  const AngularGrid hg = BuildAngularGrid(p.nzeta, p.npsi);
  Kokkos::Array<short, kMaxAng> olist;
  Kokkos::Array<short, 9> ooff;
  {
    int n = 0;
    for (int o = 0; o < 8; ++o) {
      ooff[o] = n;
      for (int a = 0; a < nang; ++a) {
        const int oa = (hg.nx[a] > 0.0 ? 1 : 0) + (hg.ny[a] > 0.0 ? 2 : 0) + (hg.nz[a] > 0.0 ? 4 : 0);
        if (oa == o) olist[n++] = a;
      }
    }
    ooff[8] = n;
  }
  Kokkos::Array<short, MAX_GROUP> gl;
  const int ngr = static_cast<int>(groups.size());
  for (int q = 0; q < ngr; ++q) gl[q] = static_cast<short>(groups[q]);
  const int ndiag = Nj + Nk - 1;
  using scratch_t = Kokkos::View<Real *, parthenon::DevExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  const bool scan = p.wf5_scan;   // radiation/sn_wf5_scan: Hillis-Steele scan of the row's affine maps (5 rounds at 32)
  // F, A, B, D, Y (+ two 6-component buffers of affine maps for the scan): one scratch area per team member (= warp)
  const int per_warp = 5 * Ni + (scan ? 12 * Ni : 0);
  // vector lanes = consecutive i on a GPU (one warp per row); host backends run the vector range serially
  constexpr bool host = std::is_same<parthenon::DevExecSpace, Kokkos::DefaultHostExecutionSpace>::value;
  const int vlen = host ? 1 : 32;
  // radiation/sn_wf5_team_size (speed exploration): team members (warps) per team; 0 = Kokkos::AUTO
  const int tsz = host ? 0 : p.wf5_team_size;
  Kokkos::TeamPolicy<parthenon::DevExecSpace> pol = (tsz > 0)
      ? Kokkos::TeamPolicy<parthenon::DevExecSpace>(nb * 8 * ngr, tsz, vlen)
      : Kokkos::TeamPolicy<parthenon::DevExecSpace>(nb * 8 * ngr, Kokkos::AUTO, vlen);
  pol.set_scratch_size(0, Kokkos::PerThread(scratch_t::shmem_size(per_warp)));
  Kokkos::parallel_for(
      "Sn::PrecondSweepWave5Groups", pol, KOKKOS_LAMBDA(const team_t &m) {
        const int b = m.league_rank() / (8 * ngr), rem = m.league_rank() - b * (8 * ngr);
        const int g = gl[rem / 8], o = rem % 8;
        const int a0 = ooff[o], na = ooff[o + 1] - a0;
        if (na == 0) return;
        const int ir = ir0 + g * rstep, iy = iy0 + g * ystep, iun = iun0 + g * nang;
        const int ichia = ichia0 + g, ichis = ichis0 + g;
        const int di = (o & 1) ? 1 : -1, dj = (o & 2) ? 1 : -1, dk = (o & 4) ? 1 : -1;
        scratch_t sc(m.thread_scratch(0), per_warp);
        const auto &co = pack.GetCoords(b);
        for (int s = 0; s < ndiag; ++s) {
          const int klo = Kokkos::max(0, s - (Nj - 1)), khi = Kokkos::min(Nk - 1, s);
          const int nk = khi - klo + 1;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(m, na * nk), [&](const int q) {
            const int am = q / nk, kk = klo + (q - am * nk), jj = s - kk;
            const int a = olist[a0 + am];
            const int k = (dk > 0) ? ks + kk : ke - kk, j = (dj > 0) ? js + jj : je - jj;
            Real *F = &sc(0), *A = F + Ni, *B = A + Ni, *D = B + Ni, *Y = D + Ni;
            const Real nxa = Kokkos::abs(nxv(a)), nya = Kokkos::abs(nyv(a)), nza = Kokkos::abs(nzv(a));
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(m, Ni), [&](const int ii) {
              const int i = (di > 0) ? is + ii : ie - ii;
              const Real ax = nxa / co.template Dxc<1>(k, j, i);
              const Real ay = two_d ? nya / co.template Dxc<2>(k, j, i) : 0.0;
              const Real az = three_d ? nza / co.template Dxc<3>(k, j, i) : 0.0;
              const Real l0 = lim ? pack(b, iun + a, k, j, i) : 0.0;
              Real den = rdt + Kokkos::max(0.0, pack(b, ichia, k, j, i)) + Kokkos::max(0.0, pack(b, ichis, k, j, i));
              Real num = pack(b, ir + a, k, j, i);
              {
                const Real u1 = lim ? pack(b, iun + a, k, j, i - di) : 0.0;
                const Real u2 = lim ? pack(b, iun + a, k, j, i - 2 * di) : 0.0;
                const Real q0 = lim ? WfPhi(kap, l0, u1) : kap;
                const Real q1 = lim ? WfPhi(kap, u1, u2) : kap;
                den += ax * (1.0 + 0.5 * q0);
                A[ii] = ax * (1.0 + 0.5 * (q0 + q1));
                B[ii] = -0.5 * q1 * ax;
              }
              if (two_d) {
                const Real q0 = lim ? WfPhi(kap, l0, pack(b, iun + a, k, j - dj, i)) : kap;
                const Real q1 = lim ? WfPhi(kap, pack(b, iun + a, k, j - dj, i), pack(b, iun + a, k, j - 2 * dj, i)) : kap;
                den += ay * (1.0 + 0.5 * q0);
                num += ay * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k, j - dj, i) -
                             0.5 * q1 * pack(b, iy + a, k, j - 2 * dj, i));
              }
              if (three_d) {
                const Real q0 = lim ? WfPhi(kap, l0, pack(b, iun + a, k - dk, j, i)) : kap;
                const Real q1 = lim ? WfPhi(kap, pack(b, iun + a, k - dk, j, i), pack(b, iun + a, k - 2 * dk, j, i)) : kap;
                den += az * (1.0 + 0.5 * q0);
                num += az * ((1.0 + 0.5 * (q0 + q1)) * pack(b, iy + a, k - dk, j, i) -
                             0.5 * q1 * pack(b, iy + a, k - 2 * dk, j, i));
              }
              // pre-divided here, in parallel, so the serial recurrence below is two FMAs per cell
              const Real rd = 1.0 / den;
              F[ii] = num * rd;
              A[ii] *= rd;
              B[ii] *= rd;
              D[ii] = den;
            });
            const int i0 = (di > 0) ? is : ie;
            if (!scan) {
              Kokkos::single(Kokkos::PerThread(m), [&]() {
                Real y1 = pack(b, iy + a, k, j, i0 - di), y2 = pack(b, iy + a, k, j, i0 - 2 * di);
                for (int ii = 0; ii < Ni; ++ii) {
                  const Real y = F[ii] + A[ii] * y1 + B[ii] * y2;
                  Y[ii] = y;
                  y2 = y1;
                  y1 = y;
                }
              });
            } else {
              // T_i(v) = M_i v + c_i on v = (y_{i-1}, y_{i-2}):  M_i = [[A_i, B_i], [1, 0]], c_i = (F_i, 0).
              // Inclusive scan P_i = T_i o ... o T_0 (later map applied after earlier), then y_i = (P_i v0)[0].
              Real *S0 = Y + Ni, *S1 = S0 + 6 * Ni;   // ping-pong buffers, layout [6][Ni]: m00 m01 m10 m11 c0 c1
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(m, Ni), [&](const int ii) {
                S0[ii] = A[ii]; S0[Ni + ii] = B[ii]; S0[2 * Ni + ii] = 1.0; S0[3 * Ni + ii] = 0.0;
                S0[4 * Ni + ii] = F[ii]; S0[5 * Ni + ii] = 0.0;
              });
              Real *cur = S0, *nxt = S1;
              for (int d = 1; d < Ni; d *= 2) {
                Kokkos::parallel_for(Kokkos::ThreadVectorRange(m, Ni), [&](const int ii) {
                  const Real a00 = cur[ii], a01 = cur[Ni + ii], a10 = cur[2 * Ni + ii], a11 = cur[3 * Ni + ii];
                  const Real ac0 = cur[4 * Ni + ii], ac1 = cur[5 * Ni + ii];
                  if (ii >= d) {   // later (ii) after earlier (ii - d)
                    const int e = ii - d;
                    const Real b00 = cur[e], b01 = cur[Ni + e], b10 = cur[2 * Ni + e], b11 = cur[3 * Ni + e];
                    const Real bc0 = cur[4 * Ni + e], bc1 = cur[5 * Ni + e];
                    nxt[ii] = a00 * b00 + a01 * b10;
                    nxt[Ni + ii] = a00 * b01 + a01 * b11;
                    nxt[2 * Ni + ii] = a10 * b00 + a11 * b10;
                    nxt[3 * Ni + ii] = a10 * b01 + a11 * b11;
                    nxt[4 * Ni + ii] = a00 * bc0 + a01 * bc1 + ac0;
                    nxt[5 * Ni + ii] = a10 * bc0 + a11 * bc1 + ac1;
                  } else {
                    nxt[ii] = a00; nxt[Ni + ii] = a01; nxt[2 * Ni + ii] = a10; nxt[3 * Ni + ii] = a11;
                    nxt[4 * Ni + ii] = ac0; nxt[5 * Ni + ii] = ac1;
                  }
                });
                Real *tmp = cur; cur = nxt; nxt = tmp;
              }
              const Real v0 = pack(b, iy + a, k, j, i0 - di), v1 = pack(b, iy + a, k, j, i0 - 2 * di);
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(m, Ni), [&](const int ii) {
                Y[ii] = cur[ii] * v0 + cur[Ni + ii] * v1 + cur[4 * Ni + ii];
              });
            }
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(m, Ni), [&](const int ii) {
              pack(b, iy + a, k, j, (di > 0) ? is + ii : ie - ii) = Y[ii];
            });
          });
          m.team_barrier();
        }
      });
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
