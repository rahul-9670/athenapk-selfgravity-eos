//========================================================================================
// AthenaPK - Sn transport: residual-exact matter coupling (R-F6, validation/radiation_sn R4, 2026-10-08).
//
// THE DEFECT. The Krylov / Richardson solve stops at ||A x - b||_1 < sn_rel_tol ||b||_1 with ||b||_1 = sum |rhs| and
// rhs = I^n/(chat dt) + eta + chi_s J + sum_d (|Omega_d|/dx_d) I_upwind (SnRhs): the norm CONTAINS THE STREAMING INFLOW.
// Where streaming dominates the diagonal (optically thin cells, chat dt >> dx: chat dt/dx ~ 5e3 in the D7 box) the
// accepted residual is ~rel_tol * (chat dt/dx) of the per-step radiation change, and the gas exchange, which is
// applied in full from the accepted iterate, no longer matches the radiation's own change. Measured: the D7 uniform
// 8^3 relaxation loses 2.4e-3 of e + E_rad by t = 0.4 at rel_tol 1e-8 and 3.6e-8 at 1e-12.
//
// THE IDENTITY. Angle-integrating the accepted discrete row dia x - rhs = r (weights sum to 4 pi) gives, per cell and
// group, E' - E^n = 4 pi dt (eta - chi_a,eff J') - dt sum_a w_a [streaming]_a + dt sum_a w_a r_a. The coupling charges
// the gas -(c/chat) 4 pi dt (eta - chi_a,eff J'), so e + (c/chat) E changes by the (globally cancelling, conservative
// flux form for both step and SOU) streaming term plus (c/chat) dt sum_a w_a r_a. Charging that last term to the gas,
//   e  ->  e - (c/chat) dt sum_g sum_a w_a r_{g,a},
// makes the exchange exactly conservative (round-off) at ANY tolerance; the solve's own error is unchanged.
//
// KEY radiation/sn_conserve_residual (default false: nothing registered, task graph unchanged). Groups the solve
// skipped or restored (audit P-06/P-09: coupling_off) are excluded, as in SnMatterCoupling. Applied AFTER
// SnMatterCoupling with the same internal-energy floor (counted). Step and SOU spatial schemes; WDD is refused.
//========================================================================================
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../eos/eos_table.hpp"
#include "../main.hpp"
#include "radiation.hpp"
#include "radiation_groups.hpp"
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

//! Per cell and group: sum_a w_a (dia x - rhs)_a, the angle-integrated residual of the ACCEPTED iterate, into sn.resm.
//! Same stencil, same ghosts and same order of operations per angle as SnResidual (it runs where SnKrTrue does: after
//! the final exchange of x and after SnComputeMoments).
TaskStatus SnResidualMoment(MeshData<Real> *md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");
  PARTHENON_REQUIRE(!p.diamond, "radiation/sn_conserve_residual: the WDD path is not supported");

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
  if (p.sou_limit) return SnResidualMomentSouLim(md, dt);   // R-F10
  const bool sou = p.sou;
  const Real kap = p.sou_kappa;
  auto wv = ang.w;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::ResidualMoment", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
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
            const Real ux = pack(b, iu + n, k, j, i + di);
            const Real uy = two_d ? pack(b, iu + n, k, j + dj, i) : 0.0;
            const Real uz = three_d ? pack(b, iu + n, k + dk, j, i) : 0.0;
            Real rhs, dia;
            if (sou) {
              const Real ux2 = pack(b, iu + n, k, j, i + 2 * di);
              const Real uy2 = two_d ? pack(b, iu + n, k, j + 2 * dj, i) : 0.0;
              const Real uz2 = three_d ? pack(b, iu + n, k + 2 * dk, j, i) : 0.0;
              rhs = SnSouRhs(pack(b, iun + n, k, j, i), rdt, src, kap, ax, ay, az, ux, uy, uz, ux2, uy2, uz2);
              dia = SnSouDiagonal(rdt, chi_a + chi_s, kap, ax, ay, az);
            } else {
              rhs = SnRhs(pack(b, iun + n, k, j, i), rdt, src, ax, ay, az, ux, uy, uz);
              dia = SnDiagonal(rdt, chi_a + chi_s, ax, ay, az);
            }
            acc += wv(a) * (dia * pack(b, iu + n, k, j, i) - rhs);
          }
          pack(b, irm + g, k, j, i) = acc;
        }
      });
  return TaskStatus::complete;
}

//! e -> e - (c/chat) dt sum_g sn.resm_g over the coupled groups, with SnMatterCoupling's floor (counted, warn-once per
//! rank; APK_SN_FLOOR_EVERY=1 reports every event). Writes IEN and, under sn_dual_energy, the dual component (which
//! SnMatterCoupling has just set to the same internal energy).
TaskStatus SnConserveCorrect(MeshData<Real> *md, const Real dt, std::shared_ptr<std::vector<int>> off) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  if (!p.matter_coupling) return TaskStatus::complete;
  Kokkos::Array<int, MAX_GROUP> goff;
  for (int g = 0; g < MAX_GROUP; ++g)
    goff[g] = (off != nullptr && g < static_cast<int>(off->size())) ? (*off)[g] : 0;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{ResMomName(), "cons"};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int irm = imap[ResMomName()].first;
  const int ic = imap["cons"].first;
  const int n_group = p.n_group;
  const Real c_over_chat = pkg->Param<Real>("c") / pkg->Param<Real>("chat");
  const Real tmin = pkg->Param<Real>("coupling_tmin");
  const Real T_unit = pkg->Param<Real>("T_unit");
  const Real gm1 = pkg->Param<Real>("gamma") - 1.0;
  const bool use_h2 = pkg->Param<bool>("use_h2diss");
  const auto eos_tab = pkg->Param<EOSTable::EosTable>("eos_tab");
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  const int nhydro = hydro_pkg->Param<int>("nhydro");
  const bool mhd = (nhydro > NHYDRO);
  const bool sn_dual = p.dual_energy && hydro_pkg->Param<int>("dual_idx") >= 0;
  const int idual_c = sn_dual ? hydro_pkg->Param<int>("dual_idx") : 0;
  int nfloor = 0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::ConserveCorrect", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &lfloor) {
        const Real rho = pack(b, ic + IDN, k, j, i);
        if (!(rho > 0.0)) return;
        Real corr = 0.0;
        for (int g = 0; g < n_group; ++g)
          if (!goff[g]) corr += pack(b, irm + g, k, j, i);
        corr *= c_over_chat * dt;
        const Real m1 = pack(b, ic + IM1, k, j, i), m2 = pack(b, ic + IM2, k, j, i), m3 = pack(b, ic + IM3, k, j, i);
        Real me = 0.0;
        if (mhd) {
          const Real b1 = pack(b, ic + IB1, k, j, i), b2 = pack(b, ic + IB2, k, j, i), b3 = pack(b, ic + IB3, k, j, i);
          me = 0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
        }
        const Real kin = 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / rho;
        // SnMatterCoupling just wrote IEN = kin + me + e_new (and dual = e_new): recover e_new the same way.
        const Real e_old = sn_dual ? pack(b, ic + idual_c, k, j, i) : pack(b, ic + IEN, k, j, i) - kin - me;
        Real e_new = e_old - corr;
        const Real e_min = use_h2 ? eos_tab.EintFromRhoTk(rho, tmin * T_unit, true) : rho * tmin / gm1;
        if (e_new < e_min) {
          e_new = e_min;
          lfloor += 1;
        }
        pack(b, ic + IEN, k, j, i) += e_new - e_old;
        if (sn_dual) pack(b, ic + idual_c, k, j, i) = e_new;
      },
      Kokkos::Sum<int>(nfloor));
  if (nfloor > 0) {
    static bool warned = false;
    static const bool every = std::getenv("APK_SN_FLOOR_EVERY") != nullptr;
    if (!warned || every) {
      warned = true;
      std::cout << "### WARNING Sn: sn_conserve_residual FLOORED the gas internal energy in " << nfloor
                << " cell(s) on rank " << parthenon::Globals::my_rank << " (warn-once)" << std::endl;
    }
  }
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation
