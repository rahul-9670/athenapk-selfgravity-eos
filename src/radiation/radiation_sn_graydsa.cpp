//========================================================================================
// AthenaPK - Sn transport: gray diffusion synthetic acceleration of the coupled Fleck & Cummings solve (R-F12).
// See radiation_sn_graydsa.hpp for the scheme. The diffusion operator (fluxes, coarse-fine correction, Marshak faces,
// per-level SetDiagonal) is the removed implicit-FLD solver's (radiation_fld.cpp at 1f8850d6) with the flux limiter
// fixed at 1/3, an absorption term, and the chat dt scaling folded into the coefficients.
//========================================================================================
#include "radiation_sn_graydsa.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <parthenon/parthenon.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/internal_prolongation.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>

#include "radiation.hpp"
#include "radiation_groups.hpp"
#include "radiation_sn.hpp"
#include "radiation_sn_angles.hpp"
#include "../main.hpp"

using namespace parthenon;
using namespace parthenon::package::prelude;

namespace Radiation {
namespace Sn {
namespace GrayDsa {

constexpr TopologicalElement te = TopologicalElement::CC;

namespace {
struct BndInfo {
  Real lo[3], hi[3];
  bool rad_lo[3], rad_hi[3];
};
//! Radiating (non-periodic) mesh faces, by mesh extent (a device kernel cannot read per-block boundary flags).
BndInfo GetBndInfo(MeshData<Real> *md) {
  auto *pm = md->GetMeshPointer();
  BndInfo b;
  const CoordinateDirection dirs[3] = {X1DIR, X2DIR, X3DIR};
  const auto &mbf = pm->mesh_bcs;
  for (int d = 0; d < 3; ++d) {
    b.lo[d] = pm->mesh_size.xmin(dirs[d]);
    b.hi[d] = pm->mesh_size.xmax(dirs[d]);
    b.rad_lo[d] = (mbf[2 * d] != BoundaryFlag::periodic && mbf[2 * d] != BoundaryFlag::block);
    b.rad_hi[d] = (mbf[2 * d + 1] != BoundaryFlag::periodic && mbf[2 * d + 1] != BoundaryFlag::block);
  }
  return b;
}
Real ChatDt(MeshData<Real> *md) {
  auto pkg = md->GetMeshPointer()->packages.Get("radiation");
  return pkg->Param<Real>("chat") * pkg->Param<Real>("sn_gd_dt");
}
} // namespace

//----------------------------------------------------------------------------------------
//! Face D (lower face per direction) = harmonic mean of the two cells' chat dt Dbar. One past the upper interior
//! bound so the last interior cell's upper face exists. Needs sn.gd_dc ghosts (exchanged after the load, restricted
//! on the multigrid levels).
TaskStatus ComputeD(MeshData<Real> *md) {
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
  auto desc = MakePackDescriptor<dc, D>(md);
  auto pack = desc.GetPack(md);
  parthenon::par_for(
      "SnGd::ComputeD", 0, pack.GetNBlocks() - 1, kb.s, kb.e + (ndim > 2 ? 1 : 0), jb.s, jb.e + (ndim > 1 ? 1 : 0),
      ib.s, ib.e + 1, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int d = 0; d < 3; ++d) {
          if ((d == 1 && ndim < 2) || (d == 2 && ndim < 3)) {
            pack(b, te, D(d), k, j, i) = 0.0;
            continue;
          }
          const int im = (d == 0) ? i - 1 : i, jm = (d == 1) ? j - 1 : j, km = (d == 2) ? k - 1 : k;
          const Real a = pack(b, te, dc(), km, jm, im), c = pack(b, te, dc(), k, j, i);
          pack(b, te, D(d), k, j, i) = (a > 0.0 && c > 0.0) ? 2.0 * a * c / (a + c) : 0.0;
        }
      });
  return TaskStatus::complete;
}

//! flux = D (u[i-1] - u[i]) / dxc = -D grad u on the face; Marshak vacuum on radiating mesh faces: outgoing
//! min(chat dt / 2, 2 D / dxc) u (the half-range escape of an isotropic field, capped by what the cell can deliver).
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md_mat, std::shared_ptr<MeshData<Real>> &md) {
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
  const BndInfo bi = GetBndInfo(md.get());
  // radiation/sn_gray_dsa_bc: "marshak" (default) escape min(chat dt / 2, 2 D / dxc) eps, or "dirichlet" eps = 0 on
  // the face (flux 2 D eps / dxc): h -> infinity in the same expression.
  const bool dirichlet = md->GetMeshPointer()->packages.Get("radiation")->Param<bool>("sn_gd_dirichlet");
  const Real h = dirichlet ? 1.0e300 : 0.5 * ChatDt(md.get());
  auto desc = MakePackDescriptor<u>(md.get(), {}, {PDOpt::WithFluxes});
  auto desc_m = MakePackDescriptor<D>(md_mat.get());
  auto pack = desc.GetPack(md.get());
  auto packm = desc_m.GetPack(md_mat.get());
  parthenon::par_for(
      "SnGd::CalculateFluxes", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &co = pack.GetCoordinates(b);
        {
          const Real dxc = co.Dxc<X1DIR>(k, j, i);
          const Real Dl = packm(b, te, D(0), k, j, i);
          Real f = Dl * (pack(b, te, u(), k, j, i - 1) - pack(b, te, u(), k, j, i)) / dxc;
          if (bi.rad_lo[0] && std::fabs(co.Xf<1>(i) - bi.lo[0]) < 1.0e-9 * dxc)
            f = -Kokkos::fmin(h, 2.0 * Dl / dxc) * pack(b, te, u(), k, j, i);
          pack.flux(b, X1DIR, u(), k, j, i) = f;
          if (i == ib.e) {
            const Real dxr = co.Dxc<X1DIR>(k, j, i + 1);
            const Real Dr = packm(b, te, D(0), k, j, i + 1);
            Real fr = Dr * (pack(b, te, u(), k, j, i) - pack(b, te, u(), k, j, i + 1)) / dxr;
            if (bi.rad_hi[0] && std::fabs(co.Xf<1>(i + 1) - bi.hi[0]) < 1.0e-9 * dxr)
              fr = Kokkos::fmin(h, 2.0 * Dr / dxr) * pack(b, te, u(), k, j, i);
            pack.flux(b, X1DIR, u(), k, j, i + 1) = fr;
          }
        }
        if (ndim > 1) {
          const Real dyc = co.Dxc<X2DIR>(k, j, i);
          const Real Dl = packm(b, te, D(1), k, j, i);
          Real f = Dl * (pack(b, te, u(), k, j - 1, i) - pack(b, te, u(), k, j, i)) / dyc;
          if (bi.rad_lo[1] && std::fabs(co.Xf<2>(j) - bi.lo[1]) < 1.0e-9 * dyc)
            f = -Kokkos::fmin(h, 2.0 * Dl / dyc) * pack(b, te, u(), k, j, i);
          pack.flux(b, X2DIR, u(), k, j, i) = f;
          if (j == jb.e) {
            const Real dyr = co.Dxc<X2DIR>(k, j + 1, i);
            const Real Dr = packm(b, te, D(1), k, j + 1, i);
            Real fr = Dr * (pack(b, te, u(), k, j, i) - pack(b, te, u(), k, j + 1, i)) / dyr;
            if (bi.rad_hi[1] && std::fabs(co.Xf<2>(j + 1) - bi.hi[1]) < 1.0e-9 * dyr)
              fr = Kokkos::fmin(h, 2.0 * Dr / dyr) * pack(b, te, u(), k, j, i);
            pack.flux(b, X2DIR, u(), k, j + 1, i) = fr;
          }
        }
        if (ndim > 2) {
          const Real dzc = co.Dxc<X3DIR>(k, j, i);
          const Real Dl = packm(b, te, D(2), k, j, i);
          Real f = Dl * (pack(b, te, u(), k - 1, j, i) - pack(b, te, u(), k, j, i)) / dzc;
          if (bi.rad_lo[2] && std::fabs(co.Xf<3>(k) - bi.lo[2]) < 1.0e-9 * dzc)
            f = -Kokkos::fmin(h, 2.0 * Dl / dzc) * pack(b, te, u(), k, j, i);
          pack.flux(b, X3DIR, u(), k, j, i) = f;
          if (k == kb.e) {
            const Real dzr = co.Dxc<X3DIR>(k + 1, j, i);
            const Real Dr = packm(b, te, D(2), k + 1, j, i);
            Real fr = Dr * (pack(b, te, u(), k, j, i) - pack(b, te, u(), k + 1, j, i)) / dzr;
            if (bi.rad_hi[2] && std::fabs(co.Xf<3>(k + 1) - bi.hi[2]) < 1.0e-9 * dzr)
              fr = Kokkos::fmin(h, 2.0 * Dr / dzr) * pack(b, te, u(), k, j, i);
            pack.flux(b, X3DIR, u(), k + 1, j, i) = fr;
          }
        }
      });
  return TaskStatus::complete;
}

//! out = sig u - div(D grad u)   (div of the face fluxes above is +div(D grad u))
TaskStatus ApplyOperator(std::shared_ptr<MeshData<Real>> &md_mat, std::shared_ptr<MeshData<Real>> &md,
                         std::shared_ptr<MeshData<Real>> &md_out) {
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
  auto desc = MakePackDescriptor<u>(md.get(), {}, {PDOpt::WithFluxes});
  auto desc_out = MakePackDescriptor<u>(md_out.get());
  auto desc_s = MakePackDescriptor<sig>(md_mat.get());
  auto pack = desc.GetPack(md.get());
  auto pack_out = desc_out.GetPack(md_out.get());
  auto pack_s = desc_s.GetPack(md_mat.get());
  parthenon::par_for(
      "SnGd::ApplyOperator", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &co = pack.GetCoordinates(b);
        const Real Vol = co.Volume<TopologicalElement::CC>(k, j, i);
        Real div = (pack.flux(b, X1DIR, u(), k, j, i) * co.Volume<TopologicalElement::F1>(k, j, i) -
                    pack.flux(b, X1DIR, u(), k, j, i + 1) * co.Volume<TopologicalElement::F1>(k, j, i + 1)) /
                   Vol;
        if (ndim > 1)
          div += (pack.flux(b, X2DIR, u(), k, j, i) * co.Volume<TopologicalElement::F2>(k, j, i) -
                  pack.flux(b, X2DIR, u(), k, j + 1, i) * co.Volume<TopologicalElement::F2>(k, j + 1, i)) /
                 Vol;
        if (ndim > 2)
          div += (pack.flux(b, X3DIR, u(), k, j, i) * co.Volume<TopologicalElement::F3>(k, j, i) -
                  pack.flux(b, X3DIR, u(), k + 1, j, i) * co.Volume<TopologicalElement::F3>(k + 1, j, i)) /
                 Vol;
        pack_out(b, te, u(), k, j, i) = pack_s(b, te, sig(), k, j, i) * pack(b, te, u(), k, j, i) - div;
      });
  return TaskStatus::complete;
}

TaskID GrayDsaEquation::Ax(TaskList &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md_mat,
                           std::shared_ptr<MeshData<Real>> &md_in, std::shared_ptr<MeshData<Real>> &md_out) {
  auto flx = tl.AddTask(depends_on, CalculateFluxes, md_mat, md_in);
  if (!(md_mat->grid.type() == GridType::two_level_composite)) {   // as PoissonEquation::Ax
    auto start = tl.AddTask(flx, StartReceiveFluxCorrections, md_in);
    auto send = tl.AddTask(flx, LoadAndSendFluxCorrections, md_in);
    auto recv = tl.AddTask(start, ReceiveFluxCorrections, md_in);
    flx = tl.AddTask(recv | send, SetFluxCorrections, md_in);
  }
  return tl.AddTask(flx, ApplyOperator, md_mat, md_in, md_out);
}

//! The per-level hook (coefficients are restricted onto this grid when it runs): rebuild D, then
//! diag = sig + sum_faces D A / (dxc Vol)  (>= 1, diagonally dominant).
TaskStatus GrayDsaEquation::SetDiagonal(std::shared_ptr<MeshData<Real>> &md_mat,
                                        std::shared_ptr<MeshData<Real>> &md_diag) {
  ComputeD(md_mat.get());
  const int ndim = md_mat->GetMeshPointer()->ndim;
  IndexRange ib = md_mat->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md_mat->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md_mat->GetBoundsK(IndexDomain::interior, te);
  auto desc_m = MakePackDescriptor<D, sig>(md_mat.get());
  auto desc_d = MakePackDescriptor<u>(md_diag.get());
  auto pm = desc_m.GetPack(md_mat.get());
  auto pd = desc_d.GetPack(md_diag.get());
  parthenon::par_for(
      "SnGd::SetDiagonal", 0, pd.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &co = pm.GetCoordinates(b);
        const Real Vol = co.Volume<TopologicalElement::CC>(k, j, i);
        Real acc = (pm(b, te, D(0), k, j, i) * co.Volume<TopologicalElement::F1>(k, j, i) +
                    pm(b, te, D(0), k, j, i + 1) * co.Volume<TopologicalElement::F1>(k, j, i + 1)) /
                   (co.Dxc<X1DIR>(k, j, i) * Vol);
        if (ndim > 1)
          acc += (pm(b, te, D(1), k, j, i) * co.Volume<TopologicalElement::F2>(k, j, i) +
                  pm(b, te, D(1), k, j + 1, i) * co.Volume<TopologicalElement::F2>(k, j + 1, i)) /
                 (co.Dxc<X2DIR>(k, j, i) * Vol);
        if (ndim > 2)
          acc += (pm(b, te, D(2), k, j, i) * co.Volume<TopologicalElement::F3>(k, j, i) +
                  pm(b, te, D(2), k + 1, j, i) * co.Volume<TopologicalElement::F3>(k + 1, j, i)) /
                 (co.Dxc<X3DIR>(k, j, i) * Vol);
        pd(b, te, u(), k, j, i) = pm(b, te, sig(), k, j, i) + acc;
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
namespace {
using SkipMask = Kokkos::Array<int, MAX_GROUP>;
SkipMask MaskOf(const std::vector<int> *skip, const int n_group) {
  SkipMask m;
  for (int g = 0; g < MAX_GROUP; ++g) m[g] = (skip != nullptr && g < static_cast<int>(skip->size())) ? (*skip)[g] : 0;
  (void)n_group;
  return m;
}
} // namespace

//! Per cell: xi_g, sig = chat dt sigbar, dc = chat dt Dbar from the step's coefficients (sn.chia = f chi_a,
//! sn.chis = (1 - f) chi_a + kappa_s + Rosseland excess, sn.fleck = f, sn.src = eta_g). Frozen groups: xi = 0.
TaskStatus Load(MeshData<Real> *md, const Real dt, const std::vector<int> *skip) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const Real chat = pkg->Param<Real>("chat");
  const Real rdt = 1.0 / (chat * dt);
  const Real cdt = chat * dt;
  pkg->UpdateParam<Real>("sn_gd_dt", dt);   // the matvec reads chat dt through this (it cannot take arguments)
  const int n_group = p.n_group;
  const SkipMask sk = MaskOf(skip, n_group);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{ChiaName(), ChisName(), FleckName(), SrcName(), XiName(), sig::name(),
                                 dc::name()};
  PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ia = imap[ChiaName()].first, is = imap[ChisName()].first, ifl = imap[FleckName()].first;
  const int isrc = imap[SrcName()].first, ixi = imap[XiName()].first;
  const int isg = imap[sig::name()].first, idc = imap[dc::name()].first;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SnGd::Load", DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real f = pack(b, ifl, k, j, i);
        const Real finv = (f > 0.0) ? 1.0 / f : 1.0;
        Real esum = 0.0;
        int nact = 0;
        for (int g = 0; g < n_group; ++g)
          if (!sk[g]) {
            esum += std::max(0.0, pack(b, isrc + g, k, j, i));
            ++nact;
          }
        Real xs = 0.0;
        for (int g = 0; g < n_group; ++g) {
          Real x = 0.0;
          if (!sk[g]) {
            const Real w = (esum > 0.0) ? std::max(0.0, pack(b, isrc + g, k, j, i)) / esum : 1.0 / nact;
            x = w / (rdt + std::max(0.0, pack(b, ia + g, k, j, i)) * finv);
          }
          pack(b, ixi + g, k, j, i) = x;
          xs += x;
        }
        Real sb = 0.0, db = 0.0;
        for (int g = 0; g < n_group; ++g) {
          const Real x = (xs > 0.0) ? pack(b, ixi + g, k, j, i) / xs : 0.0;
          pack(b, ixi + g, k, j, i) = x;
          const Real ca = std::max(0.0, pack(b, ia + g, k, j, i));
          const Real cs = std::max(0.0, pack(b, is + g, k, j, i));
          sb += (1.0 + cdt * ca) * x;              // chat dt (rdt + chia) xi
          db += x / (3.0 * (ca + cs + rdt));       // Dbar
        }
        pack(b, isg, k, j, i) = (sb > 0.0) ? sb : 1.0;
        pack(b, idc, k, j, i) = cdt * db;
      });
  return TaskStatus::complete;
}

//! rhs (in the solver's rhs container, field sn.gd_u) = chat dt sum_g chis_g phi_g(y), y = sn.kc_y.
TaskStatus Rhs(MeshData<Real> *md, MeshData<Real> *md_rhs, const Real dt, const std::vector<int> *skip) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real cdt = pkg->Param<Real>("chat") * dt;
  const int n_group = p.n_group, nang = p.nang;
  const SkipMask sk = MaskOf(skip, n_group);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{ChisName(), KcyName()}, imap);
  const int is = imap[ChisName()].first, iy = imap[KcyName()].first;
  auto rpack = md_rhs->PackVariables(std::vector<std::string>{u::name()});
  auto wv = ang.w;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SnGd::Rhs", DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real s = 0.0;
        for (int g = 0; g < n_group; ++g) {
          if (sk[g]) continue;
          Real ph = 0.0;
          for (int a = 0; a < nang; ++a) ph += wv(a) * pack(b, iy + g * nang + a, k, j, i);
          s += std::max(0.0, pack(b, is + g, k, j, i)) * ph;
        }
        rpack(b, 0, k, j, i) = cdt * s;
      });
  if (pkg->Param<bool>("sn_gd_debug")) {   // radiation/sn_gray_dsa_debug: keep rhs and the pre-correction y for checking
    auto dpack = md->PackVariables(std::vector<std::string>{"sn.gd_dbg_rhs", "sn.gd_dbg_y0", KcyName()}, imap);
    const int ir = imap["sn.gd_dbg_rhs"].first, iy0 = imap["sn.gd_dbg_y0"].first, iyy = imap[KcyName()].first;
    const int nfa = n_group * nang;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SnGd::Dbg", DevExecSpace(), 0, dpack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          dpack(b, ir, k, j, i) = rpack(b, 0, k, j, i);
          for (int n = 0; n < nfa; ++n) dpack(b, iy0 + n, k, j, i) = dpack(b, iyy + n, k, j, i);
        });
  }
  return TaskStatus::complete;
}

//! y_g,a += xi_g eps / 4pi on every non-skipped group (eps copied back into base sn.gd_u).
TaskStatus Apply(MeshData<Real> *md, const std::vector<int> *skip) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const int n_group = p.n_group, nang = p.nang;
  const SkipMask sk = MaskOf(skip, n_group);
  const Real omega = pkg->Param<Real>("sn_gd_damping");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{KcyName(), XiName(), u::name()}, imap);
  const int iy = imap[KcyName()].first, ixi = imap[XiName()].first, iu = imap[u::name()].first;
  const Real q = omega / (4.0 * M_PI);
  // radiation/sn_gray_dsa_tau_min (default 0 = everywhere): apply the correction only in cells whose gray optical
  // depth tau_cell = dx chibar >= tau_min, chibar = 1 / (3 Dbar) = chat dt / (3 dc). Diffusion is the wrong model of
  // the error in optically thin cells (it streams out); there the sweep is already near-exact.
  const Real tmin = pkg->Param<Real>("sn_gd_tau_min");
  const Real cdt = pkg->Param<Real>("chat") * pkg->Param<Real>("sn_gd_dt");
  PackIndexMap imap2;
  auto cpack = md->PackVariables(std::vector<std::string>{dc::name()}, imap2);
  const int idc = imap2[dc::name()].first;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SnGd::Apply", DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (tmin > 0.0) {
          const Real dx = pack.GetCoords(b).Dxc<1>(k, j, i);
          if (dx * cdt / (3.0 * std::max(cpack(b, idc, k, j, i), Real(1.0e-300))) < tmin) return;
        }
        const Real e = q * pack(b, iu, k, j, i);
        for (int g = 0; g < n_group; ++g) {
          if (sk[g]) continue;
          const Real d = pack(b, ixi + g, k, j, i) * e;
          for (int a = 0; a < nang; ++a) pack(b, iy + g * nang + a, k, j, i) += d;
        }
      });
  return TaskStatus::complete;
}

//! radiation/sn_gray_dsa_debug: copy the preconditioner's input vector (all nfa components, interior) into sn.gd_dbg_in.
TaskStatus DbgCopyIn(MeshData<Real> *md, const std::string &rn) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{rn, "sn.gd_dbg_in"}, imap);
  const int ir = imap[rn].first, io = imap["sn.gd_dbg_in"].first;
  const int n = imap[rn].second - ir + 1;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SnGd::DbgIn", DevExecSpace(), 0, pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int q = 0; q < n; ++q) pack(b, io + q, k, j, i) = pack(b, ir + q, k, j, i);
      });
  return TaskStatus::complete;
}

TaskID AddCorrection(TaskList &tl, TaskID t, std::shared_ptr<MeshData<Real>> &md,
                     std::shared_ptr<MeshData<Real>> &md_ky, const Real dt, const bool ml, Mesh *pmesh,
                     const int partition, const std::vector<int> *skip) {
  auto pkg = pmesh->packages.Get("radiation");
  auto solver = pkg->Param<std::shared_ptr<solvers::SolverBase>>("sn_gd_solver");
  auto partitions = pmesh->GetDefaultBlockPartitions();
  auto &md_u = pmesh->mesh_data.Add("sn_gd_u", partitions[partition], std::vector<std::string>{u::name()});
  auto &md_rhs = pmesh->mesh_data.Add("sn_gd_rhs", partitions[partition], std::vector<std::string>{u::name()});
  auto &md_coef = pmesh->mesh_data.AddShallow("sn_gd_coef", md, std::vector<std::string>{sig::name(), dc::name()});
  t = tl.AddTask(t, Load, md.get(), dt, skip);
  t = AddBoundaryExchangeTasks(t, tl, md_coef, ml);   // sig, dc ghosts for the face D
  t = tl.AddTask(t, Rhs, md.get(), md_rhs.get(), dt, skip);
  t = tl.AddTask(t, ComputeD, md.get());             // finest-level D (SetDiagonal rebuilds it per level)
  if (pkg->Param<bool>("sn_gd_zero_u"))              // fixed-cycle MG: every application starts from zero (linear)
    t = tl.AddTask(t, TF(solvers::utils::SetToZero<TypeList<u>>), md_u);
  t = solver->AddSetupTasks(tl, t, partition, pmesh);
  t = solver->AddTasks(tl, t, partition, pmesh);
  t = tl.AddTask(t, [](StateDescriptor *pk, std::shared_ptr<solvers::SolverBase> sv) {
    pk->UpdateParam<int>("sn_gd_it_sum", pk->Param<int>("sn_gd_it_sum") + sv->GetFinalIterations());
    pk->UpdateParam<int>("sn_gd_calls", pk->Param<int>("sn_gd_calls") + 1);
    if (!sv->GetConverged()) pk->UpdateParam<int>("sn_gd_nonconv", pk->Param<int>("sn_gd_nonconv") + 1);
    return TaskStatus::complete;
  }, pkg.get(), solver);
  t = tl.AddTask(t, TF(solvers::utils::CopyData<TypeList<u>>), md_u, md);
  t = tl.AddTask(t, Apply, md.get(), skip);
  return AddBoundaryExchangeTasks(t, tl, md_ky, ml);
}

TaskStatus Report(StateDescriptor *pkg) {
  const int calls = pkg->Param<int>("sn_gd_calls");
  if (Globals::my_rank == 0 && calls > 0)
    std::cout << "[SN/graydsa] inner diffusion solves " << calls << ", mean iterations "
              << double(pkg->Param<int>("sn_gd_it_sum")) / calls << ", not converged "
              << pkg->Param<int>("sn_gd_nonconv") << std::endl;
  pkg->UpdateParam<int>("sn_gd_it_sum", 0);
  pkg->UpdateParam<int>("sn_gd_calls", 0);
  pkg->UpdateParam<int>("sn_gd_nonconv", 0);
  return TaskStatus::complete;
}

void Initialize(ParameterInput *pin, StateDescriptor *pkg, const int n_group) {
  const std::string bn = "radiation";
  const bool on = pin->GetOrAddBoolean(bn, "sn_gray_dsa", false);
  pkg->AddParam<>("sn_gray_dsa", on);
  if (!on) return;
  PARTHENON_REQUIRE(pkg->Param<bool>("sn_fc_scatter"),
                    "radiation/sn_gray_dsa accelerates the coupled solve: it needs radiation/sn_fc_scatter = true");
  const MetadataFlag OperatorSplit = Metadata::FlagNameExists("OperatorSplit") ? Metadata::GetUserFlag("OperatorSplit")
                                                                               : Metadata::AddUserFlag("OperatorSplit");
  {
    // As the FLD solver's fld.u: Derived (scratch, not in restarts) and NOT OneCopy (BiCGSTAB's work containers
    // must be distinct copies); OperatorSplit keeps it out of the hydro integrator's {WithFluxes, Cell} pack.
    Metadata mu({Metadata::Cell, Metadata::Derived, Metadata::FillGhost, Metadata::WithFluxes, Metadata::GMGRestrict,
                 Metadata::GMGProlongate, OperatorSplit});
    mu.RegisterRefinementOps<refinement_ops::ProlongateSharedLinear, refinement_ops::RestrictAverage>();
    pkg->AddField<u>(mu);
  }
  {
    Metadata mc({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost, Metadata::GMGRestrict});
    pkg->AddField<sig>(mc);
    pkg->AddField<dc>(mc);
  }
  {
    Metadata md3({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>({3}));
    pkg->AddField<D>(md3);
    Metadata mx({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>({n_group}));
    pkg->AddField(XiName(), mx);
  }
  pkg->AddParam<>("sn_gd_dt", 0.0, Params::Mutability::Mutable);
  pkg->AddParam<>("sn_gd_it_sum", 0, Params::Mutability::Mutable);
  pkg->AddParam<>("sn_gd_calls", 0, Params::Mutability::Mutable);
  pkg->AddParam<>("sn_gd_nonconv", 0, Params::Mutability::Mutable);
  pkg->AddParam<>("sn_gd_damping", pin->GetOrAddReal(bn, "sn_gray_dsa_damping", 1.0));
  pkg->AddParam<>("sn_gd_debug", pin->GetOrAddBoolean(bn, "sn_gray_dsa_debug", false));
  {
    const std::string gbc = pin->GetOrAddString(bn, "sn_gray_dsa_bc", "marshak");
    PARTHENON_REQUIRE(gbc == "marshak" || gbc == "dirichlet", "radiation/sn_gray_dsa_bc must be marshak or dirichlet");
    pkg->AddParam<>("sn_gd_dirichlet", gbc == "dirichlet");
    pkg->AddParam<>("sn_gd_tau_min", pin->GetOrAddReal(bn, "sn_gray_dsa_tau_min", 0.0));
  }
  if (pkg->Param<bool>("sn_gd_debug")) {
    const int nang = pin->GetInteger(bn, "sn_nzeta") * pin->GetInteger(bn, "sn_npsi");
    Metadata m1({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("sn.gd_dbg_rhs", m1);
    Metadata mn({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>({n_group * nang}));
    pkg->AddField("sn.gd_dbg_y0", mn);
    pkg->AddField("sn.gd_dbg_in", mn);
  }
  // Solver keys: SET when absent (BiCGSTABSolver GetOrAdds the same keys with its own defaults, and two defaults
  // for one key abort). rtol 1e-6: measured on the 1-D model, 1e-6 is as good as exact (1e-3 costs 1.8x outer).
  const std::string sb = bn + "/sn_gray_dsa_solver";
  if (!pin->DoesParameterExist(sb, "preconditioner")) pin->SetString(sb, "preconditioner", "Diagonal");
  if (!pin->DoesParameterExist(sb, "max_iterations")) pin->SetInteger(sb, "max_iterations", 1000);
  if (!pin->DoesParameterExist(sb, "relative_residual")) pin->SetBoolean(sb, "relative_residual", true);
  if (!pin->DoesParameterExist(sb, "relative_residual_tolerance")) pin->SetReal(sb, "relative_residual_tolerance", 1e-6);
  if (!pin->DoesParameterExist(sb, "absolute_residual_tolerance")) pin->SetReal(sb, "absolute_residual_tolerance", 1e-300);
  // Inner solver type. "BiCGSTAB" (tolerance-terminated Krylov; NOT a fixed linear map of the rhs) or "MG": a FIXED
  // number of multigrid V-cycles from a zero guess (residual_tolerance 0 -> exactly max_iterations cycles), i.e. a
  // stationary LINEAR operator, which is what the outer BiCGSTAB's short recurrences assume of a preconditioner.
  const std::string type = pin->GetOrAddString(sb, "type", "BiCGSTAB");
  PARTHENON_REQUIRE(type == "BiCGSTAB" || type == "MG", "radiation/sn_gray_dsa_solver/type must be BiCGSTAB or MG");
  const bool mg = (type == "MG") || pin->GetString(sb, "preconditioner") == "Multigrid";
  PARTHENON_REQUIRE(!mg || (pin->DoesParameterExist("parthenon/mesh", "multigrid") &&
                            pin->GetBoolean("parthenon/mesh", "multigrid")),
                    "radiation/sn_gray_dsa_solver: multigrid needs <parthenon/mesh> multigrid = true");
  using Eq = GrayDsaEquation;
  using prol_t = solvers::ProlongationBlockInteriorZeroDirichlet;
  using prec_t = solvers::MGSolver<Eq, prol_t>;
  std::shared_ptr<solvers::SolverBase> sv;
  if (type == "MG") {
    if (!pin->DoesParameterExist(sb, "residual_tolerance")) pin->SetReal(sb, "residual_tolerance", 0.0);
    if (!pin->DoesParameterExist(sb, "quiet")) pin->SetBoolean(sb, "quiet", true);   // no per-V-cycle output
    auto ms = std::make_shared<prec_t>("base", "sn_gd_u", "sn_gd_rhs", pin, sb, Eq(pin, sb));
    // NOT initial_guess_is_zero: MGSolver applies that flag to EVERY cycle's first smoothing (the task graph is built
    // once), so cycle 2 would discard cycle 1. AddCorrection zeroes the solution container before each solve instead.
    ms->initial_guess_is_zero = false;
    sv = ms;
  } else {
    auto bc = std::make_shared<solvers::BiCGSTABSolver<Eq, prec_t>>("base", "sn_gd_u", "sn_gd_rhs", pin, sb, Eq(pin, sb));
    bc->initial_guess_is_zero = true;
    sv = bc;
  }
  pkg->AddParam("sn_gd_solver", sv);
  pkg->AddParam<>("sn_gd_zero_u", type == "MG");
  if (Globals::my_rank == 0)
    std::cout << "## Sn gray DSA ON (radiation/sn_gray_dsa, R-F12): one gray diffusion solve per preconditioner "
                 "apply, inner " << type << " (preconditioner " << pin->GetString(sb, "preconditioner") << "), rtol "
              << pin->GetReal(sb, "relative_residual_tolerance") << std::endl;
}

} // namespace GrayDsa
} // namespace Sn
} // namespace Radiation
