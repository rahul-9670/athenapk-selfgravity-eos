//========================================================================================
// AthenaPK - exact energy closure of the Ohmic/ambipolar heating for the dual energy
// (validation_tests/dual_energy, finding DE-F6, 2026-10-07).
//========================================================================================
//! \file dual_heat_exact.cpp
//! \brief hydro/dual_energy_heat_exact: give every cell exactly the conservative energy update of an RKL2 half.
//!
//! WHY. With hydro/dual_energy a switched cell takes its pressure from the dual energy and rebuilds E from it, so
//! the Joule/AD heat must reach the dual energy explicitly. Both previous forms lose energy (D5: force-free Ohmic decay
//! with an exact per-cell answer): the centred-J source q = eta |J_c|^2 misses (k dx)^2/4 (spatial), and ANY stage
//! source integrated by the RKL2 recurrence misses the quadratic magnetic-energy change of the stiff modes (temporal:
//! -11 % at S = eta k^2 dt = 10 .. 100, N 64, with the face-based q of hydro/dual_energy_heat_face). The heat a cell
//! receives in the conservative scheme is (E_s - E_0) - (ME_s - ME_0): the total-energy change of the flux update minus
//! the magnetic-energy change, which is quadratic in B and cannot be a linear combination of stage sources.
//!
//! WHAT. The total energy's own update, a = E_s - E_0, is linear: the RKL2 recurrence applied to M_E = -div F_E. It is
//! accumulated here in three auxiliary cell fields that nothing else touches (ConsToPrim rewrites E in switched cells
//! during the stages, so E itself cannot be used -- audit P-03): a1 = a_{j-1}, a2 = a_{j-2}, m0 = M_E(Y_0), with the
//! coefficients of RKL2StepFirst/Other and a_0 = 0. After the last stage, in EVERY interior cell:
//!     E    = E_0 + a                                (the conservative update, exactly)
//!     dual = dual_0 + a - (|B_s|^2 - |B_0|^2) / 2   (KE is unchanged by Ohmic/AD diffusion)
//! with E_0, dual_0, B_0 from the half-start snapshot ("u1"). The FillDerived that follows lets ConsToPrim decide the
//! branch; either way the cell holds the conservative total energy, so the half conserves energy to round-off
//! wherever the fluxes do. The heat may be locally negative (reversible Poynting redistribution), as in E.
//! Requires diffusion/rkl2_freeze_eta (the stage eta, hence the fluxes, must not depend on the intermediate stage
//! pressures, which a switched cell's E rewrite makes meaningless) and no conduction/viscosity (they change E or KE
//! differently; refused by the dual-energy guards anyway).
//!
//! Own translation unit (E-F9); key off = these tasks are never added.
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>
#include <parthenon/parthenon.hpp>

#include "../main.hpp"
#include "dual_heat_exact.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {
namespace {
struct ExPacks {
  parthenon::MeshBlockVarFluxPack<Real> cf;
  parthenon::MeshBlockVarPack<Real> a1, a2, m0;
};
ExPacks GetPacks(MeshData<Real> *md) {
  const std::vector<std::string> cn{"cons"};
  return ExPacks{md->PackVariablesAndFluxes(cn, cn), md->PackVariables(std::vector<std::string>{"dual_ex_a1"}),
                 md->PackVariables(std::vector<std::string>{"dual_ex_a2"}),
                 md->PackVariables(std::vector<std::string>{"dual_ex_m0"})};
}
} // namespace

TaskStatus DualExactInit(MeshData<Real> *md, const int s_rkl, const Real tau) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto p = GetPacks(md);
  const Real mu_tilde_1 = 4. / 3. / (static_cast<Real>(s_rkl) * static_cast<Real>(s_rkl) + static_cast<Real>(s_rkl) - 2.);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DualExactInit", DevExecSpace(), 0, p.cf.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = p.cf.GetCoords(b);
        const Real m0 = parthenon::Update::FluxDivHelper(IEN, k, j, i, ndim, coords, p.cf(b));
        p.m0(b, 0, k, j, i) = m0;
        p.a1(b, 0, k, j, i) = mu_tilde_1 * tau * m0; // a_1
        p.a2(b, 0, k, j, i) = 0.0;                   // a_0
      });
  return TaskStatus::complete;
}

TaskStatus DualExactStage(MeshData<Real> *md, const Real mu_j, const Real nu_j, const Real mu_tilde_j,
                          const Real gamma_tilde_j, const Real tau) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto p = GetPacks(md);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DualExactStage", DevExecSpace(), 0, p.cf.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = p.cf.GetCoords(b);
        // the flux arrays hold the diffusion fluxes of Y_{j-1} (same call RKL2StepOther made for E)
        const Real M = parthenon::Update::FluxDivHelper(IEN, k, j, i, ndim, coords, p.cf(b));
        const Real aj = mu_j * p.a1(b, 0, k, j, i) + nu_j * p.a2(b, 0, k, j, i) + mu_tilde_j * tau * M +
                        gamma_tilde_j * tau * p.m0(b, 0, k, j, i);
        p.a2(b, 0, k, j, i) = p.a1(b, 0, k, j, i);
        p.a1(b, 0, k, j, i) = aj;
      });
  return TaskStatus::complete;
}

TaskStatus DualExactFinish(MeshData<Real> *md, MeshData<Real> *md_y0) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const int idual = pkg->Param<int>("dual_idx");
  const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto cons = md->PackVariables(std::vector<std::string>{"cons"});
  auto cons0 = md_y0->PackVariables(std::vector<std::string>{"cons"});
  auto a1 = md->PackVariables(std::vector<std::string>{"dual_ex_a1"});
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DualExactFinish", DevExecSpace(), 0, cons.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real a = a1(b, 0, k, j, i);
        const Real me = 0.5 * (SQR(cons(b, IB1, k, j, i)) + SQR(cons(b, IB2, k, j, i)) + SQR(cons(b, IB3, k, j, i)));
        const Real me0 =
            0.5 * (SQR(cons0(b, IB1, k, j, i)) + SQR(cons0(b, IB2, k, j, i)) + SQR(cons0(b, IB3, k, j, i)));
        cons(b, IEN, k, j, i) = cons0(b, IEN, k, j, i) + a;
        cons(b, idual, k, j, i) = cons0(b, idual, k, j, i) + a - (me - me0);
      });
  return TaskStatus::complete;
}

} // namespace Hydro
