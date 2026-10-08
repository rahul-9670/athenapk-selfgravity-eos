//========================================================================================
// AthenaPK - dual-energy flux built from the reconstructed internal-energy DENSITY
// (validation_tests/dual_energy, finding DE-F3, 2026-10-06).
//========================================================================================
//! \file dual_flux_density.cpp
//! \brief hydro/dual_energy_flux = density: rebuild the dual-energy face flux after CalculateFluxes.
//!
//! WHY. The dual component is advected like a passive scalar: flux = mass flux x the upwind
//! RECONSTRUCTED specific energy e (hydro.cpp "Passive scalar fluxes"). At a contact (uniform v, P)
//! rho e = P/(Gamma-1) is uniform but rho and e are not, and PLM limits their slopes separately, so
//! rho_face e_face != rho e: a switched cell (whose pressure comes from the dual energy) then
//! leaves pressure equilibrium. Measured (D1a/D1x, rho 1|2 at u = 100 cs): max|P/P0 - 1| = 5.6e-2
//! with the dual energy on vs 1.6e-5 off; 0.42 at rho 1|10; bounded in time, not converging in N.
//!
//! WHAT. For every face, F_dual = F_mass * q_face / rho_face, with q = rho e and rho both PLM-
//! reconstructed (the same PLM as recon/plm_simple.hpp) on the UPWIND side (sign of the mass flux,
//! as the passive-scalar path). At a contact q_face = rho e exactly, F_mass = rho_face v, so
//! F_dual = v rho e: uniform, pressure equilibrium kept. Donor cell: q_face/rho_face = e_i, the
//! legacy flux (so the dc VL2 predictor is skipped). Positivity: PLM face values lie between the
//! neighbouring cell values, so q_face, rho_face > 0.
//!
//! Own translation unit (E-F9: changing inline bodies in shared code changed GCC's codegen and broke
//! production bit-continuity); the key off leaves the task graph unchanged.
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
using parthenon::Real; // plm_simple.hpp expects Real in scope (as in hydro.cpp)
#include "../recon/plm_simple.hpp"
#include "dual_flux_density.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {

TaskStatus DualEnergyDensityFlux(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const int idual = pkg->Param<int>("dual_idx");
  if (idual < 0) return TaskStatus::complete;
  const int ndim = pmb->pmy_mesh->ndim;
  const std::vector<std::string> cons_name{"cons"};
  auto cf = md->PackVariablesAndFluxes(cons_name, cons_name);
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  for (int dir = X1DIR; dir <= ndim; ++dir) {
    const int di = (dir == X1DIR), dj = (dir == X2DIR), dk = (dir == X3DIR);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "DualEnergyDensityFlux", DevExecSpace(), 0, cf.GetDim(5) - 1,
        kb.s, kb.e + dk, jb.s, jb.e + dj, ib.s, ib.e + di,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &c = cf(b);
          const auto &w = prim_pack(b);
          const Real fm = c.flux(dir, IDN, k, j, i);
          // upwind cell: the lower neighbour (k-dk, j-dj, i-di) for fm >= 0 (its UPPER face value),
          // else this cell (its LOWER face value) -- the same choice as the passive-scalar flux.
          const bool lower = (fm >= 0.0);
          const int kc = lower ? k - dk : k, jc = lower ? j - dj : j, ic = lower ? i - di : i;
          const Real r_m = w(IDN, kc - dk, jc - dj, ic - di), r_0 = w(IDN, kc, jc, ic),
                     r_p = w(IDN, kc + dk, jc + dj, ic + di);
          const Real q_m = r_m * w(idual, kc - dk, jc - dj, ic - di), q_0 = r_0 * w(idual, kc, jc, ic),
                     q_p = r_p * w(idual, kc + dk, jc + dj, ic + di);
          Real rl, rr, ql, qr; // PLM: (upper face value, lower face value) of the upwind cell
          PLM(r_m, r_0, r_p, rl, rr);
          PLM(q_m, q_0, q_p, ql, qr);
          c.flux(dir, idual, k, j, i) = lower ? fm * ql / rl : fm * qr / rr;
        });
  }
  return TaskStatus::complete;
}

//! hydro/dual_energy_pdv = face (validation_tests/dual_energy F3 experiment, 2026-10-07; default "centred"):
//! the -P div v work of the dual energy from the FACE velocities implied by the applied mass fluxes,
//! u_f = F_mass / rho_f (rho_f: upwind PLM face density, as in DualEnergyDensityFlux), instead of centred cell
//! velocity differences. Rationale: where the limiter clips the density update (extrema), the compression the mass
//! fluxes actually apply differs from the centred div v; the dual energy then misses that work (D2a, perpendicular
//! field: first-order max error at the density maximum).
void DualEnergyPdVFace(MeshData<Real> *md, const Real beta_dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const int idual = pkg->Param<int>("dual_idx");
  const int ndim = pmb->pmy_mesh->ndim;
  const std::vector<std::string> cons_name{"cons"}, vn{"dual_vflux"};
  auto cf = md->PackVariablesAndFluxes(cons_name, cons_name);
  auto vf = md->PackVariablesAndFluxes(vn, vn); // u_f = F_mass/rho_f, flux-corrected (DualFaceVolumeFlux)
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DualEnergyPdVFace", DevExecSpace(), 0, cf.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &c = cf(b);
        const auto &v = vf(b);
        const auto &w = prim_pack(b);
        const auto &coords = cf.GetCoords(b);
        auto uface = [&](const int dir, const int kf, const int jf, const int iff) {
          return v.flux(dir, 0, kf, jf, iff);
        };
        Real divv = (uface(X1DIR, k, j, i + 1) - uface(X1DIR, k, j, i)) / coords.template Dxc<1>(k, j, i);
        if (ndim > 1) divv += (uface(X2DIR, k, j + 1, i) - uface(X2DIR, k, j, i)) / coords.template Dxc<2>(k, j, i);
        if (ndim > 2) divv += (uface(X3DIR, k + 1, j, i) - uface(X3DIR, k, j, i)) / coords.template Dxc<3>(k, j, i);
        c(idual, k, j, i) -= beta_dt * w(IPR, k, j, i) * divv;
      });
}

} // namespace Hydro
