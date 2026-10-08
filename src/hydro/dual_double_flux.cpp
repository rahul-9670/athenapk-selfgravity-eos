//========================================================================================
// AthenaPK - double flux for the dual energy (validation_tests/dual_energy, DE-F3 part (b), 2026-10-07).
//========================================================================================
//! \file dual_double_flux.cpp
//! \brief hydro/dual_energy_double_flux: give the dual energy the own-EOS (double-flux) update in contact cells.
//!
//! WHY. hydro/double_flux makes contacts between EOS regimes pressure-exact by updating E with every cell's own frozen
//! stiffened-gas EOS (rho e = P/(g*-1) + rho e0*) and resetting E from the table at the frozen-EOS pressure. A
//! DUAL-SWITCHED cell takes its pressure from the dual energy and rebuilds E from it, so that correction is lost, and
//! its reset uses E - KE - ME, the quantity a switched cell must not trust. Measured (D1x, dual on, density flux):
//! contacts rho 1|2 keep max|P/P0 - 1| = 1.4e-3, rho 1|10 1.4e-2, vs 1.6e-5 with the dual energy off.
//!
//! WHAT. In the double-flux contact cells (the per-cell sensor of DoubleFluxCorrect), after the standard update, the
//! dual energy receives the difference between its own-EOS flux and the applied one on each face,
//!     F_own = F_mass (P_f gm1i_c + rho_f e0_c) / rho_f,    F_used = F_mass q_f / rho_f   (q = rho e),
//! with P_f, rho_f, q_f the face states the stage's reconstruction gives on the UPWIND side (as DualEnergyDensityFlux).
//! F_used is the applied dual flux itself and F_mass P_f/rho_f is stored as a flux of dual_vflux (DualFaceVolumeFlux),
//! so on refined meshes every term is the flux-corrected one (2026-10-07; see hydro.cpp dual_vflux),
//! and (g*, e0*) of THIS cell (dflux_frozen). At a contact (uniform v, P) the cell's dual energy then stays exactly on
//! its frozen EOS at P; it is finally reset to the table energy at the frozen-EOS pressure, as double flux does for E.
//! Non-switched cells are re-synced from E by the next ConsToPrim, so applying it in every contact cell is harmless.
//! Requires hydro/double_flux and hydro/dual_energy_flux = density (the face formula above is that flux's).
//!
//! Own translation unit (E-F9); key off = the task is never added.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
using parthenon::Real;
#include "../recon/plm_simple.hpp"
#include "dual_double_flux.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {
namespace {
KOKKOS_INLINE_FUNCTION Real EspFromPRef(const EOSTable::EosTable &tab, const Real rho, const Real P) {
  Real e = tab.EspFromP(rho, P);
  for (int it = 0; it < 3; ++it) { // Newton on the interpolant, as double_flux.cpp EspFromPRefined
    const Real h = 1.0e-7 * e;
    const Real f = tab.Pof(rho, e) - P;
    const Real d = (tab.Pof(rho, e + h) - tab.Pof(rho, e - h)) / (2.0 * h);
    if (!(d > 0.0)) break;
    const Real en = e - f / d;
    if (!(en > 0.0)) break;
    e = en;
  }
  return e;
}
} // namespace

TaskStatus DualFaceVolumeFlux(MeshData<Real> *md, const bool dc) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const std::vector<std::string> cn{"cons"}, vn{"dual_vflux"};
  auto cf = md->PackVariablesAndFluxes(cn, cn);
  auto vf = md->PackVariablesAndFluxes(vn, vn);
  auto prim_in = md->PackVariables(std::vector<std::string>{"prim"});
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  for (int dir = X1DIR; dir <= ndim; ++dir) {
    const int di = (dir == X1DIR), dj = (dir == X2DIR), dk = (dir == X3DIR);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "dual face volume flux", DevExecSpace(), 0, cf.GetDim(5) - 1, kb.s, kb.e + dk, jb.s,
        jb.e + dj, ib.s, ib.e + di, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &w = prim_in(b);
          const Real fm = cf(b).flux(dir, IDN, k, j, i);
          // upwind cell and face states exactly as DualEnergyDensityFlux
          const bool lower = (fm >= 0.0);
          const int kc = lower ? k - dk : k, jc = lower ? j - dj : j, ic = lower ? i - di : i;
          Real rf, pf;
          if (dc) {
            rf = w(IDN, kc, jc, ic);
            pf = w(IPR, kc, jc, ic);
          } else {
            Real rl, rr, pl, pr;
            PLM(w(IDN, kc - dk, jc - dj, ic - di), w(IDN, kc, jc, ic), w(IDN, kc + dk, jc + dj, ic + di), rl, rr);
            PLM(w(IPR, kc - dk, jc - dj, ic - di), w(IPR, kc, jc, ic), w(IPR, kc + dk, jc + dj, ic + di), pl, pr);
            rf = lower ? rl : rr;
            pf = lower ? pl : pr;
          }
          vf(b).flux(dir, 0, k, j, i) = fm / rf;
          vf(b).flux(dir, 1, k, j, i) = fm * pf / rf;
        });
  }
  return TaskStatus::complete;
}

TaskStatus DualDoubleFluxCorrect(MeshData<Real> *md, const Real beta_dt, const int stage) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const int idual = pkg->Param<int>("dual_idx");
  if (idual < 0) return TaskStatus::complete;
  const auto &tab = pkg->Param<AdiabaticGLMMHDEOS>("eos").GetEosTable();
  const int ndim = pmb->pmy_mesh->ndim;
  const Real kappa = pkg->Param<Real>("double_flux_kappa");
  const std::vector<std::string> cn{"cons"}, vn{"dual_vflux"};
  auto cf = md->PackVariablesAndFluxes(cn, cn);
  auto vf = md->PackVariablesAndFluxes(vn, vn);
  auto prim_in = md->PackVariables(std::vector<std::string>{"prim"});
  auto frz_in = md->PackVariables(std::vector<std::string>{"dflux_frozen"});
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "dual dflux correct", DevExecSpace(), 0, cf.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &c = cf(b);
        const auto &v = vf(b);
        const auto &w = prim_in(b);
        const auto &coords = cf.GetCoords(b);
        auto face_on = [&](const int k2, const int j2, const int i2) { // DoubleFluxCorrect's contact sensor
          const Real pa = w(IPR, k, j, i), pb = w(IPR, k2, j2, i2);
          const Real ra = w(IDN, k, j, i), rb = w(IDN, k2, j2, i2);
          const Real dpr = std::abs(pb - pa) / std::min(pa, pb), drr = std::abs(rb - ra) / std::min(ra, rb);
          return (dpr <= kappa * drr) || (dpr <= 1.0e-12 && drr <= 1.0e-12);
        };
        bool all_on = face_on(k, j, i - 1) && face_on(k, j, i + 1);
        if (ndim >= 2) all_on = all_on && face_on(k, j - 1, i) && face_on(k, j + 1, i);
        if (ndim >= 3) all_on = all_on && face_on(k - 1, j, i) && face_on(k + 1, j, i);
        if (!all_on) return;
        const Real g = frz_in(b, 0, k, j, i), e0c = frz_in(b, 1, k, j, i);
        const Real gm1i = 1.0 / (g - 1.0);
        // own-EOS minus APPLIED dual flux on a face: F_own = gm1i F_mass P_f/rho_f + e0c F_mass. All three are fluxes
        // Parthenon flux-corrects, so at a fine/coarse face the coarse cell corrects exactly the flux it received.
        auto dflx = [&](const int dir, const int kf, const int jf, const int iff) {
          return gm1i * v.flux(dir, 1, kf, jf, iff) + e0c * c.flux(dir, IDN, kf, jf, iff) -
                 c.flux(dir, idual, kf, jf, iff);
        };
        Real dd = (dflx(X1DIR, k, j, i) - dflx(X1DIR, k, j, i + 1)) / coords.template Dxc<1>(k, j, i);
        if (ndim >= 2) dd += (dflx(X2DIR, k, j, i) - dflx(X2DIR, k, j + 1, i)) / coords.template Dxc<2>(k, j, i);
        if (ndim >= 3) dd += (dflx(X3DIR, k, j, i) - dflx(X3DIR, k + 1, j, i)) / coords.template Dxc<3>(k, j, i);
        Real &ud = c(idual, k, j, i);
        ud += beta_dt * dd;
        const Real rho = c(IDN, k, j, i);
        const Real P = (g - 1.0) * (ud - rho * e0c);
        if (P > 0.0) ud = rho * EspFromPRef(tab, rho, P);
      });
  (void)stage;
  return TaskStatus::complete;
}

} // namespace Hydro
