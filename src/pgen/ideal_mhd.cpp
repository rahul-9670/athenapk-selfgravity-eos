//========================================================================================
// AthenaPK - validation problem generator for the IDEAL-MHD module (validation/ideal_mhd).
// Ideal-MHD validation suite (validation/ideal_mhd); not part of the collapse problem.
//
// problem/ideal_mhd/mode =
//   shocktube  1D Riemann problem along x1 split at x0 (Brio & Wu 1988; Ryu & Jones 1995): primitive left/right
//              states rhoL, vxL, vyL, vzL, PL, ByL, BzL | same with R, and a common Bx. Gamma-law EOS only (the
//              reference solutions are gamma-law).
//   noh        Noh (1987) implosion in d = ndim dimensions (planar / cylindrical / spherical): uniform rho0, cold
//              gas (P0 or, with eos = hydrogen, T0) moving at speed v0 toward the ORIGIN (put the origin on a cell
//              corner: symmetric box, even N). Optional uniform transverse field B0y, B0z (1D; MHD Noh).
//              Gamma-law answer: shock speed (gamma-1) v0 / 2, post-shock rho0 ((gamma+1)/(gamma-1))^d, upstream
//              rho0 (1 + v0 t / r)^(d-1). No inflow BC is needed if the box half-width exceeds
//              v0 t (1 + (gamma-1)/2): the boundary disturbance travels with the cold gas.
//   divb       uniform state at rest (rho0, P0 or T0, B0x/B0y/B0z) plus a Gaussian div B != 0 pulse
//              Bx += bamp exp(-r^2/rb^2), r = distance to the box centre in ndim dimensions. With bamp small and
//              B0 = 0 the extended-GLM momentum/energy sources are O(bamp^2), so div B obeys the telegraph equation
//              D_tt + lambda D_t = c_h^2 lap D, lambda = alpha c_h / mindx (dedner_source.cpp) -- solved exactly
//              by FFT in the analysis script.
// All modes: psi = 0, the dual-energy slot = the internal energy density, Heaviside-Lorentz B (P_mag = B^2/2).
//========================================================================================
#include <cmath>
#include <string>

#include <parthenon/package.hpp>

#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
#include "pgen.hpp"

namespace ideal_mhd {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  const std::string blk = "problem/ideal_mhd";
  const std::string mode = pin->GetString(blk, "mode");
  PARTHENON_REQUIRE_THROWS(mode == "shocktube" || mode == "noh" || mode == "divb",
                           "ideal_mhd: mode must be shocktube, noh or divb");
  const int imode = (mode == "shocktube") ? 0 : (mode == "noh") ? 1 : 2;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  PARTHENON_REQUIRE_THROWS(hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd,
                           "ideal_mhd expects fluid = glmmhd");
  const bool tabulated = (pin->GetString("hydro", "eos") == "hydrogen");
  PARTHENON_REQUIRE_THROWS(!(tabulated && imode == 0), "ideal_mhd shocktube: gamma-law EOS only");
  const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto tab = eos.GetEosTable();
  const Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  const int idual = hydro_pkg->Param<int>("dual_idx");
  const int ndim = pmb->pmy_mesh->ndim;

  // shocktube
  Real st[2][7] = {};
  const char *nm[7] = {"rho", "vx", "vy", "vz", "P", "By", "Bz"};
  if (imode == 0) {
    for (int s = 0; s < 2; ++s)
      for (int n = 0; n < 7; ++n)
        st[s][n] = pin->GetReal(blk, std::string(nm[n]) + (s == 0 ? "L" : "R"));
  }
  const Real Lr = st[0][0], Lvx = st[0][1], Lvy = st[0][2], Lvz = st[0][3], LP = st[0][4], Lby = st[0][5],
             Lbz = st[0][6];
  const Real Rr = st[1][0], Rvx = st[1][1], Rvy = st[1][2], Rvz = st[1][3], RP = st[1][4], Rby = st[1][5],
             Rbz = st[1][6];
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min"), x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real x0 = pin->GetOrAddReal(blk, "x0", 0.5 * (x1min + x1max));
  // noh / divb
  const Real rho0 = pin->GetOrAddReal(blk, "rho0", 1.0);
  const Real v0 = (imode == 1) ? pin->GetReal(blk, "v0") : 0.0;
  // uniform-state internal energy density: from P0 (gamma-law) or through the table from T0 [K] (eos = hydrogen)
  Real eint0 = 0.0;
  if (imode != 0) {
    if (tabulated)
      eint0 = tab.EintFromRhoTkHost(rho0, pin->GetReal(blk, "T0"));
    else
      eint0 = pin->GetReal(blk, "P0") / gm1;
  }
  const Real B0x = pin->GetOrAddReal(blk, "B0x", 0.0);
  const Real B0y = pin->GetOrAddReal(blk, "B0y", 0.0);
  const Real B0z = pin->GetOrAddReal(blk, "B0z", 0.0);
  const Real bamp = (imode == 2) ? pin->GetReal(blk, "bamp") : 0.0;
  const Real rb = (imode == 2) ? pin->GetReal(blk, "rb") : 1.0;
  const Real xc = 0.5 * (x1min + x1max);
  const Real yc = 0.5 * (pin->GetReal("parthenon/mesh", "x2min") + pin->GetReal("parthenon/mesh", "x2max"));
  const Real zc = 0.5 * (pin->GetReal("parthenon/mesh", "x3min") + pin->GetReal("parthenon/mesh", "x3max"));

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;
  const auto ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  const auto jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  const auto kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);

  pmb->par_for(
      "ideal_mhd_ic", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real dy = (ndim > 1) ? coords.Xc<2>(j) - yc : 0.0;
        const Real dz = (ndim > 2) ? coords.Xc<3>(k) - zc : 0.0;
        Real rho, vx = 0.0, vy = 0.0, vz = 0.0, eint, bx = B0x, by = B0y, bz = B0z;
        if (imode == 0) {
          const bool left = (x < x0);
          rho = left ? Lr : Rr;
          vx = left ? Lvx : Rvx;
          vy = left ? Lvy : Rvy;
          vz = left ? Lvz : Rvz;
          eint = (left ? LP : RP) / gm1;
          by = left ? Lby : Rby;
          bz = left ? Lbz : Rbz;
        } else if (imode == 1) {
          const Real dx = x - xc;
          const Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
          rho = rho0;
          eint = eint0;
          vx = -v0 * dx / r;
          vy = -v0 * dy / r;
          vz = -v0 * dz / r;
        } else {
          const Real dx = x - xc;
          rho = rho0;
          eint = eint0;
          bx += bamp * std::exp(-(dx * dx + dy * dy + dz * dz) / (rb * rb));
        }
        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * vx;
        u(IM2, k, j, i) = rho * vy;
        u(IM3, k, j, i) = rho * vz;
        u(IEN, k, j, i) = eint + 0.5 * rho * (vx * vx + vy * vy + vz * vz) + 0.5 * (bx * bx + by * by + bz * bz);
        u(IB1, k, j, i) = bx;
        u(IB2, k, j, i) = by;
        u(IB3, k, j, i) = bz;
        u(IPS, k, j, i) = 0.0;
        if (idual >= 0) u(idual, k, j, i) = eint;
      });
}

} // namespace ideal_mhd
