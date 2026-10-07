//========================================================================================
// AthenaPK - advected self-gravitating slab (self-gravity + hydro, nonlinear exact solution)
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file grav_slab.cpp
//  \brief Port of artemis/src/pgen/grav_slab.hpp (Hanawa & Mullen 2025, Sec. 4.3): a
//         NONLINEAR hydrostatic self-gravitating slab, density
//           rho = rho0 (1 + eps cos(k.x) + eps^2/3 cos(2 k.x)),
//         with the pressure that balances its own gravity exactly, advected obliquely at
//         v = (0.8, 0.6, 0). After one period (t = 3 pi) the exact solution is the initial
//         state; the analysis script computes the artemis-style volume-weighted L1 errors of
//         the conserved variables (artemis tst/scripts/self_gravity/grav_slab.py).
//         Parameters are hard-wired to the artemis values; requires four_pi_G = 1, periodic
//         box (6 pi, 3 pi, 3 pi) and the Jeans swindle.

#include <cmath>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "pgen.hpp"

namespace grav_slab {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = gam - 1.0;
  const Real rho0 = 1.0, p0 = 6.0, eps = 0.3;
  const Real vx = 0.8, vy = 0.6, vz = 0.0;
  const Real kx = 1.0 / 3.0, ky = 2.0 / 3.0, kz = 2.0 / 3.0;
  const Real k2 = kx * kx + ky * ky + kz * kz;

  auto grav_pkg = pmb->packages.Get("self_gravity");
  const Real four_pi_G = grav_pkg->Param<Real>("four_pi_G");
  PARTHENON_REQUIRE(four_pi_G == 1.0, "grav_slab requires self_gravity/four_pi_G = 1");
  const Real nperiod = pin->GetOrAddReal("problem/grav_slab", "nperiod", 1.0);
  pin->SetReal("parthenon/time", "tlim", nperiod * 3.0 * M_PI);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
  auto &data = pmb->meshblock_data.Get();
  auto &u_dev = data->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();
  auto &coords = pmb->coords;
  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real kr = kx * coords.Xc<1>(i) + ky * coords.Xc<2>(j) + kz * coords.Xc<3>(k);
        const Real c1 = std::cos(kr), c2 = std::cos(2.0 * kr);
        const Real c3 = std::cos(3.0 * kr), c4 = std::cos(4.0 * kr);
        const Real rho = rho0 * (1.0 + eps * c1 + (eps * eps / 3.0) * c2);
        const Real pres =
            p0 + (four_pi_G * eps * rho0 * rho0 / k2) *
                     ((1.0 - eps * eps / 12.0) * c1 + (eps / 3.0) * c2 +
                      (eps * eps / 12.0) * c3 + (eps * eps * eps / 144.0) * c4);
        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * vx;
        u(IM2, k, j, i) = rho * vy;
        u(IM3, k, j, i) = rho * vz;
        u(IEN, k, j, i) = pres / gm1 + 0.5 * rho * (vx * vx + vy * vy + vz * vz);
        if (mhd) {
          u(IB1, k, j, i) = 0.0;
          u(IB2, k, j, i) = 0.0;
          u(IB3, k, j, i) = 0.0;
          u(IPS, k, j, i) = 0.0;
        }
      }
    }
  }
  u_dev.DeepCopy(u);
}

} // namespace grav_slab
