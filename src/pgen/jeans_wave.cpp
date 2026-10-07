//========================================================================================
// AthenaPK - linear Jeans wave/instability (self-gravity + hydro coupling validation)
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file jeans_wave.cpp
//  \brief Oblique linear Jeans mode in a periodic box, after Athena++ src/pgen/jeans.cpp
//         (regression tests grav/jeans_3d.py, grav/unstable_jeans_3d_mg.py), with the
//         exact adiabatic eigenmode (Athena++ adds an amp^2 sin(k x1) density term; that is
//         dropped here so the analytic solution is exact to O(amp^2)).
//
//  omega^2 = cs^2 k^2 - four_pi_G rho0,  cs^2 = gamma p0 / rho0,  rho0 = 1, p0 = 1/gamma.
//    stable   (omega^2 > 0): v(0) = 0, standing wave
//             drho = rho0 amp sin(k.x) cos(wt),  v = -(w/k) amp cos(k.x) sin(wt) khat
//    unstable (omega^2 < 0): pure growing mode, g = sqrt(-omega^2)
//             drho = rho0 amp sin(k.x) e^{gt},   v = +(g/k) amp cos(k.x) e^{gt} khat
//  four_pi_G is NOT set here: it is read from the self_gravity package (deck/CLI), so the
//  analysis script derives omega from the same number the solver used. The wave direction
//  follows Athena++: ang_3 = atan(Lx/Ly), ang_2 = atan(0.5 (Lx cos a3 + Ly sin a3) / Lz),
//  wavelength = min of the projected box lengths (= 1/3 for a 1 x 0.5 x 0.5 box).

#include <cmath>
#include <iostream>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "pgen.hpp"

namespace jeans_wave {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = gam - 1.0;
  const Real amp = pin->GetOrAddReal("problem/jeans_wave", "amp", 1.0e-6);
  const Real rho0 = 1.0, p0 = 1.0 / gam, cs2 = gam * p0 / rho0;

  const Real Lx = pin->GetReal("parthenon/mesh", "x1max") -
                  pin->GetReal("parthenon/mesh", "x1min");
  const Real Ly = pin->GetReal("parthenon/mesh", "x2max") -
                  pin->GetReal("parthenon/mesh", "x2min");
  const Real Lz = pin->GetReal("parthenon/mesh", "x3max") -
                  pin->GetReal("parthenon/mesh", "x3min");
  const int nx2 = pin->GetInteger("parthenon/mesh", "nx2");
  const int nx3 = pin->GetInteger("parthenon/mesh", "nx3");
  Real ang_3 = pin->GetOrAddReal("problem/jeans_wave", "ang_3", -999.9);
  if (ang_3 == -999.9) ang_3 = std::atan(Lx / Ly);
  const Real sa3 = std::sin(ang_3), ca3 = std::cos(ang_3);
  Real ang_2 = pin->GetOrAddReal("problem/jeans_wave", "ang_2", -999.9);
  if (ang_2 == -999.9) ang_2 = std::atan(0.5 * (Lx * ca3 + Ly * sa3) / Lz);
  const Real sa2 = std::sin(ang_2), ca2 = std::cos(ang_2);
  Real lambda = Lx * ca2 * ca3;
  if (nx2 > 1 && ang_3 != 0.0) lambda = std::min(lambda, Ly * ca2 * sa3);
  if (nx3 > 1 && ang_2 != 0.0) lambda = std::min(lambda, Lz * sa2);
  const Real kw = 2.0 * M_PI / lambda;

  auto grav_pkg = pmb->packages.Get("self_gravity");
  const Real four_pi_G = grav_pkg->Param<Real>("four_pi_G");
  const Real omega2 = cs2 * kw * kw - four_pi_G * rho0;
  const Real omega = std::sqrt(std::abs(omega2));
  if (pmb->gid == 0 && parthenon::Globals::my_rank == 0) {
    std::cout << "jeans_wave: lambda " << lambda << "  k " << kw << "  four_pi_G "
              << four_pi_G << "  omega^2 " << omega2 << "  n_J = lambda/lambda_J "
              << std::sqrt(four_pi_G * rho0 / (cs2 * kw * kw)) << "  ang_2 "
              << ang_2 * 180.0 / M_PI << "  ang_3 " << ang_3 * 180.0 / M_PI << std::endl;
  }

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
        const Real xp = ca2 * (coords.Xc<1>(i) * ca3 + coords.Xc<2>(j) * sa3) +
                        coords.Xc<3>(k) * sa2;
        const Real s = std::sin(kw * xp), c = std::cos(kw * xp);
        const Real rho = rho0 * (1.0 + amp * s);
        const Real vpar = (omega2 < 0.0) ? (omega / kw) * amp * c : 0.0;
        const Real m = rho * vpar;
        const Real m1 = m * ca3 * ca2, m2 = m * sa3 * ca2, m3 = m * sa2;
        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = m1;
        u(IM2, k, j, i) = m2;
        u(IM3, k, j, i) = m3;
        u(IEN, k, j, i) = p0 / gm1 * (1.0 + gam * amp * s) +
                          0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / rho;
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

} // namespace jeans_wave
