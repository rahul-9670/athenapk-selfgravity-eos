//========================================================================================
// AthenaPK - static density fields with exactly known potentials (self-gravity validation)
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file grav_analytic.cpp
//  \brief Gas at rest, uniform pressure, and a density field whose Newtonian potential is
//         known in closed form (or by exact 1D quadrature). Only the Poisson solve is of
//         interest: run with nlim = 1 and a tiny CFL, then compare grav.phi with the exact
//         potential in validation_tests/self_gravity/scripts/exact_potentials.py.
//
//  profile = sinusoid  rho = rho_bg + amp sin(2 pi x/Lx) sin(2 pi y/Ly) sin(2 pi z/Lz)
//                      (periodic box, Jeans swindle; Athena++ poisson.cpp iprob = 1)
//          = gaussian  rho = rho_out + M (2 pi s^2)^(-3/2) exp(-r^2 / 2 s^2)
//                      (smooth, compact; phi = -G M erf(r / sqrt(2) s) / r)
//          = plummer   rho = rho_out + 3M/(4 pi a^3) (1 + r^2/a^2)^(-5/2), r < r_trunc
//                      (Athena++ poisson.cpp iprob = 2, truncated so the box holds all mass)
//          = spheres   up to two uniform spheres (Athena++ binary_gravity.cpp; one sphere =
//                      the WS-5a gate). Surface cells are volume-averaged by sub-sampling.
//          = ellipsoid homogeneous ellipsoid with semi-axes (a1, a2, a3) (Maclaurin spheroid
//                      when a1 = a2; Ricker 2008 FLASH test). Surface cells sub-sampled.
//          = uniform   rho = rho_bg everywhere (with v_homologous: ballistic converging flow,
//                      exact rho(t) = rho_bg / (1 - a t)^3 when self-gravity is off; SG-F4 test)
//  v_homologous = a    optional velocity v = -a (x - c) for every profile (default 0)

#include <cmath>
#include <string>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "pgen.hpp"

namespace grav_analytic {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  const std::string blk = "problem/grav_analytic";
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = gam - 1.0;
  const std::string profile = pin->GetOrAddString(blk, "profile", "spheres");
  const Real pres = pin->GetOrAddReal(blk, "pressure", 1.0);
  const Real rho_out = pin->GetOrAddReal(blk, "rho_out", 1.0e-10);
  const int nsub = pin->GetOrAddInteger(blk, "nsub", 8);

  // sinusoid
  const Real rho_bg = pin->GetOrAddReal(blk, "rho_bg", 2.0);
  const Real amp = pin->GetOrAddReal(blk, "amp", 1.0);
  // gaussian / plummer / ellipsoid centre (spheres have their own centres)
  const Real cx = pin->GetOrAddReal(blk, "cx", 0.0);
  const Real cy = pin->GetOrAddReal(blk, "cy", 0.0);
  const Real cz = pin->GetOrAddReal(blk, "cz", 0.0);
  const Real mass = pin->GetOrAddReal(blk, "mass", 1.0);
  const Real sigma = pin->GetOrAddReal(blk, "sigma", 1.0);
  const Real a_pl = pin->GetOrAddReal(blk, "a_plummer", 1.0);
  const Real r_trunc = pin->GetOrAddReal(blk, "r_trunc", 1.0e30);
  // spheres
  const Real m1 = pin->GetOrAddReal(blk, "m1", 1.0);
  const Real r1 = pin->GetOrAddReal(blk, "r1", 1.0);
  const Real x1c = pin->GetOrAddReal(blk, "x1", 0.0);
  const Real y1c = pin->GetOrAddReal(blk, "y1", 0.0);
  const Real z1c = pin->GetOrAddReal(blk, "z1", 0.0);
  const Real m2 = pin->GetOrAddReal(blk, "m2", 0.0);
  const Real r2s = pin->GetOrAddReal(blk, "r2", 1.0);
  const Real x2c = pin->GetOrAddReal(blk, "x2", 0.0);
  const Real y2c = pin->GetOrAddReal(blk, "y2", 0.0);
  const Real z2c = pin->GetOrAddReal(blk, "z2", 0.0);
  // ellipsoid
  const Real ea1 = pin->GetOrAddReal(blk, "a1", 1.0);
  const Real ea2 = pin->GetOrAddReal(blk, "a2", 1.0);
  const Real ea3 = pin->GetOrAddReal(blk, "a3", 0.5);
  const Real rho_ell = pin->GetOrAddReal(blk, "rho_in", 1.0);

  const Real a_hom = pin->GetOrAddReal(blk, "v_homologous", 0.0);
  PARTHENON_REQUIRE(profile == "sinusoid" || profile == "gaussian" ||
                        profile == "plummer" || profile == "spheres" ||
                        profile == "ellipsoid" || profile == "uniform",
                    "grav_analytic: unknown profile");

  const Real Lx = pin->GetReal("parthenon/mesh", "x1max") -
                  pin->GetReal("parthenon/mesh", "x1min");
  const Real Ly = pin->GetReal("parthenon/mesh", "x2max") -
                  pin->GetReal("parthenon/mesh", "x2min");
  const Real Lz = pin->GetReal("parthenon/mesh", "x3max") -
                  pin->GetReal("parthenon/mesh", "x3min");
  const Real x0 = pin->GetReal("parthenon/mesh", "x1min");
  const Real y0 = pin->GetReal("parthenon/mesh", "x2min");
  const Real z0 = pin->GetReal("parthenon/mesh", "x3min");

  const Real rho_s1 = m1 / (4.0 / 3.0 * M_PI * r1 * r1 * r1);
  const Real rho_s2 = m2 / (4.0 / 3.0 * M_PI * r2s * r2s * r2s);
  const Real rho_g0 = mass / std::pow(2.0 * M_PI * sigma * sigma, 1.5);
  const Real rho_p0 = 3.0 * mass / (4.0 * M_PI * a_pl * a_pl * a_pl);

  // Density at a point; the discontinuous profiles are sub-sampled near their surfaces.
  auto rho_point = [&](Real x, Real y, Real z) -> Real {
    if (profile == "uniform") return rho_bg;
    if (profile == "sinusoid") {
      return rho_bg + amp * std::sin(2.0 * M_PI * (x - x0) / Lx) *
                          std::sin(2.0 * M_PI * (y - y0) / Ly) *
                          std::sin(2.0 * M_PI * (z - z0) / Lz);
    }
    const Real dx = x - cx, dy = y - cy, dz = z - cz;
    const Real rr2 = dx * dx + dy * dy + dz * dz;
    if (profile == "gaussian") {
      return rho_out + rho_g0 * std::exp(-rr2 / (2.0 * sigma * sigma));
    }
    if (profile == "plummer") {
      if (rr2 >= r_trunc * r_trunc) return rho_out;
      return rho_out + rho_p0 * std::pow(1.0 + rr2 / (a_pl * a_pl), -2.5);
    }
    if (profile == "ellipsoid") {
      const Real m2e = dx * dx / (ea1 * ea1) + dy * dy / (ea2 * ea2) + dz * dz / (ea3 * ea3);
      return (m2e < 1.0) ? rho_out + rho_ell : rho_out;
    }
    // spheres
    Real rho = rho_out;
    const Real d1 = (x - x1c) * (x - x1c) + (y - y1c) * (y - y1c) + (z - z1c) * (z - z1c);
    const Real d2 = (x - x2c) * (x - x2c) + (y - y2c) * (y - y2c) + (z - z2c) * (z - z2c);
    if (d1 < r1 * r1) rho += rho_s1;
    if (m2 > 0.0 && d2 < r2s * r2s) rho += rho_s2;
    return rho;
  };
  // Does the cell (centre xc, half-diagonal hd) straddle a density discontinuity?
  auto near_surface = [&](Real x, Real y, Real z, Real hd) -> bool {
    if (profile == "spheres") {
      const Real d1 = std::sqrt((x - x1c) * (x - x1c) + (y - y1c) * (y - y1c) +
                                (z - z1c) * (z - z1c));
      const Real d2 = std::sqrt((x - x2c) * (x - x2c) + (y - y2c) * (y - y2c) +
                                (z - z2c) * (z - z2c));
      return std::abs(d1 - r1) < hd || (m2 > 0.0 && std::abs(d2 - r2s) < hd);
    }
    if (profile == "ellipsoid") {
      const Real dx = x - cx, dy = y - cy, dz = z - cz;
      const Real m = std::sqrt(dx * dx / (ea1 * ea1) + dy * dy / (ea2 * ea2) +
                               dz * dz / (ea3 * ea3));
      const Real amin = std::min(ea1, std::min(ea2, ea3));
      return std::abs(m - 1.0) < 1.01 * hd / amin;
    }
    if (profile == "plummer") {
      const Real d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
      return std::abs(d - r_trunc) < hd;
    }
    return false;
  };

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
  auto &data = pmb->meshblock_data.Get();
  auto &u_dev = data->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();
  auto &coords = pmb->coords;
  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);
  const Real hx = coords.Dxc<1>(), hy = coords.Dxc<2>(), hz = coords.Dxc<3>();
  const Real hd = 0.5 * std::sqrt(hx * hx + hy * hy + hz * hz);

  for (int k = kb.s; k <= kb.e; ++k) {
    const Real z = coords.Xc<3>(k);
    for (int j = jb.s; j <= jb.e; ++j) {
      const Real y = coords.Xc<2>(j);
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real x = coords.Xc<1>(i);
        Real rho;
        if (nsub > 1 && near_surface(x, y, z, hd)) {
          Real sum = 0.0;
          for (int kk = 0; kk < nsub; ++kk) {
            const Real zz = z + hz * ((kk + 0.5) / nsub - 0.5);
            for (int jj = 0; jj < nsub; ++jj) {
              const Real yy = y + hy * ((jj + 0.5) / nsub - 0.5);
              for (int ii = 0; ii < nsub; ++ii) {
                const Real xx = x + hx * ((ii + 0.5) / nsub - 0.5);
                sum += rho_point(xx, yy, zz);
              }
            }
          }
          rho = sum / (nsub * nsub * nsub);
        } else {
          rho = rho_point(x, y, z);
        }
        const Real vx = -a_hom * (x - cx), vy = -a_hom * (y - cy), vz = -a_hom * (z - cz);
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

} // namespace grav_analytic
