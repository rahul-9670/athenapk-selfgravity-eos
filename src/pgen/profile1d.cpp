//========================================================================================
// AthenaPK - tabulated 1D profile (self-gravity validation: literature tests L1-L9)
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file profile1d.cpp
//  \brief Initial state from a 1D table (written by validation_tests/self_gravity/scripts/
//         make_tables.py), columns: coordinate, rho, P [, v]. Used for the Spitzer sheet
//         (planar), polytropes with and without atmosphere, Evrard's collapse, the singular
//         isothermal sphere and the Boss-Bodenheimer cloud (spherical).
//
//  problem/profile1d:
//    table      path to the table (whitespace columns, '#' comments, coordinate ascending)
//    geometry   spherical (coordinate = r about (cx,cy,cz)) | planar (coordinate = x - x1min,
//               used periodically: the table must span one box length)
//    rho_out, p_out   state beyond the last table row (spherical only)
//    vx, vy, vz       uniform velocity added everywhere (advection tests)
//    omega            solid-body rotation about z for r < r_rot (default: table extent)
//    m2_amp           rho *= 1 + m2_amp cos(2 phi) for r < r_rot (Boss and Bodenheimer 1979)
//    nsub             sub-samples per cell and dimension (cell averages of rho, momentum, E)
//    b0z              uniform B_z (glmmhd, Heaviside-Lorentz: E += b0z^2/2)
//    barotropic       true: register the collapse_be Params so the driver runs the PRODUCTION
//                     barotropic enforcement (collapse_be::ApplyBarotropicCooling) every stage:
//                     P = rho sqrt(1 + (rho/rho_crit)^(2(gamma-1))), i.e. isothermal cs = 1 below
//                     rho_crit (code units). zero_mom_radius: momentum zeroed beyond this radius
//                     about the origin (default 1e30 = never).
//  A 4th column is a velocity: radial (spherical) or along x (planar).

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "pgen.hpp"

namespace profile1d {
using namespace parthenon::driver::prelude;

namespace {
struct Table {
  std::vector<Real> q, rho, p, v;
  bool has_v = false;
  bool loaded = false;
};
Table g_tab;

void Load(const std::string &path) {
  if (g_tab.loaded) return;
  std::ifstream in(path);
  PARTHENON_REQUIRE(in.good(), "profile1d: cannot open table " + path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    Real a, b, c, d;
    if (!(ss >> a >> b >> c)) continue;
    g_tab.q.push_back(a);
    g_tab.rho.push_back(b);
    g_tab.p.push_back(c);
    if (ss >> d) {
      g_tab.v.push_back(d);
      g_tab.has_v = true;
    }
  }
  PARTHENON_REQUIRE(g_tab.q.size() > 2, "profile1d: table has fewer than 3 rows");
  if (g_tab.has_v)
    PARTHENON_REQUIRE(g_tab.v.size() == g_tab.q.size(), "profile1d: ragged 4th column");
  g_tab.loaded = true;
}

// linear interpolation; i = index of the bracketing interval (binary search)
inline void Interp(Real x, Real &rho, Real &p, Real &v) {
  const auto &q = g_tab.q;
  const int n = static_cast<int>(q.size());
  int lo = 0, hi = n - 1;
  if (x <= q[0]) {
    lo = 0;
    hi = 1;
  } else {
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (q[mid] <= x)
        lo = mid;
      else
        hi = mid;
    }
  }
  const Real w = std::min(Real(1), std::max(Real(0), (x - q[lo]) / (q[hi] - q[lo])));
  rho = (1 - w) * g_tab.rho[lo] + w * g_tab.rho[hi];
  p = (1 - w) * g_tab.p[lo] + w * g_tab.p[hi];
  v = g_tab.has_v ? (1 - w) * g_tab.v[lo] + w * g_tab.v[hi] : 0.0;
}
} // namespace

void ProblemInitPackageData(parthenon::ParameterInput *pin,
                            parthenon::StateDescriptor *hydro_pkg) {
  const std::string blk = "problem/profile1d";
  if (!pin->GetOrAddBoolean(blk, "barotropic", false)) return;
  if (hydro_pkg->AllParams().hasKey("collapse_be_rhocrit")) return;
  hydro_pkg->AddParam("collapse_be_rhocrit", pin->GetOrAddReal(blk, "rho_crit", 1.0e30));
  hydro_pkg->AddParam("collapse_be_rc", pin->GetOrAddReal(blk, "zero_mom_radius", 1.0e30));
  hydro_pkg->AddParam("collapse_be_gamma", pin->GetReal("hydro", "gamma"));
  hydro_pkg->AddParam("collapse_be_mhd",
                      hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  const std::string blk = "problem/profile1d";
  Load(pin->GetString(blk, "table"));
  const std::string geom = pin->GetOrAddString(blk, "geometry", "spherical");
  PARTHENON_REQUIRE(geom == "spherical" || geom == "planar",
                    "profile1d: geometry must be spherical or planar");
  const bool sph = (geom == "spherical");
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = gam - 1.0;
  const Real rho_out = pin->GetOrAddReal(blk, "rho_out", g_tab.rho.back());
  const Real p_out = pin->GetOrAddReal(blk, "p_out", g_tab.p.back());
  const Real cx = pin->GetOrAddReal(blk, "cx", 0.0);
  const Real cy = pin->GetOrAddReal(blk, "cy", 0.0);
  const Real cz = pin->GetOrAddReal(blk, "cz", 0.0);
  const Real ux = pin->GetOrAddReal(blk, "vx", 0.0);
  const Real uy = pin->GetOrAddReal(blk, "vy", 0.0);
  const Real uz = pin->GetOrAddReal(blk, "vz", 0.0);
  const Real omega = pin->GetOrAddReal(blk, "omega", 0.0);
  const Real m2 = pin->GetOrAddReal(blk, "m2_amp", 0.0);
  const Real r_rot = pin->GetOrAddReal(blk, "r_rot", g_tab.q.back());
  const int nsub = pin->GetOrAddInteger(blk, "nsub", 2);
  const Real b0z = pin->GetOrAddReal(blk, "b0z", 0.0);
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real Lx = pin->GetReal("parthenon/mesh", "x1max") - x1min;
  const Real qmax = g_tab.q.back();

  auto point = [&](Real x, Real y, Real z, Real &rho, Real &vx, Real &vy, Real &vz,
                   Real &p) {
    Real v;
    if (sph) {
      const Real dx = x - cx, dy = y - cy, dz = z - cz;
      const Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r > qmax) {
        rho = rho_out;
        p = p_out;
        v = 0.0;
      } else {
        Interp(r, rho, p, v);
      }
      const Real ir = (r > 0.0) ? 1.0 / r : 0.0;
      vx = v * dx * ir + ux;
      vy = v * dy * ir + uy;
      vz = v * dz * ir + uz;
      if (r < r_rot) {
        vx += -omega * dy;
        vy += omega * dx;
        if (m2 != 0.0) rho *= 1.0 + m2 * std::cos(2.0 * std::atan2(dy, dx));
      }
    } else {
      Real s = std::fmod(x - x1min, Lx);
      if (s < 0) s += Lx;
      Interp(s, rho, p, v);
      vx = v + ux;
      vy = uy;
      vz = uz;
    }
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
  const Real w = 1.0 / (nsub * nsub * nsub);
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        Real d = 0, m1 = 0, m2s = 0, m3 = 0, e = 0;
        for (int kk = 0; kk < nsub; ++kk)
          for (int jj = 0; jj < nsub; ++jj)
            for (int ii = 0; ii < nsub; ++ii) {
              const Real x = coords.Xc<1>(i) + hx * ((ii + 0.5) / nsub - 0.5);
              const Real y = coords.Xc<2>(j) + hy * ((jj + 0.5) / nsub - 0.5);
              const Real z = coords.Xc<3>(k) + hz * ((kk + 0.5) / nsub - 0.5);
              Real rho, vx, vy, vz, p;
              point(x, y, z, rho, vx, vy, vz, p);
              d += w * rho;
              m1 += w * rho * vx;
              m2s += w * rho * vy;
              m3 += w * rho * vz;
              e += w * (p / gm1 + 0.5 * rho * (vx * vx + vy * vy + vz * vz));
            }
        u(IDN, k, j, i) = d;
        u(IM1, k, j, i) = m1;
        u(IM2, k, j, i) = m2s;
        u(IM3, k, j, i) = m3;
        u(IEN, k, j, i) = e;
        if (mhd) {
          u(IB1, k, j, i) = 0.0;
          u(IB2, k, j, i) = 0.0;
          u(IB3, k, j, i) = b0z;
          u(IPS, k, j, i) = 0.0;
          u(IEN, k, j, i) += 0.5 * b0z * b0z;
        }
      }
    }
  }
  u_dev.DeepCopy(u);
}

} // namespace profile1d
