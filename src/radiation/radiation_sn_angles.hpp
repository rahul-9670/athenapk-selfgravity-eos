#ifndef RADIATION_RADIATION_SN_ANGLES_HPP_
#define RADIATION_RADIATION_SN_ANGLES_HPP_
//========================================================================================
// AthenaPK - discrete-ordinates (Sn) transport: the ANGULAR QUADRATURE.
//
// Step 1 of docs/SN_TRANSPORT_DESIGN.md. Host-side construction only -- the grid is built
// once at package init and copied to device as three flat arrays (nx, ny, nz, w).
//
// WHY A PRODUCT GAUSS-LEGENDRE GRID rather than a port of Athena++'s level-symmetric set:
// we are porting the METHOD, not the quadrature. A product grid of nzeta Gauss-Legendre
// nodes in mu = cos(theta) crossed with npsi uniform azimuths is
//   * EXACT for every moment the scheme actually consumes -- the zeroth (J), first (H_i) and
//     second (K_ij) angular moments -- which is a checkable statement, not a hope;
//   * trivially refinable in each direction independently, so ray effects can be measured
//     against angular resolution rather than argued about;
//   * ~40 lines instead of ~600 (athena++/src/nr_radiation/angulargrid.cpp).
// Exactness: GL with nzeta nodes integrates polynomials in mu to degree 2*nzeta-1, so
// nzeta >= 2 is exact through K_ij; the uniform azimuthal rule is exact for trigonometric
// polynomials of degree <= npsi-1, so npsi >= 3 is exact through K_ij. Both are ASSERTED.
//
// NORMALISATION: sum_n w_n = 4*pi, so J = sum_n w_n I_n / (4*pi) is the mean intensity and
// Er = (4*pi/c) * J. Every consumer must use this convention; it is checked in
// tests/test_sn_angles.cpp.
//========================================================================================
#include <cmath>
#include <string>
#include <vector>

namespace Radiation {
namespace Sn {

//! Host-side angular grid. Directions are unit vectors; weights sum to 4*pi.
struct AngularGrid {
  int nzeta{0};     //!< Gauss-Legendre nodes in mu = cos(theta), over [-1, 1]
  int npsi{0};      //!< uniform azimuthal nodes over [0, 2*pi)
  int nang{0};      //!< nzeta * npsi
  std::vector<double> nx, ny, nz, w;
};

//----------------------------------------------------------------------------------------
//! \brief Gauss-Legendre nodes/weights on [-1,1] by Newton iteration on P_n.
//!
//! Newton on the Legendre polynomial with the standard Chebyshev-like initial guess; the
//! recurrence gives both P_n and P_n' so the derivative is exact rather than differenced.
//! Converges in <= 6 iterations for the n <= 64 we care about; the loop is capped and the
//! residual asserted by the caller's moment test rather than trusted.
inline void GaussLegendre(int n, std::vector<double> &x, std::vector<double> &wt) {
  x.assign(n, 0.0);
  wt.assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    // Initial guess: the classical asymptotic root location.
    double z = std::cos(M_PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));
    double pp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int k = 0; k < n; ++k) {   // three-term recurrence -> P_n(z)
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * k + 1.0) * z * p1 - static_cast<double>(k) * p2) / (k + 1.0);
      }
      pp = static_cast<double>(n) * (z * p0 - p1) / (z * z - 1.0);   // P_n'(z)
      const double dz = p0 / pp;
      z -= dz;
      if (std::abs(dz) < 1.0e-15) break;
    }
    x[i] = z;
    wt[i] = 2.0 / ((1.0 - z * z) * pp * pp);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Build the product grid. Returns by value; called once at package init.
inline AngularGrid BuildAngularGrid(int nzeta, int npsi) {
  AngularGrid g;
  g.nzeta = nzeta;
  g.npsi = npsi;
  g.nang = nzeta * npsi;
  g.nx.resize(g.nang);
  g.ny.resize(g.nang);
  g.nz.resize(g.nang);
  g.w.resize(g.nang);

  std::vector<double> mu, wmu;
  GaussLegendre(nzeta, mu, wmu);

  const double dphi = 2.0 * M_PI / static_cast<double>(npsi);
  int n = 0;
  for (int iz = 0; iz < nzeta; ++iz) {
    const double m = mu[iz];
    const double s = std::sqrt(std::max(0.0, 1.0 - m * m));   // sin(theta) >= 0
    for (int ip = 0; ip < npsi; ++ip) {
      // Offset by half a cell so no ordinate lies exactly on a coordinate axis. A direction
      // with n_y == 0 EXACTLY makes the y-sweep a no-op for that ordinate, which shows up as
      // an axis-aligned ray artefact on a Cartesian grid -- cheap to avoid, annoying to debug.
      const double phi = (static_cast<double>(ip) + 0.5) * dphi;
      g.nx[n] = s * std::cos(phi);
      g.ny[n] = s * std::sin(phi);
      g.nz[n] = m;
      g.w[n] = wmu[iz] * dphi;     // sum_n w = (sum wmu) * (npsi*dphi) = 2 * 2pi = 4pi
      ++n;
    }
  }
  return g;
}

//----------------------------------------------------------------------------------------
//! \brief Moment-exactness check. Returns the worst absolute error over
//!        sum w = 4pi, sum w n_i = 0, sum w n_i n_j = (4pi/3) delta_ij.
//!
//! This is THE correctness gate for the angular grid: every downstream quantity (J, H_i,
//! K_ij, and hence the Eddington tensor Sn is supposed to get right without a closure) is one
//! of these sums. If this is not at round-off, nothing downstream can be.
inline double AngularMomentError(const AngularGrid &g) {
  const double fourpi = 4.0 * M_PI;
  double s = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
  double sxx = 0.0, syy = 0.0, szz = 0.0, sxy = 0.0, sxz = 0.0, syz = 0.0;
  for (int n = 0; n < g.nang; ++n) {
    const double w = g.w[n], x = g.nx[n], y = g.ny[n], z = g.nz[n];
    s += w;
    sx += w * x; sy += w * y; sz += w * z;
    sxx += w * x * x; syy += w * y * y; szz += w * z * z;
    sxy += w * x * y; sxz += w * x * z; syz += w * y * z;
  }
  const double diag = fourpi / 3.0;
  double e = std::abs(s - fourpi);
  e = std::max(e, std::abs(sx)); e = std::max(e, std::abs(sy)); e = std::max(e, std::abs(sz));
  e = std::max(e, std::abs(sxx - diag));
  e = std::max(e, std::abs(syy - diag));
  e = std::max(e, std::abs(szz - diag));
  e = std::max(e, std::abs(sxy)); e = std::max(e, std::abs(sxz)); e = std::max(e, std::abs(syz));
  return e;
}

} // namespace Sn
} // namespace Radiation

#endif // RADIATION_RADIATION_SN_ANGLES_HPP_
