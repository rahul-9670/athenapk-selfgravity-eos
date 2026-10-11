//========================================================================================
// Host test for the Sn angular quadrature (docs/SN_TRANSPORT_DESIGN.md, section 1).
//
// Build & run with no Kokkos, no MPI, no GPU:
//   g++ -O2 -std=c++17 -Wall -Wextra src/radiation/tests/test_sn_angles.cpp -o /tmp/t && /tmp/t
//
// WHAT IT PROVES. Every quantity the scheme consumes -- J, H_i, K_ij, and hence the Eddington
// tensor that Sn is supposed to get right WITHOUT a closure -- is one of the sums checked here.
// If these are not at round-off the port cannot be correct, whatever the transport does.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <vector>

#include "../radiation_sn_angles.hpp"

using Radiation::Sn::AngularGrid;
using Radiation::Sn::AngularMomentError;
using Radiation::Sn::BuildAngularGrid;

static int failures = 0;

static void Check(bool ok, const char *what, double got, double tol) {
  if (!ok) {
    std::printf("  FAIL  %-46s  %.6e > %.1e\n", what, got, tol);
    ++failures;
  } else {
    std::printf("  ok    %-46s  %.3e\n", what, got);
  }
}

int main() {
  std::printf("Sn angular quadrature\n");

  // --- 1. Moment exactness across a range of resolutions --------------------------------
  // nzeta >= 2 and npsi >= 3 is the claimed exactness threshold (GL exact to degree
  // 2*nzeta-1 in mu; uniform azimuth exact for trig degree <= npsi-1). Test AT the threshold
  // and above; a scheme that only works at high resolution has not been shown exact.
  const int cases[][2] = {{2, 3}, {2, 4}, {4, 4}, {4, 8}, {6, 8}, {8, 16}, {12, 24}};
  for (const auto &c : cases) {
    const AngularGrid g = BuildAngularGrid(c[0], c[1]);
    const double e = AngularMomentError(g);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "nzeta=%2d npsi=%2d (nang=%3d) moments", c[0], c[1], g.nang);
    Check(e < 1.0e-13, buf, e, 1.0e-13);
  }

  // --- 2. Directions are unit vectors ---------------------------------------------------
  {
    const AngularGrid g = BuildAngularGrid(8, 16);
    double worst = 0.0;
    for (int n = 0; n < g.nang; ++n) {
      const double r = std::sqrt(g.nx[n] * g.nx[n] + g.ny[n] * g.ny[n] + g.nz[n] * g.nz[n]);
      worst = std::max(worst, std::abs(r - 1.0));
    }
    Check(worst < 1.0e-14, "|n| == 1 for every ordinate", worst, 1.0e-14);
  }

  // --- 3. All weights strictly positive -------------------------------------------------
  // A negative quadrature weight lets a positive intensity field produce a negative J. Some
  // high-order level-symmetric Sn sets DO have negative weights above S20; the product GL grid
  // never does, and this asserts it rather than assuming it.
  {
    const AngularGrid g = BuildAngularGrid(12, 24);
    double wmin = g.w[0];
    for (int n = 0; n < g.nang; ++n) wmin = std::min(wmin, g.w[n]);
    Check(wmin > 0.0, "all weights > 0 (no negative-J pathology)", wmin, 0.0);
  }

  // --- 4. No ordinate lies on a coordinate axis -----------------------------------------
  // The half-cell azimuthal offset exists for this. An ordinate with n_y == 0 exactly makes
  // the y-sweep a no-op for that direction -> axis-aligned ray artefact on a Cartesian grid.
  {
    const AngularGrid g = BuildAngularGrid(8, 16);
    double smallest = 1.0;
    for (int n = 0; n < g.nang; ++n) {
      smallest = std::min(smallest, std::abs(g.nx[n]));
      smallest = std::min(smallest, std::abs(g.ny[n]));
    }
    Check(smallest > 1.0e-3, "no ordinate on the x or y axis", smallest, 1.0e-3);
  }

  // --- 5. Isotropy round-trip: a uniform field must give J = I, H = 0, K = (1/3) I -------
  // The physical statement of test 1, in the units the kernels will actually use.
  {
    const AngularGrid g = BuildAngularGrid(6, 12);
    const double I0 = 3.7;
    double J = 0.0, Hx = 0.0, Kxx = 0.0, Kxy = 0.0;
    for (int n = 0; n < g.nang; ++n) {
      J += g.w[n] * I0;
      Hx += g.w[n] * I0 * g.nx[n];
      Kxx += g.w[n] * I0 * g.nx[n] * g.nx[n];
      Kxy += g.w[n] * I0 * g.nx[n] * g.ny[n];
    }
    const double fourpi = 4.0 * M_PI;
    J /= fourpi; Hx /= fourpi; Kxx /= fourpi; Kxy /= fourpi;
    Check(std::abs(J - I0) < 1.0e-13, "isotropic: J == I0", std::abs(J - I0), 1.0e-13);
    Check(std::abs(Hx) < 1.0e-13, "isotropic: H_x == 0", std::abs(Hx), 1.0e-13);
    Check(std::abs(Kxx - I0 / 3.0) < 1.0e-13, "isotropic: K_xx == I0/3",
          std::abs(Kxx - I0 / 3.0), 1.0e-13);
    Check(std::abs(Kxy) < 1.0e-13, "isotropic: K_xy == 0", std::abs(Kxy), 1.0e-13);
  }

  // --- 6. The property M1 CANNOT have: an exact Eddington factor for a two-beam field ----
  // Two opposed beams give H = 0 with K_xx/J = 1, i.e. chi = 1 while f = |H|/J = 0. M1 closes
  // chi from f alone and therefore returns chi = 1/3 -- the -63.3% two-source failure measured
  // in athena++/runs/m1_vs_sn/FINDINGS_m1_adequacy.md, which is the whole physics case for Sn.
  // Sn reproduces chi = 1 by construction. This test IS that claim.
  {
    const AngularGrid g = BuildAngularGrid(8, 16);
    // Put equal intensity into the two ordinates most nearly aligned with +x and -x.
    int np = 0, nm = 0;
    for (int n = 0; n < g.nang; ++n) {
      if (g.nx[n] > g.nx[np]) np = n;
      if (g.nx[n] < g.nx[nm]) nm = n;
    }
    std::vector<double> I(g.nang, 0.0);
    I[np] = 1.0 / g.w[np];
    I[nm] = 1.0 / g.w[nm];
    double J = 0.0, Hx = 0.0, Kxx = 0.0;
    for (int n = 0; n < g.nang; ++n) {
      J += g.w[n] * I[n];
      Hx += g.w[n] * I[n] * g.nx[n];
      Kxx += g.w[n] * I[n] * g.nx[n] * g.nx[n];
    }
    const double f = std::abs(Hx) / J;
    const double chi_sn = Kxx / J;
    const double chi_m1 = (3.0 + 4.0 * f * f) / (5.0 + 2.0 * std::sqrt(4.0 - 3.0 * f * f));
    const double nxp = g.nx[np];
    std::printf("  info  two-beam: f=%.3e  chi_Sn=%.4f  chi_M1=%.4f  (n_x=%.4f)\n",
                f, chi_sn, chi_m1, nxp);
    Check(f < 1.0e-13, "two-beam: reduced flux f == 0", f, 1.0e-13);
    // chi_sn is n_x^2 for the chosen pair, not 1, because the nearest ordinates to +/-x are not
    // exactly axial. The POINT is that it is nowhere near M1's 1/3.
    Check(std::abs(chi_sn - nxp * nxp) < 1.0e-13, "two-beam: chi_Sn == n_x^2 (exact)",
          std::abs(chi_sn - nxp * nxp), 1.0e-13);
    Check(std::abs(chi_m1 - 1.0 / 3.0) < 1.0e-13, "two-beam: chi_M1 collapses to 1/3",
          std::abs(chi_m1 - 1.0 / 3.0), 1.0e-13);
    Check(chi_sn > 0.75, "two-beam: Sn retains the beam (chi >> 1/3)", chi_sn, 0.75);
  }

  std::printf("%s  (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
