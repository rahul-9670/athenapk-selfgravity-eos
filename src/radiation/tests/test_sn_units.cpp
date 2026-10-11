//========================================================================================
// Host test for the Sn moment<->intensity UNIT CALIBRATION (radiation_sn_units.hpp).
//
//   g++ -O2 -std=c++17 -Wall -Wextra src/radiation/tests/test_sn_units.cpp -o /tmp/t && /tmp/t
//
// WHY THIS SUITE EXISTS. radiation_moments.cpp refused to write Sn's moments back into
// rad.Er/rad.Fr for the whole prototype phase, with the reason stated in the task list: "a
// wrong factor there would silently corrupt the gas". A wrong 4*pi or a c-for-chat does not
// crash and does not look wrong -- it yields a smooth radiation field off by a constant. The
// only defence is to close the loop numerically, through the ACTUAL angular quadrature rather
// than through the algebra the header already asserts.
//
// Every check below composes the real weights from BuildAngularGrid with the conversions, so
// a mismatch between the quadrature normalisation (sum w = 4*pi) and the unit header cannot
// hide in either file alone.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <vector>

#include "../radiation_sn_angles.hpp"
#include "../radiation_sn_units.hpp"

using Radiation::Sn::AngularGrid;
using Radiation::Sn::BuildAngularGrid;
using Radiation::Sn::kFourPi;
using Radiation::Sn::SnEmissivity;
using Radiation::Sn::SnEnergyFromJ;
using Radiation::Sn::SnEquilibriumIntensity;
using Radiation::Sn::SnCoeffs;
using Radiation::Sn::SnFleckSplit;
using Radiation::Sn::SnFluxFromH;
using Radiation::Sn::SnIntensityFromEnergy;

static int failures = 0;
static void Check(bool ok, const char *what, double got, double tol) {
  if (!ok) { std::printf("  FAIL  %-56s  %.6e > %.1e\n", what, got, tol); ++failures; }
  else     { std::printf("  ok    %-56s  %.3e\n", what, got); }
}

//! Reduce an intensity array to (J, H1, H2, H3) EXACTLY as Sn::Moments does on the device.
static void Moments(const AngularGrid &g, const double *I, double *J, double *H) {
  *J = 0.0; H[0] = H[1] = H[2] = 0.0;
  for (int n = 0; n < g.nang; ++n) {
    const double wi = g.w[n] * I[n];
    *J += wi; H[0] += wi * g.nx[n]; H[1] += wi * g.ny[n]; H[2] += wi * g.nz[n];
  }
  *J /= kFourPi; for (int d = 0; d < 3; ++d) H[d] /= kFourPi;
}

int main() {
  std::printf("Sn moment<->intensity unit calibration\n");
  const AngularGrid g = BuildAngularGrid(8, 16);
  std::vector<double> I(g.nang);
  // chat deliberately NOT 1: a unit chat makes c-vs-chat and 4*pi errors invisible. This is
  // the collapse run's own reduced speed (creduc = 1000 against c_code = 1.5789e4).
  const double chat = 15.789;

  // --- 1. ROUND TRIP through the real quadrature: E -> isotropic I -> J -> E --------------
  {
    double worst = 0.0;
    for (double E : {1.0e-15, 1.0, 3.7e8}) {
      for (int n = 0; n < g.nang; ++n) I[n] = SnIntensityFromEnergy(E, chat);
      double J, H[3];
      Moments(g, I.data(), &J, H);
      worst = std::max(worst, std::abs(SnEnergyFromJ(J, chat) - E) / E);
    }
    Check(worst < 1.0e-14, "Er -> I_iso -> J -> Er is the identity", worst, 1.0e-14);
  }

  // --- 2. An isotropic field carries ZERO net flux ----------------------------------------
  // sum w n = 0 is checked in the angles suite; here it must survive the Fr conversion, which
  // is what actually reaches the gas as a radiation force.
  {
    for (int n = 0; n < g.nang; ++n) I[n] = SnIntensityFromEnergy(2.5, chat);
    double J, H[3];
    Moments(g, I.data(), &J, H);
    const double f = std::sqrt(SnFluxFromH(H[0]) * SnFluxFromH(H[0]) +
                               SnFluxFromH(H[1]) * SnFluxFromH(H[1]) +
                               SnFluxFromH(H[2]) * SnFluxFromH(H[2]));
    Check(f / (chat * SnEnergyFromJ(J, chat)) < 1.0e-14, "isotropic field -> |Fr| = 0",
          f / (chat * SnEnergyFromJ(J, chat)), 1.0e-14);
  }

  // --- 3. REDUCED-FLUX BOUND: a fully beamed field gives |Fr| = chat*Er, not chat*Er/creduc.
  // This is the check that fails loudly if SnFluxFromH ever acquires a 1/chat "for symmetry".
  // All the energy in the single ordinate closest to +x.
  {
    int nbest = 0;
    for (int n = 1; n < g.nang; ++n) if (g.nx[n] > g.nx[nbest]) nbest = n;
    for (int n = 0; n < g.nang; ++n) I[n] = 0.0;
    I[nbest] = 1.0 / g.w[nbest];   // unit weighted intensity in one direction
    double J, H[3];
    Moments(g, I.data(), &J, H);
    const double Er = SnEnergyFromJ(J, chat);
    const double Fr = SnFluxFromH(H[0]);
    // The quadrature direction is not exactly +x, so the reduced flux is nx of that ordinate.
    Check(std::abs(Fr / (chat * Er) - g.nx[nbest]) < 1.0e-14,
          "beamed field: |Fr|/(chat*Er) = Omega_x exactly",
          std::abs(Fr / (chat * Er) - g.nx[nbest]), 1.0e-14);
    Check(Fr / (chat * Er) <= 1.0 + 1.0e-14, "reduced flux never exceeds 1",
          Fr / (chat * Er), 1.0e-14);
  }

  // --- 4. THE EQUILIBRIUM IDENTITY -- pure absorption -------------------------------------
  // The whole physical content of the calibration: a static thick non-scattering cell relaxes
  // to I = eta/chi_a isotropically, and writing that back must give Er = B_g, the SAME
  // group-integrated Planck energy density M1's matter coupling drives Er toward. If this is
  // off by any constant, the gas is heated or cooled to the wrong temperature.
  {
    double worst = 0.0;
    for (double B : {1.0e-12, 1.0, 4.4e6}) {
      for (double chi_a : {1.0e-3, 1.0, 1.0e5}) {
        const double eta = SnEmissivity(chi_a, B, chat);
        const double Ieq = SnEquilibriumIntensity(eta, chi_a);
        for (int n = 0; n < g.nang; ++n) I[n] = Ieq;
        double J, H[3];
        Moments(g, I.data(), &J, H);
        worst = std::max(worst, std::abs(SnEnergyFromJ(J, chat) - B) / B);
      }
    }
    Check(worst < 1.0e-14, "thick equilibrium: Er == B_g (a_R T^4 f_g)", worst, 1.0e-14);
  }

  // --- 5. SCATTERING: chi_a in the emissivity is what makes equilibrium come out right -----
  // With chi_s > 0 the source is chi_a*B + chi_s*J. Solving the isotropic fixed point,
  //   I = [chi_a*chat*B/(4pi) + chi_s*I] / (chi_a + chi_s)  =>  I = chat*B/(4pi),
  // i.e. Er = B INDEPENDENT of chi_s -- the correct answer, and the reason emission must use
  // the ABSORPTION coefficient. Feeding TOTAL chi to SnEmissivity instead (which is what
  // folding the multiply into the sweep did) gives Er = B*(1 + chi_s/chi_a). Both branches are
  // computed here so the header's claimed error factor is measured, not asserted.
  {
    const double B = 3.0, chi_a = 0.4, chi_s = 1.6, chi = chi_a + chi_s;
    // Right: iterate the fixed point to convergence.
    double Ii = 0.0;
    for (int it = 0; it < 20000; ++it) {
      for (int n = 0; n < g.nang; ++n) I[n] = Ii;
      double J, H[3];
      Moments(g, I.data(), &J, H);
      Ii = (SnEmissivity(chi_a, B, chat) + chi_s * J) / chi;
    }
    for (int n = 0; n < g.nang; ++n) I[n] = Ii;
    double J, H[3];
    Moments(g, I.data(), &J, H);
    Check(std::abs(SnEnergyFromJ(J, chat) - B) / B < 1.0e-12,
          "with scattering, chi_a emissivity -> Er == B_g",
          std::abs(SnEnergyFromJ(J, chat) - B) / B, 1.0e-12);
    // Wrong: total chi in the emissivity, same fixed point.
    double Iw = 0.0;
    for (int it = 0; it < 20000; ++it) {
      for (int n = 0; n < g.nang; ++n) I[n] = Iw;
      double Jw, Hw[3];
      Moments(g, I.data(), &Jw, Hw);
      Iw = (SnEmissivity(chi, B, chat) + chi_s * Jw) / chi;
    }
    for (int n = 0; n < g.nang; ++n) I[n] = Iw;
    double Jw, Hw[3];
    Moments(g, I.data(), &Jw, Hw);
    const double ratio = SnEnergyFromJ(Jw, chat) / B;
    std::printf("  info  total-chi emissivity over-emits by %.4f (predicted %.4f)\n", ratio,
                1.0 + chi_s / chi_a);
    Check(std::abs(ratio - (1.0 + chi_s / chi_a)) < 1.0e-9,
          "total-chi emissivity error factor is 1 + chi_s/chi_a",
          std::abs(ratio - (1.0 + chi_s / chi_a)), 1.0e-9);
  }

  // --- 6. chat, not c: the conversion must SCALE with its speed argument -------------------
  // Guards the RSLA trap directly. Er at chat and Er at c=creduc*chat from the same intensity
  // field must differ by exactly creduc.
  {
    const double creduc = 1000.0;
    for (int n = 0; n < g.nang; ++n) I[n] = 7.3;
    double J, H[3];
    Moments(g, I.data(), &J, H);
    const double r = SnEnergyFromJ(J, chat) / SnEnergyFromJ(J, chat * creduc);
    Check(std::abs(r - creduc) / creduc < 1.0e-14, "Er scales as 1/speed (c-vs-chat is visible)",
          std::abs(r - creduc) / creduc, 1.0e-14);
  }

  // --- 7. THE FLECK SPLIT: both invariants, on the REAL decomposition --------------------
  // THIS CHECK FAILED TO CATCH A LIVE BUG and is rewritten because of it. The first version
  // solved the fixed point in CLOSED FORM as I = eta/(f*chi_a) -- which is the algebra I
  // INTENDED, not the decomposition the code produced. The code stored the bare chi_a as the
  // absorption while the scattering already carried (1-f)*chi_a, so the sweep's total
  // extinction was too big by (1-f)*chi_a and equilibrium came out Er = f*B. G8 caught it at
  // Er = 5.96e-3 against a target of 1; this test sailed through.
  //
  // The lesson is the one check 1 of test_sn_sweep.cpp already records: a test that recomputes
  // the intended answer independently of the code grades the intention. Both invariants below
  // are now evaluated on SnFleckSplit's ACTUAL output, the same function the seed calls.
  {
    const double B = 2.75, chi_a = 0.9, chi_s = 2.1, chi_ross = 1.4;
    const double tot_expect = chi_a + chi_s + (chi_ross > chi_a ? chi_ross - chi_a : 0.0);
    double worst_tot = 0.0, worst_eq = 0.0;
    for (double f : {1.0, 0.5, 1.0e-2, 1.0e-4}) {
      const SnCoeffs c = SnFleckSplit(chi_a, chi_s, chi_ross, B, chat, f);
      // INVARIANT 1: the total extinction must not depend on f. This is the one the bug broke.
      worst_tot = std::max(worst_tot,
                           std::abs((c.chi_a_eff + c.chi_s_eff) - tot_expect) / tot_expect);
      // INVARIANT 2: chi_a_eff * J = eta at equilibrium => Er = B, for any f. Formed from the
      // returned coefficients, and pushed through the REAL quadrature rather than asserted.
      const double Ieq = c.eta / c.chi_a_eff;
      for (int n = 0; n < g.nang; ++n) I[n] = Ieq;
      double J, H[3];
      Moments(g, I.data(), &J, H);
      worst_eq = std::max(worst_eq, std::abs(SnEnergyFromJ(J, chat) - B) / B);
    }
    Check(worst_tot < 1.0e-14, "Fleck split: total extinction is independent of f",
          worst_tot, 1.0e-14);
    Check(worst_eq < 1.0e-14, "Fleck split: equilibrium is Er == B_g for every f",
          worst_eq, 1.0e-14);

    // THE COST, measured rather than argued. Source iteration contracts by chi_s_eff/chi_tot
    // per sweep, so the same parameter that buys unconditional stability drives the iteration
    // count up as ~1/f. This is the classical slow convergence of source iteration in the
    // scattering limit (radiation_sn_sweep.hpp names DSA as the escalation), and it is the
    // second, independent reason the convergence test had to become a true residual: at
    // f = 1e-4 the per-sweep increment is ~1e-4 of the remaining error.
    std::printf("  info  Fleck source-iteration cost, chi_a=%.2g chi_s=%.2g:\n", chi_a, chi_s);
    for (double f : {1.0, 0.5, 1.0e-2, 1.0e-4}) {
      const SnCoeffs c = SnFleckSplit(chi_a, chi_s, chi_ross, B, chat, f);
      const double rate = c.chi_s_eff / (c.chi_a_eff + c.chi_s_eff);
      std::printf("          f = %-8.4g rate = %.6f   sweeps for 1e-10 = %.3g\n", f, rate,
                  (rate < 1.0) ? std::log(1.0e-10) / std::log(rate) : 1.0 / 0.0);
    }
  }

  // --- 8. THE EDDINGTON SEED REPRODUCES BOTH MOMENTS EXACTLY ------------------------------
  // SnInitFromMoments adopts the problem's radiation IC as
  //   I(Omega) = (chat/4pi)*(Er + 3 Omega.Fr/chat),
  // and the reason to prefer that over an isotropic seed carrying only Er is the claim that it
  // returns BOTH Er and Fr unchanged through the quadrature. Checked here against the real
  // weights, over reduced fluxes from 0 up to the 1/3 where the expansion is still positive.
  {
    double worstE = 0.0, worstF = 0.0;
    const double Er = 2.5;
    for (double fred : {0.0, 0.1, 0.25, 1.0 / 3.0}) {
      const double Fr = fred * chat * Er;      // flux along +x
      for (int n = 0; n < g.nang; ++n)
        I[n] = SnIntensityFromEnergy(Er, chat) + 3.0 * g.nx[n] * Fr / kFourPi;
      double J, H[3];
      Moments(g, I.data(), &J, H);
      worstE = std::max(worstE, std::abs(SnEnergyFromJ(J, chat) - Er) / Er);
      const double got = SnFluxFromH(H[0]);
      worstF = std::max(worstF, (Fr > 0.0) ? std::abs(got - Fr) / Fr : std::abs(got));
      // The transverse components must vanish: sum w Om_x Om_y = 0.
      worstF = std::max(worstF, std::abs(SnFluxFromH(H[1])) / (chat * Er));
    }
    Check(worstE < 1.0e-14, "Eddington seed returns Er exactly", worstE, 1.0e-14);
    Check(worstF < 1.0e-14, "Eddington seed returns Fr exactly (and no transverse flux)",
          worstF, 1.0e-14);
    // And the documented limit: beyond |Fr| = chat*Er/3 the expansion goes negative and the
    // code clamps. Stated as a measurement so the accuracy loss is not a surprise.
    const double Fr = 0.9 * chat * Er;
    int nneg = 0;
    for (int n = 0; n < g.nang; ++n)
      if (SnIntensityFromEnergy(Er, chat) + 3.0 * g.nx[n] * Fr / kFourPi < 0.0) ++nneg;
    std::printf("  info  at reduced flux 0.9 the Eddington seed is negative in %d of %d "
                "ordinates (clamped at 0; first-step accuracy only)\n", nneg, g.nang);
  }

  std::printf("%s  (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
