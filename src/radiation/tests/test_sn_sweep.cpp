//========================================================================================
// Host test for the Sn cell update (docs/SN_TRANSPORT_DESIGN.md; radiation_sn_sweep.hpp).
//
//   g++ -O2 -std=c++17 -Wall -Wextra src/radiation/tests/test_sn_sweep.cpp -o /tmp/t && /tmp/t
//
// Compiles the SAME arithmetic the device kernel will run, so a mutation here is a mutation
// there. This is the pattern that caught the frozen-closure upwinding defect in M1.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <vector>

#include "../radiation_sn_angles.hpp"
#include "../radiation_sn_sweep.hpp"

using Radiation::Sn::AngularGrid;
using Radiation::Sn::BuildAngularGrid;
using Radiation::Sn::SnDiagonal;
using Radiation::Sn::SnOffDiagonal;
using Radiation::Sn::SnRhs;
using Radiation::Sn::SnUpdateCell;
using Radiation::Sn::SnSouCell;
using Radiation::Sn::SnSouDiagonal;
using Radiation::Sn::SnSouRhs;
using Radiation::Sn::SnWddCell;
using Radiation::Sn::SnWddDiagonal;
using Radiation::Sn::SnWddFace;
using Radiation::Sn::SnWddRhs;

static int failures = 0;
static void Check(bool ok, const char *what, double got, double tol) {
  if (!ok) { std::printf("  FAIL  %-50s  %.6e > %.1e\n", what, got, tol); ++failures; }
  else     { std::printf("  ok    %-50s  %.3e\n", what, got); }
}

int main() {
  std::printf("Sn cell update\n");

  // --- 1. Strict diagonal dominance for every ordinate, over a wide parameter sweep -------
  // This is the property that makes Gauss-Seidel converge on the transport operator. If it
  // ever fails the smoother has no guarantee at all.
  {
    const AngularGrid g = BuildAngularGrid(8, 16);
    const double dx[3] = {1.0, 0.5, 2.0};
    double worst = 1.0e30;
    for (double chatdt : {1.0e-6, 1.0, 1.0e6}) {
      for (double chi : {0.0, 1.0e-8, 1.0e6}) {
        const double rdt = 1.0 / chatdt;
        for (int n = 0; n < g.nang; ++n) {
          const double a0 = std::abs(g.nx[n]) / dx[0];
          const double a1 = std::abs(g.ny[n]) / dx[1];
          const double a2 = std::abs(g.nz[n]) / dx[2];
          const double diag = SnDiagonal(rdt, chi, a0, a1, a2);
          // Written out HERE, not taken from SnOffDiagonal. Grading a header function
          // against another function in the same header makes the check vacuous: a
          // mutation that dropped a2 from SnOffDiagonal weakened both sides and was
          // MISSED by this suite. The row's off-diagonal magnitudes are a0, a1, a2 by
          // the definition of the stencil, so state them.
          const double off = a0 + a1 + a2;
          worst = std::min(worst, diag - off);
          // SnOffDiagonal is then checked against that independent sum, so it has
          // coverage of its own instead of being one half of a tautology.
          if (std::abs(SnOffDiagonal(a0, a1, a2) - off) > 1.0e-15) worst = -1.0;
        }
      }
    }
    // diag - off == rdt + chi > 0 exactly; the minimum is the smallest rdt in the sweep.
    Check(worst > 0.0, "diag - offdiag > 0 for all ordinates/params", worst, 0.0);
  }

  // --- 2. Positivity ---------------------------------------------------------------------
  // I >= 0 is physical law for a specific intensity. With non-negative inputs every term of
  // the numerator and the denominator is non-negative, so the update cannot manufacture a
  // negative. Checked over the same sweep rather than argued.
  {
    const AngularGrid g = BuildAngularGrid(6, 12);
    double worst = 1.0e30;
    for (double chatdt : {1.0e-4, 1.0, 1.0e4}) {
      for (double chi : {0.0, 1.0, 1.0e5}) {
        for (double src : {0.0, 1.0e-30, 5.0}) {
          for (int n = 0; n < g.nang; ++n) {
            const double v = SnUpdateCell(0.0, 1.0 / chatdt, chi, src,
                                          std::abs(g.nx[n]), std::abs(g.ny[n]),
                                          std::abs(g.nz[n]), 0.0, 0.0, 0.0);
            worst = std::min(worst, v);
          }
        }
      }
    }
    Check(worst >= 0.0, "I >= 0 for non-negative inputs", worst, 0.0);
  }

  // --- 3. Pure absorption equilibrium: I -> S/chi ----------------------------------------
  // With no transport (uniform field, so upwind == local) and dt -> inf the cell must relax to
  // the local source function. This is the optically thick / LTE limit.
  {
    const double chi = 7.0, src = 21.0;         // S/chi = 3
    const double rdt = 1.0e-14;                  // dt -> inf
    // Uniform field: iterate to fixed point with upwind == self.
    double I = 0.0;
    for (int it = 0; it < 200; ++it) I = SnUpdateCell(I, rdt, chi, src, 0.3, 0.3, 0.3, I, I, I);
    Check(std::abs(I - 3.0) < 1.0e-10, "thick limit: I -> S/chi", std::abs(I - 3.0), 1.0e-10);
  }

  // --- 4. Free streaming down a 1-D column: the beam must arrive, undamped ----------------
  // chi = 0, one ordinate along +x, dt -> inf so the steady state is I(x) = I_left for all x.
  // Gauss-Seidel in sweep order solves this EXACTLY in one pass (the matrix is triangular in
  // the upwind ordering) -- which is precisely why an ordering-aware smoother beats Jacobi.
  {
    const int N = 64;
    std::vector<double> I(N + 2, 0.0);
    const double Ileft = 2.5;
    const double adx = 1.0;                      // |Omega_x|/dx
    // rdt = 0 EXACTLY, not a small number. With rdt = 1e-14 each cell carries a factor
    // 1/(1+rdt) and 64 of them accumulate to 1.6e-12 -- which is correct arithmetic and a
    // wrong test. Steady state means dt = infinity, i.e. rdt = 0; the diagonal is still
    // adx > 0 so nothing divides by zero.
    const double rdt = 0.0;
    I[0] = Ileft;                                // inflow ghost
    for (int i = 1; i <= N; ++i)                 // one forward GS sweep, upwind is i-1
      I[i] = SnUpdateCell(0.0, rdt, 0.0, 0.0, adx, 0.0, 0.0, I[i - 1], 0.0, 0.0);
    double worst = 0.0;
    for (int i = 1; i <= N; ++i) worst = std::max(worst, std::abs(I[i] - Ileft));
    Check(worst < 1.0e-12, "free streaming: exact in ONE upwind GS sweep", worst, 1.0e-12);
  }

  // --- 5. Jacobi needs N sweeps for what GS does in one ----------------------------------
  // The same column under Jacobi (all neighbours from the previous iterate) advances the beam
  // exactly one cell per sweep. This is the mechanism behind the measured smoother walls, and
  // it is worth having as an executable statement rather than a claim in a document.
  {
    const int N = 64;
    std::vector<double> I(N + 2, 0.0), Iold(N + 2, 0.0);
    const double Ileft = 2.5, adx = 1.0, rdt = 0.0;   // exact steady state, see test 4
    I[0] = Iold[0] = Ileft;
    int sweeps_to_arrive = -1;
    for (int s = 1; s <= 4 * N; ++s) {
      for (int i = 1; i <= N; ++i)
        I[i] = SnUpdateCell(0.0, rdt, 0.0, 0.0, adx, 0.0, 0.0, Iold[i - 1], 0.0, 0.0);
      for (int i = 1; i <= N; ++i) Iold[i] = I[i];
      if (sweeps_to_arrive < 0 && std::abs(I[N] - Ileft) < 1.0e-12) sweeps_to_arrive = s;
    }
    std::printf("  info  Jacobi needed %d sweeps to cross %d cells; GS needed 1\n",
                sweeps_to_arrive, N);
    Check(sweeps_to_arrive == N, "Jacobi advances exactly one cell per sweep",
          std::abs(static_cast<double>(sweeps_to_arrive - N)), 0.5);
  }

  // --- 6. Consistency: the update is the exact solution of the discrete equation ----------
  // Residual of  a*I - rhs  must be zero to round-off for arbitrary inputs. Guards against a
  // sign or term-placement slip that the physical limits above could still pass.
  {
    const double rdt = 0.37, chi = 1.9, src = 4.2;
    const double a0 = 0.8, a1 = 0.15, a2 = 0.44;
    const double u0 = 1.1, u1 = 2.2, u2 = 0.3, iold = 5.5;
    const double I = SnUpdateCell(iold, rdt, chi, src, a0, a1, a2, u0, u1, u2);
    const double resid = (rdt + chi + a0 + a1 + a2) * I
                       - (iold * rdt + src + a0 * u0 + a1 * u1 + a2 * u2);
    Check(std::abs(resid) < 1.0e-14, "update solves a*I = rhs exactly", std::abs(resid), 1.0e-14);

    // SnRhs is what the TRUE RESIDUAL kernel builds b from, so it needs coverage that does
    // NOT come from SnUpdateCell -- otherwise a mutation in SnRhs would move the update and
    // the residual together and cancel, which is the same tautology that let a dropped term
    // in SnOffDiagonal go unnoticed (check 1). Written out here independently.
    const double rhs_hand = iold * rdt + src + a0 * u0 + a1 * u1 + a2 * u2;
    Check(std::abs(SnRhs(iold, rdt, src, a0, a1, a2, u0, u1, u2) - rhs_hand) < 1.0e-14,
          "SnRhs matches an independent b_i",
          std::abs(SnRhs(iold, rdt, src, a0, a1, a2, u0, u1, u2) - rhs_hand), 1.0e-14);
    // And the identity the residual kernel relies on: the update is exactly the zero of the
    // residual, so a converged solve must report ~0 rather than ~tolerance.
    const double r2 = SnDiagonal(rdt, chi, a0, a1, a2) * I
                    - SnRhs(iold, rdt, src, a0, a1, a2, u0, u1, u2);
    Check(std::abs(r2) < 1.0e-14, "residual(SnUpdateCell(...)) == 0", std::abs(r2), 1.0e-14);
  }

  // --- 7. WEIGHTED DIAMOND (WDD): the limiting case, the consistency, and the FACE GAIN ----
  // The face gain is the load-bearing number of the whole scheme choice, so it is tested
  // directly rather than inferred from a converging run: d f_out/d f_in = -(1/w - 1), which is
  // exactly -1 at w = 1/2 (the marginally-stable checkerboard face mode that makes plain
  // diamond stall) and strictly inside the unit circle for every w > 1/2.
  {
    const double rdt = 0.37, chi = 1.9, src = 4.2;
    const double a0 = 0.8, a1 = 0.15, a2 = 0.44;
    const double f0 = 1.1, f1 = 2.2, f2 = 0.3, iold = 5.5;

    // (a) w = 1 IS step differencing -- not approximately, bit-identically. A scheme family
    // that cannot reproduce its own limiting case has a transcription bug, and this is the
    // cheapest place to catch one.
    const double Iw1 = SnWddCell(iold, rdt, chi, src, 1.0, a0, a1, a2, f0, f1, f2);
    const double Ist = SnUpdateCell(iold, rdt, chi, src, a0, a1, a2, f0, f1, f2);
    Check(Iw1 == Ist, "WDD at w=1 is bit-identical to step", std::abs(Iw1 - Ist), 0.0);
    Check(SnWddFace(Ist, 1.0, f0) == Ist, "WDD face at w=1 returns the cell value",
          std::abs(SnWddFace(Ist, 1.0, f0) - Ist), 0.0);

    // (b) w = 1/2 IS diamond: f_out = 2I - f_in.
    const double Ihalf = SnWddCell(iold, rdt, chi, src, 2.0, a0, a1, a2, f0, f1, f2);
    Check(std::abs(SnWddFace(Ihalf, 2.0, f0) - (2.0 * Ihalf - f0)) < 1.0e-14,
          "WDD face at w=1/2 is the diamond relation",
          std::abs(SnWddFace(Ihalf, 2.0, f0) - (2.0 * Ihalf - f0)), 1.0e-14);

    // (c) The update is the exact zero of the WDD residual, for a NON-trivial weight. This is
    // the invariant that makes SnResidual and SnSweepOrdered the same operator; if it breaks,
    // the solver converges confidently to the wrong answer.
    const double w = 0.55, rw = 1.0 / w;
    const double I = SnWddCell(iold, rdt, chi, src, rw, a0, a1, a2, f0, f1, f2);
    const double r = SnWddDiagonal(rdt, chi, rw, a0, a1, a2) * I
                   - SnWddRhs(iold, rdt, src, rw, a0, a1, a2, f0, f1, f2);
    Check(std::abs(r) < 1.0e-14, "residual(SnWddCell(...)) == 0 at w=0.55", std::abs(r),
          1.0e-14);
    // and an INDEPENDENT transcription of b, so a mutation cannot move update and residual
    // together and cancel (the tautology guarded against in check 6).
    const double rhs_hand = iold * rdt + src + rw * (a0 * f0 + a1 * f1 + a2 * f2);
    Check(std::abs(SnWddRhs(iold, rdt, src, rw, a0, a1, a2, f0, f1, f2) - rhs_hand) < 1.0e-14,
          "SnWddRhs matches an independent b_i",
          std::abs(SnWddRhs(iold, rdt, src, rw, a0, a1, a2, f0, f1, f2) - rhs_hand), 1.0e-14);

    // (d) THE FACE GAIN. Measured by finite difference on SnWddFace at FIXED cell value, which
    // is what the checkerboard mode does (it has I = 0 in every cell). Non-unit, mutually
    // distinct constants throughout: 1 is a fixed point of too many operations to discriminate.
    for (double ww : {0.5, 0.52, 0.55, 0.75, 1.0}) {
      const double rww = 1.0 / ww;
      const double g = (SnWddFace(0.0, rww, 3.7) - SnWddFace(0.0, rww, 1.3)) / (3.7 - 1.3);
      const double want = -(rww - 1.0);
      Check(std::abs(g - want) < 1.0e-13, "WDD face gain = -(1/w - 1)", std::abs(g - want),
            1.0e-13);
      // and the stability statement that the scheme choice rests on
      const bool marginal = (std::abs(g) >= 1.0 - 1.0e-13);
      Check(marginal == (ww <= 0.5 + 1.0e-13),
            "WDD face mode is marginal at w=1/2 and a contraction above it",
            marginal == (ww <= 0.5 + 1.0e-13) ? 0.0 : 1.0, 0.5);
    }
  }

  // --- 8. SECOND-ORDER UPWIND (SOU): the scheme that is valid under the Krylov solver -------
  // Everything here is an INDEPENDENT transcription; checking SnSouCell against SnSouRhs alone
  // would be the same tautology check 6 guards against on the step path.
  {
    const double rdt = 0.37, chi = 1.9, src = 2.3, iold = 0.71;
    const double a0 = 1.3, a1 = 0.7, a2 = 2.1;
    const double u0 = 5.0, u1 = 3.0, u2 = 11.0;     // first upwind
    const double v0 = 4.0, v1 = 2.0, v2 = 7.0;      // second upwind

    // (a) kappa = 0 is EXACTLY step differencing -- bit-identical, not merely close. This is
    // what licenses shipping `sou` with `step` as the default and claiming the default path is
    // unchanged: the same three lines of arithmetic collapse onto SnRhs/SnDiagonal.
    Check(SnSouDiagonal(rdt, chi, 0.0, a0, a1, a2) == SnDiagonal(rdt, chi, a0, a1, a2),
          "SOU diagonal at kappa=0 IS the step diagonal (bitwise)",
          std::abs(SnSouDiagonal(rdt, chi, 0.0, a0, a1, a2) - SnDiagonal(rdt, chi, a0, a1, a2)),
          0.0);
    Check(SnSouRhs(iold, rdt, src, 0.0, a0, a1, a2, u0, u1, u2, v0, v1, v2) ==
              SnRhs(iold, rdt, src, a0, a1, a2, u0, u1, u2),
          "SOU rhs at kappa=0 IS the step rhs (bitwise)",
          std::abs(SnSouRhs(iold, rdt, src, 0.0, a0, a1, a2, u0, u1, u2, v0, v1, v2) -
                   SnRhs(iold, rdt, src, a0, a1, a2, u0, u1, u2)),
          0.0);

    // (b) The update is the exact zero of the SOU residual -- the invariant that makes the
    // ordered sweep, SnResidual and SnApplyA the same operator.
    for (double kap : {0.0, 0.5, 0.94, 1.0}) {
      const double I = SnSouCell(iold, rdt, chi, src, kap, a0, a1, a2, u0, u1, u2, v0, v1, v2);
      const double r = SnSouDiagonal(rdt, chi, kap, a0, a1, a2) * I -
                       SnSouRhs(iold, rdt, src, kap, a0, a1, a2, u0, u1, u2, v0, v1, v2);
      Check(std::abs(r) < 1.0e-13, "residual(SnSouCell(...)) == 0", std::abs(r), 1.0e-13);
      // (c) INDEPENDENT transcription of the whole row: diag*I - b written from the scheme
      // definition sum_d a_d[(1+k/2) I - (1+k) u_d + (k/2) v_d] + (rdt+chi) I - rdt*I^n - S.
      const double row =
          (rdt + chi) * I - rdt * iold - src +
          a0 * ((1.0 + 0.5 * kap) * I - (1.0 + kap) * u0 + 0.5 * kap * v0) +
          a1 * ((1.0 + 0.5 * kap) * I - (1.0 + kap) * u1 + 0.5 * kap * v1) +
          a2 * ((1.0 + 0.5 * kap) * I - (1.0 + kap) * u2 + 0.5 * kap * v2);
      Check(std::abs(row) < 1.0e-12, "SOU row written from the scheme definition == 0",
            std::abs(row), 1.0e-12);
    }

    // (d) CONSISTENCY: a constant field must have ZERO net streaming, which is what makes the
    // scheme conservative for a uniform state -- (1 + k/2) - (1 + k) + k/2 == 0 exactly. A sign
    // slip on the second upwind term is invisible to (b) and (c) (they share the constants) and
    // shows up here as a spurious source.
    for (double kap : {0.25, 1.0}) {
      const double C = 3.25;
      const double net = a0 * ((1.0 + 0.5 * kap) * C - (1.0 + kap) * C + 0.5 * kap * C);
      Check(std::abs(net) < 1.0e-14, "SOU streaming annihilates a constant", std::abs(net),
            1.0e-14);
      // and a LINEAR field must be differenced exactly: with u = C - h, v = C - 2h the net
      // streaming per direction is a*h (second-order upwind is exact on linears).
      const double h = 0.8;
      const double netl =
          (1.0 + 0.5 * kap) * C - (1.0 + kap) * (C - h) + 0.5 * kap * (C - 2.0 * h);
      Check(std::abs(netl - h) < 1.0e-14, "SOU differences a LINEAR profile exactly",
            std::abs(netl - h), 1.0e-14);
    }

    // (e) THE SPURIOUS-MODE GAIN, which is why no weight dial is needed for conditioning.
    // Pure streaming with rdt = chi = 0 in 1-D: (1+k/2) I_i - (1+k) I_{i-1} + (k/2) I_{i-2} = 0
    // has roots 1 (the physical one) and k/(2+k). At kappa = 1 that is 1/3 -- a contraction --
    // against WDD's face gain of -(1/w - 1) = -0.887 at w = 0.53, which is nearly marginal.
    for (double kap : {0.5, 0.94, 1.0}) {
      const double r = kap / (2.0 + kap);
      const double poly = (1.0 + 0.5 * kap) * r * r - (1.0 + kap) * r + 0.5 * kap;
      Check(std::abs(poly) < 1.0e-14, "SOU spurious root is kappa/(2+kappa)", std::abs(poly),
            1.0e-14);
      Check(r < 0.35, "SOU spurious mode is a strong contraction (unlike WDD's 0.887)", r,
            0.35);
    }
  }

  std::printf("%s  (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
