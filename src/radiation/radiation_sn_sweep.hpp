#ifndef RADIATION_RADIATION_SN_SWEEP_HPP_
#define RADIATION_RADIATION_SN_SWEEP_HPP_
//========================================================================================
// AthenaPK - discrete-ordinates (Sn) transport: the CELL UPDATE.
//
// Extracted as a device-callable header, deliberately, so the host tests in
// tests/test_sn_sweep.cpp compile the SAME arithmetic the GPU kernel runs. That pattern is
// borrowed from radiation_m1_frozen.hpp, where it caught a real upwinding defect that every
// uniform-state gate had missed.
//
// THE EQUATION. For ordinate n with unit direction Omega, backward Euler in time and
// first-order upwind in space:
//
//   (I_i - I_i^n)/(chat*dt) + sum_d Omega_d (dI/dx_d)|_upwind = S_i - chi_i * I_i
//
// with chi = chi_a + chi_s (absorption + scattering) and S = chi_a*B + chi_s*J. Collecting
// the implicit terms gives the update this header implements:
//
//   I_i = [ I_i^n/(chat*dt) + S_i + sum_d (|Omega_d|/dx_d) * I_upwind_d ] / a_i
//   a_i =   1/(chat*dt) + chi_i + sum_d |Omega_d|/dx_d
//
// The upwind neighbour in direction d is i-1 when Omega_d > 0 and i+1 when Omega_d < 0.
//
// WHY THIS FORM CONVERGES UNDER GAUSS-SEIDEL AND WHY THAT IS NOT THE WHOLE STORY.
// Per ordinate the matrix is an M-matrix: the diagonal a_i is the sum of every off-diagonal
// magnitude PLUS 1/(chat*dt) + chi_i, so diag - sum|offdiag| = 1/(chat*dt) + chi_i EXACTLY.
// It is strictly diagonally dominant for any dt > 0 PROVIDED chi_i >= 0 -- that condition is
// not decorative, and it is not enforced here: adx_d cannot be negative (std::abs over a
// positive dx), but chi arrives from the tabulated-and-interpolated opacity chain, where a
// small negative excursion is a plausible input rather than an impossible one. The caller
// clamps it (see radiation_sn.cpp, Sn::Sweep); positivity of I depends on the same clamp.
// Jacobi and Gauss-Seidel both converge on it; GS converges ~2x faster (asymptotic rate
// rho_GS = rho_J^2 for consistently-ordered matrices), and red-black is the parallel ordering
// that realises it.
//
// That per-ordinate dominance is NOT what limits the iteration. The SCATTERING source couples
// every ordinate at a cell through J = sum_n w_n I_n / (4pi), and that coupling sits OUTSIDE
// the per-ordinate diagonal. As chi_s*dx grows the coupled iteration matrix approaches
// singularity -- the classical slow convergence of source iteration in the diffusion limit.
// This is the mechanism behind the measured smoother walls (plain Jacobi 32x the light CFL,
// red-black no wall through 81x; FINDINGS_smoother.md Finding 2): red-black's factor of ~2 on
// the transport sweep was enough to cross this problem's threshold. IT IS NOT GUARANTEED TO BE
// ENOUGH ON A THICKER PROBLEM. The literature's escalation is DSA (diffusion synthetic
// acceleration): precondition the scattering update with a cheap diffusion solve. Do not
// invent something else first -- and note AthenaPK already owns a working diffusion solver
// (the FLD path) that would supply exactly that operator.
//========================================================================================

#ifndef KOKKOS_INLINE_FUNCTION
#define SN_SWEEP_DEFINED_KIF
#define KOKKOS_INLINE_FUNCTION inline
#endif

namespace Radiation {
namespace Sn {

//----------------------------------------------------------------------------------------
//! \brief Diagonal coefficient a_i for one ordinate in one cell.
//!
//! `rdt` is 1/(chat*dt). `adx[d]` is |Omega_d|/dx_d, precomputed per ordinate per direction
//! (it is geometry x angle, constant over the whole sweep, so it never belongs inside the
//! cell loop).
KOKKOS_INLINE_FUNCTION double SnDiagonal(const double rdt, const double chi,
                                         const double adx0, const double adx1,
                                         const double adx2) {
  return rdt + chi + adx0 + adx1 + adx2;
}

//----------------------------------------------------------------------------------------
//! \brief Right-hand side b_i of the discrete row: everything not multiplying I_i.
//!
//! Factored out of SnUpdateCell so the TRUE RESIDUAL kernel and the update compute the same
//! b from the same expression. A residual measured against a separately-written b would be
//! grading the two transcriptions against each other, not the solve -- and the whole reason
//! the convergence test moved from an iterate increment to ||A x - b||/||b|| is that the
//! scattering source makes the increment unable to distinguish slow source iteration from
//! convergence. By construction SnDiagonal(...)*SnUpdateCell(...) - SnRhs(...) == 0, which
//! tests/test_sn_sweep.cpp checks directly.
KOKKOS_INLINE_FUNCTION double SnRhs(const double in_old, const double rdt, const double src,
                                    const double adx0, const double adx1, const double adx2,
                                    const double iup0, const double iup1, const double iup2) {
  return in_old * rdt + src + adx0 * iup0 + adx1 * iup1 + adx2 * iup2;
}

//----------------------------------------------------------------------------------------
//! \brief One cell, one ordinate. Returns the updated intensity.
//!
//! `iup[d]` is the UPWIND neighbour's current intensity in direction d -- the caller resolves
//! which side that is, because the caller knows the sign of Omega_d and owns the memory
//! access. Under red-black Gauss-Seidel the neighbour values are the just-updated ones for
//! the opposite colour; under Jacobi they are last iterate's. The arithmetic is identical,
//! which is why this function does not know which smoother is running.
//!
//! `src` is S_i = chi_a*B + chi_s*J, formed by the caller from the moments (two-pass source,
//! see SN_TRANSPORT_DESIGN.md section 4.2 -- never hold the angle vector per thread).
KOKKOS_INLINE_FUNCTION double SnUpdateCell(const double in_old, const double rdt,
                                           const double chi, const double src,
                                           const double adx0, const double adx1,
                                           const double adx2, const double iup0,
                                           const double iup1, const double iup2) {
  const double rhs = SnRhs(in_old, rdt, src, adx0, adx1, adx2, iup0, iup1, iup2);
  const double a = SnDiagonal(rdt, chi, adx0, adx1, adx2);
  return rhs / a;
}

//----------------------------------------------------------------------------------------
//! \brief WEIGHTED-DIAMOND (WDD) cell update: the diffusion-limit-preserving family, and the
//!        reason it is a FAMILY rather than the single diamond point.
//!
//! WHY A SECOND SCHEME EXISTS AT ALL. Step (donor-cell) differencing does not preserve the
//! asymptotic diffusion limit: its truncation error is a numerical diffusion of order c*dx that
//! ADDS to the physical D = chat/(3*chi). Gate K4 measured the consequence exactly:
//!
//!     D_measured / D_analytic  =  1 + 0.63 * tau_cell        (tau_cell = chi*dx)
//!
//! (+11.6 % at tau_cell = 0.2, +161 % at 3.0). Harmless where cells resolve a mean free path --
//! the collapse run's median tau_cell is 0.09, so +6 % -- and NOT harmless at depth.
//!
//! THE SCHEME. Carry the OUTGOING FACE value f_out,d per direction and close the cell balance
//!
//!     sum_d adx_d (f_out,d - f_in,d) + (rdt + chi) I = rdt*I^n + S
//!
//! with the auxiliary relation  I = (1-w) f_in,d + w f_out,d,  i.e.
//!
//!     f_out,d = I/w - (1/w - 1) f_in,d.
//!
//! Substituting gives the update below, which is the SAME SHAPE as SnUpdateCell with adx_d
//! replaced by adx_d/w and the incoming FACE in place of the upwind CELL:
//!
//!     I = [ rdt*I^n + S + (1/w) sum_d adx_d f_in,d ] / [ rdt + chi + (1/w) sum_d adx_d ].
//!
//! w = 1   is EXACTLY step differencing (f_out = I, so the incoming face is the upwind cell
//!         value). This is not an analogy -- it is bit-identical arithmetic, and
//!         `sn_spatial = wdd` + `sn_dd_weight = 1.0` is used as a gate against `sn_spatial =
//!         step` precisely because a scheme family that cannot reproduce its own limiting case
//!         has a transcription bug.
//! w = 1/2 is DIAMOND differencing (f_out = 2I - f_in), second order and the only member with
//!         ZERO first-order numerical diffusion.
//!
//! WHY NOT SIMPLY w = 1/2. MEASURED, and this is the whole reason the weight is exposed. At
//! w = 1/2 the face recursion is a PURE REFLECTION:
//!
//!     d f_out,d / d f_in,d  =  -(1 - w)/w  =  -1     exactly, at w = 1/2.
//!
//! So the discrete system admits a checkerboard face mode -- faces alternating as (-1)^(i+j+k)
//! with I = 0 in every cell, which satisfies every cell balance identically whenever
//! adx_0*A + adx_1*B + adx_2*C = 0 (always solvable in 2-D and 3-D). That mode is marginally
//! stable: the stationary iteration neither damps nor amplifies it, so the outer solve cannot
//! converge to a tight tolerance. It was MEASURED as a stall, not inferred: on the 64x4x4 CPU
//! reproduction (runs/sn_k4_diff/tiny_dd.in), step converged in ~100 iterations while diamond
//! ran 4000 with rel creeping 3.4e-5 -> 1.8e-5, a mode eigenvalue of ~0.9998. Two competing
//! explanations were falsified first: the positivity fixup (the stall SURVIVES with the fixup
//! off, 9.0e-5) and the scattering source iteration (it survives at chi_s = 0, pure absorption).
//!
//! Any w > 1/2 makes that gain |(1-w)/w| < 1 and the mode a genuine contraction, at the price
//! of reintroducing numerical diffusion in proportion to (w - 1/2). The weight is therefore a
//! DIRECT accuracy-vs-convergence dial, and gate K4 is what calibrates it -- do not pick it
//! from this comment.
//!
//! POSITIVITY. WDD is not positive for w < 1: f_out can go negative where the solution is
//! under-resolved, so the caller applies a set-to-zero fixup on the FACE. That fixup makes the
//! iteration non-linear; it is switchable (`sn_diamond_fixup`) so a stall can be attributed
//! rather than guessed at, which is exactly how the diamond diagnosis above was closed.
KOKKOS_INLINE_FUNCTION double SnWddDiagonal(const double rdt, const double chi,
                                            const double rw, const double adx0,
                                            const double adx1, const double adx2) {
  return rdt + chi + rw * (adx0 + adx1 + adx2);
}

KOKKOS_INLINE_FUNCTION double SnWddRhs(const double in_old, const double rdt,
                                       const double src, const double rw, const double adx0,
                                       const double adx1, const double adx2, const double f0,
                                       const double f1, const double f2) {
  return in_old * rdt + src + rw * (adx0 * f0 + adx1 * f1 + adx2 * f2);
}

//! `rw` is 1/w. `f[d]` is the INCOMING FACE value in direction d (not the upwind cell value).
KOKKOS_INLINE_FUNCTION double SnWddCell(const double in_old, const double rdt,
                                        const double chi, const double src, const double rw,
                                        const double adx0, const double adx1,
                                        const double adx2, const double f0, const double f1,
                                        const double f2) {
  return SnWddRhs(in_old, rdt, src, rw, adx0, adx1, adx2, f0, f1, f2) /
         SnWddDiagonal(rdt, chi, rw, adx0, adx1, adx2);
}

//! Outgoing face from the updated cell value and the incoming face: f_out = I/w - (1/w - 1)f_in.
//! At rw = 1 (w = 1) this returns exactly `in_new`, which is what makes the family reduce to
//! step differencing rather than merely resemble it.
KOKKOS_INLINE_FUNCTION double SnWddFace(const double in_new, const double rw,
                                        const double f_in) {
  return rw * in_new - (rw - 1.0) * f_in;
}

//----------------------------------------------------------------------------------------
//! \brief SECOND-ORDER UPWIND (SOU) cell update -- the diffusion-limit fix that is LOCAL, and
//!        therefore the one that works under the Krylov solver.
//!
//! WHY A THIRD SCHEME. Step differencing carries a numerical diffusion of 0.63*tau_cell (gate
//! K4). WDD removes it but its face closure is an upstream RECURSION (weights 0.887^k at
//! w = 0.53), so the Krylov operator cannot be assembled from a halo and `wdd` is refused under
//! `bicgstab` (SN_TRANSPORT_DESIGN.md section 16). SOU reconstructs the OUTGOING face of a cell
//! linearly from the cell and its upwind neighbour, f_out = I_i + (I_i - I_up1)/2, so the
//! streaming term becomes the 3-point, strictly upwind, cell-centred stencil
//!
//!     sum_d adx_d [ (1 + k/2) I_i - (1 + k) I_up1,d + (k/2) I_up2,d ],       k = kappa,
//!
//! kappa = 1 is second-order upwind (zero first-order numerical diffusion; the leading error is
//! an O(dx^3) HYPERdiffusion, invisible to the K4 variance measure and 3-10x smaller in L1 than
//! WDD's residual diffusion at w = 0.53 on a 4-cell pulse, tests/sn_1d_schemes.py), kappa = 0 is
//! EXACTLY step differencing (bit-identical arithmetic: the up2 term vanishes and the diagonal
//! reduces to SnDiagonal), and 0 < kappa < 1 has (1 - kappa) of step's numerical diffusion --
//! kappa = 0.94 reproduces WDD at w = 0.53 to four digits in the 1-D model.
//!
//! WHY IT IS KRYLOV-SAFE. The stencil is (i, i-1, i-2) along each direction: two upwind ghost
//! cells, which nghost >= 2 supplies. A x is then a fixed local linear map of x, so SnApplyA,
//! SnResidual and the ordered sweep can describe the SAME operator from the same three values.
//! The spurious mode of the pure-streaming recurrence is r = kappa/(2 + kappa) = 1/3 at
//! kappa = 1 -- a fast contraction, NOT WDD's marginal 0.887 face mode, so no weight dial is
//! needed for conditioning. The step sweep remains the preconditioner: on Fourier modes
//! M_step^-1 A_sou = 1.5 - 0.5 e^{-ik}, a circle of radius 0.5 about 1.5, i.e. clustered.
//!
//! WHAT IT GIVES UP. Godunov: no linear scheme above first order is monotone. Behind a sharp
//! absorber with tau_cell = d the per-cell decay has complex roots for d > 0.5 at kappa = 1, so
//! the shadow carries small sign-alternating undershoots (-2 % of the incident at d = 3, third
//! cell; WDD at w = 0.53 gives -6 % in the SECOND cell). There is deliberately NO fixup: a
//! fixup is non-linear and would invalidate the Krylov operator, and WDD's fixup was measured
//! to move the answer by 0.13 % of the step-vs-wdd difference (section 16.3).
KOKKOS_INLINE_FUNCTION double SnSouDiagonal(const double rdt, const double chi,
                                            const double kappa, const double adx0,
                                            const double adx1, const double adx2) {
  // Written as step + a kappa correction, NOT as (1 + kappa/2)*sum, so that at kappa = 0 the
  // correction is exactly +0.0 and the result is BITWISE SnDiagonal. Test 8(a) checks that
  // bitwise, and it failed on the (1 + kappa/2) form by 1.8e-15 -- a different summation
  // order, which is precisely the kind of drift that makes "step and sou at kappa=0 agree"
  // a claim that has to be tested rather than read off the algebra.
  return rdt + chi + adx0 + adx1 + adx2 + 0.5 * kappa * (adx0 + adx1 + adx2);
}

//! `iup[d]` is the first upwind neighbour and `iup2[d]` the second (two cells upwind) in
//! direction d. At kappa = 0 the iup2 terms carry a zero coefficient and this is SnRhs.
KOKKOS_INLINE_FUNCTION double SnSouRhs(const double in_old, const double rdt, const double src,
                                       const double kappa, const double adx0,
                                       const double adx1, const double adx2,
                                       const double iup0, const double iup1, const double iup2,
                                       const double iup2_0, const double iup2_1,
                                       const double iup2_2) {
  // Same reason as SnSouDiagonal: step's b plus a kappa correction, so kappa = 0 is bitwise
  // SnRhs. (1 + kappa) u - (kappa/2) v  ==  u + kappa*(u - v/2).
  return SnRhs(in_old, rdt, src, adx0, adx1, adx2, iup0, iup1, iup2) +
         kappa * (adx0 * (iup0 - 0.5 * iup2_0) + adx1 * (iup1 - 0.5 * iup2_1) +
                  adx2 * (iup2 - 0.5 * iup2_2));
}

KOKKOS_INLINE_FUNCTION double SnSouCell(const double in_old, const double rdt,
                                        const double chi, const double src, const double kappa,
                                        const double adx0, const double adx1,
                                        const double adx2, const double iup0,
                                        const double iup1, const double iup2,
                                        const double iup2_0, const double iup2_1,
                                        const double iup2_2) {
  return SnSouRhs(in_old, rdt, src, kappa, adx0, adx1, adx2, iup0, iup1, iup2, iup2_0, iup2_1,
                  iup2_2) /
         SnSouDiagonal(rdt, chi, kappa, adx0, adx1, adx2);
}

//----------------------------------------------------------------------------------------
//! \brief Off-diagonal row sum. TEST-ONLY -- the sweep kernel never calls this, so a defect
//! in it cannot affect a run, and the only thing that can catch one is a test that computes
//! the same sum independently (tests/test_sn_sweep.cpp check 1 now does).
KOKKOS_INLINE_FUNCTION double SnOffDiagonal(const double adx0, const double adx1,
                                            const double adx2) {
  return adx0 + adx1 + adx2;
}

} // namespace Sn
} // namespace Radiation

#ifdef SN_SWEEP_DEFINED_KIF
#undef KOKKOS_INLINE_FUNCTION
#undef SN_SWEEP_DEFINED_KIF
#endif

#endif // RADIATION_RADIATION_SN_SWEEP_HPP_
