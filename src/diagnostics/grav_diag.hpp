//========================================================================================
// AthenaPK - self-gravity (Poisson) solver convergence diagnostics.
//
// VALIDATION WP-5. The production deck runs the gravity solve with
//   <self_gravity/solver_params> max_iterations = 200, residual_tolerance = 1.0e-6
// and across 1975 cycles of runs/convergence_ladder/nj8/run.log there are ZERO
// occurrences of "residual", "converge" or an iteration count. We therefore could not
// tell whether the Poisson solve converges or silently bails at max_iterations.
//
// It bails SILENTLY by construction. In parthenon's
// external/parthenon/src/solvers/bicgstab_solver.hpp the completion task does
//
//     if ((rms_res < rel_tol) || (rms_res < *abs_res_tol) ||
//         (solver->iter_counter >= max_iter)) {
//       ...
//       return TaskStatus::complete;      // <-- SAME return for converged and bailed
//     }
//
// i.e. "converged" and "hit the iteration ceiling" are indistinguishable to every caller
// and neither is reported. This header adds the missing instrument.
//
// TOLERANCE SEMANTICS (verified from bicgstab_solver.hpp:49-66): the deck does not set
// `relative_residual`, so it defaults to false and the else-branch assigns
//     absolute_residual_tolerance = residual_tolerance = 1e-6
//     relative_residual_tolerance = 0.0
// The production criterion is therefore an ABSOLUTE tolerance on
//     rms_res = sqrt( sum(r^2) / pmesh->GetTotalCells() )
// of the Poisson residual for  grad^2 phi = 4 pi G (rho - rho_mean),  four_pi_G = 1.
// Because the RHS grows by many decades as the core collapses while the tolerance stays
// fixed, the *relative* accuracy demanded of the solve tightens monotonically through the
// run -- which is exactly the regime in which a fixed iteration ceiling starts to bite.
// That is the hypothesis grav-nonconv is built to test.
//
// Columns (all UserHistoryOperation::max -- the underlying scalars come from a global
// all-reduce inside the solver and are identical on every rank/partition, so max returns
// the value itself rather than a partition-count-dependent sum):
//   grav-iters    = BiCGSTAB iterations used by the most recent solve
//   grav-res      = final rms residual of the most recent solve
//   grav-nonconv  = 1.0 if that solve stopped because it hit max_iterations, else 0.0
//
// Gated on <self_gravity> solver_diag (default OFF). When off, nothing is registered and
// no task is added => bit-identical. When on, the added task only READS solver state and
// writes to package Params; it never touches field data, so it is bit-identical then too.
//========================================================================================
#ifndef DIAGNOSTICS_GRAV_DIAG_HPP_
#define DIAGNOSTICS_GRAV_DIAG_HPP_

#include <parthenon/package.hpp>
#include <solvers/solver_base.hpp>

using namespace parthenon::package::prelude;

namespace Diagnostics {

//! Which solver-convergence scalar to report.
enum class GravDiag {
  iters,   // iterations used by the most recent solve
  res,     // final rms residual of the most recent solve (ABSOLUTE -- see res_rel)
  res_rel, // AUDIT F-14: that residual divided by the tolerance actually applied, so it is
           // directly comparable to 1. The deck's residual_tolerance is RELATIVE while
           // `res` is ABSOLUTE; comparing those two is meaningless.
  nonconv, // 1.0 if the most recent solve hit max_iterations, else 0.0
  // ---- multipole exterior-validity (2026-08-15) ----
  // A multipole expansion of the interior mass is only valid at field points OUTSIDE the mass
  // distribution. The multipole BC evaluates it on the DOMAIN FACES, and the closest such
  // point sits R_ins = min(half-width) from the expansion centre. Any mass beyond R_ins is
  // mass the face-centre expansion is not entitled to expand -- and nothing in the code
  // checked for it. Truncation order (this code carries monopole + quadrupole, dipole zero by
  // construction about the COM) is the SECOND-order question; validity is the first.
  //
  // The gate this project already passed -- 0.54% against an analytic potential -- cannot see
  // this, because it used a COMPACT source. A BE sphere plus padding is not compact relative
  // to its box, so the early collapse is exactly where the assumption is weakest.
  //
  // Interpretation: <~1e-2 and monopole+quadrupole is comfortably enough; O(0.1-1) and the BC
  // is being asked for something an exterior expansion cannot give, and the answer is a bigger
  // box (or a discrete Green's-function boundary lift), NOT more multipole terms.
  mp_extmass, // fraction of total mass outside the largest sphere inscribed in the domain
  mp_comoff   // |COM - domain centre| / R_ins; says whether the expansion centre has drifted
};

//! Read back one of the recorded scalars. Reads package Params only -- `md` is used
//! solely to reach the Mesh, and no field data is touched.
Real GravDiagReport(MeshData<Real> *md, GravDiag which);

//! Post-solve hook: pull final_residual / final_iteration off the solver and record them
//! into the self_gravity package Params, warning on rank 0 the first time (and every
//! `warn_stride` occurrences thereafter) that a solve hits the iteration ceiling.
parthenon::TaskStatus GravDiagRecord(Mesh *pmesh);

//! Warn (rank 0, geometrically backed off) when the Poisson solve exhausted max_iterations.
//! Registered UNCONDITIONALLY -- unlike GravDiagRecord, which is gated on `solver_diag`
//! because it feeds hst columns. B2: silent non-convergence must not depend on a debug flag.
parthenon::TaskStatus GravConvergenceWarn(Mesh *pmesh);

} // namespace Diagnostics

#endif // DIAGNOSTICS_GRAV_DIAG_HPP_
