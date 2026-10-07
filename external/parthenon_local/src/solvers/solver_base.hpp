//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef SOLVERS_SOLVER_BASE_HPP_
#define SOLVERS_SOLVER_BASE_HPP_

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
namespace parthenon {

namespace solvers {

// Used for checking if a given equations class has a SetBoundary function
template <typename T, typename = void>
struct has_SetBoundary : std::false_type {};

template <typename T>
struct has_SetBoundary<
    T, std::void_t<decltype(std::declval<T>().SetBoundary(
           std::declval<std::shared_ptr<MeshData<Real>> &>(), std::declval<bool>()))>>
    : std::true_type {};

// Used for checking if a given equations class has a SetBoundary function
template <typename T, typename FieldTL, typename = void>
struct has_Restrict : std::false_type {};

template <typename T, typename FieldTL>
struct has_Restrict<T, FieldTL,
                    std::void_t<decltype(std::declval<T>().template Restrict<FieldTL>(
                        std::declval<TaskList &>(), std::declval<TaskID>(),
                        std::declval<std::shared_ptr<MeshData<Real>> &>()))>>
    : std::true_type {};

// Solver base class
class SolverBase {
 public:
  SolverBase(const std::string &container_base, const std::string &container_u,
             const std::string &container_rhs)
      : container_base(container_base), container_u(container_u),
        container_rhs(container_rhs) {}

  virtual ~SolverBase() {}

  virtual TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition,
                               Mesh *pmesh) = 0;
  virtual TaskID AddTasks(TaskList &tl, TaskID dependence, int partition,
                          Mesh *pmesh) = 0;

  // Provide access to the underlying matrix operator for convenience
  virtual TaskID Ax(TaskList &tl, TaskID dependence,
                    std::shared_ptr<MeshData<Real>> &md_mat,
                    std::shared_ptr<MeshData<Real>> &md_in,
                    std::shared_ptr<MeshData<Real>> &md_out) = 0;

  virtual void SetConstantProlongation(bool const_pro) {}

  Real GetInitialResidual() const { return initial_residual; }
  Real GetFinalResidual() const { return final_residual; }
  // AthenaPK patch, AUDIT F-14 (2026-08-28): the residual DIVIDED BY the tolerance actually
  // applied. GetFinalResidual() is the ABSOLUTE rms residual while the convergence test under
  // relative_residual = true compares it against rel_tol = rel_tol_coeff * sqrt(rhs2/ncells),
  // so the two are different quantities and comparing the reported residual with the deck's
  // residual_tolerance is meaningless (14 of 539 production rows read as violations while
  // being fully converged). This ratio is directly comparable to 1. -1 = not set by the solver.
  Real GetFinalResidualRel() const { return final_residual_rel; }
  int GetFinalIterations() const { return final_iteration + 1; }

  // AthenaPK WP-5/B2 (2026-08-02): the iterative solvers return the SAME TaskStatus::complete
  // whether a tolerance was met or `max_iterations` was exhausted, and report neither -- so a
  // gravity solve that silently gave up is indistinguishable from one that converged. This flag
  // records WHICH of the two happened; the solver sets it where it decides to stop. Defaults to
  // true so any solver that does not yet set it behaves exactly as before (no false alarms).
  bool GetConverged() const { return converged; }

  void SetRHSContainerLabel(const std::string &rhs) { container_rhs = rhs; }
  const std::string &GetBaseContainerLabel() const { return container_base; }
  const std::string &GetRHSContainerLabel() const { return container_rhs; }
  const std::string &GetSolutionContainerLabel() const { return container_u; }

  const std::vector<std::string> &GetFieldLabels() const { return sol_fields; }

  bool initial_guess_is_zero{false};

  static inline TimingAccumulatorDictionary solver_timings;

 protected:
  // Labels of all fields included in the vector
  std::vector<std::string> sol_fields;
  // Name of user defined container that should contain information required to
  // calculate the matrix part of the matrix vector product
  std::string container_base;
  // User defined container in which the solution will reside, only needs to contain
  // sol_fields
  // TODO(LFR): Also allow for an initial guess to come in here
  std::string container_u;
  // User defined container containing the rhs vector, only needs to contain sol_fields
  std::string container_rhs;

  Real initial_residual{-1.0};
  Real final_residual;
  Real final_residual_rel{-1.0}; // AthenaPK patch (F-14): final_residual / applied tolerance
  int final_iteration;
  // See GetConverged(). True = a tolerance was met; false = max_iterations was exhausted.
  //
  // B2-FIX (2026-08-15): default flipped true -> false. "Defaults to true so any solver that
  // does not yet set it behaves exactly as before (no false alarms)" was the wrong trade: it
  // made the SILENT failure the default, and combined with an unreachable bail-out branch it
  // meant the flag could only ever read true (job 2522013: a solve capped at 2 iterations,
  // rms residual 1.953e-02 against a converged 9.384e-07, reported converged). Defaulting
  // false makes an un-updating solver report non-convergence -- noisy, but a wrong alarm is
  // visible where a missing one is not. Every solver in this directory now sets it explicitly
  // at solve start and at the decision point, so the default is a backstop, not the behaviour.
  bool converged{false};
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_BASE_HPP_
