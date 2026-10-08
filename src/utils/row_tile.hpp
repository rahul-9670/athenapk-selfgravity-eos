//========================================================================================
// AthenaPK - inner MDRange tile of the full-row kernels (validation_tests/dual_energy gap closure, E-F8b, 2026-10-07).
//========================================================================================
#ifndef UTILS_ROW_TILE_HPP_
#define UTILS_ROW_TILE_HPP_

#include <Kokkos_Core.hpp>

//! These kernels give Kokkos an explicit tile {1, 1, 1, nx1}: one thread block per row of a meshblock. On a GPU that is
//! wrong for long rows (measured on GH200, validation_tests/dual_energy G-series):
//!   nx1 > 2048: Kokkos aborts ("MDRange tile dims exceed maximum number of threads per block");
//!   nx1 = 1024: no error, but the reduction does not execute -- EstimateHyperbolicTimestep returned DBL_MIN, the CFL
//!               limit never acted and a Noh test blew the box to vacuum while exiting rc 0.
//! Device builds cap the row tile at 256 (Parthenon's md pattern is capped identically, kokkos_abstraction.hpp); a row
//! of nx1 <= 256 (production: 32) gives the unchanged policy. Host builds return nx1 (unchanged tiles and order).
KOKKOS_INLINE_FUNCTION int RowTile(const int nx1) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  return nx1 < 256 ? nx1 : 256;
#else
  return nx1;
#endif
}

#endif // UTILS_ROW_TILE_HPP_
