//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::GLMMHD {

template <bool extended>
void DednerSource(MeshData<Real> *md, const Real beta_dt) {
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto c_h = hydro_pkg->Param<Real>("c_h");
  const auto mindx = hydro_pkg->Param<Real>("mindx");
  const auto alpha = hydro_pkg->Param<Real>("glmmhd_alpha");

  // Using an alpha based parameter here following Mignone & Tzeferacos 2010 (27)
  //
  // AUDIT F-11 (2026-08-28) -- `mindx` IS THE GLOBAL MESH MINIMUM, APPLIED TO EVERY BLOCK.
  // Examined and DELIBERATELY LEFT AS IS; the numbers are recorded so a future change can be
  // gated on them rather than argued.
  //
  // The damping per step is a pure number (alpha*cfl*dt/dt_hyp), identical on every level, but
  // the damping LENGTH in LOCAL cells is mindx/(alpha*dx_L) = 10/2^(7-L) on the production
  // mesh: 10.0 cells at L7, 1.25 at L4, 0.078 at L0. So on coarse levels psi is annihilated
  // well inside one cell and the HYPERBOLIC transport half of the mixed cleaning is
  // effectively off there; only the parabolic (local) half survives.
  //
  // Measured mean |div B| dx/|B| by level (audit section 2): 1.51e-04 (L0), 5.79e-05, 2.44e-05,
  // 1.93e-05, 1.60e-05, 1.18e-05 (L5), then 4.23e-05, 2.56e-04 at L6/L7. L0-L5 improve
  // monotonically with depth, consistent with the damping-length argument -- but L0 is ALSO
  // the turbulent envelope, so the two explanations are CONFOUNDED and this is not proof.
  //
  // WHY NOT "FIX" IT: using the LOCAL dx would give a damping length of 1/alpha = 10 cells on
  // EVERY level, i.e. WEAKER damping on the coarse levels, which are already the worst
  // (L0 mean 1.5e-04). The global-mindx choice over-damps psi where the mesh is coarse, which
  // is the safe direction. Mesh-wide the worst cell is 2.5e-02 and the volume mean 1.5e-04 --
  // good for a cell-centred GLM scheme. Undocumented choice, not a defect.
  const auto coeff = std::exp(-alpha * c_h * beta_dt / mindx);

  // EOS E-F3 (fixed 2026-10-07, validation_tests/dual_energy): in 1D the x2-derivatives read j+-1, which do not
  // exist (segfault); j_offset = 0 there makes them exactly zero, as k_offset does for x3 in 2D. 3D: unchanged.
  const int j_offset = (cons_pack.GetNdim() < 2) ? 0 : 1;
  int k_offset = 1;
  // In 2D, offset is 0 so that the second order x3-derivatives are zero
  // Should (untested) not introduce a performance penalty, as vals at k,j,i are
  // in the cache line from x1 derivative.
  if (cons_pack.GetNdim() < 3) {
    k_offset = 0;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DednerSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Use extended source terms that is non-conservative but has better
        // stability properties as reported by Dedner+ and M&T
        // TODO(pgrete) Once nvcc is fixed this could be constexpr if again
        if (extended) {
          auto &cons = cons_pack(b);
          const auto &prim = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real divB =
              0.5 * ((prim(IB1, k, j, i + 1) - prim(IB1, k, j, i - 1)) /
                         coords.Dxc<1>(k, j, i) +
                     (prim(IB2, k, j + j_offset, i) - prim(IB2, k, j - j_offset, i)) /
                         coords.Dxc<2>(k, j, i) +
                     (prim(IB3, k + k_offset, j, i) - prim(IB3, k - k_offset, j, i)) /
                         coords.Dxc<3>(k, j, i));
          cons(IM1, k, j, i) -= beta_dt * divB * prim(IB1, k, j, i);
          cons(IM2, k, j, i) -= beta_dt * divB * prim(IB2, k, j, i);
          cons(IM3, k, j, i) -= beta_dt * divB * prim(IB3, k, j, i);
          cons(IEN, k, j, i) -=
              0.5 * beta_dt *
              (prim(IB1, k, j, i) * (prim(IPS, k, j, i + 1) - prim(IPS, k, j, i - 1)) /
                   coords.Dxc<1>(k, j, i) +
               prim(IB2, k, j, i) * (prim(IPS, k, j + j_offset, i) - prim(IPS, k, j - j_offset, i)) /
                   coords.Dxc<2>(k, j, i) +
               prim(IB3, k, j, i) *
                   (prim(IPS, k + k_offset, j, i) - prim(IPS, k - k_offset, j, i)) /
                   coords.Dxc<3>(k, j, i));
        }
        cons_pack(b, IPS, k, j, i) *= coeff;
      });
}
template void DednerSource<true>(MeshData<Real> *md, const Real beta_dt);
template void DednerSource<false>(MeshData<Real> *md, const Real beta_dt);

} // namespace Hydro::GLMMHD