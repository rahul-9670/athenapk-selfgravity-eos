#ifndef RADIATION_RADIATION_SN_RESTRICT_HPP_
#define RADIATION_RADIATION_SN_RESTRICT_HPP_
//========================================================================================
// AthenaPK - discrete-ordinates (Sn) transport: FACE-LAYER RESTRICTION for coarse-fine ghosts.
//
// WHY THIS EXISTS -- measured, twice, before a line of it was written.
//
// Gate G6 found the Sn solve losing 2.085e-03 of the radiation energy at a static coarse-fine
// interface, against 1.159e-07 on a uniform mesh. The obvious reading -- "the design fuses its
// fluxes and therefore has no flux register, so of course it cannot conserve" -- is WRONG, and
// acting on it would have cost the whole memory argument of SN_TRANSPORT_DESIGN.md section 3
// (fused fluxes are what take the production mesh from 4 nodes to 2).
//
//   EXPERIMENT A (job 2730529): a UNIFORM field crosses the seam at 1.9e-10, and the drift is
//   IDENTICAL at sn_rel_tol 1e-10 and 1e-13 (2.085e-03 both, ratio 1.0x). So it is neither a
//   broken exchange nor an unconverged interface, and it requires a gradient.
//
//   EXPERIMENT B (job 2730632): rotate the refined SLAB and hold the field fixed -- same 288
//   blocks, same 69 cycles, same interface area:
//       gradient NORMAL to the seam      7.335e-03
//       gradient TANGENTIAL to the seam  1.504e-07
//       no seam at all (control)         1.159e-07
//   A factor of 48,785. A tangential seam is indistinguishable from no seam.
//
// THE MECHANISM. parthenon's RestrictAverage (prolong_restrict/pr_ops.hpp:105) fills one coarse
// ghost cell by averaging the 2x2x2 = 8 fine cells that map into it -- verified in source,
// `vol[2][2][2]`. But the energy crossing that face is carried by only the 4 fine cells in the
// LAYER TOUCHING IT. The two averages differ by exactly the variation normal to the seam, which
// is why a uniform field is fine, a tangential gradient is fine, and only a normal gradient
// leaks. This operator restricts from that face-adjacent layer instead.
//
// *** IT DOES NOT WORK, AND THE REASON IS STRUCTURAL. DEFAULT OFF. ***
//
// EXPERIMENT C (job 2731004) ran this operator against the stock one on the worst case -- the
// slab whose seam is normal to the gradient -- and got 7.335e-03 BOTH WAYS, identical to every
// digit. The operator is bit-identical to RestrictAverage in practice, because its special case
// never fires. Why, established by reading parthenon rather than by guessing:
//
//   RESTRICTION RUNS ON THE SENDER, OVER ITS OWN INTERIOR. `ProResInfo::GetInteriorRestrict`
//   (bvals/comms/bnd_info.cpp:390) builds the index range with
//   `IndexRangeType::InteriorSend`: the FINE block restricts its own interior cells into its
//   coarse buffer and ships that. So the cells this operator sees are never outside the
//   interior bounds, the `ci < cib.s` test below is always false, and the face branch is dead.
//
// The direction is not simply missing from the signature -- it is missing from the DESIGN.
// `ProResInfo` does carry a `CoordinateDirection dir`, but `Do()` is never given it, and adding
// it would not be enough: every destination buffer restricts into the SAME per-variable coarse
// buffer (`out.coarse = v->coarse_s.Get()`), and a fine cell near a corner belongs to the
// x-face send, the y-face send and the edge send at once. A direction-dependent average would
// need a different value in one slot for each destination. Fixing that means per-destination
// coarse buffers, i.e. surgery on parthenon's communication layer, not a custom stencil.
//
// KEPT, DEFAULT OFF, because a negative result that took two experiments to establish is worth
// more in the tree than deleted: it stops the next person spending the same week on the same
// idea. `radiation/sn_face_layer_restrict` enables it; nothing should.
//
// WHAT WOULD HAVE COST NOTHING, had it worked. No extra storage, no extra communication, no
// flux register -- a different average over data already being exchanged. That is why it was
// worth trying first.
//
// WHY IT IS WELL POSED. A face-neighbour ghost is read by the sweep ONLY as the upwind
// neighbour in its own normal direction -- the stencil is one cell and face-only
// (i+-1, j+-1, k+-1), so a cell at an interior corner reads two DIFFERENT face ghosts rather
// than one edge ghost. One value per ghost therefore suffices, and edge/corner ghosts are never
// read at all.
//
// WHICH LAYER. With `i = (ci - cib.s)*2 + ib.s`, a ghost on the LOW side (ci < cib.s) has the
// interface at its HIGH-x edge, so the adjacent fine layer is oi = 1; a ghost on the HIGH side
// (ci > cib.e) has it at the low edge, so the layer is oi = 0. Checked against the index
// algebra: for ci = cib.e + 1, i = ib.e + 1, and the cell touching the interior is ib.e + 1
// itself, i.e. oi = 0.
//
// FALLS BACK to the full 8-cell average whenever the target is NOT a single-face ghost: the
// interior restriction used when a block is derefined (no direction out of bounds), and
// edge/corner ghosts (two or three out of bounds), which nothing reads. That keeps this
// operator identical to RestrictAverage everywhere its special case does not apply.
//
// SCOPE: registered on `sn.u` ONLY. rad::Er keeps RestrictAverage because M1 carries a real
// flux register (Metadata::WithFluxes + LoadAndSendFluxCorrections) and does not have this
// problem.
//========================================================================================
#include <coordinates/coordinates.hpp>
#include <interface/variable_state.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/domain.hpp>

namespace Radiation {
namespace Sn {

struct RestrictAverageFaceLayer {
  static constexpr bool OperationRequired(parthenon::TopologicalElement fel,
                                          parthenon::TopologicalElement cel) {
    return fel == cel;
  }

  template <int DIM,
            parthenon::TopologicalElement el = parthenon::TopologicalElement::CC,
            parthenon::TopologicalElement = parthenon::TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const parthenon::IndexRange &ckb, const parthenon::IndexRange &cjb,
     const parthenon::IndexRange &cib, const parthenon::IndexRange &kb,
     const parthenon::IndexRange &jb, const parthenon::IndexRange &ib,
     const parthenon::Coordinates_t &coords,
     const parthenon::Coordinates_t & /*coarse_coords*/,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState> *pcoarse,
     const parthenon::ParArrayND<parthenon::Real, parthenon::VariableState> *pfine) {
    using TE = parthenon::TopologicalElement;
    using parthenon::Real;
    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3 || el == TE::E1);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1 || el == TE::E2);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::E3);
    constexpr int element_idx = static_cast<int>(el) % 3;

    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    const int i = (DIM > 0) ? (ci - cib.s) * 2 + ib.s : ib.s;
    const int j = (DIM > 1) ? (cj - cjb.s) * 2 + jb.s : jb.s;
    const int k = (DIM > 2) ? (ck - ckb.s) * 2 + kb.s : kb.s;

    // Is this a single-face ghost, and on which side?
    const bool lo1 = INCLUDE_X1 && (ci < cib.s);
    const bool hi1 = INCLUDE_X1 && (ci > cib.e);
    const bool lo2 = INCLUDE_X2 && (cj < cjb.s);
    const bool hi2 = INCLUDE_X2 && (cj > cjb.e);
    const bool lo3 = INCLUDE_X3 && (ck < ckb.s);
    const bool hi3 = INCLUDE_X3 && (ck > ckb.e);
    const int nout = static_cast<int>(lo1 || hi1) + static_cast<int>(lo2 || hi2) +
                     static_cast<int>(lo3 || hi3);
    const bool face = (nout == 1);

    // Default: the full cube, i.e. exactly RestrictAverage. Collapsed to the face-adjacent
    // layer in the one direction that is out of bounds, and only for a single-face ghost.
    int oi_s = 0, oi_e = INCLUDE_X1 ? 1 : 0;
    int oj_s = 0, oj_e = INCLUDE_X2 ? 1 : 0;
    int ok_s = 0, ok_e = INCLUDE_X3 ? 1 : 0;
    if (face && (lo1 || hi1)) oi_s = oi_e = (lo1 ? 1 : 0);
    if (face && (lo2 || hi2)) oj_s = oj_e = (lo2 ? 1 : 0);
    if (face && (lo3 || hi3)) ok_s = ok_e = (lo3 ? 1 : 0);

    Real tvol = 0.0, tsum = 0.0;
    for (int ok = ok_s; ok <= ok_e; ++ok) {
      for (int oj = oj_s; oj <= oj_e; ++oj) {
        for (int oi = oi_s; oi <= oi_e; ++oi) {
          const Real v = coords.Volume<el>(k + ok, j + oj, i + oi);
          tvol += v;
          tsum += v * fine(element_idx, l, m, n, k + ok, j + oj, i + oi);
        }
      }
    }
    coarse(element_idx, l, m, n, ck, cj, ci) = (tvol > 0.0) ? (tsum / tvol) : 0.0;
  }
};

} // namespace Sn
} // namespace Radiation

#endif // RADIATION_RADIATION_SN_RESTRICT_HPP_
