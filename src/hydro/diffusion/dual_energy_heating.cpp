//========================================================================================
// AthenaPK - dissipative heating for the dual-energy variable (increment 2).
//========================================================================================
//! \file dual_energy_heating.cpp
//! \brief Ohmic and ambipolar dissipation added to the advected internal energy.
//!
//! WHY. With hydro/dual_energy on, every operator that changes the gas thermal energy must change
//! the advected internal energy by the same amount, or the two drift apart and the switch in
//! ConsToPrim starts handing out a stale value. The hyperbolic part, the pdV work, the floors and
//! the radiation thermal exchange are increment 1. This is the non-ideal part: the RKL2 (or
//! unsplit) diffusion update changes the TOTAL energy through the Poynting flux divergence, of
//! which the irreversible share is the Joule/ambipolar dissipation
//!     Q = eta_O |J|^2 + eta_A |J_perp|^2,    J = curl B,  J_perp = J - (J.b) b
//! (the Hall EMF does no work: J . E_H = 0, so it contributes nothing here).
//!
//! WHEN IT IS APPLIED. Once per Strang half-step, AFTER the half, with that half's tau, the frozen
//! start-of-half eta cache (diffusion/rkl2_freeze_eta) and the END-of-half B: the task runs after
//! the last RKL2 stage's FillDerived, so `prim` already holds the diffused field. (Corrected by
//! audit P-03, 2026-09-23 -- this comment used to say "B from the start of the half".) In a stiff
//! cell (t_diff << tau) J has decayed by then and the advected energy is under-heated; see
//! hydro/dual_energy_heat_rkl2 below for the version integrated inside the RKL2 stages.
//!
//! WHAT IT DOES NOT DO. The total energy is untouched here: it already carries the full
//! (reversible + irreversible) diffusion update from the flux divergence. This only tells the
//! internal-energy variable what share of that was heat.
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../../main.hpp"
#include "diffusion.hpp"

using namespace parthenon::package::prelude;

//! \brief Add eta_O |J|^2 + eta_A |J_perp|^2 times `dt` to the dual-energy variable.
//!
//! J from `prim`'s B and eta from the cache: the same two the RKL2 half just deposited its
//! fluxes from.
TaskStatus DualEnergyDiffusionHeating(MeshData<Real> *md, const Real dt) {
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const int idual = pkg->Param<int>("dual_idx");
  if (idual < 0 || dt == 0.0) return TaskStatus::complete;
  const bool have_cache = pkg->Param<bool>("nonideal_eta_cache");
  PARTHENON_REQUIRE(have_cache,
                    "hydro/dual_energy with non-ideal terms needs diffusion/eta_cache=true: the "
                    "heating is evaluated from the cached cell-centred eta.");
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto eta_pack = md->PackVariables(std::vector<std::string>{"nonideal_eta"});
  const int iB = IB1;
  const int iO = NonidealEtaIdx::O;
  const int iA = NonidealEtaIdx::A;
  const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // validation_tests/dual_energy DE-F5 (2026-10-06): y/z neighbours exist only along active dimensions. In 1D/2D the
  // j+-1 / k+-1 reads below fell outside the field arrays (segfault at cycle 0); with dj/dk = 0 those differences are
  // exactly 0 (no variation along an inactive dimension). 3D (production): dj = dk = 1, the same arithmetic.
  const int ndim_h = md->GetBlockData(0)->GetBlockPointer()->pmy_mesh->ndim;
  const int dj = ndim_h > 1, dk = ndim_h > 2;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DualEnergyDiffusionHeating", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const auto &eta = eta_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        // J = curl B from the cell-centred field (same centred differences the cell-centred
        // diffusivity diagnostics use).
        const Real idx1 = 1.0 / (2.0 * coords.template Dxc<1>(k, j, i));
        const Real idx2 = 1.0 / (2.0 * coords.template Dxc<2>(k, j, i));
        const Real idx3 = 1.0 / (2.0 * coords.template Dxc<3>(k, j, i));
        const Real jx = (prim(iB + 2, k, j + dj, i) - prim(iB + 2, k, j - dj, i)) * idx2 -
                        (prim(iB + 1, k + dk, j, i) - prim(iB + 1, k - dk, j, i)) * idx3;
        const Real jy = (prim(iB + 0, k + dk, j, i) - prim(iB + 0, k - dk, j, i)) * idx3 -
                        (prim(iB + 2, k, j, i + 1) - prim(iB + 2, k, j, i - 1)) * idx1;
        const Real jz = (prim(iB + 1, k, j, i + 1) - prim(iB + 1, k, j, i - 1)) * idx1 -
                        (prim(iB + 0, k, j + dj, i) - prim(iB + 0, k, j - dj, i)) * idx2;
        const Real j2 = jx * jx + jy * jy + jz * jz;
        const Real bx = prim(iB + 0, k, j, i), by = prim(iB + 1, k, j, i),
                   bz = prim(iB + 2, k, j, i);
        const Real b2 = bx * bx + by * by + bz * bz;
        const Real jpar = (b2 > 0.0) ? (jx * bx + jy * by + jz * bz) / std::sqrt(b2) : 0.0;
        const Real jperp2 = (j2 > jpar * jpar) ? j2 - jpar * jpar : 0.0;
        const Real q = eta(iO, k, j, i) * j2 + eta(iA, k, j, i) * jperp2;
        cons(idual, k, j, i) += dt * q;
      });
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! hydro/dual_energy_heat_rkl2 (audit P-03, 2026-09-23): the Ohmic + ambipolar heating as a SOURCE
//! of the dual-energy component INSIDE the RKL2 recurrence, instead of once after the half.
//!
//! The RKL2 half computes Y_s = Y_0 + tau * sum_j w_j M(Y_j) for the field; giving the dual component
//! M_dual(Y) = q(Y) = eta_O |J|^2 + eta_A |J_perp|^2 with the SAME stage states and weights integrates
//! the heat along the same path that dissipates B, so a stiff mode that decays inside the half heats
//! the gas by what it actually lost rather than by J(end)^2 * tau (~0) or J(start)^2 * tau (too much).
//! J from the stage's `prim` B (centred differences, the stencil DualEnergyDiffusionHeating uses) and
//! eta from the frozen cache. Plumbing (see AddSTSTasks): q(Y_0) is added to MY0's dual component
//! (its flux divergence is identically 0), so RKL2StepFirst/Other carry it through the mu~ and
//! gamma~ terms unchanged; the mu~_j tau q(Y_{j-1}) term of each later stage is added by
//! DualHeatStage right after RKL2StepOther, while `prim` still describes Y_{j-1}.
namespace {
KOKKOS_INLINE_FUNCTION Real DualJouleQ(const parthenon::VariablePack<Real> &prim,
                                       const parthenon::VariablePack<Real> &eta,
                                       const parthenon::Coordinates_t &coords, const int k,
                                       const int j, const int i, const int dj, const int dk) {
  const Real idx1 = 1.0 / (2.0 * coords.template Dxc<1>(k, j, i));
  const Real idx2 = 1.0 / (2.0 * coords.template Dxc<2>(k, j, i));
  const Real idx3 = 1.0 / (2.0 * coords.template Dxc<3>(k, j, i));
  // dj/dk = 1 along active dimensions, 0 otherwise (DE-F5, see DualEnergyDiffusionHeating)
  const Real jx = (prim(IB3, k, j + dj, i) - prim(IB3, k, j - dj, i)) * idx2 -
                  (prim(IB2, k + dk, j, i) - prim(IB2, k - dk, j, i)) * idx3;
  const Real jy = (prim(IB1, k + dk, j, i) - prim(IB1, k - dk, j, i)) * idx3 -
                  (prim(IB3, k, j, i + 1) - prim(IB3, k, j, i - 1)) * idx1;
  const Real jz = (prim(IB2, k, j, i + 1) - prim(IB2, k, j, i - 1)) * idx1 -
                  (prim(IB1, k, j + dj, i) - prim(IB1, k, j - dj, i)) * idx2;
  const Real j2 = jx * jx + jy * jy + jz * jz;
  const Real bx = prim(IB1, k, j, i), by = prim(IB2, k, j, i), bz = prim(IB3, k, j, i);
  const Real b2 = bx * bx + by * by + bz * bz;
  const Real jpar = (b2 > 0.0) ? (jx * bx + jy * by + jz * bz) / std::sqrt(b2) : 0.0;
  const Real jperp2 = (j2 > jpar * jpar) ? j2 - jpar * jpar : 0.0;
  return eta(NonidealEtaIdx::O, k, j, i) * j2 + eta(NonidealEtaIdx::A, k, j, i) * jperp2;
}

//! hydro/dual_energy_heat_face (validation_tests/dual_energy DE-F6, 2026-10-07; default false = the centred-J q above).
//!
//! WHY. q = eta |J_c|^2 with the cell-centred 2-dx current is NOT the dissipation of the discrete diffusion update:
//! for a mode k, J_c^2 = cos^2(k dx/2) x the face-based operator's (k_eff B)^2, so a switched cell under-heats by
//! (k dx)^2/4 (D5a: -6.1e-4 at N 128, -2.4e-3 at N 64), a 4-cell mode by 50 %, a Nyquist mode by 100 % -- and the
//! missing energy is lost (E is rebuilt from the dual energy in switched cells).
//!
//! WHAT. The dissipation of the update that was actually applied, from the stage's own B diffusion fluxes. With the
//! flux form dB/dt = -div F_B, summation by parts gives dME/dt = sum_faces F_f . (B_R - B_L)/dx, so every face
//! dissipates D_f = -F_f . (B_R - B_L)/dx >= 0 (per unit volume), whatever eta form (Ohmic, ambipolar, tensor,
//! variable) produced F. Each cell receives half of each of its 2 ndim faces: q_i = 1/2 sum_faces D_f. The
//! cell sum equals the discrete magnetic-energy loss rate of the stage exactly (uniform mesh; at fine/coarse faces
//! the corrected coarse flux is used, approximate there). Requires the flux arrays to hold ONLY the diffusion
//! fluxes of the state in `prim` -- true for both calls (DualHeatIntoMY0 after the init CalcDiffFluxes, DualHeatStage
//! after RKL2StepOther, which leaves the flux arrays and `prim` at Y_{j-1}).
template <typename FluxPack>
KOKKOS_INLINE_FUNCTION Real DualFaceQ(const parthenon::VariablePack<Real> &prim, const FluxPack &cfb,
                                      const parthenon::Coordinates_t &coords, const int k, const int j, const int i,
                                      const int ndim) {
  Real q = 0.0;
  for (int d = 1; d <= ndim; ++d) {
    const int di = (d == 1), dj = (d == 2), dk = (d == 3);
    const Real idx = 1.0 / (d == 1 ? coords.template Dxc<1>(k, j, i)
                                   : (d == 2 ? coords.template Dxc<2>(k, j, i) : coords.template Dxc<3>(k, j, i)));
    Real dlo = 0.0, dhi = 0.0;
    for (int c = 0; c < 3; ++c) {
      const Real bm = prim(IB1 + c, k - dk, j - dj, i - di), b0 = prim(IB1 + c, k, j, i),
                 bp = prim(IB1 + c, k + dk, j + dj, i + di);
      dlo -= cfb.flux(d, IB1 + c, k, j, i) * (b0 - bm) * idx;
      dhi -= cfb.flux(d, IB1 + c, k + dk, j + dj, i + di) * (bp - b0) * idx;
    }
    q += 0.5 * (dlo + dhi);
  }
  return q;
}

//! dst.cons(idual) += coef * q(src prim, eta cache), interior cells.
TaskStatus AddDualJouleSource(MeshData<Real> *md_src, MeshData<Real> *md_dst, const Real coef) {
  auto pkg = md_src->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const int idual = pkg->Param<int>("dual_idx");
  if (idual < 0) return TaskStatus::complete;
  const auto &prim_pack = md_src->PackVariables(std::vector<std::string>{"prim"});
  const auto &eta_pack = md_src->PackVariables(std::vector<std::string>{"nonideal_eta"});
  auto dst_pack = md_dst->PackVariables(std::vector<std::string>{"cons"});
  const IndexRange ib = md_src->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = md_src->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = md_src->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const int ndim_h = md_src->GetBlockData(0)->GetBlockPointer()->pmy_mesh->ndim;
  const int dj = ndim_h > 1, dk = ndim_h > 2;
  if (pkg->Param<bool>("dual_energy_heat_face")) {
    // Packed BY NAME (flag-pack trap: grav::phi / rad fields are also WithFluxes).
    auto cf = md_src->PackVariablesAndFluxes(std::vector<std::string>{"cons"}, std::vector<std::string>{"cons"});
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "DualFaceSource", parthenon::DevExecSpace(), 0,
        dst_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          dst_pack(b, idual, k, j, i) += coef * DualFaceQ(prim_pack(b), cf(b), prim_pack.GetCoords(b), k, j, i, ndim_h);
        });
    return TaskStatus::complete;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DualJouleSource", parthenon::DevExecSpace(), 0,
      dst_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real q = DualJouleQ(prim_pack(b), eta_pack(b), prim_pack.GetCoords(b), k, j, i, dj, dk);
        dst_pack(b, idual, k, j, i) += coef * q;
      });
  return TaskStatus::complete;
}
} // namespace

TaskStatus DualHeatIntoMY0(MeshData<Real> *md_base, MeshData<Real> *md_my0) {
  return AddDualJouleSource(md_base, md_my0, 1.0);
}

TaskStatus DualHeatStage(MeshData<Real> *md_base, const Real coef) {
  return AddDualJouleSource(md_base, md_base, coef);
}
