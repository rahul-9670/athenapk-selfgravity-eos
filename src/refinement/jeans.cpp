//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <algorithm>
#include <cmath>
#include <limits>

#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/hydro.hpp"
#include "../main.hpp"
#include "refinement.hpp"

namespace refinement {
namespace jeans {

using parthenon::IndexDomain;
using parthenon::IndexRange;

// Jeans-length refinement criterion (Truelove 1997).
// Refines when the Jeans length is resolved by fewer than `njeans` cells.
//
// In code units with 4*pi*G = 1:
//   lambda_J = 2*pi * c_s / sqrt(rho)    (hydro)
//   lambda_J = 2*pi * (c_s + v_A) / sqrt(rho)   (MHD, Athena++ convention)
// where v_A = sqrt(B^2 / rho) is the Alfven speed.
//
// nj = lambda_J / dx is "Jeans length in cells"
//   nj < njeans         -> refine
//   nj > 2.5 * njeans   -> derefine
//
// AUDIT 2026-08-05 (A1) -- THE SOUND SPEED COMES FROM THE EOS, NOT FROM gamma.
// This criterion used to hard-code cs = sqrt(gamma * p / rho) with <hydro> gamma, while
// every AMR production deck also sets <hydro> eos = hydrogen (the tabulated multi-Saha
// protostellar EOS). The mesh was therefore being set by a DIFFERENT sound speed from the
// one the dynamics uses (AdiabaticGLMMHDEOS::SoundSpeed -> eos_tab.AsqFromRhoPres).
//
// Measured impact on the completed runs (runs/audit_fix_regress/measure_A1.py, on the final
// prod_v9 snapshot: 2213 blocks, 7.25e7 cells): cs_ideal/cs_table = 0.917 median, and ZERO
// blocks change tag. The ratio is BELOW one because below theta_rot = 85 K the H2 rotational
// modes are frozen out and the table returns Gamma_1 ~ 5/3, not 7/5 -- sqrt(1.4/1.65)=0.921.
// So in the regime the runs have actually occupied the old formula UNDER-estimated c_s and
// therefore over-refined: wasteful, never unsafe.
//
// It inverts where it matters: through H2 dissociation Gamma_1 -> 1.14, the old formula
// OVER-estimates c_s, over-estimates lambda_J, and UNDER-refines -- precisely the regime
// the Truelove condition exists to protect. No completed run has reached it (max T in
// prod_v9 is 530 K), so this fix changes nothing today and everything once a core is pushed
// through dissociation. Using the EOS object itself keeps one source of truth, so the
// criterion cannot drift from the dynamics again.
template <typename EOS>
KOKKOS_INLINE_FUNCTION Real JeansSpeed(const EOS &eos, const Real rho, const Real p,
                                       const Real bsq, const bool mhd) {
  Real prim_c[NHYDRO];
  prim_c[IDN] = rho;
  prim_c[IPR] = p;
  Real v = eos.SoundSpeed(prim_c); // tabulated when eos=hydrogen, gamma-law otherwise
  if (mhd) v += std::sqrt(bsq / rho);
  return v;
}

template <typename EOS>
Real MinJeansLengthInCells(parthenon::MeshBlock *pmb, const EOS &eos, const bool mhd) {
  auto w = pmb->meshblock_data.Get()->Get("prim").data;

  // AUDIT F-8 (2026-08-28): this used pmb->coords.Dxc<1>(0) alone with no check that
  // dx = dy = dz. min() is the conservative choice: identical to the old dx on a cubic mesh
  // (bit-identical there), and refining rather than under-resolving on a stretched one.
  const Real dx = std::min({pmb->coords.Dxc<1>(0), pmb->coords.Dxc<2>(0),
                            pmb->coords.Dxc<3>(0)});

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real fac = 2.0 * M_PI / dx;

  Real njmin = std::numeric_limits<Real>::max();
  pmb->par_reduce(
      "jeans refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lnjmin) {
        const Real rho = w(IDN, k, j, i);
        const Real p = w(IPR, k, j, i);
        Real bsq = 0.0;
        if (mhd) {
          bsq = w(IB1, k, j, i) * w(IB1, k, j, i) + w(IB2, k, j, i) * w(IB2, k, j, i) +
                w(IB3, k, j, i) * w(IB3, k, j, i);
        }
        const Real nj = JeansSpeed(eos, rho, p, bsq, mhd) / std::sqrt(rho);
        lnjmin = std::min(lnjmin, nj);
      },
      Kokkos::Min<Real>(njmin));

  // lambda_J = c_eff sqrt(pi/(G rho)) = 2 pi c_eff / sqrt(four_pi_G rho). `fac` assumes
  // four_pi_G = 1 (the collapse runs' units_override); any other value is divided out here (audit P-15),
  // and the == 1 case is left untouched so it stays bit-identical.
  Real nj = njmin * fac;
  auto &pkgs = pmb->pmy_mesh->packages;
  if (pkgs.AllPackages().count("self_gravity") > 0) {
    const Real fpg = pkgs.Get("self_gravity")->Param<Real>("four_pi_G");
    if (fpg != 1.0 && fpg > 0.0) nj /= std::sqrt(fpg);
  }
  return nj;
}

parthenon::AmrTag Jeans(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const Real njeans = hydro_pkg->Param<Real>("refinement/njeans");
  // This build is MHD-only (no Fluid::euler), so the EOS is always
  // the GLMMHD one and the Jeans length uses the fast magnetosonic speed -- the same speed
  // the integrator's timestep uses.
  const Real njmin =
      MinJeansLengthInCells(pmb, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"), true);

  if (njmin < njeans) return parthenon::AmrTag::refine;
  if (njmin > 2.5 * njeans) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
}

} // namespace jeans
} // namespace refinement
