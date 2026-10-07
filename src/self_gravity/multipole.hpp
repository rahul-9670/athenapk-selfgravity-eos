//========================================================================================
// AthenaPK - Self-gravity package
// Multipole (Cartesian, through quadrupole) exterior boundary condition for phi.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
#ifndef SELF_GRAVITY_MULTIPOLE_HPP_
#define SELF_GRAVITY_MULTIPOLE_HPP_

// WS-5a of PHYSICS_COMPLETION_PLAN.md. Replaces the phi=0 Dirichlet walls with the
// exterior Cartesian multipole expansion of the interior mass (monopole + traceless
// quadrupole about the center of mass), so an isolated collapsing cloud can live in a
// box only a few times its own size without the BC clipping the potential well.
//
// Units: code units, four_pi_G = 1 in production. The potential prefactor is
//   G = four_pi_G / (4 pi).
// The dipole term is identically zero about the center of mass and is omitted.
//
// Ghost fill uses FixedFace semantics (matches the zero-Dirichlet path in
// poisson_equation.hpp): the domain-boundary FACE value is pinned to the expansion,
//   ghost = 2 * Phi_mp(face) - interior_mirror.

#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/package.hpp>

#include "self_gravity.hpp"

namespace SelfGravity {

// POD carried in a mutable package param, recomputed once per step in FillPoissonRHS
// and captured by value into the (device) boundary kernels. Q is the traceless
// symmetric quadrupole about the center of mass (cx,cy,cz), stored xx,yy,zz,xy,xz,yz.
struct MultipoleMoments {
  Real M;
  Real cx, cy, cz;
  Real Qxx, Qyy, Qzz, Qxy, Qxz, Qyz;
  Real four_pi_G;
};

//========================================================================================
// SHELL-SPLIT (interior + exterior) EXPANSION -- the fix for boundary points that are NOT
// exterior to the mass distribution.  2026-08-16.
//
// WHAT WAS WRONG. MultipolePhi below is a pure EXTERIOR expansion: it assumes every gram of
// mass lies inside the field point's radius. On the domain faces that is false. Measured on
// the production geometry (L = 16 box, the new grav-mp-extmass diagnostic): 45% of the mass
// sits outside the largest inscribed sphere at t = 0, falling to 34% once the core condenses.
// Most of it is the low-density ambient medium filling the box corners -- harmless-looking,
// but the monopole term M/r counts ALL of it as interior, so it overstates the enclosed mass
// at a face-centre point by tens of percent. Measured consequence (box ladder at fixed dx,
// L = 16 against L = 32): 7.8% median / 10.2% max error in |grad phi| at r = 1-2, i.e. right
// where the collapse happens, and it grows INWARD.
//
// WHY MORE MULTIPOLE ORDER CANNOT FIX IT. The expansion is being evaluated INSIDE its own
// source distribution. Truncation order is not the problem; the formula is the wrong formula
// there, and adding l = 4 makes a divergent-in-principle series marginally different.
//
// WHY A BIGGER BOX IS NOT THE FIX EITHER. The multipole BC exists precisely so the box can be
// small -- at L = 32 plain zero-Dirichlet was already adequate. "Enlarge the box" would give
// up the thing this BC was written to buy.
//
// THE ACTUAL FIX -- the standard isolated-boundary formulation, splitting the source at the
// field radius r and expanding each part in the variable that converges for it:
//
//   Phi(r) = -G [  M</r + (p<.n)/r^2 + (Q<:nn)/(2 r^3)            <- mass INSIDE r
//                + S0 + r (n.S1) + (r^2/2)(3 n_i n_j S2_ij - T0)  ]  <- mass OUTSIDE r
//
//   inner (per shell):  M< = sum dm,  p< = sum dm x,  m_ij = sum dm x_i x_j
//                       (Q_ij = 3 m_ij - delta_ij tr m, formed at evaluation time)
//   outer (per shell):  S0 = sum dm/r',  S1_i = sum dm x_i/r'^3,
//                       S2_ij = sum dm x_i x_j/r'^5,  T0 = sum dm/r'^3
//
// Both series now have ratio (r_</r_>) <= 1 term by term, which is what makes them converge.
// With all mass interior this reduces EXACTLY to the old expression (S0 = S1 = S2 = T0 = 0,
// and p< = 0 about the centre of mass), so the two agree wherever the old one was valid.
//
// EXPANDED ABOUT THE COORDINATE ORIGIN, not the centre of mass, and the DIPOLE is carried
// explicitly. The old code dropped the dipole because it vanishes identically about the COM --
// true for the TOTAL mass, false for the mass interior to a given radius, which is what the
// split needs. Expanding about the origin also removes a chicken-and-egg: the COM is an output
// of the very reduction that would need it as an input. The measured COM offset is <= 3e-3 of
// the inscribed radius, so the two centres are nearly the same point anyway; the difference is
// that the origin-centred form stays correct when they are not.
//
// COST, and the P1 lesson it respects. Cells INSIDE the inscribed sphere (52% of them, and the
// entire dense core) are interior to every boundary point, so they need no binning at all --
// they go to the existing 10-moment tree reduction. Only cells in the shell [r_lo, r_hi] touch
// the binned accumulator, and they atomic-add 21 values spread over nshell*21 addresses. The
// pre-P1 moment kernel was 51.6% of GPU time doing 10 atomics onto TEN addresses; this does
// ~10 atomics per cell (0.48 * 21) onto ~270, i.e. far less contended per address. That is an
// argument, not a measurement -- it must be profiled on GPU before it is called cheap.
constexpr int kMPNSlot = 21; // 10 inner + 11 outer, per radial bin

//! Slot layout within one radial bin. Kept as an enum so the accumulate and evaluate sides
//! cannot drift apart -- they are in different files.
enum MPSlot {
  kM = 0, kPx, kPy, kPz, kMxx, kMyy, kMzz, kMxy, kMxz, kMyz,   // inner  (10)
  kS0, kS1x, kS1y, kS1z, kS2xx, kS2yy, kS2zz, kS2xy, kS2xz, kS2yz, kT0 // outer (11)
};

//! Cumulative shell-split moments. `cum` holds (nshell+1) bins of kMPNSlot doubles, already
//! accumulated on the host: bin index (kbin+1) where kbin is the field point's shell (-1 for
//! r <= r_lo). Inner slots are cumulative INWARD (everything at radius <= this bin's outer
//! edge); outer slots are cumulative OUTWARD (everything beyond it).
struct MultipoleShells {
  Kokkos::View<Real *> cum;
  int nshell = 0;
  Real r_lo = 0.0, r_hi = 1.0, inv_dr = 1.0;
  Real four_pi_G = 1.0;
  bool active = false;
  // Uniform-background split (see UniformBoxPhi): rho_bar is the box-mean density, and `cum`
  // holds the moments of the RESIDUAL delta-rho = rho - rho_bar, not of rho.
  Real rho_bar = 0.0;
  Real bx0 = 0.0, bx1 = 0.0, by0 = 0.0, by1 = 0.0, bz0 = 0.0, bz1 = 0.0;
  bool uniform_split = false;
};

//! Exact potential of a UNIFORM-density rectangular box, at any point inside or outside it.
//!
//! WHY THIS IS HERE -- measured, not guessed. With the shell split alone the boundary values
//! still carried 10.6% of phi's variation across the face, and a multipole-order sweep showed
//! why: l = 0,1,2,3 are all flat (1.914 -> 1.906) and only l = 4 helps (0.792). l = 4 is the
//! lowest harmonic that couples to OCTAHEDRAL symmetry -- the residual is the shape of the
//! BOX, not the shape of the core. The ambient medium fills a cube, and a cube is intrinsically
//! expensive to represent in spherical harmonics.
//! So do not represent it that way. Split rho = rho_bar + delta-rho:
//!   * rho_bar fills the box uniformly and its potential is this closed form -- exact, no
//!     truncation, no moments, and it costs 8 evaluations of an antiderivative;
//!   * delta-rho has ZERO total mass and is concentrated on the actual structure, so its
//!     shell-split multipole expansion converges at l <= 2.
//! Measured on the production density field: 10.6% -> 0.58% of phi's face variation, i.e. 18x
//! beyond the shell split alone and 57x beyond the original pure-exterior expansion.
//!
//! THE FORMULA. Phi = -G rho * sum over the 8 corners (-1)^(i+j+k+1) F(X_i, Y_j, Z_k), with
//! X_i = x_i - px etc., and F the antiderivative with d3F/dxdydz = 1/r:
//!   F = xy ln(z+r) + yz ln(x+r) + zx ln(y+r)
//!       - (x^2/2) atan(yz/(xr)) - (y^2/2) atan(zx/(yr)) - (z^2/2) atan(xy/(zr))
//! VERIFIED NUMERICALLY rather than transcribed on faith, twice: the mixed third difference of
//! F reproduces 1/r to 1e-6, and the assembled box potential matches a fine direct integration
//! to 2e-6 at seven points inside, on, and outside the face. The first transcription used
//! atan2 instead of atan -- a different function with a different branch, which failed the
//! third-difference test by factors of 2-3. atan2 here would be silently wrong.
KOKKOS_INLINE_FUNCTION Real UniformBoxAnti(Real x, Real y, Real z) {
  const Real r = std::sqrt(x * x + y * y + z * z);
  constexpr Real e = 1.0e-300;
  // Guard the atan denominators only; the logs are guarded by |.| + e. A field point exactly
  // on a box edge or face plane sends one coordinate to 0, which is a removable singularity of
  // the sum but not of the individual terms.
  const Real gx = (std::abs(x) < 1.0e-12) ? 1.0e-12 : x;
  const Real gy = (std::abs(y) < 1.0e-12) ? 1.0e-12 : y;
  const Real gz = (std::abs(z) < 1.0e-12) ? 1.0e-12 : z;
  return x * y * std::log(std::abs(z + r) + e) + y * z * std::log(std::abs(x + r) + e) +
         z * x * std::log(std::abs(y + r) + e) -
         0.5 * x * x * std::atan(y * z / (gx * r + e)) -
         0.5 * y * y * std::atan(z * x / (gy * r + e)) -
         0.5 * z * z * std::atan(x * y / (gz * r + e));
}

KOKKOS_INLINE_FUNCTION Real UniformBoxPhi(const MultipoleShells &ms, Real px, Real py,
                                          Real pz) {
  const Real xs[2] = {ms.bx0 - px, ms.bx1 - px};
  const Real ys[2] = {ms.by0 - py, ms.by1 - py};
  const Real zs[2] = {ms.bz0 - pz, ms.bz1 - pz};
  Real s = 0.0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        s += ((i + j + k) % 2 == 0 ? -1.0 : 1.0) * UniformBoxAnti(xs[i], ys[j], zs[k]);
  constexpr Real kInv4Pi = 0.25 / M_PI;
  return -(ms.four_pi_G * kInv4Pi) * ms.rho_bar * s;
}

//! Shell-split potential at (x,y,z), measured from the coordinate origin.
KOKKOS_INLINE_FUNCTION Real MultipoleShellPhi(const MultipoleShells &ms, Real x, Real y,
                                              Real z) {
  const Real r2 = x * x + y * y + z * z;
  const Real r = std::sqrt(r2);
  if (r <= 0.0) return 0.0;
  const Real invr = 1.0 / r;
  const Real nx = x * invr, ny = y * invr, nz = z * invr;
  // Which shell does the field point sit in? -1 means "inside r_lo", i.e. every binned cell
  // is exterior to it.
  int kb = -1;
  if (r > ms.r_lo) {
    kb = static_cast<int>((r - ms.r_lo) * ms.inv_dr);
    if (kb < 0) kb = 0;
    if (kb > ms.nshell - 1) kb = ms.nshell - 1;
  }
  const Real *c = ms.cum.data() + static_cast<std::size_t>(kb + 1) * kMPNSlot;

  // ---- interior part: monopole + dipole + traceless quadrupole -------------------------
  const Real invr2 = invr * invr, invr3 = invr2 * invr;
  const Real trm = c[kMxx] + c[kMyy] + c[kMzz];
  const Real Qxx = 3.0 * c[kMxx] - trm, Qyy = 3.0 * c[kMyy] - trm, Qzz = 3.0 * c[kMzz] - trm;
  const Real Qnn = Qxx * nx * nx + Qyy * ny * ny + Qzz * nz * nz +
                   2.0 * (3.0 * c[kMxy] * nx * ny + 3.0 * c[kMxz] * nx * nz +
                          3.0 * c[kMyz] * ny * nz);
  const Real phi_in = c[kM] * invr + (c[kPx] * nx + c[kPy] * ny + c[kPz] * nz) * invr2 +
                      0.5 * Qnn * invr3;

  // ---- exterior part: the same three orders in the OTHER variable -----------------------
  const Real nnS2 = c[kS2xx] * nx * nx + c[kS2yy] * ny * ny + c[kS2zz] * nz * nz +
                    2.0 * (c[kS2xy] * nx * ny + c[kS2xz] * nx * nz + c[kS2yz] * ny * nz);
  const Real phi_out = c[kS0] + r * (nx * c[kS1x] + ny * c[kS1y] + nz * c[kS1z]) +
                       0.5 * r2 * (3.0 * nnS2 - c[kT0]);

  constexpr Real kInv4Pi = 0.25 / M_PI;
  const Real phi_res = -(ms.four_pi_G * kInv4Pi) * (phi_in + phi_out);
  // `cum` holds the moments of delta-rho when the uniform split is active, so the uniform
  // background's exact potential must be added back.
  return ms.uniform_split ? phi_res + UniformBoxPhi(ms, x, y, z) : phi_res;
}

// Exterior potential at physical point (x,y,z). Valid outside the mass distribution.
KOKKOS_INLINE_FUNCTION Real MultipolePhi(const MultipoleMoments &mm, Real x, Real y,
                                         Real z) {
  constexpr Real kInv4Pi = 0.25 / M_PI;
  const Real dx = x - mm.cx;
  const Real dy = y - mm.cy;
  const Real dz = z - mm.cz;
  const Real r2 = dx * dx + dy * dy + dz * dz;
  const Real r = std::sqrt(r2);
  if (r <= 0.0) return 0.0; // degenerate; a domain boundary is never at the COM
  const Real invr = 1.0 / r;
  const Real invr5 = invr / (r2 * r2);
  const Real dQd = mm.Qxx * dx * dx + mm.Qyy * dy * dy + mm.Qzz * dz * dz +
                   2.0 * (mm.Qxy * dx * dy + mm.Qxz * dx * dz + mm.Qyz * dy * dz);
  const Real G = mm.four_pi_G * kInv4Pi;
  return -G * (mm.M * invr + 0.5 * dQd * invr5);
}

// Per-block physical-boundary function for phi enrolled as a UserBoundaryFunction (the
// packed solver-internal fast path lives in PoissonEquation::SetBoundary). Mirrors
// parthenon's GenericBC<...,FixedFace,...> loop but with a spatially varying face value
// = MultipolePhi(boundary face point). Cell-centered (phi) only.
template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE>
void MultipoleBC(std::shared_ptr<parthenon::MeshBlockData<Real>> &rc, bool coarse) {
  using namespace parthenon;
  using TE = TopologicalElement;
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BoundaryFunction::BCSide::Inner);
  constexpr auto el = TE::CC;

  MeshBlock *pmb = rc->GetBlockPointer();
  // The coarse BC is only invoked on blocks with coarser neighbors, so pmr is valid
  // there (boundary_conditions.cpp:46). The pack always carries FINE coords, so we
  // fetch the matching coordinate object explicitly.
  const Coordinates_t coords =
      (coarse && pmb->pmr) ? pmb->pmr->GetCoarseCoords() : pmb->coords;

  auto pkg = pmb->packages.Get("self_gravity");
  const MultipoleMoments mm = pkg->Param<MultipoleMoments>("grav_multipole_moments");
  // Post-solve ghost restore must use the SAME potential the RHS lift used, or the ghosts and
  // the solved interior disagree and ApplyGravitySource reads an inconsistent gradient.
  const bool mp_split_bc = pkg->Param<bool>("mp_shell_split");
  MultipoleShells ms_bc;
  if (mp_split_bc) ms_bc = pkg->Param<MultipoleShells>("grav_mp_shells");

  std::set<PDOpt> opts = coarse ? std::set<PDOpt>{PDOpt::Coarse} : std::set<PDOpt>{};
  std::vector<MetadataFlag> flags{Metadata::FillGhost, Metadata::Cell};
  const auto desc = MakePackDescriptor<grav::phi>(rc.get(), flags, opts);
  auto q = desc.GetPack(rc.get());
  const int b = 0;
  const int lstart = q.GetLowerBoundHost(b);
  const int lend = q.GetUpperBoundHost(b);
  if (lend < lstart) return; // phi not allocated on this block

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto range = X1 ? bounds.GetBoundsI(IndexDomain::interior, el)
                        : (X2 ? bounds.GetBoundsJ(IndexDomain::interior, el)
                              : bounds.GetBoundsK(IndexDomain::interior, el));
  const int ref = INNER ? range.s : range.e;
  const int offset = 2 * ref + (INNER ? -1 : 1);
  // Lower-face index (along DIR) of the domain boundary face.
  const int faceidx = INNER ? ref : ref + 1;

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  auto nb = IndexRange{lstart, lend};
  pmb->par_for_bndry(
      "SG::MultipoleBC", nb, domain, el, coarse, /*fine=*/false,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Boundary-face point aligned with this ghost column: normal coordinate at the
        // domain face, transverse coordinates at the ghost cell center.
        const Real px = X1 ? coords.Xf<X1DIR>(k, j, faceidx) : coords.Xc<X1DIR>(k, j, i);
        const Real py = X2 ? coords.Xf<X2DIR>(k, faceidx, i) : coords.Xc<X2DIR>(k, j, i);
        const Real pz = X3 ? coords.Xf<X3DIR>(faceidx, j, i) : coords.Xc<X3DIR>(k, j, i);
        const Real val = (mp_split_bc && ms_bc.active)
                             ? MultipoleShellPhi(ms_bc, px, py, pz)
                             : MultipolePhi(mm, px, py, pz);
        q(b, el, l, k, j, i) =
            2.0 * val - q(b, el, l, X3 ? offset - k : k, X2 ? offset - j : j,
                          X1 ? offset - i : i);
      });
}

} // namespace SelfGravity

#endif // SELF_GRAVITY_MULTIPOLE_HPP_
