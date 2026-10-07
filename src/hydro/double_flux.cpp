//========================================================================================
// AthenaPK -- double-flux energy update for the tabulated EOS. See double_flux.hpp for the why/what/scope.
// validation_tests/eos_tabulated E7 / E-F10 (2026-10-05).
//========================================================================================
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
#include "../recon/dc_simple.hpp"
#include "../recon/plm_simple.hpp"
#include "double_flux.hpp"

using namespace parthenon::package::prelude;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

namespace Hydro {
using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::ScratchPad2D;

namespace {
// Affine coefficients of the HLLD energy flux in the face TOTAL energies: F_E = F0 + aL E_L + aR E_R.
// Replicates exactly the wave speeds and the region selection of
// Riemann<Fluid::glmmhd, RiemannSolver::hlld>::Solve (rsolvers/glmmhd_hlld.hpp):
//   L     (S_L >= 0):            dF/dE_L = v_L
//   R     (S_R <= 0):            dF/dE_R = v_R
//   L*/L** (S_L < 0, S_M >= 0):  dF/dE_L = v_L + S_L (sdl/sdml - 1)  (U_L* .e = (sdl E_L + ...)/sdml; U_L** adds
//                                                                     E-independent terms)
//   R*/R**:                      dF/dE_R = v_R + S_R (sdr/sdmr - 1)
// (the * and ** sub-regions on one side share the coefficient, so only S_M decides the side).
KOKKOS_INLINE_FUNCTION
void HLLDEnergyCoeffs(const Real *wli, const Real *wri, const AdiabaticGLMMHDEOS &eos, const Real c_h,
                      Real &aL, Real &aR) {
  const Real bxi = 0.5 * (wli[IB1] + wri[IB1]) - 0.5 / c_h * (wri[IPS] - wli[IPS]);
  const Real bxsq = bxi * bxi;
  const Real pbl = 0.5 * (bxsq + (SQR(wli[IB2]) + SQR(wli[IB3])));
  const Real pbr = 0.5 * (bxsq + (SQR(wri[IB2]) + SQR(wri[IB3])));
  const Real cfl = eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1], wli[IB2], wli[IB3]);
  const Real cfr = eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1], wri[IB2], wri[IB3]);
  const Real s0 = std::min(wli[IV1] - cfl, wri[IV1] - cfr);
  const Real s4 = std::max(wli[IV1] + cfl, wri[IV1] + cfr);
  aL = 0.0;
  aR = 0.0;
  if (s0 >= 0.0) {
    aL = wli[IV1];
    return;
  }
  if (s4 <= 0.0) {
    aR = wri[IV1];
    return;
  }
  const Real ptl = wli[IPR] + pbl, ptr = wri[IPR] + pbr;
  const Real sdl = s0 - wli[IV1], sdr = s4 - wri[IV1];
  const Real s2 = (sdr * wri[IDN] * wri[IV1] - sdl * wli[IDN] * wli[IV1] + (ptl - ptr)) /
                  (sdr * wri[IDN] - sdl * wli[IDN]);
  const Real sdml = s0 - s2, sdmr = s4 - s2;
  if (s2 >= 0.0) {
    aL = wli[IV1] + s0 * (sdl / sdml - 1.0);
  } else {
    aR = wri[IV1] + s4 * (sdr / sdmr - 1.0);
  }
}

// Table inversion e(rho, P) refined by Newton on the interpolant (the 20-step bisection alone leaves 8e-6 in e,
// i.e. up to 1e-4 in P in the dissociation regime -- too coarse for a pressure-equilibrium reset).
KOKKOS_INLINE_FUNCTION
Real EspFromPRefined(const EOSTable::EosTable &tab, const Real rho, const Real P) {
  Real e = tab.EspFromP(rho, P);
  for (int it = 0; it < 3; ++it) {
    const Real h = 1.0e-7 * e;
    const Real f = tab.Pof(rho, e) - P;
    const Real d = (tab.Pof(rho, e + h) - tab.Pof(rho, e - h)) / (2.0 * h);
    if (!(d > 0.0)) break;
    const Real en = e - f / d;
    if (!(en > 0.0)) break;
    e = en;
  }
  return e;
}

template <Reconstruction recon, int XNDIR>
void Coeffs1D(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const auto &eos = pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto &tab = eos.GetEosTable();
  const Real c_h = pkg->Param<Real>("c_h");
  auto prim_in = md->PackVariables(std::vector<std::string>{"prim"});
  auto xyz_in = md->PackVariablesAndFluxes(std::vector<std::string>{"dflux_xyz"},
                                           std::vector<std::string>{"dflux_xyz"});
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int scratch_level = pkg->Param<int>("scratch_level");
  // Reconstruct writes EVERY prim component (passive scalars included): size the scratch rows from prim, as
  // the flux kernel does (nhydro + nscalars). Sized by the hydro count before 2026-10-06, which overran wl/wlb into
  // the next scratch array whenever a passive scalar existed.
  const int nvar = prim_in.GetDim(4);
  const size_t scratch = ScratchPad2D<Real>::shmem_size(nvar, nx1) * 3;
  // faces: x1 i in [is, ie+1] for j,k interior; x2 j in [js, je+1]; x3 k in [ks, ke+1]
  const int kl = kb.s, ku = (XNDIR == X3DIR) ? kb.e + 1 : kb.e;
  const int jl = jb.s, ju = (XNDIR == X2DIR) ? jb.e + 1 : jb.e;
  const int il = ib.s, iu = (XNDIR == X1DIR) ? ib.e + 1 : ib.e;
  constexpr int ivx = (XNDIR == X1DIR) ? IV1 : ((XNDIR == X2DIR) ? IV2 : IV3);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "dflux coeffs", DevExecSpace(), scratch, scratch_level, 0,
      prim_in.GetDim(5) - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &prim = prim_in(b);
        ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nvar, nx1);
        ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nvar, nx1);
        ScratchPad2D<Real> wlb(member.team_scratch(scratch_level), nvar, nx1);
        // nvcc: an extended __host__ __device__ lambda may not FIRST capture a variable inside `if constexpr`
        const int ilo = il, ihi = iu;
        if constexpr (XNDIR == X1DIR) {
          Reconstruct<recon, X1DIR>(member, k, j, ilo - 1, ihi, prim, wl, wr);
        } else if constexpr (XNDIR == X2DIR) {
          // reconstruction on (j-1) gives the left states at face j; on j the right states
          Reconstruct<recon, X2DIR>(member, k, j - 1, ilo, ihi, prim, wl, wlb);
          member.team_barrier();
          Reconstruct<recon, X2DIR>(member, k, j, ilo, ihi, prim, wlb, wr);
        } else {
          Reconstruct<recon, X3DIR>(member, k - 1, j, ilo, ihi, prim, wl, wlb);
          member.team_barrier();
          Reconstruct<recon, X3DIR>(member, k, j, ilo, ihi, prim, wlb, wr);
        }
        member.team_barrier();
        const int iby = (ivx == IV1) ? IB2 : ((ivx == IV2) ? IB3 : IB1);
        const int ibz = (ivx == IV1) ? IB3 : ((ivx == IV2) ? IB1 : IB2);
        const int ibx = ivx - 1 + NHYDRO;
        const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
        const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
        parthenon::par_for_inner(member, il, iu, [&](const int i) {
          Real wli[9], wri[9];
          wli[IDN] = wl(IDN, i); wli[IV1] = wl(ivx, i); wli[IV2] = wl(ivy, i); wli[IV3] = wl(ivz, i);
          wli[IPR] = wl(IPR, i); wli[IB1] = wl(ibx, i); wli[IB2] = wl(iby, i); wli[IB3] = wl(ibz, i);
          wli[IPS] = wl(IPS, i);
          wri[IDN] = wr(IDN, i); wri[IV1] = wr(ivx, i); wri[IV2] = wr(ivy, i); wri[IV3] = wr(ivz, i);
          wri[IPR] = wr(IPR, i); wri[IB1] = wr(ibx, i); wri[IB2] = wr(iby, i); wri[IB3] = wr(ibz, i);
          wri[IPS] = wr(IPS, i);
          Real aL, aR;
          HLLDEnergyCoeffs(wli, wri, eos, c_h, aL, aR);
          // The own-EOS increment of the face energy flux for ANY frozen EOS (g, e0) is affine in (1/(g-1), e0):
          //   aL (E_fr(rhoL,PL) - E_tab,L) + aR (E_fr(rhoR,PR) - E_tab,R) = X/(g-1) + e0 Y - Z,
          //   X = aL PL + aR PR,  Y = aL rhoL + aR rhoR,  Z = aL E_tab,L + aR E_tab,R  (E_tab exactly as HLLD forms it).
          // Stored as the FLUXES of dflux_xyz, so Parthenon's flux correction replaces them on a coarse face by the
          // area average over the fine faces -- the coarse cell then gets the increment of exactly the restricted
          // flux it receives (linearity makes the average exact).
          const Real ElT = eos.InternalEnergyFromPres(wli[IDN], wli[IPR]);
          const Real ErT = eos.InternalEnergyFromPres(wri[IDN], wri[IPR]);
          xyz_in(b).flux(XNDIR, 0, k, j, i) = aL * wli[IPR] + aR * wri[IPR];
          xyz_in(b).flux(XNDIR, 1, k, j, i) = aL * wli[IDN] + aR * wri[IDN];
          xyz_in(b).flux(XNDIR, 2, k, j, i) = aL * ElT + aR * ErT;
        });
      });
  (void)tab;
}

template <Reconstruction recon>
void AllDirs(MeshData<Real> *md) {
  auto pm = md->GetBlockData(0)->GetBlockPointer()->pmy_mesh;
  Coeffs1D<recon, X1DIR>(md);
  if (pm->ndim >= 2) Coeffs1D<recon, X2DIR>(md);
  if (pm->ndim >= 3) Coeffs1D<recon, X3DIR>(md);
}
} // namespace

TaskStatus DoubleFluxCoefficients(MeshData<Real> *md, const int stage) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const auto &eos = pkg->Param<AdiabaticGLMMHDEOS>("eos");
  if (stage == 1) {
    // freeze g* = rho c^2 / P and e0* = e - P/((g*-1) rho) from the STEP-START state, every cell incl. ghosts
    auto cons_in = md->PackVariables(std::vector<std::string>{"cons"});
    auto prim_in = md->PackVariables(std::vector<std::string>{"prim"});
    auto frz_in = md->PackVariables(std::vector<std::string>{"dflux_frozen"});
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "dflux freeze", DevExecSpace(), 0, cons_in.GetDim(5) - 1, kb.s, kb.e,
        jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &w = prim_in(b);
          const auto &u = cons_in(b);
          const Real rho = w(IDN, k, j, i), P = w(IPR, k, j, i);
          // the cell's CONSERVED internal energy, so the frozen EOS reproduces the step-start state exactly
          const Real eint = u(IEN, k, j, i) -
                            0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) / u(IDN, k, j, i) -
                            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
          const Real c2 = eos.GetEosTable().AsqFromRhoPres(rho, P);
          Real g = rho * c2 / P;
          g = (g > 1.0001) ? g : 1.0001;
          g = (g < 5.0) ? g : 5.0;
          frz_in(b, 0, k, j, i) = g;
          frz_in(b, 1, k, j, i) = (eint - P / (g - 1.0)) / rho;
        });
  }
  const auto stage1_dc = pkg->Param<bool>("dflux_stage1_dc");
  const auto recon = pkg->Param<Reconstruction>("reconstruction");
  if ((stage == 1 && stage1_dc) || recon == Reconstruction::dc) {
    AllDirs<Reconstruction::dc>(md);
  } else {
    AllDirs<Reconstruction::plm>(md);
  }
  return TaskStatus::complete;
}

TaskStatus DoubleFluxCorrect(MeshData<Real> *md, const Real beta_dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const auto &eos = pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto &tab = eos.GetEosTable();
  const int ndim = pmb->pmy_mesh->ndim;
  auto cons_in = md->PackVariables(std::vector<std::string>{"cons"});
  auto frz_in = md->PackVariables(std::vector<std::string>{"dflux_frozen"});
  auto xyz_in = md->PackVariablesAndFluxes(std::vector<std::string>{"dflux_xyz"},
                                           std::vector<std::string>{"dflux_xyz"});
  auto prim_in = md->PackVariables(std::vector<std::string>{"prim"});
  const Real kappa = pkg->Param<Real>("double_flux_kappa");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  // MESH REFINEMENT: nothing to do here -- the X, Y, Z face sums are fluxes of dflux_xyz and were flux-corrected
  // with the hydro fluxes (see Coeffs1D). (v5 made coarse cells at fine/coarse faces standard cells instead:
  // contacts crossing a refinement boundary kept 1e-3 (AMR) to 8e-3 (SMR) spurious P; E7r.)
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "dflux correct", DevExecSpace(), 0, cons_in.GetDim(5) - 1, kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &u = cons_in(b);
        const auto &coords = cons_in.GetCoords(b);
        // CONTACT SENSOR, PER CELL (Abgrall-Karni is a per-cell scheme: a cell either evaluates ALL its face energy
        // fluxes with its own frozen EOS and resets P from it, or it is fully standard). The frozen EOS is a tangent
        // fit at the cell state: right for a contact (P continuous) and uniform gas, wrong where the cell moves far
        // along a non-contact wave. Measured: a pressure-jump threshold (dP/P < 0.1) still caught shock feet and
        // left R1/R3 with resolution-independent energy errors (R1 plateau 1e-6 -> 1.1e-3); a per-FACE contact
        // sensor kept R1/R3 clean but applied own-EOS increments without the reset in cells with one contact face
        // and one non-contact face, and on the noisy plateaus beside the contact the faces flip on/off step to step:
        // R2/R4 plateau errors GREW with N (R4 dP* -2.9e-2 at N=3200). Shocks and adiabatic flow have
        // dP/P ~ Gamma1 drho/rho, Gamma1 >= 1.1; a contact has a density jump without a pressure jump. A cell is a
        // double-flux cell iff every face has dP/P <= kappa drho/rho (hydro/double_flux_kappa) or both jumps are at
        // round-off. Evaluated on the stage's prim (ConsToPrim has not overwritten it yet), the state the face
        // fluxes were built from.
        const auto &w = prim_in(b);
        auto face_on = [&](const int k2, const int j2, const int i2) {
          const Real pa = w(IPR, k, j, i), pb = w(IPR, k2, j2, i2);
          const Real ra = w(IDN, k, j, i), rb = w(IDN, k2, j2, i2);
          const Real dpr = std::abs(pb - pa) / std::min(pa, pb), drr = std::abs(rb - ra) / std::min(ra, rb);
          return (dpr <= kappa * drr) || (dpr <= 1.0e-12 && drr <= 1.0e-12);
        };
        bool all_on = face_on(k, j, i - 1) && face_on(k, j, i + 1);
        if (ndim >= 2) all_on = all_on && face_on(k, j - 1, i) && face_on(k, j + 1, i);
        if (ndim >= 3) all_on = all_on && face_on(k - 1, j, i) && face_on(k + 1, j, i);
        if (!all_on) return;
        // own-EOS increments of the energy flux at every face of this cell, with THIS cell's frozen EOS
        const Real gm1i = 1.0 / (frz_in(b, 0, k, j, i) - 1.0), e0c = frz_in(b, 1, k, j, i);
        const auto &xf = xyz_in(b);
        auto inc = [&](const int dir, const int k2, const int j2, const int i2) {
          return xf.flux(dir, 0, k2, j2, i2) * gm1i + e0c * xf.flux(dir, 1, k2, j2, i2) - xf.flux(dir, 2, k2, j2, i2);
        };
        Real dE = (inc(X1DIR, k, j, i) - inc(X1DIR, k, j, i + 1)) / coords.Dxc<1>(k, j, i);
        if (ndim >= 2) dE += (inc(X2DIR, k, j, i) - inc(X2DIR, k, j + 1, i)) / coords.Dxc<2>(k, j, i);
        if (ndim >= 3) dE += (inc(X3DIR, k, j, i) - inc(X3DIR, k + 1, j, i)) / coords.Dxc<3>(k, j, i);
        u(IEN, k, j, i) += beta_dt * dE;
        // reset: P from the cell's frozen EOS, E rebuilt from the table at that P (non-conservative by design)
        const Real rho = u(IDN, k, j, i);
        const Real ek = 0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) / rho;
        const Real eb = 0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
        const Real eint = u(IEN, k, j, i) - ek - eb;
        const Real g = frz_in(b, 0, k, j, i), e0 = frz_in(b, 1, k, j, i);
        const Real P = (g - 1.0) * (eint - rho * e0);
        if (P > 0.0) {
          u(IEN, k, j, i) = rho * EspFromPRefined(tab, rho, P) + ek + eb;
        }
      });
  return TaskStatus::complete;
}

namespace {
// Per block: bit (o1+1) + 3 (o2+1) + 9 (o3+1) set if the neighbour in that ghost region lives on a DIFFERENT level,
// i.e. the region was filled by prolongation (coarser neighbour) or restriction (finer neighbour).
int LevelJumpMask(const parthenon::MeshBlock *pb) {
  int mask = 0;
  for (auto const &nb : pb->GetNeighbors()) {
    if (nb.loc.level() == pb->loc.level()) continue;
    mask |= 1 << ((static_cast<int>(nb.offsets[0]) + 1) + 3 * (static_cast<int>(nb.offsets[1]) + 1) +
                  9 * (static_cast<int>(nb.offsets[2]) + 1));
  }
  return mask;
}

// Pressure of a conserved state through the table, the same expression everywhere it is compared.
template <class U>
KOKKOS_INLINE_FUNCTION Real PlainP(const EOSTable::EosTable &tab, const U &u, const int k, const int j,
                                   const int i) {
  const Real r = u(IDN, k, j, i);
  const Real ek = 0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) / r;
  const Real eb = 0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
  return tab.PresFromRhoEint(r, u(IEN, k, j, i) - ek - eb);
}

// Where the transported pressure pt and P(cons) disagree (> 1e-12 relative), rebuild E from the table at pt.
template <class U>
KOKKOS_INLINE_FUNCTION void FixE(const EOSTable::EosTable &tab, U &u, const Real pt, const int k, const int j,
                                 const int i) {
  if (!(pt > 0.0)) return;
  const Real P = PlainP(tab, u, k, j, i);
  if (std::abs(P - pt) <= 1.0e-12 * pt) return;
  const Real rho = u(IDN, k, j, i);
  const Real ek = 0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) / rho;
  const Real eb = 0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
  u(IEN, k, j, i) = rho * EspFromPRefined(tab, rho, pt) + ek + eb;
}

// Contact test of the double-flux sensor on the TRANSPORTED pressure: every neighbour inside the array (ghosts
// included) has |dP|/P <= kappa |drho|/rho, or both jumps at round-off. Used to restrict the fill to contacts
// (hydro/double_flux_fill_sensor): elsewhere the conservative interpolation of Parthenon is kept.
template <class U, class PT>
KOKKOS_INLINE_FUNCTION bool ContactLike(const U &u, const PT &pt, const int k, const int j, const int i,
                                        const int ndim, const Real kappa, const IndexRange &eib,
                                        const IndexRange &ejb, const IndexRange &ekb) {
  auto ok = [&](const int k2, const int j2, const int i2) {
    const Real pa = pt(k, j, i), pb = pt(k2, j2, i2);
    const Real ra = u(IDN, k, j, i), rb = u(IDN, k2, j2, i2);
    if (!(pa > 0.0 && pb > 0.0)) return false;
    const Real dpr = std::abs(pb - pa) / std::min(pa, pb), drr = std::abs(rb - ra) / std::min(ra, rb);
    return (dpr <= kappa * drr) || (dpr <= 1.0e-12 && drr <= 1.0e-12);
  };
  bool c = true;
  if (i > eib.s) c = c && ok(k, j, i - 1);
  if (i < eib.e) c = c && ok(k, j, i + 1);
  if (ndim > 1 && j > ejb.s) c = c && ok(k, j - 1, i);
  if (ndim > 1 && j < ejb.e) c = c && ok(k, j + 1, i);
  if (ndim > 2 && k > ekb.s) c = c && ok(k - 1, j, i);
  if (ndim > 2 && k < ekb.e) c = c && ok(k + 1, j, i);
  return c;
}
} // namespace

void DoubleFluxRefreshP(MeshData<Real> *md) {
  if (md->NumBlocks() == 0) return;
  auto mbd0 = md->GetBlockData(0);
  if (!mbd0->Contains("dflux_p") || !mbd0->Contains("cons")) return; // shallow MeshData (e.g. gravity)
  auto pmb0 = mbd0->GetBlockPointer();
  const auto &tab = pmb0->packages.Get("Hydro")->Param<AdiabaticGLMMHDEOS>("eos").GetEosTable();
  auto cons_in = md->PackVariables(std::vector<std::string>{"cons"});
  auto p_in = md->PackVariables(std::vector<std::string>{"dflux_p"});
  const IndexRange ib = pmb0->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb0->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb0->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "dflux refresh P", DevExecSpace(), 0, cons_in.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        p_in(b, 0, k, j, i) = PlainP(tab, cons_in(b), k, j, i);
      });
}

TaskStatus DoubleFluxGhostFill(MeshData<Real> *md) {
  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto &tab = pmb0->packages.Get("Hydro")->Param<AdiabaticGLMMHDEOS>("eos").GetEosTable();
  const int nblocks = md->NumBlocks();
  const int ndim = pmb0->pmy_mesh->ndim;
  parthenon::ParArray1D<int> jump("dflux level jumps", nblocks);
  auto jump_h = Kokkos::create_mirror_view(jump);
  bool any = false;
  for (int b = 0; b < nblocks; ++b) {
    jump_h(b) = LevelJumpMask(md->GetBlockData(b)->GetBlockPointer());
    any = any || (jump_h(b) != 0);
  }
  if (!any) return TaskStatus::complete;
  Kokkos::deep_copy(jump, jump_h);
  auto cons_in = md->PackVariables(std::vector<std::string>{"cons"});
  auto p_in = md->PackVariables(std::vector<std::string>{"dflux_p"});
  const bool use_sensor = pmb0->packages.Get("Hydro")->Param<bool>("dflux_fill_sensor");
  const Real kappa = pmb0->packages.Get("Hydro")->Param<Real>("double_flux_kappa");
  const IndexRange ib = pmb0->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb0->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb0->cellbounds.GetBoundsK(IndexDomain::interior);
  const IndexRange eib = pmb0->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange ejb = pmb0->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange ekb = pmb0->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "dflux ghost fill", DevExecSpace(), 0, nblocks - 1, ekb.s, ekb.e, ejb.s, ejb.e, eib.s,
      eib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int mask = jump(b);
        if (mask == 0) return;
        const int o1 = (i < ib.s) ? -1 : ((i > ib.e) ? 1 : 0);
        const int o2 = (ndim < 2) ? 0 : ((j < jb.s) ? -1 : ((j > jb.e) ? 1 : 0));
        const int o3 = (ndim < 3) ? 0 : ((k < kb.s) ? -1 : ((k > kb.e) ? 1 : 0));
        if (o1 == 0 && o2 == 0 && o3 == 0) return;
        if (!(mask & (1 << ((o1 + 1) + 3 * (o2 + 1) + 9 * (o3 + 1))))) return;
        auto &u = cons_in(b);
        if (use_sensor) {
          auto pt = [&](const int k2, const int j2, const int i2) { return p_in(b, 0, k2, j2, i2); };
          if (!ContactLike(u, pt, k, j, i, ndim, kappa, eib, ejb, ekb)) return;
        }
        FixE(tab, u, p_in(b, 0, k, j, i), k, j, i);
      });
  return TaskStatus::complete;
}

void DoubleFluxPostRegrid(parthenon::Mesh *pm) {
  for (auto &pmb : pm->block_list) {
    auto mbd = pmb->meshblock_data.Get();
    const auto &tab = pmb->packages.Get("Hydro")->Param<AdiabaticGLMMHDEOS>("eos").GetEosTable();
    auto u = mbd->Get("cons").data.Get<4>();
    auto p = mbd->Get("dflux_p").data.Get<4>();
    const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    // the hook runs BEFORE the post-regrid boundary exchange: ghosts are not valid yet, so the sensor looks at
    // interior neighbours only (with ghost neighbours the 1D AMR contact test kept 4.5e-4 instead of 2e-13, E7w3)
    const IndexRange eib = ib, ejb = jb, ekb = kb;
    const bool use_sensor = pmb->packages.Get("Hydro")->Param<bool>("dflux_fill_sensor");
    const Real kappa = pmb->packages.Get("Hydro")->Param<Real>("double_flux_kappa");
    const int ndim = pm->ndim;
    // interiors that the regrid prolongated (new fine blocks) or restricted (new coarse blocks) carry rho, momenta,
    // B and E interpolated separately; dflux_p came along through the same operators (minmod prolongation =
    // the same limited stencil applied to P; restriction = average of the children's P)
    pmb->par_for(
        "dflux post-regrid fill", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          if (use_sensor) {
            auto pt = [&](const int k2, const int j2, const int i2) { return p(0, k2, j2, i2); };
            if (!ContactLike(u, pt, k, j, i, ndim, kappa, eib, ejb, ekb)) return;
          }
          FixE(tab, u, p(0, k, j, i), k, j, i);
        });
  }
}

void ConsToPrimDoubleFlux(MeshData<Real> *md) {
  DoubleFluxGhostFill(md);
  md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro")->Param<AdiabaticGLMMHDEOS>("eos").ConservedToPrimitive(
      md);
  // ConservedToPrimitive may rewrite E (floors): the transported pressure follows cons
  DoubleFluxRefreshP(md);
}

} // namespace Hydro
