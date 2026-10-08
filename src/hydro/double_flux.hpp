#ifndef HYDRO_DOUBLE_FLUX_HPP_
#define HYDRO_DOUBLE_FLUX_HPP_
//========================================================================================
// AthenaPK -- double-flux energy update for the tabulated EOS (hydro/double_flux, default false).
// validation_tests/eos_tabulated E7 / finding E-F10 (2026-10-05).
//
// WHY. A conservative update mixes (rho, rho e) LINEARLY in the cells straddling a moving contact. With a
// nonlinear EOS (H2 dissociation, H ionization) such a mixture is not at the common pressure: measured up to
// 3-4 % spurious P in the mixed cells of a pure contact between the R4 star states, converging only as
// ~N^-0.7 -- the slow plateau convergence of the E7 Riemann problems (Abgrall 1996: no scheme that conserves
// total energy exactly can avoid this for a general EOS).
//
// WHAT (Abgrall & Karni 2001, JCP 169, 594; Billet & Abgrall 2003): at the start of each time step every cell
// freezes a stiffened-gas fit of the real EOS through its own state, rho e = P/(g*-1) + rho e0*, with
// g* = rho c^2/P (same P and sound speed as the table). With the frozen EOS rho e is LINEAR in rho at fixed P, so
// a contact stays in pressure equilibrium exactly. Each cell's energy is updated with HLLD energy fluxes in which
// the two face states' internal energies come from ITS OWN frozen EOS (HLLD's energy flux is affine in the face
// energies, so this is the standard flux plus aL dE_L + aR dE_R; mass, momentum and B fluxes are unchanged).
// After the update the pressure is taken from the frozen EOS and the total energy is rebuilt from the TABLE at
// that pressure. Price: total energy is no longer conserved exactly where g*, e0* vary (contacts, EOS
// transitions); measured and reported, not hidden.
//
// CONTACT SENSOR (hydro/double_flux_kappa, default 0.2), PER CELL: double flux only in cells whose faces ALL have a density jump but (almost)
// no pressure jump, dP/P <= kappa drho/rho (shocks and adiabatic flow have dP/P ~ Gamma1 drho/rho, Gamma1 >= 1.1).
// The frozen EOS is a tangent fit: exact for a contact, wrong across shocks and their feet (measured: a pressure-
// jump threshold left R3 with a resolution-independent energy error; a per-face gate without the reset made R2/R4
// plateau errors grow with N). Other cells are exactly the standard conservative scheme.
//
// MESH REFINEMENT (v6, 2026-10-06). The increment of a face energy flux for a frozen EOS (g, e0) is
// X/(g-1) + e0 Y - Z with face sums X = aL PL + aR PR, Y = aL rhoL + aR rhoR, Z = aL E_tab,L + aR E_tab,R. X, Y, Z
// are stored as the FLUXES of the field dflux_xyz, so Parthenon's flux correction restricts them at fine/coarse
// faces exactly like the hydro fluxes and a coarse cell gets the own-EOS increment of the flux it actually receives
// (the expression is linear in X, Y, Z, so the area average is exact).
//
// SCOPE (refused otherwise at startup): eos = hydrogen, riemann = hlld, reconstruction plm or dc, integrator vl2
// (both stages build on the step-start state, which is what the step-start freeze requires), no first-order flux
// correction. Dual energy: no extra coupling (see hydro.cpp; switched cells keep the advected energy).
//
// Everything lives in this separate translation unit on purpose: the production flux kernel and the HLLD
// header are untouched, so with the key off the build is bit-identical (E-F9: inline-body changes in shared
// headers change GCC's inlining/FMA decisions).
//========================================================================================
#include <parthenon/package.hpp>

namespace Hydro {
using parthenon::MeshData;
using parthenon::Real;
using parthenon::TaskStatus;

//! Stage s: (stage 1) freeze g*, e0* per cell from the step-start state; reconstruct the face states exactly as
//! the flux kernel does and store the face sums X, Y, Z in the fluxes of dflux_xyz. Must run BEFORE the flux
//! correction is sent.
TaskStatus DoubleFluxCoefficients(MeshData<Real> *md, const int stage);
//! After UpdateWithFluxDivergence, in double-flux cells only: E_i += beta dt sum_dir (inc_lower - inc_upper)/dx with
//! inc = X/(g_i-1) + e0_i Y - Z, then rebuild E from the table at the frozen-EOS pressure.
TaskStatus DoubleFluxCorrect(MeshData<Real> *md, const Real beta_dt);
//! EOS-CONSISTENT FILL ON REFINED MESHES (hydro/double_flux_ghost_fill, default true with double flux on a refined
//! mesh). Prolongation, restriction and regridding interpolate rho and E separately, which mixes rho e linearly (the
//! Abgrall effect, in cells the double-flux reset never sees; E7v/E7w). The PRESSURE is carried as its own field
//! dflux_p (FillGhost, same refinement operators as cons): DoubleFluxRefreshP sets it to P(cons) in block interiors
//! before every boundary send (Parthenon Mesh::PreSendBoundBufsHook) and after every ConsToPrim, so Parthenon
//! restricts it (average of the fine P), prolongates it (the limited stencil applied to P) and moves it through
//! regrids. Wherever the transported P and P(cons) disagree -- ghost regions next to a block on another level
//! (DoubleFluxGhostFill, before ConsToPrim) and regridded interiors (DoubleFluxPostRegrid, Parthenon
//! Mesh::PostRegridHook) -- E is rebuilt from the table at the transported P (rho, momenta, B as interpolated).
void DoubleFluxRefreshP(MeshData<Real> *md);
TaskStatus DoubleFluxGhostFill(MeshData<Real> *md);
void DoubleFluxPostRegrid(parthenon::Mesh *pm);
//! Hydro's FillDerivedMesh when the fill is on (registered instead of ConsToPrim, so the production ConsToPrim and
//! its translation unit stay untouched): ghost fill, ConservedToPrimitive, refresh of dflux_p.
void ConsToPrimDoubleFlux(MeshData<Real> *md);
} // namespace Hydro

#endif // HYDRO_DOUBLE_FLUX_HPP_
