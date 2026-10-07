//========================================================================================
// AthenaPK - validation problem generator for the TABULATED protostellar EOS (eos=hydrogen).
// validation_tests/eos_tabulated (2026-10-04).
//
// Every state is set THROUGH THE TABLE on the device (EintFromRhoTk / Pof / Cs2of),
// so the initial data are consistent with exactly the EOS the hydro solver uses. Densities are in
// CODE units (rho_unit of the run, 5.467e-19 g/cm^3 for the FHC units), temperatures in K.
//
//   mode = wave      planar acoustic eigenmode along x on a uniform (rho0, T0) state:
//                    drho = amp rho0 sin(kx), dv = amp cs sin(kx), dP = cs^2 drho, with cs^2 the
//                    TABLE's adiabatic sound speed -> after one period the exact solution is the IC.
//   mode = riemann   left/right states (rhoL, TL, uL) | (rhoR, TR, uR), split at x0 (along x).
//                    Optional espL_code / espR_code (2026-10-05, E10 floor probe): set the SPECIFIC
//                    internal energy directly (code units, may be <= 0 or below the table) instead of
//                    through T; this is how a sub-table-edge or negative-energy cell is planted.
//   mode = slab      (2026-10-06, double flux in 2D/3D) periodic OBLIQUE pure contacts: wave vector
//                    k = 2 pi (mx/Lx, my/Ly, mz/Lz) (problem/eos_test/mx,my,mz, integers, default 1,0,0); the
//                    state is (rhoL, espL_code | TL) where sin(k.x) >= 0 and (rhoR, espR_code | TR) elsewhere,
//                    velocity u_slab along k/|k| everywhere. Exact solution: translation along k at uniform P.
//   mode = compress  uniform (rho0, T0) with planar homologous inflow v = -alpha (x - xc): the
//                    interior stays uniform with rho(t) = rho0 / (1 - alpha t) until the boundary
//                    signal arrives, so the central T(rho) must follow the table ADIABAT.
//                    With ix1_bc/ox1_bc = eos_homologous (registered in main.cpp) the ghosts carry the
//                    exact homologous continuation (rho and esp of the edge cell, v proportional to
//                    x - xc), so the whole box stays uniform: a one-zone test of the scheme's PdV work
//                    with the tabulated pressure (E8).
//========================================================================================
#include <cmath>
#include <iostream>
#include <string>

#include <parthenon/package.hpp>

#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
#include "pgen.hpp"

namespace eos_test {
using namespace parthenon::driver::prelude;
Real homologous_xc = 0.0; // set by ProblemGenerator (problem/eos_test/xc), read by HomologousX1

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  const std::string blk = "problem/eos_test";
  const std::string mode = pin->GetString(blk, "mode");
  PARTHENON_REQUIRE_THROWS(mode == "wave" || mode == "riemann" || mode == "compress" || mode == "slab",
                           "eos_test: mode must be wave, riemann, compress or slab");
  auto hydro_pkg = pmb->packages.Get("Hydro");
  PARTHENON_REQUIRE_THROWS(hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd,
                           "eos_test expects fluid = glmmhd");
  const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto tab = eos.GetEosTable();

  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real rho0 = pin->GetOrAddReal(blk, "rho0", 1.0);
  const Real T0 = pin->GetOrAddReal(blk, "T0", 10.0);
  const Real amp = pin->GetOrAddReal(blk, "amp", 1.0e-6);
  const Real kw = 2.0 * M_PI / (x1max - x1min);
  const Real rhoL = pin->GetOrAddReal(blk, "rhoL", 1.0), TL = pin->GetOrAddReal(blk, "TL", 10.0);
  const Real rhoR = pin->GetOrAddReal(blk, "rhoR", 1.0), TR = pin->GetOrAddReal(blk, "TR", 10.0);
  const Real uL = pin->GetOrAddReal(blk, "uL", 0.0), uR = pin->GetOrAddReal(blk, "uR", 0.0);
  const bool setL = pin->DoesParameterExist(blk, "espL_code");
  const bool setR = pin->DoesParameterExist(blk, "espR_code");
  const Real espL = setL ? pin->GetReal(blk, "espL_code") : 0.0;
  const Real espR = setR ? pin->GetReal(blk, "espR_code") : 0.0;
  const Real x0 = pin->GetOrAddReal(blk, "x0", 0.5 * (x1min + x1max));
  const Real alpha = pin->GetOrAddReal(blk, "alpha", 0.0);
  const Real xc = pin->GetOrAddReal(blk, "xc", 0.5 * (x1min + x1max));
  const int imode = (mode == "wave") ? 0 : (mode == "riemann" ? 1 : (mode == "compress" ? 2 : 3));
  // slab mode only (the other modes never read these keys, so their decks/params are unchanged)
  Real kx = 0.0, ky = 0.0, kz = 0.0, uslab = 0.0;
  if (imode == 3) {
    const Real Ly = pin->GetReal("parthenon/mesh", "x2max") - pin->GetReal("parthenon/mesh", "x2min");
    const Real Lz = pin->GetReal("parthenon/mesh", "x3max") - pin->GetReal("parthenon/mesh", "x3min");
    kx = 2.0 * M_PI * pin->GetOrAddInteger(blk, "mx", 1) / (x1max - x1min);
    ky = 2.0 * M_PI * pin->GetOrAddInteger(blk, "my", 0) / Ly;
    kz = 2.0 * M_PI * pin->GetOrAddInteger(blk, "mz", 0) / Lz;
    uslab = pin->GetOrAddReal(blk, "u_slab", 0.0);
  }
  const Real kn = std::sqrt(kx * kx + ky * ky + kz * kz);
  homologous_xc = xc;

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;
  const auto ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  const auto jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  const auto kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);

  pmb->par_for(
      "eos_test_ic", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        Real rho, v = 0.0, eint;
        Real vy = 0.0, vz = 0.0;
        if (imode == 3) {
          const Real ph = kx * x + ky * coords.Xc<2>(j) + kz * coords.Xc<3>(k);
          const bool left = (std::sin(ph) >= 0.0);
          rho = left ? rhoL : rhoR;
          if (left ? setL : setR)
            eint = rho * (left ? espL : espR);
          else
            eint = tab.EintFromRhoTk(rho, left ? TL : TR);
          v = uslab * kx / kn;
          vy = uslab * ky / kn;
          vz = uslab * kz / kn;
        } else if (imode == 0) {
          const Real esp0 = tab.EintFromRhoTk(rho0, T0) / rho0;
          const Real P0 = tab.Pof(rho0, esp0);
          const Real cs2 = tab.Cs2of(rho0, esp0);
          const Real d = amp * std::sin(kw * (x - x1min));
          rho = rho0 * (1.0 + d);
          v = std::sqrt(cs2) * d;
          // Isentropic perturbation to first order WITHOUT the P->e bisection (EspFromP resolves
          // e only to ~1.6e-5 at the default 20 bisections, far above amp = 1e-6):
          // d(rho esp) = (esp0 + P0/rho0) drho = h0 drho.
          eint = rho0 * esp0 + (esp0 + P0 / rho0) * rho0 * d;
        } else if (imode == 1) {
          const bool left = (x < x0);
          rho = left ? rhoL : rhoR;
          v = left ? uL : uR;
          if (left ? setL : setR)
            eint = rho * (left ? espL : espR);
          else
            eint = tab.EintFromRhoTk(rho, left ? TL : TR);
        } else {
          rho = rho0;
          v = -alpha * (x - xc);
          eint = tab.EintFromRhoTk(rho0, T0);
        }
        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * v;
        u(IM2, k, j, i) = rho * vy;
        u(IM3, k, j, i) = rho * vz;
        u(IEN, k, j, i) = eint + 0.5 * rho * (v * v + vy * vy + vz * vz);
        u(IB1, k, j, i) = 0.0;
        u(IB2, k, j, i) = 0.0;
        u(IB3, k, j, i) = 0.0;
        u(IPS, k, j, i) = 0.0;
      });
}

//! Homologous continuation along x1 for mode = compress (E8). SINGLE meshblock in x1 (the 1D test box): every
//! ghost gets rho and esp = the MEAN of the two edge cells and v = -alpha_eff (x_g - xc), with
//! alpha_eff = -(v_ie - v_is)/(x_ie - x_is) read off the box itself (no clock needed). Symmetric by construction:
//! the first version extrapolated each side from its own edge cell, v_g = v_a (x_g - xc)/(x_a - xc), which
//! AMPLIFIES a uniform translation (ratio > 1) and grew a round-off offset to O(1) in ~5 time units (E8, 2026-10-05).
//! B = 0. Fine-grid, uniform-mesh, single-block use only.
template <bool INNER>
void HomologousX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto ir = bounds.GetBoundsI(parthenon::IndexDomain::interior);
  const int is = ir.s, ie = ir.e;
  const auto &coords = pmb->coords;
  const Real xc = homologous_xc;
  constexpr auto dom = INNER ? parthenon::IndexDomain::inner_x1 : parthenon::IndexDomain::outer_x1;
  pmb->par_for_bndry(
      "eos_test::HomologousX1", IndexRange{0, 0}, dom, parthenon::TopologicalElement::CC,
      coarse, false, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const Real rs = cons(IDN, k, j, is), re = cons(IDN, k, j, ie);
        const Real vs = cons(IM1, k, j, is) / rs, ve = cons(IM1, k, j, ie) / re;
        const Real es = (cons(IEN, k, j, is) - 0.5 * rs * vs * vs) / rs;
        const Real ee = (cons(IEN, k, j, ie) - 0.5 * re * ve * ve) / re;
        const Real rho = 0.5 * (rs + re), esp = 0.5 * (es + ee);
        const Real alpha = -(ve - vs) / (coords.Xc<1>(ie) - coords.Xc<1>(is));
        const Real vg = -alpha * (coords.Xc<1>(i) - xc);
        cons(IDN, k, j, i) = rho;
        cons(IM1, k, j, i) = rho * vg;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = 0.0;
        cons(IEN, k, j, i) = rho * esp + 0.5 * rho * vg * vg;
        cons(IB1, k, j, i) = 0.0;
        cons(IB2, k, j, i) = 0.0;
        cons(IB3, k, j, i) = 0.0;
        cons(IPS, k, j, i) = 0.0;
      });
}
void HomologousInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  HomologousX1<true>(mbd, coarse);
}
void HomologousOuterX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  HomologousX1<false>(mbd, coarse);
}

} // namespace eos_test
