
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2024, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

// Parthenon headers
#include "mesh/mesh.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "utils/error_checking.hpp"

namespace diffusion {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  const auto gamma = pin->GetReal("hydro", "gamma");
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  const auto Bx = pin->GetOrAddReal("problem/diffusion", "Bx", 0.0);
  const auto By = pin->GetOrAddReal("problem/diffusion", "By", 0.0);

  const auto iprob = pin->GetInteger("problem/diffusion", "iprob");
  // Non-ideal MHD tests of this tree (Ohmic + ambipolar). The thermal-conduction, viscosity and
  // Hall problems of the full code (iprob 0-30, 60) are not part of it.
  PARTHENON_REQUIRE_THROWS(mhd_enabled, "the diffusion pgen requires MHD (hydro/fluid = glmmhd).");
  PARTHENON_REQUIRE_THROWS(iprob == 40 || iprob == 41 || iprob == 42 || iprob == 43 || iprob == 44 ||
                               iprob == 45 || iprob == 50 || iprob == 51,
                           "diffusion pgen: iprob must be one of 40-45, 50, 51 in this tree.");
  Real t0 = 0.5;
  Real diff_coeff = 0.0;
  Real amp = 1e-6;
  // Get common parameters for Gaussian profile
  if (iprob == 40) {
    t0 = pin->GetOrAddReal("problem/diffusion", "t0", t0);
    amp = pin->GetOrAddReal("problem/diffusion", "amp", amp);
  }
  // Ohmic diffusion of 1D Gaussian
  if (iprob == 40) {
    diff_coeff = pin->GetReal("diffusion", "ohm_diff_coeff_code");
  }

  // Ambipolar diffusion of a 1D sinusoidal transverse field (eigenmode decay test).
  // With a uniform guide field B0 along x and By = amp*sin(k x), the current is purely
  // perpendicular to B, so E_A = eta_A J and By decays as exp(-eta_A k^2 t) with
  // eta_A = Q_A B0^2. The analytic decay rate is therefore Q_A * B0^2 * k^2.
  Real kpar = 0.0;
  // AUDIT 2026-09-14: optional OBLIQUE guide field B0 = (Bx, Bguide_y, 0) with the
  // perturbation moved to B_z = amp sin(kx). With Bguide_y = 0 the setup is bit-identical to
  // the historical test. With Bguide_y != 0 the current J = (0, -dBz/dx, 0) has a component
  // ALONG B, so the (J.b)b projection of the ambipolar EMF is exercised: the damping rate is
  // eta_A k^2 cos^2(theta) and the Alfven frequency k v_A cos(theta), cos(theta)=Bx/|B0|.
  // The theta = 0 test is blind to the projection term (J.B = 0 identically).
  Real bguide_y = 0.0;
  if (iprob == 50) {
    PARTHENON_REQUIRE_THROWS(mhd_enabled, "iprob=50 (ambipolar decay) requires MHD.");
    amp = pin->GetOrAddReal("problem/diffusion", "amp", amp);
    bguide_y = pin->GetOrAddReal("problem/diffusion", "Bguide_y", 0.0);
    const auto nwave = pin->GetOrAddReal("problem/diffusion", "nwave", 1.0);
    const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
    const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
    kpar = 2.0 * M_PI * nwave / (x1max - x1min);
  }
  // iprob=51: ambipolar diffusion of a LOCALIZED Gaussian By wave-packet (not the global
  // sinusoid of iprob=50 -- a periodic sin/cos has a geometric node/antinode structure in
  // |By(x)| that confounds a "near bump vs far field" amplitude comparison with genuine
  // diffusive decay) on a GUIDE FIELD that itself has a localized Gaussian "bump" in |B1|
  // around x=0: B1(x) = Bx*(1 + (bump_ratio-1)*exp(-(x/bump_width)^2)),
  // By(x,0) = amp*exp(-(x/wave_width)^2). Since AmbipolarCoeff::fixed has eta_A=Q_A*B^2,
  // the guide-field bump creates a strongly spatially-varying eta_A (peak/floor ratio =
  // bump_ratio^2) WITHOUT touching any diffusivity code -- purpose-built to isolate whether
  // the CT edge-EMF's extra spatial averaging (arithmetic mean over 4 bounding faces, each
  // itself a 2-cell eta average) under-resolves a diffusivity gradient that is sharp on the
  // grid scale (bump_width ~ few dx) relative to the validated GLM flux-form path (which
  // deposits each face's EMF directly, no edge-sharing average). bump_width >> dx is the
  // smooth control case where both schemes should agree with each other. Since J3=dBy/dx is
  // the ONLY nonzero current for a field varying purely in x (B3=0 always here), and
  // dE3/dy=dE2/dz=0 identically, ambipolar diffusion analytically leaves B1(x) exactly
  // frozen -- any drift of B1 away from its IC is a discrete-force-imbalance/PLM-truncation
  // artifact seeding real MHD dynamics (compressive/magnetosonic), NOT part of the AD
  // mechanism under test; t_fin must stay well inside the local ambipolar e-folding time yet
  // well below the bump's sound-crossing time so B1 stays numerically static (checked
  // per-run against the phdf output, not just analytically here).
  // eint(x) is chosen for EXACT total-pressure (thermal + magnetic) equilibrium:
  // P_th(x) + B1(x)^2/2 = bump_p0 + B0^2/2, i.e.
  // eint(x) = (bump_p0 - (B1(x)^2 - B0^2)/2) / (gamma-1). bump_p0 (baseline thermal
  // pressure, rho0=1) must be picked well above the peak magnetic pressure delta
  // 0.5*(Bx*bump_ratio)^2 - 0.5*Bx^2 so eint(x) stays positive (checked below).
  Real bump_ratio = 1.0, bump_width = 1.0, bump_p0 = 1.0, wave_width = 1.0;
  if (iprob == 51) {
    PARTHENON_REQUIRE_THROWS(mhd_enabled, "iprob=51 (ambipolar variable-eta) requires MHD.");
    amp = pin->GetOrAddReal("problem/diffusion", "amp", amp);
    bump_ratio = pin->GetOrAddReal("problem/diffusion", "bump_ratio", 10.0);
    bump_width = pin->GetOrAddReal("problem/diffusion", "bump_width", 0.05);
    bump_p0 = pin->GetOrAddReal("problem/diffusion", "bump_p0", 20.0);
    wave_width = pin->GetOrAddReal("problem/diffusion", "wave_width", 3.0 * bump_width);
    const Real pmag_delta = 0.5 * (SQR(Bx * bump_ratio) - SQR(Bx));
    PARTHENON_REQUIRE_THROWS(bump_p0 > pmag_delta,
                             "iprob=51: bump_p0 too small for total-pressure equilibrium "
                             "(would give negative eint at the bump peak).");
  }

  // iprob=41 (2026-10-06, validation_tests/dual_energy D5): CIRCULARLY POLARIZED transverse field
  // B = (Bx, amp sin kx, amp cos kx), v = 0, uniform rho = 1 and eint (problem/diffusion/eint0, default
  // 1/(gamma(gamma-1))). |B| is uniform, so there is no magnetic-pressure gradient and no compression.
  // Bx = 0: force-free (J = k B), Ohmic decay B ~ exp(-eta k^2 t) with NO motion; every cell gains
  //   rho de = (amp^2/2)(1 - exp(-2 eta k^2 t)) exactly; ambipolar does nothing (J_perp = 0).
  // Bx != 0: damped circular Alfven wave, exact at finite amplitude (|B| stays uniform, so eta_A = Q_A |B|^2
  //   is uniform): s^2 + eta_eff k^2 s + k^2 Bx^2/rho = 0 with eta_eff = eta_O + Q_A Bx^2, and the
  //   heating rate eta_eff k^2 |b_perp|^2 is uniform in space.
  Real eint41 = 0.0;
  bool oblique41 = false;
  Real k41[3] = {0.0, 0.0, 0.0}, n41[3] = {1.0, 0.0, 0.0}, e141[3] = {0.0, 1.0, 0.0}, e241[3] = {0.0, 0.0, 1.0};
  if (iprob == 41) {
    PARTHENON_REQUIRE_THROWS(mhd_enabled, "iprob=41 (circular decay) requires MHD.");
    amp = pin->GetOrAddReal("problem/diffusion", "amp", amp);
    const auto nwave = pin->GetOrAddReal("problem/diffusion", "nwave", 1.0);
    const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
    const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
    kpar = 2.0 * M_PI * nwave / (x1max - x1min);
    eint41 = pin->GetOrAddReal("problem/diffusion", "eint0", 1.0 / (gamma * (gamma - 1.0)));
    // OBLIQUE (2026-10-07, 3D gap closure): my, mz (default 0) give k = (kpar, 2 pi my/Ly, 2 pi mz/Lz); the guide field
    // Bx is then along k/|k| and the rotating part in the plane perpendicular to k (unit vectors e1, e2), so |B|
    // stays uniform and the exact answer holds with |k|. my = mz = 0: the original setup bit for bit.
    const int my = pin->GetOrAddInteger("problem/diffusion", "my", 0), mz = pin->GetOrAddInteger("problem/diffusion", "mz", 0);
    if (my != 0 || mz != 0) {
      k41[0] = kpar;
      k41[1] = my ? 2.0 * M_PI * my / (pin->GetReal("parthenon/mesh", "x2max") - pin->GetReal("parthenon/mesh", "x2min")) : 0.0;
      k41[2] = mz ? 2.0 * M_PI * mz / (pin->GetReal("parthenon/mesh", "x3max") - pin->GetReal("parthenon/mesh", "x3min")) : 0.0;
      const Real kn = std::sqrt(k41[0] * k41[0] + k41[1] * k41[1] + k41[2] * k41[2]);
      for (int c = 0; c < 3; ++c) n41[c] = k41[c] / kn;
      // e1 = normalize(n x z) (or n x y if n || z), e2 = n x e1
      Real ax[3] = {0.0, 0.0, 1.0};
      if (std::abs(n41[2]) > 0.9) { ax[1] = 1.0; ax[2] = 0.0; }
      Real e1[3] = {n41[1] * ax[2] - n41[2] * ax[1], n41[2] * ax[0] - n41[0] * ax[2], n41[0] * ax[1] - n41[1] * ax[0]};
      const Real e1n = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
      for (int c = 0; c < 3; ++c) e141[c] = e1[c] / e1n;
      e241[0] = n41[1] * e141[2] - n41[2] * e141[1];
      e241[1] = n41[2] * e141[0] - n41[0] * e141[2];
      e241[2] = n41[0] * e141[1] - n41[1] * e141[0];
      oblique41 = true;
    }
  }

  // iprob 42 (VALIDATION, nonideal_mhd A5, 2026-10-11): Ohmic diffusion across an eta jump. Density rho_in on
  // x_l < x < x_r and 1 outside, uniform pressure p0 (static), By = B_in inside / B_out outside (Bx = Bz = 0, tiny so
  // beta >> 1 and the hydro stays inert). With diffusion/resistivity_coeff = density_power the two media have
  // eta = c rho^p; the exact answer near each interface is the two-medium (composite) solution with interface value
  // B_i = (B_in sqrt(eta_in) + B_out sqrt(eta_out)) / (sqrt(eta_in) + sqrt(eta_out)) and erf profiles on each side.
  Real rho42 = 1.0, Bin42 = 0.0, Bout42 = 0.0, xl42 = 0.0, xr42 = 0.0, p042 = 1.0;
  if (iprob == 42) {
    PARTHENON_REQUIRE_THROWS(mhd_enabled, "iprob=42 requires MHD.");
    rho42 = pin->GetOrAddReal("problem/diffusion", "rho_in", 0.01);
    Bin42 = pin->GetOrAddReal("problem/diffusion", "B_in", 1.0e-3);
    Bout42 = pin->GetOrAddReal("problem/diffusion", "B_out", 0.0);
    xl42 = pin->GetOrAddReal("problem/diffusion", "x_l", 0.25);
    xr42 = pin->GetOrAddReal("problem/diffusion", "x_r", 0.75);
    p042 = pin->GetOrAddReal("problem/diffusion", "p0", 1.0);
  }

  // iprob 43 (VALIDATION, nonideal_mhd A6, 2026-10-11): nonlinear ambipolar spreading. Bz(x) only (B perp x, J perp B),
  // AD with eta_A = Q B^2 (diffusion/ambipolar_coeff = fixed, ambipolar_coeff_code = Q) gives the porous-medium equation
  // dBz/dt = d/dx(Q Bz^2 dBz/dx), whose Barenblatt-Pattle solution (m = 3) is
  //   Bz = B0 (t/t0)^(-1/4) sqrt(1 - (x-xc)^2 / (xf0^2 (t/t0)^(1/2)))_+ ,  t0 = xf0^2 / (4 Q B0^2).
  // The simulation starts at t0 (simulation time 0); uniform rho = 1, p = p0, beta >> 1 for B0 << 1.
  Real B043 = 0.0, xf043 = 0.0, xc43 = 0.0;
  if (iprob == 43) {
    PARTHENON_REQUIRE_THROWS(mhd_enabled, "iprob=43 requires MHD.");
    B043 = pin->GetOrAddReal("problem/diffusion", "baren_B0", 1.0e-3);
    xf043 = pin->GetOrAddReal("problem/diffusion", "baren_xf0", 0.1);
    xc43 = pin->GetOrAddReal("problem/diffusion", "baren_xc", 0.5);
    p042 = pin->GetOrAddReal("problem/diffusion", "p0", 1.0);
  }

  // iprob 44 (VALIDATION, nonideal_mhd A4, 2026-10-11): 3-D ABC (Arnold-Beltrami-Childress) field on the periodic unit
  // cube, B = amp (A sin kz + C cos ky, B sin kx + A cos kz, C sin ky + B cos kx), k = 2 pi nwave: curl B = k B, so
  // J x B = 0. Ambipolar diffusion must leave it unchanged; Ohmic (fixed eta) decays every component as exp(-eta k^2 t).
  // rho = 1, p = p0 uniform (beta >> 1 for small amp; force-free anyway).
  Real abcA = 0.0, abcB = 0.0, abcC = 0.0, abcamp = 0.0, abck = 0.0;
  if (iprob == 44) {
    PARTHENON_REQUIRE_THROWS(mhd_enabled, "iprob=44 requires MHD.");
    abcA = pin->GetOrAddReal("problem/diffusion", "abc_A", 1.0);
    abcB = pin->GetOrAddReal("problem/diffusion", "abc_B", std::sqrt(2.0 / 3.0));
    abcC = pin->GetOrAddReal("problem/diffusion", "abc_C", std::sqrt(1.0 / 3.0));
    abcamp = pin->GetOrAddReal("problem/diffusion", "amp", 1.0e-3);
    abck = 2.0 * M_PI * pin->GetOrAddReal("problem/diffusion", "nwave", 1.0);
    p042 = pin->GetOrAddReal("problem/diffusion", "p0", 1.0);
  }

  // iprob 45 (VALIDATION, nonideal_mhd A10, 2026-10-11): ambipolar C-shock. Uniform (rho0, +u, P0, By = B0) driven into a
  // reflecting wall at x1max; the shock runs back into the gas and relaxes to the steady single-fluid AD profile
  // (validation_tests/nonideal_mhd/scripts/cshock_ref.py). Use diffusion/ambipolar_coeff = fixed (eta_A = Q B^2).
  Real cs_rho = 1.0, cs_u = 0.0, cs_P = 1.0, cs_B = 0.0;
  if (iprob == 45) {
    PARTHENON_REQUIRE_THROWS(mhd_enabled, "iprob=45 requires MHD.");
    cs_rho = pin->GetOrAddReal("problem/diffusion", "cs_rho0", 1.0);
    cs_u = pin->GetOrAddReal("problem/diffusion", "cs_u", 2.0);
    cs_P = pin->GetOrAddReal("problem/diffusion", "cs_P0", 0.1);
    cs_B = pin->GetOrAddReal("problem/diffusion", "cs_B0", 2.0);
  }

  // hydro/dual_energy: the advected internal energy (last cons slot) must start equal to rho*eint
  // (validation_tests/dual_energy, 2026-10-06; left at 0 a switched cell takes the energy floor).
  const int idual = hydro_pkg->Param<int>("dual_idx");

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Diffusion", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        u(IDN, k, j, i) = 1.0;

        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        if (mhd_enabled) {
          u(IB1, k, j, i) = 0.0;
          u(IB2, k, j, i) = 0.0;
          u(IB3, k, j, i) = 0.0;
        }

        Real eint = -1.0;
        if (iprob == 40) {
          u(IB2, k, j, i) =
              amp / std::pow(std::sqrt(4. * M_PI * diff_coeff * t0), 1.0) *
              std::exp(-(std::pow(coords.Xc<1>(i), 2.)) / (4. * diff_coeff * t0));
          eint = 1.0 / (gamma * (gamma - 1.0)); // c_s = 1 everywhere
          // Circularly polarized transverse field (dual-energy heating test)
        } else if (iprob == 41) {
          if (!oblique41) {
            u(IB1, k, j, i) = Bx;
            u(IB2, k, j, i) = amp * std::sin(kpar * coords.Xc<1>(i));
            u(IB3, k, j, i) = amp * std::cos(kpar * coords.Xc<1>(i));
          } else {
            const Real ph = k41[0] * coords.Xc<1>(i) + k41[1] * coords.Xc<2>(j) + k41[2] * coords.Xc<3>(k);
            const Real s = amp * std::sin(ph), c = amp * std::cos(ph);
            u(IB1, k, j, i) = Bx * n41[0] + s * e141[0] + c * e241[0];
            u(IB2, k, j, i) = Bx * n41[1] + s * e141[1] + c * e241[1];
            u(IB3, k, j, i) = Bx * n41[2] + s * e141[2] + c * e241[2];
          }
          eint = eint41;
        } else if (iprob == 42) {
          const bool in = coords.Xc<1>(i) > xl42 && coords.Xc<1>(i) < xr42;
          u(IDN, k, j, i) = in ? rho42 : 1.0;
          u(IB2, k, j, i) = in ? Bin42 : Bout42;
          eint = p042 / (u(IDN, k, j, i) * (gamma - 1.0));
        } else if (iprob == 43) {
          const Real q = 1.0 - SQR((coords.Xc<1>(i) - xc43) / xf043);
          u(IB3, k, j, i) = q > 0.0 ? B043 * std::sqrt(q) : 0.0;
          eint = p042 / (gamma - 1.0);
        } else if (iprob == 44) {
          const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
          u(IB1, k, j, i) = abcamp * (abcA * std::sin(abck * z) + abcC * std::cos(abck * y));
          u(IB2, k, j, i) = abcamp * (abcB * std::sin(abck * x) + abcA * std::cos(abck * z));
          u(IB3, k, j, i) = abcamp * (abcC * std::sin(abck * y) + abcB * std::cos(abck * x));
          eint = p042 / (gamma - 1.0);
        } else if (iprob == 45) {
          u(IDN, k, j, i) = cs_rho;
          u(IM1, k, j, i) = cs_rho * cs_u;
          u(IB2, k, j, i) = cs_B;
          eint = cs_P / (cs_rho * (gamma - 1.0));
          // Ambipolar diffusion of a 1D sinusoidal transverse field
        } else if (iprob == 50) {
          u(IB1, k, j, i) = Bx; // uniform guide field B0 along x
          if (bguide_y != 0.0) { // oblique variant (audit 2026-09-14): perturb B_z
            u(IB2, k, j, i) = bguide_y;
            u(IB3, k, j, i) = amp * std::sin(kpar * coords.Xc<1>(i));
          } else {
            u(IB2, k, j, i) = amp * std::sin(kpar * coords.Xc<1>(i));
          }
          eint = 1.0 / (gamma * (gamma - 1.0)); // c_s = 1 everywhere
          // Ambipolar diffusion with a spatially-varying guide field |B1(x)| (localized
          // Gaussian bump at x=0) -- see comment above (variable-eta_A CT-vs-GLM probe).
        } else if (iprob == 51) {
          const Real xx = coords.Xc<1>(i);
          const Real b1x = Bx * (1.0 + (bump_ratio - 1.0) * std::exp(-SQR(xx / bump_width)));
          u(IB1, k, j, i) = b1x;
          // Localized Gaussian wave packet (not a global sin/cos): see comment above.
          u(IB2, k, j, i) = amp * std::exp(-SQR(xx / wave_width));
          // Total-pressure-balanced eint(x): see comment above (iprob=51 setup).
          eint = (bump_p0 - 0.5 * (SQR(b1x) - SQR(Bx))) / (gamma - 1.0);
        }

        PARTHENON_REQUIRE(eint > 0.0, "Missing init of eint");
        u(IEN, k, j, i) =
            u(IDN, k, j, i) * eint +
            0.5 * ((SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                   u(IDN, k, j, i));

        if (mhd_enabled) {
          u(IEN, k, j, i) +=
              0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
        }
        if (idual >= 0) u(idual, k, j, i) = u(IDN, k, j, i) * eint;
      });

}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkAfterLoop
//! \brief Quantitative check for the ambipolar diffusion eigenmode test (iprob=50).
//!
//! A transverse perturbation By = amp*sin(kx) on a uniform guide field B0 (density rho0=1,
//! velocity 0) evolves as a *damped Alfven wave*: the Lorentz force launches an Alfven
//! wave while ambipolar diffusion (a perpendicular resistivity eta_A = Q_A B0^2) damps it.
//! The linear dispersion relation is
//!     s^2 + eta_A k^2 s + k^2 v_A^2 = 0 ,   v_A^2 = B0^2/rho0 ,
//! with roots s1, s2. For the initial condition By(0)=amp, vy(0)=0 the exact solution is
//!     By(t) = amp/(s1 - s2) * (s1*exp(s1 t) - s2*exp(s2 t)) * sin(kx) .
//! We measure the By amplitude at the final time and compare to this analytic value. The
//! input is chosen to be (over)damped (eta_A k > 2 v_A) so the mode decays monotonically.
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm) {
  const auto iprob = pin->GetInteger("problem/diffusion", "iprob");
  if (iprob != 50) return;



  //--------------------------------------------------------------------------------------
  // Ambipolar diffusion damped-Alfven amplitude check (iprob=50).
  const auto B0 = pin->GetOrAddReal("problem/diffusion", "Bx", 0.0);
  const auto amp = pin->GetOrAddReal("problem/diffusion", "amp", 1e-6);
  const auto nwave = pin->GetOrAddReal("problem/diffusion", "nwave", 1.0);
  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real kpar = 2.0 * M_PI * nwave / (x1max - x1min);
  const Real QA = pin->GetReal("diffusion", "ambipolar_coeff_code");
  const Real tfin = tm.time;

  const Real rho0 = 1.0; // set in the problem generator for iprob=50
  // Oblique variant (audit 2026-09-14): B0 = (Bx, Bguide_y, 0), perturbation in B_z.
  const Real bgy = pin->GetOrAddReal("problem/diffusion", "Bguide_y", 0.0);
  const Real B0sq = B0 * B0 + bgy * bgy;
  const Real cos2 = B0 * B0 / B0sq; // cos^2(theta), = 1 for the historical test
  const Real eta_A = QA * B0sq;
  const Real vA2 = B0sq / rho0;
  const Real bb = eta_A * kpar * kpar * cos2; // = eta_A k^2 cos^2(theta)
  const Real cc = kpar * kpar * vA2 * cos2;   // = (k v_A cos(theta))^2
  const int ipert = (bgy != 0.0) ? IB3 : IB2;
  const Real disc = bb * bb - 4.0 * cc;

  // Analytic amplitude of the damped Alfven mode at t = tfin.
  Real A_pred;
  if (disc >= 0.0) { // (over)damped: two real roots
    const Real sq = std::sqrt(disc);
    const Real s1 = 0.5 * (-bb - sq); // fast
    const Real s2 = 0.5 * (-bb + sq); // slow
    A_pred = amp / (s1 - s2) * (s1 * std::exp(s1 * tfin) - s2 * std::exp(s2 * tfin));
  } else { // underdamped: complex roots, real envelope form
    const Real omega = 0.5 * std::sqrt(-disc);
    A_pred = amp * std::exp(-0.5 * bb * tfin) *
             (std::cos(omega * tfin) + 0.5 * bb / omega * std::sin(omega * tfin));
  }

  // Volume-weighted mean of By^2 over the domain. For By = A sin(kx), <By^2> = A^2/2.
  Real sum_by2 = 0.0, vol_tot = 0.0;
  for (auto &pmb : mesh->block_list) {
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    auto &rc = pmb->meshblock_data.Get();
    auto u = rc->Get("cons").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real vol = pmb->coords.CellVolume(k, j, i);
          sum_by2 += SQR(u(ipert, k, j, i)) * vol;
          vol_tot += vol;
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &sum_by2, 1, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vol_tot, 1, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  const Real A_meas = std::sqrt(2.0 * sum_by2 / vol_tot);
  const Real rel_err_amp = std::abs(A_meas - std::abs(A_pred)) / std::abs(A_pred);

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "\n=== Ambipolar diffusion damped-Alfven test (iprob=50) ===\n"
              << "  B0 = " << B0 << ", rho0 = " << rho0 << ", k = " << kpar
              << ", Q_A = " << QA << ", t = " << tfin << "\n"
              << "  Bguide_y = " << bgy << ", cos^2(theta) = " << cos2 << "\n"
              << "  eta_A = Q_A B0^2 = " << eta_A << ",  eta_A k^2 cos^2 = " << bb
              << ",  v_A = " << std::sqrt(vA2)
              << (disc >= 0.0 ? "  (overdamped)\n" : "  (underdamped)\n")
              << "  analytic amplitude A_pred = " << std::abs(A_pred) << "\n"
              << "  measured amplitude A_meas = " << A_meas << "\n"
              << "  relative error            = " << rel_err_amp << std::endl;
  }
}
} // namespace diffusion
