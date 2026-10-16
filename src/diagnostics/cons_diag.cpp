//========================================================================================
// AthenaPK - conservation-budget history diagnostics. See cons_diag.hpp.
//========================================================================================

#include <cmath>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
#include "cons_diag.hpp"
#include "../utils/row_tile.hpp"

namespace Diagnostics {

Real ConsDiagReduce(MeshData<Real> *md, ConsDiag which) {
  const auto &prim = md->PackVariables(std::vector<std::string>{"prim"});

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto *pm = pmb->pmy_mesh;
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const bool need_phi = (which == ConsDiag::Wgrav);
  // Dummy-alias to `prim` when unused: PackVariables on an unregistered field throws, and
  // the device lambda must still capture something. 
  const auto &phi =
      need_phi ? md->PackVariables(std::vector<std::string>{"grav.phi"}) : prim;

  const Real xlo = pm->mesh_size.xmin(X1DIR), xhi = pm->mesh_size.xmax(X1DIR);
  const Real ylo = pm->mesh_size.xmin(X2DIR), yhi = pm->mesh_size.xmax(X2DIR);
  const Real zlo = pm->mesh_size.xmin(X3DIR), zhi = pm->mesh_size.xmax(X3DIR);

  const Real dfloor = hydro_pkg->Param<Real>("cons_diag_dfloor");
  const Real pfloor = hydro_pkg->Param<Real>("cons_diag_pfloor");
  const int idual = hydro_pkg->Param<int>("dual_idx");
  // hydro/dual_energy_unsigned_prim: the switch flag is the `dual_switched` field, not the
  // sign of prim (the flag pack aliases prim otherwise and is not read).
  const bool dual_flag_field = hydro_pkg->AllParams().hasKey("dual_energy_unsigned_prim") &&
                               hydro_pkg->Param<bool>("dual_energy_unsigned_prim");
  const auto &dual_flag =
      (dual_flag_field && which == ConsDiag::ndual)
          ? md->PackVariables(std::vector<std::string>{"dual_switched"})
          : prim;
  const bool mhd = hydro_pkg->Param<bool>("cons_diag_mhd");
  // nTfloor: the floor pressure ConsToPrim applies for hydro/Tfloor, P_fl(rho) = P(rho, e(rho, Tfloor)) (table) or
  // (gamma-1) rho e_floor (ideal gas). The EOS is copied only for that column (a dummy otherwise, never evaluated).
  const bool want_tf = (which == ConsDiag::nTfloor);
  const AdiabaticGLMMHDEOS eosd = want_tf ? hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos")
                                          : AdiabaticGLMMHDEOS(-1.0, -1.0, -1.0, 1.0, 1.0, 1.4);

  const bool is_surf = (which == ConsDiag::Mout || which == ConsDiag::PoutX ||
                        which == ConsDiag::PoutY || which == ConsDiag::PoutZ);

  // The solver's own face fluxes. Packed BY NAME, not by the {Independent} metadata flag:
  // grav::phi is also Independent + WithFluxes, so a flag-based pack would pull phi in and
  // make the flat IDN index point at phi's flux instead of the gas mass flux whenever phi
  // sorts ahead of cons. This is the same trap SelfGravity::ApplyGravitySource documents.
  auto cons_flx_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"},
                                                  std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // validation_tests/dual_energy DE-F1 (2026-10-06): in 1D/2D the X2/X3 face-flux arrays are not allocated, but
  // the y/z "boundary face" tests below are always true for a single cell along y/z, so the solver-flux columns
  // (cons-Mout-solver*) dereferenced a null flux pointer at the first history output (segfault in every 1D run
  // with cons_diag = true). Read a solver flux only along an existing dimension. 3D (production): unchanged.
  const int ndim_cd = pmb->pmy_mesh->ndim;
  const bool has2 = ndim_cd > 1, has3 = ndim_cd > 2;
  Real sum = 0.0;
  Kokkos::parallel_reduce(
      "ConsDiagReduce",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, RowTile(ib.e + 1 - ib.s)}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &w = prim(b);
        const auto &coords = prim.GetCoords(b);
        const Real dV = coords.CellVolume(k, j, i);
        const Real rho = w(IDN, k, j, i);

        if (which == ConsDiag::Wgrav) {
          lsum += 0.5 * rho * phi(b, 0, k, j, i) * dV;
          return;
        }
        if (which == ConsDiag::nfloor || which == ConsDiag::Mfloor) {
          // A cell the floor is currently holding up sits EXACTLY at dfloor, so an
          // equality test with a relative tolerance identifies it. See the header caveat:
          // this is a proxy for floor ACTIVITY, not the mass the floor injected.
          if (dfloor > 0.0 && rho <= dfloor * (1.0 + 1.0e-9)) {
            lsum += (which == ConsDiag::nfloor) ? 1.0 : rho * dV;
          }
          return;
        }
        if (which == ConsDiag::nTfloor) {
          const Real efl = eosd.GetInternalEFloor();
#ifdef ATHENAPK_EOS_LEGACY_LOOKUPS
          const Real pfl = eosd.UseH2Diss() ? eosd.GetEosTable().PresFromRhoEint(rho, rho * efl)
                                            : (eosd.GetGamma() - 1.0) * rho * efl;
#else
          // the same floor pressure ConsToPrim applies: table energy at THIS density (EOS_VALIDATION.md O-5)
          const Real pfl = eosd.UseH2Diss() ? eosd.GetEosTable().PresFromRhoEint(rho, eosd.EintFloor(rho))
                                            : (eosd.GetGamma() - 1.0) * rho * efl;
#endif
          if (efl > 0.0 && w(IPR, k, j, i) <= pfl * (1.0 + 1.0e-9)) lsum += 1.0;
          return;
        }
        if (which == ConsDiag::npfloor) {
          if (pfloor > 0.0 && w(IPR, k, j, i) <= pfloor * (1.0 + 1.0e-9)) lsum += 1.0;
          return;
        }
        if (which == ConsDiag::ndual) {
          // Cells that took the advected value in the last ConsToPrim. Legacy: ConsToPrim marks
          // them with a negative prim sign (AdiabaticGLMMHDEOS::ConsToPrim).
          if (dual_flag_field) {
            if (dual_flag(b, 0, k, j, i) > 0.5) lsum += 1.0;
          } else if (idual >= 0 && w(idual, k, j, i) < 0.0) {
            lsum += 1.0;
          }
          return;
        }

        const Real dx = coords.Dxc<1>(k, j, i);
        const Real dy = coords.Dxc<2>(k, j, i);
        const Real dz = coords.Dxc<3>(k, j, i);

        if (which == ConsDiag::MoutSolverInner || which == ConsDiag::MoutSolverOuter ||
            which == ConsDiag::MoutInner || which == ConsDiag::MoutOuter) {
          const bool inner = (which == ConsDiag::MoutSolverInner ||
                              which == ConsDiag::MoutInner);
          const bool solver = (which == ConsDiag::MoutSolverInner ||
                               which == ConsDiag::MoutSolverOuter);
          auto &cf = cons_flx_pack(b);
          const Real vx = w(IV1, k, j, i), vy = w(IV2, k, j, i), vz = w(IV3, k, j, i);
          Real f = 0.0;
          if (inner) {
            if (std::fabs(coords.Xf<1>(i) - xlo) < 1.0e-9 * dx)
              f += -(solver ? cf.flux(X1DIR, IDN, k, j, i) : rho * vx) * (dV / dx);
            if ((has2 || !solver) && std::fabs(coords.Xf<2>(j) - ylo) < 1.0e-9 * dy)
              f += -(solver ? cf.flux(X2DIR, IDN, k, j, i) : rho * vy) * (dV / dy);
            if ((has3 || !solver) && std::fabs(coords.Xf<3>(k) - zlo) < 1.0e-9 * dz)
              f += -(solver ? cf.flux(X3DIR, IDN, k, j, i) : rho * vz) * (dV / dz);
          } else {
            if (std::fabs(coords.Xf<1>(i + 1) - xhi) < 1.0e-9 * dx)
              f += (solver ? cf.flux(X1DIR, IDN, k, j, i + 1) : rho * vx) * (dV / dx);
            if ((has2 || !solver) && std::fabs(coords.Xf<2>(j + 1) - yhi) < 1.0e-9 * dy)
              f += (solver ? cf.flux(X2DIR, IDN, k, j + 1, i) : rho * vy) * (dV / dy);
            if ((has3 || !solver) && std::fabs(coords.Xf<3>(k + 1) - zhi) < 1.0e-9 * dz)
              f += (solver ? cf.flux(X3DIR, IDN, k + 1, j, i) : rho * vz) * (dV / dz);
          }
          lsum += f;
          return;
        }

        if (which == ConsDiag::MoutSolver) {
          // Same faces as cons-Mout, but reading the SOLVER's flux array instead of
          // rebuilding the flux from the cell-centred primitive. flux(DIR,IDN,k,j,i) is the
          // LOWER face of cell i; the upper face is index i+1. Flux is a density => * dA.
          auto &cf = cons_flx_pack(b);
          Real f = 0.0;
          if (std::fabs(coords.Xf<1>(i) - xlo) < 1.0e-9 * dx)
            f += -cf.flux(X1DIR, IDN, k, j, i) * (dV / dx);
          if (std::fabs(coords.Xf<1>(i + 1) - xhi) < 1.0e-9 * dx)
            f += cf.flux(X1DIR, IDN, k, j, i + 1) * (dV / dx);
          if (has2 && std::fabs(coords.Xf<2>(j) - ylo) < 1.0e-9 * dy)
            f += -cf.flux(X2DIR, IDN, k, j, i) * (dV / dy);
          if (has2 && std::fabs(coords.Xf<2>(j + 1) - yhi) < 1.0e-9 * dy)
            f += cf.flux(X2DIR, IDN, k, j + 1, i) * (dV / dy);
          if (has3 && std::fabs(coords.Xf<3>(k) - zlo) < 1.0e-9 * dz)
            f += -cf.flux(X3DIR, IDN, k, j, i) * (dV / dz);
          if (has3 && std::fabs(coords.Xf<3>(k + 1) - zhi) < 1.0e-9 * dz)
            f += cf.flux(X3DIR, IDN, k + 1, j, i) * (dV / dz);
          lsum += f;
          return;
        }

        if (!is_surf) return;

        // --- flux through the PHYSICAL domain boundary, outward-positive ----------------
        const Real vx = w(IV1, k, j, i), vy = w(IV2, k, j, i), vz = w(IV3, k, j, i);
        const Real pres = w(IPR, k, j, i);
        Real bx = 0.0, by = 0.0, bz = 0.0;
        if (mhd) {
          bx = w(IB1, k, j, i);
          by = w(IB2, k, j, i);
          bz = w(IB3, k, j, i);
        }
        // Isotropic part of the stress: gas + magnetic pressure (Heaviside-Lorentz).
        const Real ptot = pres + 0.5 * (bx * bx + by * by + bz * bz);

        // Contribution of one face with outward unit normal nhat and area dA.
        // Mass:     rho (v.nhat)
        // Momentum: rho v_i (v.nhat) + ptot nhat_i - B_i (B.nhat)
        auto face = [&](const Real nx, const Real ny, const Real nz, const Real dA) {
          const Real vn = vx * nx + vy * ny + vz * nz;
          if (which == ConsDiag::Mout) return rho * vn * dA;
          const Real bn = bx * nx + by * ny + bz * nz;
          Real vi, ni, bi;
          if (which == ConsDiag::PoutX) {
            vi = vx; ni = nx; bi = bx;
          } else if (which == ConsDiag::PoutY) {
            vi = vy; ni = ny; bi = by;
          } else {
            vi = vz; ni = nz; bi = bz;
          }
          return (rho * vi * vn + ptot * ni - bi * bn) * dA;
        };

        Real f = 0.0;
        if (std::fabs(coords.Xf<1>(i) - xlo) < 1.0e-9 * dx) f += face(-1, 0, 0, dV / dx);
        if (std::fabs(coords.Xf<1>(i + 1) - xhi) < 1.0e-9 * dx) f += face(1, 0, 0, dV / dx);
        if (std::fabs(coords.Xf<2>(j) - ylo) < 1.0e-9 * dy) f += face(0, -1, 0, dV / dy);
        if (std::fabs(coords.Xf<2>(j + 1) - yhi) < 1.0e-9 * dy) f += face(0, 1, 0, dV / dy);
        if (std::fabs(coords.Xf<3>(k) - zlo) < 1.0e-9 * dz) f += face(0, 0, -1, dV / dz);
        if (std::fabs(coords.Xf<3>(k + 1) - zhi) < 1.0e-9 * dz) f += face(0, 0, 1, dV / dz);
        lsum += f;
      },
      sum);

  return sum;
}

//! See the header for why this exists. Deliberately delegates to the EXACT same reduction as
//! the cons-Mout-solver column rather than duplicating the face logic: the only difference
//! between the two is WHEN it is called (driver, right after the flux divergence was applied,
//! versus history, after operator-split sources have cleared the flux array). Keeping one
//! implementation means the two can never drift apart in faces, indexing or sign, so if
//! cons-Mout-applied and cons-Mout-solver ever disagree that difference is purely the timing
//! -- which is the whole measurement.
Real AppliedMassFluxSurface(MeshData<Real> *md) {
  return ConsDiagReduce(md, ConsDiag::MoutSolver);
}

} // namespace Diagnostics
