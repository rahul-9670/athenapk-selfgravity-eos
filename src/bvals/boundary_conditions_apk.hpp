//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file boundary_conditions_apk.chpp
//  \brief AthenaPK specific boundary conditions
//

#ifndef BVALS_BOUNDARY_CONDITIONS_APK_HPP_
#define BVALS_BOUNDARY_CONDITIONS_APK_HPP_

#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

#include "basic_types.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

#include "../main.hpp"
#include "../radiation/radiation.hpp"
#include "../radiation/radiation_sn.hpp"

namespace Hydro {
namespace BoundaryFunction {

using namespace parthenon::package::prelude;
using parthenon::CoordinateDirection;
// using parthenon::MeshBlockData;
// using parthenon::Real;
using parthenon::BoundaryFunction::BCSide;

// The stock hydro ReflectBC is not part of this MHD-only tree (it required fluid = euler); see
// ReflectValBC below for the closed MHD wall used by the validation suite.

template <CoordinateDirection DIR, BCSide SIDE>
void DiodeBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  MeshBlock *pmb = mbd->GetBlockPointer();

  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // FUSED PATH (hydro/fused_bc): Parthenon's GenericBC zero-gradient copy of every FillGhost field,
  // the diode clamp on `cons` and the Sn incoming-ordinate override in ONE kernel over ONE pack (the
  // values written are identical to the passes below: every ghost first takes the reference interior
  // cell, the normal-momentum component of `cons` is then zeroed if it points inward, and the
  // incoming Sn ordinates are then overwritten). The Sn Krylov vector is exchanged every BiCGSTAB
  // iteration, so this function runs very often. Only taken when every FillGhost field is
  // cell-centred and none is Metadata::Fine; otherwise the original path runs unchanged.
  if (pmb->packages.Get("Hydro")->Param<bool>("fused_bc")) {
    std::vector<std::string> names;
    bool fusable = true;
    for (const auto &v : mbd->GetVariableVector()) {
      if (!v->IsSet(Metadata::FillGhost)) continue;
      if (!v->IsSet(Metadata::Cell) || v->IsSet(Metadata::Fine)) {
        fusable = false;
        break;
      }
      names.push_back(v->label());
    }
    if (fusable) {
      if (names.empty()) return;
      parthenon::PackIndexMap imap;
      auto q = mbd->PackVariables(names, imap, coarse);
      const int nv = q.GetDim(4);
      auto rng = [&](const std::string &n, int &lo, int &cnt) {
        const auto &ip = imap[n]; // (-1,-2) when the container does not hold n
        if (ip.first < 0) {
          lo = -1;
          cnt = 0;
        } else {
          lo = ip.first;
          cnt = ip.second - ip.first + 1;
        }
      };
      int ic, nc, iu, nu, iy, ny, icy, ncy;
      rng("cons", ic, nc);
      rng(Radiation::Sn::USolName(), iu, nu);
      rng(Radiation::Sn::KyName(), iy, ny);
  rng(Radiation::Sn::KcyName(), icy, ncy);   // R-F14: the coupled-solve (sn_fc_scatter) Krylov vector, nfa-wide
      // Sn incoming override: the same parameters SnApplyIncomingBC reads.
      int mode = 0, nang = 1;
      Kokkos::View<Real *> ubg;
      Radiation::Sn::DeviceAngles ang;
      auto &all = pmb->packages.AllPackages();
      auto it = all.find("radiation");
      if (it != all.end() && it->second->AllParams().hasKey("sn_bc_mode")) {
        mode = it->second->Param<int>("sn_bc_mode");
        if (mode != 0) {
          nang = it->second->Param<Radiation::Sn::SnParams>("sn_params").nang;
          ubg = it->second->Param<Kokkos::View<Real *>>("sn_bc_ubg");
          ang = it->second->Param<Radiation::Sn::DeviceAngles>("sn_angles");
        }
      }
      if (mode == 0) nu = ny = ncy = 0;
      auto nxv = ang.nx, nyv = ang.ny, nzv = ang.nz;
      const int cmom = (ic >= 0) ? ic + DIR : -1; // IM1/IM2/IM3 == DIR within cons
      const auto nvr = IndexRange{0, nv - 1};
      pmb->par_for_bndry(
          "FusedBC", nvr, domain, parthenon::TopologicalElement::CC, coarse, false,
          KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
            Real val = q(l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
            if (l == cmom) {
              const bool inward = INNER ? (val > 0.0) : (val < 0.0);
              if (inward) val = 0.0;
            }
            if (nu > 0 && l >= iu && l < iu + nu) {
              const int n = l - iu;
              const int a = n % nang;
              const Real nc_ = X1 ? nxv(a) : (X2 ? nyv(a) : nzv(a));
              if (INNER ? (nc_ > 0.0) : (nc_ < 0.0)) val = (mode == 2) ? ubg(n / nang) : 0.0;
            } else if (ny > 0 && l >= iy && l < iy + ny && (l - iy) < nang) {
              const int a = l - iy;
              const Real nc_ = X1 ? nxv(a) : (X2 ? nyv(a) : nzv(a));
              if (INNER ? (nc_ > 0.0) : (nc_ < 0.0)) val = 0.0;
            } else if (ncy > 0 && l >= icy && l < icy + ncy) {
              const int a = (l - icy) % nang;
              const Real nc_ = X1 ? nxv(a) : (X2 ? nyv(a) : nzv(a));
              if (INNER ? (nc_ > 0.0) : (nc_ < 0.0)) val = 0.0;
            }
            q(l, k, j, i) = val;
          });
      return;
    }
  }

  // FIRST the plain zero-gradient copy for EVERY FillGhost variable, exactly as Parthenon's stock
  // `outflow` does (packing only "cons" would leave the ghost zones of every other FillGhost field
  // unwritten, which a restart exposes); THEN the diode momentum clamp on `cons`.
  parthenon::BoundaryFunction::GenericBC<DIR, SIDE, parthenon::BoundaryFunction::BCType::Outflow,
                                         parthenon::variable_names::any>(mbd, coarse);

  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const bool fine = false;

  const auto nv = IndexRange{0, cons.GetDim(4) - 1};
  pmb->par_for_bndry(
      "DiodeBC", nv, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
        // Zero-gradient copy from the nearest interior cell (identical to outflow).
        const Real val = cons(v, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        // The normal momentum component is IM1/IM2/IM3 == DIR (see main.hpp: IDN=0,
        // IM1=1, IM2=2, IM3=3, so component index v == DIR selects the normal momentum).
        // "Inward" is +normal on an inner face and -normal on an outer face.
        const bool normal_mom = (v == DIR);
        const bool inward = INNER ? (val > 0.0) : (val < 0.0);
        cons(v, k, j, i) = (normal_mom && inward) ? 0.0 : val;
      });

  // Sn incoming-ordinate condition (radiation/sn_bc_incoming != outflow). The generic copy
  // above has already filled sn.u and sn.k_y zero-gradient; this overwrites the INCOMING
  // ordinates in the same ghost region. The pointer is null on the historical path, so a run
  // without it (or without Sn at all) executes exactly the code above and nothing else.
  {
    auto &all = pmb->packages.AllPackages();
    auto it = all.find("radiation");
    if (it != all.end() && it->second->AllParams().hasKey("sn_bc_fun")) {
      using SnBcFun = void (*)(std::shared_ptr<MeshBlockData<Real>> &, bool, int, bool);
      auto fn = it->second->Param<SnBcFun>("sn_bc_fun");
      if (fn != nullptr) fn(mbd, coarse, X1 ? 1 : (X2 ? 2 : 3), INNER);
    }
  }
}

//! VALIDATION-ONLY closed wall ("reflecting_val", validation/self_gravity L6c, 2026-10-04).
//! Mirror every FillGhost field across the face (Parthenon GenericBC Reflect), then rewrite `cons`
//! as the mirror image with the NORMAL momentum and the NORMAL magnetic field negated
//! (conducting wall; psi even). Mass flux through the face is exactly zero, so a pressure-confined
//! test keeps its ambient (outflow admits inflow, diode drains a light ambient). Registered under
//! its own name, so no existing deck changes; not used by production.
template <CoordinateDirection DIR, BCSide SIDE>
void ReflectValBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");
  MeshBlock *pmb = mbd->GetBlockPointer();
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);
  parthenon::BoundaryFunction::GenericBC<DIR, SIDE, parthenon::BoundaryFunction::BCType::Reflect,
                                         parthenon::variable_names::any>(mbd, coarse);
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;
  const int offset = 2 * ref + (INNER ? -1 : 1);
  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto nv = IndexRange{0, cons.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectValBC", nv, domain, parthenon::TopologicalElement::CC, coarse, false,
      KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
        // IM1..IM3 = 1..3 (v == DIR is the normal momentum); IB1..IB3 = 5..7 (v == IB1 + DIR - 1)
        const bool odd = (v == DIR) || (v == IB1 + DIR - 1);
        cons(v, k, j, i) = (odd ? -1.0 : 1.0) * cons(v, X3 ? offset - k : k,
                                                       X2 ? offset - j : j,
                                                       X1 ? offset - i : i);
      });
}

//! Mesh-level fused diode + Sn BC (bvals/fused_bc_mesh.cpp), registered as
//! parthenon::MeshDataBoundaryHook() when all six domain faces are `diode`.
bool FusedDiodeBCMesh(std::shared_ptr<MeshData<Real>> &md, bool coarse);
void RegisterFusedDiodeBCMesh(parthenon::ParameterInput *pin);

} // namespace BoundaryFunction
} // namespace Hydro

#endif // BVALS_BOUNDARY_CONDITIONS_APK_HPP_
