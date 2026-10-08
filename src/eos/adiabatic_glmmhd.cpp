//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements functions in class EquationOfState for adiabatic
//  hydrodynamics`

// C headers

// C++ headers
#include <cmath> // sqrt()

// Parthenon headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
#include "config.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
using parthenon::IndexDomain;
using parthenon::MeshBlockVarPack;
using parthenon::ParArray4D;

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.
void AdiabaticGLMMHDEOS::ConservedToPrimitive(MeshData<Real> *md) const {
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  auto this_on_device = (*this);

  // hydro/dual_energy_unsigned_prim: the dual-energy switch flag lives in its own field
  // (`dual_switched`, 1 = took the advected internal energy) instead of the sign of prim.
  // Otherwise the pack is `prim` again and is never written (the lambda needs a capture).
  const bool write_flag = dual_unsigned_ && dual_idx_ >= 0;
  auto flag_pack = write_flag ? md->PackVariables(std::vector<std::string>{"dual_switched"})
                              : prim_pack;
  // hydro/dual_energy_eta2: per-cell stage compression written by DualEnergyPdV (aliases prim
  // otherwise and is not read).
  const bool use_comp = dual_eta2_ > 0.0 && dual_idx_ >= 0;
  auto comp_pack = use_comp ? md->PackVariables(std::vector<std::string>{"dual_comp"}) : prim_pack;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitive", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        // auto &nu = entropy_pack(b);

        const Real comp = use_comp ? comp_pack(b, 0, k, j, i) : 0.0;
        const bool sw = this_on_device.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, comp);
        if (write_flag) flag_pack(b, 0, k, j, i) = sw ? 1.0 : 0.0;
      });
}
