//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusion.cpp
//! \brief

// C++ headers
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp" // table T for the ionization model under eos=hydrogen
#include "../../main.hpp"
#include "diffusion.hpp"
#include "../../utils/row_tile.hpp"

using namespace parthenon::package::prelude;

//! Assemble the per-cell fused non-ideal evaluator (see FusedNonidealEval in
//! diffusion.hpp) from the package's active diffusivity objects, and report whether the
//! coefficient mix is fusable at all: every ACTIVE term on the "ionization" coefficient
//! (NOT "fixed"). The ionization model is shared by all active terms (identical copies,
//! including the AD closure).
bool MakeFusedNonidealEval(parthenon::StateDescriptor *pkg, FusedNonidealEval &fe) {
  const bool have_ohm = pkg->Param<Resistivity>("resistivity") != Resistivity::none;
  const bool have_ad = pkg->Param<Ambipolar>("ambipolar") != Ambipolar::none;
  fe = FusedNonidealEval(); // hall_on stays false: Hall is not part of this tree
  fe.ohm_on = have_ohm;
  fe.ad_on = have_ad;
  bool fusable = have_ohm || have_ad;
  if (have_ohm) {
    const auto &d = pkg->Param<OhmicDiffusivity>("ohm_diff");
    const auto ct = d.GetCoeffType();
    if (ct != ResistivityCoeff::ionization) {
      fusable = false;
    } else {
      fe.ion = d.GetIonModel();
      fe.ohm_cap = d.EtaCap();
    }
  }
  if (have_ad) {
    const auto &d = pkg->Param<AmbipolarDiffusivity>("ad_diff");
    const auto ct = d.GetCoeffType();
    if (ct != AmbipolarCoeff::ionization) {
      fusable = false;
    } else {
      fe.ion = d.GetIonModel();
      fe.ad_cap = d.EtaCap();
    }
  }
  // WS-4 dust -> conductivity coupling (audit item 5). Params only exist when the
  // diffusion block ran, which is guaranteed here (we already read resistivity/ambipolar).
  if (pkg->AllParams().hasKey("nonideal_dust_on") &&
      pkg->Param<bool>("nonideal_dust_on")) {
    fe.dust_on = true;
    fe.i_fdg = pkg->Param<int>("nonideal_dust_i_fdg");
    fe.i_ac = pkg->Param<int>("nonideal_dust_i_ac");
    fe.dust_a_ref = pkg->Param<Real>("nonideal_dust_a_ref");
    fe.dust_fdg_ref = pkg->Param<Real>("nonideal_dust_fdg_ref");
    fe.dust_destroyed_zero = pkg->Param<bool>("nonideal_dust_destroyed_zero");
  }
  return fusable;
}

//----------------------------------------------------------------------------------------
//! Fused non-ideal diffusive timestep (ionization coeff family). See
//! diffusion.hpp. Evaluates the minimum number of Wardle-tensor solves per cell
//! (FusedNonidealEval) and reduces the diffusive dt over the active Ohmic / ambipolar /
//! terms, replacing one full tensor solve per term with (usually) one total.
Real EstimateNonidealTimestepIonizationFused(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const auto ndim = prim_pack.GetNdim();

  const bool have_ohm = hydro_pkg->Param<Resistivity>("resistivity") != Resistivity::none;
  const bool have_ad = hydro_pkg->Param<Ambipolar>("ambipolar") != Ambipolar::none;

  // Per-term CFL factors, matching the individual estimators exactly:
  //   Ohmic/ambipolar (parabolic): 0.5 (1D), 0.25 (2D), 1/6 (3D)
  Real fac_par = 0.5;
  if (ndim == 2) {
    fac_par = 0.25;
  } else if (ndim == 3) {
    fac_par = 1.0 / 6.0;
  }
  const Real cfl_diff = hydro_pkg->Param<Real>("cfl_diff");
  const bool dt_sum = hydro_pkg->AllParams().hasKey("dt_sum_ohm_ad") &&
                      hydro_pkg->Param<bool>("dt_sum_ohm_ad"); // audit P-12
  // Fused per-cell evaluator (shared ionization model + coefficient-family flags); the
  // caller's "nonideal_dt_fused" gate guarantees the mix is fusable.
  FusedNonidealEval feval;
  MakeFusedNonidealEval(hydro_pkg.get(), feval);
  // Under eos=hydrogen the ionization model's temperature must come from the EOS (mu varies
  // with dissociation/ionization, so T_code != p/rho); ideal path uses p/rho (bit-identical).
  const auto eos_nid = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto eos_tab_nid = eos_nid.GetEosTable();
  const bool useh2_nid = eos_nid.UseH2Diss();

  Real min_dt = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      "EstimateNonidealTimestepIonizationFused",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, RowTile(ib.e + 1 - ib.s)}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &ldt) {
        const auto &coords = prim_pack.GetCoords(b);
        const auto &prim = prim_pack(b);
        const Real bmag = std::sqrt(SQR(prim(IB1, k, j, i)) + SQR(prim(IB2, k, j, i)) +
                                    SQR(prim(IB3, k, j, i)));
        const Real rho = prim(IDN, k, j, i);
        const Real temp = useh2_nid ? eos_tab_nid.TemperatureKFromPres(
                                          rho, prim(IPR, k, j, i)) /
                                          feval.ion.T_unit
                                    : prim(IPR, k, j, i) / rho; // code T (p = rho*T)

        // Minimum tensor solves -> all three diffusivities (matches per-term Get()).
        Real eta_O, eta_H, eta_A;
        feval.Eta(rho, temp, bmag, eta_O, eta_H, eta_A,
                  feval.dust_on ? prim(feval.i_fdg, k, j, i) : -1.0,
                  feval.dust_on ? prim(feval.i_ac, k, j, i) : -1.0);

        // eta is a scalar per cell, so min_d(dx_d^2/eta) = (min_d dx_d^2)/eta exactly.
        Real mindx2 = SQR(coords.Dxc<1>(k, j, i));
        if (ndim >= 2) {
          mindx2 = fmin(mindx2, SQR(coords.Dxc<2>(k, j, i)));
        }
        if (ndim >= 3) {
          mindx2 = fmin(mindx2, SQR(coords.Dxc<3>(k, j, i)));
        }
        const Real eta_O_tot = (have_ohm ? eta_O : 0.0);
        if (dt_sum) {
          // P-12: Ohmic and ambipolar act together on the perpendicular current.
          const Real eta_par = eta_O_tot + (have_ad ? eta_A : 0.0);
          if (eta_par > 0.0)
            ldt = fmin(ldt, cfl_diff * fac_par * mindx2 / (eta_par + TINY_NUMBER));
        } else {
          if (eta_O_tot > 0.0) {
            ldt = fmin(ldt, cfl_diff * fac_par * mindx2 / (eta_O_tot + TINY_NUMBER));
          }
          if (have_ad) {
            ldt = fmin(ldt, cfl_diff * fac_par * mindx2 / (eta_A + TINY_NUMBER));
          }
        }
      },
      Kokkos::Min<Real>(min_dt));

  return min_dt;
}


//----------------------------------------------------------------------------------------
//! Fill the cell-centered non-ideal diffusivity cache (see diffusion.hpp). Bounds are the
//! interior expanded by one layer in each existing dimension -- exactly the cells whose
//! values the face-averaging in the flux kernels reads. Ghost prim is valid there (the
//! per-face path already read the same cells).
TaskStatus PrecomputeNonidealEta(StateDescriptor *hydro_pkg, MeshData<Real> *md) {
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto eta_pack = md->PackVariables(std::vector<std::string>{"nonideal_eta"});

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const auto ndim = prim_pack.GetNdim();
  const int il = ib.s - 1, iu = ib.e + 1;
  const int jl = (ndim > 1) ? jb.s - 1 : jb.s, ju = (ndim > 1) ? jb.e + 1 : jb.e;
  const int kl = (ndim > 2) ? kb.s - 1 : kb.s, ku = (ndim > 2) ? kb.e + 1 : kb.e;

  const bool have_ohm = hydro_pkg->Param<Resistivity>("resistivity") != Resistivity::none;
  const bool have_ad = hydro_pkg->Param<Ambipolar>("ambipolar") != Ambipolar::none;

  // Inactive terms get dummy-constructed diffusivities (never called; the value-capture
  // into the device lambda needs *some* object).
  const auto ohm_diff =
      have_ohm ? hydro_pkg->Param<OhmicDiffusivity>("ohm_diff")
               : OhmicDiffusivity(Resistivity::none, ResistivityCoeff::none, 0.0, 0.0,
                                  0.0, 0.0);
  const auto ad_diff =
      have_ad ? hydro_pkg->Param<AmbipolarDiffusivity>("ad_diff")
              : AmbipolarDiffusivity(Ambipolar::none, AmbipolarCoeff::none, 0.0, 0.0, 0.0,
                                     0.0);

  // Ionization-family fast path: FusedNonidealEval fills all three components from ONE
  // equilibrium tensor solve per cell (bit-identical to the per-term Get calls, which each
  // compute the full tensor anyway).
  FusedNonidealEval feval;
  const bool fused = MakeFusedNonidealEval(hydro_pkg, feval);

  const auto eos_nid = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto eos_tab_nid = eos_nid.GetEosTable();
  const bool useh2_nid = eos_nid.UseH2Diss();
  const auto T_unit_nid = feval.ion.T_unit;

  // AUDIT ITEM 1: cap-activation diagnostic, filled from the SAME tensor solve
  // that fills the eta cache (no second solve). Only the fused/ionization family is
  // supported -- that is the production configuration, and hydro.cpp hard-requires it.
  const bool cap_diag = hydro_pkg->Param<bool>("nonideal_cap_diag");
  auto capd_pack = cap_diag ? md->PackVariables(std::vector<std::string>{"diff.capdiag"})
                            : eta_pack;

  // DIAGNOSTIC (nonideal_mhd A8, env APK_ETADUMP=<gid>, fused path, FIRST call only, host-execution builds): print
  // every cell of block <gid> -- the inputs the evaluator saw (rho, code T from the EOS, |B|, f_dg, a_c) and the raw and
  // clamped eta_O/H/A -- plus, once, the model parameters, so an independent evaluation can be compared cell by cell.
  static int etadump_gid = [] { const char *e = std::getenv("APK_ETADUMP"); return e ? std::atoi(e) : -1; }();
  int etadump_b = -1;
  if (etadump_gid >= 0 && fused) {
    for (int bb = 0; bb < md->NumBlocks(); bb++)
      if (md->GetBlockData(bb)->GetBlockPointer()->gid == etadump_gid) etadump_b = bb;
    if (etadump_b >= 0) {
      const auto &m = feval.ion;
      printf("[ETAPAR] rho_unit %.17e T_unit %.17e B_unit %.17e eta_unit %.17e zeta %.17e mu_n %.17e m_ion %.17e "
             "alpha0 %.17e alpha_Texp %.17e sigma_en %.17e sigv_in %.17e grain_drag %.17e f_dg %.17e rho_grain %.17e "
             "T_subl %.17e x_K %.17e chi_K %.17e x_H %.17e xe_floor %.17e closure %d gamma_AD %.17e cr_model %d "
             "cr_zeta_rn %.17e cr_N0 %.17e cr_index %.17e cr_Nc %.17e cr_mu_col %.17e cr_sigma0 %.17e cr_column_fac %.17e "
             "cr_column_ext %.17e cr_cal %d cr_cal_A %.17e cr_cal_B %.17e cr_cal_x0 %.17e cr_cal_w %.17e cr_gamma %.17e "
             "dust_on %d a_ref %.17e fdg_ref %.17e ohm_cap %.17e ad_cap %.17e\n",
             m.rho_unit, m.T_unit, m.B_unit, m.eta_unit, m.zeta, m.mu_n, m.m_ion, m.alpha0, m.alpha_Texp, m.sigma_en,
             m.sigv_in, m.grain_drag, m.f_dg, m.rho_grain, m.T_subl, m.x_K, m.chi_K, m.x_H, m.xe_floor,
             static_cast<int>(m.ad_closure), m.gamma_AD, static_cast<int>(m.cr_model), m.cr_zeta_rn, m.cr_N0,
             m.cr_index, m.cr_Nc, m.cr_mu_col, m.cr_sigma0, m.cr_column_fac, m.cr_column_ext,
             static_cast<int>(m.cr_column_calibrated), m.cr_cal_A, m.cr_cal_B, m.cr_cal_x0, m.cr_cal_w, m.cr_gamma,
             static_cast<int>(feval.dust_on), feval.dust_a_ref, feval.dust_fdg_ref, feval.ohm_cap, feval.ad_cap);
      for (int kb = 0; kb < Ionization::N_BIN; kb++) printf("[ETABIN] %d a_k %.17e mw_k %.17e\n", kb, m.a_k[kb], m.mw_k[kb]);
      etadump_gid = -2; // once
    }
  }
  const int etadump_bk = etadump_b;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrecomputeNonidealEta", DevExecSpace(), 0,
      prim_pack.GetDim(5) - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &prim = prim_pack(b);
        const Real rho = prim(IDN, k, j, i);
        const Real temp = useh2_nid ? eos_tab_nid.TemperatureKFromPres(
                                          rho, prim(IPR, k, j, i)) /
                                          T_unit_nid
                                    : prim(IPR, k, j, i) / rho; // code T (p = rho*T)
        const Real bmag = std::sqrt(SQR(prim(IB1, k, j, i)) + SQR(prim(IB2, k, j, i)) +
                                    SQR(prim(IB3, k, j, i)));
        if (fused) {
          Real eO, eH, eA;
          feval.EtaRaw(rho, temp, bmag, eO, eH, eA,
                       feval.dust_on ? prim(feval.i_fdg, k, j, i) : -1.0,
                       feval.dust_on ? prim(feval.i_ac, k, j, i) : -1.0);
          if (cap_diag) {
            // raw (pre-ceiling) magnitudes; |.| because eta_H is signed
            const Real rO = eO, rH = std::abs(eH), rA = eA;
            Real cO = eO, cH = eH, cA = eA;
            feval.Clamp(cO, cH, cA);
            const Real aO = cO, aH = std::abs(cH), aA = cA;
            // A capped cell returns the ceiling EXACTLY, so ">" on the raw value is an
            // exact test for "this ceiling bit here".
            const Real fO = (rO > aO) ? 1.0 : 0.0;
            const Real fH = (rH > aH) ? 1.0 : 0.0;
            const Real fA = (rA > aA) ? 1.0 : 0.0;
            capd_pack(b, CapDiagIdx::flagO, k, j, i) = fO;
            capd_pack(b, CapDiagIdx::flagH, k, j, i) = fH;
            capd_pack(b, CapDiagIdx::flagA, k, j, i) = fA;
            capd_pack(b, CapDiagIdx::decO, k, j, i) =
                (fO > 0.0 && aO > 0.0) ? std::log10(rO / aO) : 0.0;
            capd_pack(b, CapDiagIdx::decH, k, j, i) =
                (fH > 0.0 && aH > 0.0) ? std::log10(rH / aH) : 0.0;
            capd_pack(b, CapDiagIdx::decA, k, j, i) =
                (fA > 0.0 && aA > 0.0) ? std::log10(rA / aA) : 0.0;
          }
          const Real rawO = eO, rawH = eH, rawA = eA;
          feval.Clamp(eO, eH, eA);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          // host-execution builds only: no device printf (stack/register cost) in GPU production builds
          if (b == etadump_bk)
            printf("[ETACELL] %d %d %d rho %.17e temp %.17e bmag %.17e fdg %.17e ac %.17e raw %.17e %.17e %.17e eta %.17e %.17e %.17e\n",
                   k, j, i, rho, temp, bmag, feval.dust_on ? prim(feval.i_fdg, k, j, i) : -1.0,
                   feval.dust_on ? prim(feval.i_ac, k, j, i) : -1.0, rawO, rawH, rawA, eO, eH, eA);
#else
          (void)rawO; (void)rawH; (void)rawA; (void)etadump_bk;
#endif
          eta_pack(b, NonidealEtaIdx::O, k, j, i) = eO;
          eta_pack(b, NonidealEtaIdx::H, k, j, i) = eH;
          eta_pack(b, NonidealEtaIdx::A, k, j, i) = eA;
        } else {
          Real eO = 0.0, eH = 0.0, eA = 0.0;
          if (have_ohm) {
            eO = ohm_diff.Get(bmag, rho, temp);
          }
          if (have_ad) {
            eA = ad_diff.Get(bmag, rho, temp);
          }
          eta_pack(b, NonidealEtaIdx::O, k, j, i) = eO;
          eta_pack(b, NonidealEtaIdx::H, k, j, i) = eH;
          eta_pack(b, NonidealEtaIdx::A, k, j, i) = eA;
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// AUDIT ITEM 1 -- cap-activation history reductions over "diff.capdiag".
// Shared driver: sum over interior cells of w * dV, where w is selected by `mode`.
namespace {
enum class CapRed { volume, flag, decade, massflag };

Real CapReduce(MeshData<Real> *md, CapRed mode, int comp) {
  const auto &capd = md->PackVariables(std::vector<std::string>{"diff.capdiag"});
  const auto &prim = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  Real sum = 0.0;
  Kokkos::parallel_reduce(
      "NonidealCapHst",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {capd.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, RowTile(ib.e + 1 - ib.s)}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &coords = capd.GetCoords(b);
        const Real dV = coords.CellVolume(k, j, i);
        Real w = 1.0;
        if (mode == CapRed::flag) {
          w = capd(b, comp, k, j, i);
        } else if (mode == CapRed::decade) {
          // flag * decades: `dec*` is already 0 where the ceiling is inactive.
          w = capd(b, comp + CapDiagIdx::decO, k, j, i);
        } else if (mode == CapRed::massflag) {
          w = capd(b, comp, k, j, i) * prim(b, IDN, k, j, i);
        }
        lsum += w * dV;
      },
      sum);
  return sum;
}
} // namespace

Real NonidealCapHstVol(MeshData<Real> *md) {
  return CapReduce(md, CapRed::volume, 0);
}
Real NonidealCapHstFlag(MeshData<Real> *md, int comp) {
  return CapReduce(md, CapRed::flag, comp);
}
Real NonidealCapHstDecade(MeshData<Real> *md, int comp) {
  return CapReduce(md, CapRed::decade, comp);
}
Real NonidealCapHstMassFlag(MeshData<Real> *md, int comp) {
  return CapReduce(md, CapRed::massflag, comp);
}

TaskStatus CalcDiffFluxes(StateDescriptor *hydro_pkg, MeshData<Real> *md,
                          DiffTermSet term_set, bool refill_eta) {
  // Term-set selection (see DiffTermSet in diffusion.hpp): "all" is the unsplit integrator,
  // "parabolic" the RKL2 stages. Only Ohmic and ambipolar diffusion are part of this tree, both
  // parabolic; "dispersive" (the Hall EMF of the full code) selects nothing here.
  const bool do_parabolic = (term_set != DiffTermSet::dispersive);
  const auto &resistivity = hydro_pkg->Param<Resistivity>("resistivity");
  const auto &ambipolar = hydro_pkg->Param<Ambipolar>("ambipolar");
  const bool do_ohm = do_parabolic && (resistivity != Resistivity::none);
  const bool do_ad = do_parabolic && (ambipolar != Ambipolar::none);
  // Fill the cell-centered non-ideal diffusivity cache once per call (per stage); the Ohmic and
  // ambipolar flux kernels then face-average the cached values instead of re-running the
  // ionization solve per face. Skipped when the caller freezes eta across RKL2 stages
  // (refill_eta=false; the cache still holds the values from the first stage of this Strang half).
  if (refill_eta && hydro_pkg->Param<bool>("nonideal_eta_cache") && (do_ohm || do_ad)) {
    PrecomputeNonidealEta(hydro_pkg, md);
  }
  if (do_ohm) {
    const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
    // OhmicDiffFluxIsoFixed evaluates the diffusivity per face (or from the cache), so it
    // handles the fixed, density_power and ionization coefficients alike.
    if (resistivity == Resistivity::ohmic &&
        (ohm_diff.GetCoeffType() == ResistivityCoeff::fixed ||
         ohm_diff.GetCoeffType() == ResistivityCoeff::ionization ||
         ohm_diff.GetCoeffType() == ResistivityCoeff::density_power)) {
      OhmicDiffFluxIsoFixed(md);
    } else {
      OhmicDiffFluxGeneral(md);
    }
  }
  if (do_ad) {
    AmbipolarDiffFluxIsoFixed(md);
  }
  return TaskStatus::complete;
}
