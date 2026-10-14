//========================================================================================
// AthenaPK - radiation task assembly (discrete ordinates) and radiation diagnostics.
//
// The explicit/implicit M1, thermal M1 and FLD transports that used to live in
// radiation_moments.cpp were removed on 2026-09-24 (git history, tag/commit before
// cleanup/remove-modules); the collapse run uses discrete ordinates only (radiation_sn.cpp).
// What remains here: the per-group extinction pass behind the rslaMax history column, the
// radiation history reducers, and AddRadiationTasks.
//========================================================================================

#include <cmath>
#include <string>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../dust/dust.hpp" // DustModel / DustOpacityFactors (opacity consumer)
#include "../eos/eos_table.hpp"
#include "../main.hpp" // IDN, IM1..IM3, IEN, NHYDRO, IB1..IB3
#include "radiation.hpp"
#include "radiation_groups.hpp"
#include "radiation_opacity.hpp"
#include "radiation_sn.hpp"

namespace Radiation {

using namespace parthenon::package::prelude;

//----------------------------------------------------------------------------------------
//! Fill the per-group extinction coefficient chi_g = rho*(kappa_R,g + kappa_s)*kdust
//! [1/length, code units] for the RSLA validity monitor (rslaMax history column), over the
//! whole block including ghosts (rslaMax reads rho gradients, chi at the cell). Skipped with
//! radiation/sn_skip_extinction or rsla_diag = false.
TaskStatus ComputeRadExtinction(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->pmy_mesh->packages.Get("radiation");
  // Its only consumer is the rslaMax history column (the Sn sweep computes its own sn.chia/chis).
  if (!pkg->Param<bool>("rsla_diag") || pkg->Param<bool>("sn_skip_extinction"))
    return TaskStatus::complete;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const OpacityParams op = pkg->Param<OpacityParams>("opacity");
  const Real gm1 = pkg->Param<Real>("gamma") - 1.0;
  const Real tfloor = pkg->Param<Real>("tfloor");
  const bool use_h2 = pkg->Param<bool>("use_h2diss");
  const auto eos_tab = pkg->Param<EOSTable::EosTable>("eos_tab");
  const Real T_unit = pkg->Param<Real>("T_unit");
  const GroupOpacityTable optab = pkg->Param<GroupOpacityTable>("optable");
  const int n_group = pkg->Param<RadGroups>("groups").n_group;

  std::vector<std::string> names{"cons"};
  for (int g = 0; g < n_group; ++g) names.push_back(GroupChiName(g));
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ic = imap["cons"].first;
  const int nhydro = pmb->pmy_mesh->packages.Get("Hydro")->Param<int>("nhydro");
  const bool mhd = (nhydro > NHYDRO);
  Kokkos::Array<int, MAX_GROUP> ichi;
  for (int g = 0; g < n_group; ++g) ichi[g] = imap[GroupChiName(g)].first;

  auto &pkgs = pmb->pmy_mesh->packages;
  const bool dust_on =
      pkgs.AllPackages().count("dust") > 0 && pkgs.Get("dust")->Param<bool>("evolve");
  int dust_sidx = 0;
  Real dust_fref = 0.01, dust_aref = 1.0e-5;
  bool dust_bridged = false, dust_gassplit = false; // AUDIT F-06 / F-07 (default = historical)
  if (dust_on) {
    auto dpkg = pkgs.Get("dust");
    dust_sidx = dpkg->Param<int>("scalar_index");
    const Dust::DustModel dm = dpkg->Param<Dust::DustModel>("model");
    dust_fref = dm.f_dg_ref;
    dust_aref = dm.a_ref;
    dust_bridged = dm.opac_size_bridged;
    dust_gassplit = dm.opac_gas_split;
  }
  const int idust = ic + nhydro + dust_sidx;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Rad::Extinction", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real rho = pack(b, ic + IDN, k, j, i);
        // A ghost cell can legitimately hold rho <= 0 before the first exchange; chi = 0 there
        // makes the correction inert (eps = 1), i.e. it degrades to the OLD scheme rather than
        // to a NaN. Failing safe toward the previous behaviour is the right direction for a
        // quantity that only modulates dissipation.
        if (!(rho > 0.0)) {
          for (int g = 0; g < n_group; ++g) pack(b, ichi[g], k, j, i) = 0.0;
          return;
        }
        const Real irho = 1.0 / rho;
        const Real m1 = pack(b, ic + IM1, k, j, i);
        const Real m2 = pack(b, ic + IM2, k, j, i);
        const Real m3 = pack(b, ic + IM3, k, j, i);
        Real me = 0.0;
        if (mhd) {
          const Real b1 = pack(b, ic + IB1, k, j, i);
          const Real b2 = pack(b, ic + IB2, k, j, i);
          const Real b3 = pack(b, ic + IB3, k, j, i);
          me = 0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
        }
        const Real e0 =
            pack(b, ic + IEN, k, j, i) - 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) * irho - me;
        // Non-positive E-KE-ME -> T = tfloor as a SENTINEL (the opacity lookup then clamps to the table's
        // 3 K edge either way); keeps the clamp report able to tell it apart from genuinely cold gas.
        Real T = use_h2 ? ((e0 > 0.0) ? eos_tab.TemperatureK(rho, e0) / T_unit : tfloor)
                        : gm1 * e0 * irho;
        T = std::max(T, tfloor);
        // Same opacity combination the transport source term attenuates the flux with
        // ("Flux attenuation uses the ROSSELAND mean + scattering"), because the whole point
        // is to make the NUMERICAL diffusion consistent with the PHYSICAL one this very
        // coefficient sets.
        // AUDIT F-06/F-07: the dust factors are T-dependent once the corrections are on, so
        // they are evaluated HERE, below T, not above it. With the defaults kdustR is the
        // historical DustFactor and this block is bit-identical.
        Real kdustP = 1.0, kdustR = 1.0;
        if (dust_on) {
          DustOpacityFactors(op, rho, T, pack(b, idust, k, j, i) * irho,
                             pack(b, idust + 1, k, j, i) * irho, dust_fref, dust_aref,
                             dust_bridged, dust_gassplit, kdustP, kdustR);
        }
        const Real kR = RosselandOpacity(op, rho, T) * kdustR;
        const Real ks = ScatteringOpacity(op, rho, T) * kdustR;
        for (int g = 0; g < n_group; ++g) {
          pack(b, ichi[g], k, j, i) = rho * (kR * optab.RossMultRT(g, rho, T * T_unit) + ks);
        }
      });
  return TaskStatus::complete;
}


//! RSLA validity: chat >~ v*tau (Krumholz et al. 2007); >~ 1 anywhere means RSLA fails there.
//! tau_loc = chi * rho/|grad rho| (local, no communication), a LOWER BOUND on the true optical
//! depth, so small values are weaker evidence than large ones. Meaningless at creduc = 1.
//! ----
//! Radiation history reducers: volume-weighted sums/maxima of rad::Er and Fr (pure reads). Columns
//! are APPENDED, never inserted (inserting shifts later column indices).
Real RadErTotal(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  static const std::vector<std::string> nm{"rad.Er"};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(nm, imap);
  const int iEr = imap["rad.Er"].first;
  Real sum = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Rad::ErTotal", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        lsum += pack(b, iEr, k, j, i) * pack.GetCoords(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(sum));
  return sum;
}

//! Er summed over the DENSE gas only (rho > radiation/diag_rho_split, code units).
//!
//! The whole-box rad-Er sum is blind to the core (it occupies ~6e-11 of the volume); this splits
//! on density like WP-8's mag-dissO. Returns 0 (column not registered) when the split is 0.
Real RadErDense(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const Real split =
      pmb->packages.Get("Radiation")->Param<Real>("diag_rho_split");
  static const std::vector<std::string> nm{"rad.Er"};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(nm, imap);
  const int iEr = imap["rad.Er"].first;
  // rho comes from the hydro primitive pack, not from rad state.
  static const std::vector<std::string> pnm{"prim"};
  auto ppack = md->PackVariables(pnm);
  Real sum = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Rad::ErDense", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        if (ppack(b, IDN, k, j, i) <= split) return;
        lsum += pack(b, iEr, k, j, i) * pack.GetCoords(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(sum));
  return sum;
}

//! Companion volume, so a MEAN Er in the dense gas can be formed in analysis
//! (rad-Er-dense / rad-Vdense) rather than only a sum that grows with the refined volume.
Real RadVolDense(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const Real split =
      pmb->packages.Get("Radiation")->Param<Real>("diag_rho_split");
  static const std::vector<std::string> pnm{"prim"};
  auto ppack = md->PackVariables(pnm);
  Real sum = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Rad::VolDense", parthenon::DevExecSpace(), 0,
      ppack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        if (ppack(b, IDN, k, j, i) <= split) return;
        lsum += ppack.GetCoords(b).CellVolume(k, j, i);
      },
      Kokkos::Sum<Real>(sum));
  return sum;
}

Real RadFrMax(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  static const std::vector<std::string> nm{"rad.Fr1", "rad.Fr2", "rad.Fr3"};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(nm, imap);
  const int i1 = imap["rad.Fr1"].first;
  const int i2 = imap["rad.Fr2"].first;
  const int i3 = imap["rad.Fr3"].first;
  Real mx = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Rad::FrMax", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        const Real f1 = pack(b, i1, k, j, i);
        const Real f2 = pack(b, i2, k, j, i);
        const Real f3 = pack(b, i3, k, j, i);
        const Real f = std::sqrt(f1 * f1 + f2 * f2 + f3 * f3);
        if (f > lmax) lmax = f;
      },
      Kokkos::Max<Real>(mx));
  return mx;
}

Real RslaMaxReport(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->pmy_mesh->packages.Get("radiation");
  if (!pkg->Param<bool>("rsla_diag")) return 0.0;
  // rad.chi_g is not filled with radiation/sn_skip_extinction (audit P-08).
  if (pkg->Param<bool>("sn_skip_extinction")) return 0.0;
  const Real chat = pkg->Param<Real>("chat");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int ndim = pmb->pmy_mesh->ndim;

  // Group 0 carries essentially all the flux in this problem's 3-group split; the most opaque
  // group would report the worst case but also the least representative one.
  std::vector<std::string> names{"cons", GroupChiName(0)};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ic = imap["cons"].first;
  const int iChi = imap[GroupChiName(0)].first;

  Real worst = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Rad::RslaMax", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        const Real rho = pack(b, ic + IDN, k, j, i);
        if (!(rho > 0.0)) return;
        const auto &co = pack.GetCoords(b);
        const Real irho = 1.0 / rho;
        const Real vx = pack(b, ic + IM1, k, j, i) * irho;
        const Real vy = pack(b, ic + IM2, k, j, i) * irho;
        const Real vz = pack(b, ic + IM3, k, j, i) * irho;
        const Real vmag = std::sqrt(vx * vx + vy * vy + vz * vz);
        const Real gx = (pack(b, ic + IDN, k, j, i + 1) - pack(b, ic + IDN, k, j, i - 1)) /
                        (2.0 * co.Dxc<1>(k, j, i));
        const Real gy =
            (ndim >= 2) ? (pack(b, ic + IDN, k, j + 1, i) - pack(b, ic + IDN, k, j - 1, i)) /
                              (2.0 * co.Dxc<2>(k, j, i))
                        : 0.0;
        const Real gz =
            (ndim >= 3) ? (pack(b, ic + IDN, k + 1, j, i) - pack(b, ic + IDN, k - 1, j, i)) /
                              (2.0 * co.Dxc<3>(k, j, i))
                        : 0.0;
        const Real gmag = std::sqrt(gx * gx + gy * gy + gz * gz);
        // Uniform density (gmag -> 0) has no local scale height. Falling back to the cell size
        // gives the SMALLEST defensible L, i.e. the most conservative reported violation -- it
        // can never manufacture an alarm out of a flat region.
        // 2026-08-26: the `gmag > 0.0` test caught only EXACT zero, so a NEARLY flat region
        // still gave unbounded L (gmag=1e-30 -> L ~ 1e37 cells) -- the very alarm the line
        // above says must never be manufactured. Cap L at the block extent: a gradient this
        // block cannot resolve must not be divided by. Capping can only LOWER the reported
        // rslaMax, so it can hide a marginal violation but never invent one.
        const Real dx_loc = co.Dxc<1>(k, j, i);
        const Real L_raw = (gmag > 0.0) ? rho / gmag : dx_loc;
        const Real L_cap = dx_loc * static_cast<Real>(ib.e - ib.s + 1);
        const Real L = (L_raw < L_cap) ? L_raw : L_cap;
        const Real r = vmag * (pack(b, iChi, k, j, i) * L) / (chat + RadFuzz());
        if (r > lmax) lmax = r;
      },
      Kokkos::Max<Real>(worst));
  return worst;
}

//----------------------------------------------------------------------------------------
//! Operator-split radiation for the final stage: the extinction pass for rslaMax, then the
//! discrete-ordinates solve (seed, BiCGSTAB or Richardson, writeback, matter coupling).
void AddRadiationTasks(TaskCollection &tc, Mesh *pmesh, const Real dt) {
  using namespace parthenon;
  TaskID none(0);
  auto pkg = pmesh->packages.Get("radiation");
  const auto &snp = pkg->Param<Sn::SnParams>("sn_params");
  auto partitions = pmesh->GetDefaultBlockPartitions();
  TaskRegion &region = tc.AddRegion(partitions.size());
  for (int i = 0; i < static_cast<int>(partitions.size()); ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    TaskID prev = tl.AddTask(none, ComputeRadExtinction, md.get());
    // Shallow containers carry only the exchanged fields: sn.u (+ the WDD face fields), and
    // for the Krylov path the preconditioned vector sn.k_y. A non-shallow container would
    // allocate fresh storage and exchange ghosts that nothing reads.
    std::vector<std::string> sn_ex{Sn::USolName()};
    if (snp.diamond) {
      sn_ex.push_back(Sn::FxName());
      sn_ex.push_back(Sn::FyName());
      sn_ex.push_back(Sn::FzName());
    }
    auto &md_sn = pmesh->mesh_data.AddShallow("sn_sub", md, sn_ex);
    // R-F3: the coupled Fleck & Cummings solve exchanges its own nfa-wide preconditioned vector instead.
    const bool fc = pmesh->packages.Get("radiation")->Param<bool>("sn_fc_scatter");
    auto &md_ky = snp.krylov ? pmesh->mesh_data.AddShallow(
                                   fc ? "sn_kryfc" : "sn_kry", md,
                                   std::vector<std::string>{fc ? Sn::KcyName() : Sn::KyName()})
                             : md_sn;
    Sn::AddSnTransport(tl, prev, pmesh, i, md, md_sn, md_ky, dt);
  }
}

} // namespace Radiation
