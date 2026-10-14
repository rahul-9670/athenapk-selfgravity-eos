//========================================================================================
// AthenaPK - discrete-ordinates (Sn) transport: implementation.
// See radiation_sn.hpp for scope and docs/SN_TRANSPORT_DESIGN.md for the design rationale.
//========================================================================================
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>
#include <limits>
#include <vector>

#include <parthenon/package.hpp>
#include <parthenon/parthenon.hpp>

#include "../dust/dust.hpp"      // DustFactor: dust multiplies every radiation opacity
#include "../eos/adiabatic_glmmhd.hpp" // DualSelectEint (hydro/dual_energy_shared_select)
#include "../eos/eos_table.hpp"   // tabulated protostellar EOS -> T(rho, e_int)
#include "../main.hpp"
#include "radiation.hpp"
#include "radiation_groups.hpp"
#include "radiation_opacity.hpp"  // PlanckOpacity: emission uses kappa_P, never kappa_R
#include "utils/reductions.hpp"
#include "radiation_sn.hpp"
#include "radiation_sn_angles.hpp"
#include "radiation_sn_restrict.hpp"
#include "radiation_sn_sweep.hpp"
#include "radiation_sn_units.hpp"
#include "radiation_sn_graydsa.hpp"   // R-F12: gray DSA of the coupled solve (radiation/sn_gray_dsa)
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>

using namespace parthenon::package::prelude;
using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::Real;
using parthenon::TaskID;
using parthenon::TaskList;

namespace Radiation {
namespace Sn {

// Restart pre-check, filled by main() from parthenon's RestartReader. See the declaration in
// radiation_sn.hpp for why this is a global and not a ParameterInput key. Defaults = fresh
// start, which is also the state the host tests (which never call main) see.
bool g_restart_is_restart = false;
bool g_restart_u_present = false;
int g_restart_u_ncomp = -1;
int g_restart_ncycle = -1;

//----------------------------------------------------------------------------------------
void Initialize(ParameterInput *pin, StateDescriptor *pkg) {
  const std::string bn = "radiation";

  SnParams p;
  p.nzeta = pin->GetOrAddInteger(bn, "sn_nzeta", 4);
  p.npsi = pin->GetOrAddInteger(bn, "sn_npsi", 8);
  PARTHENON_REQUIRE(p.nzeta >= 2, "radiation/sn_nzeta must be >= 2: the Gauss-Legendre rule "
                                  "is exact through the second angular moment only for "
                                  "nzeta >= 2, and K_ij is consumed by every diagnostic.");
  PARTHENON_REQUIRE(p.npsi >= 3, "radiation/sn_npsi must be >= 3: the uniform azimuthal rule "
                                 "is exact for trigonometric degree <= npsi-1, and K_ij "
                                 "needs degree 2.");
  // The half-cell azimuthal offset in BuildAngularGrid keeps ordinates off the x and y axes
  // ONLY when npsi is a multiple of 4. Measured over npsi = 3..20 at nzeta = 8: every npsi
  // not divisible by 4 puts min(|n_x|,|n_y|) at 1.7e-17 .. 3.4e-17, i.e. exactly on an axis.
  // Algebraically sin(phi) = 0 for odd npsi and cos(phi) = 0 for npsi = 2 (mod 4). An
  // axis-aligned ordinate gets a_d = 0 in that direction and never transports along it.
  PARTHENON_REQUIRE(p.npsi % 4 == 0,
                    "radiation/sn_npsi must be a multiple of 4, or ordinates land exactly on "
                    "the x/y axes despite the half-cell offset (measured: min|n_x| ~ 2e-17) "
                    "and stop transporting in that direction.");
  // Gauss-Legendre with an ODD number of nodes has one at mu = 0 exactly, so every one of
  // those npsi ordinates gets n_z = 0 and never transports in z. Measured: min|n_z| is
  // exactly 0.0 for nzeta = 3,5,7,9,11 and 0.15-0.58 for even nzeta.
  PARTHENON_REQUIRE(p.nzeta % 2 == 0,
                    "radiation/sn_nzeta must be EVEN: an odd Gauss-Legendre rule has a node "
                    "at mu = 0, giving n_z = 0 exactly for npsi ordinates, which then never "
                    "transport in z.");
  p.nang = p.nzeta * p.npsi;
  p.n_group = pkg->Param<RadGroups>("groups").n_group;
  p.nfa = p.nang * p.n_group;
  // THE iteration cap -- there is exactly one, and it is this. An earlier revision read
  // `sn_max_iterations` here (default 2000) and then never used it again: the value that
  // actually bound was a second key, `sn_unroll` (300). A user raising the documented key
  // would have changed nothing, which is a worse version of the failure this project's own
  // rule 4 exists to prevent ("a cap measures the cap"). Both keys are now this one, and it
  // is the loop bound in AddSnTransport.
  //
  // Default 300 against a measured settled count of ~117 (Athena++ gate, Finding 4) is 2.6x
  // clear. It is NOT set higher, because iterations past convergence are not free: see the
  // cost note in AddSnTransport.
  p.max_iterations = pin->GetOrAddInteger(bn, "sn_max_iterations", 300);
  p.rel_tol = pin->GetOrAddReal(bn, "sn_rel_tol", 1.0e-6);
  // Red-black is the DEFAULT and plain Jacobi is offered only as a diagnostic. Measured on the
  // Athena++ gate (FINDINGS_smoother.md Finding 2): Jacobi walls at 32x the light CFL, SRJ at
  // 8x, red-black shows no wall through 81x = the hydro CFL. Setting this false is a way to
  // reproduce the wall, not a tuning option.
  p.red_black = pin->GetOrAddBoolean(bn, "sn_red_black", true);
  // Emission source: chi_a*B(T_gas). "planck" is the only value (the "m1er" prototype
  // placeholder was removed 2026-09-24); any other spelling is a typo and fails.
  const std::string smode = pin->GetOrAddString(bn, "sn_source", "planck");
  PARTHENON_REQUIRE(smode == "planck", "radiation/sn_source must be 'planck'.");
  // Writeback of the converged moments into rad.Er / rad.Fr. Default ON: the conversion is
  // now single-sourced in radiation_sn_units.hpp and covered by tests/test_sn_units.cpp, so
  // the reason it was off ("a wrong factor would silently corrupt the gas") has been retired
  // by a test rather than by an opinion. Setting it false leaves rad.Er untouched, which is
  // the prototype's behaviour and makes the Sn field invisible to output.
  p.writeback = pin->GetOrAddBoolean(bn, "sn_writeback", true);
  // Matter coupling. DEFAULT OFF -- see SnMatterCoupling for why an exactly conservative
  // operator can still be the wrong thing to switch on by default.
  p.matter_coupling = pin->GetOrAddBoolean(bn, "sn_matter_coupling", false);
  // radiation/sn_dual_energy (2026-09-23, default false = bit-identical). Both kernels below used to
  // take e_int = E - KE - ME. In low-beta cells (KE+ME ~99.5 % of E) that difference is exactly the
  // ill-conditioned quantity hydro/dual_energy exists to replace: at every matter-coupling FLOOR
  // event on a 3-D collapse restart the entry e_int was ALREADY negative, down to -4.4e6 against
  // e_min ~ 4.6e2 (fix/sn-matter-coupling investigation), i.e. the "floor" was this artefact,
  // not an over-cooling exchange. The coupling also never updated the dual energy, so in a
  // switched cell the next ConsToPrim rebuilt E from the pre-coupling e_dual and the radiative
  // heating/cooling of that cell was silently discarded.
  p.dual_energy = pin->GetOrAddBoolean(bn, "sn_dual_energy", false);
  // R-F6 (validation/radiation_sn R4, 2026-10-08): charge the angle-integrated residual of the accepted
  // iterate to the gas, making the exchange exactly conservative at any sn_rel_tol (radiation_sn_conserve.cpp).
  // Default false = nothing registered, task graph unchanged. A separate package Param, NOT a SnParams member, so
  // the struct every kernel captures is unchanged when the key is off (E-F9).
  pkg->AddParam<>("sn_conserve_residual", pin->GetOrAddBoolean(bn, "sn_conserve_residual", false));
  // R-F6 (radiation_sn_criterion.cpp): convergence norm "rhs" (legacy: SnResidual's sum|rhs(x)|, contains the
  // x-dependent scattering and streaming terms) or "b" (sum|I^n/(chat dt) + eta|, the true right-hand side), and the
  // local criterion sn_local_tol (0 = off). Both bicgstab-only. Defaults = legacy, task graph unchanged.
  {
    const std::string nm = pin->GetOrAddString(bn, "sn_norm", "rhs");
    PARTHENON_REQUIRE(nm == "rhs" || nm == "b", "radiation/sn_norm must be 'rhs' or 'b'");
    pkg->AddParam<>("sn_norm_b", nm == "b");
    pkg->AddParam<>("sn_local_tol", pin->GetOrAddReal(bn, "sn_local_tol", 0.0));
  }
  // R-F3 (radiation_sn_fc.cpp): Fleck & Cummings cross-group effective scattering, all groups in ONE BiCGSTAB.
  // Default false = legacy same-group split, task graph and registered fields unchanged.
  pkg->AddParam<>("sn_fc_scatter", pin->GetOrAddBoolean(bn, "sn_fc_scatter", false));
  pkg->AddParam<>("sn_gd_skip_sweep_exchange", pin->GetOrAddBoolean(bn, "sn_gd_skip_sweep_exchange", false));
  pkg->AddParam<>("sn_krylov_fused", pin->GetOrAddBoolean(bn, "sn_krylov_fused", false));
  // Speed work (2026-10-09): zero only what the coupled solve reads (radiation_sn_fastops.cpp); default false.
  pkg->AddParam<>("sn_fc_lean_zero", pin->GetOrAddBoolean(bn, "sn_fc_lean_zero", false));
  // R-F12 (radiation_sn_graydsa.cpp): gray diffusion synthetic acceleration of the coupled solve. Default false =
  // nothing registered, task graph unchanged.
  GrayDsa::Initialize(pin, pkg, pkg->Param<RadGroups>("groups").n_group);
  // R2 (Su-Olson) TEST-ONLY volume radiation source, radiation_sn_testsrc.cpp. Default false = task graph unchanged.
  pkg->AddParam<>("sn_test_source", pin->GetOrAddBoolean(bn, "sn_test_source", false));
  if (pkg->Param<bool>("sn_test_source")) {
    // R2: slab |x1| < x0, isotropic. R3: optional box "x1lo x1hi x2lo x2hi x3lo x3hi" and direction "dx dy dz"
    // (-> the ordinate with the largest dot product), and a second source sn_test_source2_* (keys space-separated:
    // a comma would be split by sbatch --export).
    std::vector<SnTestSourceDef> defs;
    const AngularGrid ag = BuildAngularGrid(pin->GetOrAddInteger(bn, "sn_nzeta", 4), pin->GetOrAddInteger(bn, "sn_npsi", 8));
    for (const std::string sfx : {"", "2"}) {
      const std::string pre = "sn_test_source" + sfx;
      if (sfx == "2" && !pin->DoesParameterExist(bn, pre + "_q")) continue;
      SnTestSourceDef d;
      d.q = pin->GetReal(bn, pre + "_q");
      d.tend = pin->GetReal(bn, pre + "_tend");
      d.x0 = pin->GetOrAddReal(bn, pre + "_x0", 0.5);
      const std::string box = pin->GetOrAddString(bn, pre + "_box", "");
      if (!box.empty()) {
        std::istringstream is(box);
        for (int q = 0; q < 6; ++q) is >> d.box[q];
        PARTHENON_REQUIRE(!is.fail(), "radiation/" + pre + "_box needs 6 numbers");
        d.slab = false;
      }
      const std::string dir = pin->GetOrAddString(bn, pre + "_dir", "");
      if (!dir.empty()) {
        std::istringstream is(dir);
        Real v[3];
        is >> v[0] >> v[1] >> v[2];
        PARTHENON_REQUIRE(!is.fail(), "radiation/" + pre + "_dir needs 3 numbers");
        Real best = -2.0;
        for (int a = 0; a < ag.nang; ++a) {
          const Real dd = ag.nx[a] * v[0] + ag.ny[a] * v[1] + ag.nz[a] * v[2];
          if (dd > best) { best = dd; d.ordinate = a; }
        }
        if (parthenon::Globals::my_rank == 0)
          std::cout << "## Sn test source" << sfx << ": ordinate " << d.ordinate << " = (" << ag.nx[d.ordinate] << ", "
                    << ag.ny[d.ordinate] << ", " << ag.nz[d.ordinate] << "), w = " << ag.w[d.ordinate] << std::endl;
      }
      defs.push_back(d);
    }
    pkg->AddParam("sn_test_source_defs", defs);
    pkg->AddParam("sn_test_source_time", std::make_shared<std::vector<Real>>());
  }
  p.radiation_force = pin->GetOrAddBoolean(bn, "sn_radiation_force", false);
  // Diagnostic switch (default true): the Rosseland excess max(0, chi_R - chi_a) is carried as
  // conservative scattering; false drops it so its share of the iteration count can be measured.
  // Not a physics option (the medium would be thinner than its table).
  p.rosseland_excess = pin->GetOrAddBoolean(bn, "sn_rosseland_excess", true);
  // Smoother: `ordered` (downstream sweep, exact within a block; PRODUCTION) or `redblack` (default,
  // the gate reference). Outer solver: the stationary iteration's rate depends only on the transport
  // Courant number (C ~ 1.5e4 at full c -> rate 0.9998 even with the ordered sweep), so production
  // uses BiCGSTAB with the sweep as preconditioner (Krylov-accelerated source iteration).
  const std::string sv = pin->GetOrAddString(bn, "sn_solver", "richardson");
  PARTHENON_REQUIRE(sv == "richardson" || sv == "bicgstab",
                    "radiation/sn_solver must be 'richardson' (the stationary iteration) or "
                    "'bicgstab' (matrix-free Krylov, preconditioned by the sweep).");
  p.krylov = (sv == "bicgstab");
  p.precond_sweeps = pin->GetOrAddInteger(bn, "sn_precond_sweeps", 1);
  // Speed work (2026-10-08): hyperplane (wavefront) form of the preconditioner sweeps, radiation_sn_wavefront.cpp.
  p.wavefront = pin->GetOrAddInteger(bn, "sn_sweep_wavefront", 0);
  p.sweep_group_batch = pin->GetOrAddBoolean(bn, "sn_sweep_group_batch", false);
  p.wf5_team_size = pin->GetOrAddInteger(bn, "sn_wf5_team_size", 0);
  p.wf5_scan = pin->GetOrAddBoolean(bn, "sn_wf5_scan", false);
  PARTHENON_REQUIRE(p.wavefront >= 0 && p.wavefront <= 5,
                    "radiation/sn_sweep_wavefront must be 0 (serial), 1, 2, 3, 4 or 5 (coalesced SOU, speed work)");
  p.verify_operator = pin->GetOrAddBoolean(bn, "sn_verify_operator", false);
  p.krylov_dot_scale = pin->GetOrAddBoolean(bn, "sn_krylov_dot_scale", false);
  p.skip_group_rel = pin->GetOrAddReal(bn, "sn_skip_group_rel", 0.0);   // audit P-09
  p.fallback = pin->GetOrAddBoolean(bn, "sn_fallback", false);          // audit P-06
  p.fallback_rel = pin->GetOrAddReal(bn, "sn_fallback_rel", 1.0e-4);
  PARTHENON_REQUIRE(p.skip_group_rel >= 0.0 && p.skip_group_rel < 1.0,
                    "radiation/sn_skip_group_rel must lie in [0, 1) (0 = off).");
  p.iter_verbose = pin->GetOrAddInteger(bn, "sn_iter_verbose", 0);
  // DIAGNOSTIC, default 0 = OFF = bit-identical. See SnParams::tau_report for why tau_cell is
  // the one number this package could not previously report about itself.
  p.tau_report = pin->GetOrAddInteger(bn, "sn_tau_report", 0);
  PARTHENON_REQUIRE(p.tau_report >= 0, "radiation/sn_tau_report must be >= 0 (0 = off).");
  // Spatial differencing: `step` (default; positive, over-diffuses as 1 + 0.63 tau_cell, gate K4),
  // `wdd` (weighted diamond, w > 1/2; Richardson only), `diamond` (w = 1/2, does not converge; kept
  // only to reproduce that), `sou` (second-order upwind: PRODUCTION; 0.974 flat over tau_cell 0.2-3).
  const std::string sp = pin->GetOrAddString(bn, "sn_spatial", "step");
  PARTHENON_REQUIRE(sp == "step" || sp == "diamond" || sp == "wdd" || sp == "sou",
                    "radiation/sn_spatial must be 'step', 'diamond' (= wdd at weight 0.5), "
                    "'wdd' (weighted diamond, weight from radiation/sn_dd_weight) or 'sou' "
                    "(second-order upwind, blend from radiation/sn_sou_kappa; the only "
                    "diffusion-limit scheme valid under sn_solver = bicgstab).");
  p.diamond = (sp == "diamond" || sp == "wdd");
  // SOU: second-order upwind, cell-centred, strictly upwind 3-point stencil (i, i-1, i-2). See
  // SnSouDiagonal in radiation_sn_sweep.hpp for the scheme, and docs/FIX_STEP_BIAS_KRYLOV_
  // 2026-09-14.md for the measurements that selected it over carrying WDD faces as Krylov
  // unknowns (2.4-3.2x memory) or a truncated face recursion (1.09x memory, 3K exchanges per
  // apply). Only `sou` reads sn_sou_kappa -- `step` must not register a key it ignores.
  p.sou = (sp == "sou");
  p.sou_kappa = p.sou ? pin->GetOrAddReal(bn, "sn_sou_kappa", 1.0) : 0.0;
  PARTHENON_REQUIRE(p.sou_kappa >= 0.0 && p.sou_kappa <= 1.0,
                    "radiation/sn_sou_kappa must lie in [0, 1]: 1 is second-order upwind "
                    "(zero first-order numerical diffusion), 0 is step (bit-identical), and "
                    "kappa in between keeps (1 - kappa) of step's 0.63*tau_cell diffusion.");
  if (p.sou) {
    // Read the key when the deck carries it, and otherwise fall back to the value Parthenon
    // has already resolved. A RESTART re-parses its stored input, which normally has the key,
    // but GetInteger on a missing key throws -- and aborting a resumed production run over a
    // diagnostic lookup is a worse failure than the one being guarded against.
    const int nghost = pin->DoesParameterExist("parthenon/mesh", "nghost")
                           ? pin->GetInteger("parthenon/mesh", "nghost")
                           : parthenon::Globals::nghost;
    PARTHENON_REQUIRE(nghost >= 2, "radiation/sn_spatial = sou reads TWO upwind cells and "
                                   "needs parthenon/mesh/nghost >= 2.");
    // R-F10 (validation/radiation_sn R3): lagged per-face positivity limiter, radiation_sn_soulim.cpp.
    const std::string lim = pin->GetOrAddString(bn, "sn_sou_limiter", "none");
    PARTHENON_REQUIRE(lim == "none" || lim == "positive",
                      "radiation/sn_sou_limiter must be 'none' (unlimited SOU) or 'positive'.");
    p.sou_limit = (lim == "positive");
    PARTHENON_REQUIRE(!p.sou_limit || p.krylov,
                      "radiation/sn_sou_limiter = positive is implemented for sn_solver = bicgstab only.");
    // R-F12 (validation/radiation_sn R5): SOU-consistent preconditioner sweep, radiation_sn_precond.cpp.
    const std::string pst = pin->GetOrAddString(bn, "sn_precond_stencil", "step");
    PARTHENON_REQUIRE(pst == "step" || pst == "sou",
                      "radiation/sn_precond_stencil must be 'step' (default) or 'sou' (the operator's own stencil).");
    p.precond_sou = (pst == "sou");
  }
  // Ghost prolongation at refinement jumps under SOU. The stencil reads the two upwind cells;
  // with ProlongatePiecewiseConstant both fine ghosts of a coarse neighbour hold the SAME
  // coarse value, so x(i-1) = x(i-2) and the streaming coefficient in the first fine cell
  // becomes 1.5*a instead of a. ProlongateSharedLinear (UNLIMITED -- a limiter would make the
  // exchange, hence the Krylov operator, non-linear) keeps the ghosts second-order. Applied to
  // sn.u AND the Krylov vector k_y, which must share one prolongation or A differs between
  // its two arguments at every coarse-fine face. Default step keeps piecewise-constant.
  p.sou_linear_prolong = p.sou ? pin->GetOrAddBoolean(bn, "sn_sou_linear_prolong", true) : false;
  // `diamond` pins the weight at 1/2 by definition; only `wdd` reads it, and `step` must not
  // read it AT ALL -- a GetOrAdd on a path that ignores the value registers a key that does
  // nothing and then persists into every restart's resolved parameters, which is precisely the
  // "an unused input key is a SILENT DEFAULT" trap this tree has already been bitten by. The
  // default weight is deliberately 0.5, so `wdd` with no weight given IS diamond: selecting a
  // scheme family must not quietly select a different discretisation as well.
  p.dd_w = (sp == "wdd") ? pin->GetOrAddReal(bn, "sn_dd_weight", 0.5) : 0.5;
  PARTHENON_REQUIRE(p.dd_w >= 0.5 && p.dd_w <= 1.0,
                    "radiation/sn_dd_weight must lie in [0.5, 1]: 0.5 is diamond (zero "
                    "numerical diffusion, marginally-stable face mode), 1.0 is step "
                    "(unconditionally positive, numerical diffusion 0.63*tau_cell). Outside "
                    "that range the scheme is neither bounded nor consistent.");
  p.diamond_fixup = pin->GetOrAddBoolean(bn, "sn_diamond_fixup", true);
  PARTHENON_REQUIRE(p.precond_sweeps >= 0, "radiation/sn_precond_sweeps must be >= 0 "
                                           "(0 = unpreconditioned, for isolating the Krylov).");
  const std::string sm = pin->GetOrAddString(bn, "sn_smoother", "redblack");
  PARTHENON_REQUIRE(sm == "redblack" || sm == "ordered" || sm == "jacobi",
                    "radiation/sn_smoother must be 'redblack', 'ordered' or 'jacobi'.");
  p.ordered_sweep = (sm == "ordered");
  // Diamond is implemented in the ORDERED sweep and in SnResidual only. Combining it with the
  // red-black/Jacobi smoothers or with the Krylov operator would run a sweep and a residual
  // that describe DIFFERENT operators -- the solver would converge confidently to the wrong
  // answer, which is the failure mode this package has already paid for twice. Refuse instead.
  PARTHENON_REQUIRE(!p.diamond || sm == "ordered",
                    "radiation/sn_spatial = diamond/wdd requires sn_smoother = ordered "
                    "(the face values are only carried by the ordered downstream sweep).");
  // SOU is implemented in the ORDERED sweep, SnResidual and SnApplyA. The red-black/Jacobi
  // colour sweeps would run a different (step) operator than the residual grades -- refuse.
  PARTHENON_REQUIRE(!p.sou || sm == "ordered",
                    "radiation/sn_spatial = sou requires sn_smoother = ordered (the 3-point "
                    "upwind stencil is implemented in the ordered sweep, SnResidual and "
                    "SnApplyA; the red-black/Jacobi colour sweeps are step-only).");
  // WDD + BiCGSTAB is refused: eliminating the WDD face unknowns gives an upstream recursion
  // (weights 0.887^k at w = 0.53), so A is dense along every ordinate; carrying the faces as Krylov
  // unknowns costs 3.17x memory, and a block-local face closure makes the discretisation depend on
  // the block size. Use `sou` under bicgstab (docs/SN_TRANSPORT_DESIGN.md section 15.5).
  PARTHENON_REQUIRE(!p.diamond || !p.krylov,
                    "radiation/sn_spatial = diamond/wdd is REFUSED under sn_solver = bicgstab, "
                    "on evidence rather than for want of an implementation. SnApplyA is "
                    "step-differenced and cannot be otherwise: eliminating the WDD face "
                    "unknowns leaves an upstream recursion whose weights decay as 0.887^k at "
                    "w = 0.53, so the operator is dense along every ordinate and no halo can "
                    "supply it. Carrying the faces as Krylov unknowns fixes that exactly and "
                    "costs 3.17x memory (152.69 -> 483.68 MiB/block) on a configuration that "
                    "already ran out of GPU memory at 3 nodes. Recomputing them by a "
                    "block-local sweep makes the DISCRETISATION depend on the block size "
                    "(1-D K4 replica at tau_cell = 1: 1.062/1.088/1.127/1.210 for blocks of "
                    "128/32/16/8, against 1.051 exact). USE radiation/sn_spatial = sou "
                    "INSTEAD: second-order upwind is a LOCAL 3-point stencil (i, i-1, i-2), so "
                    "it runs under BOTH solvers with no extra field, no extra exchange and no "
                    "extra memory, and gate K4 measures it at 0.974/0.973/0.974/0.975 over "
                    "tau_cell = 0.2/0.5/1.0/3.0 against step's 1.116/1.320/1.641/2.611 and "
                    "wdd's 0.983/0.995/1.016/1.115 (jobs 2749804/2749805; "
                    "docs/FIX_STEP_BIAS_KRYLOV_2026-09-14.md). sn_spatial = wdd remains "
                    "available under sn_solver = richardson, where it is the more accurate "
                    "scheme near tau_cell = 1 and costs roughly twice the nodes. "
                    "radiation/sn_tau_report tells you which regime you are in.");

  if (sm == "jacobi") p.red_black = false;
  // alpha = 1 is backward Euler and the only value that is unconditionally stable. 0 recovers
  // the explicit coupling EXACTLY, which exists so the G11 gate can show the difference on the
  // same deck rather than arguing it.
  p.fleck_alpha = pin->GetOrAddReal(bn, "sn_fleck_alpha", 1.0);
  PARTHENON_REQUIRE(p.fleck_alpha >= 0.0 && p.fleck_alpha <= 1.0,
                    "radiation/sn_fleck_alpha must be in [0,1] (1 = backward Euler).");
  const std::string crit = pin->GetOrAddString(bn, "sn_criterion", "residual");
  PARTHENON_REQUIRE(crit == "residual" || crit == "increment",
                    "radiation/sn_criterion must be 'residual' (true ||Ax-b||/||b||) or "
                    "'increment' (the retired iterate-difference test).");
  p.residual_criterion = (crit == "residual");
  PARTHENON_REQUIRE(!(p.radiation_force && !p.matter_coupling),
                    "radiation/sn_radiation_force requires sn_matter_coupling: the force is "
                    "applied in the same kernel, and the work it does comes out of the same "
                    "gas energy budget.");
  if (p.matter_coupling && parthenon::Globals::my_rank == 0) {
    // SCATTERING (corrected 2026-09-22). An earlier version of this block warned that the sweep
    // had "NO chi_s*J source". That described the original prototype and is no longer true:
    //  * bicgstab: SnApplyA carries -chi_s*J(x) of its own input vector, i.e. isotropic
    //    scattering is solved IMPLICITLY inside the Krylov operator (b holds only rdt*I^n + eta);
    //  * source-iteration paths: the sweep's source is S = chi_a*B + chi_s*J (radiation_sn_sweep.hpp).
    // chi_s is the effective scattering from SnFleckSplit: rho*kdust*kappa_s from the opacity
    // model, plus the Rosseland excess max(0, chi_R - chi_a), plus the Fleck remainder
    // (1-f)*chi_a. Scattering is conservative, so the matter coupling charges the gas with chi_a
    // alone. Host tests test_sn_units (Er == B independent of chi_s; f-independent total
    // extinction) cover it. Stated once so a log reader knows what is and is not in the solve.
    const OpacityParams opw = pkg->Param<OpacityParams>("opacity");
    const bool sc = (opw.model == OpacityModel::constant) ? (opw.kappa_s0 > 0.0) : true;
    if (sc)
      std::cout << "## Sn scattering: isotropic, "
                << (p.krylov ? "IMPLICIT in the Krylov operator (chi_s*J(x) in A)"
                                           : "via the sweep source S = chi_a*B + chi_s*J")
                << "; chi_s = rho*kappa_s + Rosseland excess + Fleck remainder; energy exchange "
                   "with the gas on chi_a only."
                << std::endl;
  }
  pkg->AddParam<>("sn_params", p);
  // radiation/sn_skip_extinction (audit P-08, default false = legacy). With the Planck source the
  // Sn path fills its own coefficients (sn.chia/sn.chis) in SnSeedSource; rad.chi_g from
  // ComputeRadExtinction is then read ONLY by the rslaMax history column, which at creduc = 1 says
  // nothing. The pass ran every step over the whole block with E-KE-ME temperatures and fed the
  // EOS/opacity clamp tallies. With the key on it is skipped and rslaMax reports 0.
  pkg->AddParam<>("sn_skip_extinction",
                  pin->GetOrAddBoolean(bn, "sn_skip_extinction", false));

  // One-shot, PER-PARTITION flag for SnInitFromMoments (shared_ptr so a task can clear it). False
  // whenever the ordinates were read from the restart file: re-seeding from the moments would
  // replace a converged iterate by a clamped Eddington fit and create energy (FIX_SN_RESTART).
  // R-F1 (validation/radiation_sn R0, 2026-10-08): on a restart the cycle comes from the FILE (Info/NCycle).
  // parthenon/time/ncycle in a restart's stored Input is NOT the file's cycle: parthenon updates it only after package
  // initialisation, so it holds the start cycle of the leg that wrote the file -- 0 for every restart written by a run
  // that began at t = 0. Reading it here re-seeded sn.u by P1 on the first restart of every fresh-start chain although
  // sn.u had just been read (two-leg run not bit-identical; energy created by the P1 clamp). The pin key is still
  // touched (GetOrAdd) so the Input string written to outputs is unchanged.
  const int pin_ncycle = pin->GetOrAddInteger("parthenon/time", "ncycle", 0);
  const int start_ncycle =
      (g_restart_is_restart && g_restart_ncycle >= 0) ? g_restart_ncycle : pin_ncycle;

  // ------------------------------------------------------------------------------------
  // Restart policy for sn.u -- decided from the FILE (docs/FIX_SN_RESTART_2026-09-15.md):
  //   * sn.u always carries Metadata::Restart (always written);
  //   * file has a readable sn.u -> read and used as-is; file lacks it (M1 restart) -> P1 re-seed;
  //   * file has sn.u with a DIFFERENT component count (ordinate set changed) -> cannot be read
  //     (hyperslab mismatch): re-seed and do not write sn.u this leg (self-heals next leg).
  // Parthenon skips a Restart variable that is absent on disk. sn_restart_reproject keeps its old
  // meaning (force a re-seed) and is warned about.
  const bool restart_reproject = pin->GetOrAddBoolean(bn, "sn_restart_reproject", false);
  // g_restart_* are filled by main() from the RestartReader; on a fresh start they are the
  // fresh-start defaults and every term below collapses to the pre-existing behaviour.
  const bool file_has_u = g_restart_is_restart && g_restart_u_present;
  // ncomp <= 0 means "present, but the file did not state a component count" -- trust the
  // read in that case rather than silently discarding a field we could have used.
  const bool file_u_readable = file_has_u && (g_restart_u_ncomp <= 0 ||
                                              g_restart_u_ncomp == p.nfa);
  const bool read_u_from_file = file_u_readable && !restart_reproject;
  // Drop Metadata::Restart ONLY when the file holds an sn.u we must not read: keeping it
  // would make parthenon attempt the incompatible H5Dread.
  const bool register_restart = !(file_has_u && !read_u_from_file);
  // Re-seed at t = 0 (adopt the pgen's radiation IC) or whenever we did not read ordinates.
  const bool reseed = (start_ncycle == 0) || !read_u_from_file;
  if (parthenon::Globals::my_rank == 0 && g_restart_is_restart) {
    std::cout << "### Sn restart: sn.u " << (g_restart_u_present ? "PRESENT" : "ABSENT")
              << " on disk (ncomp=" << g_restart_u_ncomp << ", need " << p.nfa << ") -> "
              << (read_u_from_file ? "READ ordinates from file"
                                   : "RE-SEED ordinates from rad.Er/rad.Fr by P1")
              << (register_restart ? "" : ", sn.u NOT written to this leg's restarts")
              << std::endl;
    if (file_u_readable && restart_reproject)
      std::cout << "### WARNING Sn: radiation/sn_restart_reproject = true is DISCARDING the "
                   "sn.u on disk and refitting the ordinates to their own first two angular "
                   "moments. That is not conservative for an anisotropic field (see "
                   "docs/FIX_SN_RESTART_2026-09-15.md); the key is no longer needed to "
                   "restart from an M1 file, which is now automatic."
                << std::endl;
    if (file_has_u && !file_u_readable)
      std::cout << "### WARNING Sn: the restart file's sn.u has " << g_restart_u_ncomp
                << " components but this deck needs " << p.nfa
                << " -- the ordinate set changed across the restart. The ordinates are being "
                   "re-seeded from the moments (a real, non-conservative loss of angular "
                   "detail) and sn.u is not written to THIS leg's restarts; the next leg "
                   "sees no sn.u on disk and resumes writing it."
                << std::endl;
  }
  pkg->AddParam("sn_init_at_start", reseed);
  pkg->AddParam("sn_restart_reproject", restart_reproject);
  pkg->AddParam("sn_register_restart", register_restart);
  pkg->AddParam("sn_need_init", std::make_shared<std::vector<char>>());
  // Report counter for the tau_cell diagnostic, PER PARTITION and held in the package because
  // the task list -- and with it any state a task-list-scope make_shared owns -- is rebuilt
  // every cycle. A counter living in the per-solve state would be reborn as 0 every step, so
  // "every N solves" would silently mean "every solve".
  pkg->AddParam("sn_tau_count", std::make_shared<std::vector<int>>());

  // --- angular grid ---------------------------------------------------------------------
  const AngularGrid g = BuildAngularGrid(p.nzeta, p.npsi);
  const double amom = AngularMomentError(g);
  // Fail LOUDLY at init rather than producing a quietly wrong Eddington tensor. Every moment
  // the scheme consumes is one of these sums (tests/test_sn_angles.cpp).
  PARTHENON_REQUIRE(amom < 1.0e-12,
                    "Sn angular quadrature failed its moment-exactness check; "
                    "sum w = 4pi, sum w n = 0, sum w n_i n_j = (4pi/3) delta_ij.");

  DeviceAngles d;
  d.nang = p.nang;
  d.nx = parthenon::ParArray1D<Real>("sn_nx", p.nang);
  d.ny = parthenon::ParArray1D<Real>("sn_ny", p.nang);
  d.nz = parthenon::ParArray1D<Real>("sn_nz", p.nang);
  d.w = parthenon::ParArray1D<Real>("sn_w", p.nang);
  auto hnx = Kokkos::create_mirror_view(d.nx);
  auto hny = Kokkos::create_mirror_view(d.ny);
  auto hnz = Kokkos::create_mirror_view(d.nz);
  auto hw = Kokkos::create_mirror_view(d.w);
  for (int n = 0; n < p.nang; ++n) {
    hnx(n) = g.nx[n];
    hny(n) = g.ny[n];
    hnz(n) = g.nz[n];
    hw(n) = g.w[n];
  }
  Kokkos::deep_copy(d.nx, hnx);
  Kokkos::deep_copy(d.ny, hny);
  Kokkos::deep_copy(d.nz, hnz);
  Kokkos::deep_copy(d.w, hw);
  pkg->AddParam<>("sn_angles", d);

  // --- physical-boundary incoming-ordinate condition (see SnApplyIncomingBC) ------------------
  {
    const std::string bc = pin->GetOrAddString(bn, "sn_bc_incoming", "outflow");
    PARTHENON_REQUIRE_THROWS(bc == "outflow" || bc == "vacuum" || bc == "planck",
                             "radiation/sn_bc_incoming must be 'outflow' (historical zero-"
                             "gradient copy), 'vacuum' (zero incoming intensity) or 'planck' "
                             "(isotropic blackbody at sn_bc_tbg_K).");
    const int mode = (bc == "outflow") ? 0 : (bc == "vacuum") ? 1 : 2;
    const Real tbg_K = pin->GetOrAddReal(bn, "sn_bc_tbg_K", 10.0);
    pkg->AddParam<>("sn_bc_mode", mode);
    // Per-group isotropic intensity u_g = chat * a_rad T^4 f_g(T) / (4 pi), the same expression
    // SnSeedSource emits with (B_g = arad T^4 PlanckFraction), so a uniform T_bg medium is in
    // exact equilibrium with this boundary. Computed once on the host; read on the device.
    Kokkos::View<Real *> ubg("sn_bc_ubg", p.n_group);
    auto hubg = Kokkos::create_mirror_view(ubg);
    {
      const Real arad = pkg->Param<Real>("arad");
      const Real chat = pkg->Param<Real>("chat");
      const Real T_unit = pkg->Param<Real>("T_unit");
      const RadGroups groups = pkg->Param<RadGroups>("groups");
      const Real T = tbg_K / T_unit;
      for (int g = 0; g < p.n_group; ++g) {
        hubg(g) = (mode == 2) ? SnIntensityFromEnergy(arad * T * T * T * T *
                                                          groups.PlanckFraction(g, tbg_K),
                                                      chat)
                              : 0.0;
      }
    }
    Kokkos::deep_copy(ubg, hubg);
    pkg->AddParam<>("sn_bc_ubg", ubg);
    pkg->AddParam<SnBcFun>("sn_bc_fun", (mode == 0) ? nullptr : &SnApplyIncomingBC);
    // R-F15 FIXED (validation/radiation_sn, 2026-10-09). The incoming-ordinate override used to be applied ONLY by
    // AthenaPK's diode boundary wrapper (bvals/boundary_conditions_apk.hpp): a face with any other condition (Parthenon
    // outflow, reflecting_val, ...) left sn.u AND the Krylov vectors at the zero-gradient copy while this banner said
    // "every physical face" (the R0-R13 1-D decks ran without the Planck inflow; the gray DSA looked broken there).
    // Now every NON-diode physical face gets SnApplyIncomingBC as a radiation-package boundary function, which
    // Parthenon calls right after the face's own condition (bvals/boundary_conditions.cpp; the fused mesh path keeps
    // that order). Diode faces keep their in-wrapper call unchanged, so an all-diode deck (production) is bit-identical.
    if (mode != 0) {
      const char *keys[6] = {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"};
      using BF = parthenon::BoundaryFace;
      const BF faces[6] = {BF::inner_x1, BF::outer_x1, BF::inner_x2, BF::outer_x2, BF::inner_x3, BF::outer_x3};
      int nreg = 0;
      for (int q = 0; q < 6; ++q) {
        // read-only: Parthenon's own default for *_bc is "outflow" (GetOrAdd with another default would abort)
        const std::string v =
            pin->DoesParameterExist("parthenon/mesh", keys[q]) ? pin->GetString("parthenon/mesh", keys[q]) : "outflow";
        if (v == "diode" || v == "periodic") continue;
        const int dir = q / 2 + 1;
        const bool inner = (q % 2 == 0);
        pkg->UserBoundaryFunctions[faces[q]].push_back(
            [dir, inner](std::shared_ptr<parthenon::MeshBlockData<Real>> &rc, bool coarse) {
              SnApplyIncomingBC(rc, coarse, dir, inner);
            });
        ++nreg;
        if (parthenon::Globals::my_rank == 0)
          std::cout << "## Sn (R-F15): radiation/sn_bc_incoming = " << bc << " applied on parthenon/mesh/" << keys[q]
                    << " = " << v << " as a radiation-package boundary function" << std::endl;
      }
      (void)nreg;
    }
    if (parthenon::Globals::my_rank == 0 && mode != 0) {
      std::cout << "## Sn incoming-ordinate BC: " << bc;
      if (mode == 2) std::cout << " (T_bg = " << tbg_K << " K)";
      std::cout << " on every physical face; Krylov corrections get zero incoming." << std::endl;
    }
  }

  // --- fields ---------------------------------------------------------------------------
  const parthenon::MetadataFlag OperatorSplit =
      parthenon::Metadata::FlagNameExists("OperatorSplit")
          ? parthenon::Metadata::GetUserFlag("OperatorSplit")
          : parthenon::Metadata::AddUserFlag("OperatorSplit");

  // The iterate sn.u: component-major (b, n, k, j, i) so a fixed ordinate coalesces in i; no
  // WithFluxes (fluxes are fused into the update); Metadata::Restart but NOT Independent (Independent
  // would put nfa components into every integrator register). Prolongation: ProlongateSharedLinear
  // under sou (sn_sou_linear_prolong, production), piecewise-constant otherwise.
  {
    // Metadata::Restart ALWAYS, except when the restart file holds an sn.u with a different
    // component count -- see the restart-policy block above. Note the asymmetry that the old
    // code got wrong: this flag controls the WRITE as well as the read, so dropping it is a
    // decision about the NEXT leg, not only about this one.
    Metadata m(pkg->Param<bool>("sn_register_restart")
                   ? std::vector<parthenon::MetadataFlag>{Metadata::Cell, Metadata::Derived,
                                                          Metadata::FillGhost,
                                                          Metadata::Restart, OperatorSplit}
                   : std::vector<parthenon::MetadataFlag>{Metadata::Cell, Metadata::Derived,
                                                          Metadata::FillGhost, OperatorSplit},
               std::vector<int>({p.nfa}));
    // DEFAULT: parthenon's RestrictAverage. RestrictAverageFaceLayer was an ATTEMPTED FIX for
    // the coarse-fine energy loss and it DOES NOT WORK -- experiment C (job 2731004) measured
    // 7.335e-03 with it and 7.335e-03 without, identical to every digit. Its special case is
    // unreachable because parthenon restricts on the SENDER over interior cells; the full
    // reasoning is in radiation_sn_restrict.hpp. Kept behind a default-off switch so the dead
    // end is recorded rather than silently repeated.
    if (pin->GetOrAddBoolean(bn, "sn_face_layer_restrict", false)) {
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                              RestrictAverageFaceLayer>();
    } else if (p.sou_linear_prolong) {
      // SOU: second-order ghosts at refinement jumps (see the sn_sou_linear_prolong note).
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                              parthenon::refinement_ops::RestrictAverage>();
    } else {
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                              parthenon::refinement_ops::RestrictAverage>();
    }
    pkg->AddField(USolName(), m);
  }
  if (p.diamond) {
    // Outgoing face values per direction. FillGhost: the sweep reads the UPWIND NEIGHBOUR's
    // outgoing face, which across a block boundary lives in the halo.
    for (const auto &nm : {FxName(), FyName(), FzName()}) {
      Metadata m({Metadata::Cell, Metadata::Derived, Metadata::FillGhost, Metadata::Restart,
                  OperatorSplit},
                 std::vector<int>({p.nfa}));
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                              parthenon::refinement_ops::RestrictAverage>();
      pkg->AddField(nm, m);
    }
  }
  // State at the beginning of the hydro step: the RHS anchor I^n. Never exchanged (read only
  // at the owning cell) and never restricted.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, OperatorSplit},
               std::vector<int>({p.nfa}));
    pkg->AddField(UnName(), m);
  }
  // Moments (J, H1, H2, H3) per group. Exists so the source pass never has to hold the angle
  // vector in registers -- see SN_TRANSPORT_DESIGN.md section 4.2 and the measured
  // per-thread-state lesson at radiation_moments.cpp:912.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({4 * p.n_group}));
    pkg->AddField(MomName(), m);
  }
  // ABSORPTION coefficient chi_a,g = rho*kappa_P,g, one per group. Stored, not recomputed,
  // so the matter coupling charges the gas with the SAME coefficient the sweep emitted with.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({p.n_group}));
    pkg->AddField(ChiaName(), m);
  }
  // Effective SCATTERING coefficient, one per group. See ChisName() in the header.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({p.n_group}));
    pkg->AddField(ChisName(), m);
  }
  // Krylov work vectors, only with bicgstab, and nang-wide (one group at a time: A is exactly
  // block-diagonal in the group index); s is formed in place in k_r. Measured 246 -> 71 MiB/block.
  // R-F3: the coupled Fleck & Cummings solve uses its own nfa-wide vectors; the per-group ones (except k_j, which
  // SnApplyA uses on either path) are then not registered, which saves 6 x nang components per cell.
  const bool fc_on = pkg->Param<bool>("sn_fc_scatter");
  if (p.krylov) {
    if (!fc_on) for (const auto &nm : {KrName(), Kr0Name(), KvName(), KtName(), KpName()}) {
      Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit},
                 std::vector<int>({p.nang}));
      pkg->AddField(nm, m);
    }
    {
      // J(x) for the operator apply. ONE component: only the group being solved is live.
      Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit},
                 std::vector<int>({1}));
      pkg->AddField(KjName(), m);
    }
    // Only the preconditioned vector has the operator and the smoother applied to it.
    if (!fc_on) for (const auto &nm : {KyName()}) {
      Metadata m({Metadata::Cell, Metadata::Derived, Metadata::FillGhost, OperatorSplit},
                 std::vector<int>({p.nang}));
      // Same prolongation as sn.u, or A(sn.u) and A(k_y) are different operators at every
      // coarse-fine face (the residual is evaluated on the former, the recurrence on the latter).
      if (p.sou_linear_prolong) {
        m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                                parthenon::refinement_ops::RestrictAverage>();
      } else {
        m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                                parthenon::refinement_ops::RestrictAverage>();
      }
      pkg->AddField(nm, m);
    }
  }
  // R-F3: nfa-wide coupled-solve work vectors (radiation/sn_fc_scatter only).
  if (pkg->Param<bool>("sn_fc_scatter")) {
    PARTHENON_REQUIRE(p.krylov, "radiation/sn_fc_scatter needs sn_solver = bicgstab");
    PARTHENON_REQUIRE(!p.diamond, "radiation/sn_fc_scatter does not support the WDD scheme");
    for (const auto &nm : {KcrName(), Kcr0Name(), KcpName(), KcvName(), KctName()}) {
      Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, OperatorSplit},
                 std::vector<int>({p.nfa}));
      pkg->AddField(nm, m);
    }
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::FillGhost, OperatorSplit},
               std::vector<int>({p.nfa}));
    if (p.sou_linear_prolong) {
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                              parthenon::refinement_ops::RestrictAverage>();
    } else {
      m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                              parthenon::refinement_ops::RestrictAverage>();
    }
    pkg->AddField(KcyName(), m);
  }
  // R-F6: angle-integrated residual per group (radiation/sn_conserve_residual only).
  if (pkg->Param<bool>("sn_conserve_residual")) {
    PARTHENON_REQUIRE(p.krylov, "radiation/sn_conserve_residual needs sn_solver = bicgstab");
    PARTHENON_REQUIRE(!p.diamond, "radiation/sn_conserve_residual does not support the WDD scheme");
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>({p.n_group}));
    pkg->AddField(ResMomName(), m);
  }
  // Fleck factor, one scalar per cell.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({1}));
    pkg->AddField(FleckName(), m);
  }
  // Lagged thermal EMISSIVITY eta_g = chi_a,g * chat * B_g(T_gas) / (4*pi), one per group.
  // An emissivity (added straight to the RHS), NOT a source function (multiplied by chi in
  // the sweep): with scattering live the source is chi_a*B + chi_s*J, which is not any single
  // coefficient times any single function. See radiation_sn_units.hpp.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({p.n_group}));
    pkg->AddField(SrcName(), m);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Initialise the ordinates from the problem's rad.Er / rad.Fr at t = 0.
//!
//! Needed because SnWriteback overwrites rad.Er/rad.Fr every step: without it the pgen's radiation
//! IC would be discarded. Expansion I = (chat/4pi)(Er + 3 Omega.Fr/chat) reproduces both moments
//! exactly under this quadrature (tests/test_sn_units.cpp check 8). It is CLAMPED at zero, and the
//! clamp ADDS energy for |Fr| > chat*Er/3 (x1.31 at f = 1, unbounded as f grows), so it runs only at
//! t = 0 and on a restart whose file has no sn.u (decided in Sn::Initialize).
TaskStatus SnInitFromMoments(MeshData<Real> *md, const int part) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  auto need = pkg->Param<std::shared_ptr<std::vector<char>>>("sn_need_init");
  if (part < 0 || part >= static_cast<int>(need->size()) || !(*need)[part])
    return TaskStatus::complete;
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> names{USolName()};
  for (int g = 0; g < p.n_group; ++g)
    for (const auto &nm : GroupFieldNames(g)) names.push_back(nm);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  Kokkos::Array<int, MAX_GROUP * 4> ir;
  for (int g = 0; g < p.n_group; ++g) {
    const auto nm = GroupFieldNames(g);
    for (int a = 0; a < 4; ++a) ir[4 * g + a] = imap[nm[a]].first;
  }
  const int nang = p.nang;
  const int nfa = p.nfa;
  const Real chat = pkg->Param<Real>("chat");
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::InitFromMoments", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, 0, nfa - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        const int g = n / nang;
        const int a = n - g * nang;
        const Real Er = std::max(0.0, pack(b, ir[4 * g + 0], k, j, i));
        const Real fdotn = nxv(a) * pack(b, ir[4 * g + 1], k, j, i)
                         + nyv(a) * pack(b, ir[4 * g + 2], k, j, i)
                         + nzv(a) * pack(b, ir[4 * g + 3], k, j, i);
        pack(b, iu + n, k, j, i) =
            std::max(0.0, SnIntensityFromEnergy(Er, chat) + 3.0 * fdotn / kFourPi);
      });
  (*need)[part] = 0;   // this partition only; see the note at the flag's construction
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! radiation/sn_dual_energy, legacy selection: E - KE - ME where it exceeds eta1*E, else the
//! advected (dual) internal energy. One-sided; see SnDualSel for the ConsToPrim-consistent rule.
KOKKOS_INLINE_FUNCTION Real SnDualEint(const Real e_E, const Real E, const Real e_dual,
                                       const Real eta1) {
  return (e_E > eta1 * E) ? e_E : ((e_dual > 0.0) ? e_dual : e_E);
}

//! hydro/dual_energy_shared_select (audit P-01, 2026-09-23). The seed and the coupling pick the
//! gas internal energy BETWEEN the hydro update and the next ConsToPrim, and the coupling writes its
//! result back into the dual energy of every cell. With the eta1-only SnDualEint that write-back
//! made ConsToPrim's two-sided eta2 test inert on the corrector stage (it then always sees
//! e_int == u_ie). With the key on, the SAME rule ConsToPrim applies is used here
//! (AdiabaticGLMMHDEOS::DualSelectEint, with this stage's `dual_comp`). Captured by value.
struct SnDualSel {
  bool on = false, use_comp = false;
  AdiabaticGLMMHDEOS eos{0.0, 0.0, 0.0, 0.0, 0.0, 1.4};
  parthenon::MeshBlockVarPack<Real> comp;
  SnDualSel(parthenon::StateDescriptor *hpkg, MeshData<Real> *md, const bool sn_dual) {
    on = sn_dual && hpkg->AllParams().hasKey("dual_energy_shared_select") &&
         hpkg->Param<bool>("dual_energy_shared_select");
    if (!on) return;
    eos = hpkg->Param<AdiabaticGLMMHDEOS>("eos");
    use_comp = eos.DualEta2() > 0.0;
    if (use_comp) comp = md->PackVariables(std::vector<std::string>{"dual_comp"});
  }
  KOKKOS_INLINE_FUNCTION Real Eint(const Real e_E, const Real E, const Real u_ie, const Real rho, const Real e_B,
                                   const int b, const int k, const int j, const int i) const {
    return eos.DualSelectEint(e_E, E, u_ie, rho, use_comp ? comp(b, 0, k, j, i) : 0.0, e_B);
  }
};

//! \brief Seed I^n and the lagged emissivity at the start of a hydro step.
//!
//! Evaluates, through the same calls as radiation_thermal_m1.cpp's Load,
//!     eta_g = chi_a,g chat B_g(T_gas)/(4 pi),  chi_a,g = rho kdust kappa_P(rho,T) PlanckMult(g,T),
//!     B_g = a_R T^4 PlanckFraction(g,T),
//! with kappa_P (Kirchhoff), not the Rosseland extinction. The gas internal energy is E - KE - ME,
//! or with radiation/sn_dual_energy the dual-energy selection (eta1 only, or ConsToPrim's two-sided
//! rule with hydro/dual_energy_shared_select).
TaskStatus SnSeedSource(MeshData<Real> *md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");

  // INTERIOR, not entire. sn.un, sn.src and sn.mom are NOT registered FillGhost, so their
  // ghost cells are never exchanged and never read: the sweep reads sn.un and sn.src only at
  // the cell it is updating, and nothing reads sn.mom at all while chi_s = 0. Seeding the
  // ghosts was writing 36^3/32^3 = 1.42x more cells than the kernel can ever use. Only sn.u
  // is FillGhost, and its ghosts come from the exchange, not from here.
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> names{USolName(), UnName(),   SrcName(),
                                 ChiaName(), ChisName(), FleckName(), "cons"};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int iun = imap[UnName()].first;
  const int isrc = imap[SrcName()].first;
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int ifl = imap[FleckName()].first;
  const int ic = imap["cons"].first;
  const int nfa = p.nfa;
  const int n_group = p.n_group;
  // chat, not c: the whole scheme runs on the reduced speed, and rad.Er/rad.Fr are the
  // reduced-speed moments. See radiation_sn_units.hpp.
  const Real chat = pkg->Param<Real>("chat");
  const Real c_light = pkg->Param<Real>("c");
  const Real fleck_alpha = p.fleck_alpha;
  const bool do_couple = p.matter_coupling;
  const bool ross_excess = p.rosseland_excess;

  // --- the EOS / opacity chain, mirroring ThermalM1::Load call for call ------------------
  const OpacityParams op = pkg->Param<OpacityParams>("opacity");
  const GroupOpacityTable optab = pkg->Param<GroupOpacityTable>("optable");
  const RadGroups groups = pkg->Param<RadGroups>("groups");
  const auto eos_tab = pkg->Param<EOSTable::EosTable>("eos_tab");
  const Real T_unit = pkg->Param<Real>("T_unit");
  const Real gm1 = pkg->Param<Real>("gamma") - 1.0;
  const Real arad = pkg->Param<Real>("arad");
  // The COUPLING clamps, not tfloor: these are the range over which the opacity and EOS
  // tables are valid, and using a different clamp here than the coupling path uses would put
  // emission and absorption on two different temperatures in the same cell.
  const Real tmin = pkg->Param<Real>("coupling_tmin");
  const Real tmax = pkg->Param<Real>("coupling_tmax");
  const bool use_h2 = pkg->Param<bool>("use_h2diss");
  const int nhydro = pmb->pmy_mesh->packages.Get("Hydro")->Param<int>("nhydro");
  const bool mhd = (nhydro > NHYDRO);

  auto &pkgs = pmb->pmy_mesh->packages;
  const bool dust_on =
      pkgs.AllPackages().count("dust") > 0 && pkgs.Get("dust")->Param<bool>("evolve");
  int dust_sidx = 0;
  Real dust_fref = 0.01, dust_aref = 1.0e-5;
  bool dust_bridged = false, dust_gassplit = false;
  if (dust_on) {
    auto dpkg = pkgs.Get("dust");
    dust_sidx = dpkg->Param<int>("scalar_index");
    const Dust::DustModel dm = dpkg->Param<Dust::DustModel>("model");
    dust_fref = dm.f_dg_ref;
    dust_aref = dm.a_ref;
    dust_bridged = dm.opac_size_bridged;   // AUDIT F-06
    dust_gassplit = dm.opac_gas_split;     // AUDIT F-07
  }
  const int idust = ic + nhydro + dust_sidx;

  // The emissivity is n_group values per cell, so the angle index is irrelevant here and the
  // cell loop is the parallel one.
  //
  // radiation/sn_dual_energy, see Initialize.
  const auto hydro_pkg_sd = pmb->packages.Get("Hydro");
  const bool sn_dual = p.dual_energy && hydro_pkg_sd->Param<int>("dual_idx") >= 0;
  const int idual_c = sn_dual ? hydro_pkg_sd->Param<int>("dual_idx") : 0;
  const Real dual_eta1 = sn_dual ? hydro_pkg_sd->Param<Real>("dual_energy_eta1") : 0.0;
  // hydro/dual_energy_shared_select (audit P-01): ConsToPrim's own two-sided rule instead of the
  // eta1-only SnDualEint. Off = legacy, bit-identical.
  const SnDualSel dsel(hydro_pkg_sd.get(), md, sn_dual);
  // par_reduce, not par_for, to carry out the EXACT STABILITY NUMBER of the matter coupling.
  // Folding it into a kernel that already visits every cell makes it essentially free; a
  // separate pass would not be.
  Real gmax = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::SeedSource", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lg) {
        const Real rho = pack(b, ic + IDN, k, j, i);
        // A cell with rho <= 0 has no matter to emit. Zero, not a floor: an emissivity is a
        // rate, and inventing one where there is no mass would be a source of energy from
        // nothing. Ghost cells can legitimately look like this before the first exchange.
        if (!(rho > 0.0)) {
          for (int g = 0; g < n_group; ++g) {
            pack(b, isrc + g, k, j, i) = 0.0;
            pack(b, ichia + g, k, j, i) = 0.0;
            pack(b, ichis + g, k, j, i) = 0.0;
          }
          pack(b, ifl, k, j, i) = 1.0;
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
        const Real e0_E =
            pack(b, ic + IEN, k, j, i) - 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) * irho - me;
        const Real e0 =
            !sn_dual ? e0_E
            : dsel.on ? dsel.Eint(e0_E, pack(b, ic + IEN, k, j, i),
                                  pack(b, ic + idual_c, k, j, i), rho, me, b, k, j, i)
                      : SnDualEint(e0_E, pack(b, ic + IEN, k, j, i),
                                   pack(b, ic + idual_c, k, j, i), dual_eta1);
        Real T = use_h2 ? eos_tab.TemperatureK(rho, std::max(e0, Real(1.0e-100))) / T_unit
                        : gm1 * e0 * irho;
        T = std::max(tmin, std::min(tmax, T));
        // AUDIT F-06/F-07: two channels. With the defaults both equal the historical
        // DustFactor and this is bit-identical to the pre-fix binary.
        Real kdustP = 1.0, kdustR = 1.0;
        if (dust_on) {
          DustOpacityFactors(op, rho, T, pack(b, idust, k, j, i) * irho,
                             pack(b, idust + 1, k, j, i) * irho, dust_fref, dust_aref,
                             dust_bridged, dust_gassplit, kdustP, kdustR);
        }
        // probe = false: this is the gas's ACTUAL temperature, not a root-finder trial, so
        // its table edge-clamps belong in the real tally that the per-cycle clamp report
        // reads. ThermalM1 passes true only because it evaluates inside a Newton iteration.
        const Real kp = rho * kdustP * PlanckOpacity(op, rho, T, false);
        const Real kr = rho * kdustR * RosselandOpacity(op, rho, T);
        const Real ks = rho * kdustR * ScatteringOpacity(op, rho, T);
        const Real T4 = T * T * T * T;
        const Real Tk = T * T_unit;

        // --- FLECK FACTOR -----------------------------------------------------------------
        // f = 1/(1 + alpha*c*dt*chi_a,P*dB/de_gas): the Fleck & Cummings linearisation of the
        // emission about T^n. This is what turns a conditionally stable explicit exchange into
        // an unconditionally stable one, and it is why this scheme needs no per-cell Newton
        // iteration the way M1's coupling does.
        //
        // dB/de_gas = 4 a_R T^3 * dT/de. For the ideal gas dT/de = (gamma-1)/rho exactly; for
        // the tabulated EOS it comes from a symmetric difference on EintFromRhoTk, because the
        // table exposes e(T) and not its derivative. The step is RELATIVE (1e-3 T) so it stays
        // meaningful across the four decades of T the table spans.
        Real dTde;
        if (use_h2) {
          const Real dT = 1.0e-3 * T;
          const Real ep = eos_tab.EintFromRhoTk(rho, (T + dT) * T_unit, true);
          const Real em = eos_tab.EintFromRhoTk(rho, (T - dT) * T_unit, true);
          const Real de = ep - em;
          // A flat or inverted table segment would give f > 1 or f < 0, either of which is
          // worse than simply not linearising. dTde = 0 => f = 1 => the explicit scheme, which
          // is at least a scheme.
          dTde = (de > 0.0) ? (2.0 * dT / de) : 0.0;
        } else {
          dTde = gm1 * irho;
        }
        const Real dBde = 4.0 * arad * T * T * T * dTde;
        // chi_a,P is the PLANCK-WEIGHTED mean absorption, sum_g chi_a,g B_g / sum_g B_g. For
        // gray that is chi_a. An unweighted mean would give equal say to groups carrying almost
        // no emission -- exactly the situation the collapse run's 3-group split creates, where
        // group 2 was measured EMPTY in 100% of cells.
        Real cab = 0.0, bsum = 0.0;
        for (int g = 0; g < n_group; ++g) {
          const Real fr = groups.PlanckFraction(g, Tk);
          cab += kp * optab.PlanckMultRT(g, rho, Tk, false) * fr;
          bsum += fr;
        }
        if (bsum > 0.0) cab /= bsum;
        // GATED ON matter_coupling. With the gas temperature FIXED there is no emission
        // response to linearise, so f < 1 buys nothing and costs everything: it moves
        // (1-f)*chi_a into effective scattering, and source iteration then contracts by
        // 1 - f per sweep. Measured on gate G8 (chi_a = 100, c = 100 => f = 2.6e-4), that took
        // a solve that converges in ~18 iterations to one needing ~9e4, so it hit the
        // iteration cap and returned a badly wrong field. Diagnosed only because G8 re-ran.
        // S is the raw stiffness of the gas-radiation exchange over this step.
        const Real S = c_light * dt * cab * dBde;
        const Real fleck = do_couple ? 1.0 / (1.0 + fleck_alpha * S) : 1.0;
        pack(b, ifl, k, j, i) = fleck;
        // THE EXACT STABILITY NUMBER, G = f*S, and it is worth deriving rather than guessing.
        // The gas update is  e <- e + f*c*dt*chi_a*(Er - B(e)),  so a perturbation is amplified
        // by  1 - f*S,  and the scheme is stable iff  0 < f*S < 2.
        //   alpha = 1 : f = 1/(1+S)  =>  G = S/(1+S) < 1  ALWAYS. Unconditional.
        //   alpha = 0 : f = 1        =>  G = S,  stable only while S < 2.
        // This replaced a |dE_gas|/e_int > 0.1 heuristic that fired on BOTH arms of gate G11 --
        // 0.166 on the stable Fleck arm and 0.496 on the unstable explicit one -- and so could
        // not tell them apart. |dE|/e_int measures how far the gas moved, which is large
        // whenever the gas is far from equilibrium and says nothing about stability.
        if (do_couple) {
          const Real G = fleck * S;
          if (G > lg) lg = G;
        }

        for (int g = 0; g < n_group; ++g) {
          const Real chi_a = kp * optab.PlanckMultRT(g, rho, Tk, false);
          const Real B_g = arad * T4 * groups.PlanckFraction(g, Tk);
          // The split -- emission scaled by f, the absorption it removes handed back as
          // effective scattering -- lives in SnFleckSplit (radiation_sn_units.hpp) so that the
          // seed and the host test share one expression. Do not inline it again here:
          //
          // sn.chia holds the EFFECTIVE absorption f*chi_a, so chia + chis is the f-independent total
          // extinction and Er = B for any f (storing the bare chi_a gave Er = f*B; gate G8).
          const Real chi_rg = kr * optab.RossMultRT(g, rho, Tk);
          const SnCoeffs co =
              SnFleckSplit(chi_a, ks, ross_excess ? chi_rg : 0.0, B_g, chat, fleck);
          pack(b, isrc + g, k, j, i) = co.eta;
          pack(b, ichia + g, k, j, i) = co.chi_a_eff;
          pack(b, ichis + g, k, j, i) = co.chi_s_eff;
        }
      },
      Kokkos::Max<Real>(gmax));

  // NON-SILENT, and now with the right criterion. Per-rank warn-once, matching the convention
  // MatterCoupling uses for its own non-convergence.
  if (do_couple && gmax > 2.0) {
    static bool warned_stab = false;
    if (!warned_stab) {
      warned_stab = true;
      std::cout << "### WARNING Sn: matter-coupling stability number f*S = " << gmax
                << " on rank " << parthenon::Globals::my_rank
                << " EXCEEDS 2. The gas update amplifies perturbations by |1 - f*S| > 1 and "
                   "will oscillate. With radiation/sn_fleck_alpha = 1 this is impossible by "
                   "construction (f*S = S/(1+S) < 1), so reaching here means alpha has been "
                   "lowered, or dt/opacity changed between the seed and the update. "
                   "(warn-once)"
                << std::endl;
    }
  }

  // The sn.u -> sn.un copy is a SEPARATE kernel with n in the PARALLEL index set, not a
  // sequential inner loop inside the cell kernel. With the angle loop inner, one thread walked
  // nfa = 240 components at a stride of a whole block -- 240 scattered accesses per thread and
  // no coalescing at all. Flattened over (b, n, k, j, i), consecutive threads read consecutive
  // i and every access coalesces. This is the same argument as design section 4.2, applied to
  // the one kernel that was still violating it.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::SeedCopy", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, 0, nfa - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        pack(b, iun + n, k, j, i) = pack(b, iu + n, k, j, i);
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Pass 1 of the source: reduce the angle vector to (J, H1, H2, H3) per group.
//!
//! Angle loop is SEQUENTIAL and INNER; parallelism is over cells. Per-thread state is four
//! scalars per group, INDEPENDENT of nang. This is the whole reason the source is two passes
//! instead of one: a fused pass would need the 240-component angle vector live per thread,
//! which is exactly the local-memory spill documented at radiation_moments.cpp:912-938 where
//! per-thread state, not parallelism, set the throughput.
TaskStatus SnComputeMoments(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");

  // INTERIOR: sn.mom is OneCopy and not FillGhost, so its ghost values are neither exchanged
  // nor read. See the note in SnSeedSource.
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> names{USolName(), MomName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int imom = imap[MomName()].first;
  const int nang = p.nang;
  const int n_group = p.n_group;
  auto wv = ang.w;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real fourpi_inv = 1.0 / (4.0 * M_PI);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::Moments", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int g = 0; g < n_group; ++g) {
          Real J = 0.0, H1 = 0.0, H2 = 0.0, H3 = 0.0;
          const int base = iu + g * nang;
          for (int a = 0; a < nang; ++a) {
            const Real wi = wv(a) * pack(b, base + a, k, j, i);
            J += wi;
            H1 += wi * nxv(a);
            H2 += wi * nyv(a);
            H3 += wi * nzv(a);
          }
          pack(b, imom + 4 * g + 0, k, j, i) = J * fourpi_inv;
          pack(b, imom + 4 * g + 1, k, j, i) = H1 * fourpi_inv;
          pack(b, imom + 4 * g + 2, k, j, i) = H2 * fourpi_inv;
          pack(b, imom + 4 * g + 3, k, j, i) = H3 * fourpi_inv;
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Write the converged moments back into rad.Er / rad.Fr.
//!
//! Er = 4 pi J/chat, Fr = 4 pi H (single-sourced in radiation_sn_units.hpp, round trip tested in
//! tests/test_sn_units.cpp). Runs after the solve, when sn.mom is current with sn.u. No ghost
//! exchange afterwards: rad.Er/rad.Fr are outputs here, the sweep transports sn.u.
TaskStatus SnWriteback(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  if (!p.writeback) return TaskStatus::complete;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> names{MomName()};
  for (int g = 0; g < p.n_group; ++g)
    for (const auto &nm : GroupFieldNames(g)) names.push_back(nm);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int imom = imap[MomName()].first;
  Kokkos::Array<int, MAX_GROUP * 4> ir;
  for (int g = 0; g < p.n_group; ++g) {
    const auto nm = GroupFieldNames(g);
    for (int a = 0; a < 4; ++a) ir[4 * g + a] = imap[nm[a]].first;
  }
  const int n_group = p.n_group;
  const Real chat = pkg->Param<Real>("chat");
  const Real efloor = pkg->Param<Real>("efloor");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::Writeback", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int g = 0; g < n_group; ++g) {
          // Er = 4*pi*J/chat, Fr_d = 4*pi*H_d. Both factors come from radiation_sn_units.hpp
          // and are written nowhere else in the package.
          pack(b, ir[4 * g + 0], k, j, i) =
              std::max(efloor, SnEnergyFromJ(pack(b, imom + 4 * g + 0, k, j, i), chat));
          for (int d = 0; d < 3; ++d)
            pack(b, ir[4 * g + 1 + d], k, j, i) =
                SnFluxFromH(pack(b, imom + 4 * g + 1 + d, k, j, i));
        }
        // The reduced-flux bound |Fr| <= chat*Er is NOT imposed here, and that is deliberate:
        // it holds identically for any non-negative intensity because |H| <= J follows from
        // |Omega_d| <= 1 under a quadrature with positive weights (checked numerically in
        // tests/test_sn_units.cpp check 3). Clamping it would convert a genuine failure --
        // negative intensities, a corrupted quadrature -- into a plausible-looking field.
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Apply the net absorption/emission energy exchange back onto the gas.
//!
//! Not MatterCoupling (that would double-count): the Sn operator already contains emission and
//! absorption, so the gas is charged with what the transport did. From the discrete moment equation
//!     (Er - Er^n)/dt + div Fr = 4 pi eta - chat f chi_a Er
//! the gas change is dE_gas = -(c/chat) dt (4 pi eta - chat chi_a,eff Er), computed from the SAME
//! stored eta and chi_a the sweep used, so energy conserves to the solver tolerance. chi_a, not the
//! total extinction (scattering exchanges no energy). With the Fleck factor (sn_fleck_alpha = 1)
//! the exchange is unconditionally stable (f*S < 1); the internal-energy floor at coupling_tmin is
//! the only place conservation is broken, and it is counted.
TaskStatus SnMatterCoupling(MeshData<Real> *md, const Real dt,
                            std::shared_ptr<std::vector<int>> off) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  if (!p.matter_coupling) return TaskStatus::complete;
  Kokkos::Array<int, MAX_GROUP> goff;
  for (int g = 0; g < MAX_GROUP; ++g)
    goff[g] = (off != nullptr && g < static_cast<int>(off->size())) ? (*off)[g] : 0;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<std::string> names{SrcName(), ChiaName(), ChisName(), FleckName(), "cons"};
  for (int g = 0; g < p.n_group; ++g)
    for (const auto &nm : GroupFieldNames(g)) names.push_back(nm);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int isrc = imap[SrcName()].first;
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int ifl = imap[FleckName()].first;
  const int ic = imap["cons"].first;
  Kokkos::Array<int, MAX_GROUP * 4> ir;
  for (int g = 0; g < p.n_group; ++g) {
    const auto nm = GroupFieldNames(g);
    for (int a = 0; a < 4; ++a) ir[4 * g + a] = imap[nm[a]].first;
  }
  const bool do_force = p.radiation_force;

  const int n_group = p.n_group;
  const Real c_light = pkg->Param<Real>("c");
  const Real chat = pkg->Param<Real>("chat");
  const Real c_over_chat = c_light / chat;
  const Real tmin = pkg->Param<Real>("coupling_tmin");
  const Real T_unit = pkg->Param<Real>("T_unit");
  const Real gm1 = pkg->Param<Real>("gamma") - 1.0;
  const bool use_h2 = pkg->Param<bool>("use_h2diss");
  const auto eos_tab = pkg->Param<EOSTable::EosTable>("eos_tab");
  const int nhydro = pmb->pmy_mesh->packages.Get("Hydro")->Param<int>("nhydro");
  const bool mhd = (nhydro > NHYDRO);

  int nfloor = 0;
  Real maxrel = 0.0;
  // radiation/sn_dual_energy, see Initialize.
  const auto hydro_pkg_mc = pmb->packages.Get("Hydro");
  const bool sn_dual = p.dual_energy && hydro_pkg_mc->Param<int>("dual_idx") >= 0;
  const int idual_c = sn_dual ? hydro_pkg_mc->Param<int>("dual_idx") : 0;
  const Real dual_eta1 = sn_dual ? hydro_pkg_mc->Param<Real>("dual_energy_eta1") : 0.0;
  const SnDualSel dsel(hydro_pkg_mc.get(), md, sn_dual);  // audit P-01, see SnSeedSource
  parthenon::par_reduce(
      // DEFAULT_LOOP_PATTERN. This kernel was first written with an explicit
      // loop_pattern_mdrange_tag -- the identical mistake that cost SnSweepColour 9.24x and is
      // documented at length there. On the GPU build the default resolves to a flattened 1D
      // range where consecutive threads take consecutive i; an MDRangePolicy tiles instead and
      // the fastest index stops being the contiguous one.
      DEFAULT_LOOP_PATTERN, "Sn::MatterCoupling", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &lfloor,
                    Real &lmax) {
        const Real rho = pack(b, ic + IDN, k, j, i);
        if (!(rho > 0.0)) return;
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
        const Real kin = 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) * irho;
        const Real e0_E = pack(b, ic + IEN, k, j, i) - kin - me;
        const Real e0 =
            !sn_dual ? e0_E
            : dsel.on ? dsel.Eint(e0_E, pack(b, ic + IEN, k, j, i),
                                  pack(b, ic + idual_c, k, j, i), rho, me, b, k, j, i)
                      : SnDualEint(e0_E, pack(b, ic + IEN, k, j, i),
                                   pack(b, ic + idual_c, k, j, i), dual_eta1);

        // Sum the discrete source over groups. Both terms are the SAME two the sweep applied:
        // 4*pi*eta is its (Fleck-scaled) emission and chat*f*chi_a*Er the absorption that
        // actually reaches the gas. The REMAINDER of the absorption, (1-f)*chi_a*Er, was
        // returned to the radiation field as effective scattering and must NOT appear here --
        // charging it to the gas would undo the Fleck linearisation and re-introduce exactly
        // the explicit overshoot it exists to prevent.
        Real src = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
        for (int g = 0; g < n_group; ++g) {
          if (goff[g]) continue;
          const Real Er = pack(b, ir[4 * g + 0], k, j, i);
          // ALREADY the effective absorption f*chi_a (see the seed). Multiplying by f again
          // here was the other half of the same bug.
          const Real chi_a = pack(b, ichia + g, k, j, i);
          src += kFourPi * pack(b, isrc + g, k, j, i) - chat * chi_a * Er;
          if (do_force) {
            // Momentum uses the TOTAL extinction: isotropic scattering deposits a photon's
            // momentum in the gas just as absorption does, even though it returns the energy.
            const Real chi_e = chi_a + pack(b, ichis + g, k, j, i);
            fx += chi_e * pack(b, ir[4 * g + 1], k, j, i);
            fy += chi_e * pack(b, ir[4 * g + 2], k, j, i);
            fz += chi_e * pack(b, ir[4 * g + 3], k, j, i);
          }
        }
        Real dE = -c_over_chat * dt * src;   // gas gains what the radiation lost

        // --- RADIATION FORCE ---------------------------------------------------------------
        // d(rho v) = dt * chi_ext * Fr / c, the standard rho*kappa*F/c. Cross-checked against
        // MatterCoupling, which writes dv = -dF/(c*chat*rho) with dF = -chat*chi*dt*Fr from the
        // flux drag: the same d(rho v) = dt*chi*Fr/c. Nothing is subtracted from Fr here --
        // the Sn transport's own -chi*I term already damped the H moment, so the momentum the
        // gas takes is momentum the radiation has ALREADY lost. Deducting it twice is the
        // exact analogue of the MatterCoupling double-count this whole operator avoids.
        Real dpx = 0.0, dpy = 0.0, dpz = 0.0;
        if (do_force) {
          const Real k_over_c = dt / c_light;
          dpx = k_over_c * fx; dpy = k_over_c * fy; dpz = k_over_c * fz;
        }
        const Real n1 = m1 + dpx, n2 = m2 + dpy, n3 = m3 + dpz;
        const Real kin_new = 0.5 * (n1 * n1 + n2 * n2 + n3 * n3) * irho;
        const Real dEk = kin_new - kin;

        // The work done by the radiation force comes OUT OF THE THERMAL BUDGET, not out of
        // thin air: total gas energy changes by dE (what the radiation lost) and the kinetic
        // share dEk is carved from it, so internal energy moves by dE - dEk. That keeps total
        // energy exactly conserved while the momentum transfer is applied, which is the only
        // split available without O(v/c) comoving-frame source terms -- and this transport has
        // none, so no more accuracy than this is on offer.
        Real e_new = e0 + dE - dEk;

        // Floor on INTERNAL energy, at the same coupling_tmin the emission temperature is
        // clamped to -- using a different floor here than the emission uses would let a cell
        // emit at one temperature and be floored at another.
        const Real e_min = use_h2 ? eos_tab.EintFromRhoTk(rho, tmin * T_unit, true)
                                  : rho * tmin / gm1;
        if (e_new < e_min) {
          e_new = e_min;
          lfloor += 1;   // conservation is deliberately broken HERE and nowhere else
        }
        if (do_force) {
          pack(b, ic + IM1, k, j, i) = n1;
          pack(b, ic + IM2, k, j, i) = n2;
          pack(b, ic + IM3, k, j, i) = n3;
        }
        pack(b, ic + IEN, k, j, i) = kin_new + me + e_new;
        // The advected internal energy moves with the gas internal energy, so a switched cell keeps
        // the radiative heating/cooling instead of having it undone by the next ConsToPrim.
        if (sn_dual) pack(b, ic + idual_c, k, j, i) = e_new;
        // The stability number. With the Fleck linearisation (alpha = 1) this is bounded by
        // construction; it is still MEASURED, because "bounded by construction" is a claim
        // about the algebra and not about the code.
        const Real den = std::max(std::abs(e0), Real(1.0e-300));
        const Real rel = std::abs(e_new - e0) / den;
        if (rel > lmax) lmax = rel;
      },
      Kokkos::Sum<int>(nfloor), Kokkos::Max<Real>(maxrel));

  // NON-SILENT, matching the convention MatterCoupling uses for its own non-convergence: a
  // coupling that floors cells or overshoots is not allowed to look like physical heating.
  // Per-rank and warn-once. Per-rank because a global reduction here would cost a collective
  // every step to police a condition that should never fire; warn-once because if it fires it
  // will fire on most steps and the first one is the diagnostic.
  if (nfloor > 0) {
    static bool warned_floor = false;
    // env APK_SN_FLOOR_EVERY=1 (diagnostic): report EVERY floor event instead of the first per
    // rank, so floor counts can be compared between runs (the warn-once log cannot be counted).
    static const bool every_floor = std::getenv("APK_SN_FLOOR_EVERY") != nullptr;
    if (!warned_floor || every_floor) {
      warned_floor = true;
      std::cout << "### WARNING Sn: matter coupling FLOORED the gas internal energy in "
                << nfloor << " cell(s) on rank " << parthenon::Globals::my_rank
                << ". Energy conservation is broken in exactly those cells. This means the "
                   "explicit exchange over-cooled them: reduce dt, reduce the opacity, or "
                   "wait for the implicit coupling. (warn-once)"
                << std::endl;
    }
  }
  // |dE_gas|/e_int is NO LONGER WARNED ON, deliberately. It measures how far the gas moved,
  // which is large whenever the gas is far from equilibrium -- a correct, converging relaxation
  // does exactly that. Measured on gate G11 it was 0.166 on the STABLE Fleck arm and 0.496 on
  // the UNSTABLE explicit one: both above the old 0.1 threshold, so it fired on both and
  // discriminated neither. The stability criterion is exact and lives in SnSeedSource
  // (f*S < 2); this value is kept only because a large move is worth being able to see.
  (void)maxrel;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief ORDERED (downstream) transport sweep -- the classical Sn smoother.
//!
//! Ordered downstream sweep: with chi_s = 0 each ordinate's operator is triangular in its own
//! direction, so sweeping downstream inverts a block exactly in one pass for any Courant number;
//! only block boundaries remain iterative. Parallel over (blocks x ordinates), sequential in space.
//! Production smoother (sn_smoother = ordered); the default stays redblack for the gate numbers.
void SnApplyIncomingBC(std::shared_ptr<parthenon::MeshBlockData<Real>> &mbd, const bool coarse,
                       const int dir, const bool inner) {
  auto pmb = mbd->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const int mode = pkg->Param<int>("sn_bc_mode");
  if (mode == 0) return;
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const auto ubg = pkg->Param<Kokkos::View<Real *>>("sn_bc_ubg");
  const int nang = p.nang;

  // The exchange that fills the Krylov vector's halo hands this function a container holding
  // ONLY sn.k_y (the driver exchanges a MeshData built from that one variable), while the
  // state exchange holds sn.u (and k_y, when registered). Pack whatever is present; packing
  // an absent name gives a meaningless index map, which left k_y's incoming ordinates at the
  // zero-gradient copy and broke the recurrence/true-residual agreement (job 2753925).
  const bool has_u = mbd->Contains(USolName());
  const bool has_ky = mbd->Contains(KyName());
  const bool has_kcy = mbd->Contains(KcyName());   // R-F3 coupled-solve preconditioned vector
  if (!has_u && !has_ky && !has_kcy) return;
  std::vector<std::string> names;
  if (has_u) names.push_back(USolName());
  if (has_ky) names.push_back(KyName());
  if (has_kcy) names.push_back(KcyName());
  parthenon::PackIndexMap imap;
  auto pack = mbd->PackVariables(names, imap, coarse);
  const int iu = has_u ? imap[USolName()].first : 0;
  const int nu = has_u ? (imap[USolName()].second - iu + 1) : 0;
  const int iy = has_ky ? imap[KyName()].first : 0;
  const int icy = has_kcy ? imap[KcyName()].first : 0;
  const int ncy = has_kcy ? (imap[KcyName()].second - icy + 1) : 0;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const IndexDomain dom =
      (dir == 1) ? (inner ? IndexDomain::inner_x1 : IndexDomain::outer_x1)
                 : (dir == 2) ? (inner ? IndexDomain::inner_x2 : IndexDomain::outer_x2)
                              : (inner ? IndexDomain::inner_x3 : IndexDomain::outer_x3);
  const IndexRange ib = bounds.GetBoundsI(dom);
  const IndexRange jb = bounds.GetBoundsJ(dom);
  const IndexRange kb = bounds.GetBoundsK(dom);
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::IncomingBC", parthenon::DevExecSpace(), kb.s, kb.e, jb.s,
      jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        for (int a = 0; a < nang; ++a) {
          const Real nc = (dir == 1) ? nxv(a) : (dir == 2) ? nyv(a) : nzv(a);
          const bool incoming = inner ? (nc > 0.0) : (nc < 0.0);
          if (!incoming) continue;
          for (int n = a; n < nu; n += nang) pack(iu + n, k, j, i) = (mode == 2) ? ubg(n / nang) : 0.0;
          if (has_ky) pack(iy + a, k, j, i) = 0.0;
          for (int n = a; n < ncy; n += nang) pack(icy + n, k, j, i) = 0.0;
        }
      });
}

TaskStatus SnSweepOrdered(MeshData<Real> *md, const Real dt, Real *res, Real *nrm,
                          Real *nclamp = nullptr) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = (pmb->pmy_mesh->ndim >= 2);
  const bool three_d = (pmb->pmy_mesh->ndim >= 3);

  const bool diamond = p.diamond;
  const bool fixup = p.diamond_fixup;
  const Real rw = 1.0 / p.dd_w;
  const bool sou = p.sou;
  const Real kap = p.sou_kappa;
  std::vector<std::string> names{USolName(), UnName(),   SrcName(),
                                 ChiaName(), ChisName(), MomName()};
  if (diamond) { names.push_back(FxName()); names.push_back(FyName()); names.push_back(FzName()); }
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int iun = imap[UnName()].first;
  const int isrc = imap[SrcName()].first;
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int imom = imap[MomName()].first;
  const int ifx = diamond ? imap[FxName()].first : 0;
  const int ify = diamond ? imap[FyName()].first : 0;
  const int ifz = diamond ? imap[FzName()].first : 0;

  const int nang = p.nang;
  const int nfa = p.nfa;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);
  const int is = ib.s, ie = ib.e, js = jb.s, je = jb.e, ks = kb.s, ke = kb.e;

  // CLAMP COUNT. The set-to-zero face fixup is the ONLY thing that makes this sweep non-linear
  // and the only thing that breaks flux telescoping between neighbouring cells -- i.e. the only
  // thing that can make WDD non-conservative, or make SnResidual describe a different operator
  // than the sweep actually solves. A run that reports ZERO clamps is thereby certified exactly
  // conservative and exactly consistent, which is a far stronger statement than "the
  // conservation check passed". Reported, never inferred.
  Real lres = 0.0, lnrm = 0.0, lclamp = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::SweepOrdered", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, 0, nfa - 1,
      KOKKOS_LAMBDA(const int b, const int n, Real &sres, Real &snrm, Real &sclamp) {
        const int g = n / nang;
        const int a = n - g * nang;
        // Downstream order: start at the UPWIND end of the block and march with the flow, so
        // every cell's upwind neighbour has already been updated in THIS pass.
        const int di = (nxv(a) > 0.0) ? 1 : -1;
        const int dj = (nyv(a) > 0.0) ? 1 : -1;
        const int dk = (nzv(a) > 0.0) ? 1 : -1;
        const int i0 = (di > 0) ? is : ie, i1 = (di > 0) ? ie : is;
        const int j0 = (dj > 0) ? js : je, j1 = (dj > 0) ? je : js;
        const int k0 = (dk > 0) ? ks : ke, k1 = (dk > 0) ? ke : ks;
        const auto &co = pack.GetCoords(b);
        for (int k = k0;; k += dk) {
          for (int j = j0;; j += dj) {
            for (int i = i0;; i += di) {
              const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
              const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
              const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
              // Upwind neighbour is the one we just came FROM: index -di, not +di.
              const Real ux = pack(b, iu + n, k, j, i - di);
              const Real uy = two_d ? pack(b, iu + n, k, j - dj, i) : 0.0;
              const Real uz = three_d ? pack(b, iu + n, k - dk, j, i) : 0.0;
              const Real chi_a = std::max(0.0, pack(b, ichia + g, k, j, i));
              const Real chi_s = std::max(0.0, pack(b, ichis + g, k, j, i));
              const Real src = pack(b, isrc + g, k, j, i)
                             + chi_s * pack(b, imom + 4 * g + 0, k, j, i);
              Real inew;
              if (sou) {
                // Second upwind neighbour, two cells back along the sweep. Both upwind values
                // were updated earlier in THIS pass (Gauss-Seidel); across a block inlet they
                // are the two exchanged ghosts, which is why sou needs nghost >= 2.
                const Real ux2 = pack(b, iu + n, k, j, i - 2 * di);
                const Real uy2 = two_d ? pack(b, iu + n, k, j - 2 * dj, i) : 0.0;
                const Real uz2 = three_d ? pack(b, iu + n, k - 2 * dk, j, i) : 0.0;
                inew = SnSouCell(pack(b, iun + n, k, j, i), rdt, chi_a + chi_s, src, kap,
                                 ax, ay, az, ux, uy, uz, ux2, uy2, uz2);
              } else if (!diamond) {
                inew = SnUpdateCell(pack(b, iun + n, k, j, i), rdt, chi_a + chi_s, src,
                                    ax, ay, az, ux, uy, uz);
              } else {
                // Incoming face = the UPWIND NEIGHBOUR's outgoing face in that direction.
                const Real fxi = pack(b, ifx + n, k, j, i - di);
                const Real fyi = two_d ? pack(b, ify + n, k, j - dj, i) : 0.0;
                const Real fzi = three_d ? pack(b, ifz + n, k - dk, j, i) : 0.0;
                inew = SnWddCell(pack(b, iun + n, k, j, i), rdt, chi_a + chi_s, src, rw,
                                 ax, ay, az, fxi, fyi, fzi);
                // POSITIVITY FIXUP. Diamond differencing is not positive: I_out = 2I - I_in can
                // go negative where the solution is under-resolved. Clamping the FACE (not the
                // cell) is the standard set-to-zero fixup; it costs formal order locally and is
                // the price of the diffusion limit. Recorded, not hidden.
                const Real fl = fixup ? 0.0 : -1.0e300;
                const Real gx = SnWddFace(inew, rw, fxi);
                const Real gy = two_d ? SnWddFace(inew, rw, fyi) : 0.0;
                const Real gz = three_d ? SnWddFace(inew, rw, fzi) : 0.0;
                if (gx < fl) sclamp += 1.0;
                if (two_d && gy < fl) sclamp += 1.0;
                if (three_d && gz < fl) sclamp += 1.0;
                pack(b, ifx + n, k, j, i) = std::max(fl, gx);
                if (two_d) pack(b, ify + n, k, j, i) = std::max(fl, gy);
                if (three_d) pack(b, ifz + n, k, j, i) = std::max(fl, gz);
              }
              // SOU NEGATIVITY DIAGNOSTIC, counted through the same reduction slot the WDD
              // face fixup uses -- but NOTHING IS CLAMPED. A fixup is non-linear and would
              // invalidate both BiCGSTAB's Krylov space and the sweep/residual/operator
              // identity, so under sou the count is a measurement and never an action.
              if (sou && inew < 0.0) sclamp += 1.0;
              const Real iold = pack(b, iu + n, k, j, i);
              pack(b, iu + n, k, j, i) = inew;
              sres += std::abs(inew - iold);
              snrm += std::abs(inew);
              if (i == i1) break;
            }
            if (j == j1) break;
          }
          if (k == k1) break;
        }
      },
      Kokkos::Sum<Real>(lres), Kokkos::Sum<Real>(lnrm), Kokkos::Sum<Real>(lclamp));
  *res = lres;
  *nrm = lnrm;
  if (nclamp != nullptr) *nclamp += lclamp;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Component width of a packed field, read from the pack index map.
//!
//! The Krylov kernels are shared between nfa-wide STATE fields (sn.u, sn.un) and nang-wide WORK
//! vectors. Deriving the width from the pack instead of passing it means a mis-registered field
//! shows up as a wrong width here rather than as a silent stride error `nang` components deep
//! into the next field.
inline int SnPackWidth(parthenon::PackIndexMap &imap, const std::string &nm) {
  return imap[nm].second - imap[nm].first + 1;
}

//! First component of group `g` of field `nm`. An nfa-wide field is offset by g*nang; a
//! nang-wide work vector holds exactly one group and starts at its own origin.
inline int SnGroupBase(parthenon::PackIndexMap &imap, const std::string &nm, const int g,
                       const int nang) {
  return imap[nm].first + ((SnPackWidth(imap, nm) > nang) ? g * nang : 0);
}

//----------------------------------------------------------------------------------------
//! \brief Zeroth angular moment J_g = (1/4pi) sum_a w_a x_a of ONE GROUP of a vector.
//!
//! Same convention as SnComputeMoments (which computes it for sn.u): weights from the product
//! Gauss-Legendre quadrature, divided by 4pi. Needed because the Krylov operator now applies
//! the scattering coupling to ITS OWN input vector, not to the last iterate's moments.
//!
//! `xn` may be an nfa-wide state field (sn.u) or a nang-wide Krylov work vector; the width is
//! read from the pack rather than assumed, and an nfa-wide field is indexed into group `g`.
//! `jn` receives a SINGLE component -- the Krylov path solves one group at a time.
TaskStatus SnMomentOf(MeshData<Real> *md, const int g, const std::string &xn,
                      const std::string &jn) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{xn, jn};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  const int base = SnGroupBase(imap, xn, g, nang);
  const int ij = imap[jn].first;
  auto wv = ang.w;
  const Real fourpi_inv = 1.0 / (4.0 * M_PI);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::MomentOf", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Real J = 0.0;
        for (int a = 0; a < nang; ++a) J += wv(a) * pack(b, base + a, k, j, i);
        pack(b, ij, k, j, i) = J * fourpi_inv;
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Matrix-free operator apply. `mode` 0: y = A*x.  `mode` 1: y = b - A*x.  `mode` 2: y = b.
//!
//! A CONTAINS THE SCATTERING COUPLING:
//!
//!     (A x)_i = (rdt + chi_a + chi_s) x_i - sum_d adx_d x_up,d - chi_s * J(x),
//!               J(x) = (1/4pi) sum_a w_a x_a        (SnMomentOf, on THIS vector)
//!
//! with b = rdt*I^n + eta alone. An earlier version LAGGED chi_s*J into b, making A
//! block-diagonal in the ordinates. That was wrong for the Krylov path -- the driver has no
//! inner moment refresh, so it converged exactly ONE source iteration and reported success --
//! and it was papered over by refusing chi_s > 0. The guard is gone because the cause is gone.
//!
//! With the scattering coupling inside A, BiCGSTAB solves transport + scattering together
//! (Krylov-accelerated source iteration). SnResidual stays the independent reference: it equals
//! (dia x - streaming - chi_s J) - b provided sn.mom is current with sn.u.
TaskStatus SnApplyA(MeshData<Real> *md, const Real dt, const int g, const std::string &xn,
                    const std::string &yn, const int mode) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = (pmb->pmy_mesh->ndim >= 2);
  const bool three_d = (pmb->pmy_mesh->ndim >= 3);

  if (p.sou_limit) return SnApplyASouLim(md, dt, g, xn, yn, mode);   // R-F10 (host-side dispatch)
  // J(x) must be the moment of THIS vector, so compute it first into k_j.
  SnMomentOf(md, g, xn, KjName());
  std::vector<std::string> names{xn, yn, UnName(), SrcName(), ChiaName(), ChisName(), KjName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  // Group bases: sn.u / sn.un are nfa-wide and are indexed INTO group g; the Krylov work
  // vectors are nang-wide and start at their own origin. SnGroupBase reads which from the pack.
  const int ix = SnGroupBase(imap, xn, g, nang);
  const int iy = SnGroupBase(imap, yn, g, nang);
  const int iun = imap[UnName()].first + g * nang;
  const int isrc = imap[SrcName()].first + g;
  const int ichia = imap[ChiaName()].first + g;
  const int ichis = imap[ChisName()].first + g;
  const int ikj = imap[KjName()].first;   // one component: the group being solved
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);
  // SOU: the streaming term reads a second upwind cell with the coefficients of SnSouRhs. The
  // stencil is written out here by hand -- NOT via SnSouRhs -- so that this kernel and
  // SnResidual remain two independent transcriptions of the same operator, which is what the
  // per-cell gate (SnKrVerifyOp) compares. At kappa = 0 the c2 terms vanish and c1 = 1.
  const bool sou = p.sou;
  const Real kap = p.sou_kappa;
  const Real c1 = sou ? 1.0 + kap : 1.0;
  const Real c2 = sou ? 0.5 * kap : 0.0;

  // TWO KERNELS, NOT ONE WITH A BRANCH -- and this is a MEASURED requirement, not tidiness.
  // The first version put `if (sou)` inside the single kernel, leaving the step arithmetic
  // textually unchanged. The default path was then NOT bit-identical under bicgstab: gate K4's
  // step arm moved by 1.45e-14 and K5's by 4.0e-15 (richardson, which never calls SnApplyA, was
  // bit-identical, which is what localised it here). Two runs of the PRE-CHANGE binary were
  // bit-identical to each other, so run-to-run non-determinism was falsified and the cause was
  // this kernel's codegen -- same expression, different FMA contraction in the presence of the
  // second branch. Keeping the step kernel a standalone, untouched lambda restores exact
  // bit-identity. Do not merge them again.
  if (!sou) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Sn::ApplyA", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0,
        nang - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) {
          const auto &co = pack.GetCoords(b);
          const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
          const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
          const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
          const int di = (nxv(a) > 0.0) ? -1 : 1;
          const int dj = (nyv(a) > 0.0) ? -1 : 1;
          const int dk = (nzv(a) > 0.0) ? -1 : 1;
          const Real ux = pack(b, ix + a, k, j, i + di);
          const Real uy = two_d ? pack(b, ix + a, k, j + dj, i) : 0.0;
          const Real uz = three_d ? pack(b, ix + a, k + dk, j, i) : 0.0;
          const Real chi_a = std::max(0.0, pack(b, ichia, k, j, i));
          const Real chi_s = std::max(0.0, pack(b, ichis, k, j, i));
          // SCATTERING IS IN THE OPERATOR, not lagged into b: subtract chi_s * J(x), the moment
          // of the INPUT vector. This is what makes A the full transport+scattering operator.
          const Real Ax = SnDiagonal(rdt, chi_a + chi_s, ax, ay, az) * pack(b, ix + a, k, j, i)
                          - (ax * ux + ay * uy + az * uz)
                          - chi_s * pack(b, ikj, k, j, i);
          if (mode == 0) {
            pack(b, iy + a, k, j, i) = Ax;
          } else {
            // b carries only the TRUE source (rdt*I^n + eta); chi_s*J moved into A above.
            const Real rhs = SnRhs(pack(b, iun + a, k, j, i), rdt, pack(b, isrc, k, j, i),
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            // mode 1: r = b - A x.  mode 2: b alone.
            pack(b, iy + a, k, j, i) = (mode == 1) ? rhs - Ax : rhs;
          }
        });
    return TaskStatus::complete;
  }

  // SECOND-ORDER UPWIND. The stencil is written out by hand here rather than routed through
  // SnSouRhs/SnSouDiagonal, so this kernel and SnResidual (which does use the header) remain
  // two INDEPENDENT transcriptions of one operator -- which is exactly what the per-cell gate
  // SnKrVerifyOp compares, and a shared helper would make that comparison a tautology.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::ApplyA_sou", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1,
      0, nang - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) {
        const auto &co = pack.GetCoords(b);
        const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
        const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
        const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
        const int di = (nxv(a) > 0.0) ? -1 : 1;
        const int dj = (nyv(a) > 0.0) ? -1 : 1;
        const int dk = (nzv(a) > 0.0) ? -1 : 1;
        const Real ux = pack(b, ix + a, k, j, i + di);
        const Real uy = two_d ? pack(b, ix + a, k, j + dj, i) : 0.0;
        const Real uz = three_d ? pack(b, ix + a, k + dk, j, i) : 0.0;
        const Real ux2 = pack(b, ix + a, k, j, i + 2 * di);
        const Real uy2 = two_d ? pack(b, ix + a, k, j + 2 * dj, i) : 0.0;
        const Real uz2 = three_d ? pack(b, ix + a, k + 2 * dk, j, i) : 0.0;
        const Real chi_a = std::max(0.0, pack(b, ichia, k, j, i));
        const Real chi_s = std::max(0.0, pack(b, ichis, k, j, i));
        const Real Ax = (rdt + chi_a + chi_s + (1.0 + 0.5 * kap) * (ax + ay + az)) *
                            pack(b, ix + a, k, j, i)
                        - c1 * (ax * ux + ay * uy + az * uz)
                        + c2 * (ax * ux2 + ay * uy2 + az * uz2)
                        - chi_s * pack(b, ikj, k, j, i);
        if (mode == 0) {
          pack(b, iy + a, k, j, i) = Ax;
        } else {
          const Real rhs = SnRhs(pack(b, iun + a, k, j, i), rdt, pack(b, isrc, k, j, i),
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
          pack(b, iy + a, k, j, i) = (mode == 1) ? rhs - Ax : rhs;
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Zero an nfa-component field over the WHOLE block, ghosts included.
//!
//! Ghosts matter. The preconditioner is applied to a zero-initialised y and reads y's ghosts on
//! the upwind face; leaving stale ghosts there would make M^-1 depend on the previous call, and
//! a preconditioner that is not a FIXED LINEAR OPERATOR silently invalidates BiCGSTAB's
//! minimisation -- it would still run, still print a falling residual, and converge to the
//! wrong answer or stagnate. This is the same class of error as the frozen-ghost bug in the M1
//! sub-cycle, which is why it gets its own kernel instead of an interior-only fill.
TaskStatus SnZeroAll(MeshData<Real> *md, const std::string &nm) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{nm}, imap);
  const int nc = SnPackWidth(imap, nm);   // nang for a Krylov work vector, nfa for a state field
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::ZeroAll", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0,
      nc - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        pack(b, n, k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief PRECONDITIONER: one ordered downstream transport sweep of M y = rhs, rhs taken from
//!        a field rather than rebuilt from (I^n, source).
//!
//! M is the same operator A is built from -- same diagonal SnDiagonal, same upwind stencil,
//! same chi_a + chi_s -- differing only in that the sweep uses already-updated downstream
//! values (Gauss-Seidel) where A uses the current iterate. Reusing SnUpdateCell with
//! in_old = 0 and src = rhs is deliberate: it makes "the preconditioner approximates THIS A"
//! true by construction rather than by two transcriptions agreeing.
//!
//! The sweep is block-local (upwind ghosts from the last exchange), a block-Gauss-Seidel-like
//! preconditioner; the Krylov method supplies the global coupling the sweep cannot (optically thin
//! domains converge at rate 0.9998 under the sweep alone).
TaskStatus SnPrecondSweep(MeshData<Real> *md, const Real dt, const int g,
                          const std::string &rn, const std::string &yn) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  // 1-D: a plane is one cell, so the wavefront kernel only adds barriers (measured 0.8-0.9x); keep the serial sweep.
  if (p.wavefront == 5 && pmb->pmy_mesh->ndim >= 2)
    return SnPrecondSweepWave5Groups(md, dt, rn, yn, p.precond_sou, std::vector<int>{g});
  if (p.wavefront > 0 && pmb->pmy_mesh->ndim >= 2) return SnPrecondSweepWave(md, dt, g, rn, yn, p.precond_sou, p.wavefront);
  if (p.precond_sou) return SnPrecondSweepSou(md, dt, g, rn, yn);   // R-F12, radiation_sn_precond.cpp
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = (pmb->pmy_mesh->ndim >= 2);
  const bool three_d = (pmb->pmy_mesh->ndim >= 3);

  std::vector<std::string> names{rn, yn, ChiaName(), ChisName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  const int ir = SnGroupBase(imap, rn, g, nang);
  const int iy = SnGroupBase(imap, yn, g, nang);
  const int ichia = imap[ChiaName()].first + g;
  const int ichis = imap[ChisName()].first + g;

  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);
  const int is = ib.s, ie = ib.e, js = jb.s, je = jb.e, ks = kb.s, ke = kb.e;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::PrecondSweep", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, 0, nang - 1, KOKKOS_LAMBDA(const int b, const int a) {
        const int di = (nxv(a) > 0.0) ? 1 : -1;
        const int dj = (nyv(a) > 0.0) ? 1 : -1;
        const int dk = (nzv(a) > 0.0) ? 1 : -1;
        const int i0 = (di > 0) ? is : ie, i1 = (di > 0) ? ie : is;
        const int j0 = (dj > 0) ? js : je, j1 = (dj > 0) ? je : js;
        const int k0 = (dk > 0) ? ks : ke, k1 = (dk > 0) ? ke : ks;
        const auto &co = pack.GetCoords(b);
        for (int k = k0;; k += dk) {
          for (int j = j0;; j += dj) {
            for (int i = i0;; i += di) {
              const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
              const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
              const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
              const Real ux = pack(b, iy + a, k, j, i - di);
              const Real uy = two_d ? pack(b, iy + a, k, j - dj, i) : 0.0;
              const Real uz = three_d ? pack(b, iy + a, k - dk, j, i) : 0.0;
              const Real chi_a = std::max(0.0, pack(b, ichia, k, j, i));
              const Real chi_s = std::max(0.0, pack(b, ichis, k, j, i));
              // in_old = 0, src = rhs: SnUpdateCell then returns exactly
              // (rhs + sum_d adx_d y_up,d) / SnDiagonal(...), the M-solve for this cell.
              pack(b, iy + a, k, j, i) =
                  SnUpdateCell(0.0, rdt, chi_a + chi_s, pack(b, ir + a, k, j, i), ax, ay, az,
                               ux, uy, uz);
              if (i == i1) break;
            }
            if (j == j1) break;
          }
          if (k == k1) break;
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Vector helpers over an nfa-component field. dst = c1*f1 + c2*f2 + c3*f3.
//!
//! Deliberately ONE general routine rather than a family of axpy/copy/scale kernels: every
//! BiCGSTAB vector update is a three-term combination, and a single kernel means one place for
//! the index arithmetic to be wrong instead of five.
TaskStatus SnVecLC(MeshData<Real> *md, const Real c1, const std::string &n1, const Real c2,
                   const std::string &n2, const Real c3, const std::string &n3,
                   const std::string &dn) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  // DEDUPLICATED. Several BiCGSTAB updates name the same field twice -- x = x + alpha*y passes
  // x as both a source and the destination, and the s/xy/xz updates pass one operand twice to
  // fill the unused third slot. Whether PackVariables collapses a repeated name or packs it
  // again decides whether every index after the repeat is shifted by nfa, and depending on
  // that is depending on luck. Dedup here and look the indices up by NAME, which is correct
  // either way. (Reading and writing the same field is fine: the kernel is elementwise, one
  // thread per cell.)
  std::vector<std::string> names;
  for (const auto &nm : {n1, n2, n3, dn})
    if (std::find(names.begin(), names.end(), nm) == names.end()) names.push_back(nm);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int i1 = imap[n1].first, i2 = imap[n2].first, i3 = imap[n3].first,
            id = imap[dn].first;
  // Width from the pack, not from p.nfa: every operand here is a nang-wide Krylov work vector
  // since the per-group split. Mixing widths silently strides off the end of the shorter one,
  // so it is asserted rather than assumed -- the x-update, which DOES mix an nfa-wide sn.u with
  // a nang-wide k_y, has its own kernel (SnKrAxpyX) for exactly this reason.
  const int nc = SnPackWidth(imap, n1);
  PARTHENON_REQUIRE(SnPackWidth(imap, n2) == nc && SnPackWidth(imap, n3) == nc &&
                        SnPackWidth(imap, dn) == nc,
                    "SnVecLC operands have different component widths; use SnKrAxpyX for the "
                    "nfa-wide/nang-wide mixed update.");
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::VecLC", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0,
      nc - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        pack(b, id + n, k, j, i) = c1 * pack(b, i1 + n, k, j, i)
                                 + c2 * pack(b, i2 + n, k, j, i)
                                 + c3 * pack(b, i3 + n, k, j, i);
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Local (this-rank) dot product of two nfa-component fields. Global sum is the caller's
//!        job, via parthenon::AllReduce -- see the note on SnCheckConvergence about why a
//!        hand-rolled MPI_Allreduce on MPI_COMM_WORLD is a latent deadlock here.
TaskStatus SnVecDot(MeshData<Real> *md, const std::string &n1, const std::string &n2,
                    Real *out, const Real scale) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  // Deduplicated: the omega step takes the dot of k_t with ITSELF, so n1 == n2 there.
  std::vector<std::string> names{n1};
  if (n2 != n1) names.push_back(n2);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int i1 = imap[n1].first, i2 = imap[n2].first;
  const int nc = SnPackWidth(imap, n1);
  PARTHENON_REQUIRE(SnPackWidth(imap, n2) == nc, "SnVecDot operand widths differ.");
  Real ldot = 0.0;
  if (scale == 1.0) {  // the original kernel, untouched: keeps sn_krylov_dot_scale=false bit-identical
    parthenon::par_reduce(
        DEFAULT_LOOP_PATTERN, "Sn::VecDot", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0,
        nc - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &d) {
          d += pack(b, i1 + n, k, j, i) * pack(b, i2 + n, k, j, i);
        },
        Kokkos::Sum<Real>(ldot));
  } else {
    // Scale BEFORE multiplying: the point is that the product of two ~1e-160 components must
    // not underflow, which scaling the sum afterwards cannot prevent.
    parthenon::par_reduce(
        DEFAULT_LOOP_PATTERN, "Sn::VecDotScaled", parthenon::DevExecSpace(), 0,
        pack.GetDim(5) - 1, 0, nc - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &d) {
          d += (scale * pack(b, i1 + n, k, j, i)) * (scale * pack(b, i2 + n, k, j, i));
        },
        Kokkos::Sum<Real>(ldot));
  }
  *out = ldot;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief TRUE RESIDUAL of the discrete system: ||A x - b||_1 and ||b||_1.
//!
//! True residual, not the iterate increment: for spectral radius rho the error is ~rho/(1-rho) x
//! the increment, so a stalled source iteration looks converged. Costs ~+50 % on the Richardson
//! path; radiation/sn_criterion = increment reproduces the old numbers. b comes from SnRhs, the same
//! function SnUpdateCell uses.
TaskStatus SnResidual(MeshData<Real> *md, const Real dt, Real *res, Real *nrm,
                      const std::string &store = "", Real *tmax = nullptr,
                      const int gsel = -1) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = (pmb->pmy_mesh->ndim >= 2);
  const bool three_d = (pmb->pmy_mesh->ndim >= 3);

  if (p.sou_limit) return SnResidualSouLim(md, dt, res, nrm, store, tmax, gsel);   // R-F10
  std::vector<std::string> names{USolName(), UnName(),   SrcName(),
                                 ChiaName(), ChisName(), MomName()};
  // OPTIONAL PER-CELL STORE. Empty (the default) leaves this path bit-identical; the Krylov
  // operator gate passes a field name so it can compare SnApplyA's output CELL BY CELL against
  // the residual this kernel actually computes -- comparing SUMS is provably not enough, see
  // SnKrVerifyOp.
  if (!store.empty()) names.push_back(store);
  // The residual MUST measure the operator the sweep actually solves, or the convergence test
  // grades a different system than the one being iterated.
  const bool diamond = p.diamond;
  const Real rw = 1.0 / p.dd_w;
  const bool sou = p.sou;
  const Real kap = p.sou_kappa;
  if (diamond) { names.push_back(FxName()); names.push_back(FyName()); names.push_back(FzName()); }
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int iun = imap[UnName()].first;
  const int isrc = imap[SrcName()].first;
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int imom = imap[MomName()].first;
  const int ist = store.empty() ? -1 : imap[store].first;
  const int ifx = diamond ? imap[FxName()].first : 0;
  const int ify = diamond ? imap[FyName()].first : 0;
  const int ifz = diamond ? imap[FzName()].first : 0;

  const int nang = p.nang;
  const int nfa = p.nfa;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);

  const int nlo = (gsel < 0) ? 0 : gsel * nang;
  const int nhi = (gsel < 0) ? nfa - 1 : gsel * nang + nang - 1;

  Real lres = 0.0, lnrm = 0.0, ltmax = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::Residual", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1,
      nlo, nhi, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                    Real &sres, Real &snrm, Real &stmax) {
        const int g = n / nang;
        const int a = n - g * nang;
        const auto &co = pack.GetCoords(b);
        const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
        const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
        const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
        const int di = (nxv(a) > 0.0) ? -1 : 1;
        const int dj = (nyv(a) > 0.0) ? -1 : 1;
        const int dk = (nzv(a) > 0.0) ? -1 : 1;
        const Real ux = pack(b, iu + n, k, j, i + di);
        const Real uy = two_d ? pack(b, iu + n, k, j + dj, i) : 0.0;
        const Real uz = three_d ? pack(b, iu + n, k + dk, j, i) : 0.0;
        const Real chi_a = std::max(0.0, pack(b, ichia + g, k, j, i));
        const Real chi_s = std::max(0.0, pack(b, ichis + g, k, j, i));
        const Real src = pack(b, isrc + g, k, j, i)
                       + chi_s * pack(b, imom + 4 * g + 0, k, j, i);
        Real rhs, dia;
        if (sou) {
          // Same stencil as the ordered sweep (through SnSouRhs/SnSouDiagonal) and as
          // SnApplyA (written out by hand there): the second upwind cell along each direction.
          const Real ux2 = pack(b, iu + n, k, j, i + 2 * di);
          const Real uy2 = two_d ? pack(b, iu + n, k, j + 2 * dj, i) : 0.0;
          const Real uz2 = three_d ? pack(b, iu + n, k + 2 * dk, j, i) : 0.0;
          rhs = SnSouRhs(pack(b, iun + n, k, j, i), rdt, src, kap, ax, ay, az, ux, uy, uz,
                         ux2, uy2, uz2);
          dia = SnSouDiagonal(rdt, chi_a + chi_s, kap, ax, ay, az);
        } else if (!diamond) {
          rhs = SnRhs(pack(b, iun + n, k, j, i), rdt, src, ax, ay, az, ux, uy, uz);
          dia = SnDiagonal(rdt, chi_a + chi_s, ax, ay, az);
        } else {
          // Incoming faces, same upwind convention as above (di = -1 for n_x > 0).
          const Real fxi = pack(b, ifx + n, k, j, i + di);
          const Real fyi = two_d ? pack(b, ify + n, k, j + dj, i) : 0.0;
          const Real fzi = three_d ? pack(b, ifz + n, k + dk, j, i) : 0.0;
          rhs = SnWddRhs(pack(b, iun + n, k, j, i), rdt, src, rw, ax, ay, az, fxi, fyi, fzi);
          dia = SnWddDiagonal(rdt, chi_a + chi_s, rw, ax, ay, az);
        }
        const Real dx_ = dia * pack(b, iu + n, k, j, i);
        const Real rr = dx_ - rhs;
        if (ist >= 0) pack(b, ist + n - nlo, k, j, i) = rr;
        sres += std::abs(rr);
        snrm += std::abs(rhs);
        // Largest INTERMEDIATE magnitude. The residual is dia*x - rhs, a difference of two
        // large and nearly equal numbers whenever streaming dominates the diagonal, so the
        // round-off floor of any two orderings of this expression scales with |dia*x|, NOT
        // with the residual. The operator gate needs this as its denominator; normalising by
        // the residual made the gate's tolerance a function of the problem's conditioning.
        stmax = std::max(stmax, std::max(std::abs(dx_), std::abs(rhs)));
      },
      Kokkos::Sum<Real>(lres), Kokkos::Sum<Real>(lnrm), Kokkos::Max<Real>(ltmax));
  *res = lres;
  *nrm = lnrm;
  if (tmax != nullptr) *tmax = ltmax;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief One red-black (or Jacobi) transport sweep over one colour. Returns the residual
//!        contribution through `res`/`nrm` so no previous-iterate array is needed.
//!
//! Flat parallelism over (b, n, k, j, i): per-thread state is ~6 scalars, and consecutive
//! threads read consecutive i, so every access coalesces. Fluxes are FUSED -- the upwind
//! neighbour is read directly rather than stored.
TaskStatus SnSweepColour(MeshData<Real> *md, const Real dt, const int colour, Real *res,
                         Real *nrm) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const DeviceAngles ang = pkg->Param<DeviceAngles>("sn_angles");
  const Real chat = pkg->Param<Real>("chat");
  const int n_group = pkg->Param<RadGroups>("groups").n_group;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = (pmb->pmy_mesh->ndim >= 2);
  const bool three_d = (pmb->pmy_mesh->ndim >= 3);

  // sn.chia + sn.chis, NOT rad.chi_g. The sweep used to take its extinction from the cached
  // rad.chi_g (Rosseland + scattering), which cannot be split into the absorbing part that
  // exchanges energy with the gas and the scattering part that does not. Both pieces are now
  // seeded explicitly, so the sweep, the matter coupling and the conservation gate all read
  // the SAME decomposition -- which is what makes G9's drift a readout of the solver tolerance
  // rather than of two opacity evaluations disagreeing.
  std::vector<std::string> names{USolName(), UnName(),   SrcName(),
                                 ChiaName(), ChisName(), MomName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int iu = imap[USolName()].first;
  const int iun = imap[UnName()].first;
  const int isrc = imap[SrcName()].first;
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int imom = imap[MomName()].first;

  const int nang = p.nang;
  const int nfa = p.nfa;
  auto nxv = ang.nx;
  auto nyv = ang.ny;
  auto nzv = ang.nz;
  const Real rdt = 1.0 / (chat * dt);

  Real lres = 0.0, lnrm = 0.0;
  // DEFAULT_LOOP_PATTERN, not an explicit MDRange: on the GPU build it is a flattened 1D range
  // (consecutive threads take consecutive i, i.e. coalesced); a 5D MDRange tiles and ran this kernel
  // at 90 GB/s (job 1698812). Do not override the default without a measurement.
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::Sweep", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, 0, nfa - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                    Real &sres, Real &snrm) {
        // Red-black colouring on the cell parity. Jacobi (colour < 0) updates every cell from
        // the previous iterate; it is a diagnostic that reproduces the measured 32x wall.
        if (colour >= 0 && (((i + j + k) & 1) != colour)) return;
        const int g = n / nang;
        const int a = n - g * nang;
        const auto &co = pack.GetCoords(b);
        const Real ax = std::abs(nxv(a)) / co.Dxc<1>(k, j, i);
        const Real ay = two_d ? std::abs(nyv(a)) / co.Dxc<2>(k, j, i) : 0.0;
        const Real az = three_d ? std::abs(nzv(a)) / co.Dxc<3>(k, j, i) : 0.0;
        // Upwind neighbour: i-1 when Omega_d > 0, i+1 when Omega_d < 0.
        const int di = (nxv(a) > 0.0) ? -1 : 1;
        const int dj = (nyv(a) > 0.0) ? -1 : 1;
        const int dk = (nzv(a) > 0.0) ? -1 : 1;
        const Real ux = pack(b, iu + n, k, j, i + di);
        const Real uy = two_d ? pack(b, iu + n, k, j + dj, i) : 0.0;
        const Real uz = three_d ? pack(b, iu + n, k + dk, j, i) : 0.0;
        // Clamped, not asserted. The M-matrix property that makes this iteration converge
        // at all -- diag - sum|offdiag| = rdt + chi -- needs chi >= 0, and positivity of I
        // needs it too (through both the diagonal and src). chi comes from the opacity chain,
        // which is tabulated and interpolated, so a small negative excursion is a plausible
        // input rather than an impossible one. A device-side assert would abort a production
        // run; clamping degrades to pure scattering-free streaming, which is the physically
        // right limit for zero opacity.
        const Real chi_a = std::max(0.0, pack(b, ichia + g, k, j, i));
        const Real chi_s = std::max(0.0, pack(b, ichis + g, k, j, i));
        const Real chi = chi_a + chi_s;
        // sn.src holds the EMISSIVITY (enters the RHS directly); S = eta + chi_s*J with the scattering
        // half lagged at the previous iteration's J (classical source iteration, Richardson path).
        const Real src = pack(b, isrc + g, k, j, i)
                       + chi_s * pack(b, imom + 4 * g + 0, k, j, i);
        const Real inew = SnUpdateCell(pack(b, iun + n, k, j, i), rdt, chi, src, ax, ay, az,
                                       ux, uy, uz);
        const Real iold = pack(b, iu + n, k, j, i);
        pack(b, iu + n, k, j, i) = inew;
        sres += std::abs(inew - iold);
        snrm += std::abs(inew);
      },
      Kokkos::Sum<Real>(lres), Kokkos::Sum<Real>(lnrm));
  *res = lres;
  *nrm = lnrm;
  return TaskStatus::complete;
}

} // namespace Sn
} // namespace Radiation

//========================================================================================
// Driver
//========================================================================================
namespace Radiation {
namespace Sn {

//! Per-partition iteration state. Heap-allocated and captured by shared_ptr so it outlives
//! task-list construction; tasks run in order, so a host flag set by iteration m is visible
//! to iteration m+1.
struct IterState {
  Real res0{0.0};      //!< residual norm of the first sweep, the relative-tolerance anchor
  Real res{0.0};
  Real nrm{0.0};
  Real rel{0.0};       //!< last relative increment, so the report says what it converged TO
  Real nclamp{0.0};    //!< face set-to-zero fixups over the solve; 0 = exactly conservative
  //! Under sn_spatial = sou the same counter holds NEGATIVE-INTENSITY CELL UPDATES, which are
  //! counted and not clamped -- see the note in SnSweepOrdered. The report must say which.
  bool sou{false};
  int iters{0};
  bool converged{false};
  //! Global (res, nrm) pair. parthenon's AllReduce owns a DUPLICATED communicator, so it
  //! cannot cross-match with any other collective in flight -- which is what made the
  //! hand-rolled MPI_Allreduce(MPI_COMM_WORLD) in the first version a latent deadlock.
  parthenon::AllReduce<std::vector<Real>> gsum;
};

//----------------------------------------------------------------------------------------
//! \brief tau_cell DIAGNOSTIC: the distribution of chi*dx over the whole mesh.
//!
//! tau_cell = (chi_a + chi_s) * dx_max per cell, max over groups, histogrammed by volume and by mass
//! (sn_tau_report > 0 only; read-only). It grades step differencing's error (1 + 0.63 tau_cell) and
//! the Krylov payoff (large at small tau). Measured at production (job 1730252): 5 % of the MASS
//! above tau_cell = 1, 0.04 % of the volume.
constexpr int kTauNBin = 24;      //!< 0.25 dex each, spanning 1e-4 .. 1e2
constexpr Real kTauLo = -4.0;     //!< log10 of the first bin edge
constexpr Real kTauDex = 0.25;

//! Reduction state for the tau_cell report. Per partition, like IterState.
struct TauState {
  //! 2*(kTauNBin+2) slots: VOLUME-weighted (under, bins, over) then MASS-weighted.
  parthenon::AllReduce<std::vector<Real>> hist;
  bool active{false};   //!< did THIS call fill the histogram, or is it a skipped cycle?
  int solve{0};         //!< which Sn solve this histogram describes, for the printed line
  int partition{0};
};

//----------------------------------------------------------------------------------------
//! \brief Fill this rank's tau_cell histogram. No-op on cycles between reports.
TaskStatus SnTauHistogram(MeshData<Real> *md, std::shared_ptr<TauState> ts) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const int every = p.tau_report;
  ts->hist.val.assign(2 * (kTauNBin + 2), 0.0);
  // ts->active was decided at task-list construction from the package-held per-partition
  // counter (see AddSnTransport); `every` is re-read here only to make the no-op explicit.
  if (every <= 0 || !ts->active) return TaskStatus::complete;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = (pmb->pmy_mesh->ndim >= 2);
  const bool three_d = (pmb->pmy_mesh->ndim >= 3);
  std::vector<std::string> names{ChiaName(), ChisName(), "cons"};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ichia = imap[ChiaName()].first;
  const int ichis = imap[ChisName()].first;
  const int ic = imap["cons"].first;
  const int n_group = p.n_group;

  // One reduction per bin, two reducers per pass (volume and mass). NBIN kernel launches on a
  // diagnostic cycle is the simple, provably race-free shape; a ScatterView histogram would be
  // one launch and one more thing to get wrong in a routine whose entire purpose is to be
  // trustworthy. Slots 0..NBIN+1 hold volume, NBIN+2..2*NBIN+3 hold mass.
  for (int nb = 0; nb < kTauNBin + 2; ++nb) {
    const Real elo = kTauLo + (nb - 1) * kTauDex;   // nb = 0 is the underflow slot
    const Real ehi = elo + kTauDex;
    const int slot = nb;
    Real cv = 0.0, cm = 0.0;
    parthenon::par_reduce(
        DEFAULT_LOOP_PATTERN, "Sn::TauHist", parthenon::DevExecSpace(), 0,
        pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &sv, Real &sm) {
          const auto &co = pack.GetCoords(b);
          const Real dx1 = co.Dxc<1>(k, j, i);
          const Real dx2 = two_d ? co.Dxc<2>(k, j, i) : dx1;
          const Real dx3 = three_d ? co.Dxc<3>(k, j, i) : dx1;
          const Real dx = std::max(dx1, std::max(dx2, dx3));
          // MAXIMUM over groups: the group that binds this cell's spatial accuracy. A mean or a
          // per-(cell,group) percentile is dominated by the empty/thin groups -- see the header.
          Real tau = 0.0;
          for (int g = 0; g < n_group; ++g) {
            const Real chi = std::max(0.0, pack(b, ichia + g, k, j, i))
                           + std::max(0.0, pack(b, ichis + g, k, j, i));
            tau = std::max(tau, chi * dx);
          }
          // log10(0) is -inf; the comparison below puts it in the underflow slot rather than
          // producing a NaN that would silently poison the whole histogram.
          const Real lt = (tau > 0.0) ? std::log10(tau) : -1.0e300;
          bool in;
          if (slot == 0) in = (lt < kTauLo);
          else if (slot == kTauNBin + 1) in = (lt >= kTauLo + kTauNBin * kTauDex);
          else in = (lt >= elo && lt < ehi);
          if (in) {
            const Real vol = dx1 * (two_d ? dx2 : 1.0) * (three_d ? dx3 : 1.0);
            sv += vol;
            sm += vol * std::max(0.0, pack(b, ic + IDN, k, j, i));
          }
        },
        Kokkos::Sum<Real>(cv), Kokkos::Sum<Real>(cm));
    ts->hist.val[slot] += cv;
    ts->hist.val[kTauNBin + 2 + slot] += cm;
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Print the reduced tau_cell distribution and the step-differencing bias it implies.
TaskStatus SnTauReport(std::shared_ptr<TauState> ts, const Real dd_w, const bool diamond,
                       const bool sou, const Real sou_kappa) {
  // ONE LINE PER PARTITION, and the partition index is printed. Each partition owns its own
  // TauState and its own once_per_region reduction, so a multi-partition run (nothing sets
  // parthenon/mesh/pack_size today, but nothing stops it) would otherwise emit a line whose
  // counts silently covered only part of the mesh.
  if (!ts->active) return TaskStatus::complete;
  if (parthenon::Globals::my_rank != 0) return TaskStatus::complete;
  const auto &h = ts->hist.val;
  const int NB = kTauNBin + 2;
  // Percentile by linear interpolation on the cumulative weight, in log10(tau). The underflow
  // slot has no lower edge, so a percentile landing in it is reported at the first edge and is
  // an UPPER BOUND, not a value.
  auto pct = [&](const int off, const Real tot, const Real q) {
    const Real target = q * tot;
    Real cum = 0.0;
    for (int nb = 0; nb < NB; ++nb) {
      if (cum + h[off + nb] >= target) {
        if (nb == 0) return std::pow(10.0, kTauLo);
        if (nb == NB - 1) return std::pow(10.0, kTauLo + kTauNBin * kTauDex);
        const Real f = (h[off + nb] > 0.0) ? (target - cum) / h[off + nb] : 0.0;
        return std::pow(10.0, kTauLo + (nb - 1 + f) * kTauDex);
      }
      cum += h[off + nb];
    }
    return std::pow(10.0, kTauLo + kTauNBin * kTauDex);
  };
  // Weight fraction above a tau threshold. INTERPOLATED inside the straddling bin, not rounded
  // up to the next bin edge: at 0.25 dex per bin, rounding would silently report "above 0.2" as
  // "above 0.316", a factor 1.6 in the threshold of a number that is meant to be quoted.
  auto frac_above = [&](const int off, const Real tot, const Real tau) {
    if (tot <= 0.0) return 0.0;
    const Real lt = std::log10(tau);
    const Real pos = (lt - kTauLo) / kTauDex;      // in bin-widths above the first edge
    const int nb = static_cast<int>(std::floor(pos)) + 1;   // slot straddling lt
    Real a = 0.0;
    for (int m = std::max(0, nb + 1); m < NB; ++m) a += h[off + m];
    if (nb >= 1 && nb < NB - 1) a += h[off + nb] * (1.0 - (pos - std::floor(pos)));
    else if (nb < 1) a = tot;                       // threshold below the histogram's floor
    return 100.0 * a / tot;
  };
  Real totv = 0.0, totm = 0.0;
  for (int nb = 0; nb < NB; ++nb) { totv += h[nb]; totm += h[NB + nb]; }
  if (totv <= 0.0) return TaskStatus::complete;
  // 1 + 0.63*tau_cell is gate K4's MEASURED fit for step differencing over tau_cell = 0.2..3,
  // not a theoretical coefficient, and it is an EXTRAPOLATION outside that range -- quoted as
  // such so nobody re-derives it from a truncation analysis and gets a different number, and so
  // nobody reads a percent figure at tau_cell = 100 as if it had been measured.
  auto bias = [](const Real t) { return 100.0 * 0.63 * t; };
  std::cout << "[SN/tau] solve " << ts->solve << " part " << ts->partition
            << ": tau_cell = max_g chi_g * max(dx), weighted\n"
            << "         by VOLUME  p50 " << pct(0, totv, 0.50) << "  p90 "
            << pct(0, totv, 0.90) << "  p99 " << pct(0, totv, 0.99)
            << "   (" << frac_above(0, totv, 0.2) << " % above 0.2, "
            << frac_above(0, totv, 1.0) << " % above 1)\n"
            << "         by MASS    p50 " << pct(NB, totm, 0.50) << "  p90 "
            << pct(NB, totm, 0.90) << "  p99 " << pct(NB, totm, 0.99)
            << "   (" << frac_above(NB, totm, 0.2) << " % above 0.2, "
            << frac_above(NB, totm, 1.0) << " % above 1)" << std::endl;
  if (sou) {
    std::cout << "[SN/tau] sn_spatial = sou at kappa = " << sou_kappa
              << ": K4 measured 0.974/0.973/0.974/0.975 at tau_cell = 0.2/0.5/1.0/3.0 -- a FLAT "
                 "-2.6 % with no tau_cell trend, against step's +11.6/+32/+64/+161 %. The "
                 "implied excess above is therefore NOT the error bar on this run; the "
                 "residual is the flat offset. Cost at the production epoch: 1.6-2.0x the "
                 "BiCGSTAB iterations and 1.62x the wall against step, same node count."
              << std::endl;
  } else if (!diamond) {
    std::cout << "[SN/tau] sn_spatial = step: implied EXCESS radiative diffusion, K4 fit "
                 "1 + 0.63*tau_cell (MEASURED over tau_cell = 0.2..3, extrapolated outside): "
                 "at the mass-weighted p50 +" << bias(pct(NB, totm, 0.50)) << " %, at p90 +"
              << bias(pct(NB, totm, 0.90)) << " %, at p99 +" << bias(pct(NB, totm, 0.99))
              << " %. This is a DISCRETISATION error: sn_rel_tol cannot see it and tightening "
                 "it will not move it. The fix is sn_spatial = sou (second-order upwind), "
                 "which runs under BOTH solvers and adds no memory; sn_spatial = wdd is more "
                 "accurate near tau_cell = 1 but requires sn_solver = richardson and roughly "
                 "doubles the node count." << std::endl;
  } else {
    std::cout << "[SN/tau] sn_spatial = wdd at w = " << dd_w
              << ": K4 measured 0.983/0.995/1.016/1.115 at tau_cell = 0.2/0.5/1.0/3.0. Check "
                 "the face-fixup count on the [SN] converged line: a non-zero count means "
                 "flux telescoping is broken in that many face updates and the solve is NOT "
                 "exactly conservative." << std::endl;
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Copy this rank's accumulated (res, nrm) into the reduction buffer.
TaskStatus SnStageReduction(std::shared_ptr<IterState> st) {
  st->gsum.val.resize(2);
  st->gsum.val[0] = st->res;
  st->gsum.val[1] = st->nrm;
  st->res = 0.0;
  st->nrm = 0.0;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief SnComputeMoments, skipped once the iteration has converged.
//!
//! The unrolled task list cannot be shortened at runtime, so every task past convergence must
//! decline for itself. This one and SnIterate do; the two AddBoundaryExchangeTasks calls
//! CANNOT (see the cost note in AddSnTransport), which is why sn_max_iterations is set near
//! the expected count rather than far above it.
TaskStatus SnMomentsIfActive(MeshData<Real> *md, std::shared_ptr<IterState> st) {
  if (st->converged) return TaskStatus::complete;
  return SnComputeMoments(md);
}

//----------------------------------------------------------------------------------------
//! \brief Replace the sweeps' increment accumulation with the TRUE RESIDUAL, when asked.
//!
//! Runs AFTER both colours and after SnComputeMoments, so the J the scattering source needs is
//! the current one. It OVERWRITES st->res / st->nrm rather than adding to them: the sweeps
//! accumulate the increment unconditionally (it is nearly free, being already in the sweep's
//! reduction) and this discards it. IterState is per-partition, so an overwrite here has the
//! same scope the += had.
TaskStatus SnResidualIfActive(MeshData<Real> *md, const Real dt,
                              std::shared_ptr<IterState> st) {
  if (st->converged) return TaskStatus::complete;
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("radiation");
  if (!pkg->Param<SnParams>("sn_params").residual_criterion) return TaskStatus::complete;
  Real r = 0.0, n = 0.0;
  SnResidual(md, dt, &r, &n);
  st->res = r;
  st->nrm = n;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief One sweep + its global reduction + the convergence test.
//!
//! The Allreduce is per sweep and that is DELIBERATE. ~230 reductions of two scalars per hydro
//! step, latency-bound at 10-100 us, is ~0.02 s against a step of order 750 GPU-s -- four
//! orders of magnitude below the noise. It was flagged as a blocker during design and the
//! arithmetic retired it. Do not "optimise" it into a strided check and reintroduce a
//! convergence bug for nothing.
TaskStatus SnIterate(MeshData<Real> *md, const Real dt, const int colour,
                     std::shared_ptr<IterState> st) {
  Real res = 0.0, nrm = 0.0;
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("radiation");
  if (pkg->Param<SnParams>("sn_params").ordered_sweep) {
    // One ordered pass replaces the red/black pair, so only colour 0 does work; the second
    // call in the red-black task pair returns immediately rather than sweeping twice.
    if (colour <= 0) SnSweepOrdered(md, dt, &res, &nrm, &st->nclamp);
  } else {
    SnSweepColour(md, dt, colour, &res, &nrm);
  }
  st->res += res;
  st->nrm += nrm;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Completion task: decide whether the enclosing sublist iterates again.
//!
//! Tagged TaskQualifier::completion | TaskQualifier::global_sync. Parthenon Iallreduces the
//! returned status with MPI_MAX, and TaskStatus is { complete=0, incomplete=1, iterate=2 },
//! so if ANY rank still wants to iterate then EVERY rank iterates. That is what makes the
//! decision collective, and it cannot deadlock by construction.
//!
//! Criterion: GLOBAL sum ratio via parthenon's AllReduce (a per-rank test is a different, stricter
//! criterion); global_sync makes the decision collective. It tests the iterate increment unless
//! sn_criterion = residual (production). `iters >= max_it` fires on the last permitted iteration.
TaskStatus SnCheckConvergence(std::shared_ptr<IterState> st, const Real rel_tol,
                              const int max_it, const int verbose) {
  ++st->iters;
  // Global sums, filled by SnStageReduction + AllReduce::CheckReduce earlier in the sublist.
  const Real gres = st->gsum.val[0];
  const Real gnrm = st->gsum.val[1];
  const Real rel = (gnrm > 0.0) ? gres / gnrm : 0.0;
  if (st->iters == 1) st->res0 = rel;
  st->rel = rel;
  if (verbose > 0 && parthenon::Globals::my_rank == 0 &&
      (st->iters % verbose == 0 || st->iters == 1)) {
    std::cout << "[SN/iter] " << st->iters << " rel " << rel << " (res " << gres << " nrm "
              << gnrm << ")" << std::endl;
  }
  if (rel < rel_tol) {
    st->converged = true;
    return TaskStatus::complete;
  }
  if (st->iters >= max_it) {
    st->converged = false;   // hit the cap: what follows is a FLOOR, not a convergence result
    return TaskStatus::complete;
  }
  return TaskStatus::iterate;
}

//----------------------------------------------------------------------------------------
//! \brief Report the iteration count, and SHOUT if the cap bound.
//!
//! A capped solve measures the cap. m1_thermal's shipped max_iterations = 200 nearly produced
//! the verdict "does not converge" when the real count was ~900 -- this exists so that failure
//! mode cannot recur silently.
TaskStatus SnReport(std::shared_ptr<IterState> st, const int max_it, const Real rel_tol) {
  if (parthenon::Globals::my_rank != 0) return TaskStatus::complete;
  if (!st->converged) {
    std::cout << "### WARNING Sn: NOT CONVERGED after " << st->iters << " iterations "
              << "(cap " << max_it << "). rel = " << st->rel << " vs rel_tol = " << rel_tol
              << ", first-sweep rel = " << st->res0
              << ". THIS IS A CAP, NOT A CONVERGENCE RESULT -- raise "
              << "radiation/sn_max_iterations before reading anything into it. If the count "
              << "rises with the cap the solve is slow; if it does not, it is STALLED, which "
              << "is what killed implicit M1." << std::endl;
  } else {
    std::cout << "[SN] converged in " << st->iters << " iterations (rel " << st->rel << ")";
    if (st->nclamp > 0.0 && st->sou)
      std::cout << " [SOU negative cell updates " << st->nclamp << " -- COUNTED, NOT CLAMPED: "
                   "no fixup is applied under sou, so the operator stays exactly linear and "
                   "the sweep, SnResidual and SnApplyA remain the same operator]";
    else if (st->nclamp > 0.0)
      std::cout << " [WDD face fixups " << st->nclamp << " -- flux telescoping broken in that "
                   "many face updates, so this solve is NOT exactly conservative]";
    std::cout << std::endl;
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//! \brief L1 norm contribution of one field, accumulated on this rank.
TaskStatus SnVecAbs(MeshData<Real> *md, const std::string &nm, Real *out) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{nm}, imap);
  const int nc = SnPackWidth(imap, nm);
  Real la = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::VecAbs", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0,
      nc - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &a) {
        a += std::abs(pack(b, n, k, j, i));
      },
      Kokkos::Sum<Real>(la));
  *out = la;
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \brief Per-cell comparison support: max|a+b| and max|a| over the interior. MAX, not a sum,
//!        and that distinction is the entire point -- see SnKrVerifyOp.
TaskStatus SnVecMaxPair(MeshData<Real> *md, const std::string &n1, const std::string &n2,
                        Real *mdiff, Real *mmax) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{n1, n2};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int i1 = imap[n1].first, i2 = imap[n2].first;
  const int nc = SnPackWidth(imap, n1);
  PARTHENON_REQUIRE(SnPackWidth(imap, n2) == nc, "SnVecMaxPair operand widths differ.");
  Real md1 = 0.0, mm1 = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::VecMaxPair", parthenon::DevExecSpace(), 0,
      pack.GetDim(5) - 1, 0, nc - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &d,
                    Real &m) {
        const Real av = pack(b, i1 + n, k, j, i), cv = pack(b, i2 + n, k, j, i);
        d = std::max(d, std::abs(av + cv));   // A x - b  and  b - A x  must cancel EXACTLY
        m = std::max(m, std::abs(av));
      },
      Kokkos::Max<Real>(md1), Kokkos::Max<Real>(mm1));
  *mdiff = md1;
  *mmax = mm1;
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \brief STAGE-A GATE: prove SnApplyA is the same operator SnResidual measures.
//!
//! A Krylov solver does not care whether A is the operator you meant -- it will drive
//! ||b - A x|| to zero for whatever A it is handed, print a beautiful convergence history, and
//! return the solution to the wrong problem. The implicit-M1 work lost weeks to exactly this
//! shape (SetDiagonal assumed Rusanov while the flux was HLL). So the operator is checked
//! against the INDEPENDENTLY WRITTEN residual kernel the Richardson path was validated with,
//! on the actual state, at runtime:
//!
//!     sum|b - A x| from SnApplyA(mode 1)  ==  sum|A x - b| from SnResidual.
//!
//! SnResidual folds the upwind terms into b; SnApplyA folds them into A x with the opposite
//! sign. Both the stencil and the diagonal have to be transcribed correctly for these to agree.
//!
//! No invariant on ||b||: SnResidual's b includes the upwind terms, the Krylov b does not.
//! On a UNIFORM field the streaming terms cancel and this gate passes for any stencil -- the gate
//! deck must carry spatial structure.
TaskStatus SnKrVerifyOp(MeshData<Real> *md, const Real dt, const int g) {
  Real res = 0.0, nrm = 0.0, mdiff = 0.0, mmax = 0.0, tscale = 0.0;
  SnResidual(md, dt, &res, &nrm, KtName(), &tscale, g);  // k_t = A x - b, per cell, group g
  SnVecMaxPair(md, KtName(), KrName(), &mdiff, &mmax);  // k_r = b - A x from SnApplyA
  // Denominator is the largest TERM, not the largest residual -- see the note in SnResidual.
  const Real den = std::max(tscale, mmax);
  const Real rel = (den > 0.0) ? mdiff / den : 0.0;
  const bool ok = (rel < 1.0e-13);
  std::cout << "[SN/verify-op] group " << g << " per-cell max|r_res + r_op| = " << mdiff
            << "  term scale = " << den << "  rel = " << rel
            << (ok ? "   PASS" : "   *** FAIL ***") << std::endl;
  PARTHENON_REQUIRE(ok, "SnApplyA does not reproduce SnResidual cell by cell: the Krylov "
                        "solver would converge on a different operator than the validated one.");
  return TaskStatus::complete;
}
//----------------------------------------------------------------------------------------
//! \brief R-F16: STAGE-A GATE for the COUPLED (Fleck & Cummings, sn_fc_scatter) solve. SnKrVerifyOp is wired only
//!        into the per-group path, so with sn_fc_scatter the coupled operator A_FC = sum_g A_g + C ran with no
//!        runtime check (R0 K1/K3/K5 regression, 2026-10-09). Same contract, all groups at once:
//!            k_c,r = b - sum_g SnApplyA_g(x) - SnFCCross(x)                 (the Krylov setup, already formed)
//!            k_c,t = SnResidual(x) + SnFCCrossMom(x)  = A_FC x - b          (two independent transcriptions)
//!        must cancel cell by cell. SnFCCrossMom reads J from sn.mom and is written from the formula, so a wrong
//!        index, sign or weight in SnFCCross FAILS here (reusing SnFCCross would make that half a tautology).
//!        Also checks sum_g C_g = 0 (the redistribution conserves energy). Runs with radiation/sn_verify_operator.
TaskStatus SnFCVerifyOp(MeshData<Real> *md, const Real dt) {
  // (1) the cross term alone: SnFCCross(x) - SnFCCrossMom(x) on a zeroed scratch, against the term scale (1-f) X.
  //     (Needed on its own: in (2) the scale is set by the 1/(chat dt) terms, which can hide an error in C.)
  Real csum = 0.0, cscale = 0.0, d2 = 0.0, cdiff = 0.0;
  SnZeroAll(md, KctName());
  SnFCCross(md, USolName(), KctName(), 1.0);
  SnFCCrossMom(md, KctName(), -1.0, &csum, &cscale);
  SnVecMaxPair(md, KctName(), KctName(), &d2, &cdiff);
  const Real crel = (cscale > 0.0) ? cdiff / cscale : 0.0;   // two transcriptions of C
  const Real srel = (cscale > 0.0) ? csum / cscale : 0.0;    // sum_g C_g = 0
  // (2) the whole coupled operator: k_c,t = SnResidual(x) + C(x) must cancel the Krylov setup k_c,r = b - A_FC x.
  Real res = 0.0, nrm = 0.0, tscale = 0.0, mdiff = 0.0, mmax = 0.0, cs2 = 0.0, cc2 = 0.0;
  SnResidual(md, dt, &res, &nrm, KctName(), &tscale, -1);   // k_c,t = A_legacy x - b, every group
  SnFCCrossMom(md, KctName(), 1.0, &cs2, &cc2);            // + C(x), second transcription
  SnVecMaxPair(md, KctName(), KcrName(), &mdiff, &mmax);
  const Real den = std::max(tscale, mmax);
  const Real rel = (den > 0.0) ? mdiff / den : 0.0;
  const bool ok = (rel < 1.0e-13) && (crel < 1.0e-12) && (srel < 1.0e-12);
  std::cout << "[SN/verify-op] coupled (all groups) per-cell max|r_res + r_op| = " << mdiff
            << "  term scale = " << den << "  rel = " << rel << " | cross term: two transcriptions " << crel
            << "  sum_g C_g " << srel << "  (of (1-f)X = " << cscale << ")"
            << (ok ? "   PASS" : "   *** FAIL ***") << std::endl;
  PARTHENON_REQUIRE(ok, "The coupled Krylov operator (sum_g SnApplyA + SnFCCross) does not reproduce SnResidual + "
                        "the independent cross term cell by cell, or the two cross-term transcriptions differ, or "
                        "sum_g C_g != 0.");
  return TaskStatus::complete;
}
//----------------------------------------------------------------------------------------
//! \brief State of one BiCGSTAB solve.
struct KrylovState {
  Real rho{1.0}, rho_old{1.0}, alpha{1.0}, omega{1.0}, beta{0.0};
  Real bnrm{0.0}, rnrm{0.0}, rel{0.0}, res0{0.0};
  int iters{0};
  bool converged{false};
  //! Non-zero on an algorithmic breakdown: (r0,v) or (t,t) underflowed. BiCGSTAB CAN break down
  //! -- it is not a minimising method -- and the honest response is to stop and say so, not to
  //! divide by zero and report a NaN residual as convergence.
  int breakdown{0};
  //! radiation/sn_krylov_dot_scale: multiplier applied to BOTH operands of every Krylov dot
  //! (1/||b||_1 of the group being solved; exactly 1.0 when the key is off).
  Real dscale{1.0};
  Real l[3]{0.0, 0.0, 0.0};                       //!< per-rank partial sums awaiting reduction
  parthenon::AllReduce<std::vector<Real>> r1, r2, r3;
  //! Group currently being solved. A is block-diagonal in the group index (see the field
  //! registration in Initialize), so the solves are sequential and ONE state serves all of them:
  //! every AllReduce does a collective MPI_Comm_dup on construction and an MPI_Comm_disconnect
  //! on destruction, and this object is built once per partition per STEP, so a state per group
  //! would triple that collective traffic every step to buy nothing.
  int group{0};
  //! R-F12 gray DSA (radiation_sn_graydsa.cpp) needs the mesh and partition for its inner diffusion solve.
  parthenon::Mesh *pmesh{nullptr};
  int partition{0};
  //! What each group's solve reported, kept so the end-of-solve report and the trust check can
  //! speak about all of them after the last one has overwritten the live scalars.
  struct GroupResult {
    int iters{0};
    Real rel{0.0}, res0{0.0};
    bool converged{false};
    int breakdown{0};
    Real bnrm{0.0};        //!< ||b||_1 of this group (audit P-09 skip test)
    bool skipped{false};   //!< not iterated: negligible ||b|| (radiation/sn_skip_group_rel)
    bool restored{false};  //!< failed and restored to I^n (radiation/sn_fallback)
  };
  std::vector<GroupResult> per_group;
  //! audit P-09: this group is not iterated (every body kernel returns at once, SnKrCheck exits).
  bool skip{false};
  //! audit P-06/P-09: per-group flag handed to SnMatterCoupling (1 = no exchange this step).
  std::shared_ptr<std::vector<int>> coupling_off;
  parthenon::StateDescriptor *pkg{nullptr};   // for the DSA inner-solve tally in SnKrReport
  //! R-F6: radiation/sn_local_tol > 0 -> r1.val[2] carries the global count of cells violating the local criterion.
  bool local_on{false};
  //! R-F3 coupled solve: per-group ||b||_1 partials (setup) and the frozen-group mask (sn_skip_group_rel).
  std::vector<Real> lg;
  std::vector<int> fc_skip;
  int nang_fc{0};   // ordinates per group, for the lean zeroing (radiation_sn_fastops.cpp)
  parthenon::AllReduce<std::vector<Real>> rg;
};

//! Clear the recurrences for a new group's solve.
TaskStatus SnKrReset(std::shared_ptr<KrylovState> st, const int g) {
  st->group = g;
  st->rho = st->rho_old = st->alpha = st->omega = 1.0;
  st->beta = 0.0;
  st->bnrm = st->rnrm = st->rel = st->res0 = 0.0;
  st->iters = 0;
  st->converged = false;
  st->breakdown = 0;
  st->dscale = 1.0;
  st->skip = false;
  for (int i = 0; i < 3; ++i) st->l[i] = 0.0;
  return TaskStatus::complete;
}

//! audit P-09 (radiation/sn_skip_group_rel): after the setup reduction, skip group g > 0 when its
//! ||b||_1 is below `rel` x the largest earlier group's. Global scalars only -> rank-uniform.
TaskStatus SnKrDecideSkip(std::shared_ptr<KrylovState> st, const int g, const Real rel) {
  st->skip = false;
  if (rel <= 0.0 || g == 0) return TaskStatus::complete;
  Real bmax = 0.0;
  for (int h = 0; h < g; ++h) bmax = std::max(bmax, st->per_group[h].bnrm);
  if (bmax > 0.0 && std::isfinite(st->bnrm) && st->bnrm < rel * bmax) st->skip = true;
  return TaskStatus::complete;
}

//! audit P-06/P-09: after group g's solve. A skipped group, or (radiation/sn_fallback) a group whose
//! solve broke down / went non-finite / stopped above sn_fallback_rel, gets its start-of-step
//! intensities back from sn.un and is excluded from the matter coupling. Decided from global
//! scalars, so every partition and rank takes the same branch.
TaskStatus SnKrFallback(MeshData<Real> *md, std::shared_ptr<KrylovState> st, const int g,
                        const bool fallback, const Real fallback_rel) {
  auto &gr = st->per_group[g];
  const bool failed = fallback && !gr.skipped &&
                      (!std::isfinite(gr.rel) || (!gr.converged && !(gr.rel < fallback_rel)));
  gr.restored = failed;
  if (st->coupling_off) (*st->coupling_off)[g] = (gr.skipped || failed) ? 1 : 0;
  if (!failed) return TaskStatus::complete;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{USolName(), UnName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  const int iu = imap[USolName()].first + g * nang;
  const int iun = imap[UnName()].first + g * nang;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::KrFallback", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0,
      nang - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) {
        pack(b, iu + a, k, j, i) = pack(b, iun + a, k, j, i);
      });
  return TaskStatus::complete;
}

//! A body task that becomes a no-op while the current group is skipped (audit P-09).
template <typename F, typename... Args>
TaskID AddUnlessSkipped(TaskList &tl, TaskID dep, std::shared_ptr<KrylovState> st, F f,
                        Args... args) {
  return tl.AddTask(dep, [=]() { return st->skip ? TaskStatus::complete : f(args...); });
}

//! Stash this group's outcome before the next group overwrites the live scalars.
TaskStatus SnKrRecord(std::shared_ptr<KrylovState> st, const int g) {
  if (g >= 0 && g < static_cast<int>(st->per_group.size()))
    st->per_group[g] = {st->iters, st->rel, st->res0, st->converged, st->breakdown,
                        st->bnrm, st->skip, false};
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief x(g, a) += c * y(a), the ONE BiCGSTAB update that mixes component widths.
//!
//! sn.u is nfa-wide and k_y is a nang-wide per-group work vector, so SnVecLC -- which now
//! requires all four operands to have the same width, precisely so a width slip cannot pass
//! silently -- cannot express it. Interior only: x's ghosts are refreshed by the exchange after
//! the solve.
TaskStatus SnKrAxpyX(MeshData<Real> *md, const int g, const Real c) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{USolName(), KyName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int nang = p.nang;
  const int ix = imap[USolName()].first + g * nang;
  const int iy = imap[KyName()].first;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::KrAxpyX", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0,
      nang - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) {
        pack(b, ix + a, k, j, i) += c * pack(b, iy + a, k, j, i);
      });
  return TaskStatus::complete;
}

//! Which of the five BiCGSTAB vector updates to apply. Coefficients come from KrylovState at
//! EXECUTION time, which is why these cannot use the constant-coefficient SnVecLC.
enum class KrUpd { p, s, xy, xz, r };

TaskStatus SnKrUpdate(MeshData<Real> *md, std::shared_ptr<KrylovState> st, const KrUpd w) {
  if (st->skip) return TaskStatus::complete;
  switch (w) {
    // p = r + beta*(p - omega*v). First pass has p = v = 0, so p = r as required.
    case KrUpd::p:
      return SnVecLC(md, 1.0, KrName(), st->beta, KpName(), -st->beta * st->omega, KvName(),
                     KpName());
    // s = r - alpha*v, IN PLACE in k_r. This is what removes the sixth work vector.
    // It is legal because r's pre-update value is never read again: the next iteration forms
    // p = r_new + beta*(p - omega*v) from the NEW r, and the only other consumer of the old r
    // was the (r0, r) dot that produced rho for THIS iteration, already taken.
    case KrUpd::s:
      return SnVecLC(md, 1.0, KrName(), -st->alpha, KvName(), 0.0, KvName(), KrName());
    case KrUpd::xy:  // x += alpha*y   -- done BEFORE k_y is reused for z
      return SnKrAxpyX(md, st->group, st->alpha);
    case KrUpd::xz:  // x += omega*z   -- k_y now holds z
      return SnKrAxpyX(md, st->group, st->omega);
    case KrUpd::r:  // r = s - omega*t, again in place: k_r still holds s
      return SnVecLC(md, 1.0, KrName(), -st->omega, KtName(), 0.0, KtName(), KrName());
  }
  return TaskStatus::complete;
}

//! Accumulate a dot product or an L1 norm into slot `slot` of this rank's partial sums.
TaskStatus SnKrDot(MeshData<Real> *md, const std::string &na, const std::string &nb,
                   std::shared_ptr<KrylovState> st, const int slot) {
  if (st->skip) return TaskStatus::complete;
  Real d = 0.0;
  SnVecDot(md, na, nb, &d, st->dscale);
  st->l[slot] += d;
  return TaskStatus::complete;
}
TaskStatus SnKrAbs(MeshData<Real> *md, const std::string &na,
                   std::shared_ptr<KrylovState> st, const int slot) {
  if (st->skip) return TaskStatus::complete;
  Real a = 0.0;
  SnVecAbs(md, na, &a);
  st->l[slot] += a;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Setup reduction: the convergence DENOMINATOR and the initial residual, both from
//!        SnResidual so `sn_rel_tol` means on the Krylov path exactly what it means on the
//!        Richardson path. Normalising by the Krylov ||b|| instead would silently change the
//!        tolerance by whatever ratio the upwind terms contribute (2.7x on the sn_k1 deck).
TaskStatus SnKrBnorm(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st) {
  Real res = 0.0, nrm = 0.0;
  SnResidual(md, dt, &res, &nrm, "", nullptr, st->group);
  st->l[0] += nrm;
  st->l[1] += res;
  return TaskStatus::complete;
}

//! R-F6 wrappers (kernels in radiation_sn_criterion.cpp).
TaskStatus SnKrBnormB(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st) {
  Real b = 0.0, r = 0.0;
  SnBnormB(md, dt, st->group, &b, &r);
  st->l[0] += b;
  st->l[1] += r;
  return TaskStatus::complete;
}
TaskStatus SnKrTrueB(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st) {
  Real res = 0.0, nrm = 0.0, b = 0.0;
  SnResidual(md, dt, &res, &nrm);
  SnBnormB(md, dt, -1, &b, nullptr);
  st->l[0] += res;
  st->l[1] += b;
  return TaskStatus::complete;
}
TaskStatus SnKrLocal(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st, const Real tol) {
  if (st->skip) return TaskStatus::complete;
  Real c = 0.0;
  SnLocalViolations(md, dt, st->group, tol, &c);
  st->l[2] += c;
  return TaskStatus::complete;
}

//! Move the per-rank partial sums into a reduction buffer and clear them. Same shape as
//! SnStageReduction, and for the same reason: each AllReduce owns a duplicated communicator, so
//! three of them can be in flight in one iteration without cross-matching.
TaskStatus SnKrStage(std::shared_ptr<KrylovState> st, const int which, const int nv) {
  auto *red = (which == 1) ? &st->r1 : (which == 2) ? &st->r2 : &st->r3;
  red->val.resize(nv);
  for (int i = 0; i < nv; ++i) red->val[i] = st->l[i];
  for (int i = 0; i < 3; ++i) st->l[i] = 0.0;
  return TaskStatus::complete;
}

//! Seed the recurrences from the initial reduce {||b||_1, ||r0||_1, (r0,r0)}.
TaskStatus SnKrInit(std::shared_ptr<KrylovState> st) {
  st->bnrm = st->r1.val[0];
  st->rnrm = st->r1.val[1];
  st->rho = st->r1.val[2];
  st->rho_old = 1.0;
  st->alpha = 1.0;
  st->omega = 1.0;
  st->res0 = (st->bnrm > 0.0) ? st->rnrm / st->bnrm : 0.0;
  return TaskStatus::complete;
}

//! radiation/sn_krylov_dot_scale setup, step 1: the norms are known (r1 = {||b||_1, ||r0||_1}),
//! so the dot scale can be fixed BEFORE the first dot product is formed.
TaskStatus SnKrSetScale(std::shared_ptr<KrylovState> st) {
  st->bnrm = st->r1.val[0];
  st->rnrm = st->r1.val[1];
  st->dscale = (st->bnrm > 0.0 && std::isfinite(st->bnrm)) ? 1.0 / st->bnrm : 1.0;
  return TaskStatus::complete;
}
//! step 2: seed the recurrences from the SCALED (r0, r0) in r3 and the norms saved in step 1.
TaskStatus SnKrInitScaled(std::shared_ptr<KrylovState> st) {
  st->rho = st->r3.val[0];
  st->rho_old = 1.0;
  st->alpha = 1.0;
  st->omega = 1.0;
  st->res0 = (st->bnrm > 0.0) ? st->rnrm / st->bnrm : 0.0;
  return TaskStatus::complete;
}

TaskStatus SnKrBeta(std::shared_ptr<KrylovState> st) {
  if (st->skip) return TaskStatus::complete;
  const Real den = st->rho_old * st->omega;
  if (std::abs(den) < std::numeric_limits<Real>::min() / std::numeric_limits<Real>::epsilon()) {
    st->breakdown = 1;
    st->beta = 0.0;
  } else {
    st->beta = (st->rho / st->rho_old) * (st->alpha / st->omega);
  }
  return TaskStatus::complete;
}

TaskStatus SnKrAlpha(std::shared_ptr<KrylovState> st) {
  if (st->skip) return TaskStatus::complete;
  const Real r0v = st->r2.val[0];
  if (r0v == 0.0) {
    st->breakdown = 2;
    st->alpha = 0.0;
  } else {
    st->alpha = st->rho / r0v;
  }
  return TaskStatus::complete;
}

TaskStatus SnKrOmega(std::shared_ptr<KrylovState> st) {
  if (st->skip) return TaskStatus::complete;
  const Real ts = st->r3.val[0], tt = st->r3.val[1];
  if (tt == 0.0) {
    // t == 0 means s was already an exact solution of M-preconditioned A z = s; omega = 0 is
    // the right answer, NOT a breakdown, but the iteration cannot continue past it because
    // beta's denominator carries omega. Flag it and let the completion task stop cleanly.
    st->breakdown = 3;
    st->omega = 0.0;
  } else {
    st->omega = ts / tt;
  }
  return TaskStatus::complete;
}

//! End-of-iteration: adopt the new rho and residual norm, then test.
TaskStatus SnKrCheck(std::shared_ptr<KrylovState> st, const Real rel_tol, const int max_it) {
  if (st->skip) {  // audit P-09: leave the sublist after its single (no-op) pass
    st->converged = true;
    st->rel = 0.0;
    return TaskStatus::complete;
  }
  ++st->iters;
  st->rho_old = st->rho;
  st->rho = st->r1.val[0];
  st->rnrm = st->r1.val[1];
  st->rel = (st->bnrm > 0.0) ? st->rnrm / st->bnrm : 0.0;
  if (st->pkg != nullptr && st->pkg->Param<bool>("sn_gray_dsa") && st->pkg->Param<bool>("sn_gd_debug") &&
      parthenon::Globals::my_rank == 0)   // R-F12 debug trace (radiation/sn_gray_dsa_debug)
    std::printf("[SN/gd-trace] it %d rel %.6e rho %.6e alpha %.6e omega %.6e\n", st->iters, st->rel, st->rho,
                st->alpha, st->omega);
  // R-F6(b): with the local criterion on, every cell must also pass (count of violators == 0).
  const bool local_ok = !st->local_on || (st->r1.val.size() > 2 && st->r1.val[2] == 0.0);
  if (st->rel < rel_tol && local_ok) {
    st->converged = true;
    return TaskStatus::complete;
  }
  // A breakdown or a NaN is a STOP, not a retry: continuing propagates the NaN into sn.u and
  // the writeback would publish it as a radiation field.
  if (st->breakdown != 0 || !std::isfinite(st->rel)) return TaskStatus::complete;
  if (st->iters >= max_it) {
    st->converged = false;
    return TaskStatus::complete;
  }
  return TaskStatus::iterate;
}

//! \brief Recompute the TRUE residual ||b - A x||_1 from scratch, once, after the solve.
//!
//! BiCGSTAB tracks the residual by a RECURRENCE (r = s - omega*t), not by evaluating b - A x.
//! The two agree in exact arithmetic and drift apart in floating point, and the drift is
//! one-directional: the recurrence keeps falling after the true residual has stagnated. A
//! solver that reports 1e-25 while the true residual sits at 1e-3 is the standard failure and
//! it is INVISIBLE from inside the iteration. So the reported convergence is audited against a
//! kernel that shares no code with the recurrence -- SnResidual, the one the Richardson path
//! was validated with. Cost is one extra pass per SOLVE (not per iteration).
TaskStatus SnKrTrue(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st) {
  Real res = 0.0, nrm = 0.0;
  SnResidual(md, dt, &res, &nrm);
  st->l[0] += res;
  st->l[1] += nrm;
  return TaskStatus::complete;
}

TaskStatus SnKrReport(std::shared_ptr<KrylovState> st, const int max_it, const Real rel_tol) {
  if (parthenon::Globals::my_rank != 0) return TaskStatus::complete;
  // R-F2 (validation/radiation_sn, 2026-10-08): the TRUE residual is divided by the SAME setup norms the
  // recurrence used (sum over groups of each solve's ||b||_1), not by SnResidual's sum|rhs| re-evaluated at the
  // converged x: that norm contains chi_s J(x) and the streaming inflow, so when J changes a lot in one step the two
  // ratios had different denominators (a 100x false alarm in a Fleck-stiff 0-D relaxation). With sn_norm = b the
  // wrapper already returns the x-independent norm, which equals the setup norm.
  Real bsetup = 0.0;
  for (const auto &gr : st->per_group) bsetup += gr.bnrm;
  const Real tr_old = (st->r1.val.size() >= 2 && st->r1.val[1] > 0.0)
                          ? st->r1.val[0] / st->r1.val[1] : 0.0;
  const Real tr = (st->r1.val.size() >= 2 && bsetup > 0.0) ? st->r1.val[0] / bsetup : tr_old;
  // The recurrence and the independent evaluation must agree, or the convergence claim is
  // worthless. 100x slack: they are different summation orders of different expressions.
  // `tr` is the true residual over ALL groups (SnKrTrue runs unrestricted), so it audits the
  // assembled solution rather than whichever group happened to finish last. The trust test only
  // bites when every group claims convergence -- if any group stopped on the cap or a
  // breakdown, a large true residual is the honest report, not a contradiction.
  bool all_conv = !st->per_group.empty();
  for (const auto &gr : st->per_group) all_conv = all_conv && gr.converged;
  const bool trust = !all_conv || (tr < 100.0 * rel_tol);
  for (std::size_t g = 0; g < st->per_group.size(); ++g) {
    const auto &gr = st->per_group[g];
    if (gr.skipped) {
      std::cout << "[SN/bicgstab] group " << g << " SKIPPED (||b||_1 = " << gr.bnrm
                << ", below radiation/sn_skip_group_rel of the largest earlier group)" << std::endl;
      continue;
    }
    if (gr.restored) {
      std::cout << "[SN/bicgstab] group " << g << " RESTORED to I^n and uncoupled this step"
                << " (radiation/sn_fallback; solve: converged=" << gr.converged << " breakdown="
                << gr.breakdown << " rel=" << gr.rel << ")" << std::endl;
    }
    if (gr.converged) {
      std::cout << "[SN/bicgstab] group " << g << " converged in " << gr.iters
                << " iters, rel = " << gr.rel << " (from " << gr.res0 << ")" << std::endl;
    } else if (gr.breakdown != 0) {
      std::cout << "[SN/bicgstab] group " << g << " BREAKDOWN code " << gr.breakdown
                << " after " << gr.iters << " iters, rel = " << gr.rel
                << " -- result is NOT converged" << std::endl;
    } else {
      std::cout << "[SN/bicgstab] group " << g << " NOT CONVERGED: hit the " << max_it
                << "-iteration cap, rel = " << gr.rel << " vs tol " << rel_tol
                << " (this is a FLOOR, not a solution)" << std::endl;
    }
  }
  std::cout << "[SN/bicgstab] all-group TRUE rel = " << tr << " (vs end-state norm " << tr_old << ")"
            << (trust ? "" : "   *** RECURRENCE DISAGREES WITH TRUE RESIDUAL ***") << std::endl;
  PARTHENON_REQUIRE(trust, "BiCGSTAB reported convergence but the independently evaluated "
                           "residual ||b - A x|| disagrees: the reported convergence is not real. "
                           "TWO causes seen so far. (1) A STALE INPUT to the residual -- e.g. "
                           "sn.mom not recomputed before the audit, which made it grade the new "
                           "solution against the previous scattering source (gate K5: 4.60e-11 "
                           "vs 1.57e-02). Check task ORDER first. (2) The attainable accuracy "
                           "limit at high cond(A) -- measured on the "
                           "optically thin K2 deck, sn_rel_tol = 1e-12 gives a recurrence of "
                           "6.17e-13 against a true residual of 4.11e-10. Raise sn_rel_tol "
                           "(~1e-10 is reachable there) rather than raising the iteration cap.");
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \brief Emit the tasks for one preconditioner application, y = M^-1 (field `rn`).
//!
//! nsw == 0 is the IDENTITY preconditioner (y = rn), not "no sweeps". The distinction matters:
//! zeroing y and doing nothing would give y = 0, hence v = A*0 = 0, hence an immediate
//! division by (r0,v) = 0 -- a breakdown that looks like a solver bug rather than a
//! misconfigured preconditioner. Unpreconditioned BiCGSTAB is the reference arm that isolates
//! the driver from the smoother, so it has to actually work.

TaskID AddPrecond(TaskList &itl, TaskID t, std::shared_ptr<MeshData<Real>> &md,
                  std::shared_ptr<MeshData<Real>> &md_ky, const Real dt, const int g,
                  const int nsw, const bool ml, const std::string &rn,
                  parthenon::Mesh *pmesh, const int partition,
                  std::shared_ptr<KrylovState> st) {
  if (nsw <= 0) {
    t = AddUnlessSkipped(itl, t, st, SnVecLC, md.get(), Real(1.0), rn, Real(0.0), rn,
                         Real(0.0), rn, KyName());
    return AddBoundaryExchangeTasks(t, itl, md_ky, ml);
  }
  // Zero FIRST so M^-1 does not depend on the previous application -- see SnZeroAll.
  t = AddUnlessSkipped(itl, t, st, SnZeroAll, md.get(), KyName());
  for (int sw = 0; sw < nsw; ++sw) {
    t = AddUnlessSkipped(itl, t, st, SnPrecondSweep, md.get(), dt, g, rn, KyName());
    t = AddBoundaryExchangeTasks(t, itl, md_ky, ml);
  }
  return t;
}

//----------------------------------------------------------------------------------------
//! \brief Preconditioned BiCGSTAB driver.
//!
//! Why Krylov: the ordered sweep solves each block in one pass but cannot supply the global
//! coupling of an optically thin domain (rate 0.9998 at full c). Per iteration: 2 preconditioner
//! sweeps, 2 operator applies, 2 halo exchanges, 3 global reductions. Scattering is inside A.
TaskID AddSnKrylov(TaskList &tl, TaskID prev, parthenon::Mesh *pmesh,
                   std::shared_ptr<MeshData<Real>> &md,
                   std::shared_ptr<MeshData<Real>> &md_sn,
                   std::shared_ptr<MeshData<Real>> &md_ky, const Real dt,
                   const int partition) {
  auto pkg = pmesh->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const int max_it = p.max_iterations;
  const int nsw = p.precond_sweeps;
  const int n_group = p.n_group;
  const bool ml = pmesh->multilevel;
  auto st = std::make_shared<KrylovState>();
  st->pkg = pkg.get();
  const bool norm_b = pkg->Param<bool>("sn_norm_b");           // R-F6(a)
  const Real local_tol = pkg->Param<Real>("sn_local_tol");     // R-F6(b)
  st->local_on = (local_tol > 0.0);
  st->per_group.assign(n_group, KrylovState::GroupResult{});
  if (p.fallback || p.skip_group_rel > 0.0)
    st->coupling_off = std::make_shared<std::vector<int>>(n_group, 0);

  // ONE EXCHANGE FOR ALL GROUPS. x's ghosts must be valid before A is applied to it, or r0 is
  // wrong on every block face. It does not have to be repeated per group: group g's solve
  // writes only components g*nang .. g*nang+nang-1 of sn.u, so the components group g+1 reads
  // are still exactly the ones this exchange communicated.
  prev = AddBoundaryExchangeTasks(prev, tl, md_sn, ml);
  if (p.sou_limit) prev = tl.AddTask(prev, SnSouLimLag, md.get());   // R-F10: lag ghosts = exchanged x0 = I^n

  // --- one independent BiCGSTAB solve per group ---------------------------------------------
  // A is block-diagonal in the group index, so this is not an approximation: it is the same
  // linear system, solved as the n_group decoupled systems it already was. Each group gets its
  // own Krylov space and its own alpha/omega/rho recurrence, which is if anything better
  // conditioned than forcing one set of scalars on all three. What it BUYS is memory: every
  // work vector is nang components instead of nfa (see the registration in Initialize).
  for (int g = 0; g < n_group; ++g) {
    // --- setup ------------------------------------------------------------------------------
    prev = tl.AddTask(prev, SnKrReset, st, g);
    // p = v = 0 so the first pass reduces to p = r; y zeroed for a linear M^-1 (see SnZeroAll).
    prev = tl.AddTask(prev, SnZeroAll, md.get(), KpName());
    prev = tl.AddTask(prev, SnZeroAll, md.get(), KvName());
    prev = tl.AddTask(prev, SnApplyA, md.get(), dt, g, USolName(), KrName(), 1);  // r = b - A x
    prev = tl.AddTask(prev, SnVecLC, md.get(), 1.0, KrName(), 0.0, KrName(), 0.0, KrName(),
                      Kr0Name());
    // Stage-A gate. Costs one extra residual pass per solve, so it is opt-in -- but it is the
    // only thing standing between a coding slip in the stencil and a confidently wrong answer.
    if (p.verify_operator) prev = tl.AddTask(prev, SnKrVerifyOp, md.get(), dt, g);
    if (!p.krylov_dot_scale) {
      prev = norm_b ? tl.AddTask(prev, SnKrBnormB, md.get(), dt, st)
                    : tl.AddTask(prev, SnKrBnorm, md.get(), dt, st);            // ||b||_1, ||r0||_1
      prev = tl.AddTask(prev, SnKrDot, md.get(), Kr0Name(), KrName(), st, 2);  // rho_1
      prev = tl.AddTask(parthenon::TaskQualifier::local_sync, prev, SnKrStage, st, 1, 3);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev,
                        &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->r1, MPI_SUM);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region |
                            parthenon::TaskQualifier::local_sync,
                        prev, &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->r1);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnKrInit, st);
      if (p.skip_group_rel > 0.0)
        prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnKrDecideSkip, st, g,
                          p.skip_group_rel);
    } else {
      // Norms first (one extra reduction per group per step), THEN the scaled rho_1.
      prev = norm_b ? tl.AddTask(prev, SnKrBnormB, md.get(), dt, st)
                    : tl.AddTask(prev, SnKrBnorm, md.get(), dt, st);            // ||b||_1, ||r0||_1
      prev = tl.AddTask(parthenon::TaskQualifier::local_sync, prev, SnKrStage, st, 1, 2);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev,
                        &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->r1, MPI_SUM);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region |
                            parthenon::TaskQualifier::local_sync,
                        prev, &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->r1);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnKrSetScale, st);
      prev = tl.AddTask(prev, SnKrDot, md.get(), Kr0Name(), KrName(), st, 0);  // scaled rho_1
      prev = tl.AddTask(parthenon::TaskQualifier::local_sync, prev, SnKrStage, st, 3, 1);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev,
                        &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->r3, MPI_SUM);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region |
                            parthenon::TaskQualifier::local_sync,
                        prev, &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->r3);
      prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnKrInitScaled, st);
      if (p.skip_group_rel > 0.0)
        prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnKrDecideSkip, st, g,
                          p.skip_group_rel);
    }

    // --- iteration --------------------------------------------------------------------------
    auto [itl, kdone] = tl.AddSublist(prev, {1, max_it});
    parthenon::TaskID none;

    auto tb = itl.AddTask(parthenon::TaskQualifier::once_per_region, none, SnKrBeta, st);
    auto t = itl.AddTask(tb, SnKrUpdate, md.get(), st, KrUpd::p);

    // y = M^-1 p. Zeroed first so M^-1 is a fixed linear operator; nsw sweeps with an exchange
    // between them make it block-Gauss-Seidel ACROSS blocks, as the red-black path does.
    t = AddPrecond(itl, t, md, md_ky, dt, g, nsw, ml, KpName(), pmesh, partition, st); // y = M^-1 p
    t = AddUnlessSkipped(itl, t, st, SnApplyA, md.get(), dt, g, KyName(), KvName(), 0); // v = A y

    t = itl.AddTask(t, SnKrDot, md.get(), Kr0Name(), KvName(), st, 0);
    t = itl.AddTask(parthenon::TaskQualifier::local_sync, t, SnKrStage, st, 2, 1);
    auto a1 = itl.AddTask(parthenon::TaskQualifier::once_per_region, t,
                          &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->r2,
                          MPI_SUM);
    auto a2 = itl.AddTask(parthenon::TaskQualifier::once_per_region |
                              parthenon::TaskQualifier::local_sync,
                          a1, &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->r2);
    t = itl.AddTask(parthenon::TaskQualifier::once_per_region, a2, SnKrAlpha, st);

    // s = r - alpha*v, IN PLACE in k_r -- k_s is gone. See SnKrUpdate for why that is legal.
    t = itl.AddTask(t, SnKrUpdate, md.get(), st, KrUpd::s);
    // x += alpha*y HERE, while k_y still holds y -- this is what licenses reusing k_y for z.
    t = itl.AddTask(t, SnKrUpdate, md.get(), st, KrUpd::xy);

    t = AddPrecond(itl, t, md, md_ky, dt, g, nsw, ml, KrName(), pmesh, partition, st); // z = M^-1 s
    t = AddUnlessSkipped(itl, t, st, SnApplyA, md.get(), dt, g, KyName(), KtName(), 0); // t = A z

    t = itl.AddTask(t, SnKrDot, md.get(), KtName(), KrName(), st, 0);          // (t, s)
    t = itl.AddTask(t, SnKrDot, md.get(), KtName(), KtName(), st, 1);
    t = itl.AddTask(parthenon::TaskQualifier::local_sync, t, SnKrStage, st, 3, 2);
    auto o1 = itl.AddTask(parthenon::TaskQualifier::once_per_region, t,
                          &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->r3,
                          MPI_SUM);
    auto o2 = itl.AddTask(parthenon::TaskQualifier::once_per_region |
                              parthenon::TaskQualifier::local_sync,
                          o1, &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->r3);
    t = itl.AddTask(parthenon::TaskQualifier::once_per_region, o2, SnKrOmega, st);

    t = itl.AddTask(t, SnKrUpdate, md.get(), st, KrUpd::xz);                   // x += omega*z
    t = itl.AddTask(t, SnKrUpdate, md.get(), st, KrUpd::r);                    // r = s - omega t

  // Next rho and the residual norm in ONE reduction, at the END of the body. The denominator
  // is the SnResidual ||b||_1 measured at setup (see SnKrBnorm), so the ratio tested here is
  // the same L1 ratio the Richardson path tests and sn_rel_tol carries the same meaning on
  // both solvers. It is fixed for the solve rather than re-evaluated per iteration, which is
  // why the end-of-solve audit (SnKrTrue) recomputes the ratio at the CONVERGED state.
    t = itl.AddTask(t, SnKrDot, md.get(), Kr0Name(), KrName(), st, 0);
    t = itl.AddTask(t, SnKrAbs, md.get(), KrName(), st, 1);
    if (local_tol > 0.0) t = itl.AddTask(t, SnKrLocal, md.get(), dt, st, local_tol);    // R-F6(b)
    t = itl.AddTask(parthenon::TaskQualifier::local_sync, t, SnKrStage, st, 1, local_tol > 0.0 ? 3 : 2);
    auto n1 = itl.AddTask(parthenon::TaskQualifier::once_per_region, t,
                          &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->r1,
                          MPI_SUM);
    auto n2 = itl.AddTask(parthenon::TaskQualifier::once_per_region |
                              parthenon::TaskQualifier::local_sync,
                          n1, &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->r1);
    itl.AddTask(parthenon::TaskQualifier::completion | parthenon::TaskQualifier::global_sync,
                n2, SnKrCheck, st, p.rel_tol, max_it);

    // Stash this group's outcome, then chain the next group's setup onto this solve's
    // completion so no two solves are ever live at once (they share one KrylovState and one
    // set of nang-wide work vectors).
    prev = tl.AddTask(kdone, SnKrRecord, st, g);
    if (p.fallback || p.skip_group_rel > 0.0)
      prev = tl.AddTask(prev, SnKrFallback, md.get(), st, g, p.fallback, p.fallback_rel);
  }

  // x's ghosts are stale after the last update; the writeback is interior-only but the NEXT
  // step's A*x0 is not, and sn.u persists across steps.
  auto ex = AddBoundaryExchangeTasks(prev, tl, md_sn, ml);
  // MOMENTS FIRST, and the order is load-bearing. SnKrTrue evaluates SnResidual, which reads
  // sn.mom for the chi_s*J term; with the moments computed AFTER the audit it graded the new
  // solution against the PREVIOUS iterate's scattering source. With chi_s = 0 that term is zero
  // and the bug is invisible -- gate K5 (chi_s/chi_t = 0.91) exposed it as a recurrence of
  // 4.60e-11 against a true residual of 1.57e-02, and the audit aborted rather than shipping it.
  auto mo = tl.AddTask(ex, SnComputeMoments, md.get());
  // Audit the convergence claim before anything acts on it. Needs x's ghosts and current moments.
  auto tv = norm_b ? tl.AddTask(mo, SnKrTrueB, md.get(), dt, st) : tl.AddTask(mo, SnKrTrue, md.get(), dt, st);
  tv = tl.AddTask(parthenon::TaskQualifier::local_sync, tv, SnKrStage, st, 1, 2);
  tv = tl.AddTask(parthenon::TaskQualifier::once_per_region, tv,
                  &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->r1, MPI_SUM);
  tv = tl.AddTask(parthenon::TaskQualifier::once_per_region |
                      parthenon::TaskQualifier::local_sync,
                  tv, &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->r1);
  // R-F6: the residual moment needs exactly what SnKrTrue needed (x's ghosts, current moments).
  const bool conserve = pkg->Param<bool>("sn_conserve_residual");
  if (conserve) tv = tl.AddTask(tv, SnResidualMoment, md.get(), dt);
  auto wb = tl.AddTask(tv, SnWriteback, md.get());
  auto mc = tl.AddTask(wb, SnMatterCoupling, md.get(), dt, st->coupling_off);
  if (conserve) mc = tl.AddTask(mc, SnConserveCorrect, md.get(), dt, st->coupling_off);
  return tl.AddTask(mc, SnKrReport, st, max_it, p.rel_tol);
}

//========================================================================================
//! R-F3: coupled Fleck & Cummings BiCGSTAB (radiation/sn_fc_scatter). One solve over ALL groups on nfa-wide work
//! vectors; operator A_FC = (legacy per-group A_g) + C(x) (SnFCCross); norm = sum|I^n/(chat dt) + eta| (R-F6a).
TaskStatus SnFCBnorm(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st) {
  Real b = 0.0, r = 0.0;
  SnBnormB(md, dt, -1, &b, nullptr);
  SnVecAbs(md, KcrName(), &r);
  st->l[0] += b;
  st->l[1] += r;
  return TaskStatus::complete;
}
//! Per-group ||b||_1 (rank partials) for the frozen-group decision.
TaskStatus SnFCGroupNorms(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st) {
  const int ng = static_cast<int>(st->fc_skip.size());
  if (static_cast<int>(st->lg.size()) < ng) st->lg.assign(ng, 0.0);
  for (int g = 0; g < ng; ++g) {
    Real b = 0.0;
    SnBnormB(md, dt, g, &b, nullptr);
    st->lg[g] += b;
  }
  return TaskStatus::complete;
}
TaskStatus SnFCStageGroups(std::shared_ptr<KrylovState> st) {
  st->rg.val = st->lg;
  std::fill(st->lg.begin(), st->lg.end(), 0.0);
  return TaskStatus::complete;
}
//! Freeze groups whose ||b||_1 is below rel x the largest (the empty UV group): not swept, not applied, residual 0.
TaskStatus SnFCDecideSkip(std::shared_ptr<KrylovState> st, const Real rel) {
  Real bmax = 0.0;
  for (const Real b : st->rg.val) bmax = std::max(bmax, b);
  for (std::size_t g = 0; g < st->fc_skip.size(); ++g) {
    st->fc_skip[g] = (rel > 0.0 && g < st->rg.val.size() && st->rg.val[g] < rel * bmax) ? 1 : 0;
    if (st->coupling_off && g < st->coupling_off->size()) (*st->coupling_off)[g] = st->fc_skip[g];  // no exchange
  }
  return TaskStatus::complete;
}
TaskStatus SnFCZeroGroup(MeshData<Real> *md, std::shared_ptr<KrylovState> st, const std::string &nm, const int g) {
  if (!st->fc_skip[g]) return TaskStatus::complete;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(std::vector<std::string>{nm}, imap);
  const int i0 = imap[nm].first + g * p.nang;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::FCZeroGroup", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, p.nang - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int a, const int k, const int j, const int i) { pack(b, i0 + a, k, j, i) = 0.0; });
  return TaskStatus::complete;
}
TaskStatus SnFCSweepG(MeshData<Real> *md, std::shared_ptr<KrylovState> st, const Real dt, const int g,
                      const std::string &rn, const std::string &yn) {
  if (st->fc_skip[g]) return TaskStatus::complete;
  return SnPrecondSweep(md, dt, g, rn, yn);
}
//! radiation/sn_sweep_group_batch: every non-frozen group's preconditioner sweep in one launch (mode 4, >= 2-D);
//! otherwise exactly the per-group SnFCSweepG loop.
TaskStatus SnFCSweepAll(MeshData<Real> *md, std::shared_ptr<KrylovState> st, const Real dt, const std::string &rn,
                        const std::string &yn) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const SnParams p = pmb->packages.Get("radiation")->Param<SnParams>("sn_params");
  std::vector<int> act;
  for (int g = 0; g < static_cast<int>(st->fc_skip.size()); ++g)
    if (!st->fc_skip[g]) act.push_back(g);
  if (p.wavefront == 5 && pmb->pmy_mesh->ndim >= 2) return SnPrecondSweepWave5Groups(md, dt, rn, yn, p.precond_sou, act);
  if (p.wavefront == 4 && pmb->pmy_mesh->ndim >= 2) return SnPrecondSweepWave4Groups(md, dt, rn, yn, p.precond_sou, act);
  for (const int g : act) SnPrecondSweep(md, dt, g, rn, yn);
  return TaskStatus::complete;
}
TaskStatus SnFCApplyG(MeshData<Real> *md, std::shared_ptr<KrylovState> st, const Real dt, const int g,
                      const std::string &xn, const std::string &yn) {
  if (st->fc_skip[g]) return TaskStatus::complete;
  return SnApplyA(md, dt, g, xn, yn, 0);
}

TaskStatus SnFCTrue(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st) {
  Real res = 0.0, b = 0.0;
  SnFCTrueResidual(md, dt, KctName(), &res);
  SnBnormB(md, dt, -1, &b, nullptr);
  st->l[0] += res;
  st->l[1] += b;
  return TaskStatus::complete;
}
TaskStatus SnFCLocal(MeshData<Real> *md, const Real dt, std::shared_ptr<KrylovState> st, const Real tol) {
  Real c = 0.0;
  SnLocalViolationsWide(md, dt, KcrName(), tol, &c);
  st->l[2] += c;
  return TaskStatus::complete;
}
TaskStatus SnFCUpdate(MeshData<Real> *md, std::shared_ptr<KrylovState> st, const KrUpd w) {
  switch (w) {
    case KrUpd::p:
      return SnVecLC(md, 1.0, KcrName(), st->beta, KcpName(), -st->beta * st->omega, KcvName(), KcpName());
    case KrUpd::s:
      return SnVecLC(md, 1.0, KcrName(), -st->alpha, KcvName(), 0.0, KcvName(), KcrName());
    case KrUpd::xy:
      return SnVecLC(md, 1.0, USolName(), st->alpha, KcyName(), 0.0, KcyName(), USolName());
    case KrUpd::xz:
      return SnVecLC(md, 1.0, USolName(), st->omega, KcyName(), 0.0, KcyName(), USolName());
    case KrUpd::r:
      return SnVecLC(md, 1.0, KcrName(), -st->omega, KctName(), 0.0, KctName(), KcrName());
  }
  return TaskStatus::complete;
}

//! radiation/sn_krylov_fused (8-group speed work, default false): the coupled solve's BiCGSTAB vector updates and their
//! reductions fused, each output written with the SAME expression SnVecLC writes (c1 a + c2 b + c3 c, c3 = 0) so the
//! iterate updates are the per-element arithmetic of the unfused path; the dot products and L1 norm are summed in one
//! kernel instead of one each (different reduction tree: round-off-level differences in alpha/omega/the norms).
//!   SX:   r = r - alpha v ;  x = x + alpha y                          (was 2 SnVecLC passes)
//!   DOTS: l0 += (s t).(s r) ; l1 += (s t).(s t)                       (was 2 SnVecDot passes)
//!   XR:   x = x + omega y ; r = r - omega t ; l0 += (s r0).(s r) ; l1 += |r|   (was 2 SnVecLC + SnVecDot + SnVecAbs)
TaskStatus SnFCFusedSX(MeshData<Real> *md, std::shared_ptr<KrylovState> st) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{KcrName(), KcvName(), USolName(), KcyName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ir = imap[KcrName()].first, iv = imap[KcvName()].first, ix = imap[USolName()].first,
            iy = imap[KcyName()].first;
  const int nc = SnPackWidth(imap, KcrName());
  PARTHENON_REQUIRE(SnPackWidth(imap, KcvName()) == nc && SnPackWidth(imap, USolName()) == nc &&
                        SnPackWidth(imap, KcyName()) == nc, "SnFCFusedSX operand widths differ.");
  const Real a = st->alpha;
  const Real c1 = 1.0, c3 = 0.0;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Sn::FCFusedSX", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nc - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        const Real v = pack(b, iv + n, k, j, i), y = pack(b, iy + n, k, j, i);
        pack(b, ir + n, k, j, i) = c1 * pack(b, ir + n, k, j, i) + (-a) * v + c3 * v;
        pack(b, ix + n, k, j, i) = c1 * pack(b, ix + n, k, j, i) + a * y + c3 * y;
      });
  return TaskStatus::complete;
}
TaskStatus SnFCFusedDots(MeshData<Real> *md, std::shared_ptr<KrylovState> st) {
  if (st->skip) return TaskStatus::complete;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{KctName(), KcrName()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int it = imap[KctName()].first, ir = imap[KcrName()].first;
  const int nc = SnPackWidth(imap, KctName());
  PARTHENON_REQUIRE(SnPackWidth(imap, KcrName()) == nc, "SnFCFusedDots operand widths differ.");
  const Real sc = st->dscale;
  Real d0 = 0.0, d1 = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::FCFusedDots", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nc - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &e0, Real &e1) {
        const Real t = sc * pack(b, it + n, k, j, i);
        e0 += t * (sc * pack(b, ir + n, k, j, i));
        e1 += t * t;
      },
      Kokkos::Sum<Real>(d0), Kokkos::Sum<Real>(d1));
  st->l[0] += d0;
  st->l[1] += d1;
  return TaskStatus::complete;
}
TaskStatus SnFCFusedXR(MeshData<Real> *md, std::shared_ptr<KrylovState> st) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  std::vector<std::string> names{USolName(), KcyName(), KcrName(), KctName(), Kcr0Name()};
  parthenon::PackIndexMap imap;
  auto pack = md->PackVariables(names, imap);
  const int ix = imap[USolName()].first, iy = imap[KcyName()].first, ir = imap[KcrName()].first,
            it = imap[KctName()].first, ir0 = imap[Kcr0Name()].first;
  const int nc = SnPackWidth(imap, KcrName());
  PARTHENON_REQUIRE(SnPackWidth(imap, USolName()) == nc && SnPackWidth(imap, KcyName()) == nc &&
                        SnPackWidth(imap, KctName()) == nc && SnPackWidth(imap, Kcr0Name()) == nc,
                    "SnFCFusedXR operand widths differ.");
  const Real w = st->omega, sc = st->dscale;
  const Real c1 = 1.0, c3 = 0.0;
  const bool red = !st->skip;
  Real d0 = 0.0, d1 = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "Sn::FCFusedXR", parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, nc - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i, Real &e0, Real &e1) {
        const Real y = pack(b, iy + n, k, j, i), t = pack(b, it + n, k, j, i);
        pack(b, ix + n, k, j, i) = c1 * pack(b, ix + n, k, j, i) + w * y + c3 * y;
        const Real r = c1 * pack(b, ir + n, k, j, i) + (-w) * t + c3 * t;
        pack(b, ir + n, k, j, i) = r;
        e0 += (sc * pack(b, ir0 + n, k, j, i)) * (sc * r);
        e1 += std::abs(r);
      },
      Kokkos::Sum<Real>(d0), Kokkos::Sum<Real>(d1));
  if (red) {
    st->l[0] += d0;
    st->l[1] += d1;
  }
  return TaskStatus::complete;
}

//! y = M^-1 rn (block-diagonal in the group index: the per-group ordered sweep), then A_FC y -> out.
static TaskID AddFCPrecondApply(TaskList &itl, TaskID t, std::shared_ptr<MeshData<Real>> &md,
                                std::shared_ptr<MeshData<Real>> &md_ky, const Real dt, const int n_group,
                                const int nsw, const bool ml, const std::string &rn, const std::string &out,
                                std::shared_ptr<KrylovState> st) {
  const bool lean_zero = st->pkg->Param<bool>("sn_fc_lean_zero");   // speed work, radiation_sn_fastops.cpp
  if (st->pkg->Param<bool>("sn_gray_dsa") && st->pkg->Param<bool>("sn_gd_debug"))
    t = itl.AddTask(t, GrayDsa::DbgCopyIn, md.get(), rn);
  if (lean_zero) {
    t = itl.AddTask(t, SnZeroGhosts, md.get(), KcyName());
    t = itl.AddTask(t, [=]() { return SnZeroSkippedGroups(md.get(), KcyName(), st->fc_skip, st->nang_fc); });
  } else {
    t = itl.AddTask(t, SnZeroAll, md.get(), KcyName());
  }
  for (int sw = 0; sw < std::max(1, nsw); ++sw) {
    if (nsw > 0 && st->pkg->Param<SnParams>("sn_params").sweep_group_batch)
      t = itl.AddTask(t, SnFCSweepAll, md.get(), st, dt, rn, KcyName());
    else
      for (int g = 0; g < n_group; ++g)
        t = (nsw > 0) ? itl.AddTask(t, SnFCSweepG, md.get(), st, dt, g, rn, KcyName())
                      : itl.AddTask(t, SnVecLC, md.get(), Real(1.0), rn, Real(0.0), rn, Real(0.0), rn, KcyName());
    // radiation/sn_gd_skip_sweep_exchange (8-group speed work, default false): with the gray DSA the halo exchange after
    // the LAST sweep is dead -- GrayDsa reads y on the interior only (Load, Rhs, Apply) and AddCorrection ends with a full
    // exchange of y that overwrites every ghost (same-level copies, coarse-fine transfers and the physical-face incoming
    // condition all recomputed from the corrected interior). Skipping it is bit-identical and saves one nfa-wide
    // exchange per application (8 groups on GPUs: ~57 ms of ~490).
    const bool dead = (sw == std::max(1, nsw) - 1) && st->pkg->Param<bool>("sn_gray_dsa") &&
                      st->pkg->Param<bool>("sn_gd_skip_sweep_exchange");
    if (!dead) t = AddBoundaryExchangeTasks(t, itl, md_ky, ml);
  }
  if (st->pkg->Param<bool>("sn_gray_dsa"))   // R-F12: y += xi eps / 4pi (gray diffusion correction), y halo
    t = GrayDsa::AddCorrection(itl, t, md, md_ky, dt, ml, st->pmesh, st->partition, &st->fc_skip);
  t = lean_zero ? itl.AddTask(t, [=]() { return SnZeroSkippedGroups(md.get(), out, st->fc_skip, st->nang_fc); })
                : itl.AddTask(t, SnZeroAll, md.get(), out);   // frozen groups' slices stay zero
  for (int g = 0; g < n_group; ++g) t = itl.AddTask(t, SnFCApplyG, md.get(), st, dt, g, KcyName(), out);
  return itl.AddTask(t, SnFCCross, md.get(), KcyName(), out, Real(1.0));
}

TaskID AddSnKrylovFC(TaskList &tl, TaskID prev, parthenon::Mesh *pmesh, std::shared_ptr<MeshData<Real>> &md,
                     std::shared_ptr<MeshData<Real>> &md_sn, std::shared_ptr<MeshData<Real>> &md_ky,
                     const Real dt, const int partition) {
  auto pkg = pmesh->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const int max_it = p.max_iterations, nsw = p.precond_sweeps, n_group = p.n_group;
  const bool ml = pmesh->multilevel;
  const Real local_tol = pkg->Param<Real>("sn_local_tol");
  auto st = std::make_shared<KrylovState>();
  st->pkg = pkg.get();
  st->per_group.assign(1, KrylovState::GroupResult{});   // ONE coupled solve; reported as "group 0"
  st->local_on = (local_tol > 0.0);
  st->fc_skip.assign(n_group, 0);
  st->nang_fc = p.nang;
  st->pmesh = pmesh;
  st->partition = partition;
  st->coupling_off = std::make_shared<std::vector<int>>(n_group, 0);   // frozen groups do not exchange with the gas
  auto red = [&](TaskList &l, TaskID t, parthenon::AllReduce<std::vector<Real>> *r, const int which,
                 const int nv) {
    t = l.AddTask(parthenon::TaskQualifier::local_sync, t, SnKrStage, st, which, nv);
    t = l.AddTask(parthenon::TaskQualifier::once_per_region, t,
                  &parthenon::AllReduce<std::vector<Real>>::StartReduce, r, MPI_SUM);
    return l.AddTask(parthenon::TaskQualifier::once_per_region | parthenon::TaskQualifier::local_sync, t,
                     &parthenon::AllReduce<std::vector<Real>>::CheckReduce, r);
  };
  prev = AddBoundaryExchangeTasks(prev, tl, md_sn, ml);
  if (p.sou_limit) prev = tl.AddTask(prev, SnSouLimLag, md.get());   // R-F10
  prev = tl.AddTask(prev, SnKrReset, st, 0);
  prev = tl.AddTask(prev, SnZeroAll, md.get(), KcpName());
  prev = tl.AddTask(prev, SnZeroAll, md.get(), KcvName());
  for (int g = 0; g < n_group; ++g) prev = tl.AddTask(prev, SnApplyA, md.get(), dt, g, USolName(), KcrName(), 1);
  prev = tl.AddTask(prev, SnFCCross, md.get(), USolName(), KcrName(), Real(-1.0));   // r0 = b - A_FC x
  if (p.verify_operator) prev = tl.AddTask(prev, SnFCVerifyOp, md.get(), dt);   // R-F16, before any group is frozen
  if (p.skip_group_rel > 0.0) {   // freeze the empty groups (sn_skip_group_rel), as the per-group path skips them
    prev = tl.AddTask(prev, SnFCGroupNorms, md.get(), dt, st);
    prev = tl.AddTask(parthenon::TaskQualifier::local_sync, prev, SnFCStageGroups, st);
    prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev,
                      &parthenon::AllReduce<std::vector<Real>>::StartReduce, &st->rg, MPI_SUM);
    prev = tl.AddTask(parthenon::TaskQualifier::once_per_region | parthenon::TaskQualifier::local_sync, prev,
                      &parthenon::AllReduce<std::vector<Real>>::CheckReduce, &st->rg);
    prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnFCDecideSkip, st, p.skip_group_rel);
    for (int g = 0; g < n_group; ++g) prev = tl.AddTask(prev, SnFCZeroGroup, md.get(), st, KcrName(), g);
  }
  prev = tl.AddTask(prev, SnVecLC, md.get(), 1.0, KcrName(), 0.0, KcrName(), 0.0, KcrName(), Kcr0Name());
  prev = tl.AddTask(prev, SnFCBnorm, md.get(), dt, st);
  prev = red(tl, prev, &st->r1, 1, 2);
  prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnKrSetScale, st);
  prev = tl.AddTask(prev, SnKrDot, md.get(), Kcr0Name(), KcrName(), st, 0);
  prev = red(tl, prev, &st->r3, 3, 1);
  prev = tl.AddTask(parthenon::TaskQualifier::once_per_region, prev, SnKrInitScaled, st);

  auto [itl, kdone] = tl.AddSublist(prev, {1, max_it});
  parthenon::TaskID none;
  auto t = itl.AddTask(parthenon::TaskQualifier::once_per_region, none, SnKrBeta, st);
  t = itl.AddTask(t, SnFCUpdate, md.get(), st, KrUpd::p);
  t = AddFCPrecondApply(itl, t, md, md_ky, dt, n_group, nsw, ml, KcpName(), KcvName(), st);   // y, v = A y
  t = itl.AddTask(t, SnKrDot, md.get(), Kcr0Name(), KcvName(), st, 0);
  t = red(itl, t, &st->r2, 2, 1);
  t = itl.AddTask(parthenon::TaskQualifier::once_per_region, t, SnKrAlpha, st);
  const bool fused = pkg->Param<bool>("sn_krylov_fused");   // radiation/sn_krylov_fused, see SnFCFusedSX
  if (fused) {
    t = itl.AddTask(t, SnFCFusedSX, md.get(), st);
  } else {
    t = itl.AddTask(t, SnFCUpdate, md.get(), st, KrUpd::s);
    t = itl.AddTask(t, SnFCUpdate, md.get(), st, KrUpd::xy);
  }
  t = AddFCPrecondApply(itl, t, md, md_ky, dt, n_group, nsw, ml, KcrName(), KctName(), st);   // z, t = A z
  if (fused) {
    t = itl.AddTask(t, SnFCFusedDots, md.get(), st);
  } else {
    t = itl.AddTask(t, SnKrDot, md.get(), KctName(), KcrName(), st, 0);
    t = itl.AddTask(t, SnKrDot, md.get(), KctName(), KctName(), st, 1);
  }
  t = red(itl, t, &st->r3, 3, 2);
  t = itl.AddTask(parthenon::TaskQualifier::once_per_region, t, SnKrOmega, st);
  if (fused) {
    t = itl.AddTask(t, SnFCFusedXR, md.get(), st);
  } else {
    t = itl.AddTask(t, SnFCUpdate, md.get(), st, KrUpd::xz);
    t = itl.AddTask(t, SnFCUpdate, md.get(), st, KrUpd::r);
    t = itl.AddTask(t, SnKrDot, md.get(), Kcr0Name(), KcrName(), st, 0);
    t = itl.AddTask(t, SnKrAbs, md.get(), KcrName(), st, 1);
  }
  if (local_tol > 0.0) t = itl.AddTask(t, SnFCLocal, md.get(), dt, st, local_tol);
  t = red(itl, t, &st->r1, 1, local_tol > 0.0 ? 3 : 2);
  itl.AddTask(parthenon::TaskQualifier::completion | parthenon::TaskQualifier::global_sync, t, SnKrCheck, st,
              p.rel_tol, max_it);
  prev = tl.AddTask(kdone, SnKrRecord, st, 0);

  auto ex = AddBoundaryExchangeTasks(prev, tl, md_sn, ml);
  auto mo = tl.AddTask(ex, SnComputeMoments, md.get());
  auto tv = tl.AddTask(mo, SnFCTrue, md.get(), dt, st);
  tv = red(tl, tv, &st->r1, 1, 2);
  auto wb = tl.AddTask(tv, SnWriteback, md.get());
  auto mc = tl.AddTask(wb, SnMatterCoupling, md.get(), dt, st->coupling_off);
  auto rep = tl.AddTask(mc, SnKrReport, st, max_it, p.rel_tol);
  if (pkg->Param<bool>("sn_gray_dsa"))
    rep = tl.AddTask(parthenon::TaskQualifier::once_per_region, rep, GrayDsa::Report, pkg.get());
  return rep;
}

TaskID AddSnTransport(TaskList &tl, TaskID prev, parthenon::Mesh *pmesh, int partition,
                      std::shared_ptr<MeshData<Real>> &md,
                      std::shared_ptr<MeshData<Real>> &md_sn,
                      std::shared_ptr<MeshData<Real>> &md_ky, const Real dt) {
  auto pkg = pmesh->packages.Get("radiation");
  const SnParams p = pkg->Param<SnParams>("sn_params");
  const int max_it = p.max_iterations;

  // Iterative sublist (AddSublist), not an unrolled loop: unrolled iterations still ran their
  // unconditional halo exchanges after convergence (90.7 % of GPU time, job 1698553).
  auto st = std::make_shared<IterState>();
  st->sou = p.sou;

  // dt reaches the seed because the FLECK FACTOR needs it: f = 1/(1 + alpha*c*dt*chi_a*dB/de)
  // is a property of the step, not of the state.
  // t = 0, and a restart whose file has no sn.u: adopt the moments into the ordinates. Without
  // it at t = 0 the writeback overwrites rad.Er/rad.Fr with whatever the ordinates relaxed to
  // from zero and the problem generator's radiation field is silently discarded; without it on
  // an M1 restart the radiation field is discarded the same way. A restart that DOES carry
  // sn.u skips it -- the decision is `sn_init_at_start`, taken from the file in
  // Sn::Initialize, and it must stay rare because the re-seed's zero-clamp creates energy.
  {
    // Size the per-partition flag lazily: the partition count is not known at Initialize time.
    auto need = pkg->Param<std::shared_ptr<std::vector<char>>>("sn_need_init");
    const int npart = pmesh->DefaultNumPartitions();
    if (static_cast<int>(need->size()) < npart)
      need->resize(npart, static_cast<char>(pkg->Param<bool>("sn_init_at_start")));
  }
  prev = tl.AddTask(prev, SnInitFromMoments, md.get(), partition);
  // R2 test-only source: into sn.u BEFORE SnSeedSource copies it to the RHS anchor sn.un.
  if (pkg->Param<bool>("sn_test_source")) prev = tl.AddTask(prev, SnTestSource, md.get(), dt, partition);
  prev = tl.AddTask(prev, SnSeedSource, md.get(), dt);
  prev = tl.AddTask(prev, SnComputeMoments, md.get());

  // tau_cell REPORT, before the solver branch so both richardson and bicgstab get it. Added to
  // the task list ONLY when the key is set, so with the default 0 the graph is unchanged and
  // the run is bit-identical. It sits after SnSeedSource because that is what fills chi_a and
  // chi_s -- reading them earlier would report the PREVIOUS step's opacity, and on a collapse
  // trajectory that is exactly the quantity that is moving.
  if (p.tau_report > 0) {
    auto tc = pkg->Param<std::shared_ptr<std::vector<int>>>("sn_tau_count");
    if (static_cast<int>(tc->size()) <= partition) tc->resize(partition + 1, 0);
    auto ts = std::make_shared<TauState>();
    ts->partition = partition;
    ts->solve = (*tc)[partition]++;
    ts->active = (ts->solve % p.tau_report == 0);
    auto th = tl.AddTask(prev, SnTauHistogram, md.get(), ts);
    auto tstart = tl.AddTask(parthenon::TaskQualifier::once_per_region |
                                 parthenon::TaskQualifier::local_sync,
                             th, &parthenon::AllReduce<std::vector<Real>>::StartReduce,
                             &ts->hist, MPI_SUM);
    auto tdone = tl.AddTask(parthenon::TaskQualifier::once_per_region |
                                parthenon::TaskQualifier::local_sync,
                            tstart, &parthenon::AllReduce<std::vector<Real>>::CheckReduce,
                            &ts->hist);
    prev = tl.AddTask(tdone, SnTauReport, ts, p.dd_w, p.diamond, p.sou, p.sou_kappa);
  }

  // BiCGSTAB replaces the whole stationary iteration below, but shares this preamble: the
  // seed builds b (sn.src, sn.un) and the moments give the lagged scattering source, both of
  // which the Krylov operator reads.
  if (p.krylov && pkg->Param<bool>("sn_fc_scatter"))   // R-F3
    return AddSnKrylovFC(tl, prev, pmesh, md, md_sn, md_ky, dt, partition);
  if (p.krylov) return AddSnKrylov(tl, prev, pmesh, md, md_sn, md_ky, dt, partition);

  auto [itl, sn_done] = tl.AddSublist(prev, {1, max_it});

  parthenon::TaskID none;
  TaskID last;
  if (p.red_black) {
    // Red, exchange, black, exchange. The exchange BETWEEN colours is what makes this
    // Gauss-Seidel ACROSS blocks rather than only within them.
    //
    // Both exchanges ship the WHOLE field. Do not try to send only one colour: the LTS work
    // found that a naive restricted-neighbour container deadlocks on the task watchdog,
    // because parthenon builds sends and receives over container membership and a dropped
    // member posts receives with no matching send.
    auto r = itl.AddTask(none, SnIterate, md.get(), dt, 0, st);
    auto rx = AddBoundaryExchangeTasks(r, itl, md_sn, pmesh->multilevel);
    auto bk = itl.AddTask(rx, SnIterate, md.get(), dt, 1, st);
    last = AddBoundaryExchangeTasks(bk, itl, md_sn, pmesh->multilevel);
  } else {
    // Jacobi: DIAGNOSTIC ONLY. Reproduces the measured 32x-light-CFL wall
    // (FINDINGS_smoother.md Finding 2). Not a tuning option.
    auto jr = itl.AddTask(none, SnIterate, md.get(), dt, -1, st);
    last = AddBoundaryExchangeTasks(jr, itl, md_sn, pmesh->multilevel);
  }
  // Moments refresh once per iteration so a scattering source (step 3) sees the current field.
  // With chi_s == 0 they are diagnostic here. Inside a sublist they now run exactly as many
  // times as the loop iterates, so SnMomentsIfActive's own guard is belt-and-braces.
  last = itl.AddTask(last, SnMomentsIfActive, md.get(), st);
  // TRUE RESIDUAL, after the moments so the scattering source it reconstructs is the current
  // one. No-op under radiation/sn_criterion = increment.
  last = itl.AddTask(last, SnResidualIfActive, md.get(), dt, st);

  // Global (res, nrm) before the test, so the criterion is the same global-sum ratio the
  // Athena++ reference uses. once_per_region is what keeps ONE reduction per region rather
  // than one per partition -- the exact shape of the deadlock the hand-rolled version had.
  auto stage = itl.AddTask(parthenon::TaskQualifier::local_sync, last, SnStageReduction, st);
  auto rstart = itl.AddTask(parthenon::TaskQualifier::once_per_region, stage,
                            &parthenon::AllReduce<std::vector<Real>>::StartReduce,
                            &st->gsum, MPI_SUM);
  auto rdone = itl.AddTask(parthenon::TaskQualifier::once_per_region |
                               parthenon::TaskQualifier::local_sync,
                           rstart, &parthenon::AllReduce<std::vector<Real>>::CheckReduce,
                           &st->gsum);

  itl.AddTask(parthenon::TaskQualifier::completion | parthenon::TaskQualifier::global_sync,
              rdone, SnCheckConvergence, st, p.rel_tol, max_it, p.iter_verbose);

  // Writeback on sn_done: the sublist has exited, and its last act before the convergence
  // test was SnComputeMoments, so sn.mom is current with sn.u. Before SnReport rather than
  // after, so that if the writeback ever throws, the "[SN] converged" line has not already
  // claimed a completed step.
  auto wb = tl.AddTask(sn_done, SnWriteback, md.get());
  // Matter coupling AFTER the writeback, because it reads the converged rad.Er the writeback
  // publishes -- that is what makes the exchange the transport operator's own source term
  // rather than an independent second one. No-op unless radiation/sn_matter_coupling.
  auto mc = tl.AddTask(wb, SnMatterCoupling, md.get(), dt, std::shared_ptr<std::vector<int>>());
  return tl.AddTask(mc, SnReport, st, max_it, p.rel_tol);
}

} // namespace Sn
} // namespace Radiation
