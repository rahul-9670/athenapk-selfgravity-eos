#ifndef RADIATION_RADIATION_SN_HPP_
#define RADIATION_RADIATION_SN_HPP_
//========================================================================================
// AthenaPK - discrete-ordinates (Sn) transport: package interface.
//
// STEP 2 of docs/SN_TRANSPORT_DESIGN.md -- the MINIMAL GPU PROTOTYPE. Scope is deliberate:
//
//   IN  : storage layout, red-black Gauss-Seidel transport sweep, two-pass moment reduction,
//         absorption with a LAGGED source, residual reduction, halo exchange.
//         That is exactly what gates G2 (bandwidth), G3 (no per-thread spill), G4 (memory)
//         and G5 (iteration count) measure -- the four numbers that decide step 3.
//   OUT : scattering (chi_s == 0 here), matter coupling back onto the gas, AMR flux
//         correction. All step 3.
//
// SOURCE AND WRITEBACK -- no longer placeholders (2026-09-07). `sn.src` now holds the true
// thermal EMISSIVITY eta_g = chi_a,g * chat * B_g(T_gas) / (4*pi), evaluated from cons through
// the same EOS + Planck-opacity chain the (removed) M1 matter coupling used, and the converged moments
// are written back to rad.Er / rad.Fr through the single calibrated conversion in
// radiation_sn_units.hpp (Er = 4*pi*J/chat, Fr = 4*pi*H). Both are covered by
// tests/test_sn_units.cpp, which closes the round trip through the ACTUAL quadrature and
// checks the thick-equilibrium identity Er == B_g. (The legacy `sn_source = m1er` placeholder
// was removed 2026-09-24 together with the M1 transport.)
//
// MATTER COUPLING (2026-09-07, `radiation/sn_matter_coupling`, DEFAULT OFF). The gas can now
// feel the radiation, through the NET exchange c*dt*chi_a*(Er - B) implied by the converged
// solution -- NOT through M1's MatterCoupling, which would double-count the emission and
// absorption the Sn transport operator already contains. It is exactly conservative against
// the transport operator's own source terms and it is EXPLICIT in the gas temperature, hence
// only conditionally stable; it reports its own stability number and refuses to be silent
// about floored cells. See SnMatterCoupling in radiation_sn.cpp for the derivation and the
// limits. Radiation FORCE (momentum exchange) is still not applied.
//
// With a frozen source the system is LINEAR and each ordinate independent, so G5 measures the
// transport smoother ALONE. Once scattering or matter coupling is added the angle-coupling
// term discussed in radiation_sn_sweep.hpp becomes live and the iteration count can only get
// worse.
//
// GATED OFF: nothing here is registered or scheduled unless <radiation> transport = sn.
//========================================================================================
#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "radiation_sn_angles.hpp"

namespace Radiation {
namespace Sn {

//! HDF5/pack names. Strings rather than typed variables because the component count
//! nfa = nang * n_group is a RUNTIME value -- parthenon::variable_names::base_t<false, N>
//! needs N at compile time. The tree already uses this pattern for multigroup
//! (GroupFieldNames / ichi[g] in radiation_moments.cpp).
//! PHYSICAL-BOUNDARY INCOMING-ORDINATE CONDITION (2026-09-16, the fresh collapse-run cost probe).
//! The domain BC (Hydro DiodeBC) fills every FillGhost field, sn.u included, by a zero-gradient
//! copy. For an ordinate pointing INTO the domain that puts the interior value in its upwind
//! ghost, so nothing pins the intensity along an incoming ray: in an optically thin box the
//! streaming operator along that ray is singular up to the 1/(chat dt) and chi terms, and the
//! Krylov solve needed 720-1142 iterations per group per step from t = 0 (production GPU run).
//! radiation/sn_bc_incoming = outflow (historical, bit-identical default) | vacuum (zero
//! incoming) | planck (isotropic blackbody at sn_bc_tbg_K in every group). The Krylov work
//! vector k_y ALWAYS gets zero incoming: the boundary data is affine, carried by sn.u alone,
//! and the operator applied to corrections must stay homogeneous (the multipole-gravity lesson).
//! Signature is a plain function pointer so the Hydro BC can call it through a Param without a
//! header dependency: (block data, coarse, dir in {1,2,3}, inner side).
using SnBcFun = void (*)(std::shared_ptr<parthenon::MeshBlockData<Real>> &, bool, int, bool);
void SnApplyIncomingBC(std::shared_ptr<parthenon::MeshBlockData<Real>> &mbd, bool coarse,
                       int dir, bool inner);

inline std::string USolName() { return "sn.u"; }    //!< the iterate, nfa components, FillGhost

//----------------------------------------------------------------------------------------
//! RESTART PRE-CHECK, filled by main() before ProcessPackages runs.
//!
//! WHY A GLOBAL AND NOT A ParameterInput KEY. The answer to "does the restart file carry
//! sn.u, and with how many ordinates?" is only available from parthenon's RestartReader,
//! which is alive ONLY between ParthenonInitEnv() and ParthenonInitPackagesAndMesh()
//! (parthenon_manager.cpp:219-223 nulls it right after RestartPackages). Sn::Initialize runs
//! inside that window, from ProcessPackages, and receives nothing but `pin`. Writing the
//! answer into `pin` would (a) fabricate a <radiation> block in decks that have none and
//! (b) put a code-generated key into every output file's Input string, so a fresh run would
//! no longer be byte-for-byte reproducible from its own deck. main.cpp already owns
//! namespace-scope callback globals for exactly this kind of pre-package wiring
//! (Hydro::ProblemSourceFirstOrder & co.), so this follows the house pattern.
//!
//! Defaults describe a FRESH start, which is also what the host tests link against.
extern bool g_restart_is_restart;  //!< true iff this process was started with -r
extern bool g_restart_u_present;   //!< true iff the restart file has an "sn.u" dataset
extern int g_restart_u_ncomp;      //!< its component count, or -1 unknown / 0 not stated
extern int g_restart_ncycle;       //!< Info/NCycle of the restart file, -1 unknown (R-F1)
inline std::string UnName() { return "sn.un"; }     //!< state at step start (RHS anchor)
inline std::string MomName() { return "sn.mom"; }   //!< 4*n_group moments (J, H1, H2, H3)
inline std::string SrcName() { return "sn.src"; }   //!< n_group lagged EMISSIVITY eta_g
//! n_group ABSORPTION coefficients chi_a,g = rho*kappa_P,g. Stored rather than recomputed
//! because the matter coupling must charge the gas with the SAME chi_a the sweep emitted
//! with -- re-evaluating the opacity chain there would let the two drift apart in exactly the
//! way that makes an energy-conservation gate un-diagnosable. It is NOT rad.chi_g, which is
//! the Rosseland-plus-scattering EXTINCTION the sweep needs on its diagonal.
inline std::string ChiaName() { return "sn.chia"; }
//! n_group effective SCATTERING coefficients. Real scattering rho*kappa_s, PLUS the Rosseland
//! excess max(0, chi_R - chi_a): treating that difference as conservative scattering keeps the
//! attenuation length Rosseland (which is the physically right mean for diffusion) while
//! leaving the ENERGY exchange with the gas on chi_a alone, so total energy still conserves
//! exactly. Absorbing it into the sink instead would over-cool the radiation by (chi_R-chi_a)Er.
inline std::string ChisName() { return "sn.chis"; }
//! FLECK FACTOR f, one per cell. f = 1/(1 + c*dt*chi_a,P*dB/de_gas) linearises the emission
//! about T^n, which is what makes the matter coupling UNCONDITIONALLY STABLE instead of merely
//! conservative. The part of the absorption it removes, (1-f)*chi_a, is returned to the field
//! as isotropic effective scattering, so equilibrium Er = B is preserved EXACTLY (see the
//! algebra in SnSeedSource).
inline std::string FleckName() { return "sn.fleck"; }
//! R-F6 (radiation/sn_conserve_residual): per cell and group, sum_a w_a (A x - b)_a of the accepted iterate.
inline std::string ResMomName() { return "sn.resm"; }
//! R-F3 (radiation/sn_fc_scatter): nfa-wide BiCGSTAB work vectors for the coupled Fleck & Cummings solve.
inline std::string KcrName()  { return "sn.kc_r"; }
inline std::string Kcr0Name() { return "sn.kc_r0"; }
inline std::string KcpName()  { return "sn.kc_p"; }
inline std::string KcvName()  { return "sn.kc_v"; }
inline std::string KctName()  { return "sn.kc_t"; }
inline std::string KcyName()  { return "sn.kc_y"; }   //!< FillGhost

//! KRYLOV WORK VECTORS, allocated ONLY when radiation/sn_solver = bicgstab, and PER GROUP:
//! each is `nang` components, NOT nfa.
//!
//! This is the memory argument, and it is what decides whether the solver can run at production
//! block counts at all. A is exactly block-diagonal in the group index -- SnApplyA reads
//! chi_a(g), chi_s(g) and J_g(x), a moment over that group's ordinates alone -- so the nfa
//! system is n_group independent nang systems, solved one at a time. Together with forming `s`
//! in place in k_r (SnKrUpdate), that is 5 x nang + 1 x nang instead of 6 x nfa + 1 x nfa.
//!
//! MEASURED on the production restart (32^3 blocks + 2 ghosts, nfa = 96, n_group = 3):
//! 246.13 -> 70.65 MiB/block, a factor 3.48, against a base Sn footprint of 82.04 MiB/block.
//! They are still OPT-IN and not registered at all on the default path.
//!
//! `k_y` is the only one that is FillGhost: it is the only vector the operator and the
//! preconditioner are applied to. The rest are local.
inline std::string KrName()  { return "sn.k_r"; }   //!< residual, and s = r - alpha*v in place
inline std::string Kr0Name() { return "sn.k_r0"; }  //!< shadow residual r-hat-0
inline std::string KpName()  { return "sn.k_p"; }   //!< search direction
inline std::string KvName()  { return "sn.k_v"; }   //!< A*y
inline std::string KtName()  { return "sn.k_t"; }   //!< A*z
//! Preconditioned vector: holds y = M^-1 p, then is REUSED for z = M^-1 s. It is the only
//! Krylov vector the operator and the preconditioner are applied to, so it is the only one
//! that needs a halo. Reuse is legal because x is updated incrementally (x += alpha*y is
//! done before z is formed); if that ordering is ever changed, y must be split back out.
inline std::string KyName()  { return "sn.k_y"; }   //!< M^-1 p, then M^-1 s (FillGhost)
//! Zeroth angular moment J of whichever Krylov vector the operator is being applied to. ONE
//! component -- only the group currently being solved is live. It exists because A CONTAINS the
//! scattering coupling chi_s*J(x) rather than lagging it into b -- see SnApplyA.
inline std::string KjName()  { return "sn.k_j"; }   //!< J(x) for the operator apply
//! Outgoing FACE values, one nfa-component field per direction. Only allocated under
//! radiation/sn_spatial = diamond, because 3*nfa is 1.5x the base Sn footprint.
inline std::string FxName()  { return "sn.fx"; }
inline std::string FyName()  { return "sn.fy"; }
inline std::string FzName()  { return "sn.fz"; }

//! Device-resident angular grid, built once on the host at package init.
struct DeviceAngles {
  parthenon::ParArray1D<parthenon::Real> nx, ny, nz, w;
  int nang{0};
};

//! Runtime parameters for the solver.
struct SnParams {
  int nzeta{4};
  int npsi{8};
  int nang{32};
  int n_group{1};
  int nfa{32};
  int max_iterations{2000};
  parthenon::Real rel_tol{1.0e-6};
  bool red_black{true};
  //! Write the converged moments back into rad.Er / rad.Fr. radiation/sn_writeback.
  bool writeback{true};
  //! Apply the net absorption/emission exchange back onto the gas. radiation/sn_matter_coupling.
  //! Exactly conservative AND, through the Fleck linearisation, unconditionally stable.
  //! Still default OFF because it changes the gas, which no existing deck expects.
  bool matter_coupling{false};
  //! Apply the radiation FORCE d(rho v) = dt*chi_ext*Fr/c. radiation/sn_radiation_force.
  bool radiation_force{false};
  //! Use the DUAL-ENERGY internal energy (same selection as ConsToPrim: E - KE - ME where it is
  //! trustworthy, the advected e_int where it is not) in SnSeedSource and SnMatterCoupling, and
  //! write the coupling's result back to the dual-energy component. radiation/sn_dual_energy.
  //! Default OFF = bit-identical. Needs hydro/dual_energy.
  bool dual_energy{false};
  //! radiation/sn_skip_group_rel (audit P-09, default 0 = off): a group g > 0 whose ||b||_1 is
  //! below this fraction of the largest earlier group's is not iterated (its intensities keep
  //! their start-of-step values and it exchanges no energy with the gas this step). Group 2 of the
  //! collapse run has an emissivity of exactly 0 below ~2000 K and ||b|| ~ 1e-150 of group 0's.
  parthenon::Real skip_group_rel{0.0};
  //! radiation/sn_fallback (audit P-06, default false): after a group's solve BREAKS DOWN, goes
  //! non-finite, or stops unconverged above sn_fallback_rel, restore that group's start-of-step
  //! intensities and exclude it from this step's matter coupling instead of publishing and
  //! coupling a partial iterate.
  bool fallback{false};
  parthenon::Real fallback_rel{1.0e-4};
  //! Carry max(0, chi_R - chi_a) as conservative scattering. DIAGNOSTIC SWITCH, default true;
  //! false drops it so its share of the source-iteration count can be measured.
  bool rosseland_excess{true};
  //! Use the ordered downstream transport sweep instead of red-black Gauss-Seidel.
  //! radiation/sn_smoother = redblack | ordered | jacobi.
  bool ordered_sweep{false};
  //! Outer solver: false = the stationary (Richardson) iteration this package shipped with,
  //! true = matrix-free BiCGSTAB with the sweep as preconditioner. radiation/sn_solver.
  bool krylov{false};
  //! Preconditioner sweeps per BiCGSTAB application. radiation/sn_precond_sweeps.
  //! radiation/sn_spatial = diamond | wdd -- the weighted-diamond family that carries face
  //! values. `step` leaves this false and allocates no face fields.
  bool diamond{false};
  //! WDD weight w in I = (1-w) f_in + w f_out. radiation/sn_dd_weight. 0.5 = diamond (zero
  //! numerical diffusion, but the face reflection gain is exactly -1 and the solve STALLS --
  //! measured, see radiation_sn_sweep.hpp); 1.0 = step, bit-identical to sn_spatial = step.
  //! Stored as w; the kernels use rw = 1/w.
  parthenon::Real dd_w{0.5};
  //! DIAGNOSTIC. Set-to-zero fixup on the outgoing diamond face. It is what makes the
  //! diamond iteration NON-LINEAR, so it must be switchable to tell a fixup-induced stall
  //! from a sweep/residual operator mismatch. Turning it off is NOT a physics option --
  //! negative intensities poison the moments -- it exists to falsify a diagnosis.
  bool diamond_fixup{true};
  //! radiation/sn_spatial = sou -- second-order upwind, the LOCAL diffusion-limit fix that is
  //! valid under sn_solver = bicgstab. No face fields, no extra exchange; needs nghost >= 2.
  //! See SnSouDiagonal in radiation_sn_sweep.hpp for the scheme and its measured properties.
  bool sou{false};
  //! SOU blend kappa in [0, 1]: 1 = second-order upwind, 0 = step (bit-identical arithmetic),
  //! in between = (1 - kappa) of step's numerical diffusion. radiation/sn_sou_kappa.
  parthenon::Real sou_kappa{1.0};
  //! R-F10: radiation/sn_sou_limiter = positive -- lagged per-face positivity limiter (radiation_sn_soulim.cpp).
  bool sou_limit{false};
  //! R-F12: radiation/sn_precond_stencil = sou -- preconditioner sweep with the operator's own SOU stencil
  //! (radiation_sn_precond.cpp); default step.
  bool precond_sou{false};
  //! Speed work: radiation/sn_sweep_wavefront = 1 | 2 | 3 -- wavefront schedules of the preconditioner sweeps
  //! (radiation_sn_wavefront.cpp, bit-identical results); default 0 = serial per (block, ordinate).
  int wavefront{0};
  //! radiation/sn_sweep_group_batch (8-group speed work): with wavefront = 4 and the coupled solve, sweep all active
  //! groups in one launch (SnPrecondSweepWave4Groups); bit-identical. Default false.
  bool sweep_group_batch{false};
  bool wf5_scan{false};   //!< radiation/sn_wf5_scan: mode 5 row recurrence as a lane-parallel scan
  int wf5_team_size{0};   //!< radiation/sn_wf5_team_size (mode 5 exploration; 0 = AUTO)
  //! Under sou: ProlongateSharedLinear (unlimited) for sn.u and k_y ghosts at refinement
  //! jumps, so x(i-2) is not a copy of x(i-1). radiation/sn_sou_linear_prolong, default true.
  bool sou_linear_prolong{false};
  bool verify_operator{false};  //!< run the Stage-A operator gate at every solve setup
  //! radiation/sn_krylov_dot_scale (default false = bit-identical): BiCGSTAB dot products are
  //! formed on vectors scaled by 1/||b||_1 of the group. alpha, beta, omega are ratios of dots of
  //! equal degree, so the iteration is unchanged in exact arithmetic; what changes is that a group
  //! whose intensities sit below ~1e-154 (group 2 at T ~ 100 K: e^{-h nu/kT} ~ 1e-161) no longer
  //! has every squared component underflow to zero -> BREAKDOWN code 1 (DEV_LOG 2026-09-23).
  bool krylov_dot_scale{false};
  int precond_sweeps{1};
  //! Fleck implicitness parameter alpha in f = 1/(1 + alpha*c*dt*chi_a*dB/de). 1 = fully
  //! backward Euler (the stable choice); 0 disables the linearisation and recovers the
  //! explicit coupling, which is how the G11 stability gate demonstrates the difference.
  parthenon::Real fleck_alpha{1.0};
  //! Convergence test: true = TRUE RESIDUAL ||A x - b||/||b||, false = iterate increment.
  //! The increment is only defensible while the ordinates are uncoupled; with scattering or
  //! the Fleck effective-scattering term live it can stall and report success.
  bool residual_criterion{true};
  //! DIAGNOSTIC. radiation/sn_iter_verbose = N > 0 prints (iter, rel) every N iterations
  //! from rank 0. It exists because the only thing the solver reported was the FINAL
  //! verdict, so a solve that had not printed yet was indistinguishable between 'slow',
  //! 'stalled' and 'diverging' -- the exact ambiguity that made the first diamond-
  //! differencing failure undiagnosable. Default 0 = silent = bit-identical.
  int iter_verbose{0};
  //! DIAGNOSTIC. radiation/sn_tau_report = N > 0 prints the mesh-wide `tau_cell = chi*dx`
  //! distribution every N Sn solves, from rank 0. Default 0 = silent = bit-identical.
  //!
  //! WHY THIS EXISTS. `tau_cell` is the single parameter that decides BOTH open questions about
  //! this package's spatial discretisation, and until now it had been measured exactly once,
  //! offline, from one dump of a production run. Gate K4 measured step
  //! differencing's error as `D_meas/D_ana = 1 + 0.63*tau_cell`, so `tau_cell` IS the error bar
  //! on the radiative diffusion coefficient; and gates K2/K2-thin showed optical depth (not the
  //! transport Courant number) decides whether a sweep converges. A run that does not report it
  //! cannot tell you which regime it is in, which is how "the exposure is DEPTH" stayed an
  //! inference through the whole full-c campaign. It is a reduction over chi and dx only -- it
  //! touches no state and returns nothing to the solver.
  int tau_report{0};
};

//! Register fields + params. Called from Radiation::Initialize.
void Initialize(parthenon::ParameterInput *pin, parthenon::StateDescriptor *pkg);

//! Add the Sn solve to the task list.
//! R-F6, radiation/sn_conserve_residual (radiation_sn_conserve.cpp): residual moment + conservative correction.
parthenon::TaskStatus SnResidualMoment(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt);
// R-F10 (radiation_sn_soulim.cpp): lagged positivity-limited SOU
parthenon::TaskStatus SnSouLimLag(parthenon::MeshData<parthenon::Real> *md);
parthenon::TaskStatus SnPrecondSweepSou(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt, const int g,
                                        const std::string &rn, const std::string &yn);
parthenon::TaskStatus SnZeroGhosts(parthenon::MeshData<parthenon::Real> *md, const std::string &nm);
parthenon::TaskStatus SnZeroSkippedGroups(parthenon::MeshData<parthenon::Real> *md, const std::string &nm,
                                          const std::vector<int> &skip, const int nang);
parthenon::TaskStatus SnPrecondSweepWave(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt, const int g,
                                         const std::string &rn, const std::string &yn, const bool sou, const int mode);
//! radiation/sn_sweep_group_batch: the mode-4 sweep of every listed group in one launch (radiation_sn_wavefront.cpp).
parthenon::TaskStatus SnPrecondSweepWave4Groups(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                                                const std::string &rn, const std::string &yn, const bool sou,
                                                const std::vector<int> &groups);
//! radiation/sn_sweep_wavefront = 5: coalesced (warp-per-row) SOU preconditioner sweep, all listed groups in one launch.
parthenon::TaskStatus SnPrecondSweepWave5Groups(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                                                const std::string &rn, const std::string &yn, const bool sou,
                                                const std::vector<int> &groups);
parthenon::TaskStatus SnApplyASouLim(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt, const int g,
                                     const std::string &xn, const std::string &yn, const int mode);
parthenon::TaskStatus SnResidualSouLim(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                                       parthenon::Real *res, parthenon::Real *nrm, const std::string &store,
                                       parthenon::Real *tmax, const int gsel);
parthenon::TaskStatus SnResidualMomentSouLim(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt);
//! R-F6 (radiation_sn_criterion.cpp): x-independent norm sum|I^n/(chat dt) + eta| and the local criterion.
parthenon::TaskStatus SnBnormB(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt, const int g,
                               parthenon::Real *bsum, parthenon::Real *rsum);
parthenon::TaskStatus SnLocalViolations(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                                        const int g, const parthenon::Real tol, parthenon::Real *count);
//! R-F3 (radiation_sn_fc.cpp).
parthenon::TaskStatus SnFCCross(parthenon::MeshData<parthenon::Real> *md, const std::string &xn,
                                const std::string &yn, const parthenon::Real sign);
parthenon::TaskStatus SnFCTrueResidual(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                                       const std::string &scratch, parthenon::Real *res);
parthenon::TaskStatus SnLocalViolationsWide(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                                            const std::string &rn, const parthenon::Real tol,
                                            parthenon::Real *count);
//! R-F16 (radiation_sn_fc.cpp): second transcription of C(x) from sn.mom, for the coupled-path operator gate.
parthenon::TaskStatus SnFCCrossMom(parthenon::MeshData<parthenon::Real> *md, const std::string &yn,
                                   const parthenon::Real sign, parthenon::Real *csum, parthenon::Real *cscale);
//! R2/R3 test-only source definition (radiation_sn_testsrc.cpp): a box (or the |x1| < x0 slab) emitting
//! isotropically (ordinate < 0) or into ONE ordinate, at energy density rate q, while t < tend.
struct SnTestSourceDef {
  parthenon::Real q{0.0}, tend{0.0}, x0{0.5};
  bool slab{true};
  parthenon::Real box[6]{0, 0, 0, 0, 0, 0};
  int ordinate{-1};
};
//! R2 test-only volume radiation source, radiation/sn_test_source (radiation_sn_testsrc.cpp).
parthenon::TaskStatus SnTestSource(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt, const int partition);
parthenon::TaskStatus SnConserveCorrect(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                                        std::shared_ptr<std::vector<int>> off);
parthenon::TaskID AddSnTransport(parthenon::TaskList &tl, parthenon::TaskID prev,
                                 parthenon::Mesh *pmesh, int partition,
                                 std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md,
                                 std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_sn,
                                 std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_ky,
                                 const parthenon::Real dt);

} // namespace Sn
} // namespace Radiation

#endif // RADIATION_RADIATION_SN_HPP_
