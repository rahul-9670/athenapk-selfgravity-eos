//========================================================================================
// AthenaPK - Sn transport: GRAY (linear multifrequency-gray) diffusion synthetic acceleration of the coupled Fleck &
// Cummings solve. R-F12 (validation/radiation_sn, 2026-10-09). Key radiation/sn_gray_dsa (default false: no
// field, no solver, no task).
//
// WHY. With the Fleck factor f << 1 in a hot, optically thick core the coupled operator's effective scattering ratio
// is ~1 - f (~1 - 4e-5 at 1000 K): its slow error modes are isotropic, spatially smooth (diffusive) and GRAY in
// frequency, because the Fleck & Cummings term re-emits every absorbed photon with one spectrum (rank one in the
// group index). The ordered sweep inverts only streaming + collision; BiCGSTAB then needs O(1/sqrt(f)) iterations
// (R5 1000 K: 421-454 with the SOU-consistent sweep, ~1000 and the 4000 cap with 8-18 groups).
// WHAT. After the sweep y = T^-1 r (sn.kc_y), the error of y is e = A^-1 S y. Its slow part is approximated by
// e_g,a ~ xi_g eps / 4pi (isotropic, spectrum xi_g), with eps the solution of ONE gray diffusion problem
//     sigbar eps - div( Dbar grad eps ) = sum_g chis_g phi_g(y)          (phi = sum_a w_a I, sum_a w_a = 4 pi)
//     xi_g   = (w_g / (rdt + chi_a,g)) / sum_k (w_k / (rdt + chi_a,k))   (infinite-medium slow mode; w_g = eta_g/sum eta)
//     sigbar = sum_g (rdt + chia_g) xi_g                                  (chia = f chi_a, the effective absorption)
//     Dbar   = sum_g xi_g / (3 (chia_g + chis_g + rdt))                   (P1 with the time term)
// The source is exact: sum_g of the angular integral of the FC scattering S y is sum_g chis_g phi_g(y) (the
// cross-group term only moves energy between groups). Solved scaled by chat dt (diagonal >= 1), cell-centred 7-point
// stencil, harmonic face D, Parthenon coarse-fine flux correction, Marshak vacuum faces; Parthenon BiCGSTAB
// (Diagonal or Multigrid preconditioner, block radiation/sn_gray_dsa_solver). The correction changes only the
// PRECONDITIONER: the outer BiCGSTAB converges on the true operator.
// Measured first (speed_work/rf12/proto1d.py, 1-D R5 problem, Semenov v12 groups): SOU sweep 156 -> 32 iterations
// at 1000 K (3 groups), 254 -> 69 (10 groups), 64 -> 33 at 300 K; insensitive to the inner tolerance down to 1e-6.
// History: the per-group FLD-based sn_dsa (2026-09-17, removed 09-24) made the production AMR mesh worse (322 -> 624);
// this one differs in being gray (the FC slow mode), using the transport chi + rdt in D without a dx^2 floor, and
// sitting after the SOU-consistent sweep (R-F12 partial fix).
//========================================================================================
#ifndef RADIATION_RADIATION_SN_GRAYDSA_HPP_
#define RADIATION_RADIATION_SN_GRAYDSA_HPP_

#include <memory>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

namespace Radiation {
namespace Sn {
namespace GrayDsa {

#define SNGD_VARIABLE(varname, ncomp)                                                    \
  struct varname : public parthenon::variable_names::base_t<false, ncomp> {               \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false, ncomp>(std::forward<Ts>(args)...) {}   \
    static std::string name() { return "sn.gd_" #varname; }                              \
  }
SNGD_VARIABLE(u, 1);     //!< the solved gray correction eps (solver variable)
SNGD_VARIABLE(sig, 1);   //!< chat dt sigbar (>= 1), GMGRestrict
SNGD_VARIABLE(dc, 1);    //!< chat dt Dbar, cell-centred, GMGRestrict
SNGD_VARIABLE(D, 3);     //!< face D (lower face per direction), rebuilt per level
#undef SNGD_VARIABLE
inline std::string XiName() { return "sn.gd_xi"; }   //!< xi_g, n_group components

//! Register fields, params and the diffusion solver when radiation/sn_gray_dsa = true (no-op otherwise).
void Initialize(parthenon::ParameterInput *pin, parthenon::StateDescriptor *pkg, const int n_group);

//! Append the correction to a preconditioner apply: y (sn.kc_y, ghosts valid) += xi eps / 4pi on every
//! non-skipped group, then the y halo. `skip` = the coupled solve's frozen-group flags (read at task run time).
parthenon::TaskID AddCorrection(parthenon::TaskList &tl, parthenon::TaskID t,
                                std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md,
                                std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_ky,
                                const parthenon::Real dt, const bool ml, parthenon::Mesh *pmesh,
                                const int partition, const std::vector<int> *skip);

//! radiation/sn_gray_dsa_debug: copy the preconditioner input rn into sn.gd_dbg_in.
parthenon::TaskStatus DbgCopyIn(parthenon::MeshData<parthenon::Real> *md, const std::string &rn);

//! Per-step tally of the inner solves (rank 0 prints, then resets).
parthenon::TaskStatus Report(parthenon::StateDescriptor *pkg);

//! The equations_t contract for Parthenon's BiCGSTABSolver / MGSolver: A = sig - div(D grad).
class GrayDsaEquation {
 public:
  using IndependentVars = parthenon::TypeList<u>;
  GrayDsaEquation() = default;
  GrayDsaEquation(parthenon::ParameterInput *, const std::string &) {}
  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_out);
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<parthenon::Real>> &md_diag);
};

} // namespace GrayDsa
} // namespace Sn
} // namespace Radiation

#endif // RADIATION_RADIATION_SN_GRAYDSA_HPP_
