//========================================================================================
// AthenaPK - first-order flux correction (FOFC) x passive-scalar consistency gate.
// A strong MHD blast (p_in/p_out >> 1, low beta) in a uniform medium drives cells to
// negative density/pressure under the high-order fluxes, so FOFC replaces their face fluxes
// with LLF. Scalar 0 is a uniform concentration r = 1: any face whose MASS flux was replaced
// but whose SCALAR flux was not would move r away from 1 (r_new = (rho r + div F_s) /
// (rho + div F_rho)), so max|r0 - 1| is a direct test of flux consistency. Scalar 1 is a
// 0/1 step (r < scalar_radius) that must stay inside [0, 1]. Density floors also break
// r0 = 1 (they raise rho without touching rho*r), so the gate is only clean when no floor
// fires -- check the log. Use periodic BCs to keep the boundary-flux clamp out of the test.
//========================================================================================

#include <cmath>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "../hydro/hydro.hpp"
#include "pgen.hpp"

namespace fofc_scalar {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  const Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  const int nhydro = static_cast<int>(Hydro::GetNVars<Fluid::glmmhd>());
  const int nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
  PARTHENON_REQUIRE_THROWS(nscalars >= 2, "fofc_scalar needs <hydro> nscalars >= 2");

  const Real rho0 = pin->GetOrAddReal("problem/fofc_scalar", "rho0", 1.0);
  const Real p_out = pin->GetOrAddReal("problem/fofc_scalar", "p_out", 0.1);
  const Real p_in = pin->GetOrAddReal("problem/fofc_scalar", "p_in", 1.0e4);
  const Real radius = pin->GetOrAddReal("problem/fofc_scalar", "radius", 0.1);
  const Real s_radius = pin->GetOrAddReal("problem/fofc_scalar", "scalar_radius", 0.2);
  const Real b0 = pin->GetOrAddReal("problem/fofc_scalar", "b0", 1.0);

  auto &data = pmb->meshblock_data.Get();
  auto u = data->Get("cons").data.GetHostMirrorAndCopy();

  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
  auto &coords = pmb->coords;

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real r = std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) +
                                 SQR(coords.Xc<3>(k)));
        const Real p = (r < radius) ? p_in : p_out;
        u(IDN, k, j, i) = rho0;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = b0;
        u(IB2, k, j, i) = 0.0;
        u(IB3, k, j, i) = 0.0;
        u(IPS, k, j, i) = 0.0;
        u(IEN, k, j, i) = p / gm1 + 0.5 * SQR(b0);
        u(nhydro, k, j, i) = rho0 * 1.0;
        u(nhydro + 1, k, j, i) = (r < s_radius) ? rho0 : 0.0;
        for (int n = 2; n < nscalars; ++n) u(nhydro + n, k, j, i) = 0.0;
      }
    }
  }
  data->Get("cons").data.DeepCopy(u);
}

} // namespace fofc_scalar
