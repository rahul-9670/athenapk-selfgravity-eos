//========================================================================================
// AthenaPK - radiation transport validation problem (M1, increment 2b).
// A Gaussian radiation-energy pulse on a static, optically-thin uniform gas, with the
// radiation flux set to a beam of reduced flux f (default f=1 => free-streaming).
// For f=1 the M1 signal speed is exactly chat, so the pulse must advect at chat
// without changing shape. Set reduced_flux=0 instead to test the diffusion-limit
// spreading speed chat/sqrt(3). Requires <physics> radiation = true.
//========================================================================================

#include <cmath>
#include <string>
#include <vector>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "../eos/eos_table.hpp"
#include "../radiation/radiation_groups.hpp" // GroupFieldNames (multigroup seeding)
#include "pgen.hpp"

namespace rad_pulse {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = gam - 1.0;

  // Static uniform background gas
  const Real rho0 = pin->GetOrAddReal("problem/rad_pulse", "rho0", 1.0);
  const Real p0 = pin->GetOrAddReal("problem/rad_pulse", "p0", 1.0);

  // Radiation pulse: Er = E0 + Eamp * exp(-(x-x0)^2 / 2 width^2); beam reduced flux fred
  const Real E0 = pin->GetOrAddReal("problem/rad_pulse", "E0", 1.0e-10);
  const Real Eamp = pin->GetOrAddReal("problem/rad_pulse", "Eamp", 1.0);
  const Real x0 = pin->GetOrAddReal("problem/rad_pulse", "x0", 0.0);
  const Real width = pin->GetOrAddReal("problem/rad_pulse", "width", 0.05);
  const Real fred = pin->GetOrAddReal("problem/rad_pulse", "reduced_flux", 1.0);
  // R8c (validation/radiation_sn, 2026-10-08): the Gaussian pulse's dimensionality -- 1 (default, every earlier
  // deck) = exp(-(x1 - x0)^2 / 2 w^2); 2 / 3 = exp(-r^2 / 2 w^2) with r^2 over (x1 - x0, x2[, x3]) (a line / point pulse).
  const int pulse_ndim = pin->GetOrAddInteger("problem/rad_pulse", "pulse_ndim", 1);
  PARTHENON_REQUIRE(pulse_ndim >= 1 && pulse_ndim <= 3, "problem/rad_pulse/pulse_ndim must be 1, 2 or 3");

  auto rad_pkg = pmb->packages.Get("radiation");
  const Real chat = rad_pkg->Param<Real>("chat");
  // Multigroup seeding: split the pulse EQUALLY across the n_group groups (each group gets
  // 1/n_group of E and F, i.e. the SAME reduced flux). Then sum_g E_g,0 = E (the gray IC),
  // and because every group has identical reduced flux the M1 closure is identical per group,
  // so under free-streaming each group evolves as (1/n_group)*gray and the group SUM tracks
  // the gray run cell-for-cell -- the multigroup==gray equivalence gate. n_group=1 => share=1
  // => group 0 carries the full pulse (bit-identical to the pre-multigroup pgen).
  const int n_group = rad_pkg->Param<int>("n_group");
  const Real share = 1.0 / static_cast<Real>(n_group);
  const bool thermal = pin->GetOrAddBoolean("problem/rad_pulse", "thermal_wave", false);
  // ISOBARIC variant of the thermal wave: hold PRESSURE uniform and vary DENSITY instead of
  // varying the internal energy at fixed density. Both give a sinusoidal T = p/rho, but only
  // this one has NO PRESSURE GRADIENT -- so the gas never accelerates and, with matter coupling
  // off, the temperature profile is frozen for the whole run.
  //
  // That is the point: it is the only configuration in this suite with a source that does not
  // decay. The Gaussian-pulse tests all let their gradients die (by relaxing to equilibrium
  // when coupling is on, or by homogenising in a periodic box when it is off), so a seam error
  // that saturates in them might be saturating because the DRIVER stopped, not because the
  // scheme bounds it. Here the driver never stops. See runs/sn_g6g/.
  const bool isobaric =
      pin->GetOrAddBoolean("problem/rad_pulse", "thermal_isobaric", false);
  // TOP-HAT instead of a cosine, and this exists because of a MEASURED FLAW in the cosine
  // version. A cell-centred midpoint sample of a periodic cosine is spectrally accurate, so the
  // volume-weighted <a_R T^4> came out IDENTICAL on a 16^3 and a 32^3 uniform mesh -- 3.5e-16
  // apart. A MIXED (AMR) mesh is not a uniform sampling and loses that property: its <T^4>
  // differed from both uniform arms by 1.45e-03. The AMR arm was therefore solving a DIFFERENT
  // SOURCE PROBLEM, and a third of the "coarse-fine seam error" it appeared to show was the
  // initial condition, not transport.
  //
  // A top-hat whose edges sit on COARSE cell faces is represented EXACTLY by every mesh in the
  // hierarchy -- no cell ever straddles the jump -- so <T^4> is identical to machine precision
  // in all arms and any remaining difference is the solver. `thermal_tophat_halfwidth` must be
  // an integer multiple of the coarse dx, and the jump must be kept away from the refinement
  // boundary so the two effects are not conflated.
  const bool tophat = pin->GetOrAddBoolean("problem/rad_pulse", "thermal_tophat", false);
  const Real tophat_hw =
      pin->GetOrAddReal("problem/rad_pulse", "thermal_tophat_halfwidth", 0.125);
  // ITEM #4 (docs/RAD_ACCURACY_ITEMS_2026-09-14.md): the top-hat's dimensionality. 1 (the
  // default, bit-identical to every existing deck) is the slab |x| < hw used by the seam
  // tests; 3 makes it the CUBE max(|x|,|y|,|z|) < hw -- a COMPACT hot source in a cold medium,
  // which is what a ray-effect measurement needs (a slab is a 1-D problem and shows none).
  // Still mesh-exact when hw is a multiple of the coarse dx. 2 = square column (|x|,|y|).
  const int tophat_ndim =
      pin->GetOrAddInteger("problem/rad_pulse", "thermal_tophat_ndim", 1);
  PARTHENON_REQUIRE(tophat_ndim >= 1 && tophat_ndim <= 3,
                    "problem/rad_pulse/thermal_tophat_ndim must be 1, 2 or 3");
  const Real Tunit = rad_pkg->Param<Real>("T_unit");
  const Real T0 = pin->GetOrAddReal("problem/rad_pulse", "temperature_K", Tunit*p0/rho0)/Tunit;
  const Real Tamp = pin->GetOrAddReal("problem/rad_pulse", "thermal_amplitude", 1e-4);
  const Real Erfactor = pin->GetOrAddReal("problem/rad_pulse", "thermal_radiation_factor", 1.0);
  // R1 (validation/radiation_sn, 2026-10-08): radiation spectrum = Planck at its OWN temperature
  // thermal_trad_K (uniform), instead of the local gas T, so a uniform box can start with T_rad != T_gas.
  // Default -1 = off (the gas T, as before).
  const Real trad_K = pin->GetOrAddReal("problem/rad_pulse", "thermal_trad_K", -1.0);
  // R5 (validation/radiation_sn, 2026-10-08): density INSIDE the top-hat = rho0 * thermal_tophat_rho_factor, so the
  // hot region can be a dense, optically thick "core" in a dilute envelope (not pressure-balanced; R5 runs a few steps,
  // short against the sound crossing). Default 1 = the uniform density of every earlier deck (x 1.0 is exact).
  const Real tophat_rho_f = pin->GetOrAddReal("problem/rad_pulse", "thermal_tophat_rho_factor", 1.0);
  PARTHENON_REQUIRE(tophat_rho_f > 0.0, "problem/rad_pulse/thermal_tophat_rho_factor must be > 0");
  PARTHENON_REQUIRE(tophat_rho_f == 1.0 || (tophat && !isobaric),
                    "problem/rad_pulse/thermal_tophat_rho_factor needs thermal_tophat = true and thermal_isobaric = false");
  const int modes[3]={pin->GetOrAddInteger("problem/rad_pulse","wave_n1",1),
                     pin->GetOrAddInteger("problem/rad_pulse","wave_n2",0),
                     pin->GetOrAddInteger("problem/rad_pulse","wave_n3",0)};
  const bool h2=rad_pkg->Param<bool>("use_h2diss");
  const auto eos=rad_pkg->Param<EOSTable::EosTable>("eos_tab");
  const auto groups=rad_pkg->Param<Radiation::RadGroups>("groups");
  const Real arad=rad_pkg->Param<Real>("arad");
  Real direction[3]={pin->GetOrAddReal("problem/rad_pulse","beam_dir1",1.0),
                     pin->GetOrAddReal("problem/rad_pulse","beam_dir2",0.0),
                     pin->GetOrAddReal("problem/rad_pulse","beam_dir3",0.0)};
  const Real dn=std::sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
  PARTHENON_REQUIRE(dn>0,"rad_pulse beam direction must be nonzero");
  for(int d=0;d<3;++d)direction[d]/=dn;
  const Real lengths[3]={pin->GetReal("parthenon/mesh","x1max")-pin->GetReal("parthenon/mesh","x1min"),
                         pin->GetReal("parthenon/mesh","x2max")-pin->GetReal("parthenon/mesh","x2min"),
                         pin->GetReal("parthenon/mesh","x3max")-pin->GetReal("parthenon/mesh","x3min")};

  auto &data = pmb->meshblock_data.Get();
  auto u = data->Get("cons").data.GetHostMirrorAndCopy();
  // Per-group host mirrors of (Er, Fr1, Fr2, Fr3).
  std::vector<decltype(data->Get("rad.Er").data.GetHostMirrorAndCopy())> gEr, gFx, gFy, gFz;
  for (int g = 0; g < n_group; ++g) {
    const auto nm = Radiation::GroupFieldNames(g);
    gEr.push_back(data->Get(nm[0]).data.GetHostMirrorAndCopy());
    gFx.push_back(data->Get(nm[1]).data.GetHostMirrorAndCopy());
    gFy.push_back(data->Get(nm[2]).data.GetHostMirrorAndCopy());
    gFz.push_back(data->Get(nm[3]).data.GetHostMirrorAndCopy());
  }

  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);
  auto &coords = pmb->coords;

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real x1 = coords.Xc<1>(i);
        // static uniform gas, v = 0
        u(IDN, k, j, i) = rho0;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = p0 / gm1;
        if (mhd) {
          u(IB1, k, j, i) = 0.0;
          u(IB2, k, j, i) = 0.0;
          u(IB3, k, j, i) = 0.0;
          u(IPS, k, j, i) = 0.0;
        }
        // radiation pulse + beam, split equally across groups
        const Real arg = (x1 - x0) / width;
        Real arg2 = arg * arg;
        if (pulse_ndim >= 2) arg2 += (coords.Xc<2>(j) / width) * (coords.Xc<2>(j) / width);
        if (pulse_ndim >= 3) arg2 += (coords.Xc<3>(k) / width) * (coords.Xc<3>(k) / width);
        const Real E = share * (E0 + Eamp * std::exp(-0.5 * arg2));
        for (int g = 0; g < n_group; ++g) {
          gEr[g](0, k, j, i) = E;
          gFx[g](0, k, j, i) = direction[0] * fred * chat * E; // reduced flux fred in +x1
          gFy[g](0, k, j, i) = direction[1] * fred * chat * E;
          gFz[g](0, k, j, i) = direction[2] * fred * chat * E;
        }
        if(thermal) {
          const Real pos[3]={coords.Xc<1>(i),coords.Xc<2>(j),coords.Xc<3>(k)};
          Real phase=0;for(int d=0;d<3;++d)phase+=2*M_PI*modes[d]*pos[d]/lengths[d];
          // Chebyshev distance over the first tophat_ndim coordinates; ndim = 1 is exactly
          // the original |pos[0]| test.
          Real cheb = std::abs(pos[0]);
          for (int d = 1; d < tophat_ndim; ++d) cheb = std::max(cheb, std::abs(pos[d]));
          const Real T = tophat ? (cheb < tophat_hw ? T0 * (1 + Tamp)
                                                    : T0 * (1 - Tamp))
                                : T0 * (1 + Tamp * std::cos(phase));
          // Isobaric: rho = p/T, so T = p/rho carries the profile while p stays uniform.
          // WRITTEN IN TERMS OF T, not of the cosine. The first version hard-coded
          // rho0/(1+Tamp*cos), which is correct for the cosine and WRONG for any other
          // profile: with the top-hat it left density on the cosine while T was the top-hat,
          // so p = rho*T varied by 56% and the gas was NOT isobaric at all. Caught by checking
          // the pressure spread on the t=0 dump before spending the run, not by the run
          // failing -- it would have produced a plausible number with a moving gas.
          // With p0 = rho0*T0 (the default T0 is exactly p0/rho0) this reduces to the old
          // expression for the cosine, so that path is unchanged.
          const Real rho_l = isobaric ? rho0 * T0 / T : (tophat && cheb < tophat_hw ? rho0 * tophat_rho_f : rho0);
          u(IDN,k,j,i)=rho_l;
          u(IEN,k,j,i)=h2?eos.EintFromRhoTkHost(rho_l,T*Tunit):rho_l*T/gm1;
          for(int g=0;g<n_group;++g){
            auto band=[&](Real tt){return arad*tt*tt*tt*tt*groups.PlanckFraction(g,tt*Tunit);};
            gEr[g](0,k,j,i)=Erfactor*band(trad_K > 0.0 ? trad_K/Tunit : T);
            (void)rho_l;
            // Zero initial flux is physical and avoids host access to device opacity tables.
            Real f[3]={0,0,0};
            gFx[g](0,k,j,i)=f[0];gFy[g](0,k,j,i)=f[1];gFz[g](0,k,j,i)=f[2];
          }
        }
      }
    }
  }
  data->Get("cons").data.DeepCopy(u);
  for (int g = 0; g < n_group; ++g) {
    const auto nm = Radiation::GroupFieldNames(g);
    data->Get(nm[0]).data.DeepCopy(gEr[g]);
    data->Get(nm[1]).data.DeepCopy(gFx[g]);
    data->Get(nm[2]).data.DeepCopy(gFy[g]);
    data->Get(nm[3]).data.DeepCopy(gFz[g]);
  }
}

} // namespace rad_pulse
