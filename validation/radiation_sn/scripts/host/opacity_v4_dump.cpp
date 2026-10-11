// R-F11 host harness (2026-10-09): load a format-v4 opacity table through the PRODUCTION loaders
// (OpacityTable::Load, BuildGroupOpacityTableFromFile in radiation_opacity.hpp / radiation_groups.hpp, compiled
// host-only through src/radiation/tests/host_shims) and print kappa_P, kappa_R, the rho-dependent group multipliers
// and the dust weights at the (rho, T) points read from stdin ("rho_cgs T_K" per line), linear and log lookup.
// scripts/r7_v12.py compares them with an independent numpy reading of the same file.
// Build: g++ -O2 -std=c++17 -I<src>/radiation/tests/host_shims -I<src> opacity_v4_dump.cpp
#include <cstdio>
#include <string>

#include "radiation/radiation_groups.hpp"
#include "radiation/radiation_opacity.hpp"

using namespace Radiation;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: opacity_v4_dump <table.bin> <ng> < points\n");
    return 2;
  }
  const std::string path = argv[1];
  const int ng = std::atoi(argv[2]);
  OpacityTable lin, lg;
  lin.Load(path, 1.0, 1.0, 1.0);   // code units = cgs: rho_unit = T_unit = opacity_unit = 1
  lg.Load(path, 1.0, 1.0, 1.0);
  lg.EnableLogInterp();
  std::ifstream f(path, std::ios::binary);
  const auto hdr = ReadOpacityHeader(f, path);
  Real edges[MAX_GROUP + 1];
  for (int g = 0; g <= ng; ++g) edges[g] = static_cast<Real>(hdr.nu_edge[g]);
  GroupOpacityTable glin = BuildGroupOpacityTableFromFile(path, ng, edges, 1.0);
  GroupOpacityTable glg = BuildGroupOpacityTableFromFile(path, ng, edges, 1.0);
  glg.EnableLogInterp();
  std::printf("# rho T kP_lin kR_lin kP_log kR_log wdP wdR [mP_lin mR_lin mP_log mR_log]*ng  rho_dep=%d dustw=%d\n",
              glin.rho_dep_ ? 1 : 0, lin.has_dust_weight_ ? 1 : 0);
  double rho, T;
  while (std::scanf("%lf %lf", &rho, &T) == 2) {
    std::printf("%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e", rho, T, lin.KappaP(rho, T), lin.KappaR(rho, T),
                lg.KappaP(rho, T), lg.KappaR(rho, T), lin.DustWeightP(rho, T), lin.DustWeightR(rho, T));
    for (int g = 0; g < ng; ++g)
      std::printf(" %.17e %.17e %.17e %.17e", glin.PlanckMultRT(g, rho, T), glin.RossMultRT(g, rho, T),
                  glg.PlanckMultRT(g, rho, T), glg.RossMultRT(g, rho, T));
    std::printf("\n");
  }
  return 0;
}
