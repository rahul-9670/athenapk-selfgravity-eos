// E7 fix check (2026-10-05): hydro/eos_rho_interp = cubic on the REAL eos_table.hpp.
//  (1) P strictly increasing in esp along a fine sweep (EspFromP's bisection needs it), both interpolations
//  (2) jump of dlnP/dln rho|_e across rho nodes: linear vs cubic (cubic must be C1 -> round-off)
//  (3) values at fixed points for the Python cross-check (stdout, "PT" lines)
//  (4) EspFromP(Pof(e)) round trip under cubic
#include <cmath>
#include <cstdio>
#include "eos_table.hpp"
using EOSTable::EosTable;
int main(int argc, char **argv) {
  const char *path = argv[1];
  for (int cub = 0; cub < 2; ++cub) {
    EosTable t; t.p_over_rho_ = true; t.extrapolate_ = true; t.p_cubic_ = (cub == 1);
    t.Load(path, 5.467e-19, 1.9e4);
    long nv = 0, n = 0; double worst = 1e300;
    for (int a = 0; a < 2000; ++a) {
      const double lr = t.lr0_ + (t.nr_ - 1) * t.dlr_ * (a + 0.5) / 2000.0;
      double Pp = -1e300;
      for (int b = 0; b < 20000; ++b) {
        const double le = t.le0_ + (t.ne_ - 1) * t.dle_ * (b + 0.5) / 20000.0;
        const double T = std::pow(10.0, t.LogTof(std::pow(10.0, lr), std::pow(10.0, le)));
        if (T >= 2.9e5) break; // generator T cap plateau (E5)
        const double P = t.Pof(std::pow(10.0, lr), std::pow(10.0, le));
        ++n; if (!(P > Pp)) ++nv; if (Pp > 0) worst = std::fmin(worst, P / Pp);
        Pp = P;
      }
    }
    printf("[%s] (1) P increasing in esp: %ld violations of %ld (T < 2.9e5 K); min ratio %.12f\n", cub ? "cubic" : "linear", nv, n, worst);
    double jmax = 0, jsum = 0; int nj = 0;
    for (int i = 2; i < t.nr_ - 3; i += 1)
      for (double le : {0.0, 0.8, 1.6, 2.4, 3.1, 3.6, 4.2}) { // code log esp: 10 K .. ~1e5 K
        const double lr = t.lr0_ + i * t.dlr_, h = 1e-6;
        auto lP = [&](double x) { return std::log(t.Pof(std::pow(10.0, x), std::pow(10.0, le))); };
        const double sl = (lP(lr) - lP(lr - h)) / h, sr = (lP(lr + h) - lP(lr)) / h;
        const double j = std::fabs(sr - sl);
        jmax = std::fmax(jmax, j); jsum += j; ++nj;
      }
    printf("[%s] (2) |jump of dlnP/dlog10rho| at rho nodes: max %.3e, mean %.3e over %d node x esp samples\n", cub ? "cubic" : "linear", jmax, jsum / nj, nj);
    {
      double jm = 0, js = 0; int nn = 0;
      for (int jn = 5; jn < t.ne_ - 5; jn += 3)
        for (double lrc : {-19.0, -16.0, -13.0, -10.3, -9.27, -8.2, -6.0, -3.0}) {
          const double lr = lrc - std::log10(5.467e-19) + 0.0123, le = t.le0_ + jn * t.dle_, h = 1e-7;
          auto lP = [&](double y) { return std::log(t.Pof(std::pow(10.0, lr), std::pow(10.0, y))); };
          const double sl = (lP(le) - lP(le - h)) / h, sr = (lP(le + h) - lP(le)) / h;
          if (!std::isfinite(sl) || !std::isfinite(sr)) continue;
          const double j = std::fabs(sr - sl);
          jm = std::fmax(jm, j); js += j; ++nn;
        }
      printf("[%s] (2e) |jump of dlnP/dlog10e| at e nodes: max %.3e, mean %.3e over %d samples\n", cub ? "cubic" : "linear", jm, js / nn, nn);
    }
    double rt = 0;
    for (int a = 0; a < 300; ++a)
      for (int b = 0; b < 300; ++b) {
        const double rho = std::pow(10.0, t.lr0_ + 0.3 + 19.5 * a / 300.0 + 0.0123);
        const double esp = std::pow(10.0, t.le0_ + 0.01 + 7.0 * b / 300.0 + 0.0071);
        const double e2 = t.EspFromP(rho, t.Pof(rho, esp));
        rt = std::fmax(rt, std::fabs(e2 / esp - 1));
      }
    printf("[%s] (4) EspFromP(Pof(e)) max rel err %.2e (bound 8.08e-6)\n", cub ? "cubic" : "linear", rt);
    for (double lrc : {-17.3, -12.71, -9.33, -8.05}) for (double le : {0.5, 2.37, 3.9})
      printf("PT %d %.17g %.17g %.17g\n", cub, std::pow(10.0, lrc) / 5.467e-19, std::pow(10.0, le), t.Pof(std::pow(10.0, lrc) / 5.467e-19, std::pow(10.0, le)));
  }
}
