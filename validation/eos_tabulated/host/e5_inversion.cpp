// E5 -- inversion consistency of the tabulated EOS, on the HOST, with the REAL device header.
// validation_tests/eos_tabulated (2026-10-05). Compiles src/eos/eos_table.hpp of the validation
// worktree unchanged (KOKKOS_INLINE_FUNCTION -> inline via ./shims), so every number below is the
// arithmetic the kernels run (GPU pow/log10 may differ in the last ulp: E9).
//
//   build: see run_e5.sh       run: ./e5_inversion TABLE > results/tables/E5_inversion.txt
//
// Sections
//   M  monotonicity of P(rho, esp) in esp at every rho node (EspFromP's bisection needs it)
//   R  T -> e -> T round trip (EintFromRhoTk then TemperatureK), off-node sweep, by regime
//   B  e -> P -> e round trip through EspFromP at 20/30/40/60 bisections, by regime
//   E  edges: rho below the table (P in both interpolation modes vs the ideal-gas continuation),
//      P below P(rho, esp_min) (EspFromP), and what each one does to the clamp counters
//   F  the production pressure floor (pfloor = 1e-8 code) through InternalEnergyFromPres
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "eos_table.hpp"

using EOSTable::EosTable;
static const double RHO_U = 5.467e-19, V_U = 1.9e4; // FHC / default units (physical_units.hpp)

struct Stat {
  std::vector<double> v;
  void add(double x) { v.push_back(std::fabs(x)); }
  double q(double p) {
    if (v.empty()) return NAN;
    std::vector<double> w = v;
    size_t k = std::min(w.size() - 1, (size_t)(p * (w.size() - 1)));
    std::nth_element(w.begin(), w.begin() + k, w.end());
    return w[k];
  }
  double max() { return v.empty() ? NAN : *std::max_element(v.begin(), v.end()); }
};

static void drain(const EosTable &t, unsigned long long c[12]) { t.ReadAndResetClampCounts(c); }
static unsigned long long sum_state(unsigned long long c[12]) {
  unsigned long long s = 0;
  for (int q = 0; q < 6; ++q) s += c[q];
  return s;
}

// regimes by T [K]
static int regime(double T) {
  if (T < 100) return 0;
  if (T < 1000) return 1;
  if (T < 5000) return 2;  // H2 dissociation
  if (T < 3e4) return 3;   // H ionization
  return 4;
}
static const char *RNAME[5] = {"T<100K (molecular, rot. frozen)", "100-1e3 K (rot./vib.)",
                               "1e3-5e3 K (H2 dissociation)", "5e3-3e4 K (H ionization)",
                               ">3e4 K (He ionization)"};

int main(int argc, char **argv) {
  const std::string path = argc > 1 ? argv[1] : "eos_table_hires_v5_cold.bin";
  for (int cfg = 0; cfg < 4; ++cfg) {
    const int mode = cfg % 2;
    const bool extrap = (cfg >= 2); // hydro/eos_extrapolate (E-F5/E-F6 fix, 2026-10-05)
    EosTable t;
    t.p_over_rho_ = (mode == 1);
    t.extrapolate_ = extrap;
    t.Load(path, RHO_U, V_U);
    if (extrap) {
      // in-table lookups must be BITWISE those of the legacy path
      EosTable L;
      L.p_over_rho_ = (mode == 1);
      L.Load(path, RHO_U, V_U);
      long nd = 0, n = 0;
      for (int a = 0; a < 397; ++a)
        for (int b = 0; b < 601; ++b) {
          const double lr = t.lr0_ + 1e-6 + (t.nr_ - 1) * t.dlr_ * (a + 0.381966) / 397.0 * 0.999999;
          const double rho = std::pow(10.0, lr);
          const double T = std::pow(10.0, std::log10(0.1) + (t.lT0_ + (t.nT_ - 1) * t.dlT_ - 0.01 - std::log10(0.1)) * (b + 0.618034) / 601.0);
          const double e1 = t.EintFromRhoTk(rho, T), e0 = L.EintFromRhoTk(rho, T);
          const double esp = e0 / rho;
          if (esp <= t.esp_min_) continue;
          const double P1 = t.Pof(rho, esp), P0 = L.Pof(rho, esp);
          nd += (e1 != e0) + (P1 != P0) + (t.EspFromP(rho, P0) != L.EspFromP(rho, P0)) +
                (t.Cs2of(rho, esp) != L.Cs2of(rho, esp)) + (t.TemperatureK(rho, e0) != L.TemperatureK(rho, e0)) +
                (t.TemperatureKFromPres(rho, P0) != L.TemperatureKFromPres(rho, P0));
          n += 6;
        }
      printf("\n[X] eos_extrapolate=true vs legacy, IN-TABLE lookups (EintFromRhoTk, Pof, EspFromP, Cs2of, TemperatureK, TemperatureKFromPres): %ld of %ld differ\n", nd, n);
    }
    unsigned long long c[12];
    drain(t, c);
    double lrl, lrh, lel, leh, lTl, lTh;
    t.AxisRanges(lrl, lrh, lel, leh, lTl, lTh);
    const double lcgs = std::log10(RHO_U);
    printf("\n================ eos_interp = %s, eos_extrapolate = %s ================\n", mode ? "p_over_rho" : "p", extrap ? "true" : "false");
    printf("axes (code): log rho [%.4f, %.4f] (cgs [%.2f, %.2f]) d %.5f | log esp [%.4f, %.4f] d %.6f | log T [%.4f, %.4f] d %.6f\n",
           lrl, lrh, lrl + lcgs, lrh + lcgs, t.dlr_, lel, leh, t.dle_, lTl, lTh, t.dlT_);

    // ---- M: monotonicity of the stored P along esp ----
    if (!extrap) {
      long nviol = 0; double minrat = 1e300; int wi = -1, wj = -1;
      for (int i = 0; i < t.nr_; ++i)
        for (int j = 0; j + 1 < t.ne_; ++j) {
          const double r = t.P_(i, j + 1) / t.P_(i, j);
          if (!(r > 1.0)) ++nviol;
          if (r < minrat) { minrat = r; wi = i; wj = j; }
        }
      printf("[M] P(rho_i, esp_j+1)/P(rho_i, esp_j): violations of > 1: %ld of %d; min ratio %.8f at i=%d j=%d"
             " (log rho_cgs %.3f, T %.1f K)\n", nviol, t.nr_ * (t.ne_ - 1), minrat, wi, wj,
             t.lr0_ + wi * t.dlr_ + lcgs, std::pow(10.0, t.logT_(wi, wj)));
      long nT = 0;
      for (int i = 0; i < t.nr_; ++i)
        for (int j = 0; j + 1 < t.nT_; ++j)
          if (!(t.espT_(i, j + 1) > t.espT_(i, j))) ++nT;
      long nL = 0;
      for (int i = 0; i < t.nr_; ++i)
        for (int j = 0; j + 1 < t.ne_; ++j)
          if (!(t.logT_(i, j + 1) > t.logT_(i, j))) ++nL;
      printf("[M] esp(rho, T) strictly increasing in T: %ld violations; log T(rho, esp) increasing in esp: %ld violations\n", nT, nL);
    }

    // sweep grid: off-node (irrational offsets), interior of the rho axis
    const int NR = 797, NT = 1201;
    std::vector<double> LR(NR), LT(NT);
    for (int a = 0; a < NR; ++a) LR[a] = lrl + 1e-6 + (lrh - lrl - 2e-6) * (a + 0.381966) / NR;
    const double lTmin = std::log10(0.1), lTmax = lTh - 0.01;
    for (int b = 0; b < NT; ++b) LT[b] = lTmin + (lTmax - lTmin) * (b + 0.618034) / NT;

    // ---- R: T -> e -> T ----
    if (!extrap) {
      Stat all[5], prod, design;
      double worst = 0; double wr = 0, wT = 0;
      for (double lr : LR)
        for (double lT : LT) {
          const double rho = std::pow(10.0, lr), T = std::pow(10.0, lT);
          const double eint = t.EintFromRhoTk(rho, T);
          const double T2 = t.TemperatureK(rho, eint);
          const double err = T2 / T - 1.0;
          all[regime(T)].add(err);
          const double lrc = lr + lcgs;
          if (lrc <= -10.0 && T >= 1.0 && T <= 200.0) prod.add(err);
          if (lrc <= -1.0 && T >= 3.0) design.add(err);
          if (std::fabs(err) > worst) { worst = std::fabs(err); wr = lrc; wT = T; }
        }
      drain(t, c);
      printf("[R] T -> e -> T, %d x %d off-node points, T in [0.1 K, %.3g K]: |T'/T - 1|\n", NR, NT, std::pow(10.0, lTmax));
      for (int g = 0; g < 5; ++g)
        printf("      %-34s n %7zu  median %.2e  99.9%% %.2e  max %.2e\n", RNAME[g], all[g].v.size(), all[g].q(0.5), all[g].q(0.999), all[g].max());
      printf("      production footprint (rho<=1e-10, 1-200 K)  n %7zu  median %.2e  max %.2e\n", prod.v.size(), prod.q(0.5), prod.max());
      printf("      design range (rho<=0.1, T>=3 K)     n %7zu  median %.2e  max %.2e\n", design.v.size(), design.q(0.5), design.max());
      printf("      worst at log rho_cgs %.3f T %.4g K; clamp counters over the sweep: state %llu (T< %llu, esp< %llu)\n", wr, wT, sum_state(c), c[4], c[2]);
    }

    // ---- B: e -> P -> e through the bisection ----
    if (!extrap) {
      const int NB[4] = {20, 30, 40, 60};
      printf("[B] e -> P -> e via EspFromP; bound 0.5 * %.4f dex * 2^-n -> relative %.2e (n=20) %.2e (30)\n", leh - lel,
             std::pow(10.0, 0.5 * (leh - lel) * std::pow(2.0, -20)) - 1, std::pow(10.0, 0.5 * (leh - lel) * std::pow(2.0, -30)) - 1);
      for (int nb : NB) {
        t.n_bisect_ = nb;
        Stat de[5], dP[5], dTp;
        for (int a = 0; a < NR; a += 2)
          for (int b = 0; b < NT; b += 2) {
            const double rho = std::pow(10.0, LR[a]);
            const double esp = t.EintFromRhoTk(rho, std::pow(10.0, LT[b])) / rho;
            if (std::log10(esp) <= lel + 1e-9 || std::log10(esp) >= leh - 1e-9) continue;
            const double P = t.Pof(rho, esp);
            const double esp2 = t.EspFromP(rho, P);
            const int g = regime(std::pow(10.0, LT[b]));
            de[g].add(esp2 / esp - 1.0);
            dP[g].add(t.Pof(rho, esp2) / P - 1.0);
          }
        printf("    n_bisect %2d:", nb);
        for (int g = 0; g < 5; ++g) printf("  [%d] de max %.2e dP max %.2e", g, de[g].max(), dP[g].max());
        printf("\n");
      }
      t.n_bisect_ = 20;
      drain(t, c);
      printf("      clamp counters over the B sweeps: state %llu\n", sum_state(c));
    }

    // ---- E: edges ----
    {
      drain(t, c);
      printf("[E] rho BELOW the table (lower edge %.2e g/cc), T = 10 K state, esp from the edge row:\n", std::pow(10.0, lrl + lcgs));
      const double rho_e = std::pow(10.0, lrl);
      const double esp10 = t.EintFromRhoTk(rho_e * 1.0000001, 10.0) / (rho_e * 1.0000001);
      const double PoR_edge = t.Pof(rho_e * 1.0000001, esp10) / (rho_e * 1.0000001);
      for (double f : {1.0, 0.5, 0.1, 1e-2, 1e-4}) {
        drain(t, c);
        const double rho = rho_e * 1.0000001 * f;
        const double P = t.Pof(rho, esp10);
        drain(t, c);
        printf("      rho/rho_edge %-7g P %.5e  ideal-gas continuation rho*(P/rho)_edge %.5e  ratio %.6f  counters rho< %llu\n", f, P,
               rho * PoR_edge, P / (rho * PoR_edge), c[0]);
      }
      // is P/rho really rho-independent near the lower edge at 10 K (ideal molecular gas)?
      const double PoR2 = t.Pof(rho_e * 100.0, esp10) / (rho_e * 100.0);
      printf("      check: (P/rho) at 100 x rho_edge / at rho_edge, same esp: %.8f\n", PoR2 / PoR_edge);

      printf("[E] P BELOW P(rho, esp_min): EspFromP result and counters (production pfloor 1e-8 code):\n");
      for (double lrc : {-19.5, -17.0, -14.0, -11.0}) {
        const double rho = std::pow(10.0, lrc - lcgs);
        const double Pmin = t.Pof(rho, std::pow(10.0, lel));
        for (double P : {0.5 * Pmin, 1e-8}) {
          drain(t, c);
          const double esp = t.EspFromP(rho, P);
          drain(t, c);
          printf("      rho %.1e g/cc  P %.3e (P(esp_min) %.3e)  -> esp/esp_min - 1 = %.2e  P(esp)/P = %.3e  T(esp) %.4f K  counters: esp< %llu, state total %llu\n",
                 std::pow(10.0, lrc), P, Pmin, esp / std::pow(10.0, lel) - 1.0, t.Pof(rho, esp) / P, std::pow(10.0, t.LogTof(rho, esp)), c[2], sum_state(c));
        }
      }
      printf("[E] esp BELOW esp_min (a cell with e_int > 0 but T < %.3f K): PresFromRhoEint and counters:\n", std::pow(10.0, t.logT_(0, 0)));
      for (double s : {0.5, 1e-3, 1e-9}) {
        const double rho = std::pow(10.0, -17.0 - lcgs);
        drain(t, c);
        const double P = t.PresFromRhoEint(rho, s * std::pow(10.0, lel) * rho);
        drain(t, c);
        printf("      esp = %.0e esp_min: P %.4e = P(esp_min) x %.6f ; counters esp< %llu\n", s, P,
               P / t.Pof(rho, std::pow(10.0, lel) * 1.0000001), c[2]);
      }
    }

    // ---- F: production pressure floor through the EOS ----
    {
      printf("[F] production pressure floor pfloor = 1e-8 code through InternalEnergyFromPres = EintFromRhoPres:\n");
      for (double lrc : {-19.9, -18.0, -15.0, -12.0, -9.0, -5.0}) {
        const double rho = std::pow(10.0, lrc - lcgs);
        drain(t, c);
        const double e_fl = t.EintFromRhoPres(rho, 1e-8);
        const double Pfl = t.PresFromRhoEint(rho, e_fl);
        drain(t, c);
        printf("      rho %.1e g/cc: e_fl/rho = esp_min x (1 + %.2e); P(e_fl) = %.4e code = %.3e x pfloor; T(e_fl) %.4f K; state clamps %llu"
               "  -> cons-npfloor test (P <= pfloor(1+1e-9)) %s\n",
               std::pow(10.0, lrc), e_fl / rho / std::pow(10.0, lel) - 1.0, Pfl, Pfl / 1e-8, t.TemperatureK(rho, e_fl), sum_state(c),
               Pfl <= 1e-8 * (1 + 1e-9) ? "COUNTS" : "does NOT count");
      }
    }
  }
  return 0;
}
