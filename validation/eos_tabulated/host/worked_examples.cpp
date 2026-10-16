// Worked examples of the tabulated EOS (EOS_VALIDATION.md Sec. W), computed by the REAL device header
// src/eos/eos_table.hpp compiled on the host (KOKKOS_INLINE_FUNCTION -> inline via ./shims), exactly as
// e5_inversion.cpp does. Every number printed is the arithmetic the kernels run on the CPU (the GPU may differ in the
// last ulp of pow/log10: E9).
//
//   build + run: see run_worked.sh        output: results/tables/W_worked_examples.txt
//
// Keys are the code defaults (hydro.cpp): eos_interp = p_over_rho, eos_extrapolate = true, eos_bisections = 20.
// Units are the defaults of src/units/physical_units.hpp (rho_unit 5.467e-19 g/cm^3, v_unit 1.9e4 cm/s,
// mu_thermal 2.29); the constants below are copied from that file.
#include <cmath>
#include <cstdio>
#include <string>

#include "eos_table.hpp"

using EOSTable::EosTable;
static const double RHO_U = 5.467e-19, V_U = 1.9e4, MU_TH = 2.29;
static const double M_H = 1.6726219e-24, K_B = 1.380649e-16;

static void counters(const EosTable &t, const char *tag) {
  unsigned long long c[12];
  t.ReadAndResetClampCounts(c);
  std::printf("    clamp counters after %s: rho< %llu rho> %llu esp< %llu esp> %llu T< %llu T> %llu"
              " | probe %llu %llu %llu %llu %llu %llu\n",
              tag, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]);
}

// Print the bilinear stencil a lookup uses (same index/weight arithmetic as EosTable::bilin).
static void stencil(const char *name, const parthenon::ParArray2D<double> &A, int n1, int n2, double x0,
                    double dx, double y0, double dy, double x, double y) {
  const double fi = (x - x0) / dx, fj = (y - y0) / dy;
  int i = static_cast<int>(fi), j = static_cast<int>(fj);
  if (i < 0) i = 0;
  if (i > n1 - 2) i = n1 - 2;
  if (j < 0) j = 0;
  if (j > n2 - 2) j = n2 - 2;
  double ti = fi - i, tj = fj - j;
  ti = ti < 0 ? 0 : (ti > 1 ? 1 : ti);
  tj = tj < 0 ? 0 : (tj > 1 ? 1 : tj);
  std::printf("    %s stencil: fi = %.6f -> i = %d, ti = %.6f;  fj = %.6f -> j = %d, tj = %.6f\n", name, fi, i,
              ti, fj, j, tj);
  std::printf("      A(i,j) = %.10e  A(i+1,j) = %.10e\n      A(i,j+1) = %.10e  A(i+1,j+1) = %.10e\n", A(i, j),
              A(i + 1, j), A(i, j + 1), A(i + 1, j + 1));
}

static void state(const EosTable &t, const char *title, double rho_code, double TK) {
  std::printf("\n== %s: rho = %.6e code (%.6e g/cm^3), T = %.6g K\n", title, rho_code, rho_code * RHO_U, TK);
  const double lr = std::log10(rho_code);
  stencil("(rho,T) grid espT", t.espT_, t.nr_, t.nT_, t.lr0_, t.dlr_, t.lT0_, t.dlT_, lr, std::log10(TK));
  const double eint = t.EintFromRhoTk(rho_code, TK); // internal-energy DENSITY [code]
  const double esp = eint / rho_code;
  std::printf("    EintFromRhoTk -> e_int = %.10e code;  esp = e_int/rho = %.10e code = %.6e erg/g\n", eint, esp,
              esp * V_U * V_U);
  stencil("(rho,esp) grid P/rho_node", t.P_, t.nr_, t.ne_, t.lr0_, t.dlr_, t.le0_, t.dle_, lr, std::log10(esp));
  const double P = t.Pof(rho_code, esp);
  const double cs2 = t.Cs2of(rho_code, esp);
  const double TKback = std::pow(10.0, t.LogTof(rho_code, esp));
  std::printf("    Pof   -> P = %.10e code = %.6e erg/cm^3;  P/rho = %.10f code\n", P, P * RHO_U * V_U * V_U,
              P / rho_code);
  std::printf("    Cs2of -> cs^2 = %.10e code (cs = %.6e cm/s);  Gamma1 = rho cs^2 / P = %.6f\n", cs2,
              std::sqrt(cs2) * V_U, rho_code * cs2 / P);
  std::printf("    e_int / P = %.6f  (ideal gas: 1/(gamma-1));  1 + P/e_int = %.6f\n", eint / P, 1.0 + P / eint);
  std::printf("    LogTof (T -> e -> T round trip) -> T = %.10g K, rel. diff %.3e\n", TKback, TKback / TK - 1.0);
  std::printf("    mu implied by P = rho k T / (mu m_H): mu = %.6f\n",
              rho_code * RHO_U * K_B * TK / (M_H * P * RHO_U * V_U * V_U));
  for (int nb : {20, 30, 60}) {
    EosTable u = t;
    u.n_bisect_ = nb;
    const double e2 = u.EspFromP(rho_code, P);
    std::printf("    EspFromP(rho, P) with %2d bisections -> esp = %.12e, rel. error %.3e\n", nb, e2, e2 / esp - 1.0);
  }
  std::printf("    AsqFromRhoPres(rho, P) = %.10e (= Cs2of at the bisected esp)\n", t.AsqFromRhoPres(rho_code, P));
  counters(t, "this state");
}

int main(int argc, char **argv) {
  const std::string path = argc > 1 ? argv[1] : "eos_table_hires_v5_cold_rho26.bin";
  const double T_unit = MU_TH * M_H * V_U * V_U / K_B;
  std::printf("# Worked examples, table %s\n", path.c_str());
  std::printf("# units: rho_unit %.4e g/cm^3, v_unit %.4e cm/s, esp_unit = v^2 = %.6e erg/g, "
              "P_unit = rho v^2 = %.6e erg/cm^3, temperature_unit = mu_th m_H v^2/k_B = %.6f K\n",
              RHO_U, V_U, V_U * V_U, RHO_U * V_U * V_U, T_unit);

  EosTable t;
  t.p_over_rho_ = true;  // hydro/eos_interp default
  t.extrapolate_ = true; // hydro/eos_extrapolate default
  t.Load(path, RHO_U, V_U);
  t.n_bisect_ = 20;      // hydro/eos_bisections default
  double lrl, lrh, lel, leh, lTl, lTh;
  t.AxisRanges(lrl, lrh, lel, leh, lTl, lTh);
  std::printf("# grid (code units after Load): nr %d ne %d nT %d\n", t.nr_, t.ne_, t.nT_);
  std::printf("#   log10 rho_code [%.6f, %.6f] step %.6f  (cgs: [%.4f, %.4f])\n", lrl, lrh, t.dlr_,
              lrl + std::log10(RHO_U), lrh + std::log10(RHO_U));
  std::printf("#   log10 esp_code [%.6f, %.6f] step %.6f  (cgs: [%.4f, %.4f])\n", lel, leh, t.dle_,
              lel + std::log10(V_U * V_U), leh + std::log10(V_U * V_U));
  std::printf("#   log10 T[K]     [%.6f, %.6f] step %.6f\n", lTl, lTh, t.dlT_);
  std::printf("#   rho_min %.6e  esp_min %.6e  T_min %.6e K\n", t.rho_min_, t.esp_min_, t.T_min_);
  std::printf("#   bisection bound 0.5 ln10 (log-esp range) 2^-20 = %.4e\n",
              0.5 * std::log(10.0) * (leh - lel) * std::pow(2.0, -20));

  // ---- Example 1: the reference state of the code units (T_code = 1) ----
  state(t, "W1 reference state (rho_code = 1, T = temperature_unit)", 1.0, T_unit);

  // ---- Example 2: H2 dissociation (Gamma1 minimum region) ----
  state(t, "W2 H2 dissociation (1e-9 g/cm^3, 2000 K)", 1.0e-9 / RHO_U, 2000.0);
  state(t, "W2b deeper in dissociation (1e-8 g/cm^3, 3000 K)", 1.0e-8 / RHO_U, 3000.0);

  // ---- Example 3: edges and floors ----
  std::printf("\n== W3 edges and floors (rho_code = 1)\n");
  const double rho = 1.0;
  const double Pmin = t.Pof(rho, t.esp_min_);
  std::printf("    P(rho, esp_min) = %.10e code;  esp_min = %.10e code (T = %.6e K)\n", Pmin, t.esp_min_,
              std::pow(10.0, t.LogTof(rho, t.esp_min_)));
  counters(t, "edge-node lookups");
  {
    EosTable L = t;
    L.extrapolate_ = false;
    const double e = 0.5 * t.esp_min_;
    std::printf("  (a) esp = 0.5 esp_min: Pof legacy clamp = %.10e, extrapolating = %.10e (ratio %.6f)\n",
                L.Pof(rho, e), t.Pof(rho, e), t.Pof(rho, e) / L.Pof(rho, e));
    std::printf("      TemperatureK extrapolating = %.6e K (half of T(esp_min))\n",
                t.TemperatureK(rho, rho * e));
    counters(t, "(a)");
  }
  {
    const double pfloor = 1.0e-8;
    const double e_fl = t.EintFromRhoPres(rho, pfloor); // = InternalEnergyFromPres under eos = hydrogen
    std::printf("  (b) pressure floor pfloor = 1e-8: EintFromRhoPres -> e_fl = %.10e code (esp/esp_min = %.6e)\n",
                e_fl, e_fl / rho / t.esp_min_);
    std::printf("      P(e_fl) = %.10e (= pfloor x %.12f);  T = %.6e K\n", t.PresFromRhoEint(rho, e_fl),
                t.PresFromRhoEint(rho, e_fl) / pfloor, t.TemperatureK(rho, e_fl));
    EosTable L = t;
    L.extrapolate_ = false;
    const double e_old = L.EintFromRhoPres(rho, pfloor);
    std::printf("      legacy clamp: e_fl = %.10e (esp/esp_min = %.8f), P(e_fl) = %.6e = pfloor x %.4e\n", e_old,
                e_old / rho / t.esp_min_, L.PresFromRhoEint(rho, e_old), L.PresFromRhoEint(rho, e_old) / pfloor);
    counters(t, "(b)");
  }
  {
    const double Tfloor_code = 0.2726;
    const double TK = Tfloor_code * T_unit;
    const double ef = t.EintFromRhoTkHost(1.0, TK);
    std::printf("  (c) hydro/Tfloor = %.4f code = %.6f K: e_floor = EintFromRhoTkHost(1, T)/1 = %.10e code\n",
                Tfloor_code, TK, ef);
    for (double r : {1.0e-6, 1.0, 1.0e6}) {
      const double Pf = t.PresFromRhoEint(r, r * ef);
      std::printf("      rho = %.0e: floor pressure P(rho, rho e_floor) = %.6e, P/rho = %.8f, T = %.6f K\n", r, Pf,
                  Pf / r, t.TemperatureK(r, r * ef));
    }
    counters(t, "(c)");
  }
  {
    const double r = 0.5 * t.rho_min_;
    const double e = t.EintFromRhoTk(1.0, 10.0) / 1.0;
    EosTable Pm = t, PmL = t;
    Pm.p_over_rho_ = false; // eos_interp = p (P stored, not P/rho) -- need a reload for the stored array
    Pm.Load(path, RHO_U, V_U);
    PmL = Pm;
    PmL.extrapolate_ = false;
    std::printf("  (d) rho = 0.5 rho_min (%.4e g/cm^3), esp of 10 K:\n", r * RHO_U);
    std::printf("      p_over_rho: P/rho = %.10f;  p + extrapolate: %.10f;  p legacy clamp: %.10f\n",
                t.Pof(r, e) / r, Pm.Pof(r, e) / r, PmL.Pof(r, e) / r);
    std::printf("      (at rho_min itself, P/rho = %.10f)\n", t.Pof(t.rho_min_, e) / t.rho_min_);
    counters(t, "(d) p_over_rho");
    counters(Pm, "(d) p");
    // E-F1: P midway between two rho nodes, p vs p_over_rho, at esp of 10 K
    const int i0 = 300;
    const double lrm = t.lr0_ + (i0 + 0.5) * t.dlr_;
    const double rm = std::pow(10.0, lrm);
    std::printf("  (e) E-F1: midway between rho nodes %d and %d (rho %.4e g/cm^3), esp of 10 K: P(p) / P(p_over_rho) = "
                "%.8f;  predicted (1 + 10^dlr)/(2 10^(dlr/2)) = %.8f\n",
                i0, i0 + 1, rm * RHO_U, Pm.Pof(rm, e) / t.Pof(rm, e),
                (1.0 + std::pow(10.0, t.dlr_)) / (2.0 * std::pow(10.0, 0.5 * t.dlr_)));
    counters(Pm, "(e)");
  }
  {
    // (f) the upper edges: esp above the esp axis (Pof), T above the T axis (EintFromRhoTk), and a pressure above
    // P(rho, esp_max) handed to EspFromP (what HLLD's InternalEnergyFromPres does with a face pressure).
    const double esp_max = std::pow(10.0, leh);
    const double Pmax = t.Pof(rho, esp_max);
    counters(t, "(f) setup");
    std::printf("  (f) esp_max = %.6e code; P(rho, esp_max) = %.6e; T(rho, esp_max) = %.6e K\n", esp_max, Pmax,
                std::pow(10.0, t.LogTof(rho, esp_max)));
    counters(t, "(f) edge-node lookups");
    std::printf("      Pof(rho, 2 esp_max) = %.6e (ratio to P(esp_max) %.6f)\n", t.Pof(rho, 2.0 * esp_max),
                t.Pof(rho, 2.0 * esp_max) / Pmax);
    counters(t, "Pof above the esp axis");
    std::printf("      EintFromRhoTk(rho, 1e6 K) / rho = %.6e (esp(T_max) = %.6e)\n", t.EintFromRhoTk(rho, 1.0e6) / rho,
                t.EintFromRhoTk(rho, std::pow(10.0, lTh)) / rho);
    counters(t, "EintFromRhoTk above the T axis");
    const double e10 = t.EspFromP(rho, 10.0 * Pmax);
    std::printf("      EspFromP(rho, 10 P(esp_max)) = %.6e = esp_max x %.6f;  P of that esp = %.6e (target %.6e)\n",
                e10, e10 / esp_max, t.Pof(rho, e10), 10.0 * Pmax);
    counters(t, "EspFromP above the P range (the Pof check above it is the esp> count)");
  }
  return 0;
}
