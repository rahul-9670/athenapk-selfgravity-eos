#!/usr/bin/env python
"""D2: sensitivity of the grown grain size to the sweep-up prefactor C (da/dt = C (f rho / rho_gr) v_rel), offline.

The law is dust.hpp's (v_B Brownian, v_t = sqrt(alpha) c_s St with the LOCAL Omega = sqrt(4 pi G rho)), integrated
along a representative collapse track: d ln(rho)/dt = 1/t_ff(rho), t_ff = sqrt(3 pi/(32 G rho)), with the barotropic
T(rho) = 10 K sqrt(1 + (rho/1e-13)^(0.8)) (isothermal -> gamma 1.4 above rhocrit; a shape, not the EOS table).
Start at rho 1e-19 g/cm^3 with a_ref = 1e-5 cm, f = 0.01. Output at rho = 1e-13, 1e-11, 1e-10, 1e-9, 1e-8.
Reports a_c/a_ref for C = 0.25 (code), 0.5 (monodisperse Smoluchowski, derivation in the report), 1.0 (Birnstiel et al.
2012 Eq. 10 / Brauer et al. 2008), the ratio to C = 0.25, the grain-area factor f/a (ionization model: per-cell
recombination area) and the bridged opacity factor g_R(a, T) (dust.hpp DustSizeFactor, Rosseland).
Also the turbulent-velocity closure: code v_t vs Ormel & Cuzzi (2007) Eq. 27 for St2 = St1/2 at Re = 1e8
(dv = V_g Re^(1/4) |St1 - St2|, V_g = sqrt(alpha) c_s) and v_B, at the same track points.
Writes results/tables/D2_prefactor.txt (+ .json); --check.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, os, sys
import numpy as np
from scipy.integrate import solve_ivp
DU = _VT + "/dust"
kB, mH, G = 1.380649e-16, 1.6726219e-24, 6.674e-8
RHO_GR, MU, ALPHA, A_REF, F_REF = 3.0, 2.33, 1e-2, 1e-5, 0.01
TB, UR = 4799.243073366221, 4.790286870161177   # dust.hpp dopac::kTBreakK, kURoss


def T_of(rho):
    return 10.0 * np.sqrt(1 + (rho / 1e-13) ** 0.8)


def vels(rho, T, a):
    cs = np.sqrt(kB * T / (MU * mH)); vth = np.sqrt(8 / np.pi) * cs
    m = 4 / 3 * np.pi * a ** 3 * RHO_GR
    vB = np.sqrt(8 * kB * T / (np.pi * 0.5 * m))
    St = RHO_GR * a / (rho * vth) * np.sqrt(4 * np.pi * G * rho)
    vt = np.sqrt(ALPHA) * cs * St
    vOC = np.sqrt(ALPHA) * cs * 1e8 ** 0.25 * 0.5 * St    # OC07 Eq. 27, St2 = St1/2, Re = 1e8
    return vB, vt, vOC, St


def track(C, stops):
    def rhs(lnr, y):
        rho = np.exp(lnr); T = T_of(rho); a = y[0]
        vB, vt, _, _ = vels(rho, T, a)
        tff = np.sqrt(3 * np.pi / (32 * G * rho))
        return [C * F_REF * rho / RHO_GR * np.sqrt(vB ** 2 + vt ** 2) * tff]   # da/dlnrho = (da/dt) t_ff
    sol = solve_ivp(rhs, (np.log(1e-19), np.log(stops[-1])), [A_REF], t_eval=np.log(stops), method="DOP853",
                    rtol=1e-10, atol=0)
    return sol.y[0]


def main():
    stops = np.array([1e-13, 1e-11, 1e-10, 1e-9, 1e-8])
    res, rows = {}, ["# D2: a_c/a_ref along a collapse track (d ln rho/dt = 1/t_ff, barotropic T), f = f_ref"]
    rows.append("%9s %7s | %8s %8s %8s | %7s %7s | %8s %8s | %8s %8s" % (
        "rho", "T[K]", "C=0.25", "C=0.5", "C=1", "r(0.5)", "r(1)", "gR(.25)", "gR(1)", "area.25", "area1"))
    A = {C: track(C, stops) for C in (0.25, 0.5, 1.0)}
    for i, rho in enumerate(stops):
        T = T_of(rho)
        a = {C: A[C][i] / A_REF for C in A}
        g = {C: (1 + UR * T / TB) / (1 + a[C] * UR * T / TB) for C in A}
        rows.append("%9.0e %7.1f | %8.3f %8.3f %8.3f | %7.3f %7.3f | %8.4f %8.4f | %8.3f %8.3f" % (
            rho, T, a[0.25], a[0.5], a[1.0], a[0.5] / a[0.25], a[1.0] / a[0.25], g[0.25], g[1.0], 1 / a[0.25], 1 / a[1.0]))
        res[f"{rho:.0e}"] = dict(T=float(T), a=[float(a[C]) for C in (0.25, 0.5, 1.0)])
    rows.append("# area = f/a in units of f_ref/a_ref (the grain-area factor of the ionization model, CellIon); gR = "
                "bridged Rosseland size factor")
    rows.append("# turbulent closure at a = a_ref and at the C = 0.25 a_c: v_B, code v_t, OC07 Eq. 27 (St2 = St1/2, Re 1e8)")
    for i, rho in enumerate(np.array([1e-19, 1e-17, 1e-15, *stops])):
        T = T_of(rho)
        for lab, a in (("a_ref", A_REF),):
            vB, vt, vOC, St = vels(rho, T, a)
            rows.append(f"  rho {rho:7.0e} T {T:6.1f} {lab}: St {St:.2e}  v_B {vB:.3e}  v_t(code) {vt:.3e}  "
                        f"v_t(OC07) {vOC:.3e} cm/s  -> v_t(OC07)/v_B {vOC / vB:.2e}")
    txt = "\n".join(rows)
    print(txt)
    jp = os.path.join(DU, "results", "tables", "D2_prefactor.json")
    if "--check" in sys.argv:
        ok = json.dumps(json.load(open(jp)), sort_keys=True) == json.dumps(json.loads(json.dumps(res)), sort_keys=True)
        print("CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
    open(jp.replace(".json", ".txt"), "w").write(txt + "\n")
    json.dump(res, open(jp, "w"), indent=1)


if __name__ == "__main__":
    main()
