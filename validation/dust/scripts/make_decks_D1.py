#!/usr/bin/env python
"""D1: 0-D grain growth IN the code (eos_test mode = compress with alpha = 0: a uniform gas at rest in a periodic box).

Production hydro + dust keys (du_common), radiation and gravity off, tabulated EOS (rho26). The only thing that
evolves is the dust pair (f_dg, a_c) through ReactDust: sublimation at t = 0 (f -> f_ref subl(T), a -> a_ref subl^1/3,
non-trivial only at 1200 K) and then sweep-up growth at fixed (rho, T). Exact answer: the ODE
da/dt = C (f rho / rho_gr) v_rel(rho, T, a) integrated to rtol 1e-12 (analyze_D1.py), cross-checked against the
Brownian closed form a^{5/2} = a0^{5/2} + (5/2) K t where v_B dominates.

States: five (rho, T) from AUDIT F-09's grid (1e-13 g/cm^3 / 30 K ... 1e-8 / 1200 K). The hydro dt of a static
uniform gas is cfl dx / c_s, so the box length sets dt; tlim = 6 t_grow(a_ref) (a grows ~2-3x).
Variants per state:
  res     dt = t_grow / 20  (growth resolved by the step; 120 React calls)
  sub     dt = 2 t_grow     (sub-cycling with growth_cap 0.1 engaged: the F-09 'rho >= 1e-10' regime)
  sub05   as sub, growth_cap 0.05;  sub025  growth_cap 0.025  (RK2 order of the sub-cycler: error ~ cap^2)
  one     dt >= tlim        (one React call for the whole run)
  pref05  as res with dust/growth_prefactor = 0.5 (the DD1 key end to end)
Writes inputs/D1/*.in, inputs/D1_runs.txt, inputs/D1_meta.json.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, sys
import numpy as np
sys.path.insert(0, _VT + "/dust/scripts")
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
from du_common import DU, EOS_TABLE, hydro_block, dust_block, mesh_block, write
import eostab

RHO_U, V_U, L_U = 5.467e-19, 1.9e4, 2.81e16
T_U = L_U / V_U
kB, mH, G = 1.380649e-16, 1.6726219e-24, 6.674e-8
RHO_GR, MU, ALPHA, A_REF, F_REF, TSUB, W = 3.0, 2.33, 1e-2, 1e-5, 0.01, 1500.0, 100.0
CFL, N = 0.3, 16


def dadt(rho, T, a, f, C=0.25):
    cs = np.sqrt(kB * T / (MU * mH)); vth = np.sqrt(8 / np.pi) * cs
    m = 4 / 3 * np.pi * a ** 3 * RHO_GR
    vB = np.sqrt(8 * kB * T / (np.pi * 0.5 * m))
    St = RHO_GR * a / (rho * vth) * np.sqrt(4 * np.pi * G * rho)
    vt = np.sqrt(ALPHA) * cs * St
    return C * f * rho / RHO_GR * np.sqrt(vB ** 2 + vt ** 2)


def main():
    tab = eostab.load(EOS_TABLE, RHO_U, V_U)
    states = [(1e-13, 30.0), (1e-11, 150.0), (1e-10, 300.0), (1e-9, 600.0), (1e-8, 1200.0)]
    runs, meta = [], {}
    for rho_cgs, T in states:
        rho = rho_cgs / RHO_U
        esp = eostab.esp_of(tab, rho, T)
        cs = np.sqrt(eostab.cs2_of(tab, rho, esp))          # code velocity
        s = 0.5 * (1 - np.tanh((T - TSUB) / W))
        f0, a0 = F_REF * s, A_REF * np.cbrt(s) if s < 1 else A_REF
        tg = a0 / dadt(rho_cgs, T, a0, f0)                 # s
        tlim = 6 * tg / T_U                                  # code
        st = f"r{int(round(-np.log10(rho_cgs)))}_T{int(T)}"
        for var, dtf, over in (("res", tg / 20, {}), ("sub", 2 * tg, {}), ("sub05", 2 * tg, {"growth_cap": 0.05}),
                               ("sub025", 2 * tg, {"growth_cap": 0.025}), ("one", 10 * tg, {}),
                               ("pref05", tg / 20, {"growth_prefactor": 0.5})):
            dt_code = dtf / T_U
            L = N * dt_code * cs / CFL
            name = f"D1_{st}_{var}"
            deck = f"""<comment>
problem = dust D1 0-D growth rho {rho_cgs:g} g/cm^3 T {T} K variant {var}: dt ~ {dtf / tg:g} t_grow, tlim = 6 t_grow

<job>
problem_id = eos_test

<problem/eos_test>
mode = compress
alpha = 0.0
rho0 = {float(rho)!r}
T0 = {T}
scalar0 = {F_REF}
scalar1 = {A_REF}

{mesh_block((N, 1, 1), (0.0, float(L)), (-0.5, 0.5), (-0.5, 0.5), (N, 1, 1))}
<parthenon/time>
integrator = vl2
cfl = {CFL}
tlim = {float(tlim)!r}
nlim = -1

{hydro_block(1, clamp=False)}
<physics>
self_gravity = false
radiation = false
dust = true

{dust_block(**over)}
<parthenon/output0>
file_type = hdf5
variables = prim, cons
dt = {float(tlim)!r}

<parthenon/output1>
file_type = hst
dn = 1
"""
            write(f"{DU}/inputs/D1/{name}.in", deck)
            runs.append(f"D1/{name} | D1/{name}.in | | 1")
            meta[name] = dict(rho_cgs=rho_cgs, T=T, rho_code=float(rho), tg_s=float(tg), tlim=float(tlim),
                              dt_over_tg=dtf / tg, L=float(L), cs_code=float(cs), over=over)
    write(f"{DU}/inputs/D1_runs.txt", "\n".join(runs) + "\n")
    write(f"{DU}/inputs/D1_meta.json", json.dumps(meta, indent=1))
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
