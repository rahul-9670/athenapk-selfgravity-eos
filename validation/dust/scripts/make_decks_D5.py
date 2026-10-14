#!/usr/bin/env python
"""D5: shock-recorded sublimation (the dual_energy T4c hand-over), with the LIVE scheme (PLM predictor, kv shock
detector + shock_eta1, glmmhd_damping midpoint) instead of T4c's dc predictor.

Planar Noh by colliding flows (eos_test riemann, uL = -uR = M cs0, 10 K at rho 1 code), the exact plateau state
(rho2, e2, T2) from ../dual_energy/inputs/D3_meta.json (M 10: T2 423.4 K; M 30: T2 1171.6 K). dust/T_subl_K = T2 + dT so
the exact plateau f_dg/f_ref = subl(T2) = 0.5 (dT = 0) -- every bit of extra sublimation is visible.
Ladder N = 256, 1024, 4096 (dT = 0); T_subl scan dT = -200, -100, +100, +200 at N 1024 (separates 'plateau hot' from
'transient overshoot': a plateau-T error dT_p shifts f by subl'(T2) dT_p, an overshoot recorded by irreversibility adds
a one-sided deficit that is largest where subl is steepest).
'frz' = freeze_growth (a_c = a_ref (f/f_ref)^(1/3) exactly); growth at rho ~ 5 code (2.7e-18 g/cm^3) is negligible.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, sys
sys.path.insert(0, _VT + "/dust/scripts")
from du_common import DU, hydro_block, dust_block, mesh_block, write

M3 = json.load(open(_VT + "/dual_energy/inputs/D3_meta.json"))["runs"]


def deck(name, m, N, Tsubl, over):
    return f"""<comment>
problem = dust {name}: planar Noh M {m['M']} N {N}, T_subl {Tsubl} K (T2 = {m['T2']} K)

<job>
problem_id = eos_test

<problem/eos_test>
mode = riemann
x0 = 0.5
rhoL = 1.0
espL_code = {m['e0']!r}
uL = {m['u']!r}
rhoR = 1.0
espR_code = {m['e0']!r}
uR = {-m['u']!r}
scalar0 = 0.01
scalar1 = 1.0e-5

{mesh_block((N, 1, 1), (0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5), (N, 1, 1), bcs={1: ('outflow', 'outflow')})}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {m['tlim']!r}
nlim = -1

{hydro_block(1, clamp=False)}
<physics>
self_gravity = false
radiation = false
dust = true

{dust_block(T_subl_K=Tsubl, **over)}
<parthenon/output0>
file_type = hdf5
variables = prim, cons, dual_switched
dt = {m['tlim'] / 4!r}

<parthenon/output1>
file_type = hst
dt = {m['tlim'] / 200!r}
"""


def main():
    runs, meta = [], {}
    for M in (10, 30):
        m = M3[f"D3_M{M}_N1024"]
        cases = [(N, 0) for N in (256, 1024, 4096)] + [(1024, d) for d in (-200, -100, 100, 200)]
        for N, dT in cases:
            name = f"D5_M{M}_N{N}_dT{dT:+d}"
            Ts = float(m["T2"]) + dT
            write(f"{DU}/inputs/D5/{name}.in", deck(name, m, N, Ts, {"freeze_growth": "true"}))
            runs.append(f"D5/{name} | D5/{name}.in | | 1")
            meta[name] = dict(M=M, N=N, dT=dT, T_subl=Ts, T2=float(m["T2"]), rho2=float(m["rho2"]),
                              e2=float(m["e2"]), S=float(m["S"]), tlim=float(m["tlim"]))
    write(f"{DU}/inputs/D5_runs.txt", "\n".join(runs) + "\n")
    write(f"{DU}/inputs/D5_meta.json", json.dumps(meta, indent=1))
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
