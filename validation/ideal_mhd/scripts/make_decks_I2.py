#!/usr/bin/env python
"""I2 circularly polarised Alfven wave (Toth 2000), pgen cpaw, S tier, production scheme keys.

rho 1, P 0.1, b_par 1, b_perp 0.1 (HL), gamma 5/3: an exact NONLINEAR solution travelling at v_A = b_par/sqrt(rho)
= 1 along k (v_par = 0) or standing (v_par = -1). beta = 2P/B^2 = 0.198; e_int/(e_int + e_B) = 0.23 > eta1, so the
dual branch does not select. cpaw's UserWorkAfterLoop compares with the INITIAL condition -> tlim = whole periods.
2D [0,2]x[0,1], 2N x N, ang_2 = 0: lambda = 2/sqrt5, period = lambda / v_A = 0.894427191. 3D [0,3]x[0,1.5]^2,
2N x N x N, pgen angles: lambda = 1, period 1.
Runs: trav 2D N 8..128 and 3D N 8..32 (1 period); stand 2D N 8..64 (1 period-length); left polarisation 2D N 32;
long 2D N 32 for 16 periods with a dump every period (amplitude decay + phase per period, analyze_I2.py).
Writes inputs/I2/*.in and inputs/I2_runs.txt.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write

P2D = 2.0 / 5 ** 0.5


def deck(dim, N, v_par=0.0, dirn=1, periods=1, dumps=True):
    if dim == 2:
        nx, box, angs, period = (2 * N, N, 1), ((0, 2), (0, 1), (-0.5, 0.5)), "ang_2 = 0.0\n", P2D
        mb = (min(2 * N, 64), min(N, 64), 1)
    else:
        nx, box, angs, period = (2 * N, N, N), ((0, 3), (0, 1.5), (0, 1.5)), "", 1.0
        mb = (2 * N, N, N) if N <= 16 else (32, 32, 32)
    nr = (nx[0] // mb[0]) * (nx[1] // mb[1]) * (nx[2] // mb[2])
    out = f"""
<parthenon/output0>
file_type = hdf5
variables = cons
dt = {period!r}
""" if dumps else ""
    s = f"""<comment>
problem = ideal_mhd I2 CPAW {dim}D N={N} v_par={v_par} dir={dirn} periods={periods}

<job>
problem_id = cpaw

<problem/cpaw>
b_par = 1.0
b_perp = 0.1
pres = 0.1
v_par = {v_par!r}
dir = {dirn}
{angs}compute_error = true

{mesh_block(nx, *box, mb)}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {periods * period!r}
nlim = -1

{hydro_block('S', clamp=False)}
<physics>
self_gravity = false
radiation = false
{out}
<parthenon/output1>
file_type = hst
dn = 10
"""
    return s, nr


def main():
    runs = []
    cases = [(f"I2_2d_trav_N{N}", 2, N, {}) for N in (8, 16, 32, 64, 128)]
    cases += [(f"I2_3d_trav_N{N}", 3, N, {}) for N in (8, 16, 32)]
    cases += [(f"I2_2d_stand_N{N}", 2, N, {"v_par": -1.0}) for N in (8, 16, 32, 64)]
    cases += [("I2_2d_left_N32", 2, 32, {"dirn": 2}), ("I2_2d_long16_N32", 2, 32, {"periods": 16, "dumps": True})]
    for name, dim, N, kw in cases:
        s, nr = deck(dim, N, **kw)
        write(f"{M}/inputs/I2/{name}.in", s)
        runs.append(f"I2/{name} | I2/{name}.in | | {nr}")
    write(f"{M}/inputs/I2_runs.txt", "\n".join(runs) + "\n")
    print("\n".join(runs))


if __name__ == "__main__":
    main()
