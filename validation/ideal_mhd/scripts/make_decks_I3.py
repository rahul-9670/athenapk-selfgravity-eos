#!/usr/bin/env python
"""I3 MHD shock tubes (pgen ideal_mhd mode = shocktube), S tier, production scheme keys. HL units.

Problems (states in exact_mhd_riemann.PROBLEMS for the regular ones):
  RJ2a      Ryu & Jones 1995 2a, gamma 5/3, t 0.2 -- all seven waves are discontinuities (exact solver, D1 (a))
  RJ2a_div  RJ2a states with u_L = -0.6, u_R = +0.6: fast + slow RAREFACTIONS (exact solver; project variant)
  BW        Brio & Wu 1988, gamma 2, Bx 0.75, By +-1, t 0.1: the numerical solution has the compound wave, the
            regular exact solution is not what schemes converge to -> converged reference N = 16384 (D1 (b))
Box [0,1], x0 0.5, N = 128 .. 4096 + reference 16384.
BCs: OUTFLOW (zero gradient) and boundary_flux_clamp = false -- NOT production (diode + clamp). RJ2a has u_L = 1.2
INTO the box: the diode refuses that inflow and launches a rarefaction from x = 0 that reaches the solution before
t = 0.2. The boundary treatment is I7's subject, not this test's.
Writes inputs/I3/*.in and inputs/I3_runs.txt.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write
from exact_mhd_riemann import PROBLEMS

NS = (128, 256, 512, 1024, 2048, 4096, 16384)


def states(name):
    if name == "BW":
        return dict(gamma=2.0, t=0.1, Bx=0.75, L=(1.0, 0, 0, 0, 1.0, 1.0, 0), R=(0.125, 0, 0, 0, 0.1, -1.0, 0))
    p = PROBLEMS[name]
    st = lambda W: (W.rho, W.u, W.vt[0], W.vt[1], W.P, W.Bt[0], W.Bt[1])
    return dict(gamma=p["gamma"], t=p["t"], Bx=p["Bx"], L=st(p["L"]), R=st(p["R"]))


def deck(name, N):
    s = states(name)
    mb = min(N, 2048)
    keys = ("rho", "vx", "vy", "vz", "P", "By", "Bz")
    st = "".join(f"{k}L = {float(s['L'][i])!r}\n{k}R = {float(s['R'][i])!r}\n" for i, k in enumerate(keys))
    h = hydro_block("S", clamp=False).replace("gamma = 1.6666666666666667", f"gamma = {s['gamma']!r}")
    h = h.replace("   # periodic box: the clamp skips periodic faces anyway (DE-F2 fixed)", "   # I3: outflow BCs, see make_decks_I3.py")
    d = f"""<comment>
problem = ideal_mhd I3 shock tube {name} N={N}

<job>
problem_id = ideal_mhd

<problem/ideal_mhd>
mode = shocktube
x0 = 0.5
B0x = {float(s['Bx'])!r}
{st}
{mesh_block((N, 1, 1), (0, 1), (-0.5, 0.5), (-0.5, 0.5), (mb, 1, 1), bc='outflow')}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {s['t']!r}
nlim = -1

{h}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = prim
dt = {s['t']!r}
"""
    return d, N // mb


def main():
    runs = []
    for name in ("RJ2a", "RJ2a_div", "BW"):
        for N in NS:
            d, nr = deck(name, N)
            r = f"I3_{name}_N{N}"
            write(f"{M}/inputs/I3/{r}.in", d)
            runs.append(f"I3/{r} | I3/{r}.in | | {nr}")
    write(f"{M}/inputs/I3_runs.txt", "\n".join(runs) + "\n")
    print(len(runs), "decks")


if __name__ == "__main__":
    main()
