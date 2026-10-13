#!/usr/bin/env python
"""I7 outgoing waves through the production boundary (diode + boundary_flux_clamp) vs a large-box reference.

Each problem runs in [0,1] (N 512) after its waves have LEFT the box, with
  diode  = production: diode BCs + boundary_flux_clamp = true
  outfl  = Parthenon outflow (zero gradient) + clamp false
  ref    = the same problem in [-4,5] at the same dx (N 4608, outflow): nothing reaches its boundary by tlim
and the [0,1] part of ref is the answer (reflections and mass changes are the error).
Problems: BW (Brio-Wu, gamma 2, at rest initially, t 0.3: all waves exit), RJ2a_div (diverging, t 0.5), and the
GLM div-B pulse (1D, a0 and prod alpha, t 0.8: both halves exit; ref is PERIODIC [-4,5]) -- psi at the boundary.
Writes inputs/I7/*.in and inputs/I7_runs.txt.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write
from make_decks_I3 import states

N = 512


def mesh(kind):
    if kind == "ref":
        return (9 * N, 1, 1), (-4, 5), (9 * N // 8, 1, 1)
    return (N, 1, 1), (0, 1), (N, 1, 1)


def tube(name, kind, tlim):
    s = states(name)
    nx, x1, mb = mesh(kind)
    bc = "diode" if kind == "diode" else "outflow"
    keys = ("rho", "vx", "vy", "vz", "P", "By", "Bz")
    st = "".join(f"{k}L = {float(s['L'][i])!r}\n{k}R = {float(s['R'][i])!r}\n" for i, k in enumerate(keys))
    h = hydro_block("S", clamp=(kind == "diode")).replace("gamma = 1.6666666666666667", f"gamma = {s['gamma']!r}")
    h = h.replace("   # periodic box: the clamp skips periodic faces anyway (DE-F2 fixed)", f"   # I7 {kind}")
    return f"""<comment>
problem = ideal_mhd I7 {name} {kind} t={tlim}

<job>
problem_id = ideal_mhd

<problem/ideal_mhd>
mode = shocktube
x0 = 0.5
B0x = {float(s['Bx'])!r}
{st}
{mesh_block(nx, x1, (-0.5, 0.5), (-0.5, 0.5), mb, bc=bc)}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {tlim!r}
nlim = -1

{h}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = prim, cons
dt = {tlim / 4!r}

<parthenon/output1>
file_type = hst
dn = 5
""", nx[0] // mb[0]


def pulse(case, kind, tlim=0.8):
    nx, x1, mb = mesh(kind)
    bc = {"diode": "diode", "outfl": "outflow", "ref": "periodic"}[kind]
    h = hydro_block("S", clamp=(kind == "diode"))
    h = h.replace("   # periodic box: the clamp skips periodic faces anyway (DE-F2 fixed)", f"   # I7 {kind}")
    if case == "a0":
        h = h.replace("glmmhd_alpha = 0.1\n", "glmmhd_alpha = 0.0   # I7 a0 (production 0.1)\n")
    return f"""<comment>
problem = ideal_mhd I7 divB pulse {case} {kind} t={tlim}

<job>
problem_id = ideal_mhd

<problem/ideal_mhd>
mode = divb
rho0 = 1.0
P0 = 0.6
bamp = 1.0e-3
rb = 0.05

{mesh_block(nx, x1, (-0.5, 0.5), (-0.5, 0.5), mb, bc=bc)}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {tlim!r}
nlim = -1

{h}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = cons
dt = 0.2
""", nx[0] // mb[0]


def main():
    runs = []
    for kind in ("diode", "outfl", "ref"):
        for name, tl in (("BW", 0.3), ("RJ2a_div", 0.5)):
            s, nr = tube(name, kind, tl)
            r = f"I7_{name}_{kind}"
            write(f"{M}/inputs/I7/{r}.in", s)
            runs.append(f"I7/{r} | I7/{r}.in | | {nr}")
        for case in ("a0", "prod"):
            s, nr = pulse(case, kind)
            r = f"I7_divb_{case}_{kind}"
            write(f"{M}/inputs/I7/{r}.in", s)
            runs.append(f"I7/{r} | I7/{r}.in | | {nr}")
    write(f"{M}/inputs/I7_runs.txt", "\n".join(runs) + "\n")
    print("\n".join(runs))


if __name__ == "__main__":
    main()
