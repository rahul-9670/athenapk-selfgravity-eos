#!/usr/bin/env python
"""I4 field-loop advection (Gardiner & Stone 2005), S tier, production scheme keys.

2D (iprob 1): box [-1,1]x[-0.5,0.5], 2N x N, loop A_z = amp (R - r), R 0.3, amp 1e-3, P 1 (beta = 2P/B^2 = 2e6,
HL units), rho 1, v = vflow (x1size, x2size) = (2, 1): the loop returns to its start at t = 1 (exact solution =
the initial condition). 3D (iprob 4): box [-0.5,0.5]^2 x [-1,1], N x N x 2N, loop cylinder rotated in the x1-x3
plane, v = (1, 1, 2), t = 1.
Cases: prod (alpha 0.1) N = 32..256 (2D), 32/64 (3D); alpha scan at 2D N = 128: a0 (0), a04 (0.4); drat10
(density 10 inside the loop, contact advected with the field).
A beta ~ 1 loop is NOT an exact test (the loop's tension is unbalanced, it evolves by itself): not included;
the low-beta GLM question is owned by I5 (pure cleaning) and I8 (production state).
Writes inputs/I4/*.in and inputs/I4_runs.txt.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write


def deck(dim, N, alpha=0.1, drat=1.0, tag=""):
    if dim == 2:
        nx, box, iprob = (2 * N, N, 1), ((-1, 1), (-0.5, 0.5), (-0.5, 0.5)), 1
        mb = (min(2 * N, 64), min(N, 64), 1)
    else:
        nx, box, iprob = (N, N, 2 * N), ((-0.5, 0.5), (-0.5, 0.5), (-1, 1)), 4
        mb = (32, 32, 32)
    nranks = (nx[0] // mb[0]) * (nx[1] // mb[1]) * (nx[2] // mb[2])
    s = f"""<comment>
problem = ideal_mhd I4 field loop {dim}D N={N} alpha={alpha!r} drat={drat!r} {tag}

<job>
problem_id = field_loop

<problem/field_loop>
iprob = {iprob}
rad = 0.3
amp = 1.0e-3
vflow = 1.0
drat = {drat!r}
p0 = 1.0

{mesh_block(nx, *box, mb)}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 1.0
nlim = -1

{hydro_block('S', clamp=False)}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = cons
dt = 0.25

<parthenon/output1>
file_type = hst
dn = 10
"""
    if alpha != 0.1:
        s = s.replace("glmmhd_alpha = 0.1\n", f"glmmhd_alpha = {alpha!r}   # I4 alpha scan (production 0.1)\n")
    return s, nranks


def main():
    runs = []
    cases = [(2, N, {}, f"I4_2d_prod_N{N}") for N in (32, 64, 128, 256)]
    cases += [(3, N, {}, f"I4_3d_prod_N{N}") for N in (32, 64)]
    cases += [(2, 128, {"alpha": 0.0}, "I4_2d_a0_N128"), (2, 128, {"alpha": 0.4}, "I4_2d_a04_N128"),
              (2, 128, {"drat": 10.0}, "I4_2d_drat10_N128")]
    for dim, N, kw, name in cases:
        s, nr = deck(dim, N, **kw)
        write(f"{M}/inputs/I4/{name}.in", s)
        runs.append(f"I4/{name} | I4/{name}.in | | {nr}")
    write(f"{M}/inputs/I4_runs.txt", "\n".join(runs) + "\n")
    print("\n".join(runs))


if __name__ == "__main__":
    main()
