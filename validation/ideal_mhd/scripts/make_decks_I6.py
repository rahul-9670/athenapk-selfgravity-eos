#!/usr/bin/env python
"""I6 Noh implosion, S tier (gamma 5/3), pgen ideal_mhd mode = noh, production scheme keys INCLUDING diode BCs +
boundary_flux_clamp = true (the box is large enough that the boundary never matters: gas that reaches the shock by
t = 0.6 started inside r = v0 t (1 + (gamma-1)/2) = 0.8 < 1).

rho0 1, v0 1 toward the origin, P0 1e-6 (Noh 1987 epsilon), box [-1,1]^d with the origin on a cell corner (even N).
Exact (gamma-law, P0 -> 0): r_s = v0 t (gamma-1)/2 = 0.2 at t = 0.6; inside rho = ((g+1)/(g-1))^d = 4 / 16 / 64,
u = 0, P = (g-1) rho v0^2 / 2; outside rho = (1 + v0 t / r)^(d-1), u = -v0.
Variants: prod (dual energy on) and nd (dual energy off) at every N; dc (vl2_predictor_recon = dc) at one N per d
(SG-F4 / NI-F1 contrast).
Writes inputs/I6/noh_*.in and inputs/I6_runs.txt.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write


def deck(dim, N, dual=True, pred="same"):
    nx = (N, N if dim > 1 else 1, N if dim > 2 else 1)
    box = tuple((-1, 1) if d < dim else (-0.5, 0.5) for d in range(3))
    m = N if dim == 1 else 32   # 1D: one block, one rank
    mb = tuple(min(n, m) if n > 1 else 1 for n in nx)
    nranks = min(64, (nx[0] // mb[0]) * (nx[1] // mb[1]) * (nx[2] // mb[2]))
    h = hydro_block("S", clamp=True, dual=dual)
    if pred != "same":
        h = h.replace("vl2_predictor_recon = same\n", f"vl2_predictor_recon = {pred}   # contrast (production = same)\n")
    s = f"""<comment>
problem = ideal_mhd I6 Noh {dim}D N={N} dual={dual} pred={pred}

<job>
problem_id = ideal_mhd

<problem/ideal_mhd>
mode = noh
rho0 = 1.0
v0 = 1.0
P0 = 1.0e-6

{mesh_block(nx, *box, mb, bc='diode')}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 0.6
nlim = -1

{h}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = prim
dt = 0.2

<parthenon/output1>
file_type = hst
dn = 10
"""
    return s, nranks


def main():
    runs = []
    for dim, Ns, ndc in ((1, (128, 256, 512, 1024, 2048), 1024), (2, (64, 128, 256, 512), 256), (3, (32, 64, 128), 64)):
        for N in Ns:
            for tag, kw in (("prod", {}), ("nd", {"dual": False})) + ((("dc", {"pred": "dc"}),) if N == ndc else ()):
                name = f"noh_{dim}d_{tag}_N{N}"
                s, nr = deck(dim, N, **kw)
                write(f"{M}/inputs/I6/{name}.in", s)
                runs.append(f"I6/{name} | I6/{name}.in | | {nr}")
    write(f"{M}/inputs/I6_runs.txt", "\n".join(runs) + "\n")
    print("\n".join(runs))


if __name__ == "__main__":
    main()
