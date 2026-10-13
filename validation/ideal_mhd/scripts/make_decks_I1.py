#!/usr/bin/env python
"""I1 / I1b linear MHD waves (Gardiner & Stone 2005; Stone et al. 2008 setup), S tier, production scheme keys.

Background rho 1, B = (1, sqrt 2, 1/2) along the wave frame, P0 = 1/gamma (beta = 2 P0 / B^2 = 0.37, HL units);
I1b: P0 = 0.04875 (beta = 0.030: e_int / (e_int + e_B) = 0.043 < eta1 = 0.05, so the dual-energy branch can
select). Amplitude 1e-6, ONE period (problem/linear_wave/test = true reinterprets tlim as periods).
wave_flag 0 fast, 1 Alfven, 2 slow (left-going), 3 entropy (with vflow = 1).
1D: x1 in [0,1], N = 16..512. 2D: [0,2]x[0,1], 2N x N, ang_2 = 0 (wave in the x1-x2 plane, k along (1,2)/sqrt5).
3D: [0,3]x[0,1.5]^2, 2N x N x N, angles from the pgen (lambda = 1, Stone et al. 2008).
Variants: _nd = dual energy off (pairs with I1b to isolate the dual branch); _dc = vl2_predictor_recon = dc
(1D fast + slow only; the SG-F4 contrast).
Writes inputs/I1/*.in and inputs/I1_runs.txt ("rundir | deck | overrides | nranks").
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write

WAVES = {0: "fast", 1: "alfven", 2: "slow", 3: "entropy"}
P0_LB = 0.04875


def deck(dim, wf, N, p0=None, dual=True, pred="same"):
    if dim == 1:
        nx, box, mb, angs = (N, 1, 1), ((0, 1), (-0.5, 0.5), (-0.5, 0.5)), (N, 1, 1), "ang_2 = 0.0\nang_3 = 0.0\n"
    elif dim == 2:
        nx, box, angs = (2 * N, N, 1), ((0, 2), (0, 1), (-0.5, 0.5)), "ang_2 = 0.0\n"
        mb = (2 * N, N, 1) if N <= 64 else (64, 64, 1)
    else:
        nx, box, angs = (2 * N, N, N), ((0, 3), (0, 1.5), (0, 1.5)), ""
        mb = (2 * N, N, N) if N <= 16 else (32, 32, 32)
    nranks = min(64, (nx[0] // mb[0]) * (nx[1] // mb[1]) * (nx[2] // mb[2]))
    s = f"""<comment>
problem = ideal_mhd I1{'b' if p0 else ''} linear wave {WAVES[wf]} {dim}D N={N}{' P0='+str(p0) if p0 else ''}{'' if dual else ' dual off'}{'' if pred=='same' else ' pred='+pred}

<job>
problem_id = linear_wave_mhd

<problem/linear_wave>
wave_flag = {wf}
amp = 1.0e-6
vflow = {1.0 if wf == 3 else 0.0}
{angs}{'p0 = %r' % p0 + chr(10) if p0 else ''}test = true
compute_error = true

{mesh_block(nx, *box, mb)}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 1.0   # periods (test = true)
nlim = -1

{hydro_block('S', clamp=False, dual=dual)}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hst
dn = 10
"""
    if pred != "same":
        s = s.replace("vl2_predictor_recon = same\n", "vl2_predictor_recon = dc   # SG-F4 contrast (production = same)\n")
    return s, nranks


def main():
    runs = []
    for dim, Ns in ((1, (16, 32, 64, 128, 256, 512)), (2, (8, 16, 32, 64, 128)), (3, (8, 16, 32, 64, 128))):
        for wf in WAVES:
            for N in Ns:
                for tag, kw in (("", {}), ("_lb", {"p0": P0_LB}), ("_lb_nd", {"p0": P0_LB, "dual": False})):
                    if dim == 2 and tag:
                        continue  # low-beta variants in 1D and 3D only
                    if dim == 3 and N == 128 and tag:
                        continue  # 3D N 128 rung (added 2026-10-12, N 64 not yet asymptotic): production keys only
                    name = f"I1_{dim}d_{WAVES[wf]}_N{N}{tag}"
                    s, nr = deck(dim, wf, N, **kw)
                    write(f"{M}/inputs/I1/{name}.in", s)
                    runs.append(f"I1/{name} | I1/{name}.in | | {nr}")
            if dim == 1 and wf in (0, 2):
                for N in Ns:
                    name = f"I1_{dim}d_{WAVES[wf]}_N{N}_dc"
                    s, nr = deck(dim, wf, N, pred="dc")
                    write(f"{M}/inputs/I1/{name}.in", s)
                    runs.append(f"I1/{name} | I1/{name}.in | | {nr}")
    write(f"{M}/inputs/I1_runs.txt", "\n".join(runs) + "\n")
    print(len(runs), "decks; ranks total", sum(int(r.split("|")[3]) for r in runs))


if __name__ == "__main__":
    main()
