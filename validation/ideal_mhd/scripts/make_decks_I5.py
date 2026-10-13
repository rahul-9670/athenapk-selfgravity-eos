#!/usr/bin/env python
"""I5 / I5b GLM div-B pulse (pgen ideal_mhd mode = divb), S tier, production scheme keys.

Uniform rho 1, P0 0.6 (cs = 1 at gamma 5/3), v = 0, B0 = 0, plus Bx = bamp exp(-r^2/rb^2), bamp 1e-3, rb 0.05,
centred in the periodic unit box. With B0 = 0 the extended-GLM momentum/energy sources are O(bamp^2) = 1e-6 of
the signal, so (B, psi) obey the linear GLM system: irrotational part of B and psi follow, per Fourier mode,
  a' = -i|k| psi,   psi' = -i c_h^2 |k| a - lambda psi,   lambda = alpha c_h / mindx  (dedner_source.cpp)
with c_h = max fast speed = 1 (1D: cf = cs along x; 2D: sqrt(cs^2 + B^2) = 1 + 5e-7). Exact reference by FFT
(analyze_I5.py).
Cases: a0 = alpha 0 (pure c_h transport), lam10 = alpha = 10 mindx / c_h (lambda = 10 fixed: a convergence
test), prod = alpha 0.1 (the production key: lambda = 0.1 N, grows with resolution).
I5b: 2D base N = 128 with the central square [0.25, 0.75]^2 refined by 1 or 2 levels (pulse inside the fine
region; the ring crosses the coarse-fine boundary at t ~ 0.25), a0 and prod.
Outputs: cons at t = 0, 0.05, ..., 0.3.
Writes inputs/I5/*.in and inputs/I5_runs.txt.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
from im_common import M, hydro_block, mesh_block, write

ALPHA = {"a0": lambda N: 0.0, "lam10": lambda N: 10.0 / N, "prod": lambda N: 0.1}


def deck(dim, N, case, smr_levels=0, patch_levels=0, mb_override=None):
    mindx_N = N * 2 ** max(smr_levels, patch_levels)
    alpha = ALPHA[case](mindx_N)
    if dim == 1:
        nx, mb = (N, 1, 1), (min(N, 64), 1, 1)
        box = ((0, 1), (-0.5, 0.5), (-0.5, 0.5))
    else:
        nx, mb = (N, N, 1), (min(N, 32), min(N, 32), 1)
        box = ((0, 1), (0, 1), (-0.5, 0.5))
    smr = [(smr_levels, (0.25, 0.75), (0.25, 0.75), (-0.5, 0.5))] if smr_levels else None
    if patch_levels:   # I5c: refined corner patch far from the pulse -- only mindx (and so lambda) changes there
        smr = [(patch_levels, (0.0, 0.1), (0.0, 0.1), (-0.5, 0.5))]
    if mb_override:
        mb = mb_override
    s = f"""<comment>
problem = ideal_mhd I5{'c' if patch_levels else ('b' if smr else '')} GLM div-B pulse {dim}D N={N}{' +%d SMR levels' % smr_levels if smr_levels else ''}{' corner patch %d levels' % patch_levels if patch_levels else ''} case {case} (alpha {alpha!r})

<job>
problem_id = ideal_mhd

<problem/ideal_mhd>
mode = divb
rho0 = 1.0
P0 = 0.6
bamp = 1.0e-3
rb = 0.05

{mesh_block(nx, *box, mb, smr=smr)}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = 0.3
nlim = -1

{hydro_block('S', clamp=False)}"""
    s = s.replace("glmmhd_alpha = 0.1\n", f"glmmhd_alpha = {alpha!r}{'' if case == 'prod' else '   # I5 case ' + case + ' (production 0.1)'}\n")
    s += """
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = cons
dt = 0.05

<parthenon/output1>
file_type = hst
dn = 5
"""
    return s


def main():
    runs = []
    for dim, Ns in ((1, (64, 128, 256, 512, 1024)), (2, (64, 128, 256, 512))):
        for case in ALPHA:
            for N in Ns:
                name = f"I5_{dim}d_{case}_N{N}"
                write(f"{M}/inputs/I5/{name}.in", deck(dim, N, case))
                runs.append(f"I5/{name} | I5/{name}.in | | 1")
    for lev in (1, 2):
        for case in ("a0", "prod"):
            name = f"I5b_2d_{case}_N128_smr{lev}"
            write(f"{M}/inputs/I5/{name}.in", deck(2, 128, case, smr_levels=lev))
            runs.append(f"I5/{name} | I5/{name}.in | | 1")
    # I5c: pulse on the COARSE level (base N 64, rb = 3.2 cells), refined corner patch of 0..3 levels: lambda =
    # alpha c_h / mindx grows 2x per patch level although the pulse's own cells never change (audit F-11 question)
    for lev in (0, 1, 2, 3):
        name = f"I5c_2d_prod_N64_patch{lev}"
        write(f"{M}/inputs/I5/{name}.in", deck(2, 64, "prod", patch_levels=lev, mb_override=(8, 8, 1)))
        runs.append(f"I5/{name} | I5/{name}.in | | 1")
    # time-splitting test of the psi damping: fixed lambda 10, N 512, cfl 0.3 / 0.15 / 0.075 (first order in dt?)
    for cfl in (0.15, 0.075):
        for dim in (1, 2):
            name = f"I5_{dim}d_lam10_N512_cfl{cfl}"
            write(f"{M}/inputs/I5/{name}.in", deck(dim, 512, "lam10").replace("cfl = 0.3\n", f"cfl = {cfl}   # I5 splitting test (production 0.3)\n"))
            runs.append(f"I5/{name} | I5/{name}.in | | 1")
    write(f"{M}/inputs/I5_runs.txt", "\n".join(runs) + "\n")
    print(len(runs), "decks")


if __name__ == "__main__":
    main()
