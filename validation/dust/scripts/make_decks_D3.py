#!/usr/bin/env python
"""D3: one-zone sublimation in the code (eos_test mode = compress, eos_homologous ghosts carrying the scalars).

D3a  homologous COMPRESSION from (1e-9 g/cm^3, 1250 K) by 8x: the table adiabat crosses T_subl = 1500 K at ~1.6x and
     reaches ~2000 K (H2 dissociation zone, Gamma_1 ~ 1.1). Exact: T rises monotonically, so after every React
     f_dg = f_ref subl(T) and a_c = a_growth (f/f_ref)^(1/3).
D3b  homologous EXPANSION (alpha < 0) from (1e-9, 1900 K) by 8x: the first React destroys the grains down to
     f_ref subl(T_1); the gas then cools through T_subl and below; irreversibility => f_dg, a_c frozen at their
     first-React values for the rest of the run (bitwise).
Variants: 'frz' = dust/freeze_growth = true (a_c is then exactly a_ref (f/f_ref)^(1/3)); 'prod' = production keys
(growth on; f_dg is unaffected by growth, so its exact answer is the same).
Dumps every ~10 % of the run (prim + cons) -> T(t) of the box from the table, compared dump by dump.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, sys
sys.path.insert(0, _VT + "/dust/scripts")
from du_common import DU, hydro_block, dust_block, mesh_block, write

RHO_U = 5.467e-19


def deck(name, rho_cgs, T0, alpha, tlim, frz):
    over = {"freeze_growth": "true"} if frz else {}
    return f"""<comment>
problem = dust {name}: homologous {'compression' if alpha > 0 else 'expansion'} rho0 {rho_cgs:g} g/cm^3 T0 {T0} K alpha {alpha}

<job>
problem_id = eos_test

<problem/eos_test>
mode = compress
alpha = {alpha}
xc = 0.5
rho0 = {rho_cgs / RHO_U!r}
T0 = {T0}
scalar0 = 0.01
scalar1 = 1.0e-5

{mesh_block((16, 1, 1), (0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5), (16, 1, 1), bcs={1: ('eos_homologous', 'eos_homologous')})}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {tlim!r}
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
dt = {tlim / 20!r}

<parthenon/output1>
file_type = hst
dn = 1
"""


def main():
    runs, meta = [], {}
    for name, rho, T0, alpha, tlim in (("D3a", 1e-9, 1250.0, 1.0, 0.875), ("D3b", 1e-9, 1900.0, -1.0, 7.0)):
        for v in ("frz", "prod"):
            n = f"{name}_{v}"
            write(f"{DU}/inputs/D3/{n}.in", deck(n, rho, T0, alpha, tlim, v == "frz"))
            runs.append(f"D3/{n} | D3/{n}.in | | 1")
            meta[n] = dict(rho_cgs=rho, T0=T0, alpha=alpha, tlim=tlim, freeze=(v == "frz"))
    write(f"{DU}/inputs/D3_runs.txt", "\n".join(runs) + "\n")
    write(f"{DU}/inputs/D3_meta.json", json.dumps(meta, indent=1))
    print("\n".join(runs))


if __name__ == "__main__":
    main()
