#!/usr/bin/env python
"""D7: the dust pair at the PRODUCTION box boundary (diode on every face + boundary_flux_clamp + clamp_wall, IM-F2).

eos_test mode = divb with bamp = 0: a uniform state (1e-12 g/cm^3, 20 K, f_ref, a_ref) in uniform oblique motion
(u, v) = (2, 1) c_s through a 2D box, 64^2, 32^2 blocks, 4 ranks, 2 box crossings. The x1-inner / x2-inner faces are
INFLOW faces (the clamp makes them walls: zero mass flux, so zero scalar flux), the outer faces outflow.
Exact for the dust: f_dg = f_ref in every cell at every time, whatever the gas does at the walls (no dust-free gas
can enter if no gas enters). So sum(rho f dV) / (f_ref sum(rho dV)) - 1 = 0 and max |f/f_ref - 1| = 0 (to
round-off). a_c: growth on (production) differs only through the local (rho, T) near the walls; also a
freeze_growth variant where a_c must stay a_ref everywhere.
Variant 'noclamp': boundary_flux_clamp = false (diode alone lets the inflow in; the diode ghost copies the interior
scalars, so the inflowing gas still carries f_ref) -- contrast, not production.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, sys
sys.path.insert(0, _VT + "/dust/scripts")
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
import numpy as np
from du_common import DU, EOS_TABLE, hydro_block, dust_block, mesh_block, write
import eostab

RHO_U, V_U = 5.467e-19, 1.9e4


def main():
    tab = eostab.load(EOS_TABLE, RHO_U, V_U)
    rho = 1e-12 / RHO_U
    cs = float(np.sqrt(eostab.cs2_of(tab, rho, eostab.esp_of(tab, rho, 20.0))))
    tlim = float(2.0 / (2 * cs))
    runs, meta = [], {}
    for tag, clamp, over in (("prod", True, {}), ("frz", True, {"freeze_growth": "true"}), ("noclamp", False, {})):
        name = f"D7_{tag}"
        d = f"""<comment>
problem = dust {name}: uniform oblique flow (2, 1) c_s through diode faces, clamp {clamp}

<job>
problem_id = eos_test

<problem/eos_test>
mode = divb
rho0 = {float(rho)!r}
T0 = 20.0
u0 = {2 * cs!r}
v0y = {cs!r}
bamp = 0.0
rb = 0.1
scalar0 = 0.01
scalar1 = 1.0e-5

{mesh_block((64, 64, 1), (0.0, 1.0), (0.0, 1.0), (-0.5, 0.5), (32, 32, 1), bcs={1: ('diode', 'diode'), 2: ('diode', 'diode'), 3: ('periodic', 'periodic')})}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {tlim!r}
nlim = -1

{hydro_block(2, clamp=clamp)}
<physics>
self_gravity = false
radiation = false
dust = true

{dust_block(**over)}
<parthenon/output0>
file_type = hdf5
variables = prim, cons
dt = {tlim / 4!r}

<parthenon/output1>
file_type = hst
dn = 10
"""
        write(f"{DU}/inputs/D7/{name}.in", d)
        runs.append(f"D7/{name} | D7/{name}.in | | 4")
        meta[name] = dict(clamp=clamp, over=over, tlim=tlim, cs=cs)
    write(f"{DU}/inputs/D7_runs.txt", "\n".join(runs) + "\n")
    write(f"{DU}/inputs/D7_meta.json", json.dumps(meta, indent=1))
    print("\n".join(runs))


if __name__ == "__main__":
    main()
