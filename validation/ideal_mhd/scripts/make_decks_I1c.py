#!/usr/bin/env python
"""I1c linear waves with the PRODUCTION tabulated EOS (P tier: eos = hydrogen, rho26 table, double flux, Tfloor,
eos_interp = p, dual energy keys): pgen eos_test mode = wave, state S1 of the EOS module (rho0 1.8291567587342235
code = 1e-18 g/cc, T0 10 K), amplitude 1e-6, one period on [0,1], N 16 .. 512.
  ac      acoustic (B = 0): the EOS module's E6 wave, now in the production key set
  fast    perpendicular fast magnetosonic mode (wave_fast = true) in B0y = cs sqrt(rho0) (v_A = cs, beta ~ 1.2)
  fastlb  B0y = 7 cs sqrt(rho0) (beta ~ 0.025: e_int/E < eta1, the dual branch selects); and fastlb_nd (dual off)
cs^2 from the rho26 table exactly as the pgen evaluates it (bilinear in log rho, log esp; esp0 from (rho0, T0)).
Writes inputs/I1c/*.in and inputs/I1c_runs.txt.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import math, sys
sys.path.insert(0, _VT + "/ideal_mhd/scripts")
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
from im_common import M, EOS_TABLE, hydro_block, mesh_block, write
import eostab as E

RHO0, T0 = 1.8291567587342235, 10.0


def deck(N, B0y, dual=True):
    t = E.load(EOS_TABLE)
    esp0 = E.esp_of(t, RHO0, T0)
    cs2 = E.cs2_of(t, RHO0, esp0)
    cf = math.sqrt(cs2 + B0y * B0y / RHO0)
    period = 1.0 / cf
    h = hydro_block("P", clamp=False, dual=dual)
    s = f"""<comment>
problem = ideal_mhd I1c tabulated-EOS wave N={N} B0y={B0y!r} cs={math.sqrt(cs2)!r} cf={cf!r} dual={dual}

<job>
problem_id = eos_test

<problem/eos_test>
mode = wave
rho0 = {RHO0!r}
T0 = {T0!r}
amp = 1.0e-6
B0y = {float(B0y)!r}
wave_fast = {'true' if B0y else 'false'}

{mesh_block((N, 1, 1), (0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5), (N, 1, 1))}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {period!r}
nlim = -1

{h}
<physics>
self_gravity = false
radiation = false

<parthenon/output0>
file_type = hdf5
variables = prim
dt = {period!r}
"""
    return s, math.sqrt(cs2)


def main():
    runs = []
    _, cs = deck(16, 0.0)
    for tag, B0y, dual in (("ac", 0.0, True), ("fast", cs * math.sqrt(RHO0), True),
                           ("fastlb", 7 * cs * math.sqrt(RHO0), True), ("fastlb_nd", 7 * cs * math.sqrt(RHO0), False)):
        for N in (16, 32, 64, 128, 256, 512):
            s, _ = deck(N, float(B0y), dual)
            name = f"I1c_{tag}_N{N}"
            write(f"{M}/inputs/I1c/{name}.in", s)
            runs.append(f"I1c/{name} | I1c/{name}.in | | 1")
    write(f"{M}/inputs/I1c_runs.txt", "\n".join(runs) + "\n")
    print(len(runs), "decks; cs =", cs)


if __name__ == "__main__":
    main()
