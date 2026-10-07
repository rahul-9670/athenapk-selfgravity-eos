#!/usr/bin/env python
"""Decks for the tabulated-EOS suite. Production EOS keys verbatim (eos = hydrogen, v5_cold table, hlld, plm,
eos_floor_consistent, default 20 bisections)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import eostab as E
INP = os.path.join(os.path.dirname(__file__), "..", "inputs")
os.makedirs(INP, exist_ok=True)
t = E.load()
STATES = {"S1": (1e-18, 10.0), "S2": (1e-15, 100.0), "S3": (1e-12, 500.0), "S4": (1e-9, 2000.0), "S5": (1e-8, 1.0e4)}
meta = {}
HYDRO = f"""<hydro>
fluid = glmmhd
eos = hydrogen
eos_table_file = {E.TABLE}
gamma = 1.4
riemann = hlld
reconstruction = plm
glmmhd_source = dedner_plain   # 1D decks: dedner_extended (production, 3D) indexes y/z neighbours and segfaults in 1D; B = 0 here
eos_floor_consistent = true
dfloor = 1.0e-12
pfloor = 1.0e-14
"""


def deck(name, comment, mode_block, nx, tlim, extra_hydro="", bc="periodic", dt_out=None):
    txt = f"""<comment>
problem = {comment}

<job>
problem_id = eos_test

<problem/eos_test>
{mode_block}
<parthenon/mesh>
refinement = none
nx1 = {nx}
x1min = 0.0
x1max = 1.0
ix1_bc = {bc}
ox1_bc = {bc}
nx2 = 1
x2min = -0.5
x2max = 0.5
ix2_bc = periodic
ox2_bc = periodic
nx3 = 1
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic

<parthenon/meshblock>
nx1 = {nx}
nx2 = 1
nx3 = 1

<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {tlim!r}
nlim = -1

{HYDRO}{extra_hydro}
<physics>
self_gravity = false

<parthenon/output0>
file_type = hdf5
variables = prim
dt = {dt_out if dt_out else tlim!r}
"""
    open(os.path.join(INP, name + ".in"), "w").write(txt)


for s, (rho_cgs, T) in STATES.items():
    rho = rho_cgs / E.RHO_U
    esp = E.esp_of(t, rho, T)
    cs = float(np.sqrt(E.cs2_of(t, rho, esp)))
    P = float(E.P_of(t, rho, esp))
    meta[s] = dict(rho_cgs=rho_cgs, T=T, rho_code=rho, esp_code=float(esp), P_code=P, cs_code=cs,
                   gamma1=cs * cs * rho / P, T_check=float(E.T_of(t, rho, esp)))
    for nx in (16, 32, 64, 128, 256):
        deck(f"E6_wave_{s}_n{nx}", f"tabulated-EOS acoustic wave, state {s}: rho {rho_cgs:g} g/cc, T {T:g} K, cs {cs:.6g} code, one period",
             f"mode = wave\nrho0 = {rho!r}\nT0 = {T!r}\namp = 1.0e-6\n", nx, 1.0 / cs)
    for nx in (64, 256):
        deck(f"E6_wave_{s}_n{nx}_prho", f"as E6_wave_{s}_n{nx} with hydro/eos_interp = p_over_rho (E-F1 fix candidate)",
             f"mode = wave\nrho0 = {rho!r}\nT0 = {T!r}\namp = 1.0e-6\n", nx, 1.0 / cs, extra_hydro="eos_interp = p_over_rho\n")
    deck(f"E6_wave_{s}_n128_bis60", f"as E6_wave_{s}_n128 with 60 P->e bisections (production: 20)",
         f"mode = wave\nrho0 = {rho!r}\nT0 = {T!r}\namp = 1.0e-6\n", 128, 1.0 / cs, extra_hydro="eos_bisections = 60\n")
json.dump(meta, open(os.path.join(INP, "eos_states.json"), "w"), indent=1)
for s, m in meta.items():
    print(s, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items() if k in ("rho_code", "T", "cs_code", "gamma1", "T_check")})
