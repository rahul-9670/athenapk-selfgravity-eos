"""D1x: isolate the D1a contact pressure error with dual energy on (DE-F3). Base = D1a decks (rho 1|2 at P0, N 256,
u = M cs0), one variable changed per run (CLI overrides or a new density ratio):
  dc      hydro/reconstruction=dc  (first order: rho_face e_face = (rho e)_i -> exact if the PLM-product mechanism holds)
  N<n>    resolution 64 .. 1024
  x5      five box crossings (growth in time?)
  eta1e-3 hydro/dual_energy_eta1=1e-3 (M = 10: e/E ~ 0.03 > 1e-3 -> nothing switches -> must equal dual OFF)
  r1.1 / r10  density ratio 1|1.1 and 1|10 (T_L = 100 K so that T_R = 10 K stays above the 2.73 K Tfloor)
Writes inputs/D1x_*.in (new ratios only), inputs/D1x_runs.txt, inputs/D1x_meta.json."""
import json
import os
import numpy as np
from de_common import INP, deck, fmt, hydro_block, mesh_block, table_eos
import eostab as E  # noqa: E402
from de_common import TABLE

eos = table_eos()
m1 = json.load(open(os.path.join(INP, "D1_meta.json")))
cs0, P0, e0 = m1["cs0"], m1["P0"], m1["e0"]
OFF = "hydro/dual_energy=false"
runs, meta = [], {}


def pair(name, deckname, ov, m):
    meta[name] = m
    for tag, o in (("on", ov), ("off", (OFF + " " + ov).strip())):
        runs.append((f"D1x/{name}_{tag}", deckname, o))


for M in (10, 100):
    base = f"D1a_M{M}.in"
    u = M * cs0
    pair(f"D1x_M{M}_dc", base, "hydro/reconstruction=dc", dict(M=M, var="dc", rhoR=2.0, N=256, ncross=1))
    pair(f"D1x_M{M}_x5", base, f"parthenon/time/tlim={fmt(5.0 / u)}", dict(M=M, var="x5", rhoR=2.0, N=256, ncross=5))
    for N in (64, 128, 512, 1024):
        pair(f"D1x_M{M}_N{N}", base, f"parthenon/mesh/nx1={N} parthenon/meshblock/nx1={N}",
             dict(M=M, var=f"N{N}", rhoR=2.0, N=N, ncross=1))
pair("D1x_M10_eta1e-3", "D1a_M10.in", "hydro/dual_energy_eta1=1e-3", dict(M=10, var="eta1e-3", rhoR=2.0, N=256, ncross=1))

for ratio, TL in ((1.1, 10.0), (10.0, 100.0)):
    eL = float(eos.esp_of_T(1.0, TL)); PL = float(eos.P(1.0, eL)); eR = eos.e_of_P(ratio, PL)
    for M in (10, 100):
        csL = float(np.sqrt(E.cs2_of(E.load(TABLE), 1.0, eL)))
        u = M * cs0   # same speed as D1a (Mach w.r.t. the 10 K reference gas)
        name = f"D1x_M{M}_r{ratio:g}"
        prob = f"""<problem/eos_test>
mode = slab
mx = 1
rhoL = 1.0
espL_code = {fmt(eL)}
rhoR = {fmt(ratio)}
espR_code = {fmt(eR)}
u_slab = {fmt(u)}
"""
        txt = deck(f"D1x contact 1|{ratio:g} (T_L {TL:g} K) at equal P, u = {M} cs0(10 K), one crossing", prob,
                   mesh_block(256), 1.0 / u, 1.0 / u, hydro_block(1), hst_dt=1.0 / u / 200)
        open(os.path.join(INP, name + ".in"), "w").write(txt)
        pair(name, name + ".in", "", dict(M=M, var=f"r{ratio:g}", rhoR=ratio, N=256, ncross=1, P0=PL, TL=TL, csL=csL))

with open(os.path.join(INP, "D1x_runs.txt"), "w") as f:
    for r in runs:
        f.write(" | ".join(r) + "\n")
json.dump(dict(runs=meta, P0=P0), open(os.path.join(INP, "D1x_meta.json"), "w"), indent=1)
print(len(runs), "runs")
