"""D1 decks: hypersonic flow of cold molecular gas (rho = 1 code = 5.467e-19 g/cc, T = 10 K, rho26 table, eos_interp p).

D1a  slab: periodic pure contact rho 1 | 2 (10 K | ~5 K) at equal P (e_R from the interpolant's exact inverse), uniform velocity
     u = M cs0, M in {0, 1, 10, 1e2, 1e3, 1e4}, N = 256, one box crossing. Exact: translation at P = P0.
D1b  sinus: v = u0 + M cs0 sin(2 pi x), uniform rho/T, to 0.8 t_c (t_c = 1/(vamp 2 pi) = caustic). Smooth before
     the caustic -> every cell stays on the ONE table adiabat through (1, 10 K). M in {1, 3, 10, 30, 1e2, 1e3, 1e4},
     N in {64 .. 1024} (+2048 .. 16384 for M = 10 .. 1000: the resolved regime needs dv per cell < cs, N >~ 10 M), u0 = 0; boosted u0 = 10 vamp at N = 256.
Writes inputs/D1*.in and inputs/D1_runs.txt (run dir | deck | overrides); dual OFF = override hydro/dual_energy=false."""
import json
import os
import numpy as np
from de_common import INP, deck, fmt, hydro_block, mesh_block, table_eos

eos = table_eos()
RHO0, T0 = 1.0, 10.0
e0 = float(eos.esp_of_T(RHO0, T0))
P0 = float(eos.P(RHO0, e0))
import eostab as E
from de_common import TABLE
cs0 = float(np.sqrt(E.cs2_of(E.load(TABLE), RHO0, e0)))
runs, meta = [], {}
OFF = "hydro/dual_energy=false"

# ---- D1a
rhoR = 2.0   # T_R ~ 5 K: the first version (rho 4, 2.5 K) sat BELOW the production Tfloor 2.73 K
eR = eos.e_of_P(rhoR, P0)
for M in (0, 1, 10, 100, 1000, 10000):
    u = M * cs0
    tl = 1.0 / u if M > 0 else 1.0 / cs0
    name = f"D1a_M{M:g}"
    prob = f"""<problem/eos_test>
mode = slab
mx = 1
rhoL = {fmt(RHO0)}
espL_code = {fmt(e0)}
rhoR = {fmt(rhoR)}
espR_code = {fmt(eR)}
u_slab = {fmt(u)}
"""
    txt = deck(f"D1a pure contact 1|2 at P0, Mach {M} (u = {u:.6g}), one crossing", prob, mesh_block(256), tl, tl,
               hydro_block(1), hst_dt=tl / 200)
    open(os.path.join(INP, name + ".in"), "w").write(txt)
    meta[name] = dict(test="D1a", M=M, u=u, tlim=tl, rhoL=RHO0, eL=e0, rhoR=rhoR, eR=eR, P0=P0, N=256)
    for tag, ov in (("on", ""), ("off", OFF), ("on_nodf", "hydro/double_flux=false"),
                    ("off_nodf", OFF + " hydro/double_flux=false")):
        runs.append((f"D1a/{name}_{tag}", name + ".in", ov))

# ---- D1b
def d1b(M, N, boost):
    vamp = M * cs0
    tc = 1.0 / (vamp * 2 * np.pi)
    u0 = 10.0 * vamp if boost else 0.0
    name = f"D1b_M{M:g}_N{N}" + ("_boost" if boost else "")
    prob = f"""<problem/eos_test>
mode = sinus
rho0 = {fmt(RHO0)}
T0 = {fmt(T0)}
vamp = {fmt(vamp)}
u0 = {fmt(u0)}
mx = 1
"""
    txt = deck(f"D1b cold sinusoid Mach {M} (vamp {vamp:.6g}, u0 {u0:.6g}), N {N}, to 0.8 t_c (t_c {tc:.6g})", prob,
               mesh_block(N), 0.8 * tc, 0.2 * tc, hydro_block(1), hst_dt=0.01 * tc)
    open(os.path.join(INP, name + ".in"), "w").write(txt)
    meta[name] = dict(test="D1b", M=M, N=N, vamp=vamp, u0=u0, tc=tc, rho0=RHO0, T0=T0, e0=e0)
    for tag, ov in (("on", ""), ("off", OFF)):
        runs.append((f"D1b/{name}_{tag}", name + ".in", ov))

for M in (1, 3, 10, 30, 100, 1000, 10000):
    for N in (64, 128, 256, 512, 1024) + ((2048, 4096, 8192, 16384) if M in (10, 30, 100, 1000) else ()):
        d1b(M, N, False)
    d1b(M, 256, True)

with open(os.path.join(INP, "D1_runs.txt"), "w") as f:
    for r in runs:
        f.write(" | ".join(r) + "\n")
json.dump(dict(P0=P0, e0=e0, cs0=cs0, runs=meta), open(os.path.join(INP, "D1_meta.json"), "w"), indent=1)
print(f"e0 {e0:.10g} P0 {P0:.10g} cs0 {cs0:.10g} eR {eR:.10g}; {len(runs)} runs")
