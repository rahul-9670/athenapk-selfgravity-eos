"""D2 decks: magnetically dominated cold gas (rho = 1 code, T = 10 K, rho26 table, eos_interp p). Field strength by the
plasma beta of the initial state, beta = 2 P0 / B0^2 (Heaviside-Lorentz code units: P_mag = B^2/2).

D2a  1D, the D1b cold sinusoid v = M cs0 sin(2 pi x) at M = 3 (resolved, never switched kinetically) with a uniform
     field, beta in {1, 1e-1, 1e-2, 1e-3, 1e-4}:
       par : B0 = B0x along the flow -> no force, the hydro is D1b exactly; only e/E ~ beta switches the cells.
             Exact: the one table adiabat, to 0.8 t_c.
       perp: B0 = B0y across the flow -> B_y/rho is frozen in, magnetic pressure resists (smooth magnetosonic flow while
             the fast Mach number stays < 1). Exact: the adiabat AND B_y/rho = B0 (flux freezing), to 0.4 t_c.
     N in {256, 1024}.
D2c  1D contact rho 1 | 2 at equal P (D1a) with a uniform field, par (Bx) and perp (By, then P + B^2/2 is still uniform),
     u = 10 cs0, beta in {1, 1e-2, 1e-4}, N = 256, one crossing. Exact: translation at P = P0.
D2b  2D div B blob (eos_test mode divb): uniform rho, T, v = (u0, u0) with |v| = M cs0, uniform B0z (out of plane, sets
     beta), plus Bx += bamp (r^8 - 2 r^4 + 1), r = |x|/rb < 1, rb = 0.125 (Dedner et al. 2002 shape). Production
     glmmhd_source = dedner_extended. Nothing physical heats the gas: every cell must stay on the adiabat through the
     initial state (the extended GLM momentum source -B div B can push the gas, reversibly). beta in {1, 1e-2, 1e-4},
     bamp/B0z in {0.1, 1}, M in {0, 10}; N = 128^2 (+256^2 for beta 1e-4, bamp 0.1, M 0); 32^2 meshblocks.
Writes inputs/D2*.in, inputs/D2_runs.txt, inputs/D2_meta.json; dual OFF = override hydro/dual_energy=false."""
import json
import os

import numpy as np


from de_common import INP, TABLE, deck, fmt, hydro_block, mesh_block, table_eos
import eostab as E  # noqa: E402 (path set by de_common)

eos = table_eos()
RHO0, T0 = 1.0, 10.0
e0 = float(eos.esp_of_T(RHO0, T0))
P0 = float(eos.P(RHO0, e0))
cs0 = float(np.sqrt(E.cs2_of(E.load(TABLE), RHO0, e0)))
OFF = "hydro/dual_energy=false"
runs, meta = [], {}


def add(name, txt, m, nranks=1):
    open(os.path.join(INP, name + ".in"), "w").write(txt)
    meta[name] = m
    for tag, ov in (("on", ""), ("off", OFF)):
        runs.append((f"{m['test']}/{name}_{tag}", name + ".in", ov, str(nranks)))


# ---- D2a
M = 3
vamp = M * cs0
tc = 1.0 / (vamp * 2 * np.pi)
for geo in ("par", "perp"):
    for beta in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        B0 = np.sqrt(2 * P0 / beta)
        for N in (256, 1024):
            tl = (0.8 if geo == "par" else 0.4) * tc
            name = f"D2a_{geo}_b{beta:g}_N{N}"
            prob = f"""<problem/eos_test>
mode = sinus
rho0 = {fmt(RHO0)}
T0 = {fmt(T0)}
vamp = {fmt(vamp)}
u0 = 0.0
mx = 1
B0x = {fmt(B0 if geo == "par" else 0.0)}
B0y = {fmt(B0 if geo == "perp" else 0.0)}
"""
            txt = deck(f"D2a cold sinusoid M {M}, {geo} field beta {beta:g} (B0 {B0:.6g}), N {N}, to {tl / tc:.1f} t_c",
                       prob, mesh_block(N), tl, tl / 4, hydro_block(1), hst_dt=tl / 100)
            add(name, txt, dict(test="D2a", geo=geo, beta=beta, B0=B0, N=N, M=M, vamp=vamp, tc=tc, e0=e0, P0=P0))

# ---- D2c
rhoR = 2.0
eR = eos.e_of_P(rhoR, P0)
u = 10 * cs0
for geo in ("par", "perp"):
    for beta in (1.0, 1e-2, 1e-4):
        B0 = np.sqrt(2 * P0 / beta)
        name = f"D2c_{geo}_b{beta:g}"
        prob = f"""<problem/eos_test>
mode = slab
mx = 1
rhoL = {fmt(RHO0)}
espL_code = {fmt(e0)}
rhoR = {fmt(rhoR)}
espR_code = {fmt(eR)}
u_slab = {fmt(u)}
B0x = {fmt(B0 if geo == "par" else 0.0)}
B0y = {fmt(B0 if geo == "perp" else 0.0)}
"""
        txt = deck(f"D2c contact 1|2 at P0, u = 10 cs0, {geo} field beta {beta:g}, one crossing", prob, mesh_block(256),
                   1.0 / u, 1.0 / u, hydro_block(1), hst_dt=1.0 / u / 200)
        add(name, txt, dict(test="D2c", geo=geo, beta=beta, B0=B0, u=u, rhoR=rhoR, eR=eR, e0=e0, P0=P0))

# ---- D2b
rb = 0.125
for beta in (1.0, 1e-2, 1e-4):
    B0z = np.sqrt(2 * P0 / beta)
    for brel in (0.1, 1.0):
        for M2 in (0, 10):
            for N in ((128, 256) if (beta == 1e-4 and brel == 0.1 and M2 == 0) else (128,)):
                u0 = M2 * cs0 / np.sqrt(2)
                tl = 0.25 / cs0 if M2 == 0 else 1.0 / u0   # static: ~3 cleaning-wave crossings of the box; moving: one period
                name = f"D2b_b{beta:g}_br{brel:g}_M{M2}_N{N}"
                prob = f"""<problem/eos_test>
mode = divb
rho0 = {fmt(RHO0)}
T0 = {fmt(T0)}
u0 = {fmt(u0)}
v0y = {fmt(u0)}
B0z = {fmt(B0z)}
bamp = {fmt(brel * B0z)}
rb = {fmt(rb)}
"""
                txt = deck(f"D2b div B blob, beta {beta:g}, bamp/B0z {brel:g}, |v| = {M2} cs0, N {N}^2", prob,
                           mesh_block(N, x1=(-0.5, 0.5), nx2=N, x2=(-0.5, 0.5), mb=(32, 32, 1)), tl, tl / 4,
                           hydro_block(2), hst_dt=tl / 200)
                add(name, txt, dict(test="D2b", beta=beta, B0z=B0z, bamp=brel * B0z, rb=rb, M=M2, u0=u0, N=N, tlim=tl,
                                    e0=e0, P0=P0), nranks=16 if N == 256 else 1)

with open(os.path.join(INP, "D2_runs.txt"), "w") as f:
    for r in runs:
        f.write(" | ".join(r) + "\n")
json.dump(dict(P0=P0, e0=e0, cs0=cs0, runs=meta), open(os.path.join(INP, "D2_meta.json"), "w"), indent=1)
print(f"{len(runs)} runs")
