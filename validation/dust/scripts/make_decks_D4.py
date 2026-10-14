#!/usr/bin/env python
"""D4: advection of the dust pair through periodic moving slabs (eos_test mode = slab, per-side scalars).

Exact solution for every variant: pure translation at uniform P (after an integer number of crossings, the IC).
  D4a  SCALAR front: uniform rho (1e-12 g/cm^3), T 20 K; f_dg = f_ref | 0.5 f_ref with a_c = a_ref | a_ref 0.5^(1/3)
       (fixed grain number). Checks: bounds (f <= f_ref, a in [a_R, a_L]), dust-mass budget (periodic => any change
       in sum(rho f dV) is React's doing: the f > f_ref clamp), L1 profile error vs N.
  D4b  DENSITY contact x10 at uniform P (1e-12 | 1e-13 g/cm^3, 20 K | T_R from the table): uniform f_ref, a_ref,
       freeze_growth = true (the two sides would otherwise grow at different rates). Exact: f, a uniform forever.
  D4c  TEMPERATURE contact across T_subl at uniform P: cold 700 K at 1e-9 g/cm^3 | hot 1900 K (rho_H from the table),
       each side seeded at its own equilibrium (f_ref subl(T), a_ref subl^(1/3)); freeze_growth = true (variant
       'prod': growth on). Exact: no dust is destroyed (the gas never changes temperature). Numerically the
       contact smears T; cells whose mixed T exceeds ~1200 K sublimate, IRREVERSIBLY: the measured dust loss is a
       pure numerical artefact (PLAN open question 3).
  D4d  restart: D4c N 256 run in two legs (restart at tlim/2) vs one leg -> bitwise (separate script run_D4d.sh).
1D: mx = 1 (two contacts), N = 64 .. 1024, 10 crossings. 2D: mx = my = 1 oblique, N = 128^2, 32^2 blocks, 16 ranks.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, sys
import numpy as np
sys.path.insert(0, _VT + "/dust/scripts")
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
from du_common import DU, EOS_TABLE, hydro_block, dust_block, mesh_block, write
import eostab

RHO_U, V_U = 5.467e-19, 1.9e4
F_REF, A_REF, TSUB, W = 0.01, 1e-5, 1500.0, 100.0
subl = lambda T: 0.5 * (1 - np.tanh((T - TSUB) / W))


def rho_for_P(tab, P, T, guess):
    lo, hi = np.log10(guess) - 3, np.log10(guess) + 3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        r = 10 ** mid
        if eostab.P_of(tab, r, eostab.esp_of(tab, r, T)) > P: hi = mid
        else: lo = mid
    return 10 ** (0.5 * (lo + hi))


def deck(name, dim, N, L, R, scL, scR, u, tlim, over, nr, extra_out=""):
    # plain Python floats only: a numpy float prints as "np.float64(x)", which the deck parser reads as 0 (trap)
    L, R, scL, scR, u, tlim = [float(v) for v in L], [float(v) for v in R], [float(v) for v in scL], \
        [float(v) for v in scR], float(u), float(tlim)
    nx = (N, N if dim == 2 else 1, 1)
    mb = (min(N, 32 if dim == 2 else N), min(N, 32) if dim == 2 else 1, 1)
    return f"""<comment>
problem = dust {name}

<job>
problem_id = eos_test

<problem/eos_test>
mode = slab
mx = 1
my = {1 if dim == 2 else 0}
mz = 0
u_slab = {u!r}
rhoL = {L[0]!r}
TL = {L[1]!r}
rhoR = {R[0]!r}
TR = {R[1]!r}
scalarL0 = {scL[0]!r}
scalarL1 = {scL[1]!r}
scalarR0 = {scR[0]!r}
scalarR1 = {scR[1]!r}

{mesh_block(nx, (0.0, 1.0), (0.0, 1.0) if dim == 2 else (-0.5, 0.5), (-0.5, 0.5), mb)}
<parthenon/time>
integrator = vl2
cfl = 0.3
tlim = {tlim!r}
nlim = -1

{hydro_block(dim, clamp=False)}
<physics>
self_gravity = false
radiation = false
dust = true

{dust_block(**over)}
<parthenon/output0>
file_type = hdf5
variables = prim, cons
dt = {tlim / 10!r}

<parthenon/output1>
file_type = hst
dn = 1
{extra_out}"""


def main():
    tab = eostab.load(EOS_TABLE, RHO_U, V_U)
    runs, meta = [], {}
    # --- states (code units) ---
    r12 = 1e-12 / RHO_U
    cs20 = np.sqrt(eostab.cs2_of(tab, r12, eostab.esp_of(tab, r12, 20.0)))
    P_b = eostab.P_of(tab, r12, eostab.esp_of(tab, r12, 20.0))
    r13 = 1e-13 / RHO_U
    lo, hi = 20.0, 2000.0   # T_R with P(r13, T_R) = P_b
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eostab.P_of(tab, r13, eostab.esp_of(tab, r13, mid)) > P_b: hi = mid
        else: lo = mid
    T_R = 0.5 * (lo + hi)
    rC = 1e-9 / RHO_U
    P_c = eostab.P_of(tab, rC, eostab.esp_of(tab, rC, 700.0))
    rH = rho_for_P(tab, P_c, 1900.0, rC / 3)
    csC = np.sqrt(eostab.cs2_of(tab, rC, eostab.esp_of(tab, rC, 700.0)))
    sC, sH = subl(700.0), subl(1900.0)
    variants = {
        "D4a": dict(L=(r12, 20.0), R=(r12, 20.0), scL=(F_REF, A_REF), scR=(0.5 * F_REF, A_REF * 0.5 ** (1 / 3)),
                    u=float(cs20), over={}),
        "D4afrz": dict(L=(r12, 20.0), R=(r12, 20.0), scL=(F_REF, A_REF), scR=(0.5 * F_REF, A_REF * 0.5 ** (1 / 3)),
                       u=float(cs20), over={"freeze_growth": "true"}),   # D4a with growth frozen: a_c bounds
        "D4b": dict(L=(r12, 20.0), R=(r13, T_R), scL=(F_REF, A_REF), scR=(F_REF, A_REF), u=float(cs20),
                    over={"freeze_growth": "true"}),
        "D4c": dict(L=(rC, 700.0), R=(rH, 1900.0), scL=(F_REF * sC, A_REF * np.cbrt(sC)),
                    scR=(F_REF * sH, A_REF * np.cbrt(sH)), u=float(csC), over={"freeze_growth": "true"}),
        "D4cprod": dict(L=(rC, 700.0), R=(rH, 1900.0), scL=(F_REF * sC, A_REF * np.cbrt(sC)),
                        scR=(F_REF * sH, A_REF * np.cbrt(sH)), u=float(csC), over={}),
    }
    for v, p in variants.items():
        for dim, Ns in ((1, (64, 128, 256, 512, 1024)), (2, (128,))):
            if v == "D4afrz":
                Ns = (64, 256, 1024) if dim == 1 else ()
            if v == "D4cprod":
                Ns = (256,) if dim == 1 else ()
            for N in Ns:
                lam = 1.0 if dim == 1 else 1.0 / np.sqrt(2.0)
                tlim = 10 * lam / p["u"]
                name = f"{v}_{dim}d_N{N}"
                nr = 1 if dim == 1 else 16
                write(f"{DU}/inputs/D4/{name}.in", deck(name, dim, N, p["L"], p["R"], p["scL"], p["scR"], p["u"],
                                                         float(tlim), p["over"], nr))
                runs.append(f"D4/{name} | D4/{name}.in | | {nr}")
                meta[name] = dict(variant=v, dim=dim, N=N, tlim=float(tlim), u=p["u"], L=list(map(float, p["L"])),
                                  R=list(map(float, p["R"])), scL=list(map(float, p["scL"])),
                                  scR=list(map(float, p["scR"])), over=p["over"])
    # D4d: two-leg restart of D4c 1d N256 (restart output at tlim/2)
    m = meta["D4c_1d_N256"]
    p = variants["D4c"]
    rst = f"""
<parthenon/output2>
file_type = rst
dt = {m['tlim'] / 2!r}
"""
    write(f"{DU}/inputs/D4/D4d_1d_N256.in", deck("D4d_1d_N256 (two-leg restart of D4c)", 1, 256, p["L"], p["R"],
                                                  p["scL"], p["scR"], p["u"], m["tlim"], p["over"], 1, rst))
    meta["D4d_1d_N256"] = dict(m, variant="D4d")
    write(f"{DU}/inputs/D4_runs.txt", "\n".join(runs) + "\n")
    write(f"{DU}/inputs/D4_meta.json", json.dumps(meta, indent=1))
    print("\n".join(runs)); print("T_R(D4b) =", T_R, " rho_H/rho_C(D4c) =", rH / rC)


if __name__ == "__main__":
    main()
