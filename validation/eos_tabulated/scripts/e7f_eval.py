#!/usr/bin/env python
"""E7f: plateau errors (P*, u*) and L1(rho) for the bisection / high-resolution legs in runs/E7f, against the same
exact solutions as analyze_E7.py. -> results/tables/E7f_bisect.txt"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
from analyze_E7 import read, exact_profile, NFINE, X0
HERE = os.path.dirname(os.path.abspath(__file__))
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json")))
t = E.load(); eos = RX.TableEOS(t, "p_over_rho")
cache, lines = {}, []
SUB = sys.argv[1] if len(sys.argv) > 1 else "E7f"; OUTN = sys.argv[2] if len(sys.argv) > 2 else "E7f_bisect.txt"
for d in sorted(glob.glob(os.path.join(HERE, "..", "runs", SUB, "*")), key=lambda q: (os.path.basename(q).rsplit("_n", 1)[0], int(os.path.basename(q).rsplit("_n", 1)[1]) if "_n" in os.path.basename(q) and os.path.basename(q).rsplit("_n", 1)[1].isdigit() else 0)):
    f = glob.glob(os.path.join(d, "*.out0.final.phdf"))
    if not f:
        lines.append(f"{os.path.basename(d):22s} (no final dump)"); continue
    prob = {"R1": "R1_molecular", "R2": "R2_dissociation", "R3": "R3_dissoc_shock", "R4": "R4_ionization"}[os.path.basename(d)[:2]]
    rec = ex[prob + "|p_over_rho"]
    tt, rho, u, P, nc = read(f[0])
    if prob not in cache:
        cache[prob] = exact_profile(eos, rec, tt) + (tt,)
    so, fine, tf = cache[prob]
    n = rho.size; x = (np.arange(n) + 0.5) / n; xi = (x - X0) / tt
    edge = lambda s: so[s]["S"] if so[s]["type"] == "shock" else so[s]["tail"]
    aL, bL, aR, bR = edge("L"), so["u"], so["u"], edge("R")
    m = ((xi > aL + 0.25 * (bL - aL)) & (xi < bL - 0.25 * (bL - aL))) | ((xi > aR + 0.25 * (bR - aR)) & (xi < bR - 0.25 * (bR - aR)))
    L1 = np.nan
    if NFINE % n == 0:
        L1 = float(np.mean(np.abs(rho - fine[0].reshape(n, -1).mean(1))) / np.abs(fine[0]).max())
    lines.append(f"{os.path.basename(d):22s} N {n:6d} cycles {nc:7d} | plateau dP* {np.median(P[m]) / so['P'] - 1:+.3e}  du*/vmax {(np.median(u[m]) - so['u']) / max(abs(so['u']), rec['vmax']):+.3e} | L1 rho {L1:.3e}")
txt = "\n".join(lines); print(txt)
open(os.path.join(HERE, "..", "results", "tables", OUTN), "w").write(txt + "\n")
