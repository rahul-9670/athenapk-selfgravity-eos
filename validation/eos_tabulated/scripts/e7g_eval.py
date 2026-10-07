#!/usr/bin/env python
"""E7g: plateau P* error and L1(rho) of R2/R4 on the finer tables (runs/E7g), each against the exact solution of
its own table, next to the production-table numbers (E7e/E7f). -> results/tables/E7g_tables.txt"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
from analyze_E7 import read, exact_profile, NFINE, X0
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
exg = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7g_exact.json")))


def plateau(so, rho, u, P, tt):
    n = rho.size; x = (np.arange(n) + 0.5) / n; xi = (x - X0) / tt
    edge = lambda s: so[s]["S"] if so[s]["type"] == "shock" else so[s]["tail"]
    aL, bL, aR, bR = edge("L"), so["u"], so["u"], edge("R")
    m = ((xi > aL + 0.25 * (bL - aL)) & (xi < bL - 0.25 * (bL - aL))) | ((xi > aR + 0.25 * (bR - aR)) & (xi < bR - 0.25 * (bR - aR)))
    return np.median(P[m]) / so["P"] - 1


lines = []
for key, rec in sorted(exg.items()):
    tab, prob = key.split("|")
    t = E.load(rec["table"]); eos = RX.TableEOS(t, "p_over_rho"); cache = None
    for d in sorted(glob.glob(os.path.join(RUNS, "E7g", f"E7g_{tab}_{prob}_n*")), key=lambda q: int(q.rsplit("_n", 1)[1])):
        f = glob.glob(os.path.join(d, "*.out0.final.phdf"))
        if not f:
            lines.append(f"{tab:3s} {prob:16s} {os.path.basename(d)} (no final dump)"); continue
        tt, rho, u, P, nc = read(f[0])
        if cache is None:
            cache = exact_profile(eos, rec, tt)
        so, fine = cache
        n = rho.size; L1 = float(np.mean(np.abs(rho - fine[0].reshape(n, -1).mean(1))) / np.abs(fine[0]).max())
        lines.append(f"{tab:3s} {prob:16s} N {n:5d} | plateau dP* {plateau(so, rho, u, P, tt):+.3e} | L1 rho {L1:.3e}")
txt = "\n".join(lines); print(txt)
open(os.path.join(HERE, "..", "results", "tables", "E7g_tables.txt"), "w").write(txt + "\n")
