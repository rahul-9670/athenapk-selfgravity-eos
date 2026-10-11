#!/usr/bin/env python
"""R2: convergence of the semi-analytic Su-Olson reference itself in box length Lp and mode count nk (S64, cell
averages h = 0.05). usage: r2_refcheck.py run|check -> results/tables/R2_ref_convergence.txt"""
import os, sys
import numpy as np
import su_olson_ref as SO
from rad_common import MOD
OUT = os.path.join(MOD, "results", "tables", "R2_ref_convergence.txt")
z = np.arange(-6.0 + 0.025, 6.0, 0.05)
mu, w = SO.gauss_slab(64)
base = SO.solve([1.01, 10.0], z, 0.05, mu, w, Lp=80.0, nk=4096)
L = ["# R2 reference convergence (scripts/r2_refcheck.py): max |U - U_base| / max|U_base| vs Lp=80, nk=4096, S64"]
for Lp, nk in ((80.0, 8192), (160.0, 8192), (160.0, 16384)):
    U, V = SO.solve([1.01, 10.0], z, 0.05, mu, w, Lp=Lp, nk=nk)
    L.append(f"Lp {Lp:5.0f} nk {nk:6d}: tau 1.01 U {np.max(np.abs(U[0]-base[0][0]))/np.max(base[0][0]):.2e} V {np.max(np.abs(V[0]-base[1][0]))/np.max(base[1][0]):.2e}"
             f" | tau 10 U {np.max(np.abs(U[1]-base[0][1]))/np.max(base[0][1]):.2e} V {np.max(np.abs(V[1]-base[1][1]))/np.max(base[1][1]):.2e}")
for N in (32, 96):
    mu2, w2 = SO.gauss_slab(N)
    U, V = SO.solve([10.0], z, 0.05, mu2, w2)
    L.append(f"S{N} vs S64 at tau 10: U {np.max(np.abs(U[0]-base[0][1]))/np.max(base[0][1]):.2e}")
txt = "\n".join(L) + "\n"
if sys.argv[1] == "run":
    open(OUT, "w").write(txt); print(txt)
else:
    print("check:", "BYTE-IDENTICAL" if open(OUT).read() == txt else "DIFFERS")
