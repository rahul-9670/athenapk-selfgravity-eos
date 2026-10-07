#!/usr/bin/env python
"""E7g: where does the slow-converging error live? (P_num - P_exact)/P* and (u_num - u_exact)/|u*| against
xi = (x - x0)/t for R4 (and R2) at N = 1600..12800 and R1 at N = 1600/3200 -> results/figures/E7g_error_profiles.png"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
from analyze_E7 import read, exact_profile, NFINE, X0
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json"))); t = E.load(); eos = RX.TableEOS(t, "p_over_rho")
sets = {"R4_ionization": [("E7e/E7e_R4_ionization_p_over_rho_n1600", 1600), ("E7e/E7e_R4_ionization_p_over_rho_n3200", 3200),
                          ("E7f/R4_ionization_n6400", 6400), ("E7f/R4_ionization_n12800", 12800)],
        "R2_dissociation": [("E7e/E7e_R2_dissociation_p_over_rho_n1600", 1600), ("E7e/E7e_R2_dissociation_p_over_rho_n3200", 3200),
                            ("E7f/R2_dissociation_n6400", 6400), ("E7f/R2_dissociation_n12800", 12800)],
        "R1_molecular": [("E7e/E7e_R1_molecular_p_over_rho_n1600", 1600), ("E7e/E7e_R1_molecular_p_over_rho_n3200", 3200)]}
fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex="col")
for j, (prob, runs) in enumerate(sets.items()):
    rec = ex[prob + "|p_over_rho"]
    for d, n in runs:
        tt, rho, u, P, _ = read(glob.glob(os.path.join(RUNS, d, "*.out0.final.phdf"))[0])
        so, fine = exact_profile(eos, rec, tt)
        avg = fine.reshape(3, n, NFINE // n).mean(axis=2) if NFINE % n == 0 else None
        x = (np.arange(n) + 0.5) / n; xi = (x - X0) / tt
        if avg is None:   # N = 12800 > 25600/2 still divides; keep the guard
            continue
        ax[0, j].plot(xi, (P - avg[2]) / so["P"], lw=0.8, label=f"N={n}")
        ax[1, j].plot(xi, (u - avg[1]) / max(abs(so["u"]), 1e-30), lw=0.8)
    st = so["L"]
    for a in ax[:, j]:
        for v, c in ((st.get("head", np.nan), "g"), (st.get("tail", np.nan), "g"), (so["u"], "k"), (so["R"].get("S", np.nan), "r")):
            a.axvline(v, color=c, lw=0.5, ls=":")
    ax[0, j].set_ylim(-3e-3, 3e-3); ax[1, j].set_ylim(-3e-3, 3e-3)
    ax[0, j].set_title(prob + " (p_over_rho)"); ax[0, j].legend(fontsize=7)
    ax[1, j].set_xlabel("xi = (x - x0)/t  [code]")
ax[0, 0].set_ylabel("(P - P_exact)/P*"); ax[1, 0].set_ylabel("(u - u_exact)/|u*|")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "results", "figures", "E7g_error_profiles.png"), dpi=120)
print("ok")
