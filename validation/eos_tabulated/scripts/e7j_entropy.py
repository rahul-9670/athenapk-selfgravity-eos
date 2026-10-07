#!/usr/bin/env python
"""E7j: where is the R4/R2 plateau error made? Departure from the LEFT isentrope, de = e_num/e_ise(rho_num) - 1, across
the left fan + left star region (exactly isentropic in the exact solution), PLM N = 800..6400 (and LimO3 for contrast),
plus the departure of P at fixed rho. -> results/figures/E7j_entropy.png, results/tables/E7j_entropy.txt"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
from analyze_E7 import read, X0
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json"))); t = E.load(); eos = RX.TableEOS(t, "p_over_rho")
out = []; fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for j, prob in enumerate(("R4_ionization", "R2_dissociation")):
    rec = ex[prob + "|p_over_rho"]; so = RX.solve(eos, tuple(rec["L"]), tuple(rec["R"]))
    W = so["WL"]; ise = W.isentrope(np.log(so["L"]["rho"] * 0.9))
    lnr_grid = np.linspace(ise.lo, np.log(rec["L"][0]), 4000); e_grid = np.array([ise(x)[0] for x in lnr_grid])
    runs = [(f"E7h/R4_{k}_n{n}" if prob.startswith("R4") else f"E7i/R2_limo3_n{n}" if k == "limo3" else None, k, n)
            for k in ("plmref", "limo3") for n in (800, 1600, 3200, 6400)]
    for d, k, n in runs:
        if prob.startswith("R4"):
            d = f"E7h/R4_limo3_n{n}" if k == "limo3" else (f"E7e/E7e_R4_ionization_p_over_rho_n{n}" if n <= 3200 else f"E7f/R4_ionization_n{n}")
        else:
            d = f"E7i/R2_limo3_n{n}" if k == "limo3" else (f"E7e/E7e_R2_dissociation_p_over_rho_n{n}" if n <= 3200 else f"E7f/R2_dissociation_n{n}")
        fs = glob.glob(os.path.join(RUNS, d, "*.out0.final.phdf"))
        if not fs: continue
        tt, rho, u, P, _ = read(fs[0])
        x = (np.arange(n) + 0.5) / n; xi = (x - X0) / tt
        m = (xi > so["L"]["head"] + 2) & (xi < so["u"] - 2)          # left fan + left star region, off the edges
        e = np.array([eos.e_of_P(r, p) for r, p in zip(rho[m], P[m])])
        eis = np.interp(np.log(rho[m]), lnr_grid, e_grid)
        de = e / eis - 1
        star = xi[m] > so["L"]["tail"] + 2
        out.append(f"{prob} {k:6s} N {n:5d}: de in left star region median {np.median(de[star]):+.3e} (range {de[star].min():+.2e}..{de[star].max():+.2e}); in fan median {np.median(de[~star]):+.3e}")
        ax[j].plot(xi[m], de, lw=0.8, ls="-" if k == "plmref" else "--", label=f"{k} N={n}")
    ax[j].axvline(so["L"]["tail"], color="k", lw=0.5, ls=":"); ax[j].set_title(prob + ": e/e_isentrope(rho) - 1 (left fan + star)")
    ax[j].set_xlabel("xi"); ax[j].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "results", "figures", "E7j_entropy.png"), dpi=120)
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "..", "results", "tables", "E7j_entropy.txt"), "w").write(txt + "\n")
