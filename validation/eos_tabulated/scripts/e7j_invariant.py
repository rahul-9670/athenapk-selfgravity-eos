#!/usr/bin/env python
"""E7j: Riemann invariant J = u + int_{rho_L}^{rho} c/rho drho along the left isentrope (exactly u_L through the left fan
and left star region). (J_num - u_L)/|u*| vs xi for PLM and LimO3, R4 and R2. -> results/figures/E7j_invariant.png"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
from analyze_E7 import read, X0
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json"))); t = E.load(); eos = RX.TableEOS(t, "p_over_rho")
fig, ax = plt.subplots(2, 2, figsize=(14, 7)); out = []
for j, prob in enumerate(("R4_ionization", "R2_dissociation")):
    rec = ex[prob + "|p_over_rho"]; so = RX.solve(eos, tuple(rec["L"]), tuple(rec["R"])); W = so["WL"]
    ise = W.isentrope(np.log(so["L"]["rho"] * 0.9))
    lnr = np.linspace(ise.lo, np.log(rec["L"][0]), 6000); I = np.array([ise(x)[1] for x in lnr])  # I = int c dlnrho (<=0)
    for k, sty in (("plm", "-"), ("limo3", "--")):
        for n in (1600, 3200, 6400):
            if prob.startswith("R4"):
                d = f"E7h/R4_limo3_n{n}" if k == "limo3" else (f"E7e/E7e_R4_ionization_p_over_rho_n{n}" if n <= 3200 else f"E7f/R4_ionization_n{n}")
            else:
                d = f"E7i/R2_limo3_n{n}" if k == "limo3" else (f"E7e/E7e_R2_dissociation_p_over_rho_n{n}" if n <= 3200 else f"E7f/R2_dissociation_n{n}")
            fs = glob.glob(os.path.join(RUNS, d, "*.out0.final.phdf"))
            if not fs: continue
            tt, rho, u, P, _ = read(fs[0]); x = (np.arange(n) + 0.5) / n; xi = (x - X0) / tt
            m = (xi > so["L"]["head"] - 3) & (xi < so["u"] - 1)
            Jn = u[m] - np.interp(np.log(rho[m]), lnr, I)      # left fan: u = u_L - I  ->  J = u + I = u_L ; here J = u + I
            Jn = u[m] + np.interp(np.log(rho[m]), lnr, I)
            dJ = (Jn - rec["L"][1]) / abs(so["u"])
            row = 0 if k == "plm" else 1
            ax[row, j].plot(xi[m], dJ, sty, lw=0.8, label=f"{k} N={n}")
            star = xi[m] > so["L"]["tail"] + 1; head = xi[m] < so["L"]["head"] + 0.15 * (so["L"]["tail"] - so["L"]["head"])
            out.append(f"{prob} {k:5s} N {n:5d}: dJ/|u*| left star median {np.median(dJ[star]):+.3e} | first 15% of fan median {np.median(dJ[head]):+.3e}")
    for row in (0, 1):
        for v in (so["L"]["head"], so["L"]["tail"]): ax[row, j].axvline(v, color="k", lw=0.5, ls=":")
        ax[row, j].set_ylim(-2e-3, 2e-3); ax[row, j].legend(fontsize=7); ax[row, j].set_title(f"{prob}: (J - u_L)/|u*|")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "results", "figures", "E7j_invariant.png"), dpi=120)
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "..", "results", "tables", "E7j_invariant.txt"), "w").write(txt + "\n")
