"""D6 analysis: eta1 / eta2 sensitivity on the D1a/D1b/D2b/D3 known answers + the conservation cost of switching.
Per run: the parent test's metric (D1a max|P/P0-1|; D1b max and L1 of dP = P/P_ad(rho) - 1 at the last dump; D3
plateau P/P2 - 1 and pre-shock max|P/P0-1|; D2b max|dP| and spurious heat / blob excess magnetic energy), the total-
energy drift tot-E(end)/tot-E(0) - 1 from the hst (mass and momentum are conserved by construction; the switch rewrites
E), and the time-averaged switched fraction <cons-ndual>/Ncells. -> results/tables/D6_threshold.{txt,json}"""
import glob
import json
import os

import numpy as np

from analyze_D1 import hst, load as load1
from analyze_D2 import cells, Pad as Pad2, e_ad as ead2, eos_e
from de_common import INP, RUNS, TAB, table_eos

eos = table_eos()
m1 = json.load(open(os.path.join(INP, "D1_meta.json")))
m3 = json.load(open(os.path.join(INP, "D3_meta.json")))["runs"]
m2 = json.load(open(os.path.join(INP, "D2_meta.json")))
NCELL = {"D1a_M100": 256, "D1b_M100_N4096": 4096, "D1b_M1000_N1024": 1024, "D1b_M100_N256_boost": 256,
         "D3_M10_N1024": 1024, "D3_M30_N1024": 1024, "D2b_b0.01_br1_M0_N128": 128 * 128}


def metric(case, d):
    f = glob.glob(os.path.join(d, "*.out0.final.phdf"))[0]
    t, c, dV = cells(f)
    if case.startswith("D1a"):
        return f"max|P/P0-1| {np.abs(c['P'] / m1['P0'] - 1).max():.2e}"
    if case.startswith("D1b") or case.startswith("D2b"):
        dP = c["P"] / Pad2(c["rho"]) - 1
        s = f"max|dP| {np.abs(dP).max():.2e} L1 {np.abs(dP).mean():.2e}"
        if case.startswith("D2b"):
            mm = m2["runs"][case]
            f0 = sorted(glob.glob(os.path.join(d, "*.out0.00000.phdf")))[0]
            _, c0, _ = cells(f0)
            EB = float((0.5 * (c0["Bx"] ** 2 + c0["By"] ** 2 + c0["Bz"] ** 2) - 0.5 * mm["B0z"] ** 2).sum() * dV)
            heat = float((c["rho"] * (eos_e(c["rho"], c["P"]) - ead2(c["rho"]))).sum() * dV)
            s += f" heat/E_B,exc {heat / EB:+.1e}"
        return s
    if case.startswith("D3"):
        m = m3[case]; P0 = float(eos.P(1.0, m["e0"]))
        o = np.argsort(c["x"]); x, P = c["x"][o], c["P"][o]
        r = np.abs(x - 0.5) / (m["S"] * t)
        pl = (r > 0.15) & (r < 0.8)
        return f"plateau P/P2-1 {np.mean(P[pl] / m['P2'] - 1):+.2e} (max {np.abs(P[pl] / m['P2'] - 1).max():.1e}) pre-shock {np.abs(P[r > 1.2] / P0 - 1).max():.1e}"


rows, res = [], {}
for line in open(os.path.join(INP, "D6_runs.txt")):
    d, k, o = [x.strip() for x in line.split("|")]
    name = os.path.basename(d)
    case = name.split("_eta1_")[0]
    rd = os.path.join(RUNS, d)
    if not glob.glob(os.path.join(rd, "*.out0.final.phdf")):
        rows.append(f"{name:40s} MISSING"); continue
    h = hst(rd)
    dE = float(h["tot-E"][-1] / h["tot-E"][0] - 1)
    frac = float(np.mean(h["cons-ndual"]) / NCELL[case])
    s = metric(case, rd)
    res[name] = dict(metric=s, totE=dE, swfrac=frac, nTfloor=float(h.get("cons-nTfloor", np.zeros(1)).sum()))
    rows.append(f"{name:40s} {s} | tot-E drift {dE:+.2e} | switched frac {frac:.3f} | nTfloor {res[name]['nTfloor']:.0f}")
txt = "\n".join(rows)
print(txt)
open(os.path.join(TAB, "D6_threshold.txt"), "w").write(
    "# D6 eta1 scan (eta2 0.1) + eta2 off at eta1 5e-2 (production = eta1 5e-2, eta2 0.1)\n" + txt + "\n")
json.dump(res, open(os.path.join(TAB, "D6_threshold.json"), "w"), indent=1)
