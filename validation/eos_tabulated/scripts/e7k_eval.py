#!/usr/bin/env python
"""E7k: pure-contact test. Exact solution: uniform P = P* and u = u*. Reports the spurious pressure/velocity
(L1 and max over the domain, relative to P* and to max(|u*|, c)) for PLM and LimO3 vs N. -> results/tables/E7k_contact.txt"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_E7 import read
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs", "E7k")
meta = json.load(open(os.path.join(HERE, "..", "inputs", "E7k_contact.json"))); out = []
for key, m in meta.items():
    prob, utag = key.split("|")
    for rec in ("plm", "limo3"):
        row = []
        for n in (400, 800, 1600, 3200, 6400):
            fs = glob.glob(os.path.join(RUNS, f"E7k_{prob}_{utag}_n{n}_{rec}", "*.out0.final.phdf"))
            if not fs:
                row.append("     --     "); continue
            tt, rho, u, P, _ = read(fs[0])
            dp = np.abs(P / m["P"] - 1); du = np.abs(u - m["u"]) / max(abs(m["u"]), 1.0)
            row.append(f"{dp.mean():.2e}/{dp.max():.1e}")
        out.append(f"{prob:16s} {utag:4s} {rec:5s} L1/max |P/P*-1| N=400..6400: " + "  ".join(row))
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "..", "results", "tables", "E7k_contact.txt"), "w").write(txt + "\n")
