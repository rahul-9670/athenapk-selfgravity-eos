#!/usr/bin/env python
"""E7g: one-sided derivative jumps of the bilinear P interpolant (p_over_rho) at the grid lines each rarefaction
crosses: |jump of dlnP/dln e|_rho| at e lines and |jump of dlnP/dln rho|_e| at rho lines, summed over the fan.
-> results/tables/E7g_kinks2.txt"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
HERE = os.path.dirname(os.path.abspath(__file__))
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json"))); t = E.load()
out = []
for prob in ("R1_molecular", "R2_dissociation", "R4_ionization"):
    r = ex[f"{prob}|p_over_rho"]; eos = RX.TableEOS(t, "p_over_rho")
    W = RX.WaveCurve(eos, *r["L"], -1); ise = W.isentrope(np.log(r["Lstar"]["rho"]))
    lnr = np.linspace(np.log(r["L"][0]), np.log(r["Lstar"]["rho"]), 20001)
    le = np.log10([ise(x)[0] for x in lnr]); lr = lnr / np.log(10)
    h = 1e-7
    lP = lambda a, b: np.log(eos.P(10 ** a, 10 ** b))
    je, jr = [], []
    ie = np.floor((le - t["le0"]) / t["dle"]); ir = np.floor((lr - t["lr0"]) / t["dlr"])
    for k in np.nonzero(np.diff(ie) != 0)[0]:
        node = t["le0"] + max(ie[k], ie[k + 1]) * t["dle"]; a = lr[k]
        sl = (lP(a, node) - lP(a, node - h)) / h; sr = (lP(a, node + h) - lP(a, node)) / h
        je.append(abs(sr - sl) / np.log(10))
    for k in np.nonzero(np.diff(ir) != 0)[0]:
        node = t["lr0"] + max(ir[k], ir[k + 1]) * t["dlr"]; b = le[k]
        sl = (lP(node, b) - lP(node - h, b)) / h; sr = (lP(node + h, b) - lP(node, b)) / h
        jr.append(abs(sr - sl) / np.log(10))
    out.append(f"{prob:16s} e lines {len(je):3d}: sum |jump dlnP/dlne| {np.sum(je):.3f} (max {np.max(je):.3f}) | "
               f"rho lines {len(jr):3d}: sum |jump dlnP/dlnrho| {np.sum(jr):.3f} (max {np.max(jr):.3f})")
print("\n".join(out)); open(os.path.join(HERE, "..", "results", "tables", "E7g_kinks2.txt"), "w").write("\n".join(out) + "\n")
