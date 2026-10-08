"""D1y: hydro/dual_energy_flux = density (DE-F3 candidate fix) vs the legacy specific-energy flux (dual ON) and dual OFF.
Contacts (D1a/D1x decks): max|P/P0 - 1| at the last dump, L1 density error. Smooth flows (D1b decks): adiabat error at
t/tc ~ 0.4 and 0.8 (max, L1). Legacy numbers are read from D1a_contact.json / D1x_contact.json / D1b_adiabat.json.
-> results/tables/D1y_density_flux.{txt,json}"""
import glob, json, os
import numpy as np
from analyze_D1 import load, adiabat_table, meta as m1, eos
from de_common import INP, RUNS, TAB

L, Ee = adiabat_table(1.0, m1["e0"], 0.2, 20.0)
Pad = lambda rho: eos.P(rho, np.interp(np.log(rho), L, Ee))
ja = json.load(open(os.path.join(TAB, "D1a_contact.json")))
jx = json.load(open(os.path.join(TAB, "D1x_contact.json")))
jb = json.load(open(os.path.join(TAB, "D1b_adiabat.json")))
mx = json.load(open(os.path.join(INP, "D1x_meta.json")))
rows, res = [], {}
for line in open(os.path.join(INP, "D1y_runs.txt")):
    d, k, o = [x.strip() for x in line.split("|")]
    name = os.path.basename(d)
    rd = os.path.join(RUNS, d)
    fs = sorted(glob.glob(os.path.join(rd, "*.out0.0*.phdf"))) + glob.glob(os.path.join(rd, "*.out0.final.phdf"))
    if not glob.glob(os.path.join(rd, "*.out0.final.phdf")):
        rows.append(f"{name:28s} MISSING"); continue
    if k.startswith("D1b"):
        key = name.replace("_den", "")
        mm = m1["runs"][key]
        out = []
        for f in fs[1:]:
            t, X, rho, v, P = load(f)
            dP = P / Pad(rho) - 1
            out.append((t / mm["tc"], float(np.abs(dP).max()), float(np.abs(dP).mean())))
        def pick(seq, tt):
            return min(seq, key=lambda s: abs(s[0] - tt))
        a4, a8 = pick(out, 0.4), pick(out, 0.8)
        lo, lf = jb[key + "_on"], jb[key + "_off"]
        l4 = min(lo["dumps"], key=lambda p: abs(p["t_tc"] - 0.4)); l8 = lo["dumps"][-1]
        f4 = min(lf["dumps"], key=lambda p: abs(p["t_tc"] - 0.4)); f8 = lf["dumps"][-1]
        res[name] = dict(kind="smooth", den=[a4, a8], legacy=[[l4["maxdP"], l4["L1dP"]], [l8["maxdP"], l8["L1dP"]]],
                         off=[[f4["maxdP"], f4["L1dP"]], [f8["maxdP"], f8["L1dP"]]])
        rows.append(f"{key:28s} smooth | t/tc 0.4 max/L1: density {a4[1]:.2e}/{a4[2]:.2e}  legacy {l4['maxdP']:.2e}/{l4['L1dP']:.2e}"
                    f"  off {f4['maxdP']:.2e}/{f4['L1dP']:.2e} | t/tc 0.8: density {a8[1]:.2e}/{a8[2]:.2e}  legacy {l8['maxdP']:.2e}/{l8['L1dP']:.2e}"
                    f"  off {f8['maxdP']:.2e}/{f8['L1dP']:.2e}")
    else:
        if name.startswith("D1y_M") and "_den" not in name:      # from D1x
            key = name.replace("D1y_", "D1x_").replace("_on", "")
            P0 = mx["runs"][key].get("P0", mx["P0"])
            leg, off = jx[key + "_on"]["per"][-1], jx[key + "_off"]["per"][-1]
        else:                                                      # from D1a
            key = name.replace("D1y_", "D1a_").replace("_den", "")
            key = name.replace("_den", "")
            P0 = m1["P0"]
            leg = dict(maxdP=ja[key + "_on"]["maxdP"], L1rho=ja[key + "_on"]["L1rho"])
            off = dict(maxdP=ja[key + "_off"]["maxdP"], L1rho=ja[key + "_off"]["L1rho"])
        _, X, r0, _, _ = load(fs[0])
        seq = []
        for f in fs[1:]:
            t, X1, r1, v1, P1 = load(f)
            seq.append(float(np.abs(P1 / P0 - 1).max()))
        L1 = float(np.abs(r1 - r0).mean())
        res[name] = dict(kind="contact", den=seq, L1rho=L1, legacy=leg, off=off)
        rows.append(f"{key:28s} contact | max|P/P0-1| density {' '.join(f'{s:.2e}' for s in seq)}  legacy {leg['maxdP']:.2e}"
                    f"  off {off['maxdP']:.2e} | L1(rho) density {L1:.3e} legacy {leg['L1rho']:.3e} off {off['L1rho']:.3e}")
txt = "\n".join(rows)
print(txt)
open(os.path.join(TAB, "D1y_density_flux.txt"), "w").write(
    "# D1y: hydro/dual_energy_flux = density (candidate fix for DE-F3) vs legacy (specific) flux with dual ON, and dual OFF\n" + txt + "\n")
json.dump(res, open(os.path.join(TAB, "D1y_density_flux.json"), "w"), indent=1)
