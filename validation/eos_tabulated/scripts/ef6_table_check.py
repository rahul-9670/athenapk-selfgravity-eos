#!/usr/bin/env python
"""E-F6 closure: the production table extended down to 1e-26 g/cc (gen_eos_table.py rho_pad_pts = 120).
(1) axes and original rows vs the production table; (2) new rows vs direct Saha at off-node (rho, T), compared with
what the runtime continuation gave there. -> results/tables/EF6_table_check.txt"""
import os, sys, importlib.util
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
spec = importlib.util.spec_from_file_location("gen", _REPO + "/src/eos/gen_eos_table.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
old = E.load(); new = E.load(os.path.join(os.path.dirname(__file__), "..", "runs", "EF6_table", "v5cold_rho26.bin"))
out = []
for k in ("nr", "ne", "nT"): out.append(f"{k}: production {old[k]}  extended {new[k]}")
for k in ("lr0", "dlr", "le0", "dle", "lT0", "dlT"): out.append(f"{k}: production {old[k]:.15g}  extended {new[k]:.15g}  diff {new[k] - old[k]:+.3e}")
pad = new["nr"] - old["nr"]
for k in ("P", "cs2", "logT", "espT"):
    a, b = old[k], new[k][pad:]
    d = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
    rows = np.nonzero((a != b).any(axis=1))[0]
    out.append(f"original rows, {k}: {int((a != b).sum())} of {a.size} values differ; rows with differences {rows.tolist()[:6]}; max rel {d.max():.2e}")
eosN = RX.TableEOS(new, "p_over_rho"); eosO = RX.TableEOS(old, "p_over_rho")
PU = E.RHO_U * E.V_U ** 2; EU = E.V_U ** 2
out.append("new rows vs direct Saha (off-node rho): P_table/P_Saha - 1 (extended table)  [continuation with the production table]")
Ts = (3, 10, 100, 1000, 1200, 2000, 3000, 4000, 5000, 1e4, 3e4)
out.append("rho [g/cc] " + "".join(f"{T:>24g}" for T in Ts))
worst = 0.0
for rc in (2.3e-21, 1.7e-22, 3.1e-24, 1.3e-26):
    row = f"{rc:9.1e} "
    for T in Ts:
        P, e = G.P_e_of_rho_T(rc, T); rho = rc / E.RHO_U; esp = e / rc / EU
        pn = float(eosN.P(rho, esp)) * PU / P - 1; po = float(eosO.P(rho, esp)) * PU / P - 1
        worst = max(worst, abs(pn)); row += f"  {pn:+9.1e} [{po:+8.1e}]"
    out.append(row)
out.append(f"max |P_table/P_Saha - 1| over the new rows sampled: {worst:.2e}")
txt = "\n".join(out); print(txt)
open(os.path.join(os.path.dirname(__file__), "..", "results", "tables", "EF6_table_check.txt"), "w").write(txt + "\n")
