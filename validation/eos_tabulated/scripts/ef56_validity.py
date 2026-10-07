#!/usr/bin/env python
"""E-F5/E-F6 closure: physical validity of the out-of-table continuation (hydro/eos_extrapolate).
(1) below the rho edge (1e-20 g/cc): continuation P = rho (P/rho)(rho_edge, esp), T = T(rho_edge, esp), against the
    generator's own Saha solver evaluated directly at rho < 1e-20 (same physics, no table);
(2) below the esp edge: P/(rho e) at e_min = 2/3 (translational H2+He, rotation frozen) -> the continuation P ~ e is exact.
-> results/tables/EF56_validity.txt"""
import os, sys, importlib.util
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
spec = importlib.util.spec_from_file_location("gen", _REPO + "/src/eos/gen_eos_table.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
t = E.load(); eos = RX.TableEOS(t, "p_over_rho")
PU = E.RHO_U * E.V_U ** 2; EU = E.V_U ** 2
out = ["(1) below the rho edge: continuation vs direct Saha; cells = P_cont/P_Saha - 1 | T_cont/T - 1"]
Ts = (3, 10, 30, 100, 300, 1000, 1500, 2000, 3000, 5000, 1e4, 3e4)
out.append("rho [g/cc] " + "".join(f"{T:>22g}" for T in Ts))
for rc in (1e-21, 1e-22, 1e-24):
    row = f"{rc:9.0e} "
    for T in Ts:
        P, e = G.P_e_of_rho_T(rc, T)
        esp = e / rc / EU; rho = rc / E.RHO_U
        Pc = float(eos.P(rho, esp)) * PU               # p_over_rho bilinear clamps rho -> exactly rho (P/rho)_edge
        Tc = 10 ** float(eos.logT(rho, esp))
        row += f"  {Pc / P - 1:+9.1e} | {Tc / T - 1:+8.1e}"
    out.append(row)
r = t["P"][:, 0] / (10 ** (t["lr0"] + np.arange(t["nr"]) * t["dlr"]))
out.append(f"(2) P/(rho e) on the bottom esp row (e_min, T 0.063 K), all {t['nr']} rho nodes: min {r.min() / 10 ** t['le0']:.6f} max {r.max() / 10 ** t['le0']:.6f} (2/3 = 0.666667)")
txt = "\n".join(out); print(txt)
open(os.path.join(os.path.dirname(__file__), "..", "results", "tables", "EF56_validity.txt"), "w").write(txt + "\n")
