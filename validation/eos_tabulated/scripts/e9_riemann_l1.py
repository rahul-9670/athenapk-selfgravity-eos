#!/usr/bin/env python
"""E9 for the Riemann legs: GPU-CPU difference in L1 (normalised by max|q|), next to each run's L1 error vs the exact
solution (analyze_E7.exact_profile). Pointwise maxima are meaningless at discontinuities. -> results/tables/E9_riemann_L1.txt"""
import glob, json, os, sys
import h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
from analyze_E7 import exact_profile, NFINE
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "runs")
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json"))); t = E.load()


def load(f):
    with h5py.File(f) as h:
        p = h["prim"][()].astype(float); o = np.argsort(h["Locations/x"][:, 0])
        tt = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
    return tt, np.concatenate([p[b, [0, 1, 4], 0, 0, :] for b in o], axis=1)


lines = []
for g in sorted(glob.glob(os.path.join(R, "E9_gpu", "E7e_*_mb100"))):
    name = os.path.basename(g); prob, mode = name[4:].split("_p")[0], "p_over_rho" if "_p_over_rho_" in name else "p"
    tg, a = load(glob.glob(os.path.join(g, "*.out0.final.phdf"))[0])
    tc, b = load(glob.glob(os.path.join(R, "E9_cpu_mb100", name, "*.out0.final.phdf"))[0])
    so, fine = exact_profile(RX.TableEOS(t, mode), ex[f"{prob}|{mode}"], tc)
    n = b.shape[1]; avg = fine.reshape(3, n, NFINE // n).mean(axis=2); s = np.abs(fine).max(axis=1)
    d = [np.mean(np.abs(a[k] - b[k])) / s[k] for k in range(3)]
    eg = [np.mean(np.abs(a[k] - avg[k])) / s[k] for k in range(3)]; ec = [np.mean(np.abs(b[k] - avg[k])) / s[k] for k in range(3)]
    lines.append(f"{name:44s} L1(GPU-CPU) rho {d[0]:.1e} u {d[1]:.1e} P {d[2]:.1e} | L1 err vs exact GPU rho {eg[0]:.3e} CPU {ec[0]:.3e}"
                 f" | P GPU {eg[2]:.3e} CPU {ec[2]:.3e} | t_gpu - t_cpu {tg - tc:+.1e}")
txt = "\n".join(lines); print(txt)
open(os.path.join(HERE, "..", "results", "tables", "E9_riemann_L1.txt"), "w").write(txt + "\n")
