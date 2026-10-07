#!/usr/bin/env python
"""E9: GPU (H100, 1 rank) vs CPU (1 rank) for the same decks and the same source. The t = 0 dump compares the
table LOOKUPS themselves (the IC energy is set through EintFromRhoTk on the device and P = Pof(rho, e) comes
from the first ConsToPrim); the final dump compares the whole evolution. Reports per run the number of
differing values and max |q_gpu/q_cpu - 1| for rho, v1, P, plus cycle counts.
-> results/tables/E9_gpu_cpu.txt"""
import glob, os
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "runs")
CPU_DIR = {"E10": "E10", "E7e": "E9_cpu_mb100", "E6": "E6cpu", "E8": "E8"}


def load(f):
    with h5py.File(f) as h:
        p = h["prim"][()]
        return (p.dtype, p[:, [0, 1, 4]].astype(float), int(np.atleast_1d(h["Info"].attrs["NCycle"])[0]),
                float(np.atleast_1d(h["Info"].attrs["Time"])[0]))


lines = []
for g in sorted(glob.glob(os.path.join(R, "E9_gpu", "E*"))):
    name = os.path.basename(g)
    c = os.path.join(R, CPU_DIR[name.split("_")[0]], name)
    row = [f"{name:34s}"]
    for tag in ("00000", "final"):
        fg = glob.glob(os.path.join(g, f"*.out0.{tag}.phdf")); fc = glob.glob(os.path.join(c, f"*.out0.{tag}.phdf"))
        if not fg or not fc:
            row.append(f"{tag}: missing ({'gpu' if not fg else 'cpu'})"); continue
        dt, a, na, ta = load(fg[0]); _, b, nb, tb = load(fc[0])
        if a.shape != b.shape:
            row.append(f"{tag}: DECK MISMATCH shapes {a.shape} vs {b.shape} (not compared)"); continue
        nd = int(np.sum(a != b))
        rel = [float(np.max(np.abs(a[:, k] - b[:, k]) / np.maximum(np.abs(b[:, k]), 1e-300 + np.abs(b[:, k]).max() * 1e-12))) for k in range(3)]
        row.append(f"{tag}: cycles {na}/{nb} t {'=' if ta == tb else '!='} | differing {nd}/{a.size} ({dt}) | max rel rho {rel[0]:.1e} v {rel[1]:.1e} P {rel[2]:.1e}")
    lines.append("  ".join(row))
txt = "\n".join(lines)
print(txt)
open(os.path.join(HERE, "..", "results", "tables", "E9_gpu_cpu.txt"), "w").write(txt + "\n")
