#!/usr/bin/env python
"""R6: GPU (H100, binary athenaPK_gpu_65e6a2f9) vs CPU (athenaPK_cpu_e90ae249), same source, same deck, 1 rank
(scripts/make_lists_R6.py). Per gate: completion, the first two dt (E-F8b: a broken CFL reduction showed up as a wrong
first dt), number of cycles, Sn iterations per solve (mean), and the final-state difference of the radiation energy
(all groups summed): max |dE| / max E, and max / median |gpu/cpu - 1| over cells above 1e-12 of the max; and the gas
temperature p/rho (max |gpu/cpu - 1|).
usage: analyze_R6.py run | check   -> results/tables/R6_gpu_cpu.txt
"""
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rad_common import MOD  # noqa: E402

RUNS = os.path.join(MOD, "runs", "R6")
OUT = os.path.join(MOD, "results", "tables", "R6_gpu_cpu.txt")
GATES = ("G9", "K4", "R1", "S1", "S3", "R5", "R8a", "R8c")


def info(d):
    s = open(os.path.join(d, "run.log"), errors="replace").read()
    dts = re.findall(r"^cycle=(\d+) time=\S+ dt=(\S+)", s, re.M)
    it = [int(v) for v in re.findall(r"converged in (\d+) iters", s)]
    rc = re.findall(r"^rc (\d+) wall (\S+)", open(os.path.join(d, "cmd.txt")).read(), re.M)
    return dict(dt=[float(v) for _, v in dts[:2]], ncyc=int(dts[-1][0]) if dts else -1, it=np.mean(it) if it else 0.0,
                rc=int(rc[0][0]) if rc else -1, wall=float(rc[0][1]) if rc else float("nan"),
                nc=len(re.findall(r"NOT CONVERGED", s)), err="PARTHENON ERROR" in s)


def final(d):
    with h5py.File(os.path.join(d, "parthenon.out0.final.phdf"), "r") as h:
        E = sum(np.array(h[k]) for k in h.keys() if k == "rad.Er" or k.startswith("rad.Er_g"))
        p = np.array(h["prim"])
        return E, p[:, 4] / p[:, 0]


def main(mode):
    L = ["# R6 GPU (H100) vs CPU, same source / deck / 1 rank (scripts/analyze_R6.py)"]
    for g in GATES:
        c, q = os.path.join(RUNS, "cpu", g), os.path.join(RUNS, "gpu", g)
        a, b = info(c), info(q)
        line = (f"{g:4s}: cpu rc {a['rc']} cycles {a['ncyc']} dt0 {a['dt'][0]:.6e} dt1 {a['dt'][1] if len(a['dt']) > 1 else float('nan'):.6e}"
                f" it {a['it']:.1f} wall {a['wall']:.1f} s | gpu rc {b['rc']} cycles {b['ncyc']} dt0 {b['dt'][0]:.6e} dt1 "
                f"{b['dt'][1] if len(b['dt']) > 1 else float('nan'):.6e} it {b['it']:.1f} wall {b['wall']:.1f} s")
        if a["rc"] == 0 and b["rc"] == 0:
            Ea, Ta = final(c)
            Eb, Tb = final(q)
            m = Ea > 1e-12 * Ea.max()
            line += (f" | final E_rad max |dE|/max E {np.max(np.abs(Eb - Ea)) / Ea.max():.2e}; max rel {np.max(np.abs(Eb[m] / Ea[m] - 1)):.2e} median "
                     f"{np.median(np.abs(Eb[m] / Ea[m] - 1)):.2e}; T max rel {np.max(np.abs(Tb / Ta - 1)):.2e}")
        else:
            line += f" | NOT COMPARED (gpu NOT CONVERGED x{b['nc']}, error {b['err']}; cpu NOT CONVERGED x{a['nc']})"
        L.append(line)
    txt = "\n".join(L) + "\n"
    if mode == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
