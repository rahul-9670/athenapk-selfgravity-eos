#!/usr/bin/env python
"""R-F13 fix study analysis (decks: make_decks_R13.py; table v12; every arm with sn_precond_stencil = sou, binary
athenaPK_cpu_g24_2ee77644 = 14f27ea1 with ATHENAPK_MAX_GROUP 24). Observables and metrics exactly as analyze_R5.py R5a:
envelope gas T (|x| in [3, 18] AU) at the last dump vs the reference, envelope heat Q = int (e - e_0), the 20 / 50 K
crossing radii, cost (s/cycle, Sn iterations/cycle, 1 rank). Reference per core temperature: F18 (0.125 dex over
[1e12, 1e14]); L18 (0.25 dex over [1e11, 1e15]) listed as well, with Q relative to F18, so the [1e12, 1e14] window
itself is tested (it was adequate for v11 at 300 K, R5: L18 = F10 to 0.1 %).
usage: analyze_R13.py run | check   -> results/tables/R13_groups_v12.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_R5 as A  # noqa: E402
from make_decks_R13 import FAMS, TCORE, XFAMS  # noqa: E402
from rad_common import MOD  # noqa: E402

# R13D=1: the re-run with diode faces (R-F15: the R13 decks' outflow faces silently skipped the Planck inflow)
R13D = os.environ.get("R13D", "0") == "1"
RUNS = os.path.join(MOD, "runs", "R13d" if R13D else "R13")
OUT = os.path.join(MOD, "results", "tables", "R13d_groups_v12_diode.txt" if R13D else "R13_groups_v12.txt")
ARMS = ["G1", "P3", "E4", "E5", "X5", "E6", "W6", "X6", "X6b", "X7", "E8", "X8", "F10", "L18", "F18", "M34",
        "F34"]   # reference = last complete (F34); X* = explicit edges (make_decks_R13.XFAMS, radiation/nu_edges_hz)


def main(mode):
    L = ["# R-F13 on table v12 (Semenov 2003 + Helling 2001): group partitions, R5 problem (scripts/analyze_R13.py)",
         "# envelope T_gas vs the reference F34 (0.0625 dex over [1e12, 1e14]); Q = envelope heat int(e - e0) / Q_ref; x20/x50 = where T falls to 20/50 K "
         "[AU]; wall = s/cycle (1 rank, 1-D); it = Sn iterations/cycle"]
    for Tc in TCORE:
        ref, rows = None, []
        for a in ARMS[::-1]:
            d = os.path.join(RUNS, f"{a}_T{int(Tc)}")
            if not A.complete(d):
                rows.append(f"  {a:4s}: FAILED -- {A.failure(d)}")
                continue
            D, D0 = A.dump(d, -1), A.dump(d, 0)
            env = (np.abs(D["X"]) > 3.0) & (np.abs(D["X"]) < 18.0)
            Q = float((D["e"] - D0["e"])[env].sum())
            if ref is None:
                ref, refname = (D, Q), a
            r = np.abs(D["T"][env] / ref[0]["T"][env] - 1.0)
            w, it, nf = A.cost(d)
            n = FAMS[a][0] if a in FAMS else (len(XFAMS[a]) + 1 if a in XFAMS else 1)
            rows.append(f"  {a:4s} ng {n:2d}: max |dT/T| {r.max():.3e}  median {np.median(r):.3e}  Q/Q_ref "
                        f"{Q / ref[1]:.4f}  x20 {A.crossing(D['X'], D['T'], 20.0):6.3f}  x50 "
                        f"{A.crossing(D['X'], D['T'], 50.0):6.3f}  T(3 AU) {np.interp(3.0, *A._side(D)):7.2f} K  wall "
                        f"{w:.3f}  it {it:.0f}" + (f"  NOT-CONVERGED {nf}" if nf else ""))
        L.append(f"T_core {Tc:.0f} K, end (t = {ref[0]['t']:.3e} code), reference {refname}:" if ref else
                 f"T_core {Tc:.0f} K: no complete arm")
        L += rows[::-1]
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
