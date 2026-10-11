#!/usr/bin/env python
"""R-F3 fix check: radiation/sn_fc_scatter = true (coupled Fleck & Cummings BiCGSTAB) on the R1 0-D problems.

Per run (runs/RF3/<pair>_<op>_k<k>, binary 66e045f6): code vs the Fleck & Cummings discrete step (r1_model.imc_step,
the implementation check) and vs the exact ODE; the legacy split from R1 (runs/R1/<pair>_<op>_a1.0_k<k>) alongside.
Errors at the dumps nearest t = 1, 2, 3, 4 x 1.1e-5: max relative over live groups, and group 0 (far-IR) separately.
usage: analyze_RF3.py run | check   -> results/tables/RF3_fc_scatter.txt
"""
import os
import sys

import numpy as np

import analyze_R1 as A
import r1_model as M
from make_decks_R1 import DT_PROD
from rad_common import MOD

RUNS = os.path.join(MOD, "runs")
OUT = os.path.join(MOD, "results", "tables", "RF3_fc_scatter.txt")
SUB = "RF3"
FAMS = ["g100r30_c2", "g100r30_tab", "g30r100_c2", "g30r100_tab", "g15r10_tab", "g100r30_c0"]


def evaluate(d, op, k, step):
    rows, P = A.load(d)
    t0, rho, T0, E0 = rows[0]
    Pm, rep, _ = A.problem(op, rho, P)
    e0 = rho * T0 / Pm.gm1
    dt = DT_PROD * 4.0 ** -k
    dumps = [A.at(rows, t) for t in A.TM]
    td = [r[0] for r in dumps]
    nd = [int(round(t / dt)) for t in td]
    ex = (M.exact_ode_tab if rep is None else M.exact_ode)(Pm, e0, E0, td)
    mult = (lambda Tk: (None, None)) if rep is None else rep
    o = M.march(step, Pm, e0, E0, dt, nd[-1], 1.0, mult, every=1)
    model = [o[n] for n in nd]
    live = [g for g in range(3) if ex[1 + g, -1] > 1e-10 * ex[1:, -1].sum()]
    cE = np.array([r[3] for r in dumps]).T
    impl = max(A.rel(cE[g], [m[2][g] for m in model]) for g in live)
    err = [max(A.rel(cE[g, m], ex[1 + g, m]) for g in live) for m in range(4)]
    g0 = [A.rel(cE[0, m], ex[1, m]) for m in range(4)]
    return impl, err, g0


def main(mode):
    L = ["# R-F3: coupled Fleck & Cummings (sn_fc_scatter) vs legacy same-group split, 0-D R1 problems (analyze_RF3.py)",
         "# impl = code vs its discrete model (FC: imc_step; legacy: code_step); errors vs exact ODE at t = 1..4 x 1.1e-5"]
    for fam in FAMS:
        pn, op = fam.rsplit("_", 1)
        L.append(f"== {fam}")
        for k in (0, 1, 2):
            if not os.path.isdir(os.path.join(RUNS, SUB, f"{fam}_k{k}")):
                continue
            fc = evaluate(os.path.join(RUNS, SUB, f"{fam}_k{k}"), op, k, M.imc_step)
            lg = evaluate(os.path.join(RUNS, "R1", f"{fam}_a1.0_k{k}"), op, k, M.code_step)
            L.append(f"  k={k} FC     impl {fc[0]:.1e} | live groups " + " ".join(f"{x:.2e}" for x in fc[1])
                     + " | group 0 " + " ".join(f"{x:.2e}" for x in fc[2]))
            L.append(f"  k={k} legacy impl {lg[0]:.1e} | live groups " + " ".join(f"{x:.2e}" for x in lg[1])
                     + " | group 0 " + " ".join(f"{x:.2e}" for x in lg[2]))
    txt = "\n".join(L) + "\n"
    if mode == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 2:   # analyze_RF3.py run|check <subdir> <out>  (e.g. RF3s: the frozen-group-skip re-check)
        SUB, OUT = sys.argv[2], sys.argv[3]
    main(sys.argv[1])
