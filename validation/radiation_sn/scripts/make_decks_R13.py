#!/usr/bin/env python
"""R-F13 fix study (2026-10-09): group partitions on the R-F11 opacity table v12 (Semenov 2003 + Helling 2001, format
v4, gen_opacity_table.py --model semenov). R5's problem and deck builder unchanged (scripts/make_decks_R5.py: hot top-hat
core 1e-11 g/cm^3 in a 1e-13 g/cm^3, 10 K envelope, 1-D 640 cells over +-20 AU, adopted solver keys, 60 forced steps),
with three changes:
  * tables: one v12-recipe table per partition (inputs/R13/tables.txt -> scripts/gen_tables_R13.sh);
  * core temperatures 150 K (the collapse run's own first-core range: T ~ 60-390 K up to rho_max 1e-11 in a 3-D collapse restart, ~150 K
    by its stop at 1e-10, and Semenov's water-ice front at 130-160 K), 300 K and 1000 K;
  * partitions: the R5 set (gray, P3, E4, E8, F10, F18, L18) plus E5 / E6 (4 / 5 log intervals over [1e12, 1e14])
    and W6 (0.75 dex over [10^11.5, 10^14.5]) to find the cheapest partition near the reference.
References: F18 (0.125 dex over [1e12, 1e14]) at 150 and 300 K; L18 (0.25 dex over [1e11, 1e15]) as well, because a
1000 K core puts photons above 1e14 Hz. Every arm runs the R-F12 preconditioner (sn_precond_stencil = sou) and the test
build with ATHENAPK_MAX_GROUP = 24 (all arms, so the binary is not a variable).
writes inputs/R13/*.in, inputs/R13/tables.txt, inputs/R13_runs.txt
"""
import os

import make_decks_R5 as R5
from rad_common import INP, RUNS, adopted_overlay

R5.TAB = os.path.join(RUNS, "R13", "tables")
FAMS = {"P3": (3, 1e12, 1e15), "E4": (4, 1e12, 1e14), "E5": (5, 1e12, 1e14), "E6": (6, 1e12, 1e14),
        "W6": (6, 10 ** 11.5, 10 ** 14.5), "E8": (8, 10 ** 11.5, 10 ** 14.5), "F10": (10, 1e12, 1e14),
        "F18": (18, 1e12, 1e14), "L18": (18, 1e11, 1e15),
        # R13b (after the first pass: 0.5 -> 0.25 -> 0.125 dex was not converging): 0.0625 dex over [1e12, 1e14] and
        # 0.125 dex over [1e11, 1e15], 34 groups each, ATHENAPK_MAX_GROUP = 40 build
        "F34": (34, 1e12, 1e14), "M34": (34, 1e11, 1e15)}
R5.FAMS = FAMS
# R13x (explicit interior edges, radiation/nu_edges_hz, binary with BuildRadGroupsFromEdges): resolution where a
# 150-400 K core's photons are absorbed by a 10-30 K envelope (5e12 - 6e13 Hz, silicate 10 / 18 um features); one group
# for the envelope's own emission below ~2e12 Hz; the top group (> 3-4e13 Hz) carries < 1e-3 of a 300 K Planck function.
XFAMS = {"X5": (2e12, 6e12, 1.5e13, 3e13), "X6": (2e12, 5e12, 1e13, 2e13, 4e13), "X6b": (3e12, 7e12, 1.2e13, 2e13, 3.5e13),
         "X7": (2e12, 4e12, 7e12, 1.2e13, 2e13, 3.5e13), "X8": (1.5e12, 3e12, 5e12, 8e12, 1.2e13, 1.8e13, 3e13)}


def xdeck(Tc, name):
    e = XFAMS[name]
    n = len(e) + 1
    R5.FAMS = dict(FAMS, **{name: (n, 1e12, 1e14)})   # n and the table name; nu_min/nu_max are overridden by nu_edges_hz
    txt = R5.deck(Tc, name, 1)
    R5.FAMS = FAMS
    txt = txt.replace(f"n_group = {n}\n", f"n_group = {n}\nnu_edges_hz = {' '.join(repr(x) for x in e)}\n", 1)
    return txt
TCORE = (150.0, 300.0, 1000.0)
PC = "radiation/sn_precond_stencil=sou"


def main():
    d = os.path.join(INP, "R13")
    os.makedirs(d, exist_ok=True)
    os.makedirs(R5.TAB, exist_ok=True)
    open(os.path.join(d, "tables.txt"), "w").write("".join(f"{f} | {R5.edges(*FAMS[f])}\n" for f in FAMS))
    runs, runs_b = [], []
    for Tc in TCORE:
        for fam in list(FAMS) + ["G1"]:
            name = f"{fam}_T{int(Tc)}"
            open(os.path.join(d, name + ".in"), "w").write(adopted_overlay(R5.deck(Tc, fam, 1)))
            line = f"R13/{name} | R13/{name}.in | {PC} | 1"
            (runs_b if fam in ("F34", "M34") else runs).append(line)
    runs_x = []
    for Tc in TCORE:
        for name in XFAMS:
            nm = f"{name}_T{int(Tc)}"
            open(os.path.join(d, nm + ".in"), "w").write(adopted_overlay(xdeck(Tc, name)))
            runs_x.append(f"R13/{nm} | R13/{nm}.in | {PC} | 1")
    open(os.path.join(INP, "R13x_runs.txt"), "w").write("\n".join(runs_x) + "\n")
    open(os.path.join(INP, "R13_runs.txt"), "w").write("\n".join(runs) + "\n")
    open(os.path.join(INP, "R13b_runs.txt"), "w").write("\n".join(runs_b) + "\n")
    print(len(runs), "R13 runs,", len(runs_b), "R13b runs")


if __name__ == "__main__":
    main()
