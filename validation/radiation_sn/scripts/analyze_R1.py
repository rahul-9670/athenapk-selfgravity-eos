#!/usr/bin/env python
"""R1: 0-D multigroup relaxation -- code vs (a) a replica of its own discrete step (implementation), (b) the exact
multigroup ODE (method error), (c) the Fleck & Cummings multifrequency step with cross-group re-emission (the P-07
alternative), all at the code's dt.

usage: analyze_R1.py run | check      (check = re-derive results/tables/R1_relax.txt byte for byte)
Reads runs/R1/<pair>_<op>_a<alpha>_k<k>/ (make_decks_R1.py). Quantities: box means of T = p/rho and E_g (rad.Er,
rad.Er_g1, rad.Er_g2) at t_m = m DT_PROD, m = 1..4. Errors are relative; E_tot = e + sum_g E_g (c/chat = 1).
"""
import glob
import os
import sys

import h5py
import numpy as np

import r1_model as M
from make_decks_R1 import PAIRS, OPS, KS, DT_PROD, TABLE, RHO_UNIT, OPAC_UNIT
from rad_common import MOD

RUNS = os.path.join(MOD, "runs", "R1")
OUT = os.path.join(MOD, "results", "tables", "R1_relax.txt")
EDGES = [0.0, 1e12, 1e15, 1e30]
TM = [m * DT_PROD for m in (1, 2, 3, 4)]


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))
    rows = []
    P = None
    for f in fs:
        with h5py.File(f, "r") as h:
            p = np.array(h["prim"])
            P = dict(h["Params"].attrs)
            E = np.array([np.array(h[n]).mean() for n in ("rad.Er", "rad.Er_g1", "rad.Er_g2")])
            rows.append((float(h["Info"].attrs["Time"]), p[:, 0].mean(), (p[:, 4] / p[:, 0]).mean(), E))
    return rows, P


def at(rows, t):
    """The first dump at or after t (Parthenon writes on the first step that crosses an output time; at fine dt the
    summed dt lands one step past t). Every comparison uses THIS dump's own time and step count."""
    later = [r for r in rows if r[0] >= t * (1 - 1e-9)]
    return min(later, key=lambda x: x[0]) if later else None


def problem(op, rho, P):
    a, c, chat, Tu, gam = P["radiation/arad"], P["radiation/c"], P["radiation/chat"], P["radiation/T_unit"], \
        P["radiation/gamma"]
    if OPS[op][0] == "tabulated":
        tab = M.TabOpacity(TABLE, OPAC_UNIT, RHO_UNIT, EDGES)
        return M.TabProblem(tab, rho, gam, a, c, chat, Tu, EDGES), None, None
    sh = M.Shape(OPS[op][1])
    Pm = M.Problem(rho, P["radiation/kappa_a"], gam, a, c, chat, Tu, EDGES, sh)
    return Pm, M.Replica(sh, EDGES), sh


def rel(x, y):
    return float(np.max(np.abs(np.asarray(x) - np.asarray(y)) / np.maximum(np.abs(np.asarray(y)), 1e-300)))


def family(pn, op, alpha):
    lines = []
    ref = None
    for k in KS:
        d = os.path.join(RUNS, f"{pn}_{op}_a{alpha}_k{k}")
        rows, P = load(d)
        rc = open(os.path.join(d, "cmd.txt")).read().split("rc ")[-1].split()[0] if os.path.exists(
            os.path.join(d, "cmd.txt")) else "?"
        if not rows or at(rows, TM[-1]) is None:
            lines.append(f"  k={k} dt={DT_PROD * 4.0 ** -k:.3e}  RUN DID NOT REACH t = 4 DT_PROD (rc {rc}, "
                         f"{len(rows)} dumps)")
            continue
        t0, rho, T0, E0 = rows[0]
        Pm, rep, sh = problem(op, rho, P)
        e0 = rho * T0 / Pm.gm1
        dt = DT_PROD * 4.0 ** -k
        dumps = [at(rows, t) for t in TM]
        td = [r[0] for r in dumps]
        nd = [int(round(t / dt)) for t in td]
        ode = M.exact_ode_tab if rep is None else M.exact_ode
        ex = ode(Pm, e0, E0, td)
        refk = {"T": Pm.T_of_e(ex[0]), "E": ex[1:]}
        if ref is None:
            ex4 = ode(Pm, e0, E0, TM)
            ref = {"T": Pm.T_of_e(ex4[0]), "E": ex4[1:], "Etot0": e0 + E0.sum()}
            lines.append(f"  exact ODE: T(t_m) = {np.array2string(ref['T'], precision=10)}  E_g(t_4) = "
                         f"{np.array2string(ref['E'][:, -1], precision=6)}  (IC T = {T0:.6g}, E_g = "
                         f"{np.array2string(E0, precision=6)})")
        mult = (lambda Tk: (None, None)) if rep is None else rep
        ser = {}
        for name, step in (("replica", M.code_step), ("imc", M.imc_step)):
            o = M.march(step, Pm, e0, E0, dt, nd[-1], float(alpha), mult, every=1)
            ser[name] = [o[n] for n in nd]
        S = M.fleck_f(Pm, T0, *Pm.coeffs(T0, mult)[::2], dt, float(alpha))[1]
        cT = [r[2] for r in dumps]
        cE = np.array([r[3] for r in dumps]).T
        # live groups: equilibrium share > 1e-10 of the total (the empty UV group integrates to +-1e-13 noise)
        live = [g for g in range(3) if ref["E"][g, -1] > 1e-10 * ref["E"][:, -1].sum()]
        impl = max(rel(cT, [Pm.T_of_e(x[1]) for x in ser["replica"]]),
                   max(rel(cE[g], [x[2][g] for x in ser["replica"]]) for g in live))
        errT = [rel(cT[m], refk["T"][m]) for m in range(4)]
        errE = [max(rel(cE[g, m], refk["E"][g, m]) for g in live) for m in range(4)]
        errEg = {g: [rel(cE[g, m], refk["E"][g, m]) for m in range(4)] for g in live}
        imcEg = {g: [rel(ser["imc"][m][2][g], refk["E"][g, m]) for m in range(4)] for g in live}
        imcT = [rel(Pm.T_of_e(ser["imc"][m][1]), refk["T"][m]) for m in range(4)]
        imcE = [max(rel(ser["imc"][m][2][g], refk["E"][g, m]) for g in live) for m in range(4)]
        etot = rho * cT[-1] / Pm.gm1 + cE[:, -1].sum()
        lines.append(f"  k={k} dt={dt:.3e} S={S:9.3g} | code-replica {impl:.1e} | code vs exact: T "
                     + " ".join(f"{x:.2e}" for x in errT) + " ; E_g " + " ".join(f"{x:.2e}" for x in errE)
                     + " | IMC vs exact: T " + " ".join(f"{x:.2e}" for x in imcT) + " ; E_g "
                     + " ".join(f"{x:.2e}" for x in imcE) + f" | dE_tot/E_tot {abs(etot / ref['Etot0'] - 1):.1e}")
        for g in live:
            lines.append(f"        group {g}: code " + " ".join(f"{x:.2e}" for x in errEg[g]) + "  | IMC "
                         + " ".join(f"{x:.2e}" for x in imcEg[g]))
    return lines


def render():
    L = ["# R1 0-D multigroup relaxation (scripts/analyze_R1.py; decks make_decks_R1.py; binary athenaPK_cpu_4135083d)",
         "# columns: S = Fleck stiffness at t=0; code-replica = max rel diff of T, E_g vs the code's own discrete step",
         "# (implementation; floor = sn_rel_tol 1e-8); errors vs the exact ODE at t = 1,2,3,4 x 1.1e-5 (T; max over live",
         "# groups of E_g); IMC = Fleck & Cummings cross-group re-emission at the same dt; dE_tot = energy conservation."]
    for pn in PAIRS:
        for op in OPS:
            for a in (("1.0", "0.5") if (pn, op) == ("g100r30", "c2") else ("1.0",)):
                L.append(f"== {pn} {op} alpha={a}")
                L += family(pn, op, a)
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    txt = render()
    if sys.argv[1] == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)
