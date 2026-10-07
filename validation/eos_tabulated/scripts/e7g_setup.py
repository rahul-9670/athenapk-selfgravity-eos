#!/usr/bin/env python
"""E7g (hypothesis H1': the rho-direction kinks of the bilinear P interpolant slow the recombination fans).
R2 and R4 (p_over_rho, linear) on finer rebuilds of the production table: r4 = 4x rho nodes only (e, T axes
identical), x2 / x4 = 2x / 4x on every axis. Exact solutions on each table's own interpolant (same states in
(rho, T, u)), decks E7g_<tab>_<prob>_n<N> pointing at that table. usage: e7g_setup.py TAB [TAB ...]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
import e7_exact as X
import make_decks_eos2 as D
HERE = os.path.dirname(os.path.abspath(__file__))
TABDIR = os.path.abspath(os.path.join(HERE, "..", "runs", "E7g_tables"))
dst = os.path.join(HERE, "..", "results", "tables", "E7g_exact.json")
res = json.load(open(dst)) if os.path.exists(dst) else {}
for tab in sys.argv[1:]:
    path = os.path.join(TABDIR, f"eos_table_v5cold_{tab}.bin")
    t = E.load(path); eos = RX.TableEOS(t, "p_over_rho")
    for prob in ("R2_dissociation", "R4_ionization"):
        pr = X.PROBLEMS[prob]
        st = [(rc / E.RHO_U, u / X.VU_KMS, float(eos.esp_of_T(rc / E.RHO_U, T))) for rc, T, u in (pr["L"], pr["R"])]
        so = RX.solve(eos, *st)
        vmax = max(abs(so["L"]["head"]), abs(so["R"]["S"]))
        rec = dict(table=path, L=st[0], R=st[1], P=so["P"], u=so["u"], tlim=0.40 / vmax, vmax=vmax,
                   L_T=pr["L"][1], R_T=pr["R"][1], Lstar=dict(rho=so["L"]["rho"], tail=so["L"]["tail"], head=so["L"]["head"], type="rarefaction"),
                   Rstar=dict(rho=so["R"]["rho"], S=so["R"]["S"], type="shock"), what=pr["what"])
        res[f"{tab}|{prob}"] = rec
        print(tab, prob, f"P* {so['P']:.6e} u* {so['u']:.6e} tlim {rec['tlim']:.5g}", flush=True)
        hyd = D.HYDRO.replace(E.TABLE, path)
        mb = (f"mode = riemann\nx0 = 0.5\nrhoL = {st[0][0]!r}\nTL = {pr['L'][1]!r}\nuL = {st[0][1]!r}\n"
              f"rhoR = {st[1][0]!r}\nTR = {pr['R'][1]!r}\nuR = {st[1][1]!r}\n"
              f"espL_code = {st[0][2]!r}\nespR_code = {st[1][2]!r}\n")
        for nx in (800, 1600, 3200, 6400):
            D.deck(f"E7g_{tab}_{prob}_n{nx}", f"E7g {prob} on table {tab}, p_over_rho linear, N = {nx}", mb, nx, rec["tlim"],
                   extra_hydro="eos_interp = p_over_rho\n", hydro=hyd)
json.dump(res, open(dst, "w"), indent=1, default=float)
