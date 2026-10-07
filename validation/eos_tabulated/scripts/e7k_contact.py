#!/usr/bin/env python
"""E7k: pure contact discontinuity with the tabulated EOS (hypothesis: spurious pressure from mixed cells at a contact
between EOS regimes drives the slow R2/R4 convergence). Left/right = the exact STAR states of R4 (and R2, R1 as
control): same P* and u*, different (rho, e). Exact solution: uniform P and u, rho/e advected. Decks
E7k_<prob>_<u>_n<N> for u = u* (moving) and 0 (resting contact), PLM and LimO3."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
import make_decks_eos2 as D
HERE = os.path.dirname(os.path.abspath(__file__))
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json"))); t = E.load(); eos = RX.TableEOS(t, "p_over_rho")
meta = {}
for prob in ("R4_ionization", "R2_dissociation", "R1_molecular"):
    r = ex[prob + "|p_over_rho"]; so = RX.solve(eos, tuple(r["L"]), tuple(r["R"]))
    rL, rR = so["L"]["rho"], so["R"]["rho"]; P = so["P"]
    eL, eR = eos.e_of_P(rL, P), eos.e_of_P(rR, P)
    TL, TR = 10 ** float(eos.logT(rL, eL)), 10 ** float(eos.logT(rR, eR))
    for utag, u in (("mov", so["u"]), ("rest", 0.0)):
        cs = max(RX.c2(eos, rL, eL), RX.c2(eos, rR, eR)) ** 0.5
        tlim = 0.3 / abs(u) if u != 0 else 0.3 / cs      # contact moves 0.3 (or one sound crossing of 0.3 at rest)
        x0 = 0.35 if u != 0 else 0.5
        mb = (f"mode = riemann\nx0 = {x0!r}\nrhoL = {float(rL)!r}\nTL = {TL!r}\nuL = {float(u)!r}\n"
              f"rhoR = {float(rR)!r}\nTR = {TR!r}\nuR = {float(u)!r}\nespL_code = {float(eL)!r}\nespR_code = {float(eR)!r}\n")
        for nx in (400, 800, 1600, 3200, 6400):
            D.deck(f"E7k_{prob}_{utag}_n{nx}", f"E7k pure contact, star states of {prob}, u = {utag}, N = {nx}", mb, nx, float(tlim),
                   extra_hydro="eos_interp = p_over_rho\n", bc="outflow")
        meta[f"{prob}|{utag}"] = dict(rhoL=float(rL), rhoR=float(rR), eL=float(eL), eR=float(eR), P=float(P), u=float(u), x0=x0, tlim=float(tlim), TL=TL, TR=TR)
        print(prob, utag, f"rho {rL:.4e}|{rR:.4e}  T {TL:.0f}|{TR:.0f} K  P {P:.5e}  u {u:.4g}  tlim {tlim:.4g}")
json.dump(meta, open(os.path.join(HERE, "..", "inputs", "E7k_contact.json"), "w"), indent=1)
