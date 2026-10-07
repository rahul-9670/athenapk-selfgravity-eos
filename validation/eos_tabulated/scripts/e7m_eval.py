#!/usr/bin/env python
"""E7m: hydro/double_flux (E-F10 fix) vs the standard conservative PLM.
(1) pure contacts (E7k states): spurious |P/P* - 1|, L1 and max;
(2) Riemann problems: plateau P* error and L1(rho) vs the exact solution;
(3) energy: total E (from prim through the exact table inverse) of the double-flux run minus that of the conservative
    run at the same time, relative to the total -- both runs see the same boundary fluxes, so this is the
    double flux's non-conservation. -> results/tables/E7m_dflux.txt"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E, riemann_exact as RX
from analyze_E7 import read, exact_profile, NFINE, X0


def plateau(so, rho, u, P, tt):
    n = rho.size; x = (np.arange(n) + 0.5) / n; xi = (x - X0) / tt
    edge = lambda s_: so[s_]["S"] if so[s_]["type"] == "shock" else so[s_]["tail"]
    aL, bL, aR, bR = edge("L"), so["u"], so["u"], edge("R")
    m = ((xi > aL + 0.25 * (bL - aL)) & (xi < bL - 0.25 * (bL - aL))) | ((xi > aR + 0.25 * (bR - aR)) & (xi < bR - 0.25 * (bR - aR)))
    return np.median(P[m]) / so["P"] - 1

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "runs")
t = E.load(); eos = RX.TableEOS(t, "p_over_rho")
ex = json.load(open(os.path.join(HERE, "..", "results", "tables", "E7_exact.json")))
meta = json.load(open(os.path.join(HERE, "..", "inputs", "E7k_contact.json")))
SUB = sys.argv[1] if len(sys.argv) > 1 else "E7m"; OUTN = sys.argv[2] if len(sys.argv) > 2 else "E7m_dflux.txt"
out = []


def Etot(rho, u, P):
    e = np.array([eos.e_of_P(r, p) for r, p in zip(rho, P)])
    return float(np.sum(rho * e + 0.5 * rho * u * u)) / rho.size


def std_dir(kind, prob, n):
    if kind == "contact":
        return os.path.join(RUNS, "E7k", f"E7k_{prob}_mov_n{n}_plm")
    if n <= 3200:
        return os.path.join(RUNS, "E7e", f"E7e_{prob}_p_over_rho_n{n}")
    return os.path.join(RUNS, "E7f", f"{prob}_n{n}")


out.append("(1) pure moving contact: L1 / max |P/P*-1|   [standard PLM]   | energy: (E_dflux - E_std)/E_std")
for prob in ("R4_ionization", "R2_dissociation", "R1_molecular"):
    m = meta[prob + "|mov"]
    for n in (400, 800, 1600, 3200, 6400):
        fd = glob.glob(os.path.join(RUNS, SUB, f"E7k_{prob}_mov_n{n}_dflux", "*.out0.final.phdf"))
        fs = glob.glob(os.path.join(std_dir("contact", prob, n), "*.out0.final.phdf"))
        if not fd:
            continue
        td, rd, ud, Pd, _ = read(fd[0]); dp = np.abs(Pd / m["P"] - 1)
        s = f"{prob:16s} N {n:5d}: {dp.mean():.2e} / {dp.max():.1e}"
        if fs:
            ts, rs, us, Ps, _ = read(fs[0]); dps = np.abs(Ps / m["P"] - 1)
            Ed, Es = Etot(rd, ud, Pd), Etot(rs, us, Ps)
            s += f"   [{dps.mean():.2e} / {dps.max():.1e}]   | dE/E {Ed / Es - 1:+.2e}"
        out.append(s)
out.append("(2) Riemann: plateau P* error and L1(rho) vs exact   [standard PLM]   | energy (E_dflux - E_std)/E_std")
for prob in ("R4_ionization", "R2_dissociation", "R1_molecular", "R3_dissoc_shock"):
    rec = ex[prob + "|p_over_rho"]; cache = None
    for n in (400, 800, 1600, 3200, 6400):
        fd = glob.glob(os.path.join(RUNS, SUB, f"E7e_{prob}_n{n}_dflux", "*.out0.final.phdf"))
        if not fd:
            continue
        td, rd, ud, Pd, _ = read(fd[0])
        if cache is None:
            cache = exact_profile(eos, rec, td)
        so, fine = cache
        L1 = lambda r: float(np.mean(np.abs(r - fine[0].reshape(n, -1).mean(1))) / np.abs(fine[0]).max())
        s = f"{prob:16s} N {n:5d}: dP* {plateau(so, rd, ud, Pd, td):+.2e}  L1 {L1(rd):.2e}"
        fs = glob.glob(os.path.join(std_dir("riemann", prob, n), "*.out0.final.phdf"))
        if fs:
            ts, rs, us, Ps, _ = read(fs[0])
            s += f"   [dP* {plateau(so, rs, us, Ps, ts):+.2e}  L1 {L1(rs):.2e}]   | dE/E {Etot(rd, ud, Pd) / Etot(rs, us, Ps) - 1:+.2e}"
        out.append(s)
txt = "\n".join(out); print(txt)
open(os.path.join(HERE, "..", "results", "tables", OUTN), "w").write(txt + "\n")
