#!/usr/bin/env python
"""E10 floor-visibility probe: which diagnostics see a cell whose internal energy is below the table
(0 < esp < esp_min) or non-positive, with the production floor keys (pfloor 1e-8, dfloor 1e-6), for
eos_floor_consistent on/off. Per leg: EOS clamp WARNING lines in the log (CELL-STATE esp< counts),
hst cons-npfloor / cons-nfloor per cycle, and the pressure / temperature the right half ends with.
-> results/tables/E10_probe.txt"""
import glob, os, re, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E
import riemann_exact as RX

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs", sys.argv[1] if len(sys.argv) > 1 else "E10")
PAT = sys.argv[2] if len(sys.argv) > 2 else "E10_probe_*"
OUT = sys.argv[3] if len(sys.argv) > 3 else "E10_probe.txt"
t = E.load(); eos = RX.TableEOS(t, "p_over_rho"); esp_min = 10 ** t["le0"]
out = [f"esp_min (code) {esp_min:.6e}; P(rho, esp_min)/rho = {float(eos.P(18.29, esp_min)) / 18.29:.4e}; pfloor 1e-8"]
for d in sorted(glob.glob(os.path.join(RUNS, PAT))):
    log = open(os.path.join(d, "run.log"), errors="replace").read()
    warn = re.findall(r"### (WARNING|NOTE) EOS table: (\d+) CELL-STATE .*?esp<[^:]*: (\d+)", log)
    hst = glob.glob(os.path.join(d, "*.hst"))
    hrows = []
    if hst:
        L = open(hst[0]).read().splitlines()
        hdr = [l for l in L if l.startswith("#") and "[1]" in l][0]
        cols = {n: int(i) - 1 for i, n in re.findall(r"\[(\d+)\]=(\S+)", hdr)}
        for l in L:
            if l.startswith("#"):
                continue
            v = l.split()
            hrows.append((int(float(v[cols["cycle"]])), float(v[cols.get("cons-npfloor", 0)]) if "cons-npfloor" in cols else None,
                          float(v[cols["tot-E"]])))
    f = glob.glob(os.path.join(d, "*.out0.final.phdf"))
    with h5py.File(f[0]) as h:
        p = h["prim"][()].astype(float)
    rho = p[0, 0, 0, 0, :]; P = p[0, 4, 0, 0, :]
    n = rho.size; R = slice(n // 2 + 4, n)   # right half away from the interface
    PoR = P[R] / rho[R]

    out.append(f"{os.path.basename(d):30s} EOS-warning lines {len(warn)} (esp< per cycle {[int(w[2]) for w in warn]})"
               f" | hst (cycle, npfloor, tot-E) {hrows}"
               f" | right half: P/rho in [{PoR.min():.4e}, {PoR.max():.4e}], P/pfloor {P[R].min() / 1e-8:.3e}")
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "..", "results", "tables", OUT), "w").write(txt + "\n")
