"""D3 decks: planar Noh problem = two cold streams colliding (eos_test riemann, rho = 1 code, T0 = 10 K, u = +-M cs0,
rho26 table, eos_interp p). Two shocks run outward from x = 0.5, leaving gas at rest; star state from the exact
general-EOS Riemann solver (eos_tabulated/scripts/riemann_exact.py, on the runtime interpolant). Pre-shock gas is cold,
hypersonic and DUAL-SWITCHED (e/E ~ 1/(1 + 0.56 M^2)); a shocked cell must take its energy from E (the eta2
shock exemption: comp = -div v dt > shock_comp), otherwise it stays cold. Outflow boundaries keep the inflow coming
(boundary_flux_clamp off: it would block the inflow, DE-F2). Run to t with the shocks at +-0.3 from the centre.
Variants (CLI): prod; off; eta2 0 (one-sided switch); exempt off (shock_comp 1e30 -> the exemption never fires);
decomposition N 1024: 1 block / 32 blocks x 1 rank / 32 blocks x 8 ranks, comp_ghost on (prod) and off (P-02).
M in {3, 10, 30}; N in {256, 1024}.
Writes inputs/D3*.in, inputs/D3_runs.txt (4th field = ranks), inputs/D3_meta.json."""
import json
import os

import numpy as np

from de_common import INP, deck, fmt, hydro_block, mesh_block, table_eos
import riemann_exact as RX  # noqa: E402 (path set by de_common)

eos = table_eos()
m1 = json.load(open(os.path.join(INP, "D1_meta.json")))
cs0, e0 = m1["cs0"], m1["e0"]
VAR = {"prod": "", "off": "hydro/dual_energy=false", "eta2off": "hydro/dual_energy_eta2=0.0",
       "noexempt": "hydro/dual_energy_shock_comp=1.0e30"}
runs, meta = [], {}
for M in (3, 10, 30):
    u = M * cs0
    so = RX.solve(eos, (1.0, u, e0), (1.0, -u, e0))
    S = abs(so["R"]["S"])
    T2 = float(10 ** eos.logT(so["R"]["rho"], so["R"]["e"]))
    tl = 0.3 / S
    for N in (256, 1024):
        name = f"D3_M{M}_N{N}"
        prob = f"""<problem/eos_test>
mode = riemann
x0 = 0.5
rhoL = 1.0
espL_code = {fmt(e0)}
uL = {fmt(u)}
rhoR = 1.0
espR_code = {fmt(e0)}
uR = {fmt(-u)}
"""
        txt = deck(f"D3 planar Noh, u = +-{M} cs0, T0 10 K, N {N}; exact rho2 {so['R']['rho']:.6g} P2 {so['P']:.6g} T2 {T2:.4g} K",
                   prob, mesh_block(N, bc="outflow"), tl, tl / 4, hydro_block(1), hst_dt=tl / 200)
        open(os.path.join(INP, name + ".in"), "w").write(txt)
        meta[name] = dict(M=M, N=N, u=u, e0=e0, rho2=float(so["R"]["rho"]), e2=float(so["R"]["e"]), P2=float(so["P"]),
                          T2=T2, S=S, tlim=tl)
        for v, ov in VAR.items():
            runs.append((f"D3/{name}_{v}", name + ".in", ov, "1"))
        if N == 1024:
            mb = "parthenon/meshblock/nx1=32"
            runs.append((f"D3/{name}_mb32", name + ".in", mb, "1"))
            runs.append((f"D3/{name}_mb32_np8", name + ".in", mb, "8"))
            runs.append((f"D3/{name}_mb32_np8_noghost", name + ".in", mb + " hydro/dual_comp_ghost=false", "8"))
    print(f"M {M}: rho2 {so['R']['rho']:.4f} P2 {so['P']:.4g} T2 {T2:.1f} K S {S:.4g}")
with open(os.path.join(INP, "D3_runs.txt"), "w") as f:
    for r in runs:
        f.write(" | ".join(r) + "\n")
json.dump(dict(runs=meta), open(os.path.join(INP, "D3_meta.json"), "w"), indent=1)
print(len(runs), "runs")
