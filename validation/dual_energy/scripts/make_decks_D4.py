"""D4 decks: homologous compression of low-beta gas along the table adiabat (EOS module E8 harness: eos_test compress +
eos_homologous ghosts, 33 cells, one block) with a transverse field B_y = B0 (beta0 = 2 P0/B0^2) that the compression
carries (B_y/rho conserved). rho 1e-13 -> 1e-11 g/cc (x 101) from 10 K, alpha >= E8 leg 0's, scaled with the fast speed for cost (the solution is alpha-independent). Exact: the centre
cell (v = 0) follows the table adiabat through (rho0, e0) and B_y/rho = B0/rho0. Production keys (rho26, eos_interp p);
boundary_flux_clamp off (the homologous inflow enters through the boundary). beta0 in {1, 1e-2, 1e-4}; dual on / off.
Writes inputs/D4*.in, inputs/D4_runs.txt, inputs/D4_meta.json."""
import json, os
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root
import numpy as np
from de_common import INP, deck, fmt, hydro_block, mesh_block, table_eos
eos = table_eos()
e8 = json.load(open(_REPO + "/validation/eos_tabulated/inputs/E8_legs.json"))["E8_leg0"]
rho0, T0, alpha, tl = e8["rho0"], e8["T0"], e8["alpha"], e8["tlim"]
e0 = float(eos.esp_of_T(rho0, T0)); P0 = float(eos.P(rho0, e0))
runs, meta = [], {}
for beta in (1.0, 1e-2, 1e-4):
    B0 = np.sqrt(2 * P0 / beta)
    # The homologous solution is exact for ANY alpha (uniform box, only pdV work), so alpha is chosen for cost: E8's
    # alpha (0.0091) at beta0 1e-4 needs ~1e7 cycles (c_f ~ 140 code, growing as sqrt(rho)). Target ~2e4 steps:
    # steps ~ c_f,mean / (alpha 0.3 dx), c_f,mean ~ 5 c_f0 over the x101 leg. Never slower than E8.
    cf0 = np.sqrt(1.67 * P0 / rho0 + B0 ** 2 / rho0)
    alpha_b = max(alpha, 5.0 * cf0 * 33 / (0.3 * 2.0e4))
    tl_b = (1.0 - 1.0 / 101.0) / alpha_b
    name = f"D4_b{beta:g}"
    prob = f"""<problem/eos_test>
mode = compress
rho0 = {fmt(rho0)}
T0 = {fmt(T0)}
alpha = {fmt(alpha_b)}
xc = 0.5
B0y = {fmt(B0)}
"""
    txt = deck(f"D4 homologous compression x101 from (1e-13 g/cc, 10 K), transverse B beta0 {beta:g} (B0 {B0:.6g})", prob,
               mesh_block(33, bc="eos_homologous"), tl_b, tl_b / 60, hydro_block(1), hst_dt=tl_b / 400)
    open(os.path.join(INP, name + ".in"), "w").write(txt)
    meta[name] = dict(beta=beta, B0=B0, rho0=rho0, e0=e0, P0=P0, alpha=alpha_b, tlim=tl_b)
    print(name, "alpha", alpha_b, "tlim", tl_b)
    for v, ov in (("on", ""), ("off", "hydro/dual_energy=false")):
        runs.append((f"D4/{name}_{v}", name + ".in", ov))
open(os.path.join(INP, "D4_runs.txt"), "w").write("".join(" | ".join(r) + "\n" for r in runs))
json.dump(dict(runs=meta), open(os.path.join(INP, "D4_meta.json"), "w"), indent=1)
print(len(runs), "runs; P0", P0, "e0", e0)
