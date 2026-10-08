"""D2y: (i) resolution study of the D2a low-beta sinusoid (par/perp, beta 1e-2 and 1e-3, N 256..4096; on / off /
dual_energy_flux = density), max and L1 of dP = P/P_ad(rho) - 1 at the last dump, and the observed order between
successive N; (ii) D2c low-beta contacts with dual_energy_flux = density vs D2c on/off.
-> results/tables/D2y_resolution.{txt,json}"""
import glob, json, os
import numpy as np
from analyze_D2 import cells, Pad, meta
from de_common import RUNS, TAB
rows, res = [], {}
def last(d):
    f = glob.glob(os.path.join(d, "*.out0.final.phdf")); return f[0] if f else None
for g in ("par", "perp"):
    for b in ("0.01", "0.001"):
        for v in ("on", "off", "den"):
            seq = []
            for N in (256, 1024, 2048, 4096):
                base = "D2a" if (N in (256, 1024) and v != "den") else "D2y"
                d = os.path.join(RUNS, base, f"D2a_{g}_b{b}_N{N}_{v}")
                f = last(d)
                if not f: continue
                t, c, _ = cells(f); dP = c["P"] / Pad(c["rho"]) - 1
                seq.append((N, float(np.abs(dP).max()), float(np.abs(dP).mean())))
            res[f"D2a_{g}_b{b}_{v}"] = seq
            txt = " ".join(f"N{N}: {mx:.2e}/{l1:.2e}" for N, mx, l1 in seq)
            ords = " ".join(f"{np.log(seq[i][1] / seq[i + 1][1]) / np.log(seq[i + 1][0] / seq[i][0]):.2f}" for i in range(len(seq) - 1))
            rows.append(f"D2a_{g}_b{b:6s} {v:4s} max/L1 {txt} | order(max) {ords}")
jc = json.load(open(os.path.join(TAB, "D2c_lowbeta_contact.json")))
for g in ("par", "perp"):
    for b in ("1", "0.01", "0.0001"):
        name = f"D2c_{g}_b{b}"; m = meta["runs"][name]
        f = last(os.path.join(RUNS, "D2y", f"{name}_den"))
        if not f: rows.append(f"{name} den MISSING"); continue
        _, c1, _ = cells(f)
        mx = float(np.abs(c1["P"] / m["P0"] - 1).max())
        on = jc.get(f"{name}_on", {}).get("maxdP", float("nan")); off = jc.get(f"{name}_off", {}).get("maxdP", float("nan"))
        res[f"{name}_den"] = dict(maxdP=mx, legacy=on, off=off)
        rows.append(f"{name:18s} contact max|P/P0-1|: density {mx:.2e}  legacy {on:.2e}  off {off:.2e}")
txt = "\n".join(rows); print(txt)
open(os.path.join(TAB, "D2y_resolution.txt"), "w").write("# D2y: D2a resolution study (max/L1 of dP at the last dump) and D2c contacts with dual_energy_flux = density\n" + txt + "\n")
json.dump(res, open(os.path.join(TAB, "D2y_resolution.json"), "w"), indent=1)
