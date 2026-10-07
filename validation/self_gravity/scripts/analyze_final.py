#!/usr/bin/env python
"""FINAL-configuration regression (runs/FINAL, suite_final.sh): production gravity keys only, final binary
(PLM VL2 predictor, NaN guard, SG-F1 floor + fallback). Reuses the round-1 analysis functions and prints the
round-1 numbers next to them."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import sgv, analyze_static as ast, analyze_jeans as aj, analyze_slab as asl, analyze_dynamic as ad  # noqa: E402
F = os.path.join(ast.RUNS, "FINAL")
OUT = os.path.join(ast.OUT, "tables", "final.json")
res = {}


def legs(pat):
    return sorted((d for d in glob.glob(os.path.join(F, pat)) if sgv.run_rc(d) == 0),
                  key=lambda d: int("".join(c for c in d.split("_n")[-1] if c.isdigit()) or 0))


def hist(rd):
    hs = glob.glob(os.path.join(rd, "*.hst"))
    if not hs:   # single-cycle static runs: no history; count non-convergence in the log
        log = open(sorted(glob.glob(os.path.join(rd, "run.log*")))[-1], errors="replace").read()
        return None, log.count("did NOT converge") + log.count("NOT CONVERGED")
    H = sgv.hst(hs[0]); ok = H["grav-iters"] >= 0
    return int(H["grav-iters"][ok].max()), int(H["grav-nonconv"][ok].sum())


print("S1 periodic sinusoid, production keys ONLY (round 1, hand-set floor: 4.86e-4, 1.21e-4, 3.01e-5, 7.53e-6; production keys alone: NaN at 16^3 and 64^3):")
res["S1"] = []
for rd in legs("S1_sinusoid/n*"):
    r = ast.analyse(rd); it, nc = hist(rd)
    res["S1"].append(dict(run=os.path.basename(rd), eps_abs=r["eps_abs"], iters_max=it, nonconv=nc))
    print(f"   {os.path.basename(rd):5s} rms err {r['eps_abs']:.3e}  its<= {it} nonconv {nc}")
for nj in ("0.5", "1.5"):
    print(f"D{1 if nj == '0.5' else 2} Jeans n_J = {nj} (round 1: " + ("4.09e-7, 7.95e-8, 1.13e-8, 1.51e-9" if nj == "0.5" else "2.40e-6, 6.68e-7, 1.63e-7, 3.88e-8") + " with a hand-set tolerance):")
    rows = [aj.errors(rd) for rd in legs(f"D12_jeans/nJ{nj}_n*")]
    res[f"D12_nJ{nj}"] = rows
    for i, r in enumerate(rows):
        o = f" order {np.log2(rows[i-1]['rms_l1'] / r['rms_l1']):.2f}" if i else ""
        print(f"   n={round(r['ncells'] ** (1/3) * 2 ** (2/3))}: RMS-L1 {r['rms_l1']:.3e}{o}  its<= {r.get('grav_iters_max')} nonconv {r.get('grav_nonconv')}")
print("D4 slab vl2 cfl 0.3 (round 1: 1.7685e-1, 4.0261e-2, 1.0171e-2, 2.6192e-3):")
rows = [asl.analyse(rd) for rd in legs("D4_slab/vl2_cfl0.3_n*")]
res["D4"] = rows
for i, r in enumerate(rows):
    o = f" order {np.log2(rows[i-1]['rms_l1'] / r['rms_l1']):.2f}" if i else ""
    print(f"   RMS-L1 {r['rms_l1']:.4e}{o}  mass drift {r['mass_rel_drift']:+.1e}  its<= {r['grav_iters_max']} nonconv {r['grav_nonconv']}")
print("D7 free fall 64^3, dt ladder (round 1 vl2/donor-cell predictor: +0.0531/+0.0263/+0.0128/+0.0060 % at t = 1.0):")
res["D7"] = {}
for rd in sorted(glob.glob(os.path.join(F, "D7_freefall/n64_dtceil*")), key=lambda d: -float(d.split("dtceil")[1])):
    if sgv.run_rc(rd) != 0:
        continue
    ff = ad.freefall(rd)
    pick = {}
    for tt in (0.6, 1.0, 1.4):
        q = min(ff["series"], key=lambda s: abs(s["t"] - tt)); pick[tt] = q["err_c_pct"]
    res["D7"][os.path.basename(rd)] = pick
    print(f"   {os.path.basename(rd):18s} " + "  ".join(f"t={k}: {v:+.4f} %" for k, v in pick.items()))
json.dump(res, open(OUT, "w"), indent=1, default=float)
