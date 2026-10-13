#!/usr/bin/env python
"""Fix campaign (make_runs_fix.py; binary d4ca78e9): IM-F3 GLM damping, IM-F1 shock detector, IM-F2 boundary.
Reuses the module's scorers (analyze_I4/I5/I6/I6b/I7) on runs/F/*; adds the exact no-inflow wall test:
  B_x = 0, uniform rho0 1, P0 0.6, By0, gas leaving the wall at u0: the wall state is the fast rarefaction with u* = 0,
  u0 = int_{rho*}^{rho0} c_f(rho) drho / rho, c_f^2 = gamma P0 (rho/rho0)^gamma / rho + By0^2 (rho/rho0)^2 / rho (B ~ rho).
  Score: mean rho of the first 3 cells at each wall at t = 0.3 vs rho* (and the legacy clamp / wall-only arms).
  analyze_fix.py [--check] -> results/tables/F.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, math, os, sys
import h5py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, _VT + "/ideal_mhd/scripts")
import analyze_I4 as A4  # noqa: E402
import analyze_I5 as A5  # noqa: E402
import analyze_I6 as A6  # noqa: E402
import analyze_I6b as A6b  # noqa: E402

M = _VT + "/ideal_mhd"
F = f"{M}/runs/F"
G = 5 / 3


def bitcmp(da, db):
    fa = sorted(glob.glob(f"{da}/*.out0.final.phdf")) or sorted(glob.glob(f"{da}/*.out0.*.phdf"))
    fb = sorted(glob.glob(f"{db}/*.out0.final.phdf")) or sorted(glob.glob(f"{db}/*.out0.*.phdf"))
    if not fa or not fb:
        ea, eb = f"{da}/linearwave-errors.dat", f"{db}/linearwave-errors.dat"
        if os.path.exists(ea) and os.path.exists(eb):
            return "IDENTICAL" if open(ea).read() == open(eb).read() else "DIFFER"
        return "MISSING"
    with h5py.File(fa[-1]) as a, h5py.File(fb[-1]) as b:
        for k in a:
            if k in ("Info", "Params", "Input") or not isinstance(a[k], h5py.Dataset):
                continue
            if not np.array_equal(a[k][()].view(np.uint8), b[k][()].view(np.uint8)):
                return f"DIFFER ({k})"
    return "IDENTICAL"


def wall_exact(u0, By):
    if u0 == 0:
        return 1.0
    cf = lambda r: math.sqrt(G * 0.6 * r**G / r + By**2 * r**2 / r)
    f = lambda rs: quad(lambda r: cf(r) / r, rs, 1.0)[0] - u0
    return brentq(f, 1e-6, 1.0)


def wall(d, u0, By):
    f = sorted(glob.glob(f"{d}/parthenon.out0.*.phdf"))[-1]
    with h5py.File(f) as h:
        p = h["prim"][()][0, :, 0, 0, :]
    rs = wall_exact(u0, By)
    lo, hi = p[0, :3].mean(), p[0, -3:].mean()
    return {"rho_star": rs, "rho_wall_lo": float(lo), "rho_wall_hi": float(hi),
            "err_lo": float(lo / rs - 1), "err_hi": float(hi / rs - 1), "max_abs_u_wall": float(abs(p[1, :3]).max())}


def d1b_reg():
    """dual_energy D1b (cold sinusoid, M 1000, N 256): max |P/P_ad - 1| at the final dump, table adiabat through
    (rho 1, e0) exactly as dual_energy/scripts/analyze_D1.py d1b() (production reference there: 4.429e-03)."""
    sys.path.insert(0, _VT + "/dual_energy/scripts")
    import analyze_D1 as D1
    L, Ee = D1.adiabat_table(1.0, D1.meta["e0"], 0.2, 20.0)
    out = {}
    for tag, d in (("prod", f"{F}/f1reg_D1b_M1000_N256_prod"), ("pjump", f"{F}/f1reg_D1b_M1000_N256_f1"),
                   ("kjump", f"{F}/f1kreg_D1b_M1000_N256_f1"), ("kv", f"{F}/f1vreg_D1b_M1000_N256_f1")):
        fs = D1.dumps(d)
        if not fs:
            continue
        t, X, rho, v, P = D1.load(fs[-1])
        Pad = D1.eos.P(rho, np.interp(np.log(rho), L, Ee))
        out[tag] = {"t": t, "max_dP": float(np.abs(P / Pad - 1).max()), "L1_dP": float(np.abs(P / Pad - 1).mean())}
    return out


def d3_f1():
    """I6c scoring (plateau rho vs the exact general-EOS star state) of the tabulated D3 decks run with F1."""
    sys.path.insert(0, _VT + "/dual_energy/scripts")
    from analyze_D2 import cells
    meta = json.load(open(f"{M}/inputs/I6/D3_meta.json"))["runs"]
    out = {}
    for pre, m in [(p, m) for p in ("f1", "f1k", "f1v") for m in (3, 10, 30)]:
        f = glob.glob(f"{F}/{pre}_D3_M{m}/*.out0.final.phdf")
        if not f:
            continue
        mt = meta[f"D3_M{m}_N1024"]
        t, c, _ = cells(f[0])
        xi = np.abs(c["x"] - 0.5) / (mt["S"] * t)
        pl = (xi > 0.15) & (xi < 0.8)
        out[f"{pre}_M{m}"] = {"rho_plateau": float((c["rho"][pl] / mt["rho2"] - 1).mean()),
                        "P_plateau": float((c["P"][pl] / mt["P2"] - 1).mean())}
    return out


def main():
    res = {}
    res["keyoff"] = {d: bitcmp(f"{F}/keyoff_{d}", f"{M}/runs/{ref}") for d, ref in (
        ("I7_BW_diode", "I7/I7_BW_diode"), ("I5_2d_prod_N128", "I5/I5_2d_prod_N128"),
        ("noh_1d_prod_N1024", "I6/noh_1d_prod_N1024"), ("I1_1d_fast_N64", "I1/I1_1d_fast_N64"))}
    # GLM
    g = {}
    for d in sorted(glob.glob(f"{F}/glm_I5*")):
        n = os.path.basename(d)[4:]
        m = A5.PAT.match(n)
        if not m or not glob.glob(d + "/*.phdf"):
            continue
        dim, case, N, lev = int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5) or m.group(6) or 0)
        new = A5.score(d, dim, N, lev, case)["rows"][-1]
        old_d = f"{M}/runs/I5/{n}"
        old = A5.score(old_d, dim, N, lev, case)["rows"][-1] if glob.glob(old_d + "/*.phdf") else None
        g[n] = {"L1_Bx_midpoint": new["L1_Bx"], "L1_Bx_lie": old["L1_Bx"] if old else None,
                "L1_psi_midpoint": new["L1_psi"], "L1_psi_lie": old["L1_psi"] if old else None,
                "rel_midpoint": new["rel_err_Bx"]}
    res["glm_I5"] = g
    res["glm_I4"] = {}
    for n in ("I4_2d_prod_N128", "I4_3d_prod_N64"):
        d = f"{F}/glm_{n}"
        if len(glob.glob(d + "/*.phdf")) >= 5:
            r = A4.score(d, int(n[3]))["rows"][-1]
            o = A4.score(f"{M}/runs/I4/{n}", int(n[3]))["rows"][-1]
            res["glm_I4"][n] = {"midpoint": r, "lie": o}
    # F1
    f1 = {}
    for d in sorted(glob.glob(f"{F}/f1*_noh1d_*")):
        n = os.path.basename(d)
        if glob.glob(d + "/*.out0.*.phdf"):
            P0 = float(n.split("_P")[1].split("_")[0]) if "_P" in n else 1e-6
            r = A6b.one(d, P0)
            f1[n] = {k: r[k] for k in ("Mach", "plateau", "r_shock_rel") if k in r}
    for d in sorted(glob.glob(f"{F}/f1*_noh_*")):
        n = os.path.basename(d)
        if glob.glob(d + "/*.phdf"):
            f1[n] = A6.score(d, int(n.split("_")[2][0]))
    res["f1"] = f1
    res["f1_D1b_regression"] = d1b_reg()
    res["f1_D3_tabulated"] = d3_f1()
    # F2: I7 against the big-box references (analyze_I7.compare reads runs/I7/I7_<prob>_<kind>)
    import analyze_I7 as A7
    f2 = {}
    for p in ("BW", "RJ2a_div", "divb_a0", "divb_prod"):
        src = f"{F}/f2_I7_{p}"
        link = f"{M}/runs/I7/I7_{p}_f2fix"
        if glob.glob(src + "/*.phdf"):
            if not os.path.islink(link):
                os.symlink(src, link)
            f2[p] = A7.compare(p, "f2fix")[-1]
    res["f2_I7"] = f2
    w = {}
    for d in sorted(glob.glob(f"{F}/wall_*")):
        n = os.path.basename(d)
        if glob.glob(d + "/*.phdf"):
            u0 = float(n.split("_u")[1].split("_")[0]); By = float(n.split("_By")[1].split("_")[0])
            w[n] = wall(d, u0, By)
    res["f2_wall"] = w
    js = f"{M}/results/tables/F.json"
    if "--check" in sys.argv:
        ok = json.load(open(js)) == json.loads(json.dumps(res)); print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    json.dump(res, open(js, "w"), indent=1, default=float)
    L = ["Fix campaign (binary d4ca78e9)", "key-off bitwise vs the earlier runs: " + str(res["keyoff"]), "",
         "GLM midpoint vs lie (I5, L1 Bx / L1 psi at t 0.3):"]
    for n, r in g.items():
        L.append(f"  {n:<30} Bx {r['L1_Bx_midpoint']:.3e} (lie {r['L1_Bx_lie'] or float('nan'):.3e})  "
                 f"psi {r['L1_psi_midpoint']:.3e} (lie {r['L1_psi_lie'] or float('nan'):.3e})  rel {r['rel_midpoint']:.2e}")
    for n, r in res["glm_I4"].items():
        L.append(f"  {n}: ME {r['midpoint']['ME_ME0']:.4f} (lie {r['lie']['ME_ME0']:.4f}) mean divB {r['midpoint']['divB_mean']:.2e} "
                 f"(lie {r['lie']['divB_mean']:.2e}) max {r['midpoint']['divB_max']:.2e} (lie {r['lie']['divB_max']:.2e})")
    L += ["", "F1 (shock_eta1 + pjump 1.0):"]
    for n, r in f1.items():
        L.append(f"  {n:<28} " + " ".join(f"{k} {v:+.3e}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()
                                            if k in ("Mach", "plateau", "r_shock_rel", "pre_min", "pre_max", "aniso")))
    L.append("  D3 tabulated with F1 (plateau rho/P): " + json.dumps(res["f1_D3_tabulated"]))
    L.append("  D1b regression (max|P/P_ad-1|): " + json.dumps(res["f1_D1b_regression"]))
    L += ["", "F2 I7 (last dump vs big box): " + json.dumps({k: {q: round(v, 6) if isinstance(v, float) else v
                                                                for q, v in r.items() if q.startswith(('max_', 'mass'))}
                                                            for k, r in f2.items()})]
    L.append("F2 wall test (rho of the 3 wall cells / exact rho* - 1):")
    for n, r in w.items():
        L.append(f"  {n:<26} rho* {r['rho_star']:.5f}  lo {r['err_lo']:+.2e}  hi {r['err_hi']:+.2e}  |u| wall {r['max_abs_u_wall']:.1e}")
    open(js.replace(".json", ".txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
