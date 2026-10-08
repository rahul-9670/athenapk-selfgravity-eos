"""D1 analysis.

D1b (cold sinusoid): every cell must stay on the table adiabat through (rho0, e0) until the caustic. Metric per cell
dP = P / P_ad(rho) - 1, P_ad(rho) = P(rho, e_ad(rho)) with e_ad from de/dln rho = P(rho, e)/rho integrated (RK4,
h = 1e-4) on the SAME interpolant the code uses (eos_interp = p). Kinematics vs the ballistic (pressureless) map
x = q + vamp sin(kq) t, rho = rho0 / (1 + vamp k t cos kq) (valid to O(M^-2)) as a sanity check.
D1a (contact): max |P/P0 - 1| and L1 density error vs the exact translated profile (= the IC after one crossing).
Diagnostics from the hst: cons-ndual (switched-cell count, summed over hst rows), cons-npfloor, cons-nTfloor.
-> results/tables/D1a_contact.{txt,json}, D1b_adiabat.{txt,json}"""
import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy.optimize import brentq

from de_common import INP, RUNS, TAB, table_eos

eos = table_eos()
meta = json.load(open(os.path.join(INP, "D1_meta.json")))


def dumps(d):
    fs = sorted(glob.glob(os.path.join(d, "*.out0.0*.phdf")))
    fin = glob.glob(os.path.join(d, "*.out0.final.phdf"))
    return fs + fin


def load(f, want_sw=False):
    with h5py.File(f) as h:
        p = h["prim"][()].astype(float)
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        x = h["Locations"]["x"][()]
        sw = h["dual_switched"][()].astype(float).reshape(p.shape[0], -1) if "dual_switched" in h else None
    xc = 0.5 * (x[:, 1:] + x[:, :-1])
    nb = p.shape[0]
    rho = np.concatenate([p[b, 0, 0, 0, :] for b in range(nb)])
    v = np.concatenate([p[b, 1, 0, 0, :] for b in range(nb)])
    P = np.concatenate([p[b, 4, 0, 0, :] for b in range(nb)])
    X = np.concatenate([xc[b] for b in range(nb)])
    o = np.argsort(X)
    if want_sw:
        S = np.concatenate([sw[b] for b in range(nb)])[o] if sw is not None else np.zeros_like(rho)
        return t, X[o], rho[o], v[o], P[o], S
    return t, X[o], rho[o], v[o], P[o]


def hst(d):
    fs = glob.glob(os.path.join(d, "*.out1.hst"))
    if not fs:
        return {}
    hdr = [l for l in open(fs[0]) if l.startswith("# [1]")][0]
    cols = [s.split("=")[1].strip() for s in hdr.replace("#", "").split("[")[1:]]
    h = np.atleast_2d(np.loadtxt(fs[0]))
    return {n: h[:, i] for i, n in enumerate(cols)}


def hcount(c, key):
    return float(c[key].sum()) if key in c else float("nan")


def adiabat_table(r0, e0, lo, hi, h=1e-4):
    f = lambda lr, e: float(eos.P(np.exp(lr), e)) / np.exp(lr)
    out = {}
    for sgn, end in ((+1, np.log(hi)), (-1, np.log(lo))):
        lr, e, L, Ee = np.log(r0), e0, [np.log(r0)], [e0]
        while sgn * (end - lr) > 1e-14:
            hh = sgn * min(h, sgn * (end - lr))
            k1 = f(lr, e); k2 = f(lr + hh / 2, e + hh / 2 * k1); k3 = f(lr + hh / 2, e + hh / 2 * k2)
            k4 = f(lr + hh, e + hh * k3)
            e += hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4); lr += hh
            L.append(lr); Ee.append(e)
        out[sgn] = (np.array(L), np.array(Ee))
    L = np.concatenate([out[-1][0][::-1], out[1][0][1:]]); Ee = np.concatenate([out[-1][1][::-1], out[1][1][1:]])
    return L, Ee


def ballistic_rho(X, t, vamp, u0, rho0):
    """Vectorised bisection on the (pre-caustic, monotone) Lagrangian map q + vamp t sin(kq) = x - u0 t."""
    k = 2 * np.pi
    xs = (X - u0 * t) % 1.0
    a, b = xs - vamp * t - 1e-12, xs + vamp * t + 1e-12
    for _ in range(80):
        c = 0.5 * (a + b)
        gc = c + vamp * np.sin(k * c) * t - xs
        a = np.where(gc < 0, c, a); b = np.where(gc < 0, b, c)
    q = 0.5 * (a + b)
    return rho0 / (1 + vamp * k * t * np.cos(k * q))


def d1a():
    rows, res = [], {}
    for name, m in sorted(meta["runs"].items(), key=lambda kv: (kv[1]["test"], kv[1].get("M", 0))):
        if m["test"] != "D1a":
            continue
        for tag in ("on", "off", "on_nodf", "off_nodf"):
            d = os.path.join(RUNS, "D1a", f"{name}_{tag}")
            fs = dumps(d)
            if not glob.glob(os.path.join(d, "*.out0.final.phdf")):
                rows.append(f"{name:12s} {tag:9s} MISSING/INCOMPLETE"); continue
            t0, X, r0, _, _ = load(fs[0]); t, X1, r1, v1, P1 = load(fs[-1])
            dP = P1 / meta["P0"] - 1
            c = hst(d)
            r = dict(M=m["M"], tag=tag, t=t, maxdP=float(np.abs(dP).max()), L1rho=float(np.abs(r1 - r0).mean()),
                     maxdv=float(np.abs(v1 / m["u"] - 1).max()) if m["u"] > 0 else float(np.abs(v1).max()),
                     ndual=hcount(c, "cons-ndual"), npfloor=hcount(c, "cons-npfloor"), nTfloor=hcount(c, "cons-nTfloor"))
            res[f"{name}_{tag}"] = r
            rows.append(f"{name:12s} {tag:9s} t {t:.4g}: max|P/P0-1| {r['maxdP']:.3e}  L1(rho-rho_exact) {r['L1rho']:.3e}  "
                        f"max|v/u-1| {r['maxdv']:.2e}  ndual(sum) {r['ndual']:.0f} npfloor {r['npfloor']:.0f} nTfloor {r['nTfloor']:.0f}")
    txt = "\n".join(rows)
    print(txt)
    open(os.path.join(TAB, "D1a_contact.txt"), "w").write(
        "# D1a pure contact rho 1|2 (10 K | ~5 K) at P0 (rho26 table, eos_interp p), N = 256, one box crossing; exact = IC\n" + txt + "\n")
    json.dump(res, open(os.path.join(TAB, "D1a_contact.json"), "w"), indent=1)


def d1b():
    import eostab as ET
    from de_common import TABLE
    tb = ET.load(TABLE)
    L, Ee = adiabat_table(1.0, meta["e0"], 0.2, 20.0)
    Pad = lambda rho: eos.P(rho, np.interp(np.log(rho), L, Ee))
    import riemann_exact as RXm
    E_cs2 = lambda rho: RXm.TableEOS._bilin(tb["cs2"], tb["lr0"], tb["dlr"], tb["le0"], tb["dle"], np.log10(rho),
                                            np.log10(np.interp(np.log(rho), L, Ee)))
    rows, res = [], {}
    keys = sorted([k for k, m in meta["runs"].items() if m["test"] == "D1b"],
                  key=lambda k: (meta["runs"][k]["u0"] > 0, meta["runs"][k]["M"], meta["runs"][k]["N"]))
    for name in keys:
        m = meta["runs"][name]
        for tag in ("on", "off"):
            d = os.path.join(RUNS, "D1b", f"{name}_{tag}")
            fs = dumps(d)
            if not glob.glob(os.path.join(d, "*.out0.final.phdf")):
                rows.append(f"{name:22s} {tag:3s} MISSING/INCOMPLETE"); continue
            per = []
            for f in fs[1:]:
                t, X, rho, v, P, S = load(f, want_sw=True)
                dP = P / Pad(rho) - 1
                # local resolution of the velocity field: s = |v_{i+1} - v_{i-1}| / (2 cs_ad), cs from the adiabat state
                cs = np.sqrt(np.maximum(E_cs2(rho), 1e-300))
                sres = np.abs(np.roll(v, -1) - np.roll(v, 1)) / (2 * cs)
                cls = {}
                for cn, msk in (("sw_res", (S > 0.5) & (sres < 1)), ("sw_unres", (S > 0.5) & (sres >= 1)),
                                ("E_res", (S <= 0.5) & (sres < 1)), ("E_unres", (S <= 0.5) & (sres >= 1))):
                    cls[cn] = dict(n=int(msk.sum()), max=float(np.abs(dP[msk]).max()) if msk.any() else 0.0,
                                   L1=float(np.abs(dP[msk]).mean()) if msk.any() else 0.0)
                kin = rho / ballistic_rho(X, t, m["vamp"], m["u0"], 1.0) - 1 if m["M"] >= 100 else np.full_like(rho, np.nan)
                per.append(dict(t_tc=t / m["tc"], maxdP=float(np.abs(dP).max()), L1dP=float(np.abs(dP).mean()), cls=cls,
                                smax=float(sres.max()),
                                rhomax=float(rho.max()), rhomin=float(rho.min()), kin=float(np.abs(kin).mean())))
            c = hst(d)
            r = dict(M=m["M"], N=m["N"], u0=m["u0"], tag=tag, dumps=per, ndual=hcount(c, "cons-ndual"),
                     ncell_rows=float(len(c.get("cons-ndual", [])) * m["N"]), npfloor=hcount(c, "cons-npfloor"),
                     nTfloor=hcount(c, "cons-nTfloor"))
            res[f"{name}_{tag}"] = r
            fin = per[-1]; mid = [p for p in per if abs(p["t_tc"] - 0.4) < 0.05]
            mid = mid[0] if mid else per[len(per) // 2]
            frac = r["ndual"] / r["ncell_rows"] if r["ncell_rows"] else float("nan")
            cc = lambda p, k: f"{p['cls'][k]['n']}:{p['cls'][k]['max']:.1e}"
            rows.append(f"{name:22s} {tag:3s} | t/tc {mid['t_tc']:.2f}: max|dP| {mid['maxdP']:.3e} L1 {mid['L1dP']:.3e} "
                        f"[sw_res {cc(mid,'sw_res')} sw_unres {cc(mid,'sw_unres')} E_res {cc(mid,'E_res')} E_unres {cc(mid,'E_unres')}; smax {mid['smax']:.2g}] | "
                        f"t/tc {fin['t_tc']:.2f}: max|dP| {fin['maxdP']:.3e} L1 {fin['L1dP']:.3e} rho {fin['rhomin']:.3f}..{fin['rhomax']:.3f}"
                        f" kin {fin['kin']:.1e} | switched frac {frac:.3f} npfloor {r['npfloor']:.0f} nTfloor {r['nTfloor']:.0f}")
    txt = "\n".join(rows)
    print(txt)
    open(os.path.join(TAB, "D1b_adiabat.txt"), "w").write(
        "# D1b cold sinusoid, dP = P/P_ad(rho) - 1 (table adiabat through rho 1, 10 K; eos_interp p); kin = mean|rho/rho_ballistic - 1| (M >= 100)\n"
        "# switched frac = sum(cons-ndual)/(hst rows x N) = time-averaged fraction of switched cells\n"
        "# [class n:max|dP|] at t/tc ~ 0.4: sw = dual-switched (dual_switched field), E = total-energy branch; res = s < 1, s = |v_i+1 - v_i-1|/(2 cs_ad)\n" + txt + "\n")
    json.dump(res, open(os.path.join(TAB, "D1b_adiabat.json"), "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b"]
    if "a" in which:
        d1a()
    if "b" in which:
        d1b()
