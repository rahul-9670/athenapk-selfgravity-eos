#!/usr/bin/env python
"""SG-D1/D2 (Jeans wave / instability, convergence) and SG-D3 (dispersion relation sweep).

Exact solution (linear, adiabatic, rho0 = 1, cs^2 = gamma p0/rho0 = 1, k = 6 pi along the
Athena++ oblique direction; see src/pgen/jeans_wave.cpp):
  stable:   rho = 1 + A sin(kx) cos(wt),  v = -(w/k) A cos(kx) sin(wt),  p = p0 (1 + gamma A sin(kx) cos(wt))
  unstable: rho = 1 + A sin(kx) e^{gt},   v = +(g/k) A cos(kx) e^{gt},   p = p0 (1 + gamma A sin(kx) e^{gt})
Errors as in Athena++ src/pgen/jeans.cpp: L1 = sum |q - q*| / Ncells for rho, rho v_1..3 and P;
RMS-L1 = sqrt(sum of the squares).
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import sgv  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results")
GAM = 5.0 / 3.0


def params(f):
    import h5py
    with h5py.File(f, "r") as h:
        s = h["Input"].attrs["File"]
    s = s.decode() if isinstance(s, bytes) else s
    get = lambda blk, key: float(re.search(rf"<{re.escape(blk)}>[^<]*?\n{key}\s*=\s*([^\s#]+)", s).group(1))
    return {"four_pi_G": get("self_gravity", "four_pi_G"), "amp": get("problem/jeans_wave", "amp")}


def geometry(Lx=1.0, Ly=0.5, Lz=0.5):
    a3 = np.arctan(Lx / Ly)
    a2 = np.arctan(0.5 * (Lx * np.cos(a3) + Ly * np.sin(a3)) / Lz)
    lam = min(Lx * np.cos(a2) * np.cos(a3), Ly * np.cos(a2) * np.sin(a3), Lz * np.sin(a2))
    return a2, a3, 2 * np.pi / lam


def exact(d, t, four_pi_G, amp):
    a2, a3, k = geometry()
    w2 = k * k - four_pi_G
    w = np.sqrt(abs(w2))
    xp = np.cos(a2) * (d["x"] * np.cos(a3) + d["y"] * np.sin(a3)) + d["z"] * np.sin(a2)
    s, c = np.sin(k * xp), np.cos(k * xp)
    if w2 > 0:
        fr, fv = np.cos(w * t), -np.sin(w * t)
    else:
        fr = fv = np.exp(w * t)
    rho = 1.0 + amp * s * fr
    vpar = (w / k) * amp * c * fv
    p = (1.0 / GAM) * (1.0 + GAM * amp * s * fr)
    m = rho * vpar
    return rho, [m * np.cos(a3) * np.cos(a2), m * np.sin(a3) * np.cos(a2), m * np.sin(a2)], p, xp, k, w2


def errors(run_dir):
    f = sgv.final_dump(run_dir)
    d = sgv.load(f, ["prim"])
    P = params(f)
    rho, mom, p, *_ , w2 = exact(d, d["time"], P["four_pi_G"], P["amp"])
    r = d["prim"][0]
    m_num = [r * d["prim"][i] for i in (1, 2, 3)]
    N = r.size
    l1 = {"d": np.abs(r - rho).sum() / N, "M1": np.abs(m_num[0] - mom[0]).sum() / N,
          "M2": np.abs(m_num[1] - mom[1]).sum() / N, "M3": np.abs(m_num[2] - mom[2]).sum() / N,
          "P": np.abs(d["prim"][4] - p).sum() / N}
    l1 = {k: float(v) for k, v in l1.items()}
    out = {"run": os.path.relpath(run_dir, RUNS), "t": d["time"], "ncells": N, "nJ": float(np.sqrt(P["four_pi_G"]) / geometry()[2]),
           "omega2": float(w2), "rms_l1": float(np.sqrt(sum(v * v for v in l1.values()))), **{"L1_" + k: v for k, v in l1.items()}}
    hs = glob.glob(os.path.join(run_dir, "*.hst"))
    if hs:
        H = sgv.hst(hs[0])
        ok = H["grav-iters"] >= 0
        out.update(grav_iters_max=int(H["grav-iters"][ok].max()), grav_iters_mean=float(H["grav-iters"][ok].mean()),
                   grav_nonconv=int(H["grav-nonconv"][ok].sum()), grav_res_rel_max=float(H["grav-res-rel"][ok].max()))
    return out


def fit_dispersion(run_dir):
    """Projection a(t) = <drho sin(kx)> / <sin^2> from every dump; fit omega (stable) or g (unstable)."""
    from scipy.optimize import curve_fit
    fs = sgv.dumps(run_dir)
    P = params(fs[0])
    ts, aa = [], []
    for f in fs:
        d = sgv.load(f, ["prim"])
        a2, a3, k = geometry()
        xp = np.cos(a2) * (d["x"] * np.cos(a3) + d["y"] * np.sin(a3)) + d["z"] * np.sin(a2)
        s = np.sin(k * xp)
        ts.append(d["time"])
        aa.append(np.sum((d["prim"][0] - 1.0) * s) / np.sum(s * s))
    ts, aa = np.array(ts), np.array(aa)
    k = geometry()[2]
    w2 = k * k - P["four_pi_G"]
    if w2 > 0:
        popt, _ = curve_fit(lambda t, A, w: A * np.cos(w * t), ts, aa, p0=[P["amp"], np.sqrt(w2)])
        meas = popt[1] ** 2
    else:
        sel = aa > 0
        g = np.polyfit(ts[sel], np.log(aa[sel]), 1)[0]
        meas = -g * g
    return {"run": os.path.relpath(run_dir, RUNS), "nJ": float(np.sqrt(P["four_pi_G"]) / k),
            "omega2_exact": float(w2), "omega2_meas": float(meas), "rel_err": float((meas - w2) / abs(w2)),
            "t": ts.tolist(), "a": aa.tolist()}


def main():
    res = {"convergence": [], "dispersion": []}
    for rd in sorted(glob.glob(os.path.join(RUNS, "D12_jeans", "*"))):
        rc = sgv.run_rc(rd)
        if rc != 0:
            print(f"[not graded] {os.path.relpath(rd, RUNS)} rc={rc}")
            continue
        if sgv.final_dump(rd):
            try:
                r = errors(rd)
            except Exception as e:
                r = {"run": rd, "error": repr(e)}
            res["convergence"].append(r)
            print(json.dumps({k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in r.items()}))
    for rd in sorted(glob.glob(os.path.join(RUNS, "D3_dispersion", "*"))):
        rc = sgv.run_rc(rd)
        if rc != 0:
            print(f"[not graded] {os.path.relpath(rd, RUNS)} rc={rc}")
            continue
        if len(sgv.dumps(rd)) > 5:
            r = fit_dispersion(rd)
            res["dispersion"].append(r)
            print(f"{r['run']:28s} nJ={r['nJ']:.3f} w2 exact {r['omega2_exact']:+.5e} meas {r['omega2_meas']:+.5e} rel {r['rel_err']:+.2e}")
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "tables", "jeans.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
