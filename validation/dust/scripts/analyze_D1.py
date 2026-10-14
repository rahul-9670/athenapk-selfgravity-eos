#!/usr/bin/env python
"""D1 analysis: a_c (and f_dg) of every D1 run vs the ODE of the same law at the cell's own (rho, T).

Reference: scipy solve_ivp (DOP853, rtol 1e-12, atol 0) of da/dt = C (f rho/rho_gr) v_rel(rho, T, a) over
tlim * t_unit [s], from (f0, a0) = (f_ref s, a_ref s^1/3) with s = subl_factor(T) (= the code's first-call
sublimation; T is constant, so later calls do nothing). Units from the run's own Params (Hydro/phys_units/*).
T: the EOS table's logT bilinear at (rho, e/rho) of the final cons state (static gas: E = rho e, B = 0).
Also: Brownian closed form a^{5/2} = a0^{5/2} + (5/2) K t (meaningful where v_B >> v_t), and the sub-cycler
order from the growth_cap ladder (sub / sub05 / sub025).
Gate (pre-registered in PLAN.md): |a/a_ODE - 1| <= 1e-3 for every run; f_dg exact to 1e-12.
Writes results/tables/D1_growth.txt and .json.  --check compares with the saved json.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import h5py
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
import eostab
DU = _VT + "/dust"
TABLE = _REPO + "/src/eos/eos_table_hires_v5_cold_rho26.bin"
kB, mH, G = 1.380649e-16, 1.6726219e-24, 6.674e-8
RHO_GR, MU, ALPHA, A_REF, F_REF, TSUB, W = 3.0, 2.33, 1e-2, 1e-5, 0.01, 1500.0, 100.0


def v_rel(rho, T, a):
    cs = np.sqrt(kB * T / (MU * mH)); vth = np.sqrt(8 / np.pi) * cs
    m = 4 / 3 * np.pi * a ** 3 * RHO_GR
    vB = np.sqrt(8 * kB * T / (np.pi * 0.5 * m))
    vt = np.sqrt(ALPHA) * cs * RHO_GR * a / (rho * vth) * np.sqrt(4 * np.pi * G * rho)
    return np.sqrt(vB ** 2 + vt ** 2), vB, vt


def load(run):
    fs = sorted(glob.glob(os.path.join(DU, "runs", run, "*.out0.final.phdf")))
    if not fs:
        return None
    with h5py.File(fs[0]) as h:
        names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
        pn = [n for n in names if n.startswith("prim_")]
        cn = [n for n in names if n.startswith("cons_")]
        p, c = h["prim"][()], h["cons"][()]
        P = h["Params"].attrs
        U = dict(rho=float(P["Hydro/phys_units/rho_unit_cgs"]), t=float(P["Hydro/phys_units/time_unit_cgs"]),
                 v=float(P["Hydro/phys_units/v_unit_cgs"]))
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        ncyc = int(np.atleast_1d(h["Info"].attrs["NCycle"])[0])
    rho = p[:, pn.index("prim_density")].ravel()
    f = p[:, pn.index("prim_scalar_0")].ravel()
    a = p[:, pn.index("prim_scalar_1")].ravel()
    E = c[:, cn.index("cons_total_energy_density")].ravel()
    return dict(rho=rho, f=f, a=a, esp=E / rho, U=U, t=t, ncyc=ncyc)


def main():
    meta = json.load(open(os.path.join(DU, "inputs", "D1_meta.json")))
    out, rows = {}, []
    rows.append("# D1 0-D growth in the code vs ODE (rtol 1e-12). err = a_code/a_ODE - 1 (cell mean; spread = max-min)")
    rows.append("%-24s %8s %6s %5s %10s %10s %11s %9s %11s %9s %7s" % (
        "run", "rho", "T[K]", "ncyc", "a0/a_ref", "a/a_ref", "err", "spread", "err_closed", "vB/vt", "f_err"))
    tab = None
    for name, m in meta.items():
        r = load(f"D1/{name}")
        if r is None:
            rows.append(f"{name:24s} MISSING"); continue
        if tab is None or tab[1] != r["U"]["rho"]:
            tab = (eostab.load(TABLE, r["U"]["rho"], r["U"]["v"]), r["U"]["rho"])
        T = float(eostab.T_of(tab[0], float(r["rho"][0]), float(r["esp"][0])))
        rho_cgs = float(r["rho"][0]) * r["U"]["rho"]
        C = float(m["over"].get("growth_prefactor", 0.25))
        s = 0.5 * (1 - np.tanh((T - TSUB) / W))
        f0 = F_REF * s
        a0 = A_REF * np.cbrt(s) if f0 < F_REF * (1 - 1e-12) else A_REF
        ts = r["t"] * r["U"]["t"]
        sol = solve_ivp(lambda t, y: [C * f0 * rho_cgs / RHO_GR * v_rel(rho_cgs, T, y[0])[0]], (0, ts), [a0],
                        method="DOP853", rtol=1e-12, atol=0.0)
        a_ode = sol.y[0, -1]
        _, vB, vt = v_rel(rho_cgs, T, a_ode)
        K = C * f0 * rho_cgs / RHO_GR * np.sqrt(12 * kB * T / (np.pi ** 2 * RHO_GR))
        a_cf = (a0 ** 2.5 + 2.5 * K * ts) ** 0.4
        err = r["a"] / a_ode - 1
        ferr = float(np.max(np.abs(r["f"] / f0 - 1)))
        out[name] = dict(T=T, rho_cgs=rho_cgs, ncyc=r["ncyc"], a0=a0, a_ode=a_ode, err=float(err.mean()),
                         spread=float(err.max() - err.min()), err_closed=float(a_ode / a_cf - 1),
                         vB_over_vt=float(vB / vt), f_err=ferr, dt_over_tg=m["dt_over_tg"])
        rows.append("%-24s %8.1e %6.1f %5d %10.6f %10.6f %+11.3e %9.1e %+11.3e %9.1f %7.0e" % (
            name, rho_cgs, T, r["ncyc"], a0 / A_REF, r["a"].mean() / A_REF, err.mean(), err.max() - err.min(),
            a_ode / a_cf - 1, vB / vt, ferr))
    # sub-cycler order: err(cap) at dt = 2 t_grow
    rows.append("# sub-cycler order p from err(sub, cap 0.1) / err(sub05, 0.05) / err(sub025, 0.025): p = log2 ratio")
    for st in sorted({k.rsplit("_", 1)[0] for k in out}):
        e = [out.get(f"{st}_{v}", {}).get("err") for v in ("sub", "sub05", "sub025")]
        if None not in e and e[1] != 0 and e[2] != 0:
            rows.append(f"{st:16s} err {e[0]:+.3e} {e[1]:+.3e} {e[2]:+.3e}  p = {np.log2(abs(e[0] / e[1])):.2f}, "
                        f"{np.log2(abs(e[1] / e[2])):.2f}")
    worst = max(abs(v["err"]) for v in out.values()) if out else np.nan
    fworst = max(v["f_err"] for v in out.values()) if out else np.nan
    gate = worst <= 1e-3 and fworst <= 1e-12 and len(out) == len(meta)
    rows.append(f"GATE |a/a_ODE - 1| max {worst:.3e} (<= 1e-3), f_dg max err {fworst:.1e} (<= 1e-12), "
                f"{len(out)}/{len(meta)} runs: {'PASS' if gate else 'FAIL'}")
    txt = "\n".join(rows)
    print(txt)
    jpath = os.path.join(DU, "results", "tables", "D1_growth.json")
    if "--check" in sys.argv:
        ref = json.load(open(jpath))
        same = json.dumps(ref, sort_keys=True) == json.dumps(json.loads(json.dumps(out)), sort_keys=True)
        print("CHECK", "PASS" if same else "FAIL"); sys.exit(0 if same else 1)
    open(os.path.join(DU, "results", "tables", "D1_growth.txt"), "w").write(txt + "\n")
    json.dump(out, open(jpath, "w"), indent=1)


if __name__ == "__main__":
    main()
