#!/usr/bin/env python
"""E8: homologous compression legs (eos_test compress + eos_homologous ghosts) vs the exact adiabat of the SAME
interpolant, de/dln rho = P(rho, e)/rho, integrated (RK4, h = 1e-4) from each leg's initial state
(esp_IC = esp_of_T(rho0, T0), exactly what the pgen sets). Centre cell (at rest, odd N). Reports, per leg:
  e/e_ad - 1 and T/T_ad - 1 at the end (100x compression) and max over the leg; the kinematics
  rho_c/(rho0/(1 - alpha t)) - 1; the box uniformity max|rho/rho_c - 1|.
-> results/tables/E8_compress.{json,txt}; figure results/figures/E8_adiabat.png"""
import glob, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E
import riemann_exact as RX

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs", "E8")
TAB = os.path.join(HERE, "..", "results", "tables")
FIG = os.path.join(HERE, "..", "results", "figures")


def adiabat(eos, r0, e0, r1, h=1e-4):
    lnr, e = np.log(r0), e0
    L, Ecol = [lnr], [e]
    f = lambda x, y: float(eos.P(np.exp(x), y)) / np.exp(x)
    while lnr < np.log(r1) - 1e-14:
        hh = min(h, np.log(r1) - lnr)
        k1 = f(lnr, e); k2 = f(lnr + hh / 2, e + hh / 2 * k1); k3 = f(lnr + hh / 2, e + hh / 2 * k2); k4 = f(lnr + hh, e + hh * k3)
        e += hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4); lnr += hh
        L.append(lnr); Ecol.append(e)
    return np.array(L), np.array(Ecol)


def main():
    t = E.load()
    meta = json.load(open(os.path.join(HERE, "..", "inputs", "E8_legs.json")))
    eoss = {m: RX.TableEOS(t, m) for m in ("p_over_rho", "p")}
    res, lines, curves = {}, [], {}
    for name, m in meta.items():
        d = os.path.join(RUNS, name)
        fs = sorted(glob.glob(os.path.join(d, "*.out0.0*.phdf"))) + glob.glob(os.path.join(d, "*.out0.final.phdf"))
        if not glob.glob(os.path.join(d, "*.out0.final.phdf")):
            continue
        eos = eoss[m["mode"]]
        r0 = m["rho0"]
        eIC = float(eos.esp_of_T(r0, m["T0"]))
        Lr, Ee = adiabat(eos, r0, eIC, r0 * 101.0)
        rows = []
        for f in fs:
            with h5py.File(f) as h:
                p = h["prim"][()].astype(float)
                tt = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
            rho = p[0, 0, 0, 0, :]; P = p[0, 4, 0, 0, :]; v = p[0, 1, 0, 0, :]
            c = rho.size // 2
            e = eos.e_of_P(rho[c], P[c])
            e_ad = float(np.interp(np.log(rho[c]), Lr, Ee))
            T = 10 ** float(eos.logT(rho[c], e)); T_ad = 10 ** float(eos.logT(rho[c], e_ad))
            rows.append(dict(t=tt, rho_c=float(rho[c]), comp=float(rho[c] / r0), de=e / e_ad - 1, dT=T / T_ad - 1, T=T,
                             kin=float(rho[c] * (1 - m["alpha"] * tt) / r0 - 1), unif=float(np.abs(rho / rho[c] - 1).max()),
                             vc=float(v[c])))
        fin = rows[-1]
        res[name] = dict(meta=m, rows=rows, end=dict(comp=fin["comp"], de=fin["de"], dT=fin["dT"], kin=fin["kin"], unif=fin["unif"]),
                         max_abs_de=float(max(abs(r["de"]) for r in rows)))
        curves[name] = rows
        lines.append(f"{name:18s} [{m['mode']:10s} cfl {m['cfl']:.2f} alpha {m['alpha']:.4g}] rho {r0 * E.RHO_U:.3g} -> x{fin['comp']:.2f}:"
                     f" T {m['T0']:8.2f} -> {fin['T']:8.2f} K | end e/e_ad-1 {fin['de']:+.3e} T/T_ad-1 {fin['dT']:+.3e}"
                     f" | max|de| {res[name]['max_abs_de']:.3e} | kinematics {fin['kin']:+.2e} | uniformity {fin['unif']:.1e} | v_c {fin['vc']:.1e}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(TAB, "E8_compress.txt"), "w").write(txt + "\n")
    json.dump(res, open(os.path.join(TAB, "E8_compress.json"), "w"), indent=1)
    plot(t, curves, meta)


def plot(t, curves, meta):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    eos = RX.TableEOS(t, "p_over_rho")
    r0 = 1e-13 / E.RHO_U
    Lr, Ee = adiabat(eos, r0, float(eos.esp_of_T(r0, 10.0)), 1e-1 / E.RHO_U, h=2e-3)
    rho = np.exp(Lr)
    T = 10 ** eos.logT(rho, Ee)
    Pp = eos.P(rho, Ee)
    gam = np.gradient(np.log(Pp), Lr)   # Gamma_1 along the adiabat = dlnP/dlnrho|_s
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(rho * E.RHO_U, T, "k-", lw=1, label="table adiabat from (1e-13 g/cc, 10 K)")
    for name, rows in curves.items():
        if name.count("_") != 1:
            continue
        ax[0].loglog([r["rho_c"] * E.RHO_U for r in rows], [r["T"] for r in rows], "o", ms=2.5, label=name)
    ax[0].set_xlabel(r"$\rho$ [g cm$^{-3}$]"); ax[0].set_ylabel("T [K]"); ax[0].legend(fontsize=7)
    ax[1].semilogx(rho * E.RHO_U, gam, "k-", lw=1)
    for g in (5 / 3, 7 / 5, 1.1):
        ax[1].axhline(g, color="0.7", lw=0.6, ls="--")
    ax[1].set_xlabel(r"$\rho$ [g cm$^{-3}$]"); ax[1].set_ylabel(r"$\Gamma_1 = d\ln P/d\ln\rho|_s$ (interpolant)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E8_adiabat.png"), dpi=130)
    np.savetxt(os.path.join(TAB, "E8_adiabat_path.txt"), np.c_[rho * E.RHO_U, T, gam][::25],
               header="table adiabat from (1e-13 g/cc, 10 K), eos_interp = p_over_rho: rho [g/cc]  T [K]  Gamma1", fmt="%.6e")


if __name__ == "__main__":
    main()
