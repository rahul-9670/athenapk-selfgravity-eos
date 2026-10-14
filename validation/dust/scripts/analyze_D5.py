#!/usr/bin/env python
"""D5 analysis: shock-recorded sublimation in planar Noh (colliding flows), live scheme.

Plateau = cells with |x - 1/2| / (S t) in 0.15 .. 0.8 (as T1/T4c). Per run:
  dev      = mean f/(f_ref s(T2)) - 1     the deficit vs the EXACT plateau value (s = tanh switch about T_subl)
  excess   = mean f/(f_ref s(T_cell)) - 1  deficit vs the cell's OWN final temperature (0 if nothing was recorded)
  T_p      = mean T_cell / T2 - 1          plateau temperature error
  T_rec    = T_subl + W atanh(1 - 2 f/f_ref): the hottest temperature the parcel was SEEN at (irreversible
             sublimation is a max-temperature recorder); dT_rec = mean (T_rec - T_cell) [K] is the transient
             post-shock overshoot each parcel carried through the shock.
Convergence: dev, excess and dT_rec vs N (256, 1024, 4096). If they do not fall with N, the deficit is a per-cell
shock transient (every parcel crosses the numerical shock once, whatever the resolution).
Writes results/tables/D5_shock_sublimation.txt (+ .json); optional --tag/--runs to analyse other run sets
(the DD2 fix runs); --check compares with the saved json.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import argparse, glob, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
import eostab
DU = _VT + "/dust"
TABLE = _REPO + "/src/eos/eos_table_hires_v5_cold_rho26.bin"
F_REF, W = 0.01, 100.0


def load(d):
    fs = glob.glob(os.path.join(DU, "runs", d, "*.out0.final.phdf"))
    if not fs:
        return None
    with h5py.File(fs[0]) as h:
        names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
        pn = [n for n in names if n.startswith("prim_")]
        cn = [n for n in names if n.startswith("cons_")]
        p, c = h["prim"][()].astype(float), h["cons"][()].astype(float)
        x = h["Locations"]["x"][()]
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        U = h["Params"].attrs
        ru, vu = float(U["Hydro/phys_units/rho_unit_cgs"]), float(U["Hydro/phys_units/v_unit_cgs"])
    xc = (0.5 * (x[:, 1:] + x[:, :-1]))[:, None, None, :]
    X = np.broadcast_to(xc, p[:, 0].shape).ravel()
    g = lambda n: p[:, pn.index(n)].ravel()
    rho = g("prim_density")
    # T React read: the dual-selected internal energy; the final dump's cons_dual (if present) or E - KE
    E = c[:, cn.index("cons_total_energy_density")].ravel()
    ke = 0.5 * c[:, cn.index("cons_momentum_density_1")].ravel() ** 2 / rho
    return dict(t=t, X=X, rho=rho, f=g("prim_scalar_0"), a=g("prim_scalar_1"), e=(E - ke) / rho, ru=ru, vu=vu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="D5_shock_sublimation")
    ap.add_argument("--meta", default=os.path.join(DU, "inputs", "D5_meta.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    tab, rows, res = None, [], {}
    rows.append(f"# {a.tag}: Noh plateau |x-1/2|/(S t) in 0.15..0.8; T_subl = T2 + dT")
    rows.append("%-28s %6s %10s %11s %11s %10s %9s %6s" % ("run", "N", "s_exact", "dev", "excess", "T_p", "dT_rec[K]",
                                                          "ncell"))
    for name, m in meta.items():
        r = load(f"{m.get('dir', 'D5')}/{name}")
        if r is None:
            rows.append(f"{name:28s} MISSING"); continue
        if tab is None:
            tab = eostab.load(TABLE, r["ru"], r["vu"])
        sel = np.abs(r["X"] - 0.5) / (m["S"] * r["t"])
        pl = (sel > 0.15) & (sel < 0.8)
        T = np.array([eostab.T_of(tab, rr, ee) for rr, ee in zip(r["rho"][pl], r["e"][pl])])
        Ts = m["T_subl"]
        s = lambda TT: 0.5 * (1 - np.tanh((TT - Ts) / W))
        f = r["f"][pl] / F_REF
        s_ex = float(s(m["T2"]))
        dev = f / s_ex - 1
        exc = f / s(T) - 1
        Trec = Ts + W * np.arctanh(np.clip(1 - 2 * f, -1 + 1e-15, 1 - 1e-15))
        out = dict(N=m["N"], M=m["M"], dT=m["dT"], s_exact=s_ex, dev=float(dev.mean()), dev_max=float(np.abs(dev).max()),
                   excess=float(exc.mean()), T_p=float(T.mean() / m["T2"] - 1), dT_rec=float((Trec - T).mean()),
                   dT_rec_max=float((Trec - T).max()), ncell=int(pl.sum()))
        res[name] = out
        rows.append("%-28s %6d %10.4f %+11.3e %+11.3e %+10.2e %9.2f %6d" % (name, m["N"], s_ex, out["dev"], out["excess"],
                                                                         out["T_p"], out["dT_rec"], out["ncell"]))
    txt = "\n".join(rows)
    print(txt)
    jp = os.path.join(DU, "results", "tables", a.tag + ".json")
    if a.check:
        same = json.dumps(json.load(open(jp)), sort_keys=True) == json.dumps(json.loads(json.dumps(res)), sort_keys=True)
        print("CHECK", "PASS" if same else "FAIL"); sys.exit(0 if same else 1)
    open(jp.replace(".json", ".txt"), "w").write(txt + "\n")
    json.dump(res, open(jp, "w"), indent=1)


if __name__ == "__main__":
    main()
