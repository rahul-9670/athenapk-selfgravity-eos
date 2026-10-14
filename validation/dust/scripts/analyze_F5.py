#!/usr/bin/env python
"""F5 (DD5 candidate dust/subl_model = semenov): one-zone compression (the D3a deck), per cell, every dump.

Exact: the gas is heated monotonically AND T - T5(rho) grows monotonically (checked here), so after every React
f = f_ref s(T, rho), s = Semenov's compact ramp about T5(rho) (dust.hpp SublSemenov, re-implemented here from
opacity_semenov.py evap_temperatures()[4]); frz: a = a_ref (f/f_ref)^(1/3).
Also reports where the box crosses T5 (vs the fixed 1500 K of production) and the T at which f/f_ref = 0.5 in both.
Writes results/tables/F5_semenov.txt (+ .json); --check.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
sys.path.insert(0, _REPO + "/src/radiation")
import eostab
import opacity_semenov as SM
DU = _VT + "/dust"
TABLE = _REPO + "/src/eos/eos_table_hires_v5_cold_rho26.bin"
F_REF, A_REF, W = 0.01, 1e-5, 100.0


def s_sem(T, rho_cgs):
    x = (T - SM.evap_temperatures(rho_cgs)[4]) / W
    s = np.where(x <= -1, 1.0, np.where(x >= 1, 1e-12, 0.5 * (1 - np.sin(0.5 * np.pi * x))))
    return np.maximum(s, 1e-12)


def main():
    res, rows = {}, ["# F5 subl_model = semenov, one-zone compression 1e-9 g/cm^3, 1250 K -> x8 (per cell, every dump)"]
    tab = None
    for name in json.load(open(os.path.join(DU, "inputs", "F5_meta.json"))):
        fs = sorted(glob.glob(os.path.join(DU, "runs", "F", name, "*.out0.0*.phdf")))
        if not fs:
            rows.append(f"{name}: MISSING"); continue
        ef, ea, xs, trace = 0.0, 0.0, [], []
        for k, fn in enumerate(fs):
            with h5py.File(fn) as h:
                names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
                pn = [n for n in names if n.startswith("prim_")]
                cn = [n for n in names if n.startswith("cons_")]
                p, c = h["prim"][()], h["cons"][()]
                ru = float(h["Params"].attrs["Hydro/phys_units/rho_unit_cgs"])
                vu = float(h["Params"].attrs["Hydro/phys_units/v_unit_cgs"])
            if tab is None:
                tab = eostab.load(TABLE, ru, vu)
            rho = p[:, pn.index("prim_density")].ravel()
            f = p[:, pn.index("prim_scalar_0")].ravel(); a = p[:, pn.index("prim_scalar_1")].ravel()
            E = c[:, cn.index("cons_total_energy_density")].ravel()
            ke = 0.5 * c[:, cn.index("cons_momentum_density_1")].ravel() ** 2 / rho
            T = np.array([eostab.T_of(tab, r, (e - q) / r) for r, e, q in zip(rho, E, ke)])
            rc = rho * ru
            x = (T - SM.evap_temperatures(rc)[4]) / W
            xs.append(float(x.mean())); trace.append((float(T.mean()), float(rc.mean()), float(f.mean() / F_REF)))
            if k > 0:
                ef = max(ef, float(np.abs(f / (F_REF * s_sem(T, rc)) - 1).max()))
                ea = max(ea, float(np.abs(a / (A_REF * np.cbrt(f / F_REF)) - 1).max()))
        mono = bool(np.all(np.diff(xs) > 0))
        r = dict(f_err_max=ef, a_err_max=ea, x_monotone=mono, trace=trace)
        res[name] = r
        rows.append(f"{name:24s} max |f/(f_ref s_sem) - 1| {ef:.2e} | max |a/(a_ref (f/f_ref)^1/3) - 1| {ea:.2e} | "
                    f"(T - T5)/w monotone: {mono}")
        for T_, r_, f_ in trace[::4]:
            rows.append(f"    T {T_:7.1f} K  rho {r_:.3e}  T5 {float(SM.evap_temperatures(r_)[4]):7.1f} K  f/f_ref {f_:.4e}"
                        f"  (fixed 1500 K model: {0.5 * (1 - np.tanh((T_ - 1500) / 100)):.4e})")
    txt = "\n".join(rows)
    print(txt)
    jp = os.path.join(DU, "results", "tables", "F5_semenov.json")
    if "--check" in sys.argv:
        ok = json.dumps(json.load(open(jp)), sort_keys=True) == json.dumps(json.loads(json.dumps(res)), sort_keys=True)
        print("CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
    open(jp.replace(".json", ".txt"), "w").write(txt + "\n")
    json.dump(res, open(jp, "w"), indent=1)


if __name__ == "__main__":
    main()
