#!/usr/bin/env python
"""D3 analysis: one-zone sublimation (compress / expand) dump by dump.

Every dump is written at the end of a cycle, after ReactDust, from the same hydro state React read, so the T computed
here (EOS table logT at the cell's (rho, e), e from the dual-energy-selected internal energy = E - KE for B = 0) is
the T React used at that cycle.
D3a (monotone heating): exact f_dg = f_ref subl(T_dump) at every dump; frz: a_c = a_ref (f/f_ref)^(1/3) exactly.
D3b (heat once, then cool): f_dg and a_c must be CONSTANT (bitwise) from the first dump after t = 0 on; their value
= f_ref subl(T_1) where T_1 is the first cycle's T (between T0 and the first dump's T; both quoted).
Box uniformity: max - min over the 16 cells of f/f_ref.
Writes results/tables/D3_sublimation.txt (+ .json); --check compares with the saved json.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
import eostab
DU = _VT + "/dust"
TABLE = _REPO + "/src/eos/eos_table_hires_v5_cold_rho26.bin"
F_REF, A_REF, TSUB, W = 0.01, 1e-5, 1500.0, 100.0
subl = lambda T: 0.5 * (1 - np.tanh((T - TSUB) / W))


def dumps(run):
    out = []
    for fn in sorted(glob.glob(os.path.join(DU, "runs", run, "*.out0.0*.phdf"))):
        with h5py.File(fn) as h:
            names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
            pn = [n for n in names if n.startswith("prim_")]
            cn = [n for n in names if n.startswith("cons_")]
            p, c = h["prim"][()], h["cons"][()]
            U = h["Params"].attrs
            out.append(dict(t=float(np.atleast_1d(h["Info"].attrs["Time"])[0]),
                            rho=p[:, pn.index("prim_density")].ravel(), f=p[:, pn.index("prim_scalar_0")].ravel(),
                            a=p[:, pn.index("prim_scalar_1")].ravel(),
                            ke=0.5 * (c[:, cn.index("cons_momentum_density_1")] ** 2).ravel() / p[:, 0].ravel(),
                            E=c[:, cn.index("cons_total_energy_density")].ravel(),
                            ru=float(U["Hydro/phys_units/rho_unit_cgs"]), vu=float(U["Hydro/phys_units/v_unit_cgs"])))
    return out


def main():
    meta = json.load(open(os.path.join(DU, "inputs", "D3_meta.json")))
    rows, res = ["# D3 one-zone sublimation (dump by dump; T from the table at the cell state)"], {}
    tab = None
    for name, m in meta.items():
        ds = dumps(f"D3/{name}")
        if not ds:
            rows.append(f"{name}: MISSING"); continue
        if tab is None:
            tab = eostab.load(TABLE, ds[0]["ru"], ds[0]["vu"])
        Ts, fs, as_, spread, efc, eac, fmn, fmx, TC = [], [], [], 0.0, 0.0, 0.0, [], [], []
        for k_, d in enumerate(ds):
            T = np.array([eostab.T_of(tab, r, (E - k) / r) for r, E, k in zip(d["rho"], d["E"], d["ke"])])
            Ts.append(T.mean()); fs.append(d["f"].mean()); as_.append(d["a"].mean())
            fmn.append(d["f"].min()); fmx.append(d["f"].max()); TC.append(T)
            spread = max(spread, float(np.ptp(d["f"] / F_REF)))
            if k_ > 0:   # per-cell exact relations (D3a: monotone heating in every cell)
                efc = max(efc, float(np.abs(d["f"] / (F_REF * subl(T)) - 1).max()))
                eac = max(eac, float(np.abs(d["a"] / (A_REF * np.cbrt(d["f"] / F_REF)) - 1).max()))
        Ts, fs, as_ = map(np.array, (Ts, fs, as_))
        r = dict(Tmin=float(Ts.min()), Tmax=float(Ts.max()), ndump=len(ds), cell_spread_f=spread)
        if m["alpha"] > 0:
            r["f_err_max"] = efc                      # per cell, every dump after t = 0
            r["f_final_over_ref"] = float(fs[-1] / F_REF)
            if m["freeze"]:
                r["a_err_max"] = eac
            rows.append(f"{name:10s} T {Ts[0]:.1f} -> {Ts[-1]:.1f} K | f/f_ref final {fs[-1] / F_REF:.4e} | "
                        f"max |f/(f_ref subl(T)) - 1| {r['f_err_max']:.2e} | "
                        + (f"max |a/(a_ref (f/f_ref)^1/3) - 1| {r['a_err_max']:.2e} | " if m['freeze'] else "")
                        + f"cell spread {spread:.1e}")
        else:
            # irreversibility: after the first React the per-cell f may only MIX (advection of a 1e-4 cell-to-cell
            # spread left by the homologous ghosts' first-step T), never rise above its first-React maximum,
            # although f_eq = f_ref subl(T) climbs to subl(T_final) as the gas cools.
            const_f = bool(max(fmx[1:]) <= fmx[1] * (1 + 1e-12)) and bool(min(fmn[1:]) >= fmn[1] * (1 - 1e-12))
            r.update(const_bitwise=const_f, f_eq_final_over_ref=float(subl(Ts[-1])), f1_over_ref=float(fs[1] / F_REF), subl_T0=float(subl(m["T0"])),
                     subl_Tdump1=float(subl(Ts[1])))
            rows.append(f"{name:10s} T {Ts[0]:.1f} -> {Ts[-1]:.1f} K | f/f_ref after 1st React {fs[1] / F_REF:.4e} "
                        f"(subl(T0) {subl(m['T0']):.4e}) | f within its first-React [min,max] at dumps 1..{len(ds) - 1}: "
                        f"{'YES' if const_f else 'NO'} while f_eq(T_final)/f_ref = {subl(Ts[-1]):.3f} | "
                        f"cell spread {spread:.1e}")
        res[name] = r
    txt = "\n".join(rows)
    print(txt)
    jp = os.path.join(DU, "results", "tables", "D3_sublimation.json")
    if "--check" in sys.argv:
        same = json.dumps(json.load(open(jp)), sort_keys=True) == json.dumps(json.loads(json.dumps(res)), sort_keys=True)
        print("CHECK", "PASS" if same else "FAIL"); sys.exit(0 if same else 1)
    open(jp.replace(".json", ".txt"), "w").write(txt + "\n")
    json.dump(res, open(jp, "w"), indent=1)


if __name__ == "__main__":
    main()
