"""D2 analysis (magnetically dominated cold gas; HL code units, beta = 2P/B^2).

D2a  1D sinusoid + uniform field: dP = P/P_ad(rho) - 1 (table adiabat through rho 1, 10 K), split switched / E-branch
     cells; perp also flux freezing max|(B_y/rho)/(B0/rho0) - 1|.
D2c  1D contact + uniform field: max|P/P0 - 1| at the end (one crossing).
D2b  2D div B blob: dP vs the adiabat in every cell (max, L1, cells > 1e-2) at each dump; the blob's excess magnetic
     energy (initial) vs the volume integral of rho (e - e_ad(rho)) = spurious heat; relDivB from the hst.
-> results/tables/D2{a,b,c}_*.{txt,json}"""
import glob
import json
import os
import sys

import h5py
import numpy as np

from analyze_D1 import adiabat_table, hst, hcount
from de_common import INP, RUNS, TAB, table_eos

eos = table_eos()
meta = json.load(open(os.path.join(INP, "D2_meta.json")))
L, Ee = adiabat_table(1.0, meta["e0"], 0.05, 50.0)
e_ad = lambda rho: np.interp(np.log(rho), L, Ee)
Pad = lambda rho: eos.P(rho, e_ad(rho))


def dumps(d):
    return sorted(glob.glob(os.path.join(d, "*.out0.0*.phdf"))) + glob.glob(os.path.join(d, "*.out0.final.phdf"))


def cells(f):
    """All interior cells flattened: t, dict(rho, P, Bx, By, Bz, sw, x, y), cell volume."""
    with h5py.File(f) as h:
        p = h["prim"][()].astype(float)
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        sw = h["dual_switched"][()].astype(float) if "dual_switched" in h else None
        x = h["Locations"]["x"][()]; y = h["Locations"]["y"][()]
    nb = p.shape[0]
    xc = 0.5 * (x[:, 1:] + x[:, :-1]); yc = 0.5 * (y[:, 1:] + y[:, :-1])
    X = np.broadcast_to(xc[:, None, None, :], (nb,) + p.shape[2:]).ravel()
    Y = np.broadcast_to(yc[:, None, :, None], (nb,) + p.shape[2:]).ravel()
    out = dict(rho=p[:, 0].ravel(), P=p[:, 4].ravel(), Bx=p[:, 5].ravel(), By=p[:, 6].ravel(), Bz=p[:, 7].ravel(),
               x=X, y=Y, sw=sw.reshape(nb, -1).ravel() if sw is not None else np.zeros(p[:, 0].size))
    dV = (x[0, 1] - x[0, 0]) * (y[0, 1] - y[0, 0]) if p.shape[3] > 1 else (x[0, 1] - x[0, 0])
    return t, out, dV


def d2a():
    rows, res = [], {}
    for name, m in meta["runs"].items():
        if m["test"] != "D2a":
            continue
        for tag in ("on", "off"):
            d = os.path.join(RUNS, "D2a", f"{name}_{tag}")
            fs = dumps(d)
            if not glob.glob(os.path.join(d, "*.final.phdf")):
                rows.append(f"{name:24s} {tag:3s} MISSING"); continue
            per = []
            for f in fs[1:]:
                t, c, _ = cells(f)
                dP = c["P"] / Pad(c["rho"]) - 1
                sw = c["sw"] > 0.5
                ff = float(np.abs((c["By"] / c["rho"]) / m["B0"] - 1).max()) if m["geo"] == "perp" else float("nan")
                per.append(dict(t_tc=t / m["tc"], max=float(np.abs(dP).max()), L1=float(np.abs(dP).mean()),
                                nsw=int(sw.sum()), maxsw=float(np.abs(dP[sw]).max()) if sw.any() else 0.0,
                                maxE=float(np.abs(dP[~sw]).max()) if (~sw).any() else 0.0, ff=ff,
                                rhomax=float(c["rho"].max())))
            h = hst(d)
            res[f"{name}_{tag}"] = dict(meta=m, per=per, nTfloor=hcount(h, "cons-nTfloor"), npfloor=hcount(h, "cons-npfloor"))
            f_ = per[-1]
            rows.append(f"{name:24s} {tag:3s} t/tc {f_['t_tc']:.2f}: max|dP| {f_['max']:.3e} L1 {f_['L1']:.3e} | switched {f_['nsw']}"
                        f" (max {f_['maxsw']:.2e}) E-branch max {f_['maxE']:.2e} | flux-freeze {f_['ff']:.2e} | rho_max {f_['rhomax']:.3f}"
                        f" | nTfloor {res[f'{name}_{tag}']['nTfloor']:.0f} npfloor {res[f'{name}_{tag}']['npfloor']:.0f}")
    txt = "\n".join(rows); print(txt)
    open(os.path.join(TAB, "D2a_lowbeta_sinus.txt"), "w").write(
        "# D2a cold sinusoid M 3 + uniform field (par = Bx, perp = By), dP = P/P_ad(rho) - 1 at the last dump\n" + txt + "\n")
    json.dump(res, open(os.path.join(TAB, "D2a_lowbeta_sinus.json"), "w"), indent=1)


def d2c():
    rows, res = [], {}
    for name, m in meta["runs"].items():
        if m["test"] != "D2c":
            continue
        for tag in ("on", "off"):
            d = os.path.join(RUNS, "D2c", f"{name}_{tag}")
            fs = dumps(d)
            if not glob.glob(os.path.join(d, "*.final.phdf")):
                rows.append(f"{name:18s} {tag:3s} MISSING"); continue
            _, c0, _ = cells(fs[0]); _, c1, _ = cells(fs[-1])
            r = dict(maxdP=float(np.abs(c1["P"] / m["P0"] - 1).max()), L1rho=float(np.abs(c1["rho"] - c0["rho"]).mean()))
            res[f"{name}_{tag}"] = r
            rows.append(f"{name:18s} {tag:3s} max|P/P0-1| {r['maxdP']:.3e}  L1(rho) {r['L1rho']:.3e}")
    txt = "\n".join(rows); print(txt)
    open(os.path.join(TAB, "D2c_lowbeta_contact.txt"), "w").write(
        "# D2c contact rho 1|2 at P0, u = 10 cs0, uniform field par/perp, N 256, one crossing\n" + txt + "\n")
    json.dump(res, open(os.path.join(TAB, "D2c_lowbeta_contact.json"), "w"), indent=1)


def d2b():
    rows, res = [], {}
    for name, m in meta["runs"].items():
        if m["test"] != "D2b":
            continue
        for tag in ("on", "off"):
            d = os.path.join(RUNS, "D2b", f"{name}_{tag}")
            fs = dumps(d)
            # beta 1e-4 static runs timed out twice (4 h at 16 ranks): analyse up to the last dump, flagged PARTIAL --
            # the known answer (every cell on the adiabat) holds at every time.
            partial = not glob.glob(os.path.join(d, "*.final.phdf"))
            if len(fs) < 2:
                rows.append(f"{name:28s} {tag:3s} MISSING"); continue
            t0, c0, dV = cells(fs[0])
            dB2 = 0.5 * (c0["Bx"] ** 2 + c0["By"] ** 2 + c0["Bz"] ** 2) - 0.5 * m["B0z"] ** 2
            EBexc = float(dB2.sum() * dV)               # blob's excess magnetic energy at t = 0
            per = []
            for f in fs[1:]:
                t, c, _ = cells(f)
                dP = c["P"] / Pad(c["rho"]) - 1
                heat = float((c["rho"] * (eos_e(c["rho"], c["P"]) - e_ad(c["rho"]))).sum() * dV)
                per.append(dict(t=t, max=float(np.abs(dP).max()), L1=float(np.abs(dP).mean()),
                                n1e2=int((np.abs(dP) > 1e-2).sum()), heat_over_EB=heat / EBexc,
                                nsw=int((c["sw"] > 0.5).sum())))
            h = hst(d)
            dvb = h.get("relDivB", np.array([np.nan]))
            res[f"{name}_{tag}"] = dict(meta=m, per=per, EBexc=EBexc, relDivB0=float(dvb[0]), relDivBend=float(dvb[-1]),
                                        nTfloor=hcount(h, "cons-nTfloor"), npfloor=hcount(h, "cons-npfloor"))
            seq = " ".join(f"{p['max']:.1e}/{p['heat_over_EB']:+.1e}" for p in per)
            rows.append(f"{name:28s} {tag:3s}{' PARTIAL to t/tlim ' + format(per[-1]['t'] / m['tlim'], '.2f') if partial else ''} [max|dP| / heat/E_B,exc per dump] {seq} | cells>1e-2 {per[-1]['n1e2']} switched {per[-1]['nsw']}"
                        f" | relDivB {dvb[0]:.1e}->{dvb[-1]:.1e} | nTfloor {res[f'{name}_{tag}']['nTfloor']:.0f} npfloor {res[f'{name}_{tag}']['npfloor']:.0f}")
    txt = "\n".join(rows); print(txt)
    open(os.path.join(TAB, "D2b_divb_blob.txt"), "w").write(
        "# D2b div B blob (2D, dedner_extended), dP = P/P_ad(rho) - 1; heat = sum rho (e - e_ad(rho)) dV / blob excess magnetic energy\n"
        + txt + "\n")
    json.dump(res, open(os.path.join(TAB, "D2b_divb_blob.json"), "w"), indent=1)


def eos_e(rho, P):
    """e(rho, P) for many cells: Newton on the interpolant from the adiabat value (smooth, monotone in e)."""
    e = e_ad(rho).copy()
    for _ in range(40):
        f = eos.P(rho, e) - P
        de = np.maximum(1e-6 * e, 1e-14)
        fp = (eos.P(rho, e + de) - eos.P(rho, e - de)) / (2 * de)
        e = np.maximum(e - f / fp, 1e-6 * e)
    return e


if __name__ == "__main__":
    w = sys.argv[1:] or ["a", "b", "c"]
    if "a" in w: d2a()
    if "c" in w: d2c()
    if "b" in w: d2b()
