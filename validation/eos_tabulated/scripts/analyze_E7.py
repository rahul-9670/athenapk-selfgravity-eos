#!/usr/bin/env python
"""E7: AthenaPK Riemann problems on the tabulated EOS vs the exact general-EOS solution of the SAME interpolant
(riemann_exact.py). Exact cell averages from a 25600-subcell sampling (8 per cell at N = 3200). Reports
  * L1 errors of rho, u, P (normalised by max|q|) and their convergence orders;
  * plateau states (median over the middle half of each star region) vs P*, u*, rho*_L, rho*_R;
  * shock position vs S t.
-> results/tables/E7_riemann.json, E7_riemann.txt; figure results/figures/E7_riemann.png"""
import glob, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E
import riemann_exact as RX

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.environ.get("E7_PREFIX", "E7e")
RUNS = os.environ.get("E7_RUNS", os.path.join(HERE, "..", "runs", PREFIX))
TAB = os.environ.get("E7_TAB", os.path.join(HERE, "..", "results", "tables"))
FIG = os.environ.get("EOS_FIG", os.path.join(HERE, "..", "results", "figures"))
NFINE = 25600
X0 = 0.5


def read(f):
    with h5py.File(f) as h:
        p = h["prim"][()].astype(float)
        t = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        nc = int(np.atleast_1d(h["Info"].attrs["NCycle"])[0])
    return t, p[0, 0, 0, 0, :], p[0, 1, 0, 0, :], p[0, 4, 0, 0, :], nc


def exact_profile(eos, rec, t):
    L, R = tuple(rec["L"]), tuple(rec["R"])
    so = RX.solve(eos, L, R)
    xs = (np.arange(NFINE) + 0.5) / NFINE
    xi = (xs - X0) / t
    out = np.zeros((3, NFINE))
    us, Ps = so["u"], so["P"]
    for side, K, sgn in (("L", L, -1), ("R", R, +1)):
        st, W = so[side], so["W" + side]
        m = (xi < us) if side == "L" else (xi >= us)
        PK = float(eos.P(K[0], K[2]))
        if st["type"] == "shock":
            behind = m & ((xi - st["S"]) * sgn < 0)
            ahead = m & ~behind
            out[:, behind] = np.array([st["rho"], us, Ps])[:, None]
            out[:, ahead] = np.array([K[0], K[1], PK])[:, None]
        else:
            ahead = m & ((xi - st["head"]) * sgn >= 0)
            star = m & ((xi - st["tail"]) * sgn <= 0)
            fan = m & ~ahead & ~star
            out[:, ahead] = np.array([K[0], K[1], PK])[:, None]
            out[:, star] = np.array([st["rho"], us, Ps])[:, None]
            if fan.any():
                ise = W.isentrope(np.log(st["rho"]))
                lnr = np.linspace(np.log(st["rho"]), np.log(K[0]), 40001)
                eI = np.array([ise(q) for q in lnr])
                rho = np.exp(lnr)
                c = np.sqrt([RX.c2(eos, r, e) for r, e in zip(rho, eI[:, 0])])
                u = K[1] + sgn * eI[:, 1]
                char = u + sgn * c
                o = np.argsort(char)
                q = np.interp(xi[fan], char[o], lnr[o])
                out[0, fan] = np.exp(q)
                out[1, fan] = np.interp(q, lnr, u)
                out[2, fan] = np.array([float(eos.P(np.exp(a), float(np.interp(a, lnr, eI[:, 0])))) for a in q])
    return so, out


def main():
    ex = json.load(open(os.environ.get("E7_EXACT", os.path.join(TAB, "E7_exact.json"))))
    t = E.load()
    res, lines = {}, []
    profiles = {}
    for key, rec in ex.items():
        prob, mode = key.split("|")
        eos = RX.TableEOS(t, mode)
        runs = sorted(glob.glob(os.path.join(RUNS, f"{PREFIX}_{prob}_{mode}_n*")), key=lambda d: int(d.rsplit("_n", 1)[1]))
        runs = [d for d in runs if glob.glob(os.path.join(d, "*.out0.final.phdf"))]
        if not runs:
            continue
        tf = read(glob.glob(os.path.join(runs[0], "*.out0.final.phdf"))[0])[0]
        so, fine = exact_profile(eos, rec, tf)
        profiles[key] = (so, fine, tf)
        rows = []
        scale = np.abs(fine).max(axis=1)
        for d in runs:
            n = int(d.rsplit("_n", 1)[1])
            tt, rho, u, P, nc = read(glob.glob(os.path.join(d, "*.out0.final.phdf"))[0])
            assert abs(tt - tf) < 1e-12 * max(1.0, tf), (d, tt, tf)
            avg = fine.reshape(3, n, NFINE // n).mean(axis=2)
            L1 = [float(np.mean(np.abs(q - a)) / s) for q, a, s in zip((rho, u, P), avg, scale)]
            x = (np.arange(n) + 0.5) / n; xi = (x - X0) / tt
            # plateaus: middle half of [left star edge, contact] and [contact, right star edge]
            def edge(side):
                st = so[side]
                return st["S"] if st["type"] == "shock" else st["tail"]
            aL, bL = edge("L"), so["u"]; aR, bR = so["u"], edge("R")
            mL = (xi > aL + 0.25 * (bL - aL)) & (xi < bL - 0.25 * (bL - aL))
            mR = (xi > aR + 0.25 * (bR - aR)) & (xi < bR - 0.25 * (bR - aR))
            pl = dict(P=float(np.median(P[mL | mR]) / so["P"] - 1), u=float(np.median(u[mL | mR]) - so["u"]) / max(abs(so["u"]), rec["vmax"]),
                      rhoL=float(np.median(rho[mL]) / so["L"]["rho"] - 1) if mL.sum() else None,
                      rhoR=float(np.median(rho[mR]) / so["R"]["rho"] - 1) if mR.sum() else None,
                      ncellsL=int(mL.sum()), ncellsR=int(mR.sum()))
            # shock positions: half-way crossing between pre- and post-shock density
            shocks = {}
            for side in ("L", "R"):
                st = so[side]
                if st["type"] != "shock":
                    continue
                K = rec[side]
                half = 0.5 * (K[0] + st["rho"])
                xs_exact = X0 + st["S"] * tt
                if side == "R":
                    idx = np.nonzero((rho[:-1] >= half) & (rho[1:] < half))[0]
                else:
                    idx = np.nonzero((rho[:-1] < half) & (rho[1:] >= half))[0]
                if len(idx):
                    i = idx[np.argmin(np.abs(x[idx] - xs_exact))]
                    xc = x[i] + (half - rho[i]) / (rho[i + 1] - rho[i]) / n
                    shocks[side] = float((xc - xs_exact) * n)  # in cells
            rows.append(dict(n=n, cycles=nc, L1_rho=L1[0], L1_u=L1[1], L1_P=L1[2], plateau=pl, shock_offset_cells=shocks))
        for i in range(1, len(rows)):
            for q in ("rho", "u", "P"):
                rows[i]["order_" + q] = float(np.log2(rows[i - 1]["L1_" + q] / rows[i]["L1_" + q]) / np.log2(rows[i]["n"] / rows[i - 1]["n"]))
        res[key] = dict(exact=dict(P=so["P"], u=so["u"], L=so["L"]["type"], R=so["R"]["type"]), runs=rows)
        lines.append(f"{prob} [{mode}]  t = {tf:.5g}  exact P* {so['P']:.6e} u* {so['u']:+.6e}  waves {so['L']['type']}/{so['R']['type']}")
        for r in rows:
            o = "" if "order_rho" not in r else f"  orders {r['order_rho']:.2f} {r['order_u']:.2f} {r['order_P']:.2f}"
            pl = r["plateau"]
            lines.append(f"   N {r['n']:5d}  L1 rho {r['L1_rho']:.3e} u {r['L1_u']:.3e} P {r['L1_P']:.3e}{o} | plateau dP* {pl['P']:+.2e} du* {pl['u']:+.2e}"
                         f" drho*L {pl['rhoL'] if pl['rhoL'] is None else format(pl['rhoL'], '+.2e')} drho*R {pl['rhoR'] if pl['rhoR'] is None else format(pl['rhoR'], '+.2e')}"
                         f" | shock offset [cells] {', '.join(f'{k} {v:+.2f}' for k, v in r['shock_offset_cells'].items())}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(TAB, f"{PREFIX}_riemann.txt"), "w").write(txt + "\n")
    json.dump(res, open(os.path.join(TAB, f"{PREFIX}_riemann.json"), "w"), indent=1)
    plot(profiles, ex)


def plot(profiles, ex):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    probs = sorted({k.split("|")[0] for k in profiles})
    fig, ax = plt.subplots(3, len(probs), figsize=(4.2 * len(probs), 8.5), squeeze=False)
    t = E.load()
    for j, prob in enumerate(probs):
        key = prob + "|p_over_rho"
        if key not in profiles:
            continue
        so, fine, tf = profiles[key]
        xs = (np.arange(NFINE) + 0.5) / NFINE
        eos = RX.TableEOS(t, "p_over_rho")
        for n, sty in ((200, "o"), (3200, "-")):
            fs = glob.glob(os.path.join(RUNS, f"{PREFIX}_{prob}_p_over_rho_n{n}", "*.out0.final.phdf"))
            if not fs:
                continue
            tt, rho, u, P, _ = read(fs[0])
            x = (np.arange(n) + 0.5) / n
            esp = np.array([eos.e_of_P(r, p) for r, p in zip(rho, P)]) if n <= 400 else None
            for i, q in enumerate((rho * E.RHO_U, u * E.V_U / 1e5)):
                ax[i, j].plot(x, q, sty, ms=2.5, lw=1.0, label=f"AthenaPK N={n}")
            if esp is not None:
                ax[2, j].plot(x, 10 ** eos.logT(rho, esp), sty, ms=2.5, label=f"AthenaPK N={n}")
            else:
                T = np.array([10 ** eos.logT(r, eos.e_of_P(r, p)) for r, p in zip(rho[::4], P[::4])])
                ax[2, j].plot(x[::4], T, sty, lw=1.0, label=f"AthenaPK N={n}")
        Tex = np.array([10 ** eos.logT(r, eos.e_of_P(r, p)) for r, p in zip(fine[0, ::16], fine[2, ::16])])
        ax[0, j].plot(xs, fine[0] * E.RHO_U, "k--", lw=0.8, label="exact")
        ax[1, j].plot(xs, fine[1] * E.V_U / 1e5, "k--", lw=0.8)
        ax[2, j].plot(xs[::16], Tex, "k--", lw=0.8)
        ax[0, j].set_title(prob.replace("_", " "), fontsize=10)
        ax[0, j].set_ylabel(r"$\rho$ [g cm$^{-3}$]"); ax[1, j].set_ylabel("u [km/s]"); ax[2, j].set_ylabel("T [K]")
        ax[2, j].set_xlabel("x")
        ax[0, j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{PREFIX}_riemann.png"), dpi=130)


if __name__ == "__main__":
    main()
