#!/usr/bin/env python
"""Analysis of the literature tests L1-L10 (see ../README.md). Writes results/tables/lit.json.

  python analyze_lit.py [L1 L2 ...]     (default: all whose runs exist)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(__file__))
import sgv  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "inputs", "tables")
META = json.load(open(os.path.join(TAB, "tables_meta.json")))
REF = os.path.join(OUT, "ref1d")


def ok_runs(pattern):
    return [rd for rd in sorted(glob.glob(os.path.join(RUNS, pattern))) if sgv.dumps(rd)]


def radius(d, c=(0, 0, 0)):
    return np.sqrt((d["x"] - c[0]) ** 2 + (d["y"] - c[1]) ** 2 + (d["z"] - c[2]) ** 2)


def series(rd, fn, var=("prim",)):
    out = []
    for f in sgv.dumps(rd):
        d = sgv.load(f, list(var))
        out.append(fn(d))
    return out


# ------------------------------------------------------------------ L1
def L1():
    m = META["L1_polytrope_atm"]
    g = m["gamma"]
    res = {}
    for rd in ok_runs("L1_polytrope_atm/*"):
        def fn(d):
            r = radius(d)
            v = np.sqrt(d["prim"][1] ** 2 + d["prim"][2] ** 2 + d["prim"][3] ** 2)
            body, atm = r < 0.9, (r > 1.1) & (r < 3.0)
            eth = np.sum(d["prim"][4][r < 1.0] * d["vol"][r < 1.0]) / (g - 1)
            return {"t": d["time"], "rho_c": float(d["prim"][0][np.argmin(r)]), "vmax_body": float(v[body].max()),
                    "vmax_atm": float(v[atm].max()), "vrms_atm": sgv.rms(v[atm], d["vol"][atm]), "Eth_body": float(eth)}
        s = series(rd, fn)
        res[os.path.basename(rd)] = s
        print(f"L1 {os.path.basename(rd):12s} t/tau={s[-1]['t'] / m['tau']:.1f} rho_c/rho_c0={s[-1]['rho_c'] / s[0]['rho_c']:.4f} "
              f"max vbody={max(x['vmax_body'] for x in s):.3e} max vatm={max(x['vmax_atm'] for x in s):.3e} "
              f"Eth_body/Eth0-1={s[-1]['Eth_body'] / s[0]['Eth_body'] - 1:+.3e}  (v_esc = 1.414)")
    return res


# ------------------------------------------------------------------ L2
def L2():
    tab = np.loadtxt(os.path.join(TAB, "L2_spitzer.txt"))
    res = []
    for rd in ok_runs("L2_spitzer/*"):
        if sgv.run_rc(rd) != 0:
            continue
        d = sgv.load(sgv.final_dump(rd), ["prim"])
        x = d["x"]
        xs = np.unique(np.round(x, 12))
        dx = float(d["dx"][0])
        rho1d = np.array([d["prim"][0][np.isclose(x, xx)].mean() for xx in xs])
        # exact cell average (the sheet returns to its initial position after t = 4 = one period)
        sub = (np.arange(64) + 0.5) / 64 - 0.5
        ex = np.array([np.interp(np.mod(xx + dx * sub, 4.0), tab[:, 0], tab[:, 1]).mean() for xx in xs])
        l1 = float(np.sum(np.abs(rho1d - ex)) * dx)
        H = sgv.hst(glob.glob(os.path.join(rd, "*.hst"))[0])
        p0 = H["1-mom"][0]
        res.append({"n": len(xs), "t": d["time"], "L1": l1, "px_rel_drift": float(np.max(np.abs(H["1-mom"] - p0)) / abs(p0))})
        print(f"L2 n={len(xs):4d} t={d['time']:.3f} L1={l1:.4e} max|dP_x|/P_x={res[-1]['px_rel_drift']:.2e}")
    res.sort(key=lambda q: q["n"])
    if len(res) > 1:
        print("   orders:", ["%.2f" % o for o in sgv.order([q["L1"] for q in res], [q["n"] for q in res])])
    return res


# ------------------------------------------------------------------ L3a
def L3a():
    res = {}
    for rd in ok_runs("L3_n5_polytrope/*"):
        def fn(d):
            r = radius(d)
            v = np.sqrt(d["prim"][1] ** 2 + d["prim"][2] ** 2 + d["prim"][3] ** 2)
            ex = (1 + r ** 2 / 3) ** -2.5
            sel = r < 3
            return {"t": d["time"], "rho_c": float(d["prim"][0][np.argmin(r)]), "vmax": float(v[sel].max()),
                    "L1_rho": float(np.sum(np.abs(d["prim"][0][sel] - ex[sel]) * d["vol"][sel]) / np.sum(ex[sel] * d["vol"][sel]))}
        s = series(rd, fn)
        res[os.path.basename(rd)] = s
        print(f"L3a {os.path.basename(rd):6s} t={s[-1]['t']:.1f} rho_c={s[-1]['rho_c']:.5f} max v(r<3)={max(x['vmax'] for x in s):.3e} "
              f"L1(rho, r<3, final)={s[-1]['L1_rho']:.3e}  (c_s,c = {np.sqrt(5 / 3 / 6):.3f})")
    return res


# ------------------------------------------------------------------ L3b / L5 / L6 / L7: vs 1D reference
def ref(name):
    p = os.path.join(REF, name + ".npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def rho_c_series(rd):
    return np.array(series(rd, lambda d: (d["time"], float(d["prim"][0][np.argmin(radius(d))]))))


def compare_rho_c(label, rd, refname, times):
    s = rho_c_series(rd)
    R = ref(refname)
    out = {"t": s[:, 0].tolist(), "rho_c": s[:, 1].tolist()}
    if R is not None:
        rc_ref = np.interp(s[:, 0], R["t"], R["rho_c"])
        out["rho_c_ref"] = rc_ref.tolist()
        msg = " ".join(f"t={t:.2f}: {100 * (np.interp(t, s[:, 0], s[:, 1]) / np.interp(t, R['t'], R['rho_c']) - 1):+.2f}%"
                       for t in times if t <= s[-1, 0])
        print(f"{label} {os.path.basename(rd)}: rho_c(3D)/rho_c(1D) - 1: {msg}")
    return out


def L3b():
    return {os.path.basename(rd): compare_rho_c("L3b", rd, "L3b_N2000", [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
            for rd in ok_runs("L3_n5_collapse/*")}


def L5():
    out = {}
    R = ref("L5_N2000")
    for rd in ok_runs("L5_evrard/*"):
        H = sgv.hst(glob.glob(os.path.join(rd, "*.hst"))[0])
        i0 = int(np.argmax(H["cons-W"] != 0))
        t, ek, eth, w = H["time"][i0:], H["KE"][i0:], (H["tot-E"] - H["KE"])[i0:], H["cons-W"][i0:]
        etot = ek + eth + w
        rec = {"t": t.tolist(), "Ek": ek.tolist(), "Eth": eth.tolist(), "W": w.tolist(),
               "Etot_drift_max": float(np.max(np.abs(etot - etot[0])) / abs(w[0]))}
        msg = ""
        if R is not None:
            Rt, RE = R["t"], R["E"]
            for k, (nm, a) in enumerate([("Ek", ek), ("Eth", eth), ("W", w)]):
                rr = np.interp(t, Rt, RE[:, k])
                rec[nm + "_ref"] = rr.tolist()
                rec[nm + "_maxdev_over_W0"] = float(np.max(np.abs(a - rr)) / abs(R["E"][0, 2]))
            msg = " | ".join(f"{nm}: max|3D-1D|/|W0| = {rec[nm + '_maxdev_over_W0']:.3e}" for nm in ("Ek", "Eth", "W"))
        out[os.path.basename(rd)] = rec
        print(f"L5 {os.path.basename(rd)}: max|E_tot - E_tot0|/|W0| = {rec['Etot_drift_max']:.3e}  {msg}")
    return out


def L6():
    out = {}
    for rd in ok_runs("L6_be/*"):
        nm = os.path.basename(rd)  # xi3_f1.1 ...
        xi, f = nm[2:].split("_f")
        out[nm] = compare_rho_c("L6", rd, f"L6_xi{float(xi):g}_f{float(f):g}_N1000", [1, 2, 5, 10, 15, 20, 25, 30])
    return out


def L6b():
    """BE spheres in a light hot ambient: rho_c(t) vs 1D, oscillation extrema (2nd half of the run),
    and the mass inside the initial BE radius (must stay constant if the sphere is stable)."""
    out = {}
    for rd in ok_runs("L6b_be/*"):
        nm = os.path.basename(rd)  # xi3_f1.1[_nj64]
        xi, f = nm[2:].split("_nj")[0].split("_f")
        rec = compare_rho_c("L6b", rd, f"L6_xi{float(xi):g}_f{float(f):g}_N1000", [1, 2, 5, 10, 15, 20, 25, 30])
        m_in = series(rd, lambda d: (d["time"], float(np.sum(d["prim"][0] * d["dx"] ** 3 * (radius(d) < float(xi))))))
        m_in = np.array(m_in)
        rec["M_in"] = m_in.tolist()
        t, r = np.array(rec["t"]), np.array(rec["rho_c"])
        h = t >= 0.5 * t[-1]
        R = ref(f"L6_xi{float(xi):g}_f{float(f):g}_N1000")
        hr = (R["t"] >= 0.5 * t[-1]) & (R["t"] <= t[-1])
        rec["late_rho_c_minmax"] = [float(r[h].min()), float(r[h].max())]
        rec["late_rho_c_minmax_1D"] = [float(R["rho_c"][hr].min()), float(R["rho_c"][hr].max())] if hr.any() else None
        rec["M_in_range_rel"] = [float(m_in[:, 1].min() / m_in[0, 1] - 1), float(m_in[:, 1].max() / m_in[0, 1] - 1)]
        print(f"L6b {nm}: t_end {t[-1]:.2f}  rho_c(0) {r[0]:.3f}  late min/max 3D {rec['late_rho_c_minmax']} "
              f"1D {rec['late_rho_c_minmax_1D']}  final rho_c 3D {r[-1]:.3f}  M(<xi) range {rec['M_in_range_rel']}")
        out[nm] = rec
    return out


def L6c():
    """L6' with production hydro BCs: the box must stay closed (mass, confining pressure flat) and rho_c(t)
    must follow the 1D model with the same light hot ambient and a constant outer pressure (L6amb_a)."""
    out = {}
    P0 = META["L6b_be_xi3_f1.1"]["rho_edge"]
    for rd in ok_runs("L6c_be/*") + ok_runs("L6d_be/*"):
        R = ref("L6rigid" if "/L6d_be/" in rd else "L6amb_a")  # closed box <-> rigid 1D wall
        H = sgv.hst(glob.glob(os.path.join(rd, "*.hst"))[0])
        rows = []
        for f in sgv.dumps(rd):
            d = sgv.load(f, ["prim"])
            r = radius(d)
            s1 = (r > 3.5) & (r < 4.5)
            rows.append((d["time"], float(d["prim"][0][np.argmin(r)]), float(np.median(d["prim"][4][s1]) / P0)))
        a = np.array(rows)
        rec = {"t": a[:, 0].tolist(), "rho_c": a[:, 1].tolist(), "P_conf_rel": a[:, 2].tolist(),
               "box_mass_rel_change": float(H["mass"][-1] / H["mass"][0] - 1), "t_end": float(a[-1, 0]), "windows": []}
        for w0, w1 in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)):
            s = (a[:, 0] >= w0) & (a[:, 0] < w1)
            sr = (R["t"] >= w0) & (R["t"] < w1)
            if s.any():
                rec["windows"].append([w0, w1, float(a[s, 1].min()), float(a[s, 1].max()),
                                       float(R["rho_c"][sr].min()), float(R["rho_c"][sr].max())])
        out[rd.split("/")[-2][:3] + "_" + os.path.basename(rd)] = rec
        print(f"{rd.split('/')[-2][:3]} {os.path.basename(rd)}: t_end {rec['t_end']:.1f}, box mass change {100 * rec['box_mass_rel_change']:+.2f} %, "
              f"confining P/P_edge {a[0, 2]:.4f} -> {a[-1, 2]:.4f}")
        for w in rec["windows"]:
            print(f"   t in [{w[0]},{w[1]}): 3D min {w[2]:.3f} max {w[3]:.3f} | 1D min {w[4]:.3f} max {w[5]:.3f}")
    return out


def L7():
    """rho_c(t) vs the 1D reference, and the Larson-Penston test: with tau from rho_c = D0/(4 pi G tau^2),
    D(x) = 4 pi G rho tau^2 vs x = r/(cs tau) must approach the LP solution."""
    import similarity
    D0, xs, xLP, DLP, VLP, *_ = similarity.larson_penston()
    out = {"LP": {"D0": D0, "x": xLP.tolist(), "D": DLP.tolist(), "V": VLP.tolist()}}
    for rd in ok_runs("L7_be_collapse/*"):
        rec = compare_rho_c("L7", rd, "L7_N2000", [0.2, 0.4, 0.6, 0.7, 0.75, 0.8])
        R = ref("L7_N2000")
        t3, r3 = np.array(rec["t"]), np.array(rec["rho_c"])
        def cross(t, r, lev):
            """First time ln rho_c reaches ln lev: quadratic in t through the three dumps around the
            crossing (ln rho_c(t) is convex, a chord would bias t early by up to ~1e-3 at dn = 8)."""
            r = np.maximum.accumulate(r)
            if r.max() < lev:
                return np.nan
            i = int(np.argmax(r >= lev))
            if i == 0:
                return float(t[0])
            j = slice(max(i - 2, 0), max(i - 2, 0) + 3) if i + 1 >= len(t) else slice(i - 1, i + 2)
            tt, yy = t[j], np.log(r[j])
            c = np.polyfit(tt - t[i], yy, 2) if len(tt) == 3 else np.polyfit(tt - t[i], yy, 1)
            rts = np.roots(c - np.r_[np.zeros(len(c) - 1), np.log(lev)])
            rts = rts[np.isreal(rts)].real + t[i]
            rts = rts[(rts >= t[i - 1] - 1e-12) & (rts <= t[i] + 1e-12)]
            return float(rts.min()) if len(rts) else float(np.interp(np.log(lev), np.log(r), t))
        rec["t_cross"] = {f"{lev:g}": [cross(t3, r3, lev), cross(R["t"], R["rho_c"], lev)] for lev in (1e2, 1e3, 1e4, 1e5)}
        print("L7 " + os.path.basename(rd) + " t(3D) - t(1D) at rho_c = " +
              " ".join(f"{k}: {v[0] - v[1]:+.4f}" for k, v in rec["t_cross"].items()))
        snaps = []
        for f in sgv.dumps(rd)[::4]:
            d = sgv.load(f, ["prim"])
            r = radius(d)
            rc = float(d["prim"][0][np.argmin(r)])
            if rc < 50 or rc > 1e4:  # LP is isothermal; barotropic P is +4.8 % at rho_c = 1e4
                continue
            tau = np.sqrt(D0 / rc)
            x = r / tau
            sel = x < 30
            bins = np.geomspace(0.05, 30, 40)
            idx = np.digitize(x[sel], bins)
            Dm = [np.median(d["prim"][0][sel][idx == i] * tau ** 2) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
            xb = np.sqrt(bins[1:] * bins[:-1])
            Dlp = np.interp(xb, xLP, DLP)
            good = np.isfinite(Dm)
            dev = np.nanmedian(np.abs(np.array(Dm)[good] / Dlp[good] - 1))
            snaps.append({"t": d["time"], "rho_c": rc, "x": xb.tolist(), "D": Dm, "median_rel_dev_vs_LP": float(dev)})
        rec["LP_profiles"] = snaps
        if snaps:
            print("L7 " + os.path.basename(rd) + " LP profile deviation (median over x<30): " +
                  " ".join(f"rho_c={q['rho_c']:.0e}:{100 * q['median_rel_dev_vs_LP']:.1f}%" for q in snaps))
        out[os.path.basename(rd)] = rec
    return out


# ------------------------------------------------------------------ L4 Lin-Mestel-Shu
def lms_exact(a0=4.0, c0=2.0, rho_in=1.0, rho_b=0.01, four_pi_G=1.0, t_end=2.5):
    G = four_pi_G / (4 * np.pi)
    def A(a, c):
        e = np.sqrt(max(1 - c * c / (a * a), 1e-14))
        as_ = np.arcsin(e)
        A1 = np.sqrt(1 - e * e) / e ** 3 * as_ - (1 - e * e) / e ** 2
        return A1, 2 - 2 * A1
    def rhs(t, y):
        a, c, va, vc = y
        rho = rho_in * a0 * a0 * c0 / (a * a * c)
        A1, A3 = A(a, c)
        gb = 4 * np.pi * G * rho_b / 3          # uniform background (cube interior, leading order)
        return [va, vc, -2 * np.pi * G * rho * A1 * a - gb * a, -2 * np.pi * G * rho * A3 * c - gb * c]
    ev = lambda t, y: y[1] - 0.02 * c0
    ev.terminal = True
    return solve_ivp(rhs, (0, t_end), [a0, c0, 0, 0], rtol=1e-11, atol=1e-13, dense_output=True, events=ev)


def L4():
    sol = lms_exact()
    out = {}
    for rd in ok_runs("L4_lms_spheroid/*"):
        rows = []
        for f in sgv.dumps(rd):
            d = sgv.load(f, ["prim"])
            rho = d["prim"][0]
            sel = rho > 0.1
            w = (rho[sel] - 0.01) * d["vol"][sel]
            a = np.sqrt(5 * np.sum(w * (d["x"][sel] ** 2 + d["y"][sel] ** 2) / 2) / np.sum(w))
            c = np.sqrt(5 * np.sum(w * d["z"][sel] ** 2) / np.sum(w))
            if d["time"] > sol.t[-1]:
                break
            ae, ce = sol.sol(d["time"])[:2]
            rows.append({"t": d["time"], "a": a, "c": c, "a_ex": float(ae), "c_ex": float(ce)})
        out[os.path.basename(rd)] = rows
        for q in rows[::5]:
            print(f"L4 t={q['t']:.2f} a={q['a']:.4f} (exact {q['a_ex']:.4f}, {100 * (q['a'] / q['a_ex'] - 1):+.2f}%)  "
                  f"c={q['c']:.4f} (exact {q['c_ex']:.4f}, {100 * (q['c'] / q['c_ex'] - 1):+.2f}%)")
    return out


# ------------------------------------------------------------------ L8 Shu
def L8():
    import similarity
    m0, xs, a_s, v_s = similarity.shu()
    out = {}
    for rd in ok_runs("L8_sis/*"):
        snaps = []
        for f in sgv.dumps(rd):
            d = sgv.load(f, ["prim"])
            t = d["time"]
            if t < 0.5:
                continue
            r = radius(d)
            vr = (d["prim"][1] * d["x"] + d["prim"][2] * d["y"] + d["prim"][3] * d["z"]) / np.maximum(r, 1e-12)
            x = r / t
            bins = np.linspace(0.1, 1.4, 27)
            idx = np.digitize(x, bins)
            xb = 0.5 * (bins[1:] + bins[:-1])
            alpha = np.array([np.median(d["prim"][0][idx == i] * t ** 2) for i in range(1, len(bins))])
            v = np.array([np.median(vr[idx == i]) for i in range(1, len(bins))])
            a_ex = np.where(xb < 1, np.interp(xb, xs, a_s), 2 / xb ** 2)
            v_ex = np.where(xb < 1, np.interp(xb, xs, v_s), 0.0)
            inner = xb < 0.9
            snaps.append({"t": t, "x": xb.tolist(), "alpha": alpha.tolist(), "v": v.tolist(), "alpha_ex": a_ex.tolist(),
                          "v_ex": v_ex.tolist(), "alpha_med_dev": float(np.median(np.abs(alpha[inner] / a_ex[inner] - 1))),
                          "v_med_dev": float(np.median(np.abs(v[inner] - v_ex[inner])))})
            print(f"L8 t={t:.2f}: median |alpha/alpha_Shu - 1| (0.1<x<0.9) = {100 * snaps[-1]['alpha_med_dev']:.1f}%  "
                  f"median |v - v_Shu| = {snaps[-1]['v_med_dev']:.3f} c_s")
        out[os.path.basename(rd)] = {"m0": m0, "snaps": snaps}
    return out


# ------------------------------------------------------------------ L9 / L10: peaks
def clumps(d, thresh, rsep):
    """Density peaks above thresh: (1) local maxima = densest cell within 3 of its OWN cell widths,
    (2) greedy suppression in descending density: a maximum within rsep of a denser accepted peak is
    dropped (mirror-symmetric twins and prolongation plateaus count once; a bar with one maximum = 1)."""
    from scipy.spatial import cKDTree
    rho = d["prim"][0]
    sel = rho > thresh
    if not np.any(sel):
        return []
    pts = np.stack([d["x"][sel], d["y"][sel], d["z"][sel]], 1)
    rr = rho[sel]
    tree = cKDTree(pts)
    nb = tree.query_ball_point(pts, 3.0 * d["dx"][sel])
    cand = sorted((i for i in range(len(rr)) if rr[i] >= rr[nb[i]].max()), key=lambda i: -rr[i])
    peaks = []
    for i in cand:
        if all(np.sum((pts[i] - q[:3]) ** 2) > rsep ** 2 for q in peaks):
            peaks.append(np.append(pts[i], rr[i]))
    return [list(map(float, q)) for q in peaks]


def L9_L10(name, pattern, thresh, rsep):
    out = {}
    for rd in ok_runs(pattern):
        rows = []
        for f in sgv.dumps(rd):
            d = sgv.load(f, ["prim"])
            pk = clumps(d, thresh, rsep)
            rows.append({"t": d["time"], "rho_max": float(d["prim"][0].max()), "n_peaks": len(pk), "peaks": pk[:10],
                         "max_level": int(d["level"].max()) if "level" in d else 0})
        out[os.path.basename(rd)] = rows
        last = rows[-1]
        for q in rows[-6:]:
            pk = np.array(q["peaks"])
            sep = (f" sep(1-2) {np.linalg.norm(pk[0, :3] - pk[1, :3]):.4f}" if len(pk) > 1 else "")
            print(f"{name} {os.path.basename(rd)}: t={q['t']:.4f} rho_max={q['rho_max']:.3e} peaks(>{thresh:g}, rsep {rsep:g})="
                  f"{q['n_peaks']} max level {q['max_level']}{sep} top: " +
                  " ".join(f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f};{p[3]:.2e})" for p in pk[:4]))
    return out


if __name__ == "__main__":
    want = sys.argv[1:] or ["L1", "L2", "L3a", "L3b", "L4", "L5", "L6", "L7", "L8", "L9", "L10"]
    p = os.path.join(OUT, "tables", "lit.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    fns = {"L1": L1, "L2": L2, "L3a": L3a, "L3b": L3b, "L4": L4, "L5": L5, "L6": L6, "L6b": L6b, "L6c": L6c, "L7": L7, "L8": L8,
           "L9": lambda: L9_L10("L9", "L9_boss_bodenheimer/nj*", 1e3, 0.1),  # 0.1 = 62 au
           "L10": lambda: L9_L10("L10", "L10_hennebelle/*", 1e5, 0.08)}  # l0 = 11700 au: 0.08 = 940 au
    for k in want:
        try:
            res[k] = fns[k]()
        except Exception as e:  # report, keep going
            print(f"{k}: ERROR {e!r}")
    json.dump(res, open(p, "w"), indent=1)
