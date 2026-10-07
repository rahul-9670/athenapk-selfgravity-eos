#!/usr/bin/env python
"""Figures for results/RESULTS.md from results/tables/*.json (run the analyze_*.py scripts first).

Palette: validated categorical slots 1-4 (#2a78d6 blue, #eb6834 orange, #1baf7a aqua, #eda100 yellow;
dataviz validate_palette.js light: CVD worst adjacent dE 9.1, contrast WARN for aqua/yellow -> every
series also carries a distinct marker and a legend, and RESULTS.md has the numbers as tables).
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
T = os.path.join(ROOT, "results", "tables")
F = os.path.join(ROOT, "results", "figures")
os.makedirs(F, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "lines.markersize": 6, "legend.frameon": False, "savefig.dpi": 200})


def n_ticks(ax, ns):
    """Explicit cell-count ticks on a log x axis (the default minor labels collide)."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator
    ns = sorted(set(int(n) for n in ns))
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in ns]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def load(name):
    p = os.path.join(T, name)
    return json.load(open(p)) if os.path.exists(p) else None


def slope_guide(ax, x, y0, p, label):
    x = np.asarray(x, float)
    ax.plot(x, y0 * (x / x[0]) ** (-p), ls="--", lw=1, color=MUTED)
    ax.annotate(label, (x[-1], y0 * (x[-1] / x[0]) ** (-p)), color=MUTED, fontsize=8, xytext=(4, 0),
                textcoords="offset points", va="center")


def n_of(run):
    m = re.search(r"(?:^|/)(?:n|root|rk2_cfl0\.327_n|vl2_cfl0\.3_n)(\d+)", run)
    return int(m.group(1)) if m else None


def fig_static():
    rows = load("static.json")
    if not rows:
        return
    rows = [r for r in rows if "error" not in r]
    # S1 convergence + tolerance sweep
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    s1 = sorted([(n_of(r["run"]), r["eps_abs"]) for r in rows if re.search(r"S1_sinusoid/n\d+_prodabs$", r["run"])])
    if s1:
        n, e = zip(*s1)
        ax[0].loglog(n, e, marker=MK[0], color=C[0], label="production keys + abs. floor")
        slope_guide(ax[0], n, e[0] * 1.6, 2, "N$^{-2}$")
        ax[0].set(xlabel="cells per dimension N", ylabel=r"rms $|\phi-\phi^*|$", title="S1 periodic sinusoid")
        n_ticks(ax[0], n)
        ax[0].legend(loc="lower left")
    sw = sorted([(float(r["run"].split("_")[-1]), r["eps_abs"]) for r in rows if "cold_n128_" in r["run"]])
    if sw:
        t, e = zip(*sw)
        ax[1].loglog(t, e, marker=MK[1], color=C[1])
        ax[1].invert_xaxis()
        ax[1].set(xlabel="residual_tolerance (cold start, rel. to rms rhs)", ylabel=r"rms $|\phi-\phi^*|$",
                  title="S1 tolerance sweep, 128$^3$")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "S1_sinusoid.png"))
    plt.close(fig)
    # isolated problems: core force error, production BC vs splits off
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for k, suffix, title in [(0, "_mp", "multipole BC, production (splits on)"),
                             (1, "_mp_nosplit", "multipole BC, splits off")]:
        for i, (tst, lab) in enumerate([("S2_gaussian", "Gaussian"), ("S3_sphere", "uniform sphere"),
                                        ("S5_maclaurin", "Maclaurin e=0.9"), ("S7_be_ic", "production BE IC")]):
            pts = sorted([(n_of(r["run"]), r["force_rel_core"]) for r in rows
                          if r["run"].startswith(tst + "/") and re.search(rf"/n\d+{suffix}$", r["run"]) and "force_rel_core" in r])
            if pts:
                n, e = zip(*pts)
                ax[k].loglog(n, e, marker=MK[i], color=C[i], label=lab)
        ax[k].set(xlabel="cells per dimension N", title=title)
        n_ticks(ax[k], [32, 64, 128, 256])
    ax[0].set_ylabel(r"median $|\Delta g|/|g|$, r < L/4")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(F, "S_force_error.png"))
    plt.close(fig)


def fig_dynamic_convergence():
    J, S = load("jeans.json"), load("slab.json")
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
    if J:
        for k, (nj, title) in enumerate([("0.5", "D1 stable Jeans, n$_J$=0.5"), ("1.5", "D2 unstable, n$_J$=1.5")]):
            pts = sorted([(n_of(r["run"].split("_", 2)[-1]), r["rms_l1"]) for r in J["convergence"]
                          if f"nJ{nj}_" in r["run"] and r["run"].endswith("_prod")])
            if pts:
                n, e = zip(*pts)
                ax[k].loglog(n, e, marker=MK[k], color=C[k], label="AthenaPK")
                slope_guide(ax[k], n, e[0] * 1.6, 2, "N$^{-2}$")
                ax[k].set(xlabel="N$_x$", ylabel="RMS L1 error", title=title)
                n_ticks(ax[k], n)
    if S:
        for i, (tag, lab) in enumerate([("rk2_cfl0.327", "rk2, artemis CFL"), ("vl2_cfl0.3", "vl2 (prod.)")]):
            pts = sorted([(r["nx"], r["rms_l1"]) for r in S if tag in r["run"]])
            if pts:
                n, e = zip(*pts)
                ax[2].loglog(n, e, marker=MK[i], color=C[i], label=lab)
        ax[2].loglog([16, 32], [1.850706e-01, 4.414474e-02], ls="none", marker="x", ms=9, mew=2, color=INK,
                     label="artemis golden")
        slope_guide(ax[2], [16, 128], 0.3, 2, "N$^{-2}$")
        ax[2].set(xlabel="N$_x$", ylabel="RMS L1 error", title="D4 advected slab")
        n_ticks(ax[2], [16, 32, 64, 128])
        ax[2].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "D_convergence.png"))
    plt.close(fig)


def fig_dispersion():
    J = load("jeans.json")
    if not J or not J["dispersion"]:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    nn = np.linspace(0.0, 3.1, 300)
    ax.plot(nn, 1 - nn ** 2, color=MUTED, lw=1.5, label=r"exact $1-n_J^2$")
    d = sorted(J["dispersion"], key=lambda r: r["nJ"])
    k2 = (6 * np.pi) ** 2
    ax.plot([r["nJ"] for r in d], [r["omega2_meas"] / k2 for r in d], ls="none", marker=MK[0], color=C[0],
            label="AthenaPK, 64x32x32")
    ax.axhline(0, color=GRID, lw=1)
    ax.set(xlabel=r"$n_J=\lambda/\lambda_J$", ylabel=r"$\omega^2/(c_s^2k^2)$", title="D3 Jeans dispersion relation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(F, "D3_dispersion.png"))
    plt.close(fig)


def fig_freefall_poly():
    D = load("dynamic.json")
    if not D:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    order = ["D7_freefall/n64", "D7_freefall/n128", "D7_freefall/n256", "D7_freefall/n64_nsub1"]
    labels = ["64$^3$", "128$^3$", "256$^3$", "64$^3$ stair-step (WP-15 IC)"]
    for i, (run, lab) in enumerate(zip(order, labels)):
        s = [x for r in D["freefall"] if r["run"] == run for x in [r["series"]]]
        if s:
            s = s[0]
            ax[0].plot([x["t"] for x in s], [x["err_c_pct"] for x in s], marker=MK[i], ms=4, color=C[i], label=lab)
            if i == 0:
                ax[0].plot([x["t"] for x in s], [x["err_pct"] for x in s], ls=":", lw=1.2, color=C[i],
                           label=r"64$^3$, $\rho_{max}$ (WP-15 metric)")
    ax[0].set(xlabel="t (t$_{ff}$ = 1.92)", ylabel=r"$\rho_c/\rho^*-1$ (%)", title="D7 pressure-free collapse")
    ax[0].legend()
    for i, run in enumerate(["D5_polytrope/n32_mp", "D5_polytrope/n64_mp", "D5_polytrope/n128_mp"]):
        s = [r["series"] for r in D["polytrope"] if r["run"] == run]
        if s:
            ax[1].plot([x["t"] for x in s[0]], [x["rho_c"] / s[0][0]["rho_c"] for x in s[0]], marker=MK[i], ms=4,
                       color=C[i], label=run.split("/")[1].split("_")[0][1:] + "$^3$")
    ax[1].set(xlabel="t", ylabel=r"$\rho_c(t)/\rho_c(0)$", title="D5 n=1 polytrope (artemis IC)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(F, "D57_freefall_polytrope.png"))
    plt.close(fig)


REF = os.path.join(ROOT, "results", "ref1d")


def ref1d(name):
    q = os.path.join(REF, name + ".npz")
    return dict(np.load(q)) if os.path.exists(q) else None


def fig_L7():
    """Deep-AMR collapse vs the exact 1D solution: rho_c(t), lag at fixed rho_c, Larson-Penston profile."""
    L = (load("lit.json") or {}).get("L7")
    R = ref1d("L7_N2000")
    if not L or R is None:
        return
    runs = [("nj4", "n$_J$ = 4"), ("nj8", "n$_J$ = 8 (production)"), ("nj16", "n$_J$ = 16")]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ax[0].semilogy(R["t"], R["rho_c"], color=INK, lw=1.2, label="1D Lagrangian (exact)")
    for i, (k, lab) in enumerate(runs):
        if k in L:
            ax[0].semilogy(L[k]["t"], L[k]["rho_c"], color=C[i], marker=MK[i], ms=3, lw=1.2, label=lab)
    ax[0].set(xlim=(0.8, 1.035), ylim=(10, 3e5), xlabel="t (code)", ylabel=r"$\rho_c$ (code)",
              title="L7 central density")
    ax[0].legend(fontsize=7)
    lev = [1e2, 1e3, 1e4, 1e5]
    var = [("nj8", "n$_J$ = 8, vl2"), ("nj8_vl2same", "n$_J$ = 8, vl2 + PLM predictor"),
           ("nj8_rk2", "n$_J$ = 8, rk2"), ("nj16", "n$_J$ = 16, vl2")]
    for i, (k, lab) in enumerate(var):
        if k in L and "t_cross" in L[k]:
            tc = L[k]["t_cross"]
            y = [1e3 * (tc[f"{v:g}"][0] - tc[f"{v:g}"][1]) for v in lev]
            ax[1].semilogx(lev, y, color=C[i], marker=MK[i], label=lab)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set(xlabel=r"$\rho_c$ (code)", ylabel=r"$10^3\,(t_{3D}-t_{1D})$", title="lag behind the exact collapse",
              ylim=(-1, 23))
    ax[1].legend(fontsize=7, loc="upper left", ncol=2)
    LP = L["LP"]
    ax[2].loglog(LP["x"], LP["D"], color=INK, lw=1.2, label="Larson-Penston")
    for i, (k, lab) in enumerate(runs[1:], start=1):
        prof = [q for q in L.get(k, {}).get("LP_profiles", []) if q["rho_c"] <= 1e4]
        if prof:
            q = prof[-1]
            ax[2].loglog(q["x"], q["D"], color=C[i], marker=MK[i], ms=3, lw=1.2,
                         label=f"{lab}, " + r"$\rho_c$" + f" = {q['rho_c']:.1e}")
    ax[2].set(xlim=(0.05, 30), xlabel=r"$x = r/(c_s\tau)$", ylabel=r"$D = 4\pi G\rho\tau^2$",
              title="similarity profile (isothermal phase)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "L7_deep_amr_collapse.png"))
    plt.close(fig)


def fig_L6b():
    L = load("lit.json") or {}
    Lb, Lc = L.get("L6b", {}), L.get("L6c", {})
    if not Lb:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    rg = ref1d("L6rigid")
    if rg is not None:
        ax[0].plot(rg["t"], rg["rho_c"], color=INK, lw=1.0, label="1D, same ambient, rigid wall")
    for i, (k, lab) in enumerate([("L6d_xi3_f1.1_nj8", "3D closed box, n$_J$ = 8"),
                                  ("L6d_xi3_f1.1_nj32", "3D closed box, n$_J$ = 32")]):
        if k in Lc:
            ax[0].plot(Lc[k]["t"], Lc[k]["rho_c"], color=C[i], marker=MK[i], ms=3, lw=1.2, label=lab)
    ax[0].set(xlabel="t", ylabel=r"$\rho_c$", title=r"L6d: BE sphere $\xi$ = 3, f = 1.1, closed box", ylim=(0.9, 2.5))
    ax[0].legend(fontsize=7, loc="upper left")
    ramp, amb = ref1d("L6ramp_k0.00857"), ref1d("L6amb_a")
    if amb is not None:
        ax[1].plot(amb["t"], amb["rho_c"], color=INK, lw=1.0, label="1D, constant outer P")
    if ramp is not None:
        ax[1].plot(ramp["t"], ramp["rho_c"], color=MUTED, lw=1.0, ls="--", label="1D, outer P ramped as measured in 3D")
    for i, (k, lab) in enumerate([("xi3_f1.1", "3D open (outflow) box, n$_J$ = 8"), ("xi3_f1.1_nj64", "3D open box, n$_J$ = 64")]):
        if k in Lb:
            ax[1].plot(Lb[k]["t"], Lb[k]["rho_c"], color=C[i], marker=MK[i], ms=3, lw=1.2, label=lab)
    ax[1].set(xlabel="t", ylabel=r"$\rho_c$", title="L6': drift = ambient inflow (diagnosis)", ylim=(0.9, 3.2))
    ax[1].legend(fontsize=7, loc="upper left")
    for i, (k, lab) in enumerate([("xi3_f1.1", r"$\xi$ = 3, f = 1.1"), ("xi6.45_f1", r"$\xi$ = 6.45, f = 1"),
                                  ("xi9_f1", r"$\xi$ = 9, f = 1"), ("xi9_f1.1", r"$\xi$ = 9, f = 1.1")]):
        if k in Lb:
            ax[2].semilogy(Lb[k]["t"], Lb[k]["rho_c"], color=C[i], marker=MK[i], ms=3, lw=1.2, label=lab)
    ax[2].set(xlabel="t", ylabel=r"$\rho_c$", title="stability boundary (3D)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "L6b_bonnor_ebert.png"))
    plt.close(fig)


def fig_equilibria():
    """L1 (n = 1 polytrope + atmosphere), L3a (n = 5 polytrope), L2 (Spitzer sheet convergence)."""
    L = load("lit.json") or {}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for i, (k, lab) in enumerate([("n40", "R/dx = 5"), ("n80", "R/dx = 10"), ("n160", "R/dx = 20")]):
        if k in L.get("L1", {}):
            r = [q for q in L["L1"][k] if q["vmax_body"] > 0]  # t = 0 has v = 0 (off a log axis)
            ax[0].semilogy([q["t"] / 0.7071 for q in r], [q["vmax_body"] for q in r], color=C[i], marker=MK[i], ms=4, lw=1.2,
                           label=lab)
    if "n80_100tau" in L.get("L1", {}):
        r = [q for q in L["L1"]["n80_100tau"] if q["vmax_body"] > 0]
        ax[0].semilogy([q["t"] / 0.7071 for q in r], [q["vmax_body"] for q in r], color=C[1], ls=":", lw=1.2,
                       label=r"R/dx = 10, to 100 $\tau$")
    ax[0].set(xlabel=r"t / $\tau$  ($\tau$ = R/v$_{esc}$ = 0.707)", ylabel="max |v| in the body (code)",
              title="L1 n=1 polytrope + atmosphere: spurious flow")
    ax[0].legend(fontsize=7)
    for i, (k, lab) in enumerate([("n32", "32$^3$"), ("n64", "64$^3$"), ("n128", "128$^3$")]):
        if k in L.get("L3a", {}):
            r = L["L3a"][k]
            ax[1].plot([q["t"] for q in r], [100 * (q["rho_c"] - 1) for q in r], color=C[i], marker=MK[i], ms=4,
                       lw=1.2, label=lab)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set(xlabel="t", ylabel=r"$\rho_c/\rho_{c,exact} - 1$ (%)",
              title="L3a n=5 polytrope (t=0 offset = cell average)")
    ax[1].legend(fontsize=7)
    r2 = L.get("L2")
    if r2:
        n = [q["n"] for q in r2]
        e = [q["L1"] for q in r2]
        ax[2].loglog(n, e, color=C[0], marker=MK[0], label="L1 error of rho")
        slope_guide(ax[2], n[2:], e[2] * 1.6, 2, "2nd order")
        n_ticks(ax[2], n)
    ax[2].set(xlabel="N cells", ylabel="L1(rho) error", title="L2 advected Spitzer sheet")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "L1_L2_L3a_equilibria.png"))
    plt.close(fig)


def fig_collapse_1d():
    """L3b (n = 5 collapse + bounce) and L5 (Evrard) vs the 1D Lagrangian references."""
    L = load("lit.json") or {}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    R = ref1d("L3b_N2000")
    if R is not None:
        ax[0].plot(R["t"], R["rho_c"], color=INK, lw=1.0, label="1D (N = 2000)")
    for i, (k, lab) in enumerate([("n128", "128$^3$"), ("n256", "256$^3$")]):
        if k in L.get("L3b", {}):
            ax[0].plot(L["L3b"][k]["t"], L["L3b"][k]["rho_c"], color=C[i], marker=MK[i], ms=2.5, lw=1.0, label=lab)
    ax[0].set(xlabel="t", ylabel=r"$\rho_c$", title="L3b n=5 polytrope: collapse and bounce")
    ax[0].legend(fontsize=7)
    r5 = L.get("L5", {})
    if "n128" in r5:
        q = r5["n128"]
        w0 = abs(q["W_ref"][0]) if q.get("W_ref") else abs(q["W"][0])
        for comp, (c, mk) in zip(("Ek", "Eth", "W"), zip(C, MK)):
            ax[1].plot(q["t"], np.array(q[comp]) / w0, color=c, lw=1.4, label=f"{comp} (3D 128$^3$)")
            if comp + "_ref" in q:
                ax[1].plot(q["t"], np.array(q[comp + "_ref"]) / w0, color=c, lw=1.0, ls="--")
        et = (np.array(q["Ek"]) + np.array(q["Eth"]) + np.array(q["W"])) / w0
        ax[1].plot(q["t"], et, color=INK, lw=1.2, label="total (3D)")
    ax[1].set(xlabel="t", ylabel=r"E / |W$_0$|", title="L5 Evrard collapse (dashed: 1D)")
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "L3b_L5_collapse_vs_1D.png"))
    plt.close(fig)


def fig_deep():
    D = load("deep_amr.json") or {}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    rows = D.get("DA1", {}).get("root64_L6", [])
    if rows:
        lv = [r["level"] for r in rows]
        ax[0].semilogy(lv, [r["force_rel_median"] for r in rows], color=C[0], marker=MK[0], label="median")
        ax[0].semilogy(lv, [r["force_rel_p99"] for r in rows], color=C[1], marker=MK[1], label="99th percentile")
        ax[0].set_xticks(lv)
    ax[0].set(xlabel="AMR level (dx = 0.25 / 2$^{level}$)", ylabel="|g - g$_{exact}$| / |g$_{exact}$|",
              title="DA1 production IC, 6 static levels")
    ax[0].legend(fontsize=7)
    p = os.path.join(T, "DA2_time_offset.txt")
    if os.path.exists(p):
        rc, lead = [], []
        for line in open(p):
            m = re.search(r"rho_c/rho0=\s*([\d.]+).*\(([-+\d.]+)% of t_ff\)", line)
            if m:
                rc.append(float(m.group(1)))
                lead.append(float(m.group(2)))
        ax[1].semilogx(rc, lead, color=C[0], marker=MK[0], label="time lead of 3D collapse")
        for x in (4.1, 33, 262):
            ax[1].axvline(x, color=GRID, lw=1.0)
        ax[1].annotate("sphere edge crosses level boundaries", (4.3, max(lead) * 0.15), fontsize=7, color=MUTED)
    ax[1].set(xlabel=r"$\rho_c/\rho_0$", ylabel="lead (% of t$_{ff}$)", title="DA2 free fall through 4 static levels",
              ylim=(0, 0.035))
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "DA_deep_amr.png"))
    plt.close(fig)


def fig_L4():
    L = (load("lit.json") or {}).get("L4")
    if not L:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for i, k in enumerate(["n128", "n256"]):
        if k in L:
            t = [r["t"] for r in L[k]]
            ax.plot(t, [100 * (r["a"] / r["a_ex"] - 1) for r in L[k]], color=C[i], marker=MK[i], ls="--", lw=1.2,
                    mfc="white", label=f"{k[1:]}$^3$, a (long axis)")
            ax.plot(t, [100 * (r["c"] / r["c_ex"] - 1) for r in L[k]], color=C[i], marker=MK[i], lw=1.6,
                    label=f"{k[1:]}$^3$, c (short axis)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="t", ylabel="axis / exact - 1 (%)", title="L4 Lin-Mestel-Shu spheroid collapse",
           xlim=(-0.03, 1.53), ylim=(-1, 10))
    ax.annotate("c error grows steeply as the pancake thins toward ~5 cells (t = 1.5)", (0.02, 0.6), xycoords="axes fraction",
                fontsize=7, color=MUTED)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "L4_lms_spheroid.png"))
    plt.close(fig)


def fig_L9():
    """Equatorial density slice of the Boss-Bodenheimer runs at t = 2.45 (n_J = 4 vs 8), AMR cells rasterised."""
    import glob as _g
    import sys as _s
    _s.path.insert(0, os.path.dirname(__file__))
    import sgv
    import analyze_lit as al
    from matplotlib.colors import LogNorm
    runs = [("n$_J$ = 4", os.path.join(ROOT, "runs", "L9_boss_bodenheimer", "nj4", "parthenon.out0.00049.phdf")),
            ("n$_J$ = 8", os.path.join(ROOT, "runs", "F7_hang", "L9armB", "try1", "parthenon.out0.00049.phdf"))]
    runs = [(l, f) for l, f in runs if os.path.exists(f)]
    if not runs:
        return
    W, npx = 0.6, 600
    fig, ax = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.4))
    ax = np.atleast_1d(ax)
    for a, (lab, f) in zip(ax, runs):
        d = sgv.load(f, ["prim"])
        rho, x, y, z, dx = d["prim"][0], d["x"], d["y"], d["z"], d["dx"]
        sel = (z - 0.5 * dx <= 0) & (z + 0.5 * dx > 0) & (np.abs(x) - 0.5 * dx < W) & (np.abs(y) - 0.5 * dx < W)
        img = np.full((npx, npx), np.nan)
        h = 2 * W / npx
        order = np.argsort(-dx[sel])                       # coarse first, fine cells overwrite
        for xs, ys, ds, rs in zip(x[sel][order], y[sel][order], dx[sel][order], rho[sel][order]):
            i0, i1 = int((xs - ds / 2 + W) / h), int(np.ceil((xs + ds / 2 + W) / h))
            j0, j1 = int((ys - ds / 2 + W) / h), int(np.ceil((ys + ds / 2 + W) / h))
            img[max(j0, 0):min(j1, npx), max(i0, 0):min(i1, npx)] = rs
        im = a.imshow(img, origin="lower", extent=(-W, W, -W, W), norm=LogNorm(vmin=10, vmax=max(1e6, np.nanmax(img))),
                      cmap="magma", interpolation="nearest")
        pk = np.array(al.clumps(d, 1e3, 0.1))
        if len(pk):
            a.scatter(pk[:, 0], pk[:, 1], s=40, facecolors="none", edgecolors="#2a78d6", linewidths=1.2)
        a.grid(False)
        a.set(xlabel="x (code; 1 = 620 au)", ylabel="y (code)", title=f"L9 Boss-Bodenheimer, {lab}, t = {d['time']:.3f}")
        fig.colorbar(im, ax=a, shrink=0.85, label=r"$\rho$ (code)")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "L9_fragments_t245.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_static()
    fig_dynamic_convergence()
    fig_dispersion()
    fig_freefall_poly()
    fig_L7()
    fig_L6b()
    fig_L4()
    fig_equilibria()
    fig_collapse_1d()
    fig_deep()
    fig_L9()
    print(sorted(os.listdir(F)))
