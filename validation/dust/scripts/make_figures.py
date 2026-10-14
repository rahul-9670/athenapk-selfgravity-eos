#!/usr/bin/env python
"""Figures of the dust report (results/figures/*.png), from the saved tables and the F2 trace run.
  fig_shock_deficit.png  D5/F2: plateau dust deficit vs N for the production rule and each candidate key (M 10, M 30)
  fig_parcel_trace.png   F2_M10_N256_trace: T - T2 of post-shock parcels vs time since the shock reached them
  fig_contact_loss.png   D4c: dust destroyed by contact mixing vs N (order 0.70)
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _VT + "/eos_tabulated/scripts")
import eostab
DU = _VT + "/dust"
T_ = os.path.join(DU, "results", "tables")
OUT = os.path.join(DU, "results", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"hold0": "#1a1a19", "hold1": "#8a8980", "hold2": "#5b8def", "filt": "#e8a33d", "tau": "#2e9e6b"}
LAB = {"hold0": "production (instantaneous)", "hold1": "subl_shock_hold = 1", "hold2": "subl_shock_hold = 2",
       "filt": "subl_T_filter", "tau": "subl_tau_s (finite time)"}

# 1. deficit vs N
F = json.load(open(os.path.join(T_, "F2_shock_hold.json")))
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
for ax, M in zip(axs, (10, 30)):
    for key in ("hold0", "hold1", "hold2", "filt", "tau"):
        pts = sorted((v["N"], abs(v["dev"])) for k, v in F.items()
                     if v["M"] == M and (k.endswith("_" + key) or k.endswith("_" + key.replace("hold", "hold"))))
        pts = [(n, d) for n, d in pts if d > 0]
        if pts:
            ax.loglog(*zip(*pts), "o-", color=COL[key], label=LAB[key], lw=1.6, ms=5)
    ax.set_title(f"Noh Mach {M}: |plateau dust deficit|")
    ax.set_xlabel("cells N")
    ax.set_xticks([256, 1024, 4096]); ax.set_xticklabels(["256", "1024", "4096"])
axs[0].set_ylabel("|f / f_exact - 1|")
axs[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_shock_deficit.png"), dpi=130); plt.close(fig)

# 2. parcel trace
m = json.load(open(os.path.join(DU, "inputs", "D5_meta.json")))["D5_M10_N256_dT+0"]
fs = sorted(glob.glob(os.path.join(DU, "runs", "F", "F2_M10_N256_trace", "*.out0.0*.phdf")))
tab, Ts, ts = None, [], []
for fn in fs:
    with h5py.File(fn) as h:
        names = [n.decode() if isinstance(n, bytes) else n for n in h["Info"].attrs["ComponentNames"]]
        cn = [n for n in names if n.startswith("cons_")]
        p, c = h["prim"][()], h["cons"][()]
        if tab is None:
            tab = eostab.load(_REPO + "/src/eos/eos_table_hires_v5_cold_rho26.bin",
                              float(h["Params"].attrs["Hydro/phys_units/rho_unit_cgs"]),
                              float(h["Params"].attrs["Hydro/phys_units/v_unit_cgs"]))
            x = h["Locations"]["x"][()]; xc = 0.5 * (x[0, 1:] + x[0, :-1])
        rho, v = p[0, 0, 0, 0, :], p[0, 1, 0, 0, :]
        E = c[0, cn.index("cons_total_energy_density"), 0, 0, :]
        Ts.append([eostab.T_of(tab, r, (e - 0.5 * r * vv * vv) / r) for r, e, vv in zip(rho, E, v)])
        ts.append(float(h["Info"].attrs["Time"]))
Ts, ts = np.array(Ts), np.array(ts)
dtc = (xc[1] - xc[0]) / m["S"]
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, cc in zip((140, 160, 180, 200), ("#1a1a19", "#5b8def", "#e8a33d", "#2e9e6b")):
    ti = (xc[i] - 0.5) / m["S"]; sel = ts > ti
    ax.plot((ts[sel] - ti) / dtc, Ts[sel, i] - m["T2"], color=cc, lw=1.3, label=f"cell {i}")
ax.axhline(0, color="#8a8980", lw=0.8)
ax.set_xlim(0, 12); ax.set_ylim(-8, 4)
ax.set_xlabel("time since the shock reached the cell [shock cell-crossing times dx/S]")
ax.set_ylabel("T - T2 [K]")
ax.set_title("Mach 10 Noh, N 256: every parcel overshoots T2 by ~2 K for ~1 dx/S")
ax.legend(fontsize=8, frameon=False, ncol=4)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_parcel_trace.png"), dpi=130); plt.close(fig)

# 3. contact loss
D4 = json.load(open(os.path.join(T_, "D4_advection.json")))
pts = sorted((v["N"], -v["dM_dust"]) for k, v in D4.items() if k.startswith("D4c_1d"))
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.loglog(*zip(*pts), "o-", color="#5b8def", lw=1.6)
n = np.array([p[0] for p in pts]); ax.loglog(n, pts[0][1] * (n / n[0]) ** -0.7, "--", color="#8a8980", lw=1,
                                              label="N^-0.7")
ax.set_xlabel("cells N"); ax.set_ylabel("dust mass destroyed (fraction)")
ax.set_title("D4c: contact 700 K | 1900 K, 10 crossings")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_contact_loss.png"), dpi=130); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
