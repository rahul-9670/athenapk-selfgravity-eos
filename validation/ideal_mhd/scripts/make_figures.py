#!/usr/bin/env python
"""Figures for IDEAL_MHD_VALIDATION.md, from the saved tables only (results/tables/*.json) -> results/fig_*.png.
Palette: the dataviz reference categorical slots 1-4 in fixed order (validated: CVD dE >= 9.1; slots 3-4 below 3:1
contrast -> every series is direct-labelled and has its own marker shape; the numbers are in the report tables)."""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = _VT + "/ideal_mhd"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MK = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3dd"
T = lambda n: json.load(open(f"{M}/results/tables/{n}.json"))


def style(ax, xl, yl):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_xlabel(xl, color=INK2, fontsize=9)
    ax.set_ylabel(yl, color=INK2, fontsize=9)


def label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def label_ends(ax, items, xfac=1.08, sep=1.45):
    """direct labels at the line ends, nudged apart vertically (log axis) so they never overlap."""
    items = sorted(items, key=lambda q: q[1])
    ys = []
    for x, y, txt in items:
        y2 = max(y, ys[-1] * sep) if ys else y
        ys.append(y2)
        ax.annotate(txt, (x, y), xytext=(x * xfac, y2), fontsize=8, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6) if y2 != y else None)


def fig_waves():
    t = T("I1")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, dim in zip(axes, (1, 2, 3)):
        ends = []
        for i, w in enumerate(("fast", "alfven", "slow", "entropy")):
            rows = t[f"{dim}d_{w}_prod"]
            N = [r["N"] for r in rows]; e = [r["RMS_L1"] for r in rows]
            ax.loglog(N, e, color=C[i], marker=MK[i], ms=5, lw=2, label=w)
            ends.append((N[-1], e[-1], w))
        label_ends(ax, ends)
        ax.set_xlim(right=N[-1] * 2.2)
        n0, e0 = rows[0]["N"], 2 * rows[0]["RMS_L1"]
        ax.loglog([n0, N[-1]], [e0, e0 * (n0 / N[-1]) ** 2], color=INK2, lw=1, ls="--")
        ax.set_title(f"{dim}D", fontsize=10, color=INK)
        style(ax, "N", "RMS L1 error (amp 1e-6)" if dim == 1 else "")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("I1 linear MHD waves, production scheme keys (dashed: second order)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(f"{M}/results/fig_I1_waves.png", dpi=130); plt.close(fig)


def fig_noh():
    t = T("I6b")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (tag, nm) in enumerate((("dualtrue", "dual energy on (production keys before 2026-10-12)"), ("dualfalse", "dual energy off"))):
        pts = sorted((r["Mach"], abs(r["plateau"])) for k, r in t.items() if k.endswith(tag))
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], color=C[i], marker=MK[i], ms=6, lw=2, label=nm)
        if i == 0:
            label_end(ax, pts[-1][0], pts[-1][1], "dual energy on")
    fx = T("F")["f1"]
    kv = sorted((r["Mach"], abs(r["plateau"])) for k, r in fx.items() if k.startswith("f1v_noh1d_P") and k.count("_") == 2)
    ax.loglog([p[0] for p in kv], [p[1] for p in kv], color=C[2], marker=MK[2], ms=6, lw=2,
              label="adopted: shock_eta1 + detector kv")
    style(ax, "upstream Mach number", "|plateau density error|")
    ax.set_xlim(5, 3000)
    ax.legend(fontsize=8, frameon=False, loc="center", bbox_to_anchor=(0.55, 0.38))
    ax.set_title("I6 planar Noh, gamma 5/3, N 1024: post-shock density error vs exact", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(f"{M}/results/fig_I6_noh_mach.png", dpi=130); plt.close(fig)


def fig_glm():
    t = T("I5")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    pts = sorted((r["lambda"], r["rows"][-1]["exact_divB_L2_ratio"]) for k, r in t.items()
                 if k.startswith("I5_2d_prod_N"))
    ax.semilogx([p[0] for p in pts], [p[1] for p in pts], color=C[0], marker=MK[0], ms=6, lw=2,
                label="exact, alpha 0.1")
    for (x, y), n in zip(pts, (64, 128, 256, 512)):
        ax.annotate(f"N {n}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([6.4, 12.8, 25.6, 51.2]); ax.minorticks_off()
    # single series: no legend box; the title and axis name it (the code follows this curve: report section 8)
    style(ax, "GLM damping rate lambda = alpha c_h / mindx", "||div B||_2 (t 0.3) / (t 0)")
    ax.set_title("I5 GLM pulse (exact telegraph solution): over-damping past lambda ~ c_h k", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(f"{M}/results/fig_I5_glm.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    fig_waves(); fig_noh(); fig_glm()
    print("figures written")
