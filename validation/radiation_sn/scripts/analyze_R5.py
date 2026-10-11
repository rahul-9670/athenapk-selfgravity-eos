#!/usr/bin/env python
"""R5 analysis: group-count / edge convergence (R5a, 1-D) and ordinate convergence (R5b, 3-D), decks by make_decks_R5.py.

Observable = the gas temperature of the ENVELOPE heated by the hot core (T_gas = p/rho x T_unit, ideal gas), at the
last dump (t = 1.2e-5 code) and at mid-run, in |x| in [3, 18] AU (R5a; the core edge, which moves <= 0.3 AU, and the
outer boundary layer are excluded) or in the shell 3 AU < r < 5.5 AU (R5b). Metrics per arm vs the reference (R5a: L18,
0.25 dex; R5b: 12x24): max and median |T/T_ref - 1|; the envelope heat Q = int (e - e_0) dV relative to the reference;
the position where T falls to 20 / 50 K (R5a). R5b adds the ray-effect anisotropy: the rms of T over the shell cells at
fixed r (cells binned in r, 0.25 AU) relative to the shell mean. Cost: mean wall per cycle (wsec_step, cycles >= 2) and
Sn iterations per cycle (sum of the "converged in N iters" lines).
usage: analyze_R5.py run | check [a|c]   -> results/tables/R5_groups_ordinates.txt (a = first pass, step-sweep
       preconditioner; default) or R5c_groups_ordinates.txt (c = every arm with sn_precond_stencil = sou, R-F12)
"""
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_decks_R5 import AU, FAMS, ORDS, T_UNIT, TCORE  # noqa: E402
from rad_common import MOD  # noqa: E402

RUNS = os.path.join(MOD, "runs", "R5")
PFX = sys.argv[2] if len(sys.argv) > 2 else "a"
OUT = os.path.join(MOD, "results", "tables", "R5_groups_ordinates.txt" if PFX == "a" else "R5c_groups_ordinates.txt")
ARMS_A = ["G1", "P3", "L3", "E3", "L4", "E4", "L6", "E8", "L10", "L18", "F10", "F18"]


def dump(d, which):
    fs = sorted(glob.glob(os.path.join(d, "parthenon.out0.0*.phdf"))) + [os.path.join(d, "parthenon.out0.final.phdf")]
    f = fs[which]
    with h5py.File(f, "r") as h:
        p = np.array(h["prim"])
        x = np.array(h["Locations/x"])
        y = np.array(h["Locations/y"])
        z = np.array(h["Locations/z"])
        t = float(h["Info"].attrs["Time"])
        gam = float(dict(h["Params"].attrs)["radiation/gamma"])
    xc = 0.5 * (x[:, 1:] + x[:, :-1])
    yc = 0.5 * (y[:, 1:] + y[:, :-1])
    zc = 0.5 * (z[:, 1:] + z[:, :-1])
    X = np.broadcast_to(xc[:, None, None, :], p[:, 0].shape)
    Y = np.broadcast_to(yc[:, None, :, None], p[:, 0].shape)
    Z = np.broadcast_to(zc[:, :, None, None], p[:, 0].shape)
    return dict(t=t, T=(p[:, 4] / p[:, 0] * T_UNIT).ravel(), e=(p[:, 4] / (gam - 1)).ravel(),
                X=X.ravel() / AU, Y=Y.ravel() / AU, Z=Z.ravel() / AU)


def cost(d):
    s = open(os.path.join(d, "run.log"), errors="replace").read()
    w = [float(v) for c, v in re.findall(r"^cycle=(\d+) .*wsec_step=([0-9.e+-]+)", s, re.M) if int(c) >= 2]
    it = sum(int(v) for v in re.findall(r"converged in (\d+) iters", s))
    nc = len(re.findall(r"^cycle=", s, re.M)) - 1
    nf = len(re.findall(r"NOT converged|not converged", s))
    return (np.mean(w) if w else float("nan")), it / max(nc, 1), nf


def complete(d):
    c = os.path.join(d, "cmd.txt")
    return os.path.exists(os.path.join(d, "parthenon.out0.final.phdf")) and os.path.exists(c) and "rc 0 " in open(c).read()


def failure(d):
    if not os.path.exists(os.path.join(d, "run.log")):
        return "not run"
    if "rc " not in open(os.path.join(d, "cmd.txt")).read():
        return "STILL RUNNING"
    s = open(os.path.join(d, "run.log"), errors="replace").read()
    nc = len(re.findall(r"NOT CONVERGED", s))
    dis = len(re.findall(r"RECURRENCE DISAGREES", s))
    cyc = re.findall(r"^cycle=(\d+)", s, re.M)
    why = "NaN guard" if "NaN guard" in s else ("trust audit (recurrence vs true residual)" if dis else "abort")
    it = [int(v) for v in re.findall(r"converged in (\d+) iters", s)]
    return (f"{why} at cycle {cyc[-1] if cyc else '?'}; NOT CONVERGED x{nc}, recurrence disagreements x{dis}; Sn iterations "
            f"per step mean {np.mean(it):.0f} max {max(it)}")


def crossing(x, T, Tq):
    """Outermost |x| (AU) on the +x side where T >= Tq (linear interpolation between cells)."""
    s = x > 0
    xs, Ts = x[s], T[s]
    o = np.argsort(xs)
    xs, Ts = xs[o], Ts[o]
    i = np.nonzero(Ts >= Tq)[0]
    if i.size == 0 or i[-1] == xs.size - 1:
        return float("nan")
    j = i[-1]
    return float(xs[j] + (Tq - Ts[j]) * (xs[j + 1] - xs[j]) / (Ts[j + 1] - Ts[j]))


def main(mode):
    L = ["# R5 group-count / edge (R5a, 1-D) and ordinate (R5b, 3-D) convergence (scripts/analyze_R5.py)",
         "# envelope T_gas vs the reference; Q = envelope heat int(e - e0) relative to the reference; x20/x50 = where T "
         "falls to 20/50 K [AU]; wall = s/cycle (1 rank 1-D; 27 ranks 3-D), it = Sn iterations/cycle"]
    for Tc in TCORE:
        for which, lab in ((2, "mid-run"), (-1, "end")):
            ref = None
            rows = []
            for a in ARMS_A[::-1]:
                d = os.path.join(RUNS, f"{PFX}_{a}_T{int(Tc)}")
                if not complete(d):
                    rows.append(f"  {a:4s}: FAILED -- {failure(d)}")
                    continue
                D = dump(d, which)
                D0 = dump(d, 0)
                env = (np.abs(D["X"]) > 3.0) & (np.abs(D["X"]) < 18.0)
                Q = float((D["e"] - D0["e"])[env].sum())
                if ref is None:
                    ref = (D, Q)
                    refname = a
                r = np.abs(D["T"][env] / ref[0]["T"][env] - 1.0)
                w, it, nf = cost(d)
                n = FAMS[a][0] if a in FAMS else 1
                rows.append(f"  {a:4s} ng {n:2d}: max |dT/T| {r.max():.3e}  median {np.median(r):.3e}  Q/Q_ref "
                            f"{Q / ref[1]:.4f}  x20 {crossing(D['X'], D['T'], 20.0):6.3f}  x50 "
                            f"{crossing(D['X'], D['T'], 50.0):6.3f}  T(3 AU) {np.interp(3.0, *_side(D)):7.2f} K  wall "
                            f"{w:.3f}  it {it:.0f}" + (f"  NOT-CONVERGED {nf}" if nf else ""))
            L.append(f"R5a T_core {Tc:.0f} K, {lab} (t = {ref[0]['t']:.3e} code), reference {refname}:")
            L += rows[::-1]
    # R5b
    ref = None
    rows = []
    for nz, npsi in (ORDS[::-1] if PFX == "c" else ()):
        d = os.path.join(RUNS, f"b_o{nz}x{npsi}")
        if not complete(d) and os.path.isdir(d + "L"):
            d = d + "L"
        if not complete(d):
            rows.append(f"  {nz:2d}x{npsi:2d}: not complete")
            continue
        D = dump(d, -1)
        D0 = dump(d, 0)
        R = np.sqrt(D["X"] ** 2 + D["Y"] ** 2 + D["Z"] ** 2)
        sh = (R > 3.0) & (R < 5.5)
        Q = float((D["e"] - D0["e"])[sh].sum())
        if ref is None:
            ref = (D, Q)
        r = np.abs(D["T"][sh] / ref[0]["T"][sh] - 1.0)
        bins = np.arange(3.0, 5.51, 0.25)
        an = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (R >= lo) & (R < hi)
            an.append(np.std(D["T"][s]) / np.mean(D["T"][s]))
        w, it, nf = cost(d)
        rows.append(f"  {nz:2d}x{npsi:2d} ({nz * npsi:3d} ordinates): max |dT/T| {r.max():.3e}  median {np.median(r):.3e}"
                    f"  Q/Q_ref {Q / ref[1]:.4f}  shell rms(T)/<T> {np.mean(an):.3e} (max {np.max(an):.3e})  wall "
                    f"{w:.2f}  it {it:.0f}" + (f"  NOT-CONVERGED {nf}" if nf else ""))
    if rows:
        L.append(f"R5b T_core 300 K, 3-D, end (t = {ref[0]['t']:.3e} code), shell 3-5.5 AU, reference 12x24, groups P3:")
        L += rows[::-1]
    txt = "\n".join(L) + "\n"
    if mode == "run":
        open(OUT, "w").write(txt)
        print(txt)
    else:
        ok = open(OUT).read() == txt
        print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
        sys.exit(0 if ok else 1)


def _side(D):
    s = D["X"] > 0
    o = np.argsort(D["X"][s])
    return D["X"][s][o], D["T"][s][o]


if __name__ == "__main__":
    main(sys.argv[1])
