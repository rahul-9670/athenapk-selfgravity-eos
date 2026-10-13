#!/usr/bin/env python
"""I2 analysis: circularly polarised Alfven wave (Toth 2000).

(a) one-period runs: RMS L1 error vs the initial condition from cpaw-errors.dat (written by pgen cpaw), order =
    log2 ratio of successive N.
(b) I2_2d_long16_N32: per dump (every period) the complex amplitude of the transverse field in the wave frame,
    w = b_z' + i b_y' = A exp(i (k x' - phi)); A/b_perp (numerical dissipation) and phi (phase error; exact = 0 at
    whole periods). 2D wave frame: ang_2 = 0, ang_3 = atan(2), k = 2 pi / lambda, lambda = 2/sqrt5.
  analyze_I2.py [--check] -> results/tables/I2.{json,txt}
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, math, os, re, sys
import h5py
import numpy as np

M = _VT + "/ideal_mhd"
BPERP = 0.1
PAT = re.compile(r"I2_(\d)d_(\w+?)_N(\d+)$")


def ladder():
    out = {}
    for d in sorted(glob.glob(f"{M}/runs/I2/I2_*")):
        m = PAT.match(os.path.basename(d))
        f = os.path.join(d, "cpaw-errors.dat")
        if not m or not os.path.exists(f) or m.group(2).startswith("long"):
            continue
        rows = [l.split() for l in open(f) if l.strip() and not l.startswith("#")]
        out.setdefault(f"{m.group(1)}d_{m.group(2)}", {})[int(m.group(3))] = float(rows[-1][4])
    res = {}
    for k, lad in out.items():
        Ns = sorted(lad)
        res[k] = [{"N": N, "RMS_L1": lad[N],
                   "order": math.log2(lad[Ns[i - 1]] / lad[N]) if i and Ns[i - 1] * 2 == N else None}
                  for i, N in enumerate(Ns)]
    return res


def longrun(d, dim=2):
    sgn = -1.0 if "left" in os.path.basename(d) else 1.0   # dir = 2: b_z' = -b_perp cos(k x')
    if dim == 3:
        return []   # 3D wave frame not decomposed (the 3D ladder is scored by the L1 error only)
    a3 = math.atan(2.0)
    lam = 2.0 / math.sqrt(5.0)
    k = 2 * math.pi / lam
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf"))):
        with h5py.File(f, "r") as h:
            t = float(h["Info"].attrs["Time"])
            c = h["cons"][()]
            x = h["Locations"]["x"][()]
            y = h["Locations"]["y"][()]
        num = 0j
        n = 0
        for b in range(c.shape[0]):
            X = 0.5 * (x[b, 1:] + x[b, :-1])[None, :]
            Y = 0.5 * (y[b, 1:] + y[b, :-1])[:, None]
            xp = X * math.cos(a3) + Y * math.sin(a3)
            b1, b2, b3 = c[b, 5, 0], c[b, 6, 0], c[b, 7, 0]
            byp = -b1 * math.sin(a3) + b2 * math.cos(a3)
            w = sgn * b3 + 1j * byp   # IC: by' = b_perp sin(k x'), bz' = b_perp cos(k x')  -> w = b_perp exp(i k x')
            num += (w * np.exp(-1j * k * xp)).sum()
            n += w.size
        z = num / n
        rows.append({"t": round(t, 6), "periods": round(t / lam, 4), "A_rel": float(abs(z) / BPERP),
                     "phase": float(-np.angle(z))})
    return rows


def main():
    res = {"ladder": ladder()}
    d = f"{M}/runs/I2/I2_2d_long16_N32"
    if glob.glob(d + "/*.phdf"):
        res["long16_N32"] = longrun(d)
    # amplitude / phase after one period for every 2D run (separates dissipation from dispersion, which the L1 norm mixes)
    res["ap_1period"] = {}
    for d in sorted(glob.glob(f"{M}/runs/I2/I2_2d_*")):
        name = os.path.basename(d)
        if "long" not in name and glob.glob(d + "/*.phdf"):
            r = longrun(d)
            res["ap_1period"][name] = {"A0": r[0]["A_rel"], "A1": r[-1]["A_rel"], "A1_A0": r[-1]["A_rel"] / r[0]["A_rel"],
                                       "phase": r[-1]["phase"]}
    js = f"{M}/results/tables/I2.json"
    if "--check" in sys.argv:
        old = json.load(open(js))
        ok = old == json.loads(json.dumps(res))
        print("check:", "OK" if ok else "DIFF"); sys.exit(0 if ok else 1)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    L = ["I2 CPAW (Toth 2000): RMS L1 vs the initial condition after one period; order = log2 ratio"]
    for k, rows in res["ladder"].items():
        L += [f"  {k:<10} N {r['N']:4d}  {r['RMS_L1']:.3e}  " + (f"{r['order']:.2f}" if r["order"] else "-") for r in rows]
    if "long16_N32" in res:
        L.append("  long run 2D N 32: periods, A/b_perp, phase error [rad]")
        L += [f"    {r['periods']:6.2f}  {r['A_rel']:.5f}  {r['phase']:+.4e}" for r in res["long16_N32"]]
    L.append("  one period, 2D: A(t=0)/b_perp (discrete-curl init), A(1)/A(0), phase error [rad]")
    L += [f"    {k:<20} {v['A0']:.6f} {v['A1_A0']:.6f} {v['phase']:+.3e}" for k, v in res["ap_1period"].items()]
    open(js.replace(".json", ".txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
