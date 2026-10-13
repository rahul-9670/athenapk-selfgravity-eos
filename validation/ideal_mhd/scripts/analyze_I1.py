#!/usr/bin/env python
"""I1 / I1b analysis: L1 error of the linear MHD waves after one period, convergence order.

Reads runs/I1/<name>/linearwave-errors.dat (written by pgen linear_wave_mhd, UserWorkAfterLoop):
RMS over the 8 conserved components of the volume-averaged L1 error. Order between successive N:
log2(e(N/2)/e(N)) (the N ladders double).  Writes results/tables/I1.{json,txt}.
  analyze_I1.py           make the tables
  analyze_I1.py --check   recompute and compare with the saved json (exit 1 on any difference > 1e-12 rel.)
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, math, os, re, sys

M = _VT + "/ideal_mhd"
PAT = re.compile(r"I1_(\d)d_(\w+?)_N(\d+)(_lb_nd|_lb|_dc)?$")
COLS = ["Nx1", "Nx2", "Nx3", "Ncycle", "RMS", "d", "M1", "M2", "M3", "E", "B1", "B2", "B3"]


def collect():
    out = {}
    for d in sorted(glob.glob(f"{M}/runs/I1/I1_*")):
        m = PAT.match(os.path.basename(d))
        f = os.path.join(d, "linearwave-errors.dat")
        if not m or not os.path.exists(f):
            continue
        rows = [l.split() for l in open(f) if l.strip() and not l.startswith("#")]
        if len(rows) != 1:
            print("WARN", d, len(rows), "rows")
        r = dict(zip(COLS, map(float, rows[-1][:len(COLS)])))
        dim, wave, N, var = int(m.group(1)), m.group(2), int(m.group(3)), (m.group(4) or "")[1:] or "prod"
        out.setdefault(f"{dim}d_{wave}_{var}", {})[N] = {"RMS": r["RMS"], "ncycle": int(r["Ncycle"])}
    res = {}
    for key, ladder in sorted(out.items()):
        Ns = sorted(ladder)
        rows = []
        for i, N in enumerate(Ns):
            e = ladder[N]["RMS"]
            o = math.log2(ladder[Ns[i - 1]]["RMS"] / e) if i and Ns[i - 1] * 2 == N and e > 0 else None
            rows.append({"N": N, "RMS_L1": e, "order": o, "ncycle": ladder[N]["ncycle"]})
        res[key] = rows
    return res


def text(res):
    lines = ["I1 linear waves, RMS L1 error after one period (amp 1e-6), order = log2 ratio of successive N",
             f"{'case':<22} " + " ".join(f"{'N':>5} {'RMS L1':>10} {'order':>6}" for _ in range(1))]
    for key, rows in res.items():
        for r in rows:
            o = f"{r['order']:6.2f}" if r["order"] is not None else "     -"
            lines.append(f"{key:<22} {r['N']:5d} {r['RMS_L1']:10.3e} {o}  ({r['ncycle']} cycles)")
        lines.append("")
    return "\n".join(lines)


def main():
    res = collect()
    js = f"{M}/results/tables/I1.json"
    if "--check" in sys.argv:
        old = json.load(open(js))
        bad = 0
        for k, rows in old.items():
            for a, b in zip(rows, res.get(k, [])):
                if a["N"] != b["N"] or abs(a["RMS_L1"] - b["RMS_L1"]) > 1e-12 * abs(a["RMS_L1"]):
                    print("DIFF", k, a, b); bad += 1
        print("check:", "OK" if not bad and old.keys() == res.keys() else f"{bad} differences / key mismatch")
        sys.exit(1 if bad or old.keys() != res.keys() else 0)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    t = text(res)
    open(js.replace(".json", ".txt"), "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
