#!/usr/bin/env python
"""R0: grade every consolidated Sn gate with its (copied) checker and write results/tables/R0_gates.txt.

usage: analyze_R0.py run     # run all checkers, write results/tables/R0_logs/<gate>_<family>.txt + R0_gates.txt
       analyze_R0.py check   # re-derive everything and compare byte for byte with the files on disk

Each checker runs in runs/R0/<GATE>/<family>/ (deck.in -> the family deck, made by this script). Checker copies in
scripts/r0_checkers/ differ from the archived originals only where stated in their headers (G8 arm_noabs; K1/K3/K5
exempt the three keys retired with M1). Verdicts are the checkers' own, verbatim; NOTE lines add measured context.
"""
import glob
import os
import re
import subprocess
import sys

from rad_common import MOD, PY
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root

R0 = os.path.join(MOD, "runs", "R0")
CHK = os.path.join(MOD, "scripts", "r0_checkers")
TABD = os.path.join(MOD, "results", "tables")
LOGD = os.path.join(TABD, "R0_logs")
# gate -> (checker, families, regex of the lines kept in the summary)
GATES = [
    ("G8", "check_g8.py", ["orig", "prod"], r"PASS :|FAIL :|relative error|^G8 "),
    ("G9", "check_g9.py", ["orig", "prod"], r"drift|  T  :|PASS :|FAIL :|^G9 "),
    ("G10", "check_g10.py", ["orig", "prod"], r"relative difference|PASS :|FAIL :|^G10 "),
    ("G11", "check_g11.py", ["orig", "prod"], r"monotone=|PASS :|FAIL :|^G11 "),
    ("K1", "check_k1.py", ["step", "sou", "prod"], r"PASS, .* FAIL|arm_kry[01] *:|^GATE"),
    ("K3", "check_k3.py", ["step", "sou", "prod"], r"PASS, .* FAIL|arm_kry[01] *:|^GATE"),
    ("K5", "check_k5.py", ["step", "sou", "prod"], r"PASS, .* FAIL|arm_kry[01] *:|^GATE"),
    ("SEAM", "check_g6h.py", ["step", "sou", "prod"], r"error|ratio AMR|=>|levels"),
    ("FLECK", "check_fleck5.py", ["orig", "prod"], r"fitted order|\|errT\|"),
    ("K4", "check_k4sou.py", ["orig", "prod"], r"^f_(sou1|step)_tau(0p2|0p5|1) |violations"),
    ("RAY", "check_ray4.py", ["orig"], r"^  z\S+ +\d+ +\||CHECKER EXIT"),
]
NOTES = {
    ("G11", "prod"): "NOTE: alpha1 non-monotone only after equilibrium: |dT|/T <= 6e-12 from cycle 7 (solver noise at "
                     "sn_rel_tol 1e-8; criterion 1e-12); T_end err 2.7e-9, checkerboard 0. Not an instability.",
}


def link_decks():
    for g, _, fams, _ in GATES:
        for f in fams:
            d = os.path.join(R0, g, f)
            if not os.path.isdir(d):
                continue
            tgt = os.path.relpath(os.path.join(MOD, "inputs", "R0", f"{g}_{f}.in"), d)
            ln = os.path.join(d, "deck.in")
            if os.path.lexists(ln):
                os.remove(ln)
            os.symlink(tgt, ln)


def grade():
    out = {}
    for g, chk, fams, rx in GATES:
        for f in fams:
            d = os.path.join(R0, g, f)
            if not os.path.isdir(d):
                out[(g, f)] = ("MISSING", "")
                continue
            if g == "SEAM":   # check_g6h.py expects exactly arm_amr / arm_coarse / arm_fine (present)
                pass
            r = subprocess.run([PY, os.path.join(CHK, chk)], cwd=d, capture_output=True, text=True)
            txt = r.stdout + r.stderr
            out[(g, f)] = (f"exit {r.returncode}", txt)
    return out


def host_and_restart():
    rows = []
    h = os.path.join(R0, "host", "host_tests.log")
    if os.path.exists(h):
        rows += ["HOST " + l.strip() for l in open(h) if l.strip() and "logs:" not in l]
    for tag, sub in (("c69ee834", "RSTB"), ("4135083d", "RSTB_fix")):
        for v in sorted(os.listdir(os.path.join(R0, sub))) if os.path.isdir(os.path.join(R0, sub)) else []:
            for name, a, b, cyc in (("cont vs cont2", "cont", "cont2", 20), ("cont vs legB", "cont", "legB", 12),
                                    ("cont vs legB", "cont", "legB", 20)):
                pa, pb = os.path.join(R0, sub, v, a), os.path.join(R0, sub, v, b)
                if not (os.path.isdir(pa) and os.path.isdir(pb)):
                    continue
                r = subprocess.run([PY, _REPO + "/validation/radiation_sn/scripts/cmp_bitwise.py",
                                    pa, pb, str(cyc)], capture_output=True, text=True)
                rows.append(f"RESTART bin {tag} variant {v:11s} {name} @{cyc}: {r.stdout.strip().splitlines()[-1]}")
    return rows


def render(res, extra):
    L = ["# R0 consolidated Sn gates (scripts/analyze_R0.py). Binary athenaPK_cpu_c69ee834 unless stated; families:",
         "#   orig/step/sou = archived deck (+ the scheme keys its submit script passed), prod = + PROD_RAD + dual keys.",
         "# Full checker outputs: results/tables/R0_logs/<gate>_<family>.txt"]
    for g, _, fams, rx in GATES:
        for f in fams:
            st, txt = res[(g, f)]
            L.append(f"== {g}/{f} ({st})")
            bins = sorted({l.split()[1][:8] for c in glob.glob(os.path.join(R0, g, f, "*", "cmd.txt")) + glob.glob(os.path.join(R0, g, f, "cmd.txt"))
                           for l in open(c) if l.startswith("bin ")})
            if bins and bins != ["c69ee834"]:
                L.append("   binaries (cmd.txt): " + " ".join(bins))
            L += ["   " + l.rstrip() for l in txt.splitlines() if re.search(rx, l)]
            if (g, f) in NOTES:
                L.append("   " + NOTES[(g, f)])
    L.append("== host tests and restart exactness")
    L += ["   " + r for r in extra]
    return "\n".join(L) + "\n"


def main(mode):
    link_decks()
    res = grade()
    extra = host_and_restart()
    files = {os.path.join(TABD, "R0_gates.txt"): render(res, extra)}
    for (g, f), (st, txt) in res.items():
        files[os.path.join(LOGD, f"{g}_{f}.txt")] = txt
    if mode == "run":
        os.makedirs(LOGD, exist_ok=True)
        for p, t in files.items():
            open(p, "w").write(t)
        print(files[os.path.join(TABD, "R0_gates.txt")])
    else:
        bad = [p for p, t in files.items() if not os.path.exists(p) or open(p).read() != t]
        print("check:", "BYTE-IDENTICAL" if not bad else "DIFFERS: " + " ".join(bad))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
