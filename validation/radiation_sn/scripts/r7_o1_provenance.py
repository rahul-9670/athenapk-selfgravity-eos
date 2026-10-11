#!/usr/bin/env python
"""R7-O1: opacity-table provenance. Regenerate the production table (v9, --evap pollack) and the previous
production table (v7_f23, --evap bl94) with the generator in the validation worktree, using the recipe
recorded in DEV_LOG 2026-09-22 (afternoon), and compare md5 against the committed binaries.

usage:  r7_o1_provenance.py run     # regenerate into runs/R7_O1/<arm>/, write results/tables/R7_O1_provenance.txt
        r7_o1_provenance.py check   # re-derive the table from the run dirs and compare byte for byte
"""
import hashlib, os, platform, subprocess, sys, datetime
import os as _os
_REPO = _os.environ.get("REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../../..")))  # repository root

MOD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = _REPO + "/src/radiation"
GEN = os.path.join(SRC, "gen_opacity_table.py")
PY = sys.executable
RUNS = os.path.join(MOD, "runs", "R7_O1")
TABLE = os.path.join(MOD, "results", "tables", "R7_O1_provenance.txt")
RECIPE = ["--nr", "101", "--nT", "233", "--rho_min", "1e-20", "--rho_max", "1",
          "--T_min", "0.4755370", "--T_max", "299920"]
# arm -> (extra generator args, committed reference binary)
ARMS = {
    "v9_pollack_fmt3": (["--evap", "pollack", "--format", "3"], "opacity_table_v9_pollack_fmt3.bin"),
    "v7_bl94_fmt3":    (["--evap", "bl94", "--format", "3"], "opacity_table_v7_f23_fmt3.bin"),
    "v7_bl94_fmt2":    (["--evap", "bl94", "--format", "2"], "opacity_table_v7_f23_fmt2.bin"),
}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def first_diff(a, b):
    A, B = open(a, "rb").read(), open(b, "rb").read()
    if A == B:
        return "-"
    n = min(len(A), len(B))
    i = next((k for k in range(n) if A[k] != B[k]), n)
    return f"byte {i} (sizes {len(A)} vs {len(B)})"


def table_text():
    lines = ["# R7-O1 opacity-table provenance (written by scripts/r7_o1_provenance.py)",
             f"# generator {GEN} md5 {md5(GEN)}",
             "# recipe " + " ".join(RECIPE),
             "# arm | generator args | regenerated md5 | reference | reference md5 | bytes | first difference | verdict"]
    for arm, (extra, ref) in ARMS.items():
        out = os.path.join(RUNS, arm, "table.bin")
        refp = os.path.join(SRC, ref)
        m, r = md5(out), md5(refp)
        lines.append(" | ".join([arm, " ".join(extra), m, ref, r, str(os.path.getsize(out)),
                                 first_diff(out, refp), "BIT-IDENTICAL" if m == r else "DIFFERS"]))
    return "\n".join(lines) + "\n"


def run():
    for arm, (extra, _) in ARMS.items():
        d = os.path.join(RUNS, arm)
        os.makedirs(d, exist_ok=True)
        cmd = [PY, "-I", GEN, "--out", os.path.join(d, "table.bin")] + RECIPE + extra
        with open(os.path.join(d, "gen.log"), "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=d)
        with open(os.path.join(d, "cmd.txt"), "w") as f:
            f.write(f"date {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                    f"node {platform.node()}\njob none (front-end, python only)\n"
                    f"generator md5 {md5(GEN)}\ncommand {' '.join(cmd)}\n")
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    open(TABLE, "w").write(table_text())
    print(open(TABLE).read())


def check():
    ok = open(TABLE).read() == table_text()
    print("check:", "BYTE-IDENTICAL" if ok else "DIFFERS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    {"run": run, "check": check}[sys.argv[1]]()
