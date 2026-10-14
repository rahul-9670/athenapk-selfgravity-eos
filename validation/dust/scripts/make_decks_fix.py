#!/usr/bin/env python
"""Decks for the two candidate fixes (default-off keys):
  F2 (DD2)  dust/subl_shock_hold = 1 | 2 on the D5 Noh decks (M 10, M 30; N 256, 1024, 4096; T_subl = T2). Pass if the
            plateau deficit and the recorded overshoot dT_rec FALL with N (convergent), and the key-off decks reproduce
            D5 bitwise (key-off = the D5 runs on the new binary: F2_*_hold0).
  F5 (DD5)  dust/subl_model = semenov on the D3a one-zone compression (1e-9 g/cm^3, 1250 -> 1928 K): exact
            f = f_ref s(T, rho) with Semenov's compact ramp about T5(rho) (dust.hpp SublSemenov), per cell, every dump.
Writes inputs/F/*.in, inputs/F_runs.txt, inputs/F2_meta.json (analyze_D5.py format), inputs/F5_meta.json.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import json, sys
sys.path.insert(0, _VT + "/dust/scripts")
from du_common import DU, write
import make_decks_D5 as D5
import make_decks_D3 as D3

M3 = D5.M3


def main():
    runs, meta2, meta5 = [], {}, {}
    for M in (10, 30):
        m = M3[f"D3_M{M}_N1024"]
        for N in (256, 1024, 4096):
            for hold in (0, 1, 2, "filt", "filthold2", "tau"):
                name = f"F2_M{M}_N{N}_hold{hold}" if isinstance(hold, int) else f"F2_M{M}_N{N}_{hold}"
                over = {"freeze_growth": "true"}
                if hold in (1, 2):
                    over["subl_shock_hold"] = hold
                if hold in ("filt", "filthold2"):   # 2026-10-13: dust/subl_T_filter (post-shock 2 dx noise)
                    over["subl_T_filter"] = "true"
                if hold == "filthold2":
                    over["subl_shock_hold"] = 2
                if hold == "tau":   # 2026-10-13: finite sublimation time, tau = 2e-3 code = 2.958e9 s (t_unit l/v)
                    over["subl_tau_s"] = 2.0e-3 * 2.81e16 / 1.9e4
                write(f"{DU}/inputs/F/{name}.in", D5.deck(name, m, N, float(m["T2"]), over))
                runs.append(f"F/{name} | F/{name}.in | | 1")
                meta2[name] = dict(dir="F", M=M, N=N, dT=0, hold=str(hold), T_subl=float(m["T2"]), T2=float(m["T2"]),
                                   rho2=float(m["rho2"]), e2=float(m["e2"]), S=float(m["S"]), tlim=float(m["tlim"]))
    for v, frz in (("frz", True), ("prod", False)):
        name = f"F5_D3a_semenov_{v}"
        d = D3.deck(name, 1e-9, 1250.0, 1.0, 0.875, frz)
        d = d.replace("dual_energy = true\n\n<parthenon/output0>",
                      "dual_energy = true\nsubl_model = semenov   # DD5 candidate (production: absent = fixed)\n\n"
                      "<parthenon/output0>")
        assert "subl_model = semenov" in d
        write(f"{DU}/inputs/F/{name}.in", d)
        runs.append(f"F/{name} | F/{name}.in | | 1")
        meta5[name] = dict(rho_cgs=1e-9, T0=1250.0, alpha=1.0, tlim=0.875, freeze=frz)
    write(f"{DU}/inputs/F_runs.txt", "\n".join(runs) + "\n")
    write(f"{DU}/inputs/F2_meta.json", json.dumps(meta2, indent=1))
    write(f"{DU}/inputs/F5_meta.json", json.dumps(meta5, indent=1))
    print(len(runs), "runs")


if __name__ == "__main__":
    main()
