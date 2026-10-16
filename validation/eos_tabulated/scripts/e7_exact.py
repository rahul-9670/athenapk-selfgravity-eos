#!/usr/bin/env python
"""E7: exact solutions of the tabulated-EOS Riemann problems (both eos_interp modes) -> results/tables/E7_exact.json.
States are (rho [g/cc], T [K], u [km/s]); the IC energy is esp_of_T exactly as the eos_test pgen sets it."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E
import riemann_exact as RX

PROBLEMS = {
    "R1_molecular": dict(L=(1e-14, 300.0, 0.0), R=(1e-15, 30.0, 0.0),
                         what="cold molecular shock tube across H2 rotational excitation (30-300 K)"),
    "R2_dissociation": dict(L=(1e-9, 4000.0, 0.0), R=(1e-10, 1000.0, 0.0),
                            what="shock tube with a partially dissociated driver (rarefaction through recombination)"),
    "R3_dissoc_shock": dict(L=(1e-10, 100.0, 16.0), R=(1e-10, 100.0, -16.0),
                            what="colliding molecular streams +-16 km/s: two shocks heating the gas into H2 dissociation (T* 2530 K, compression 19)"),
    "R4_ionization": dict(L=(1e-8, 3.0e4, 0.0), R=(1e-9, 6000.0, 0.0),
                          what="shock tube with an ionized driver (rarefaction through H recombination)"),
}
VU_KMS = E.V_U / 1e5


def run(names=None, modes=("p_over_rho", "p")):
    t = E.load()
    out = {}
    for mode in modes:
        eos = RX.TableEOS(t, mode)
        for name, pr in PROBLEMS.items():
            if names and name not in names: continue
            t0 = time.time()
            st = []
            for (rc, T, ukms) in (pr["L"], pr["R"]):
                rho = rc / E.RHO_U
                st.append((rho, ukms / VU_KMS, float(eos.esp_of_T(rho, T))))
            so = RX.solve(eos, *st)
            rec = dict(mode=mode, what=pr["what"], L=st, R=st and st[1], P=so["P"], u=so["u"])
            rec["L"] = st[0]; rec["R"] = st[1]
            speeds = []
            for side in ("L", "R"):
                d = so[side]
                rec[side + "star"] = {k: (float(v) if not isinstance(v, str) else v) for k, v in d.items()}
                rec[side + "star"]["T"] = float(10 ** eos.logT(d["rho"], d["e"]))
                if d["type"] == "shock":
                    speeds.append(abs(d["S"]))
                else:
                    speeds += [abs(d["head"]), abs(d["tail"])]
                    W = so["W" + side]
                    rec[side + "Gmin"] = RX.fundamental_derivative_min(eos, W, d["rho"])
            # convexity along the Hugoniot is not needed for admissibility here; report G at the shocked state
            vmax = max(speeds)
            rec["tlim"] = 0.40 / vmax
            rec["vmax"] = vmax
            for side in ("L", "R"):
                rec[side + "_T"] = float(10 ** eos.logT(rec[side][0], rec[side][2]))
            out[f"{name}|{mode}"] = rec
            ls, rs = rec["Lstar"], rec["Rstar"]
            print(f"{name:16s} {mode:10s} P* {so['P']:.6e} u* {so['u']:+.6e} | L: {ls['type']:11s} rho* {ls['rho']:.5e} T* {ls['T']:8.1f} K"
                  f" | R: {rs['type']:11s} rho* {rs['rho']:.5e} T* {rs['T']:8.1f} K | Gmin L {rec.get('LGmin')} R {rec.get('RGmin')}"
                  f" | vmax {vmax:.4g} tlim {rec['tlim']:.4g}  ({time.time() - t0:.0f} s)", flush=True)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    modes = ("p_over_rho", "p")
    if args and args[0].startswith("--modes="):
        modes = tuple(args[0].split("=", 1)[1].split(",")); args = args[1:]
    res = run(args or None, modes)
    dst = os.environ.get("E7_EXACT", os.path.join(os.path.dirname(__file__), "..", "results", "tables", "E7_exact.json"))
    old = json.load(open(dst)) if os.path.exists(dst) else {}
    old.update(res)
    json.dump(old, open(dst, "w"), indent=1, default=float)
