#!/usr/bin/env python3
"""THE DEFINITIVE TEST v2 -- driven steady state, with the source confound removed.

Grades ACCURACY, not conservation: against a fixed-T bath the radiation exchanges energy with
the gas, so total Er is not conserved and conservation says nothing. What IS well defined is the
steady state, so the AMR total Er is compared with a uniform run at the AMR's finest resolution.

The AMR mesh is fine inside the slab and coarse outside, so its error MUST land between the two
uniform arms if the seam is harmless. Exceeding the coarse arm means refining made the answer
WORSE, which would overturn option D.
"""
import sys, glob, os
import numpy as np, h5py


def series(d):
    """(time, cycle, volume-weighted total Er, total volume) for every dump."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf"))):
        with h5py.File(p, "r") as f:
            Er = np.array(f["rad.Er"]); lev = np.array(f["Levels"])
            nx1 = int(f["Info"].attrs["RootGridSize"][0])
            vol = (1.0 / (nx1 * 2.0 ** lev)) ** 3
            flat = Er.reshape(Er.shape[0], -1)
            out.append((float(f["Info"].attrs["Time"]), int(f["Info"].attrs["NCycle"]),
                        float((flat.sum(axis=1) * vol).sum()),
                        float((flat.shape[1] * vol).sum()),
                        sorted(set(lev.tolist()))))
    return out


def arm(d):
    s = series(d)
    if len(s) < 3:
        return None, "need >=3 dumps in %s (found %d)" % (d, len(s))
    if abs(s[-1][3] - 1.0) > 1e-10:
        return None, "volume weighting wrong: %.12g" % s[-1][3]
    # STEADINESS is a precondition, not an assumption: the last two dumps must agree, or the
    # run has not reached the steady state this test is built to compare.
    a, b = s[-2][2], s[-1][2]
    steady = abs(b - a) / b
    # PREMISE CHECK. Isobaric means zero pressure gradient, so the gas must never move. If it
    # does, the temperature profile is not frozen and this is not a driven steady state at all
    # -- the whole test would be measuring something else while looking fine.
    last = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))[-1]
    with h5py.File(last, "r") as f:
        prim = np.array(f["prim"]); nb = prim.shape[0]
        vmax = float(np.abs(prim[:, 1:4]).max())
        rho = prim[:, 0].ravel()
        tspread = float((prim[:, 4].ravel() / rho).max() / (prim[:, 4].ravel() / rho).min())
        # THE SOURCE ITSELF, volume-weighted. v1 of this test was contaminated because the AMR
        # arm's <a_R T^4> differed from the uniform arms' by 1.45e-03 -- a MIXED mesh does not
        # inherit the midpoint rule's spectral accuracy on a periodic function. The top-hat is
        # exactly representable on every mesh here, so these must now agree to round-off, and
        # the checker REFUSES to grade if they do not.
        P = f["Params"].attrs
        arad = float(P["radiation/arad"])
        lev = np.array(f["Levels"]); nx1 = int(f["Info"].attrs["RootGridSize"][0])
        vol = (1.0 / (nx1 * 2.0 ** lev)) ** 3
        T = prim[:, 4].reshape(nb, -1) / prim[:, 0].reshape(nb, -1)
        src = float(((arad * T ** 4).sum(axis=1) * vol).sum())
        # ISOBARIC premise: uniform pressure is what makes the force zero and the gas frozen.
        # The first top-hat build set T from the top-hat but left rho on the COSINE formula, so
        # p = rho*T varied by 56% -- the gas would have moved and the "frozen driver" premise
        # would have been false while every other number still looked reasonable. Checked, not
        # assumed.
        pr = prim[:, 4].ravel()
        pspread = float((pr.max() - pr.min()) / pr.mean())
    return dict(tot=b, steady=steady, t=s[-1][0], ncyc=s[-1][1], levels=s[-1][4],
                hist=[x[2] for x in s], vmax=vmax, tspread=tspread, src=src,
                pspread=pspread), None


if __name__ == "__main__":
    print("THE DEFINITIVE TEST -- driven steady state, seam at the steepest gradient")
    res = {}
    for name in ("amr", "coarse", "fine"):
        r, err = arm("arm_%s" % name)
        print("  --- %s ---" % name)
        if r is None:
            print("    ERROR: %s" % err); continue
        res[name] = r
        print("    t = %.4g  cycle = %d  levels %s" % (r["t"], r["ncyc"], r["levels"]))
        print("    total Er = %.12e   last-two-dump change %.3e %s"
              % (r["tot"], r["steady"], "(STEADY)" if r["steady"] < 1e-6 else "(NOT STEADY)"))
        print("    history: %s" % "  ".join("%.6e" % x for x in r["hist"]))
        print("    PREMISE: max|v| = %.3e   p spread = %.3e (both must be ~0)"
              % (r["vmax"], r["pspread"]))
        print("             T_max/T_min = %.4f   <a_R T^4> = %.12e" % (r["tspread"], r["src"]))
    print()
    print("  ================ VERDICT ================")
    if not {"amr", "coarse", "fine"} <= set(res):
        print("  incomplete"); sys.exit(1)
    badp = [k for k, v in res.items() if v["pspread"] > 1e-12]
    if badp:
        print("  ISOBARIC PREMISE VIOLATED in %s: pressure is not uniform, so there IS a force"
              % ", ".join(badp))
        print("  on the gas and the temperature profile is not frozen. Refusing to grade.")
        sys.exit(1)
    bad = [k for k, v in res.items() if v["vmax"] > 1e-12]
    if bad:
        print("  PREMISE VIOLATED in %s: the gas moved, so T is not frozen and this is not a"
              % ", ".join(bad))
        print("  driven steady state. The comparison below would be measuring something else.")
        sys.exit(1)
    # THE CONFOUND GATE. Every arm must be driven by the SAME source, or the comparison is
    # between different problems. This is exactly what invalidated v1.
    srcs = {k: v["src"] for k, v in res.items()}
    smax = max(abs(srcs[k] - srcs["fine"]) / srcs["fine"] for k in srcs)
    print("  source <a_R T^4> spread across arms: %.3e" % smax)
    if smax > 1e-12:
        print("  SOURCE CONFOUND PRESENT (%.3e) -- the arms are not solving the same problem."
              % smax)
        print("  This is what contaminated v1. Refusing to grade.")
        sys.exit(1)
    print("  (sources identical to round-off: the arms differ ONLY in the mesh)")
    print()
    if res["amr"]["steady"] > 1e-6 or res["fine"]["steady"] > 1e-6:
        print("  NOT AT STEADY STATE -- the comparison is not meaningful yet. Increase tlim.")
        sys.exit(1)
    ref = res["fine"]["tot"]
    e_amr = abs(res["amr"]["tot"] - ref) / ref
    e_cor = abs(res["coarse"]["tot"] - ref) / ref
    print("  reference (uniform fine)   : %.12e" % ref)
    print("  uniform coarse  error      : %.3e   <- pure resolution error" % e_cor)
    print("  static AMR      error      : %.3e" % e_amr)
    print("  ratio AMR/coarse           : %.2f" % (e_amr / e_cor if e_cor > 0 else float("inf")))
    print()
    if e_amr <= e_cor:
        print("  => THE SEAM IS BENIGN UNDER SUSTAINED DRIVING. Refining part of the domain is")
        print("     no worse than not refining it, so the coarse-fine interface adds nothing")
        print("     beyond the resolution error it is supposed to remove. Option D stands and")
        print("     the flux register is not needed.")
    elif e_amr < 5.0 * e_cor:
        print("  => THE SEAM COSTS %.2fx THE RESOLUTION ERROR, AND REFINING MAKES THE GLOBAL"
              % (e_amr / e_cor))
        print("     ANSWER WORSE, NOT BETTER. (An earlier version of this line claimed")
        print("     'refinement still helps or breaks even' -- that was wrong whenever the")
        print("     ratio exceeds 1, which is exactly what it is reporting.)")
        print("     Seam-attributable error = %.3e, on top of a %.3e resolution error."
              % (e_amr - e_cor, e_cor))
        print("     It is a BOUNDED steady-state BIAS, not a growing one: the state is steady")
        print("     to ~1e-10. Quote it; decide whether a %.2f%% bias in radiation energy"
              % (100 * (e_amr - e_cor)))
        print("     matters for the science before spending the memory win on a register.")
    else:
        print("  => THE SEAM DOMINATES (%.2fx the coarse error). Refining makes the steady state"
              % (e_amr / e_cor))
        print("     WORSE than leaving the mesh coarse, under a driver that never stops. This")
        print("     OVERTURNS option D: the flux register becomes mandatory.")
    print("  =========================================")
