#!/usr/bin/env python3
"""GATE G10 -- scattering must not exchange energy with the gas.

Both arms must land on the SAME coupled equilibrium and conserve e_gas + (c/chat)Er. If the
energy accounting used the TOTAL extinction instead of chi_a, the kappa_s arm would equilibrate
somewhere else entirely (by 1 + chi_s/chi_a, measured exactly in test_sn_units.cpp check 5).
"""
import sys, glob, os, re
import numpy as np, h5py


def solver_tol(deck="deck.in"):
    """sn_rel_tol, read from the DECK rather than hardcoded, so the two cannot drift apart."""
    with open(deck) as f:
        m = re.search(r"^\s*sn_rel_tol\s*=\s*([0-9eE.+-]+)", f.read(), re.M)
    if not m:
        raise SystemExit("check: sn_rel_tol not found in %s -- refusing to guess" % deck)
    return float(m.group(1))


def cons_tol(ncycles, deck="deck.in"):
    """Conservation tolerance DERIVED from the solve, not chosen.

    The coupling is charged from the same eta and chi_a the sweep applied, so the discrete
    moment equation is an identity and the only error is that Er is converged to sn_rel_tol
    rather than exactly. That error accumulates once per cycle, so the drift scales as
    ncycles * sn_rel_tol -- VERIFIED, not assumed: job 2729191 tightened sn_rel_tol 100x at an
    identical 182 cycles and the drift fell 82x (4.05e-09 -> 4.94e-11).

    A fixed 1e-8 was the first version and it failed G9 at 1.18e-08 for no better reason than
    that the convergence criterion had changed from the iterate increment to the true residual,
    which is a different measure of the same solve. 10x the expected accumulation keeps ~15x
    margin on the measured value while staying six orders of magnitude tighter than any real
    accounting bug: charging the gas with the TOTAL extinction instead of chi_a moves the
    answer by 1 + chi_s/chi_a, i.e. O(1).
    """
    return min(1.0e-4, 10.0 * max(1, ncycles) * solver_tol(deck))


RTOL_MATCH = 1.0e-6   # RTOL_CONS is derived per run; see cons_tol()

def read(path):
    with h5py.File(path, "r") as f:
        P = f["Params"].attrs
        w = float(P["radiation/c"]) / float(P["radiation/chat"])
        gam = float(P["Hydro/AdiabaticIndex"])
        prim = np.array(f["prim"])
        return dict(w=w, gam=gam, t=float(f["Info"].attrs["Time"]),
                    ncyc=int(f["Info"].attrs["NCycle"]),
                    rho=prim[:, 0].ravel(), p=prim[:, 4].ravel(),
                    Er=np.array(f["rad.Er"]).ravel())

def arm(d):
    du = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))
    if len(du) < 2: return None, "need >=2 dumps in %s" % d
    a, b = read(du[0]), read(du[-1])
    if a["t"] != 0.0 or b["ncyc"] <= 0: return None, "dumps did not advance"
    gm1 = b["gam"] - 1.0
    t0 = (a["p"] / gm1 + a["w"] * a["Er"]).mean()
    t1 = (b["p"] / gm1 + b["w"] * b["Er"]).mean()
    T1 = (b["p"] / b["rho"]).mean()
    return dict(T=T1, Er=b["Er"].mean(), cons=abs(t1 - t0) / t0, t=b["t"], ncyc=b["ncyc"],
                rtol=cons_tol(b["ncyc"])), None

if __name__ == "__main__":
    print("GATE G10 -- scattering does not exchange energy with the gas")
    res, fails = {}, 0
    for name, d in [("ks0", "arm_ks0"), ("ks2", "arm_ks2")]:
        r, err = arm(d)
        if r is None:
            print("  --- arm %s ---\n    FAIL : %s" % (name, err)); fails += 1; continue
        res[name] = r
        ok = r["cons"] < r["rtol"]
        print("  --- arm %s ---" % name)
        print("    t = %.4g  cycle = %d   T = %.9f   Er = %.9f   conservation drift %.3e (tol %.2e)"
              % (r["t"], r["ncyc"], r["T"], r["Er"], r["cons"], r["rtol"]))
        print("    %s : conservation" % ("PASS" if ok else "FAIL"))
        if not ok: fails += 1
    if len(res) == 2:
        dT = abs(res["ks2"]["T"] - res["ks0"]["T"]) / res["ks0"]["T"]
        # If the gas were charged with total chi, chi_s = 2 against chi_a = 0.5 would move the
        # equilibrium by a factor of 5 in the emission balance -- so this is a wide-open test.
        ok = dT < RTOL_MATCH
        print("  --- invariance ---")
        print("    T(kappa_s=0) = %.9f   T(kappa_s=2) = %.9f   relative difference %.3e (tol %.0e)"
              % (res["ks0"]["T"], res["ks2"]["T"], dT, RTOL_MATCH))
        print("    %s : equilibrium is independent of kappa_s" % ("PASS" if ok else "FAIL"))
        if not ok: fails += 1
    else:
        fails += 1
    print("G10 %s (%d failure%s)" % ("PASS" if fails == 0 else "FAIL", fails, "" if fails == 1 else "s"))
    sys.exit(0 if fails == 0 else 1)
