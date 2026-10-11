#!/usr/bin/env python3
"""GATE G8 -- Sn thick-equilibrium unit calibration.

Grades one arm's final dump against the ANALYTIC answer:

    Er -> a_R T^4 = 1.0 exactly   (arad_code = 1, p0 = rho0 = 1, gamma = 5/3 => T = p/rho = 1)
    Fr -> 0                       (an isotropic field carries no net flux)

REFUSES a t = 0 dump. A previous gate in this campaign graded the INITIAL CONDITION and
reported passes for legs that had crashed, because it fell back to the newest dump without
checking that the run had advanced. The time and cycle are asserted, not assumed.
"""
import sys, glob, os
import numpy as np
import h5py

RTOL_ER   = 1.0e-6       # the solve is driven to sn_rel_tol = 1e-10 and relaxes ~1e-2 / step
ATOL_FR   = 1.0e-10      # |Fr| against chat*Er = 100


def grade(rundir, expect_equilibrium, target):
    # `parthenon`, not the problem_id: AthenaPK's output prefix is fixed (CLAUDE.md documents
    # the science dumps as parthenon.out1.NNNNN.phdf) while job/problem_id selects the PROBLEM
    # GENERATOR. Lexical sort puts "final" after the numbered dumps, which is what we want.
    dumps = sorted(glob.glob(os.path.join(rundir, "parthenon.out0.*.phdf")))
    if not dumps:
        return False, "no phdf dumps in %s" % rundir
    path = dumps[-1]
    with h5py.File(path, "r") as f:
        t = float(f["Info"].attrs["Time"])
        ncycle = int(f["Info"].attrs["NCycle"])
        if t <= 0.0 or ncycle <= 0:
            return False, "graded a t=0 dump (t=%g, cycle=%d) -- run did not advance" % (t, ncycle)
        Er = np.array(f["rad.Er"]).ravel()
        Fr = np.stack([np.array(f["rad.Fr%d" % d]).ravel() for d in (1, 2, 3)])

    emin, emax, emean = Er.min(), Er.max(), Er.mean()
    frmax = np.abs(Fr).max()
    spread = (emax - emin) / emean if emean > 0 else float("inf")
    err = abs(emean - target) / target

    print("  %s" % os.path.basename(path))
    print("    t = %.6g   cycle = %d   cells = %d" % (t, ncycle, Er.size))
    print("    Er   min %.12e  max %.12e  mean %.12e" % (emin, emax, emean))
    print("    Er   relative error vs a_R T^4 = %-10.6g: %.3e   (tol %.0e)" % (target, err, RTOL_ER))
    print("    Er   cell-to-cell spread          : %.3e" % spread)
    print("    |Fr| max                          : %.3e   (tol %.0e)" % (frmax, ATOL_FR))

    if expect_equilibrium:
        ok = (err < RTOL_ER) and (spread < RTOL_ER) and (frmax < ATOL_FR)
        return ok, ("Er == a_R T^4, uniform, zero flux" if ok else
                    "err=%.3e spread=%.3e |Fr|=%.3e" % (err, spread, frmax))
    # Falsification arm: the retired m1er source is its own fixed point, so Er must STAY at
    # the initial 1e-30 and must NOT reach 1. A pass here would mean the gate cannot tell the
    # placeholder from the physical source, i.e. it grades nothing.
    ok = err > 0.5
    return ok, ("stayed away from equilibrium as required (Er_mean=%.3e)" % emean if ok else
                "REACHED equilibrium WITHOUT absorption/emission -- gate does not discriminate")


if __name__ == "__main__":
    # (name, dir, expect equilibrium, target Er = a_R T^4)
    #
    # THE THIRD ARM EXISTS BECAUSE THE FIRST TWO ARE DEGENERATE. arm_planck uses arad = 1 and
    # T = 1, which was chosen to make the expected answer exactly 1.0 with no unit derivation
    # -- and 1 is a fixed point of every power, so that arm passes identically for a_R T^4,
    # a_R T^3, T^4 with no a_R, or a_R alone. It constrains the 4*pi and the chat (both of
    # which are far from 1) and NOTHING about the temperature dependence.
    #
    # arm_t2 sets p0 = 2 (=> T = 2) and arad_code = 3, so the target is 3 * 2^4 = 48 and every
    # one of those confusions gives a different number: T^3 -> 24, T^5 -> 96, no a_R -> 16,
    # a_R alone -> 3. Same binary, same deck, two CLI overrides.
    arms = [("planck", "arm_planck", True, 1.0),
            ("noabs", "arm_noabs", False, 1.0),   # R0 2026-10-08: replaces the retired m1er arm (kappa_a = 0: Er must stay 1e-30)
            ("t2", "arm_t2", True, 48.0)]
    if len(sys.argv) > 1:
        arms = [a for a in arms if a[0] in sys.argv[1:]]
    fails = 0
    print("GATE G8 -- Sn thick-equilibrium unit calibration")
    for name, d, expect, target in arms:
        print("  --- arm %s (expect %s, target Er = %g) ---"
              % (name, "EQUILIBRIUM" if expect else "NO equilibrium", target))
        ok, msg = grade(d, expect, target)
        print("    %s : %s" % ("PASS" if ok else "FAIL", msg))
        if not ok:
            fails += 1
    print("G8 %s (%d arm failure%s)" % ("PASS" if fails == 0 else "FAIL", fails,
                                        "" if fails == 1 else "s"))
    sys.exit(0 if fails == 0 else 1)
