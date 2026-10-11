#!/usr/bin/env python
"""GATE K1: BiCGSTAB solves the SAME system the validated Richardson path solves.

WHAT THIS GRADES, and what it deliberately does NOT.

(1) OPERATOR. The Stage-A gate (in-code) asserts sum|b-Ax| from SnApplyA equals sum|Ax-b| from
    SnResidual, the kernel the Richardson path was validated with. Reported here as a count.

(2) SOURCE EQUALITY, ASSERTED NOT ASSUMED. The arms are only comparable if they solve the same
    problem. matter_coupling is off, so the gas should evolve identically and the Planck source
    with it -- but "should" is how a previous AMR test spent a run comparing two different
    source problems. So `prim` is compared across arms first, and a mismatch is a FAIL of the
    gate's premise rather than of the solver.

(3) ANSWER, against a MEASURED noise floor. Comparing to a fixed "100x tolerance" threshold is
    meaningless: the tolerance is on a global L1 residual, the comparison is per-cell, and the
    map between them runs through cond(A), which is unknown. Instead the reference solver is
    re-run 1000x tighter (arm_tight) and every arm is measured against THAT. The discriminator
    is like-for-like: does BiCGSTAB deviate more than the reference solver does at the same
    nominal tolerance?

(4) BOTH MOMENTS. Er is the ZEROTH angular moment -- a 32-ordinate field projected onto one
    scalar, so two different intensity fields can share it. Fr (first moment) is compared too.
    Neither is the ordinate field itself; the statement that the FULL field solves the system
    is the in-code true-residual audit (SnKrTrue), not this script.

OUT OF SCOPE, STATED SO IT IS NOT READ IN: this deck runs at transport Courant number C ~ 3,
where every solver converges. It establishes CORRECTNESS ONLY. It says nothing about the
the collapse run's C = 15,231, which is the entire reason Krylov was written. That is gate K2.
"""
import sys, glob, re, h5py, numpy as np

FIELDS = ["rad.Er", "rad.Fr1", "rad.Fr2", "rad.Fr3"]

def load(arm, names):
    f = sorted(glob.glob(f"{arm}/parthenon.out0.*.phdf"))
    if not f: return None, None
    out = {}
    with h5py.File(f[-1], "r") as h:
        for n in names:
            if n in h: out[n] = np.array(h[n])
    return out, f[-1]

def dev(x, ref):
    return (np.abs(x - ref).sum() / max(np.abs(ref).sum(), 1e-300),
            (np.abs(x - ref) / np.maximum(np.abs(ref), 1e-300)).max())

arms = ["arm_ref", "arm_kry0", "arm_kry1", "arm_tight"]
data = {}
for a in arms:
    d, fn = load(a, FIELDS + ["prim"])
    if d is None: print(f"FAIL: {a} produced no dump"); sys.exit(1)
    data[a] = d
    print(f"{a:10s} {fn}")

ok = True

# --- (2) premise: identical source ---------------------------------------------------------
print("\n--- PREMISE: do the arms solve the same problem? (gas state must match) ---")
pref = data["arm_ref"]["prim"]
for a in arms[1:]:
    l1, mx = dev(data[a]["prim"], pref)
    good = mx < 1e-12
    print(f"  prim {a:10s} vs arm_ref: max/cell = {mx:.3e}   " +
          ("OK" if good else "*** ARMS SOLVE DIFFERENT PROBLEMS -- comparison is void ***"))
    ok &= good

# --- blindness guard: the field must have 3-D structure ------------------------------------
er = data["arm_tight"]["rad.Er"]
print(f"\n--- CONTENT: source structure ---")
sp = (er.max() - er.min()) / abs(er.mean())
print(f"  Er spread (max-min)/mean = {sp:.4f}", end="   ")
if sp < 1e-3: print("*** UNIFORM -- gate is vacuous ***"); sys.exit(1)
print("OK")
# per-axis variation: a field flat along an axis cannot test that axis's upwind stencil
e = er.reshape(er.shape[0], -1) if er.ndim < 4 else er
ax_names = {1: "k(z)", 2: "j(y)", 3: "i(x)"}
if er.ndim == 4:
    for axis, nm in ax_names.items():
        others = tuple(x for x in (1, 2, 3) if x != axis)
        prof = er.mean(axis=others)              # profile along this axis, per block
        v = (prof.max(axis=-1) - prof.min(axis=-1)).max() / abs(er.mean())
        good = v > 1e-3
        print(f"  variation along {nm}: {v:.4e}   " +
              ("OK" if good else "*** FLAT -- this transport direction is NOT graded ***"))
        ok &= good

# --- (1) operator gate ---------------------------------------------------------------------
txt = "".join(open(p, errors="ignore").read() for p in glob.glob("k1_*.out") + glob.glob("*/run.log"))

# UNUSED-PARAMETER CHECK. Parthenon prints "The following input parameters are set but unused"
# for any key the code never reads, and a mistyped key is silently the DEFAULT. This cost a
# whole K2 comparison: the decks wrote `sn_ordered_sweep = true`, which does not exist, so the
# reference arms ran the default redblack smoother and the Krylov speedup was measured against
# the weaker baseline. The warning was printed every time and read past. Now it FAILS the gate.
# R0 2026-10-08: radiation/{cfl,closure,matter_coupling} were read ONLY by the M1 path removed in 117e2fe8
# (git show 117e2fe8: the three GetOrAdd* lines are deleted); the archived decks still carry them. Exempted by
# name; any OTHER unused key still fails the gate.
RETIRED = {"radiation/cfl", "radiation/closure", "radiation/matter_coupling"}
unused = re.findall(r"set but unused:\n((?:[^\n]+\n)+?)\n", txt)
keys = sorted({k.strip() for blk in unused for k in blk.splitlines()
               if k.strip() and not k.strip().startswith("comment/")}
              - RETIRED)
print("\n--- UNUSED INPUT KEYS (a mistyped key is silently the default) ---")
if keys:
    for k in keys: print(f"  *** {k} ***")
else:
    print("  none")
nver = len(re.findall(r"\[SN/verify-op\].*PASS", txt))
nfail = len(re.findall(r"\[SN/verify-op\].*FAIL", txt))
print(f"\n--- OPERATOR: SnApplyA == SnResidual ---\n  {nver} PASS, {nfail} FAIL")
ok &= (nfail == 0 and nver > 0)
ok &= (len(keys) == 0)

# --- (3)+(4) answer vs measured control ----------------------------------------------------
print(f"\n--- ANSWER vs arm_tight (sn_rel_tol=1e-13), both moments ---")
for fld in FIELDS:
    if fld not in data["arm_ref"]: continue
    ref, tight = data["arm_ref"][fld], data["arm_tight"][fld]
    cl1, cmx = dev(ref, tight)
    print(f"  {fld}:  CONTROL arm_ref vs tight: L1={cl1:.3e} max/cell={cmx:.3e}")
    for a in ["arm_kry0", "arm_kry1"]:
        l1, mx = dev(data[a][fld], tight)
        r1 = l1 / cl1 if cl1 > 0 else np.inf
        rm = mx / cmx if cmx > 0 else np.inf
        good = (r1 < 10.0) and (rm < 10.0)
        print(f"      {a:9s}: L1={l1:.3e} ({r1:6.2f}x)  max/cell={mx:.3e} ({rm:6.2f}x)   " +
              ("PASS" if good else "*** FAIL ***"))
        ok &= good

print("\n--- CONVERGENCE ---")
for m in re.findall(r"=== ARM [^\n]*|\[SN[^\n]*(?:converged|NOT CONVERGED|BREAKDOWN)[^\n]*", txt):
    print("   ", m)
print("\nGATE K1:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
