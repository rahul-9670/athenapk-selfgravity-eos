#!/usr/bin/env python3
"""GATE G11 -- the Fleck linearisation buys unconditional stability.

alpha=1 (backward Euler) must be well behaved on a stiff deck where alpha=0 (the explicit
coupling G9 used) must not. A gate only its intended configuration can pass is not evidence,
so the alpha0 arm is REQUIRED to misbehave: if the explicit scheme were fine here too, the
Fleck machinery would be buying nothing.

"Well behaved" is made specific: the temperature history must be MONOTONE (a relaxation toward
equilibrium cannot overshoot and come back), must stay positive, and must land on the coupled
equilibrium that conserves e_gas + (c/chat)Er.
"""
import sys, glob, os
import numpy as np, h5py

def series(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf"))):
        with h5py.File(p, "r") as f:
            P = f["Params"].attrs
            w = float(P["radiation/c"]) / float(P["radiation/chat"])
            gam = float(P["Hydro/AdiabaticIndex"]); arad = float(P["radiation/arad"])
            prim = np.array(f["prim"])
            rho = prim[:, 0].ravel(); pr = prim[:, 4].ravel()
            Er = np.array(f["rad.Er"]).ravel()
            out.append(dict(t=float(f["Info"].attrs["Time"]), w=w, gam=gam, arad=arad,
                            T=(pr / rho).mean(), rho=rho.mean(),
                            tot=(pr / (gam - 1.0) + w * Er).mean(), Er=Er.mean()))
    return out

def solve_T(rho, gam, arad, w, E0):
    f = lambda T: rho * T / (gam - 1.0) + w * arad * T ** 4 - E0
    lo, hi = 0.0, 1.0
    while f(hi) < 0.0: hi *= 2.0
    for _ in range(300):
        m = 0.5 * (lo + hi)
        if f(m) < 0.0: lo = m
        else: hi = m
    return 0.5 * (lo + hi)

def parity_split(d):
    """Amplitude of the grid-scale CHECKERBOARD mode in the final dump.

    This is the sharpest discriminator in the gate, and it was found by chasing an apparent
    contradiction rather than designed in: the unstable arm's density was changing while
    max|v| was EXACTLY 0.000e+00 in all three components, which is impossible for a flow.

    The resolution: the explicit coupling's instability selects the odd-even mode -- the very
    (i+j+k)&1 parity the red-black sweep colours by. Measured on the final dump, pressure is
    0.413691 on even cells and 0.938802 on odd, with ZERO spread within each parity. A perfect
    checkerboard is in the null space of the centred pressure gradient (both neighbours in any
    direction carry the same value), so the net force is exactly zero and momentum never moves
    -- while the Riemann solver still produces mass flux across the pressure jump, which is why
    density develops structure anyway.

    An instability that leaves velocity bit-zero would be invisible to any gate watching
    momentum. This one watches the mode itself.
    """
    p = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))[-1]
    with h5py.File(p, "r") as f:
        pres = np.array(f["prim"])[0, 4]
    k, j, i = np.indices(pres.shape)
    par = (i + j + k) & 1
    e, o = pres[par == 0].mean(), pres[par == 1].mean()
    return abs(e - o) / (0.5 * (e + o))


def grade(d, expect_stable):
    s = series(d)
    if len(s) < 3: return False, "need >=3 dumps in %s (found %d)" % (d, len(s))
    T = np.array([x["T"] for x in s])
    finite = np.all(np.isfinite(T)) and np.all(T > 0)
    # Monotone to within a hair: a true relaxation never reverses direction.
    dT = np.diff(T)
    mono = np.all(dT <= 1e-12 * np.abs(T[:-1])) or np.all(dT >= -1e-12 * np.abs(T[:-1]))
    Teq = solve_T(s[-1]["rho"], s[-1]["gam"], s[-1]["arad"], s[-1]["w"], s[0]["tot"])
    err = abs(T[-1] - Teq) / Teq if finite else float("inf")
    cons = abs(s[-1]["tot"] - s[0]["tot"]) / s[0]["tot"] if finite else float("inf")
    print("    T history: %s" % "  ".join("%.6g" % x for x in T[:8]))
    chk = parity_split(d)
    print("    finite&positive=%s  monotone=%s  T_end err vs equilibrium %.3e  conservation %.3e"
          % (finite, mono, err, cons))
    print("    checkerboard (odd-even pressure split): %.3e" % chk)
    stable = bool(finite and mono and err < 1.0e-3 and cons < 1.0e-6 and chk < 1.0e-8)
    if expect_stable:
        return stable, ("stable, monotone, on the equilibrium" if stable
                        else "alpha=1 FAILED to be stable -- the Fleck factor is not working")
    return (not stable), ("explicit arm misbehaves as required (this is the discrimination)"
                          if not stable else
                          "explicit arm was ALSO fine -- this deck is not stiff enough to "
                          "demonstrate anything, and the gate proves nothing")

if __name__ == "__main__":
    print("GATE G11 -- Fleck linearisation vs the explicit coupling")
    fails = 0
    for name, d, exp in [("alpha1", "arm_alpha1", True), ("alpha0", "arm_alpha0", False)]:
        print("  --- arm %s (expect %s) ---" % (name, "STABLE" if exp else "UNSTABLE"))
        ok, msg = grade(d, exp)
        print("    %s : %s" % ("PASS" if ok else "FAIL", msg))
        if not ok: fails += 1
    print("G11 %s (%d failure%s)" % ("PASS" if fails == 0 else "FAIL", fails, "" if fails == 1 else "s"))
    sys.exit(0 if fails == 0 else 1)
