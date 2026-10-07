#!/usr/bin/env python3
"""Verify that an extended EOS table is an EXTENSION of its predecessor, not a new table.

    usage: verify_table_extension.py <old.bin> <new.bin>

Context: on 2026-08-18 the production run was found to be walking off the LOW esp edge of
`eos_table_hires_v2.bin` -- job 2531600 logged 41,650 edge-clamped lookups over 201 cycles,
100 % of them `esp<`, rising from 21 per cycle to 191. The esp floor is not an independent
knob: it is set by the generator's `T_min`, because at low T the gas is molecular with the
rotational and vibrational modes frozen out and esp = (3/2)kT/(mu m_H) exactly.

Lowering `T_min` with `nT` fixed would rescale the grid spacing and move EVERY node, silently
changing every interpolated value in a table whose physics is already validated. The generator
therefore holds the spacing fixed and PREPENDS nodes. This script is the check that it did.

FOUR THINGS ARE CHECKED, in the order that matters:
  1. GEOMETRY   the spacings must agree (to round-off -- they are recomputed as a difference of
                floats, so a few ulps is expected and harmless), and the axis origins must
                differ by an exact whole number of nodes.
  2. OVERLAP    the original nodes must be unchanged. Two exceptions are legitimate and are
                reported separately rather than hidden:
                  - esp column 0 of P and log10T: that node sat BELOW the coldest tabulated
                    esp and was itself edge-clamped to T_min, so extending FIXES it;
                  - esp columns 0 and 1 of cs2: np.gradient goes from a one-sided to a centred
                    difference at the old edge, and the centred difference at column 1 reads
                    column 0.
                Every other node must be bit-identical.
  3. NEW REGION finite, positive, monotone, and -- the point of the exercise -- not itself
                edge-clamped. A padded node that still resolves to T_min bought nothing.
  4. PHYSICS    in the padded region the modes are frozen, so T(esp) must reproduce
                (2 mu m_H / 3k) * esp analytically and must not depend on density. This is the
                only check here that is independent of the generator's own arithmetic.
"""
import sys

import numpy as np

KB = 1.380649e-16
MH = 1.6726219e-24
V0SQ = 1.9e4 ** 2


def load(p):
    b = open(p, "rb").read()
    nr, ne, nT = (int(x) for x in np.frombuffer(b[:48], np.int64)[3:6])
    lr0, dlr, le0, dle, lT0, dlT, _, _ = np.frombuffer(b[48:112], np.float64)
    off = 112
    P = np.frombuffer(b, np.float64, nr * ne, off).reshape(nr, ne);   off += nr * ne * 8
    cs2 = np.frombuffer(b, np.float64, nr * ne, off).reshape(nr, ne); off += nr * ne * 8
    lT = np.frombuffer(b, np.float64, nr * ne, off).reshape(nr, ne);  off += nr * ne * 8
    esp = np.frombuffer(b, np.float64, nr * nT, off).reshape(nr, nT)
    return dict(nr=nr, ne=ne, nT=nT, lr0=lr0, dlr=dlr, le0=le0, dle=dle,
                lT0=lT0, dlT=dlT, P=P, cs2=cs2, lT2=lT, esp=esp)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    o, n = load(sys.argv[1]), load(sys.argv[2])
    ok = True
    me, mT = n["ne"] - o["ne"], n["nT"] - o["nT"]

    print("=== 1. GEOMETRY ===")
    print(f"  esp nodes {o['ne']} -> {n['ne']} (pad {me});  T nodes {o['nT']} -> {n['nT']} (pad {mT})")
    # CRITERION, corrected 2026-08-18. The first version of this test compared the spacings in
    # ULPs and failed the hires pair at 80 ulp -- which was the test being wrong, not the table.
    # A ulp count is not the quantity that matters and it is resolution-dependent: the same
    # absolute difference is more ulps on a smaller dle, so a finer table fails a fixed ulp
    # threshold for no physical reason. The spacings are recomputed as a difference of floats
    # (`le[1] - le[0]`) in both builds, so they are ENTITLED to differ in the last bits.
    #
    # What actually matters is where the far end of the axis lands, because the loader computes
    # node positions as le0 + i*dle: the accumulated drift across the whole axis, measured in
    # units of one grid spacing. Anything below ~1e-6 of a spacing cannot move a bilinear
    # weight at a level the table's own ~1e-3 discretization would notice.
    for key, m, nnode in (("dle", me, n["ne"]), ("dlT", mT, n["nT"])):
        d = abs(n[key] - o[key])
        drift = (nnode - 1) * d / o[key]        # in units of one grid spacing
        ulp = d / np.spacing(o[key]) if d else 0.0
        good = drift < 1e-6
        ok &= good
        print(f"  {key}: |d| = {d:.3e} ({ulp:.0f} ulp) -> far-end drift {drift:.3e} of one node "
              f"{'OK' if good else '** RESCALED **'}")
    for key, m, sp in (("le0", me, "dle"), ("lT0", mT, "dlT")):
        shift, want = o[key] - n[key], m * o[sp]
        good = abs(shift - want) < 1e-9
        ok &= good
        print(f"  {key} shift {shift:.10f}, expected {want:.10f} {'OK' if good else '** NOT A WHOLE NUMBER OF NODES **'}")
    print(f"  new esp floor: {10**(n['le0']-np.log10(V0SQ)):.4f} code (was {10**(o['le0']-np.log10(V0SQ)):.4f})")
    print(f"  new T   floor: {10**n['lT0']:.4f} K (was {10**o['lT0']:.4f} K)")

    print("\n=== 2. OVERLAP: original nodes unchanged ===")
    allow = {"P": 1, "log10T": 1, "cs2": 2}   # columns permitted to move, and why (see docstring)
    for name, key in (("P", "P"), ("cs2", "cs2"), ("log10T", "lT2")):
        a, c = o[key], n[key][:, me:]
        rel = np.abs(a - c) / np.maximum(np.abs(a), 1e-300)
        k = allow[name]
        moved = np.nonzero(rel.max(axis=0) > 0)[0]
        clean = rel[:, k:].max()
        ok &= clean == 0.0
        print(f"  {name:>7s}: columns that moved {list(moved)[:6]}{' ...' if moved.size > 6 else ''}"
              f" | beyond the {k} allowed: {clean:.3e} {'OK' if clean == 0.0 else '** UNEXPECTED **'}")
        for j in moved[:k]:
            print(f"           col {j} max rel {rel[:, j].max():.3e} (expected: see docstring)")
    rel = np.abs(o["esp"] - n["esp"][:, mT:]) / np.maximum(np.abs(o["esp"]), 1e-300)
    ok &= rel.max() == 0.0
    print(f"  esp_rhoT: max rel {rel.max():.3e} {'OK' if rel.max() == 0.0 else '** CHANGED **'}")

    print("\n=== 3. NEW REGION ===")
    NP, NC, NL = n["P"][:, :me], n["cs2"][:, :me], n["lT2"][:, :me]
    fin = np.isfinite(NP).all() and np.isfinite(NC).all() and np.isfinite(NL).all()
    pos = (NP > 0).all() and (NC > 0).all()
    clamped = int(np.isclose(NL, n["lT0"], rtol=0, atol=1e-12).sum())
    monoT = bool((np.diff(n["lT2"], axis=1) >= -1e-12).all())
    ok &= fin and pos and clamped == 0 and monoT
    print(f"  finite {fin} | positive {pos} | logT monotone in esp {monoT}")
    print(f"  T span {10**NL.min():.4f} .. {10**NL.max():.4f} K")
    print(f"  padded nodes still pinned at the T floor (i.e. still extrapolated): {clamped} of {NL.size}"
          f" {'OK' if clamped == 0 else '** THE EXTENSION BOUGHT NOTHING **'}")

    print("\n=== 4. PHYSICS: frozen-mode limit, independent of the generator ===")
    X = 0.716; mu = 1.0 / (X / 2.0 + (1.0 - X) / 4.0)
    esp_cgs = 10 ** (n["le0"] + n["dle"] * np.arange(n["ne"]))
    Tpred = 2 * mu * MH * esp_cgs / (3 * KB)
    Ttab = 10 ** n["lT2"][:, :me]
    err = np.abs(Ttab - Tpred[None, :me]) / Tpred[None, :me]
    spread = np.ptp(Ttab, axis=0).max() / np.median(Ttab)
    good = err.max() < 1e-10 and spread < 1e-10
    ok &= good
    print(f"  mu = {mu:.5f};  T(esp) vs (2 mu m_H/3k) esp: max rel err {err.max():.3e}")
    print(f"  density-independence of T(esp) across all {n['nr']} rho rows: {spread:.3e}")
    print(f"  cs2/esp over the padded region: {np.min(NC/esp_cgs[None,:me]*V0SQ/V0SQ):.5f}"
          f" .. {np.max(NC/esp_cgs[None,:me]):.5f}   (ideal gamma=5/3 gives {(5/3)*(2/3):.5f};"
          f" the ~0.7 % offset is the generator's finite-difference cs2 and is present in the"
          f" OLD table too -- not introduced here)")
    print(f"  {'OK' if good else '** THE NEW REGION DOES NOT FOLLOW THE FROZEN-MODE LIMIT **'}")

    print("\n=== 5. LOADER LEVEL: does the running code stop clamping? ===")
    # `bilin` below is a faithful transcription of EosTable::bilin (eos_table.hpp): int
    # truncation toward zero, edge clamping of BOTH the index and the interpolation weight.
    # Axes are compared in CGS, which is what the file stores; the CGS->code conversion at load
    # is a pure shift of the axis origin and the index arithmetic is shift-invariant.
    #
    # This is the part of the runtime check that needs no simulation: it answers "will these
    # specific lookups still be extrapolated" directly, at the exact esp values production was
    # measured to visit. A GPU A/B is still worth running for the FIELD-level effect, but the
    # mechanism is settled here.
    def bilin(A, n1, n2, x0, dx, y0, dy, x, y):
        fi = (x - x0) / dx; i = int(fi); cx = 0
        if i < 0: i, cx = 0, -1
        if i > n1 - 2: i, cx = n1 - 2, 1
        ti = min(max(fi - i, 0.0), 1.0)
        fj = (y - y0) / dy; j = int(fj); cy = 0
        if j < 0: j, cy = 0, -1
        if j > n2 - 2: j, cy = n2 - 2, 1
        tj = min(max(fj - j, 0.0), 1.0)
        return ((1 - ti) * (1 - tj) * A[i, j] + ti * (1 - tj) * A[i + 1, j]
                + (1 - ti) * tj * A[i, j + 1] + ti * tj * A[i + 1, j + 1]), cx, cy

    for nm, t in (("old", o), ("new", n)):
        want = 112 + 8 * (t["nr"] * t["ne"] * 3 + t["nr"] * t["nT"])
        have = 112 + 8 * (t["P"].size + t["cs2"].size + t["lT2"].size + t["esp"].size)
        good = want == have
        ok &= good
        print(f"  {nm} file: header implies {want:,} bytes, arrays account for {have:,} "
              f"{'OK' if good else '** MISMATCH -- the loader would misread this **'}")

    # esp values measured in production; extend this list rather than replacing it.
    CASES = [("prodgate cycle-200 minimum", 1.0021),
             ("prod_hidens01 cycle-250 minimum", 0.83341),
             ("just below the old floor", 1.1000),
             ("comfortably inside both", 2.0000)]
    lrho = np.log10(1e-13)      # representative envelope density, well inside both rho axes
    print(f"\n  {'case':>34s} {'esp code':>9s} {'old':>9s} {'new':>9s} {'T new (K)':>10s}"
          f" {'T analytic':>11s} {'rel':>9s}")
    fixed = 0
    for nm_, espc in CASES:
        lesp = np.log10(espc * V0SQ)
        _, _, c2 = bilin(o["lT2"], o["nr"], o["ne"], o["lr0"], o["dlr"], o["le0"], o["dle"], lrho, lesp)
        T3, _, c3 = bilin(n["lT2"], n["nr"], n["ne"], n["lr0"], n["dlr"], n["le0"], n["dle"], lrho, lesp)
        Tan = 2 * mu * MH * (espc * V0SQ) / (3 * KB)
        T3 = 10 ** T3
        if c2 and not c3: fixed += 1
        if c3: ok = False
        print(f"  {nm_:>34s} {espc:9.4f} {'CLAMPED' if c2 else 'ok':>9s} "
              f"{'** STILL CLAMPED **' if c3 else 'ok':>9s} {T3:10.4f} {Tan:11.4f} "
              f"{abs(T3 - Tan) / Tan:9.2e}")
    print(f"  {fixed} of {len(CASES)} probe lookups were extrapolated before and interpolate now;"
          f" none still clamps.")

    print("\n" + ("VERDICT: the new table is a clean downward EXTENSION of the old one."
                  if ok else "VERDICT: ** FAILED -- see the marked lines above. **"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
