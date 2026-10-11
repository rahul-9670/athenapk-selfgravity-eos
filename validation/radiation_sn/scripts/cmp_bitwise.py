#!/usr/bin/env python
"""Bit-exact comparison of two AthenaPK restart files, field by field.

  cmp_bitwise.py <dirA> <dirB> [ncycle]

Picks the restart in each directory whose Info/NCycle matches `ncycle` (default: the highest
cycle present in BOTH), then compares every variable dataset bit-for-bit. The .hst cannot be
used for this: it prints only 6 significant figures.
"""
import glob, os, sys
import h5py, numpy as np

def index(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.rhdf")):
        try:
            with h5py.File(f, "r") as h:
                out[int(h["Info"].attrs["NCycle"])] = f
        except Exception:
            pass
    return out

da, db = sys.argv[1], sys.argv[2]
ia, ib = index(da), index(db)
if not ia or not ib:
    print(f"NO RESTARTS: {da}={sorted(ia)} {db}={sorted(ib)}"); sys.exit(2)
common = sorted(set(ia) & set(ib))
if len(sys.argv) > 3:
    nc = int(sys.argv[3])
elif common:
    nc = common[-1]
else:
    print(f"NO COMMON CYCLE: {da}={sorted(ia)} {db}={sorted(ib)}"); sys.exit(2)
print(f"comparing cycle {nc}\n  A {ia[nc]}\n  B {ib[nc]}")

ha, hb = h5py.File(ia[nc], "r"), h5py.File(ib[nc], "r")
ta, tb = float(ha["Info"].attrs["Time"]), float(hb["Info"].attrs["Time"])
print(f"  time A {ta!r}\n  time B {tb!r}\n  time identical: {ta == tb}")

skip = {"Blocks", "Info", "Input", "Levels", "Locations", "LogicalLocations",
        "Params", "VolumeLocations"}
names = sorted((set(ha.keys()) | set(hb.keys())) - skip)
bad = 0
for n in names:
    if n not in ha or n not in hb:
        print(f"  {n:16s} MISSING in {'A' if n not in ha else 'B'}"); bad += 1; continue
    a, b = ha[n], hb[n]
    if a.shape != b.shape:
        print(f"  {n:16s} SHAPE {a.shape} vs {b.shape}"); bad += 1; continue
    d = 0; mx = 0.0; mr = 0.0
    step = max(1, a.shape[0] // 8)
    for i in range(0, a.shape[0], step):
        j = min(i + step, a.shape[0])
        x, y = a[i:j], b[i:j]
        neq = x != y
        c = int(neq.sum())
        if c:
            d += c
            diff = np.abs(x[neq].astype(np.float64) - y[neq].astype(np.float64))
            den = np.maximum(np.abs(x[neq].astype(np.float64)), 1e-300)
            mx = max(mx, float(diff.max())); mr = max(mr, float((diff / den).max()))
    tot = int(np.prod(a.shape))
    if d:
        print(f"  {n:16s} DIFFERS  {d}/{tot} cells, max abs {mx:.3e}, max rel {mr:.3e}"); bad += 1
    else:
        print(f"  {n:16s} identical ({tot} values)")
print("\nRESULT:", "BIT-IDENTICAL" if bad == 0 else f"NOT identical ({bad} dataset(s) differ)")
sys.exit(0 if bad == 0 else 1)
