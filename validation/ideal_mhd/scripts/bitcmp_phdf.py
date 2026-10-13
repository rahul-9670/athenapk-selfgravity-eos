#!/usr/bin/env python
"""Bitwise comparison of the final phdf of two run directories (every dataset with a numeric dtype except Info /
Params / Input metadata). Usage: bitcmp_phdf.py <dirA> <dirB>; prints IDENTICAL or the first differences."""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, os, sys
import h5py, numpy as np

def last(d):
    fs = sorted(glob.glob(os.path.join(d, "*.out0.final.phdf"))) or sorted(glob.glob(os.path.join(d, "*.out0.*.phdf")))
    return fs[-1]

fa, fb = last(sys.argv[1]), last(sys.argv[2])
bad = 0
with h5py.File(fa) as a, h5py.File(fb) as b:
    for k in a.keys():
        if k in ("Info", "Params", "Input") or not isinstance(a[k], h5py.Dataset):
            continue
        x, y = a[k][()], b[k][()]
        same = x.shape == y.shape and np.array_equal(x.view(np.uint8), y.view(np.uint8))
        if not same:
            bad += 1
            print(f"  {k}: DIFFER (max abs {np.abs(x - y).max():.3e})")
print(os.path.basename(fa), "vs", os.path.basename(fb), "->", "IDENTICAL" if not bad else f"{bad} datasets differ")
sys.exit(1 if bad else 0)
