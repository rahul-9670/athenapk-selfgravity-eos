"""Bitwise comparison of every dataset in the final phdf of two run directories (key-off identity checks).
usage: bitcmp_dirs.py dirA dirB [dirA2 dirB2 ...]"""
import glob, sys
import h5py, numpy as np
a = sys.argv[1:]
for A, B in zip(a[::2], a[1::2]):
    fa = sorted(glob.glob(A + "/*.out0.final.phdf"))[0]; fb = sorted(glob.glob(B + "/*.out0.final.phdf"))[0]
    with h5py.File(fa) as ha, h5py.File(fb) as hb:
        bad = []
        for k in ha:
            if isinstance(ha[k], h5py.Dataset) and k in hb:
                x, y = ha[k][()], hb[k][()]
                if x.shape != y.shape or not np.array_equal(x.view(np.uint8), y.view(np.uint8)):
                    bad.append(k)
        print(f"{'BITWISE' if not bad else 'DIFFER ' + str(bad)}  {A}  vs  {B}")
