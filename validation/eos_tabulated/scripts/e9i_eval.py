#!/usr/bin/env python
"""E9i: GPU vs CPU for double flux + EOS-consistent fill on refined meshes (E7r decks), all snapshots."""
import glob, os
import h5py
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def flat(f):
    with h5py.File(f) as h:
        p = h["prim"][()].astype(float)
        xs, ys, zs = h["Locations/x"][()], h["Locations/y"][()], h["Locations/z"][()]
        nc = int(np.atleast_1d(h["Info"].attrs["NCycle"])[0])
    K, V = [], []
    for b in range(p.shape[0]):
        Z, Y, X = np.meshgrid(0.5 * (zs[b, 1:] + zs[b, :-1]), 0.5 * (ys[b, 1:] + ys[b, :-1]),
                              0.5 * (xs[b, 1:] + xs[b, :-1]), indexing="ij")
        K.append(np.stack([Z.ravel(), Y.ravel(), X.ravel()]))
        V.append(p[b][[0, 4]].reshape(2, -1))
    k = np.concatenate(K, 1)
    v = np.concatenate(V, 1)
    o = np.lexsort((k[2], k[1], k[0]))
    return v[:, o], nc


lines = ["E9i: double flux + transported-pressure fill on GPU vs CPU 85b625bf (runs/E7w4/gf), all snapshots",
         "run                          | cycles GPU/CPU | max over snaps |P/P*-1| GPU, CPU | max rel diff rho, P"]
for d in sorted(glob.glob(os.path.join(R, "E9i_gpu", "*"))):
    n = os.path.basename(d)
    fg = sorted(glob.glob(d + "/*.out0.*.phdf"))
    fc = sorted(glob.glob(os.path.join(R, "E7w4", "gf", n, "*.out0.*.phdf")))
    if len(fg) != len(fc) or not fg:
        lines.append(f"{n} snapshot count {len(fg)} vs {len(fc)}"); continue
    P0 = np.median(flat(fg[0])[0][1])
    eg = ec = dr = dp = 0.0
    for a, b in zip(fg, fc):
        (G, ng), (C, nc) = flat(a), flat(b)
        if G.shape != C.shape:
            lines.append(f"{n} mesh differs at {os.path.basename(a)}"); break
        eg = max(eg, np.abs(G[1] / P0 - 1).max()); ec = max(ec, np.abs(C[1] / P0 - 1).max())
        dr = max(dr, np.abs(G[0] / C[0] - 1).max()); dp = max(dp, np.abs(G[1] / C[1] - 1).max())
    lines.append(f"{n:28s} | {ng}/{nc} | {eg:.2e}, {ec:.2e} | {dr:.1e}, {dp:.1e}")
out = os.path.join(R, "..", "results", "tables", "E9i_gf_gpu.txt")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
