#!/usr/bin/env python
"""R-F13 adoption (user 2026-10-09: "go for 8"): remap an AthenaPK restart from one frequency-group partition to another
(production 3 groups {0, 1e12, 1e15, 1e30} Hz -> X8 {0, 1.5e12, 3e12, 5e12, 8e12, 1.2e13, 1.8e13, 3e13, 1e30} Hz).

Parthenon reads each Restart variable with the RUN's component count and skips one that is absent on disk, so an
8-group run cannot simply resume from a 3-group file (the code's own fallback re-seeds sn.u by P1 from rad.Er/rad.Fr and
would leave the five new groups' rad.* at zero). This writes a new file in which every group-dependent dataset has the
new shape; everything else (cons, grav.phi, mesh, Params, Input, Info) is copied unchanged except the Info attributes
that list the datasets and their components.

THE MAP (per cell, linear, the same for every ordinate of sn.u and for rad.Er / rad.Fr1-3):
    new_g' = sum_G W[g', G] old_G,   W[g', G] = (Planck energy of G /\\ g') / (Planck energy of G)   at T_c,
so sum_g' W[g', G] = 1: each old group's energy, flux and every ordinate's intensity are conserved EXACTLY, and an old
group that lies inside one new group goes there whole. T_c is the cell's radiation COLOUR temperature: the Planck
temperature whose energy ratio between the first two old groups equals the cell's own E_0 / E_1 (clipped to
[T_min, T_max]; cells with E_1 or E_0 at the floor take the clip). Only old groups that straddle new edges need it.

usage: remap_restart_groups.py IN.rhdf OUT.rhdf --old 0,1e12,1e15,1e30 --new 0,1.5e12,...,1e30 [--nang 32]
       remap_restart_groups.py --check IN.rhdf OUT.rhdf --old ... --new ...   (re-reads both files: conservation per
       cell and old group of E, F and every ordinate; untouched datasets byte-identical; prints the T_c distribution)
"""
import argparse
import sys

import h5py
import numpy as np

H_K = 6.62607015e-27 / 1.380649e-16   # h / k_B [K s]
EFLOOR_REL = 1e-12                     # E_0 or E_1 below this x (E_0 + E_1): colour temperature not defined -> clip


def group_names(base, ng):
    return [base] + [f"{base}_g{g}" for g in range(1, ng)]


def planck_fraction_table(lo, hi, cuts, Tgrid):
    """Fraction of the Planck energy in [lo, hi] that falls into each piece [cuts[k], cuts[k+1]] (cuts span lo..hi),
    per T in Tgrid. Log-nu trapezoid on 40000 nodes with the integrand normalised by its maximum on [lo, hi] (no
    underflow at x = h nu / k T >> 1)."""
    lo_ = max(lo, 1e8)
    hi_ = min(hi, 1e18)
    lnu = np.linspace(np.log(lo_), np.log(hi_), 40001)
    nu = np.exp(lnu)
    out = np.zeros((len(Tgrid), len(cuts) - 1))
    for it, T in enumerate(Tgrid):
        x = H_K * nu / T
        lb = 4.0 * lnu - np.log(np.expm1(np.minimum(x, 700.0))) - np.where(x > 700.0, x - 700.0, 0.0)  # nu B_nu, d ln nu
        f = np.exp(lb - lb.max())
        c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lnu))])
        tot = c[-1]
        for k in range(len(cuts) - 1):
            a, b = np.log(max(cuts[k], lo_)), np.log(min(cuts[k + 1], hi_))
            out[it, k] = (np.interp(b, lnu, c) - np.interp(a, lnu, c)) / tot
    return out


class Remap:
    def __init__(self, old, new, Tmin=2.0, Tmax=1.0e4, nT=600):
        self.old, self.new = np.array(old, float), np.array(new, float)
        self.nG, self.ng = len(old) - 1, len(new) - 1
        self.Tgrid = np.geomspace(Tmin, Tmax, nT)
        self.lT = np.log(self.Tgrid)
        # per old group: the new groups it touches and (if it straddles) the fraction table
        self.parts = []
        for G in range(self.nG):
            lo, hi = self.old[G], self.old[G + 1]
            idx = [g for g in range(self.ng) if min(hi, self.new[g + 1]) > max(lo, self.new[g])]
            if len(idx) == 1:
                self.parts.append((idx, None))
            else:
                cuts = [lo] + [self.new[g + 1] for g in idx[:-1]] + [hi]
                self.parts.append((idx, planck_fraction_table(lo, hi, cuts, self.Tgrid)))
        # colour temperature from E_0 / E_1: ratio of the Planck energies of old groups 0 and 1
        f01 = planck_fraction_table(self.old[0], self.old[2], [self.old[0], self.old[1], self.old[2]], self.Tgrid)
        self.ratio = f01[:, 0] / np.maximum(f01[:, 1], 1e-300)   # decreasing in T
        assert np.all(np.diff(self.ratio) < 0), "E_0/E_1 Planck ratio not monotone on the T grid"

    def colour_T(self, E0, E1):
        s = np.abs(E0) + np.abs(E1)
        ok = (E0 > EFLOOR_REL * s) & (E1 > EFLOOR_REL * s)
        r = np.where(ok, E0 / np.where(ok, E1, 1.0), 1.0)
        # ratio decreasing in T: interpolate log T against -log ratio
        lt = np.interp(-np.log(np.clip(r, 1e-300, None)), -np.log(self.ratio), self.lT)
        lt = np.where(ok, lt, np.where(E0 >= E1, self.lT[0], self.lT[-1]))
        return np.exp(lt)

    def weights(self, T):
        """W[..., g', G] for cell temperatures T (any shape)."""
        W = np.zeros(T.shape + (self.ng, self.nG))
        lt = np.log(np.clip(T, self.Tgrid[0], self.Tgrid[-1]))
        for G, (idx, tab) in enumerate(self.parts):
            if tab is None:
                W[..., idx[0], G] = 1.0
                continue
            fr = np.stack([np.interp(lt, self.lT, tab[:, k]) for k in range(len(idx))], axis=-1)
            fr /= fr.sum(axis=-1, keepdims=True)   # exact partition of unity after interpolation
            for k, g in enumerate(idx):
                W[..., g, G] = fr[..., k]
        return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--nang", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    old = [float(x) for x in a.old.split(",")]
    new = [float(x) for x in a.new.split(",")]
    R = Remap(old, new)
    nG, ng, nang = R.nG, R.ng, a.nang
    remapped = set(["sn.u"] + sum([group_names(b, nG) for b in ("rad.Er", "rad.Fr1", "rad.Fr2", "rad.Fr3")], []))
    if a.check:
        return check(a, R, remapped)
    src = h5py.File(a.inp, "r")
    assert src["sn.u"].shape[1] == nG * nang, f"sn.u has {src['sn.u'].shape[1]} components, expected {nG}x{nang}"
    for nm in remapped:
        assert nm in src, f"{nm} missing in {a.inp}"
    dst = h5py.File(a.out, "w-")
    for k in src.keys():
        if k not in remapped:
            src.copy(src[k], dst, name=k)
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    nb = src["sn.u"].shape[0]
    cell = src["rad.Er"].shape[1:]
    comp = dict(compression=src["sn.u"].compression, compression_opts=src["sn.u"].compression_opts)
    out_names = {b: group_names(b, ng) for b in ("rad.Er", "rad.Fr1", "rad.Fr2", "rad.Fr3")}
    for b, names in out_names.items():
        for nm in names:
            d = dst.create_dataset(nm, (nb,) + cell, dtype=src["rad.Er"].dtype, chunks=(1,) + cell, **comp)
            d.attrs["TopologicalLocation"] = src["rad.Er"].attrs["TopologicalLocation"]
    du = dst.create_dataset("sn.u", (nb, ng * nang) + cell, dtype=src["sn.u"].dtype, chunks=(1, 1) + cell, **comp)
    du.attrs["TopologicalLocation"] = src["sn.u"].attrs["TopologicalLocation"]
    Tc_hist = np.zeros(len(R.Tgrid))
    for b0 in range(0, nb, a.chunk):
        b1 = min(nb, b0 + a.chunk)
        E = np.stack([np.array(src[nm][b0:b1]) for nm in group_names("rad.Er", nG)], axis=1)   # (nb, G, k, j, i)
        T = R.colour_T(E[:, 0], E[:, 1])
        W = R.weights(T)                                                                     # (nb, k, j, i, g, G)
        Tc_hist += np.histogram(np.log(T), bins=np.concatenate([R.lT, [np.inf]]))[0]
        for b, names in out_names.items():
            X = np.stack([np.array(src[nm][b0:b1]) for nm in group_names(b, nG)], axis=-1)    # (nb, k, j, i, G)
            Y = np.einsum("...gG,...G->...g", W, X)
            for g, nm in enumerate(names):
                dst[nm][b0:b1] = Y[..., g]
        U = np.array(src["sn.u"][b0:b1])                                                     # (nb, G*nang, k, j, i)
        U = U.reshape(U.shape[0], nG, nang, *cell)
        V = np.einsum("bkjigG,bGakji->bgakji", W, U)
        dst["sn.u"][b0:b1] = V.reshape(V.shape[0], ng * nang, *cell)
        print(f"  blocks {b0}-{b1 - 1} of {nb}", flush=True)
    # Info: dataset list, component counts and names follow the new datasets (Parthenon writes them sorted by name)
    info = dst["Info"]
    old_names = [x.decode() if isinstance(x, bytes) else str(x) for x in src["Info"].attrs["OutputDatasetNames"]]
    old_nc = [int(x) for x in src["Info"].attrs["NumComponents"]]   # uint64 in production files (int + uint64 -> float)
    old_cn = [x.decode() if isinstance(x, bytes) else str(x) for x in src["Info"].attrs["ComponentNames"]]
    cn_of, pos = {}, 0
    for nm, nc in zip(old_names, old_nc):
        cn_of[nm] = old_cn[pos:pos + nc]
        pos += nc
    assert pos == len(old_cn), "Info/ComponentNames does not match NumComponents"
    names = sorted([n for n in old_names if n not in remapped] + ["sn.u"] + sum(out_names.values(), []))
    nc_new, cn_new = [], []
    for nm in names:
        if nm == "sn.u":
            nc_new.append(ng * nang)
            cn_new += [f"sn.u_{i}" for i in range(ng * nang)]
        elif nm in cn_of:
            nc_new.append(len(cn_of[nm]))
            cn_new += cn_of[nm]
        else:
            nc_new.append(1)
            cn_new.append(nm)
    del info.attrs["OutputDatasetNames"], info.attrs["NumComponents"], info.attrs["ComponentNames"]
    info.attrs["OutputDatasetNames"] = np.array(names, dtype=h5py.string_dtype(encoding="ascii"))
    info.attrs["NumComponents"] = np.array(nc_new, dtype=src["Info"].attrs["NumComponents"].dtype)
    info.attrs["ComponentNames"] = np.array(cn_new, dtype=h5py.string_dtype(encoding="ascii"))
    dst.attrs["remap_groups"] = f"old {a.old} -> new {a.new} (validation/radiation_sn/scripts/remap_restart_groups.py)"
    dst.close()
    q = np.cumsum(Tc_hist) / Tc_hist.sum()
    edges = R.Tgrid
    print("colour temperature T_c percentiles (K): " +
          "  ".join(f"p{int(p * 100)} {edges[np.searchsorted(q, p)]:.1f}" for p in (0.01, 0.5, 0.99)))
    print("WROTE", a.out)


def check(a, R, remapped):
    """Conservation per cell and old group: sum over new groups of each output = sum over old groups of the input
    (E, F1-3 and every ordinate); untouched datasets identical; Info consistent; new group values >= 0 where old >= 0."""
    nG, ng, nang = R.nG, R.ng, a.nang
    s, d = h5py.File(a.inp, "r"), h5py.File(a.out, "r")
    ok = True
    for k in s.keys():
        if k in remapped or not isinstance(s[k], h5py.Dataset):
            continue
        same = np.array_equal(np.array(s[k]), np.array(d[k]))
        ok &= same
        if not same:
            print("  DIFFERS (should be untouched):", k)
    print("  untouched datasets identical:", ok)
    worst = {}
    nb = s["sn.u"].shape[0]
    for b0 in range(0, nb, a.chunk):
        b1 = min(nb, b0 + a.chunk)
        for base in ("rad.Er", "rad.Fr1", "rad.Fr2", "rad.Fr3"):
            X = sum(np.array(s[nm][b0:b1]) for nm in group_names(base, nG))
            Y = sum(np.array(d[nm][b0:b1]) for nm in group_names(base, ng))
            sc = sum(np.abs(np.array(s[nm][b0:b1])) for nm in group_names(base, nG))
            r = np.max(np.abs(Y - X) / np.maximum(sc, 1e-300))
            worst[base] = max(worst.get(base, 0.0), r)
        U = np.array(s["sn.u"][b0:b1]).reshape(b1 - b0, nG, nang, *s["rad.Er"].shape[1:])
        V = np.array(d["sn.u"][b0:b1]).reshape(b1 - b0, ng, nang, *s["rad.Er"].shape[1:])
        r = np.max(np.abs(V.sum(1) - U.sum(1)) / np.maximum(np.abs(U).sum(1), 1e-300))
        worst["sn.u per ordinate"] = max(worst.get("sn.u per ordinate", 0.0), r)
        worst["sn.u negative (new) count"] = worst.get("sn.u negative (new) count", 0) + int((V < 0).sum())
        worst["sn.u negative (old) count"] = worst.get("sn.u negative (old) count", 0) + int((U < 0).sum())
    for k, v in worst.items():
        print(f"  {k:28s} max rel |sum_new - sum_old| = {v:.3e}" if "count" not in k else f"  {k:28s} {v}")
    cons_ok = all(v < 1e-13 for k, v in worst.items() if "count" not in k)
    nc = [int(x) for x in d["Info"].attrs["NumComponents"]]
    nm = list(d["Info"].attrs["OutputDatasetNames"])
    info_ok = (len(nc) == len(nm) and sum(nc) == len(d["Info"].attrs["ComponentNames"]) and
               all((d[(x.decode() if isinstance(x, bytes) else x)].shape[1] if d[(x.decode() if isinstance(x, bytes) else x)].ndim == 5 else 1) == c
                   for x, c in zip(nm, nc)))
    print("  conservation < 1e-13:", cons_ok, "  Info consistent:", info_ok)
    ok = ok and cons_ok and info_ok
    print("CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
