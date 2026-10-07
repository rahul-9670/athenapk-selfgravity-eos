"""Shared helpers for the self-gravity validation suite: Parthenon phdf reader, error norms.

Reads a Parthenon .phdf into flat per-cell arrays (works for uniform and AMR meshes):
    d = load(fname, ["grav.phi", "prim"])   -> d["x"], d["y"], d["z"], d["vol"], d["level"],
                                               d["grav.phi"], d["prim"][comp], d["time"], ...
AthenaPK prim components: 0 rho, 1-3 v, 4 P (glmmhd: 5-7 B, 8 psi).
"""
import glob
import os
import re

import h5py
import numpy as np


def _attr(h, k):
    v = h["Info"].attrs[k]
    return v.decode() if isinstance(v, bytes) else v


def load(fname, variables=("grav.phi",)):
    out = {}
    with h5py.File(fname, "r") as h:
        out["time"] = float(np.atleast_1d(h["Info"].attrs["Time"])[0])
        out["ncycle"] = int(np.atleast_1d(h["Info"].attrs["NCycle"])[0])
        xf = h["Locations/x"][()]
        yf = h["Locations/y"][()]
        zf = h["Locations/z"][()]
        nb, nx = xf.shape[0], xf.shape[1] - 1
        ny, nz = yf.shape[1] - 1, zf.shape[1] - 1
        xc = 0.5 * (xf[:, 1:] + xf[:, :-1])
        yc = 0.5 * (yf[:, 1:] + yf[:, :-1])
        zc = 0.5 * (zf[:, 1:] + zf[:, :-1])
        dx = xf[:, 1] - xf[:, 0]
        dy = yf[:, 1] - yf[:, 0]
        dz = zf[:, 1] - zf[:, 0]
        # cell arrays ordered [block, k, j, i]
        out["x"] = np.broadcast_to(xc[:, None, None, :], (nb, nz, ny, nx)).ravel()
        out["y"] = np.broadcast_to(yc[:, None, :, None], (nb, nz, ny, nx)).ravel()
        out["z"] = np.broadcast_to(zc[:, :, None, None], (nb, nz, ny, nx)).ravel()
        out["dx"] = np.broadcast_to(dx[:, None, None, None], (nb, nz, ny, nx)).ravel()
        out["vol"] = np.broadcast_to((dx * dy * dz)[:, None, None, None], (nb, nz, ny, nx)).ravel()
        if "Levels" in h:
            lev = h["Levels"][()]
            out["level"] = np.broadcast_to(lev[:, None, None, None], (nb, nz, ny, nx)).ravel()
        out["nblocks"] = nb
        out["block_shape"] = (nz, ny, nx)
        for v in variables:
            a = h[v][()].astype(np.float64)
            if a.ndim == 4:                       # [nb, k, j, i]
                out[v] = a.ravel()
            elif a.ndim == 5 and a.shape[1] == 1 and v != "prim":
                out[v] = a[:, 0].ravel()
            else:                                 # [nb, comp, k, j, i]
                out[v] = [a[:, c].ravel() for c in range(a.shape[1])]
    return out


def final_dump(run_dir, out="out0"):
    """Newest phdf of an output stream (the 'final' dump if present)."""
    fs = glob.glob(os.path.join(run_dir, f"*.{out}.final.phdf"))
    if fs:
        return fs[0]
    fs = sorted(glob.glob(os.path.join(run_dir, f"*.{out}.[0-9]*.phdf")))
    return fs[-1] if fs else None


def dumps(run_dir, out="out0"):
    fs = sorted(glob.glob(os.path.join(run_dir, f"*.{out}.[0-9]*.phdf")))
    return fs


def rms(err, vol):
    return float(np.sqrt(np.sum(err ** 2 * vol) / np.sum(vol)))


def l1(err, vol):
    return float(np.sum(np.abs(err) * vol) / np.sum(vol))


def order(errs, ns):
    """Pairwise convergence orders log(e_i/e_{i+1}) / log(n_{i+1}/n_i)."""
    e, n = np.asarray(errs, float), np.asarray(ns, float)
    return list(np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1]))


def hst(fname):
    names = None
    rows = []
    for line in open(fname):
        if line.startswith("#"):
            m = re.findall(r"\[(\d+)\]=(\S+)", line)
            if m:
                names = [n for _, n in m]
            continue
        if line.strip():
            rows.append([float(v) for v in line.split()])
    a = np.array(rows)
    return {n: a[:, i] for i, n in enumerate(names)} if names else a


def solver_stats(run_log):
    """Gravity-solver diagnostics from run.log (solver_diag = true prints grav-iters/grav-res)."""
    txt = open(run_log, errors="replace").read()
    its = [int(v) for v in re.findall(r"grav-iters[=: ]+(-?\d+)", txt)]
    res = [float(v) for v in re.findall(r"grav-res[=: ]+(-?[0-9.eE+-]+)", txt)]
    nonconv = len(re.findall(r"NOT CONVERGED|grav-nonconv[=: ]+[1-9]", txt))
    return {"iters": its, "res": res, "nonconv": nonconv}


def run_rc(run_dir):
    """Exit code recorded by run_one.sh in cmd.txt (None if the run has not finished)."""
    try:
        for line in open(os.path.join(run_dir, "cmd.txt")):
            if line.startswith("rc "):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def to_global(d, field):
    """Uniform mesh only: scatter a flat per-cell field into a global [nz, ny, nx] array.
    Returns (array, (xc, yc, zc)) or None if the mesh has more than one level."""
    if "level" in d and len(np.unique(d["level"])) > 1:
        return None
    dx = float(d["dx"][0])
    xs, ys, zs = np.unique(np.round(d["x"] / dx, 6)), np.unique(np.round(d["y"] / dx, 6)), np.unique(np.round(d["z"] / dx, 6))
    ix = np.searchsorted(xs, np.round(d["x"] / dx, 6))
    iy = np.searchsorted(ys, np.round(d["y"] / dx, 6))
    iz = np.searchsorted(zs, np.round(d["z"] / dx, 6))
    a = np.full((zs.size, ys.size, xs.size), np.nan)
    a[iz, iy, ix] = field
    return a, (xs * dx, ys * dx, zs * dx)
