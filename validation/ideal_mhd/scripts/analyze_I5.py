#!/usr/bin/env python
"""I5 / I5b analysis: GLM div-B pulse against the exact solution of the linear GLM system.

Exact reference (periodic unit box, B0 = 0, v = 0; extended-GLM sources O(bamp^2) dropped):
  B = B_sol + khat a,  per Fourier mode k:  d/dt (a, psi) = A (a, psi),  A = [[0, -i|k|], [-i c^2 |k|, -lambda]]
  exp(A t) = e^{-lambda t/2} [cosh(mu t) I + sinh(mu t)/mu (A + lambda/2 I)],  mu^2 = lambda^2/4 - c^2 k^2.
  lambda = alpha c_h / mindx (dedner_source.cpp: psi *= exp(-alpha c_h dt / mindx) every step), c_h = 1 (cs; the
  bamp^2 correction to the fast speed is 5e-7, below every measured error).
The reference is sampled on the grid x = j h, h = dx_finest / 2: it contains the cell centres of EVERY level
(level-L centres are (2i+1) dx_L / 2), so uniform and SMR runs are scored with one array, no interpolation.
Initial data: the analytic Gaussian on that grid (aliasing of exp(-r^2/rb^2) at h <= 1/256: < 1e-30).
Errors: L1 (cell-volume weighted) of Bx, By and psi vs the reference, and the div-B norm
  ||D||_1 with D = central-difference div B (the operator dedner_source.cpp uses), per output time.
Writes results/tables/I5.{json,txt};  --check recomputes and compares with the saved json.
"""
import os as _os  # public repository: paths relative to this script
_VT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_REPO = _os.path.dirname(_VT)
import glob, json, math, os, re, sys
import h5py
import numpy as np

M = _VT + "/ideal_mhd"
BAMP, RB, CH = 1.0e-3, 0.05, 1.0
PAT = re.compile(r"I5([bc]?)_(\d)d_(a0|lam10|prod)_N(\d+)(?:_smr(\d)|_patch(\d))?(?:_cfl[0-9.]+)?$")


def alpha_of(case, mindx_N):
    return {"a0": 0.0, "lam10": 10.0 / mindx_N, "prod": 0.1}[case]


def reference(dim, nref, lam, t, return_div=False):
    """Exact (Bx, By, psi) on the nref^dim grid x_j = j / nref (j = 0..nref-1)."""
    x = np.arange(nref) / nref
    k1 = 2 * np.pi * np.fft.fftfreq(nref, d=1.0 / nref)
    if dim == 1:
        g = BAMP * np.exp(-((x - 0.5) ** 2) / RB**2)
        kx, ky = k1, np.zeros_like(k1)
    else:
        X, Y = np.meshgrid(x, x, indexing="ij")
        g = BAMP * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / RB**2)
        kx, ky = np.meshgrid(k1, k1, indexing="ij")
    G = np.fft.fftn(g)
    kk = np.sqrt(kx**2 + ky**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        hx = np.where(kk > 0, kx / kk, 0.0)
        hy = np.where(kk > 0, ky / kk, 0.0)
    a0 = hx * G                        # irrotational amplitude khat . B_hat (psi_hat = 0 at t = 0)
    bsx, bsy = G - hx * a0, -hy * a0   # solenoidal remainder (static); k = 0 mode stays in it
    mu = np.sqrt((lam**2 / 4 - CH**2 * kk**2).astype(complex))
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(np.abs(mu) > 1e-14, np.sinh(mu * t) / mu, t)
    e = np.exp(-lam * t / 2)
    # (A + lam/2 I) applied to (a0, 0): first row lam/2 a0, second row -i c^2 |k| a0
    a = e * (np.cosh(mu * t) * a0 + sh * (lam / 2) * a0)
    ps = e * (sh * (-1j * CH**2 * kk * a0))
    if return_div:
        return float(np.sqrt((np.abs(kk * a) ** 2).sum() / max((np.abs(kk * a0) ** 2).sum(), 1e-300)))
    Bx = np.real(np.fft.ifftn(bsx + hx * a))
    By = np.real(np.fft.ifftn(bsy + hy * a))
    psi = np.real(np.fft.ifftn(ps))
    return Bx, By, psi


def load(f):
    with h5py.File(f, "r") as h:
        t = float(h["Info"].attrs["Time"])
        c = h["cons"][()]
        x = h["Locations"]["x"][()]
        y = h["Locations"]["y"][()]
    return t, c, x, y


def score(d, dim, N, lev, case):
    mindx_N = N * 2**lev
    lam = alpha_of(case, mindx_N) * CH * mindx_N
    nref = 2 * mindx_N
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf"))):
        t, c, x, y = load(f)
        Bxr, Byr, psr = reference(dim, nref, lam, t)
        e = {"Bx": 0.0, "By": 0.0, "psi": 0.0}
        sig = 0.0   # L1 of the reference Bx - B0 (the pulse signal), same cells
        D1, vol = 0.0, 0.0
        for b in range(c.shape[0]):
            xc = 0.5 * (x[b, 1:] + x[b, :-1])
            ix = np.rint(xc * nref).astype(int) % nref
            dx = x[b, 1] - x[b, 0]
            bx, by, ps = c[b, 5, 0], c[b, 6, 0], c[b, 8, 0]   # (j, i)
            if dim == 1:
                rx, ry, rp = Bxr[ix][None, :], Byr[ix][None, :], psr[ix][None, :]
                dV = dx
            else:
                yc = 0.5 * (y[b, 1:] + y[b, :-1])
                iy = np.rint(yc * nref).astype(int) % nref
                rx, ry, rp = (A[np.ix_(ix, iy)].T for A in (Bxr, Byr, psr))
                dV = dx * (y[b, 1] - y[b, 0])
            e["Bx"] += np.abs(bx - rx).sum() * dV
            sig += np.abs(rx).sum() * dV
            e["By"] += np.abs(by - ry).sum() * dV
            e["psi"] += np.abs(ps - rp).sum() * dV
            # div B on interior cells of the block (central differences, as dedner_source.cpp)
            if dim == 1:
                Dv = (bx[0, 2:] - bx[0, :-2]) / (2 * dx)
                D1 += np.abs(Dv).sum() * dV
            else:
                dy = y[b, 1] - y[b, 0]
                Dv = (bx[1:-1, 2:] - bx[1:-1, :-2]) / (2 * dx) + (by[2:, 1:-1] - by[:-2, 1:-1]) / (2 * dy)
                D1 += np.abs(Dv).sum() * dV
            vol += dV * bx.size
        rows.append({"t": round(t, 6), "L1_Bx": e["Bx"], "L1_By": e["By"], "L1_psi": e["psi"], "divB_L1": D1,
                     "rel_err_Bx": e["Bx"] / sig if sig > 0 else None,
                     "exact_divB_L2_ratio": reference(dim, min(nref, 1024), lam, t, return_div=True)})
    return {"lambda": lam, "rows": rows}


def collect():
    res = {}
    def key(d):   # (I5 / I5b / I5c, dim, case, smr, N): ladders in increasing N, so the order column is right
        m = PAT.match(os.path.basename(d))
        return (m.group(1), int(m.group(2)), m.group(3), int(m.group(5) or 0), int(m.group(4)), d) if m else ("z",)
    for d in sorted(glob.glob(f"{M}/runs/I5/I5*"), key=key):
        m = PAT.match(os.path.basename(d))
        if not m or not glob.glob(os.path.join(d, "*.phdf")):
            continue
        dim, case, N, lev = int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5) or m.group(6) or 0)
        res[os.path.basename(d)] = dict(dim=dim, case=case, N=N, smr=lev, **score(d, dim, N, lev, case))
    return res


def text(res):
    L = ["I5 GLM div-B pulse vs exact linear GLM solution (bamp 1e-3, rb 0.05, c_h 1). L1 errors at t = 0.3;",
         "order = log2 ratio of successive N at fixed case. rel = L1 Bx error / L1 Bx signal; exact D2 = exact",
         "||div B||_2(t)/||div B||_2(0) of the reference (spectral; the code's own interior div-B L1 is NOT used: without",
         "ghosts it skips block rims and depends on the block size).",
         f"{'run':<28} {'lambda':>8} {'L1 Bx':>10} {'ord':>5} {'L1 psi':>10} {'ord':>5} {'rel':>8} {'exact D2':>9}"]
    prev = {}
    for name, r in res.items():
        last, first = r["rows"][-1], r["rows"][0]
        key = (r["dim"], r["case"], r["smr"])
        oB = oP = "    -"
        if key in prev and prev[key][0] * 2 == r["N"] and r["smr"] == 0 and "cfl" not in name:
            oB = f"{math.log2(prev[key][1] / last['L1_Bx']):5.2f}"
            oP = f"{math.log2(prev[key][2] / last['L1_psi']):5.2f}" if last["L1_psi"] > 0 and prev[key][2] > 0 else oP
        prev[key] = (r["N"], last["L1_Bx"], last["L1_psi"])
        L.append(f"{name:<28} {r['lambda']:8.2f} {last['L1_Bx']:10.3e} {oB} {last['L1_psi']:10.3e} {oP} "
                 f"{last['rel_err_Bx']:8.2e} {last['exact_divB_L2_ratio']:9.3e}")
    return "\n".join(L)


def main():
    res = collect()
    js = f"{M}/results/tables/I5.json"
    if "--check" in sys.argv:
        old = json.load(open(js))
        bad = [k for k in old if k not in res or any(
            abs(a[q] - b[q]) > 1e-10 * max(abs(a[q]), 1e-300) for a, b in zip(old[k]["rows"], res[k]["rows"])
            for q in ("L1_Bx", "L1_By", "L1_psi", "divB_L1"))]
        print("check:", "OK" if not bad and old.keys() == res.keys() else f"DIFF {bad}")
        sys.exit(1 if bad or old.keys() != res.keys() else 0)
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(res, open(js, "w"), indent=1)
    t = text(res)
    open(js.replace(".json", ".txt"), "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
