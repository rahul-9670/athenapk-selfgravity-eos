#!/usr/bin/env python
"""1-D periodic S2 model of gate K4 extended to grade CELL-CENTRED Krylov-compatible schemes.

Companion to wdd_1d_model.py (same physics, same calibration: 'step' and 'wdd_exact' arms
reproduce that file). New arms, all with the unknown = cell intensity only:

  trwdd_K   truncated-recursion WDD: the face recurrence f(i) = rw x(i) - (rw-1) f(i-1) is run
            block-locally from a STEP closure (inlet := upwind ghost cell), and the faces are
            exchanged K times (K+1 block-local recomputes).  The incoming face of a cell is then
            the exact WDD face truncated at (i - i_first + 1) + K*nb cells upstream, with the
            closure (-(rw-1))^m x(i-m).  Fixed linear map -> valid Krylov operator.
  sou_k     second-order upwind kappa-family: a[(1+k/2) x_i - (1+k) x_{i-1} + (k/2) x_{i-2}]
  theta_t   central blended with step: a[(1-t)(x_{i+1}-x_{i-1})/2 + t (x_i - x_{i-1})]

Measures: (1) K4 ratio D_meas/D_ana, Er conservation; (2) decomposition dependence of trwdd
(nb = 8/16/32/128, K = 0/1/2/3); (3) a thin/thick INTERFACE test: min intensity (negativity)
downstream/upstream of a bright thick slab embedded in a dark thin medium.
"""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl

MU = np.array([+1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)])
WT = np.array([0.5, 0.5])

def face_map(N, rw, di, nb, K):
    """T (N x N): outgoing face of each cell as a linear map of the cell values, for one
    ordinate with sweep direction di, blocks of nb cells, K exchange rounds.  Returns T_int
    (faces after the LAST recompute, used for upwind cells inside the same block) and T_gh
    (faces after the last EXCHANGE = one round behind, used at block inlets)."""
    q = -(rw - 1.0)
    def T_round(r):
        T = np.zeros((N, N))
        for i in range(N):
            # position within block in sweep order
            if di > 0:
                pos = i % nb
            else:
                pos = (nb - 1) - (i % nb)
            m = pos + 1 + r * nb            # truncation distance
            for k in range(m):
                T[i, (i - k * di) % N] += rw * q**k
            T[i, (i - m * di) % N] += q**m   # step closure at the truncation cell
        return T
    return T_round(K), T_round(max(K - 1, 0)) if K >= 1 else None

def build(N, dx, chi_s, chi_a, rdt, scheme, w=0.53, nb=None, K=0, kappa=1.0, theta=0.06):
    nm = 2
    n = nm * N
    A = np.zeros((n, n))
    def ci(m, i): return m * N + (i % N)
    rw = 1.0 / w
    chi = chi_s + chi_a
    for m in range(nm):
        a = abs(MU[m]) / dx
        di = 1 if MU[m] > 0 else -1
        rows = [ci(m, i) for i in range(N)]
        if scheme == "step":
            for i in range(N):
                A[ci(m, i), ci(m, i)] += rdt + chi + a
                A[ci(m, i), ci(m, i - di)] -= a
        elif scheme == "trwdd":
            T_int, T_gh = face_map(N, rw, di, nb, K)
            for i in range(N):
                r = ci(m, i)
                A[r, r] += rdt + chi + rw * a
                up = (i - di) % N
                same_block = (up // nb) == (i // nb)
                T = T_int if (same_block or K == 0) else T_gh
                if K == 0 and not same_block:
                    # no exchange at all: inlet := ghost cell value (step closure)
                    A[r, ci(m, up)] -= rw * a
                else:
                    for jx in range(N):
                        if T[up, jx] != 0.0:
                            A[r, ci(m, jx)] -= rw * a * T[up, jx]
        elif scheme == "sou":
            k = kappa
            for i in range(N):
                r = ci(m, i)
                A[r, r] += rdt + chi + a * (1.0 + 0.5 * k)
                A[r, ci(m, i - di)] -= a * (1.0 + k)
                A[r, ci(m, i - 2 * di)] += a * 0.5 * k
        elif scheme == "theta":
            t = theta
            for i in range(N):
                r = ci(m, i)
                A[r, r] += rdt + chi + a * t
                A[r, ci(m, i - di)] -= a * (0.5 * (1.0 - t) + t)
                A[r, ci(m, i + di)] += a * 0.5 * (1.0 - t)
        else:
            raise ValueError(scheme)
        # isotropic scattering source
        for i in range(N):
            for mm in range(nm):
                A[ci(m, i), ci(mm, i)] -= chi_s * WT[mm]
    return sp.csr_matrix(A)

def k4(N, L, chi, scheme, nstep, tend, chat=1.0, **kw):
    dx = L / N
    x = (np.arange(N) + 0.5) * dx - L / 2
    sigma0 = 0.4
    I0 = np.exp(-0.5 * (x / sigma0) ** 2) + 1e-12
    I = np.concatenate([I0, I0]) * 0.5
    D_ana = chat / (3.0 * chi)
    dt = tend / nstep
    rdt = 1.0 / (chat * dt)
    A = build(N, dx, chi, 0.0, rdt, scheme, **kw)
    lu = spl.splu(sp.csc_matrix(A))
    ts, vs, tot, mins = [], [], [], []
    def moments(Iv):
        J = WT[0] * Iv[0:N] + WT[1] * Iv[N:2 * N]
        W = J.sum(); xb = (J * x).sum() / W
        return W, (J * (x - xb) ** 2).sum() / W
    t = 0.0
    W, v = moments(I); ts.append(t); vs.append(v); tot.append(W)
    for s in range(nstep):
        I = lu.solve(rdt * I)
        t += dt
        W, v = moments(I); ts.append(t); vs.append(v); tot.append(W); mins.append(I.min())
    ts = np.array(ts); vs = np.array(vs); tot = np.array(tot)
    k0 = max(1, len(ts) // 5)
    D_meas = np.polyfit(ts[k0:], vs[k0:], 1)[0] / 2.0
    cons = abs(tot[-1] - tot[0]) / abs(tot[0])
    return D_meas / D_ana, cons, min(mins) / I.max()

def interface(N, L, tau_thick, tau_thin, scheme, chat=1.0, **kw):
    """Steady state: a bright emitting THICK slab (cells N/2-8 .. N/2+8, chi_a = tau_thick/dx,
    source S = chi_a * 1) in a dark THIN absorbing medium (chi_a = tau_thin/dx, S = 0).  rdt = 0
    (steady).  Reports min I / max I and the location of the most negative value relative to
    the slab, for the mu > 0 ordinate."""
    dx = L / N
    chi_a = np.full(N, tau_thin / dx)
    lo, hi = N // 2 - 8, N // 2 + 8
    chi_a[lo:hi] = tau_thick / dx
    # build with spatially varying chi_a: rebuild the diagonal by hand
    rdt = 1.0e-12
    A = build(N, dx, 0.0, 0.0, rdt, scheme, **kw).toarray()
    for m in range(2):
        for i in range(N):
            A[m * N + i, m * N + i] += chi_a[i]
    b = np.zeros(2 * N)
    for m in range(2):
        b[m * N + lo:m * N + hi] = chi_a[lo:hi] * 1.0
    I = np.linalg.solve(A, b)
    Ip = I[:N]
    imin = int(np.argmin(Ip))
    side = "inside" if lo <= imin < hi else ("UPSTREAM" if imin < lo else "downstream")
    return Ip.min() / Ip.max(), imin - (hi - 1), side

if __name__ == "__main__":
    L, N = 4.0, 512
    dx = L / N
    NSTEP = 200
    print(f"1-D S2 periodic K4 model, N={N} dx={dx:.4e}\n")
    taus = (0.2, 0.5, 1.0, 3.0)
    arms = [("step", {}), ("trwdd_exact(nb=N)", dict(scheme="trwdd", nb=N, K=0)),
            ]
    for nb in (8, 16, 32, 128):
        for K in (0, 1, 2, 3):
            arms.append((f"trwdd nb={nb} K={K}", dict(scheme="trwdd", nb=nb, K=K)))
    for kap in (1.0, 0.94, 0.8):
        arms.append((f"sou kappa={kap}", dict(scheme="sou", kappa=kap)))
    for th in (0.0, 0.06, 0.2):
        arms.append((f"theta={th}", dict(scheme="theta", theta=th)))
    print(f"{'arm':>24s}" + "".join(f"{'tau='+str(t):>12s}" for t in taus) + "   [cons, minI/maxI at tau=1]")
    for name, kw in arms:
        kw = dict(kw); sch = kw.pop("scheme", "step")
        out = []
        for tau in taus:
            chi = tau / dx
            tend = 0.12 * chi
            r, c, mn = k4(N, L, chi, sch, NSTEP, tend, **kw)
            out.append((r, c, mn))
        print(f"{name:>24s}" + "".join(f"{o[0]:12.4f}" for o in out) + f"   [{out[2][1]:.1e}, {out[2][2]:+.2e}]")
        sys.stdout.flush()
    print("\nINTERFACE test (steady, bright thick slab in dark thin medium), mu>0 ordinate:")
    print(f"{'arm':>24s} {'tau_thick':>9s} {'tau_thin':>8s} {'minI/maxI':>10s} {'where':>10s}")
    for name, kw in [("step", {}), ("trwdd exact", dict(scheme="trwdd", nb=512, K=0)),
                     ("sou kappa=1", dict(scheme="sou", kappa=1.0)),
                     ("sou kappa=0.94", dict(scheme="sou", kappa=0.94)),
                     ("theta=0.06", dict(scheme="theta", theta=0.06))]:
        kw = dict(kw); sch = kw.pop("scheme", "step")
        for tt, tn in ((3.0, 1e-3), (10.0, 1e-3), (1.0, 1e-3), (3.0, 0.3)):
            mn, off, side = interface(128, 1.0, tt, tn, sch, **kw)
            print(f"{name:>24s} {tt:9.1f} {tn:8.1e} {mn:+10.3e} {side:>10s} (offset {off:+d} cells from slab exit)")
