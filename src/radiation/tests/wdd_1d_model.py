#!/usr/bin/env python
"""1-D periodic Sn model of gate K4, built to grade options A / B / C for WDD under Krylov.

Replicates the K4 physics EXACTLY in structure (pure scattering, periodic box, Gaussian pulse,
Var(t) = Var0 + 2*D*t with D = chat/(3*chi_s)) but in 1-D with S2 so every discrete operator can
be formed EXPLICITLY as a matrix and inspected.  That is the point: on the GPU code these
operators are only ever applied, never seen.

Arms:
  step      the operator SnApplyA implements today (local, Krylov-compatible)
  wdd_exact the operator the ordered sweep + SnResidual converge to (faces exchanged across
            every block boundary, so the answer is decomposition-INDEPENDENT)
  wdd_blk   option B: WDD inside a block, incoming face at a block inlet taken from the upwind
            CELL (step closure) because the exact inlet face is not evaluable block-locally
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl

MU = np.array([+1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)])   # S2: int mu^2 = 1/3 exactly
WT = np.array([0.5, 0.5])                                # sum w = 1

def build(N, dx, chi, rdt, w, scheme, nb=None):
    """Return (Aop, cellrows) for one backward-Euler step.

    Unknown vector: [I(m=0,i), I(m=1,i)] then, for wdd, [f(m,i)] (outgoing face of cell i in
    that ordinate's sweep direction).  Returns the sparse matrix and the row/col index of the
    cell unknowns so the RHS and the solution can be extracted.
    """
    nm = 2
    ncell = nm * N
    if scheme == "step":
        n = ncell
    else:
        n = 2 * ncell
    A = sp.lil_matrix((n, n))
    def ci(m, i): return m * N + (i % N)
    def fi(m, i): return ncell + m * N + (i % N)
    rw = 1.0 / w
    for m in range(nm):
        a = abs(MU[m]) / dx
        di = 1 if MU[m] > 0 else -1          # sweep direction
        for i in range(N):
            up = (i - di) % N                # upwind cell
            r = ci(m, i)
            if scheme == "step":
                A[r, r] += rdt + chi + a
                A[r, ci(m, up)] -= a
            else:
                inlet = (nb is not None) and (i % nb == (0 if di > 0 else nb - 1))
                A[r, r] += rdt + chi + rw * a
                if scheme == "wdd_blk" and inlet:
                    # option B: the exact inlet face is a non-local function of the whole
                    # upstream chain, so it is replaced by the upwind CELL value (step closure).
                    A[r, ci(m, up)] -= rw * a
                else:
                    A[r, fi(m, up)] -= rw * a
                # face-closure row for cell i
                rf = fi(m, i)
                A[rf, rf] += 1.0
                A[rf, r] -= rw
                if scheme == "wdd_blk" and inlet:
                    A[rf, ci(m, up)] += (rw - 1.0)
                else:
                    A[rf, fi(m, up)] += (rw - 1.0)
            # scattering: -chi_s * J_i  (chi_a = 0 so chi = chi_s)
            for mm in range(nm):
                A[r, ci(mm, i)] -= chi * WT[mm]
    return sp.csr_matrix(A), ncell

def run(N, L, chi, w, scheme, nb, nstep, tend, chat=1.0):
    dx = L / N
    x = (np.arange(N) + 0.5) * dx - L / 2
    sigma0 = 0.4
    I0 = np.exp(-0.5 * (x / sigma0) ** 2) + 1e-12
    I = np.concatenate([I0, I0]) * 0.5      # isotropic
    D_ana = chat / (3.0 * chi)
    dt = tend / nstep
    rdt = 1.0 / (chat * dt)
    A, ncell = build(N, dx, chi, rdt, w, scheme, nb)
    lu = spl.splu(sp.csc_matrix(A))
    ts, vs, tot = [], [], []
    def moments(Iv):
        J = WT[0] * Iv[0:N] + WT[1] * Iv[N:2 * N]
        W = J.sum()
        xb = (J * x).sum() / W
        return W, (J * (x - xb) ** 2).sum() / W
    t = 0.0
    W, v = moments(I); ts.append(t); vs.append(v); tot.append(W)
    for s in range(nstep):
        rhs = np.zeros(A.shape[0]); rhs[:ncell] = rdt * I
        sol = lu.solve(rhs)
        I = sol[:ncell]
        t += dt
        W, v = moments(I); ts.append(t); vs.append(v); tot.append(W)
    ts = np.array(ts); vs = np.array(vs); tot = np.array(tot)
    k0 = max(1, len(ts) // 5)
    D_meas = np.polyfit(ts[k0:], vs[k0:], 1)[0] / 2.0
    cons = abs(tot[-1] - tot[0]) / abs(tot[0])
    return D_meas / D_ana, cons

if __name__ == "__main__":
    L, N = 4.0, 512
    dx = L / N
    NSTEP = 200
    print(f"1-D S2 periodic model of gate K4:  N={N}  dx={dx:.4e}  sigma0=0.4 (sigma0/dx={0.4/dx:.0f})")
    print("End time chosen per arm so Var grows ~50% of Var0, matching deck_b.in's arms.")
    print("ratio = D_measured / D_analytic;  D_analytic = chat/(3 chi_s), chat = 1\n")
    hdr = f"{'tau_cell':>9s} {'step':>7s} {'wdd w=1':>8s} {'wdd_exact':>10s}" + "".join(f"{'blk'+str(nb):>8s}" for nb in (8,16,32,128))
    print(hdr)
    for tau in (0.2, 0.5, 1.0, 3.0):
        chi = tau / dx
        tend = 0.12 * chi          # => 2*D*tend = 0.08 = 0.5*Var0
        out = [tau]
        out.append(run(N, L, chi, 1.0,  "step", None, NSTEP, tend)[0])
        out.append(run(N, L, chi, 1.0,  "wdd",  None, NSTEP, tend)[0])
        ex = run(N, L, chi, 0.53, "wdd", None, NSTEP, tend)
        out.append(ex[0])
        for nb in (8, 16, 32, 128):
            out.append(run(N, L, chi, 0.53, "wdd_blk", nb, NSTEP, tend)[0])
        print(f"{out[0]:9.2f} {out[1]:7.3f} {out[2]:8.3f} {out[3]:10.3f}" + "".join(f"{v:8.3f}" for v in out[4:])
              + f"   [Er cons {ex[1]:.1e}]")
