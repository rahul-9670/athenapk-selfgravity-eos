#!/usr/bin/env python3
"""Item #5: Fleck temporal error, dt-refinement on the 0-D relaxation (uniform G11 physics).
Exact reference: the ODE  dEr/dt = c*chi*(T^4 - Er),  dT/dt = -(c*chi/1.5)*(T^4 - Er)
(rho=1, gamma=5/3 => e = 1.5 T; c = chat = 10, chi_a = 20, arad = 1) from (T, Er) = (1, 1e-30)."""
import glob, os, sys
import numpy as np, h5py
from scipy.integrate import solve_ivp

C, CHI = 10.0, 20.0
def rhs(t, y):
    T, Er = y
    x = C * CHI * (T**4 - Er)
    return [-x / 1.5, x]
TEND = 2.0**-6
sol = solve_ivp(rhs, [0, TEND], [1.0, 1e-30], method="DOP853", rtol=1e-13, atol=1e-16,
                dense_output=True)
def exact(t):
    return sol.sol(t)

def final_state(d):
    fs = sorted(glob.glob(os.path.join(d, "parthenon.out0.*.phdf")))
    out = []
    for p in fs:
        with h5py.File(p, "r") as f:
            t = float(f["Info"].attrs["Time"])
            prim = np.array(f["prim"]); Er = np.array(f["rad.Er"]).ravel()
            rho = prim[:, 0].ravel(); pr = prim[:, 4].ravel()
            T = pr / rho
            fl = np.array(f["sn.fleck"]).ravel() if "sn.fleck" in f else None
            out.append(dict(t=t, T=T.mean(), Tspread=T.max()-T.min(), Er=Er.mean(),
                            Erspread=Er.max()-Er.min(), fleck=(fl.mean() if fl is not None else np.nan)))
    return out

Te, Ere = exact(TEND)
print("exact at t=%.6g: T = %.12f  Er = %.12f   (T_eq=0.76698, tau_relax~2.3e-3 => %.1f e-folds)"
      % (TEND, Te, Ere, TEND*440))
print("%-6s %-5s %-12s %-14s %-11s %-11s %-9s %-9s %-8s" % ("alpha","N","dt","T_end","errT","errEr","S_T1","f_last","spread"))
rows = {}
for A in ("1.0", "0.5", "0.0"):
    prev = None
    for N in (1,2,4,8,16,32,64,128,256,512,1024):
        d = "arm_a%s_N%d" % (A, N)
        if not os.path.isdir(d): continue
        st = final_state(d)
        s = st[-1]
        if abs(s["t"] - TEND) > 1e-12:
            print("  %s: last dump at t=%.6g != tlim" % (d, s["t"])); continue
        dt = TEND / N
        errT = (s["T"] - Te) / Te
        errE = (s["Er"] - Ere) / Ere
        S1 = C * dt * CHI * (8.0/3.0)   # stiffness at T=1
        rows[(A, N)] = (dt, errT, errE, s["T"], s["Er"])
        ratio = ""
        if prev is not None and prev != 0:
            ratio = " ratio %.3f -> p=%.2f" % (prev/errT, np.log2(abs(prev/errT))) if errT != 0 else ""
        print("%-6s %-5d %-12.5g %-14.10f %-11.3e %-11.3e %-9.3g %-9.4f %-8.1e%s"
              % (A, N, dt, s["T"], errT, errE, S1, s["fleck"], s["Tspread"], ratio))
        prev = errT
    print()
# Order from the ladder tail (Richardson-style): fit log|err| vs log dt on N>=64
for A in ("1.0","0.5","0.0"):
    ks = [k for k in rows if k[0]==A and k[1] >= 64]
    if len(ks) < 3: continue
    x = np.log([rows[k][0] for k in ks]); y = np.log([abs(rows[k][1]) for k in ks])
    p = np.polyfit(x, y, 1)[0]
    print("alpha=%s: fitted order over N>=64 (dt <= %.3g): p = %.3f" % (A, TEND/64, p))
# production dt
for A in ("1.0","0.5"):
    print("alpha=%s: |errT| at dt=1.53e-5 (N=1024, closest to production dt 1.1e-5): %.3e"
          % (A, abs(rows[(A,1024)][1])))
