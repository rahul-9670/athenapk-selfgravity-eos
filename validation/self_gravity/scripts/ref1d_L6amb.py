#!/usr/bin/env python
"""Falsification for L6' (xi = 3, f = 1.1): is the 3D-vs-1D late drift the ambient/EOS model?
(a) 1D adiabatic gamma = 1.001 (as the 3D deck) with the 3D deck's light hot ambient (rho_edge/100 at
    p_edge) out to R = 9.92 (volume-equivalent radius of the [-8, 8]^3 box), constant P at R;
(b) 1D adiabatic gamma = 1.001, constant external pressure at the BE edge (no ambient).
Writes results/ref1d/L6amb_{a,b}.npz."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lagr1d import Lagr1D, mass_grid  # noqa: E402
from ref1d import le_isothermal, OUT  # noqa: E402

xi, f, g = 3.0, 1.1, 1.001
prof = le_isothermal(xi)
rho_of = lambda r: f * prof(r)
rho_e = float(rho_of(np.array(xi)))
e_be, rho_be = mass_grid(rho_of, xi, 1000, geometric=5.0)
for case in sys.argv[1:] or ["a", "b"]:
    if case == "a":
        R = 8.0 * (6 / np.pi) ** (1 / 3)
        s_ = np.linspace(0, 1, 201)[1:]
        dr0 = e_be[-1] - e_be[-2]
        q = 1.02
        n = 200
        # geometric spacing from xi to R starting at the last BE zone width
        k = np.arange(1, n + 1)
        w = dr0 * q ** k
        w *= (R - xi) / w.sum()
        e_amb = xi + np.cumsum(w)
        edges = np.r_[e_be, e_amb]
        rho = np.r_[rho_be, np.full(n, rho_e / 100)]
        P = np.r_[rho_be, np.full(n, rho_e)]
        p_ext = rho_e
    else:
        edges, rho, P, p_ext = e_be, rho_be, rho_be.copy(), rho_e
    s = Lagr1D(edges, rho, four_pi_G=1.0, eos="adiabatic", gamma=g, e=P / ((g - 1) * rho), p_ext=p_ext,
               outer="pressure")
    t, rc, mbe = [], [], []
    nbe = len(rho_be)
    def cb(q):
        t.append(q.t); rc.append(q.rho()[0]); mbe.append(q.r[nbe])
    s.run(t_end=30.0, every=0.02, callback=cb)
    np.savez_compressed(os.path.join(OUT, f"L6amb_{case}.npz"), t=np.array(t), rho_c=np.array(rc), r_be=np.array(mbe))
    print(case, "done; rho_c at t=2,5,10,15,20,25,28:",
          " ".join(f"{np.interp(x, t, rc):.3f}" for x in (2, 5, 10, 15, 20, 25, 28)),
          "| late (t>=15) min/max %.3f %.3f" % (min(np.array(rc)[np.array(t) >= 15]), max(np.array(rc)[np.array(t) >= 15])))
