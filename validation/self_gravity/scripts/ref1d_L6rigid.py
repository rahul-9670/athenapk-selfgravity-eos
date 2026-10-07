#!/usr/bin/env python
"""L6d 1D reference: the ambient model of ref1d_L6amb.py case a, but closed by a RIGID wall at R (the 3D box
is closed by reflecting_val walls; R = volume-equivalent radius of [-8, 8]^3). No pressure ramp.
"""
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
R = 8.0 * (6 / np.pi) ** (1 / 3)
n = 200
w = (e_be[-1] - e_be[-2]) * 1.02 ** np.arange(1, n + 1)
w *= (R - xi) / w.sum()
edges = np.r_[e_be, xi + np.cumsum(w)]
rho = np.r_[rho_be, np.full(n, rho_e / 100)]
P = np.r_[rho_be, np.full(n, rho_e)]
s = Lagr1D(edges, rho, four_pi_G=1.0, eos="adiabatic", gamma=g, e=P / ((g - 1) * rho), p_ext=0.0, outer="rigid")
t, rc = [], []
def cb(q):
    t.append(q.t); rc.append(q.rho()[0])
s.run(t_end=30.0, every=0.01, callback=cb)
t, rc = np.array(t), np.array(rc)
np.savez_compressed(os.path.join(OUT, "L6rigid.npz"), t=t, rho_c=rc)
for a, b in ((5, 10), (10, 15), (15, 20), (20, 25), (25, 30)):
    sel = (t >= a) & (t < b)
    print(f"rigid  t in [{a},{b}): rho_c min {rc[sel].min():.3f} max {rc[sel].max():.3f}")
