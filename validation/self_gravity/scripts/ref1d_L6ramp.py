#!/usr/bin/env python
"""Falsification of the L6' residual: 1D ambient model (ref1d_L6amb case a) with the outer pressure
RAMPED as measured in 3D, P(t) = P_edge (1 + k t), k = 0.00857 (3D: +21.3 % at t = 25 from ambient inflow
through the outflow faces). If the 3D late-time rise of rho_c is that inflow, this 1D run reproduces it."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lagr1d import Lagr1D, mass_grid  # noqa: E402
from ref1d import le_isothermal, OUT  # noqa: E402

xi, f, g, k = 3.0, 1.1, 1.001, float(sys.argv[1]) if len(sys.argv) > 1 else 0.00857
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
s = Lagr1D(edges, rho, four_pi_G=1.0, eos="adiabatic", gamma=g, e=P / ((g - 1) * rho), p_ext=rho_e, outer="pressure")
t, rc = [], []
def cb(q):
    q.p_ext = rho_e * (1 + k * q.t)
    t.append(q.t); rc.append(q.rho()[0])
s.run(t_end=30.0, every=0.01, callback=cb)
t, rc = np.array(t), np.array(rc)
np.savez_compressed(os.path.join(OUT, f"L6ramp_k{k:g}.npz"), t=t, rho_c=rc)
for a, b in ((5, 10), (10, 15), (15, 20), (20, 25), (25, 30)):
    sel = (t >= a) & (t < b)
    print(f"k={k:g}  t in [{a},{b}): rho_c min {rc[sel].min():.3f} max {rc[sel].max():.3f}")
