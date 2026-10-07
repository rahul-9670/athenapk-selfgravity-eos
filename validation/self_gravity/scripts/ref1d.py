#!/usr/bin/env python
"""1D Lagrangian reference runs (scripts/lagr1d.py) with the SAME initial states as the 3D decks.

  ref1d.py L7 [N]          production collapse_be IC (Tomida, f = 5), barotropic rho_crit = 1e-13 g/cc
  ref1d.py L6 xi f [N]      exact BE sphere xi_max, overdensity f, isothermal
  ref1d.py L3b [N]          n = 5 polytrope with P uniform inside 0.9 R, gamma = 1.3
  ref1d.py L5 [N]           Evrard collapse
Writes results/ref1d/<case>_N<N>.npz: time series (t, rho_c[, energies]) and snapshots.
The 3D ambient outside a pressure-bounded sphere is static (collapse_be zeroes momentum there), so the
1D outer boundary is a constant external pressure equal to the ambient's.
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(__file__))
from lagr1d import Lagr1D, mass_grid  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "ref1d")
os.makedirs(OUT, exist_ok=True)
RHO0_FHC = 5.46682784129e-19  # g/cc, code density unit of the FHC IC (mass 6, T 10 K, f 5)


def le_isothermal(xmax):
    s = solve_ivp(lambda x, y: [y[1], np.exp(-y[0]) - 2 * y[1] / x], (1e-6, xmax + 1e-9),
                  [1e-12 / 6, 1e-6 / 3], rtol=1e-12, atol=1e-14, dense_output=True)
    return lambda r: np.exp(-s.sol(np.clip(np.ravel(r), 1e-6, xmax))[0]).reshape(np.shape(r))


def snapshots(sim, rho_marks):
    """Callback factory: store a profile each time rho_c first exceeds the next mark."""
    store = {"t": [], "rho_c": [], "snaps": []}
    marks = list(rho_marks)
    def cb(s):
        rc = s.rho()[0]
        store["t"].append(s.t)
        store["rho_c"].append(rc)
        if s.e is not None:
            store.setdefault("E", []).append(s.energies())
        while marks and rc >= marks[0]:
            r, rho, u = s.centres()
            store["snaps"].append((marks.pop(0), s.t, r.copy(), rho.copy(), u.copy()))
    return store, cb


def save(name, store, extra=None):
    d = {"t": np.array(store["t"]), "rho_c": np.array(store["rho_c"])}
    if "E" in store:
        d["E"] = np.array(store["E"])
    for i, (mark, t, r, rho, u) in enumerate(store["snaps"]):
        d[f"snap{i}_mark"], d[f"snap{i}_t"], d[f"snap{i}_r"], d[f"snap{i}_rho"], d[f"snap{i}_u"] = mark, t, r, rho, u
    if extra:
        d.update(extra)
    np.savez_compressed(os.path.join(OUT, name + ".npz"), **d)
    print("saved", name, "steps recorded", len(d["t"]), "final t", d["t"][-1], "rho_c", d["rho_c"][-1])


def run_L7(N):
    f, rc, rcsq = 5.0, 6.45, 26.0 / 3.0
    rho_of = lambda r: f * (1 + r ** 2 / rcsq) ** -1.5
    edges, rho = mass_grid(rho_of, rc, N, geometric=60.0)
    p_ext = float(rho_of(np.array(rc)))
    s = Lagr1D(edges, rho, four_pi_G=1.0, eos="barotropic", cs=1.0, gamma=1.4, rho_crit=1e-13 / RHO0_FHC,
               p_ext=p_ext, outer="pressure")
    store, cb = snapshots(s, [f * 10 ** k for k in np.arange(0.5, 6.01, 0.5)])
    s.run(stop=lambda q: q.rho()[0] > 2e5 or q.t > 3.0, every=1e-5, callback=cb)  # fine: crossing times
    save(f"L7_N{N}", store, {"rho_crit_code": 1e-13 / RHO0_FHC})


def run_L6(xi, f, N):
    prof = le_isothermal(xi)
    rho_of = lambda r: f * prof(r)
    edges, rho = mass_grid(rho_of, xi, N, geometric=5.0)
    s = Lagr1D(edges, rho, four_pi_G=1.0, eos="barotropic", cs=1.0, gamma=1.4, rho_crit=np.inf,
               p_ext=float(rho_of(np.array(xi))), outer="pressure")
    store, cb = snapshots(s, [10.0, 100.0, 1000.0])
    s.run(stop=lambda q: q.rho()[0] > 1e4 or q.t > 30.0, every=0.02, callback=cb)
    save(f"L6_xi{xi:g}_f{f:g}_N{N}", store)


def run_L3b(N):
    rho_of = lambda r: (1 + r ** 2 / 3) ** -2.5
    Pc = 1 / 6
    R = 5.12
    edges, rho = mass_grid(rho_of, R, N, geometric=8.0)
    rc = 0.5 * (edges[1:] + edges[:-1])
    P = np.where(rc < 0.9, Pc * (1 + 0.81 / 3) ** -3, Pc * (1 + rc ** 2 / 3) ** -3)
    g = 1.3
    s = Lagr1D(edges, rho, four_pi_G=1.0, eos="adiabatic", gamma=g, e=P / ((g - 1) * rho),
               p_ext=float(Pc * (1 + R ** 2 / 3) ** -3), outer="pressure", q_quad=2.0, q_lin=0.3)
    store, cb = snapshots(s, [2.0, 5.0, 10.0, 30.0, 100.0])
    s.run(t_end=6.0, every=0.01, callback=cb)
    save(f"L3b_N{N}", store)


def run_L5(N):
    # equal-mass shells: M(<r) = r^2 for rho = 1/(2 pi r), M = R = 1
    m = np.linspace(0, 1, N + 1)
    edges = np.sqrt(m)
    rc = 0.5 * (edges[1:] + edges[:-1])
    vol = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = np.diff(m) / vol
    g = 5 / 3
    s = Lagr1D(edges, rho, four_pi_G=4 * np.pi, eos="adiabatic", gamma=g, e=0.05 * np.ones_like(rho),
               p_ext=0.0, outer="pressure", q_quad=2.0, q_lin=0.3)
    store, cb = snapshots(s, [])
    s.run(t_end=3.0, every=0.01, callback=cb)
    save(f"L5_N{N}", store)


if __name__ == "__main__":
    case = sys.argv[1]
    if case == "L7":
        run_L7(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    elif case == "L6":
        run_L6(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1000)
    elif case == "L3b":
        run_L3b(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    elif case == "L5":
        run_L5(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
