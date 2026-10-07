#!/usr/bin/env python
"""1D spherical Lagrangian self-gravitating hydrodynamics -- the independent REFERENCE for the
spherically symmetric collapse tests (L3b n=5 polytrope collapse, L5 Evrard, L6 BE stability,
L7 BE -> Larson-Penston collapse).

Why it is a reference: in spherical symmetry gravity is exact, g = -G M(<r)/r^2, with no Poisson
solve, no boundary condition and no Cartesian sphere, and a Lagrangian mass grid follows the
collapse with no remapping. Scheme: staggered mesh (radii and velocities at shell boundaries,
density/pressure/energy in zones), kick-drift-kick leapfrog (2nd order), von Neumann-Richtmyer +
linear artificial viscosity for shocks. Converge it by doubling N.

EOS modes
  barotropic : P = cs^2 rho sqrt(1 + (rho/rho_crit)^(2(gamma-1)))   (= collapse_be's enforcement)
  adiabatic  : P = (gamma-1) rho e, e evolved with PdV + viscous heating
Outer boundary: 'pressure' (constant P_ext acting on the last shell) or 'rigid'.
Units: four_pi_G given; G = four_pi_G/(4 pi).
"""
import numpy as np


class Lagr1D:
    def __init__(self, r_edges, rho, *, four_pi_G=1.0, eos="barotropic", cs=1.0, gamma=1.4,
                 rho_crit=np.inf, e=None, p_ext=0.0, outer="pressure", q_lin=0.1, q_quad=2.0,
                 cfl=0.25):
        self.G = four_pi_G / (4 * np.pi)
        self.r = np.asarray(r_edges, float).copy()          # N+1 edges, r[0] = 0
        vol = 4 * np.pi / 3 * (self.r[1:] ** 3 - self.r[:-1] ** 3)
        self.dm = np.asarray(rho, float) * vol               # zone masses (fixed)
        self.menc = np.concatenate([[0.0], np.cumsum(self.dm)])  # mass inside each edge
        self.u = np.zeros_like(self.r)
        self.eos, self.cs, self.gamma, self.rho_crit = eos, cs, gamma, rho_crit
        self.e = None if e is None else np.asarray(e, float).copy()
        self.p_ext, self.outer = p_ext, outer
        self.ql, self.qq, self.cfl = q_lin, q_quad, cfl
        self.t = 0.0
        self.q = np.zeros_like(self.dm)

    # ------------------------------------------------------------------ state
    def rho(self):
        return self.dm / (4 * np.pi / 3 * (self.r[1:] ** 3 - self.r[:-1] ** 3))

    def pressure(self, rho):
        if self.eos == "barotropic":
            return self.cs ** 2 * rho * np.sqrt(1.0 + (rho / self.rho_crit) ** (2 * (self.gamma - 1)))
        return (self.gamma - 1) * rho * self.e

    def sound(self, rho):
        if self.eos == "barotropic":
            return self.cs * np.ones_like(rho)
        return np.sqrt(self.gamma * (self.gamma - 1) * np.maximum(self.e, 1e-300))

    def accel(self, rho):
        p = self.pressure(rho) + self.q
        a = np.zeros_like(self.r)
        # interface j (1..N-1): pressure force from zones j-1, j, half-zone masses
        mj = 0.5 * (self.dm[1:] + self.dm[:-1])
        a[1:-1] = -4 * np.pi * self.r[1:-1] ** 2 * (p[1:] - p[:-1]) / mj
        a[1:] -= self.G * self.menc[1:] / self.r[1:] ** 2
        if self.outer == "pressure":
            a[-1] += -4 * np.pi * self.r[-1] ** 2 * (self.p_ext - p[-1]) / (0.5 * self.dm[-1])
        else:
            a[-1] = 0.0
        a[0] = 0.0
        return a

    def visc(self, rho):
        du = self.u[1:] - self.u[:-1]
        comp = du < 0
        c = self.sound(rho)
        self.q = np.where(comp, rho * (self.qq * du ** 2 + self.ql * c * np.abs(du)), 0.0)

    def dt(self, rho):
        dr = self.r[1:] - self.r[:-1]
        c = self.sound(rho)
        du = np.abs(self.u[1:] - self.u[:-1])
        dt_cfl = self.cfl * np.min(dr / (c + du + 1e-300))
        # dynamical-time limit: without it a (near-)pressureless collapse is one giant step
        dt_dyn = 0.02 / np.sqrt(4 * np.pi * self.G * np.max(rho))
        return min(dt_cfl, dt_dyn)

    # ------------------------------------------------------------------ step
    def step(self, dt):
        rho = self.rho()
        self.visc(rho)
        a = self.accel(rho)
        self.u += 0.5 * dt * a
        if self.outer == "rigid":
            self.u[-1] = 0.0
        vol_old = self.r[1:] ** 3 - self.r[:-1] ** 3
        self.r += dt * self.u
        self.r[0] = 0.0
        if np.any(np.diff(self.r) <= 0):
            raise FloatingPointError(f"shell crossing at t = {self.t}")
        rho_new = self.rho()
        if self.e is not None:  # PdV with time-centred pressure (one predictor iteration)
            vol_new = self.r[1:] ** 3 - self.r[:-1] ** 3
            dV = 4 * np.pi / 3 * (vol_new - vol_old)
            p_old = self.pressure(rho) + self.q
            e_pred = self.e - p_old * dV / self.dm
            e_save, self.e = self.e, e_pred
            p_new = self.pressure(rho_new) + self.q
            self.e = e_save - 0.5 * (p_old + p_new) * dV / self.dm
        a = self.accel(rho_new)
        self.u += 0.5 * dt * a
        if self.outer == "rigid":
            self.u[-1] = 0.0
        self.t += dt

    def run(self, t_end=None, stop=None, every=None, callback=None, max_steps=10 ** 8):
        """Advance to t_end, or until stop(self) is True. callback(self) every `every` time units."""
        nxt = 0.0 if every else None
        for _ in range(max_steps):
            rho = self.rho()
            if every is not None and self.t >= nxt:
                callback(self)
                nxt += every
            if (t_end is not None and self.t >= t_end) or (stop is not None and stop(self)):
                break
            dt = self.dt(rho)
            if t_end is not None:
                dt = min(dt, t_end - self.t)
            if every is not None:
                dt = min(dt, max(nxt - self.t, 1e-14))
            self.step(dt)
        return self

    # ------------------------------------------------------------------ diagnostics
    def energies(self):
        rho = self.rho()
        um = 0.5 * (self.u[1:] + self.u[:-1])
        ek = 0.5 * np.sum(self.dm * um ** 2)
        eth = np.sum(self.dm * self.e) if self.e is not None else np.nan
        rc = 0.5 * (self.r[1:] + self.r[:-1])
        mc = 0.5 * (self.menc[1:] + self.menc[:-1])
        w = -self.G * np.sum(self.dm * mc / rc)
        return ek, eth, w

    def centres(self):
        return 0.5 * (self.r[1:] + self.r[:-1]), self.rho(), 0.5 * (self.u[1:] + self.u[:-1])


def mass_grid(rho_of_r, R, N, r_in_frac=0.0, geometric=1.0):
    """Edges for N zones on [0, R]; geometric > 1 concentrates zones toward the centre."""
    if geometric == 1.0:
        edges = np.linspace(0.0, R, N + 1)
    else:
        s = np.linspace(0.0, 1.0, N + 1)
        edges = R * (geometric ** s - 1) / (geometric - 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    # zone-average density by 8-point Gauss-like midpoint sub-sampling in r^2 dr
    sub = np.linspace(0, 1, 9)[:-1] + 1 / 16
    rr = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * sub[None, :]
    w = rr ** 2
    rho = np.sum(rho_of_r(rr) * w, axis=1) / np.sum(w, axis=1)
    return edges, rho


if __name__ == "__main__":
    # Self-test 1: pressure-free uniform sphere vs the exact cycloid (rho_c at t = 0.6 t_ff).
    G4 = 1.0
    edges, rho = mass_grid(lambda r: np.ones_like(r), 1.0, 400)
    s = Lagr1D(edges, rho, four_pi_G=G4, eos="barotropic", cs=1e-4, outer="pressure", p_ext=0.0)
    tff = np.pi * np.sqrt(3.0 / 8.0)
    s.run(t_end=0.6 * tff)
    from scipy.optimize import brentq
    b = brentq(lambda b: (2 / np.pi) * tff * (b + np.sin(b) * np.cos(b)) - s.t, 0, np.pi / 2)
    print(f"free fall: rho_c(0.6 t_ff) = {s.rho()[0]:.6f}  exact {np.cos(b) ** -6:.6f}")
    # Self-test 2: hydrostatic exact BE sphere (xi = 3, stable) stays put for 10 sound times.
    from scipy.integrate import solve_ivp
    le = solve_ivp(lambda x, y: [y[1], np.exp(-y[0]) - 2 * y[1] / x], (1e-6, 3.0), [1e-12 / 6, 1e-6 / 3],
                   rtol=1e-12, atol=1e-14, dense_output=True)
    rho_be = lambda r: np.exp(-le.sol(np.maximum(np.ravel(r), 1e-6))[0]).reshape(np.shape(r))
    edges, rho = mass_grid(rho_be, 3.0, 400)
    s = Lagr1D(edges, rho, four_pi_G=1.0, eos="barotropic", cs=1.0, p_ext=float(rho_be(np.array(3.0))),
               outer="pressure")
    s.run(t_end=10.0)
    print(f"BE xi=3 after 10 t_sound: rho_c = {s.rho()[0]:.5f} (1)  max|u| = {np.abs(s.u).max():.2e}")
