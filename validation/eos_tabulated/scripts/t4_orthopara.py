#!/usr/bin/env python3
"""T-4 (EOS_VALIDATION.md C4): how much of the E2 Maxwell-identity residual (2.46e-3) comes from the ortho/para
modelling of src/eos/gen_eos_table.py (A3.5)?

The generator takes the H2 dissociation equilibrium from the EQUILIBRIUM spin-weighted rotational partition function
Zrot_ns (para + ortho, levels measured from para J = 0), but the rotational ENERGY from a FROZEN 3:1 ortho:para mixture
with each species measured from its own ground state. Gate 3 of the consistency test checks the identity
    -rho^2 (du/drho)_T = T (dP/dT)_rho - P          (u = e_int/rho)
which holds exactly for any P, u derived from one free energy. Here it is evaluated, on the gate's own states and with
the generator's own Saha solver, for three rotational-energy models (P and the composition are the same in all three):
  frozen    the shipped model (gen_eos_table.erot_per_H2)
  equil     u_rot = k T^2 d ln Zrot_ns / dT, the energy of the SAME partition function K_d uses (consistent)
  frozen0   frozen 3:1 with ortho measured from para J = 0 (diagnostic: frozen + the ortho zero point 2 k theta_rot)
  chieff    the shipped frozen energy, but each dissociated molecule is charged chi_d - (3/4) 2 k theta_rot (the
            dissociation energy from the frozen 3:1 ground mixture instead of from para J = 0); K_d unchanged. Equal to
            frozen0 up to a constant per unit mass, so the cold limit u -> 0 is kept (candidate minimal fix)
each at three finite-difference steps (truncation error scales as h^2, a model inconsistency does not).
Also reported: what switching the energy model would change in the EOS itself (u, P/u, Gamma1) at selected states.
No table is rebuilt; the shipped generator is imported unchanged and only its erot_per_H2 is swapped in memory.

  python3 scripts/t4_orthopara.py > results/tables/T4_orthopara.txt
"""
import importlib.util
import os

import numpy as np

_REPO = os.environ.get("REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
spec = importlib.util.spec_from_file_location("gen_eos_table", _REPO + "/src/eos/gen_eos_table.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
k_B, th = G.k_B, G.theta_rot
frozen_shipped = G.erot_per_H2


def erot_equil(T):
    # mean rotational energy of the spin-weighted equilibrium sum used in K_d (levels from para J = 0)
    J = np.arange(0, 60)
    w = np.where(J % 2 == 0, 1.0, 3.0)
    E = J * (J + 1) * k_B * th
    x = J * (J + 1) * th / np.maximum(T, 1.0)
    g = w * (2 * J + 1) * np.exp(-x)
    return np.sum(g * E) / np.sum(g)


def erot_frozen0(T):
    # frozen 3:1, ortho measured from para J = 0 (adds 2 k theta_rot per ortho molecule)
    return frozen_shipped(T) + 0.75 * 2.0 * k_B * th


MODELS = {"frozen": frozen_shipped, "equil": erot_equil, "frozen0": erot_frozen0}
P_e_shipped = G.P_e_of_rho_T


def P_e_chieff(rho, T):
    P, e = P_e_shipped(rho, T)
    s = G.solve_saha(rho, T)
    return P, e - 0.75 * 2.0 * k_B * th * (s["HI"] + s["HII"]) / 2.0
rhos = np.logspace(-16, -4, 13)   # the states of gate 3 (src/eos/tests/test_eos_consistency.py)
Ts = np.logspace(1.2, 4.3, 16)


def residual(rho, T, h):
    Pp, ep = G.P_e_of_rho_T(rho * (1 + h), T)
    Pm, em = G.P_e_of_rho_T(rho * (1 - h), T)
    dudrho = (ep / (rho * (1 + h)) - em / (rho * (1 - h))) / (2 * h * rho)
    lhs = -rho * rho * dudrho
    PTp, _ = G.P_e_of_rho_T(rho, T * (1 + h))
    PTm, _ = G.P_e_of_rho_T(rho, T * (1 - h))
    P0, _ = G.P_e_of_rho_T(rho, T)
    rhs = T * (PTp - PTm) / (2 * h * T) - P0
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), P0)


def gamma1(rho, T, h=1e-4):
    # c_s^2 = P_rho|u + (P/rho^2) P_u|rho from central differences in (rho, T) (as scripts/worked_saha.py)
    P0, E0 = G.P_e_of_rho_T(rho, T)
    Pr1, Er1 = G.P_e_of_rho_T(rho * (1 + h), T); Pr0, Er0 = G.P_e_of_rho_T(rho * (1 - h), T)
    PT1, ET1 = G.P_e_of_rho_T(rho, T * (1 + h)); PT0, ET0 = G.P_e_of_rho_T(rho, T * (1 - h))
    P_rT = (Pr1 - Pr0) / (2 * h * rho); u_rT = (Er1 / (rho * (1 + h)) - Er0 / (rho * (1 - h))) / (2 * h * rho)
    P_Tr = (PT1 - PT0) / (2 * h * T); u_Tr = (ET1 - ET0) / rho / (2 * h * T)
    cs2 = (P_rT - P_Tr * u_rT / u_Tr) + P0 / rho ** 2 * (P_Tr / u_Tr)
    return rho * cs2 / P0


print("# T-4: Maxwell identity -rho^2 (du/drho)_T = T (dP/dT)_rho - P on the 13 x 16 states of gate 3 (interior 11 x 14 =")
print("# 154), relative residual |lhs - rhs| / max(|lhs|, |rhs|, P); generator src/eos/gen_eos_table.py unchanged except")
print("# erot_per_H2 swapped in memory. Bands by T [K].")
bands = [(0, 300, "T < 300"), (300, 1000, "300-1e3"), (1000, 5000, "1e3-5e3 (H2 dissociation)"),
         (5000, 3e4, "5e3-2e4 (H ionization)")]
for name, f in list(MODELS.items()) + [("chieff", frozen_shipped)]:
    G.erot_per_H2 = f
    G.P_e_of_rho_T = P_e_chieff if name == "chieff" else P_e_shipped
    for h in (2e-3, 1e-3, 5e-4):
        res = {b[2]: [] for b in bands}
        worst, where = 0.0, None
        for rho in rhos[1:-1]:
            for T in Ts[1:-1]:
                r = residual(rho, T, h)
                if r > worst: worst, where = r, (rho, T)
                for lo, hi, lab in bands:
                    if lo <= T < hi: res[lab].append(r)
        line = "  ".join(f"{lab}: {max(v):.2e}" for lab, v in res.items() if v)
        print(f"{name:8s} h={h:.0e}  worst {worst:.3e} at rho {where[0]:.0e} g/cc, T {where[1]:.0f} K | max by band  {line}")
G.erot_per_H2 = frozen_shipped
G.P_e_of_rho_T = P_e_shipped

print("\n# What the energy model changes in the EOS (frozen = shipped; equil = consistent with K_d)")
print("# rho [g/cc]  T [K]   u_frozen [erg/g]  u_equil/u_frozen-1  P/u frozen  P/u equil  Gamma1 frozen  Gamma1 equil"
      "  | chieff: u/u_frozen-1  Gamma1")
for rho, T in [(1e-14, 10.0), (1e-14, 30.0), (1e-14, 50.0), (1e-14, 100.0), (1e-14, 200.0), (1e-14, 500.0),
               (1e-10, 1000.0), (1e-9, 2000.0), (1e-8, 3000.0), (1e-6, 5000.0)]:
    out = []
    for f in (frozen_shipped, erot_equil):
        G.erot_per_H2 = f
        P, E = G.P_e_of_rho_T(rho, T)
        out.append((E / rho, P / E, gamma1(rho, T)))
    G.erot_per_H2 = frozen_shipped
    G.P_e_of_rho_T = P_e_chieff
    P, E = G.P_e_of_rho_T(rho, T)
    ce = (E / rho, gamma1(rho, T))
    G.P_e_of_rho_T = P_e_shipped
    print(f"{rho:9.0e} {T:7.0f}   {out[0][0]:.6e}   {out[1][0] / out[0][0] - 1:+.3e}        "
          f"{out[0][1]:.5f}    {out[1][1]:.5f}    {out[0][2]:.5f}       {out[1][2]:.5f}"
          f"       | {ce[0] / out[0][0] - 1:+.3e}  {ce[1]:.5f}")
