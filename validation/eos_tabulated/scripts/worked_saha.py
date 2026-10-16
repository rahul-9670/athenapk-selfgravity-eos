#!/usr/bin/env python3
"""Worked examples, physics side (EOS_VALIDATION.md Sec. W): the generator's own functions
(src/eos/gen_eos_table.py: solve_saha, P_e_of_rho_T) evaluated at the example states, with the
internal energy split into its terms and the adiabatic sound speed from the same identity the
generator uses, cs^2 = (dP/drho)_esp + (P/rho^2)(dP/desp)_rho, here by central differences in
(rho, T) of the exact Saha EOS instead of the table. Compare with host/worked_examples.cpp (table).

  python3 scripts/worked_saha.py > results/tables/W_worked_saha.txt
"""
import importlib.util
import os

import numpy as np

_REPO = os.environ.get("REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
spec = importlib.util.spec_from_file_location("gen_eos_table", _REPO + "/src/eos/gen_eos_table.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

RHO_U, V_U, MU_TH = 5.467e-19, 1.9e4, 2.29          # src/units/physical_units.hpp defaults
T_UNIT = MU_TH * G.m_H * V_U**2 / G.k_B


def energy_terms(rho, T):
    s = G.solve_saha(rho, T)
    n_tot = sum(s[k] for k in ("H2", "HI", "HII", "e", "HeI", "HeII", "HeIII"))
    terms = {
        "translational 3/2 n_tot k T": 1.5 * G.k_B * T * n_tot,
        "H2 rotation n_H2 e_rot(T)": s["H2"] * G.erot_per_H2(T),
        "H2 vibration n_H2 e_vib(T)": s["H2"] * G.evib_per_H2(T),
        "dissociation chi_d (n_HI + n_HII)/2": G.chi_d * (s["HI"] + s["HII"]) / 2.0,
        "H ionization chi_H n_HII": G.chi_H * s["HII"],
        "He ionization": G.chi_He1 * (s["HeII"] + s["HeIII"]) + G.chi_He2 * s["HeIII"],
    }
    return s, n_tot, terms


def cs2_exact(rho, T, h=1e-4):
    """cs^2 = P_rho|e + (P/rho^2) P_e|rho from central differences in (rho, T); e SPECIFIC."""
    P0, E0 = G.P_e_of_rho_T(rho, T)
    Pr1, Er1 = G.P_e_of_rho_T(rho * (1 + h), T)
    Pr0, Er0 = G.P_e_of_rho_T(rho * (1 - h), T)
    PT1, ET1 = G.P_e_of_rho_T(rho, T * (1 + h))
    PT0, ET0 = G.P_e_of_rho_T(rho, T * (1 - h))
    u = lambda E, r: E / r
    P_r_T = (Pr1 - Pr0) / (2 * h * rho)
    u_r_T = (u(Er1, rho * (1 + h)) - u(Er0, rho * (1 - h))) / (2 * h * rho)
    P_T_r = (PT1 - PT0) / (2 * h * T)
    u_T_r = (u(ET1, rho) - u(ET0, rho)) / (2 * h * T)
    P_e_r = P_T_r / u_T_r
    P_r_e = P_r_T - P_T_r * u_r_T / u_T_r
    return P_r_e + P0 / rho**2 * P_e_r, u_T_r


def report(title, rho, T):
    print(f"\n== {title}: rho = {rho:.6e} g/cm^3 (rho_code {rho / RHO_U:.6e}), T = {T:.6g} K")
    s, n_tot, terms = energy_terms(rho, T)
    nH = rho * G.X / G.m_H
    nHe = rho * G.Y / (4 * G.m_H)
    print(f"  nuclei: n_H = {nH:.6e}, n_He = {nHe:.6e} cm^-3")
    print(f"  species [cm^-3]: " + ", ".join(f"{k} {v:.4e}" for k, v in s.items()))
    print(f"  fractions: 2 n_H2/n_H = {2 * s['H2'] / nH:.6f}, n_HI/n_H = {s['HI'] / nH:.6e}, "
          f"n_HII/n_H = {s['HII'] / nH:.6e}, n_e/n_H = {s['e'] / nH:.6e}")
    # |n_e - (n_HII + n_HeII + 2 n_HeIII)| / n_H: the n_e bisection brackets log10 n_e from -40, so a fully
    # neutral state returns n_e = 1e-40 cm^-3 with zero ions (an absolute residual of 1e-40, not a relative one).
    chk_e = abs(s["e"] - (s["HII"] + s["HeII"] + 2 * s["HeIII"])) / nH
    chk_H = (2 * s["H2"] + s["HI"] + s["HII"]) / nH - 1
    print(f"  checks: |charge residual|/n_H {chk_e:.2e}, H-nucleus conservation {chk_H:.2e}")
    P, E = G.P_e_of_rho_T(rho, T)
    mu = rho / (G.m_H * n_tot)
    print(f"  n_tot = {n_tot:.6e} cm^-3, mu = rho/(m_H n_tot) = {mu:.6f}")
    print(f"  P = n_tot k T = {P:.6e} erg/cm^3 = {P / (RHO_U * V_U**2):.6e} code")
    for k, v in terms.items():
        print(f"    e term {k:38s} {v:.6e} erg/cm^3 ({v / E * 100:7.3f} %)")
    print(f"  e_int = {E:.6e} erg/cm^3; esp = e_int/rho = {E / rho:.6e} erg/g = {E / rho / V_U**2:.6e} code")
    cs2, cv = cs2_exact(rho, T)
    print(f"  e_int/P = {E / P:.6f}; 1 + P/e_int = {1 + P / E:.6f}; c_v = (du/dT)_rho = {cv:.6e} erg/g/K")
    print(f"  cs^2 (exact Saha, finite diff.) = {cs2:.6e} cm^2/s^2 = {cs2 / V_U**2:.6e} code; "
          f"Gamma1 = rho cs^2/P = {rho * cs2 / P:.6f}")


if __name__ == "__main__":
    print(f"# generator {_REPO}/src/eos/gen_eos_table.py; X = {G.X}, Y = {G.Y:.3f}; "
          f"temperature_unit = {T_UNIT:.6f} K")
    report("W1 reference state (rho_code = 1, T = temperature_unit)", RHO_U, T_UNIT)
    report("W2 H2 dissociation", 1e-9, 2000.0)
    report("W2b deeper in dissociation", 1e-8, 3000.0)
