#!/usr/bin/env python
"""Decks for E7 (Riemann problems), E8 (homologous compression legs) and E10 (floor-visibility probe).
Production EOS keys verbatim (same HYDRO block as make_decks_eos.py, which made the E6 decks and is left untouched
so their md5s stay valid). Binary default eos_interp = p_over_rho; `p` legs set it explicitly."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eostab as E
import riemann_exact as RX

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "..", "inputs")
TAB = os.path.join(HERE, "..", "results", "tables")
HYDRO = f"""<hydro>
fluid = glmmhd
eos = hydrogen
eos_table_file = {E.TABLE}
gamma = 1.4
riemann = hlld
reconstruction = plm
glmmhd_source = dedner_plain   # 1D decks: dedner_extended (production, 3D) segfaults in 1D (E-F3); B = 0 here
eos_floor_consistent = true
dfloor = 1.0e-12
pfloor = 1.0e-14
"""


def deck(name, comment, mode_block, nx, tlim, extra_hydro="", bc="outflow", dt_out=None, cfl=0.3, hst=False,
         x1=(0.0, 1.0), nlim=-1, hydro=HYDRO, nyz=1):
    hst_blk = "\n<parthenon/output1>\nfile_type = hst\ndt = -1\ndn = 1\n" if hst else ""
    txt = f"""<comment>
problem = {comment}

<job>
problem_id = eos_test

<problem/eos_test>
{mode_block}
<parthenon/mesh>
refinement = none
nx1 = {nx}
x1min = {float(x1[0])!r}
x1max = {float(x1[1])!r}
ix1_bc = {bc}
ox1_bc = {bc}
nx2 = {nyz}
x2min = -0.5
x2max = 0.5
ix2_bc = periodic
ox2_bc = periodic
nx3 = {nyz}
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic

<parthenon/meshblock>
nx1 = {nx}
nx2 = {nyz}
nx3 = {nyz}

<parthenon/time>
integrator = vl2
cfl = {float(cfl)!r}
tlim = {float(tlim)!r}
nlim = {nlim}

{hydro}{extra_hydro}
<physics>
self_gravity = false

<parthenon/output0>
file_type = hdf5
variables = prim
dt = {float(dt_out) if dt_out else float(tlim)!r}
{hst_blk}"""
    open(os.path.join(INP, name + ".in"), "w").write(txt)
    return name


def e7(prefix="E7e"):
    """prefix E7e (2026-10-05): the IC energies are set EXACTLY (espL_code/espR_code = the exact solver's states).
    The first set (prefix E7, runs/E7) passed T = 10^logT(rho, e) and the pgen re-derived e through the (rho, T)
    grid: the T -> e -> T round trip (E5) shifted P_L by -2.8e-4 (R2) and P_L/P_R by -2.1e-4/-4.6e-4 (R4), which
    was exactly the plateau error those legs stalled at."""
    ex = json.load(open(os.path.join(TAB, "E7_exact.json")))
    names = []
    for key, r in ex.items():
        prob, mode = key.split("|")
        (rL, uL, eL), (rR, uR, eR) = r["L"], r["R"]
        mb = (f"mode = riemann\nx0 = 0.5\nrhoL = {rL!r}\nTL = {r['L_T']!r}\nuL = {uL!r}\n"
              f"rhoR = {rR!r}\nTR = {r['R_T']!r}\nuR = {uR!r}\n")
        if prefix == "E7e":
            mb += f"espL_code = {float(eL)!r}\nespR_code = {float(eR)!r}\n"
        for nx in (100, 200, 400, 800, 1600, 3200):
            extra = "eos_interp = p\n" if mode == "p" else "eos_interp = p_over_rho\n"
            names.append(deck(f"{prefix}_{prob}_{mode}_n{nx}", f"{prefix} {prob} ({r['what']}), eos_interp = {mode}, N = {nx}",
                              mb, nx, r["tlim"], extra_hydro=extra))
    return names


def e7c():
    """E7d (2026-10-05; the first version E7c was rho-only and changed nothing): the E7e problems with hydro/eos_p_interp = cubic (C1 P along log rho AND log e), exact energies,
    against the exact solutions of the cubic interpolant (E7_exact.json keys '<prob>|p_over_rho_cubic')."""
    ex = json.load(open(os.path.join(TAB, "E7_exact.json")))
    names = []
    for key, r in ex.items():
        prob, mode = key.split("|")
        if mode != "p_over_rho_cubic":
            continue
        (rL, uL, eL), (rR, uR, eR) = r["L"], r["R"]
        mb = (f"mode = riemann\nx0 = 0.5\nrhoL = {rL!r}\nTL = {r['L_T']!r}\nuL = {uL!r}\n"
              f"rhoR = {rR!r}\nTR = {r['R_T']!r}\nuR = {uR!r}\n"
              f"espL_code = {float(eL)!r}\nespR_code = {float(eR)!r}\n")
        for nx in (100, 200, 400, 800, 1600, 3200, 6400, 12800):
            if prob.startswith("R3") and nx > 3200:
                continue
            names.append(deck(f"E7d_{prob}_{mode}_n{nx}", f"E7d {prob} ({r['what']}), eos_interp = p_over_rho, eos_p_interp = cubic (C1 in rho and e), N = {nx}",
                              mb, nx, r["tlim"], extra_hydro="eos_interp = p_over_rho\neos_p_interp = cubic\n"))
    return names


def e8():
    """Legs of 100x homologous compression along the table adiabat through (1e-13 g/cc, 10 K)."""
    t = E.load()
    eos = RX.TableEOS(t, "p_over_rho")
    # reference adiabat (exact for the interpolant) from (1e-13, 10 K): de/dlnrho = P/rho, RK4, h = 1e-4
    rho = 1e-13 / E.RHO_U
    e = float(eos.esp_of_T(rho, 10.0))
    starts = []
    lnr = np.log(rho); h = 1e-4
    targets = [np.log(1e-13 * 100 ** k / E.RHO_U) for k in range(6)]  # 1e-13 .. 1e-3 g/cc
    f = lambda x, y: float(eos.P(np.exp(x), y)) / np.exp(x)
    k = 0
    while k < len(targets):
        if lnr >= targets[k] - 1e-12:
            starts.append((np.exp(lnr), e)); k += 1
            continue
        hh = min(h, targets[k] - lnr)
        k1 = f(lnr, e); k2 = f(lnr + hh / 2, e + hh / 2 * k1); k3 = f(lnr + hh / 2, e + hh / 2 * k2); k4 = f(lnr + hh, e + hh * k3)
        e += hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4); lnr += hh
    meta, names = {}, []
    nx, L = 33, 1.0           # odd N: the centre cell sits at xc and stays at rest
    dx = L / nx; C = 100.0
    for k, (r0, e0) in enumerate(starts[:-1]):
        T0 = float(10 ** eos.logT(r0, e0))
        cs0 = float(np.sqrt(RX.c2(eos, r0, e0)))
        # cell-average KE mistaken for e_int ~ (alpha dx C)^2 / 24 (see README); keep it <= 1e-6 e at the end
        # (1e-8 cost ~3.6e5 cycles per leg; the _half_alpha leg measures this term: it scales as alpha^2)
        e_end_est = e0 * C ** (2.0 / 3.0)
        alpha = np.sqrt(24e-6 * e_end_est) / (dx * C)
        tlim = (1.0 - 1.0 / C) / alpha
        r0, e0, T0, alpha = float(r0), float(e0), float(T0), float(alpha)
        for tag, a, cfl, extra in (("", alpha, 0.3, ""), ("_cfl015", alpha, 0.15, ""), ("_half_alpha", alpha / 2, 0.3, ""),
                                   ("_bis60", alpha, 0.3, "eos_bisections = 60\n"), ("_p", alpha, 0.3, "eos_interp = p\n")):
            tl = float((1.0 - 1.0 / C) / a); a = float(a)
            mb = f"mode = compress\nrho0 = {r0!r}\nT0 = {T0!r}\nalpha = {a!r}\nxc = {L / 2!r}\n"
            nm = deck(f"E8_leg{k}{tag}", f"E8 homologous compression leg {k}: rho {r0 * E.RHO_U:.3g} -> {100 * r0 * E.RHO_U:.3g} g/cc"
                      f" from T {T0:.4g} K on the table adiabat; alpha {a:.4g}{tag}", mb, nx, tl, extra_hydro=extra,
                      bc="eos_homologous", dt_out=tl / 60.0, cfl=cfl)
            names.append(nm)
            meta[nm] = dict(rho0=r0, e0_adiabat=e0, T0=T0, alpha=a, tlim=tl, cs0=cs0, cfl=cfl, mode="p" if tag == "_p" else "p_over_rho")
    json.dump(meta, open(os.path.join(INP, "E8_legs.json"), "w"), indent=1)
    return names


def e10():
    """Floor-visibility probe (3 cycles) on a thin 3D box 32 x 4 x 4 (hydro/cons_diag segfaults in 1D: E-F4) whose right half carries a sub-table or negative
    specific internal energy; production floor keys (pfloor 1e-8, dfloor 1e-6), eos_floor_consistent on/off."""
    t = E.load()
    esp_min = 10 ** t["le0"]
    names = []
    hyd = HYDRO.replace("dfloor = 1.0e-12\npfloor = 1.0e-14\n", "dfloor = 1.0e-6\npfloor = 1.0e-8\ncons_diag = true   # production: floor counters in the hst\n")
    rho = 1e-17 / E.RHO_U
    for tag, espR in (("subedge", float(0.5 * esp_min)), ("negative", -1.0e-3), ("control", None)):
        for fc in ("true", "false"):
            mb = f"mode = riemann\nx0 = 0.5\nrhoL = {rho!r}\nTL = 10.0\nrhoR = {rho!r}\nTR = 10.0\n"
            if espR is not None:
                mb += f"espR_code = {espR!r}\n"
            h = hyd.replace("eos_floor_consistent = true", f"eos_floor_consistent = {fc}")
            names.append(deck(f"E10_probe_{tag}_fc{fc}", f"E10 floor probe: right half esp = {espR}, eos_floor_consistent = {fc}",
                              mb, 32, 1.0, dt_out=None, hst=True, nlim=3, hydro=h, nyz=4))
    return names


if __name__ == "__main__":
    which = sys.argv[1:] or ["e7", "e8", "e10"]
    for w in which:
        n = globals()[w]()
        print(w, len(n), "decks:", n[0], "...", n[-1])
