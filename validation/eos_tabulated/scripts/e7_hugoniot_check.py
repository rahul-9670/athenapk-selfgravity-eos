#!/usr/bin/env python
"""E7 support: is the shock branch of each problem single-valued and admissible on the table interpolant?
Hugoniot parametrised by P: at each P all roots rho of g(rho) = P(rho, e_H(rho;P)) - P on (rho0, 60 rho0) are found
on a fine grid; Liu's condition (chord between the states must not cross the Hugoniot) is checked in the (v, P) plane."""
import sys, numpy as np
import eostab as E, riemann_exact as RX
t = E.load()


def hugoniot(eos, rho0, e0, Pmax, nP=400, nr=6000):
    P0 = float(eos.P(rho0, e0))
    Ps = P0 * np.logspace(1e-4, np.log10(Pmax / P0), nP)
    rr = rho0 * np.logspace(1e-6, np.log10(60.0), nr)
    out = []
    for P in Ps:
        e = e0 + 0.5 * (P + P0) * (1 / rho0 - 1 / rr)
        g = eos.P(rr, e) - P
        idx = np.nonzero(np.sign(g[1:]) != np.sign(g[:-1]))[0]
        out.append((P, rr[idx] / rho0))
    return out


if __name__ != "__main__":
    raise SystemExit
for mode in ("p_over_rho", "p"):
    eos = RX.TableEOS(t, mode)
    for name, rc, T, u in (("R3 +-16 km/s", 1e-10, 100.0, 16.0), ("R1 right", 1e-15, 30.0, None), ("R2 right", 1e-10, 1000.0, None),
                           ("R4 right", 1e-9, 6000.0, None)):
        rho0 = rc / E.RHO_U; e0 = float(eos.esp_of_T(rho0, T))
        if u is not None:
            so = RX.solve(eos, (rho0, u / 0.19, e0), (rho0, -u / 0.19, e0))
        else:
            import json
            ex = json.load(open("../results/tables/E7_exact.json"))
            key = {"R1 right": "R1_molecular", "R2 right": "R2_dissociation", "R4 right": "R4_ionization"}[name] + "|" + mode
            so = dict(P=ex[key]["P"], L=None)
        Pst = so["P"]
        H = hugoniot(eos, rho0, e0, 3 * Pst)
        multi = [(P, r) for P, r in H if len(r) != 1]
        Pc = np.array([P for P, r in H if len(r) == 1]); rc_ = np.array([r[0] for P, r in H if len(r) == 1])
        mono = np.all(np.diff(rc_) > 0)
        # Liu's chord condition with EXACT shock states (the grid roots above are only good to ~7e-4 in rho,
        # which flips the sign where the chord and the Hugoniot are within ~1e-3 P* of each other)
        W = RX.WaveCurve(eos, rho0, 0.0, e0, +1); P0 = float(eos.P(rho0, e0))
        rs = W.shock_state(Pst)[0]; v0, vs = 1 / rho0, 1 / rs
        PP = P0 * np.logspace(0.002, np.log10(Pst / P0) - 0.002, 150)
        d = np.array([P - (P0 + (Pst - P0) * (v0 - 1 / W.shock_state(P)[0]) / (v0 - vs)) for P in PP])
        s = np.sign(d)
        dec = np.nonzero(np.diff(rc_) <= 0)[0]
        where = "" if mono else f" (rho_H decreases for P/P0 in [{Pc[dec[0]] / P0:.3g}, {Pc[dec[-1] + 1] / P0:.3g}], i.e. {'BELOW' if Pc[dec[0]] < Pst else 'above'} P*)"
        print(f"{mode:10s} {name:13s} P*/P0 {Pst / P0:9.3g}  rho*/rho0 {rs / rho0:7.3f}  P values with !=1 root: {len(multi)} of {len(H)}"
              f"{'' if not multi else ' e.g. P/P0 %.3g roots %s' % (multi[0][0] / P0, np.round(multi[0][1], 3))}"
              f" | rho_H(P) increasing: {mono}{where} | max|P_H-chord|/P* {np.abs(d).max() / Pst:.2e} | Liu sign(P_H - chord) in [{int(s.min())},{int(s.max())}]", flush=True)
