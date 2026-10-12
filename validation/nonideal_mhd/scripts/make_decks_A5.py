#!/usr/bin/env python
"""A5 (PLAN.md): Ohmic diffusion across an eta jump. iprob 42: rho_in on 0.25 < x < 0.75, 1 outside, uniform p0 = 1,
By = 1e-3 inside / 0 outside (beta ~ 2e6: hydro inert). diffusion/resistivity_coeff = density_power (validation-only),
eta = 100 rho^1 -> eta_out = 100, eta_in = 100 rho_in (contrast 1/rho_in). End time t = 2.5e-5 (sqrt(eta_out t) = 0.05,
fronts from the two interfaces never meet). Schemes: unsplit, rkl2 at ratio caps 100 and 1000. N = 64..1024.
usage: make_decks_A5.py   (writes inputs/A5/*.in + runs.txt)"""
import os
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'inputs', 'A5')
TPL = open(os.path.join(H, 'A5_template.in')).read()
TLIM, C = 2.5e-5, 100.0
runs = []
HARM = '--harmonic' in __import__('sys').argv   # falsification arm: diffusion/eta_face_harmonic = true
for rho_in in (0.1, 0.01):
    for integ, r in ((('unsplit', 1000), ('rkl2', 1000)) if HARM else (('unsplit', 1000), ('rkl2', 100), ('rkl2', 1000))):
        for N in (64, 128, 256, 512, 1024):
            tag = f"A5_c{int(round(1/rho_in))}_{integ}{'' if integ == 'unsplit' else r}{'H' if HARM else ''}_N{N}"
            diff = f"resistivity = ohmic\nresistivity_coeff = density_power\nohm_diff_coeff_code = {C!r}\nohm_rho_power = 1.0" + ("\neta_face_harmonic = true" if HARM else "")
            txt = TPL.format(tag=tag, N=N, rho_in=rho_in, tlim=TLIM, integ=integ, r=r, diff=diff)
            if integ == 'unsplit':   # rkl2-only keys are rejected / meaningless under unsplit
                txt = txt.replace('rkl2_freeze_eta = true\n', '').replace('dual_energy_heat_exact = true', 'dual_energy_heat_exact = false').replace('dual_energy_heat_rkl2 = true', 'dual_energy_heat_rkl2 = false')
            open(os.path.join(D, tag + '.in'), 'w').write(txt); runs.append(tag)
open(os.path.join(D, 'runs_H.txt' if HARM else 'runs.txt'), 'w').write('\n'.join(runs) + '\n'); print(len(runs), 'decks')
