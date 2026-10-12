#!/usr/bin/env python
"""A6 (PLAN.md): nonlinear ambipolar spreading, Barenblatt-Pattle m = 3 (iprob 43). Q = 1e6, B0 = 1e-3 (eta_A peak 1,
beta ~ 2e6), xf0 = 0.1 at t0 = xf0^2/(4 Q B0^2) = 2.5e-3; run 0.2 = 80 t0 (front 0.1 -> 0.3). Arms: unsplit (N <= 512),
rkl2 ratio 100 / 1000 with freeze_eta (production), ratio 1000 refilling eta every stage, ratio 1000 + harmonic faces.
usage: make_decks_A6.py   (writes inputs/A6/*.in + runs.txt)"""
import os
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'inputs', 'A6')
TPL = open(os.path.join(H, 'A6_template.in')).read()
arms = [('unsplit', 1000, 'true', False), ('rkl2', 100, 'true', False), ('rkl2', 1000, 'true', False),
        ('rkl2', 1000, 'false', False), ('rkl2', 1000, 'true', True)]
runs = []
for integ, r, freeze, harm in arms:
    for N in (128, 256, 512, 1024):
        if integ == 'unsplit' and N > 512: continue
        tag = f"A6_{integ}{'' if integ == 'unsplit' else r}{'' if freeze == 'true' else 'nofreeze'}{'H' if harm else ''}_N{N}"
        diff = "ambipolar = ambipolar\nambipolar_coeff = fixed\nambipolar_coeff_code = 1.0e6" + ("\neta_face_harmonic = true" if harm else "")
        txt = TPL.format(tag=tag, N=N, integ=integ, r=r, freeze=freeze, diff=diff)
        if integ == 'unsplit' or freeze == 'false':   # heat_exact needs rkl2 + freeze_eta
            txt = txt.replace('dual_energy_heat_exact = true', 'dual_energy_heat_exact = false')
        if integ == 'unsplit':
            txt = txt.replace('rkl2_freeze_eta = true\n', '').replace('dual_energy_heat_rkl2 = true', 'dual_energy_heat_rkl2 = false')
        open(os.path.join(D, tag + '.in'), 'w').write(txt); runs.append(tag)
open(os.path.join(D, 'runs.txt'), 'w').write('\n'.join(runs) + '\n'); print(len(runs), 'decks')
