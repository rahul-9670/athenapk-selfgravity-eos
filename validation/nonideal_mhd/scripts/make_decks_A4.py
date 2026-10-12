#!/usr/bin/env python
"""A4 (PLAN.md): 3-D ABC (Beltrami) field, iprob 44, periodic unit cube, nwave 1 (k = 2 pi), amp 1e-3, rho = p = 1.
curl B = k B -> J x B = 0: AD (Q = 1e6, eta_A = Q B^2 ~ 0.3..2.4) must do nothing; Ohmic (eta = 1) decays B as
exp(-eta k^2 t). tlim 0.025 (Ohmic: eta k^2 t = 0.99). Arms unsplit / rkl2 cap 1000 (production keys of the module).
N = 16, 32, 64 (meshblock N/2). usage: make_decks_A4.py"""
import os, re
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'inputs', 'A4')
T = open(os.path.join(H, 'A7_template.in')).read()
runs = []
for term in ('ohm', 'ad'):
    for integ in ('unsplit', 'rkl2'):
        for N in (16, 32, 64):
            tag = f'A4_{term}_{integ}_N{N}'
            diff = ("resistivity = ohmic\nresistivity_coeff = fixed\nohm_diff_coeff_code = 1.0" if term == 'ohm'
                    else "ambipolar = ambipolar\nambipolar_coeff = fixed\nambipolar_coeff_code = 1.0e6")
            s = T.format(tag=tag, N=N, amp=1.0e-3, m=1, r=1000, diff=diff)
            s = s.replace('iprob = 41', 'iprob = 44').replace('Bx = 0.0\n', '').replace('tlim = 1.0', 'tlim = 0.025').replace('nlim = 1\n', 'nlim = -1\n')
            s = s.replace('problem = ' + tag, 'problem = ' + tag + ' 3-D ABC field (nonideal_mhd A4)')
            for ax in ('2', '3'):
                s = re.sub(rf'nx{ax} = 1\nx{ax}min = -0.5\nx{ax}max = 0.5', f'nx{ax} = {N}\nx{ax}min = 0.0\nx{ax}max = 1.0', s)
                s = re.sub(rf'(<parthenon/meshblock>[\s\S]*?)nx{ax} = 1', lambda m: m.group(1) + f'nx{ax} = {N // 2}', s)
            s = re.sub(r'(<parthenon/meshblock>\n)nx1 = \d+', lambda m: m.group(1) + f'nx1 = {N // 2}', s)
            s = s.replace('glmmhd_source = dedner_plain', 'glmmhd_source = dedner_extended').replace('dt = 10.0', 'dt = 1.0e9')
            if integ == 'unsplit':
                s = s.replace('integrator = rkl2\n', 'integrator = unsplit\n').replace('rkl2_freeze_eta = true\n', '')
                s = s.replace('dual_energy_heat_exact = true', 'dual_energy_heat_exact = false').replace('dual_energy_heat_rkl2 = true', 'dual_energy_heat_rkl2 = false')
            open(os.path.join(D, tag + '.in'), 'w').write(s); runs.append(tag)
open(os.path.join(D, 'runs.txt'), 'w').write('\n'.join(runs) + '\n'); print(len(runs), 'decks')
