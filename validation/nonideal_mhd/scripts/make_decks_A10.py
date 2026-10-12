#!/usr/bin/env python
"""A10 (PLAN.md): ambipolar C-shock, iprob 45. Uniform rho0 = 1, u = 2, P0 = 0.1, By = B0 = 2 (c_s 0.41, v_A 2, gamma 5/3)
hitting a reflecting wall at x = 1 (outflow at x = 0); AD fixed Q = 0.01 (eta_A = Q B^2). cshock_ref.py: shock speed
V 1.911, fast Mach 1.92, compression 2.05, C-type, 10-90 width 0.117. t = 0.3 (shock at x ~ 0.43). Production keys of
the module; unsplit vs rkl2 cap 1000; N = 128..1024. usage: make_decks_A10.py"""
import os, re
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'inputs', 'A10')
T = open(os.path.join(H, 'A7_template.in')).read()
runs = []
for integ in ('unsplit', 'rkl2'):
    for N in (128, 256, 512, 1024):
        tag = f'A10_{integ}_N{N}'
        diff = "ambipolar = ambipolar\nambipolar_coeff = fixed\nambipolar_coeff_code = 0.01"
        s = T.format(tag=tag, N=N, amp=0.0, m=1, r=1000, diff=diff)
        s = s.replace('iprob = 41\nBx = 0.0\namp = 0.0\nnwave = 1', 'iprob = 45\ncs_rho0 = 1.0\ncs_u = 2.0\ncs_P0 = 0.1\ncs_B0 = 2.0')
        s = s.replace('ix1_bc = periodic\nox1_bc = periodic', 'ix1_bc = outflow\nox1_bc = reflecting_val')
        s = s.replace('tlim = 1.0', 'tlim = 0.3').replace('nlim = 1\n', 'nlim = -1\n').replace('dt = 10.0', 'dt = 1.0e9')
        s = s.replace('problem = ' + tag, 'problem = ' + tag + ' ambipolar C-shock (nonideal_mhd A10)')
        if integ == 'unsplit':
            s = s.replace('integrator = rkl2\n', 'integrator = unsplit\n').replace('rkl2_freeze_eta = true\n', '')
            s = s.replace('dual_energy_heat_exact = true', 'dual_energy_heat_exact = false').replace('dual_energy_heat_rkl2 = true', 'dual_energy_heat_rkl2 = false')
        assert 'iprob = 45' in s and 'reflecting' in s
        open(os.path.join(D, tag + '.in'), 'w').write(s); runs.append(tag)
open(os.path.join(D, 'runs.txt'), 'w').write('\n'.join(runs) + '\n'); print(len(runs), 'decks')
