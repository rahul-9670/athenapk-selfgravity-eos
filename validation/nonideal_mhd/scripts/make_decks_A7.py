#!/usr/bin/env python
"""A7 (PLAN.md): RKL2 amplification of single Fourier modes vs the exact stability polynomial, at the production
super-step ratios. iprob 41 with Bx = 0: B = amp (0, sin kx, cos kx), |B| uniform -> no Lorentz force, the hydro step
is inert and one cycle = two RKL2 halves (tau = dt/2 each) of pure diffusion of the mode. dt is set by the ratio cap
(diffusion/rkl2_max_dt_ratio = r, eta large so dt_hyp/dt_diff >> r). Terms: ohm (resistivity fixed eta) and ad
(ambipolar Q B^2 with B perp k, = Ohmic with eta_A = Q amp^2). Production keys of this module verbatim (rkl2,
freeze_eta, cfl 0.3, dual energy + heat closure). usage: make_decks_A7.py  (writes inputs/A7/*.in + runs.txt)"""
import os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inputs', 'A7')
N, ETA, AMP = 64, 100.0, 1.0
RATIOS = [1, 10, 100, 300, 500, 1000]
MODES = [1, 4, 8, 16, 24, 31]
TPL = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'A7_template.in')).read()
runs = []
for term in ('ohm', 'ad'):
    if term == 'ohm':
        diff = f"resistivity = ohmic\nresistivity_coeff = fixed\nohm_diff_coeff_code = {ETA!r}"
    else:
        diff = f"ambipolar = ambipolar\nambipolar_coeff = fixed\nambipolar_coeff_code = {ETA / AMP**2!r}"
    for r in RATIOS:
        for m in MODES:
            tag = f"A7_{term}_r{r}_m{m}"
            open(os.path.join(D, tag + '.in'), 'w').write(TPL.format(tag=tag, N=N, amp=AMP, m=m, r=r, diff=diff))
            runs.append(tag)
open(os.path.join(D, 'runs.txt'), 'w').write('\n'.join(runs) + '\n')
print(len(runs), 'decks')
