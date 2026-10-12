#!/usr/bin/env python
"""A1 (Ohmic Gaussian, iprob 40) and A3 (AD damped Alfven eigenmode, iprob 50, aligned and 45 deg oblique via
Bguide_y = 1) re-run in the PRODUCTION key set of this module. Base decks = the 2026-09-14 audit decks
(inputs/base/diffusion_{ohmic,ad}_glm.in); added: dual energy block + heat closure, rkl2
freeze_eta + eta_cache + diffusion cfl 0.3. Arms: unsplit and rkl2 (ratio cap 1000). N = 64..512 (meshblock N/2).
usage: make_decks_A1A3.py"""
import os, re
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'inputs', 'A1A3')
SRC = os.path.join(H, '..', 'inputs', 'base')
DUAL = """eos_table_allow_no_radiation = true
dual_energy = true
dual_energy_eta1 = 5e-2
dual_energy_sync_step_end = true
dual_energy_unsigned_prim = true
dual_energy_exact_resync = true
dual_energy_eta2 = 0.1
dual_energy_kappa = 3.0
dual_energy_shock_comp = 0.02
dual_energy_shared_select = true
dual_comp_ghost = true
dual_energy_heat_rkl2 = {hr}
dual_energy_heat_exact = {he}
boundary_clamp_dual = true
boundary_flux_clamp = false
"""
def deck(base, tag, N, integ, extra_prob=''):
    s = open(f'{SRC}/{base}').read()
    s = re.sub(r'problem = .*', f'problem = {tag} (nonideal_mhd A1/A3, production keys)', s, count=1)
    s = re.sub(r'(<parthenon/mesh>[\s\S]*?)nx1 = \d+', lambda m: m.group(1) + f'nx1 = {N}', s, count=1)
    s = re.sub(r'(<parthenon/meshblock>[\s\S]*?)nx1 = \d+', lambda m: m.group(1) + f'nx1 = {N // 2}', s, count=1)
    rk = integ == 'rkl2'
    s = s.replace('divergence_control = glm', 'divergence_control = glm\n' + DUAL.format(hr='true' if rk else 'false', he='true' if rk else 'false'))
    s = re.sub(r'<diffusion>\s*\nintegrator = unsplit', '<diffusion>\nintegrator = ' + integ + ('\nrkl2_max_dt_ratio = 1000\nrkl2_freeze_eta = true' if rk else '') + '\neta_cache = true', s)
    if 'cfl = 0.3' not in s.split('<diffusion>')[1]: s = s.replace('<diffusion>\n', '<diffusion>\ncfl = 0.3\n', 1)
    if extra_prob: s = s.replace('<problem/diffusion>\n', '<problem/diffusion>\n' + extra_prob + '\n', 1)
    open(os.path.join(D, tag + '.in'), 'w').write(s); return tag
runs = []
for integ in ('unsplit', 'rkl2'):
    for N in (64, 128, 256, 512):
        runs.append(deck('diffusion_ohmic_glm.in', f'A1_{integ}_N{N}', N, integ))
        runs.append(deck('diffusion_ad_glm.in', f'A3_aligned_{integ}_N{N}', N, integ))
        runs.append(deck('diffusion_ad_glm.in', f'A3_oblique_{integ}_N{N}', N, integ, 'Bguide_y = 1.0'))
open(os.path.join(D, 'runs.txt'), 'w').write('\n'.join(runs) + '\n'); print(len(runs), 'decks')
