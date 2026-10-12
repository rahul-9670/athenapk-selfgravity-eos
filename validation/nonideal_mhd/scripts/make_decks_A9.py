#!/usr/bin/env python
"""A9 (PLAN.md): A1's Ohmic Gaussian (iprob 40, eta 0.25, periodic [-4,4]) with a STATIC refinement level on
0 < x < 4 (blocks of width 1; coarse-fine faces at x = 0, the Gaussian's peak, and x = +-4). Production keys as A1.
Arms: unsplit / rkl2 cap 1000. N = 64..512 = base resolution (fine side has 2N-equivalent). usage: make_decks_A9.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location('m', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_decks_A1A3.py'))
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'inputs', 'A9')
src = open(os.path.join(H, 'make_decks_A1A3.py')).read().split('runs = []')[0].replace("'inputs', 'A1A3'", "'inputs', 'A9'")
exec(src)
runs = []
for integ in ('unsplit', 'rkl2'):
    for N in (64, 128, 256, 512):
        tag = deck('diffusion_ohmic_glm.in', f'A9_{integ}_N{N}', N, integ)
        p = os.path.join(D, tag + '.in'); s = open(p).read()
        s = s.replace('refinement = none', 'refinement = static\nnumlevel = 2')
        import re
        s = re.sub(r'(<parthenon/meshblock>\n)nx1 = \d+', lambda m: m.group(1) + f'nx1 = {N // 8}', s)
        s += "\n<parthenon/static_refinement0>\nx1min = 0.1\nx1max = 3.9\nx2min = -0.25\nx2max = 0.25\nlevel = 1\n"
        open(p, 'w').write(s); runs.append(tag)
open(os.path.join(D, 'runs.txt'), 'w').write('\n'.join(runs) + '\n'); print(len(runs), 'decks')
