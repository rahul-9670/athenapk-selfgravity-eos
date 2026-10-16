#!/bin/bash
# Rebuild eos_table_hires_v6_rho26.bin (see TABLE_PROVENANCE.md): the v5_cold_rho26 grid (400 + 120 rows down to
# 1e-26 g/cm^3, esp / T axes padded by 399 / 450 nodes) with the CONSISTENT energy model (EOS_VALIDATION.md A3.5, T-4):
# a dissociated H2 molecule is charged chi_d - (3/4) 2 k theta_rot. The esp axis is pinned to the legacy energies, so
# every node below ~1000 K is the same as in eos_table_hires_v5_cold_rho26.bin.
set -o pipefail
PY=${PY:-python3}
cd "$(dirname "$0")"
OUT=$PWD/eos_table_hires_v6_rho26.h5
echo "generator md5: $(md5sum gen_eos_table.py | cut -c1-32)"
time $PY gen_eos_table.py build 400 1000 920 "$OUT" 399 450 3.0e5 0 0 120 consistent
md5sum eos_table_hires_v6_rho26.bin
