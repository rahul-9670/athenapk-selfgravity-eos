#!/bin/bash
# Rebuild eos_table_hires_v5_cold_rho26.bin (see TABLE_PROVENANCE.md). Expected md5: cb68bd0f919a9a929af3601c5befb686.
# The v5-cold build plus rho_pad_pts = 120 rows prepended at the SAME dlog rho (0.05 dex), down to 1e-26 g/cm^3.
# The esp and T axes are taken from the ORIGINAL rows only, so every original node keeps its value;
# rho_pad_pts = 0 reproduces eos_table_hires_v5_cold.bin byte-identically.
set -o pipefail
PY=${PY:-python3}
cd "$(dirname "$0")"
OUT=$PWD/eos_table_hires_v5_cold_rho26.h5
echo "generator md5: $(md5sum gen_eos_table.py | cut -c1-32)"
time $PY gen_eos_table.py build 400 1000 920 "$OUT" 399 450 3.0e5 0 0 120 legacy
md5sum eos_table_hires_v5_cold_rho26.bin
