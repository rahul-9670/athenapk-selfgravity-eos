#!/bin/bash
# Rebuild eos_table_hires_v5_cold.bin (see TABLE_PROVENANCE.md). Expected md5: f712d6a806cb82889214fec95a6c2681.
# Base resolution 400 x 1000 x 920 in (rho, esp, T) with the esp / T axes padded downward by 399 / 450 nodes at the
# same spacing (cold-end extension to T ~ 0.06 K). Takes ~7 min on one core.
set -o pipefail
PY=${PY:-python3}
cd "$(dirname "$0")"
OUT=$PWD/eos_table_hires_v5_cold.h5
echo "generator md5: $(md5sum gen_eos_table.py | cut -c1-32)"
time $PY gen_eos_table.py build 400 1000 920 "$OUT" 399 450
md5sum eos_table_hires_v5_cold.bin
