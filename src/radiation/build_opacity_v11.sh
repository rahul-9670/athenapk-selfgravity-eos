#!/bin/bash
# Opacity table v11 (R-F9, validation/radiation_sn, 2026-10-08): the v10b recipe (R-F4 accurate quadrature,
# R-F5 grid: T 8x finer than v9 with nodes exactly on 140 and 170 K, rho 2x finer; pollack evaporation) plus
# --shape gasfrac (the gray part of the spectral shape = the BL94 gas fraction of the opacity, no greying at the
# ice-sublimation front). Validated by validation/radiation_sn/scripts/r7_v11.py (results/tables/R7_v11_gasfrac.txt).
# usage: build_opacity_v11.sh [out.bin]     check: prints OK when the md5 matches the adopted table.
set -eu
P=${PY:-python3}
D=$(dirname "$(readlink -f "$0")")
OUT=${1:-$D/opacity_table_v11_pollack_acc_gasfrac_fmt3.bin}
"$P" -I -W ignore "$D/gen_opacity_table.py" --out "$OUT" --nr 201 --nT 1859 --rho_min 1e-20 --rho_max 1 \
  --T_min 0.47406553651197825 --T_max 300860.0467910191 --evap pollack --quad accurate --format 3 --shape gasfrac
M=$(md5sum < "$OUT" | cut -c1-32)
[ "$M" = 162bb0cf92370d44182eb9082f2e0454 ] && echo "OK $M" || { echo "MISMATCH $M (expected 162bb0cf92370d44182eb9082f2e0454)"; exit 1; }
