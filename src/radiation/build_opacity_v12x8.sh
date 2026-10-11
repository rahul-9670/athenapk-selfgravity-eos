#!/bin/bash
# R-F13 (2026-10-09): table v12 recipe on the X8 group edges {0, 1.5, 3, 5, 8, 12, 18, 30 THz, 1e30} (8 groups;
# validation/radiation_sn R13: envelope heat 0.964/0.986 of the 34-group reference at 150/300 K vs 0.848/0.910 for
# the 3 production groups). Regenerates validation/radiation_sn/runs/R13/tables/opac_X8.bin bit-identically.
# Opacity table v12 (R-F11, validation/radiation_sn 2026-10-09): Semenov et al. 2003 dust (monochromatic nrm_h_s,
# opacity_semenov.py) + Helling 2001 gas, format v4 (rho-dependent group multipliers, dust weights for
# dust/opacity_evap = table), here on the X8 edges. Expected md5 609ef26365e730b6f5efcbf65745bea9 (25 s).
# Semenov's download (opacity.zip + nrm.zip) must be at --semenov_dir (downloaded from the authors' web page; see validation/radiation_sn/README.md).
set -eu
P=${PY:-python3}
D=$(cd "$(dirname "$0")" && pwd)
$P -I $D/gen_opacity_table.py --model semenov --out ${1:-$D/opacity_table_v12x8_semenov_hs_fmt4.bin} --nr 201 --nT 1859 \
  --rho_min 1e-20 --rho_max 1 --T_min 0.47406553651197825 --T_max 300860.0467910191 --edges 0.0,1.5e12,3e12,5e12,8e12,1.2e13,1.8e13,3e13,1e30
md5sum ${1:-$D/opacity_table_v12x8_semenov_hs_fmt4.bin}
