#!/bin/bash
# Opacity table v12 (R-F11, validation/radiation_sn 2026-10-09): Semenov et al. 2003 dust (monochromatic nrm_h_s,
# opacity_semenov.py) + Helling 2001 gas, format v4 (rho-dependent group multipliers, dust weights for
# dust/opacity_evap = table), production 3-group edges. Expected md5 413baf4f99a3a8d07ef614e0861e0f09 (18 s).
# Semenov's download (opacity.zip + nrm.zip) must be at --semenov_dir (downloaded from the authors' web page; see validation/radiation_sn/README.md).
set -eu
P=${PY:-python3}
D=$(cd "$(dirname "$0")" && pwd)
$P -I $D/gen_opacity_table.py --model semenov --out ${1:-$D/opacity_table_v12_semenov_hs_fmt4.bin} --nr 201 --nT 1859 \
  --rho_min 1e-20 --rho_max 1 --T_min 0.47406553651197825 --T_max 300860.0467910191 --edges 0,1e12,1e15,1e30
md5sum ${1:-$D/opacity_table_v12_semenov_hs_fmt4.bin}
