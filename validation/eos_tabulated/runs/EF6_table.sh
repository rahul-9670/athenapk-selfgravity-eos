#!/bin/bash
#SBATCH --job-name=eos_EF6tab
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E-F6 closure: the production table extended DOWN in rho with real Saha physics (rho_pad_pts = 120 rows at the
# production dlog rho = 6.0 dex -> 1e-26 g/cc), and the pad-0 rebuild that must stay byte-identical to f712d6a8.
export TMPDIR=/tmp
source ~/athenapk_env.sh >/dev/null 2>&1
G=$REPO/src/eos/gen_eos_table.py
O=$REPO/validation/eos_tabulated/runs/EF6_table
md5sum $G
PY=python3
( time $PY $G build 400 1000 920 $O/pad0.h5 399 450 3.0e5 0 0 0 ) > $O/pad0.log 2>&1 &
( time $PY $G build 400 1000 920 $O/v5cold_rho26.h5 399 450 3.0e5 0 0 120 ) > $O/rho26.log 2>&1 &
wait
md5sum $O/pad0.bin $O/v5cold_rho26.bin $REPO/src/eos/eos_table_hires_v5_cold.bin
cmp $O/pad0.bin $REPO/src/eos/eos_table_hires_v5_cold.bin && echo PAD0-BYTE-IDENTICAL
