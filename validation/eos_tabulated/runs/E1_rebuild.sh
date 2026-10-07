#!/bin/bash
#SBATCH --job-name=eos_E1
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/E1_rebuild/%x_%j.out
# E1: rebuild the production EOS table from the production generator with the documented command
# (src/eos/build_hires_v5_cold.sh) and compare md5 with the shipped table f712d6a8.
source ~/athenapk_env.sh
G=$REPO/src/eos/gen_eos_table.py
O=$REPO/validation/eos_tabulated/runs/E1_rebuild/eos_table_hires_v5_cold.h5   # generator writes .h5 and the .bin next to it
md5sum $G
time python3 $G build 400 1000 920 $O 399 450
md5sum ${O%.h5}.bin $REPO/src/eos/eos_table_hires_v5_cold.bin
cmp ${O%.h5}.bin $REPO/src/eos/eos_table_hires_v5_cold.bin && echo BYTE-IDENTICAL
