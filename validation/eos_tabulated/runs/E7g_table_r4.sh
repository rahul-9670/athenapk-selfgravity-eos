#!/bin/bash
#SBATCH --job-name=eos_tabr4
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7g H1': the production table with 4x the RHO nodes only (same rho range; e and T axes unchanged: esp_max comes
# from the lowest rho row, which is the same 1e-20 g/cc).
source ~/athenapk_env.sh
G=$REPO/src/eos/gen_eos_table.py
O=$REPO/validation/eos_tabulated/runs/E7g_tables/eos_table_v5cold_r4.h5
md5sum $G
time python3 $G build 1600 1000 920 $O 399 450
ls -la ${O%.h5}.bin; md5sum ${O%.h5}.bin
