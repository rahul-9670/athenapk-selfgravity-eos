#!/bin/bash
#SBATCH --job-name=eos_tab2
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7g (hypothesis H1: bilinear-interpolant kinks slow the recombination fans): the production table rebuilt with
# 2x the nodes on every axis (same generator, same ranges: node counts and pads scaled together).
source ~/athenapk_env.sh
G=$REPO/src/eos/gen_eos_table.py
O=$REPO/validation/eos_tabulated/runs/E7g_tables/eos_table_v5cold_x2.h5
md5sum $G
time python3 $G build 800 2000 1840 $O 798 900
ls -la ${O%.h5}.bin; md5sum ${O%.h5}.bin
