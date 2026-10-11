#!/bin/bash
#SBATCH --job-name=r5tab
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R5/tables/gen_%j.out
# R5 (group-count / edge convergence): one v11-recipe opacity table (build_opacity_v11.sh arguments, --shape gasfrac)
# per group partition of make_decks_R5.py (inputs/R5/tables.txt: name | --edges). Group multipliers are rho-independent
# by format; the gray part is identical to v11 (checked by analyze_R5.py: gray kappa_P/kappa_R blocks == v11's).
set -u
P=python3
G=$REPO/src/radiation/gen_opacity_table.py
D=$REPO/validation/radiation_sn/runs/R5/tables
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
md5sum $G
while IFS='|' read -r name edges; do
  name=$(echo $name); edges=$(echo $edges)
  ( "$P" -I -W ignore $G --out $D/opac_$name.bin --nr 201 --nT 1859 --rho_min 1e-20 --rho_max 1 \
      --T_min 0.47406553651197825 --T_max 300860.0467910191 --evap pollack --quad accurate --format 3 --shape gasfrac \
      --edges "$edges" > $D/gen_$name.log 2>&1; echo "$name rc=$? $(md5sum < $D/opac_$name.bin | cut -c1-12)" ) &
done < $REPO/validation/radiation_sn/inputs/R5/tables.txt
wait
echo TABLES-DONE
