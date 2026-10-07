#!/bin/bash
#SBATCH --job-name=sg_ref1d
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# 1D Lagrangian references, each case at N and 2N (convergence of the reference itself).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P=python3; cd $REPO/validation/self_gravity/scripts
for N in 1000 2000; do
  $P ref1d.py L7 $N > ../results/ref1d_L7_$N.log 2>&1 &
  $P ref1d.py L3b $N > ../results/ref1d_L3b_$N.log 2>&1 &
  $P ref1d.py L5 $N > ../results/ref1d_L5_$N.log 2>&1 &
done
for c in "3 1.1" "6.45 1" "9 1" "9 1.1"; do $P ref1d.py L6 $c 1000 > "../results/ref1d_L6_${c// /_}.log" 2>&1 & done
wait; echo REF1D-DONE
