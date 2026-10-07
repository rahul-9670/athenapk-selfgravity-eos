#!/bin/bash
#SBATCH --job-name=sg_jeansD3
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-D1 stable (n_J = 0.5) and SG-D2 unstable (n_J = 1.5) Jeans; k = 6 pi (lambda = 1/3), cs^2 = 1
# Prints four_pi_G, tlim (1 period or 2 e-folds) and ABSTOL = 1e-6 rms(4 pi G drho) = 1e-6 four_pi_G amp/sqrt2:
# BiCGSTAB's relative_residual is relative to the WARM-START residual (bicgstab_solver.hpp:229), which
# under the swindle stalls on the round-off floor of the mean density (README, finding SG-F1).
J() { python3 -c "import math;n=$1;k=6*math.pi;w2=k*k*(1-n*n);g=n*n*k*k;print(f'{g:.17g}', f'{(2*math.pi/math.sqrt(w2)) if w2>0 else 2.0/math.sqrt(-w2):.17g}', f'{1e-6*g*1e-6/math.sqrt(2):.6e}')"; }
# ---- SG-D3 dispersion relation sweep at 64x32x32: stable runs 2 periods, unstable 3 e-folds; 40 dumps
for nj in 0.2 0.4 0.6 0.8 0.9 1.1 1.25 1.5 2.0 3.0; do read G T AT <<< "$(J $nj)"
  T2=$(python3 -c "print($T*(2 if $nj<1 else 1.5))")
  $R $W/D3_dispersion/nJ$nj 16 $I/jeans_wave.in $(mesh 64 32 32 16) self_gravity/four_pi_G=$G parthenon/time/tlim=$T2 parthenon/output0/dt=$(python3 -c "print($T2/40)") self_gravity/solver_params/absolute_residual_tolerance=$AT
done
echo SUITE-JEANS-DONE
