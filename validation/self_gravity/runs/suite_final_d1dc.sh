#!/bin/bash
#SBATCH --job-name=sg_finaldc
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# Attribution of the D1 error change in the final regression: final binary + production keys, but the OLD
# donor-cell VL2 predictor. If errors return to round 1 (1.13e-8, 1.51e-9) the predictor is the cause; if they
# stay at the final values (2.0e-8, 4.4e-9) the solver-tolerance change is.
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_11a642a4
F=$W/FINAL_dc
J() { python3 -c "import math;n=$1;k=6*math.pi;w2=k*k*(1-n*n);g=n*n*k*k;print(f'{g:.17g}', f'{(2*math.pi/math.sqrt(w2)) if w2>0 else 2.0/math.sqrt(-w2):.17g}')"; }
read G T <<< "$(J 0.5)"
for n in 64 128; do mb=16
  $R $F/D12_jeans/nJ0.5_n$n $(np_for $n $((n/2)) $((n/2)) $mb) $I/jeans_wave.in $(mesh $n $((n/2)) $((n/2)) $mb) \
    self_gravity/four_pi_G=$G parthenon/time/tlim=$T parthenon/output0/dt=$(python3 -c "print($T/20)") hydro/vl2_predictor_recon=dc
done
