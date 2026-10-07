#!/bin/bash
#SBATCH --job-name=sg_final
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# FINAL-CONFIGURATION regression (2026-10-04): the gravity-hydro coupling tests re-run with the final
# validation binary = production defaults (VL2 with PLM predictor, NaN guard, SG-F1 round-off floor +
# safe fallback) and the PRODUCTION gravity keys ONLY -- no hand-set absolute_residual_tolerance, which
# round 1 needed for every periodic leg (SG-F1). MPI: suite_common.sh (sm fast boxes off, SG-F7).
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_11a642a4
F=$W/FINAL
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN; env | grep OMPI_MCA
J() { python3 -c "import math;n=$1;k=6*math.pi;w2=k*k*(1-n*n);g=n*n*k*k;print(f'{g:.17g}', f'{(2*math.pi/math.sqrt(w2)) if w2>0 else 2.0/math.sqrt(-w2):.17g}')"; }
# S1 periodic sinusoid (round 1: NaN on production keys at 16^3 and 64^3)
P="problem/grav_analytic/profile=sinusoid problem/grav_analytic/rho_bg=2 problem/grav_analytic/amp=1 self_gravity/four_pi_G=12.566370614359172"
for n in 16 32 64 128; do mb=$(( n<64 ? 8 : 16 ))
  $R $F/S1_sinusoid/n$n $(np_for $n $n $n $mb) $I/static_poisson.in $(mesh $n $n $n $mb) $cube01 $periodic $P
done
# D1 / D2 Jeans waves
for nj in 0.5 1.5; do read G T <<< "$(J $nj)"
  for n in 16 32 64 128; do mb=$(( n<=32 ? 8 : 16 ))
    $R $F/D12_jeans/nJ${nj}_n$n $(np_for $n $((n/2)) $((n/2)) $mb) $I/jeans_wave.in $(mesh $n $((n/2)) $((n/2)) $mb) \
      self_gravity/four_pi_G=$G parthenon/time/tlim=$T parthenon/output0/dt=$(python3 -c "print($T/20)")
  done; done
# D4 advected slab, production integrator
for n in 16 32 64 128; do mb=$(( n/4 )); [ $mb -gt 32 ] && mb=32
  $R $F/D4_slab/vl2_cfl0.3_n$n $(np_for $n $((n/2)) $((n/2)) $mb) $I/grav_slab.in $(mesh $n $((n/2)) $((n/2)) $mb) parthenon/time/integrator=vl2 parthenon/time/cfl=0.3
done
# D7 free fall, dt ladder at 64^3 (SG-F4: round-1 vl2 with the donor-cell predictor was first order in dt)
for dt in 0.002 0.001 0.0005 0.00025; do
  $R $F/D7_freefall/n64_dtceil$dt 64 $I/freefall.in $(mesh 64 64 64 16) parthenon/time/dt_ceil=$dt
done
echo SUITE-FINAL-DONE
