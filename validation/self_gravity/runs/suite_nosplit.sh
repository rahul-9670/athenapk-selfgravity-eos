#!/bin/bash
#SBATCH --job-name=sg_nosplit
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-S1 periodic sinusoid (Tomida & Stone 2023 Sec 4.1; Athena++ poisson iprob 1): G = 1
# Multipole BC formulation decomposition: shell split + uniform split OFF (= the 2026-07 WS-5a
# formulation, audit 2026-09-15 T1). Prediction: compact sources (S2, S3, S5, S6) converge with
# resolution once the splits are off; the production field S7 (mass at the wall) gets WORSE.
NOSPLIT="self_gravity/mp_shell_split=false self_gravity/mp_uniform_split=false"
for n in 32 64 128 256; do mb=$(( n<=64 ? 16 : 32 )); M="$(mesh $n $n $n $mb)"; NP=$(np_for $n $n $n $mb)
  RS $W/S2_gaussian/n${n}_mp_nosplit $NP $I/static_poisson.in $M $NOSPLIT problem/grav_analytic/profile=gaussian problem/grav_analytic/mass=1 problem/grav_analytic/sigma=1
  RS $W/S3_sphere/n${n}_mp_nosplit $NP $I/static_poisson.in $M $NOSPLIT
  RS $W/S5_maclaurin/n${n}_mp_nosplit $NP $I/static_poisson.in $M $NOSPLIT problem/grav_analytic/profile=ellipsoid problem/grav_analytic/a1=4 problem/grav_analytic/a2=4 problem/grav_analytic/a3=1.7435595774162693
  RS $W/S7_be_ic/n${n}_mp_nosplit $NP $I/be_ic.in $M $NOSPLIT
done
echo SUITE-NOSPLIT-DONE
