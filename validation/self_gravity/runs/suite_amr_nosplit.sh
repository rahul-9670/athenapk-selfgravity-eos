#!/bin/bash
#SBATCH --job-name=sg_amrnosplit
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
echo "job $SLURM_JOB_ID $(date)"; md5sum $REPO/build_cpu/bin/athenaPK
# ---- SG-S1 periodic sinusoid (Tomida & Stone 2023 Sec 4.1; Athena++ poisson iprob 1): G = 1
NOSPLIT="self_gravity/mp_shell_split=false self_gravity/mp_uniform_split=false"
for n in 32 64 128; do
  RS $W/S4_binary/root${n}_nosplit 8 $I/binary_amr.in $(mesh $n $n $n 16) $NOSPLIT
  RS $W/S3_sphere_amr/root${n}_nosplit 8 $I/sphere_amr.in $(mesh $n $n $n 16) $NOSPLIT
done
echo SUITE-AMRNOSPLIT-DONE
