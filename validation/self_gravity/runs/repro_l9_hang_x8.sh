#!/bin/bash
#SBATCH --job-name=sg_L9hangx8
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
# Hang frequency: 8 repeats of the critical window c1400 -> c1446 (remesh at the end of c1439; the original
# job 2956209 hung in the first GMG solve after it; the bit-identical repro 2956493 did not).
source $REPO/validation/self_gravity/scripts/suite_common.sh
BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
D=$REPO/validation/self_gravity/runs/L9_boss_bodenheimer/_repro_hang_nj4
echo "job $SLURM_JOB_ID $(hostname) $(date)"; env | grep OMPI_MCA
for k in 1 2 3 4 5 6 7 8; do
  mkdir -p $D/rep$k; cd $D/rep$k
  timeout 1200 mpirun -n 64 --bind-to core $BIN -r ../parthenon.out2.00006.rhdf parthenon/time/nlim=1446 \
     parthenon/output2/dn=100000 parthenon/output0/dt=1e30 > run.log 2>&1
  rc=$?
  echo "rep $k rc $rc last: $(grep -a '^cycle=' run.log | tail -1 | cut -c1-30) hang: $(grep -a -c 'probably hanging' run.log) $(date +%T)"
done
