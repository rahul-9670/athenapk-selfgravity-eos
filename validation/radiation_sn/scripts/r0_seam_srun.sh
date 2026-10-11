#!/bin/bash
#SBATCH --job-name=rad_seams
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R0/seamsrun_%j.out
# R0 SEAM AMR arm (FAM env, default prod), srun --mpi=pmix --cpu-bind=cores, btl tcp,self (R-F7).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
ulimit -c 0
# R-F7 (2026-10-08, runs/R7F4): with the sm BTL this arm hangs in 13 of 14 properly bound runs (srun AND mpirun), with
# tcp in 0 of 8 -- so the launcher was never the variable; force tcp here (cannot go through sbatch --export: commas).
export OMPI_MCA_btl=tcp,self
M=$REPO/validation/radiation_sn; F=${FAM:-prod}
BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_4135083d}
d=$M/runs/R0/SEAM/$F/arm_amr; mkdir -p $d; cd $d; rm -f *.phdf *.xdmf *.hst run.log
{ echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"; echo "deck $M/inputs/R0/SEAM_$F.in $(md5sum < $M/inputs/R0/SEAM_$F.in | cut -c1-32)"; echo "cmd  srun --mpi=pmix --cpu-bind=cores -n 8 $BIN -i $M/inputs/R0/SEAM_$F.in"; } > cmd.txt
T0=$(date +%s); srun --mpi=pmix --cpu-bind=cores -n 8 $BIN -i $M/inputs/R0/SEAM_$F.in > run.log 2>&1; echo "rc $? wall $(( $(date +%s)-T0 )) s" >> cmd.txt
tail -1 cmd.txt; grep -aE '^cycle=' run.log | tail -1 | cut -c1-80
