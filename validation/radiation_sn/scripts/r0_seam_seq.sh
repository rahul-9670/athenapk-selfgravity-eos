#!/bin/bash
#SBATCH --job-name=rad_seam
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=03:30:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R0/seamseq_%j.out
# R0 SEAM hang, isolation round 2 (round 1, job 2987510: 3 concurrent 8-rank mpirun of the AMR arm all hang before the
# first solve; each alone completes). Concurrent variants, nlim = 3, 300 s cap:
#   C1  srun --mpi=pmix x3 (no mpirun/prterun)          C2  mpirun x3 with OMPI_MCA_btl=tcp,self (no shared memory)
# then the three SEAM AMR arms ONE AT A TIME, full length (the R0 result).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false
ulimit -c 0
M=$REPO/validation/radiation_sn
export BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_4135083d}
echo "BIN $(md5sum $BIN)"
conc () { tag=$1; for f in step sou prod; do
  d=$M/runs/R0/HANG/${tag}_$f; mkdir -p $d; cd $d; rm -f *.phdf *.xdmf *.hst run.log
  ( T0=$(date +%s)
    if [ $tag = C1 ]; then timeout 300 srun --mpi=pmix -n 8 --exact $BIN -i $M/inputs/R0/SEAM_$f.in parthenon/time/nlim=3 > run.log 2>&1
    else OMPI_MCA_btl=tcp,self timeout 300 mpirun -n 8 --bind-to none $BIN -i $M/inputs/R0/SEAM_$f.in parthenon/time/nlim=3 > run.log 2>&1; fi
    echo "${tag}_$f rc=$? wall=$(( $(date +%s)-T0 ))s solves=$(grep -ac 'TRUE rel' run.log)" ) &
done; wait; }
conc C1
conc C2
for f in step sou prod; do
  $REPO/validation/eos_tabulated/scripts/run_one_nobind.sh $M/runs/R0/SEAM/$f/arm_amr 8 $M/inputs/R0/SEAM_$f.in
done
echo SEAMSEQ-DONE
