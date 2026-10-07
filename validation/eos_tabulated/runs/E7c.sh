#!/bin/bash
#SBATCH --job-name=eos_E7c
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E7c: E7e problems with hydro/eos_rho_interp = cubic (C1 P along log rho), N 100-12800.
source $REPO/validation/self_gravity/scripts/suite_common.sh   # incl. btl_sm_fbox_max=0
# Many independent 1-rank runs: athenapk_env.sh sets OMP_PROC_BIND=spread + OMP_PLACES=threads, which pins EVERY
# process's single OpenMP thread to the first core of the job cpuset (64 runs on one core: 0.6 s/cycle vs 0.01 s alone).
export OMP_PROC_BIND=false; unset OMP_PLACES
BIN=$REPO/validation/_build/bin/athenaPK_cpu_8a49161e
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs
echo "job $SLURM_JOB_ID $(date) $(hostname)"; md5sum $BIN
one() { n=$(basename $1 .in); g=${n%%_*}; d=$W/$g/$n; mkdir -p $d; cd $d
        if ls *.out0.final.phdf >/dev/null 2>&1; then echo "$n skip"; return; fi
        { echo "date $(date -Is) node $(hostname) job $SLURM_JOB_ID"; echo "bin $(md5sum $BIN)"; echo "deck $1 $(md5sum < $1 | cut -c1-32)";
          echo "cmd mpirun -n 1 --bind-to none $BIN -i $1"; } > cmd.txt
        mpirun -n 1 --bind-to none $BIN -i $1 > run.log 2>&1; echo "rc $?" >> cmd.txt; echo "$n $(tail -1 cmd.txt)"; }
export -f one; export BIN W
{ ls $I/E7c_*_n12800.in $I/E7c_*_n6400.in $I/E7c_*_n3200.in $I/E7c_*_n1600.in; ls $I/E7c_*_n800.in $I/E7c_*_n400.in $I/E7c_*_n200.in $I/E7c_*_n100.in; } \
  | xargs -P 32 -I{} bash -c 'one {}'
echo E7c-DONE $(date)
