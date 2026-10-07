#!/bin/bash
#SBATCH --job-name=eos_E6
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/eos_tabulated/runs/%x_%j.out
# E6: acoustic waves on the tabulated EOS, 5 states x N = 16..256 (+ 60-bisection legs), 1 rank each.
source $REPO/validation/self_gravity/scripts/suite_common.sh   # incl. btl_sm_fbox_max=0
BIN=$REPO/validation/_build/bin/athenaPK_cpu_39c0d06a
I=$REPO/validation/eos_tabulated/inputs; W=$REPO/validation/eos_tabulated/runs/E6_wave
echo "job $SLURM_JOB_ID $(date)"; md5sum $BIN
one() { d=$W/$(basename $1 .in); mkdir -p $d; cd $d; { echo "bin $(md5sum $BIN)"; echo "deck $1 $(md5sum < $1 | cut -c1-32)"; } > cmd.txt
        mpirun -n 1 --bind-to none $BIN -i $1 > run.log 2>&1; echo "rc $?" >> cmd.txt; echo "$(basename $1 .in) rc $(tail -1 cmd.txt)"; }
export -f one; export BIN W
ls $I/E6_wave_*.in | xargs -P 32 -I{} bash -c 'one {}'
echo E6-DONE
