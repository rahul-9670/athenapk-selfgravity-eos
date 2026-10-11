#!/bin/bash
#SBATCH --job-name=rad_list
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/list_%j.out
# Run every line "rundir | deck | overrides [| nranks]" of LIST whose rundir matches FILTER (regex, default all) as a 1-rank
# run, NPAR at a time (run_one_nobind.sh: --bind-to none + OMP_PROC_BIND=false). Usage:
#   sbatch --export=ALL,TMPDIR=/tmp,LIST=<file>,FILTER=<re>,BIN=<binary> run_list.sh
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
export OMP_PROC_BIND=false
ulimit -c 0   # no core dumps in the run dirs (quota)
M=$REPO/validation/radiation_sn
export BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_c69ee834}
R1=$REPO/validation/eos_tabulated/scripts/run_one_nobind.sh
NPAR=${NPAR:-64}
echo "BIN $(md5sum $BIN)  LIST $LIST  FILTER ${FILTER:-.}"
grep -E "^${FILTER:-.}" "$LIST" | while IFS='|' read -r d k o n; do   # optional 4th field = MPI ranks (default 1)
  d=$(echo $d); k=$(echo $k); o=$(echo $o); n=$(echo $n); echo "$M/runs/$d ${n:-1} $M/inputs/$k $o" | sed "s/ *$//"
done | xargs -P $NPAR -L 1 bash -c '$0 "$@"' $R1
echo LIST-DONE
