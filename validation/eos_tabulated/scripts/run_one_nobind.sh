#!/bin/bash
# As self_gravity/scripts/run_one.sh but for MANY CONCURRENT 1-rank runs in one job: --bind-to none (each independent
# mpirun would otherwise bind its rank to the job's first core) -- pair with OMP_PROC_BIND=false.
set -u
BIN=${BIN:?}
D=$1; NP=$2; DECK=$3; shift 3
mkdir -p "$D"; cd "$D" || exit 1
rm -f *.phdf *.rhdf *.xdmf *.hst run.log
{ echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"; echo "deck $DECK $(md5sum < $DECK | cut -c1-32)"; echo "cmd  mpirun -n $NP --bind-to none $BIN -i $DECK $*"; } > cmd.txt
T0=$(date +%s.%N)
mpirun -n $NP --bind-to none "$BIN" -i "$DECK" "$@" > run.log 2>&1
RC=$?
echo "rc $RC wall $(echo "$(date +%s.%N) - $T0" | bc) s" >> cmd.txt
echo "[$RC] $(basename $D) $(grep -aE '^cycle=' run.log | tail -1 | cut -c1-100)"
