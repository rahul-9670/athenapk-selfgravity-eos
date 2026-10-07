#!/bin/bash
# run_one.sh <run_dir> <np> <deck> [overrides...]  -- one AthenaPK run of the validation binary.
# Writes run.log, cmd.txt (exact command + binary md5) in <run_dir>.
set -u
BIN=${BIN:-$REPO/build_cpu/bin/athenaPK}
D=$1; NP=$2; DECK=$3; shift 3
mkdir -p "$D"; cd "$D" || exit 1
rm -f *.phdf *.rhdf *.xdmf *.hst run.log
{ echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"; echo "deck $DECK $(md5sum < $DECK | cut -c1-32)"; echo "cmd  mpirun -n $NP $BIN -i $DECK $*"; } > cmd.txt
T0=$(date +%s.%N)
mpirun -n $NP --bind-to core "$BIN" -i "$DECK" "$@" > run.log 2>&1
RC=$?
echo "rc $RC wall $(echo "$(date +%s.%N) - $T0" | bc) s" >> cmd.txt
echo "[$RC] $(basename $D) $(grep -aE '^cycle=' run.log | tail -1 | cut -c1-100)"
