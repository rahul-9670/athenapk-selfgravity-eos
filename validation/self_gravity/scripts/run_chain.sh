#!/bin/bash
# run_chain.sh <run_dir> <np> <deck> [overrides...] -- restartable leg for jobs longer than the 12 h
# SLURM limit. Fresh start, or resume from the newest restart; Parthenon stops cleanly at -t $WALL and
# writes restarts every $RDN cycles. Writes <run_dir>/.done ONLY when the log shows time >= tlim.
# Exit 0 = done, 99 = needs another leg, other = failure.
set -u
BIN=${BIN:?pinned binary}; WALL=${WALL:-11:20:00}; RDN=${RDN:-500}
D=$1; NP=$2; DECK=$3; shift 3
mkdir -p "$D"; cd "$D" || exit 1
[ -f .done ] && { echo "[done] $(basename $D)"; exit 0; }
[ -f .stop ] && { echo "[stopped] $(basename $D) (.stop present; remove it and resubmit to continue)"; exit 0; }
RST=$(ls -t *.out2.*.rhdf 2>/dev/null | head -1)
LEG=$(ls run.log.* 2>/dev/null | wc -l)
if [ -n "$RST" ]; then ARGS="-r $RST -t $WALL $*"; else ARGS="-i $DECK -t $WALL $* parthenon/output2/file_type=rst parthenon/output2/dn=$RDN"; fi
{ echo "leg $LEG date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"; echo "deck $DECK $(md5sum < $DECK | cut -c1-32)"; echo "cmd  mpirun -n $NP $BIN $ARGS"; } >> cmd.txt
mpirun -n $NP --bind-to core "$BIN" $ARGS > run.log.$LEG 2>&1
RC=$?
cp run.log.$LEG run.log
T=$(grep -a "^time=" run.log | tail -1 | sed -E 's/time=([^ ]+).*/\1/'); TL=$(grep -a "^tlim=" run.log | tail -1 | sed -E 's/tlim=([^ ]+).*/\1/')
echo "rc $RC leg $LEG t=$T tlim=$TL" >> cmd.txt
if [ $RC -ne 0 ]; then echo "[$RC] $(basename $D) leg $LEG FAILED"; exit $RC; fi
if python3 -c "import sys; sys.exit(0 if float('$T') >= float('$TL')*(1-1e-9) else 1)" 2>/dev/null; then
  touch .done; echo "rc 0" >> cmd.txt; echo "[0] $(basename $D) done at t=$T"; exit 0
fi
echo "[99] $(basename $D) leg $LEG stopped at t=$T < tlim=$TL"; exit 99
