#!/bin/bash
# Run every line "rundir | deck | overrides | nranks" of a run list (inputs/<name>.txt) whose rundir matches FILTER
# (anchored regex, default all): runs/<rundir>/ gets run.log and the outputs; deck = inputs/<deck>. NPAR runs at a time,
# each `mpirun -n <nranks> --bind-to none` (set OMP_PROC_BIND=false; one OpenMP thread per rank).
# Usage (from anywhere):  BIN=/path/to/athenaPK LIST=I1_runs.txt [FILTER=I1/I1_1d] [NPAR=8] bash run_list.sh
# Decks name tables as $REPO/...: substitute first, e.g. sed -i "s#\$REPO#$REPO#g" inputs/*/*.in
set -u; ulimit -c 0
H=$(cd "$(dirname "$0")" && pwd); M=$(dirname "$H")
BIN=${BIN:?path to athenaPK}; LIST=${LIST:?run list in inputs/}; NPAR=${NPAR:-8}
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
one() { local d=$1 k=$2 n=$3; shift 3; mkdir -p "$M/runs/$d" && cd "$M/runs/$d" || return
  rm -f *.phdf *.rhdf *.xdmf *.hst run.log
  mpirun -n "$n" --bind-to none "$BIN" -i "$M/inputs/$k" "$@" > run.log 2>&1 < /dev/null; echo "[$?] $d"; }
export -f one; export BIN M
# one line per run, newline-delimited (xargs -L would join a line that ends in a blank, i.e. empty overrides)
grep -E "^${FILTER:-.}" "$M/inputs/$LIST" | while IFS='|' read -r d k o n; do
  n=$(echo $n); echo "$(echo $d) $(echo $k) ${n:-1} $(echo $o)"; done | xargs -d '\n' -P "$NPAR" -I{} bash -c 'one {}'
echo LIST-DONE
