#!/bin/bash
# Run every deck of one test set: inputs/<SET>/<LIST> (default runs.txt) -> runs/<SET>/<deck>/, one MPI rank per deck,
# NPAR decks at a time. Usage (from anywhere):  BIN=/path/to/athenaPK SET=A7 [LIST=runs.txt] [NPAR=8] bash run_set.sh
# Then: python3 analyze_<SET>.py [--check]   (A2 uses the A7 runs: analyze_A2_budget.py)
set -u; ulimit -c 0
H=$(cd "$(dirname "$0")" && pwd); M=$(dirname "$H")
BIN=${BIN:?path to athenaPK}; SET=${SET:?test set, e.g. A7}; LIST=${LIST:-runs.txt}; NPAR=${NPAR:-8}
I=$M/inputs/$SET; R=$M/runs/$SET; mkdir -p "$R"
run() { mkdir -p "$R/$1" && cd "$R/$1" && mpirun -n 1 "$BIN" -i "$I/$1.in" > run.log 2>&1 < /dev/null; echo "$1 rc=$?"; }
export -f run; export BIN I R
xargs -P "$NPAR" -I{} bash -c 'run "$@"' _ {} < "$I/$LIST"
