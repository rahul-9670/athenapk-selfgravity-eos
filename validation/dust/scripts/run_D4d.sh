#!/bin/bash
# D4d: two-leg restart of the D4c 1D N256 temperature-contact deck (production dust keys, freeze_growth).
# leg1 = one uninterrupted run that also writes a restart at tlim/2; leg2 = restart from that file to tlim.
# Pass = leg2's final phdf BIT-IDENTICAL to leg1's (every dataset, bitcmp_phdf.py).
# Usage: BIN=/path/to/athenaPK bash run_D4d.sh   (decks: substitute $REPO first, see ../README.md)
set -u; ulimit -c 0
export OMP_PROC_BIND=false OMP_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); DU=$(dirname "$H")
BIN=${BIN:?path to athenaPK}
DECK=$DU/inputs/D4/D4d_1d_N256.in
R=$DU/runs/D4d; rm -rf $R; mkdir -p $R/leg1 $R/leg2
cd $R/leg1 && mpirun -n 1 --bind-to none $BIN -i $DECK > run.log 2>&1; echo "leg1 rc $?"
RST=$(ls $R/leg1/*.out2.00001.rhdf); echo "restart $RST"
cd $R/leg2 && mpirun -n 1 --bind-to none $BIN -r $RST > run.log 2>&1; echo "leg2 rc $?"
python3 $H/bitcmp_phdf.py $R/leg1 $R/leg2 | tee $R/bitcmp.txt
