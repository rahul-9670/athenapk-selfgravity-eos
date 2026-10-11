#!/bin/bash
#SBATCH --job-name=rad_rst
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R0/RST/rst_%j.out
# R0 restart exactness of Sn (replaces athenapk/runs/sn_restart_fix, whose leg 0 was an M1 restart; M1 is removed).
# Pure Sn from t = 0, production solver keys (inputs/R0/RST_prod.in = sn_restart_fix/deck.in + PROD_RAD + dual keys):
#   cont : one leg, cycles 0 -> 20
#   legA : cycles 0 -> 12 ;  legB : restart from legA's cycle-12 file -> 20
# PASS = legB and cont BIT-IDENTICAL at cycle 20 (every dataset, cmp_bitwise.py). Same binary, same 8 ranks.
# Also legC: restart from cycle 12 with sn_nzeta 4 -> 6 (ordinate-set change must re-seed, not abort).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
ulimit -c 0
M=$REPO/validation/radiation_sn
BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_c69ee834}
DECK=$M/inputs/R0/RST_prod.in
W=$M/runs/R0/RST
leg () { d=$W/$1; shift; mkdir -p $d; cd $d; rm -f *.rhdf *.phdf *.xdmf *.hst run.log
  { echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"
    echo "deck $DECK $(md5sum < $DECK | cut -c1-32)"; echo "cmd  mpirun -n 8 $BIN $*"; } > cmd.txt
  mpirun -n 8 $BIN "$@" > run.log 2>&1; echo "rc $?" >> cmd.txt
  echo "[$(tail -1 cmd.txt)] $(basename $d) $(grep -aE '^cycle=' run.log | tail -1 | cut -c1-90)"; }
leg cont -i $DECK parthenon/time/nlim=20
leg legA -i $DECK parthenon/time/nlim=12
R12=$(ls $W/legA/parthenon.out1.*.rhdf | grep -v final | sort | tail -1); echo "legA restart: $R12"
leg legB -r $R12 parthenon/time/nlim=20
leg legC -r $R12 parthenon/time/nlim=16 radiation/sn_nzeta=6
PYB=python3
$PYB $REPO/validation/radiation_sn/scripts/cmp_bitwise.py $W/cont $W/legB 20 | tee $W/cmp_cont_legB.txt
echo "legC: rc=$(grep rc $W/legC/cmd.txt) reseed lines: $(grep -aci 'seed' $W/legC/run.log)"
echo RST-DONE
