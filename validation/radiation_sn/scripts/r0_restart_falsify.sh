#!/bin/bash
#SBATCH --job-name=rad_rstf
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:20:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R0/RSTB/rstf_%j.out
# Falsification of the R0 restart diagnosis (stale parthenon/time/ncycle = 0 in a fresh-start run's restart Input
# makes Sn::Initialize re-seed sn.u by P1 although it was read). Uses the 'orig' variant of r0_restart_bisect.sh.
#  T1: restart from legB's cycle-15 file (its Input says ncycle = 11) -> 20; predicted BIT-IDENTICAL to legB @20.
#  T2: legA cycle-11 restart with CLI parthenon/time/ncycle=11 -> 20; predicted BIT-IDENTICAL to cont @12 and @20.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
ulimit -c 0
M=$REPO/validation/radiation_sn; W=$M/runs/R0/RSTB/orig
BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_c69ee834}
PYB=python3; C=$REPO/validation/radiation_sn/scripts/cmp_bitwise.py
leg () { d=$1; shift; mkdir -p $d; cd $d; rm -f *.rhdf *.phdf *.xdmf *.hst run.log
  { echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"; echo "cmd  mpirun -n 8 $BIN $*"; } > cmd.txt
  mpirun -n 8 $BIN "$@" > run.log 2>&1; echo "rc $?" >> cmd.txt; }
leg $W/legT1 -r $W/legB/parthenon.out1.00015.rhdf parthenon/time/nlim=20
leg $W/legT2 -r $W/legA/parthenon.out1.00011.rhdf parthenon/time/nlim=20 parthenon/time/ncycle=11
echo "T1 legB vs legT1 @20: $($PYB $C $W/legB $W/legT1 20 | tail -1)"
echo "T2 cont vs legT2 @12: $($PYB $C $W/cont $W/legT2 12 | tail -1)"
echo "T2 cont vs legT2 @20: $($PYB $C $W/cont $W/legT2 20 | tail -1)"
echo "T2 first cycle line: $(grep -aE '^cycle=' $W/legT2/run.log | head -1 | cut -c1-60)"
echo RSTF-DONE
