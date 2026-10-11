#!/bin/bash
#SBATCH --job-name=rad_rstb
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R0/rstb_%j.out
# R0 restart bisection (finding from r0_restart.sh 2026-10-08: prod-overlay two-leg run NOT bit-identical one step
# after the restart). Each variant: cont 0->20, cont2 0->20 (run-to-run determinism), legA 0->12, legB from the
# cycle-11 restart -> 20 (ONE CYCLE AFTER THE READ is compared: cycle 12, and cycle 20). Every Sn floor event is
# counted (APK_SN_FLOOR_EVERY=1). VARIANTS env: space-separated list of names defined below.
export TMPDIR=/tmp APK_SN_FLOOR_EVERY=1
source $REPO/validation/self_gravity/scripts/suite_common.sh
ulimit -c 0
M=$REPO/validation/radiation_sn
BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_c69ee834}
PYB=python3
C=$REPO/validation/radiation_sn/scripts/cmp_bitwise.py
DUAL="hydro/dual_energy=true hydro/dual_energy_eta1=5e-2 hydro/dual_energy_sync_step_end=true hydro/dual_energy_unsigned_prim=true hydro/dual_energy_exact_resync=true hydro/dual_energy_eta2=0.1 hydro/dual_energy_kappa=3.0 hydro/dual_energy_shock_comp=0.02 hydro/dual_energy_shared_select=true hydro/dual_comp_ghost=true hydro/dual_energy_heat_rkl2=true hydro/boundary_clamp_dual=true hydro/ghost_sync_rounds=1 radiation/sn_dual_energy=true"
SOU="radiation/sn_spatial=sou radiation/sn_sou_kappa=1.0 radiation/sn_sou_linear_prolong=true"
SOLV="radiation/sn_krylov_dot_scale=true radiation/sn_fallback=true radiation/sn_skip_extinction=true radiation/sn_skip_group_rel=1e-12"
declare -A DK=( [orig]=RST_orig.in [sou]=RST_orig.in [dual]=RST_orig.in [solv]=RST_orig.in [prod]=RST_prod.in [prodstep]=RST_prod.in [prodnodual]=RST_prod.in )
declare -A EX=( [orig]="" [sou]="$SOU" [dual]="$DUAL" [solv]="$SOLV" [prod]="" [prodstep]="radiation/sn_spatial=step"
  [prodnodual]="hydro/dual_energy=false radiation/sn_dual_energy=false" )
leg () { d=$1; shift; mkdir -p $d; cd $d; rm -f *.rhdf *.phdf *.xdmf *.hst run.log
  { echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"
    echo "deck $DECK $(md5sum < $DECK | cut -c1-32)"; echo "cmd  mpirun -n 8 $BIN $*"; } > cmd.txt
  mpirun -n 8 $BIN "$@" > run.log 2>&1; echo "rc $?" >> cmd.txt; }
for V in ${VARIANTS:-orig sou dual solv prod prodstep prodnodual}; do
  W=$M/runs/R0/${RSTB_DIR:-RSTB}/$V; DECK=$M/inputs/R0/${DK[$V]}; X=${EX[$V]}
  leg $W/cont -i $DECK parthenon/time/nlim=20 $X
  leg $W/cont2 -i $DECK parthenon/time/nlim=20 $X
  leg $W/legA -i $DECK parthenon/time/nlim=12 $X
  leg $W/legB -r $W/legA/parthenon.out1.00011.rhdf parthenon/time/nlim=20 $X
  echo "######## $V  (deck ${DK[$V]}; extra: $X)"
  for d in cont cont2 legA legB; do echo "  $d: $(tail -1 $W/$d/cmd.txt) floors=$(grep -a 'FLOORED' $W/$d/run.log | grep -oE 'in [0-9]+ cell' | awk '{s+=$2} END {print s+0}') notconv=$(grep -aci 'not conv' $W/$d/run.log) snread=$(grep -ac 'READ ordinates' $W/$d/run.log)"; done
  echo "  cont vs cont2 @20: $($PYB $C $W/cont $W/cont2 20 | tail -1)"
  echo "  cont vs legB  @12: $($PYB $C $W/cont $W/legB 12 | tail -1)"
  echo "  cont vs legB  @20: $($PYB $C $W/cont $W/legB 20 | tail -1)"
done
echo RSTB-DONE
