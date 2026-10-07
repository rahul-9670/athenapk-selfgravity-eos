#!/bin/bash
#SBATCH --job-name=sg_smoke3
#SBATCH --partition=std
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --output=$REPO/validation/self_gravity/runs/%x_%j.out
source $REPO/validation/self_gravity/scripts/suite_common.sh
export BIN=$REPO/validation/_build/bin/athenaPK_cpu_50cbbdc5
for d in L7_be_collapse_nj8 L10_hennebelle_mu2 L9_boss_bodenheimer_nj8 L8_sis; do $R $W/smoke3/$d 64 $I/$d.in parthenon/time/nlim=5; grep -a -m2 "PARTHENON ERROR\|Message:     [A-Z]" $W/smoke3/$d/run.log | grep -v ModifyFrom | head -3; done
