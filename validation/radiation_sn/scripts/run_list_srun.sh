#!/bin/bash
#SBATCH --job-name=rad_lists
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=02:00:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/lists_%j.out
# As run_list.sh, but every run is an `srun --mpi=pmix --exact -n <nranks>` job step. NOTE R-F7: the Open MPI sm BTL
# (on via suite_common.sh) hangs multi-rank mesh-refined Sn runs with srun AND mpirun; such lists need btl tcp,self. Lines "rundir | deck | overrides | nranks"; FILTER anchored; at most NPAR
# concurrent steps.
# --cpu-bind=cores is REQUIRED (R-F7, 2026-10-08): athenapk_env.sh exports OMP_PROC_BIND=spread OMP_PLACES=threads;
# without a per-rank mask every rank of a step gets the whole step cpuset and libgomp pins its one thread to the
# first place of it -> all ranks of a step on ONE hardware thread (measured: 1.0 of 8 cores, 8x slower).
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
ulimit -c 0
# BTL_TCP=1: btl tcp,self (R-F7: sm hangs multi-rank mesh-refined Sn runs; commas cannot go through --export)
[ "${BTL_TCP:-0}" = 1 ] && export OMPI_MCA_btl=tcp,self
M=${RAD_MOD:-$REPO/validation/radiation_sn}
BIN=${BIN:?}
NPAR=${NPAR:-8}
echo "BIN $(md5sum $BIN)  LIST $LIST  FILTER ${FILTER:-.}  btl $OMPI_MCA_btl"
one () { d=$M/runs/$1; deck=$M/inputs/$2; np=$4; shift 4; mkdir -p $d; cd $d; rm -f *.phdf *.rhdf *.xdmf *.hst run.log
  { echo "date $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none}"; echo "bin  $(md5sum $BIN)"; echo "deck $deck $(md5sum < $deck | cut -c1-32)"; echo "cmd  srun --mpi=pmix --exact --cpu-bind=cores ${SRUN_EXTRA:-} -n $np $BIN -i $deck $*"; } > cmd.txt
  T0=$(date +%s.%N); timeout ${CAP:-100000} srun --mpi=pmix --exact --cpu-bind=cores ${SRUN_EXTRA:-} -n $np $BIN -i $deck "$@" > run.log 2>&1; rc=$?
  echo "rc $rc wall $(echo "$(date +%s.%N) - $T0" | bc) s" >> cmd.txt
  echo "[$rc] $(basename $d) $(grep -aE '^cycle=' run.log | tail -1 | cut -c1-90)"; }
export -f one; export M BIN
# one line per run, newline-delimited (xargs -L would join a line that ends in a blank, i.e. empty overrides)
grep -E "^${FILTER:-.}" "$LIST" | while IFS='|' read -r d k o n; do echo "$(echo $d) $(echo $k) x ${n:-1} $(echo $o)"; done |
  xargs -d '\n' -P $NPAR -I{} bash -c 'one {}'
echo LIST-DONE
