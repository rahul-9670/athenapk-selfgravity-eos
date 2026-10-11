#!/bin/bash
#SBATCH --job-name=rad_rst0
#SBATCH --partition=std
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:15:00
#SBATCH --output=$REPO/validation/radiation_sn/runs/R0/rst0_%j.out
# R-F1 fix, the case the original logic existed for: a restart from a CYCLE-0 file must still re-seed sn.u from the
# moments (sn_init_at_start = 1), and the result should equal the uninterrupted run. Fixed binary only.
export TMPDIR=/tmp
source $REPO/validation/self_gravity/scripts/suite_common.sh
M=$REPO/validation/radiation_sn; W=$M/runs/R0/RSTB_fix/prod
BIN=${BIN:-$REPO/validation/_build/bin/athenaPK_cpu_4135083d}
PYB=python3; C=$REPO/validation/radiation_sn/scripts/cmp_bitwise.py
d=$W/legR0; mkdir -p $d; cd $d; rm -f *.rhdf *.phdf *.xdmf run.log
echo "bin $(md5sum $BIN)" > cmd.txt; echo "cmd mpirun -n 8 $BIN -r $W/legA/parthenon.out1.00000.rhdf parthenon/time/nlim=20" >> cmd.txt
mpirun -n 8 $BIN -r $W/legA/parthenon.out1.00000.rhdf parthenon/time/nlim=20 > run.log 2>&1; echo "rc $?" >> cmd.txt
grep -a 'Sn restart' run.log
$PYB -c "import h5py; print('sn_init_at_start =', h5py.File('$d/parthenon.out1.final.rhdf','r')['Params'].attrs['radiation/sn_init_at_start'])"
echo "cont vs legR0 @20: $($PYB $C $W/cont $d 20 | tail -1)"
