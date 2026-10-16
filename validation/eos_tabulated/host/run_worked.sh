#!/bin/bash
# Build + run the worked examples of EOS_VALIDATION.md (Sec. W) on the host against the REAL eos_table.hpp.
#   REPO=/path/to/clone ./run_worked.sh > ../results/tables/W_worked_examples.txt
set -e
cd "$(dirname "$0")"
REPO=${REPO:-$(cd ../../.. && pwd)}
SRC=$REPO/src/eos
TABLE=${TABLE:-$REPO/src/eos/eos_table_hires_v5_cold_rho26.bin}
g++ -O2 -std=c++17 -Wall -D_GTHREAD_USE_MUTEX_INIT_FUNC -D_GTHREAD_USE_COND_INIT_FUNC -Ishims -I$SRC worked_examples.cpp -o worked_examples
echo "# eos_table.hpp md5 $(md5sum $SRC/eos_table.hpp | cut -c1-8)  table md5 $(md5sum $TABLE | cut -c1-8)"
./worked_examples $TABLE
