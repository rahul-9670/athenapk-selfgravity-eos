#!/bin/bash
# Build + run E5 on the host against the validation worktree's REAL eos_table.hpp.
set -e
cd "$(dirname "$0")"
SRC=$REPO/src/eos
TABLE=$REPO/src/eos/eos_table_hires_v5_cold.bin
source ~/athenapk_env.sh >/dev/null 2>&1
g++ -O2 -std=c++17 -Wall -D_GTHREAD_USE_MUTEX_INIT_FUNC -D_GTHREAD_USE_COND_INIT_FUNC -Ishims -I$SRC e5_inversion.cpp -o e5_inversion
echo "# E5 $(date -Is)  eos_table.hpp md5 $(md5sum $SRC/eos_table.hpp | cut -c1-8)  table md5 $(md5sum $TABLE | cut -c1-8)"
./e5_inversion $TABLE
