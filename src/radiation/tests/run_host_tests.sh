#!/bin/bash
# Compile and run every radiation HOST test suite, and report one line each.
#
# WHY THIS EXISTS. There are nine of these and, until 2026-09-07, the only instruction for
# running any of them was a `g++ ...` line in a comment at the top of its own file. So the
# suites were run when someone remembered the file existed -- which is not a regression check,
# it is a habit. This makes "did I break a host test?" one command.
#
# -Ihost_shims is REQUIRED and is why five of these looked broken when this runner was first
# written with a plain g++ line: host_shims/ carries stand-in Kokkos_Core.hpp / Kokkos_Pair.hpp
# / basic_types.hpp so the real device headers compile on the host. The Sn suites do not need
# it (their headers pull in no Kokkos), which is exactly why the omission was easy to miss.
#
# These are deliberately NOT CMake targets: each one #includes a device-callable header
# directly and compiles the SAME arithmetic the GPU kernel runs, with KOKKOS_INLINE_FUNCTION
# falling back to `inline` on the host. That is the pattern that caught the frozen-closure
# upwinding defect in M1, and it works precisely because it does not need the Kokkos build.
#
#   ./run_host_tests.sh          run all
#   ./run_host_tests.sh sn       run only suites whose name matches 'sn'
#
# Exit status is the number of suites that failed to build or failed their checks.
set -u
cd "$(dirname "$0")"
OUT=${TMPDIR:-/tmp}/rad_host_tests.$$
mkdir -p "$OUT"
FILTER=${1:-}
fails=0; ran=0
for f in test_*.cpp; do
  name=${f%.cpp}
  [ -n "$FILTER" ] && case "$name" in *"$FILTER"*) ;; *) continue ;; esac
  ran=$((ran+1))
  # GLIBC 2.41 WORKAROUND (2026-09-14): glibc 2.41 changed the pthread_mutex_t / pthread_cond_t
  # static initializers to nested-brace
  # forms that GCC 13.3.0's libstdc++ (configured against the old glibc) cannot brace-initialize:
  # every TU that reaches <mutex> fails at std_mutex.h:206 / concurrence.h:252. These two
  # defines are libstdc++'s own escape hatches (gthr-default.h:72,80): they switch to
  # pthread_*_init() calls. Same compiler, same ABI.
  # build_cpu/CMakeCache.txt carries the same two flags in CMAKE_CXX_FLAGS.
  GLIBC241="-D_GTHREAD_USE_MUTEX_INIT_FUNC -D_GTHREAD_USE_COND_INIT_FUNC"
  if ! g++ -O2 -std=c++17 -Wall -Wextra $GLIBC241 -Ihost_shims "$f" -o "$OUT/$name" > "$OUT/$name.build" 2>&1; then
    printf "  %-24s BUILD FAILED  (see %s)\n" "$name" "$OUT/$name.build"
    fails=$((fails+1)); continue
  fi
  if "$OUT/$name" > "$OUT/$name.log" 2>&1; then
    printf "  %-24s %s\n" "$name" "$(tail -1 "$OUT/$name.log")"
  else
    printf "  %-24s %s\n" "$name" "$(tail -1 "$OUT/$name.log")"
    printf "      failing checks:\n"
    grep -a "  FAIL" "$OUT/$name.log" | sed 's/^/      /'
    fails=$((fails+1))
  fi
done
echo "---"
echo "$ran suite(s) run, $fails failed.  logs: $OUT"
exit $fails
