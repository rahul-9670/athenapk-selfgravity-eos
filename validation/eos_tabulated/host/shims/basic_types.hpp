#pragma once
#include <cstdint>
namespace parthenon { using Real = double; }

// Globals::my_rank -- table loaders gate their diagnostic prints on it.
// Rank 0 on the host, so the suites see the same messages a real rank-0 run would.
namespace parthenon {
struct Globals {
  static inline int my_rank = 0;
  static inline int nranks = 1;
};
} // namespace parthenon
