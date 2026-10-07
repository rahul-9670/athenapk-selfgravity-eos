#pragma once
// HOST SHIM for Kokkos_Core.hpp. Just enough Kokkos for the host test harnesses to
// compile the REAL device headers (src/eos/eos_table.hpp) on the host with plain g++ -- no
// Kokkos build, no MPI. The View/atomic half supports the clamp counters.
//
// SEMANTICS ON THE HOST. Views are reference-counted heap arrays, zero-initialised like the
// real thing; a default-constructed View has data() == nullptr, which is the exact condition
// the clamp counters test to decide whether to tally. create_mirror_view returns the SAME
// view (host space is device space here) and deep_copy is a copy or a fill. atomic_fetch_add
// is a plain add: the suites are single-threaded.
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <cmath>
#include <initializer_list>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
#ifndef KOKKOS_FORCEINLINE_FUNCTION
#define KOKKOS_FORCEINLINE_FUNCTION inline
#endif

namespace Kokkos {

//! View<T*> only -- the rank the table headers use for counters.
template <typename T>
class View;

template <typename T>
class View<T *> {
 public:
  View() = default;
  View(const std::string & /*label*/, const size_t n)
      : n_(n), data_(new T[n]()) {}
  inline T &operator()(const size_t i) const { return data_[i]; }
  inline T *data() const { return data_.get(); }
  inline size_t extent(const int) const { return n_; }
  inline size_t size() const { return n_; }

 private:
  size_t n_{0};
  std::shared_ptr<T[]> data_{};
};

template <typename T>
inline View<T *> create_mirror_view(const View<T *> &v) { return v; }

//! Generic overload so ParArray2D (host_shims/parthenon_arrays.hpp) mirrors too. Host space
//! IS device space here, so a mirror is the same object and deep_copy between them is a no-op
//! -- which is exactly why the tabulated-table builders work unchanged on the host.
template <typename V>
inline V create_mirror_view(const V &v) { return v; }

template <typename V>
inline void deep_copy(const V &dst, const V &src) { (void)dst; (void)src; }

template <typename T>
inline void deep_copy(const View<T *> &dst, const View<T *> &src) {
  if (dst.data() == src.data()) return;
  std::copy(src.data(), src.data() + src.size(), dst.data());
}

template <typename T, typename S>
inline void deep_copy(const View<T *> &dst, const S value) {
  std::fill(dst.data(), dst.data() + dst.size(), static_cast<T>(value));
}

template <typename T, typename S>
inline T atomic_fetch_add(T *addr, const S incr) {
  const T old = *addr;
  *addr += static_cast<T>(incr);
  return old;
}

// ---- EOS-suite additions (validation_tests/eos_tabulated/host): what eos_table.hpp's p_over_rho
// loader needs on the host. Serial, row-major, same arithmetic order.
template <unsigned N> struct Rank {};
template <typename R> struct MDRangePolicy {
  int lo[2], hi[2];
  MDRangePolicy(std::initializer_list<int> a, std::initializer_list<int> b) {
    std::copy(a.begin(), a.end(), lo); std::copy(b.begin(), b.end(), hi);
  }
};
template <typename R, typename F>
inline void parallel_for(const std::string &, const MDRangePolicy<R> &p, const F &f) {
  for (int i = p.lo[0]; i < p.hi[0]; ++i)
    for (int j = p.lo[1]; j < p.hi[1]; ++j) f(i, j);
}
inline void fence() {}
inline double pow(double a, double b) { return std::pow(a, b); }
} // namespace Kokkos
#ifndef KOKKOS_LAMBDA
#define KOKKOS_LAMBDA [=]
#endif
