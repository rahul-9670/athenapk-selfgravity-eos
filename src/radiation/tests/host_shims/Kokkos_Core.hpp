#pragma once
// HOST SHIM for Kokkos_Core.hpp. Just enough Kokkos for the radiation host test suites to
// compile the REAL device headers on the host with plain g++ -- no Kokkos build, no MPI.
//
// The macro half of this file has been here since the suites were written. The View/atomic
// half was added 2026-09-07: radiation_groups.hpp and radiation_opacity.hpp had grown a
// Kokkos::View clamp-counter and atomic_fetch_add, which quietly made test_opacity and
// test_rad_groups unbuildable. Nothing caught it because there was no runner -- see
// run_host_tests.sh.
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

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
#ifndef KOKKOS_FORCEINLINE_FUNCTION
#define KOKKOS_FORCEINLINE_FUNCTION inline
#endif

namespace Kokkos {

//! View<T*> only -- the rank the radiation headers use for counters.
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

//! 2026-10-09: R-F5's EnableLogInterp (radiation_opacity.hpp, radiation_groups.hpp) uses
//! create_mirror_view_and_copy(HostSpace(), v); the shims lacked it, so test_opacity and
//! test_rad_groups had stopped building on 2026-10-08 unnoticed. Host IS device here: the mirror is v.
struct HostSpace {};
template <typename V>
inline V create_mirror_view_and_copy(const HostSpace &, const V &v) { return v; }

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

} // namespace Kokkos
