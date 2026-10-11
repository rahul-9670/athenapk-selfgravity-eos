#pragma once
// HOST SHIM for parthenon_arrays.hpp -- just enough of ParArray2D<T> for the host test
// suites to compile radiation_groups.hpp and radiation_opacity.hpp.
//
// WHY IT EXISTS. test_opacity and test_rad_groups document themselves as buildable with
// `g++ -Ihost_shims <file>.cpp`, and both had stopped being buildable: the two headers they
// include grew an `#include <parthenon_arrays.hpp>` for the tabulated ParArray2D tables, and
// host_shims was never extended to match. Nothing noticed, because there was no runner --
// each suite's only instruction lived in a comment in its own file. Found 2026-09-07 when
// run_host_tests.sh was added and immediately reported 2 of 9 unbuildable.
//
// Deliberately a plain heap array with reference-counted sharing, NOT a Kokkos View: the
// point of the host suites is to compile the SAME arithmetic without the Kokkos build. Only
// what the two headers actually use is provided.
#include <memory>
#include <string>

namespace parthenon {

template <typename T>
class ParArray2D {
 public:
  ParArray2D() = default;
  ParArray2D(const std::string & /*label*/, const int n0, const int n1)
      : n0_(n0), n1_(n1), data_(new T[static_cast<size_t>(n0) * n1]()) {}
  // (slow index, fast index) -- same order as Kokkos's LayoutRight default on the host.
  inline T &operator()(const int i, const int j) const {
    return data_[static_cast<size_t>(i) * n1_ + j];
  }
  inline int extent_int(const int d) const { return d == 0 ? n0_ : n1_; }
  inline size_t size() const { return static_cast<size_t>(n0_) * n1_; }
  inline T *data() const { return data_.get(); }
  inline bool is_allocated() const { return static_cast<bool>(data_); }

 private:
  int n0_{0}, n1_{0};
  std::shared_ptr<T[]> data_{};
};

//! R-F11 (2026-10-09): rank-3 for the format-v4 rho-dependent group multipliers (radiation_groups.hpp).
template <typename T>
class ParArray3D {
 public:
  ParArray3D() = default;
  ParArray3D(const std::string & /*label*/, const int n0, const int n1, const int n2)
      : n0_(n0), n1_(n1), n2_(n2), data_(new T[static_cast<size_t>(n0) * n1 * n2]()) {}
  inline T &operator()(const int i, const int j, const int k) const {
    return data_[(static_cast<size_t>(i) * n1_ + j) * n2_ + k];
  }
  inline int extent_int(const int d) const { return d == 0 ? n0_ : (d == 1 ? n1_ : n2_); }
  inline size_t size() const { return static_cast<size_t>(n0_) * n1_ * n2_; }
  inline T *data() const { return data_.get(); }

 private:
  int n0_{0}, n1_{0}, n2_{0};
  std::shared_ptr<T[]> data_{};
};

//! Rank-0 device scalar used for the opacity/EOS clamp counters.
template <typename T>
class ParArray1D {
 public:
  ParArray1D() = default;
  ParArray1D(const std::string & /*label*/, const int n)
      : n_(n), data_(new T[n]()) {}
  inline T &operator()(const int i) const { return data_[i]; }
  inline int extent_int(const int) const { return n_; }
  inline T *data() const { return data_.get(); }

 private:
  int n_{0};
  std::shared_ptr<T[]> data_{};
};

} // namespace parthenon
