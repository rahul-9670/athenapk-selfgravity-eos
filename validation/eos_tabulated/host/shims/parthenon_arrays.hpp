#pragma once
// HOST SHIM for parthenon_arrays.hpp -- just enough of ParArray2D<T> for the host test
// harnesses to compile the table headers with `g++ -Ishims <file>.cpp`.
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

//! Rank-0 device scalar used for the EOS clamp counters.
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
