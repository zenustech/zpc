#pragma once
#include <algorithm>
#include <vector>
#include <zensim/math/ValueSequence.hpp>

namespace zs {

#if 1
  template <typename T, typename Tn, Tn dim> struct DenseGrid : integrals<Tn, dim> {
    using IV = vec<Tn, dim>;
    using indexer = integrals<Tn, dim>;

    DenseGrid(const IV &dimsIn, T val) noexcept
        : indexer{indexer{dimsIn}.seqExclSfxProd()},
          dims{dimsIn},
          grid(indexer{dimsIn}.prod(), val) {}
    DenseGrid(IV &&dimsIn, T val) noexcept
        : indexer{indexer{dimsIn}.seqExclSfxProd()},
          dims{dimsIn},
          grid(indexer{dimsIn}.prod(), val) {}

    T operator()(const IV &indices) const { return grid[this->offset(indices)]; }
    T &operator()(const IV &indices) { return grid[this->offset(indices)]; }

    Tn domain(std::size_t d) const noexcept { return dims(d); }

    indexer dims{};
    std::vector<T> grid{};
  };

#else

  template <typename T, typename Tn, Tn dim> struct DenseGrid : integrals<Tn, dim> {
    using IV = vec<Tn, dim>;
    using indexer = integrals<Tn, dim>;

    DenseGrid(const IV &dimsIn, T val) : indexer{indexer{dimsIn}.seqExclSfxProd()}, dims{dimsIn} {
      grid = new T[indexer{dimsIn}.prod()];
      std::fill_n(grid, indexer{dimsIn}.prod(), val);
    }
    DenseGrid(IV &&dimsIn, T val) : indexer{indexer{dimsIn}.seqExclSfxProd()}, dims{dimsIn} {
      grid = new T[indexer{dimsIn}.prod()];
      std::fill_n(grid, indexer{dimsIn}.prod(), val);
    }
    ~DenseGrid() { delete[] grid; }

    T operator()(const IV &indices) const { return grid[this->offset(indices)]; }
    T &operator()(const IV &indices) { return grid[this->offset(indices)]; }

    Tn domain(std::size_t d) const noexcept { return dims(d); }

    indexer dims{};
    T *grid;
  };
#endif

}  // namespace zs
