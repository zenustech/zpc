#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/execution/Atomics.hpp"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  template <typename Table> struct CleanSparsity;
  template <typename T, typename Table, typename Position> struct ComputeSparsity;
  template <typename Table> struct EnlargeSparsity;
  template <typename T, typename Table, typename Position, typename Count> struct SpatiallyCount;
  template <typename T, typename Table, typename Position, typename Indices>
  struct SpatiallyDistribute;

  template <execspace_e space, typename Table> CleanSparsity(wrapv<space>, Table)
      -> CleanSparsity<HashTableView<space, Table>>;

  template <execspace_e space, typename T, typename Table, typename X, typename... Args>
  ComputeSparsity(wrapv<space>, T, int, Table, const X&, Args...)
      -> ComputeSparsity<T, HashTableView<space, Table>, VectorView<space, const X>>;

  template <execspace_e space, typename Table>
  EnlargeSparsity(wrapv<space>, Table, vec<int, Table::dim>, vec<int, Table::dim>)
      -> EnlargeSparsity<HashTableView<space, Table>>;

  template <execspace_e space, typename T, typename Table, typename X, typename Count,
            typename... Args>
  SpatiallyCount(wrapv<space>, T, Table, const X&, Count, Args...)
      -> SpatiallyCount<T, HashTableView<space, Table>, VectorView<space, const X>,
                        VectorView<space, Count>>;

  template <execspace_e space, typename T, typename Table, typename X, typename Indices,
            typename... Args>
  SpatiallyDistribute(wrapv<space>, T, Table, const X&, Indices counts, Indices offsets,
                      Indices indices, Args...)
      -> SpatiallyDistribute<T, HashTableView<space, Table>, VectorView<space, const X>,
                             VectorView<space, Indices>>;

  /// implementations
  template <execspace_e space, typename Table> struct CleanSparsity<HashTableView<space, Table>> {
    using table_t = HashTableView<space, Table>;

    explicit CleanSparsity(wrapv<space>, Table& table) : table{proxy<space>(table)} {}

    constexpr void operator()(typename Table::value_t entry) noexcept {
      using namespace placeholders;
      table._table.keys[entry] = Table::key_t::constant(Table::key_scalar_sentinel_v);
      table._table.indices[entry] = Table::sentinel_v;  // necessary for query to terminate
      table._table.status[entry] = -1;
      if (entry == 0) *table._cnt = 0;
    }

    table_t table;
  };

  template <execspace_e space, typename T, typename Table, typename X>
  struct ComputeSparsity<T, HashTableView<space, Table>, VectorView<space, const X>> {
    using table_t = HashTableView<space, Table>;
    using positions_t = VectorView<space, const X>;

    explicit ComputeSparsity(wrapv<space>, T dx, int blockLen, Table& table, const X& pos,
                             int offset = -2, T displacement = (T)0.5f)
        : table{proxy<space>(table)},
          pos{proxy<space>(pos)},
          dxinv{(T)1.0 / dx},
          displacement{displacement},
          blockLen{blockLen},
          offset{offset} {}

    constexpr void operator()(typename positions_t::size_type parid) noexcept {
      vec<int, table_t::dim> coord{};
      for (int d = 0; d != table_t::dim; ++d)
        coord[d] = lower_trunc(pos(parid)[d] * dxinv + displacement) + offset;
      auto blockid = coord;
      for (int d = 0; d != table_t::dim; ++d) blockid[d] += (coord[d] < 0 ? -blockLen + 1 : 0);
      blockid = blockid / blockLen;
      table.insert(blockid);
    }

    table_t table;
    positions_t pos;
    T dxinv, displacement;
    int blockLen;
    int offset;
  };

  template <execspace_e space, typename Table> struct EnlargeSparsity<HashTableView<space, Table>> {
    static constexpr int dim = Table::dim;
    using table_t = HashTableView<space, Table>;

    explicit EnlargeSparsity(wrapv<space>, Table& table, vec<int, dim> lo, vec<int, dim> hi)
        : table{proxy<space>(table)}, lo{lo}, hi{hi} {}

    constexpr void operator()(typename table_t::value_t i) noexcept {
      if constexpr (table_t::dim == 3) {
        auto blockid = table._activeKeys[i];
        for (int dx = lo[0]; dx < hi[0]; ++dx)
          for (int dy = lo[1]; dy < hi[1]; ++dy)
            for (int dz = lo[2]; dz < hi[2]; ++dz) {
              table.insert(blockid + vec<int, 3>{dx, dy, dz});
            }
      } else if constexpr (table_t::dim == 2) {
        auto blockid = table._activeKeys[i];
        for (int dx = lo[0]; dx < hi[0]; ++dx)
          for (int dy = lo[1]; dy < hi[1]; ++dy) table.insert(blockid + vec<int, 2>{dx, dy});
      }
    }

    table_t table;
    vec<int, dim> lo, hi;
  };

  template <execspace_e space, typename T, typename Table, typename X, typename CountT>
  struct SpatiallyCount<T, HashTableView<space, Table>, VectorView<space, const X>,
                        VectorView<space, CountT>> {
    using table_t = HashTableView<space, Table>;
    using positions_t = VectorView<space, const X>;
    using counters_t = VectorView<space, CountT>;
    using counter_interger_type = zs::make_unsigned_t<typename CountT::value_type>;

    explicit SpatiallyCount(wrapv<space>, T dx, Table& table, const X& pos, CountT& cnts,
                            int blockLen = 1, int offset = 0, T displacement = (T)0.5f)
        : table{proxy<space>(table)},
          pos{proxy<space>(pos)},
          counts{proxy<space>(cnts)},
          dxinv{(T)1.0 / dx},
          displacement{displacement},
          blockLen{blockLen},
          offset{offset} {}

    constexpr void operator()(typename positions_t::size_type parid) noexcept {
      vec<int, table_t::dim> coord{};
      for (int d = 0; d != table_t::dim; ++d)
        coord[d] = lower_trunc(pos(parid)[d] * dxinv + displacement) + offset;
      auto blockid = coord;
      for (int d = 0; d != table_t::dim; ++d) blockid[d] += (coord[d] < 0 ? -blockLen + 1 : 0);
      blockid = blockid / blockLen;
      /// guarantee counts are non-negative, thus perform this explicit type conversion for cuda
      /// atomic overload
      atomic_add(wrapv<space>{}, (counter_interger_type*)&counts(table.query(blockid)),
                 (counter_interger_type)1);
    }

    table_t table;
    positions_t pos;
    counters_t counts;
    T dxinv, displacement;
    int blockLen;
    int offset;
  };

  template <execspace_e space, typename T, typename Table, typename X, typename Indices>
  struct SpatiallyDistribute<T, HashTableView<space, Table>, VectorView<space, const X>,
                             VectorView<space, Indices>> {
    using table_t = HashTableView<space, Table>;
    using positions_t = VectorView<space, const X>;
    using indices_t = VectorView<space, Indices>;
    using counter_interger_type = zs::make_unsigned_t<typename Indices::value_type>;

    explicit SpatiallyDistribute(wrapv<space>, T dx, Table& table, const X& pos, Indices& cnts,
                                 Indices& offsets, Indices& indices, int blockLen = 1,
                                 int offset = 0, T displacement = (T)0.5f)
        : table{proxy<space>(table)},
          pos{proxy<space>(pos)},
          counts{proxy<space>(cnts)},
          offsets{proxy<space>(offsets)},
          indices{proxy<space>(indices)},
          dxinv{(T)1.0 / dx},
          displacement{displacement},
          blockLen{blockLen},
          offset{offset} {}

    constexpr void operator()(typename positions_t::size_type parid) noexcept {
      vec<int, table_t::dim> coord{};
      for (int d = 0; d != table_t::dim; ++d)
        coord[d] = lower_trunc(pos(parid)[d] * dxinv + displacement) + offset;
      auto blockid = coord;
      for (int d = 0; d < table_t::dim; ++d) blockid[d] += (coord[d] < 0 ? -blockLen + 1 : 0);
      blockid = blockid / blockLen;
      auto cellno = table.query(blockid);
      /// guarantee counts are non-negative, thus perform this explicit type conversion for cuda
      /// atomic overload
      auto dst = atomic_add(wrapv<space>{}, (counter_interger_type*)&counts(cellno),
                            (counter_interger_type)1);
      indices(offsets(cellno) + dst) = parid;
    }

    table_t table;
    positions_t pos;
    indices_t counts, offsets, indices;
    T dxinv, displacement;
    int blockLen;
    int offset;
  };

}  // namespace zs