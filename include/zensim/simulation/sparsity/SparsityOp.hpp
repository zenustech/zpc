#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  template <typename Table> struct CleanSparsity;
  template <typename T, typename Table, typename Position> struct ComputeSparsity;
  template <typename Table> struct EnlargeSparsity;

  template <execspace_e space, typename Table> CleanSparsity(wrapv<space>, Table)
      -> CleanSparsity<HashTableProxy<space, Table>>;

  template <execspace_e space, typename T, typename Table, typename X>
  ComputeSparsity(wrapv<space>, T, int, Table, X)
      -> ComputeSparsity<T, HashTableProxy<space, Table>, VectorProxy<space, X>>;

  template <execspace_e space, typename Table>
  EnlargeSparsity(wrapv<space>, Table, vec<int, Table::dim>, vec<int, Table::dim>)
      -> EnlargeSparsity<HashTableProxy<space, Table>>;

  /// implementations
  template <execspace_e space, typename Table> struct CleanSparsity<HashTableProxy<space, Table>> {
    using table_t = HashTableProxy<space, Table>;

    explicit CleanSparsity(wrapv<space>, Table& table) : table{proxy<space>(table)} {}

    constexpr void operator()(typename Table::value_t entry) noexcept {
      using namespace placeholders;
#if 0
      if (entry <= 2) {
        std::intptr_t addr = reinterpret_cast<std::intptr_t>((void*)&table._table(_1, entry));
        if (addr != table._table.template channel_offset<1>())
          printf("WTF? entry: %d, addr: %lld, real: %lld, expected: %lld, calc: %lld\n", (int)entry,
                 (long long int)addr, (long long int)addr - table._table.template address<0>(),
                 (long long int)table._table.template channel_offset<1>(),
                 (long long int)table._table.template element_offset(_1, 0, entry));
      }
#endif
      table._table(_0, entry) = Table::key_t::uniform(Table::key_scalar_sentinel_v);
      table._table(_1, entry) = Table::sentinel_v;  // necessary for query to terminate
      table._table(_2, entry) = -1;
      if (entry == 0) *table._cnt = 0;
    }

    table_t table;
  };

  template <execspace_e space, typename T, typename Table, typename X>
  struct ComputeSparsity<T, HashTableProxy<space, Table>, VectorProxy<space, X>> {
    using table_t = HashTableProxy<space, Table>;
    using positions_t = VectorProxy<space, X>;

    explicit ComputeSparsity(wrapv<space>, T dx, int blockLen, Table& table, X& pos)
        : dxinv{(T)1.0 / dx},
          blockLen{blockLen},
          table{proxy<space>(table)},
          pos{proxy<space>(pos)} {}

    constexpr void operator()(typename positions_t::size_type parid) noexcept {
      vec<int, table_t::dim> coord{};
      for (int d = 0; d < table_t::dim; ++d)
        coord[d] = lower_trunc(pos(parid)[d] * dxinv + 0.5) - 2;
      auto blockid = coord;
      for (int d = 0; d < table_t::dim; ++d) blockid[d] += (coord[d] < 0 ? -blockLen + 1 : 0);
      blockid = blockid / blockLen;
      table.insert(blockid);
    }

    T dxinv;
    int blockLen;
    table_t table;
    positions_t pos;
  };

  template <execspace_e space, typename Table>
  struct EnlargeSparsity<HashTableProxy<space, Table>> {
    static constexpr int dim = Table::dim;
    using table_t = HashTableProxy<space, Table>;

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

}  // namespace zs