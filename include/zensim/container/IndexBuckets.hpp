#pragma once
#include <utility>

#include "zensim/container/HashTable.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <int dim_ = 3, typename Index = i64, typename Tn_ = i32> struct IndexBuckets {
    static constexpr int dim = dim_;
    using allocator_type = ZSPmrAllocator<>;
    using value_type = f32;
    using index_type = std::make_signed_t<Index>;
    using size_type = std::make_unsigned_t<index_type>;
    using Tn = std::make_signed_t<Tn_>;
    using TV = vec<value_type, dim>;
    using IV = vec<Tn, dim>;
    using table_t = HashTable<Tn, dim, index_type>;
    using vector_t = Vector<index_type>;

    constexpr IndexBuckets() = default;

    IndexBuckets clone(const allocator_type &allocator) const {
      IndexBuckets ret{};
      ret._table = _table.clone(allocator);
      ret._indices = _indices.clone(allocator);
      ret._offsets = _offsets.clone(allocator);
      ret._counts = _counts.clone(allocator);
      ret._dx = _dx;
      return ret;
    }
    IndexBuckets clone(const MemoryLocation &mloc) const {
      return clone(get_memory_source(mloc.memspace(), mloc.devid()));
    }

    constexpr auto numEntries() const noexcept { return _indices.size(); }
    constexpr auto numBuckets() const noexcept { return _counts.size() - 1; }

    table_t _table{};
    vector_t _indices{}, _offsets{}, _counts{};
    value_type _dx{1};
  };

  using GeneralIndexBuckets
      = variant<IndexBuckets<3, i32, i32>, IndexBuckets<3, i64, i32>, IndexBuckets<3, i32, i64>,
                IndexBuckets<3, i64, i64>, IndexBuckets<2, i32, i32>, IndexBuckets<2, i64, i32>,
                IndexBuckets<2, i32, i64>, IndexBuckets<2, i64, i64>>;

  template <execspace_e Space, typename IndexBucketsT, typename = void> struct IndexBucketsView {
    using value_type = typename IndexBucketsT::value_type;
    using index_type = typename IndexBucketsT::index_type;
    using TV = typename IndexBucketsT::TV;
    using IV = typename IndexBucketsT::IV;
    using table_t = typename IndexBucketsT::table_t;
    using vector_t = typename IndexBucketsT::vector_t;
    static constexpr int dim = IndexBucketsT::dim;

    constexpr IndexBucketsView() = default;
    ~IndexBucketsView() = default;
    constexpr IndexBucketsView(IndexBucketsT &ibs)
        : table{proxy<Space>(ibs._table)},
          indices{proxy<Space>(ibs._indices)},
          offsets{proxy<Space>(ibs._offsets)},
          counts{proxy<Space>(ibs._counts)},
          dx{ibs._dx} {}

    constexpr auto coord(const index_type bucketno) const noexcept {
      return table._activeKeys[bucketno];
    }
    template <typename T> constexpr auto bucketCoord(const vec<T, dim> &pos) const {
      return world_to_index<typename table_t::Tn>(pos, 1.0 / dx, 0);
    }
    constexpr auto bucketNo(const vec<typename table_t::Tn, dim> &coord) const {
      return table.query(coord);
    }

    HashTableView<Space, table_t> table;  // activekeys, table
    VectorView<Space, vector_t> indices, offsets, counts;
    value_type dx;
  };

  template <execspace_e ExecSpace, int dim, typename Index, typename Tn>
  constexpr decltype(auto) proxy(IndexBuckets<dim, Index, Tn> &indexBuckets) {
    return IndexBucketsView<ExecSpace, IndexBuckets<dim, Index, Tn>>{indexBuckets};
  }

}  // namespace zs