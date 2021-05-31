#pragma once
#include <utility>

#include "zensim/container/HashTable.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <int dim_ = 3, int lane_width_ = 32> struct IndexBuckets {
    static constexpr int dim = dim_;
    static constexpr int lane_width = lane_width_;
    using value_type = f32;
    using index_type = i64;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;
    using table_t = HashTable<int, dim, index_type>;
    using vector_t = Vector<index_type>;

    constexpr IndexBuckets() = default;

    IndexBuckets clone(const MemoryHandle mh) const {
      IndexBuckets ret{};
      ret._table = _table.clone(mh);
      ret._indices = _indices.clone(mh);
      ret._counts = _counts.clone(mh);
      ret._dx = _dx;
      return ret;
    }

    table_t _table{};
    vector_t _indices{}, _counts{};
    value_type _dx{1};
  };

  using GeneralIndexBuckets = variant<IndexBuckets<3, 32>, IndexBuckets<3, 8>>;

  template <execspace_e Space, typename IndexBucketsT, typename = void> struct IndexBucketsProxy {
    using value_type = typename IndexBucketsT::value_type;
    using index_type = typename IndexBucketsT::index_type;
    using TV = typename IndexBucketsT::TV;
    using IV = typename IndexBucketsT::IV;
    using table_t = typename IndexBucketsT::table_t;
    using vector_t = typename IndexBucketsT::vector_t;

    constexpr IndexBucketsProxy() = default;
    ~IndexBucketsProxy() = default;
    IndexBucketsProxy(IndexBucketsT &ibs)
        : table{ibs._table}, indices{ibs._indices}, counts{ibs._counts}, dx{ibs._dx} {}

    HashTableProxy<Space, table_t> table;
    VectorProxy<Space, vector_t> indices, counts;
    value_type dx;
  };

  template <execspace_e ExecSpace, int dim, int lane_width>
  decltype(auto) proxy(IndexBuckets<dim, lane_width> &indexBuckets) {
    return IndexBucketsProxy<ExecSpace, IndexBuckets<dim, lane_width>>{indexBuckets};
  }

}  // namespace zs