#pragma once
#include <utility>

#include "zensim/container/HashTable.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"

namespace zs {

  template <int dim_ = 3> struct IndexBuckets {
    static constexpr int dim = dim_;
    static constexpr int lane_width = 32;
    using value_type = f32;
    using index_type = i64;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;

    constexpr IndexBuckets() = default;

    IndexBuckets clone(const MemoryHandle mh) const {
      IndexBuckets ret{};
      ret._table = _table.clone(mh);
      ret._indices = _indices.clone(mh);
      ret._counts = _counts.clone(mh);
      ret._dx = _dx;
      return ret;
    }

    HashTable<int, dim, index_type> _table{};
    Vector<index_type> _indices{}, _counts{};
    value_type _dx{1};
  };

}  // namespace zs