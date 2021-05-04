#pragma once
#include "zensim/cuda/container/HashTable.hpp"
#include "zensim/geometry/SparseLevelSet.hpp"

namespace zs {

#if 0
  template <typename SparseLevelSetT>
  struct SparseLevelSetProxy<execspace_e::cuda, SparseLevelSetT>
      : LevelSetInterface<SparseLevelSetProxy<execspace_e::cuda, SparseLevelSetT>,
                          typename SparseLevelSetT::value_type, SparseLevelSetT::dim> {
    using table_t = typename SparseLevelSetT::table_t;
    using tiles_t = typename SparseLevelSetT::tiles_t;
    using T = typename SparseLevelSetT::value_type;
    static constexpr int dim = SparseLevelSetT::dim;
    using TV = vec<T, dim>;
    static constexpr auto Space = execspace_e::cuda;

    constexpr SparseLevelSetProxy() = default;
    ~SparseLevelSetProxy() = default;
    explicit SparseLevelSetProxy(const std::vector<SmallString> &tagNames, SparseLevelSetT &ls)
        : table{proxy<Space>(ls._table)}, tiles{proxy<Space>(tagNames, ls._tiles)} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {}
    constexpr TV getNormal(const TV &X) const noexcept {}
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {}
    constexpr decltype(auto) getBoundingBox() const noexcept {}

    HashTableProxy<Space, table_t> table;
    TileVectorProxy<Space, tiles_t> tiles;
  };
#endif

}  // namespace zs