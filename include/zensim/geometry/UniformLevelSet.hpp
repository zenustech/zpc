#pragma once
#include "LevelSetInterface.h"
#include "VdbLevelSet.h"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/TileVector.hpp"

namespace zs {

  struct UniformLevelSet {
    static constexpr int dim = 3;
    using value_type = f32;
    using tiles_t = TileVector<value_type, 32>;
    using table_t = HashTable<i64, dim, i64>;

    ;

    table_t _table;
    tiles_t _tiles;
  };

  template <execspace_e, typename UniformLevelSetT, typename = void> struct UniformLevelSetProxy;

#if 0
  template <execspace_e Space, typename UniformLevelSetT>
  struct UniformLevelSetProxy<Space, UniformLevelSetT>
      : LevelSetInterface<UniformLevelSetProxy<Space, UniformLevelSetT>,
                          typename UniformLevelSetT::value_type, UniformLevelSetT::dim> {
    using table_t = typename UniformLevelSetT::table_t;
    using tiles_t = typename UniformLevelSetT::tiles_t;
    using T = typename UniformLevelSetT::value_type;
    static constexpr int dim = UniformLevelSetT::dim;
    using TV = vec<T, dim>;

    constexpr UniformLevelSetProxy() = default;
    ~UniformLevelSetProxy() = default;
    explicit UniformLevelSetProxy(const std::vector<SmallString> &tagNames, UniformLevelSetT &ls)
        : table{proxy<Space>(ls._table)}, tiles{proxy<Space>(tagNames, ls._tiles)} {}

    constexpr T getSignedDistance(const TV &X) const noexcept {}
    constexpr TV getNormal(const TV &X) const noexcept {}
    constexpr TV getMaterialVelocity(const TV &X) const noexcept {}
    constexpr decltype(auto) getBoundingBox() const noexcept {}

    HashTableProxy<Space, table_t> table;
    TileVectorProxy<Space, tiles_t> tiles;
  };

  template <execspace_e ExecSpace>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames, UniformLevelSet &levelset) {
    return UniformLevelSetProxy<ExecSpace, UniformLevelSet>{tagNames, levelset};
  }
#endif

}  // namespace zs