#pragma once
#include "LevelSetInterface.h"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/TileVector.hpp"

namespace zs {

  template <int dim_ = 3> struct SparseLevelSet {
    static constexpr int dim = dim_;
    using value_type = f32;
    using tiles_t = TileVector<value_type, 32>;
    using table_t = HashTable<i64, dim, i64>;

    explicit SparseLevelSet(int sideLengthBits = 3, value_type dx = 1.f)
        : _sideLength{1 << sideLengthBits}, _space{1 << (sideLengthBits * dim)}, _dx{dx} {}

    SparseLevelSet clone(const MemoryHandle mh) const {
      SparseLevelSet ret{};
      ret._sideLength = _sideLength;
      ret._space = _space;
      ret._dx = _dx;
      ret._defaultValue = _defaultValue;
      ret._table = _table.clone(mh);
      ret._tiles = _tiles.clone(mh);
      return ret;
    }

    int _sideLength{8};  // tile side length
    int _space{512};     // voxels per tile
    value_type _dx;
    value_type _defaultValue;
    table_t _table;
    tiles_t _tiles;
  };

  template <execspace_e, typename SparseLevelSetT, typename = void> struct SparseLevelSetProxy;

#if 0
  template <execspace_e Space, typename SparseLevelSetT>
  struct SparseLevelSetProxy<Space, SparseLevelSetT>
      : LevelSetInterface<SparseLevelSetProxy<Space, SparseLevelSetT>,
                          typename SparseLevelSetT::value_type, SparseLevelSetT::dim> {
    using table_t = typename SparseLevelSetT::table_t;
    using tiles_t = typename SparseLevelSetT::tiles_t;
    using T = typename SparseLevelSetT::value_type;
    static constexpr int dim = SparseLevelSetT::dim;
    using TV = vec<T, dim>;

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

  template <execspace_e ExecSpace>
  decltype(auto) proxy(const std::vector<SmallString> &tagNames, SparseLevelSet &levelset) {
    return SparseLevelSetProxy<ExecSpace, SparseLevelSet>{tagNames, levelset};
  }
#endif

}  // namespace zs