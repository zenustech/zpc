#pragma once
#include <utility>

#include "Structure.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/LevelSetInterface.h"

namespace zs {

  template <int dim_ = 3, grid_e category_ = grid_e::collocated> struct SparseLevelSet {
    static constexpr int dim = dim_;
    static constexpr int side_length = 8;
    static constexpr auto category = category_;
    using value_type = f32;
    using index_type = i32;
    using TV = vec<value_type, dim>;
    using IV = vec<index_type, dim>;
    using Affine = vec<value_type, dim + 1, dim + 1>;
    using table_t = HashTable<index_type, dim, i64>;
    using grid_t = Grid<value_type, dim, side_length, category>;

    constexpr SparseLevelSet() = default;

    SparseLevelSet clone(const MemoryHandle mh) const {
      SparseLevelSet ret{};
      ret._sideLength = _sideLength;
      ret._space = _space;
      ret._dx = _dx;
      ret._backgroundValue = _backgroundValue;
      ret._backgroundVecValue = _backgroundVecValue;
      ret._table = _table.clone(mh);
      ret._grid = _grid.clone(mh);
      ret._min = _min;
      ret._max = _max;
      ret._w2v = _w2v;
      return ret;
    }

    int _sideLength{8};  // tile side length
    int _space{512};     // voxels per tile
    value_type _dx{1};
    value_type _backgroundValue{0};
    TV _backgroundVecValue{TV::zeros()};
    table_t _table{};
    grid_t _grid{};
    TV _min{}, _max{};
    Affine _w2v{};
  };

  using GeneralSparseLevelSet
      = variant<SparseLevelSet<3, grid_e::collocated>, SparseLevelSet<2, grid_e::collocated>>;

  template <execspace_e, typename SparseLevelSetT, typename = void> struct SparseLevelSetView;

  template <execspace_e Space, typename SparseLevelSetT>
  struct SparseLevelSetView<Space, SparseLevelSetT>
      : LevelSetInterface<SparseLevelSetView<Space, SparseLevelSetT>,
                          typename SparseLevelSetT::value_type, SparseLevelSetT::dim> {
    using table_t = typename SparseLevelSetT::table_t;
    using grid_t = typename SparseLevelSetT::grid_t;
    using grid_view_t =
        typename GridsView<Space,
                           typename grid_t::grids_t>::template Grid<SparseLevelSetT::category>;
    using T = typename SparseLevelSetT::value_type;
    using Ti = typename SparseLevelSetT::index_type;
    using TV = typename SparseLevelSetT::TV;
    using IV = typename SparseLevelSetT::IV;
    using Affine = typename SparseLevelSetT::Affine;
    static constexpr int dim = SparseLevelSetT::dim;

    template <typename Val, std::size_t... Is>
    static constexpr auto arena_type_impl(index_seq<Is...>) {
      return vec<Val, (Is + 1 > 0 ? 2 : 2)...>{};
    }
    template <typename Val, int d> static constexpr auto arena_type() {
      return arena_type_impl<Val>(std::make_index_sequence<d>{});
    }

    template <typename Val> using Arena = decltype(arena_type<Val, dim>());

    constexpr SparseLevelSetView() = default;
    ~SparseLevelSetView() = default;
    constexpr SparseLevelSetView(const std::vector<SmallString> &tagNames, SparseLevelSetT &ls)
        : _sideLength{ls._sideLength},
          _space{ls._space},
          _dx{ls._dx},
          _backgroundValue{ls._backgroundValue},
          _backgroundVecValue{ls._backgroundVecValue},
          table{proxy<Space>(ls._table)},
          _grid{proxy<Space>(ls._grid)},
          _min{ls._min},
          _max{ls._max},
          _w2v{ls._w2v} {}
    constexpr SparseLevelSetView(SparseLevelSetT &ls)
        : _sideLength{ls._sideLength},
          _space{ls._space},
          _dx{ls._dx},
          _backgroundValue{ls._backgroundValue},
          _backgroundVecValue{ls._backgroundVecValue},
          table{proxy<Space>(ls._table)},
          _grid{proxy<Space>(ls._grid)},
          _min{ls._min},
          _max{ls._max},
          _w2v{ls._w2v} {}

    constexpr T getSignedDistance(const TV &x) const noexcept {
      /// world to local
      auto arena = Arena<T>::uniform(_backgroundValue);
      IV loc{};
      TV X = x * _w2v;
      // TV X = x / _dx;
      for (int d = 0; d < dim; ++d) loc(d) = gcem::floor(X(d));
      // for (int d = 0; d < dim; ++d) loc(d) = gcem::floor((X(d) - _min(d)) / _dx);
      TV diff = X - loc;
      if constexpr (dim == 2) {
        for (auto &&[dx, dy] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            arena(dx, dy) = _grid("sdf", blockno, coord - blockid * _sideLength);
          }
        }
      } else if constexpr (dim == 3) {
        for (auto &&[dx, dy, dz] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy, loc(2) + dz};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            arena(dx, dy, dz) = _grid("sdf", blockno, coord - blockid * _sideLength);
          }
        }
      }
      return trilinear_interop<0>(diff, arena);
    }
    constexpr TV getNormal(const TV &X) const noexcept {
      TV diff{}, v1{}, v2{};
      T eps = (T)1e-6;
      /// compute a local partial derivative
      for (int i = 0; i < dim; i++) {
        v1 = X;
        v2 = X;
        v1(i) = X(i) + eps;
        v2(i) = X(i) - eps;
        diff(i) = (getSignedDistance(v1) - getSignedDistance(v2)) / (eps + eps);
      }
      return diff.normalized();
    }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept {
      if (!_grid.hasProperty("vel")) return TV::zeros();
      /// world to local
      auto arena = Arena<TV>::uniform(_backgroundVecValue);
      IV loc{};
      TV X = x * _w2v;
      // TV X = x / _dx;
      for (int d = 0; d < dim; ++d) loc(d) = gcem::floor(X(d));
      // for (int d = 0; d < dim; ++d) loc(d) = gcem::floor((X(d) - _min(d)) / _dx);
      TV diff = X - loc;
      if constexpr (dim == 2) {
        for (auto &&[dx, dy] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            arena(dx, dy) = _grid.template pack<dim>("vel", blockno, coord - blockid * _sideLength);
          }
        }
      } else if constexpr (dim == 3) {
        for (auto &&[dx, dy, dz] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy, loc(2) + dz};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            arena(dx, dy, dz)
                = _grid.template pack<dim>("vel", blockno, coord - blockid * _sideLength);
          }
        }
      }
      return trilinear_interop<0>(diff, arena);
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return std::make_tuple(_min, _max); }

  protected:
    template <std::size_t d, typename Field, enable_if_t<(d == dim - 1)> = 0>
    constexpr auto trilinear_interop(const TV &diff, const Field &arena) const noexcept {
      return linear_interop(diff(d), arena(0), arena(1));
    }
    template <std::size_t d, typename Field, enable_if_t<(d != dim - 1)> = 0>
    constexpr auto trilinear_interop(const TV &diff, const Field &arena) const noexcept {
      return linear_interop(diff(d), trilinear_interop<d + 1>(diff, arena[0]),
                            trilinear_interop<d + 1>(diff, arena[1]));
    }

    int _sideLength;  // tile side length
    int _space;       // voxels per tile
    T _dx;
    T _backgroundValue;
    TV _backgroundVecValue;
    HashTableView<Space, table_t> table;
    grid_view_t _grid;
    TV _min, _max;
    Affine _w2v;
  };

  template <execspace_e ExecSpace, int dim>
  constexpr decltype(auto) proxy(const std::vector<SmallString> &tagNames,
                                 SparseLevelSet<dim> &levelset) {
    return SparseLevelSetView<ExecSpace, SparseLevelSet<dim>>{tagNames, levelset};
  }

  template <execspace_e ExecSpace, int dim>
  constexpr decltype(auto) proxy(SparseLevelSet<dim> &levelset) {
    return SparseLevelSetView<ExecSpace, SparseLevelSet<dim>>{levelset};
  }

}  // namespace zs