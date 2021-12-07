#pragma once
#include <utility>

#include "Structure.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/geometry/LevelSetInterface.h"
#include "zensim/math/Vec.h"
#include "zensim/math/matrix/Transform.hpp"
#include "zensim/tpls/fmt/color.h"

namespace zs {

  template <int dim_ = 3, grid_e category_ = grid_e::collocated> struct SparseLevelSet {
    static constexpr int dim = dim_;
    static constexpr int side_length = 8;
    static constexpr auto category = category_;
    using value_type = f32;
    using index_type = i32;
    using IV = vec<index_type, dim>;
    using TV = vec<value_type, dim>;
    using TM = vec<value_type, dim, dim>;
    using Affine = vec<value_type, dim + 1, dim + 1>;
    using table_t = HashTable<index_type, dim, i64>;
    using grid_t = Grid<value_type, dim, side_length, category>;
    using size_type = typename grid_t::size_type;

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
      ret._i2wSinv = _i2wSinv;
      ret._i2wRinv = _i2wRinv;
      ret._i2wT = _i2wT;
      ret._i2wShat = _i2wShat;
      ret._i2wRhat = _i2wRhat;
      return ret;
    }

    template <typename VecTM,
              enable_if_all<VecTM::dim == 2, VecTM::template get_range<0>() == dim + 1,
                            VecTM::template get_range<1>() == dim + 1,
                            std::is_floating_point_v<typename VecTM::value_type>> = 0>
    void resetTransformation(const VecInterface<VecTM> &i2w) {
      math::decompose_transform(i2w, _i2wSinv, _i2wRinv, _i2wT, 0);
      _i2wSinv = inverse(_i2wSinv);
      _i2wRinv = _i2wRinv.transpose();  // equal to inverse
      _i2wShat = TM::identity();
      _i2wRhat = TM::identity();
    }
    auto getIndexToWorldTransformation() const {
      Affine ret{Affine::identity()};
      {
        auto S = inverse(_i2wSinv);
        for (int i = 0; i != dim; ++i)
          for (int j = 0; j != dim; ++j) ret(i, j) = S(i, j);
      }
      {
        Affine tmp{Affine::identity()};
        auto R = _i2wRinv.transpose();
        for (int i = 0; i != dim; ++i)
          for (int j = 0; j != dim; ++j) tmp(i, j) = R(i, j);
        ret = ret * tmp;
      }
      {
        Affine tmp{Affine::identity()};
        for (int j = 0; j != dim; ++j) tmp(dim, j) = _i2wT[j];
        ret = ret * tmp;
      }
      return ret;
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    void translate(const VecInterface<VecT> &t) noexcept {
      _i2wT += t;
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template get_range<0>() == dim,
                                           VecT::template get_range<1>() == dim> = 0>
    void rotate(const VecInterface<VecT> &r) noexcept {
      _i2wRhat = _i2wRhat * r;
      _i2wRinv = r.transpose() * _i2wRinv;
    }
    template <typename VecT, enable_if_all<VecT::dim == 2, VecT::template get_range<0>() == dim,
                                           VecT::template get_range<1>() == dim> = 0>
    void scale(const VecInterface<VecT> &s) {
      _i2wShat = _i2wShat * s;
      _i2wSinv = inverse(s) * _i2wSinv;
    }
    void scale(const value_type s) { scale(s * TM::identity()); }

    int _sideLength{8};  // tile side length
    int _space{512};     // voxels per tile
    value_type _dx{1};
    value_type _backgroundValue{0};
    TV _backgroundVecValue{TV::zeros()};
    table_t _table{};
    grid_t _grid{};
    TV _min{}, _max{};
    // initial index-to-world affine transformation
    TM _i2wSinv{TM::identity()}, _i2wRinv{TM::identity()};
    TV _i2wT{TV::zeros()};
    // additional
    TM _i2wShat{TM::identity()}, _i2wRhat{TM::identity()};
  };

  using GeneralSparseLevelSet
      = variant<SparseLevelSet<3, grid_e::collocated>, SparseLevelSet<2, grid_e::collocated>>;

  template <execspace_e, typename SparseLevelSetT, typename = void> struct SparseLevelSetView;

  template <execspace_e Space, typename SparseLevelSetT>
  struct SparseLevelSetView<Space, SparseLevelSetT>
      : LevelSetInterface<SparseLevelSetView<Space, SparseLevelSetT>,
                          typename SparseLevelSetT::value_type, SparseLevelSetT::dim> {
    using value_type = typename SparseLevelSetT::value_type;
    using size_type = typename SparseLevelSetT::size_type;
    using table_t = typename SparseLevelSetT::table_t;
    using grid_t = typename SparseLevelSetT::grid_t;
    using grid_view_t = RM_CVREF_T(proxy<Space>({}, std::declval<grid_t &>()));
    using T = typename SparseLevelSetT::value_type;
    using Ti = typename SparseLevelSetT::index_type;
    using TV = typename SparseLevelSetT::TV;
    using TM = typename SparseLevelSetT::TM;
    using IV = typename SparseLevelSetT::IV;
    using Affine = typename SparseLevelSetT::Affine;
    static constexpr int dim = SparseLevelSetT::dim;
    static constexpr auto side_length = SparseLevelSetT::side_length;

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
    constexpr SparseLevelSetView(SparseLevelSetT &ls)
        : _sideLength{ls._sideLength},
          _space{ls._space},
          _dx{ls._dx},
          _backgroundValue{ls._backgroundValue},
          _backgroundVecValue{ls._backgroundVecValue},
          _table{proxy<Space>(ls._table)},
          _grid{proxy<Space>(ls._grid)},
          _min{ls._min},
          _max{ls._max},
          _i2wT{ls._i2wT},
          _i2wRinv{ls._i2wRinv},
          _i2wSinv{ls._i2wSinv},
          _i2wRhat{ls._i2wRhat},
          _i2wShat{ls._i2wShat} {}

    void print() {
      if constexpr (dim == 2) {
        auto blockCnt = *_table._cnt;
        using Ti = RM_CVREF_T(blockCnt);
        using IV = vec<Ti, dim>;
        for (Ti bno = 0; bno != blockCnt; ++bno) {
          auto blockCoord = _table._activeKeys[bno];
          fmt::print(fg(fmt::color::orange), "\nblock [{}] ({}, {})\n", bno, blockCoord[0],
                     blockCoord[1]);
          if (blockCoord[0] >= 0 || blockCoord[1] >= 0) {
            fmt::print("skip\n");
            continue;
          }

          auto block = _grid.block(bno);
          for (Ti cno = 0, ed = grid_t::block_space(); cno != ed; ++cno) {
            IV cellCoord{cno % side_length, side_length - 1 - cno / side_length};
            auto val = block("sdf", cellCoord);
            auto tag = block("tag", cellCoord);
            auto mask = block("mask", cellCoord);
            auto tagmask = block("tagmask", cellCoord);
            auto c = fg(fmt::color::white);
            if (mask == 0 || tagmask == 0) {
            } else {
              if (tag > 1)
                c = fg(fmt::color::dark_olive_green);
              else if (tag > 0)
                c = fg(fmt::color::light_sea_green);
              else
                c = fg(fmt::color::yellow_green);
            }
            // if (tag < 0) fmt::print("WTF at {} ({}, {})??\n", (int)bno, (int)cellCoord[0],
            // (int)cellCoord[1]);
            auto candi = fmt::format("{:.4f}", val);
            auto candi1 = fmt::format("{}", tag);
            fmt::print(c, "[{}{}({})] ", val < 0 ? "" : " ", mask ? candi : "------",
                       tagmask ? candi1 : " ");
            if (cno % side_length == side_length - 1) fmt::print("\n");
          }
          fmt::print("\n");
        }
      }
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto getReferencePosition(const VecInterface<VecT> &x) const noexcept {
      // world-to-view: minus trans, div rotation, div scale
      return (x - _i2wT) * _i2wRinv * _i2wSinv;
    }
    constexpr T getSignedDistance(const TV &x) const noexcept {
      if (!_grid.hasProperty("sdf")) return limits<T>::max();
      /// world to local
      auto arena = Arena<T>::uniform(_backgroundValue);
      IV loc{};
      TV X = getReferencePosition(x);
      for (int d = 0; d < dim; ++d) loc(d) = zs::floor(X(d));
      TV diff = X - loc;
      if constexpr (dim == 2) {
        for (auto &&[dx, dy] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = _table.query(blockid);
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
          auto blockno = _table.query(blockid);
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
      TV X = getReferencePosition(x);
      for (int d = 0; d < dim; ++d) loc(d) = zs::floor(X(d));
      TV diff = X - loc;
      if constexpr (dim == 2) {
        for (auto &&[dx, dy] : ndrange<dim>(2)) {
          IV coord{loc(0) + dx, loc(1) + dy};
          auto blockid = coord;
          for (int d = 0; d < dim; ++d) blockid[d] += (coord[d] < 0 ? -_sideLength + 1 : 0);
          blockid = blockid / _sideLength;
          auto blockno = _table.query(blockid);
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
          auto blockno = _table.query(blockid);
          if (blockno != table_t::sentinel_v) {
            arena(dx, dy, dz)
                = _grid.template pack<dim>("vel", blockno, coord - blockid * _sideLength);
          }
        }
      }
      return trilinear_interop<0>(diff, arena) * _i2wShat * _i2wRhat;
    }
    constexpr decltype(auto) getBoundingBox() const noexcept { return std::make_tuple(_min, _max); }

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
    HashTableView<Space, table_t> _table;
    grid_view_t _grid;
    TV _min, _max;

    TV _i2wT;
    TM _i2wRinv, _i2wSinv;
    TM _i2wRhat, _i2wShat;
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