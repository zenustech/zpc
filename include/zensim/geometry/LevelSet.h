#pragma once
#include <fstream>
#include <type_traits>

#include "LevelSetInterface.h"
#include "VdbLevelSet.h"
// #include "zensim/execution/Concurrency.h"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  /***************************************************************************/
  /****************************** height field *******************************/
  /***************************************************************************/

  template <typename DataType, typename Tn_ = int> struct HeightField
      : LevelSetInterface<HeightField<DataType, Tn_>, DataType, 3> {
    using T = DataType;
    using Tn = Tn_;
    static constexpr int dim = 3;
    using TV = vec<T, dim>;
    using IV = vec<Tn, dim>;

    using Arena = vec<T, 2, 2, 2>;
    // using VecArena = vec<TV, 2, 2, 2>;

    template <typename Allocator>
    HeightField(Allocator &&allocator, T dx, Tn const &x, Tn const &y, Tn const &z)
        : _dx{dx}, _extent{x, y, z}, _min{TV::zeros()} {
      _field = (T *)allocator.allocate((std::size_t)x * z * sizeof(T));
    }

    constexpr void setDx(T dx) noexcept { _dx = dx; }
    constexpr void setOffset(T dx, T dy, T dz) noexcept { _min = TV{dx, dy, dz}; }

    template <typename Allocator = heap_allocator>
    void constructFromTxtFile(const std::string &fn, Allocator &&allocator = heap_allocator{}) {
      std::ifstream is(fn, std::ios::in);
      if (!is.is_open()) {
        printf("%s not found!\n", fn.c_str());
        return;
      }
      for (int x = 0; x < _extent(0); ++x)
        for (int z = 0; z < _extent(2); ++z) is >> entry(x, z);
      is.close();
    }

    constexpr auto &entry(Tn const &x, Tn const &z) noexcept { return _field[x * _extent(2) + z]; }
    constexpr auto const &entry(Tn const &x, Tn const &z) const noexcept {
      return _field[x * _extent(2) + z];
    }
    constexpr bool inside(const IV &X) const noexcept {
      if (X(0) >= _extent(0) || X(1) >= _extent(1) || X(2) >= _extent(2)) return false;
      return true;
    }
    template <std::size_t d, typename Field>
    constexpr auto trilinear_interop(const TV &diff, const Field &arena) const noexcept {
      if constexpr (d == dim - 1) {
        return linear_interop(diff(d), arena(0), arena(1));
      } else
        return linear_interop(diff(d), trilinear_interop<d + 1>(diff, arena[0]),
                              trilinear_interop<d + 1>(diff, arena[1]));
      return (T)1;
    }

    constexpr T getSignedDistance(const TV &x) const noexcept {
      /// world to local
      Arena arena{};
      IV loc{};
      for (int d = 0; d < dim; ++d) loc(d) = zs::floor((x(d) - _min(d)) / _dx);
      TV diff = (x - _min) / _dx - loc;
      {
        for (Tn dx = 0; dx < 2; dx++)
          for (Tn dy = 0; dy < 2; dy++)
            for (Tn dz = 0; dz < 2; dz++) {
              if (inside(IV{loc(0) + dx, loc(1) + dy, loc(2) + dz})) {
                T h = entry(loc(0) + dx, loc(2) + dz);
                arena(dx, dy, dz) = (loc(1) + dy) * _dx - h;
              } else
                arena(dx, dy, dz) = 2 * _dx;
            }
      }
      return trilinear_interop<0>(diff, arena);
    }
    constexpr TV getNormal(const TV &x) const noexcept { return TV{0, 1, 0}; }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept { return TV::zeros(); }
    constexpr decltype(auto) getBoundingBox() const noexcept {
      return std::make_tuple(_min, _min + _extent * _dx);
    }

    IV _extent;
    T _dx;

  private:
    T *_field;
    TV _min;
  };

  ///
  /// special purpose levelset views
  ///
  template <typename SdfLsView, typename VelLsView> struct SdfVelField
      : LevelSetInterface<SdfVelField<SdfLsView, VelLsView>, typename SdfLsView::T,
                          SdfLsView::dim> {
    static_assert(SdfLsView::dim == VelLsView::dim, "dimension mismatch!");
    static_assert(std::is_floating_point_v<
                      typename SdfLsView::T> && std::is_floating_point_v<typename VelLsView::T>,
                  "levelset not in floating point type!");

    using T = typename SdfLsView::T;
    static constexpr int dim = SdfLsView::dim;
    using TV = vec<T, dim>;

    constexpr SdfVelField(const SdfLsView &sdf, const VelLsView &vel) noexcept
        : _sdf(sdf), _vel(vel) {}

    /// bounding volume interface
    constexpr std::tuple<TV, TV> do_getBoundingBox() const noexcept {
      return _sdf.getBoundingBox();
    }
    constexpr TV do_getBoxCenter() const noexcept { return _sdf.getBoxCenter(); }
    constexpr TV do_getBoxSideLengths() const noexcept { return _sdf.getBoxSideLengths(); }
    constexpr TV do_getUniformCoord(const TV &pos) const noexcept {
      return _sdf.getUniformCoord(pos);
    }
    /// levelset interface
    constexpr T getSignedDistance(const TV &x) const noexcept { return _sdf.getSignedDistance(x); }
    constexpr TV getNormal(const TV &x) const noexcept { return _sdf.getNormal(x); }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept {
      return _vel.getMaterialVelocity(x);  // this is special
    }

    SdfLsView _sdf;
    VelLsView _vel;
  };

  template <typename LsView> struct TransitionLevelSet
      : LevelSetInterface<TransitionLevelSet<LsView>, typename LsView::T, LsView::dim> {
    static_assert(std::is_floating_point_v<typename LsView::T>,
                  "levelset not in floating point type!");

    using T = typename LsView::T;
    static constexpr int dim = LsView::dim;
    using TV = vec<T, dim>;

    constexpr TransitionLevelSet(const LsView &lsvSrc, const LsView &lsvDst, const T stepDt,
                                 const T alpha = (T)0) noexcept
        : _lsvSrc{lsvSrc}, _lsvDst{lsvDst}, _stepDt{stepDt}, _alpha{alpha} {}

    /// bounding volume interface
    constexpr std::tuple<TV, TV> do_getBoundingBox() const noexcept {
      return _lsvSrc.getBoundingBox();
    }
    constexpr TV do_getBoxCenter() const noexcept { return _lsvSrc.getBoxCenter(); }
    constexpr TV do_getBoxSideLengths() const noexcept { return _lsvSrc.getBoxSideLengths(); }
    constexpr TV do_getUniformCoord(const TV &pos) const noexcept {
      return _lsvSrc.getUniformCoord(pos);
    }
    /// levelset interface
    constexpr T getSignedDistance(const TV &x) const noexcept {
      TV v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) / 2;
      TV x0 = x - _alpha * _stepDt * v, x1 = x + (1 - _alpha) * _stepDt * v;
      return ((T)1 - _alpha) * _lsvSrc.getSignedDistance(x0)
             + _alpha * _lsvDst.getSignedDistance(x1);
    }
    constexpr TV getNormal(const TV &x) const noexcept { return _lsvSrc.getNormal(x); }
    constexpr TV getMaterialVelocity(const TV &x) const noexcept {
      TV v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) / 2;
      TV x0 = x - _alpha * _stepDt * v, x1 = x + (1 - _alpha) * _stepDt * v;
      return ((T)1 - _alpha) * _lsvSrc.getMaterialVelocity(x0)
             + _alpha * _lsvDst.getMaterialVelocity(x1);
    }

    LsView _lsvSrc, _lsvDst;
    T _stepDt, _alpha;
  };

}  // namespace zs
