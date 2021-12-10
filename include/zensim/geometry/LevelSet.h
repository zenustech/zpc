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

  ///
  /// special purpose levelsets
  ///
  // WIP

  ///
  /// special purpose levelset views
  ///
  template <typename SdfLsView, typename VelLsView> struct SdfVelField
      : LevelSetInterface<SdfVelField<SdfLsView, VelLsView>> {
    static_assert(SdfLsView::dim == VelLsView::dim, "dimension mismatch!");
    static_assert(std::is_floating_point_v<
                      typename SdfLsView::T> && std::is_floating_point_v<typename VelLsView::T>,
                  "levelset not in floating point type!");

    using value_type = typename SdfLsView::value_type;
    static constexpr int dim = SdfLsView::dim;
    using TV = vec<value_type, dim>;

    constexpr SdfVelField(const SdfLsView &sdf, const VelLsView &vel) noexcept
        : _sdf(sdf), _vel(vel) {}

    /// bounding volume interface
    constexpr std::tuple<TV, TV> do_getBoundingBox() const noexcept {
      return _sdf.getBoundingBox();
    }
    constexpr TV do_getBoxCenter() const noexcept { return _sdf.getBoxCenter(); }
    constexpr TV do_getBoxSideLengths() const noexcept { return _sdf.getBoxSideLengths(); }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getUniformCoord(const VecInterface<VecT> &x) const noexcept {
      return _sdf.getUniformCoord(x);
    }
    /// levelset interface
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return _sdf.getSignedDistance(x);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getNormal(const VecInterface<VecT> &x) const noexcept {
      return _sdf.getNormal(x);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      return _vel.getMaterialVelocity(x);  // this is the motive
    }

    SdfLsView _sdf;
    VelLsView _vel;
  };

  template <typename LsView> struct TransitionLevelSet
      : LevelSetInterface<TransitionLevelSet<LsView>> {
    static_assert(std::is_floating_point_v<typename LsView::T>,
                  "levelset not in floating point type!");

    using value_type = typename LsView::value_type;
    static constexpr int dim = LsView::dim;
    using TV = vec<value_type, dim>;

    constexpr TransitionLevelSet(const LsView &lsvSrc, const LsView &lsvDst,
                                 const value_type stepDt,
                                 const value_type alpha = (value_type)0) noexcept
        : _lsvSrc{lsvSrc}, _lsvDst{lsvDst}, _stepDt{stepDt}, _alpha{alpha} {}

    /// bounding volume interface
    constexpr std::tuple<TV, TV> do_getBoundingBox() const noexcept {
      return _lsvSrc.getBoundingBox();
    }
    constexpr TV do_getBoxCenter() const noexcept { return _lsvSrc.getBoxCenter(); }
    constexpr TV do_getBoxSideLengths() const noexcept { return _lsvSrc.getBoxSideLengths(); }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getUniformCoord(const VecInterface<VecT> &pos) const noexcept {
      return _lsvSrc.getUniformCoord(pos);
    }
    /// levelset interface
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr value_type do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      TV v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) / (value_type)2;
      TV x0 = x - _alpha * _stepDt * v, x1 = x + ((value_type)1 - _alpha) * _stepDt * v;
      return ((value_type)1 - _alpha) * _lsvSrc.getSignedDistance(x0)
             + _alpha * _lsvDst.getSignedDistance(x1);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getNormal(const VecInterface<VecT> &x) const noexcept {
      TV v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) / (value_type)2;
      TV x0 = x - _alpha * _stepDt * v, x1 = x + ((value_type)1 - _alpha) * _stepDt * v;
      return ((value_type)1 - _alpha) * _lsvSrc.getNormal(x0) + _alpha * _lsvDst.getNormal(x1);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr TV do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      TV v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) / (value_type)2;
      TV x0 = x - _alpha * _stepDt * v, x1 = x + ((value_type)1 - _alpha) * _stepDt * v;
      return ((value_type)1 - _alpha) * _lsvSrc.getMaterialVelocity(x0)
             + _alpha * _lsvDst.getMaterialVelocity(x1);
    }

    LsView _lsvSrc, _lsvDst;
    value_type _stepDt, _alpha;
  };

}  // namespace zs
