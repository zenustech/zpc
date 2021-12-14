#pragma once
#include <fstream>
#include <type_traits>

#include "AnalyticLevelSet.h"
#include "LevelSetInterface.h"
#include "SparseLevelSet.hpp"
// #include "zensim/execution/Concurrency.h"
#include "zensim/math/Vec.h"
#include "zensim/memory/Allocator.h"

namespace zs {

///
/// special purpose levelsets
///
#if 0
  template <typename T, int d> struct LevelSetRefs {
    using value_type = T;
    static constexpr int dim = d;
    using ls_t = variant<AnalyticLevelSet<analytic_geometry_e::Plane, value_type, dim> *,
                         AnalyticLevelSet<analytic_geometry_e::Cuboid, value_type, dim> *,
                         AnalyticLevelSet<analytic_geometry_e::Sphere, value_type, dim> *,
                         AnalyticLevelSet<analytic_geometry_e::Cylinder, value_type, dim> *,
                         SparseLevelSet<dim, grid_e::collocated> *>;

    ls_t _ls{};
  };
  template <typename T, int d> struct SdfVelField {
    using value_type = T;
    static constexpr int dim = d;
    using TV = vec<value_type, dim>;

    template <typename SdfField, typename VelField>
    constexpr SdfVelField(SdfField &sdf, VelField &vel) noexcept : _sdfPtr{&sdf}, _velPtr{&vel} {}

    LevelSetRefs<T, d> _sdfPtr{nullptr};
    LevelSetRefs<T, d> _velPtr{nullptr};
  };

  template <typename Ls> struct TransitionLevelSet {
    using ls_t = remove_cvref_t<Ls>;
    using value_type = typename ls_t::value_type;
    static constexpr int dim = ls_t::dim;
    using TV = vec<value_type, dim>;

    void setStepDt(const value_type dt) noexcept { _stepDt = dt; }
    void push(Ls *ls) noexcept {
      if (_lsPtrs[0] == nullptr)
        _lsPtrs[0] = ls;
      else if (_lsPtrs[1] == nullptr)
        _lsPtrs[1] = ls;
      else
        _lsPtrs[0] = _lsPtrs[1];
      _alpha = 0;
    }

    std::array<Ls *, 2> _lsPtrs{nullptr, nullptr};
    value_type _stepDt{0}, _alpha{0};
  };
#endif

  ///
  /// special purpose levelset views
  ///
  template <typename SdfLsView, typename VelLsView> struct SdfVelFieldView
      : LevelSetInterface<SdfVelFieldView<SdfLsView, VelLsView>> {
    static_assert(SdfLsView::dim == VelLsView::dim, "dimension mismatch!");
    static_assert(std::is_floating_point_v<
                      typename SdfLsView::
                          value_type> && std::is_floating_point_v<typename VelLsView::value_type>,
                  "levelset not in floating point type!");

    using value_type = typename SdfLsView::value_type;
    static constexpr int dim = SdfLsView::dim;
    using TV = vec<value_type, dim>;

    constexpr SdfVelFieldView(const SdfLsView &sdf, const VelLsView &vel) noexcept
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

  template <typename LsView> struct TransitionLevelSetView
      : LevelSetInterface<TransitionLevelSetView<LsView>> {
    static_assert(std::is_floating_point_v<typename LsView::T>,
                  "levelset not in floating point type!");

    using value_type = typename LsView::value_type;
    static constexpr int dim = LsView::dim;
    using TV = vec<value_type, dim>;

    constexpr TransitionLevelSetView(const LsView &lsvSrc, const LsView &lsvDst,
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
