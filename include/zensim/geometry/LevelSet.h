#pragma once
#include <deque>
#include <fstream>
#include <type_traits>

#include "AdaptiveGrid.hpp"
#include "AnalyticLevelSet.h"
#include "LevelSetInterface.h"
#include "SparseGrid.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  template <execspace_e space, typename LsT>
  constexpr auto get_level_set_view(const std::shared_ptr<LsT> lsPtr) noexcept {
    using ls_t = remove_cvref_t<LsT>;
    if constexpr (is_spg_v<ls_t>)      // SparseGrid<ls_t::dim, ...>
      return proxy<space>(*lsPtr);     // const & non-const view
    else if constexpr (is_ag_v<ls_t>)  // VdbGrid<ls_t::dim, ...>
      return proxy<space>(*lsPtr);     // const & non-const view
    else
      return ls_t{*lsPtr};
  }

  ///
  /// special purpose levelsets
  ///
  namespace detail {
    template <execspace_e space> struct ls_view_helper {
      template <typename LsSharedPtr> constexpr auto operator()(LsSharedPtr) noexcept {
        return decltype(get_level_set_view<space>(declval<LsSharedPtr>())){};
      }
    };
  }  // namespace detail
  template <typename T, int d> struct DummyLevelSet
      : public LevelSetInterface<DummyLevelSet<T, d>> {
    using value_type = T;
    static constexpr int dim = d;
  };
  template <typename T, int d> struct UniformVelocityLevelSet
      : public LevelSetInterface<UniformVelocityLevelSet<T, d>> {
    using value_type = T;
    static constexpr int dim = d;
    constexpr UniformVelocityLevelSet() noexcept : vel{vec<T, d>::zeros()} {}
    template <typename VecT> constexpr UniformVelocityLevelSet(const VecInterface<VecT> &v) noexcept
        : vel{} {
      vel = v;
    }
    template <typename V, enable_if_t<is_floating_point_v<V>> = 0>
    constexpr UniformVelocityLevelSet(V v) noexcept : vel{vec<T, d>::constant(v)} {}

    constexpr auto do_getMaterialVelocity(...) const noexcept { return vel; }

    vec<T, d> vel;
  };

  template <typename T, int d> struct BasicLevelSet {
    using value_type = T;
    static constexpr int dim = d;
    using dummy_ls_t = DummyLevelSet<T, d>;
    using uniform_vel_ls_t = UniformVelocityLevelSet<T, d>;
    using spls_t = SparseGrid<3, T, 8>;                      // 8x8x8
    using spvdb_t = VdbGrid<3, T, index_sequence<3, 4, 5>>;  // FloatGrid alike
    template <analytic_geometry_e type = analytic_geometry_e::Plane> using analytic_ls_t
        = AnalyticLevelSet<type, value_type, dim>;
    /// raw levelset type list
    using raw_sdf_ls_tl = type_seq<spls_t, spvdb_t
#if 0
                                   ,
                                   analytic_ls_t<analytic_geometry_e::Cuboid>,
                                   analytic_ls_t<analytic_geometry_e::Sphere>,
                                   analytic_ls_t<analytic_geometry_e::Cylinder>
#endif
                                   >;
    using raw_vel_ls_tl = type_seq<dummy_ls_t, spls_t, spvdb_t, uniform_vel_ls_t>;
    // should automatically compute from the above two typelists
    using raw_ls_tl = type_seq<dummy_ls_t, spls_t, spvdb_t,
                               // analytic_ls_t<analytic_geometry_e::Plane>,
                               uniform_vel_ls_t>;
    /// shared_ptr of const raw levelsets
    using basic_ls_ptr_t = assemble_t<variant, map_t<std::shared_ptr, raw_ls_tl>>;
    using const_basic_ls_ptr_t
        = assemble_t<variant, map_t<std::shared_ptr, map_t<add_const_t, raw_ls_tl>>>;

    using const_sdf_ls_ptr_t
        = assemble_t<variant, map_t<std::shared_ptr, map_t<add_const_t, raw_sdf_ls_tl>>>;
    using const_vel_ls_ptr_t
        = assemble_t<variant, map_t<std::shared_ptr, map_t<add_const_t, raw_vel_ls_tl>>>;

    constexpr BasicLevelSet() noexcept = default;

    template <typename Ls,
              enable_if_t<raw_ls_tl::template occurencies_t<remove_cvref_t<Ls>>::value == 1> = 0>
    BasicLevelSet(Ls &&ls) : _ls{std::make_shared<remove_cvref_t<Ls>>(FWD(ls))} {}
    template <typename Ls,
              enable_if_t<raw_ls_tl::template occurencies_t<remove_cvref_t<Ls>>::value == 1> = 0>
    BasicLevelSet(const std::shared_ptr<Ls> &ls) : _ls{ls} {}

#if 0
    BasicLevelSet(const BasicLevelSet &ls) : _ls{} {
      match([this](const auto &lsPtr) {
        using LsT = typename RM_REF_T(lsPtr)::element_type;
        static_assert(std::is_copy_constructible_v<LsT>,
                      "the levelset should be copy constructible");
        _ls = std::make_shared<LsT>(*lsPtr);
      })(ls._ls);
    }
    BasicLevelSet &operator=(const BasicLevelSet &ls) {
      if (this != &ls) return *this;
      BasicLevelSet tmp(ls);
      std::swap(_ls, tmp._ls);
      return *this;
    }
#endif

    template <typename LsT> bool holdsLevelSet() const noexcept {
      return std::holds_alternative<std::shared_ptr<LsT>>(_ls);
    }
    template <typename LsT> decltype(auto) getLevelSet() const noexcept {
      return *std::get<std::shared_ptr<LsT>>(_ls);
    }
    template <typename LsT> decltype(auto) getLevelSet() noexcept {
      return *std::get<std::shared_ptr<LsT>>(_ls);
    }

    basic_ls_ptr_t _ls{};
  };

  template <typename T, int d> struct ConstSdfVelFieldPtr {
    using value_type = T;
    static constexpr int dim = d;
    using basic_level_set_t = BasicLevelSet<value_type, dim>;

    using dummy_ls_t = typename basic_level_set_t::dummy_ls_t;
    using uniform_vel_ls_t = typename basic_level_set_t::uniform_vel_ls_t;
    using spls_t = typename basic_level_set_t::spls_t;

    using const_sdf_ls_ptr_t = typename basic_level_set_t::const_sdf_ls_ptr_t;
    using const_vel_ls_ptr_t = typename basic_level_set_t::const_vel_ls_ptr_t;

    template <typename TSeq> using tseq_to_tuple = assemble_t<zs::tuple, TSeq>;
    template <execspace_e space> using const_sdf_ls_view_tl
        = map_op_t<detail::ls_view_helper<space>, get_ttal_t<const_sdf_ls_ptr_t>>;
    template <execspace_e space> using const_vel_ls_view_tl
        = map_op_t<detail::ls_view_helper<space>, get_ttal_t<const_vel_ls_ptr_t>>;

    // for levelset sequence view
    template <execspace_e space> using field_view_tl
        = map_t<tseq_to_tuple, compose_t<const_sdf_ls_view_tl<space>, const_vel_ls_view_tl<space>>>;
    template <execspace_e space> using const_field_view_t
        = assemble_t<variant, field_view_tl<space>>;

    template <typename TT> using duplicate_t = zs::tuple<TT, TT>;
    template <execspace_e space> using const_field_seq_view_t
        = assemble_t<variant, map_t<duplicate_t, field_view_tl<space>>>;

    /// ctor
    template <typename SdfField = spls_t, typename VelField = DummyLevelSet<T, d>>
    constexpr ConstSdfVelFieldPtr(std::shared_ptr<const SdfField> sdf = {},
                                  std::shared_ptr<const VelField> vel = {}) noexcept
        : _sdfConstPtr{sdf}, _velConstPtr{vel} {}
    template <typename SdfField = spls_t, typename VelField = DummyLevelSet<T, d>>
    constexpr ConstSdfVelFieldPtr(const SdfField *sdf, const VelField *vel = nullptr) noexcept
        : _sdfConstPtr{std::shared_ptr(sdf, [](...) {})},
          _velConstPtr{std::shared_ptr(vel, [](...) {})} {}

    constexpr ConstSdfVelFieldPtr(const basic_level_set_t &sdf,
                                  const basic_level_set_t &vel = dummy_ls_t{}) noexcept
        : _sdfConstPtr{}, _velConstPtr{} {
      match(
          [this](const auto &sdfPtr, const auto &velPtr) -> decltype(_sdfConstPtr = sdfPtr,
                                                                     _velConstPtr = velPtr,
                                                                     void()) {
            _sdfConstPtr = sdfPtr;
            _velConstPtr = velPtr;
          },
          [this](...) { throw std::runtime_error("sdf-velocity field construction failed."); })(
          sdf._ls, vel._ls);
    }

    /// view
    template <execspace_e space = execspace_e::host>
    constexpr const_field_view_t<space> getView(wrapv<space> = {}) const noexcept {
      return match([](const auto &sdfPtr,
                      const auto &velPtr) noexcept -> const_field_view_t<space> {
        return zs::make_tuple(get_level_set_view<space>(sdfPtr), get_level_set_view<space>(velPtr));
      })(_sdfConstPtr, _velConstPtr);
    }
    const_sdf_ls_ptr_t _sdfConstPtr{};
    const_vel_ls_ptr_t _velConstPtr{};
  };

  template <typename T, int d> struct ConstTransitionLevelSetPtr {
    using value_type = T;
    static constexpr int dim = d;
    using const_sdf_vel_ls_t = ConstSdfVelFieldPtr<value_type, dim>;
    template <execspace_e space> using const_field_view_t =
        typename const_sdf_vel_ls_t::template const_field_view_t<space>;
    template <execspace_e space> using const_field_seq_view_t =
        typename const_sdf_vel_ls_t::template const_field_seq_view_t<space>;

    void setStepDt(const value_type dt) noexcept { _stepDt = dt; }
    void advance(const value_type ratio) noexcept {
      _alpha += ratio;
      constexpr auto threshold
          = (value_type)1 - (value_type)128 * detail::deduce_numeric_epsilon<value_type>();
      while (_alpha > threshold) {
        _alpha -= (value_type)1;
        if (_fields.size()) pop();
      }
    }
    void push(const const_sdf_vel_ls_t ls) { _fields.push_back(ls); }
    void pop() { _fields.pop_front(); }

    /// view
    template <execspace_e space = execspace_e::host>
    constexpr const_field_seq_view_t<space> getView(wrapv<space> = {}) const {
      const_field_seq_view_t<space> ret{};
      const_field_view_t<space> ls0{}, ls1{};
      if (_fields.size() > 0) {
        ls0 = _fields[0].template getView<space>();
        if (_fields.size() > 1)
          ls1 = _fields[1].template getView<space>();
        else
          ls1 = ls0;
      } else
        throw std::runtime_error("the levelset transition queue is empty.");
      // only allows a pair of the same type of levelsets
      match([&ret](auto &&src, auto &&dst) {
        if constexpr (is_same_v<RM_CVREF_T(src), RM_CVREF_T(dst)>)
          ret = zs::make_tuple(std::move(src), std::move(dst));
        else
          throw std::runtime_error("heterogeneous levelset queue is not supported.");
      })(std::move(ls0), std::move(ls1));
      return ret;
    }

    // better use custom circular (rolling) array
    std::deque<const_sdf_vel_ls_t> _fields{};  // tuple<Plane, Plane> by default
    value_type _stepDt{0}, _alpha{0};
  };

  ///
  /// special purpose levelset views
  ///
  template <typename SdfLsView, typename VelLsView> struct SdfVelFieldView
      : LevelSetInterface<SdfVelFieldView<SdfLsView, VelLsView>> {
    static_assert(SdfLsView::dim == VelLsView::dim, "dimension mismatch!");
    static_assert(is_floating_point_v<typename SdfLsView::value_type>
                      && is_floating_point_v<typename VelLsView::value_type>,
                  "levelset not in floating point type!");

    using value_type = typename SdfLsView::value_type;
    static constexpr int dim = SdfLsView::dim;

    constexpr SdfVelFieldView() noexcept = default;
    ~SdfVelFieldView() noexcept = default;
    constexpr SdfVelFieldView(const SdfLsView &sdf, const VelLsView &vel) noexcept
        : _sdf(sdf), _vel(vel) {}
    constexpr SdfVelFieldView(const zs::tuple<SdfLsView, VelLsView> &field) noexcept
        : _sdf(zs::get<0>(field)), _vel(zs::get<1>(field)) {}

    /// bounding volume interface
    constexpr auto do_getBoundingBox() const noexcept { return _sdf.getBoundingBox(); }
    /// levelset interface
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      return _sdf.getSignedDistance(x);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getNormal(const VecInterface<VecT> &x) const noexcept {
      return _sdf.getNormal(x);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      // if vel ls is dummy, use sdf ls instead
      if constexpr (is_same_v<VelLsView, DummyLevelSet<value_type, dim>>)
        return _sdf.getMaterialVelocity(x);
      else
        return _vel.getMaterialVelocity(x);  // this is the motive
    }

    SdfLsView _sdf{};
    VelLsView _vel{};
  };

  template <typename FieldView> struct TransitionLevelSetView;
  template <typename SdfLsView, typename VelLsView>
  struct TransitionLevelSetView<SdfVelFieldView<SdfLsView, VelLsView>>
      : LevelSetInterface<TransitionLevelSetView<SdfVelFieldView<SdfLsView, VelLsView>>> {
    using ls_t = SdfVelFieldView<SdfLsView, VelLsView>;
    using value_type = typename ls_t::value_type;
    static constexpr int dim = ls_t::dim;

    constexpr TransitionLevelSetView() noexcept = default;
    ~TransitionLevelSetView() noexcept = default;

    constexpr TransitionLevelSetView(const ls_t &lsvSrc, const ls_t &lsvDst,
                                     const value_type stepDt = (value_type)0,
                                     const value_type alpha = (value_type)0) noexcept
        : _lsvSrc{lsvSrc}, _lsvDst{lsvDst}, _stepDt{stepDt}, _alpha{alpha} {}

    /// bounding volume interface
    constexpr auto do_getBoundingBox() const noexcept {
      auto srcBvp = _lsvSrc.getBoundingBox();
      auto dstBvp = _lsvDst.getBoundingBox();
      using TV = RM_CVREF_T(get<0>(srcBvp));  // ADL
      using Ti = typename TV::index_type;
      return zs::make_tuple(
          TV::init([&](Ti i) {
            return get<0>(srcBvp)[i] < get<0>(dstBvp)[i] ? get<0>(srcBvp)[i] : get<0>(dstBvp)[i];
          }),
          TV::init([&](Ti i) {
            return get<1>(srcBvp)[i] > get<1>(dstBvp)[i] ? get<1>(srcBvp)[i] : get<1>(dstBvp)[i];
          }));
    }
    /// levelset interface
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getSignedDistance(const VecInterface<VecT> &x) const noexcept {
      auto v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) * (value_type)0.5;
      auto x0 = x - _alpha * _stepDt * v, x1 = x + ((value_type)1 - _alpha) * _stepDt * v;
      return ((value_type)1 - _alpha) * _lsvSrc.getSignedDistance(x0)
             + _alpha * _lsvDst.getSignedDistance(x1);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getNormal(const VecInterface<VecT> &x) const noexcept {
      auto v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) * (value_type)0.5;
      auto x0 = x - _alpha * _stepDt * v, x1 = x + ((value_type)1 - _alpha) * _stepDt * v;
      return ((value_type)1 - _alpha) * _lsvSrc.getNormal(x0) + _alpha * _lsvDst.getNormal(x1);
    }
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr auto do_getMaterialVelocity(const VecInterface<VecT> &x) const noexcept {
      auto v = (_lsvSrc.getMaterialVelocity(x) + _lsvDst.getMaterialVelocity(x)) * (value_type)0.5;
      auto x0 = x - _alpha * _stepDt * v, x1 = x + ((value_type)1 - _alpha) * _stepDt * v;
      return ((value_type)1 - _alpha) * _lsvSrc.getMaterialVelocity(x0)
             + _alpha * _lsvDst.getMaterialVelocity(x1);
    }

    ls_t _lsvSrc{}, _lsvDst{};
    value_type _stepDt{0}, _alpha{0};
  };
  template <typename SdfLsView, typename VelLsView, typename... Args>
  TransitionLevelSetView(SdfVelFieldView<SdfLsView, VelLsView>,
                         SdfVelFieldView<SdfLsView, VelLsView>,
                         Args...) -> TransitionLevelSetView<SdfVelFieldView<SdfLsView, VelLsView>>;

}  // namespace zs
