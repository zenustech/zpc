#pragma once
#include "zensim/ZpcIterator.hpp"
#include "zensim/geometry/LevelSetInterface.h"
#include "zensim/math/MathUtils.h"
#include "zensim/math/Vec.h"
#include "zensim/types/Property.h"
#include "zensim/types/Tuple.h"

namespace zs {

  /// temporary
  template <typename T> constexpr vec<T, 3> bspline_weight(T p, T const dx_inv) noexcept {
    vec<T, 3> dw{};
    T d = p * dx_inv - (lower_trunc(p * dx_inv + 0.5) - 1);
    dw[0] = 0.5f * (1.5 - d) * (1.5 - d);
    d -= 1.0f;
    dw[1] = 0.75 - d * d;
    d = 0.5f + d;
    dw[2] = 0.5 * d * d;
    return dw;
  }
  template <typename T, auto dim>
  constexpr vec<T, dim, 3> bspline_weight(const vec<T, dim> &p, T const dx_inv) noexcept {
    vec<T, dim, 3> dw{};
    for (int i = 0; i < dim; ++i) {
      T d = p(i) * dx_inv - (lower_trunc(p(i) * dx_inv + 0.5) - 1);
      dw(i, 0) = 0.5f * (1.5 - d) * (1.5 - d);
      d -= 1.0f;
      dw(i, 1) = 0.75 - d * d;
      d = 0.5f + d;
      dw(i, 2) = 0.5 * d * d;
    }
    return dw;
  }
  template <kernel_e kt> constexpr int get_lerp_degree(wrapv<kt> = {}) noexcept {
    if constexpr (kt == kernel_e::linear)
      return 0;
    else if constexpr (kt == kernel_e::quadratic)
      return 1;
    else if constexpr (kt == kernel_e::cubic)
      return 2;
    return -1;
  }
  // ref: ziran2020
  // https://github.com/penn-graphics-research/ziran2020
  template <int interpolation_degree, typename T,
            typename Ti = conditional_t<sizeof(T) <= sizeof(f32), i32, i64>,
            enable_if_t<is_floating_point_v<T>> = 0>
  constexpr auto base_node(T x, wrapt<Ti> tag = {}) noexcept {
    // linear: 0
    // quadratic: 1
    // cubic: 2
    constexpr auto offset = (T)0.5 * interpolation_degree;
    return lower_trunc(x - offset, tag);
  }

  template <int order, typename VecT,
            enable_if_all<VecT::dim == 1, is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto linear_bspline_weights(const VecInterface<VecT> &x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;
    // weights: 0
    // 1nd deriv weights: 1
    // 2nd deriv weights: 2
    static_assert(order == 0 || order == 1 || order == 2,
                  "wrong order for linear bspline weight pads");
    using Pad =
        typename VecT::template variant_vec<T, integer_sequence<typename VecT::index_type, dim, 2>>;
    using RetT = conditional_t<order == 0, tuple<Pad>,
                               conditional_t<order == 1, tuple<Pad, Pad>, tuple<Pad, Pad, Pad>>>;
    RetT weights{};
    auto &w{get<0>(weights)};
    for (int i = 0; i != dim; ++i) {
      T dx = x(i) - base_node<0>(x(i));
      w(i, 0) = (T)1 - dx;
      w(i, 1) = dx;
      if constexpr (order > 0) {
        auto &dw{get<1>(weights)};
        dw(i, 0) = (T)-1;
        dw(i, 1) = (T)1;
      }
      if constexpr (order == 2) {
        auto &ddw{get<2>(weights)};
        ddw(i, 0) = (T)0;
        ddw(i, 1) = (T)0;
      }
    }
    return weights;
  }

  template <int order, typename VecT,
            enable_if_all<VecT::dim == 1, is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto quadratic_bspline_weights(const VecInterface<VecT> &x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;
    // weights: 0
    // 1nd deriv weights: 1
    // 2nd deriv weights: 2
    static_assert(order == 0 || order == 1 || order == 2,
                  "wrong order for quadratic bspline weight pads");
    using Pad =
        typename VecT::template variant_vec<T, integer_sequence<typename VecT::index_type, dim, 3>>;
    using RetT = conditional_t<order == 0, tuple<Pad>,
                               conditional_t<order == 1, tuple<Pad, Pad>, tuple<Pad, Pad, Pad>>>;
    RetT weights{};
    auto &w{get<0>(weights)};
    for (int i = 0; i != dim; ++i) {
      T d0 = x(i) - base_node<1>(x(i));
      w(i, 0) = (T)0.5 * ((T)1.5 - d0) * ((T)1.5 - d0);
      T d1 = d0 - (T)1.0;
      w(i, 1) = (T)0.75 - d1 * d1;
      T zz = (T)0.5 + d1;
      w(i, 2) = (T)0.5 * zz * zz;
      if constexpr (order > 0) {
        auto &dw{get<1>(weights)};
        dw(i, 0) = d0 - (T)1.5;
        dw(i, 1) = -(d1 + d1);
        dw(i, 2) = zz;
      }
      if constexpr (order == 2) {
        auto &ddw{get<2>(weights)};
        ddw(i, 0) = 1;
        ddw(i, 1) = -2;
        ddw(i, 2) = 1;
      }
    }
    return weights;
  }

  template <int order, typename VecT,
            enable_if_all<VecT::dim == 1, is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto cubic_bspline_weights(const VecInterface<VecT> &x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;
    // weights: 0
    // 1nd deriv weights: 1
    // 2nd deriv weights: 2
    static_assert(order == 0 || order == 1 || order == 2,
                  "wrong order for cubic bspline weight pads");
    using Pad =
        typename VecT::template variant_vec<T, integer_sequence<typename VecT::index_type, dim, 4>>;
    using RetT = conditional_t<order == 0, tuple<Pad>,
                               conditional_t<order == 1, tuple<Pad, Pad>, tuple<Pad, Pad, Pad>>>;
    RetT weights{};
    auto &w{get<0>(weights)};
    for (int i = 0; i != dim; ++i) {
      T d0 = x(i) - base_node<2>(x(i));
      T z = (T)2 - d0;
      T z3 = z * z * z;
      w(i, 0) = z3 / (T)6;

      T d1 = d0 - (T)1.0;  // d0 - 1
      w(i, 1) = ((T)0.5 * d1 - (T)1) * d1 * d1 + (T)2 / (T)3;

      T d2 = (T)1 - d1;  // 2 - d0
      w(i, 2) = ((T)0.5 * d2 - (T)1) * d2 * d2 + (T)2 / (T)3;

      T d3 = (T)1 + d2;  // d0 - 1
      T zzzz = (T)2 - d3;
      w(i, 3) = zzzz * zzzz * zzzz / (T)6;

      if constexpr (order > 0) {
        auto &dw{get<1>(weights)};
        dw(i, 0) = (T)-0.5 * z * z;
        dw(i, 1) = ((T)1.5 * d1 - (T)2) * d1;
        dw(i, 2) = ((T)-1.5 * d2 + (T)2) * d2;
        dw(i, 3) = (T)0.5 * zzzz * zzzz;
      }
      if constexpr (order == 2) {
        auto &ddw{get<2>(weights)};
        ddw(i, 0) = (T)2 - d0;
        ddw(i, 1) = (T)-2 + (T)3 * (d0 - (T)1);
        ddw(i, 2) = (T)-2 + (T)3 * ((T)2 - d0);
        ddw(i, 3) = (T)-1 + d0;
      }
    }
    return weights;
  }

  template <typename VecT,
            enable_if_all<VecT::dim == 1, is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto delta_2point_weights(const VecInterface<VecT> &x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;

    using Pad =
        typename VecT::template variant_vec<T, integer_sequence<typename VecT::index_type, dim, 2>>;
    using RetT = tuple<Pad>;

    RetT weights{};
    auto &w{get<0>(weights)};
    for (int i = 0; i != dim; ++i) {
      T base = base_node<0>(x(i));

      for (int off = 0; off < 2; ++off) {
        T r = abs(x(i) - (base + (T)off));
        w(i, off) = 0;

        if (r < (T)1) {
          w(i, off) = 1 - r;
        }
      }
    }
    return weights;
  }

  template <typename VecT,
            enable_if_all<VecT::dim == 1, is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto delta_3point_weights(const VecInterface<VecT> &x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;

    using Pad =
        typename VecT::template variant_vec<T, integer_sequence<typename VecT::index_type, dim, 3>>;
    using RetT = tuple<Pad>;

    RetT weights{};
    auto &w{get<0>(weights)};

    for (int i = 0; i != dim; ++i) {
      T base = base_node<1>(x(i));

      for (int off = 0; off < 3; ++off) {
        T r = abs(x(i) - (base + (T)off));
        w(i, off) = 0;

        if (r <= (T)0.5) {
          w(i, off) = (T)1. / (T)3. * ((T)1. + sqrt((T)-3. * r * r + (T)1.));
        } else if (r > (T)0.5 && r < (T)1.5) {
          w(i, off) = (T)1. / (T)6.
                      * ((T)5. - (T)3. * r - sqrt((T)-3. * ((T)1. - r) * ((T)1. - r) + (T)1.));
        }
      }
    }
    return weights;
  }

  template <typename VecT,
            enable_if_all<VecT::dim == 1, is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto delta_4point_weights(const VecInterface<VecT> &x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;

    using Pad =
        typename VecT::template variant_vec<T, integer_sequence<typename VecT::index_type, dim, 4>>;
    using RetT = tuple<Pad>;

    RetT weights{};
    auto &w{get<0>(weights)};

    for (int i = 0; i != dim; ++i) {
      T base = base_node<2>(x(i));

      for (int off = 0; off < 4; ++off) {
        T r = abs(x(i) - (base + (T)off));
        w(i, off) = 0;

        if (r <= (T)1.) {
          w(i, off) = (T)1. / (T)8. * ((T)3. - (T)2. * r + sqrt((T)1. + (T)4. * r - (T)4. * r * r));
        } else if (r > (T)1. && r < (T)2.) {
          w(i, off)
              = (T)1. / (T)8. * ((T)5. - (T)2. * r - sqrt((T)-7. + (T)12. * r - (T)4. * r * r));
        }
      }
    }
    return weights;
  }

  template <typename T> struct is_grid_accessor : false_type {};

  template <typename GridViewT, kernel_e kt_ = kernel_e::linear, int drv_order = 0>
  struct GridArena {
    using grid_view_type = GridViewT;
    using value_type = typename grid_view_type::value_type;
    using size_type = typename grid_view_type::size_type;

    using integer_coord_component_type = typename grid_view_type::integer_coord_component_type;
    using integer_coord_type = typename grid_view_type::integer_coord_type;
    using coord_component_type = typename grid_view_type::coord_component_type;
    using coord_type = typename grid_view_type::coord_type;

    static constexpr int dim = grid_view_type::dim;
    static constexpr kernel_e kt = kt_;
    static constexpr int width = [](kernel_e kt) {
      if (kt == kernel_e::linear || kt == kernel_e::delta2)
        return 2;
      else if (kt == kernel_e::quadratic || kt == kernel_e::delta3)
        return 3;
      else if (kt == kernel_e::cubic || kt == kernel_e::delta4)
        return 4;
      return -1;
    }(kt);
    static constexpr int deriv_order = drv_order;

    using TWM = vec<coord_component_type, dim, width>;

    static_assert(deriv_order >= 0 && deriv_order <= 2,
                  "weight derivative order should be an integer within [0, 2]");
    static_assert(((kt == kernel_e::delta2 || kt == kernel_e::delta3 || kt == kernel_e::delta4)
                   && deriv_order == 0)
                      || (kt == kernel_e::linear || kt == kernel_e::quadratic
                          || kt == kernel_e::cubic),
                  "weight derivative order should be 0 when using delta kernel");

    using WeightScratchPad
        = conditional_t<deriv_order == 0, tuple<TWM>,
                        conditional_t<deriv_order == 1, tuple<TWM, TWM>, tuple<TWM, TWM, TWM>>>;

    template <typename ValT, size_t... Is>
    static constexpr auto deduce_arena_type_impl(index_sequence<Is...>) {
      return vec<ValT, (Is + 1 > 0 ? width : width)...>{};
    }
    template <typename ValT, int d> static constexpr auto deduce_arena_type() {
      return deduce_arena_type_impl<ValT>(make_index_sequence<d>{});
    }
    template <typename ValT> using arena_type = RM_CVREF_T(deduce_arena_type<ValT, dim>());

    /// constructors
    /// index-space ctors
    // collocated grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(false_type, grid_view_type *gv, const VecInterface<VecT> &X) noexcept
        : gridPtr{gv}, weights{}, iLocalPos{}, iCorner{} {
      constexpr int lerp_degree
          = ((kt == kernel_e::linear || kt == kernel_e::delta2)
                 ? 0
                 : ((kt == kernel_e::quadratic || kt == kernel_e::delta3) ? 1 : 2));
      for (int d = 0; d != dim; ++d)
        iCorner[d] = base_node<lerp_degree>(X[d], wrapt<integer_coord_component_type>{});
      iLocalPos = X - iCorner;
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::delta2)
        weights = delta_2point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta3)
        weights = delta_3point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta4)
        weights = delta_4point_weights(iLocalPos);
    }
    // staggered grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(false_type, grid_view_type *gv, const VecInterface<VecT> &X, int f) noexcept
        : gridPtr{gv}, weights{}, iLocalPos{}, iCorner{} {
      constexpr int lerp_degree
          = ((kt == kernel_e::linear || kt == kernel_e::delta2)
                 ? 0
                 : ((kt == kernel_e::quadratic || kt == kernel_e::delta3) ? 1 : 2));
      const auto delta = coord_type::init([f = f % dim](int d) {
        return d != f ? (coord_component_type)0 : (coord_component_type)-0.5;
      });
      for (int d = 0; d != dim; ++d) iCorner[d] = base_node<lerp_degree>(X[d] - delta[d]);
      iLocalPos = X - (iCorner + delta);
      if constexpr (kt == kernel_e::linear)
        weights = linear_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::quadratic)
        weights = quadratic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::cubic)
        weights = cubic_bspline_weights<deriv_order>(iLocalPos);
      else if constexpr (kt == kernel_e::delta2)
        weights = delta_2point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta3)
        weights = delta_3point_weights(iLocalPos);
      else if constexpr (kt == kernel_e::delta4)
        weights = delta_4point_weights(iLocalPos);
    }
    /// world-space ctors
    // collocated grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(true_type, grid_view_type *gv, const VecInterface<VecT> &x) noexcept
        : GridArena{false_c, gv, gv->worldToIndex(x)} {}
    // staggered grid
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::extent == dim> = 0>
    constexpr GridArena(true_type, grid_view_type *gv, const VecInterface<VecT> &x, int f) noexcept
        : GridArena{false_c, gv, gv->worldToIndex(x), f} {}

    /// scalar arena
    constexpr arena_type<value_type> arena(size_type chn,
                                           value_type defaultVal = {}) const noexcept {
      // ensure that chn's orientation is aligned with initialization if within a staggered grid
      arena_type<value_type> pad{};
      for (auto offset : ndrange<dim>(width)) {
        if constexpr (is_grid_accessor<remove_cvref_t<grid_view_type>>::value) {
          value_type val{};
          bool found = const_cast<remove_cvref_t<grid_view_type> *>(gridPtr)->probeValue(
              chn, iCorner + make_vec<integer_coord_component_type>(offset), val);
          // if (!found) val = defaultVal;
          pad.val(offset) = val;
        } else if constexpr (is_ag_v<typename grid_view_type::container_type>) {
          pad.val(offset) = gridPtr->value(
              false_c, chn, iCorner + make_vec<integer_coord_component_type>(offset));
        } else {
          pad.val(offset) = gridPtr->valueOr(
              false_c, chn, iCorner + make_vec<integer_coord_component_type>(offset), defaultVal);
        }
      }
      return pad;
    }
    constexpr arena_type<value_type> arena(const SmallString &propName, size_type chn = 0,
                                           value_type defaultVal = {}) const noexcept {
      return arena(gridPtr->propertyOffset(propName) + chn, defaultVal);
    }

    /// helpers
    constexpr auto range() const noexcept { return ndrange<dim>(width); }

    template <typename... Tn> constexpr auto offset(const std::tuple<Tn...> &loc) const noexcept {
      return make_vec<int>(loc);
    }
    template <typename... Tn> constexpr auto offset(const tuple<Tn...> &loc) const noexcept {
      return make_vec<int>(loc);
    }

    template <typename... Tn> constexpr auto coord(const std::tuple<Tn...> &loc) const noexcept {
      return iCorner + offset(loc);
    }
    template <typename... Tn> constexpr auto coord(const tuple<Tn...> &loc) const noexcept {
      return iCorner + offset(loc);
    }

    /// minimum
    constexpr value_type minimum(size_type chn = 0) const noexcept {
      auto pad = arena(chn, detail::deduce_numeric_max<value_type>());
      value_type ret = detail::deduce_numeric_max<value_type>();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v < ret) ret = v;
      return ret;
    }
    constexpr value_type minimum(const SmallString &propName, size_type chn = 0) const noexcept {
      return minimum(gridPtr->propertyOffset(propName) + chn);
    }

    /// maximum
    constexpr value_type maximum(size_type chn = 0) const noexcept {
      auto pad = arena(chn, detail::deduce_numeric_lowest<value_type>());
      value_type ret = detail::deduce_numeric_lowest<value_type>();
      for (auto offset : ndrange<dim>(width))
        if (const auto &v = pad.val(offset); v > ret) ret = v;
      return ret;
    }
    constexpr value_type maximum(const SmallString &propName, size_type chn = 0) const noexcept {
      return maximum(gridPtr->propertyOffset(propName) + chn);
    }

    /// isample
    constexpr value_type isample(size_type chn, value_type defaultVal = {}) const noexcept {
      auto pad = arena(chn, defaultVal);
      if constexpr (kt == kernel_e::linear)
        return xlerp(iLocalPos, pad);
      else {
        value_type ret = 0;
        for (auto offset : ndrange<dim>(width)) ret += weight(offset) * pad.val(offset);
        return ret;
      }
    }
    constexpr value_type isample(const SmallString &propName, size_type chn,
                                 value_type defaultVal = {}) const noexcept {
      return isample(gridPtr->propertyOffset(propName) + chn, defaultVal);
    }

    /// weight
    template <typename... Tn>
    constexpr value_type weight(const std::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, index_sequence_for<Tn...>{});
    }
    template <typename... Tn>
    constexpr value_type weight(const zs::tuple<Tn...> &loc) const noexcept {
      return weight_impl(loc, index_sequence_for<Tn...>{});
    }
    template <typename... Tn, enable_if_all<((!is_tuple_v<Tn> && !is_std_tuple<Tn>()) && ...
                                             && (sizeof...(Tn) == dim))>
                              = 0>
    constexpr auto weight(Tn &&...is) const noexcept {
      return weight(zs::forward_as_tuple(FWD(is)...));
    }
    /// weight gradient
    template <zs::size_t I, typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_component_type> weightGradient(
        const std::tuple<Tn...> &loc) const noexcept {
      return weightGradient_impl<I>(loc, index_sequence_for<Tn...>{});
    }
    template <zs::size_t I, typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_component_type> weightGradient(
        const zs::tuple<Tn...> &loc) const noexcept {
      return weightGradient_impl<I>(loc, index_sequence_for<Tn...>{});
    }
    /// weight gradient
    template <typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_type> weightsGradient(
        const std::tuple<Tn...> &loc) const noexcept {
      return weightGradients_impl(loc, index_sequence_for<Tn...>{});
    }
    template <typename... Tn, auto ord = deriv_order>
    constexpr enable_if_type<(ord > 0), coord_type> weightsGradient(
        const zs::tuple<Tn...> &loc) const noexcept {
      return weightGradients_impl(loc, index_sequence_for<Tn...>{});
    }

    void printWeights() {
      value_type sum = 0;
      for (int d = 0; d != dim; ++d) {
        for (int w = 0; w != width; ++w) {
          sum += get<0>(weights)(d, w);
          fmt::print("weights({}, {}): [{}]\t", d, w, get<0>(weights)(d, w));
          if constexpr (deriv_order > 0) fmt::print("[{}]\t", get<1>(weights)(d, w));
          if constexpr (deriv_order > 1) fmt::print("[{}]\t", get<2>(weights)(d, w));
          fmt::print("\n");
        }
      }
      fmt::print("weight sum: {}\n", sum);
    }

    grid_view_type *gridPtr{nullptr};
    WeightScratchPad weights{};
    coord_type iLocalPos{coord_type::zeros()};                // index-space local offset
    integer_coord_type iCorner{integer_coord_type::zeros()};  // index-space global coord

  protected:
    /// std tuple
    template <typename... Tn, size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr coord_component_type weight_impl(const std::tuple<Tn...> &loc,
                                               index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= get<0>(weights)(Is, std::get<Is>(loc))), ...);
      return ret;
    }
    template <zs::size_t I, typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_component_type weightGradient_impl(const std::tuple<Tn...> &loc,
                                                       index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= (I == Is ? get<1>(weights)(Is, std::get<Is>(loc))
                              : get<0>(weights)(Is, std::get<Is>(loc)))),
       ...);
      return ret;
    }
    template <typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_type weightGradients_impl(const std::tuple<Tn...> &loc,
                                              index_sequence<Is...>) const noexcept {
      return coord_type{weightGradient_impl<Is>(loc, index_sequence<Is...>{})...};
    }
    /// zs tuple
    template <typename... Tn, size_t... Is,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim)> = 0>
    constexpr coord_component_type weight_impl(const zs::tuple<Tn...> &loc,
                                               index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= get<0>(weights)(Is, zs::get<Is>(loc))), ...);
      return ret;
    }
    template <zs::size_t I, typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_component_type weightGradient_impl(const zs::tuple<Tn...> &loc,
                                                       index_sequence<Is...>) const noexcept {
      coord_component_type ret{1};
      ((void)(ret *= (I == Is ? get<1>(weights)(Is, zs::get<Is>(loc))
                              : get<0>(weights)(Is, zs::get<Is>(loc)))),
       ...);
      return ret;
    }
    template <typename... Tn, size_t... Is, auto ord = deriv_order,
              enable_if_all<(sizeof...(Is) == dim), (sizeof...(Tn) == dim), (ord > 0)> = 0>
    constexpr coord_type weightGradients_impl(const zs::tuple<Tn...> &loc,
                                              index_sequence<Is...>) const noexcept {
      return coord_type{weightGradient_impl<Is>(loc, index_sequence<Is...>{})...};
    }
  };

  template <typename AdaptiveGridViewT, int NumDepths> struct AdaptiveGridAccessor {
    static constexpr int dim = AdaptiveGridViewT::dim;
    static constexpr int num_levels = AdaptiveGridViewT::num_levels;
    static constexpr int cache_depths = NumDepths;
    static_assert(cache_depths <= num_levels, "???");

    using value_type = typename AdaptiveGridViewT::value_type;
    using size_type = typename AdaptiveGridViewT::size_type;
    using index_type = typename AdaptiveGridViewT::index_type;
    static_assert(is_signed_v<index_type>, "???");

    static constexpr index_type sentinel_v = AdaptiveGridViewT::sentinel_v;

    using integer_coord_component_type = typename AdaptiveGridViewT::integer_coord_component_type;
    using integer_coord_type = typename AdaptiveGridViewT::integer_coord_type;
    using coord_component_type = typename AdaptiveGridViewT::coord_component_type;
    using coord_type = typename AdaptiveGridViewT::coord_type;
    using coord_mask_type = typename AdaptiveGridViewT::coord_mask_type;

    static constexpr integer_coord_type get_key_sentinel() noexcept {
      return integer_coord_type::constant(
          detail::deduce_numeric_max<integer_coord_component_type>());
    }

    struct CacheRecord {
      integer_coord_type coord{get_key_sentinel()};
      index_type blockNo{sentinel_v};
    };
    using cache_type =
        typename build_seq<cache_depths>::template uniform_types_t<zs::tuple, CacheRecord>;

    // template <size_t I, typename AgT = typename AdaptiveGridViewT, enable_if_t<is_ag_v<typename
    // AgT::container_type>> = 0> static auto deduce_level_grid_tile_type()
    //     -> decltype(declval<typename AgT::template level_view_type<I>::grid_type>().tile(0));
    // template <size_t I, typename SpgT = typename AdaptiveGridViewT, enable_if_t<is_spg_v<typename
    // SpgT::container_type>> = 0> static auto deduce_level_grid_tile_type() ->
    // decltype(declval<SpgT>()._grid.tile(0)); template <size_t I> using grid_tile_type =
    // decltype(deduce_level_grid_tile_type<cache_depths - 1 - I>()); using cached_tiles_type =
    // assemble_t<zs::tuple, typename build_seq<cache_depths>::template type_seq_t<grid_tile_type>>;

    constexpr AdaptiveGridAccessor() noexcept = default;
    constexpr AdaptiveGridAccessor(AdaptiveGridViewT *gridPtr) noexcept
        : _gridPtr{gridPtr}, _cache{} {}
    ~AdaptiveGridAccessor() = default;

    template <int D> static constexpr coord_mask_type origin_mask
        = AdaptiveGridViewT::template level_view_type<num_levels - 1 - D>::origin_mask;

    template <int D> constexpr CacheRecord &cache() const { return zs::get<D>(_cache); }
    template <int D> constexpr CacheRecord &cache() { return zs::get<D>(_cache); }

#if 0
    template <typename T, int D>
    constexpr bool eval(size_type chn, const integer_coord_type &coord, T &val, wrapv<D>) {
      if constexpr (D == 0) {
        return _gridPtr->probeValueAndCache(*this, chn, coord, val, sentinel_v,
                                            /*Ordered*/ true_c, wrapv<num_levels - 1>{},
                                            wrapv<true>{});
      } else
        return false;
    }
    template <typename T, int D, size_t... Is>
    constexpr bool evalFirstCached(size_type chn, const integer_coord_type &coord, T &val, wrapv<D>,
                                   index_sequence<Is...>) {
      bool ret = false;
      ((isHashed(coord, wrapv<D - (int)Is>{})
            ? /*0*/ (
                ret = _gridPtr->probeValueAndCache(
                    *this, chn, coord, val, retrieveCache<D - (int)Is>(),
                    /*Ordered*/ true_c, wrapv<num_levels - 1 - (D - (int)Is)>{}, wrapv<false>{}),
                true)
            : (/*1*/ (ret = eval(chn, coord, val, wrapv<D - (int)Is>{})), false))
       || ...);
      return ret;
    }
    template <typename T, int D = cache_depths - 1, enable_if_t<!is_const_v<T>> = 0>
    constexpr bool probeValue(size_type chn, const integer_coord_type &coord, T &val,
                              wrapv<D> = {}) {
      return evalFirstCached(chn, coord, val, wrapv<D>{}, typename build_seq<D + 1>::ascend{});
    }
#else

    template <typename T, int D = cache_depths - 1, enable_if_t<!is_const_v<T>> = 0>
    constexpr bool probeValue(size_type chn, const integer_coord_type &coord, T &val,
                              wrapv<D> = {}) {
      if (isHashed(coord, wrapv<D>{})) {
        // check cache first
        return _gridPtr->probeValueAndCache(*this, chn, coord, val, retrieveCache<D>(),
                                            /*Ordered*/ true_c, wrapv<num_levels - 1 - D>{},
                                            wrapv<false>{});
      } else {
        if constexpr (D == 0)  // the last cache level
          return _gridPtr->probeValueAndCache(*this, chn, coord, val, sentinel_v,
                                              /*Ordered*/ true_c, wrapv<num_levels - 1>{},
                                              wrapv<true>{});
      }
      if constexpr (D != 0) return this->probeValue(chn, coord, val, wrapv<D - 1>{});
    }
#endif

    /// @note I is depth index here rather than level
    template <int D> constexpr index_type retrieveCache() const { return cache<D>().blockNo; }
    template <int D> constexpr void eraseCache() { cache<D>() = CacheRecord{}; }
    template <int D = 0> constexpr void clear(wrapv<D> = {}) {
      eraseCache<D>();
      if constexpr (D + 1 < cache_depths) clear<D + 1>();
    }

    template <int D>
    constexpr void insert(const integer_coord_type &coord, index_type index, wrapv<D>) const {
      if constexpr (D >= 0 && D < cache_depths) {
        auto &c = cache<D>();
        c.coord = coord & origin_mask<D>;
        c.blockNo = index;
      }
    }
    template <int D> constexpr bool isHashed(const integer_coord_type &coord, wrapv<D>) const {
      if constexpr (D < 0 || D >= cache_depths)
        return false;
      else {
        const auto &cc = cache<D>().coord;
        for (int d = 0; d < dim; ++d)
          if ((coord[d] & origin_mask<D>) != cc[d]) return false;
        return true;
      }
    }

    AdaptiveGridViewT *_gridPtr{nullptr};
    mutable cache_type _cache{};
    // mutable cached_tiles_type _cachedTiles{};
  };

  template <typename AdaptiveGridViewT, int NumDepths>
  struct is_grid_accessor<AdaptiveGridAccessor<AdaptiveGridViewT, NumDepths>> : true_type {};

}  // namespace zs