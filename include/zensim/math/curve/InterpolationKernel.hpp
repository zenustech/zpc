#pragma once
#include "zensim/math/MathUtils.h"
#include "zensim/math/Vec.h"
#include "zensim/types/Property.h"

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
  constexpr vec<T, dim, 3> bspline_weight(const vec<T, dim>& p, T const dx_inv) noexcept {
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
            enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr auto base_node(T x, wrapt<Ti> tag = {}) noexcept {
    // linear: 0
    // quadratic: 1
    // cubic: 2
    constexpr auto offset = (T)0.5 * interpolation_degree;
    return lower_trunc(x - offset, tag);
  }

  template <int order, typename VecT,
            enable_if_all<VecT::dim == 1, std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto linear_bspline_weights(const VecInterface<VecT>& x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;
    // weights: 0
    // 1nd deriv weights: 1
    // 2nd deriv weights: 2
    static_assert(order == 0 || order == 1 || order == 2,
                  "wrong order for linear bspline weight pads");
    using Pad =
        typename VecT::template variant_vec<T, integer_seq<typename VecT::index_type, dim, 2>>;
    using RetT = conditional_t<order == 0, tuple<Pad>,
                               conditional_t<order == 1, tuple<Pad, Pad>, tuple<Pad, Pad, Pad>>>;
    RetT weights{};
    auto& w{get<0>(weights)};
    for (int i = 0; i != dim; ++i) {
      T dx = x(i) - base_node<0>(x(i));
      w(i, 0) = (T)1 - dx;
      w(i, 1) = dx;
      if constexpr (order > 0) {
        auto& dw{get<1>(weights)};
        dw(i, 0) = (T)-1;
        dw(i, 1) = (T)1;
      }
      if constexpr (order == 2) {
        auto& ddw{get<2>(weights)};
        ddw(i, 0) = (T)0;
        ddw(i, 1) = (T)0;
      }
    }
    return weights;
  }

  template <int order, typename VecT,
            enable_if_all<VecT::dim == 1, std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto quadratic_bspline_weights(const VecInterface<VecT>& x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;
    // weights: 0
    // 1nd deriv weights: 1
    // 2nd deriv weights: 2
    static_assert(order == 0 || order == 1 || order == 2,
                  "wrong order for quadratic bspline weight pads");
    using Pad =
        typename VecT::template variant_vec<T, integer_seq<typename VecT::index_type, dim, 3>>;
    using RetT = conditional_t<order == 0, tuple<Pad>,
                               conditional_t<order == 1, tuple<Pad, Pad>, tuple<Pad, Pad, Pad>>>;
    RetT weights{};
    auto& w{get<0>(weights)};
    for (int i = 0; i != dim; ++i) {
      T d0 = x(i) - base_node<1>(x(i));
      w(i, 0) = (T)0.5 * ((T)1.5 - d0) * ((T)1.5 - d0);
      T d1 = d0 - (T)1.0;
      w(i, 1) = (T)0.75 - d1 * d1;
      T zz = (T)0.5 + d1;
      w(i, 2) = (T)0.5 * zz * zz;
      if constexpr (order > 0) {
        auto& dw{get<1>(weights)};
        dw(i, 0) = d0 - (T)1.5;
        dw(i, 1) = -(d1 + d1);
        dw(i, 2) = zz;
      }
      if constexpr (order == 2) {
        auto& ddw{get<2>(weights)};
        ddw(i, 0) = 1;
        ddw(i, 1) = -2;
        ddw(i, 2) = 1;
      }
    }
    return weights;
  }

  template <int order, typename VecT,
            enable_if_all<VecT::dim == 1, std::is_floating_point_v<typename VecT::value_type>> = 0>
  constexpr auto cubic_bspline_weights(const VecInterface<VecT>& x) noexcept {
    using T = typename VecT::value_type;
    constexpr auto dim = VecT::extent;
    // weights: 0
    // 1nd deriv weights: 1
    // 2nd deriv weights: 2
    static_assert(order == 0 || order == 1 || order == 2,
                  "wrong order for cubic bspline weight pads");
    using Pad =
        typename VecT::template variant_vec<T, integer_seq<typename VecT::index_type, dim, 4>>;
    using RetT = conditional_t<order == 0, tuple<Pad>,
                               conditional_t<order == 1, tuple<Pad, Pad>, tuple<Pad, Pad, Pad>>>;
    RetT weights{};
    auto& w{get<0>(weights)};
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
        auto& dw{get<1>(weights)};
        dw(i, 0) = (T)-0.5 * z * z;
        dw(i, 1) = ((T)1.5 * d1 - (T)2) * d1;
        dw(i, 2) = ((T)-1.5 * d2 + (T)2) * d2;
        dw(i, 3) = (T)0.5 * zzzz * zzzz;
      }
      if constexpr (order == 2) {
        auto& ddw{get<2>(weights)};
        ddw(i, 0) = (T)2 - d0;
        ddw(i, 1) = (T)-2 + (T)3 * (d0 - (T)1);
        ddw(i, 2) = (T)-2 + (T)3 * ((T)2 - d0);
        ddw(i, 3) = (T)-1 + d0;
      }
    }
    return weights;
  }

}  // namespace zs