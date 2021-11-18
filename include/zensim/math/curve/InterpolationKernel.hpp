#pragma once
#include "zensim/math/MathUtils.h"
#include "zensim/math/Vec.h"

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
  // ref: ziran2020
  template <int interpolation_degree, typename T> constexpr auto base_node(T x) noexcept {
    // linear: 0
    // quadratic: 1
    // cubic: 2
    constexpr auto offset = (T)0.5 * interpolation_degree;
    return lower_trunc(x - offset);
  }
  template <int order = 0, typename T = float, auto dim = 3>
  constexpr auto quadratic_bspline_weights(const vec<T, dim>& x) noexcept {
    // weights: 0
    // 1nd deriv weights: 1
    // 2nd deriv weights: 2
    static_assert(order == 0 || order == 1 || order == 2,
                  "wrong order for quadratic bspline weight pads");
    using Pad = vec<T, dim, 3>;
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

}  // namespace zs