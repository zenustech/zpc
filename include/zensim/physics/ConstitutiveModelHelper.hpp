#pragma once
#include "zensim/math/Vec.h"

namespace zs {

  /// I, dI/dF, d2I/dF2

  // I1
  template <int Opt = 1, typename T0, typename T1, typename Tn, Tn dim>
  constexpr auto eval_I1_deriv_hessian(const vec_t<T0, Tn, dim, dim>& F,
                                       const vec_t<T1, Tn, dim>& a) {
    if constexpr (Opt == 1)
      return std::make_tuple(mul(F, a).l2NormSqr());
    else {
      auto dyadicAa = dyadic_prod(a, a);
      if constexpr (Opt == 2)
        return std::make_tuple(mul(F, a).l2NormSqr(), 2 * vectorize(mul(F, dyadicAa)));
      else {
        using R = math::op_result_t<T0, T1>;
        auto H = vec_t<R, Tn, dim * dim, dim * dim>::zeros();
        for (Tn i = 0; i != dim; ++i)
          for (Tn j = 0, bi = i * dim; j != dim; ++j) {
            const auto v = dyadicAa(i, j);
            for (Tn d = 0, bj = j * dim; d != dim; ++d) H(bi + d, bj + d) = v;
          }
        return std::make_tuple(mul(F, a).l2NormSqr(), 2 * vectorize(mul(F, dyadicAa)), H * 2);
      }
    }
  }

  // I2
  template <int Opt = 1, typename T0, typename T1, typename T2, typename Tn, int dim>
  constexpr auto eval_I2_deriv_hessian(const vec_t<T0, Tn, dim, dim>& F,
                                       const vec_t<T1, Tn, dim>& a1, const vec_t<T2, Tn, dim>& a2) {
    if constexpr (Opt == 1)
      return std::make_tuple(mul(F, a1).dot(mul(F, a2)));
    else {
      auto dyadicA12 = dyadic_prod(a1, a2);
      dyadicA12 = (dyadicA12 + dyadicA12.transpose());
      if constexpr (Opt == 2)
        return std::make_tuple(mul(F, a1).dot(mul(F, a2)), vectorize(F * dyadicA12));
      else {
        using R = math::op_result_t<T0, T1, T2>;
        auto H = vec_t<R, Tn, dim * dim, dim * dim>::zeros();
        for (Tn i = 0; i != dim; ++i)
          for (Tn j = 0, bi = i * dim; j != dim; ++j) {
            const auto v = dyadicA12(i, j);
            for (Tn d = 0, bj = j * dim; d != dim; ++d) H(bi + d, bj + d) = v;
          }
        return std::make_tuple(mul(F, a1).dot(mul(F, a2)), vectorize(F * dyadicA12), H);
      }
    }
  }

  // I3
  template <int Opt = 1, typename T, typename Tn>
  constexpr auto eval_I3_deriv_hessian(const vec_t<T, Tn, 3, 3>& F) {
    if constexpr (Opt == 1)
      return std::make_tuple(determinant(F));
    else {
      auto f0 = col(F, 0);
      auto f1 = col(F, 1);
      auto f2 = col(F, 2);
      auto f12 = cross(f1, f2);
      auto f20 = cross(f2, f0);
      auto f01 = cross(f0, f1);
      if constexpr (Opt == 2) {
        return std::make_tuple(
            determinant(F), vec_t<T, Tn, 9>{f12(0), f12(1), f12(2), f20(0), f20(1), f20(2), f01(0),
                                            f01(1), f01(2)});
      } else {
        auto H = vec_t<T, Tn, 9, 9>::zeros();
        auto asym = asymmetric(f0);
        for (Tn i = 0; i != 3; ++i)
          for (Tn j = 0; j != 3; ++j) {
            H(6 + i, 3 + j) = asym(i, j);
            H(3 + i, 6 + j) = -asym(i, j);
          }
        asym = asymmetric(f1);
        for (Tn i = 0; i != 3; ++i)
          for (Tn j = 0; j != 3; ++j) {
            H(6 + i, j) = -asym(i, j);
            H(i, 6 + j) = asym(i, j);
          }
        asym = asymmetric(f2);
        for (Tn i = 0; i != 3; ++i)
          for (Tn j = 0; j != 3; ++j) {
            H(3 + i, j) = asym(i, j);
            H(i, 3 + j) = -asym(i, j);
          }
        return std::make_tuple(
            determinant(F),
            vec_t<T, Tn, 9>{f12(0), f12(1), f12(2), f20(0), f20(1), f20(2), f01(0), f01(1), f01(2)},
            H);
      }
    }
  }

  template <typename T, typename Tn>
  constexpr auto get_I1_sigma(const vec_t<T, Tn, (Tn)3>& al) noexcept {
    return vec_t<T, Tn, 3, 3>{al[0] * al[0], 0, 0, 0, al[1] * al[1], 0, 0, 0, al[2] * al[2]};
  }

  template <typename T, typename Tn>
  constexpr auto get_I2_sigma(const vec_t<T, Tn, (Tn)3>& al) noexcept {
    return vec_t<T, Tn, 3, 3>{0, al[0] * al[1], al[0] * al[2], al[1] * al[0],
                              0, al[1] * al[2], al[2] * al[0], al[2] * al[1],
                              0};
  }

  template <typename T0, typename T1, typename T2, typename Tn>
  constexpr auto eval_I1_delta(const vec_t<T0, Tn, (Tn)3>& aw,
                               const vec_t<T1, Tn, (Tn)3, (Tn)3>& fiber_dir,
                               const vec_t<T2, Tn, (Tn)3, (Tn)3>& F) {
    auto awSqr = aw * aw;
    auto dM = diag_mul(mul(mul(fiber_dir.transpose(), F), fiber_dir), awSqr);
    return 2 * 3 * dM.trace() / aw.squaredNorm();
  }

  template <typename T0, typename T1, typename Tn>
  constexpr auto eval_I1_delta_deriv(const vec_t<T0, Tn, (Tn)3>& aw,
                                     const vec_t<T0, Tn, (Tn)3, (Tn)3>& fiber_dir) {
    auto awSqr = aw * aw;
    auto dM = mul(diag_mul(fiber_dir, awSqr), fiber_dir.transpose());
    return 2 * 3 * dM / aw.squaredNorm();
  }

}  // namespace zs