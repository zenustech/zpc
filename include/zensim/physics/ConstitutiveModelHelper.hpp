#pragma once
#include "zensim/math/Vec.h"

namespace zs {

  /// I, dI/dF, d2I/dF2

  // I1
  template <int Opt = 1, typename T0, typename T1, typename Tn, Tn dim>
  constexpr auto eval_I1_deriv_hessian(const vec_t<T0, Tn, dim, dim>& F,
                                       const vec_t<T1, Tn, dim>& a) {
    if constexpr (Opt == 1)
      return std::make_tuple((F * a).l2NormSqr());
    else {
      auto dyadicAa = dyadic_prod(a, a);
      if constexpr (Opt == 2)
        return std::make_tuple((F * a).l2NormSqr(), 2 * (F * dyadicAa).vectorize());
      else {
        using R = math::op_result_t<T0, T1>;
        auto H = vec_t<R, Tn, dim * dim, dim * dim>::zeros();
        for (Tn i = 0; i != dim; ++i)
          for (Tn j = 0, bi = i * dim; j != dim; ++j) {
            const auto v = dyadicAa(i, j);
            for (Tn d = 0, bj = j * dim; d != dim; ++d) H(bi + d, bj + d) = v;
          }
        return std::make_tuple((F * a).l2NormSqr(), 2 * vectorize(F * dyadicAa), H * 2);
      }
    }
  }

  // I2
  template <int Opt = 1, typename T0, typename T1, typename T2, typename Tn, int dim>
  constexpr auto eval_I2_deriv_hessian(const vec_t<T0, Tn, dim, dim>& F,
                                       const vec_t<T1, Tn, dim>& a1, const vec_t<T2, Tn, dim>& a2) {
    if constexpr (Opt == 1)
      return std::make_tuple((F * a1).dot((F * a2)));
    else {
      auto dyadicA12 = dyadic_prod(a1, a2);
      dyadicA12 = (dyadicA12 + dyadicA12.transpose());
      if constexpr (Opt == 2)
        return std::make_tuple((F * a1).dot(F * a2), (F * dyadicA12).vectorize());
      else {
        using R = math::op_result_t<T0, T1, T2>;
        auto H = vec_t<R, Tn, dim * dim, dim * dim>::zeros();
        for (Tn i = 0; i != dim; ++i)
          for (Tn j = 0, bi = i * dim; j != dim; ++j) {
            const auto v = dyadicA12(i, j);
            for (Tn d = 0, bj = j * dim; d != dim; ++d) H(bi + d, bj + d) = v;
          }
        return std::make_tuple((F * a1).dot(F * a2), (F * dyadicA12).vectorize(), H);
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
        auto asym = cross_matrix(f0);
        for (Tn i = 0; i != 3; ++i)
          for (Tn j = 0; j != 3; ++j) {
            H(6 + i, 3 + j) = asym(i, j);
            H(3 + i, 6 + j) = -asym(i, j);
          }
        asym = cross_matrix(f1);
        for (Tn i = 0; i != 3; ++i)
          for (Tn j = 0; j != 3; ++j) {
            H(6 + i, j) = -asym(i, j);
            H(i, 6 + j) = asym(i, j);
          }
        asym = cross_matrix(f2);
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
                               const vec_t<T2, Tn, (Tn)3, (Tn)3>& F) noexcept {
    auto awSqr = aw * aw;
    auto dM = diag_mul(fiber_dir.transpose() * F * fiber_dir, awSqr);
    return 2 * 3 * trace(dM) / aw.l2NormSqr();
  }

  template <typename T0, typename T1, typename Tn>
  constexpr auto eval_I1_delta_deriv(const vec_t<T0, Tn, (Tn)3>& aw,
                                     const vec_t<T1, Tn, (Tn)3, (Tn)3>& fiber_dir) noexcept {
    auto awSqr = aw * aw;
    auto dM = diag_mul(fiber_dir, awSqr) * fiber_dir.transpose();
    return 2 * 3 * dM / aw.l2NormSqr();
  }

  template <typename T, typename Tn>
  constexpr auto eval_dFact_dF(const vec_t<T, Tn, (Tn)3, (Tn)3>& actInv) noexcept {
    auto M = vec_t<T, Tn, (Tn)9, (Tn)9>::zeros();

    M(0, 0) = M(1, 1) = M(2, 2) = actInv(0, 0);
    M(3, 0) = M(4, 1) = M(5, 2) = actInv(0, 1);
    M(6, 0) = M(7, 1) = M(8, 2) = actInv(0, 2);

    M(0, 3) = M(1, 4) = M(2, 5) = actInv(1, 0);
    M(3, 3) = M(4, 4) = M(5, 5) = actInv(1, 1);
    M(6, 3) = M(7, 4) = M(8, 5) = actInv(1, 2);

    M(0, 6) = M(1, 7) = M(2, 8) = actInv(2, 0);
    M(3, 6) = M(4, 7) = M(5, 8) = actInv(2, 1);
    M(6, 6) = M(7, 7) = M(8, 8) = actInv(2, 2);

    return M;
  }

}  // namespace zs