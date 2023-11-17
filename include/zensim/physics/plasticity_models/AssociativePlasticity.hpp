#pragma once
#include "../ConstitutiveModel.hpp"
#include "zensim/math/MathUtils.h"

namespace zs {

  template <typename T = float> struct AssociativeVonMises
      : PlasticityModelInterface<AssociativeVonMises<T>> {
    using base_t = PlasticityModelInterface<AssociativeVonMises<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    /// vonmises
    // yielding criteria: f_vm(cauchy, k, alpha) = sqrt(3/2 * (cauchy - alpha)^T P (cauchy - alpha))
    // - (ys0 + h k) plastic flow: m_vm(cauchy, alpha) = 3 P (cauchy - alpha) / sqrt(6 * (cauchy -
    // alpha)^T P (cauchy - alpha))

    value_type ys0, h;  // initial yield surface, hardening modulus

    constexpr AssociativeVonMises(value_type ys0, value_type h = 0) noexcept : ys0{ys0}, h{h} {}

    template <typename VecT, auto... Ns> using vec_t =
        typename VecT::template variant_vec<typename VecT::value_type,
                                            integer_sequence<typename VecT::index_type, Ns...>>;
    // project_strain
    template <
        typename VecT, typename Model, 
        enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                      is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr bool do_project_sigma(VecInterface<VecT>& S, const Model& model) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      using mat_type = vec_t<VecT, dim, dim>;
      using vec_type = vec_t<VecT, dim>;
      using hessian_type = vec_t<VecT, dim + 1, dim + 1>;
      using residual_type = vec_t<VecT, dim + 1>;

      // [k] isotropic hardening (1)
      // [alpha] kinematic hardening (dim), position of the centroid of the yield function
      constexpr auto P = ([]() noexcept {  // only works in 3d
        constexpr auto dim = VecT::template range_t<0>::value;
        auto m = mat_type::constant((T)-1);
        for (int d = 0; d != dim; ++d) m(d, d) = (T)2;
        return m;
      })();
      auto principal_cauchy_stress
          = [&model](const auto& S) { return model.dpsi_dsigma(S) * S / S.prod(); };
      auto von_mises
          = [&P](const auto& sigma) { return zs::sqrt((value_type)0.5 * dot(sigma, P * sigma)); };

      auto sigma = principal_cauchy_stress(S);
      auto vm = von_mises(sigma);
      if (vm < 0) return false;  // already inside yield surface

      auto gradient = [&von_mises](const auto& sigma) {
        const auto vm = von_mises(sigma);
        return P * sigma / (vm + vm);
      };

      // dof
      residual_type dof{};
      for (Ti d = 0; d != dim; ++d)  // sigma -> sigmaTr
        dof[d] = sigma[d];
      dof[dim] = (T)0;

      auto eps = [&principal_cauchy_stress, &von_mises, &gradient, &dof, &S, ys0 = this->ys0]() {
        residual_type ret{};
        // sigma
        auto sigma = principal_cauchy_stress(S);
        auto m = gradient(sigma);
        for (Ti d = 0; d != dim; ++d) ret[d] = sigma[d] - (dof[d] + dof[dim] * m[d]);
        // criteria
        ret[dim] = von_mises(sigma) - ys0;
        return ret;
      };
      auto hessian_inverse = [&P, &principal_cauchy_stress, &von_mises, &gradient, &dof, &S]() {
        hessian_type H{};
        auto sigma = principal_cauchy_stress(S);
        auto fvm = von_mises(sigma);
        auto Psigma = P * sigma;
        auto dm_dsigma
            = (T)1 / (fvm + fvm) * (P - (T)0.5 / (fvm * fvm) * dyadic_prod(Psigma, Psigma));
        for (Ti i = 0; i != dim; ++i)
          for (Ti j = 0; j != dim; ++j) H(i, j) = dm_dsigma(i, j) + (i == j ? (T)1 : (T)0);

        auto m = gradient(sigma);
        for (Ti d = 0; d != dim; ++d) H(dim, d) = H(d, dim) = m(d);
        H(dim, dim) = (T)0;
        return inverse(H);
      };

      auto get_ce = [&model](auto& S) { return (T)1 / S.prod() * model.dpsi_dsigma(S); };
      auto residual = eps();
      auto ce = get_ce(S);

      int iter = 0;

      while (!residual.norm() < 1e-6) {
        // printf("iter: %d: \nresidual: %f; dof: %f, %f, %f, %f; S: %f, %f, %f\n", iter,
        // (float)residual.norm(), (float)dof[0], (float)dof[1], (float)dof[2], (float)dof[3],
        // (float)S[0], (float)S[1], (float)S[2]);

        dof = dof - hessian_inverse() * residual;
        for (Ti d = 0; d != dim; ++d) {
          S[d] = (T)0;
          for (Ti i = 0; i != dim; ++i) S[d] += dof[i] / ce(i);
        }

        // printf("\titer end: \ndof: %f, %f, %f, %f; S: %f, %f, %f\n", (float)dof[0],
        // (float)dof[1], (float)dof[2], (float)dof[3], (float)S[0], (float)S[1], (float)S[2]);

        ce = get_ce(S);
      }
      return true;
    }

  };

}  // namespace zs