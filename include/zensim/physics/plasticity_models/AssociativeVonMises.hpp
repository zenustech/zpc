#pragma once
#include "../ConstitutiveModel.hpp"
#include "zensim/math/MathUtils.h"

namespace zs {

  // ref:
  // An adaptive generalized interpolation material point method for simulating elastoplastidc
  // materials,
  // sec 4.2.2
  template <typename T = float> struct AssociativeVonMises
      : PlasticityModelInterface<AssociativeVonMises<T>> {
    using base_t = PlasticityModelInterface<AssociativeVonMises<T>>;
    using value_type = T;

    static_assert(std::is_floating_point_v<value_type>, "value type should be floating point");

    value_type initialStress;

    constexpr AssociativeVonMises(value_type initialStress) noexcept
        : initialStress{initialStress} {}

    // project_strain
    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 1, (VecT::template range<0>() <= 3),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_sigma(VecInterface<VecT>& S, const Model& model) const noexcept {
      using value_type = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template range<0>();
    }

    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_strain(VecInterface<VecT>& F, const Model& model) const noexcept {
      auto [U, S, V] = math::svd(F);
      do_project_sigma(S, model);
      F = diag_mul(U, S) * V.transpose();
    }

    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto project_strain(VecInterface<VecT>& F, const Model& model,
                                  int pi) const noexcept {
      auto [U_, S, V_] = math::svd(F);  // S is the principal strain
      const auto& U = U_;
      const auto& V = V_;

      if constexpr (VecT::extent == 9) {
        // using vec3 = RM_CVREF_T(S);
        using vec3 = vec<typename RM_CVREF_T(S)::value_type, 3>;
        using mat3 =
            typename VecT::template variant_vec<typename vec3::value_type, typename VecT::extents>;
        mat3 P{2, -1, -1, -1, 2, -1, -1, -1, 2};
        auto getResidual = [&model, &P, &U, &V, this](const auto& S) noexcept {
          auto cauchy = model.dpsi_dsigma(S) * S / S.prod();
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
          auto vonMisesStress = ::sqrt(0.5 * cauchy.dot(P * cauchy));
#else
          auto vonMisesStress = std::sqrt(0.5 * cauchy.dot(P * cauchy));
#endif
          return vonMisesStress - initialStress;
        };

        auto res = getResidual(S);

        if (res < 0) return S;
        res = gcem::abs(res);

        // d f_vm(stress) / d stress(lambda) = P stress / sqrt(2 * stress^T * P * stress)
        auto getDeriv = [&model, &P, &U, &V](const auto& S) noexcept {
          auto cauchy = model.dpsi_dsigma(S) * S / S.prod();
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
          return P * cauchy / ::sqrt(2 * cauchy.dot(P * cauchy));
#else
          return P * cauchy / std::sqrt(2 * cauchy.dot(P * cauchy));
#endif
        };
        vec3 Pflow = getDeriv(S);

#if 0
        if (pi == 0) {
          auto printvec = [&pi](const char* msg, auto&& v) {
            printf("pi[%d] %s (%f, %f, %f)\n", pi, msg, (float)v(0), (float)v(1), (float)v(2));
          };
          auto printmat = [&pi](const char* msg, auto&& m) {
            printf("pi[%d] %s (%f, %f, %f; %f, %f, %f; %f, %f, %f)\n", pi, msg, (float)m(0, 0),
                   (float)m(0, 1), (float)m(0, 2), (float)m(1, 0), (float)m(1, 1), (float)m(1, 2),
                   (float)m(2, 0), (float)m(2, 1), (float)m(2, 2));
          };
          printvec("pflow", Pflow);
        }
#endif

        // d f_vm(stress) / d stress * (d stress / d strain) * (d strain / d lambda)
        auto getJacobi = [&getDeriv, &model, &U, &V, &Pflow](auto&& S) noexcept {
          auto H = model.d2psi_dsigma2(S);  // d stress / d sigma
          return getDeriv(S).dot(H * Pflow);
        };

        value_type lambda{0};
        vec3 SeHat = S;
        int cnt = 0;

        if (pi == 0)
          printf("pi[%d] cur strain (%f, %f, %f), vms (%f) <--> ys (%f)\n", pi, SeHat(0), SeHat(1),
                 SeHat(2), getResidual(SeHat) + initialStress, initialStress);

        for (; gcem::abs(res) > 1e-6 && cnt < 10; ++cnt) {
          auto oldRes = res;
          auto Jac = getJacobi(SeHat);
          lambda = lambda - res / Jac;  // Newton's Method
          SeHat = SeHat - lambda * Pflow;
          res = getResidual(SeHat);
          if (pi == 0)
            printf("pi[%d] round %d, jacobi %f, residual: %f -> %f\n", pi, cnt, (float)Jac,
                   (float)oldRes, (float)res);
        }

        F = diag_mul(U, SeHat) * V.transpose();
        return SeHat;
      } else {
        // TODO
        F = diag_mul(U, S) * V.transpose();
        return S;
      }
    }
  };

}  // namespace zs