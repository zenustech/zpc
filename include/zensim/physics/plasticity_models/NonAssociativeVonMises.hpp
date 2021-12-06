#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  // ref:
  // https://github.com/penn-graphics-research/ziran2020/blob/5c52c26936b30d445e668cd7ef700bcbf9e6ff7e/Lib/Ziran/Physics/PlasticityApplier.cpp#L197
  template <typename T = float> struct NonAssociativeVonMises
      : PlasticityModelInterface<NonAssociativeVonMises<T>> {
    using base_t = PlasticityModelInterface<NonAssociativeVonMises<T>>;
    using value_type = T;

    static_assert(std::is_floating_point_v<value_type>, "value type should be floating point");

    value_type tauY, alpha, hardeningCoeff;

    constexpr NonAssociativeVonMises(value_type tauY, value_type alpha = 0,
                                     value_type hardeningCoeff = 0) noexcept
        : tauY{tauY}, alpha{alpha}, hardeningCoeff{hardeningCoeff} {}

    // project_strain
    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 1, VecT::template get_range<0>() <= 3,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_sigma(VecInterface<VecT>& S, const Model& model) const noexcept {
      using value_type = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template get_range<0>();

      // Compute scaled tauY
      auto scaledTauY = math::sqrtNewtonRaphson((value_type)2 / ((value_type)6 - (value_type)dim))
                        * (tauY + hardeningCoeff * alpha);
      // Compute B hat trial based on the singular values from F^trial --> sigma^2
      // are the singular vals of be hat trial
      auto B_hat_trial = S * S;
#if 1
      // mu * J^(-2/dim)
      auto scaledMu = static_cast<const Model&>(model).mu
                      * zs::pow(S.prod(), -(value_type)2 / (value_type)dim);
      // Compute s hat trial using b hat trial
      auto s_hat_trial = scaledMu * B_hat_trial.deviatoric();
      // Compute s hat trial's L2 norm (since it appears in our expression for
      // y(tau)
      auto s_hat_trial_norm = s_hat_trial.norm();
#else
      // auto s_hat_trial = model.first_piola(S);
      auto dE_dsigma = model.dpsi_dsigma(S);
      auto s_hat_trial = dE_dsigma * S;
      auto shifted = s_hat_trial;
      if constexpr (dim == 2)
        std::swap(shifted(0), shifted(1));
      else if constexpr (dim == 3) {
        shifted(0) = s_hat_trial(1);
        shifted(1) = s_hat_trial(2);
        shifted(2) = s_hat_trial(0);
      }
      auto s_hat_trial_norm
          = math::sqrtNewtonRaphson((s_hat_trial - shifted).l2NormSqr() * (value_type)0.5);
#endif
      // Compute y using sqrt(s:s) and scaledTauY
      auto y = s_hat_trial_norm - scaledTauY;

      if (y < 1e-4) return;  // within the yield surface

      auto z = y / scaledMu;
      // auto z = y / B_hat_trial.deviatoric();
      // Compute new Bhat
      auto B_hat_new = B_hat_trial - ((z / s_hat_trial_norm) * s_hat_trial);

      // printf("S (%f) (%f, %f, %f) -> %f, %f, %f\n", s_hat_trial_norm, S(0), S(1), S(2),
      // B_hat_new(0), B_hat_new(1),
      // B_hat_new(2));
      // Now compute new sigmas by taking sqrt of B hat new, then set strain to be
      // this new F^{n+1} value
      for (int i = 0; i != dim; ++i) S(i) = math::sqrtNewtonRaphson(B_hat_new(i));
    }

    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 2, VecT::template get_range<0>() <= 3,
                            VecT::template get_range<0>() == VecT::template get_range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_strain(VecInterface<VecT>& F, const Model& model) const noexcept {
      auto [U, S, V] = math::svd(F);
      do_project_sigma(S, model);
      F = diag_mul(U, S) * V.transpose();
    }

    template <typename VecT, typename VecTV, typename Model,
              enable_if_all<VecT::dim == 2, VecT::template get_range<0>() <= 3,
                            VecT::template get_range<0>() == VecT::template get_range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto project_strain(VecInterface<VecT>& F, const Model& model,
                                  const VecInterface<VecTV>& oldS,
                                  typename VecT::value_type dt) const noexcept {
      auto [U, S, V] = math::svd(F);

      using value_type = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template get_range<0>();

      // Compute scaled tauY
      // auto P = (value_type)0.33;
      auto P = (value_type)10;
      // auto C = (value_type)1.2;
      auto C = (value_type)100;
      auto ys = tauY * (1 + zs::pow(((S - oldS) / oldS).norm() / dt / C, (value_type)1 / P));
      auto scaledTauY = zs::sqrt((value_type)2 / ((value_type)6 - (value_type)dim))
                        * (ys /*tauY*/ + hardeningCoeff * alpha);
      // Compute B hat trial based on the singular values from F^trial --> sigma^2
      // are the singular vals of be hat trial
      auto B_hat_trial = S * S;
      // mu * J^(-2/dim)
      auto scaledMu = static_cast<const Model&>(model).mu
                      * zs::pow(S.prod(), -(value_type)2 / (value_type)dim);
      // Compute s hat trial using b hat trial
      auto s_hat_trial = scaledMu * B_hat_trial.deviatoric();
      // Compute s hat trial's L2 norm (since it appears in our expression for
      // y(tau)
      auto s_hat_trial_norm = s_hat_trial.norm();

      // Compute y using sqrt(s:s) and scaledTauY
      auto y = s_hat_trial_norm - scaledTauY;

#if 0
      if (ys != tauY)
        printf("sigma_0: %f -> sigma_yield: %f, candidate: %f, scaledMu: %f\n", tauY, ys,
               s_hat_trial_norm, scaledMu);
#endif

      if (y < 1e-4) return S;  // within the yield surface

      printf("sigma_0: %f -> sigma_yield: %f, candidate: %f, scaledMu: %f\n", tauY, ys,
             s_hat_trial_norm, scaledMu);

      auto z = y / scaledMu;
      // auto z = y / B_hat_trial.deviatoric();
      // Compute new Bhat
      auto B_hat_new = B_hat_trial - ((z / s_hat_trial_norm) * s_hat_trial);

      // printf("S (%f) (%f, %f, %f) -> %f, %f, %f\n", s_hat_trial_norm, S(0), S(1), S(2),
      // B_hat_new(0), B_hat_new(1),
      // B_hat_new(2));
      // Now compute new sigmas by taking sqrt of B hat new, then set strain to be
      // this new F^{n+1} value
      for (int i = 0; i != dim; ++i) S(i) = math::sqrtNewtonRaphson(B_hat_new(i));

      F = diag_mul(U, S) * V.transpose();
      return S;
    }
  };

}  // namespace zs