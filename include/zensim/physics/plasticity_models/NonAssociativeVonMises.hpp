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

    constexpr NonAssociativeVonMises(value_type tauY, value_type alpha,
                                     value_type hardeningCoeff) noexcept
        : tauY{tauY}, alpha{alpha}, hardeningCoeff{hardeningCoeff} {}

    // project_strain
    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 1, (VecT::template range<0>() <= 3),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_sigma(VecInterface<VecT>& S, const Model& model) const noexcept {
      using value_type = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template range<0>();

      // Compute scaled tauY
      auto scaledTauY = math::sqrtNewtonRaphson((value_type)2 / ((value_type)6 - (value_type)dim))
                        * (tauY + hardeningCoeff * alpha);
      // Compute B hat trial based on the singular values from F^trial --> sigma^2
      // are the singular vals of be hat trial
      auto B_hat_trial = S * S;
      // mu * J^(-2/dim)
      auto scaledMu = static_cast<const Model&>(model).mu
                      * gcem::pow(S.prod(), -(value_type)2 / (value_type)dim);
      // Compute s hat trial using b hat trial
      auto s_hat_trial = scaledMu * B_hat_trial.deviatoric();
      // Compute s hat trial's L2 norm (since it appears in our expression for
      // y(tau)
      auto s_hat_trial_norm = s_hat_trial.norm();
      // Compute y using sqrt(s:s) and scaledTauY
      auto y = s_hat_trial_norm - scaledTauY;

      if (y < 1e-4) return;  // within the yield surface

      auto z = y / scaledMu;
      // Compute new Bhat
      auto B_hat_new = B_hat_trial - ((z / s_hat_trial_norm) * s_hat_trial);
      // Now compute new sigmas by taking sqrt of B hat new, then set strain to be
      // this new F^{n+1} value
      for (int i = 0; i != dim; ++i) S(i) = math::sqrtNewtonRaphson(B_hat_new(i));
    }

    template <typename VecT, template <typename> class ModelInterface, typename Model,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_strain(VecInterface<VecT>& F, const Model& model) const noexcept {
      auto [U, S, V] = math::svd(F);
      do_project_sigma(S, model);
      F = diag_mul(U, S) * V.transpose();
    }
  };

}  // namespace zs