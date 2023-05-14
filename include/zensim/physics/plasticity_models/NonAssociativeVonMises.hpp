#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  // ref:
  // https://github.com/penn-graphics-research/ziran2020/blob/5c52c26936b30d445e668cd7ef700bcbf9e6ff7e/Lib/Ziran/Physics/PlasticityApplier.cpp#L197
  template <typename T = float> struct NonAssociativeVonMises
      : PlasticityModelInterface<NonAssociativeVonMises<T>> {
    using base_t = PlasticityModelInterface<NonAssociativeVonMises<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type tauY, alpha, hardeningCoeff;

    constexpr NonAssociativeVonMises(value_type tauY, value_type alpha = 0,
                                     value_type hardeningCoeff = 0) noexcept
        : tauY{tauY}, alpha{alpha}, hardeningCoeff{hardeningCoeff} {}

    // project_strain
    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr bool do_project_sigma(VecInterface<VecT>& S, const Model& model) const noexcept {
      using value_type = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template range_t<0>::value;

      // Compute scaled tauY
      constexpr auto coeff
          = math::sqrtNewtonRaphson((value_type)2 / ((value_type)6 - (value_type)dim));
      auto scaledTauY = coeff * (tauY + hardeningCoeff * alpha);
      // Compute B hat trial based on the singular values from F^trial --> sigma^2
      // are the singular vals of be hat trial
      auto B_hat_trial = S * S;
      // mu * J^(-2/dim)
      auto scaledMu = model.mu * zs::pow(S.prod(), -(value_type)2 / (value_type)dim);
      // Compute s hat trial using b hat trial
      auto s_hat_trial = scaledMu * B_hat_trial.deviatoric();
      // Compute s hat trial's L2 norm (since it appears in our expression for
      // y(tau)
      auto s_hat_trial_norm = s_hat_trial.norm();
      // Compute y using sqrt(s:s) and scaledTauY
      auto y = s_hat_trial_norm - scaledTauY;

      if (y < 1e-4) return false;  // within the yield surface

      auto z = y / scaledMu;
      // auto z = y / B_hat_trial.deviatoric();
      // Compute new Bhat
      auto B_hat_new = B_hat_trial - ((z / s_hat_trial_norm) * s_hat_trial);

      // Now compute new sigmas by taking sqrt of B hat new, then set strain to be
      // this new F^{n+1} value
      S.assign(B_hat_new.sqrt());  // S = SS is not correct, since S is a VecInterface object.
      return true;
    }
  };

}  // namespace zs