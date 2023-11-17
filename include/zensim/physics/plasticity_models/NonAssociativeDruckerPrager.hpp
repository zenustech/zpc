#pragma once
#include "../ConstitutiveModel.hpp"
#include "zensim/math/MathUtils.h"

namespace zs {

  // ref:
  // An adaptive generalized interpolation material point method for simulating elastoplastidc
  // materials,
  // sec 4.2.2
  template <typename T_ = float> struct NonAssociativeDruckerPrager
      : PlasticityModelInterface<NonAssociativeDruckerPrager<T_>> {
    using base_t = PlasticityModelInterface<NonAssociativeDruckerPrager<T_>>;
    using value_type = T_;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type tau0;

    NonAssociativeDruckerPrager(value_type frictionAngle = 35) {
      constexpr value_type coeff = math::sqrtNewtonRaphson((value_type)2 / (value_type)3);
      const auto sinFa = std::sin(frictionAngle / (value_type)180 * g_pi);
      tau0 = coeff * (sinFa + sinFa) / ((value_type)3 - sinFa);
    }

    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr bool do_project_sigma(VecInterface<VecT>& S, const Model& model) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      auto eps = S.log();
      auto eps_trace = eps.sum();
      auto dev_eps = eps - eps_trace / (value_type)dim;  // equivalent to eps.deviatoric()
      const auto dev_eps_norm = dev_eps.norm();
      if (math::near_zero(dev_eps_norm) || eps_trace > 0) {  // Case II (all stress removed)
        S.assign(S.ones());
        return true;
        // return eps.norm();
      }
      const auto _2mu = model.mu + model.mu;
      const auto delta_gamma = dev_eps_norm + (dim * model.lam + _2mu) / _2mu * eps_trace * tau0;
      if (delta_gamma <= 0)  // Case I (already within yield surface)
        return false;
      // return 0;
      auto H
          = eps
            - delta_gamma / dev_eps_norm * dev_eps;  // Case III (project to cone, preserve volume)
      S.assign(H.exp());
      return true;
      // return delta_gamma;
    }
  };

}  // namespace zs