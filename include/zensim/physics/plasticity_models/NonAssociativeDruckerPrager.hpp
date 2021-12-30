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

    static_assert(std::is_floating_point_v<value_type>, "value type should be floating point");

    // h0 > h3 >= 0
    // h1, h2 >= 0
    value_type h0, h1, h2, h3;

    constexpr NonAssociativeDruckerPrager(value_type h0 = 35, value_type h1 = 9,
                                          value_type h2 = 0.2, value_type h3 = 10) noexcept
        : h0{h0}, h1{h1}, h2{h2}, h3{h3} {}

    constexpr value_type yieldSurface(value_type q) const noexcept {
      constexpr value_type coeff = math::sqrtNewtonRaphson((value_type)2 / (value_type)3);
      const auto frictionAngle = h0 + (h1 * q - h3) * zs::exp(-h2 * q);
      const auto sinFa = zs::sin(frictionAngle);
      return coeff * (sinFa + sinFa) / ((value_type)3 - sinFa);
    }

    template <typename VecT, typename Model, typename T,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr typename VecT::value_type do_project_sigma(VecInterface<VecT>& S, const Model& model,
                                                         T alpha) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      auto eps = S.log();
      auto eps_trace = eps.sum();
      auto dev_eps = eps - eps_trace / (value_type)dim;  // equivalent to eps.deviatoric()
      const auto dev_eps_norm = dev_eps.norm();
      if (math::near_zero(dev_eps_norm) || eps_trace > 0) {  // Case II (all stress removed)
        S = S.ones();
        return eps.norm();
      }
      const auto _2mu = model.mu + model.mu;
      const auto delta_gamma = dev_eps_norm + (dim * model.lam + _2mu) / _2mu * eps_trace * alpha;
      if (delta_gamma <= 0)  // Case I (already within yield surface)
        return 0;
      auto H
          = eps
            - delta_gamma / dev_eps_norm * dev_eps;  // Case III (project to cone, preserve volume)
      S = H.exp();
      return delta_gamma;
    }
  };

}  // namespace zs