#pragma once
#include "../ConstitutiveModel.hpp"
#include "zensim/math/MathUtils.h"

namespace zs {

  template <typename T = float> struct SnowPlasticity
      : PlasticityModelInterface<SnowPlasticity<T>> {
    using base_t = PlasticityModelInterface<SnowPlasticity<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type thetaC, thetaS, xi;  // compression, stretch, hardening coefficient

    constexpr SnowPlasticity(value_type c = 2e-2, value_type s = 7.5e-3,
                             value_type xi = 10) noexcept
        : thetaC{c}, thetaS{s}, xi{xi} {}

    // project_strain
    template <typename VecT, typename Tp,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_project_sigma(VecInterface<VecT>& S, Tp mu0, Tp lam0) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      using Ti = typename VecT::index_type;

      auto J = S.prod();
      for (Ti d = 0; d != dim; ++d) S[d] = zs::max(zs::min(S[d], (T)1 + thetaS), (T)1 - thetaC);
      auto Jp = J / zs::max(S.prod(), (T)1e-6);
      const auto e = zs::exp(xi * ((T)1 - Jp));
      return zs::make_tuple(mu0 * e, lam0 * e);
    }
  };

}  // namespace zs