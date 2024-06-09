#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  template <typename T = float> struct AnisotropicArap
      : InvariantConstitutiveModelInterface<AnisotropicArap<T>> {
    using base_t = InvariantConstitutiveModelInterface<AnisotropicArap<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type mu;

    AnisotropicArap() noexcept = default;
    constexpr AnisotropicArap(value_type E, value_type nu, value_type fiberStrength = 10) noexcept {
      value_type lam{};
      zs::tie(mu, lam) = lame_parameters(E, nu);
      mu *= fiberStrength;
    }

    template <typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto I4_sign(const VecInterface<VecTM>& F,
                           const VecInterface<VecTV>& a) const noexcept {
      const auto I4 = zs::get<0>(base_t::I_wrt_F_a<4, 0>(F, a));
      return math::near_zero(I4) ? (value_type)0 : (I4 > 0 ? (value_type)1 : (value_type)-1);
    }
    template <typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto do_psi(const VecInterface<VecTM>& F,
                          const VecInterface<VecTV>& a) const noexcept {
      const auto v = zs::sqrt(zs::get<0>(base_t::I_wrt_F_a<5, 0>(F, a))) - I4_sign(F, a);
      return (value_type)0.5 * mu * v * v;
    }
    template <typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto do_first_piola(const VecInterface<VecTM>& F,
                                  const VecInterface<VecTV>& a) const noexcept {
      const auto A = dyadic_prod(a, a);
      const auto coeff
          = (value_type)0.5 * mu
            * ((value_type)1 - I4_sign(F, a) / zs::sqrt(zs::get<0>(base_t::I_wrt_F_a<5, 0>(F, a))));
      return coeff * F * A;
    }
    template <typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto do_first_piola_derivative(const VecInterface<VecTM>& F,
                                             const VecInterface<VecTV>& a) const noexcept {
      typename VecTM::value_type I5{};
      typename base_t::template gradient_t<VecTM> g5{};
      typename base_t::template hessian_t<VecTM> H5{};
      zs::tie(I5, g5, H5) = base_t::I_wrt_F_a<5, 2>(F, a);
      const auto S_I4 = I4_sign(F, a);
      const auto Sqrt_I5 = zs::sqrt(I5);

      auto alpha = (1 - S_I4 / Sqrt_I5);
      auto beta = (S_I4 + S_I4) / (Sqrt_I5 * I5);
      alpha = alpha < 0 ? 0 : alpha;
      beta = beta < 0 ? 0 : beta;

      // return mu
      //        * ((1 - S_I4 / Sqrt_I5) * H5 + (S_I4 + S_I4) / (Sqrt_I5 * I5) * dyadic_prod(g5, g5));

      return mu
             * (alpha * H5 + beta * dyadic_prod(g5, g5));
    }
  };

}  // namespace zs