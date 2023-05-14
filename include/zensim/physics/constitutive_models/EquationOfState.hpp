#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  template <typename T = float> struct EquationOfState
      : IsotropicConstitutiveModelInterface<EquationOfState<T>> {
    using base_t = IsotropicConstitutiveModelInterface<EquationOfState<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type bulk, gamma;  // gamma is 7

    constexpr EquationOfState() noexcept = default;
    constexpr EquationOfState(value_type E, value_type nu, value_type gamma = 7) noexcept
        : bulk{E / (3 - 6 * nu)}, gamma{7} {}

    // psi
    template <typename V> constexpr V psi(V J) const noexcept {
      const auto J2 = J * J;
      const auto J6 = J2 * J2 * J2;
      // return -bulk * ((T)1 / (T)(-6) / J6 - J);
      return bulk * ((V)1 / (V)6 / J6 + J);
    }
    // first_piola
    template <typename V> constexpr V first_piola(V J) const noexcept {
      const auto J2 = J * J;
      const auto J4 = J2 * J2;
      const auto J7 = J4 * J2 * J;
      // return -bulk * ((T)1 / J7 - (T)1);
      return bulk - bulk / J7;
    }
    // first_piola_derivative
    template <typename V> constexpr V first_piola_derivative(V J) const noexcept {
      const auto J2 = J * J;
      const auto J4 = J2 * J2;
      const auto J8 = J4 * J4;
      // return bulk * ((T)1 / J8 * (T)7);
      return bulk * (V)7 / J8;
    }
  };

}  // namespace zs