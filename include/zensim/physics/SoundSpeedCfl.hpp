#pragma once
#include "zensim/tpls/gcem/gcem.hpp"

namespace zs {

  template <class T>
  constexpr T evaluate_soundspeed_linear_elasticity(const T E, const T nu, const T rho) {
    return gcem::sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho));
  }

  template <class T> constexpr T evaluate_timestep_linear_elasticity(const T E, const T nu,
                                                                     const T rho, const T dx,
                                                                     const T cfl) {
    return cfl * dx / evaluate_soundspeed_linear_elasticity(E, nu, rho);
  }

}  // namespace zs