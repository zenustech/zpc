#pragma once
#include <tuple>

#include "zensim/tpls/gcem.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum struct constitutive_model_e : char {
    EquationOfState = 0,
    NeoHookean,
    FixedCorotated,
    VonMisesFixedCorotated,
    DruckerPrager,
    NACC,
    NumConstitutiveModels
  };

  template <typename T> std::tuple<T, T> lame_parameters(T E, T nu) {
    T mu = 0.5 * E / (1 + nu);
    T lam = E * nu / ((1 + nu) * (1 - 2 * nu));
    return std::make_tuple(mu, lam);
  }

  struct MaterialConfig {
    float rho{1e3};
    float volume{1};
    int dim{3};
  };
  struct EquationOfStateConfig : MaterialConfig {
    float bulk{4e4f};
    float gamma{7.15f};
    float viscosity{0.01f};
  };
  struct NeoHookeanConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
  };
  struct FixedCorotatedConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
  };
  struct VonMisesFixedCorotatedConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
    float yieldStress{240e6};
  };
  struct DruckerPragerConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
    float logJp0{0.f};
    float fa{30.f};  ///< friction angle
    float cohesion{0.f};
    float beta{1.f};
    float yieldSurface{0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f)};
  };
  struct NACCConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
    float logJp0{-0.01f};  ///< alpha
    float fa{45.f};
    float xi{0.8f};  ///< hardening factor
    float beta{0.5f};
    constexpr float bulk() const noexcept {
      return 2.f / 3.f * (E / (2 * (1 + nu))) + (E * nu / ((1 + nu) * (1 - 2 * nu)));
    }
    constexpr float mohrColumbFriction() const noexcept {
      // 0.503599787772409
      float sin_phi = gcem::sin(fa);
      return gcem::sqrt(2.f / 3.f) * 2.f * sin_phi / (3.f - sin_phi);
    }
    constexpr float M() const noexcept {
      // 1.850343771924453
      return mohrColumbFriction() * dim / gcem::sqrt(2.f / (6.f - dim));
    }
    constexpr float Msqr() const noexcept {
      // 3.423772074299613
      auto ret = M();
      return ret * ret;
    }
  };

  using ConstitutiveModelConfig
      = variant<EquationOfStateConfig, NeoHookeanConfig, FixedCorotatedConfig,
                VonMisesFixedCorotatedConfig, DruckerPragerConfig, NACCConfig>;

}  // namespace zs