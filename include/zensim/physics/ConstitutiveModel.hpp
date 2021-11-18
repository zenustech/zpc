#pragma once
#include <tuple>

#include "zensim/math/Vec.h"
#include "zensim/math/curve/InterpolationKernel.hpp"
#include "zensim/math/matrix/SVD.hpp"
#include "zensim/tpls/fmt/core.h"
#include "zensim/tpls/gcem/gcem.hpp"
#include "zensim/types/Polymorphism.h"

namespace zs {

  enum struct constitutive_model_e : char {
    EquationOfState = 0,
    NeoHookean,
    FixedCorotated,
    StvkWithHencky,
    VonMisesFixedCorotated,
    DruckerPrager,
    NACC,
    NumConstitutiveModels
  };

  enum struct plasticity_model_e : char {
    NonAssociativeVonMises = 0,
    VonMisesCapped,
    NonAssociativeCamClay,
    NonAssociativeDruckerPrager,
    DruckerPrager,
    SnowPlascitity,
    NumPlasticityModels
  };

  template <typename T> constexpr std::tuple<T, T> lame_parameters(T E, T nu) {
    T mu = 0.5 * E / (1 + nu);
    T lam = E * nu / ((1 + nu) * (1 - 2 * nu));
    return std::make_tuple(mu, lam);
  }

  template <typename Model> struct IsotropicConstitutiveModelInterface {
#define DECLARE_ISOTROPIC_CONSTITUTIVE_MODEL_INTERFACE_ATTRIBUTES \
  using value_type = typename Model::value_type;

    using model_type = Model;
    template <typename VecT> using vec_type = typename VecT::template variant_vec<
        typename VecT::value_type,
        integer_seq<typename VecT::index_type, VecT::template range<0>()>>;
    template <typename VecT> using mat_type = typename VecT::template variant_vec<
        typename VecT::value_type, integer_seq<typename VecT::index_type, VecT::template range<0>(),
                                               VecT::template range<0>()>>;

    // psi_sigma
    template <typename VecT, enable_if_all<VecT::dim == 1, (VecT::template range<0>() <= 3),
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) psi_sigma(const VecInterface<VecT>& S) const noexcept {
      return static_cast<const Model*>(this)->do_psi_sigma(S);
    }
    // dpsi_dsigma
    template <typename VecT, enable_if_all<VecT::dim == 1, (VecT::template range<0>() <= 3),
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) dpsi_dsigma(const VecInterface<VecT>& S) const noexcept {
      return static_cast<const Model*>(this)->do_dpsi_dsigma(S);
    }
    // d2psi_dsigma2
    template <typename VecT, enable_if_all<VecT::dim == 1, (VecT::template range<0>() <= 3),
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) d2psi_dsigma2(const VecInterface<VecT>& S) const noexcept {
      return static_cast<const Model*>(this)->do_d2psi_dsigma2(S);
    }
    // Bij_neg_coeff
    template <typename VecT,
              enable_if_all<VecT::dim == 1,
                            (VecT::template range<0>() == 2 || VecT::template range<0>() == 3),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) Bij_neg_coeff(const VecInterface<VecT>& S) const noexcept {
      return static_cast<const Model*>(this)->do_Bij_neg_coeff(S);
    }

    // details (default impls)
    template <typename VecT>
    constexpr typename VecT::value_type do_psi_sigma(const VecInterface<VecT>& S) const noexcept {
      return (typename VecT::value_type)0;
    }
    template <typename VecT>
    constexpr auto do_dpsi_dsigma(const VecInterface<VecT>& S) const noexcept {
      return vec_type<VecT>::zeros();
    }
    template <typename VecT>
    constexpr auto do_d2psi_dsigma2(const VecInterface<VecT>& S) const noexcept {
      return mat_type<VecT>::zeros();
    }
    template <typename VecT>
    constexpr auto do_Bij_neg_coeff(const VecInterface<VecT>& S) const noexcept {
      using RetT = typename VecT::template variant_vec<
          typename VecT::value_type,
          integer_seq<typename VecT::index_type, (VecT::template range<0>() == 3 ? 3 : 1)>>;
      return RetT::zeros();
    }

    // psi
    template <typename VecT, enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                                           VecT::template range<0>() == VecT::template range<1>(),
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto psi(const VecInterface<VecT>& F) const noexcept {
      auto [U, S, V] = math::svd(F);
      return static_cast<const Model*>(this)->psi_sigma(S);
    }
    // first piola
    template <typename VecT, enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                                           VecT::template range<0>() == VecT::template range<1>(),
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto first_piola(const VecInterface<VecT>& F) const noexcept {
      auto [U, S, V] = math::svd(F);
      auto dE_dsigma = static_cast<const Model*>(this)->dpsi_dsigma(S);
      return diag_mul(U, dE_dsigma) * V.transpose();
    }
    // first piola derivative
    template <typename VecT, enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                                           VecT::template range<0>() == VecT::template range<1>(),
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto first_piola_derivative(const VecInterface<VecT>& F) const noexcept {
      using T = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template range<0>();

      auto [U, S, V] = math::svd(F);
      auto dE_dsigma = static_cast<const Model*>(this)->dpsi_dsigma(S);
      // A
      auto d2E_dsigma2 = static_cast<const Model*>(this)->d2psi_dsigma2(S);
      // Bij
      using MatB = typename VecT::template variant_vec<T, integer_seq<Ti, 2, 2>>;
      auto ComputeBij = [&dE_dsigma, &S = S,
                         Bij_left_coeffs = Bij_neg_coeff(S)](int i) -> MatB {  // i -> i, i + 1
        int j = (i + 1) % dim;
        T leftCoeff = Bij_left_coeffs[i];
        T rightDenom = gcem::max(S[i] + S[j], (T)1e-6);  // prevents division instability
        T rightCoeff = (dE_dsigma[i] + dE_dsigma[j]) / (rightDenom + rightDenom);
        return MatB{leftCoeff + rightCoeff, leftCoeff - rightCoeff, leftCoeff - rightCoeff,
                    leftCoeff + rightCoeff};
      };
      using MatH = typename VecT::template variant_vec<T, integer_seq<Ti, dim * dim, dim * dim>>;
      auto M = MatH::zeros();
      MatH dPdF{};

      if constexpr (is_same_v<typename VecT::dims, sindex_seq<3, 3>>) {
        auto B0 = ComputeBij(0) /*B12*/, B1 = ComputeBij(1) /*B23*/, B2 = ComputeBij(2) /*B13*/;
        // fmt::print("B0: [{}, {}], [{}, {}]\n", B0(0, 0), B0(0, 1), B0(1, 0), B0(1, 1));
        // fmt::print("B1: [{}, {}], [{}, {}]\n", B1(0, 0), B1(0, 1), B1(1, 0), B1(1, 1));
        // fmt::print("B2: [{}, {}], [{}, {}]\n", B2(0, 0), B2(0, 1), B2(1, 0), B2(1, 1));
        // [A_00, 0,      0] [0,      A_01, 0     ] [0,       0,      A_02]
        // [0,    B12_00, 0] [B12_01, 0,    0     ] [0,       0,      0   ]
        // [0,    0, B13_11] [0,      0,    0     ] [B13_10,  0,      0   ]
        // [0,    B12_10, 0] [B12_11, 0,    0     ] [0,       0,      0   ]
        // [A_10, 0,      0] [0,      A_11, 0     ] [0,       0,      A_12]
        // [0,    0,      0] [0,      0,    B23_00] [0,       B23_01, 0   ]
        // [0,    0, B13_01] [0,      0,    0     ] [B13_00,  0,      0   ]
        // [0,    0,      0] [0,      0,    B23_10] [0,       B23_11, 0   ]
        // [A_20, 0,      0] [0,      A_21, 0     ] [0,       0,      A_22]
        for (int ji = 0; ji != dim * dim; ++ji) {
          int j = ji / dim;
          int i = ji - j * dim;
          for (int sr = 0; sr <= ji; ++sr) {
            int s = sr / dim;
            int r = sr - s * dim;
            dPdF(ji, sr) = dPdF(sr, ji)
                = d2E_dsigma2(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0)
                  + d2E_dsigma2(0, 1) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1)
                  + d2E_dsigma2(0, 2) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2)
                  + d2E_dsigma2(1, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0)
                  + d2E_dsigma2(1, 1) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1)
                  + d2E_dsigma2(1, 2) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2)
                  + d2E_dsigma2(2, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0)
                  + d2E_dsigma2(2, 1) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1)
                  + d2E_dsigma2(2, 2) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2)
                  + B0(0, 0) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1)
                  + B0(0, 1) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0)
                  + B0(1, 0) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1)
                  + B0(1, 1) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0)
                  + B1(0, 0) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2)
                  + B1(0, 1) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1)
                  + B1(1, 0) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2)
                  + B1(1, 1) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1)
                  + B2(1, 1) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2)
                  + B2(1, 0) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0)
                  + B2(0, 1) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2)
                  + B2(0, 0) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);
          }
        }
      } else if constexpr (is_same_v<typename VecT::dims, sindex_seq<2, 2>>) {
        auto B = ComputeBij(0);
        for (int ji = 0; ji != dim * dim; ++ji) {
          int j = ji / dim;
          int i = ji - j * dim;
          for (int sr = 0; sr <= ji; ++sr) {
            int s = sr / dim;
            int r = sr - s * dim;
            dPdF(ji, sr) = dPdF(sr, ji)
                = d2E_dsigma2(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0)
                  + d2E_dsigma2(0, 1) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1)
                  + B(0, 0) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1)
                  + B(0, 1) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0)
                  + B(1, 0) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1)
                  + B(1, 1) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0)
                  + d2E_dsigma2(1, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0)
                  + d2E_dsigma2(1, 1) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1);
          }
        }
      } else if constexpr (is_same_v<typename VecT::dims, sindex_seq<1, 1>>) {
        dPdF(0, 0) = d2E_dsigma2(0, 0);  // U = V = [1]
      }
      return dPdF;
    }
  };

  template <typename Model> struct PlasticityModelInterface {
    using model_type = Model;

    // project_sigma
    template <typename VecT, typename... Args,
              enable_if_all<VecT::dim == 1, (VecT::template range<0>() <= 3),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) project_sigma(VecInterface<VecT>& S, Args&&... args) const noexcept {
      return static_cast<const Model*>(this)->do_project_sigma(S, FWD(args)...);
    }
    // project_strain
    template <typename VecT, typename... Args,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) project_strain(VecInterface<VecT>& F, Args&&... args) const noexcept {
      return static_cast<const Model*>(this)->do_project_strain(F, FWD(args)...);
    }

    // details (default impls)
    template <typename VecT, typename... Args>
    constexpr auto do_project_sigma(const VecInterface<VecT>& S, Args&&... args) const noexcept {
      typename VecT::template variant_vec<typename VecT::value_type, typename VecT::extents> r{};
      for (typename VecT::index_type i = 0; i != VecT::extent; ++i) r.val(i) = S.val(i);
      return r;
    }
    template <typename VecT, typename... Args>
    constexpr auto do_project_strain(const VecInterface<VecT>& F, Args&&... args) const noexcept {
      typename VecT::template variant_vec<typename VecT::value_type, typename VecT::extents> r{};
      for (typename VecT::index_type i = 0; i != VecT::extent; ++i) r.val(i) = F.val(i);
      return r;
    }
  };

  struct MaterialConfig {
    float rho{1e3};
    float volume{1};
    int dim{3};
  };
  struct EquationOfStateConfig : MaterialConfig {
    float bulk{4e4f};
    float gamma{7.15f};  ///< set to 7 by force
    float viscosity{0.f};
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
    bool volumeCorrection{true};
    float yieldSurface{0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f)};
  };
  struct NACCConfig : MaterialConfig {
    float E{5e4f};
    float nu{0.4f};
    float logJp0{-0.01f};  ///< alpha
    float fa{45.f};
    float xi{0.8f};  ///< hardening factor
    float beta{0.5f};
    bool hardeningOn{true};
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

  constexpr bool particleHasF(const ConstitutiveModelConfig& model) noexcept {
    return model.index() != 0;
  }
  constexpr bool particleHasJ(const ConstitutiveModelConfig& model) noexcept {
    return model.index() == 0;
  }

  inline void displayConfig(ConstitutiveModelConfig& config) {
    match(
        [](EquationOfStateConfig& config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("bulk {}, gamma {}, viscosity{}\n", config.bulk, config.gamma,
                     config.viscosity);
        },
        [](NeoHookeanConfig& config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}\n", config.E, config.nu);
        },
        [](FixedCorotatedConfig& config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}\n", config.E, config.nu);
        },
        [](VonMisesFixedCorotatedConfig& config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}, yieldStress {}\n", config.E, config.nu, config.yieldStress);
        },
        [](DruckerPragerConfig& config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}, logJp0 {}, fric_angle {}, cohesion {}, beta, yieldSurface {}\n",
                     config.E, config.nu, config.logJp0, config.fa, config.cohesion, config.beta,
                     config.yieldSurface);
        },
        [](NACCConfig& config) {
          fmt::print("rho {}, volume {}, dim {}\n", config.rho, config.volume, config.dim);
          fmt::print("E {}, nu {}, logJp0 {}, fric_angle {}, xi {}, beta {}, mohrColumbFric {}\n",
                     config.E, config.nu, config.logJp0, config.fa, config.xi, config.beta,
                     config.mohrColumbFriction());
        })(config);
  }

}  // namespace zs