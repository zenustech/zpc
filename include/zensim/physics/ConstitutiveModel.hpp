#pragma once
#include <tuple>

#include "zensim/math/Vec.h"
#include "zensim/math/curve/InterpolationKernel.hpp"
#include "zensim/math/matrix/SVD.hpp"
#include "zensim/tpls/fmt/core.h"
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
        integer_seq<typename VecT::index_type, VecT::template range_t<0>::value>>;
    template <typename VecT> using mat_type = typename VecT::template variant_vec<
        typename VecT::value_type,
        integer_seq<typename VecT::index_type, VecT::template range_t<0>::value,
                    VecT::template range_t<0>::value>>;

    // psi_sigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) psi_sigma(const VecInterface<VecT>& S) const noexcept {
      return static_cast<const Model*>(this)->do_psi_sigma(S);
    }
    // dpsi_dsigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) dpsi_dsigma(const VecInterface<VecT>& S) const noexcept {
      return static_cast<const Model*>(this)->do_dpsi_dsigma(S);
    }
    // d2psi_dsigma2
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) d2psi_dsigma2(const VecInterface<VecT>& S) const noexcept {
      return static_cast<const Model*>(this)->do_d2psi_dsigma2(S);
    }
    // Bij_neg_coeff
    template <typename VecT, enable_if_all<VecT::dim == 1,
                                           VecT::template range_t<0>::value == 2
                                               || VecT::template range_t<0>::value == 3,
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
          integer_seq<typename VecT::index_type, (VecT::template range_t<0>::value == 3 ? 3 : 1)>>;
      return RetT::zeros();
    }

    // psi
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto psi(const VecInterface<VecT>& F) const noexcept {
      auto [U, S, V] = math::svd(F);
      return static_cast<const Model*>(this)->psi_sigma(S);
    }
    // first piola
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto first_piola(const VecInterface<VecT>& F) const noexcept {
      auto [U, S, V] = math::svd(F);
      auto dE_dsigma = static_cast<const Model*>(this)->dpsi_dsigma(S);
      return diag_mul(U, dE_dsigma) * V.transpose();
    }
    // first piola derivative
    template <typename VecT, bool project_SPD = false,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto first_piola_derivative(const VecInterface<VecT>& F,
                                          wrapv<project_SPD> = {}) const noexcept {
      using T = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      constexpr int dim = VecT::template range_t<0>::value;

      auto [U, S, V] = math::svd(F);
      auto dE_dsigma = static_cast<const Model*>(this)->dpsi_dsigma(S);
      // A
      auto d2E_dsigma2 = static_cast<const Model*>(this)->d2psi_dsigma2(S);
      if constexpr (project_SPD) make_pd(d2E_dsigma2);
      // Bij
      using MatB = typename VecT::template variant_vec<T, integer_seq<Ti, 2, 2>>;
      auto ComputeBij = [&dE_dsigma, &S = S,
                         Bij_left_coeffs = Bij_neg_coeff(S)](int i) -> MatB {  // i -> i, i + 1
        constexpr int dim = VecT::template range_t<0>::value;
        int j = (i + 1) % dim;
        T leftCoeff = Bij_left_coeffs[i];
        T rightDenom = math::max(S[i] + S[j], (T)1e-6);  // prevents division instability
        T rightCoeff = (dE_dsigma[i] + dE_dsigma[j]) / (rightDenom + rightDenom);
        return MatB{leftCoeff + rightCoeff, leftCoeff - rightCoeff, leftCoeff - rightCoeff,
                    leftCoeff + rightCoeff};
      };
      using MatH = typename VecT::template variant_vec<T, integer_seq<Ti, dim * dim, dim * dim>>;
      MatH dPdF{};

      if constexpr (is_same_v<typename VecT::dims, sindex_seq<3, 3>>) {
        auto B0 = ComputeBij(0) /*B12*/, B1 = ComputeBij(1) /*B23*/, B2 = ComputeBij(2) /*B13*/;
        if constexpr (project_SPD) {
          make_pd(B0);
          make_pd(B1);
          make_pd(B2);
        }
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
        if constexpr (project_SPD) make_pd(B);
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

  template <typename Model> struct InvariantConstitutiveModelInterface {
#define DECLARE_INVARIANT_CONSTITUTIVE_MODEL_INTERFACE_ATTRIBUTES \
  using value_type = typename Model::value_type;

    using model_type = Model;

    template <typename VecT> using dim_t = typename VecT::template range_t<0>;
    template <typename VecT> using vec_type = typename VecT::template variant_vec<
        typename VecT::value_type, integer_seq<typename VecT::index_type, dim_t<VecT>::value>>;
    template <typename VecT> using mat_type = typename VecT::template variant_vec<
        typename VecT::value_type,
        integer_seq<typename VecT::index_type, dim_t<VecT>::value, dim_t<VecT>::value>>;

    template <typename VecT> using gradient_t = typename VecT::template variant_vec<
        typename VecT::value_type,
        integer_seq<typename VecT::index_type, dim_t<VecT>::value * dim_t<VecT>::value>>;
    template <typename VecT> using hessian_t = typename VecT::template variant_vec<
        typename VecT::value_type,
        integer_seq<typename VecT::index_type, dim_t<VecT>::value * dim_t<VecT>::value,
                    dim_t<VecT>::value * dim_t<VecT>::value>>;
    template <typename VecT, int deriv_order = 0> using pack_t = conditional_t<
        deriv_order == 0, std::tuple<typename VecT::value_type>,
        conditional_t<deriv_order == 1, std::tuple<typename VecT::value_type, gradient_t<VecT>>,
                      std::tuple<typename VecT::value_type, gradient_t<VecT>, hessian_t<VecT>>>>;

    // isotropic invariants
    // I_1 = tr(S)
    // I_2 = tr(F^T F)
    // I_3 = det(F)
    // I_i(F)
    template <int I, int deriv_order = 0, typename VecT,
              enable_if_all<VecT::dim == 2,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            VecT::template range_t<0>::value <= 3,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto I_wrt_F(const VecInterface<VecT>& F) const noexcept {
      constexpr auto dim = dim_t<VecT>::value;
      using ScalarT = typename VecT::value_type;
      using index_type = typename VecT::index_type;
      using GradientT = gradient_t<VecT>;
      using HessianT = hessian_t<VecT>;
      using RetT = pack_t<VecT, deriv_order>;

      using MatT = vec<ScalarT, dim, dim>;
      RetT ret{};
      ///  I1 (length)
      if constexpr (I == 0) {
        if constexpr (deriv_order == 0) {
          auto [R, S] = math::polar_decomposition(F);
          std::get<0>(ret) = trace(S);
        } else if constexpr (deriv_order == 1) {
          auto [R, S] = math::polar_decomposition(F);
          std::get<0>(ret) = trace(S);
          std::get<1>(ret) = vectorize(R);
        } else if constexpr (deriv_order == 2) {
          auto [U, S, V] = math::qr_svd(F);
          std::get<0>(ret) = S.sum();
          auto R = U * V.transpose();
          std::get<1>(ret) = vectorize(R);
          // auto Ssym = diag_mul(V, S) * V.transpose();
          auto& dRdF = std::get<2>(ret);
          dRdF = HessianT::zeros();
          constexpr auto sqrt2Inv = (ScalarT)1 / g_sqrt2;
          if constexpr (dim == 2) {
            constexpr MatT T0{0, -1, 1, 0};
            const auto vecQ0 = vectorize(sqrt2Inv * U * T0 * V.transpose());
            dRdF
                += (ScalarT)2 / math::max((S(0) + S(1)), (ScalarT)1e-6) * dyadic_prod(vecQ0, vecQ0);
          } else if constexpr (dim == 3) {
            constexpr MatT Ti[3] = {{0, -1, 0, 1, 0, 0, 0, 0, 0},
                                    {0, 0, 0, 0, 0, 1, 0, -1, 0},
                                    {0, 0, 1, 0, 0, 0, -1, 0, 0}};
            for (int d = 0; d != 3; ++d) {
              const auto vecQi = vectorize(sqrt2Inv * U * Ti[d] * V.transpose());
              dRdF += ((ScalarT)2 / math::max((S(d) + S(d + 1 == 3 ? 0 : d + 1)), (ScalarT)1e-6)
                       * dyadic_prod(vecQi, vecQi));
            }
          }
        }
      }
      ///  I2 (area)
      else if constexpr (I == 1) {
        std::get<0>(ret) = trace(F.transpose() * F);
        if constexpr (deriv_order > 0) {
          std::get<1>(ret) = vectorize(F + F);
          if constexpr (deriv_order > 1) {
            constexpr auto I9x9 = HessianT::identity();
            std::get<2>(ret) = I9x9 + I9x9;
          }
        }
      }
      ///  I3 (volume)
      else if constexpr (I == 2) {
        std::get<0>(ret) = determinant(F);
        auto f0 = col(F, 0);
        auto f1 = col(F, 1);
        auto f2 = col(F, 2);
        // gradient
        if constexpr (deriv_order > 0) {
          if constexpr (dim == 1)
            std::get<1>(ret) = GradientT{1};
          else if constexpr (dim == 2)
            std::get<1>(ret) = GradientT{F(1, 1), -F(0, 1), -F(1, 0), F(0, 0)};
          else if constexpr (dim == 3) {
            const auto f1f2 = cross(f1, f2);
            const auto f2f0 = cross(f2, f0);
            const auto f0f1 = cross(f0, f1);
            std::get<1>(ret) = GradientT{f1f2(0), f1f2(1), f1f2(2), f2f0(0), f2f0(1),
                                         f2f0(2), f0f1(0), f0f1(1), f0f1(2)};
          }
          // hessian
          if constexpr (deriv_order > 1) {
            if constexpr (dim == 1)
              std::get<2>(ret) = HessianT{0};
            else if constexpr (dim == 2)
              std::get<2>(ret) = HessianT{0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0};
            else if constexpr (dim == 3) {
              auto& H = std::get<2>(ret);
              H = HessianT::zeros();
              auto asym = cross_matrix(f0);
              for (index_type i = 0; i != 3; ++i)
                for (index_type j = 0; j != 3; ++j) {
                  H(6 + i, 3 + j) = asym(i, j);
                  H(3 + i, 6 + j) = -asym(i, j);
                }
              asym = cross_matrix(f1);
              for (index_type i = 0; i != 3; ++i)
                for (index_type j = 0; j != 3; ++j) {
                  H(6 + i, j) = -asym(i, j);
                  H(i, 6 + j) = asym(i, j);
                }
              asym = cross_matrix(f2);
              for (index_type i = 0; i != 3; ++i)
                for (index_type j = 0; j != 3; ++j) {
                  H(3 + i, j) = asym(i, j);
                  H(i, 3 + j) = -asym(i, j);
                }
            }
          }  // hessian
        }    // gradient
      }
      return ret;
    }

    // anisotropic invariants
    // I_4 = aT S a (mainly use its sign info)
    // I_5 = aT FT F a
    // I_i(F, a)  // a is (uniformed) fiber direction
    template <int I, int deriv_order = 0, typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            std::is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto I_wrt_F_a(const VecInterface<VecTM>& F,
                             const VecInterface<VecTV>& a) const noexcept {
      constexpr auto dim = dim_t<VecTM>::value;
      // using ScalarT = typename VecTM::value_type;
      using index_type = typename VecTM::index_type;
      // using GradientT = gradient_t<VecTM>;
      using HessianT = hessian_t<VecTM>;
      using RetT = pack_t<VecTM, deriv_order>;

      RetT ret{};
      ///  I4 = a^T S a
      if constexpr (I == 4) {
        auto [R, S] = math::polar_decomposition(F);
        std::get<0>(ret) = dot(a, S * a);
        static_assert(
            !(I == 4 && deriv_order > 0),
            "the author haven\'t figure it out yet how to compute derivative and hessian of I4.");
#if 0
        using MatT = vec<ScalarT, dim, dim>;
        if constexpr (deriv_order > 0) {
          if constexpr (dim == 2) {
            // dR_dF : Faa^T
            auto gi = (VT * a).sum() * (S(0) - S(1)) / math::max(S(0) + S(1), (ScalarT)1e-6) * U
                          * MatT{0, -1, 1, 0} * VT
                      + R * A;
            std::get<1>(ret) = vectorize(gi);
          }
        }
#endif
      }
      ///  I5 = a^T S^T S a
      if constexpr (I == 5) {
        const auto Fa = F * a;  // equal to (Sa)^T Sa
        std::get<0>(ret) = dot(Fa, Fa);
        if constexpr (deriv_order > 0) {
          const auto A = dyadic_prod(a, a);
          auto FA = F * A;
          std::get<1>(ret) = vectorize(FA + FA);
          if constexpr (deriv_order > 1) {
            auto& H = std::get<2>(ret);
            H = HessianT::zeros();
            for (index_type i = 0; i != dim; ++i)
              for (index_type j = 0; j != dim; ++j) {
                const auto _2aij = A(i, j) + A(i, j);
                for (index_type ii = i * dim, jj = j * dim, d = (index_type)0; d != dim; ++d)
                  H(ii + d, jj + d) = _2aij;
              }
          }
        }
      }
      return ret;
    }

    // psi_I
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value == 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) psi_I(const VecInterface<VecT>& Is) const noexcept {
      return static_cast<const Model*>(this)->do_psi_I(Is);
    }
    // dpsi_dI
    template <int I, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value == 3,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) dpsi_dI(const VecInterface<VecT>& Is) const noexcept {
      return static_cast<const Model*>(this)->template do_dpsi_dI<I>(Is);
    }
    // d2psi_dI2 -> dim [dimxdim] matrices
    template <int I, typename VecT,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value == 3,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) d2psi_dI2(const VecInterface<VecT>& Is) const noexcept {
      return static_cast<const Model*>(this)->template do_d2psi_dI2<I>(Is);
    }

    // details (default impls)
    template <typename VecT>
    constexpr typename VecT::value_type do_psi_I(const VecInterface<VecT>&) const noexcept {
      return (typename VecT::value_type)0;
    }
    template <int I, typename VecT>
    constexpr typename VecT::value_type do_dpsi_dI(const VecInterface<VecT>&) const noexcept {
      return (typename VecT::value_type)0;
    }
    template <int I, typename VecT>
    constexpr typename VecT::value_type do_d2psi_dI2(const VecInterface<VecT>&) const noexcept {
      return (typename VecT::value_type)0;
    }

    /// isotropic
    // psi
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto psi(const VecInterface<VecT>& F) const noexcept {
      typename VecT::template variant_vec<typename VecT::value_type,
                                          integer_seq<typename VecT::index_type, 3>>
          Is{};
      Is[0] = std::get<0>(I_wrt_F<0, 0>(F));
      Is[1] = std::get<0>(I_wrt_F<1, 0>(F));
      Is[2] = std::get<0>(I_wrt_F<2, 0>(F));
      return psi_I(Is);
    }
    // first piola
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto first_piola(const VecInterface<VecT>& F) const noexcept {
      // sum_i ((dPsi / dI_i) g_i)
      typename VecT::template variant_vec<typename VecT::value_type,
                                          integer_seq<typename VecT::index_type, 3>>
          Is{};
      gradient_t<VecT> gi[3]{};
      zs::tie(Is(0), gi[0]) = I_wrt_F<0, 1>(F);
      zs::tie(Is[1], gi[1]) = I_wrt_F<1, 1>(F);
      zs::tie(Is[2], gi[2]) = I_wrt_F<2, 1>(F);
      auto res = gradient_t<VecT>::zeros();
      res += dpsi_dI<0>(Is) * gi[0];
      res += dpsi_dI<1>(Is) * gi[1];
      res += dpsi_dI<2>(Is) * gi[2];
      constexpr auto dim = dim_t<VecT>::value;
      auto m
          = VecT::template variant_vec<typename VecT::value_type,
                                       integer_seq<typename VecT::index_type, dim, dim>>::zeros();
      // auto m = vec<typename VecT::value_type, dim, dim>::zeros();
      // gradient convention order: column-major
      for (typename VecT::index_type j = 0, no = 0; j != dim; ++j)
        for (typename VecT::index_type i = 0; i != dim; ++i) m(i, j) = res(no++);
      return m;
    }
    // first piola derivative
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto first_piola_derivative(const VecInterface<VecT>& F) const noexcept {
      // sum_i ((d2Psi / dI_i2) g_i g_i^T + ((dPsi / dI_i) H_i))
      typename VecT::template variant_vec<typename VecT::value_type,
                                          integer_seq<typename VecT::index_type, 3>>
          Is{};
      gradient_t<VecT> gi[3]{};
      hessian_t<VecT> Hi[3]{};
      zs::tie(Is[0], gi[0], Hi[0]) = I_wrt_F<0, 2>(F);
      zs::tie(Is[1], gi[1], Hi[1]) = I_wrt_F<1, 2>(F);
      zs::tie(Is[2], gi[2], Hi[2]) = I_wrt_F<2, 2>(F);
      auto dPdF = hessian_t<VecT>::zeros();
      dPdF += d2psi_dI2<0>(Is) * dyadic_prod(gi[0], gi[0]) + dpsi_dI<0>(Is) * Hi[0];
      dPdF += d2psi_dI2<1>(Is) * dyadic_prod(gi[1], gi[1]) + dpsi_dI<1>(Is) * Hi[1];
      dPdF += d2psi_dI2<2>(Is) * dyadic_prod(gi[2], gi[2]) + dpsi_dI<2>(Is) * Hi[2];
      return dPdF;
    }

    /// anisotropic
    // psi
    template <typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            std::is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto psi(const VecInterface<VecTM>& F, const VecInterface<VecTV>& a) const noexcept {
      return static_cast<const Model*>(this)->do_psi(F, a);
    }
    template <typename VecTM, typename VecTV>
    constexpr auto do_psi(const VecInterface<VecTM>&, const VecInterface<VecTV>&) const noexcept {
      return (typename VecTM::value_type)0;
    }
    // first piola
    template <typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            std::is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto first_piola(const VecInterface<VecTM>& F,
                               const VecInterface<VecTV>& a) const noexcept {
      return static_cast<const Model*>(this)->do_first_piola(F, a);
    }
    template <typename VecTM, typename VecTV>
    constexpr auto do_first_piola(const VecInterface<VecTM>&,
                                  const VecInterface<VecTV>&) const noexcept {
      return typename VecTM::template variant_vec<
          typename VecTM::value_type, integer_seq<typename VecTM::index_type, dim_t<VecTM>::value,
                                                  dim_t<VecTM>::value>>::zeros();
    }
    // first piola derivative
    template <typename VecTM, typename VecTV,
              enable_if_all<VecTM::dim == 2, VecTV::dim == 1,
                            VecTM::template range_t<0>::value == VecTM::template range_t<1>::value,
                            VecTM::template range_t<0>::value == VecTV::template range_t<0>::value,
                            VecTM::template range_t<0>::value <= 3,
                            std::is_floating_point_v<typename VecTM::value_type>> = 0>
    constexpr auto first_piola_derivative(const VecInterface<VecTM>& F,
                                          const VecInterface<VecTV>& a) const noexcept {
      return static_cast<const Model*>(this)->do_first_piola_derivative(F, a);
    }
    template <typename VecTM, typename VecTV>
    constexpr auto do_first_piola_derivative(const VecInterface<VecTM>&,
                                             const VecInterface<VecTV>&) const noexcept {
      return hessian_t<VecTM>::zeros();
    }
  };

  template <typename Model> struct PlasticityModelInterface {
    using model_type = Model;

    // project_sigma
    template <typename VecT, typename... Args,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) project_sigma(VecInterface<VecT>& S, Args&&... args) const noexcept {
      return static_cast<const Model*>(this)->do_project_sigma(S, FWD(args)...);
    }
    // project_strain
    template <typename VecT, typename... Args,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value <= 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr decltype(auto) project_strain(VecInterface<VecT>& F, Args&&... args) const noexcept {
      return static_cast<const Model*>(this)->do_project_strain(F, FWD(args)...);
    }

    // details (default impls)
    // return delta_gamma (projection distance)
    template <typename VecT, typename... Args>
    constexpr bool do_project_sigma(VecInterface<VecT>& S, Args&&... args) const noexcept {
      return false;
    }
    template <typename VecT, typename... Args>
    constexpr auto do_project_strain(VecInterface<VecT>& F, Args&&... args) const noexcept {
      auto [U, S, V] = math::svd(F);
      using result_t = decltype(static_cast<const Model*>(this)->project_sigma(S, FWD(args)...));
      if constexpr (!is_same_v<result_t, void>) {
        auto res = static_cast<const Model*>(this)->project_sigma(S, FWD(args)...);
        F.assign(diag_mul(U, S) * V.transpose());
        return res;
      } else {
        static_cast<const Model*>(this)->project_sigma(S, FWD(args)...);
        F.assign(diag_mul(U, S) * V.transpose());
        return;
      }
    }
  };

  template <typename VecTM,
            enable_if_all<VecTM::dim == 2, VecTM::template range_t<0>::value
                                               == VecTM::template range_t<1>::value> = 0>
  constexpr auto dFdXMatrix(const VecInterface<VecTM>& DmInv) noexcept {
    using value_type = typename VecTM::value_type;
    using index_type = typename VecTM::index_type;
    constexpr int dim = VecTM::template range_t<0>::value;
    constexpr int dimp1 = dim + 1;
    using RetT =
        typename VecTM::template variant_vec<value_type,
                                             integer_seq<index_type, dim * dim, dim * dimp1>>;

    vec<value_type, dim> t{};  // negative col-sum
    for (int d = 0; d != dim; ++d) {
      t[d] = -DmInv(0, d);
      for (int vi = 1; vi != dim; ++vi) t[d] -= DmInv(vi, d);
    }
    auto ret = RetT::zeros();
    for (int vi = 0; vi != dimp1; ++vi) {
      index_type c = vi * dim;
      for (int j = 0; j != dim; ++j) {
        index_type r = j * dim;
        const auto v = vi != 0 ? DmInv(vi - 1, j) : t(j);
        for (int d = 0; d != dim; ++d) ret(r + d, c + d) = v;
      }
    }
    return ret;
  }

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
      float sin_phi = zs::sin(fa);
      return zs::sqrt(2.f / 3.f) * 2.f * sin_phi / (3.f - sin_phi);
    }
    constexpr float M() const noexcept {
      // 1.850343771924453
      return mohrColumbFriction() * dim / zs::sqrt(2.f / (6.f - dim));
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