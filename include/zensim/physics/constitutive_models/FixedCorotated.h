#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  template <typename T = float> struct FixedCorotated
      : IsotropicConstitutiveModelInterface<FixedCorotated<T>> {
    using base_t = IsotropicConstitutiveModelInterface<FixedCorotated<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type mu, lam;

    constexpr FixedCorotated() noexcept = default;
    constexpr FixedCorotated(value_type E, value_type nu) noexcept {
      zs::tie(mu, lam) = lame_parameters(E, nu);
    }

    // do_psi_sigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr typename VecT::value_type do_psi_sigma(const VecInterface<VecT>& S) const noexcept {
      auto S_sum_m1 = (S.prod() - (value_type)1);
      return mu * (S - 1).l2NormSqr() + (typename VecT::value_type)0.5 * lam * S_sum_m1 * S_sum_m1;
    }
    // do_dpsi_dsigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_dpsi_dsigma(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;

      typename base_t::template vec_type<VecT> S_prod_i{};
      if constexpr (dim == 1) {
        S_prod_i[0] = (value_type)1;
      } else if constexpr (dim == 2) {
        S_prod_i[0] = S[1];
        S_prod_i[1] = S[0];
      } else if constexpr (dim == 3) {
        S_prod_i[0] = S[1] * S[2];
        S_prod_i[1] = S[0] * S[2];
        S_prod_i[2] = S[0] * S[1];
      }
      return (mu + mu) * (S - (value_type)1) + lam * (S.prod() - (value_type)1) * S_prod_i;
    }
    // do_d2psi_dsigma2
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_d2psi_dsigma2(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;

      typename base_t::template vec_type<VecT> S_prod_i{};
      typename base_t::template mat_type<VecT> d2E_dsigma2{};
      auto S_prod = S.prod();
      auto _2mu = mu + mu;
      if constexpr (dim == 1) {
        d2E_dsigma2(0, 0) = (value_type)1;
      } else if constexpr (dim == 2) {
        S_prod_i[0] = S[1];
        S_prod_i[1] = S[0];
        d2E_dsigma2(0, 0) = _2mu + lam * S_prod_i[0] * S_prod_i[0];
        d2E_dsigma2(1, 1) = _2mu + lam * S_prod_i[1] * S_prod_i[1];
        d2E_dsigma2(0, 1) = d2E_dsigma2(1, 0) = lam * ((S_prod - 1) + S_prod_i[0] * S_prod_i[1]);
      } else if constexpr (dim == 3) {
        S_prod_i[0] = S[1] * S[2];
        S_prod_i[1] = S[0] * S[2];
        S_prod_i[2] = S[0] * S[1];
        d2E_dsigma2(0, 0) = _2mu + lam * S_prod_i[0] * S_prod_i[0];
        d2E_dsigma2(1, 1) = _2mu + lam * S_prod_i[1] * S_prod_i[1];
        d2E_dsigma2(2, 2) = _2mu + lam * S_prod_i[2] * S_prod_i[2];
        d2E_dsigma2(0, 1) = d2E_dsigma2(1, 0)
            = lam * (S[2] * (S_prod - 1) + S_prod_i[0] * S_prod_i[1]);
        d2E_dsigma2(0, 2) = d2E_dsigma2(2, 0)
            = lam * (S[1] * (S_prod - 1) + S_prod_i[0] * S_prod_i[2]);
        d2E_dsigma2(1, 2) = d2E_dsigma2(2, 1)
            = lam * (S[0] * (S_prod - 1) + S_prod_i[1] * S_prod_i[2]);
      }
      return d2E_dsigma2;
    }
    // do_Bij_neg_coeff
    template <typename VecT, enable_if_all<VecT::dim == 1,
                                           VecT::template range_t<0>::value == 2
                                               || VecT::template range_t<0>::value == 3,
                                           is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_Bij_neg_coeff(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;
      using RetT = typename VecT::template variant_vec<
          typename VecT::value_type,
          integer_sequence<typename VecT::index_type, (VecT::template range_t<0>::value == 3 ? 3 : 1)>>;
      RetT coeffs{};
      if constexpr (dim == 2)
        coeffs[0] = mu - (value_type)0.5 * lam * (S.prod() - (value_type)1);
      else if constexpr (dim == 3) {
        const auto S_prod = S.prod();
        coeffs[0] = mu - (value_type)0.5 * lam * S[2] * (S_prod - (value_type)1);
        coeffs[1] = mu - (value_type)0.5 * lam * S[0] * (S_prod - (value_type)1);
        coeffs[2] = mu - (value_type)0.5 * lam * S[1] * (S_prod - (value_type)1);
      }
      return coeffs;
    }
  };

}  // namespace zs