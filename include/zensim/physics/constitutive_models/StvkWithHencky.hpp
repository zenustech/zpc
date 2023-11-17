#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  template <typename T = float> struct StvkWithHencky
      : IsotropicConstitutiveModelInterface<StvkWithHencky<T>> {
    using base_t = IsotropicConstitutiveModelInterface<StvkWithHencky<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type mu, lam;

    constexpr StvkWithHencky() noexcept = default;
    constexpr StvkWithHencky(value_type E, value_type nu) noexcept {
      zs::tie(mu, lam) = lame_parameters(E, nu);
    }

    // do_psi_sigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr typename VecT::value_type do_psi_sigma(const VecInterface<VecT>& S) const noexcept {
      const auto S_log = S.abs().log();
      const auto S_log_trace = S_log.sum();
      return mu * S_log.square().sum() + (value_type)0.5 * lam * S_log_trace * S_log_trace;
    }
    // do_dpsi_dsigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_dpsi_dsigma(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      // constexpr auto dim = VecT::template range_t<0>::value;

      const auto S_log = S.abs().log();
      const auto S_log_trace = S_log.sum();
      return ((mu + mu) * S_log + lam * S_log_trace) / S;
    }
    // do_d2psi_dsigma2
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_d2psi_dsigma2(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;

      const auto S_log = S.abs().log();
      const auto S_log_trace = S_log.sum();
      const auto _2mu = mu + mu;
      const auto _1_m_S_log_trace = ((value_type)1 - S_log_trace);
      typename base_t::template mat_type<VecT> d2E_dsigma2{};
      d2E_dsigma2(0, 0)
          = (_2mu * ((value_type)1 - S_log(0)) + lam * _1_m_S_log_trace) / (S[0] * S[0]);
      if constexpr (dim > 1) {
        d2E_dsigma2(1, 1)
            = (_2mu * ((value_type)1 - S_log(1)) + lam * _1_m_S_log_trace) / (S[1] * S[1]);
        d2E_dsigma2(0, 1) = d2E_dsigma2(1, 0) = lam / (S[0] * S[1]);
      }
      if constexpr (dim > 2) {
        d2E_dsigma2(2, 2)
            = (_2mu * ((value_type)1 - S_log(2)) + lam * _1_m_S_log_trace) / (S[2] * S[2]);
        d2E_dsigma2(0, 2) = d2E_dsigma2(2, 0) = lam / (S[0] * S[2]);
        d2E_dsigma2(1, 2) = d2E_dsigma2(2, 1) = lam / (S[1] * S[2]);
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

      const auto S_log = S.abs().log();
      constexpr value_type eps = 1e-6;
      if constexpr (dim == 2) {
        auto q = zs::max(S[0] / S[1] - 1, -1 + eps);
        auto h = zs::abs(q) < eps ? (value_type)1 : (zs::log1p(q) / q);
        auto t = h / S[1];
        auto z = S_log[1] - t * S[1];
        coeffs[0] = -(lam * (S_log[0] + S_log[1]) + (mu + mu) * z) / S.prod() * (value_type)0.5;
      } else if constexpr (dim == 3) {
        const auto S_log_trace = S_log.sum();
        const auto _2mu = mu + mu;
        coeffs[0]
            = -(lam * S_log_trace
                + _2mu * math::diff_interlock_log_over_diff(S(0), zs::abs(S(1)), S_log(1), eps))
              / (S[0] * S[1]) * (value_type)0.5;
        coeffs[1]
            = -(lam * S_log_trace
                + _2mu * math::diff_interlock_log_over_diff(S(1), zs::abs(S(2)), S_log(2), eps))
              / (S[1] * S[2]) * (value_type)0.5;
        coeffs[2]
            = -(lam * S_log_trace
                + _2mu * math::diff_interlock_log_over_diff(S(0), zs::abs(S(2)), S_log(2), eps))
              / (S[0] * S[2]) * (value_type)0.5;
      }
      return coeffs;
    }
  };

}  // namespace zs