#pragma once
#include "../ConstitutiveModel.hpp"
#include "zensim/math/MathUtils.h"

namespace zs {

  // ref:
  // An adaptive generalized interpolation material point method for simulating elastoplastidc
  // materials,
  // sec 4.2.2
  template <typename T_ = float> struct NonAssociativeCamClay
      : PlasticityModelInterface<NonAssociativeCamClay<T_>> {
    using base_t = PlasticityModelInterface<NonAssociativeCamClay<T_>>;
    using value_type = T_;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type M, beta, xi;
    bool hardeningOn;
    // use qhard

    NonAssociativeCamClay(value_type frictionAngle = 35, value_type beta = 2, value_type xi = 3,
                          int dim = 3, bool hardeningOn = true)
        : beta{beta}, xi{xi}, hardeningOn{hardeningOn} {
      constexpr value_type coeff = math::sqrtNewtonRaphson((value_type)2 / (value_type)3);
      const auto sinFa = std::sin(frictionAngle / (value_type)180 * g_pi);
      const auto mohr_columb_friction = coeff * (sinFa + sinFa) / ((value_type)3 - sinFa);
      M = mohr_columb_friction * (value_type)dim / std::sqrt((value_type)2 / (value_type)(6 - dim));
    }

    template <typename VecT, typename Model, typename T,
              enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                            is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr bool do_project_sigma(VecInterface<VecT>& S, const Model& model,
                                    T& logJp) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      const auto kappa = model.lam + model.mu * (value_type)2 / (value_type)3;

      auto p0 = kappa * ((value_type)1e-5 + xi * zs::sinh(zs::max(-logJp, (value_type)0)));
      auto J = S.prod();
      auto B_hat_trial = S * S;
      auto side_len_sqr = zs::pow(J, (value_type)2 / (value_type)dim);
      auto s_hat_trial = model.mu / side_len_sqr * B_hat_trial.deviatoric();

      auto prime = kappa / (value_type)2 * (J - (value_type)1 / J);
      auto p_trial = -prime * J;

      // Cases 1 and 2 (Ellipsoid Tips)
      // Project to the tips
      auto pMin = beta * p0;
      auto pMax = p0;
      int caseNo = p_trial > pMax ? 1 : (p_trial < -pMin ? 2 : 3);
      if (caseNo < 3) {
        auto JeNew = caseNo == 1 ? zs::sqrt(-(pMax + pMax) / kappa + (value_type)1)
                                 : zs::sqrt((pMin + pMin) / kappa + (value_type)1);
        S = zs::pow(JeNew, (value_type)1 / (value_type)dim);  // assigned the scalar
        // hardening
        if (hardeningOn) logJp += zs::log(J / JeNew);
        return true;
      }

      // Case 3 --> check if inside or outside YS
      auto y_s_half_coeff
          = ((value_type)6 - dim) * (value_type)0.5 * ((value_type)1 + (value_type)2 * beta);
      auto y_p_half = M * M * (p_trial + pMin) * (p_trial - pMax);
      auto y = y_s_half_coeff * s_hat_trial.l2NormSqr() + y_p_half;

      // Case 3a (Inside Yield Surface)
      // Do nothing
      if (y < (value_type)1e-4) return false;

      // Case 3b (Outside YS)
      // project to yield surface
      auto s_hat_trial_norm = s_hat_trial.norm();
      auto B_hat_new = side_len_sqr / model.mu * zs::sqrt(-y_p_half / y_s_half_coeff) * s_hat_trial
                       / s_hat_trial_norm;
      B_hat_new += (value_type)1 / (value_type)dim * B_hat_trial.sum();
      S.assign(B_hat_new.sqrt());

      // Step 2: Hardening
      // q-based hardening:
      if (p0 > (value_type)1e-4 && p_trial < pMax - (value_type)1e-4
          && p_trial > (value_type)1e-4 - pMin) {
        constexpr auto sqrt_3_m_half_dim
            = math::sqrtNewtonRaphson((value_type)(6 - dim) / (value_type)2);
        auto p_center = p0 * (((value_type)1 - beta) * (value_type)0.5);
        auto q_trial = sqrt_3_m_half_dim * s_hat_trial_norm;
        vec<value_type, 2> direction{p_center - p_trial, -q_trial};
        direction /= direction.norm();

        const auto MSqr = M * M;
        auto C = MSqr * (p_center + beta * p0) * (p_center - p0);
        auto B = MSqr * direction(0) * (p_center + p_center - p0 + beta * p0);
        auto A
            = MSqr * direction(0) * direction(0) + (1 + beta + beta) * direction(1) * direction(1);

        // roots of quadratic
        const auto deltaSqrt = zs::sqrt(B * B - 4 * A * C);
        auto l1 = (-B + deltaSqrt) / (A + A);
        auto l2 = (-B - deltaSqrt) / (A + A);

        auto p1 = p_center + l1 * direction(0);
        auto p2 = p_center + l2 * direction(0);
        auto p_fake = (p_trial - p_center) * (p1 - p_center) > 0 ? p1 : p2;

        // Only for pFake Hardening
        // auto Je_new_fake = sqrt(std::abs(-2 * p_fake / c.kappa + 1));
        // dAlpha = log(J / Je_new_fake);

        // Only for q Hardening
        auto qNPlus
            = zs::sqrt(M * M * (p_trial + pMin) * (pMax - p_trial) / ((value_type)1 + beta + beta));
        // Jtrial
        auto zTrial = zs::sqrt((q_trial * side_len_sqr / (model.mu * sqrt_3_m_half_dim)) + 1);
        auto zNPlus = zs::sqrt((qNPlus * side_len_sqr / (model.mu * sqrt_3_m_half_dim)) + 1);

        // value_type dAlpha{};  // change in logJp, or the change in volumetric plastic strain
        value_type dOmega{};  // change in logJp from q hardening (only for q hardening)
        if (p_trial > p_fake) {
          dOmega = (value_type)-1 * zs::log(zTrial / zNPlus);
        } else
          dOmega = zs::log(zTrial / zNPlus);

        if (hardeningOn)
          // if (Je_new_fake > 1e-4) logJp += dAlpha;
          if (zNPlus > 1e-4) logJp += dOmega;
      }
      return true;
    }
  };

}  // namespace zs