#pragma once
#include "../ConstitutiveModel.hpp"
#include "../ConstitutiveModelHelper.hpp"

namespace zs {

  template <typename T = float> struct NeoHookean
      : IsotropicConstitutiveModelInterface<NeoHookean<T>> {  // BW08, P80
    using base_t = IsotropicConstitutiveModelInterface<NeoHookean<T>>;
    using value_type = T;

    static_assert(std::is_floating_point_v<value_type>, "value type should be floating point");

    value_type mu, lam;

    NeoHookean() noexcept = default;
    constexpr NeoHookean(value_type E, value_type nu) noexcept {
      zs::tie(mu, lam) = lame_parameters(E, nu);
    }

    // do_psi_sigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr typename VecT::value_type do_psi_sigma(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;

      auto S_prod_log = zs::log(S.prod());
      return (value_type)0.5 * mu * (S.l2NormSqr() - dim)
             - (mu - (value_type)0.5 * lam * S_prod_log) * S_prod_log;
    }
    // do_dpsi_dsigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_dpsi_dsigma(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;

      auto S_prod_log = zs::log(S.prod());
      auto S_inv = (value_type)1 / S;
      return mu * (S - S_inv) + lam * S_inv * S_prod_log;
    }
    // do_d2psi_dsigma2
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_d2psi_dsigma2(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;

      auto S_prod_log = zs::log(S.prod());
      typename base_t::template mat_type<VecT> d2E_dsigma2{};
      auto S2_inv = (value_type)1 / (S * S);
      d2E_dsigma2(0, 0)
          = mu * ((value_type)1 + S2_inv[0]) - lam * S2_inv[0] * (S_prod_log - (value_type)1);
      if constexpr (dim > 1) {
        d2E_dsigma2(1, 1)
            = mu * ((value_type)1 + S2_inv[1]) - lam * S2_inv[1] * (S_prod_log - (value_type)1);
        d2E_dsigma2(0, 1) = d2E_dsigma2(1, 0) = lam / (S[0] * S[1]);
      }
      if constexpr (dim > 2) {
        d2E_dsigma2(2, 2)
            = mu * ((value_type)1 + S2_inv[2]) - lam * S2_inv[2] * (S_prod_log - (value_type)1);
        d2E_dsigma2(0, 2) = d2E_dsigma2(2, 0) = lam / (S[0] * S[2]);
        d2E_dsigma2(1, 2) = d2E_dsigma2(2, 1) = lam / (S[1] * S[2]);
      }
      return d2E_dsigma2;
    }
    // do_Bij_neg_coeff
    template <typename VecT, enable_if_all<VecT::dim == 1,
                                           VecT::template range_t<0>::value == 2
                                               || VecT::template range_t<0>::value == 3,
                                           std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto do_Bij_neg_coeff(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;
      using RetT = typename VecT::template variant_vec<
          typename VecT::value_type,
          integer_seq<typename VecT::index_type, (VecT::template range_t<0>::value == 3 ? 3 : 1)>>;
      RetT coeffs{};
      const auto S_prod = S.prod();
      if constexpr (dim == 2)
        coeffs[0] = (mu + (mu - lam * zs::log(S_prod)) / S_prod) * (value_type)0.5;
      else if constexpr (dim == 3) {
        const auto tmp = mu - lam * zs::log(S_prod);
        coeffs[0] = (mu + tmp / (S[0] * S[1])) * (value_type)0.5;
        coeffs[1] = (mu + tmp / (S[1] * S[2])) * (value_type)0.5;
        coeffs[2] = (mu + tmp / (S[2] * S[0])) * (value_type)0.5;
      }
      return coeffs;
    }
  };

  template <typename T = float> struct NeoHookeanInvariant
      : InvariantConstitutiveModelInterface<NeoHookeanInvariant<T>> {
    using base_t = InvariantConstitutiveModelInterface<NeoHookean<T>>;
    using value_type = T;

    static_assert(std::is_floating_point_v<value_type>, "value type should be floating point");

    value_type mu, lam;

    NeoHookeanInvariant() noexcept = default;
    constexpr NeoHookeanInvariant(value_type E, value_type nu) noexcept {
      zs::tie(mu, lam) = lame_parameters(E, nu);
    }

    // details (default impls)
    template <typename VecT>
    constexpr typename VecT::value_type do_psi_I(const VecInterface<VecT>& Is) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      auto logI3 = zs::log(Is[2]); // logI3
      return (value_type)0.5 * mu * (Is[1] - dim) - (mu - (value_type)0.5 * lam * logI3) * logI3;
    }
    template <int I, typename VecT>
    constexpr typename VecT::value_type do_dpsi_dI(const VecInterface<VecT>& Is) const noexcept {
      if constexpr (I == 1)
        return mu * (typename VecT::value_type)0.5;
      else if constexpr (I == 2)
        return (lam * zs::log(Is[2]) - mu) / Is[2];
      else
        return (typename VecT::value_type)0;
    }
    template <int I, typename VecT>
    constexpr typename VecT::value_type do_d2psi_dI2(const VecInterface<VecT>& Is) const noexcept {
      if constexpr (I == 2)
        return (lam * ((typename VecT::value_type)1 - zs::log(Is[2])) + mu) / (Is[2] * Is[2]);
      else
        return (typename VecT::value_type)0;
    }
  };

  /**
   * @brief The traditional Neohookean model is formulated using Cauchy-Green Invarients, but this
   * set of invarients can not cover all the isotropic material, take Corated-linear model for
   * example. The Smith Invarients is a supper set of Cauchy-Green invarients and can describe all
   * the isotropic material.
   * @param orient The orientation of the fibers
   * @param weights the activation level of the three fiber direction
   * @param F the stretching matrix
   */
  template <int Opt = 1, typename T0, typename T1, typename T2, typename Tn>
  constexpr auto compute_anisotrpic_invariant_deriv_hesssian(
      const vec_t<T0, Tn, (Tn)3, (Tn)3>& orient, const vec_t<T1, Tn, (Tn)3>& weights,
      const vec_t<T2, Tn, (Tn)3, (Tn)3>& F) {
    using R = math::op_result_t<T0, T1, T2>;
    using vec9 = vec_t<R, Tn, (Tn)9>;  // 3x3 flattened
    using mat9 = vec_t<R, Tn, (Tn)9, (Tn)9>;
    using Ret = conditional_t<
        Opt == 1, std::tuple<std::array<R, 3>>,
        conditional_t<Opt == 2, std::tuple<std::array<R, 3>, std::array<vec9, 3>>,
                      std::tuple<std::array<R, 3>, std::array<vec9, 3>, std::array<mat9, 3>>>>;
    Ret ret{};
    for (auto&& v : std::get<0>(ret)) v = (R)0;
    if constexpr (Opt > 1)
      for (auto&& v : std::get<1>(ret)) v = vec9::zeros();
    if constexpr (Opt > 2)
      for (auto&& v : std::get<2>(ret)) v = mat9::zeros();
    T1 weightSum = 0;
    for (Tn i = 0; i != 3; ++i) {
      auto pack = eval_I1_deriv_hessian<Opt>(F, col(orient, i));
      auto weight = weights[i] * weights[i];
      std::get<0>(ret)[0] += weight * std::get<0>(pack);
      if constexpr (Opt > 1) std::get<1>(ret)[0] += weight * std::get<1>(pack);
      if constexpr (Opt > 2) std::get<2>(ret)[0] += weight * std::get<2>(pack);
      weightSum += weight;
    }
    std::get<0>(ret)[0] *= 3 / weightSum;
    if constexpr (Opt > 1) std::get<1>(ret)[0] *= 3 / weightSum;
    if constexpr (Opt > 2) std::get<2>(ret)[0] *= 3 / weightSum;

    weightSum = 0;
    for (Tn i = 0; i != 3; ++i)
      for (Tn j = i + 1; j != 3; ++j) {
        auto pack = eval_I2_deriv_hessian<Opt>(F, col(orient, i), col(orient, j));
        auto weight = weights[i] * weights[j];
        std::get<0>(ret)[1] += weight * std::get<0>(pack);
        if constexpr (Opt > 1) std::get<1>(ret)[1] += weight * std::get<1>(pack);
        if constexpr (Opt > 2) std::get<2>(ret)[1] += weight * std::get<2>(pack);
        weightSum += weight;
      }
    std::get<0>(ret)[1] /= weightSum;
    if constexpr (Opt > 1) std::get<1>(ret)[1] = std::get<1>(ret)[1] / weightSum;
    if constexpr (Opt > 2) std::get<2>(ret)[1] = std::get<2>(ret)[1] / weightSum;

    {
      auto pack = eval_I3_deriv_hessian<Opt>(F);
      std::get<0>(ret)[2] = std::get<0>(pack);
      if constexpr (Opt > 1) std::get<1>(ret)[2] = std::get<1>(pack);
      if constexpr (Opt > 2) std::get<2>(ret)[2] = std::get<2>(pack);
    }
    return ret;
  }

  /**
   * @brief An interface for defining the potential energy of force model, all the force models
   * should inherit this method and implement their own version of element-wise potential energy
   * defination, element-wise energy gradient and 12x12 element-wise energy hessian w.r.t deformed
   * shape.
   * @param activation the activation level along the three orthogonal fiber directions
   * @param fiber_direction the three orthogonal fiber directions
   * @param <F> the deformation gradient
   * @param <energy> the potential energy output
   * @param <derivative> the derivative of potential energy w.r.t the deformation gradient
   * @param <Hessian> the hessian of potential energy w.r.t the deformed shape for elasto model or
   * nodal velocities for damping model
   * @param <enforcing_spd> decide whether we should enforce the SPD of hessian matrix
   */
  template <int Opt = 1, typename T0, typename T1, typename T2, typename Tn>
  constexpr auto compute_psi_deriv_hessian(const vec_t<T0, Tn, (Tn)3, (Tn)3>& act,
                                           const vec_t<T1, Tn, (Tn)3>& weights,
                                           const vec_t<T1, Tn, (Tn)3, (Tn)3>& fiberDirection,
                                           const T2 E, const T2 nu, vec_t<T2, Tn, (Tn)3, (Tn)3> F) {
    using R = math::op_result_t<T0, T1, T2>;
    using vec3 = vec_t<R, Tn, (Tn)3>;
    using vec9 = vec_t<R, Tn, (Tn)9>;
    using mat3 = vec_t<R, Tn, (Tn)3, (Tn)3>;
    using mat9 = vec_t<R, Tn, (Tn)9, (Tn)9>;
    using Ret
        = conditional_t<Opt == 1, std::tuple<R>,
                        conditional_t<Opt == 2, std::tuple<R, vec9>, std::tuple<R, vec9, mat9>>>;
    Ret ret{};

    const auto actInv = inverse(act);
    F = F * actInv;  // fiber-space deformation gradient

    // Is, Ds, Hs
    auto pack = compute_anisotrpic_invariant_deriv_hesssian<Opt>(fiberDirection, weights, F);

    auto I1_d = eval_I1_delta(weights, fiberDirection, F);                      // scalar
    auto I1_d_deriv = eval_I1_delta_deriv(weights, fiberDirection).vectorize();  // vec9
    // chain-rule
    auto dFact_dF = eval_dFact_dF(actInv);  // mat9

    auto lambda = (nu * E) / ((1 + nu) * (1 - 2 * nu));  // Enu2Lambda(E, nu);
    auto halfMu = E / (2 * (1 + nu)) * (T2)0.5;
    auto I2_minus_1 = (std::get<0>(pack)[2] - 1);
    // Psi
    std::get<0>(ret)
        = halfMu * (std::get<0>(pack)[0] - I1_d) + lambda * (T2)0.5 * I2_minus_1 * I2_minus_1;
    if constexpr (Opt > 1)
      // dPsi / dF
      std::get<1>(ret) = dFact_dF.transpose()
                         * (halfMu * (std::get<1>(pack)[0] - I1_d_deriv)
                            + lambda * I2_minus_1 * std::get<1>(pack)[2]);
    if constexpr (Opt > 2) {
      // ddPsi / dF^2
      std::get<2>(ret) = halfMu * std::get<2>(pack)[0]
                         + lambda * dyadic_prod(std::get<1>(pack)[2], std::get<1>(pack)[2])
                         + lambda * I2_minus_1 * std::get<2>(pack)[2];
      std::get<2>(ret) = dFact_dF.transpose() * std::get<2>(ret) * dFact_dF;
    }
    return ret;
  }

}  // namespace zs