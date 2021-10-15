#pragma once
#include "../ConstitutiveModel.hpp"
#include "../ConstitutiveModelHelper.hpp"

namespace zs {

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
      auto pack = eval_I1_deriv_hessian<Opt>(F, orient.col(i));
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
        auto pack = eval_I2_deriv_hessian<Opt>(F, orient.col(i), orient.col(j));
        auto weight = weights[i] * weights[j];
        std::get<0>(ret)[1] += weight * std::get<0>(pack);
        if constexpr (Opt > 1) std::get<1>(ret)[1] += weight * std::get<1>(pack);
        if constexpr (Opt > 2) std::get<2>(ret)[1] += weight * std::get<2>(pack);
        weightSum += weight;
      }
    std::get<0>(ret)[1] /= weightSum;
    if constexpr (Opt > 1) std::get<1>(ret)[1] /= weightSum;
    if constexpr (Opt > 2) std::get<2>(ret)[1] /= weightSum;

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
                                           const T2 E, const T2 nu,
                                           const vec_t<T2, Tn, (Tn)3, (Tn)3>& F,
                                           bool enforcingSpd = true) {
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
    auto pack = compute_anisotrpic_invariant_deriv_hesssian<1>(fiberDirection, weights, F);

    auto I1_d = eval_I1_delta(weights, fiberDirection, F);                      // scalar
    auto I1_d_deriv = vectorize(eval_I1_delta_deriv(weights, fiberDirection));  // vec9
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
  }

}  // namespace zs