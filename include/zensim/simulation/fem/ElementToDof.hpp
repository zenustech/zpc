#pragma once
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/physics/ConstitutiveModelHelper.hpp"
#include "zensim/physics/constitutive_models/DirichletDamping.hpp"
#include "zensim/physics/constitutive_models/NeoHookean.hpp"

namespace zs {

  template <int Opt = 1, typename ForceModel, typename DampingModel, typename T, typename T0,
            typename T1, typename Tn>
  constexpr void eval_tet_force(
      T vol, T rho, T E, T nu, T dt, const vec_t<T, Tn, (Tn)3>& gravity,
      const vec_t<T, Tn, (Tn)9, (Tn)12>& dFdX, const ForceModel& forceModel,
      const vec_t<T0, Tn, (Tn)12>& u_n,
      const vec_t<T0, Tn, (Tn)3, (Tn)3>& activation, const vec_t<T0, Tn, (Tn)3>& weights,
      const vec_t<T0, Tn, (Tn)3, (Tn)3>& orientation, const DampingModel& dampingModel,
      const T1 dampingCoeff, const vec_t<T1, Tn, (Tn)12>& v_n) {
    using R = math::op_result_t<T, T0, T1>;
    using vec12 = vec_t<R, Tn, (Tn)12>;
    using mat3 = vec_t<R, Tn, (Tn)3, (Tn)3>;
    using mat9 = vec_t<R, Tn, (Tn)9, (Tn)9>;
    using mat12 = vec_t<R, Tn, (Tn)12, (Tn)12>;

    auto nodalMass = vol * rho * (T)0.25;  // evenly distributed to vertices

    mat3 Ds{}, Dv{};
    for (Tn i = 0; i != 3; ++i) {
      auto dx = u_n[i + 1] - u_n[0];
      for (Tn d = 0; d != 3; ++d) {
        Ds(d, i) = dx[d];
        Dv(d, i) = v_n[(i + 1) * 3 + d] - v_n[d];
      }
    }
    auto F = Ds * DmInv;
    auto L = Dv * DmInv;

    // psi
    auto elasticPack = compute_psi_deriv_hessian<Opt>(activation, weights, orientation, E, nu, F);
    auto dampingPack = compute_psi_deriv_hessian<Opt>(dampingCoeff, L);

    vec_t<T1, Tn, (Tn)12> gravVec{};
    for (Tn i = 0; i != 12; ++i) gravVec[i] = gravity(i % 3);

    return -nodalMass * gravVec
            + dFdX.transpose() * (std::get<1>(elasticPack) + std::get<1>(dampingPack)) * vol;
  }

  /// currently forceModel&dampingModel are just placeholders
  template <int Opt = 1, typename ForceModel, typename DampingModel, typename T, typename T0,
            typename T1, typename Tn>
  constexpr void eval_tet_obj_deriv_jacobi(
      T vol, T rho, T E, T nu, T dt, const vec_t<T, Tn, (Tn)3>& gravity,
      const vec_t<T, Tn, (Tn)9, (Tn)12>& dFdX, const ForceModel& forceModel,
      const vec_t<T0, Tn, (Tn)12>& u_n, const vec_t<T0, Tn, (Tn)12>& u,
      const vec_t<T0, Tn, (Tn)3, (Tn)3>& activation, const vec_t<T0, Tn, (Tn)3>& weights,
      const vec_t<T0, Tn, (Tn)3, (Tn)3>& orientation, const DampingModel& dampingModel,
      const T1 dampingCoeff, const vec_t<T1, Tn, (Tn)12>& v_n) {
    using R = math::op_result_t<T, T0, T1>;
    using vec9 = vec_t<R, Tn, (Tn)9>;
    using vec12 = vec_t<R, Tn, (Tn)12>;
    using mat3 = vec_t<R, Tn, (Tn)3, (Tn)3>;
    using mat9 = vec_t<R, Tn, (Tn)9, (Tn)9>;
    using mat12 = vec_t<R, Tn, (Tn)12, (Tn)12>;
    using Ret
        = conditional_t<Opt == 1, std::tuple<R>,
                        conditional_t<Opt == 2, std::tuple<R, vec12>, std::tuple<R, vec12, mat9>>>;
    // obj, deriv, H
    Ret ret{};
    auto nodalMass = vol * rho * (T)0.25;  // evenly distributed to vertices

    mat3 Ds{}, Dv{};
    for (Tn i = 0; i != 3; ++i) {
      auto dx = u[i + 1] - u[0];
      for (Tn d = 0, base = (i + 1) * 3; d != 3; ++d) {
        Ds(d, i) = dx[d];
        Dv(d, i) = v_n[base + d] - v_n[d];
      }
    }
    auto F = Ds * DmInv;
    auto L = Dv * DmInv;

    // psi
    auto elasticPack = compute_psi_deriv_hessian<Opt>(activation, weights, orientation, E, nu, F);
    auto dampingPack = compute_psi_deriv_hessian<Opt>(dampingCoeff, L);

    vec12 gravVec{};
    for (Tn i = 0; i != 12; ++i) gravVec[i] = gravity(i % 3);
    auto y = u - u_n - v_n * dt - dt * dt * gravVec;
    auto PhiI = y.squaredNorm() * nodalMass / (2 * dt);

    std::get<0>(ret) = PhiI + dt * (std::get<0>(elasticPack) + dt * std::get<0>(dampingPack)) * vol;

    if constexpr (Opt > 1) {
      std::get<1>(ret)
          = nodalMass * y / dt
            + dFdX.transpose() * (std::get<1>(elasticPack) + std::get<1>(dampingPack)) * dt * vol;

      if constexpr (Opt > 2)
        std::get<2>(ret) = mat12::identity() * nodalMass / dt
                           + vol * transpose(dFdX)
                                 * (dt * std::get<2>(elasticPack) + std::get<2>(dampingPack))
                                 * dFdX;
    }
    return ret;
  }

}  // namespace zs