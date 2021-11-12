#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  // ref:
  // An adaptive generalized interpolation material point method for simulating elastoplastidc
  // materials,
  // sec 4.2.2
  template <typename T = float> struct VonMisesCapped
      : PlasticityModelInterface<VonMisesCapped<T>> {
    using base_t = PlasticityModelInterface<VonMisesCapped<T>>;
    using value_type = T;

    static_assert(std::is_floating_point_v<value_type>, "value type should be floating point");

    value_type k1Compress, k1Stretch, yieldStress /*i.e. k2, sigma_yield*/;
    // Z(G) = k1 * |tr(G)| + k2 * FrobeniusNorm(G')

    constexpr VonMisesCapped(value_type k1Compress, value_type k1Stretch, value_type k2) noexcept
        : k1Compress{k1Compress}, k1Stretch{k1Stretch}, yieldStress{k2} {}

    // project_strain
    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 1, (VecT::template range<0>() <= 3),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_sigma(VecInterface<VecT>& S, const Model& model) const noexcept {
      using value_type = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template range<0>();

      const auto _2mu = (static_cast<const Model&>(model).mu + static_cast<const Model&>(model).mu);
      const auto dim_mul_lam = (value_type)dim * (value_type) static_cast<const Model&>(model).lam;

      auto eps = S.log();
      auto eps_trace = eps.sum();
      auto dev_eps = (eps - eps_trace / (value_type)dim);  // equivalent to eps.deviatoric()
      auto dev_eps_norm = dev_eps.norm();
      auto delta_gamma = dev_eps_norm - yieldStress / _2mu;
      if (delta_gamma > 0) {
        auto H = eps - (delta_gamma / dev_eps_norm) * dev_eps;
        S = H.exp();
      }

      if (eps_trace > k1Stretch / (dim_mul_lam + _2mu))
        S = S * gcem::exp(k1Stretch / (dim * dim_mul_lam + dim * _2mu) - eps_trace / dim);
      else if (eps_trace < -k1Compress / (dim_mul_lam + _2mu))
        S = S * gcem::exp(-k1Compress / (dim * dim_mul_lam + dim * _2mu) - eps_trace / dim);
    }

    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_strain(VecInterface<VecT>& F, const Model& model) const noexcept {
      auto [U, S, V] = math::svd(F);
      do_project_sigma(S, model);
      F = diag_mul(U, S) * V.transpose();
    }

    template <typename VecT, typename Model,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void project_strain(VecInterface<VecT>& F, const Model& model,
                                  typename VecT::value_type strainRate, int pi) const noexcept {
      auto [U, S, V] = math::svd(F);

      using value_type = typename VecT::value_type;
      using Ti = typename VecT::index_type;
      using extents = typename VecT::extents;
      constexpr int dim = VecT::template range<0>();

      const auto _2mu = (static_cast<const Model&>(model).mu + static_cast<const Model&>(model).mu);
      const auto dim_mul_lam = (value_type)dim * (value_type) static_cast<const Model&>(model).lam;

      auto eps = S.log();
      auto eps_trace = eps.sum();
      auto dev_eps = eps - eps_trace / (value_type)dim;  // equivalent to eps.deviatoric()

#if 0
      if (pi < 0)
        printf("pi[%d]: dev(eps) = %f, %f, %f; eps = %f, %f, %f; trace %f; normSqr %f; norm %f\n",
               pi, (float)dev_eps(0), (float)dev_eps(1), (float)dev_eps(2), (float)eps(0),
               (float)eps(1), (float)eps(2), (float)eps_trace, (float)dev_eps.l2NormSqr(),
               math::sqrtNewtonRaphson((float)dev_eps.l2NormSqr()));
#endif

      auto dev_eps_norm = dev_eps.norm();

      // Cowper-Symonds
      auto P = (value_type)dim;
      auto C = (value_type)1;
      auto coef = gcem::pow((strainRate) / C, (value_type)1 / P);
#if 1
      auto ys = yieldStress * (1 + gcem::pow((strainRate) / C, (value_type)1 / P));
#else
      auto ys = yieldStress;
#endif

      auto delta_gamma = dev_eps_norm - ys / _2mu;
// printf("%f -> %f\n", yieldStress, ys);
#if 0
      if (dev_eps_norm > 0.2f && pi < 10)
        printf(
            "pi[%d]: sigma_0: %f -> sigma_yield: %f (coeff: %f), mu: %f, candidate: %f, right: "
            "%f\n",
            pi, (float)yieldStress, (float)ys, (float)coef, (float)(_2mu * 0.5),
            (float)dev_eps_norm, (float)(ys / _2mu));
      if (pi < 5)
        printf(
            "pi[%d]: ys(%f -> %f), CHat[%f, %f] (%f, %f, %f), mu: %f, coeff: %f, delta_gamma: %f, "
            "left: %f, "
            "right: %f\n",
            pi, (float)yieldStress, (float)ys, (float)Chat.infNorm(),
            (float)(Chat.sum() / (value_type)dim), (float)Chat(0), (float)Chat(1), (float)Chat(2),
            (float)_2mu * 0.5f, (float)coef, (float)delta_gamma, (float)dev_eps_norm,
            (float)(ys / _2mu));
#endif

      if (delta_gamma > 0) {
        auto H = eps - (delta_gamma / dev_eps_norm) * dev_eps;
        S = H.exp();
#if 0
        if (dev_eps_norm > 0.2f)
          printf("H: %f, %f, %f; S %f, %f, %f -> %f, %f, %f\n", (float)H[0], (float)H[1],
                 (float)H[2], (float)oldS[0], (float)oldS[1], (float)oldS[2], (float)S[0],
                 (float)S[1], (float)S[2]);
#endif
      }

#if 0
      if (eps_trace > k1Stretch / (dim_mul_lam + _2mu))
        S = S * gcem::exp(k1Stretch / (dim * dim_mul_lam + dim * _2mu) - eps_trace / (value_type)dim);
      else if (eps_trace < -k1Compress / (dim_mul_lam + _2mu))
        S = S * gcem::exp(-k1Compress / (dim * dim_mul_lam + dim * _2mu) - eps_trace / (value_type)dim);
#endif

      F = diag_mul(U, S) * V.transpose();
    }
  };

}  // namespace zs