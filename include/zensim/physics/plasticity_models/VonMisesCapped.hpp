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
    template <typename VecT, template <typename> class ModelInterface, typename Model,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_sigma(VecInterface<VecT>& S,
                                    const ModelInterface<Model>& model) const noexcept {
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

    template <typename VecT, template <typename> class ModelInterface, typename Model,
              enable_if_all<VecT::dim == 2, (VecT::template range<0>() <= 3),
                            VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr void do_project_strain(VecInterface<VecT>& F,
                                     const ModelInterface<Model>& model) const noexcept {
      auto [U, S, V] = math::svd(F);
      do_project_sigma(S);
      F = diag_mul(U, S) * V.transpose();
    }
  };

}  // namespace zs