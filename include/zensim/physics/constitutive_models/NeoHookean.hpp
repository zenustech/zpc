#pragma once
#include "../ConstitutiveModel.hpp"

namespace zs {

  template <typename T = float> struct NeoHookean
      : IsotropicConstitutiveModelInterface<NeoHookean<T>> {  // BW08, P80
    using base_t = IsotropicConstitutiveModelInterface<NeoHookean<T>>;
    using value_type = T;

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type mu, lam;

    NeoHookean() noexcept = default;
    constexpr NeoHookean(value_type E, value_type nu) noexcept {
      zs::tie(mu, lam) = lame_parameters(E, nu);
    }

    // do_psi_sigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>>
                             = 0>
    constexpr typename VecT::value_type do_psi_sigma(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;

      auto S_prod_log = zs::log(S.prod());
      return (value_type)0.5 * mu * (S.l2NormSqr() - dim)
             - (mu - (value_type)0.5 * lam * S_prod_log) * S_prod_log;
    }
    // do_dpsi_dsigma
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>>
                             = 0>
    constexpr auto do_dpsi_dsigma(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      // constexpr auto dim = VecT::template range_t<0>::value;

      auto S_prod_log = zs::log(S.prod());
      auto S_inv = (value_type)1 / S;
      return mu * (S - S_inv) + lam * S_inv * S_prod_log;
    }
    // do_d2psi_dsigma2
    template <typename VecT, enable_if_all<VecT::dim == 1, VecT::template range_t<0>::value <= 3,
                                           is_floating_point_v<typename VecT::value_type>>
                             = 0>
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
                                           is_floating_point_v<typename VecT::value_type>>
                             = 0>
    constexpr auto do_Bij_neg_coeff(const VecInterface<VecT>& S) const noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto dim = VecT::template range_t<0>::value;
      using RetT = typename VecT::template variant_vec<
          typename VecT::value_type,
          integer_sequence<typename VecT::index_type,
                           (VecT::template range_t<0>::value == 3 ? 3 : 1)>>;
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

    static_assert(is_floating_point_v<value_type>, "value type should be floating point");

    value_type mu, lam;

    NeoHookeanInvariant() noexcept = default;
    constexpr NeoHookeanInvariant(value_type E, value_type nu) noexcept {
      zs::tie(mu, lam) = lame_parameters(E, nu);
    }

    // details (default impls)
    template <typename VecT>
    constexpr typename VecT::value_type do_psi_I(const VecInterface<VecT>& Is) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      auto logI3 = zs::log(Is[2]);  // logI3
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
    template <int I, int J = I, typename VecT>
    constexpr typename VecT::value_type do_d2psi_dI2(const VecInterface<VecT>& Is) const noexcept {
      if constexpr (I == 2 && J == 2)
        return (lam * ((typename VecT::value_type)1 - zs::log(Is[2])) + mu) / (Is[2] * Is[2]);
      else
        return (typename VecT::value_type)0;
    }
  };

  template <typename T = float> struct StableNeohookeanInvarient
      : InvariantConstitutiveModelInterface<StableNeohookeanInvarient<T>> {
    using base_t = InvariantConstitutiveModelInterface<StableNeohookeanInvarient<T>>;
    using value_type = T;
    static_assert(is_floating_point_v<value_type>, "value type should be floating point");
    value_type mu, lam;

    StableNeohookeanInvarient() noexcept = default;
    constexpr StableNeohookeanInvarient(value_type E, value_type nu) noexcept {
      zs::tie(mu, lam) = lame_parameters(E, nu);
    }

    template <typename VecT>
    constexpr typename VecT::value_type do_psi_I(const VecInterface<VecT>& Is) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      value_type l2m1 = (Is[2] - (value_type)1.0);
      return (value_type)0.5 * mu * (Is[1] - (value_type)3.0) - mu * l2m1
             + (value_type)0.5 * lam * l2m1 * l2m1;
    }

    template <int I, typename VecT>
    constexpr typename VecT::value_type do_dpsi_dI(const VecInterface<VecT>& Is) const noexcept {
      constexpr auto dim = VecT::template range_t<0>::value;
      if constexpr (I == 0)
        return (value_type)0.0;
      else if constexpr (I == 1)
        return (value_type)0.5 * mu;
      else  // I == 2
        return -mu + lam * (Is[2] - (value_type)1.0);
    }

    template <int I, int J, typename VecT>
    constexpr typename VecT::value_type do_d2psi_dI2(const VecInterface<VecT>&) const noexcept {
      if constexpr (I == 2 && J == 2)
        return (value_type)lam;
      else
        return (value_type)0.0;
    }

    template <typename VecT, typename VecS,
              enable_if_all<VecT::dim == 1, VecS::dim == 1, VecT::template range_t<0>::value == 3,
                            VecS::template range_t<0>::value == 3,
                            is_floating_point_v<typename VecT::value_type>,
                            is_floating_point_v<typename VecS::value_type>>
              = 0>
    constexpr auto eval_stretching_matrix(const VecInterface<VecT>& Is,
                                          const VecInterface<VecS>& sigma) const noexcept {
      typename VecT::template variant_vec<typename VecT::value_type,
                                          integer_sequence<typename VecT::index_type, 3, 3>>
          A{};
      A(0, 0) = mu + lam * Is[2] * Is[2] / sigma[0] / sigma[0];
      A(0, 1) = sigma[2] * (lam * (2 * Is[2] - 1) - mu);
      A(0, 2) = sigma[1] * (lam * (2 * Is[2] - 1) - mu);
      A(1, 1) = mu + lam * Is[2] * Is[2] / sigma[1] / sigma[1];
      A(1, 2) = sigma[0] * (lam * (2 * Is[2] - 1) - mu);
      A(2, 2) = mu + lam * Is[2] * Is[2] / sigma[2] / sigma[2];
      A(1, 0) = A(0, 1);
      A(2, 0) = A(0, 2);
      A(2, 1) = A(1, 2);

      return A;
    }

    //   // first piola derivative
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range_t<0>::value == 3,
                            VecT::template range_t<0>::value == VecT::template range_t<1>::value,
                            is_floating_point_v<typename VecT::value_type>>
              = 0>
    constexpr auto do_first_piola_derivative_spd(const VecInterface<VecT>& F) const noexcept {
      // sum_i ((d2Psi / dI_i2) g_i g_i^T + ((dPsi / dI_i) H_i))
      // printf("do_first_piola_derivative_spd get called\n");

      typename VecT::template variant_vec<typename VecT::value_type,
                                          integer_sequence<typename VecT::index_type, 3>>
          Is{};

      auto [U, S, V] = math::qr_svd(F);

      Is[0] = zs::get<0>(base_t::I_wrt_F<0, 0>(F));
      Is[1] = zs::get<0>(base_t::I_wrt_F<1, 0>(F));
      Is[2] = zs::get<0>(base_t::I_wrt_F<2, 0>(F));

      auto A = eval_stretching_matrix(Is, S);
      auto [A_eigvals, A_eigvecs] = zs::eigen_decomposition(A.template cast<double>());

      typename base_t::template gradient_t<VecT> eigen_vals{};
      typename base_t::template gradient_t<VecT> eigen_vecs[9]{};

      using Ti = typename VecT::index_type;
      // Ti offset = 0;
      // scale
      eigen_vals[0] = A_eigvals[0];
      eigen_vals[1] = A_eigvals[1];
      eigen_vals[2] = A_eigvals[2];
      // flip
      // eigen_vals[offset++] = (double)mu + (double)S[2] * (lam * (Is[2] - 1) - mu);
      eigen_vals[3] = mu * (1 - S[2]) + S[2] * (lam * (Is[2] - 1));
      // eigen_vals[offset++] = (double)mu + (double)S[0] * (lam * (Is[2] - 1) - mu);
      eigen_vals[4] = mu * (1 - S[0]) + S[0] * (lam * (Is[2] - 1));
      // eigen_vals[offset++] = (double)mu + (double)S[1] * (lam * (Is[2] - 1) - mu);
      eigen_vals[5] = mu * (1 - S[1]) + S[1] * (lam * (Is[2] - 1));
      // twist
      eigen_vals[6] = mu - S[2] * (lam * (Is[2] - 1) - mu);
      eigen_vals[7] = mu - S[0] * (lam * (Is[2] - 1) - mu);
      eigen_vals[8] = mu - S[1] * (lam * (Is[2] - 1) - mu);

      using mat3 =
          typename VecT::template variant_vec<typename VecT::value_type,
                                              integer_sequence<typename VecT::index_type, 3, 3>>;
      constexpr double sqrt2 = 1.4142135623730950488016887242096980785697L;
      constexpr mat3 Qs[9]
          = {mat3::init([](int i) { return i == 0 ? 1 : 0; }),
             mat3::init([](int i) { return i == 4 ? 1 : 0; }),
             mat3::init([](int i) { return i == 8 ? 1 : 0; }),
             mat3::init([](int i) { return i == 1 ? -1 / sqrt2 : (i == 3 ? 1 / sqrt2 : 0); }),
             mat3::init([](int i) { return i == 5 ? 1 / sqrt2 : (i == 7 ? -1 / sqrt2 : 0); }),
             mat3::init([](int i) { return i == 2 ? 1 / sqrt2 : (i == 6 ? -1 / sqrt2 : 0); }),
             mat3::init([](int i) { return i == 1 ? 1 / sqrt2 : (i == 3 ? 1 / sqrt2 : 0); }),
             mat3::init([](int i) { return i == 5 ? 1 / sqrt2 : (i == 7 ? 1 / sqrt2 : 0); }),
             mat3::init([](int i) { return i == 2 ? 1 / sqrt2 : (i == 6 ? 1 / sqrt2 : 0); })};
      typename base_t::template gradient_t<VecT> projspace[3]
          = {vectorize(U * Qs[0] * V.transpose(), wrapv<true>{}),
             vectorize(U * Qs[1] * V.transpose(), wrapv<true>{}),
             vectorize(U * Qs[2] * V.transpose(), wrapv<true>{})};
      for (Ti col = 0; col != 3; ++col) {
        for (int i = 0; i != 9; ++i) {
          eigen_vecs[col](i) = 0;
          for (int d = 0; d != 3; ++d) eigen_vecs[col](i) += projspace[d](i) * A_eigvecs(d, col);
        }
      }
      for (Ti i = 3; i != 9; ++i) {
        auto eigen_M = U * Qs[i] * V.transpose();
        // if(elm_id == 0){
        //   printf("Eigen_VECS<%d>:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",i,
        //     (double)eigen_M(0,0),(double)eigen_M(0,1),(double)eigen_M(0,2),
        //     (double)eigen_M(1,0),(double)eigen_M(1,1),(double)eigen_M(1,2),
        //     (double)eigen_M(2,0),(double)eigen_M(2,1),(double)eigen_M(2,2)
        //   );
        //   printf("U<%d>:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",i,
        //     (double)U(0,0),(double)U(0,1),(double)U(0,2),
        //     (double)U(1,0),(double)U(1,1),(double)U(1,2),
        //     (double)U(2,0),(double)U(2,1),(double)U(2,2)
        //   );
        //   printf("V<%d>:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",i,
        //     (double)V(0,0),(double)V(0,1),(double)V(0,2),
        //     (double)V(1,0),(double)V(1,1),(double)V(1,2),
        //     (double)V(2,0),(double)V(2,1),(double)V(2,2)
        //   );
        //   printf("QS<%d>:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",i,
        //     (double)Qs[i](0,0),(double)Qs[i](0,1),(double)Qs[i](0,2),
        //     (double)Qs[i](1,0),(double)Qs[i](1,1),(double)Qs[i](1,2),
        //     (double)Qs[i](2,0),(double)Qs[i](2,1),(double)Qs[i](2,2)
        //   );
        // }
        for (Ti d = 0; d != 9; ++d) eigen_vecs[i](d) = eigen_M(d % 3, d / 3);
      }

      // if(elm_id == 0){
      //   printf("Is_GPU: %f %f %f\n",(float)Is[0],(float)Is[1],(float)Is[2]);
      //   printf("PARAMS: %f %f\n",(float)lam,(float)mu);
      //   printf("S: %f %f %f\n",S[0],S[1],S[2]);
      //   printf("A:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
      //     (double)A(0,0),(double)A(0,1),(double)A(0,2),
      //     (double)A(1,0),(double)A(1,1),(double)A(1,2),
      //     (double)A(2,0),(double)A(2,1),(double)A(2,2)
      //   );
      //   auto A_rcmp = diag_mul(A_eigvecs, A_eigvals) * A_eigvecs.transpose();
      //   printf("A_rcmp:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
      //     (double)A_rcmp(0,0),(double)A_rcmp(0,1),(double)A_rcmp(0,2),
      //     (double)A_rcmp(1,0),(double)A_rcmp(1,1),(double)A_rcmp(1,2),
      //     (double)A_rcmp(2,0),(double)A_rcmp(2,1),(double)A_rcmp(2,2)
      //   );
      //   printf("EIGEN_VALS_GPU : %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
      //     (double)eigen_vals[0],(double)eigen_vals[1],(double)eigen_vals[2],
      //     (double)eigen_vals[3],(double)eigen_vals[4],(double)eigen_vals[5],
      //     (double)eigen_vals[6],(double)eigen_vals[7],(double)eigen_vals[8]
      //   );
      //   printf("Eigen_VECS_A_GPU:\n%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
      //     (double)A_eigvecs(0,0),(double)A_eigvecs(0,1),(double)A_eigvecs(0,2),
      //     (double)A_eigvecs(1,0),(double)A_eigvecs(1,1),(double)A_eigvecs(1,2),
      //     (double)A_eigvecs(2,0),(double)A_eigvecs(2,1),(double)A_eigvecs(2,2)
      //   );
      //   printf("EIGEN_VECS_GPU:\n");
      //   for(int i = 0;i < 9;++i)
      //     printf("EIGVEC<%d> : \n%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",i,
      //         (double)eigen_vecs[i][0],(double)eigen_vecs[i][1],(double)eigen_vecs[i][2],
      //         (double)eigen_vecs[i][3],(double)eigen_vecs[i][4],(double)eigen_vecs[i][5],
      //         (double)eigen_vecs[i][6],(double)eigen_vecs[i][7],(double)eigen_vecs[i][8]
      //     );
      // }

      auto dPdF = base_t::template hessian_t<VecT>::zeros();
      // if(elm_id == 0){
      //   printf("TEST_ELM<%d> : EVAL_VALS\n %f %f %f %f %f %f %f %f %f\n",
      //     (float)eigen_vals[0],
      //     (float)eigen_vals[1],
      //     (float)eigen_vals[2],
      //     (float)eigen_vals[3],
      //     (float)eigen_vals[4],
      //     (float)eigen_vals[5],
      //     (float)eigen_vals[6],
      //     (float)eigen_vals[7],
      //     (float)eigen_vals[8]
      //   );

      //   printf("PARAMS : %f %f %f %f
      //   %f\n",(float)lam,(float)mu,(float)Is[0],(float)Is[1],(float)Is[2]);

      //   printf("EIGEN_VECS:\n %f %f %f %f %f %f %f %f %f\n",
      //     (float)eigen_vecs[0].norm(),
      //     (float)eigen_vecs[1].norm(),
      //     (float)eigen_vecs[2].norm(),
      //     (float)eigen_vecs[3].norm(),
      //     (float)eigen_vecs[4].norm(),
      //     (float)eigen_vecs[5].norm(),
      //     (float)eigen_vecs[6].norm(),
      //     (float)eigen_vecs[7].norm(),
      //     (float)eigen_vecs[8].norm()
      //   );
      // }

      for (Ti i = 0; i != 9; ++i) {
        // eigen_vals[i] = eigen_vals[i] < detail::deduce_numeric_epsilon<value_type>() ?
        // detail::deduce_numeric_epsilon<value_type>()  : eigen_vals[i];
        eigen_vals[i] = zs::abs(eigen_vals[i]);
      }

      // if(elm_id == 0)
      //   printf("epsilon : %e
      //   %e\n",(double)detail::deduce_numeric_epsilon<value_type>(),(double)deduce_numeric_min<value_type>());

      for (Ti i = 0; i != 9; ++i)
        dPdF += eigen_vals[i] * zs::dyadic_prod(eigen_vecs[i], eigen_vecs[i]);

      // if(elm_id == 0){
      //   printf("dPdF<%d>:\n",elm_id);
      //   for(Ti i = 0;i < 9;++i)
      //     printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
      //       (double)dPdF(i,0),
      //       (double)dPdF(i,1),
      //       (double)dPdF(i,2),
      //       (double)dPdF(i,3),
      //       (double)dPdF(i,4),
      //       (double)dPdF(i,5),
      //       (double)dPdF(i,6),
      //       (double)dPdF(i,7),
      //       (double)dPdF(i,8)
      //     );
      // }

      return dPdF;
    }
  };

}  // namespace zs