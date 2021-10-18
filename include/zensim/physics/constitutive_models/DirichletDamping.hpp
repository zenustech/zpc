#pragma once
#include "../ConstitutiveModel.hpp"
#include "../ConstitutiveModelHelper.hpp"

namespace zs {

  template <int Opt = 1, typename T0, typename T1, typename Tn>
  constexpr auto compute_psi_deriv_hessian(const T0 v, const vec_t<T1, Tn, (Tn)3, (Tn)3>& L) {
    using R = math::op_result_t<T0, T1>;
    using vec9 = vec_t<R, Tn, (Tn)9>;
    using mat3 = vec_t<R, Tn, (Tn)3, (Tn)3>;
    using mat9 = vec_t<R, Tn, (Tn)9, (Tn)9>;
    using Ret
        = conditional_t<Opt == 1, std::tuple<R>,
                        conditional_t<Opt == 2, std::tuple<R, vec9>, std::tuple<R, vec9, mat9>>>;
    Ret ret{};

    std::get<0>(ret) = L.l2NormSqr() * (R)v * (T1)0.5;
    if constexpr (Opt > 1) std::get<1>(ret) = v * vectorize(L);
    if constexpr (Opt > 2) std::get<2>(ret) = v * mat9::identity();
    return ret;
  }

}  // namespace zs