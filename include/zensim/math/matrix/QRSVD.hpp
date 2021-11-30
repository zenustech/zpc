#pragma once
#include "Givens.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  namespace math {

    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range<0>() == VecT::template range<1>(),
                            VecT::template range<0>() == 2,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto polar_decomposition(const VecInterface<VecT>& A,
                                       GivensRotation<typename VecT::value_type>& R) noexcept {
      constexpr auto N = VecT::template range<0>();
      using value_type = typename VecT::value_type;
      typename VecT::template variant_vec<value_type, typename VecT::extents> S = A;
      vec<value_type, 2> x{A(0, 0) + A(1, 1), A(1, 0) - A(0, 1)};
      auto d = x.norm();
      if (d != 0) {
        R.c = x(0) / d;
        R.s = -x(1) / d;
      } else {
        R.c = 1;
        R.s = 0;
      }
      R.rowRotation(S);
      return S;
    }

    // Polar guarantees negative sign is on the small magnitude singular value.
    // S is guaranteed to be the closest one to identity.
    // R is guaranteed to be the closest rotation to A.
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range<0>() == VecT::template range<1>(),

                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto polar_decomposition(const VecInterface<VecT>& A) noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto N = VecT::template range<0>();
      typename VecT::template variant_vec<value_type, typename VecT::extents> R{};
      if constexpr (N == 1) {
        R(0, 0) = 1;
        return std::make_tuple(R, A.clone());
      } else if constexpr (N == 2) {
        GivensRotation<value_type> r{0, 1};
        auto S = polarDecomposition(A, r);
        r.fill(R);
        return std::make_tuple(R, S);
      } else if constexpr (N == 3) {
        ;
      } else
        ;
    }

  }  // namespace math

}  // namespace zs
