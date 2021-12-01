#pragma once
#include "Givens.hpp"
#include "zensim/math/Vec.h"
#include "zensim/tpls/fmt/core.h"

namespace zs {

  namespace math {

    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range<0>() == VecT::template range<1>(),
                            VecT::template range<0>() == 2,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto polar_decomposition(const VecInterface<VecT>& A,
                                       GivensRotation<typename VecT::value_type>& R) noexcept {
      using value_type = typename VecT::value_type;
      constexpr auto N = VecT::template range<0>();
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

    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range<0>() == VecT::template range<1>(),
                            VecT::template range<0>() == 2,
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto qr_svd(const VecInterface<VecT>& A, GivensRotation<typename VecT::value_type>& U,
                          GivensRotation<typename VecT::value_type>& V) noexcept {
      using value_type = typename VecT::value_type;
      using index_type = typename VecT::index_type;
      constexpr auto N = VecT::template range<0>();
      typename VecT::template variant_vec<value_type, integer_seq<index_type, N>> S{};

      auto S_sym = polar_decomposition(A, U);
      value_type cosine{}, sine{};
      auto x{S_sym(0, 0)}, y{S_sym(0, 1)}, z{S_sym(1, 1)};
      auto y2 = y * y;

      if (y2 == 0) {  // S is already diagonal
        cosine = 1;
        sine = 0;
        S(0) = x;
        S(1) = z;
      } else {
        auto tau = (value_type)0.5 * (x - z);
        value_type w{zs::sqrt(tau * tau + y2)}, t{};
        if (tau > 0)  // tau + w > w > y > 0 ==> division is safe
          t = y / (tau + w);
        else  // tau - w < -w < -y < 0 ==> division is safe
          t = y / (tau - w);
        cosine = zs::rsqrt(t * t + (value_type)1);
        sine = -t * cosine;
        /*
          V = [cosine -sine; sine cosine]
          Sigma = V'SV. Only compute the diagonals for efficiency.
          Also utilize symmetry of S and don't form V yet.
        */
        value_type c2 = cosine * cosine;
        value_type _2csy = 2 * cosine * sine * y;
        value_type s2 = sine * sine;

        S(0) = c2 * x - _2csy + s2 * z;
        S(1) = s2 * x + _2csy + c2 * z;
      }
      // Sorting
      // Polar already guarantees negative sign is on the small magnitude singular value.
      if (S(0) < S(1)) {
        std::swap(S(0), S(1));
        V.c = -sine;
        V.s = cosine;
      } else {
        V.c = cosine;
        V.s = sine;
      }
      U *= V;
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
        auto S = polar_decomposition(A, r);
        r.fill(R);
        return std::make_tuple(R, S);
      } else if constexpr (N == 3) {
        ;
      } else
        ;
    }

    namespace detail {}  // namespace detail

    // Polar guarantees negative sign is on the small magnitude singular value.
    // S is guaranteed to be the closest one to identity.
    // R is guaranteed to be the closest rotation to A.
    template <typename VecT,
              enable_if_all<VecT::dim == 2, VecT::template range<0>() == VecT::template range<1>(),
                            std::is_floating_point_v<typename VecT::value_type>> = 0>
    constexpr auto qr_svd(const VecInterface<VecT>& A) noexcept {
      using value_type = typename VecT::value_type;
      using index_type = typename VecT::index_type;
      constexpr auto N = VecT::template range<0>();
      typename VecT::template variant_vec<value_type, typename VecT::extents> U{}, V{};
      if constexpr (N == 1) {
        typename VecT::template variant_vec<value_type, integer_seq<index_type, N>> S{A(0, 0)};
        U(0, 0) = V(0, 0) = 1;
        return std::make_tuple(U, S, V);
      } else if constexpr (N == 2) {
        GivensRotation<value_type> gu{0, 1}, gv{0, 1};
        auto S = qr_svd(A, gu, gv);
        gu.fill(U);
        gv.fill(V);
        return std::make_tuple(U, S, V);
      } else if constexpr (N == 3) {
        auto B = A.clone();
        U = U.identity();
        V = V.identity();
        upper_bidiagonalize(B, U, V);

        if constexpr (false) {
          auto printMat = [](auto&& mat, std::string msg = "") {
            using Mat = RM_CVREF_T(mat);
            if (!msg.empty()) fmt::print("## msg: {}\n", msg);
            // if constexpr (Mat::extent == 9)
            fmt::print("mat3[{}] ==\n{}, {}, {}\n{}, {}, {}\n{}, {}, {}\n", (void*)&mat, mat(0, 0),
                       mat(0, 1), mat(0, 2), mat(1, 0), mat(1, 1), mat(1, 2), mat(2, 0), mat(2, 1),
                       mat(2, 2));
          };
          printMat(A, "A");
          printMat(U, "U");
          printMat(B, "B");
          printMat(V.transpose(), "V^T");
          printMat(U * B * V.transpose(), "Achk (U B V^T)");
        }
        return std::make_tuple(U, A, V);  // wrong
      } else
        ;
    }

  }  // namespace math

}  // namespace zs
