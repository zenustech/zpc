
namespace zs {

  template <typename T, typename Tn>
  constexpr auto asymmetric(const vec_impl<T, std::integer_sequence<Tn, (Tn)3>> &v) {
    return vec_t<T, Tn, (Tn)3, (Tn)3>{0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0};
  }

  template <typename T, typename Tn, Tn Nr, Tn Nc>
  constexpr auto row(const vec_impl<T, std::integer_sequence<Tn, Nr, Nc>> &mat, Tn i) {
    vec_t<T, Tn, Nc> ret{};
    for (Tn j = 0; j != Nc; ++j) ret(j) = mat(i, j);
    return ret;
  }
  template <typename T, typename Tn, Tn Nr, Tn Nc>
  constexpr auto col(const vec_impl<T, std::integer_sequence<Tn, Nr, Nc>> &mat, Tn j) {
    vec_t<T, Tn, Nr> ret{};
    for (Tn i = 0; i != Nr; ++i) ret(i) = mat(i, j);
    return ret;
  }
  template <typename T, typename Tn, Tn N>
  constexpr auto trace(const vec_impl<T, std::integer_sequence<Tn, N, N>> &mat) {
    constexpr Tn N2 = N * N;
    auto sum = (T)0;
    for (Tn i = 0; i < N2; i += (N + 1)) sum += mat.val(i);
    return sum;
  }

  template <typename T0, typename T1, typename Tn>
  constexpr auto cross(const vec_impl<T0, std::integer_sequence<Tn, (Tn)2>> &lhs,
                       const vec_impl<T1, std::integer_sequence<Tn, (Tn)2>> &rhs) noexcept {
    using R = math::op_result_t<T0, T1>;
    return (R)lhs(0) * (R)rhs(1) - (R)lhs(1) * (R)rhs(0);
  }
  template <typename T0, typename T1, typename Tn>
  constexpr auto cross(const vec_impl<T0, std::integer_sequence<Tn, (Tn)3>> &lhs,
                       const vec_impl<T1, std::integer_sequence<Tn, (Tn)3>> &rhs) noexcept {
    using R = math::op_result_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, (Tn)3>> res{};
    res(0) = lhs(1) * rhs(2) - lhs(2) * rhs(1);
    res(1) = lhs(2) * rhs(0) - lhs(0) * rhs(2);
    res(2) = lhs(0) * rhs(1) - lhs(1) * rhs(0);
    return res;
  }

  /// vector-vector product
  template <typename T0, typename T1, typename Tn, Tn N>
  constexpr auto dot(vec_impl<T0, std::integer_sequence<Tn, N>> const &row,
                     vec_impl<T1, std::integer_sequence<Tn, N>> const &col) noexcept {
    using R = math::op_result_t<T0, T1>;
    R sum = 0;
    for (Tn i = 0; i != N; ++i) sum += row(i) * col(i);
    return sum;
  }
  template <typename T0, typename T1, typename Tn, Tn Nc, Tn Nr>
  constexpr auto dyadic_prod(vec_impl<T0, std::integer_sequence<Tn, Nc>> const &col,
                             vec_impl<T1, std::integer_sequence<Tn, Nr>> const &row) noexcept {
    using R = math::op_result_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, Nc, Nr>> r{};
    for (Tn i = 0; i != Nc; ++i)
      for (Tn j = 0; j != Nr; ++j) r(i, j) = col(i) * row(j);
    return r;
  }

  /// matrix-vector product
  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T0, std::integer_sequence<Tn, N0, N1>> const &A,
                           vec_impl<T1, std::integer_sequence<Tn, N1>> const &x) noexcept {
    using R = math::op_result_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N0>> r{};
    for (Tn i = 0; i != N0; ++i) {
      r(i) = 0;
      for (Tn j = 0; j != N1; ++j) r(i) += A(i, j) * x(j);
    }
    return r;
  }
  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T1, std::integer_sequence<Tn, N0>> const &x,
                           vec_impl<T0, std::integer_sequence<Tn, N0, N1>> const &A) noexcept {
    using R = math::op_result_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N1>> r{};
    for (Tn i = 0; i != N1; ++i) {
      r(i) = 0;
      for (Tn j = 0; j != N0; ++j) r(i) += A(j, i) * x(j);
    }
    return r;
  }
  template <typename T0, typename T1, typename Tn, Tn Ni, Tn Nk, Tn Nj>
  constexpr auto operator*(vec_impl<T0, std::integer_sequence<Tn, Ni, Nk>> const &A,
                           vec_impl<T1, std::integer_sequence<Tn, Nk, Nj>> const &B) noexcept {
    using R = math::op_result_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, Ni, Nj>> r{};
    for (Tn i = 0; i != Ni; ++i)
      for (Tn j = 0; j != Nj; ++j) {
        r(i, j) = 0;
        for (Tn k = 0; k != Nk; ++k) r(i, j) += A(i, k) * B(k, j);
      }
    return r;
  }
  template <int i0, int i1, typename T, typename Tn, Tn N0, Tn N1>
  constexpr T det2(vec_impl<T, std::integer_sequence<Tn, N0, N1>> const &A) noexcept {
    return A(i0, 0) * A(i1, 1) - A(i1, 0) * A(i0, 1);
  }
  template <int i0, int i1, int i2, typename T, typename Tn>
  constexpr T det3(vec_impl<T, std::integer_sequence<Tn, (Tn)4, (Tn)4>> const &A, const T &d0,
                   const T &d1, const T &d2) noexcept {
    return A(i0, 2) * d0 + (-A(i1, 2) * d1 + A(i2, 2) * d2);
  }
  template <typename T, typename Tn>
  constexpr T determinant(vec_impl<T, std::integer_sequence<Tn, 1, 1>> const &A) noexcept {
    return A.val(0);
  }
  template <typename T, typename Tn>
  constexpr T determinant(vec_impl<T, std::integer_sequence<Tn, 2, 2>> const &A) noexcept {
    return A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1);
  }
  template <typename T, typename Tn>
  constexpr T determinant(vec_impl<T, std::integer_sequence<Tn, 3, 3>> const &A) noexcept {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
           - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
           + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  }
  template <typename T, typename Tn>
  constexpr T determinant(vec_impl<T, std::integer_sequence<Tn, 4, 4>> const &A) noexcept {
    T d2_01 = det2<0, 1>(A);
    T d2_02 = det2<0, 2>(A);
    T d2_03 = det2<0, 3>(A);
    T d2_12 = det2<1, 2>(A);
    T d2_13 = det2<1, 3>(A);
    T d2_23 = det2<2, 3>(A);
    T d3_0 = det3<1, 2, 3>(A, d2_23, d2_13, d2_12);
    T d3_1 = det3<0, 2, 3>(A, d2_23, d2_03, d2_02);
    T d3_2 = det3<0, 1, 3>(A, d2_13, d2_03, d2_01);
    T d3_3 = det3<0, 1, 2>(A, d2_12, d2_02, d2_01);
    return -A(0, 3) * d3_0 + A(1, 3) * d3_1 + -A(2, 3) * d3_2 + A(3, 3) * d3_3;
  }
  template <typename T, typename Tn>
  constexpr auto inverse(vec_impl<T, std::integer_sequence<Tn, 1, 1>> const &A) noexcept {
    return vec_impl<T, std::integer_sequence<Tn, 1, 1>>{(T)1 / A.val(0)};
  }
  template <typename T, typename Tn>
  constexpr auto inverse(vec_impl<T, std::integer_sequence<Tn, 2, 2>> const &A) noexcept {
    vec_impl<T, std::integer_sequence<Tn, 2, 2>> ret{};
    auto invdet = (T)1 / determinant(A);
    ret(0, 0) = A(1, 1) * invdet;
    ret(1, 0) = -A(1, 0) * invdet;
    ret(0, 1) = -A(0, 1) * invdet;
    ret(1, 1) = A(0, 0) * invdet;
    return ret;
  }
  template <int i, int j, typename T, typename Tn>
  constexpr T cofactor(vec_impl<T, std::integer_sequence<Tn, 3, 3>> const &A) noexcept {
    constexpr int i1 = (i + 1) % 3;
    constexpr int i2 = (i + 2) % 3;
    constexpr int j1 = (j + 1) % 3;
    constexpr int j2 = (j + 2) % 3;
    return A(i1, j1) * A(i2, j2) - A(i1, j2) * A(i2, j1);
  }
  template <int i1, int i2, int i3, int j1, int j2, int j3, typename T, typename Tn>
  constexpr T det3(vec_impl<T, std::integer_sequence<Tn, (Tn)4, (Tn)4>> const &A) noexcept {
    return A(i1, j1) * (A(i2, j2) * A(i3, j3) - A(i2, j3) * A(i3, j2));
  }
  template <int i, int j, typename T, typename Tn>
  constexpr T cofactor(vec_impl<T, std::integer_sequence<Tn, 4, 4>> const &A) noexcept {
    constexpr int i1 = (i + 1) % 4;
    constexpr int i2 = (i + 2) % 4;
    constexpr int i3 = (i + 3) % 4;
    constexpr int j1 = (j + 1) % 4;
    constexpr int j2 = (j + 2) % 4;
    constexpr int j3 = (j + 3) % 4;
    return det3<i1, i2, i3, j1, j2, j3>(A) + det3<i2, i3, i1, j1, j2, j3>(A)
           + det3<i3, i1, i2, j1, j2, j3>(A);
  }
  template <typename T, typename Tn>
  constexpr auto inverse(vec_impl<T, std::integer_sequence<Tn, 3, 3>> const &A) noexcept {
    vec_impl<T, std::integer_sequence<Tn, 3, 3>> ret{};
    ret(0, 0) = cofactor<0, 0>(A);
    ret(0, 1) = cofactor<1, 0>(A);
    ret(0, 2) = cofactor<2, 0>(A);
    const T invdet = (T)1 / (ret(0, 0) * A(0, 0) + ret(0, 1) * A(1, 0) + ret(0, 2) * A(2, 0));
    T c01 = cofactor<0, 1>(A) * invdet;
    T c11 = cofactor<1, 1>(A) * invdet;
    T c02 = cofactor<0, 2>(A) * invdet;
    ret(1, 2) = cofactor<2, 1>(A) * invdet;
    ret(2, 1) = cofactor<1, 2>(A) * invdet;
    ret(2, 2) = cofactor<2, 2>(A) * invdet;
    ret(1, 0) = c01;
    ret(1, 1) = c11;
    ret(2, 0) = c02;
    ret(0, 0) *= invdet;
    ret(0, 1) *= invdet;
    ret(0, 2) *= invdet;
    return ret;
  }
  template <typename T, typename Tn>
  constexpr auto inverse(vec_impl<T, std::integer_sequence<Tn, 4, 4>> const &A) noexcept {
    vec_impl<T, std::integer_sequence<Tn, 4, 4>> ret{};
    ret(0, 0) = cofactor<0, 0>(A);
    ret(1, 0) = -cofactor<0, 1>(A);
    ret(2, 0) = cofactor<0, 2>(A);
    ret(3, 0) = -cofactor<0, 3>(A);

    ret(0, 2) = cofactor<2, 0>(A);
    ret(1, 2) = -cofactor<2, 1>(A);
    ret(2, 2) = cofactor<2, 2>(A);
    ret(3, 2) = -cofactor<2, 3>(A);

    ret(0, 1) = -cofactor<1, 0>(A);
    ret(1, 1) = cofactor<1, 1>(A);
    ret(2, 1) = -cofactor<1, 2>(A);
    ret(3, 1) = cofactor<1, 3>(A);

    ret(0, 3) = -cofactor<3, 0>(A);
    ret(1, 3) = cofactor<3, 1>(A);
    ret(2, 3) = -cofactor<3, 2>(A);
    ret(3, 3) = cofactor<3, 3>(A);
    return ret
           / (A(0, 0) * ret(0, 0) + A(1, 0) * ret(0, 1) + A(2, 0) * ret(0, 2)
              + A(3, 0) * ret(0, 3));
  }
  template <typename T0, typename T1, typename Tn, Tn Nr, Tn Nc>
  constexpr auto diag_mul(vec_impl<T0, std::integer_sequence<Tn, Nr, Nc>> const &A,
                          vec_impl<T1, std::integer_sequence<Tn, Nc>> const &diag) noexcept {
    using R = math::op_result_t<T0, T1>;
    using extentsT = std::integer_sequence<Tn, Nr, Nc>;
    vec_impl<R, extentsT> r{};
    for (Tn i = 0; i != Nr; ++i)
      for (Tn j = 0; j != Nc; ++j) r(i, j) = A(i, j) * diag(j);
    return r;
  }
  template <typename T0, typename T1, typename Tn, Tn Nr, Tn Nc>
  constexpr auto diag_mul(vec_impl<T1, std::integer_sequence<Tn, Nr>> const &diag,
                          vec_impl<T0, std::integer_sequence<Tn, Nr, Nc>> const &A) noexcept {
    using R = math::op_result_t<T0, T1>;
    using extentsT = std::integer_sequence<Tn, Nr, Nc>;
    vec_impl<R, extentsT> r{};
    for (Tn i = 0; i != Nr; ++i)
      for (Tn j = 0; j != Nc; ++j) r(i, j) = A(i, j) * diag(i);
    return r;
  }
  /// affine transform
  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T0, std::integer_sequence<Tn, N0, N1 + 1>> const &A,
                           vec_impl<T1, std::integer_sequence<Tn, N1>> const &x) noexcept {
    using R = math::op_result_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N0 - 1>> r{};
    for (Tn i = 0; i != N0 - 1; ++i) {
      r(i) = 0;
      for (Tn j = 0; j != N1 + 1; ++j) r(i) += (j == N1 ? A(i, j) : A(i, j) * x(j));
    }
    return r;
  }
  template <typename T0, typename T1, typename Tn, Tn N0, Tn N1>
  constexpr auto operator*(vec_impl<T1, std::integer_sequence<Tn, N0>> const &x,
                           vec_impl<T0, std::integer_sequence<Tn, N0 + 1, N1>> const &A) noexcept {
    using R = math::op_result_t<T0, T1>;
    vec_impl<R, std::integer_sequence<Tn, N1 - 1>> r{};
    for (Tn i = 0; i != N1 - 1; ++i) {
      r(i) = 0;
      for (Tn j = 0; j != N0 + 1; ++j) r(i) += (j == N0 ? A(j, i) : A(j, i) * x(j));
    }
    return r;
  }

}  // namespace zs