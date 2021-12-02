#pragma once

#include <cmath>
#include <functional>
#include <type_traits>
#include <utility>

#include "zensim/math/bit/Bits.h"
#include "zensim/meta/ControlFlow.h"
#include "zensim/meta/Meta.h"
#include "zensim/meta/Relationship.h"
#include "zensim/meta/Sequence.h"

namespace zs {

  namespace mathutil_impl {
    // constexpr scan only available in c++20:
    // https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
    template <typename... Args, std::size_t... Is>
    constexpr auto incl_prefix_sum_impl(std::make_signed_t<std::size_t> I,
                                        std::index_sequence<Is...>, Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is <= I ? std::forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr auto excl_prefix_sum_impl(std::size_t I, std::index_sequence<Is...>,
                                        Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is < I ? std::forward<Args>(args) : 0) + ...);
    }
    template <typename... Args, std::size_t... Is>
    constexpr auto excl_suffix_mul_impl(std::make_signed_t<std::size_t> I,
                                        std::index_sequence<Is...>, Args &&...args) noexcept {
      return (((std::make_signed_t<std::size_t>)Is > I ? std::forward<Args>(args) : 1) * ...);
    }
  }  // namespace mathutil_impl

  namespace math {

    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr bool near_zero(T v) noexcept {
      constexpr auto eps = (T)128 * limits<T>::epsilon();
      return v >= -eps && v <= eps;
    }

    /// binary_op_result
    template <typename T0, typename T1> struct binary_op_result {
      static constexpr auto determine_type() noexcept {
        if constexpr (std::is_integral_v<T0> && std::is_integral_v<T1>) {
          using bigger_type = conditional_t<(sizeof(T0) >= sizeof(T1)), T0, T1>;
          if constexpr (std::is_signed_v<T0> || std::is_signed_v<T1>)
            return std::make_signed_t<bigger_type>{};
          else
            return bigger_type{};
        } else
          return std::common_type_t<T0, T1>{};
      }
      using type = decltype(determine_type());
    };
    template <typename T0, typename T1> using binary_op_result_t =
        typename binary_op_result<T0, T1>::type;

    /// op_result
    template <typename... Ts> struct op_result;
    template <typename T> struct op_result<T> { using type = T; };
    template <typename T, typename... Ts> struct op_result<T, Ts...> {
      using type = binary_op_result_t<T, typename op_result<Ts...>::type>;
    };
    template <typename... Args> using op_result_t = typename op_result<Args...>::type;
  }  // namespace math

  /**
   *  math intrinsics (not constexpr at all! just cheating the compiler)
   */
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T abs(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fabsf(v);
    else
      return ::fabs(v);
#else
    return std::abs(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T max(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fmaxf(x, y);
    else
      return ::fmax(x, y);
#else
    return std::max(x, y);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T min(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fminf(x, y);
    else
      return ::fmin(x, y);
#else
    return std::min(x, y);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T fma(T x, T y, T z) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fmaf(x, y, z);
    else
      return ::fma(x, y, z);
#else
    return std::fma(x, y, z);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T fmod(T x, T y) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::fmodf(x, y);
    else
      return ::fmod(x, y);
#else
    return std::fmod(x, y);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T ceil(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::ceilf(v);
    else
      return ::ceil(v);
#else
    return std::ceil(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T floor(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::floorf(v);
    else
      return ::floor(v);
#else
    return std::floor(v);
#endif
  }

  // different from math::sqrt
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T sqrt(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::sqrtf(v);
    else
      return ::sqrt(v);
#else
    return std::sqrt(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T rsqrt(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::rsqrtf(v);
    else
      return ::rsqrt(v);
#else
    return (T)1 / (T)std::sqrt(v);
#endif
  }

  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T log(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::logf(v);
    else
      return ::log(v);
#else
    return std::log(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T log1p(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::log1pf(v);
    else
      return ::log1p(v);
#else
    return std::log1p(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T exp(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::expf(v);
    else
      return ::exp(v);
#else
    return std::exp(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T pow(T base, T exp) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::powf(base, exp);
    else
      return ::pow(base, exp);
#else
    return std::pow(base, exp);
#endif
  }

  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T sin(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::sinf(v);
    else
      return ::sin(v);
#else
    return std::sin(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T asin(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::asinf(v);
    else
      return ::asin(v);
#else
    return std::asin(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T cos(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::cosf(v);
    else
      return ::cos(v);
#else
    return std::cos(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T acos(T v) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::acosf(v);
    else
      return ::acos(v);
#else
    return std::acos(v);
#endif
  }
  template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
  constexpr T atan2(T y, T x) noexcept {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
    if constexpr (is_same_v<T, float>)
      return ::atan2f(y, x);
    else
      return ::atan2(y, x);
#else
    return std::atan2(y, x);
#endif
  }

  /// customized zpc calls
  namespace math {
    template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0>
    constexpr T min(T x, T y) noexcept {
      return y < x ? y : x;
    }
    template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0>
    constexpr T max(T x, T y) noexcept {
      return y > x ? y : x;
    }
    template <typename T, enable_if_t<std::is_fundamental_v<T>> = 0> constexpr T abs(T x) noexcept {
      return x < 0 ? -x : x;
    }

#if 0
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T sqrtNewtonRaphson(T x, T curr = 1, T prev = 0) noexcept {
      return curr == prev ? curr : sqrtNewtonRaphson(x, (T)0.5 * (curr + x / curr), curr);
    }
#else
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T sqrtNewtonRaphson(T n, T relTol = (T)(sizeof(T) > 4 ? 1e-9 : 1e-6)) noexcept {
      constexpr auto eps = (T)128 * limits<T>::epsilon();
      if (n < -eps) return (T)limits<T>::quiet_NaN();
      if (n < (T)eps) return (T)0;

      T xn = (T)1;
      T xnp1 = (T)0.5 * (xn + n / xn);
      for (const auto tol = max(n * relTol, eps); abs(xnp1 - xn) > tol;
           xnp1 = (T)0.5 * (xn + n / xn))
        xn = xnp1;
      return xnp1;
    }
#endif
    /// ref: http://www.lomont.org/papers/2003/InvSqrt.pdf
    /// ref: https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    /// ref: https://community.wolfram.com/groups/-/m/t/1108896
    /// ref:
    /// https://www.codeproject.com/Articles/69941/Best-Square-Root-Method-Algorithm-Function-Precisi
    constexpr float q_rsqrt(float number) noexcept {
      uint32_t i{};
      float x2 = number * 0.5f, y = number;
      // i = *(uint32_t *)&y;
      i = reinterpret_bits<uint32_t>(y);
      i = 0x5f375a86 - (i >> 1);
      // y = *(float *)&i;
      y = reinterpret_bits<float>(i);
      y = y * (1.5f - (x2 * y * y));
      y = y * (1.5f - (x2 * y * y));
      return y;
    }
    constexpr double q_rsqrt(double number) noexcept {
      uint64_t i{};
      double x2 = number * 0.5, y = number;
      // i = *(uint64_t *)&y;
      i = reinterpret_bits<uint64_t>(y);
      i = 0x5fe6eb50c7b537a9 - (i >> 1);
      // y = *(double *)&i;
      y = reinterpret_bits<double>(i);
      y = y * (1.5 - (x2 * y * y));
      y = y * (1.5 - (x2 * y * y));
      return y;
    }
    constexpr float q_sqrt(float x) noexcept { return 1.f / q_rsqrt(x); }
    // best guess starting square
    constexpr double q_sqrt(double fp) noexcept { return 1.0 / q_rsqrt(fp); }
    /// ref:
    /// https://stackoverflow.com/questions/66752842/ieee-754-conformant-sqrtf-implementation-taking-into-account-hardware-restrict
    /* square root computation suitable for all IEEE-754 binary32 arguments */
    constexpr float sqrt(float arg) noexcept {
      const float FP32_INFINITY = reinterpret_bits<float>(0x7f800000u);
      const float FP32_QNAN = reinterpret_bits<float>(0x7fffffffu); /* system specific */
      const float scale_in = 0x1.0p+26f;
      const float scale_out = 0x1.0p-13f;
      float rsq{}, err{}, sqt{};

      if (arg < 0.0f) {
        return FP32_QNAN;
      }
      // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g1c6fe34b4ac091e40eceeb0bae58459f
      else if ((arg == 0.0f) || !(abs(arg) < FP32_INFINITY)) { /* Inf, NaN */
        return arg + arg;
      } else {
        /* scale subnormal arguments towards unity */
        arg = arg * scale_in;

        /* generate low-accuracy approximation to rsqrt(arg) */
        rsq = q_rsqrt(arg);

        /* apply two Newton-Raphson iterations with quadratic convergence */
        rsq = ((-0.5f * arg * rsq) * rsq + 0.5f) * rsq + rsq;
        rsq = ((-0.5f * arg * rsq) * rsq + 0.5f) * rsq + rsq;

        /* compute sqrt from rsqrt, round to nearest or even */
        sqt = rsq * arg;
        err = sqt * -sqt + arg;
        sqt = (0.5f * rsq * err + sqt);

        /* compensate scaling of argument by counter-scaling the result */
        sqt = sqt * scale_out;

        return sqt;
      }
    }

    namespace detail {
      template <typename T, typename Tn, enable_if_t<std::is_integral_v<Tn>> = 0>
      constexpr T pow_integral_recursive(T base, T val, Tn exp) noexcept {
        return exp > (Tn)1
                   ? ((exp & (Tn)1) ? pow_integral_recursive(base * base, val * base, exp / (Tn)2)
                                    : pow_integral_recursive(base * base, val, exp / (Tn)2))
                   : (exp == (Tn)1 ? val * base : val);
      }
    }  // namespace detail
    template <typename T, typename Tn,
              enable_if_all<std::is_arithmetic_v<T>, std::is_integral_v<Tn>> = 0>
    constexpr auto pow_integral(T base, Tn exp) noexcept {
      using R = T;  // math::op_result_t<T0, T1>;
      return exp == (Tn)3
                 ? base * base * base
                 : (exp == (Tn)2
                        ? base * base
                        : (exp == (Tn)1
                               ? base
                               : (exp == (Tn)0 ? (R)1
                                               : (exp == limits<Tn>::max()
                                                      ? limits<R>::infinity()
                                                      // make signed to get rid of compiler warn
                                                      : ((std::make_signed_t<Tn>)exp < 0
                                                             ? (R)0
                                                             : detail::pow_integral_recursive(
                                                                 (R)base, (R)1, (Tn)exp))))));
    }

    /**
     * Robustly computing log(x+1)/x
     */
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T log_1px_over_x(const T x, const T eps = 1e-6) noexcept {
      if (abs(x) < eps)
        return (T)1 - x / (T)2 + x * x / (T)3 - x * x * x / (T)4;
      else {
#if ZS_ENABLE_CUDA && defined(__CUDACC__)
        if constexpr (is_same_v<T, float>)
          return ::log1pf(x) / x;
        else
          return ::log1p(x) / x;
#else
        if constexpr (is_same_v<T, float>)
          return std::log1pf(x) / x;
        else
          return std::log1p(x) / x;
#endif
      }
    }
    /**
     * Robustly computing (logx-logy)/(x-y)
     */
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T diff_log_over_diff(const T x, const T y, const T eps = 1e-6) noexcept {
      return log_1px_over_x(x / y - (T)1, eps) / y;
    }
    /**
     * Robustly computing (x logy- y logx)/(x-y)
     */
    template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
    constexpr T diff_interlock_log_over_diff(const T x, const T y, const T logy,
                                             const T eps = 1e-6) noexcept {
      return logy - y * diff_log_over_diff(x, y, eps);
    }

  }  // namespace math

  template <typename... Args>
  constexpr auto incl_prefix_sum(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::incl_prefix_sum_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename... Args>
  constexpr auto excl_prefix_sum(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::excl_prefix_sum_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename... Args>
  constexpr auto excl_suffix_mul(std::size_t I, Args &&...args) noexcept {
    return mathutil_impl::excl_suffix_mul_impl(I, std::index_sequence_for<Args...>{},
                                               std::forward<Args>(args)...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto incl_prefix_sum(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return incl_prefix_sum(I, Ns...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto excl_prefix_sum(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return excl_prefix_sum(I, Ns...);
  }
  template <typename Tn, Tn... Ns>
  constexpr auto excl_suffix_mul(std::size_t I, std::integer_sequence<Tn, Ns...>) noexcept {
    return excl_suffix_mul(I, Ns...);
  }

  template <typename T, typename Data>
  constexpr auto linear_interop(T &&alpha, Data &&a, Data &&b) noexcept {
    return a + (b - a) * alpha;
  }

  template <typename T, enable_if_t<is_same_v<T, double>> = 0>
  constexpr auto lower_trunc(T v) noexcept {
    return v >= 0 ? (i64)v : ((i64)v) - 1;
  }
  template <typename T, enable_if_t<is_same_v<T, float>> = 0>
  constexpr auto lower_trunc(T v) noexcept {
    return v >= 0 ? (i32)v : ((i32)v) - 1;
  }

}  // namespace zs
